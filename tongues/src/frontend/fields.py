"""Phase 6: Field type collection."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .ast_compat import ASTNode, dict_walk, is_type
from .collection import CollectionCallbacks
from ..ir import (
    BOOL,
    BoolLit,
    FieldInfo,
    INT,
    IntLit,
    Pointer,
    STRING,
    StringLit,
    StructRef,
)

if TYPE_CHECKING:
    from ..ir import Expr, StructInfo, SymbolTable, Type


_PASCAL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _pascal_to_kebab(name: str) -> str:
    """PascalCase to kebab-case: BinaryOp → binary-op."""
    return _PASCAL_RE.sub("-", name).lower()


def _is_dataclass_class(node: ASTNode) -> tuple[bool, bool]:
    """Check decorator_list for @dataclass. Returns (is_dataclass, kw_only)."""
    for dec in node.get("decorator_list", []):
        if is_type(dec, ["Name"]) and dec.get("id") == "dataclass":
            return True, False
        if is_type(dec, ["Call"]):
            func = dec.get("func", {})
            if is_type(func, ["Name"]) and func.get("id") == "dataclass":
                kw_only = False
                for kw in dec.get("keywords", []):
                    if kw.get("arg") == "kw_only":
                        val = kw.get("value", {})
                        if is_type(val, ["Constant"]):
                            kw_only = bool(val.get("value"))
                return True, kw_only
    return False, False


def _unwrap_field_type(typ: Type) -> Type:
    """Strip Pointer(StructRef(X)) → StructRef(X) for field types."""
    if isinstance(typ, Pointer) and isinstance(typ.target, StructRef):
        return typ.target
    return typ


def _make_default_expr(value: ASTNode) -> Expr | None:
    """Convert constant AST node to IR literal."""
    if not is_type(value, ["Constant"]):
        return None
    v = value.get("value")
    if isinstance(v, bool):
        return BoolLit(typ=BOOL, value=v)
    if isinstance(v, int):
        return IntLit(typ=INT, value=v)
    if isinstance(v, str):
        return StringLit(typ=STRING, value=v)
    return None


def _is_field_call_default_factory(value: ASTNode) -> bool:
    """Detect field(default_factory=...)."""
    if not is_type(value, ["Call"]):
        return False
    func = value.get("func", {})
    if not (is_type(func, ["Name"]) and func.get("id") == "field"):
        return False
    for kw in value.get("keywords", []):
        if kw.get("arg") == "default_factory":
            return True
    return False


def _check_no_field_assign_in_block(block: list[ASTNode]) -> str | None:
    """Return field name if self.x = ... found inside block, else None."""
    for stmt in dict_walk({"_type": "_wrapper", "body": block}):
        if is_type(stmt, ["Assign"]):
            for target in stmt.get("targets", []):
                if (
                    is_type(target, ["Attribute"])
                    and is_type(target.get("value"), ["Name"])
                    and target.get("value", {}).get("id") == "self"
                ):
                    return target.get("attr")
        if is_type(stmt, ["AnnAssign"]):
            target = stmt.get("target", {})
            if (
                is_type(target, ["Attribute"])
                and is_type(target.get("value"), ["Name"])
                and target.get("value", {}).get("id") == "self"
            ):
                return target.get("attr")
    return None


def _check_no_new_fields_outside_init(
    func: ASTNode, known_fields: set[str]
) -> str | None:
    """Return field name if method introduces new self.x not in known_fields."""
    for stmt in dict_walk(func):
        if is_type(stmt, ["Assign"]):
            for target in stmt.get("targets", []):
                if (
                    is_type(target, ["Attribute"])
                    and is_type(target.get("value"), ["Name"])
                    and target.get("value", {}).get("id") == "self"
                ):
                    fname = target.get("attr")
                    if fname not in known_fields:
                        return fname
        if is_type(stmt, ["AnnAssign"]):
            target = stmt.get("target", {})
            if (
                is_type(target, ["Attribute"])
                and is_type(target.get("value"), ["Name"])
                and target.get("value", {}).get("id") == "self"
            ):
                fname = target.get("attr")
                if fname not in known_fields:
                    return fname
    return None


def _type_kind(typ: Type) -> str:
    """Return a short string for a type for error messages."""
    from ..ir import Primitive
    if isinstance(typ, Primitive):
        kind = typ.kind
        if kind == "string":
            return "str"
        return kind
    return str(type(typ).__name__)


def collect_init_fields(
    init: ASTNode,
    info: StructInfo,
    callbacks: CollectionCallbacks,
) -> None:
    """Collect fields assigned in __init__."""
    param_types: dict[str, str] = {}
    args = init.get("args", {})
    args_list = args.get("args", [])
    for arg in args_list:
        if arg.get("arg") != "self":
            info.init_params.append(arg.get("arg"))
            if arg.get("annotation"):
                param_types[arg.get("arg")] = callbacks.annotation_to_str(
                    arg.get("annotation")
                )
    has_computed_init = False
    for stmt in init.get("body", []):
        # Check for conditional field assignments
        if is_type(stmt, ["If", "For", "While"]):
            body_stmts = stmt.get("body", [])
            orelse_stmts = stmt.get("orelse", [])
            bad = _check_no_field_assign_in_block(body_stmts)
            if bad is None:
                bad = _check_no_field_assign_in_block(orelse_stmts)
            if bad is not None:
                raise RuntimeError(
                    "conditional field assignment not allowed: " + bad
                )
            continue
        if is_type(stmt, ["AnnAssign"]):
            target = stmt.get("target", {})
            if (
                is_type(target, ["Attribute"])
                and is_type(target.get("value"), ["Name"])
                and target.get("value", {}).get("id") == "self"
            ):
                field_name = target.get("attr")
                py_type = callbacks.annotation_to_str(stmt.get("annotation"))
                typ = _unwrap_field_type(
                    callbacks.py_type_to_ir(py_type, True)
                )
                if field_name in info.fields:
                    # Check type mismatch
                    existing = info.fields[field_name]
                    existing_kind = _type_kind(existing.typ)
                    new_kind = _type_kind(typ)
                    if existing_kind != new_kind:
                        raise RuntimeError(
                            f"field '{field_name}' declared as {existing_kind} "
                            f"but assigned {new_kind}"
                        )
                else:
                    info.fields[field_name] = FieldInfo(
                        name=field_name, typ=typ, py_name=field_name
                    )
                value = stmt.get("value")
                if value is not None:
                    if not (
                        is_type(value, ["Name"])
                        and value.get("id") in info.init_params
                    ):
                        has_computed_init = True
        elif is_type(stmt, ["Assign"]):
            for target in stmt.get("targets", []):
                if (
                    is_type(target, ["Attribute"])
                    and is_type(target.get("value"), ["Name"])
                    and target.get("value", {}).get("id") == "self"
                ):
                    field_name = target.get("attr")
                    value = stmt.get("value", {})
                    is_simple_param = (
                        is_type(value, ["Name"])
                        and value.get("id") in info.init_params
                    )
                    is_const_str = is_type(value, ["Constant"]) and isinstance(
                        value.get("value"), str
                    )
                    if is_simple_param:
                        info.param_to_field[value.get("id")] = field_name
                    elif is_const_str:
                        info.const_fields[field_name] = value.get("value")
                    else:
                        has_computed_init = True
                    if field_name in info.fields and not is_simple_param:
                        assert callbacks.infer_type_from_value is not None
                        inferred = _unwrap_field_type(
                            callbacks.infer_type_from_value(
                                value, param_types
                            )
                        )
                        existing_kind = _type_kind(info.fields[field_name].typ)
                        new_kind = _type_kind(inferred)
                        if existing_kind != new_kind:
                            raise RuntimeError(
                                f"field '{field_name}' declared as "
                                f"{existing_kind} but assigned {new_kind}"
                            )
                    elif field_name not in info.fields:
                        assert callbacks.infer_type_from_value is not None
                        typ = _unwrap_field_type(
                            callbacks.infer_type_from_value(value, param_types)
                        )
                        info.fields[field_name] = FieldInfo(
                            name=field_name, typ=typ, py_name=field_name
                        )
    if info.init_params:
        info.needs_constructor = True
    elif has_computed_init:
        info.needs_constructor = True
    if info.is_exception:
        info.needs_constructor = True


def collect_class_fields(
    node: ASTNode,
    symbols: SymbolTable,
    callbacks: CollectionCallbacks,
    hierarchy_root: str | None = None,
) -> None:
    """Collect fields from class body and __init__."""
    info = symbols.structs[node.get("name")]
    is_dc, kw_only = _is_dataclass_class(node)
    info.is_dataclass = is_dc
    info.kw_only = kw_only
    seen_fields: set[str] = set()
    # Collect class-level annotations
    for stmt in node.get("body", []):
        if is_type(stmt, ["AnnAssign"]) and is_type(stmt.get("target"), ["Name"]):
            target = stmt.get("target", {})
            field_name = target.get("id")
            if field_name in seen_fields:
                raise RuntimeError(
                    f"field '{field_name}' already declared"
                )
            seen_fields.add(field_name)
            py_type = callbacks.annotation_to_str(stmt.get("annotation"))
            typ = _unwrap_field_type(callbacks.py_type_to_ir(py_type, True))
            has_default = False
            default_expr: Expr | None = None
            value_node = stmt.get("value")
            if value_node is not None:
                if _is_field_call_default_factory(value_node):
                    raise RuntimeError(
                        "field(default_factory=...) not allowed"
                    )
                has_default = True
                default_expr = _make_default_expr(value_node)
            info.fields[field_name] = FieldInfo(
                name=field_name,
                typ=typ,
                py_name=field_name,
                has_default=has_default,
                default=default_expr,
            )
    # For dataclasses without __init__, populate init_params from body annotations
    has_init = False
    for stmt in node.get("body", []):
        if is_type(stmt, ["FunctionDef"]) and stmt.get("name") == "__init__":
            has_init = True
            collect_init_fields(stmt, info, callbacks)
    if is_dc and not has_init:
        for field_name in info.fields:
            info.init_params.append(field_name)
    # Check non-__init__ methods for new field introductions
    known_fields = set(info.fields.keys()) | set(info.const_fields.keys())
    for stmt in node.get("body", []):
        if is_type(stmt, ["FunctionDef"]) and stmt.get("name") != "__init__":
            bad = _check_no_new_fields_outside_init(stmt, known_fields)
            if bad is not None:
                raise RuntimeError(
                    f"field '{bad}' must be assigned in __init__"
                )
    # Exception classes always need constructors
    if info.is_exception:
        info.needs_constructor = True
    # Auto-generate kind from class name unless already set or class is hierarchy root
    if (
        info.init_params is not None
        and "kind" not in info.const_fields
        and info.name != hierarchy_root
    ):
        # Only generate kind if no init param maps to kind
        kind_from_param = any(
            info.param_to_field.get(p) == "kind" or p == "kind"
            for p in info.init_params
        )
        if not kind_from_param and has_init:
            info.const_fields["kind"] = _pascal_to_kebab(info.name)
    # Build field-to-structs mapping for Node subclasses
    if info.is_node:
        for field_name in info.fields:
            if field_name not in symbols.field_to_structs:
                symbols.field_to_structs[field_name] = []
            if info.name not in symbols.field_to_structs[field_name]:
                symbols.field_to_structs[field_name].append(info.name)


def collect_fields(
    tree: ASTNode,
    symbols: SymbolTable,
    callbacks: CollectionCallbacks,
) -> None:
    """Pass 6: Collect struct fields from class definitions."""
    from .hierarchy import find_hierarchy_root

    hierarchy_root = find_hierarchy_root(symbols)
    for node in tree.get("body", []):
        if is_type(node, ["ClassDef"]):
            collect_class_fields(node, symbols, callbacks, hierarchy_root)
