"""Phase 6: Field type collection.

Collect field types from class definitions. Fields are declared in class bodies
(dataclass-style annotations) or assigned in __init__. Also determines
constructor parameters, constant discriminator fields, and auto-generated kind
values.

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from __future__ import annotations

from typing import Callable

from .signatures import (
    SignatureResult,
    annotation_to_str,
    py_type_to_type_dict,
)

# Type alias for AST dict nodes
ASTNode = dict[str, object]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class FieldInfo:
    """A single class field."""

    def __init__(
        self,
        name: str,
        typ: dict[str, object],
        py_name: str,
        has_default: bool,
        default: dict[str, object] | None,
    ) -> None:
        self.name: str = name
        self.typ: dict[str, object] = typ
        self.py_name: str = py_name
        self.has_default: bool = has_default
        self.default: dict[str, object] | None = default

    def to_dict(self) -> dict[str, object]:
        """Serialize to a dict for test assertions."""
        d: dict[str, object] = {
            "typ": self.typ,
            "py_name": self.py_name,
            "has_default": self.has_default,
            "default": self.default,
        }
        return d


class ClassInfo:
    """Field and constructor info for a single class."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.fields: dict[str, FieldInfo] = {}
        self.init_params: list[str] = []
        self.param_to_field: dict[str, str] = {}
        self.const_fields: dict[str, str] = {}
        self.is_dataclass: bool = False
        self.kw_only: bool = False
        self.needs_constructor: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize to a dict for test assertions."""
        fields: dict[str, object] = {}
        fkeys = list(self.fields.keys())
        i = 0
        while i < len(fkeys):
            fields[fkeys[i]] = self.fields[fkeys[i]].to_dict()
            i += 1
        d: dict[str, object] = {
            "fields": fields,
            "init_params": list(self.init_params),
            "param_to_field": dict(self.param_to_field),
            "const_fields": dict(self.const_fields),
            "is_dataclass": self.is_dataclass,
            "kw_only": self.kw_only,
            "needs_constructor": self.needs_constructor,
        }
        return d


class FieldError:
    """An error found during field collection."""

    def __init__(self, lineno: int, col: int, message: str) -> None:
        self.lineno: int = lineno
        self.col: int = col
        self.message: str = message

    def __repr__(self) -> str:
        return (
            "error:"
            + str(self.lineno)
            + ":"
            + str(self.col)
            + ": [fields] "
            + self.message
        )


class FieldResult:
    """Result of field collection."""

    def __init__(self) -> None:
        self.classes: dict[str, ClassInfo] = {}
        self.field_to_structs: dict[str, list[str]] = {}
        self._errors: list[FieldError] = []

    def add_error(self, lineno: int, col: int, message: str) -> None:
        self._errors.append(FieldError(lineno, col, message))

    def errors(self) -> list[FieldError]:
        return self._errors

    def to_dict(self) -> dict[str, object]:
        """Serialize to nested dicts for test assertions."""
        classes: dict[str, object] = {}
        ckeys = list(self.classes.keys())
        i = 0
        while i < len(ckeys):
            classes[ckeys[i]] = self.classes[ckeys[i]].to_dict()
            i += 1
        return {"classes": classes}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _is_type(node: object, type_names: list[str]) -> bool:
    """Check if node is one of the given AST types."""
    if not isinstance(node, dict):
        return False
    t = node.get("_type")
    i = 0
    while i < len(type_names):
        if t == type_names[i]:
            return True
        i += 1
    return False


def _dict_walk(node: ASTNode) -> list[ASTNode]:
    """Walk dict-based AST, returns list of all nodes."""
    result: list[ASTNode] = [node]
    keys = list(node.keys())
    i = 0
    while i < len(keys):
        key = keys[i]
        if not key.startswith("_"):
            value = node[key]
            if isinstance(value, dict) and "_type" in value:
                result = result + _dict_walk(value)
            elif isinstance(value, list):
                j = 0
                while j < len(value):
                    item = value[j]
                    if isinstance(item, dict) and "_type" in item:
                        result = result + _dict_walk(item)
                    j += 1
        i += 1
    return result


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def _unwrap_field_type(typ: dict[str, object]) -> dict[str, object]:
    """Unwrap Pointer(StructRef(X)) → StructRef(X) and Slice(byte) → bytes."""
    if typ.get("_type") == "Pointer":
        target = typ.get("target")
        if isinstance(target, dict) and target.get("_type") == "StructRef":
            return target
    if typ.get("_type") == "Slice":
        element = typ.get("element")
        if isinstance(element, dict) and element.get("kind") == "byte":
            return {"kind": "bytes"}
    return typ


def _type_kind_str(typ: dict[str, object]) -> str:
    """Short string for a type, used in error messages."""
    kind = typ.get("kind")
    if isinstance(kind, str):
        if kind == "string":
            return "str"
        return kind
    t = typ.get("_type")
    if t == "Slice":
        return "list"
    if t == "Map":
        return "dict"
    if t == "Set":
        return "set"
    if t == "Tuple":
        return "tuple"
    if t == "Optional":
        return "Optional"
    if t == "StructRef":
        name = typ.get("name", "")
        if isinstance(name, str):
            return name
        return "struct"
    if t == "InterfaceRef":
        return "interface"
    if t == "FuncType":
        return "Callable"
    if t == "Pointer":
        return "Pointer"
    return "unknown"


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def _make_default_expr(node: ASTNode) -> dict[str, object] | None:
    """Convert a constant AST node to a default value dict."""
    if not _is_type(node, ["Constant"]):
        return None
    v = node.get("value")
    if isinstance(v, bool):
        return {"_type": "BoolLit", "value": v}
    if isinstance(v, int):
        return {"_type": "IntLit", "value": v}
    if isinstance(v, str):
        return {"_type": "StringLit", "value": v}
    return None


# ---------------------------------------------------------------------------
# PascalCase → kebab-case
# ---------------------------------------------------------------------------


def _pascal_to_kebab(name: str) -> str:
    """PascalCase to kebab-case: BinaryOp → binary-op."""
    result: list[str] = []
    i = 0
    while i < len(name):
        ch = name[i]
        if ch.isupper() and i > 0:
            prev = name[i - 1]
            if prev.islower() or prev.isdigit():
                result.append("-")
            elif prev.isupper() and i + 1 < len(name) and name[i + 1].islower():
                result.append("-")
        result.append(ch)
        i += 1
    return "".join(result).lower()


# ---------------------------------------------------------------------------
# Dataclass detection
# ---------------------------------------------------------------------------


def _is_dataclass_class(node: ASTNode) -> tuple[bool, bool]:
    """Check decorator_list for @dataclass. Returns (is_dataclass, kw_only)."""
    dec_list = node.get("decorator_list", [])
    if not isinstance(dec_list, list):
        return (False, False)
    i = 0
    while i < len(dec_list):
        dec = dec_list[i]
        if not isinstance(dec, dict):
            i += 1
            continue
        if _is_type(dec, ["Name"]) and dec.get("id") == "dataclass":
            return (True, False)
        if _is_type(dec, ["Call"]):
            func = dec.get("func", {})
            if isinstance(func, dict) and _is_type(func, ["Name"]) and func.get("id") == "dataclass":
                kw_only = False
                keywords = dec.get("keywords", [])
                if isinstance(keywords, list):
                    j = 0
                    while j < len(keywords):
                        kw = keywords[j]
                        if isinstance(kw, dict) and kw.get("arg") == "kw_only":
                            val = kw.get("value", {})
                            if isinstance(val, dict) and _is_type(val, ["Constant"]):
                                kw_only = bool(val.get("value"))
                        j += 1
                return (True, kw_only)
        i += 1
    return (False, False)


# ---------------------------------------------------------------------------
# field(default_factory=...) detection
# ---------------------------------------------------------------------------


def _is_field_call_default_factory(node: ASTNode) -> bool:
    """Detect field(default_factory=...)."""
    if not _is_type(node, ["Call"]):
        return False
    func = node.get("func", {})
    if not isinstance(func, dict):
        return False
    if not (_is_type(func, ["Name"]) and func.get("id") == "field"):
        return False
    keywords = node.get("keywords", [])
    if not isinstance(keywords, list):
        return False
    i = 0
    while i < len(keywords):
        kw = keywords[i]
        if isinstance(kw, dict) and kw.get("arg") == "default_factory":
            return True
        i += 1
    return False


# ---------------------------------------------------------------------------
# Conditional field assignment check
# ---------------------------------------------------------------------------


def _check_no_field_assign_in_block(block: list[ASTNode]) -> str | None:
    """Return field name if self.x = ... found inside block, else None."""
    wrapper: ASTNode = {"_type": "_wrapper", "body": block}
    all_nodes = _dict_walk(wrapper)
    i = 0
    while i < len(all_nodes):
        stmt = all_nodes[i]
        if _is_type(stmt, ["Assign"]):
            targets = stmt.get("targets", [])
            if isinstance(targets, list):
                j = 0
                while j < len(targets):
                    target = targets[j]
                    if isinstance(target, dict) and _is_type(target, ["Attribute"]):
                        val_node = target.get("value")
                        if isinstance(val_node, dict) and _is_type(val_node, ["Name"]):
                            if val_node.get("id") == "self":
                                attr = target.get("attr")
                                if isinstance(attr, str):
                                    return attr
                    j += 1
        if _is_type(stmt, ["AnnAssign"]):
            target = stmt.get("target", {})
            if isinstance(target, dict) and _is_type(target, ["Attribute"]):
                val_node = target.get("value")
                if isinstance(val_node, dict) and _is_type(val_node, ["Name"]):
                    if val_node.get("id") == "self":
                        attr = target.get("attr")
                        if isinstance(attr, str):
                            return attr
        i += 1
    return None


# ---------------------------------------------------------------------------
# Outside-init field check
# ---------------------------------------------------------------------------


def _check_no_new_fields_outside_init(
    func: ASTNode, known_fields: set[str]
) -> str | None:
    """Return field name if method introduces new self.x not in known_fields."""
    all_nodes = _dict_walk(func)
    i = 0
    while i < len(all_nodes):
        stmt = all_nodes[i]
        if _is_type(stmt, ["Assign"]):
            targets = stmt.get("targets", [])
            if isinstance(targets, list):
                j = 0
                while j < len(targets):
                    target = targets[j]
                    if isinstance(target, dict) and _is_type(target, ["Attribute"]):
                        val_node = target.get("value")
                        if isinstance(val_node, dict) and _is_type(val_node, ["Name"]):
                            if val_node.get("id") == "self":
                                fname = target.get("attr")
                                if isinstance(fname, str) and fname not in known_fields:
                                    return fname
                    j += 1
        if _is_type(stmt, ["AnnAssign"]):
            target = stmt.get("target", {})
            if isinstance(target, dict) and _is_type(target, ["Attribute"]):
                val_node = target.get("value")
                if isinstance(val_node, dict) and _is_type(val_node, ["Name"]):
                    if val_node.get("id") == "self":
                        fname = target.get("attr")
                        if isinstance(fname, str) and fname not in known_fields:
                            return fname
        i += 1
    return None


# ---------------------------------------------------------------------------
# Type inference from expressions
# ---------------------------------------------------------------------------


def _infer_type_from_value(
    node: ASTNode,
    param_types: dict[str, str],
    known_classes: set[str],
    func_return_types: dict[str, str],
    errors: list[FieldError],
    lineno: int,
) -> dict[str, object] | None:
    """Infer a type dict from an expression AST node. Returns None if cannot infer."""
    if not isinstance(node, dict):
        return None
    t = node.get("_type")
    # Literal constants
    if t == "Constant":
        v = node.get("value")
        if v is None:
            return {"kind": "void"}
        if isinstance(v, bool):
            return {"kind": "bool"}
        if isinstance(v, int):
            return {"kind": "int"}
        if isinstance(v, float):
            return {"kind": "float"}
        if isinstance(v, str):
            return {"kind": "string"}
    # Name reference — could be a param
    if t == "Name":
        name = node.get("id")
        if isinstance(name, str) and name in param_types:
            py_type = param_types[name]
            dummy_errors: list[object] = []
            from .signatures import SignatureError
            sig_errors: list[SignatureError] = []
            typ = py_type_to_type_dict(py_type, known_classes, sig_errors, lineno, 0)
            return _unwrap_field_type(typ)
        if isinstance(name, str):
            errors.append(
                FieldError(lineno, 0, "cannot infer type for field from '" + name + "'")
            )
        return None
    # Call — constructor or function call
    if t == "Call":
        func = node.get("func")
        if isinstance(func, dict) and _is_type(func, ["Name"]):
            func_name = func.get("id")
            if isinstance(func_name, str):
                # Constructor call
                if func_name in known_classes:
                    return {"_type": "StructRef", "name": func_name}
                # Known function call
                if func_name in func_return_types:
                    py_ret = func_return_types[func_name]
                    from .signatures import SignatureError
                    sig_errors = []
                    typ = py_type_to_type_dict(py_ret, known_classes, sig_errors, lineno, 0)
                    return _unwrap_field_type(typ)
        return None
    # Binary op — infer from operands
    if t == "BinOp":
        left = node.get("left")
        right = node.get("right")
        if isinstance(left, dict) and isinstance(right, dict):
            left_t = _infer_type_from_value(
                left, param_types, known_classes, func_return_types, errors, lineno
            )
            # Remove any errors added during operand inference (we just want the type)
            if left_t is not None:
                return left_t
        return None
    # Unary op — infer from operand
    if t == "UnaryOp":
        operand = node.get("operand")
        if isinstance(operand, dict):
            return _infer_type_from_value(
                operand, param_types, known_classes, func_return_types, errors, lineno
            )
    # List/Dict/Set/Tuple literal — basic type without element info
    if t == "List":
        return {"_type": "Slice", "element": {"_type": "InterfaceRef", "name": "any"}}
    if t == "Dict":
        return {"_type": "Map", "key": {"_type": "InterfaceRef", "name": "any"}, "value": {"_type": "InterfaceRef", "name": "any"}}
    if t == "Set":
        return {"_type": "Set", "element": {"_type": "InterfaceRef", "name": "any"}}
    if t == "Tuple":
        return {"_type": "Tuple", "elements": [], "variadic": False}
    return None


# ---------------------------------------------------------------------------
# Init field collection
# ---------------------------------------------------------------------------


def _collect_init_fields(
    init: ASTNode,
    info: ClassInfo,
    known_classes: set[str],
    func_return_types: dict[str, str],
    errors: list[FieldError],
) -> None:
    """Collect fields assigned in __init__."""
    args = init.get("args", {})
    if not isinstance(args, dict):
        args = {}
    # Collect param names and py_types
    param_types: dict[str, str] = {}
    args_list = args.get("args", [])
    if not isinstance(args_list, list):
        args_list = []
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if isinstance(arg, dict):
            arg_name = arg.get("arg")
            if isinstance(arg_name, str) and arg_name != "self":
                info.init_params.append(arg_name)
                annotation = arg.get("annotation")
                if annotation is not None:
                    param_types[arg_name] = annotation_to_str(annotation)
        i += 1
    # Also collect posonlyargs and kwonlyargs
    posonlyargs = args.get("posonlyargs", [])
    if isinstance(posonlyargs, list):
        i = 0
        while i < len(posonlyargs):
            arg = posonlyargs[i]
            if isinstance(arg, dict):
                arg_name = arg.get("arg")
                if isinstance(arg_name, str) and arg_name != "self":
                    info.init_params.append(arg_name)
                    annotation = arg.get("annotation")
                    if annotation is not None:
                        param_types[arg_name] = annotation_to_str(annotation)
            i += 1
    kwonlyargs = args.get("kwonlyargs", [])
    if isinstance(kwonlyargs, list):
        i = 0
        while i < len(kwonlyargs):
            arg = kwonlyargs[i]
            if isinstance(arg, dict):
                arg_name = arg.get("arg")
                if isinstance(arg_name, str) and arg_name != "self":
                    info.init_params.append(arg_name)
                    annotation = arg.get("annotation")
                    if annotation is not None:
                        param_types[arg_name] = annotation_to_str(annotation)
            i += 1
    has_computed_init = False
    body = init.get("body", [])
    if not isinstance(body, list):
        body = []
    lineno_val = init.get("lineno", 0)
    lineno = lineno_val if isinstance(lineno_val, int) else 0
    i = 0
    while i < len(body):
        stmt = body[i]
        if not isinstance(stmt, dict):
            i += 1
            continue
        stmt_lineno_val = stmt.get("lineno", lineno)
        stmt_lineno = stmt_lineno_val if isinstance(stmt_lineno_val, int) else lineno
        # Check for conditional field assignments
        if _is_type(stmt, ["If", "For", "While"]):
            body_stmts = stmt.get("body", [])
            orelse_stmts = stmt.get("orelse", [])
            if not isinstance(body_stmts, list):
                body_stmts = []
            if not isinstance(orelse_stmts, list):
                orelse_stmts = []
            bad = _check_no_field_assign_in_block(body_stmts)
            if bad is None:
                bad = _check_no_field_assign_in_block(orelse_stmts)
            if bad is not None:
                errors.append(
                    FieldError(
                        stmt_lineno,
                        0,
                        "conditional field assignment not allowed: " + bad,
                    )
                )
                return
            i += 1
            continue
        # AnnAssign: self.x: T = expr
        if _is_type(stmt, ["AnnAssign"]):
            target = stmt.get("target", {})
            if isinstance(target, dict) and _is_type(target, ["Attribute"]):
                val_node = target.get("value")
                if isinstance(val_node, dict) and _is_type(val_node, ["Name"]) and val_node.get("id") == "self":
                    field_name = target.get("attr")
                    if isinstance(field_name, str):
                        ann = stmt.get("annotation")
                        py_type = annotation_to_str(ann)
                        from .signatures import SignatureError
                        sig_errors: list[SignatureError] = []
                        typ = py_type_to_type_dict(py_type, known_classes, sig_errors, stmt_lineno, 0)
                        typ = _unwrap_field_type(typ)
                        if field_name in info.fields:
                            existing_kind = _type_kind_str(info.fields[field_name].typ)
                            new_kind = _type_kind_str(typ)
                            if existing_kind != new_kind:
                                errors.append(
                                    FieldError(
                                        stmt_lineno,
                                        0,
                                        "field '"
                                        + field_name
                                        + "' declared as "
                                        + existing_kind
                                        + " but assigned "
                                        + new_kind,
                                    )
                                )
                                return
                        else:
                            info.fields[field_name] = FieldInfo(
                                name=field_name,
                                typ=typ,
                                py_name=field_name,
                                has_default=False,
                                default=None,
                            )
                        value = stmt.get("value")
                        if value is not None:
                            if not (isinstance(value, dict) and _is_type(value, ["Name"]) and value.get("id") in param_types):
                                has_computed_init = True
            i += 1
            continue
        # Assign: self.x = value
        if _is_type(stmt, ["Assign"]):
            targets = stmt.get("targets", [])
            if not isinstance(targets, list):
                targets = []
            j = 0
            while j < len(targets):
                target = targets[j]
                if isinstance(target, dict) and _is_type(target, ["Attribute"]):
                    val_node = target.get("value")
                    if isinstance(val_node, dict) and _is_type(val_node, ["Name"]) and val_node.get("id") == "self":
                        field_name = target.get("attr")
                        if isinstance(field_name, str):
                            value = stmt.get("value", {})
                            if not isinstance(value, dict):
                                value = {}
                            is_simple_param = (
                                _is_type(value, ["Name"])
                                and isinstance(value.get("id"), str)
                                and value.get("id") in param_types
                            )
                            is_const_str = (
                                _is_type(value, ["Constant"])
                                and isinstance(value.get("value"), str)
                            )
                            if is_simple_param:
                                param_name = value.get("id")
                                if isinstance(param_name, str):
                                    info.param_to_field[param_name] = field_name
                            elif is_const_str:
                                const_val = value.get("value")
                                if isinstance(const_val, str):
                                    info.const_fields[field_name] = const_val
                            else:
                                has_computed_init = True
                            # Determine the field type
                            if field_name not in info.fields:
                                if is_simple_param:
                                    param_name = value.get("id")
                                    if isinstance(param_name, str) and param_name in param_types:
                                        py_type = param_types[param_name]
                                        from .signatures import SignatureError
                                        sig_errors = []
                                        typ = py_type_to_type_dict(
                                            py_type, known_classes, sig_errors, stmt_lineno, 0
                                        )
                                        typ = _unwrap_field_type(typ)
                                        info.fields[field_name] = FieldInfo(
                                            name=field_name,
                                            typ=typ,
                                            py_name=field_name,
                                            has_default=False,
                                            default=None,
                                        )
                                elif is_const_str:
                                    info.fields[field_name] = FieldInfo(
                                        name=field_name,
                                        typ={"kind": "string"},
                                        py_name=field_name,
                                        has_default=False,
                                        default=None,
                                    )
                                else:
                                    typ = _infer_type_from_value(
                                        value,
                                        param_types,
                                        known_classes,
                                        func_return_types,
                                        errors,
                                        stmt_lineno,
                                    )
                                    if typ is not None:
                                        typ = _unwrap_field_type(typ)
                                        info.fields[field_name] = FieldInfo(
                                            name=field_name,
                                            typ=typ,
                                            py_name=field_name,
                                            has_default=False,
                                            default=None,
                                        )
                            elif not is_simple_param:
                                # Check type mismatch
                                inferred = _infer_type_from_value(
                                    value,
                                    param_types,
                                    known_classes,
                                    func_return_types,
                                    errors,
                                    stmt_lineno,
                                )
                                if inferred is not None:
                                    inferred = _unwrap_field_type(inferred)
                                    existing_kind = _type_kind_str(info.fields[field_name].typ)
                                    new_kind = _type_kind_str(inferred)
                                    if existing_kind != new_kind:
                                        errors.append(
                                            FieldError(
                                                stmt_lineno,
                                                0,
                                                "field '"
                                                + field_name
                                                + "' declared as "
                                                + existing_kind
                                                + " but assigned "
                                                + new_kind,
                                            )
                                        )
                                        return
                j += 1
        i += 1
    if len(info.init_params) > 0:
        info.needs_constructor = True
    elif has_computed_init:
        info.needs_constructor = True


# ---------------------------------------------------------------------------
# Class field collection
# ---------------------------------------------------------------------------


def _collect_class_fields(
    node: ASTNode,
    known_classes: set[str],
    node_classes: set[str],
    hierarchy_roots: set[str],
    func_return_types: dict[str, str],
    result: FieldResult,
) -> None:
    """Collect fields from a single class body and __init__."""
    class_name = node.get("name", "")
    if not isinstance(class_name, str):
        class_name = ""
    lineno_val = node.get("lineno", 0)
    lineno = lineno_val if isinstance(lineno_val, int) else 0
    info = ClassInfo(class_name)
    is_dc, kw_only = _is_dataclass_class(node)
    info.is_dataclass = is_dc
    info.kw_only = kw_only
    seen_fields: set[str] = set()
    body = node.get("body", [])
    if not isinstance(body, list):
        body = []
    # Collect class-level annotations
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, dict) and _is_type(stmt, ["AnnAssign"]):
            target = stmt.get("target", {})
            if isinstance(target, dict) and _is_type(target, ["Name"]):
                field_name = target.get("id")
                if isinstance(field_name, str):
                    if field_name in seen_fields:
                        result.add_error(lineno, 0, "field '" + field_name + "' already declared")
                        return
                    seen_fields.add(field_name)
                    ann = stmt.get("annotation")
                    py_type = annotation_to_str(ann)
                    from .signatures import SignatureError
                    sig_errors: list[SignatureError] = []
                    typ = py_type_to_type_dict(py_type, known_classes, sig_errors, lineno, 0)
                    typ = _unwrap_field_type(typ)
                    has_default = False
                    default_expr: dict[str, object] | None = None
                    value_node = stmt.get("value")
                    if value_node is not None and isinstance(value_node, dict):
                        if _is_field_call_default_factory(value_node):
                            result.add_error(lineno, 0, "field(default_factory=...) not allowed")
                            return
                        has_default = True
                        default_expr = _make_default_expr(value_node)
                    info.fields[field_name] = FieldInfo(
                        name=field_name,
                        typ=typ,
                        py_name=field_name,
                        has_default=has_default,
                        default=default_expr,
                    )
        i += 1
    # Walk __init__ for field assignments
    has_init = False
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, dict) and _is_type(stmt, ["FunctionDef"]) and stmt.get("name") == "__init__":
            has_init = True
            _collect_init_fields(stmt, info, known_classes, func_return_types, result._errors)
            if len(result._errors) > 0:
                return
        i += 1
    # For dataclasses without __init__, populate init_params from body annotations
    if is_dc and not has_init:
        fkeys = list(info.fields.keys())
        j = 0
        while j < len(fkeys):
            info.init_params.append(fkeys[j])
            j += 1
    # Check non-__init__ methods for new field introductions
    known_field_set: set[str] = set()
    fkeys = list(info.fields.keys())
    j = 0
    while j < len(fkeys):
        known_field_set.add(fkeys[j])
        j += 1
    ckeys = list(info.const_fields.keys())
    j = 0
    while j < len(ckeys):
        known_field_set.add(ckeys[j])
        j += 1
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, dict) and _is_type(stmt, ["FunctionDef"]) and stmt.get("name") != "__init__":
            bad = _check_no_new_fields_outside_init(stmt, known_field_set)
            if bad is not None:
                result.add_error(lineno, 0, "field '" + bad + "' must be assigned in __init__")
                return
        i += 1
    # Auto-generate kind from class name
    if has_init and class_name not in hierarchy_roots:
        if "kind" not in info.const_fields:
            # Check if any init param maps to kind
            kind_from_param = False
            j = 0
            while j < len(info.init_params):
                p = info.init_params[j]
                mapped = info.param_to_field.get(p)
                if mapped == "kind" or p == "kind":
                    kind_from_param = True
                j += 1
            if not kind_from_param:
                info.const_fields["kind"] = _pascal_to_kebab(class_name)
    # Build field-to-structs mapping for Node subclasses
    if class_name in node_classes:
        fkeys = list(info.fields.keys())
        j = 0
        while j < len(fkeys):
            fname = fkeys[j]
            if fname not in result.field_to_structs:
                result.field_to_structs[fname] = []
            if class_name not in result.field_to_structs[fname]:
                result.field_to_structs[fname].append(class_name)
            j += 1
    result.classes[class_name] = info


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def collect_fields(
    tree: ASTNode,
    known_classes: set[str],
    node_classes: set[str],
    hierarchy_roots: set[str],
    sig_result: SignatureResult,
) -> FieldResult:
    """Collect field information from all classes in the module AST.

    Args:
        tree: The module AST dict.
        known_classes: Set of known class names from the name table.
        node_classes: Set of class names that are Node subclasses.
        hierarchy_roots: Set of class names that are hierarchy roots.
        sig_result: Signature result for function return types.
    """
    # Build func_return_types for type inference
    func_return_types: dict[str, str] = {}
    fkeys = list(sig_result.functions.keys())
    i = 0
    while i < len(fkeys):
        func = sig_result.functions[fkeys[i]]
        func_return_types[fkeys[i]] = func.return_py_type
        i += 1
    result = FieldResult()
    body = tree.get("body", [])
    if not isinstance(body, list):
        return result
    i = 0
    while i < len(body):
        node = body[i]
        if isinstance(node, dict) and _is_type(node, ["ClassDef"]):
            _collect_class_fields(
                node, known_classes, node_classes, hierarchy_roots,
                func_return_types, result,
            )
            if len(result._errors) > 0:
                return result
        i += 1
    return result
