"""Type validation pass (Phase 8b).

Runs after compute_expr_types() annotates the AST but before lowering.
Walks the typed AST with a flow-sensitive type environment, raising
ParseError on type violations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ast_compat import ASTNode, is_type, op_type
from .parse import ParseError
from ..ir import (
    BOOL,
    BYTE,
    FLOAT,
    INT,
    STRING,
    VOID,
    FuncType,
    InterfaceRef,
    Map,
    Optional,
    Pointer,
    Set,
    Slice,
    StructRef,
    Tuple,
)

if TYPE_CHECKING:
    from ..ir import FuncInfo, SymbolTable, Type


# ============================================================
# SOURCE TYPE HELPERS (pre-erasure type strings)
# ============================================================


def _split_union_parts(source_type: str) -> list[str]:
    """Split a source type string on | respecting brackets."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for c in source_type:
        if c == "[":
            depth += 1
            current.append(c)
        elif c == "]":
            depth -= 1
            current.append(c)
        elif c == "|" and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(c)
    parts.append("".join(current).strip())
    return [p for p in parts if p]


def _is_union_source(source_type: str) -> bool:
    """Does this source type contain a union (A | B)?"""
    parts = _split_union_parts(source_type)
    return len(parts) > 1


def _is_optional_source(source_type: str) -> bool:
    """Is this source type T | None?"""
    parts = _split_union_parts(source_type)
    return "None" in parts and len(parts) >= 2


def _non_none_parts(source_type: str) -> list[str]:
    """Get the non-None parts of a union source type."""
    return [p for p in _split_union_parts(source_type) if p != "None"]


def _needs_narrowing(source_type: str) -> bool:
    """Does this source type require narrowing before member access?"""
    if source_type == "object":
        return True
    return _is_union_source(source_type)


def _build_source_types(
    stmts: list[ASTNode],
    func_info: "FuncInfo | None",
) -> dict[str, str]:
    """Build source_types map from function params and annotated locals."""
    from .frontend import annotation_to_str

    source_types: dict[str, str] = {}
    if func_info:
        for p in func_info.params:
            if p.name != "self" and p.py_type:
                source_types[p.name] = p.py_type
    # Walk body for AnnAssign nodes (annotated locals)
    for stmt in stmts:
        _collect_ann_assigns(stmt, source_types, annotation_to_str)
    return source_types


def _collect_ann_assigns(
    node: ASTNode,
    source_types: dict[str, str],
    annotation_to_str: object,
) -> None:
    """Recursively collect annotated assignments from a statement."""
    if not isinstance(node, dict):
        return
    node_t = node.get("_type")
    if node_t == "AnnAssign":
        target = node.get("target", {})
        if is_type(target, ["Name"]):
            ann = node.get("annotation")
            if ann is not None:
                py_type = annotation_to_str(ann)
                if py_type:
                    source_types[target.get("id")] = py_type
        return
    # Recurse into compound statements
    for key in ("body", "orelse", "handlers", "finalbody"):
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                _collect_ann_assigns(child, source_types, annotation_to_str)
    # Handler body
    if node_t == "ExceptHandler":
        for child in node.get("body", []):
            _collect_ann_assigns(child, source_types, annotation_to_str)


# ============================================================
# TYPE ENVIRONMENT
# ============================================================


class TypeEnv:
    """Flow-sensitive type environment for validation."""

    def __init__(
        self,
        var_types: dict[str, Type],
        return_type: Type | None,
        return_py_type: str,
        symbols: SymbolTable,
        func_info: FuncInfo | None,
        class_name: str,
        node_types: set[str],
        hierarchy_root: str | None,
        kind_to_struct: dict[str, str],
        param_types: dict[str, Type],
        source_types: dict[str, str],
    ) -> None:
        self.var_types: dict[str, Type] = var_types
        self.return_type: Type | None = return_type
        self.return_py_type: str = return_py_type
        self.symbols: SymbolTable = symbols
        self.func_info: FuncInfo | None = func_info
        self.class_name: str = class_name
        self.node_types: set[str] = node_types
        self.hierarchy_root: str | None = hierarchy_root
        self.kind_to_struct: dict[str, str] = kind_to_struct
        self.param_types: dict[str, Type] = param_types
        self.source_types: dict[str, str] = source_types
        self.guarded_attrs: set[str] = set()  # attribute paths guarded by is not None

    def copy(self) -> TypeEnv:
        env = TypeEnv(
            var_types=self.var_types.copy(),
            return_type=self.return_type,
            return_py_type=self.return_py_type,
            symbols=self.symbols,
            func_info=self.func_info,
            class_name=self.class_name,
            node_types=self.node_types,
            hierarchy_root=self.hierarchy_root,
            kind_to_struct=self.kind_to_struct,
            param_types=self.param_types,
            source_types=self.source_types.copy(),
        )
        env.guarded_attrs = self.guarded_attrs.copy()
        return env

    def guard_attr(self, attr_path: str) -> TypeEnv:
        env = self.copy()
        env.guarded_attrs.add(attr_path)
        return env

    def narrow(self, var: str, typ: Type) -> TypeEnv:
        env = self.copy()
        env.var_types[var] = typ
        return env

    def narrow_source(self, var: str, typ: Type, source_type: str) -> TypeEnv:
        """Narrow both IR type and source type."""
        env = self.copy()
        env.var_types[var] = typ
        env.source_types[var] = source_type
        return env

    def lookup(self, name: str) -> Type:
        if name in self.var_types:
            return self.var_types[name]
        if name in self.param_types:
            return self.param_types[name]
        if name in self.symbols.constants:
            return self.symbols.constants[name]
        return InterfaceRef("any")

    def source_type(self, name: str) -> str | None:
        """Get the original Python type string for a variable."""
        return self.source_types.get(name)

    def raw_param_type(self, name: str) -> Type | None:
        """Get the original (un-transformed) parameter type."""
        return self.param_types.get(name)


# ============================================================
# ENTRY POINT
# ============================================================


def validate_function(
    stmts: list[ASTNode],
    var_types: dict[str, Type],
    symbols: SymbolTable,
    func_info: FuncInfo | None,
    class_name: str,
    node_types: set[str],
    hierarchy_root: str | None,
    kind_to_struct: dict[str, str],
) -> None:
    """Validate types in a function body. Raises ParseError on violations."""
    ret_type = func_info.return_type if func_info else None
    ret_py_type = func_info.return_py_type if func_info else ""
    param_types: dict[str, Type] = {}
    if func_info:
        for p in func_info.params:
            if p.name != "self":
                param_types[p.name] = p.typ
    source_types = _build_source_types(stmts, func_info)
    env = TypeEnv(
        var_types=var_types.copy(),
        return_type=ret_type,
        return_py_type=ret_py_type,
        symbols=symbols,
        func_info=func_info,
        class_name=class_name,
        node_types=node_types,
        hierarchy_root=hierarchy_root,
        kind_to_struct=kind_to_struct,
        param_types=param_types,
        source_types=source_types,
    )
    _validate_stmts(stmts, env)


# ============================================================
# STATEMENT VALIDATION
# ============================================================


def _validate_stmts(stmts: list[ASTNode], env: TypeEnv) -> tuple[TypeEnv, bool]:
    """Validate statements, returning (env, always_returns)."""
    for stmt in stmts:
        env, returned = _validate_stmt(stmt, env)
        if returned:
            return env, True
    return env, False


def _validate_stmt(stmt: ASTNode, env: TypeEnv) -> tuple[TypeEnv, bool]:
    """Validate a single statement. Returns (updated_env, always_returns)."""
    stmt_type = stmt.get("_type")
    if stmt_type == "Return":
        _validate_return(stmt, env)
        return env, True
    if stmt_type == "If":
        return _validate_if(stmt, env)
    if stmt_type == "For":
        _validate_for(stmt, env)
        return env, False
    if stmt_type == "While":
        _validate_while(stmt, env)
        return env, False
    if stmt_type == "Assign":
        _validate_assign(stmt, env)
        return env, False
    if stmt_type == "AnnAssign":
        _validate_ann_assign(stmt, env)
        return env, False
    if stmt_type == "Expr":
        value = stmt.get("value", {})
        _validate_expr(value, env)
        _check_iterator_escape_expr_stmt(value, env)
        return env, False
    if stmt_type == "Assert":
        return _validate_assert(stmt, env)
    if stmt_type == "Try":
        _validate_try(stmt, env)
        return env, False
    if stmt_type == "With":
        body = stmt.get("body", [])
        _validate_stmts(body, env)
        return env, False
    return env, False


# ============================================================
# RETURN VALIDATION
# ============================================================


def _validate_return(stmt: ASTNode, env: TypeEnv) -> None:
    value = stmt.get("value")
    if value is None or not isinstance(value, dict):
        return
    _validate_expr(value, env)
    _check_iterator_escape_return(value, env)
    if env.return_type is None:
        return
    actual = _expr_type(value, env)
    # If return type is a union, allow any member type
    if env.return_py_type and _is_union_source(env.return_py_type):
        return
    if not _is_assignable(actual, env.return_type, env):
        lineno = stmt.get("lineno", 0)
        raise ParseError(
            f"type error: cannot return {_type_name(actual)} as {_type_name(env.return_type)}",
            lineno,
            0,
        )


# ============================================================
# IF VALIDATION (with narrowing)
# ============================================================


def _validate_if(stmt: ASTNode, env: TypeEnv) -> tuple[TypeEnv, bool]:
    test = stmt.get("test", {})
    body = stmt.get("body", [])
    orelse = stmt.get("orelse", [])
    _validate_expr(test, env)
    _validate_truthiness(test, env)
    then_env, else_env = _extract_narrowing(test, env)
    _, then_returns = _validate_stmts(body, then_env)
    _, else_returns = _validate_stmts(orelse, else_env)
    if then_returns and else_returns:
        return env, True
    if then_returns:
        return else_env, False
    if else_returns:
        return then_env, False
    return env, False


# ============================================================
# FOR VALIDATION
# ============================================================


def _validate_for(stmt: ASTNode, env: TypeEnv) -> None:
    body = stmt.get("body", [])
    _validate_stmts(body, env)


# ============================================================
# WHILE VALIDATION
# ============================================================


def _validate_while(stmt: ASTNode, env: TypeEnv) -> None:
    test = stmt.get("test", {})
    body = stmt.get("body", [])
    _validate_expr(test, env)
    _validate_truthiness(test, env)
    _validate_stmts(body, env)


# ============================================================
# ASSIGN VALIDATION
# ============================================================


def _validate_assign(stmt: ASTNode, env: TypeEnv) -> None:
    targets = stmt.get("targets", [])
    value = stmt.get("value", {})
    if not isinstance(value, dict):
        return
    _validate_expr(value, env)
    _check_iterator_escape_assign(value, env, stmt)
    if len(targets) != 1:
        return
    target = targets[0]
    if is_type(target, ["Tuple"]):
        _validate_tuple_unpack(target, value, env, stmt)
        return
    if not is_type(target, ["Name"]):
        if is_type(target, ["Subscript"]):
            _validate_subscript_assign(target, value, env, stmt)
        return
    # Empty collection without annotation requires type annotation
    var_name = target.get("id")
    if not env.source_type(var_name):
        val_t = value.get("_type") if isinstance(value, dict) else None
        if val_t == "List" and not value.get("elts"):
            lineno = stmt.get("lineno", 0)
            raise ParseError(
                f"type error: empty list needs type annotation",
                lineno,
                0,
            )
        if val_t == "Dict" and not value.get("keys"):
            lineno = stmt.get("lineno", 0)
            raise ParseError(
                f"type error: empty dict needs type annotation",
                lineno,
                0,
            )


def _validate_ann_assign(stmt: ASTNode, env: TypeEnv) -> None:
    target = stmt.get("target", {})
    value = stmt.get("value")
    if value is None or not isinstance(value, dict):
        return
    _validate_expr(value, env)
    if not is_type(target, ["Name"]):
        return
    var_name = target.get("id")
    declared = env.var_types.get(var_name)
    if declared is None:
        return
    # If source type is union/optional, allow any member type or None
    src = env.source_type(var_name)
    if src and _is_union_source(src):
        return
    # None assigned to a primitive type is an error (but not for strings/interfaces/optionals)
    if _is_none_literal(value):
        if declared == INT or declared == FLOAT or declared == BOOL:
            lineno = stmt.get("lineno", 0)
            raise ParseError(
                f"type error: cannot assign None to {_type_name(declared)}",
                lineno,
                0,
            )
        return  # None is OK for STRING (sentinel), Optional, Interface, Pointer, etc.
    actual = _expr_type(value, env)
    if not _is_assignable(actual, declared, env):
        lineno = stmt.get("lineno", 0)
        raise ParseError(
            f"type error: cannot assign {_type_name(actual)} to {_type_name(declared)}",
            lineno,
            0,
        )


def _validate_subscript_assign(
    target: ASTNode, value: ASTNode, env: TypeEnv, stmt: ASTNode
) -> None:
    """Validate d[key] = val assignments for type correctness."""
    container = target.get("value", {})
    container_type = _expr_type(container, env)
    actual = _expr_type(value, env)
    if isinstance(container_type, Map):
        key_expr = target.get("slice", {})
        key_type = _expr_type(key_expr, env)
        if not _is_assignable(key_type, container_type.key, env):
            lineno = stmt.get("lineno", 0)
            raise ParseError(
                f"type error: dict key type mismatch: {_type_name(key_type)} vs {_type_name(container_type.key)}",
                lineno,
                0,
            )
        if not _is_assignable(actual, container_type.value, env):
            lineno = stmt.get("lineno", 0)
            raise ParseError(
                f"type error: dict value type mismatch: {_type_name(actual)} vs {_type_name(container_type.value)}",
                lineno,
                0,
            )


# ============================================================
# TUPLE UNPACK VALIDATION
# ============================================================


def _validate_tuple_unpack(
    target: ASTNode, value: ASTNode, env: TypeEnv, stmt: ASTNode
) -> None:
    """Validate tuple unpacking: a, b = expr."""
    elts = target.get("elts", [])
    val_type = _expr_type(value, env)
    # Check source type for optional tuple detection
    if is_type(value, ["Name"]):
        src = env.source_type(value.get("id", ""))
        if src and _is_optional_source(src):
            lineno = stmt.get("lineno", 0)
            raise ParseError(
                f"cannot unpack optional tuple without guard",
                lineno,
                0,
            )
    # Optional tuple cannot be unpacked without guard
    if isinstance(val_type, Optional) and isinstance(val_type.inner, Tuple):
        lineno = stmt.get("lineno", 0)
        raise ParseError(
            f"cannot unpack optional tuple without guard",
            lineno,
            0,
        )
    if isinstance(val_type, Tuple) and not val_type.variadic:
        if len(val_type.elements) > 0 and len(elts) != len(val_type.elements):
            lineno = stmt.get("lineno", 0)
            raise ParseError(
                f"cannot unpack tuple of {len(val_type.elements)} elements into {len(elts)} targets",
                lineno,
                0,
            )


# ============================================================
# ASSERT VALIDATION (for narrowing)
# ============================================================


def _validate_assert(stmt: ASTNode, env: TypeEnv) -> tuple[TypeEnv, bool]:
    test = stmt.get("test", {})
    _validate_expr(test, env)
    from .inference import is_isinstance_call
    isinstance_check = is_isinstance_call(test)
    if isinstance_check:
        var_name, type_name = isinstance_check
        narrowed_type = _name_to_type(type_name, env)
        if narrowed_type is not None:
            return env.narrow_source(var_name, narrowed_type, type_name), False
    is_not_none = _extract_is_not_none(test)
    if is_not_none:
        var_type = env.lookup(is_not_none)
        src = env.source_type(is_not_none)
        if isinstance(var_type, Optional):
            narrowed_src = _non_none_parts(src)[0] if src and _is_optional_source(src) else ""
            return env.narrow_source(is_not_none, var_type.inner, narrowed_src), False
        # Source type is optional but IR type was erased (e.g., str | None → STRING)
        if src and _is_optional_source(src):
            non_none = _non_none_parts(src)
            if len(non_none) == 1:
                return env.narrow_source(is_not_none, var_type, non_none[0]), False
    return env, False


# ============================================================
# TRY VALIDATION
# ============================================================


def _validate_try(stmt: ASTNode, env: TypeEnv) -> None:
    body = stmt.get("body", [])
    _validate_stmts(body, env)
    for handler in stmt.get("handlers", []):
        _validate_stmts(handler.get("body", []), env)


# ============================================================
# EXPRESSION VALIDATION
# ============================================================


def _validate_expr(expr: ASTNode, env: TypeEnv) -> None:
    """Recursively validate expressions."""
    if not isinstance(expr, dict):
        return
    node_t = expr.get("_type")
    if node_t == "Call":
        _validate_call(expr, env)
    elif node_t == "BinOp":
        _validate_binop(expr, env)
    elif node_t == "Attribute":
        _validate_attribute(expr, env)
    elif node_t == "Subscript":
        _validate_subscript(expr, env)
    elif node_t == "IfExp":
        _validate_ifexp(expr, env)
    elif node_t == "UnaryOp":
        _validate_expr(expr.get("operand", {}), env)
    elif node_t == "BoolOp":
        _validate_boolop(expr, env)
    elif node_t == "Compare":
        _validate_expr(expr.get("left", {}), env)
        for comp in expr.get("comparators", []):
            _validate_expr(comp, env)
    elif node_t == "List":
        _validate_list_literal(expr, env)
    elif node_t == "Dict":
        _validate_dict_literal(expr, env)
    elif node_t == "Tuple":
        for elt in expr.get("elts", []):
            _validate_expr(elt, env)
    elif node_t == "NamedExpr":
        _validate_expr(expr.get("value", {}), env)
    elif node_t == "ListComp":
        _validate_expr(expr.get("elt", {}), env)
    elif node_t == "SetComp":
        _validate_expr(expr.get("elt", {}), env)
    elif node_t == "DictComp":
        _validate_expr(expr.get("key", {}), env)
        _validate_expr(expr.get("value", {}), env)
    elif node_t == "GeneratorExp":
        _validate_expr(expr.get("elt", {}), env)


def _validate_call(expr: ASTNode, env: TypeEnv) -> None:
    func = expr.get("func", {})
    args = expr.get("args", [])
    for arg in args:
        _validate_expr(arg, env)
    if is_type(func, ["Attribute"]):
        _validate_method_call(expr, env)
    elif is_type(func, ["Name"]):
        _validate_func_call(expr, env)


def _validate_func_call(expr: ASTNode, env: TypeEnv) -> None:
    func = expr.get("func", {})
    func_name = func.get("id")
    args = expr.get("args", [])
    # Callable-typed variable invocation: check arg types
    var_type = env.lookup(func_name)
    if isinstance(var_type, FuncType):
        _check_callable_args(args, var_type, env, expr)
        return
    # Builtin function arg type checks
    if func_name == "len" and args:
        arg_type = _expr_type(args[0], env)
        if arg_type in (INT, FLOAT, BOOL):
            lineno = expr.get("lineno", 0)
            raise ParseError(
                f"type error: len() requires a sized type, got {_type_name(arg_type)}",
                lineno,
                0,
            )
    # Check function signatures from symbol table
    if func_name in env.symbols.functions:
        func_info = env.symbols.functions[func_name]
        _check_func_info_args(args, func_info, env, expr)


def _validate_method_call(expr: ASTNode, env: TypeEnv) -> None:
    func = expr.get("func", {})
    method = func.get("attr")
    obj = func.get("value", {})
    args = expr.get("args", [])
    obj_type = _expr_type(obj, env)
    # Check source type for un-narrowed union/object method calls
    if is_type(obj, ["Name"]):
        src = env.source_type(obj.get("id", ""))
        if src and _needs_narrowing(src):
            if src == "object":
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"cannot call method on object without narrowing",
                    lineno,
                    0,
                )
            parts = _non_none_parts(src)
            if "None" in _split_union_parts(src):
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"cannot call method on optional type (may be None)",
                    lineno,
                    0,
                )
            if len(parts) > 1:
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"cannot call method on union type without narrowing",
                    lineno,
                    0,
                )
    # Check hierarchy interface method availability
    if isinstance(obj_type, InterfaceRef) and obj_type.name != "any":
        root = obj_type.name
        if root in env.symbols.structs:
            struct_info = env.symbols.structs[root]
            if method not in struct_info.methods and method not in ("to_sexp", "get_kind", "ToSexp", "GetKind"):
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"method '{method}' not available on base type {root}",
                    lineno,
                    0,
                )
    # Unwrap Pointer for collection mutation checks (list params are Pointer(Slice))
    inner_type = obj_type.target if isinstance(obj_type, Pointer) else obj_type
    # Collection mutation checks
    if isinstance(inner_type, Slice):
        if method == "append" and args:
            _check_collection_element(args[0], inner_type.element, env, expr, "list")
        elif method == "extend" and args:
            arg_type = _expr_type(args[0], env)
            ext_type = arg_type.target if isinstance(arg_type, Pointer) else arg_type
            if isinstance(ext_type, Slice):
                if not _is_assignable(ext_type.element, inner_type.element, env):
                    lineno = expr.get("lineno", 0)
                    raise ParseError(
                        f"type error: cannot extend list[{_type_name(inner_type.element)}] with list[{_type_name(ext_type.element)}]",
                        lineno,
                        0,
                    )
        elif method == "insert" and len(args) >= 2:
            _check_collection_element(args[1], inner_type.element, env, expr, "list")
    elif isinstance(inner_type, Set):
        if method == "add" and args:
            _check_collection_element(args[0], inner_type.element, env, expr, "set")
    # Unbound method call: ClassName.method(args) — missing self
    if is_type(obj, ["Name"]) and obj.get("id") in env.symbols.structs:
        class_name = obj.get("id")
        struct_info = env.symbols.structs[class_name]
        if method in struct_info.methods:
            method_info = struct_info.methods[method]
            if method_info.is_method:
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"cannot call method without self: {class_name}.{method}",
                    lineno,
                    0,
                )


def _check_collection_element(
    arg: ASTNode, expected: "Type", env: TypeEnv, expr: ASTNode, coll_name: str
) -> None:
    actual = _expr_type(arg, env)
    if not _is_assignable(actual, expected, env):
        lineno = expr.get("lineno", 0)
        raise ParseError(
            f"type error: cannot add {_type_name(actual)} to {coll_name}[{_type_name(expected)}]",
            lineno,
            0,
        )


def _check_callable_args(
    args: list[ASTNode], func_type: FuncType, env: TypeEnv, expr: ASTNode
) -> None:
    expected_params = func_type.params
    if len(args) != len(expected_params):
        lineno = expr.get("lineno", 0)
        raise ParseError(
            f"type error: expected {len(expected_params)} arguments, got {len(args)}",
            lineno,
            0,
        )
    for i, (arg, expected) in enumerate(zip(args, expected_params)):
        actual = _expr_type(arg, env)
        if not _is_assignable(actual, expected, env):
            lineno = expr.get("lineno", 0)
            raise ParseError(
                f"type error: argument {i + 1} has type {_type_name(actual)}, expected {_type_name(expected)}",
                lineno,
                0,
            )


def _check_func_info_args(
    args: list[ASTNode], func_info: "FuncInfo", env: TypeEnv, expr: ASTNode
) -> None:
    params = func_info.params
    param_list = [p for p in params if p.name != "self"]
    required = sum(1 for p in param_list if not p.has_default)
    if len(args) < required or len(args) > len(param_list):
        return
    for i, arg in enumerate(args):
        if i >= len(param_list):
            break
        expected = param_list[i].typ
        actual = _expr_type(arg, env)
        # When expected is FuncType and arg is a function reference, resolve its type
        if isinstance(expected, FuncType) and actual == InterfaceRef("any"):
            resolved = _resolve_func_ref(arg, env)
            if resolved is not None:
                actual = resolved
        if isinstance(expected, FuncType) and isinstance(actual, FuncType):
            if not _callable_assignable(actual, expected):
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"type error: argument {i + 1} callable type mismatch",
                    lineno,
                    0,
                )
        elif not _is_assignable(actual, expected, env):
            lineno = expr.get("lineno", 0)
            raise ParseError(
                f"type error: argument {i + 1} has type {_type_name(actual)}, expected {_type_name(expected)}",
                lineno,
                0,
            )


def _resolve_func_ref(arg: ASTNode, env: TypeEnv) -> "FuncType | None":
    """Resolve a function name reference to its FuncType."""
    if not is_type(arg, ["Name"]):
        return None
    name = arg.get("id", "")
    if name in env.symbols.functions:
        fi = env.symbols.functions[name]
        param_types = tuple(p.typ for p in fi.params if p.name != "self")
        return FuncType(params=param_types, ret=fi.return_type)
    # Builtins
    if name == "len":
        return FuncType(params=(InterfaceRef("any"),), ret=INT)
    if name == "str":
        return FuncType(params=(InterfaceRef("any"),), ret=STRING)
    if name == "int":
        return FuncType(params=(InterfaceRef("any"),), ret=INT)
    if name == "bool":
        return FuncType(params=(InterfaceRef("any"),), ret=BOOL)
    return None


def _callable_assignable(actual: FuncType, expected: FuncType) -> bool:
    if len(actual.params) != len(expected.params):
        return False
    for a, e in zip(actual.params, expected.params):
        if a != e and e != InterfaceRef("any") and a != InterfaceRef("any"):
            return False
    if actual.ret != expected.ret and expected.ret != InterfaceRef("any") and actual.ret != InterfaceRef("any"):
        return False
    return True


# ============================================================
# BINOP VALIDATION
# ============================================================


def _validate_binop(expr: ASTNode, env: TypeEnv) -> None:
    left = expr.get("left", {})
    right = expr.get("right", {})
    _validate_expr(left, env)
    _validate_expr(right, env)
    op = op_type(expr.get("op"))
    # Check source types for un-narrowed union/object arithmetic
    if is_type(left, ["Name"]):
        src = env.source_type(left.get("id", ""))
        if src and _needs_narrowing(src):
            if src == "object":
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"cannot use object in arithmetic without narrowing",
                    lineno,
                    0,
                )
            all_parts = _split_union_parts(src)
            non_none = _non_none_parts(src)
            if "None" in all_parts:
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"cannot use {src} in arithmetic (may be None)",
                    lineno,
                    0,
                )
            if len(non_none) > 1:
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"cannot use union type {src} in arithmetic without narrowing",
                    lineno,
                    0,
                )
    left_type = _expr_type(left, env)
    right_type = _expr_type(right, env)
    if op == "Add":
        # String + non-string or non-string + string
        if (left_type == STRING) != (right_type == STRING):
            if left_type == STRING or right_type == STRING:
                other = right_type if left_type == STRING else left_type
                if other not in (STRING, InterfaceRef("any")):
                    lineno = expr.get("lineno", 0)
                    raise ParseError(
                        f"type error: cannot add {_type_name(left_type)} and {_type_name(right_type)}",
                        lineno,
                        0,
                    )
        if isinstance(left_type, Slice) and isinstance(right_type, Slice):
            if not _is_assignable(right_type.element, left_type.element, env):
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"type error: cannot concatenate list[{_type_name(left_type.element)}] and list[{_type_name(right_type.element)}]",
                    lineno,
                    0,
                )


# ============================================================
# BOOLOP VALIDATION
# ============================================================


def _validate_boolop(expr: ASTNode, env: TypeEnv) -> None:
    """Validate BoolOp with narrowing propagation for 'and'."""
    op = expr.get("op", {})
    values = expr.get("values", [])
    if is_type(op, ["And"]):
        # In `a and b`, b is evaluated only if a is truthy → narrow from a
        current = env
        for val in values:
            _validate_expr(val, current)
            then_env, _ = _extract_narrowing(val, current)
            current = then_env
    elif is_type(op, ["Or"]):
        for val in values:
            _validate_expr(val, env)
    else:
        for val in values:
            _validate_expr(val, env)


# ============================================================
# ATTRIBUTE VALIDATION
# ============================================================


def _validate_attribute(expr: ASTNode, env: TypeEnv) -> None:
    obj = expr.get("value", {})
    _validate_expr(obj, env)
    attr = expr.get("attr")
    obj_type = _expr_type(obj, env)
    # Check source type for un-narrowed union/object/optional attribute access
    if is_type(obj, ["Name"]):
        name = obj.get("id", "")
        src = env.source_type(name)
        if src and _needs_narrowing(src):
            # Kind check on union (node.kind) — allow it for discrimination
            if attr == "kind":
                return
            if src == "object":
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"cannot access attribute '{attr}' on object without narrowing",
                    lineno,
                    0,
                )
            parts = _non_none_parts(src)
            if "None" in _split_union_parts(src):
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"cannot access '{attr}' on optional type (may be None)",
                    lineno,
                    0,
                )
            # Multi-variant union without None — check if all parts share the attribute
            if len(parts) > 1:
                if not _all_parts_have_field(parts, attr, env):
                    lineno = expr.get("lineno", 0)
                    raise ParseError(
                        f"attribute '{attr}' not available on all union members",
                        lineno,
                        0,
                    )
                return
        # Already narrowed or non-union — fall through
    # Optional without guard (IR-level) — skip if guarded by is not None
    if isinstance(obj_type, Optional):
        expr_path = _attr_path(obj)
        if expr_path and expr_path in env.guarded_attrs:
            return  # guarded by is not None check
        lineno = expr.get("lineno", 0)
        raise ParseError(
            f"cannot access '{attr}' on optional type (may be None)",
            lineno,
            0,
        )


def _all_parts_have_field(parts: list[str], attr: str, env: TypeEnv) -> bool:
    """Check if all union member types share a field."""
    for part in parts:
        if part in env.symbols.structs:
            info = env.symbols.structs[part]
            if attr not in info.fields and attr not in info.methods:
                return False
        else:
            return False
    return True


# ============================================================
# SUBSCRIPT VALIDATION
# ============================================================


def _validate_subscript(expr: ASTNode, env: TypeEnv) -> None:
    obj = expr.get("value", {})
    _validate_expr(obj, env)
    obj_type = _expr_type(obj, env)
    # Check source type for un-narrowed object subscript
    if is_type(obj, ["Name"]):
        src = env.source_type(obj.get("id", ""))
        if src == "object":
            lineno = expr.get("lineno", 0)
            raise ParseError(
                f"cannot subscript object without narrowing",
                lineno,
                0,
            )
    # Tuple index out of bounds (only for fixed-element tuples with actual elements)
    if isinstance(obj_type, Tuple) and not obj_type.variadic and len(obj_type.elements) > 0:
        slice_node = expr.get("slice", {})
        if is_type(slice_node, ["Constant"]) and isinstance(slice_node.get("value"), int):
            idx = slice_node.get("value")
            if idx < 0 or idx >= len(obj_type.elements):
                lineno = expr.get("lineno", 0)
                raise ParseError(
                    f"tuple index {idx} out of bounds for tuple of length {len(obj_type.elements)}",
                    lineno,
                    0,
                )


# ============================================================
# IFEXP (TERNARY) VALIDATION
# ============================================================


def _validate_ifexp(expr: ASTNode, env: TypeEnv) -> None:
    test = expr.get("test", {})
    body = expr.get("body", {})
    orelse = expr.get("orelse", {})
    _validate_expr(test, env)
    _validate_truthiness(test, env)
    _validate_expr(body, env)
    _validate_expr(orelse, env)


# ============================================================
# LIST AND DICT LITERAL VALIDATION
# ============================================================


def _validate_list_literal(expr: ASTNode, env: TypeEnv) -> None:
    elts = expr.get("elts", [])
    if not elts:
        return
    for elt in elts:
        _validate_expr(elt, env)
    # Check if the containing context expects a union list type
    # If so, skip homogeneity check (e.g., list[int | str] = [1, "hello"])
    # We detect this by checking if the return type or declared variable type is a union list
    if _parent_expects_union_list(expr, env):
        return
    # Only check homogeneity if all elements resolve to concrete types
    first_type = _expr_type(elts[0], env)
    if first_type == InterfaceRef("any"):
        return  # Can't validate if types are unknown
    for elt in elts[1:]:
        elt_type = _expr_type(elt, env)
        if elt_type == InterfaceRef("any"):
            return
        if not _is_assignable(elt_type, first_type, env) and not _is_assignable(first_type, elt_type, env):
            lineno = expr.get("lineno", 0)
            raise ParseError(
                f"type error: mixed types in list literal: {_type_name(first_type)} and {_type_name(elt_type)}",
                lineno,
                0,
            )


def _parent_expects_union_list(expr: ASTNode, env: TypeEnv) -> bool:
    """Check if the list literal is in a context that expects a union-typed list."""
    # If return type is a union list, allow mixed types
    if env.return_py_type:
        rp = env.return_py_type
        if rp.startswith("list[") and _is_union_source(rp[5:-1]):
            return True
    return False


def _validate_dict_literal(expr: ASTNode, env: TypeEnv) -> None:
    keys = expr.get("keys", [])
    values = expr.get("values", [])
    for k in keys:
        if isinstance(k, dict):
            _validate_expr(k, env)
    for v in values:
        if isinstance(v, dict):
            _validate_expr(v, env)
    # Check key/value type consistency
    if not keys:
        return
    first_key_type = _expr_type(keys[0], env) if isinstance(keys[0], dict) else None
    first_val_type = _expr_type(values[0], env) if values and isinstance(values[0], dict) else None
    if first_key_type and first_key_type != InterfaceRef("any"):
        for k in keys[1:]:
            if isinstance(k, dict):
                kt = _expr_type(k, env)
                if kt != InterfaceRef("any") and not _is_assignable(kt, first_key_type, env):
                    lineno = expr.get("lineno", 0)
                    raise ParseError(
                        f"type error: mixed key types in dict literal",
                        lineno,
                        0,
                    )
    if first_val_type and first_val_type != InterfaceRef("any"):
        for v in values[1:]:
            if isinstance(v, dict):
                vt = _expr_type(v, env)
                if vt != InterfaceRef("any") and not _is_assignable(vt, first_val_type, env):
                    lineno = expr.get("lineno", 0)
                    raise ParseError(
                        f"type error: mixed value types in dict literal",
                        lineno,
                        0,
                    )


# ============================================================
# TRUTHINESS VALIDATION
# ============================================================


_EAGER_CONSUMERS = frozenset({
    "list", "tuple", "set", "dict", "frozenset",
    "sum", "min", "max", "any", "all", "sorted",
})


def _validate_truthiness(expr: ASTNode, env: TypeEnv) -> None:
    """Validate that the expression has unambiguous truthiness."""
    if not isinstance(expr, dict):
        return
    node_t = expr.get("_type")
    if node_t == "BoolOp":
        for val in expr.get("values", []):
            _validate_truthiness(val, env)
        return
    if node_t == "UnaryOp" and op_type(expr.get("op")) == "Not":
        _validate_truthiness(expr.get("operand", {}), env)
        return
    if node_t in ("Compare", "Call"):
        return
    if node_t == "NamedExpr":
        # Walrus in boolean context: if (val := expr) — skip truthiness check
        # since the pattern is always used for optional narrowing
        return
    # For Name nodes, use source type if available
    if node_t == "Name":
        name = expr.get("id")
        src = env.source_type(name)
        if src:
            _check_source_truthiness(src, expr)
            return
        param_type = env.raw_param_type(name)
        if param_type is not None:
            _check_type_truthiness(param_type, None, expr)
            return
    typ = _expr_type(expr, env)
    _check_type_truthiness(typ, None, expr)


def _check_source_truthiness(source_type: str, expr: ASTNode) -> None:
    """Check truthiness using the original Python type string."""
    if source_type == "bool":
        return
    # Non-union types — delegate to IR-level check
    if not _is_union_source(source_type):
        if source_type in ("int", "float"):
            lineno = expr.get("lineno", 0) if isinstance(expr, dict) else 0
            raise ParseError(
                f"truthiness of {source_type} not allowed (zero is valid data)",
                lineno,
                0,
            )
        return
    # Union types
    parts = _split_union_parts(source_type)
    non_none = [p for p in parts if p != "None"]
    has_none = "None" in parts
    if not has_none:
        # Pure union like int | str — truthiness of int is ambiguous
        for p in non_none:
            if p in ("int", "float"):
                lineno = expr.get("lineno", 0) if isinstance(expr, dict) else 0
                raise ParseError(
                    f"truthiness of {p} not allowed (zero is valid data)",
                    lineno,
                    0,
                )
        return
    # Optional type T | None
    if len(non_none) == 1:
        inner = non_none[0]
        # Optional[str] — ambiguous (None vs empty string)
        if inner == "str":
            lineno = expr.get("lineno", 0) if isinstance(expr, dict) else 0
            raise ParseError(
                f"ambiguous truthiness for optional str (could be None or empty)",
                lineno,
                0,
            )
        # Optional[collection] — ambiguous
        if inner.startswith("list[") or inner.startswith("dict[") or inner.startswith("set["):
            lineno = expr.get("lineno", 0) if isinstance(expr, dict) else 0
            inner_name = inner.split("[")[0]
            raise ParseError(
                f"ambiguous truthiness for optional {inner_name} (could be None or empty)",
                lineno,
                0,
            )
        # Optional[int], Optional[float] — allowed (truthiness distinguishes None)
        return
    # Multi-variant union with None — check each non-None part
    for p in non_none:
        if p == "str" or p.startswith("list[") or p.startswith("dict[") or p.startswith("set["):
            lineno = expr.get("lineno", 0) if isinstance(expr, dict) else 0
            raise ParseError(
                f"ambiguous truthiness for optional union (could be None or empty)",
                lineno,
                0,
            )


def _check_type_truthiness(typ: "Type", source_type: str | None, expr: ASTNode) -> None:
    if typ == BOOL:
        return
    if isinstance(typ, Optional):
        inner = typ.inner
        if isinstance(inner, (Slice, Map, Set)) or inner == STRING:
            lineno = expr.get("lineno", 0) if isinstance(expr, dict) else 0
            raise ParseError(
                f"ambiguous truthiness for optional {_type_name(inner)} (could be None or empty)",
                lineno,
                0,
            )
        return
    if isinstance(typ, (Slice, Map, Set)) or typ == STRING:
        return
    if typ == INT:
        lineno = expr.get("lineno", 0) if isinstance(expr, dict) else 0
        raise ParseError(
            f"truthiness of int not allowed (zero is valid data)",
            lineno,
            0,
        )
    if typ == FLOAT:
        lineno = expr.get("lineno", 0) if isinstance(expr, dict) else 0
        raise ParseError(
            f"truthiness of float not allowed (zero is valid data)",
            lineno,
            0,
        )


# ============================================================
# NARROWING EXTRACTION
# ============================================================


def _extract_narrowing(test: ASTNode, env: TypeEnv) -> tuple[TypeEnv, TypeEnv]:
    """Extract narrowing from a test expression. Returns (then_env, else_env)."""
    from .inference import is_isinstance_call, extract_isinstance_or_chain

    isinstance_check = is_isinstance_call(test)
    if isinstance_check:
        var_name, type_name = isinstance_check
        narrowed_type = _name_to_type(type_name, env)
        if narrowed_type is not None:
            then_env = env.narrow_source(var_name, narrowed_type, type_name)
            else_env = _narrow_else_isinstance(var_name, type_name, env)
            return then_env, else_env

    or_chain = extract_isinstance_or_chain(test, env.kind_to_struct)
    if or_chain:
        var_name, type_names = or_chain
        if len(type_names) == 1:
            narrowed_type = _name_to_type(type_names[0], env)
            if narrowed_type is not None:
                then_env = env.narrow_source(var_name, narrowed_type, type_names[0])
                else_env = _narrow_else_isinstance(var_name, type_names[0], env)
                return then_env, else_env

    is_none = _extract_is_none(test)
    if is_none:
        var_type = env.lookup(is_none)
        src = env.source_type(is_none)
        if isinstance(var_type, Optional):
            else_env = env.narrow_source(is_none, var_type.inner,
                                         _non_none_parts(src)[0] if src and _is_optional_source(src) else "")
            return env, else_env
        # Source type says optional but IR erased it (e.g., str | None → STRING)
        if src and _is_optional_source(src):
            non_none = _non_none_parts(src)
            if len(non_none) == 1:
                else_env = env.narrow_source(is_none, var_type, non_none[0])
                return env, else_env

    is_not_none = _extract_is_not_none(test)
    if is_not_none:
        var_type = env.lookup(is_not_none)
        src = env.source_type(is_not_none)
        if isinstance(var_type, Optional):
            narrowed_src = _non_none_parts(src)[0] if src and _is_optional_source(src) else ""
            then_env = env.narrow_source(is_not_none, var_type.inner, narrowed_src)
            return then_env, env
        # Source type says optional but IR erased it
        if src and _is_optional_source(src):
            non_none = _non_none_parts(src)
            if len(non_none) == 1:
                then_env = env.narrow_source(is_not_none, var_type, non_none[0])
                return then_env, env

    # x.attr is not None — guard the attr path for the then-branch
    attr_path = _extract_attr_is_not_none_path(test)
    if attr_path:
        then_env = env.guard_attr(attr_path)
        return then_env, env

    # x.attr is None — guard the attr path for the else-branch
    attr_path_none = _extract_attr_is_none_path(test)
    if attr_path_none:
        else_env = env.guard_attr(attr_path_none)
        return env, else_env

    # not isinstance(x, T) → then gets complement, else gets T
    if is_type(test, ["UnaryOp"]) and op_type(test.get("op")) == "Not":
        operand = test.get("operand", {})
        isinstance_check = is_isinstance_call(operand)
        if isinstance_check:
            var_name, type_name = isinstance_check
            narrowed_type = _name_to_type(type_name, env)
            if narrowed_type is not None:
                else_env = env.narrow_source(var_name, narrowed_type, type_name)
                then_env = _narrow_else_isinstance(var_name, type_name, env)
                return then_env, else_env

    # Truthiness narrows optional (both IR Optional and source optional)
    if is_type(test, ["Name"]):
        var_name = test.get("id")
        var_type = env.lookup(var_name)
        src = env.source_type(var_name)
        if isinstance(var_type, Optional):
            narrowed_src = _non_none_parts(src)[0] if src and _is_optional_source(src) else ""
            then_env = env.narrow_source(var_name, var_type.inner, narrowed_src)
            return then_env, env
        # Source says optional but IR erased
        if src and _is_optional_source(src):
            non_none = _non_none_parts(src)
            if len(non_none) == 1:
                then_env = env.narrow_source(var_name, var_type, non_none[0])
                return then_env, env

    # Walrus operator: if (val := expr) narrows val based on expr's return type
    if is_type(test, ["NamedExpr"]):
        target = test.get("target", {})
        if is_type(target, ["Name"]):
            var_name = target.get("id")
            val = test.get("value", {})
            val_type = _expr_type(val, env)
            if isinstance(val_type, Optional):
                then_env = env.narrow_source(var_name, val_type.inner, "")
                return then_env, env
            # Check if the call's return py_type is optional
            if is_type(val, ["Call"]):
                ret_src = _call_return_source_type(val, env)
                if ret_src and _is_optional_source(ret_src):
                    non_none = _non_none_parts(ret_src)
                    if len(non_none) == 1:
                        narrowed_type = _name_to_type(non_none[0], env)
                        if narrowed_type is not None:
                            then_env = env.narrow_source(var_name, narrowed_type, non_none[0])
                            return then_env, env
                        # If type can't be resolved to IR, still narrow source
                        then_env = env.narrow_source(var_name, val_type, non_none[0])
                        return then_env, env

    # Compound AND: propagate narrowing through
    if is_type(test, ["BoolOp"]) and is_type(test.get("op"), ["And"]):
        values = test.get("values", [])
        current_then = env
        for val in values:
            then_sub, _ = _extract_narrowing(val, current_then)
            current_then = then_sub
        return current_then, env

    # Kind check narrowing: x.kind == "value"
    if _is_kind_check(test, env):
        _validate_kind_value(test, env)
        return env, env  # Kind checks are allowed, narrowing handled by inference

    return env, env


def _call_return_source_type(call: ASTNode, env: TypeEnv) -> str | None:
    """Get the return py_type of a function call."""
    func = call.get("func", {})
    if is_type(func, ["Name"]):
        func_name = func.get("id")
        if func_name in env.symbols.functions:
            return env.symbols.functions[func_name].return_py_type
    return None


def _narrow_else_isinstance(var_name: str, type_name: str, env: TypeEnv) -> TypeEnv:
    """Narrow else-branch after isinstance check."""
    src = env.source_type(var_name)
    if src and _is_union_source(src):
        parts = _split_union_parts(src)
        remaining = [p for p in parts if p != type_name]
        if len(remaining) == 1:
            narrowed_type = _name_to_type(remaining[0], env)
            if narrowed_type is not None:
                return env.narrow_source(var_name, narrowed_type, remaining[0])
            if remaining[0] == "None":
                return env  # narrowed to None — just don't narrow IR
        elif len(remaining) > 1:
            # Still a union, narrow source type
            new_src = " | ".join(remaining)
            return env.narrow_source(var_name, env.lookup(var_name), new_src)
    return env


def _is_kind_check(test: ASTNode, env: TypeEnv) -> bool:
    """Check if test is a kind-based discrimination: x.kind == "value"."""
    if not is_type(test, ["Compare"]):
        return False
    ops = test.get("ops", [])
    if len(ops) != 1 or not is_type(ops[0], ["Eq"]):
        return False
    left = test.get("left", {})
    if is_type(left, ["Attribute"]) and left.get("attr") == "kind":
        return True
    return False


def _validate_kind_value(test: ASTNode, env: TypeEnv) -> None:
    """Validate that a kind check value maps to a known struct."""
    comparators = test.get("comparators", [])
    if not comparators:
        return
    right = comparators[0]
    if not (is_type(right, ["Constant"]) and isinstance(right.get("value"), str)):
        return
    kind_value = right.get("value")
    # Check against hierarchy kind_to_struct
    if env.kind_to_struct and kind_value not in env.kind_to_struct:
        lineno = test.get("lineno", 0)
        raise ParseError(
            f"kind value '{kind_value}' does not match any known type",
            lineno,
            0,
        )
        return
    # Check against source type union members
    left = test.get("left", {})
    if is_type(left, ["Attribute"]) and left.get("attr") == "kind":
        obj = left.get("value", {})
        if is_type(obj, ["Name"]):
            src = env.source_type(obj.get("id", ""))
            if src and _is_union_source(src):
                parts = _non_none_parts(src)
                valid_kinds = {p.lower() for p in parts}
                if kind_value not in valid_kinds:
                    lineno = test.get("lineno", 0)
                    raise ParseError(
                        f"kind value '{kind_value}' does not match any union member",
                        lineno,
                        0,
                    )


def _extract_is_none(test: ASTNode) -> str | None:
    if not is_type(test, ["Compare"]):
        return None
    ops = test.get("ops", [])
    if len(ops) != 1 or not is_type(ops[0], ["Is"]):
        return None
    left = test.get("left", {})
    comparators = test.get("comparators", [])
    if not comparators:
        return None
    right = comparators[0]
    if is_type(left, ["Name"]) and _is_none_literal(right):
        return left.get("id")
    return None


def _extract_is_not_none(test: ASTNode) -> str | None:
    if not is_type(test, ["Compare"]):
        return None
    ops = test.get("ops", [])
    if len(ops) != 1 or not is_type(ops[0], ["IsNot"]):
        return None
    left = test.get("left", {})
    comparators = test.get("comparators", [])
    if not comparators:
        return None
    right = comparators[0]
    if is_type(left, ["Name"]) and _is_none_literal(right):
        return left.get("id")
    return None


def _extract_attr_is_not_none_path(test: ASTNode) -> str | None:
    """Extract 'x.attr' path from 'x.attr is not None'."""
    if not is_type(test, ["Compare"]):
        return None
    ops = test.get("ops", [])
    if len(ops) != 1 or not is_type(ops[0], ["IsNot"]):
        return None
    left = test.get("left", {})
    comparators = test.get("comparators", [])
    if not comparators:
        return None
    if is_type(left, ["Attribute"]) and _is_none_literal(comparators[0]):
        return _attr_path(left)
    return None


def _extract_attr_is_none_path(test: ASTNode) -> str | None:
    """Extract 'x.attr' path from 'x.attr is None'."""
    if not is_type(test, ["Compare"]):
        return None
    ops = test.get("ops", [])
    if len(ops) != 1 or not is_type(ops[0], ["Is"]):
        return None
    left = test.get("left", {})
    comparators = test.get("comparators", [])
    if not comparators:
        return None
    if is_type(left, ["Attribute"]) and _is_none_literal(comparators[0]):
        return _attr_path(left)
    return None


def _attr_path(node: ASTNode) -> str:
    """Build a dot-separated attribute path like 'n.next'."""
    if is_type(node, ["Name"]):
        return node.get("id", "")
    if is_type(node, ["Attribute"]):
        base = _attr_path(node.get("value", {}))
        return f"{base}.{node.get('attr', '')}" if base else node.get("attr", "")
    return ""


# ============================================================
# ITERATOR / GENERATOR ESCAPE
# ============================================================

_ITERATOR_FUNCS = frozenset({"enumerate", "zip", "reversed"})


def _is_iterator_call(expr: ASTNode) -> str | None:
    if not is_type(expr, ["Call"]):
        return None
    func = expr.get("func", {})
    if is_type(func, ["Name"]) and func.get("id") in _ITERATOR_FUNCS:
        return func.get("id")
    return None


def _is_generator_expr(expr: ASTNode) -> bool:
    return is_type(expr, ["GeneratorExp"])


def _is_eager_consumer_call(parent_func_name: str) -> bool:
    return parent_func_name in _EAGER_CONSUMERS


def _check_iterator_escape_assign(value: ASTNode, env: TypeEnv, stmt: ASTNode) -> None:
    iter_name = _is_iterator_call(value)
    if iter_name and iter_name != "range":
        lineno = stmt.get("lineno", 0)
        raise ParseError(
            f"cannot assign {iter_name}() to variable (single-use iterator)",
            lineno,
            0,
        )
    if _is_generator_expr(value):
        lineno = stmt.get("lineno", 0)
        raise ParseError(
            f"cannot assign generator expression to variable",
            lineno,
            0,
        )


def _check_iterator_escape_return(value: ASTNode, env: TypeEnv) -> None:
    # Allow if the return value is wrapped in an eager consumer (e.g., tuple(enumerate(xs)))
    if _is_eager_consumer_wrapping(value):
        return
    iter_name = _is_iterator_call(value)
    if iter_name and iter_name != "range":
        lineno = value.get("lineno", 0)
        raise ParseError(
            f"cannot return {iter_name}() (single-use iterator)",
            lineno,
            0,
        )
    if _is_generator_expr(value):
        lineno = value.get("lineno", 0)
        raise ParseError(
            f"cannot return generator expression",
            lineno,
            0,
        )


def _is_eager_consumer_wrapping(expr: ASTNode) -> bool:
    """Check if expr is an eager consumer call wrapping iterators/generators."""
    if not is_type(expr, ["Call"]):
        return False
    func = expr.get("func", {})
    if is_type(func, ["Name"]) and func.get("id") in _EAGER_CONSUMERS:
        return True
    return False


def _check_iterator_escape_expr_stmt(value: ASTNode, env: TypeEnv) -> None:
    if not is_type(value, ["Call"]):
        return
    func = value.get("func", {})
    args = value.get("args", [])
    caller_name = None
    if is_type(func, ["Name"]):
        caller_name = func.get("id")
    elif is_type(func, ["Attribute"]):
        caller_name = func.get("attr")
    for arg in args:
        iter_name = _is_iterator_call(arg)
        if iter_name and iter_name != "range":
            if caller_name and _is_eager_consumer_call(caller_name):
                continue
            if caller_name == "join":
                continue
            lineno = value.get("lineno", 0)
            raise ParseError(
                f"cannot pass {iter_name}() to non-consumer function (single-use iterator)",
                lineno,
                0,
            )
        if _is_generator_expr(arg):
            if caller_name and _is_eager_consumer_call(caller_name):
                continue
            if caller_name == "join":
                continue
            lineno = value.get("lineno", 0)
            raise ParseError(
                f"cannot pass generator expression to non-consumer function",
                lineno,
                0,
            )


# ============================================================
# TYPE HELPERS
# ============================================================


def _expr_type(expr: ASTNode, env: TypeEnv) -> "Type":
    """Get the type of an expression."""
    if not isinstance(expr, dict):
        return InterfaceRef("any")
    node_t = expr.get("_type")
    # Name nodes always go through env.lookup for flow-sensitive narrowing
    if node_t == "Name":
        return env.lookup(expr.get("id", ""))
    if node_t == "Constant":
        value = expr.get("value")
        if isinstance(value, bool):
            return BOOL
        if isinstance(value, int):
            return INT
        if isinstance(value, str):
            return STRING
        if isinstance(value, float):
            return FLOAT
        if value is None:
            return InterfaceRef("any")
    stored = expr.get("_expr_type")
    if stored is not None:
        return stored
    if node_t == "List":
        elts = expr.get("elts", [])
        if elts:
            return Slice(_expr_type(elts[0], env))
        return Slice(InterfaceRef("any"))
    if node_t == "Tuple":
        elts = expr.get("elts", [])
        return Tuple(tuple(_expr_type(e, env) for e in elts))
    if node_t == "Dict":
        keys = expr.get("keys", [])
        values = expr.get("values", [])
        if keys and isinstance(keys[0], dict):
            key_type = _expr_type(keys[0], env)
            val_type = _expr_type(values[0], env) if values and isinstance(values[0], dict) else InterfaceRef("any")
            return Map(key_type, val_type)
        return Map(InterfaceRef("any"), InterfaceRef("any"))
    if node_t == "NamedExpr":
        return _expr_type(expr.get("value", {}), env)
    return InterfaceRef("any")


def _is_none_literal(node: ASTNode) -> bool:
    if not isinstance(node, dict):
        return False
    return is_type(node, ["Constant"]) and node.get("value") is None


def _name_to_type(type_name: str, env: TypeEnv) -> "Type | None":
    if type_name == "int":
        return INT
    if type_name == "str":
        return STRING
    if type_name == "bool":
        return BOOL
    if type_name == "float":
        return FLOAT
    if type_name in env.symbols.structs:
        return Pointer(StructRef(type_name))
    return None


def _is_assignable(actual: "Type", expected: "Type", env: TypeEnv) -> bool:
    if actual == expected:
        return True
    if expected == InterfaceRef("any"):
        return True
    if actual == InterfaceRef("any"):
        return True
    if actual == BOOL and expected == INT:
        return True
    if actual == INT and expected == FLOAT:
        return True
    if actual == BOOL and expected == FLOAT:
        return True
    if isinstance(expected, Optional):
        if _is_assignable(actual, expected.inner, env):
            return True
    if isinstance(actual, Pointer) and isinstance(actual.target, StructRef):
        struct_name = actual.target.name
        if isinstance(expected, InterfaceRef):
            if expected.name == env.hierarchy_root and struct_name in env.node_types:
                return True
    if isinstance(actual, InterfaceRef) and isinstance(expected, InterfaceRef):
        if actual.name == expected.name:
            return True
    if isinstance(actual, Slice) and isinstance(expected, Slice):
        return _is_assignable(actual.element, expected.element, env)
    if isinstance(actual, Map) and isinstance(expected, Map):
        return _is_assignable(actual.key, expected.key, env) and _is_assignable(
            actual.value, expected.value, env
        )
    if isinstance(actual, Set) and isinstance(expected, Set):
        return _is_assignable(actual.element, expected.element, env)
    if isinstance(actual, Tuple) and isinstance(expected, Tuple):
        if actual.variadic and expected.variadic:
            if len(actual.elements) > 0 and len(expected.elements) > 0:
                return _is_assignable(actual.elements[0], expected.elements[0], env)
            return True
        if not actual.variadic and expected.variadic and len(expected.elements) == 1:
            if not actual.elements:
                return True
            return all(
                _is_assignable(a, expected.elements[0], env) for a in actual.elements
            )
        if actual.variadic != expected.variadic:
            return False
        if len(actual.elements) != len(expected.elements):
            return False
        return all(
            _is_assignable(a, e, env) for a, e in zip(actual.elements, expected.elements)
        )
    if isinstance(actual, FuncType) and isinstance(expected, FuncType):
        return _callable_assignable(actual, expected)
    if isinstance(actual, StructRef) and isinstance(expected, InterfaceRef):
        if expected.name == env.hierarchy_root and actual.name in env.node_types:
            return True
    # Pointer[StructA] assignable to Pointer[StructB] if both are in node hierarchy
    if (
        isinstance(actual, Pointer)
        and isinstance(expected, Pointer)
        and isinstance(actual.target, StructRef)
        and isinstance(expected.target, StructRef)
    ):
        if (
            actual.target.name in env.node_types
            and expected.target.name in env.node_types
        ):
            return True
        if expected.target.name == env.hierarchy_root and actual.target.name in env.node_types:
            return True
    return False


def _type_name(typ: "Type") -> str:
    if typ == STRING:
        return "str"
    if typ == INT:
        return "int"
    if typ == BOOL:
        return "bool"
    if typ == FLOAT:
        return "float"
    if typ == VOID:
        return "None"
    if typ == BYTE:
        return "byte"
    if isinstance(typ, Slice):
        return f"list[{_type_name(typ.element)}]"
    if isinstance(typ, Map):
        return f"dict[{_type_name(typ.key)}, {_type_name(typ.value)}]"
    if isinstance(typ, Set):
        return f"set[{_type_name(typ.element)}]"
    if isinstance(typ, Tuple):
        if typ.variadic and typ.elements:
            return f"tuple[{_type_name(typ.elements[0])}, ...]"
        if not typ.elements:
            return "tuple[()]"
        return f"tuple[{', '.join(_type_name(e) for e in typ.elements)}]"
    if isinstance(typ, Optional):
        return f"{_type_name(typ.inner)} | None"
    if isinstance(typ, Pointer) and isinstance(typ.target, StructRef):
        return typ.target.name
    if isinstance(typ, StructRef):
        return typ.name
    if isinstance(typ, InterfaceRef):
        if typ.name == "any":
            return "object"
        return typ.name
    if isinstance(typ, FuncType):
        params = ", ".join(_type_name(p) for p in typ.params)
        return f"Callable[[{params}], {_type_name(typ.ret)}]"
    return str(typ)
