"""Phase 9: Type inference and validation.

Bidirectional type inference with flow-sensitive narrowing. Computes types for
all expressions, infers local variable types from assignments, enforces type
safety constraints, and validates iterator/generator consumption.

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from __future__ import annotations


from .signatures import (
    FuncInfo,
    SignatureResult,
    annotation_to_str,
    py_type_to_type_dict,
    SignatureError,
)
from .fields import FieldResult
from .hierarchy import HierarchyResult
from .types import (
    TypeNode,
    PrimitiveType,
    SliceType,
    MapType,
    SetType,
    TupleType,
    OptionalType,
    PointerType,
    StructRef,
    InterfaceRef,
    FuncType,
    ANY_TYPE,
    INT_TYPE,
    FLOAT_TYPE,
    BOOL_TYPE,
    STR_TYPE,
    VOID_TYPE,
    BYTES_TYPE,
    is_any,
    type_name as _type_name_fn,
)

# Type alias for AST dict nodes
ASTNode = dict[str, object]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class InferenceError:
    """An error found during inference."""

    def __init__(
        self, lineno: int, col: int, message: str, source_file: str = ""
    ) -> None:
        self.lineno: int = lineno
        self.col: int = col
        self.message: str = message
        self.source_file: str = source_file

    def __repr__(self) -> str:
        file_prefix = ""
        if self.source_file != "":
            file_prefix = self.source_file + ":"
        return (
            file_prefix
            + "error:"
            + str(self.lineno)
            + ":"
            + str(self.col)
            + ": [inference] "
            + self.message
        )


class InferenceResult:
    """Result of inference analysis."""

    def __init__(self) -> None:
        self._errors: list[InferenceError] = []

    def add_error(
        self, lineno: int, col: int, message: str, source_file: str = ""
    ) -> None:
        self._errors.append(InferenceError(lineno, col, message, source_file))

    def errors(self) -> list[InferenceError]:
        return self._errors


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def _is_type(node: object, type_names: list[str]) -> bool:
    if not isinstance(node, dict):
        return False
    t = node.get("_type")
    i = 0
    while i < len(type_names):
        if t == type_names[i]:
            return True
        i += 1
    return False


def _is_bytes_type(t: TypeNode) -> bool:
    """Check if t represents bytes (either PrimitiveType("bytes") or SliceType(byte))."""
    if isinstance(t, PrimitiveType) and t.kind == "bytes":
        return True
    if isinstance(t, SliceType):
        if isinstance(t.element, PrimitiveType) and t.element.kind == "byte":
            return True
    return False


def _type_eq(a: TypeNode, b: TypeNode) -> bool:
    """Check structural equality of two TypeNodes."""
    if _is_bytes_type(a) and _is_bytes_type(b):
        return True
    if isinstance(a, PrimitiveType) and isinstance(b, PrimitiveType):
        return a.kind == b.kind
    if isinstance(a, PrimitiveType) or isinstance(b, PrimitiveType):
        return False
    if isinstance(a, SliceType) and isinstance(b, SliceType):
        return _type_eq(a.element, b.element)
    if isinstance(a, MapType) and isinstance(b, MapType):
        return _type_eq(a.key, b.key) and _type_eq(a.value, b.value)
    if isinstance(a, SetType) and isinstance(b, SetType):
        return _type_eq(a.element, b.element)
    if isinstance(a, OptionalType) and isinstance(b, OptionalType):
        return _type_eq(a.inner, b.inner)
    if isinstance(a, TupleType) and isinstance(b, TupleType):
        if a.variadic != b.variadic:
            return False
        if len(a.elements) != len(b.elements):
            return False
        j = 0
        while j < len(a.elements):
            if not _type_eq(a.elements[j], b.elements[j]):
                return False
            j += 1
        return True
    if isinstance(a, PointerType) and isinstance(b, PointerType):
        return _type_eq(a.target, b.target)
    if isinstance(a, StructRef) and isinstance(b, StructRef):
        return a.name == b.name
    if isinstance(a, InterfaceRef) and isinstance(b, InterfaceRef):
        return a.name == b.name
    if isinstance(a, FuncType) and isinstance(b, FuncType):
        if len(a.params) != len(b.params):
            return False
        j = 0
        while j < len(a.params):
            if not _type_eq(a.params[j], b.params[j]):
                return False
            j += 1
        return _type_eq(a.ret, b.ret)
    return a == b


def _type_name(t: TypeNode) -> str:
    return _type_name_fn(t)


# ---------------------------------------------------------------------------
# Assignability
# ---------------------------------------------------------------------------


def _is_assignable(
    actual: TypeNode,
    expected: TypeNode,
    hier: HierarchyResult,
) -> bool:
    """Check if actual is assignable to expected."""
    if _type_eq(actual, expected):
        return True
    if is_any(actual) or is_any(expected):
        return True
    # void (None literal) assignable to Optional
    if isinstance(actual, PrimitiveType) and actual.kind == "void":
        if isinstance(expected, OptionalType):
            return True
        if isinstance(expected, InterfaceRef):
            return True
        return False
    # bool <: int <: float
    if isinstance(actual, PrimitiveType) and isinstance(expected, PrimitiveType):
        if actual.kind == "bool" and expected.kind == "int":
            return True
        if actual.kind == "bool" and expected.kind == "float":
            return True
        if actual.kind == "int" and expected.kind == "float":
            return True
    # T assignable to T?
    if isinstance(expected, OptionalType):
        if _is_assignable(actual, expected.inner, hier):
            return True
    # Collection assignable to Pointer(collection) (mutated param)
    if isinstance(expected, PointerType):
        if not isinstance(expected.target, StructRef):
            if _is_assignable(actual, expected.target, hier):
                return True
    # Pointer(collection) assignable to plain collection (forwarding mutated param)
    if isinstance(actual, PointerType):
        if not isinstance(actual.target, StructRef):
            if _is_assignable(actual.target, expected, hier):
                return True
    # Struct hierarchy: subclass assignable to base / interface
    a_name = _struct_name(actual)
    e_name = _struct_name(expected)
    if isinstance(actual, PointerType) and isinstance(actual.target, StructRef):
        if isinstance(expected, PointerType) and isinstance(expected.target, StructRef):
            if a_name == e_name:
                return True
            if _is_ancestor(a_name, e_name, hier):
                return True
        if isinstance(expected, InterfaceRef):
            if hier.is_node(a_name):
                return True
    # StructRef without Pointer wrapper
    if isinstance(actual, StructRef):
        if isinstance(expected, InterfaceRef):
            if hier.is_node(a_name):
                return True
        if isinstance(expected, StructRef):
            if a_name == e_name:
                return True
            if _is_ancestor(a_name, e_name, hier):
                return True
        if isinstance(expected, PointerType) and isinstance(expected.target, StructRef):
            if a_name == e_name:
                return True
            if _is_ancestor(a_name, e_name, hier):
                return True
    # Collection element assignability (invariant but check same direction)
    if isinstance(actual, SliceType) and isinstance(expected, SliceType):
        return _is_assignable(actual.element, expected.element, hier)
    if isinstance(actual, MapType) and isinstance(expected, MapType):
        return _is_assignable(actual.key, expected.key, hier) and _is_assignable(
            actual.value, expected.value, hier
        )
    if isinstance(actual, SetType) and isinstance(expected, SetType):
        return _is_assignable(actual.element, expected.element, hier)
    # Tuple assignability
    if isinstance(actual, TupleType) and isinstance(expected, TupleType):
        if actual.variadic and expected.variadic:
            if len(actual.elements) > 0 and len(expected.elements) > 0:
                return _is_assignable(actual.elements[0], expected.elements[0], hier)
        if not actual.variadic and expected.variadic:
            if len(expected.elements) > 0:
                be = expected.elements[0]
                j = 0
                while j < len(actual.elements):
                    if not _is_assignable(actual.elements[j], be, hier):
                        return False
                    j += 1
                return True
        if not actual.variadic and not expected.variadic:
            if len(actual.elements) != len(expected.elements):
                return False
            j = 0
            while j < len(actual.elements):
                if not _is_assignable(actual.elements[j], expected.elements[j], hier):
                    return False
                j += 1
            return True
    # FuncType assignability
    if isinstance(actual, FuncType) and isinstance(expected, FuncType):
        if len(actual.params) != len(expected.params):
            return False
        j = 0
        while j < len(actual.params):
            if not _is_assignable(expected.params[j], actual.params[j], hier):
                return False
            j += 1
        return _is_assignable(actual.ret, expected.ret, hier)
    return False


def _is_ancestor(child: str, ancestor: str, hier: HierarchyResult) -> bool:
    """Check if ancestor is transitively a base of child."""
    visited: set[str] = set()
    current: str = child
    while True:
        if current in visited:
            return False
        visited.add(current)
        bases = hier.ancestors.get(current)
        if bases is None or len(bases) == 0:
            return False
        j = 0
        while j < len(bases):
            if bases[j] == ancestor:
                return True
            j += 1
        current = bases[0]


def _struct_name(t: TypeNode) -> str:
    """Extract struct name from Pointer(StructRef(name)) or StructRef(name)."""
    if isinstance(t, PointerType) and isinstance(t.target, StructRef):
        return t.target.name
    if isinstance(t, StructRef):
        return t.name
    return ""


# ---------------------------------------------------------------------------
# Source type tracking
# ---------------------------------------------------------------------------


def _split_union_parts(py_type: str) -> list[str]:
    """Split 'int | str | None' into ['int', 'str', 'None']."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(py_type):
        c = py_type[i]
        if c == "[":
            depth += 1
            current.append(c)
        elif c == "]":
            depth -= 1
            current.append(c)
        elif (
            c == " "
            and depth == 0
            and i + 2 < len(py_type)
            and py_type[i + 1] == "|"
            and py_type[i + 2] == " "
        ):
            parts.append("".join(current).strip())
            current: list[str] = []
            i += 3
            continue
        else:
            current.append(c)
        i += 1
    tail = "".join(current).strip()
    if tail != "":
        parts.append(tail)
    return parts


def _is_union_source(py_type: str) -> bool:
    parts = _split_union_parts(py_type)
    return len(parts) > 1


def _is_optional_source(py_type: str) -> bool:
    parts = _split_union_parts(py_type)
    if len(parts) < 2:
        return False
    i = 0
    while i < len(parts):
        if parts[i] == "None":
            return True
        i += 1
    return False


def _non_none_parts(py_type: str) -> list[str]:
    parts = _split_union_parts(py_type)
    result: list[str] = []
    i = 0
    while i < len(parts):
        if parts[i] != "None":
            result.append(parts[i])
        i += 1
    return result


def _needs_narrowing(py_type: str) -> bool:
    return _is_union_source(py_type) or py_type == "object"


# ---------------------------------------------------------------------------
# Type environment
# ---------------------------------------------------------------------------


class TypeEnv:
    """Flow-sensitive type environment for a function body."""

    def __init__(self) -> None:
        self.types: dict[str, TypeNode] = {}
        self.source_types: dict[str, str] = {}
        self.guarded_attrs: set[str] = set()

    def copy(self) -> TypeEnv:
        env = TypeEnv()
        tkeys = list(self.types.keys())
        i = 0
        while i < len(tkeys):
            env.types[tkeys[i]] = self.types[tkeys[i]]
            i += 1
        skeys = list(self.source_types.keys())
        i = 0
        while i < len(skeys):
            env.source_types[skeys[i]] = self.source_types[skeys[i]]
            i += 1
        gkeys = list(self.guarded_attrs)
        i = 0
        while i < len(gkeys):
            env.guarded_attrs.add(gkeys[i])
            i += 1
        return env

    def set(self, name: str, typ: TypeNode, source: str) -> None:
        self.types[name] = typ
        self.source_types[name] = source

    def get_type(self, name: str) -> TypeNode | None:
        return self.types.get(name)

    def get_source(self, name: str) -> str:
        return self.source_types.get(name, "")

    def narrow(self, name: str, typ: TypeNode, source: str) -> None:
        self.types[name] = typ
        self.source_types[name] = source

    def guard_attr(self, path: str) -> None:
        self.guarded_attrs.add(path)

    def is_attr_guarded(self, path: str) -> bool:
        return path in self.guarded_attrs


# ---------------------------------------------------------------------------
# Expression type synthesis
# ---------------------------------------------------------------------------

_EAGER_CONSUMERS: set[str] = {
    "list",
    "tuple",
    "set",
    "dict",
    "frozenset",
    "sum",
    "min",
    "max",
    "any",
    "all",
    "sorted",
}

_ITERATOR_FUNCS: set[str] = {"enumerate", "zip", "reversed"}


def _synth_expr(
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
) -> TypeNode:
    """Synthesize the type of an expression node."""
    if not isinstance(node, dict):
        return ANY_TYPE
    t = node.get("_type")
    if t == "Constant":
        return _synth_constant(node)
    if t == "Name":
        return _synth_name(node, env, ctx)
    if t == "Attribute":
        return _synth_attribute(node, env, ctx)
    if t == "Call":
        return _synth_call(node, env, ctx)
    if t == "Subscript":
        return _synth_subscript(node, env, ctx)
    if t == "BinOp":
        return _synth_binop(node, env, ctx)
    if t == "UnaryOp":
        return _synth_unaryop(node, env, ctx)
    if t == "Compare":
        return BOOL_TYPE
    if t == "BoolOp":
        return _synth_boolop(node, env, ctx)
    if t == "IfExp":
        return _synth_ifexp(node, env, ctx)
    if t == "List":
        return _synth_list(node, env, ctx)
    if t == "Dict":
        return _synth_dict(node, env, ctx)
    if t == "Set":
        return _synth_set(node, env, ctx)
    if t == "Tuple":
        return _synth_tuple(node, env, ctx)
    if t == "ListComp":
        return _synth_listcomp(node, env, ctx)
    if t == "SetComp":
        return _synth_setcomp(node, env, ctx)
    if t == "DictComp":
        return _synth_dictcomp(node, env, ctx)
    if t == "GeneratorExp":
        return ANY_TYPE
    if t == "JoinedStr":
        return STR_TYPE
    if t == "FormattedValue":
        return STR_TYPE
    if t == "NamedExpr":
        return _synth_namedexpr(node, env, ctx)
    if t == "Starred":
        return ANY_TYPE
    return ANY_TYPE


def _synth_constant(node: ASTNode) -> TypeNode:
    v = node.get("value")
    if v is None:
        return VOID_TYPE
    if isinstance(v, bool):
        return BOOL_TYPE
    if isinstance(v, int):
        return INT_TYPE
    if isinstance(v, float):
        return FLOAT_TYPE
    if isinstance(v, str):
        return STR_TYPE
    if isinstance(v, bytes):
        return BYTES_TYPE
    return ANY_TYPE


def _synth_name(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    name = node.get("id")
    if not isinstance(name, str):
        return ANY_TYPE
    typ = env.get_type(name)
    if typ is not None:
        return typ
    # User-defined function reference -> FuncType
    func_info = ctx.sig_result.functions.get(name)
    if func_info is not None:
        params: list[TypeNode] = []
        j = 0
        while j < len(func_info.params):
            params.append(func_info.params[j].typ)
            j += 1
        return FuncType(params, func_info.return_type)
    # Builtin function references
    if name == "len":
        return FuncType([ANY_TYPE], INT_TYPE)
    if name == "str":
        return FuncType([ANY_TYPE], STR_TYPE)
    if name == "int":
        return FuncType([ANY_TYPE], INT_TYPE)
    if name == "bool":
        return FuncType([ANY_TYPE], BOOL_TYPE)
    # Module-level variable
    mod_var = ctx.module_vars.get(name)
    if mod_var is not None:
        return mod_var
    return ANY_TYPE


def _synth_attribute(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    value = node.get("value")
    attr = node.get("attr")
    if not isinstance(value, dict) or not isinstance(attr, str):
        return ANY_TYPE
    path = _attr_path(node)
    if path != "":
        narrowed = env.get_type(path)
        if narrowed is not None:
            return narrowed
    obj_type = _synth_expr(value, env, ctx)
    result = _resolve_attr(obj_type, attr, value, env, ctx)
    if isinstance(result, OptionalType):
        if path != "" and env.is_attr_guarded(path):
            return result.inner
    return result


def _resolve_attr(
    obj_type: TypeNode,
    attr: str,
    value_node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
) -> TypeNode:
    """Resolve attribute access on a type."""
    # Unwrap Pointer to collection (not struct)
    if isinstance(obj_type, PointerType) and not isinstance(obj_type.target, StructRef):
        obj_type = obj_type.target
    # String methods
    if isinstance(obj_type, PrimitiveType) and obj_type.kind == "string":
        if (
            attr == "upper"
            or attr == "lower"
            or attr == "strip"
            or attr == "lstrip"
            or attr == "rstrip"
        ):
            return FuncType([], STR_TYPE)
        if attr == "split":
            return FuncType([], SliceType(STR_TYPE))
        if attr == "join":
            return FuncType([SliceType(STR_TYPE)], STR_TYPE)
        if attr == "replace" or attr == "format":
            return FuncType([STR_TYPE], STR_TYPE)
        if attr == "startswith" or attr == "endswith":
            return FuncType([STR_TYPE], BOOL_TYPE)
        if attr == "find" or attr == "index" or attr == "count":
            return FuncType([STR_TYPE], INT_TYPE)
        return ANY_TYPE
    # List methods
    if isinstance(obj_type, SliceType):
        elem = obj_type.element
        if attr == "append":
            return FuncType([elem], VOID_TYPE)
        if attr == "extend":
            return FuncType([SliceType(elem)], VOID_TYPE)
        if attr == "insert":
            return FuncType([INT_TYPE, elem], VOID_TYPE)
        if attr == "pop":
            return FuncType([], elem)
        if attr == "copy":
            return FuncType([], obj_type)
        if attr == "sort":
            return FuncType([], VOID_TYPE)
        if attr == "reverse":
            return FuncType([], VOID_TYPE)
        if attr == "clear":
            return FuncType([], VOID_TYPE)
        if attr == "count":
            return FuncType([elem], INT_TYPE)
        if attr == "index":
            return FuncType([elem], INT_TYPE)
        if attr == "remove":
            return FuncType([elem], VOID_TYPE)
        return ANY_TYPE
    # Dict methods
    if isinstance(obj_type, MapType):
        key_t = obj_type.key
        val_t = obj_type.value
        if attr == "get":
            return FuncType([key_t], OptionalType(val_t))
        if attr == "keys":
            return FuncType([], SliceType(key_t))
        if attr == "values":
            return FuncType([], SliceType(val_t))
        if attr == "items":
            items_elem = TupleType([key_t, val_t], False)
            return FuncType([], SliceType(items_elem))
        if attr == "pop":
            return FuncType([key_t], val_t)
        if attr == "update":
            return FuncType([obj_type], VOID_TYPE)
        if attr == "copy":
            return FuncType([], obj_type)
        if attr == "clear":
            return FuncType([], VOID_TYPE)
        if attr == "setdefault":
            return FuncType([key_t, val_t], val_t)
        return ANY_TYPE
    # Set methods
    if isinstance(obj_type, SetType):
        elem = obj_type.element
        if attr == "add":
            return FuncType([elem], VOID_TYPE)
        if attr == "remove" or attr == "discard":
            return FuncType([elem], VOID_TYPE)
        if attr == "union" or attr == "intersection" or attr == "difference":
            return FuncType([obj_type], obj_type)
        if attr == "copy":
            return FuncType([], obj_type)
        if attr == "clear":
            return FuncType([], VOID_TYPE)
        return ANY_TYPE
    # Struct field access
    sname = _struct_name(obj_type)
    if sname != "":
        return _resolve_struct_attr(sname, attr, ctx)
    # Optional: error handled by validator
    if isinstance(obj_type, OptionalType):
        return _resolve_attr(obj_type.inner, attr, value_node, env, ctx)
    return ANY_TYPE


def _resolve_struct_attr(sname: str, attr: str, ctx: _InferCtx) -> TypeNode:
    """Resolve attribute on a struct type."""
    cls = ctx.field_result.classes.get(sname)
    if cls is not None:
        fld = cls.fields.get(attr)
        if fld is not None:
            return fld.typ
        if attr in cls.const_fields:
            return STR_TYPE
    methods = ctx.sig_result.methods.get(sname)
    if methods is not None:
        method = methods.get(attr)
        if method is not None:
            params: list[TypeNode] = []
            j = 0
            while j < len(method.params):
                if method.params[j].name != "self":
                    params.append(method.params[j].typ)
                j += 1
            return FuncType(params, method.return_type)
    return ANY_TYPE


def _synth_call(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    """Synthesize return type of a call."""
    func = node.get("func")
    if not isinstance(func, dict):
        return ANY_TYPE
    args = node.get("args", [])
    if not isinstance(args, list):
        args: list[ASTNode] = []
    # Direct name call
    if _is_type(func, ["Name"]):
        fname = func.get("id")
        if isinstance(fname, str):
            return _synth_name_call(fname, args, node, env, ctx)
    # Method call
    if _is_type(func, ["Attribute"]):
        return _synth_method_call(func, args, node, env, ctx)
    # Callable variable
    func_type = _synth_expr(func, env, ctx)
    if isinstance(func_type, FuncType):
        return func_type.ret
    return ANY_TYPE


def _synth_name_call(
    fname: str,
    args: list[object],
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
) -> TypeNode:
    """Synthesize return type for a direct function call."""
    if fname == "int":
        return INT_TYPE
    if fname == "float":
        return FLOAT_TYPE
    if fname == "str":
        return STR_TYPE
    if fname == "bool":
        return BOOL_TYPE
    if fname == "len":
        return INT_TYPE
    if fname == "abs":
        return INT_TYPE
    if fname == "ord":
        return INT_TYPE
    if fname == "chr":
        return STR_TYPE
    if fname == "repr":
        return STR_TYPE
    if fname == "round":
        return INT_TYPE
    if fname == "sum":
        return INT_TYPE
    if fname == "min" or fname == "max":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                if isinstance(ft, SliceType):
                    return ft.element
                if len(args) >= 2:
                    has_int = False
                    has_bool = False
                    j = 0
                    while j < len(args):
                        a = args[j]
                        if isinstance(a, dict):
                            at = _synth_expr(a, env, ctx)
                            if isinstance(at, PrimitiveType) and at.kind == "int":
                                has_int = True
                            if isinstance(at, PrimitiveType) and at.kind == "bool":
                                has_bool = True
                        j += 1
                    if has_int or has_bool:
                        return INT_TYPE
                return ft
        return INT_TYPE
    if fname == "isinstance":
        return BOOL_TYPE
    if fname == "hash":
        return INT_TYPE
    if fname == "range":
        return ANY_TYPE
    if fname == "enumerate":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                elem = _element_type(ft)
                return {
                    "_type": "_Iterator",
                    "element": TupleType([INT_TYPE, elem], False),
                    "source": "enumerate",
                }
        return {
            "_type": "_Iterator",
            "element": TupleType([INT_TYPE, ANY_TYPE], False),
            "source": "enumerate",
        }
    if fname == "zip":
        elems: list[TypeNode] = []
        j = 0
        while j < len(args):
            a = args[j]
            if isinstance(a, dict):
                ft = _synth_expr(a, env, ctx)
                elems.append(_element_type(ft))
            j += 1
        return {
            "_type": "_Iterator",
            "element": TupleType(elems, False),
            "source": "zip",
        }
    if fname == "reversed":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                elem = _element_type(ft)
                return {"_type": "_Iterator", "element": elem, "source": "reversed"}
        return {"_type": "_Iterator", "element": ANY_TYPE, "source": "reversed"}
    if fname == "sorted":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                if isinstance(ft, dict) and ft.get("_type") == "_Iterator":
                    elem = ft.get("element")
                    if isinstance(elem, TypeNode):
                        return SliceType(elem)
                elem = _element_type(ft)
                return SliceType(elem)
        return SliceType(ANY_TYPE)
    if fname == "list":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                if isinstance(ft, dict) and ft.get("_type") == "_Iterator":
                    elem = ft.get("element")
                    if isinstance(elem, TypeNode):
                        return SliceType(elem)
                elem = _element_type(ft)
                return SliceType(elem)
        return SliceType(ANY_TYPE)
    if fname == "tuple":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                if isinstance(ft, dict) and ft.get("_type") == "_Iterator":
                    elem = ft.get("element")
                    if isinstance(elem, TypeNode):
                        return TupleType([elem], True)
                elem = _element_type(ft)
                return TupleType([elem], True)
        return TupleType([], False)
    if fname == "set":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                elem = _element_type(ft)
                return SetType(elem)
        return SetType(ANY_TYPE)
    if fname == "dict":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                if isinstance(ft, dict) and ft.get("_type") == "_Iterator":
                    elem = ft.get("element")
                    if isinstance(elem, TupleType):
                        if len(elem.elements) >= 2:
                            return MapType(elem.elements[0], elem.elements[1])
        return MapType(ANY_TYPE, ANY_TYPE)
    if fname == "any" or fname == "all":
        return BOOL_TYPE
    if fname == "divmod":
        return TupleType([INT_TYPE, INT_TYPE], False)
    if fname == "print":
        return VOID_TYPE
    # User-defined function
    func_info = ctx.sig_result.functions.get(fname)
    if func_info is not None:
        return func_info.return_type
    # Class constructor
    if fname in ctx.known_classes:
        return PointerType(StructRef(fname))
    # Callable variable
    typ = env.get_type(fname)
    if typ is not None and isinstance(typ, FuncType):
        return typ.ret
    return ANY_TYPE


def _synth_method_call(
    func: ASTNode,
    args: list[object],
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
) -> TypeNode:
    """Synthesize return type of a method call (obj.method(...))."""
    obj = func.get("value")
    attr = func.get("attr")
    if not isinstance(obj, dict) or not isinstance(attr, str):
        return ANY_TYPE
    obj_type = _synth_expr(obj, env, ctx)
    # String join special case
    if (
        isinstance(obj_type, PrimitiveType)
        and obj_type.kind == "string"
        and attr == "join"
    ):
        return STR_TYPE
    attr_type = _resolve_attr(obj_type, attr, obj, env, ctx)
    if isinstance(attr_type, FuncType):
        if attr == "get" and len(args) >= 2 and isinstance(attr_type.ret, OptionalType):
            return attr_type.ret.inner
        return attr_type.ret
    # Direct method return type from sig table
    sname = _struct_name(obj_type)
    if sname != "":
        methods = ctx.sig_result.methods.get(sname)
        if methods is not None:
            method = methods.get(attr)
            if method is not None:
                return method.return_type
    return attr_type


def _element_type(t: TypeNode) -> TypeNode:
    """Get the element type of a collection."""
    if isinstance(t, SliceType):
        return t.element
    if isinstance(t, SetType):
        return t.element
    if isinstance(t, MapType):
        return t.key
    if isinstance(t, TupleType):
        if len(t.elements) > 0:
            return t.elements[0]
    if isinstance(t, PrimitiveType) and t.kind == "string":
        return STR_TYPE
    return ANY_TYPE


def _synth_subscript(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    value = node.get("value")
    slc = node.get("slice")
    if not isinstance(value, dict):
        return ANY_TYPE
    obj_type = _synth_expr(value, env, ctx)
    # String indexing
    if isinstance(obj_type, PrimitiveType) and obj_type.kind == "string":
        return STR_TYPE
    # List indexing
    if isinstance(obj_type, SliceType):
        if isinstance(slc, dict) and _is_type(slc, ["Slice"]):
            return obj_type
        return obj_type.element
    # Dict indexing
    if isinstance(obj_type, MapType):
        return obj_type.value
    # Tuple indexing
    if isinstance(obj_type, TupleType):
        if obj_type.variadic and len(obj_type.elements) > 0:
            return obj_type.elements[0]
        if isinstance(slc, dict) and _is_type(slc, ["Constant"]):
            idx = slc.get("value")
            if isinstance(idx, int) and not isinstance(idx, bool):
                if 0 <= idx < len(obj_type.elements):
                    return obj_type.elements[idx]
                elif idx < 0 and -idx <= len(obj_type.elements):
                    return obj_type.elements[len(obj_type.elements) + idx]
                else:
                    t_lineno = node.get("lineno", 0)
                    if not isinstance(t_lineno, int):
                        t_lineno = 0
                    ctx.result.add_error(
                        t_lineno,
                        0,
                        "tuple index "
                        + str(idx)
                        + " out of bounds for tuple of length "
                        + str(len(obj_type.elements)),
                    )
    return ANY_TYPE


def _synth_binop(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    left = node.get("left")
    right = node.get("right")
    op = node.get("op", {})
    if not isinstance(left, dict) or not isinstance(right, dict):
        return ANY_TYPE
    lt = _synth_expr(left, env, ctx)
    rt = _synth_expr(right, env, ctx)
    op_type = ""
    if isinstance(op, dict):
        op_type = str(op.get("_type", ""))
    # String concatenation
    if (
        isinstance(lt, PrimitiveType)
        and lt.kind == "string"
        and isinstance(rt, PrimitiveType)
        and rt.kind == "string"
    ):
        return STR_TYPE
    # List concatenation
    if isinstance(lt, SliceType) and isinstance(rt, SliceType):
        if not _is_assignable(
            lt.element, rt.element, ctx.hier_result
        ) and not _is_assignable(rt.element, lt.element, ctx.hier_result):
            b_lineno = node.get("lineno", 0)
            if not isinstance(b_lineno, int):
                b_lineno = 0
            ctx.result.add_error(
                b_lineno,
                0,
                "cannot concatenate list["
                + _type_name(lt.element)
                + "] and list["
                + _type_name(rt.element)
                + "]",
            )
        return lt
    # Numeric
    lt_num = isinstance(lt, PrimitiveType) and lt.kind in ("int", "float", "bool")
    rt_num = isinstance(rt, PrimitiveType) and rt.kind in ("int", "float", "bool")
    if lt_num and rt_num:
        if op_type in ("BitAnd", "BitOr", "BitXor"):
            if (
                isinstance(lt, PrimitiveType)
                and lt.kind == "bool"
                and isinstance(rt, PrimitiveType)
                and rt.kind == "bool"
            ):
                return BOOL_TYPE
            return INT_TYPE
        if (isinstance(lt, PrimitiveType) and lt.kind == "float") or (
            isinstance(rt, PrimitiveType) and rt.kind == "float"
        ):
            return FLOAT_TYPE
        return INT_TYPE
    # String * int
    if (
        isinstance(lt, PrimitiveType)
        and lt.kind == "string"
        and isinstance(rt, PrimitiveType)
        and (rt.kind == "int" or rt.kind == "bool")
    ):
        return STR_TYPE
    if (
        isinstance(lt, PrimitiveType)
        and (lt.kind == "int" or lt.kind == "bool")
        and isinstance(rt, PrimitiveType)
        and rt.kind == "string"
    ):
        return STR_TYPE
    return ANY_TYPE


def _synth_unaryop(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    operand = node.get("operand")
    op = node.get("op", {})
    if not isinstance(operand, dict):
        return ANY_TYPE
    ot = _synth_expr(operand, env, ctx)
    if isinstance(op, dict):
        op_type = op.get("_type", "")
        if op_type == "Not":
            return BOOL_TYPE
        if op_type == "USub" or op_type == "UAdd":
            if isinstance(ot, PrimitiveType) and ot.kind == "bool":
                return INT_TYPE
            return ot
        if op_type == "Invert":
            return INT_TYPE
    return ot


def _synth_boolop(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    values = node.get("values", [])
    if not isinstance(values, list) or len(values) == 0:
        return ANY_TYPE
    last = values[len(values) - 1]
    if isinstance(last, dict):
        return _synth_expr(last, env, ctx)
    return ANY_TYPE


def _synth_ifexp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    test = node.get("test")
    body = node.get("body")
    then_env = env.copy()
    if isinstance(test, dict):
        dummy_else = env.copy()
        _extract_narrowing(test, then_env, dummy_else, ctx)
    if isinstance(body, dict):
        return _synth_expr(body, then_env, ctx)
    return ANY_TYPE


def _synth_list(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    elts = node.get("elts", [])
    if not isinstance(elts, list) or len(elts) == 0:
        return SliceType(ANY_TYPE)
    first = elts[0]
    if isinstance(first, dict):
        return SliceType(_synth_expr(first, env, ctx))
    return SliceType(ANY_TYPE)


def _synth_dict(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    keys = node.get("keys", [])
    values = node.get("values", [])
    if not isinstance(keys, list) or not isinstance(values, list):
        return MapType(ANY_TYPE, ANY_TYPE)
    if len(keys) == 0:
        return MapType(ANY_TYPE, ANY_TYPE)
    k = keys[0]
    v = values[0]
    kt = ANY_TYPE
    vt = ANY_TYPE
    if isinstance(k, dict):
        kt = _synth_expr(k, env, ctx)
    if isinstance(v, dict):
        vt = _synth_expr(v, env, ctx)
    return MapType(kt, vt)


def _synth_set(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    elts = node.get("elts", [])
    if not isinstance(elts, list) or len(elts) == 0:
        return SetType(ANY_TYPE)
    first = elts[0]
    if isinstance(first, dict):
        return SetType(_synth_expr(first, env, ctx))
    return SetType(ANY_TYPE)


def _synth_tuple(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    elts = node.get("elts", [])
    if not isinstance(elts, list):
        return TupleType([], False)
    elems: list[TypeNode] = []
    i = 0
    while i < len(elts):
        e = elts[i]
        if isinstance(e, dict):
            elems.append(_synth_expr(e, env, ctx))
        else:
            elems.append(ANY_TYPE)
        i += 1
    return TupleType(elems, False)


def _synth_listcomp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    elt = node.get("elt")
    generators = node.get("generators", [])
    comp_env = env.copy()
    if isinstance(generators, list):
        _bind_comprehension_vars(generators, comp_env, ctx)
    if isinstance(elt, dict):
        return SliceType(_synth_expr(elt, comp_env, ctx))
    return SliceType(ANY_TYPE)


def _synth_setcomp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    elt = node.get("elt")
    generators = node.get("generators", [])
    comp_env = env.copy()
    if isinstance(generators, list):
        _bind_comprehension_vars(generators, comp_env, ctx)
    if isinstance(elt, dict):
        return SetType(_synth_expr(elt, comp_env, ctx))
    return SetType(ANY_TYPE)


def _synth_dictcomp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    key = node.get("key")
    value = node.get("value")
    generators = node.get("generators", [])
    comp_env = env.copy()
    if isinstance(generators, list):
        _bind_comprehension_vars(generators, comp_env, ctx)
    kt = ANY_TYPE
    vt = ANY_TYPE
    if isinstance(key, dict):
        kt = _synth_expr(key, comp_env, ctx)
    if isinstance(value, dict):
        vt = _synth_expr(value, comp_env, ctx)
    return MapType(kt, vt)


def _bind_comprehension_vars(
    generators: list[object], env: TypeEnv, ctx: _InferCtx
) -> None:
    """Bind iteration variables from comprehension generators."""
    i = 0
    while i < len(generators):
        gen = generators[i]
        if isinstance(gen, dict):
            target = gen.get("target")
            iter_node = gen.get("iter")
            if isinstance(iter_node, dict):
                iter_type = _synth_expr(iter_node, env, ctx)
                elem = _iteration_element(iter_type)
                _bind_target(target, elem, env)
            ifs = gen.get("ifs", [])
            if isinstance(ifs, list):
                j = 0
                while j < len(ifs):
                    cond = ifs[j]
                    if isinstance(cond, dict):
                        dummy_else = env.copy()
                        _extract_narrowing(cond, env, dummy_else, ctx)
                    j += 1
        i += 1


def _synth_namedexpr(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    target = node.get("target")
    value = node.get("value")
    if not isinstance(value, dict):
        return ANY_TYPE
    vt = _synth_expr(value, env, ctx)
    if isinstance(target, dict) and _is_type(target, ["Name"]):
        name = target.get("id")
        if isinstance(name, str):
            env.set(name, vt, _type_name(vt))
    return vt


# ---------------------------------------------------------------------------
# Iteration element type
# ---------------------------------------------------------------------------


def _iteration_element(t: TypeNode) -> TypeNode:
    """Get the element type when iterating over a type."""
    # Handle _Iterator dicts (these are still dicts, not TypeNode)
    if isinstance(t, dict) and t.get("_type") == "_Iterator":
        elem = t.get("element")
        if isinstance(elem, TypeNode):
            return elem
        return ANY_TYPE
    if isinstance(t, SliceType):
        return t.element
    if isinstance(t, SetType):
        return t.element
    if isinstance(t, MapType):
        return t.key
    if isinstance(t, TupleType):
        if t.variadic and len(t.elements) > 0:
            return t.elements[0]
    if isinstance(t, PrimitiveType) and t.kind == "string":
        return STR_TYPE
    return ANY_TYPE


def _bind_target(target: object, typ: TypeNode, env: TypeEnv) -> None:
    """Bind an assignment target (Name or Tuple) to a type."""
    if not isinstance(target, dict):
        return
    if _is_type(target, ["Name"]):
        name = target.get("id")
        if isinstance(name, str):
            env.set(name, typ, _type_name(typ))
    elif _is_type(target, ["Tuple", "List"]):
        elts = target.get("elts", [])
        if isinstance(elts, list) and isinstance(typ, TupleType):
            j = 0
            while j < len(elts) and j < len(typ.elements):
                _bind_target(elts[j], typ.elements[j], env)
                j += 1


# ---------------------------------------------------------------------------
# Inference context
# ---------------------------------------------------------------------------


class _InferCtx:
    """Shared context for inference within a module."""

    def __init__(
        self,
        sig_result: SignatureResult,
        field_result: FieldResult,
        hier_result: HierarchyResult,
        known_classes: set[str],
        class_bases: dict[str, list[str]],
        result: InferenceResult,
    ) -> None:
        self.sig_result: SignatureResult = sig_result
        self.field_result: FieldResult = field_result
        self.hier_result: HierarchyResult = hier_result
        self.known_classes: set[str] = known_classes
        self.class_bases: dict[str, list[str]] = class_bases
        self.result: InferenceResult = result
        self.module_vars: dict[str, TypeNode] = {}


# ---------------------------------------------------------------------------
# Statement validation
# ---------------------------------------------------------------------------


def _validate_func(func_node: ASTNode, ctx: _InferCtx, receiver: str) -> None:
    """Validate a single function/method body."""
    func_name = func_node.get("name", "")
    if not isinstance(func_name, str):
        func_name = ""
    func_info: FuncInfo | None = None
    if receiver != "":
        methods = ctx.sig_result.methods.get(receiver)
        if methods is not None:
            func_info = methods.get(func_name)
    else:
        func_info = ctx.sig_result.functions.get(func_name)
    if func_info is None:
        return
    env = TypeEnv()
    i = 0
    while i < len(func_info.params):
        p = func_info.params[i]
        env.set(p.name, p.typ, p.py_type)
        i += 1
    if receiver != "":
        self_type = PointerType(StructRef(receiver))
        env.set("self", self_type, receiver)
    body = func_node.get("body", [])
    if not isinstance(body, list):
        return
    _validate_stmts(body, env, func_info, ctx)


def _validate_stmts(
    stmts: list[object],
    env: TypeEnv,
    func_info: FuncInfo,
    ctx: _InferCtx,
) -> bool:
    """Validate a list of statements. Returns True if all paths return."""
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        if not isinstance(stmt, dict):
            i += 1
            continue
        if len(ctx.result._errors) > 0:
            return False
        returned = _validate_stmt(stmt, env, func_info, ctx)
        if returned:
            return True
        i += 1
    return False


def _validate_stmt(
    stmt: ASTNode,
    env: TypeEnv,
    func_info: FuncInfo,
    ctx: _InferCtx,
) -> bool:
    """Validate a single statement. Returns True if it always returns."""
    t = stmt.get("_type")
    if t == "Return":
        _validate_return(stmt, env, func_info, ctx)
        return True
    if t == "Assign":
        _validate_assign(stmt, env, func_info, ctx)
        return False
    if t == "AnnAssign":
        _validate_ann_assign(stmt, env, func_info, ctx)
        return False
    if t == "AugAssign":
        _validate_aug_assign(stmt, env, func_info, ctx)
        return False
    if t == "Expr":
        _validate_expr_stmt(stmt, env, func_info, ctx)
        return False
    if t == "If":
        return _validate_if(stmt, env, func_info, ctx)
    if t == "While":
        _validate_while(stmt, env, func_info, ctx)
        return False
    if t == "For":
        _validate_for(stmt, env, func_info, ctx)
        return False
    if t == "Assert":
        _validate_assert(stmt, env, func_info, ctx)
        return False
    if t == "Pass":
        return False
    if t == "Break" or t == "Continue":
        return True
    if t == "Raise":
        return True
    if t == "Try":
        _validate_try(stmt, env, func_info, ctx)
        return False
    if t == "Match":
        _validate_match(stmt, env, func_info, ctx)
        return False
    if t == "FunctionDef":
        lineno = stmt.get("lineno", 0)
        if not isinstance(lineno, int):
            lineno = 0
        ctx.result.add_error(lineno, 0, "nested function definitions are not allowed")
        return False
    return False


def _validate_return(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    value = stmt.get("value")
    if value is None:
        return
    if not isinstance(value, dict):
        return
    lineno = stmt.get("lineno", 0)
    if not isinstance(lineno, int):
        lineno = 0
    if _check_iterator_escape_return(value, env, ctx, lineno):
        return
    if _check_generator_escape_return(value, env, ctx, lineno):
        return
    _validate_expr_access(value, env, ctx, lineno)
    if len(ctx.result._errors) > 0:
        return
    if _is_type(value, ["Call"]):
        _validate_call_args(value, env, ctx, lineno)
        if len(ctx.result._errors) > 0:
            return
    _validate_return_value(value, func_info.return_type, env, ctx, lineno)
    if len(ctx.result._errors) > 0:
        return
    actual = _synth_expr(value, env, ctx)
    expected = func_info.return_type
    if not _is_assignable(actual, expected, ctx.hier_result):
        ctx.result.add_error(
            lineno,
            0,
            "cannot return " + _type_name(actual) + " as " + _type_name(expected),
        )


def _validate_assign(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    targets = stmt.get("targets", [])
    value = stmt.get("value")
    if not isinstance(targets, list) or not isinstance(value, dict):
        return
    lineno = stmt.get("lineno", 0)
    if not isinstance(lineno, int):
        lineno = 0
    if len(targets) == 1:
        tgt = targets[0]
        if isinstance(tgt, dict) and _is_type(tgt, ["Name"]):
            if _check_iterator_escape_assign(value, env, ctx, lineno):
                return
            if _check_generator_escape_assign(value, env, ctx, lineno):
                return
    val_type = _synth_expr(value, env, ctx)
    i = 0
    while i < len(targets):
        tgt = targets[i]
        if isinstance(tgt, dict):
            if _is_type(tgt, ["Name"]):
                name = tgt.get("id")
                if isinstance(name, str):
                    existing = env.get_type(name)
                    if existing is not None:
                        if not _type_eq(val_type, existing):
                            source = _infer_source(value, env, ctx)
                            env.set(name, val_type, source)
                    else:
                        if _is_empty_collection(value) and is_any(
                            _element_type(val_type)
                        ):
                            ctx.result.add_error(
                                lineno,
                                0,
                                "empty "
                                + _collection_name(value)
                                + " needs type annotation",
                            )
                            return
                        source = _infer_source(value, env, ctx)
                        env.set(name, val_type, source)
            elif _is_type(tgt, ["Tuple", "List"]):
                _validate_unpack(tgt, val_type, value, env, ctx, lineno)
            elif _is_type(tgt, ["Subscript"]):
                _validate_subscript_assign(tgt, val_type, env, ctx, lineno)
            elif _is_type(tgt, ["Attribute"]):
                pass
        i += 1


def _is_empty_collection(node: ASTNode) -> bool:
    t = node.get("_type")
    if t == "List":
        elts = node.get("elts", [])
        return isinstance(elts, list) and len(elts) == 0
    if t == "Dict":
        keys = node.get("keys", [])
        return isinstance(keys, list) and len(keys) == 0
    if t == "Set":
        elts = node.get("elts", [])
        return isinstance(elts, list) and len(elts) == 0
    return False


def _collection_name(node: ASTNode) -> str:
    t = node.get("_type")
    if t == "List":
        return "list"
    if t == "Dict":
        return "dict"
    if t == "Set":
        return "set"
    return "collection"


def _infer_source(value: ASTNode, env: TypeEnv, ctx: _InferCtx) -> str:
    """Infer the source type string for an expression."""
    t = value.get("_type")
    if t == "Constant":
        v = value.get("value")
        if v is None:
            return "None"
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "str"
    if t == "Name":
        name = value.get("id")
        if isinstance(name, str):
            return env.get_source(name)
    if t == "Call":
        func = value.get("func")
        if isinstance(func, dict) and _is_type(func, ["Name"]):
            fname = func.get("id")
            if isinstance(fname, str):
                fi = ctx.sig_result.functions.get(fname)
                if fi is not None:
                    return fi.return_py_type
    return ""


def _validate_unpack(
    target: ASTNode,
    val_type: TypeNode,
    value: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Validate tuple unpacking."""
    elts = target.get("elts", [])
    if not isinstance(elts, list):
        return
    if isinstance(val_type, OptionalType):
        ctx.result.add_error(lineno, 0, "cannot unpack optional tuple without guard")
        return
    if not isinstance(val_type, TupleType):
        j = 0
        while j < len(elts):
            _bind_target(elts[j], ANY_TYPE, env)
            j += 1
        return
    if val_type.variadic:
        elem = ANY_TYPE
        if len(val_type.elements) > 0:
            elem = val_type.elements[0]
        j = 0
        while j < len(elts):
            _bind_target(elts[j], elem, env)
            j += 1
        return
    if len(elts) != len(val_type.elements):
        ctx.result.add_error(
            lineno,
            0,
            "cannot unpack tuple of "
            + str(len(val_type.elements))
            + " elements into "
            + str(len(elts))
            + " targets",
        )
        return
    j = 0
    while j < len(elts):
        _bind_target(elts[j], val_type.elements[j], env)
        j += 1


def _validate_subscript_assign(
    target: ASTNode,
    val_type: TypeNode,
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Validate d[k] = v assignment."""
    value = target.get("value")
    slc = target.get("slice")
    if not isinstance(value, dict):
        return
    obj_type = _synth_expr(value, env, ctx)
    if isinstance(obj_type, MapType):
        if isinstance(slc, dict):
            key_actual = _synth_expr(slc, env, ctx)
            if not _is_assignable(key_actual, obj_type.key, ctx.hier_result):
                ctx.result.add_error(
                    lineno,
                    0,
                    "cannot assign "
                    + _type_name(key_actual)
                    + " key to "
                    + _type_name(obj_type.key),
                )
                return
        if not _is_assignable(val_type, obj_type.value, ctx.hier_result):
            ctx.result.add_error(
                lineno,
                0,
                "cannot assign "
                + _type_name(val_type)
                + " value to "
                + _type_name(obj_type.value),
            )
    elif isinstance(obj_type, SliceType):
        if not _is_assignable(val_type, obj_type.element, ctx.hier_result):
            ctx.result.add_error(
                lineno,
                0,
                "cannot assign "
                + _type_name(val_type)
                + " to list element "
                + _type_name(obj_type.element),
            )


def _validate_ann_assign(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    target = stmt.get("target")
    annotation = stmt.get("annotation")
    value = stmt.get("value")
    if not isinstance(target, dict):
        return
    lineno = stmt.get("lineno", 0)
    if not isinstance(lineno, int):
        lineno = 0
    if not isinstance(annotation, dict):
        return
    ann_str = annotation_to_str(annotation)
    sig_errors: list[SignatureError] = []
    ann_type = py_type_to_type_dict(ann_str, ctx.known_classes, sig_errors, lineno, 0)
    if _is_type(target, ["Name"]):
        name = target.get("id")
        if isinstance(name, str):
            env.set(name, ann_type, ann_str)
            if value is not None and isinstance(value, dict):
                val_type = _synth_expr(value, env, ctx)
                if not _is_assignable(val_type, ann_type, ctx.hier_result):
                    ctx.result.add_error(
                        lineno,
                        0,
                        "cannot assign "
                        + _type_name(val_type)
                        + " to "
                        + _type_name(ann_type),
                    )


def _validate_aug_assign(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    target = stmt.get("target")
    value = stmt.get("value")
    if not isinstance(target, dict) or not isinstance(value, dict):
        return
    lineno = stmt.get("lineno", 0)
    if not isinstance(lineno, int):
        lineno = 0
    _synth_expr(value, env, ctx)


def _validate_expr_stmt(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    value = stmt.get("value")
    if not isinstance(value, dict):
        return
    lineno = stmt.get("lineno", 0)
    if not isinstance(lineno, int):
        lineno = 0
    if _is_type(value, ["Call"]):
        func = value.get("func")
        if isinstance(func, dict) and _is_type(func, ["Name"]):
            fname = func.get("id")
            if (
                isinstance(fname, str)
                and fname not in _EAGER_CONSUMERS
                and fname not in _ITERATOR_FUNCS
            ):
                args = value.get("args", [])
                if isinstance(args, list):
                    j = 0
                    while j < len(args):
                        arg = args[j]
                        if isinstance(arg, dict):
                            _check_iterator_escape_arg(arg, fname, env, ctx, lineno)
                            _check_generator_escape_arg(arg, fname, env, ctx, lineno)
                        j += 1
        if isinstance(func, dict) and _is_type(func, ["Attribute"]):
            attr = func.get("attr")
            if isinstance(attr, str):
                args = value.get("args", [])
                if isinstance(args, list):
                    j = 0
                    while j < len(args):
                        arg = args[j]
                        if isinstance(arg, dict):
                            if attr == "join":
                                pass
                            else:
                                _check_generator_escape_arg(arg, attr, env, ctx, lineno)
                        j += 1
    _synth_expr(value, env, ctx)
    _validate_call_args(value, env, ctx, lineno)


def _validate_call_args(
    node: ASTNode, env: TypeEnv, ctx: _InferCtx, lineno: int
) -> None:
    """Validate argument types in function/method calls."""
    if not _is_type(node, ["Call"]):
        return
    func = node.get("func")
    args = node.get("args", [])
    if not isinstance(func, dict) or not isinstance(args, list):
        return
    if _is_type(func, ["Name"]):
        fname = func.get("id")
        if not isinstance(fname, str):
            return
        func_info = ctx.sig_result.functions.get(fname)
        if func_info is not None:
            _check_call_args(func_info, args, env, ctx, lineno)
            return
        if fname in ctx.known_classes:
            return
        ftype = env.get_type(fname)
        if ftype is not None and isinstance(ftype, FuncType):
            _check_func_type_args(ftype, args, env, ctx, lineno)
            return
        if fname == "len":
            if len(args) > 0:
                a = args[0]
                if isinstance(a, dict):
                    at = _synth_expr(a, env, ctx)
                    if isinstance(at, PrimitiveType) and at.kind in (
                        "int",
                        "float",
                        "bool",
                    ):
                        ctx.result.add_error(
                            lineno,
                            0,
                            "len() requires a sized type, got " + _type_name(at),
                        )
    if _is_type(func, ["Attribute"]):
        obj = func.get("value")
        attr = func.get("attr")
        if not isinstance(obj, dict) or not isinstance(attr, str):
            return
        obj_type = _synth_expr(obj, env, ctx)
        if _is_type(obj, ["Name"]):
            obj_name = obj.get("id")
            if isinstance(obj_name, str) and obj_name in ctx.known_classes:
                methods = ctx.sig_result.methods.get(obj_name)
                if methods is not None and attr in methods:
                    ctx.result.add_error(
                        lineno,
                        0,
                        "cannot call method without self: " + obj_name + "." + attr,
                    )
                    return
        sname = _struct_name(obj_type)
        if sname != "":
            methods = ctx.sig_result.methods.get(sname)
            method: FuncInfo | None = None
            if methods is not None:
                method = methods.get(attr)
            if method is not None:
                _check_call_args(method, args, env, ctx, lineno)
                return
            if _subclass_has_method(sname, attr, ctx):
                ctx.result.add_error(
                    lineno, 0, "method '" + attr + "' not accessible on " + sname
                )
                return
        check_type = obj_type
        if isinstance(obj_type, PointerType) and not isinstance(
            obj_type.target, StructRef
        ):
            check_type = obj_type.target
        _validate_collection_method_args(check_type, attr, args, env, ctx, lineno)


def _check_call_args(
    func_info: FuncInfo,
    args: list[object],
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Check argument types against function parameters."""
    params = func_info.params
    n_positional = 0
    n_required = 0
    j = 0
    while j < len(params):
        if params[j].modifier != "keyword":
            n_positional += 1
            if not params[j].has_default:
                n_required += 1
        j += 1
    if len(args) < n_required or len(args) > n_positional:
        ctx.result.add_error(
            lineno,
            0,
            "expected " + str(len(params)) + " arguments, got " + str(len(args)),
        )
        return
    j = 0
    while j < len(args):
        arg = args[j]
        if isinstance(arg, dict) and j < len(params):
            actual = _synth_expr(arg, env, ctx)
            expected = params[j].typ
            if not _is_assignable(actual, expected, ctx.hier_result):
                ctx.result.add_error(
                    lineno,
                    0,
                    "argument "
                    + str(j + 1)
                    + " has type "
                    + _type_name(actual)
                    + ", expected "
                    + _type_name(expected),
                )
                return
        j += 1


def _check_func_type_args(
    ftype: FuncType,
    args: list[object],
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Check args against a FuncType."""
    if len(args) != len(ftype.params):
        ctx.result.add_error(
            lineno,
            0,
            "expected " + str(len(ftype.params)) + " arguments, got " + str(len(args)),
        )
        return
    j = 0
    while j < len(args):
        arg = args[j]
        if isinstance(arg, dict) and j < len(ftype.params):
            actual = _synth_expr(arg, env, ctx)
            expected = ftype.params[j]
            if not _is_assignable(actual, expected, ctx.hier_result):
                ctx.result.add_error(
                    lineno,
                    0,
                    "argument "
                    + str(j + 1)
                    + " has type "
                    + _type_name(actual)
                    + ", expected "
                    + _type_name(expected),
                )
                return
        j += 1


def _validate_collection_method_args(
    obj_type: TypeNode,
    method: str,
    args: list[object],
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Validate collection method argument types."""
    if isinstance(obj_type, SliceType):
        elem = obj_type.element
        if method == "append":
            if len(args) > 0:
                a = args[0]
                if isinstance(a, dict):
                    at = _synth_expr(a, env, ctx)
                    if not _is_assignable(at, elem, ctx.hier_result):
                        ctx.result.add_error(
                            lineno,
                            0,
                            "cannot assign "
                            + _type_name(at)
                            + " to list element "
                            + _type_name(elem),
                        )
        elif method == "extend":
            if len(args) > 0:
                a = args[0]
                if isinstance(a, dict):
                    at = _synth_expr(a, env, ctx)
                    aelem = _element_type(at)
                    if not _is_assignable(aelem, elem, ctx.hier_result):
                        ctx.result.add_error(
                            lineno,
                            0,
                            "cannot assign "
                            + _type_name(aelem)
                            + " to list element "
                            + _type_name(elem),
                        )
        elif method == "insert":
            if len(args) > 1:
                a = args[1]
                if isinstance(a, dict):
                    at = _synth_expr(a, env, ctx)
                    if not _is_assignable(at, elem, ctx.hier_result):
                        ctx.result.add_error(
                            lineno,
                            0,
                            "cannot assign "
                            + _type_name(at)
                            + " to list element "
                            + _type_name(elem),
                        )
    elif isinstance(obj_type, SetType):
        elem = obj_type.element
        if method == "add":
            if len(args) > 0:
                a = args[0]
                if isinstance(a, dict):
                    at = _synth_expr(a, env, ctx)
                    if not _is_assignable(at, elem, ctx.hier_result):
                        ctx.result.add_error(
                            lineno,
                            0,
                            "cannot assign "
                            + _type_name(at)
                            + " to set element "
                            + _type_name(elem),
                        )


def _validate_if(
    stmt: ASTNode,
    env: TypeEnv,
    func_info: FuncInfo,
    ctx: _InferCtx,
) -> bool:
    """Validate if/elif/else. Returns True if all branches return."""
    test = stmt.get("test")
    body = stmt.get("body", [])
    orelse = stmt.get("orelse", [])
    if not isinstance(body, list):
        body: list[ASTNode] = []
    if not isinstance(orelse, list):
        orelse: list[ASTNode] = []
    lineno = stmt.get("lineno", 0)
    if not isinstance(lineno, int):
        lineno = 0
    if isinstance(test, dict):
        _check_truthiness(test, env, ctx, lineno)
    then_env = env.copy()
    else_env = env.copy()
    if isinstance(test, dict):
        _extract_narrowing(test, then_env, else_env, ctx)
    then_returns = _validate_stmts(body, then_env, func_info, ctx)
    if len(ctx.result._errors) > 0:
        return False
    else_returns = False
    if len(orelse) > 0:
        else_returns = _validate_stmts(orelse, else_env, func_info, ctx)
    if then_returns and not else_returns:
        ekeys = list(else_env.types.keys())
        j = 0
        while j < len(ekeys):
            env.types[ekeys[j]] = else_env.types[ekeys[j]]
            j += 1
        skeys = list(else_env.source_types.keys())
        j = 0
        while j < len(skeys):
            env.source_types[skeys[j]] = else_env.source_types[skeys[j]]
            j += 1
        gkeys = list(else_env.guarded_attrs)
        j = 0
        while j < len(gkeys):
            env.guarded_attrs.add(gkeys[j])
            j += 1
    elif not then_returns and len(orelse) == 0:
        ekeys = list(else_env.types.keys())
        j = 0
        while j < len(ekeys):
            k = ekeys[j]
            else_t = else_env.types[k]
            then_t = then_env.types.get(k)
            if then_t is not None and _type_eq(else_t, then_t):
                env.types[k] = else_t
                es = else_env.source_types.get(k, "")
                if es != "":
                    env.source_types[k] = es
            j += 1
    return then_returns and else_returns


def _validate_while(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    test = stmt.get("test")
    body = stmt.get("body", [])
    if not isinstance(body, list):
        body: list[ASTNode] = []
    lineno = stmt.get("lineno", 0)
    if not isinstance(lineno, int):
        lineno = 0
    if isinstance(test, dict):
        _check_truthiness(test, env, ctx, lineno)
    loop_env = env.copy()
    _validate_stmts(body, loop_env, func_info, ctx)


def _validate_for(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    target = stmt.get("target")
    iter_node = stmt.get("iter")
    body = stmt.get("body", [])
    if not isinstance(body, list):
        body: list[ASTNode] = []
    if isinstance(iter_node, dict):
        iter_type = _synth_expr(iter_node, env, ctx)
        elem = _iteration_element(iter_type)
        if isinstance(target, dict):
            _bind_target(target, elem, env)
    _validate_stmts(body, env, func_info, ctx)


def _validate_assert(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    test = stmt.get("test")
    if not isinstance(test, dict):
        return
    dummy_else = env.copy()
    _extract_narrowing(test, env, dummy_else, ctx)


def _validate_try(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    body = stmt.get("body", [])
    handlers = stmt.get("handlers", [])
    orelse = stmt.get("orelse", [])
    finalbody = stmt.get("finalbody", [])
    if isinstance(body, list):
        _validate_stmts(body, env, func_info, ctx)
    if isinstance(handlers, list):
        j = 0
        while j < len(handlers):
            h = handlers[j]
            if isinstance(h, dict):
                hbody = h.get("body", [])
                if isinstance(hbody, list):
                    _validate_stmts(hbody, env.copy(), func_info, ctx)
            j += 1
    if isinstance(orelse, list):
        _validate_stmts(orelse, env, func_info, ctx)
    if isinstance(finalbody, list):
        _validate_stmts(finalbody, env, func_info, ctx)


def _validate_match(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    cases = stmt.get("cases", [])
    if not isinstance(cases, list):
        return
    j = 0
    while j < len(cases):
        case = cases[j]
        if isinstance(case, dict):
            case_body = case.get("body", [])
            if isinstance(case_body, list):
                _validate_stmts(case_body, env.copy(), func_info, ctx)
        j += 1


# ---------------------------------------------------------------------------
# Truthiness checking
# ---------------------------------------------------------------------------


def _check_truthiness(test: ASTNode, env: TypeEnv, ctx: _InferCtx, lineno: int) -> None:
    """Check that a condition expression has unambiguous truthiness."""
    if not isinstance(test, dict):
        return
    t = test.get("_type")
    if t == "Compare":
        return
    if t == "Call":
        func = test.get("func")
        if (
            isinstance(func, dict)
            and _is_type(func, ["Name"])
            and func.get("id") == "isinstance"
        ):
            return
    if t == "BoolOp":
        values = test.get("values", [])
        if isinstance(values, list):
            j = 0
            while j < len(values):
                v = values[j]
                if isinstance(v, dict):
                    _check_truthiness(v, env, ctx, lineno)
                j += 1
        return
    if t == "UnaryOp":
        op = test.get("op", {})
        if isinstance(op, dict) and op.get("_type") == "Not":
            operand = test.get("operand")
            if isinstance(operand, dict):
                _check_truthiness(operand, env, ctx, lineno)
            return
    if t == "NamedExpr":
        value = test.get("value")
        if isinstance(value, dict):
            vt = _synth_expr(value, env, ctx)
            _check_type_truthiness(vt, env, test, ctx, lineno)
        return
    expr_type = _synth_expr(test, env, ctx)
    _check_type_truthiness(expr_type, env, test, ctx, lineno)


def _check_type_truthiness(
    typ: TypeNode,
    env: TypeEnv,
    node: ASTNode,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Check if a type has unambiguous truthiness."""
    if isinstance(typ, PrimitiveType) and typ.kind == "bool":
        return
    if isinstance(typ, PrimitiveType) and typ.kind == "int":
        ctx.result.add_error(
            lineno, 0, "truthiness of int not allowed (zero is valid data)"
        )
        return
    if isinstance(typ, PrimitiveType) and typ.kind == "float":
        ctx.result.add_error(
            lineno, 0, "truthiness of float not allowed (zero is valid data)"
        )
        return
    if isinstance(typ, OptionalType):
        inner = typ.inner
        if isinstance(inner, PrimitiveType) and inner.kind in ("int", "float", "bool"):
            return
        if isinstance(inner, PrimitiveType) and inner.kind == "string":
            ctx.result.add_error(lineno, 0, "ambiguous truthiness for optional str")
            return
        if isinstance(inner, SliceType):
            ctx.result.add_error(lineno, 0, "ambiguous truthiness for optional list")
            return
        if isinstance(inner, MapType):
            ctx.result.add_error(lineno, 0, "ambiguous truthiness for optional dict")
            return
        if isinstance(inner, SetType):
            ctx.result.add_error(lineno, 0, "ambiguous truthiness for optional set")
            return
        return
    source = ""
    if _is_type(node, ["Name"]):
        name = node.get("id")
        if isinstance(name, str):
            source = env.get_source(name)
    if source != "" and _is_optional_source(source):
        non_none = _non_none_parts(source)
        if len(non_none) == 1:
            nn = non_none[0]
            if nn == "str":
                ctx.result.add_error(lineno, 0, "ambiguous truthiness for optional str")
                return
    if isinstance(typ, PrimitiveType) and typ.kind == "string":
        return
    if isinstance(typ, SliceType):
        return
    if isinstance(typ, MapType):
        return
    if isinstance(typ, SetType):
        return
    if is_any(typ):
        return
    if isinstance(typ, InterfaceRef):
        return


# ---------------------------------------------------------------------------
# Narrowing
# ---------------------------------------------------------------------------


def _extract_narrowing(
    test: ASTNode,
    then_env: TypeEnv,
    else_env: TypeEnv,
    ctx: _InferCtx,
) -> None:
    """Extract type narrowing from a condition into then/else environments."""
    if not isinstance(test, dict):
        return
    t = test.get("_type")
    if t == "Call":
        func = test.get("func")
        if (
            isinstance(func, dict)
            and _is_type(func, ["Name"])
            and func.get("id") == "isinstance"
        ):
            _narrow_isinstance(test, then_env, else_env, ctx)
            return
    if t == "Compare":
        _narrow_compare(test, then_env, else_env, ctx)
        return
    if t == "UnaryOp":
        op = test.get("op", {})
        if isinstance(op, dict) and op.get("_type") == "Not":
            operand = test.get("operand")
            if isinstance(operand, dict):
                _extract_narrowing(operand, else_env, then_env, ctx)
            return
    if t == "BoolOp":
        op = test.get("op", {})
        if isinstance(op, dict):
            op_t = op.get("_type")
            if op_t == "And":
                values = test.get("values", [])
                if isinstance(values, list):
                    j = 0
                    while j < len(values):
                        v = values[j]
                        if isinstance(v, dict):
                            _extract_narrowing(v, then_env, else_env, ctx)
                        j += 1
                return
            if op_t == "Or":
                values = test.get("values", [])
                if isinstance(values, list):
                    _narrow_or_isinstance(values, then_env, ctx)
                    j = 0
                    while j < len(values):
                        v = values[j]
                        if isinstance(v, dict):
                            _extract_narrowing(v, then_env, else_env, ctx)
                        j += 1
                return
    if t == "NamedExpr":
        target = test.get("target")
        value = test.get("value")
        if (
            isinstance(target, dict)
            and isinstance(value, dict)
            and _is_type(target, ["Name"])
        ):
            name = target.get("id")
            if isinstance(name, str):
                vt = _synth_expr(value, then_env, ctx)
                then_env.set(name, vt, _type_name(vt))
                else_env.set(name, vt, _type_name(vt))
                if isinstance(vt, OptionalType):
                    then_env.set(name, vt.inner, _type_name(vt.inner))
        return
    if t == "Name":
        name = test.get("id")
        if isinstance(name, str):
            typ = then_env.get_type(name)
            source = then_env.get_source(name)
            if typ is not None and isinstance(typ, OptionalType):
                then_env.narrow(name, typ.inner, _type_name(typ.inner))
            elif source != "" and _is_optional_source(source):
                non_none = _non_none_parts(source)
                if len(non_none) == 1:
                    sig_errors: list[SignatureError] = []
                    narrowed = py_type_to_type_dict(
                        non_none[0], ctx.known_classes, sig_errors, 0, 0
                    )
                    then_env.narrow(name, narrowed, non_none[0])
        return


def _attr_path(node: ASTNode) -> str:
    """Build dotted path from nested Attribute nodes, e.g. 'expr.obj'."""
    parts: list[str] = []
    cur = node
    while _is_type(cur, ["Attribute"]):
        a = cur.get("attr")
        if not isinstance(a, str):
            return ""
        parts.append(a)
        v = cur.get("value")
        if not isinstance(v, dict):
            return ""
        cur = v
    if not _is_type(cur, ["Name"]):
        return ""
    base = cur.get("id")
    if not isinstance(base, str) or base == "":
        return ""
    result = base
    i = len(parts) - 1
    while i >= 0:
        result = result + "." + parts[i]
        i -= 1
    return result


def _narrow_isinstance(
    test: ASTNode,
    then_env: TypeEnv,
    else_env: TypeEnv,
    ctx: _InferCtx,
) -> None:
    """Narrow from isinstance(x, T)."""
    args = test.get("args", [])
    if not isinstance(args, list) or len(args) < 2:
        return
    target = args[0]
    type_arg = args[1]
    if not isinstance(target, dict) or not isinstance(type_arg, dict):
        return
    name = ""
    if _is_type(target, ["Name"]):
        name = str(target.get("id", ""))
    elif _is_type(target, ["Attribute"]):
        name = _attr_path(target)
    if name == "":
        return
    narrow_name = ""
    if _is_type(type_arg, ["Name"]):
        narrow_name = str(type_arg.get("id", ""))
    if narrow_name == "":
        return
    sig_errors: list[SignatureError] = []
    narrowed = py_type_to_type_dict(narrow_name, ctx.known_classes, sig_errors, 0, 0)
    then_env.narrow(name, narrowed, narrow_name)
    source = else_env.get_source(name)
    if source != "" and _is_union_source(source):
        parts = _split_union_parts(source)
        remaining: list[str] = []
        j = 0
        while j < len(parts):
            if parts[j] != narrow_name:
                remaining.append(parts[j])
            j += 1
        if len(remaining) == 1:
            sig_errors2: list[SignatureError] = []
            rem_type = py_type_to_type_dict(
                remaining[0], ctx.known_classes, sig_errors2, 0, 0
            )
            else_env.narrow(name, rem_type, remaining[0])
        elif len(remaining) > 1:
            new_source = " | ".join(remaining)
            sig_errors2: list[SignatureError] = []
            rem_type = py_type_to_type_dict(
                new_source, ctx.known_classes, sig_errors2, 0, 0
            )
            else_env.narrow(name, rem_type, new_source)


def _narrow_compare(
    test: ASTNode,
    then_env: TypeEnv,
    else_env: TypeEnv,
    ctx: _InferCtx,
) -> None:
    """Narrow from comparison (x is None, x is not None, x.kind == "foo")."""
    left = test.get("left")
    ops = test.get("ops", [])
    comparators = test.get("comparators", [])
    if (
        not isinstance(left, dict)
        or not isinstance(ops, list)
        or not isinstance(comparators, list)
    ):
        return
    if len(ops) == 0 or len(comparators) == 0:
        return
    op = ops[0]
    comp = comparators[0]
    if not isinstance(op, dict) or not isinstance(comp, dict):
        return
    op_type = op.get("_type", "")
    if op_type == "Is" and _is_type(comp, ["Constant"]) and comp.get("value") is None:
        if _is_type(left, ["Name"]):
            name = str(left.get("id", ""))
            if name != "":
                _narrow_to_non_none(name, else_env, ctx)
        if _is_type(left, ["Attribute"]):
            path = _attr_path(left)
            if path != "":
                else_env.guard_attr(path)
        return
    if (
        op_type == "IsNot"
        and _is_type(comp, ["Constant"])
        and comp.get("value") is None
    ):
        if _is_type(left, ["Name"]):
            name = str(left.get("id", ""))
            if name != "":
                _narrow_to_non_none(name, then_env, ctx)
        if _is_type(left, ["Attribute"]):
            path = _attr_path(left)
            if path != "":
                then_env.guard_attr(path)
        return
    if op_type == "Is" and _is_type(comp, ["Constant"]) and comp.get("value") is None:
        if _is_type(left, ["Attribute"]):
            path = _attr_path(left)
            if path != "":
                else_env.guard_attr(path)
        return
    if op_type == "Eq":
        if _is_type(left, ["Attribute"]):
            attr = left.get("attr")
            if isinstance(attr, str) and attr == "kind":
                comp_value = comp.get("value")
                if isinstance(comp_value, str):
                    obj_node = left.get("value")
                    is_union = False
                    if isinstance(obj_node, dict) and _is_type(obj_node, ["Name"]):
                        obj_name = obj_node.get("id")
                        if isinstance(obj_name, str):
                            src = then_env.get_source(obj_name)
                            if src != "" and _is_union_source(src):
                                is_union = True
                    if is_union:
                        found = False
                        all_classes = list(ctx.known_classes)
                        j = 0
                        while j < len(all_classes):
                            if all_classes[j].lower() == comp_value.lower():
                                found = True
                                break
                            j += 1
                        if not found:
                            k_lineno = test.get("lineno", 0)
                            if not isinstance(k_lineno, int):
                                k_lineno = 0
                            ctx.result.add_error(
                                k_lineno,
                                0,
                                "kind value '"
                                + comp_value
                                + "' does not match any known type",
                            )
                return


def _narrow_to_non_none(name: str, env: TypeEnv, ctx: _InferCtx) -> None:
    """Narrow a variable to its non-None part."""
    typ = env.get_type(name)
    source = env.get_source(name)
    if typ is not None and isinstance(typ, OptionalType):
        env.narrow(name, typ.inner, _type_name(typ.inner))
        return
    if source != "" and _is_optional_source(source):
        non_none = _non_none_parts(source)
        if len(non_none) == 1:
            sig_errors: list[SignatureError] = []
            narrowed = py_type_to_type_dict(
                non_none[0], ctx.known_classes, sig_errors, 0, 0
            )
            env.narrow(name, narrowed, non_none[0])
        elif len(non_none) > 1:
            new_source = " | ".join(non_none)
            sig_errors3: list[SignatureError] = []
            narrowed = py_type_to_type_dict(
                new_source, ctx.known_classes, sig_errors3, 0, 0
            )
            env.narrow(name, narrowed, new_source)


def _narrow_or_isinstance(
    values: list[object], then_env: TypeEnv, ctx: _InferCtx
) -> None:
    """Handle isinstance(x,A) or isinstance(x,B) in then branch."""
    pass


# ---------------------------------------------------------------------------
# Iterator/generator escape checking
# ---------------------------------------------------------------------------


def _is_iterator_call(node: ASTNode) -> str:
    """If node is enumerate/zip/reversed call, return the func name."""
    if not _is_type(node, ["Call"]):
        return ""
    func = node.get("func")
    if isinstance(func, dict) and _is_type(func, ["Name"]):
        fname = func.get("id")
        if isinstance(fname, str) and fname in _ITERATOR_FUNCS:
            return fname
    return ""


def _is_generator_expr(node: ASTNode) -> bool:
    return _is_type(node, ["GeneratorExp"])


def _check_iterator_escape_assign(
    value: ASTNode, env: TypeEnv, ctx: _InferCtx, lineno: int
) -> bool:
    fname = _is_iterator_call(value)
    if fname != "":
        ctx.result.add_error(lineno, 0, "cannot assign " + fname + "() to variable")
        return True
    return False


def _check_iterator_escape_return(
    value: ASTNode, env: TypeEnv, ctx: _InferCtx, lineno: int
) -> bool:
    fname = _is_iterator_call(value)
    if fname != "":
        ctx.result.add_error(lineno, 0, "cannot return " + fname + "()")
        return True
    if _is_type(value, ["Call"]):
        func = value.get("func")
        if isinstance(func, dict) and _is_type(func, ["Name"]):
            wrapper = func.get("id")
            if isinstance(wrapper, str) and wrapper in _EAGER_CONSUMERS:
                return False
    return False


def _check_iterator_escape_arg(
    arg: ASTNode, caller: str, env: TypeEnv, ctx: _InferCtx, lineno: int
) -> None:
    fname = _is_iterator_call(arg)
    if fname != "" and caller not in _EAGER_CONSUMERS:
        ctx.result.add_error(
            lineno, 0, "cannot pass " + fname + "() to non-consumer function"
        )


def _check_generator_escape_assign(
    value: ASTNode, env: TypeEnv, ctx: _InferCtx, lineno: int
) -> bool:
    if _is_generator_expr(value):
        ctx.result.add_error(
            lineno, 0, "cannot assign generator expression to variable"
        )
        return True
    return False


def _check_generator_escape_return(
    value: ASTNode, env: TypeEnv, ctx: _InferCtx, lineno: int
) -> bool:
    if _is_generator_expr(value):
        ctx.result.add_error(lineno, 0, "cannot return generator expression")
        return True
    if _is_type(value, ["Call"]):
        func = value.get("func")
        args = value.get("args", [])
        if (
            isinstance(func, dict)
            and _is_type(func, ["Name"])
            and isinstance(args, list)
        ):
            wrapper = func.get("id")
            if isinstance(wrapper, str) and wrapper in _EAGER_CONSUMERS:
                return False
        if isinstance(func, dict) and _is_type(func, ["Attribute"]):
            attr = func.get("attr")
            if isinstance(attr, str) and attr == "join":
                return False
        if isinstance(args, list):
            j = 0
            while j < len(args):
                a = args[j]
                if isinstance(a, dict) and _is_generator_expr(a):
                    if isinstance(func, dict) and _is_type(func, ["Name"]):
                        wrapper = func.get("id")
                        if isinstance(wrapper, str) and wrapper in _EAGER_CONSUMERS:
                            return False
                    ctx.result.add_error(
                        lineno, 0, "cannot return generator expression"
                    )
                    return True
                j += 1
    return False


def _check_generator_escape_arg(
    arg: ASTNode, caller: str, env: TypeEnv, ctx: _InferCtx, lineno: int
) -> None:
    if _is_generator_expr(arg) and caller not in _EAGER_CONSUMERS and caller != "join":
        ctx.result.add_error(
            lineno, 0, "cannot pass generator expression to non-consumer"
        )


# ---------------------------------------------------------------------------
# Literal validation
# ---------------------------------------------------------------------------


def _validate_list_literal(
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
    expected: TypeNode | None = None,
) -> None:
    """Check list literal for mixed types."""
    if expected is not None and isinstance(expected, SliceType):
        if is_any(expected.element):
            return
    exp_elem: TypeNode | None = None
    if expected is not None and isinstance(expected, SliceType):
        exp_elem = expected.element
    elts = node.get("elts", [])
    if not isinstance(elts, list) or len(elts) < 2:
        return
    first = elts[0]
    if not isinstance(first, dict):
        return
    first_type = _synth_expr(first, env, ctx)
    j = 1
    while j < len(elts):
        e = elts[j]
        if isinstance(e, dict):
            et = _synth_expr(e, env, ctx)
            if exp_elem is not None:
                if not _is_assignable(et, exp_elem, ctx.hier_result):
                    ctx.result.add_error(
                        lineno,
                        0,
                        "mixed types in list literal: "
                        + _type_name(et)
                        + " not assignable to "
                        + _type_name(exp_elem),
                    )
                    return
            elif not _is_assignable(
                et, first_type, ctx.hier_result
            ) and not _is_assignable(first_type, et, ctx.hier_result):
                ctx.result.add_error(
                    lineno,
                    0,
                    "mixed types in list literal: "
                    + _type_name(first_type)
                    + " and "
                    + _type_name(et),
                )
                return
        j += 1


def _validate_dict_literal(
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
    expected: TypeNode | None,
) -> None:
    """Check dict literal for mixed key/value types."""
    if expected is not None:
        check_exp = expected
        if isinstance(check_exp, OptionalType):
            check_exp = check_exp.inner
        if isinstance(check_exp, MapType):
            if is_any(check_exp.value):
                return
    keys = node.get("keys", [])
    values = node.get("values", [])
    if not isinstance(keys, list) or not isinstance(values, list) or len(keys) < 2:
        return
    first_k = keys[0]
    first_v = values[0]
    if not isinstance(first_k, dict) or not isinstance(first_v, dict):
        return
    first_kt = _synth_expr(first_k, env, ctx)
    first_vt = _synth_expr(first_v, env, ctx)
    j = 1
    while j < len(keys):
        k = keys[j]
        v = values[j]
        if isinstance(k, dict):
            kt = _synth_expr(k, env, ctx)
            if not _is_assignable(kt, first_kt, ctx.hier_result):
                ctx.result.add_error(lineno, 0, "mixed key types in dict literal")
                return
        if isinstance(v, dict):
            vt = _synth_expr(v, env, ctx)
            if not _is_assignable(vt, first_vt, ctx.hier_result):
                ctx.result.add_error(lineno, 0, "mixed value types in dict literal")
                return
        j += 1


# ---------------------------------------------------------------------------
# Additional validation during return
# ---------------------------------------------------------------------------


def _validate_return_value(
    value: ASTNode,
    expected: TypeNode,
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Extra validation on return expressions: literal checking, etc."""
    if _is_type(value, ["List"]):
        _validate_list_literal(value, env, ctx, lineno, expected)
    if _is_type(value, ["Dict"]):
        _validate_dict_literal(value, env, ctx, lineno, expected)


# ---------------------------------------------------------------------------
# Object / union access validation
# ---------------------------------------------------------------------------


def _check_needs_narrowing(
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
    context: str,
    attr_name: str,
) -> None:
    """Check if a Name node refers to a variable that needs narrowing."""
    if not _is_type(node, ["Name"]):
        return
    name = node.get("id")
    if not isinstance(name, str):
        return
    source = env.get_source(name)
    typ = env.get_type(name)
    if typ is None:
        return
    if isinstance(typ, OptionalType):
        if context == "arithmetic":
            ctx.result.add_error(
                lineno, 0, "cannot use optional type in arithmetic (may be None)"
            )
        elif context == "attribute":
            ctx.result.add_error(
                lineno,
                0,
                "cannot access '" + attr_name + "' on optional type (may be None)",
            )
        elif context == "subscript":
            ctx.result.add_error(
                lineno, 0, "cannot subscript optional type (may be None)"
            )
        return
    if source != "" and _is_optional_source(source):
        if context == "arithmetic":
            ctx.result.add_error(
                lineno, 0, "cannot use " + source + " in arithmetic (may be None)"
            )
        elif context == "attribute":
            ctx.result.add_error(
                lineno,
                0,
                "cannot access '" + attr_name + "' on optional type (may be None)",
            )
        elif context == "subscript":
            ctx.result.add_error(
                lineno, 0, "cannot subscript optional type (may be None)"
            )
        return
    if source != "" and _is_union_source(source):
        if context == "attribute" and attr_name != "":
            if _all_members_have_attr(source, attr_name, ctx):
                return
            ctx.result.add_error(
                lineno,
                0,
                "attribute '" + attr_name + "' not available on all union members",
            )
            return
        if context == "arithmetic":
            ctx.result.add_error(
                lineno, 0, "cannot use union type in arithmetic without narrowing"
            )
        elif context == "attribute":
            ctx.result.add_error(
                lineno, 0, "cannot access attribute on union type without narrowing"
            )
        elif context == "subscript":
            ctx.result.add_error(
                lineno, 0, "cannot subscript union type without narrowing"
            )
        return
    if source == "object":
        if context == "arithmetic":
            ctx.result.add_error(
                lineno, 0, "cannot use object in arithmetic without narrowing"
            )
        elif context == "attribute":
            ctx.result.add_error(
                lineno, 0, "cannot access attribute on object without narrowing"
            )
        elif context == "subscript":
            ctx.result.add_error(lineno, 0, "cannot subscript object without narrowing")
        return


def _validate_expr_access(
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Check for un-narrowed access on object/union/optional types in an expression."""
    if not isinstance(node, dict):
        return
    if len(ctx.result._errors) > 0:
        return
    t = node.get("_type")
    if t == "BinOp":
        left = node.get("left")
        right = node.get("right")
        if isinstance(left, dict):
            _check_needs_narrowing(left, env, ctx, lineno, "arithmetic", "")
        if len(ctx.result._errors) > 0:
            return
        if isinstance(right, dict):
            _check_needs_narrowing(right, env, ctx, lineno, "arithmetic", "")
        if len(ctx.result._errors) > 0:
            return
        if isinstance(left, dict):
            _validate_expr_access(left, env, ctx, lineno)
        if isinstance(right, dict):
            _validate_expr_access(right, env, ctx, lineno)
        return
    if t == "Attribute":
        value = node.get("value")
        attr = node.get("attr")
        attr_str = ""
        if isinstance(attr, str):
            attr_str = attr
        if isinstance(value, dict) and attr_str != "kind":
            _check_needs_narrowing(value, env, ctx, lineno, "attribute", attr_str)
        if isinstance(value, dict):
            _validate_expr_access(value, env, ctx, lineno)
        return
    if t == "Subscript":
        value = node.get("value")
        if isinstance(value, dict):
            _check_needs_narrowing(value, env, ctx, lineno, "subscript", "")
            _validate_expr_access(value, env, ctx, lineno)
        return
    if t == "Call":
        func = node.get("func")
        if isinstance(func, dict):
            _validate_expr_access(func, env, ctx, lineno)
        if len(ctx.result._errors) > 0:
            return
        args = node.get("args", [])
        if isinstance(args, list):
            j = 0
            while j < len(args):
                arg = args[j]
                if isinstance(arg, dict):
                    _validate_expr_access(arg, env, ctx, lineno)
                if len(ctx.result._errors) > 0:
                    return
                j += 1
        return


def _subclass_has_method(base_name: str, method_name: str, ctx: _InferCtx) -> bool:
    """Check if any subclass of base_name has the given method."""
    all_classes = list(ctx.class_bases.keys())
    i = 0
    while i < len(all_classes):
        cls = all_classes[i]
        bases = ctx.class_bases.get(cls, [])
        j = 0
        while j < len(bases):
            if bases[j] == base_name:
                methods = ctx.sig_result.methods.get(cls)
                if methods is not None and method_name in methods:
                    return True
            j += 1
        i += 1
    return False


def _all_members_have_attr(source: str, attr_name: str, ctx: _InferCtx) -> bool:
    """Check if all union members have the given attribute (field, const_field, or method)."""
    parts = _split_union_parts(source)
    i = 0
    while i < len(parts):
        p = parts[i]
        if p == "None":
            i += 1
            continue
        found = False
        cls = ctx.field_result.classes.get(p)
        if cls is not None:
            if attr_name in cls.fields or attr_name in cls.const_fields:
                found = True
        if not found:
            methods = ctx.sig_result.methods.get(p)
            if methods is not None and attr_name in methods:
                found = True
        if not found:
            return False
        i += 1
    return len(parts) > 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_inference(
    tree: ASTNode,
    sig_result: SignatureResult,
    field_result: FieldResult,
    hier_result: HierarchyResult,
    known_classes: set[str],
    class_bases: dict[str, list[str]],
) -> InferenceResult:
    """Run type inference and validation on the module AST."""
    result = InferenceResult()
    ctx = _InferCtx(
        sig_result, field_result, hier_result, known_classes, class_bases, result
    )
    body = tree.get("body", [])
    if not isinstance(body, list):
        return result
    i = 0
    while i < len(body):
        node = body[i]
        if isinstance(node, dict) and node.get("_type") == "AnnAssign":
            target = node.get("target")
            annotation = node.get("annotation")
            if (
                isinstance(target, dict)
                and target.get("_type") == "Name"
                and isinstance(annotation, dict)
            ):
                var_name = target.get("id")
                if isinstance(var_name, str):
                    ann_str = annotation_to_str(annotation)
                    if ann_str != "":
                        sig_errors: list[SignatureError] = []
                        var_type = py_type_to_type_dict(
                            ann_str, known_classes, sig_errors, 0, 0
                        )
                        if len(sig_errors) == 0:
                            ctx.module_vars[var_name] = var_type
        i += 1
    i = 0
    while i < len(body):
        node = body[i]
        if not isinstance(node, dict):
            i += 1
            continue
        t = node.get("_type")
        sf = node.get("_source_file", "")
        if not isinstance(sf, str):
            sf = ""
        if t == "FunctionDef":
            err_before = len(result._errors)
            _validate_func(node, ctx, "")
            ei = err_before
            while ei < len(result._errors):
                result._errors[ei].source_file = sf
                ei += 1
            if len(result._errors) > 0:
                return result
        elif t == "ClassDef":
            class_name = node.get("name", "")
            if not isinstance(class_name, str):
                class_name = ""
            class_body = node.get("body", [])
            if isinstance(class_body, list):
                j = 0
                while j < len(class_body):
                    stmt = class_body[j]
                    if isinstance(stmt, dict) and _is_type(stmt, ["FunctionDef"]):
                        stmt_sf = stmt.get("_source_file", "")
                        if not isinstance(stmt_sf, str):
                            stmt_sf = ""
                        if stmt_sf == "":
                            stmt_sf = sf
                        err_before = len(result._errors)
                        _validate_func(stmt, ctx, class_name)
                        ei = err_before
                        while ei < len(result._errors):
                            result._errors[ei].source_file = stmt_sf
                            ei += 1
                        if len(result._errors) > 0:
                            return result
                    j += 1
        i += 1
    return result
