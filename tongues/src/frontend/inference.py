"""Phase 9: Type inference and validation.

Bidirectional type inference with flow-sensitive narrowing. Computes types for
all expressions, infers local variable types from assignments, enforces type
safety constraints, and validates iterator/generator consumption.

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from __future__ import annotations


from .cfg import FlowGraph, lookup_alias
from .typecollect import (
    FuncInfo,
    TypeCollectResult,
    annotation_to_str,
    py_type_to_type_dict,
    TypeCollectError,
)
from .hierarchy import HierarchyResult
from .types import (
    TypeNode,
    PrimitiveType,
    SliceType,
    MapType,
    SetType,
    TupleType,
    OptionalType,
    UnionType,
    PointerType,
    StructRef,
    InterfaceRef,
    FuncType,
    IteratorType,
    LiteralType,
    ANY_TYPE,
    INT_TYPE,
    FLOAT_TYPE,
    BOOL_TYPE,
    STR_TYPE,
    VOID_TYPE,
    BYTES_TYPE,
    is_any,
    type_eq as _type_eq,
    combine_types,
    remove_from_union,
    union_variant_names,
    _variant_name,
    map_subtypes,
    get_subtypes,
    type_name as _type_name_fn,
    JStr,
    JInt,
    JBool,
    JFloat,
    JNull,
    ASTNode,
    get_str,
    get_int,
    get_bool,
    get_node,
    get_nodes,
)


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
        self._reveals: list[tuple[int, str]] = []

    def add_error(
        self, lineno: int, col: int, message: str, source_file: str = ""
    ) -> None:
        self._errors.append(InferenceError(lineno, col, message, source_file))

    def add_reveal(self, lineno: int, type_str: str) -> None:
        self._reveals.append((lineno, type_str))

    def errors(self) -> list[InferenceError]:
        return self._errors

    def reveals(self) -> list[tuple[int, str]]:
        return self._reveals


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def _is_null_value(node: ASTNode) -> bool:
    """Check if a Constant node has a None/null value."""
    v = node.get("value")
    return v is None or isinstance(v, JNull)


def _is_type(node: ASTNode, type_names: list[str]) -> bool:
    if not isinstance(node, dict):
        return False
    t = get_str(node, "_type")
    i = 0
    while i < len(type_names):
        if t == type_names[i]:
            return True
        i += 1
    return False


def _type_name(t: TypeNode) -> str:
    return _type_name_fn(t)


def _prim_kind(t: TypeNode) -> str:
    """Extract primitive kind from PrimitiveType or LiteralType."""
    if isinstance(t, PrimitiveType):
        return t.kind
    if isinstance(t, LiteralType):
        return t.base.kind
    return ""


def _extract_literal(node: ASTNode) -> LiteralType | None:
    """Extract a LiteralType from a Constant AST node."""
    v = node.get("value")
    if isinstance(v, JStr):
        if not get_bool(node, "_is_bytes"):
            return LiteralType(v.value, PrimitiveType("string"))
    if isinstance(v, JInt):
        return LiteralType(str(v.value), PrimitiveType("int"))
    if isinstance(v, JBool):
        val = "true" if v.value else "false"
        return LiteralType(val, PrimitiveType("bool"))
    return None


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
    if isinstance(actual, PrimitiveType) and actual.kind == "never":
        return True
    # void (None literal) assignable to Optional
    if isinstance(actual, PrimitiveType) and actual.kind == "void":
        if isinstance(expected, OptionalType):
            return True
        if isinstance(expected, InterfaceRef):
            return True
        return False
    # LiteralType delegates to its base
    if isinstance(actual, LiteralType):
        return _is_assignable(actual.base, expected, hier)
    # bool <: int <: float, byte <: int
    if isinstance(actual, PrimitiveType) and isinstance(expected, PrimitiveType):
        if actual.kind == "bool" and expected.kind == "int":
            return True
        if actual.kind == "bool" and expected.kind == "float":
            return True
        if actual.kind == "int" and expected.kind == "float":
            return True
        if actual.kind == "byte" and expected.kind == "int":
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
    # T assignable to Union if assignable to any variant
    if isinstance(expected, UnionType):
        ev = get_subtypes(expected)
        j = 0
        while j < len(ev):
            if _is_assignable(actual, ev[j], hier):
                return True
            j += 1
    # Union assignable to T if every variant is assignable
    if isinstance(actual, UnionType):
        av = get_subtypes(actual)
        j = 0
        while j < len(av):
            if not _is_assignable(av[j], expected, hier):
                return False
            j += 1
        return True
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
# Type environment
# ---------------------------------------------------------------------------


class CondAlias:
    """A condition alias: flag = isinstance(x, T) or x is None, etc."""

    def __init__(
        self, target: str, narrow_type: str, type_name: str, field_name: str
    ) -> None:
        self.target: str = target
        self.narrow_type: str = narrow_type
        self.type_name: str = type_name
        self.field_name: str = field_name


class TypeEnv:
    """Flow-sensitive type environment for a function body."""

    def __init__(self) -> None:
        self.types: dict[str, TypeNode] = {}
        self.guarded_attrs: set[str] = set()

    def copy(self) -> TypeEnv:
        env = TypeEnv()
        tkeys = list(self.types.keys())
        i = 0
        while i < len(tkeys):
            env.types[tkeys[i]] = self.types[tkeys[i]]
            i += 1
        gkeys = list(self.guarded_attrs)
        i = 0
        while i < len(gkeys):
            env.guarded_attrs.add(gkeys[i])
            i += 1
        return env

    def set(self, name: str, typ: TypeNode) -> None:
        self.types[name] = typ

    def get_type(self, name: str) -> TypeNode | None:
        return self.types.get(name)

    def narrow(self, name: str, typ: TypeNode) -> None:
        self.types[name] = typ

    def guard_attr(self, path: str) -> None:
        self.guarded_attrs.add(path)

    def is_attr_guarded(self, path: str) -> bool:
        return path in self.guarded_attrs

    def evict_guarded_attrs(self, name: str) -> None:
        """Remove all guarded/narrowed attr paths rooted at name."""
        prefix = name + "."
        to_remove: list[str] = []
        keys = list(self.guarded_attrs)
        i = 0
        while i < len(keys):
            k = keys[i]
            if k == name or k.startswith(prefix):
                to_remove.append(k)
            i += 1
        j = 0
        while j < len(to_remove):
            self.guarded_attrs.discard(to_remove[j])
            j += 1
        type_keys = list(self.types.keys())
        i = 0
        while i < len(type_keys):
            k = type_keys[i]
            if k.startswith(prefix):
                self.types.pop(k)
            i += 1


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

_BUILTIN_FUNCS: set[str] = {
    "len",
    "int",
    "str",
    "float",
    "bool",
    "abs",
    "ord",
    "chr",
    "repr",
    "hash",
    "range",
    "print",
    "sum",
    "min",
    "max",
    "sorted",
    "list",
    "tuple",
    "set",
    "dict",
    "type",
}


def _synth_expr(
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
) -> TypeNode:
    """Synthesize the type of an expression node."""
    if not isinstance(node, dict):
        return ANY_TYPE
    t = get_str(node, "_type")
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
    if v is None or isinstance(v, JNull):
        return VOID_TYPE
    if isinstance(v, JBool):
        return BOOL_TYPE
    if isinstance(v, JInt):
        return INT_TYPE
    if isinstance(v, JFloat):
        return FLOAT_TYPE
    if isinstance(v, JStr):
        if get_bool(node, "_is_bytes"):
            return BYTES_TYPE
        return STR_TYPE
    return ANY_TYPE


def _synth_name(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    name = get_str(node, "id")
    if name == "":
        return ANY_TYPE
    typ = env.get_type(name)
    if typ is not None:
        return typ
    # User-defined function reference -> FuncType
    func_info = ctx.tc_result.functions.get(name)
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
    if name == "pow":
        return FuncType([INT_TYPE, INT_TYPE], INT_TYPE)
    # Module-level variable
    mod_var = ctx.module_vars.get(name)
    if mod_var is not None:
        return mod_var
    return ANY_TYPE


def _synth_attribute(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    value = get_node(node, "value")
    attr = get_str(node, "attr")
    if len(value) == 0 or attr == "":
        return ANY_TYPE
    if _is_type(value, ["Name"]) and get_str(value, "id") == "sys" and attr == "argv":
        return SliceType(STR_TYPE)
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
    if _prim_kind(obj_type) == "string":
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
            return FuncType([], SetType(key_t))
        if attr == "values":
            return FuncType([], SliceType(val_t))
        if attr == "items":
            items_elem = TupleType([key_t, val_t], False)
            return FuncType([], SetType(items_elem))
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
    # Union: resolve on each variant
    if isinstance(obj_type, UnionType):
        subs = get_subtypes(obj_type)
        resolved: list[TypeNode] = []
        i = 0
        while i < len(subs):
            r = _resolve_attr(subs[i], attr, value_node, env, ctx)
            if is_any(r):
                return ANY_TYPE
            resolved.append(r)
            i += 1
        if len(resolved) == 0:
            return ANY_TYPE
        return map_subtypes(obj_type, resolved)
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
    cls = ctx.tc_result.classes.get(sname)
    if cls is not None:
        fld = cls.fields.get(attr)
        if fld is not None:
            return fld.typ
        if attr in cls.const_fields:
            return LiteralType(cls.const_fields[attr], PrimitiveType("string"))
    methods = ctx.tc_result.methods.get(sname)
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
    func = get_node(node, "func")
    if len(func) == 0:
        return ANY_TYPE
    args = get_nodes(node, "args")
    # Direct name call
    if _is_type(func, ["Name"]):
        fname = get_str(func, "id")
        if fname != "":
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
    args: list[ASTNode],
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
            ft = _synth_expr(first, env, ctx)
            if isinstance(ft, SliceType):
                return ft.element
            if len(args) >= 2:
                has_int = False
                has_bool = False
                j = 0
                while j < len(args):
                    a = args[j]
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
    if fname == "pow":
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
            ft = _synth_expr(first, env, ctx)
            elem = _element_type(ft)
            return IteratorType(TupleType([INT_TYPE, elem], False))
        return IteratorType(TupleType([INT_TYPE, ANY_TYPE], False))
    if fname == "zip":
        elems: list[TypeNode] = []
        j = 0
        while j < len(args):
            a = args[j]
            ft = _synth_expr(a, env, ctx)
            elems.append(_element_type(ft))
            j += 1
        return IteratorType(TupleType(elems, False))
    if fname == "reversed":
        if len(args) > 0:
            first = args[0]
            ft = _synth_expr(first, env, ctx)
            elem = _element_type(ft)
            return IteratorType(elem)
        return IteratorType(ANY_TYPE)
    if fname == "sorted":
        if len(args) > 0:
            first = args[0]
            ft = _synth_expr(first, env, ctx)
            if isinstance(ft, IteratorType):
                return SliceType(ft.element)
            elem = _element_type(ft)
            return SliceType(elem)
        return SliceType(ANY_TYPE)
    if fname == "list":
        if len(args) > 0:
            first = args[0]
            ft = _synth_expr(first, env, ctx)
            if isinstance(ft, IteratorType):
                return SliceType(ft.element)
            elem = _element_type(ft)
            return SliceType(elem)
        return SliceType(ANY_TYPE)
    if fname == "tuple":
        if len(args) > 0:
            first = args[0]
            ft = _synth_expr(first, env, ctx)
            if isinstance(ft, IteratorType):
                return TupleType([ft.element], True)
            elem = _element_type(ft)
            return TupleType([elem], True)
        return TupleType([], False)
    if fname == "set":
        if len(args) > 0:
            first = args[0]
            ft = _synth_expr(first, env, ctx)
            elem = _element_type(ft)
            return SetType(elem)
        return SetType(ANY_TYPE)
    if fname == "dict":
        if len(args) > 0:
            first = args[0]
            ft = _synth_expr(first, env, ctx)
            if isinstance(ft, IteratorType):
                elem = ft.element
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
    func_info = ctx.tc_result.functions.get(fname)
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
    args: list[ASTNode],
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
) -> TypeNode:
    """Synthesize return type of a method call (obj.method(...))."""
    obj = get_node(func, "value")
    attr = get_str(func, "attr")
    if len(obj) == 0 or attr == "":
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
    if isinstance(attr_type, UnionType):
        ret_types: list[TypeNode] = []
        all_func = True
        ui = 0
        while ui < len(attr_type.variants):
            v = attr_type.variants[ui]
            if isinstance(v, FuncType):
                ret_types.append(v.ret)
            else:
                all_func = False
            ui += 1
        if all_func and len(ret_types) > 0:
            return combine_types(ret_types)
    # Direct method return type from sig table
    sname = _struct_name(obj_type)
    if sname != "":
        methods = ctx.tc_result.methods.get(sname)
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
    if _prim_kind(t) == "string":
        return STR_TYPE
    return ANY_TYPE


def _synth_subscript(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    value = get_node(node, "value")
    slc = get_node(node, "slice")
    if len(value) == 0:
        return ANY_TYPE
    obj_type = _synth_expr(value, env, ctx)
    # String indexing
    if _prim_kind(obj_type) == "string":
        return STR_TYPE
    # List indexing
    if isinstance(obj_type, SliceType):
        if len(slc) > 0 and _is_type(slc, ["Slice"]):
            return obj_type
        return obj_type.element
    # Dict indexing
    if isinstance(obj_type, MapType):
        return obj_type.value
    # Tuple indexing
    if isinstance(obj_type, TupleType):
        if obj_type.variadic and len(obj_type.elements) > 0:
            return obj_type.elements[0]
        if len(slc) > 0 and _is_type(slc, ["Constant"]):
            slc_v = slc.get("value")
            if isinstance(slc_v, JInt):
                idx = slc_v.value
                if 0 <= idx < len(obj_type.elements):
                    return obj_type.elements[idx]
                elif idx < 0 and -idx <= len(obj_type.elements):
                    return obj_type.elements[len(obj_type.elements) + idx]
                else:
                    t_lineno = get_int(node, "lineno")
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
    left = get_node(node, "left")
    right = get_node(node, "right")
    op = get_node(node, "op")
    if len(left) == 0 or len(right) == 0:
        return ANY_TYPE
    lt = _synth_expr(left, env, ctx)
    rt = _synth_expr(right, env, ctx)
    op_type = get_str(op, "_type")
    # String concatenation
    if _prim_kind(lt) == "string" and _prim_kind(rt) == "string":
        return STR_TYPE
    # Set operations (&, |, ^)
    if isinstance(lt, SetType) and isinstance(rt, SetType):
        if op_type in ("BitAnd", "BitOr", "BitXor"):
            return lt
    # List concatenation
    if isinstance(lt, SliceType) and isinstance(rt, SliceType):
        if not _is_assignable(
            lt.element, rt.element, ctx.hier_result
        ) and not _is_assignable(rt.element, lt.element, ctx.hier_result):
            b_lineno = get_int(node, "lineno")
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
    lt_num = _prim_kind(lt) in ("int", "float", "bool")
    rt_num = _prim_kind(rt) in ("int", "float", "bool")
    if lt_num and rt_num:
        if op_type in ("BitAnd", "BitOr", "BitXor"):
            if _prim_kind(lt) == "bool" and _prim_kind(rt) == "bool":
                return BOOL_TYPE
            return INT_TYPE
        if _prim_kind(lt) == "float" or _prim_kind(rt) == "float":
            return FLOAT_TYPE
        return INT_TYPE
    # String * int
    lk = _prim_kind(lt)
    rk = _prim_kind(rt)
    if lk == "string" and (rk == "int" or rk == "bool"):
        return STR_TYPE
    if (lk == "int" or lk == "bool") and rk == "string":
        return STR_TYPE
    return ANY_TYPE


def _synth_unaryop(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    operand = get_node(node, "operand")
    op = get_node(node, "op")
    if len(operand) == 0:
        return ANY_TYPE
    ot = _synth_expr(operand, env, ctx)
    op_type = get_str(op, "_type")
    if op_type == "Not":
        return BOOL_TYPE
    if op_type == "USub" or op_type == "UAdd":
        if _prim_kind(ot) == "bool":
            return INT_TYPE
        return ot
    if op_type == "Invert":
        return INT_TYPE
    return ot


def _synth_boolop(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    values = get_nodes(node, "values")
    if len(values) == 0:
        return ANY_TYPE
    op = get_node(node, "op")
    op_t = get_str(op, "_type")
    if op_t == "And":
        work = env.copy()
        j = 0
        while j < len(values) - 1:
            dummy = env.copy()
            _extract_narrowing(values[j], work, dummy, ctx)
            j += 1
        return _synth_expr(values[len(values) - 1], work, ctx)
    if op_t == "Or":
        work = env.copy()
        j = 0
        while j < len(values) - 1:
            dummy = env.copy()
            _extract_narrowing(values[j], dummy, work, ctx)
            j += 1
        return _synth_expr(values[len(values) - 1], work, ctx)
    return _synth_expr(values[len(values) - 1], env, ctx)


def _synth_ifexp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    test = get_node(node, "test")
    body = get_node(node, "body")
    then_env = env.copy()
    if len(test) > 0:
        dummy_else = env.copy()
        _extract_narrowing(test, then_env, dummy_else, ctx)
    if len(body) > 0:
        return _synth_expr(body, then_env, ctx)
    return ANY_TYPE


def _synth_list(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    elts = get_nodes(node, "elts")
    if len(elts) == 0:
        return SliceType(ANY_TYPE)
    first = elts[0]
    return SliceType(_synth_expr(first, env, ctx))


def _synth_dict(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    keys = get_nodes(node, "keys")
    values = get_nodes(node, "values")
    if len(keys) == 0:
        return MapType(ANY_TYPE, ANY_TYPE)
    k = keys[0]
    v = values[0]
    kt = _synth_expr(k, env, ctx)
    vt = ANY_TYPE
    if len(values) > 0:
        vt = _synth_expr(v, env, ctx)
    return MapType(kt, vt)


def _synth_set(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    elts = get_nodes(node, "elts")
    if len(elts) == 0:
        return SetType(ANY_TYPE)
    first = elts[0]
    return SetType(_synth_expr(first, env, ctx))


def _synth_tuple(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    elts = get_nodes(node, "elts")
    elems: list[TypeNode] = []
    i = 0
    while i < len(elts):
        elems.append(_synth_expr(elts[i], env, ctx))
        i += 1
    return TupleType(elems, False)


def _synth_listcomp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    elt = get_node(node, "elt")
    generators = get_nodes(node, "generators")
    comp_env = env.copy()
    _bind_comprehension_vars(generators, comp_env, ctx)
    if len(elt) > 0:
        lineno = get_int(node, "lineno")
        if lineno == 0 and len(generators) > 0:
            lineno = get_int(generators[0], "lineno")
        err_snap = len(ctx.result._errors)
        _validate_expr_access(elt, comp_env, ctx, lineno)
        if _has_new_errors(ctx, err_snap):
            return SliceType(ANY_TYPE)
        return SliceType(_synth_expr(elt, comp_env, ctx))
    return SliceType(ANY_TYPE)


def _synth_setcomp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    elt = get_node(node, "elt")
    generators = get_nodes(node, "generators")
    comp_env = env.copy()
    _bind_comprehension_vars(generators, comp_env, ctx)
    if len(elt) > 0:
        return SetType(_synth_expr(elt, comp_env, ctx))
    return SetType(ANY_TYPE)


def _synth_dictcomp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    key = get_node(node, "key")
    value = get_node(node, "value")
    generators = get_nodes(node, "generators")
    comp_env = env.copy()
    _bind_comprehension_vars(generators, comp_env, ctx)
    kt = ANY_TYPE
    vt = ANY_TYPE
    if len(key) > 0:
        kt = _synth_expr(key, comp_env, ctx)
    if len(value) > 0:
        vt = _synth_expr(value, comp_env, ctx)
    return MapType(kt, vt)


def _bind_comprehension_vars(
    generators: list[ASTNode], env: TypeEnv, ctx: _InferCtx
) -> None:
    """Bind iteration variables from comprehension generators."""
    i = 0
    while i < len(generators):
        gen = generators[i]
        target = get_node(gen, "target")
        iter_node = get_node(gen, "iter")
        if len(iter_node) > 0:
            iter_type = _synth_expr(iter_node, env, ctx)
            elem = _iteration_element(iter_type)
            _bind_target(target, elem, env)
        ifs = get_nodes(gen, "ifs")
        j = 0
        while j < len(ifs):
            cond = ifs[j]
            dummy_else = env.copy()
            _extract_narrowing(cond, env, dummy_else, ctx)
            j += 1
        i += 1


def _synth_namedexpr(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> TypeNode:
    target = get_node(node, "target")
    value = get_node(node, "value")
    if len(value) == 0:
        return ANY_TYPE
    vt = _synth_expr(value, env, ctx)
    if len(target) > 0 and _is_type(target, ["Name"]):
        name = get_str(target, "id")
        if name != "":
            env.set(name, vt)
    return vt


# ---------------------------------------------------------------------------
# Iteration element type
# ---------------------------------------------------------------------------


def _iteration_element(t: TypeNode) -> TypeNode:
    """Get the element type when iterating over a type."""
    if isinstance(t, IteratorType):
        return t.element
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


def _bind_target(target: ASTNode, typ: TypeNode, env: TypeEnv) -> None:
    """Bind an assignment target (Name or Tuple) to a type."""
    if not isinstance(target, dict):
        return
    if _is_type(target, ["Name"]):
        name = get_str(target, "id")
        if name != "":
            env.set(name, typ)
    elif _is_type(target, ["Tuple", "List"]):
        elts = get_nodes(target, "elts")
        if isinstance(typ, TupleType):
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
        tc_result: TypeCollectResult,
        hier_result: HierarchyResult,
        known_classes: set[str],
        class_bases: dict[str, list[str]],
        result: InferenceResult,
        flow_graphs: dict[str, FlowGraph],
    ) -> None:
        self.tc_result: TypeCollectResult = tc_result
        self.hier_result: HierarchyResult = hier_result
        self.known_classes: set[str] = known_classes
        self.class_bases: dict[str, list[str]] = class_bases
        self.result: InferenceResult = result
        self.flow_graphs: dict[str, FlowGraph] = flow_graphs
        self.current_graph: FlowGraph | None = None
        self.module_vars: dict[str, TypeNode] = {}
        self._func_err_start: int = 0
        self._func_err_limit: int = 0


def _has_new_errors(ctx: _InferCtx, snapshot: int) -> bool:
    return len(ctx.result._errors) > snapshot


def _func_err_budget_exhausted(ctx: _InferCtx) -> bool:
    return (len(ctx.result._errors) - ctx._func_err_start) >= ctx._func_err_limit


# ---------------------------------------------------------------------------
# Statement validation
# ---------------------------------------------------------------------------


def _validate_func(func_node: ASTNode, ctx: _InferCtx, receiver: str) -> None:
    """Validate a single function/method body."""
    func_name = get_str(func_node, "name")
    func_info: FuncInfo | None = None
    if receiver != "":
        methods = ctx.tc_result.methods.get(receiver)
        if methods is not None:
            func_info = methods.get(func_name)
    else:
        func_info = ctx.tc_result.functions.get(func_name)
    if func_info is None:
        return
    if receiver != "":
        graph_key = receiver + "::" + func_name
    else:
        graph_key = "module::" + func_name
    ctx.current_graph = ctx.flow_graphs.get(graph_key)
    env = TypeEnv()
    i = 0
    while i < len(func_info.params):
        p = func_info.params[i]
        env.set(p.name, p.typ)
        i += 1
    if receiver != "":
        self_type = PointerType(StructRef(receiver))
        env.set("this", self_type)
    body = get_nodes(func_node, "body")
    if len(body) == 0:
        return
    ctx._func_err_start = len(ctx.result._errors)
    ctx._func_err_limit = 5
    _validate_stmts(body, env, func_info, ctx)


def _validate_stmts(
    stmts: list[ASTNode],
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
        if _func_err_budget_exhausted(ctx):
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
    t = get_str(stmt, "_type")
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
        test = get_node(stmt, "test")
        if len(test) > 0 and _is_type(test, ["Constant"]):
            val = test.get("value")
            if isinstance(val, JBool) and not val.value:
                return True
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
        lineno = get_int(stmt, "lineno")
        ctx.result.add_error(lineno, 0, "nested function definitions are not allowed")
        return False
    return False


def _validate_return(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    value = get_node(stmt, "value")
    if len(value) == 0:
        return
    lineno = get_int(stmt, "lineno")
    if _check_iterator_escape_return(value, env, ctx, lineno):
        return
    if _check_generator_escape_return(value, env, ctx, lineno):
        return
    err_snap = len(ctx.result._errors)
    _validate_expr_access(value, env, ctx, lineno)
    if _has_new_errors(ctx, err_snap):
        return
    _validate_expr_calls(value, env, ctx, lineno)
    if _has_new_errors(ctx, err_snap):
        return
    _validate_return_value(value, func_info.return_type, env, ctx, lineno)
    if _has_new_errors(ctx, err_snap):
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
    targets = get_nodes(stmt, "targets")
    value = get_node(stmt, "value")
    if len(targets) == 0 or len(value) == 0:
        return
    lineno = get_int(stmt, "lineno")
    err_snap = len(ctx.result._errors)
    if len(targets) == 1:
        tgt = targets[0]
        if _is_type(tgt, ["Name"]):
            if _check_iterator_escape_assign(value, env, ctx, lineno):
                return
            if _check_generator_escape_assign(value, env, ctx, lineno):
                return
            _validate_expr_access(value, env, ctx, lineno)
            _validate_expr_calls(value, env, ctx, lineno)
    if _has_new_errors(ctx, err_snap):
        val_type = ANY_TYPE
    else:
        val_type = _synth_expr(value, env, ctx)
    i = 0
    while i < len(targets):
        tgt = targets[i]
        if _is_type(tgt, ["Name"]):
            name = get_str(tgt, "id")
            if name != "":
                env.evict_guarded_attrs(name)
                existing = env.get_type(name)
                if existing is not None:
                    if not _type_eq(val_type, existing):
                        env.set(name, val_type)
                else:
                    if _is_empty_collection(value) and is_any(_element_type(val_type)):
                        ctx.result.add_error(
                            lineno,
                            0,
                            "empty "
                            + _collection_name(value)
                            + " needs type annotation",
                        )
                        return
                    env.set(name, val_type)
        elif _is_type(tgt, ["Tuple", "List"]):
            _validate_unpack(tgt, val_type, value, env, ctx, lineno)
        elif _is_type(tgt, ["Subscript"]):
            _validate_subscript_assign(tgt, val_type, env, ctx, lineno)
        elif _is_type(tgt, ["Attribute"]):
            pass
        i += 1


def _is_empty_collection(node: ASTNode) -> bool:
    t = get_str(node, "_type")
    if t == "List":
        elts = get_nodes(node, "elts")
        return len(elts) == 0
    if t == "Dict":
        keys = get_nodes(node, "keys")
        return len(keys) == 0
    if t == "Set":
        elts = get_nodes(node, "elts")
        return len(elts) == 0
    return False


def _collection_name(node: ASTNode) -> str:
    t = get_str(node, "_type")
    if t == "List":
        return "list"
    if t == "Dict":
        return "dict"
    if t == "Set":
        return "set"
    return "collection"


def _validate_unpack(
    target: ASTNode,
    val_type: TypeNode,
    value: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Validate tuple unpacking."""
    elts = get_nodes(target, "elts")
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
    value = get_node(target, "value")
    slc = get_node(target, "slice")
    if len(value) == 0:
        return
    obj_type = _synth_expr(value, env, ctx)
    if isinstance(obj_type, MapType):
        if len(slc) > 0:
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
        is_slice = len(slc) > 0 and get_str(slc, "_type") == "Slice"
        if is_slice:
            if not _is_assignable(val_type, obj_type, ctx.hier_result):
                ctx.result.add_error(
                    lineno,
                    0,
                    "cannot assign "
                    + _type_name(val_type)
                    + " to "
                    + _type_name(obj_type),
                )
        elif not _is_assignable(val_type, obj_type.element, ctx.hier_result):
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
    target = get_node(stmt, "target")
    annotation = get_node(stmt, "annotation")
    value = get_node(stmt, "value")
    if len(target) == 0:
        return
    lineno = get_int(stmt, "lineno")
    if len(annotation) == 0:
        return
    ann_str = annotation_to_str(annotation)
    sig_errors: list[TypeCollectError] = []
    ann_type = py_type_to_type_dict(ann_str, ctx.known_classes, sig_errors, lineno, 0)
    if _is_type(target, ["Name"]):
        name = get_str(target, "id")
        if name != "":
            env.set(name, ann_type)
            if len(value) > 0:
                err_snap = len(ctx.result._errors)
                _validate_expr_access(value, env, ctx, lineno)
                if _has_new_errors(ctx, err_snap):
                    return
                _validate_expr_calls(value, env, ctx, lineno)
                if _has_new_errors(ctx, err_snap):
                    return
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
    target = get_node(stmt, "target")
    value = get_node(stmt, "value")
    if len(target) == 0 or len(value) == 0:
        return
    lineno = get_int(stmt, "lineno")
    err_snap = len(ctx.result._errors)
    _check_needs_narrowing(value, env, ctx, lineno, "arithmetic", "")
    if _has_new_errors(ctx, err_snap):
        return
    _validate_expr_access(value, env, ctx, lineno)
    if _has_new_errors(ctx, err_snap):
        return
    _validate_expr_calls(value, env, ctx, lineno)
    _synth_expr(value, env, ctx)


def _validate_expr_stmt(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    value = get_node(stmt, "value")
    if len(value) == 0:
        return
    lineno = get_int(stmt, "lineno")
    if _is_type(value, ["Call"]):
        func = get_node(value, "func")
        if len(func) > 0 and _is_type(func, ["Name"]):
            fname = get_str(func, "id")
            if fname == "reveal_type":
                args = get_nodes(value, "args")
                if len(args) == 1:
                    typ = _synth_expr(args[0], env, ctx)
                    ctx.result.add_reveal(lineno, _type_name(typ))
                return
            if (
                fname != ""
                and fname not in _EAGER_CONSUMERS
                and fname not in _ITERATOR_FUNCS
            ):
                args = get_nodes(value, "args")
                j = 0
                while j < len(args):
                    arg = args[j]
                    _check_iterator_escape_arg(arg, fname, env, ctx, lineno)
                    _check_generator_escape_arg(arg, fname, env, ctx, lineno)
                    j += 1
        if len(func) > 0 and _is_type(func, ["Attribute"]):
            attr = get_str(func, "attr")
            if attr != "":
                args = get_nodes(value, "args")
                j = 0
                while j < len(args):
                    arg = args[j]
                    if attr == "join":
                        pass
                    else:
                        _check_generator_escape_arg(arg, attr, env, ctx, lineno)
                    j += 1
    err_snap = len(ctx.result._errors)
    _validate_expr_access(value, env, ctx, lineno)
    if _has_new_errors(ctx, err_snap):
        return
    _synth_expr(value, env, ctx)
    _validate_expr_calls(value, env, ctx, lineno)


def _validate_expr_calls(
    node: ASTNode, env: TypeEnv, ctx: _InferCtx, lineno: int
) -> None:
    """Walk an expression and validate args for every Call found."""
    if not isinstance(node, dict):
        return
    t = get_str(node, "_type")
    if t == "Call":
        _validate_call_args(node, env, ctx, lineno)
        args = get_nodes(node, "args")
        j = 0
        while j < len(args):
            _validate_expr_calls(args[j], env, ctx, lineno)
            j += 1
        return
    if t == "BinOp":
        _validate_expr_calls(get_node(node, "left"), env, ctx, lineno)
        _validate_expr_calls(get_node(node, "right"), env, ctx, lineno)
        return
    if t == "UnaryOp":
        _validate_expr_calls(get_node(node, "operand"), env, ctx, lineno)
        return
    if t == "BoolOp":
        op = get_node(node, "op")
        op_t = get_str(op, "_type")
        values = get_nodes(node, "values")
        if op_t == "And":
            work = env.copy()
            j = 0
            while j < len(values):
                _validate_expr_calls(values[j], work, ctx, lineno)
                if j < len(values) - 1:
                    dummy = env.copy()
                    _extract_narrowing(values[j], work, dummy, ctx)
                j += 1
            return
        if op_t == "Or":
            work = env.copy()
            j = 0
            while j < len(values):
                _validate_expr_calls(values[j], work, ctx, lineno)
                if j < len(values) - 1:
                    dummy = env.copy()
                    _extract_narrowing(values[j], dummy, work, ctx)
                j += 1
            return
        return


def _validate_call_args(
    node: ASTNode, env: TypeEnv, ctx: _InferCtx, lineno: int
) -> None:
    """Validate argument types in function/method calls."""
    if not _is_type(node, ["Call"]):
        return
    func = get_node(node, "func")
    args = get_nodes(node, "args")
    if len(func) == 0:
        return
    if _is_type(func, ["Name"]):
        fname = get_str(func, "id")
        if fname == "":
            return
        func_info = ctx.tc_result.functions.get(fname)
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
        obj = get_node(func, "value")
        attr = get_str(func, "attr")
        if len(obj) == 0 or attr == "":
            return
        obj_type = _synth_expr(obj, env, ctx)
        if _is_type(obj, ["Name"]):
            obj_name = get_str(obj, "id")
            if obj_name != "" and obj_name in ctx.known_classes:
                methods = ctx.tc_result.methods.get(obj_name)
                if methods is not None and attr in methods:
                    ctx.result.add_error(
                        lineno,
                        0,
                        "cannot call method without self: " + obj_name + "." + attr,
                    )
                    return
        sname = _struct_name(obj_type)
        if sname != "":
            methods = ctx.tc_result.methods.get(sname)
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
    args: list[ASTNode],
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
        if j < len(params):
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
    args: list[ASTNode],
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
        if j < len(ftype.params):
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
    args: list[ASTNode],
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Validate collection method argument types."""
    if isinstance(obj_type, SliceType):
        elem = obj_type.element
        if method == "append":
            if len(args) > 0:
                at = _synth_expr(args[0], env, ctx)
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
                at = _synth_expr(args[0], env, ctx)
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
                at = _synth_expr(args[1], env, ctx)
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
                at = _synth_expr(args[0], env, ctx)
                if not _is_assignable(at, elem, ctx.hier_result):
                    ctx.result.add_error(
                        lineno,
                        0,
                        "cannot assign "
                        + _type_name(at)
                        + " to set element "
                        + _type_name(elem),
                    )


def _flatten_for_merge(t: TypeNode, out: list[TypeNode]) -> None:
    """Expand OptionalType and UnionType into flat list for combine_types."""
    if isinstance(t, OptionalType):
        _flatten_for_merge(t.inner, out)
        out.append(VOID_TYPE)
    elif isinstance(t, UnionType):
        i = 0
        while i < len(t.variants):
            _flatten_for_merge(t.variants[i], out)
            i += 1
    else:
        out.append(t)


def _merge_no_else(
    then_env: TypeEnv, else_env: TypeEnv, out: TypeEnv, hier: HierarchyResult
) -> None:
    """Merge if-without-else: only update vars present in both branches."""
    ekeys = list(else_env.types.keys())
    j = 0
    while j < len(ekeys):
        k = ekeys[j]
        then_t = then_env.types.get(k)
        else_t = else_env.types[k]
        if then_t is not None:
            if _type_eq(else_t, then_t):
                out.types[k] = else_t
            elif _is_assignable(then_t, else_t, hier):
                out.types[k] = else_t
            elif _is_assignable(else_t, then_t, hier):
                out.types[k] = then_t
            else:
                parts: list[TypeNode] = []
                _flatten_for_merge(then_t, parts)
                _flatten_for_merge(else_t, parts)
                out.types[k] = combine_types(parts)
        j += 1


def _merge_branch_envs(
    env_a: TypeEnv, env_b: TypeEnv, out: TypeEnv, hier: HierarchyResult
) -> None:
    """Merge two branch environments into out using combine_types."""
    all_keys: list[str] = []
    akeys = list(env_a.types.keys())
    j = 0
    while j < len(akeys):
        all_keys.append(akeys[j])
        j += 1
    bkeys = list(env_b.types.keys())
    j = 0
    while j < len(bkeys):
        k = bkeys[j]
        found = False
        ki = 0
        while ki < len(all_keys):
            if all_keys[ki] == k:
                found = True
            ki += 1
        if not found:
            all_keys.append(k)
        j += 1
    j = 0
    while j < len(all_keys):
        k = all_keys[j]
        in_a = k in env_a.types
        in_b = k in env_b.types
        if in_a and in_b:
            ta = env_a.types[k]
            tb = env_b.types[k]
            if _type_eq(ta, tb):
                out.types[k] = ta
            elif _is_assignable(ta, tb, hier):
                out.types[k] = tb
            elif _is_assignable(tb, ta, hier):
                out.types[k] = ta
            else:
                parts: list[TypeNode] = []
                _flatten_for_merge(ta, parts)
                _flatten_for_merge(tb, parts)
                out.types[k] = combine_types(parts)
        elif in_a:
            out.types[k] = env_a.types[k]
        elif in_b:
            out.types[k] = env_b.types[k]
        j += 1


def _has_never_narrowing(pre_env: TypeEnv, post_env: TypeEnv) -> bool:
    """Check if any variable was narrowed to never between pre and post envs."""
    keys = list(post_env.types.keys())
    i = 0
    while i < len(keys):
        k = keys[i]
        t = post_env.types[k]
        if isinstance(t, PrimitiveType) and t.kind == "never":
            pre_t = pre_env.types.get(k)
            if pre_t is not None and not (
                isinstance(pre_t, PrimitiveType) and pre_t.kind == "never"
            ):
                return True
        i += 1
    return False


def _validate_if(
    stmt: ASTNode,
    env: TypeEnv,
    func_info: FuncInfo,
    ctx: _InferCtx,
) -> bool:
    """Validate if/elif/else. Returns True if all branches return."""
    test = get_node(stmt, "test")
    body = get_nodes(stmt, "body")
    orelse = get_nodes(stmt, "orelse")
    lineno = get_int(stmt, "lineno")
    if len(test) > 0:
        err_snap = len(ctx.result._errors)
        _validate_expr_access(test, env, ctx, lineno)
        if _has_new_errors(ctx, err_snap):
            return False
        _validate_expr_calls(test, env, ctx, lineno)
        if _has_new_errors(ctx, err_snap):
            return False
        _check_truthiness(test, env, ctx, lineno)
    then_env = env.copy()
    else_env = env.copy()
    if len(test) > 0:
        _extract_narrowing(test, then_env, else_env, ctx)
    then_returns = _validate_stmts(body, then_env, func_info, ctx)
    else_returns = False
    if len(orelse) > 0:
        is_elif = len(orelse) == 1 and _is_type(orelse[0], ["If"])
        if not is_elif and _has_never_narrowing(env, else_env):
            else_lineno = get_int(orelse[0], "lineno")
            ctx.result.add_error(
                else_lineno, 0, "unreachable code: all union variants already handled"
            )
        else_returns = _validate_stmts(orelse, else_env, func_info, ctx)
    if then_returns and not else_returns:
        ekeys = list(else_env.types.keys())
        j = 0
        while j < len(ekeys):
            env.types[ekeys[j]] = else_env.types[ekeys[j]]
            j += 1
        gkeys = list(else_env.guarded_attrs)
        j = 0
        while j < len(gkeys):
            env.guarded_attrs.add(gkeys[j])
            j += 1
    elif else_returns and not then_returns:
        tkeys = list(then_env.types.keys())
        j = 0
        while j < len(tkeys):
            env.types[tkeys[j]] = then_env.types[tkeys[j]]
            j += 1
        gkeys = list(then_env.guarded_attrs)
        j = 0
        while j < len(gkeys):
            env.guarded_attrs.add(gkeys[j])
            j += 1
    elif not then_returns and not else_returns and len(orelse) > 0:
        _merge_branch_envs(then_env, else_env, env, ctx.hier_result)
    elif not then_returns and len(orelse) == 0:
        _merge_no_else(then_env, else_env, env, ctx.hier_result)
    return then_returns and else_returns


def _validate_while(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    test = get_node(stmt, "test")
    body = get_nodes(stmt, "body")
    lineno = get_int(stmt, "lineno")
    if len(test) > 0:
        err_snap = len(ctx.result._errors)
        _validate_expr_access(test, env, ctx, lineno)
        if _has_new_errors(ctx, err_snap):
            return
        _validate_expr_calls(test, env, ctx, lineno)
        _check_truthiness(test, env, ctx, lineno)
    loop_env = env.copy()
    else_env = env.copy()
    if len(test) > 0:
        _extract_narrowing(test, loop_env, else_env, ctx)
    body_returns = _validate_stmts(body, loop_env, func_info, ctx)
    has_break = False
    j = 0
    while j < len(body):
        bt = get_str(body[j], "_type") if isinstance(body[j], dict) else ""
        if bt == "Break":
            has_break = True
        j += 1
    if body_returns and not has_break:
        ekeys = list(else_env.types.keys())
        j = 0
        while j < len(ekeys):
            env.types[ekeys[j]] = else_env.types[ekeys[j]]
            j += 1
    elif not body_returns:
        _merge_branch_envs(else_env, loop_env, env, ctx.hier_result)


def _validate_for(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    target = get_node(stmt, "target")
    iter_node = get_node(stmt, "iter")
    body = get_nodes(stmt, "body")
    if len(iter_node) > 0:
        err_snap = len(ctx.result._errors)
        _validate_expr_calls(iter_node, env, ctx, get_int(stmt, "lineno"))
        if _has_new_errors(ctx, err_snap):
            return
        iter_type = _synth_expr(iter_node, env, ctx)
        elem = _iteration_element(iter_type)
        if len(target) > 0:
            _bind_target(target, elem, env)
    loop_env = env.copy()
    if len(target) > 0:
        tname = get_str(target, "id")
        if tname != "":
            t = env.get_type(tname)
            if t is not None:
                loop_env.set(tname, t)
    _validate_stmts(body, loop_env, func_info, ctx)
    _merge_branch_envs(env, loop_env, env, ctx.hier_result)


def _validate_assert(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    test = get_node(stmt, "test")
    if len(test) == 0:
        return
    lineno = get_int(stmt, "lineno")
    err_snap = len(ctx.result._errors)
    _validate_expr_calls(test, env, ctx, lineno)
    if _has_new_errors(ctx, err_snap):
        return
    dummy_else = env.copy()
    _extract_narrowing(test, env, dummy_else, ctx)


def _validate_try(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    body = get_nodes(stmt, "body")
    handlers = get_nodes(stmt, "handlers")
    orelse = get_nodes(stmt, "orelse")
    finalbody = get_nodes(stmt, "finalbody")
    _validate_stmts(body, env, func_info, ctx)
    j = 0
    while j < len(handlers):
        h = handlers[j]
        hbody = get_nodes(h, "body")
        _validate_stmts(hbody, env.copy(), func_info, ctx)
        j += 1
    _validate_stmts(orelse, env, func_info, ctx)
    _validate_stmts(finalbody, env, func_info, ctx)


def _validate_match(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    subject = get_node(stmt, "subject")
    subj_name = ""
    if len(subject) > 0 and _is_type(subject, ["Name"]):
        subj_name = get_str(subject, "id")
    cases = get_nodes(stmt, "cases")
    j = 0
    while j < len(cases):
        case = cases[j]
        case_env = env.copy()
        if subj_name != "":
            pattern = get_node(case, "pattern")
            if len(pattern) > 0 and _is_type(pattern, ["MatchClass"]):
                cls = get_node(pattern, "cls")
                if len(cls) > 0 and _is_type(cls, ["Name"]):
                    cls_name = get_str(cls, "id")
                    if cls_name != "":
                        sig_errors: list[TypeCollectError] = []
                        narrowed = py_type_to_type_dict(
                            cls_name, ctx.known_classes, sig_errors, 0, 0
                        )
                        case_env.narrow(subj_name, narrowed)
        case_body = get_nodes(case, "body")
        _validate_stmts(case_body, case_env, func_info, ctx)
        j += 1


# ---------------------------------------------------------------------------
# Truthiness checking
# ---------------------------------------------------------------------------


def _check_truthiness(test: ASTNode, env: TypeEnv, ctx: _InferCtx, lineno: int) -> None:
    """Check that a condition expression has unambiguous truthiness."""
    if not isinstance(test, dict):
        return
    t = get_str(test, "_type")
    if t == "Compare":
        return
    if t == "Call":
        func = get_node(test, "func")
        if (
            len(func) > 0
            and _is_type(func, ["Name"])
            and get_str(func, "id") == "isinstance"
        ):
            return
    if t == "BoolOp":
        values = get_nodes(test, "values")
        j = 0
        while j < len(values):
            _check_truthiness(values[j], env, ctx, lineno)
            j += 1
        return
    if t == "UnaryOp":
        op = get_node(test, "op")
        if get_str(op, "_type") == "Not":
            operand = get_node(test, "operand")
            if len(operand) > 0:
                _check_truthiness(operand, env, ctx, lineno)
            return
    if t == "NamedExpr":
        value = get_node(test, "value")
        if len(value) > 0:
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
    pk = _prim_kind(typ)
    if pk == "bool":
        return
    if pk == "int":
        return
    if pk == "float":
        return
    if isinstance(typ, OptionalType):
        inner = typ.inner
        inner_pk = _prim_kind(inner)
        if inner_pk in ("int", "float", "bool"):
            return
        if inner_pk == "string":
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
    if pk == "string":
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
    t = get_str(test, "_type")
    if t == "Call":
        func = get_node(test, "func")
        if (
            len(func) > 0
            and _is_type(func, ["Name"])
            and get_str(func, "id") == "isinstance"
        ):
            _narrow_isinstance(test, then_env, else_env, ctx)
            return
    if t == "Compare":
        _narrow_compare(test, then_env, else_env, ctx)
        return
    if t == "UnaryOp":
        op = get_node(test, "op")
        if get_str(op, "_type") == "Not":
            operand = get_node(test, "operand")
            if len(operand) > 0:
                _extract_narrowing(operand, else_env, then_env, ctx)
            return
    if t == "BoolOp":
        op = get_node(test, "op")
        op_t = get_str(op, "_type")
        if op_t == "And":
            values = get_nodes(test, "values")
            j = 0
            while j < len(values):
                _extract_narrowing(values[j], then_env, else_env, ctx)
                j += 1
            return
        if op_t == "Or":
            values = get_nodes(test, "values")
            _narrow_or_isinstance(values, then_env, ctx)
            dummy_then = else_env.copy()
            j = 0
            while j < len(values):
                _extract_narrowing(values[j], dummy_then, else_env, ctx)
                j += 1
            return
    if t == "NamedExpr":
        target = get_node(test, "target")
        value = get_node(test, "value")
        if len(target) > 0 and len(value) > 0 and _is_type(target, ["Name"]):
            name = get_str(target, "id")
            if name != "":
                vt = _synth_expr(value, then_env, ctx)
                then_env.set(name, vt)
                else_env.set(name, vt)
                if isinstance(vt, OptionalType):
                    then_env.set(name, vt.inner)
        return
    if t == "Name":
        name = get_str(test, "id")
        if name != "":
            if ctx.current_graph is not None:
                flow_alias = lookup_alias(ctx.current_graph, name)
                if flow_alias is not None:
                    alias = CondAlias(
                        flow_alias.target,
                        flow_alias.narrow_type,
                        flow_alias.type_name,
                        flow_alias.field_name,
                    )
                    _apply_alias_narrowing(alias, then_env, else_env, ctx)
                    return
            typ = then_env.get_type(name)
            if typ is not None and isinstance(typ, OptionalType):
                then_env.narrow(name, typ.inner)
        return
    if t == "Attribute":
        path = _attr_path(test)
        if path != "":
            then_env.guard_attr(path)
        return


def _attr_path(node: ASTNode) -> str:
    """Build dotted path from nested Attribute nodes, e.g. 'expr.obj'."""
    parts: list[str] = []
    cur = node
    while _is_type(cur, ["Attribute"]):
        a = get_str(cur, "attr")
        if a == "":
            return ""
        parts.append(a)
        v = get_node(cur, "value")
        if len(v) == 0:
            return ""
        cur = v
    if not _is_type(cur, ["Name"]):
        return ""
    base = get_str(cur, "id")
    if base == "":
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
    args = get_nodes(test, "args")
    if len(args) < 2:
        return
    target = args[0]
    type_arg = args[1]
    name = ""
    if _is_type(target, ["Name"]):
        name = get_str(target, "id")
    elif _is_type(target, ["Attribute"]):
        name = _attr_path(target)
    elif _is_type(target, ["NamedExpr"]):
        walrus_target = get_node(target, "target")
        if len(walrus_target) > 0 and _is_type(walrus_target, ["Name"]):
            name = get_str(walrus_target, "id")
            walrus_value = get_node(target, "value")
            if name != "" and len(walrus_value) > 0:
                vt = _synth_expr(walrus_value, then_env, ctx)
                then_env.set(name, vt)
                else_env.set(name, vt)
    if name == "":
        return
    narrow_names: list[str] = []
    if _is_type(type_arg, ["Name"]):
        n = get_str(type_arg, "id")
        if n != "":
            narrow_names.append(n)
    elif _is_type(type_arg, ["Tuple"]):
        elts = get_nodes(type_arg, "elts")
        j = 0
        while j < len(elts):
            if _is_type(elts[j], ["Name"]):
                n = get_str(elts[j], "id")
                if n != "":
                    narrow_names.append(n)
            j += 1
    if len(narrow_names) == 0:
        return
    if len(narrow_names) == 1:
        narrow_name = narrow_names[0]
        sig_errors: list[TypeCollectError] = []
        narrowed = py_type_to_type_dict(
            narrow_name, ctx.known_classes, sig_errors, 0, 0
        )
        then_env.narrow(name, narrowed)
    elif len(narrow_names) > 1:
        sig_errors: list[TypeCollectError] = []
        variants: list[TypeNode] = []
        ni = 0
        while ni < len(narrow_names):
            variants.append(
                py_type_to_type_dict(
                    narrow_names[ni], ctx.known_classes, sig_errors, 0, 0
                )
            )
            ni += 1
        then_env.narrow(name, UnionType(variants))
    else_type = else_env.get_type(name)
    if else_type is not None:
        remove_types: list[TypeNode] = []
        ri = 0
        while ri < len(narrow_names):
            sig_e_rm: list[TypeCollectError] = []
            remove_types.append(
                py_type_to_type_dict(
                    narrow_names[ri], ctx.known_classes, sig_e_rm, 0, 0
                )
            )
            ri += 1
        remaining_type = remove_from_union(else_type, remove_types)
        else_env.narrow(name, remaining_type)


def _pascal_to_kebab(name: str) -> str:
    """PascalCase to kebab-case: BinaryOp -> binary-op."""
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


def _find_variant_by_const_field(
    union_type: UnionType,
    field_name: str,
    field_value: str,
    ctx: _InferCtx,
) -> TypeNode | None:
    """Find the union variant whose const_fields[field_name] == field_value."""
    i = 0
    while i < len(union_type.variants):
        vname = _variant_name(union_type.variants[i])
        if vname != "":
            cls = ctx.tc_result.classes.get(vname)
            if cls is not None:
                cf_val = cls.const_fields.get(field_name)
                if cf_val is not None and cf_val == field_value:
                    return union_type.variants[i]
                if cf_val is None and field_name == "kind":
                    if _pascal_to_kebab(vname) == field_value:
                        return union_type.variants[i]
        i += 1
    return None


def _narrow_compare(
    test: ASTNode,
    then_env: TypeEnv,
    else_env: TypeEnv,
    ctx: _InferCtx,
) -> None:
    """Narrow from comparison (x is None, x is not None, x.attr == "foo")."""
    left = get_node(test, "left")
    ops = get_nodes(test, "ops")
    comparators = get_nodes(test, "comparators")
    if len(left) == 0 or len(ops) == 0 or len(comparators) == 0:
        return
    if _is_type(left, ["NamedExpr"]):
        ne_target = get_node(left, "target")
        ne_value = get_node(left, "value")
        if len(ne_target) > 0 and _is_type(ne_target, ["Name"]):
            ne_name = get_str(ne_target, "id")
            if ne_name != "" and len(ne_value) > 0:
                vt = _synth_expr(ne_value, then_env, ctx)
                then_env.set(ne_name, vt)
                else_env.set(ne_name, vt)
            left = ne_target
    op = ops[0]
    comp = comparators[0]
    op_type = get_str(op, "_type")
    comp_is_none = _is_type(comp, ["Constant"]) and _is_null_value(comp)
    if op_type == "Is" and comp_is_none:
        if _is_type(left, ["Name"]):
            name = get_str(left, "id")
            if name != "":
                _narrow_to_non_none(name, else_env, ctx)
        if _is_type(left, ["Attribute"]):
            path = _attr_path(left)
            if path != "":
                else_env.guard_attr(path)
        return
    if op_type == "IsNot" and comp_is_none:
        if _is_type(left, ["Name"]):
            name = get_str(left, "id")
            if name != "":
                _narrow_to_non_none(name, then_env, ctx)
        if _is_type(left, ["Attribute"]):
            path = _attr_path(left)
            if path != "":
                then_env.guard_attr(path)
        return
    if op_type == "Eq" and comp_is_none:
        if _is_type(left, ["Name"]):
            name = get_str(left, "id")
            if name != "":
                _narrow_to_non_none(name, else_env, ctx)
        if _is_type(left, ["Attribute"]):
            path = _attr_path(left)
            if path != "":
                else_env.guard_attr(path)
        return
    if op_type == "NotEq" and comp_is_none:
        if _is_type(left, ["Name"]):
            name = get_str(left, "id")
            if name != "":
                _narrow_to_non_none(name, then_env, ctx)
        if _is_type(left, ["Attribute"]):
            path = _attr_path(left)
            if path != "":
                then_env.guard_attr(path)
        return
    if op_type == "Eq" and not comp_is_none:
        if _is_type(left, ["Name"]):
            name = get_str(left, "id")
            if name != "":
                _narrow_to_non_none(name, then_env, ctx)
                if _is_type(comp, ["Constant"]):
                    lit = _extract_literal(comp)
                    if lit is not None:
                        cur = then_env.get_type(name)
                        if cur is not None and _prim_kind(cur) == lit.base.kind:
                            then_env.narrow(name, lit)
    if op_type == "Eq":
        if _is_type(left, ["Attribute"]):
            attr = get_str(left, "attr")
            comp_v = comp.get("value")
            if isinstance(comp_v, JStr):
                comp_value = comp_v.value
                obj_node = get_node(left, "value")
                obj_name = ""
                obj_type: TypeNode | None = None
                if len(obj_node) > 0 and _is_type(obj_node, ["Name"]):
                    obj_name = get_str(obj_node, "id")
                    if obj_name != "":
                        obj_type = then_env.get_type(obj_name)
                if len(obj_node) > 0 and _is_type(obj_node, ["Attribute"]):
                    attr_path = _attr_path(obj_node)
                    if attr_path != "":
                        then_env.guard_attr(attr_path)
                if obj_type is not None and isinstance(obj_type, UnionType):
                    matched = _find_variant_by_const_field(
                        obj_type,
                        attr,
                        comp_value,
                        ctx,
                    )
                    if matched is None:
                        k_lineno = get_int(test, "lineno")
                        ctx.result.add_error(
                            k_lineno,
                            0,
                            attr
                            + " value '"
                            + comp_value
                            + "' does not match any known type",
                        )
                    elif obj_name != "":
                        then_env.narrow(obj_name, matched)
                        else_remaining = remove_from_union(obj_type, [matched])
                        else_env.narrow(obj_name, else_remaining)
                elif obj_type is not None and obj_name != "":
                    sn = _struct_name(obj_type)
                    if sn != "":
                        cls = ctx.tc_result.classes.get(sn)
                        if cls is not None:
                            cf_val = cls.const_fields.get(attr)
                            matches = False
                            if cf_val is not None and cf_val == comp_value:
                                matches = True
                            if cf_val is None and attr == "kind":
                                if _pascal_to_kebab(sn) == comp_value:
                                    matches = True
                            if matches:
                                else_env.narrow(obj_name, PrimitiveType("never"))
            return
    if op_type == "NotEq" and not comp_is_none:
        if _is_type(left, ["Name"]):
            name = get_str(left, "id")
            if name != "":
                if _is_type(comp, ["Constant"]):
                    lit = _extract_literal(comp)
                    if lit is not None:
                        cur = else_env.get_type(name)
                        if cur is not None and _prim_kind(cur) == lit.base.kind:
                            else_env.narrow(name, lit)
        if _is_type(left, ["Attribute"]):
            attr = get_str(left, "attr")
            comp_v = comp.get("value")
            if isinstance(comp_v, JStr):
                comp_value = comp_v.value
                obj_node = get_node(left, "value")
                obj_name = ""
                obj_type2: TypeNode | None = None
                if len(obj_node) > 0 and _is_type(obj_node, ["Name"]):
                    obj_name = get_str(obj_node, "id")
                    if obj_name != "":
                        obj_type2 = else_env.get_type(obj_name)
                if (
                    obj_name != ""
                    and obj_type2 is not None
                    and isinstance(obj_type2, UnionType)
                ):
                    matched = _find_variant_by_const_field(
                        obj_type2,
                        attr,
                        comp_value,
                        ctx,
                    )
                    if matched is not None:
                        else_env.narrow(obj_name, matched)
                        then_remaining = remove_from_union(obj_type2, [matched])
                        then_env.narrow(obj_name, then_remaining)
                elif obj_name != "" and obj_type2 is not None:
                    sn = _struct_name(obj_type2)
                    if sn != "":
                        cls = ctx.tc_result.classes.get(sn)
                        if cls is not None:
                            cf_val = cls.const_fields.get(attr)
                            matches = False
                            if cf_val is not None and cf_val == comp_value:
                                matches = True
                            if cf_val is None and attr == "kind":
                                if _pascal_to_kebab(sn) == comp_value:
                                    matches = True
                            if matches:
                                then_env.narrow(obj_name, PrimitiveType("never"))
            return


def _apply_alias_narrowing(
    alias: CondAlias,
    then_env: TypeEnv,
    else_env: TypeEnv,
    ctx: _InferCtx,
) -> None:
    """Apply narrowing from a condition alias (flag = isinstance(x, T), etc)."""
    target = alias.target
    if alias.narrow_type == "isinstance":
        sig_errors: list[TypeCollectError] = []
        narrowed = py_type_to_type_dict(
            alias.type_name, ctx.known_classes, sig_errors, 0, 0
        )
        then_env.narrow(target, narrowed)
        else_type = else_env.get_type(target)
        if else_type is not None:
            sig_e_rm: list[TypeCollectError] = []
            remove_t = py_type_to_type_dict(
                alias.type_name, ctx.known_classes, sig_e_rm, 0, 0
            )
            remaining = remove_from_union(else_type, [remove_t])
            else_env.narrow(target, remaining)
        return
    if alias.narrow_type == "is_none":
        _narrow_to_non_none(target, else_env, ctx)
        return
    if alias.narrow_type == "is_not_none":
        _narrow_to_non_none(target, then_env, ctx)
        return
    if alias.narrow_type == "const_field":
        obj_type = then_env.get_type(target)
        if obj_type is not None and isinstance(obj_type, UnionType):
            matched = _find_variant_by_const_field(
                obj_type, alias.field_name, alias.type_name, ctx
            )
            if matched is not None:
                then_env.narrow(target, matched)
                else_remaining = remove_from_union(obj_type, [matched])
                else_env.narrow(target, else_remaining)


def _narrow_to_non_none(name: str, env: TypeEnv, ctx: _InferCtx) -> None:
    """Narrow a variable to its non-None part."""
    typ = env.get_type(name)
    if typ is not None and isinstance(typ, OptionalType):
        env.narrow(name, typ.inner)


def _narrow_or_isinstance(
    values: list[ASTNode], then_env: TypeEnv, ctx: _InferCtx
) -> None:
    """Handle isinstance(x,A) or isinstance(x,B) in then branch."""
    name = ""
    type_names: list[str] = []
    i = 0
    while i < len(values):
        v = values[i]
        if not _is_type(v, ["Call"]):
            return
        func = get_node(v, "func")
        if not (
            len(func) > 0
            and _is_type(func, ["Name"])
            and get_str(func, "id") == "isinstance"
        ):
            return
        args = get_nodes(v, "args")
        if len(args) < 2:
            return
        target = args[0]
        if not _is_type(target, ["Name"]):
            return
        tname = get_str(target, "id")
        if tname == "":
            return
        if name == "":
            name = tname
        elif tname != name:
            return
        type_arg = args[1]
        if _is_type(type_arg, ["Name"]):
            tn = get_str(type_arg, "id")
            if tn != "":
                type_names.append(tn)
        elif _is_type(type_arg, ["Tuple"]):
            elts = get_nodes(type_arg, "elts")
            j = 0
            while j < len(elts):
                if _is_type(elts[j], ["Name"]):
                    tn = get_str(elts[j], "id")
                    if tn != "":
                        type_names.append(tn)
                j += 1
        i += 1
    if name == "" or len(type_names) == 0:
        return
    sig_errors: list[TypeCollectError] = []
    if len(type_names) == 1:
        narrowed = py_type_to_type_dict(
            type_names[0], ctx.known_classes, sig_errors, 0, 0
        )
        then_env.narrow(name, narrowed)
    else:
        variants: list[TypeNode] = []
        ni = 0
        while ni < len(type_names):
            variants.append(
                py_type_to_type_dict(
                    type_names[ni], ctx.known_classes, sig_errors, 0, 0
                )
            )
            ni += 1
        then_env.narrow(name, UnionType(variants))


# ---------------------------------------------------------------------------
# Iterator/generator escape checking
# ---------------------------------------------------------------------------


def _is_iterator_call(node: ASTNode) -> str:
    """If node is enumerate/zip/reversed call, return the func name."""
    if not _is_type(node, ["Call"]):
        return ""
    func = get_node(node, "func")
    if len(func) > 0 and _is_type(func, ["Name"]):
        fname = get_str(func, "id")
        if fname != "" and fname in _ITERATOR_FUNCS:
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
        func = get_node(value, "func")
        if len(func) > 0 and _is_type(func, ["Name"]):
            wrapper_name = get_str(func, "id")
            if wrapper_name != "" and wrapper_name in _EAGER_CONSUMERS:
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
        call_func = get_node(value, "func")
        call_args = get_nodes(value, "args")
        if len(call_func) > 0 and _is_type(call_func, ["Name"]):
            wrapper_name = get_str(call_func, "id")
            if wrapper_name != "" and wrapper_name in _EAGER_CONSUMERS:
                return False
        if len(call_func) > 0 and _is_type(call_func, ["Attribute"]):
            attr_name = get_str(call_func, "attr")
            if attr_name == "join":
                return False
        j = 0
        while j < len(call_args):
            call_arg = call_args[j]
            if _is_generator_expr(call_arg):
                if len(call_func) > 0 and _is_type(call_func, ["Name"]):
                    wrapper_name = get_str(call_func, "id")
                    if wrapper_name != "" and wrapper_name in _EAGER_CONSUMERS:
                        return False
                ctx.result.add_error(lineno, 0, "cannot return generator expression")
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
    elts = get_nodes(node, "elts")
    if len(elts) < 2:
        return
    first = elts[0]
    first_type = _synth_expr(first, env, ctx)
    j = 1
    while j < len(elts):
        et = _synth_expr(elts[j], env, ctx)
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
        elif not _is_assignable(et, first_type, ctx.hier_result) and not _is_assignable(
            first_type, et, ctx.hier_result
        ):
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
    exp_vt: TypeNode | None = None
    if expected is not None:
        check_exp = expected
        if isinstance(check_exp, OptionalType):
            check_exp = check_exp.inner
        if isinstance(check_exp, MapType):
            if is_any(check_exp.value):
                return
            exp_vt = check_exp.value
    keys = get_nodes(node, "keys")
    values = get_nodes(node, "values")
    if len(keys) < 2:
        return
    first_kt = _synth_expr(keys[0], env, ctx)
    first_vt = ANY_TYPE
    if len(values) > 0:
        first_vt = _synth_expr(values[0], env, ctx)
    j = 1
    while j < len(keys):
        kt = _synth_expr(keys[j], env, ctx)
        if not _is_assignable(kt, first_kt, ctx.hier_result):
            ctx.result.add_error(lineno, 0, "mixed key types in dict literal")
            return
        if j < len(values):
            vt = _synth_expr(values[j], env, ctx)
            if exp_vt is not None:
                if not _is_assignable(vt, exp_vt, ctx.hier_result):
                    ctx.result.add_error(lineno, 0, "mixed value types in dict literal")
                    return
            elif not _is_assignable(vt, first_vt, ctx.hier_result):
                if _is_assignable(first_vt, vt, ctx.hier_result):
                    first_vt = vt
                else:
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
    name = get_str(node, "id")
    if name == "":
        return
    typ = env.get_type(name)
    if typ is None:
        return
    if isinstance(typ, PrimitiveType) and typ.kind == "never":
        return
    if isinstance(typ, PrimitiveType) and typ.kind == "void":
        if context == "arithmetic":
            ctx.result.add_error(lineno, 0, "cannot use None in arithmetic")
        elif context == "attribute":
            ctx.result.add_error(lineno, 0, "cannot access '" + attr_name + "' on None")
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
    if isinstance(typ, UnionType):
        if context == "attribute" and attr_name != "":
            if _all_union_members_have_attr(typ, attr_name, ctx):
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
    if isinstance(typ, InterfaceRef) and ctx.hier_result.is_hierarchy_root(typ.name):
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
    if context == "attribute" and attr_name != "":
        sname = _struct_name(typ)
        if sname != "" and sname in ctx.known_classes:
            if not _class_has_attr(sname, attr_name, ctx):
                if not _all_subclasses_have_attr(sname, attr_name, ctx):
                    ctx.result.add_error(
                        lineno, 0, "cannot access '" + attr_name + "' on " + sname
                    )


def _validate_expr_access(
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Check for un-narrowed access on object/union/optional types in an expression."""
    t = get_str(node, "_type")
    if t == "BinOp":
        err_snap = len(ctx.result._errors)
        binop_left = get_node(node, "left")
        binop_right = get_node(node, "right")
        if len(binop_left) > 0:
            _check_needs_narrowing(binop_left, env, ctx, lineno, "arithmetic", "")
        if _has_new_errors(ctx, err_snap):
            return
        if len(binop_right) > 0:
            _check_needs_narrowing(binop_right, env, ctx, lineno, "arithmetic", "")
        if _has_new_errors(ctx, err_snap):
            return
        if len(binop_left) > 0 and not _is_type(binop_left, ["Name"]):
            lt = _synth_expr(binop_left, env, ctx)
            if isinstance(lt, OptionalType):
                ctx.result.add_error(
                    lineno, 0, "cannot use optional type in arithmetic (may be None)"
                )
                return
        if len(binop_right) > 0 and not _is_type(binop_right, ["Name"]):
            rt = _synth_expr(binop_right, env, ctx)
            if isinstance(rt, OptionalType):
                ctx.result.add_error(
                    lineno, 0, "cannot use optional type in arithmetic (may be None)"
                )
                return
        binop_op = get_node(node, "op")
        binop_op_t = get_str(binop_op, "_type")
        if binop_op_t != "Mult":
            ltype = _synth_expr(binop_left, env, ctx)
            rtype = _synth_expr(binop_right, env, ctx)
            l_str = _prim_kind(ltype) == "string"
            r_str = _prim_kind(rtype) == "string"
            r_num = _prim_kind(rtype) in ("int", "float", "bool")
            l_num = _prim_kind(ltype) in ("int", "float", "bool")
            if l_str and r_num:
                ctx.result.add_error(lineno, 0, "cannot use str in arithmetic")
                return
            if r_str and l_num:
                ctx.result.add_error(lineno, 0, "cannot use str in arithmetic")
                return
        if len(binop_left) > 0:
            _validate_expr_access(binop_left, env, ctx, lineno)
        if len(binop_right) > 0:
            _validate_expr_access(binop_right, env, ctx, lineno)
        return
    if t == "Compare":
        err_snap = len(ctx.result._errors)
        ops = get_nodes(node, "ops")
        is_ordering = False
        oi = 0
        while oi < len(ops):
            op_type = get_str(ops[oi], "_type")
            if op_type in ("Lt", "LtE", "Gt", "GtE"):
                is_ordering = True
            oi += 1
        cmp_left = get_node(node, "left")
        if is_ordering:
            if len(cmp_left) > 0:
                _check_needs_narrowing(cmp_left, env, ctx, lineno, "arithmetic", "")
            if _has_new_errors(ctx, err_snap):
                return
            comparators = get_nodes(node, "comparators")
            ci = 0
            while ci < len(comparators):
                _check_needs_narrowing(
                    comparators[ci], env, ctx, lineno, "arithmetic", ""
                )
                if _has_new_errors(ctx, err_snap):
                    return
                ci += 1
        if len(cmp_left) > 0:
            _validate_expr_access(cmp_left, env, ctx, lineno)
            if _has_new_errors(ctx, err_snap):
                return
        comparators = get_nodes(node, "comparators")
        ci = 0
        while ci < len(comparators):
            _validate_expr_access(comparators[ci], env, ctx, lineno)
            if _has_new_errors(ctx, err_snap):
                return
            ci += 1
        return
    if t == "Attribute":
        err_snap = len(ctx.result._errors)
        value = get_node(node, "value")
        attr_str = get_str(node, "attr")
        if len(value) > 0 and attr_str != "kind":
            _check_needs_narrowing(value, env, ctx, lineno, "attribute", attr_str)
        if _has_new_errors(ctx, err_snap):
            return
        if (
            len(value) > 0
            and _is_type(value, ["Name"])
            and get_str(value, "id") == "self"
            and attr_str != ""
        ):
            self_type = env.get_type("self")
            use_this = self_type is None
            if self_type is not None and is_any(self_type):
                use_this = True
            if not use_this and self_type is not None:
                check_sname = _struct_name(self_type)
                if check_sname != "" and check_sname in ctx.known_classes:
                    if not _class_has_attr(check_sname, attr_str, ctx):
                        if not _all_subclasses_have_attr(check_sname, attr_str, ctx):
                            ctx.result.add_error(
                                lineno,
                                0,
                                "cannot access '" + attr_str + "' on " + check_sname,
                            )
                            return
            if use_this:
                this_type = env.get_type("this")
                if this_type is not None:
                    sname = _struct_name(this_type)
                    if sname != "" and sname in ctx.known_classes:
                        if not _class_has_attr(sname, attr_str, ctx):
                            if not _all_subclasses_have_attr(sname, attr_str, ctx):
                                ctx.result.add_error(
                                    lineno,
                                    0,
                                    "cannot access '" + attr_str + "' on " + sname,
                                )
                                return
        if len(value) > 0 and not _is_type(value, ["Name"]) and attr_str != "kind":
            val_path = _attr_path(value)
            is_guarded = val_path != "" and env.is_attr_guarded(val_path)
            obj_type = _synth_expr(value, env, ctx)
            if not is_guarded and isinstance(obj_type, OptionalType):
                ctx.result.add_error(
                    lineno,
                    0,
                    "cannot access '" + attr_str + "' on optional type (may be None)",
                )
                return
            if _is_type(value, ["Attribute"]):
                check_type = obj_type
                if isinstance(check_type, OptionalType):
                    check_type = check_type.inner
                sname = _struct_name(check_type)
                if sname != "" and sname in ctx.known_classes:
                    if not _class_has_attr(sname, attr_str, ctx):
                        if not _all_subclasses_have_attr(sname, attr_str, ctx):
                            ctx.result.add_error(
                                lineno,
                                0,
                                "cannot access '" + attr_str + "' on " + sname,
                            )
                            return
                if not is_guarded and isinstance(check_type, UnionType):
                    vi = 0
                    while vi < len(check_type.variants):
                        v = check_type.variants[vi]
                        vsname = _struct_name(v)
                        if vsname != "" and vsname in ctx.known_classes:
                            if not _class_has_attr(vsname, attr_str, ctx):
                                ctx.result.add_error(
                                    lineno,
                                    0,
                                    "cannot access '" + attr_str + "' on " + vsname,
                                )
                                return
                        vi += 1
        if len(value) > 0:
            _validate_expr_access(value, env, ctx, lineno)
        return
    if t == "Subscript":
        value = get_node(node, "value")
        if len(value) > 0:
            _check_needs_narrowing(value, env, ctx, lineno, "subscript", "")
            _validate_expr_access(value, env, ctx, lineno)
        return
    if t == "Call":
        err_snap = len(ctx.result._errors)
        call_func = get_node(node, "func")
        if len(call_func) > 0:
            _validate_expr_access(call_func, env, ctx, lineno)
        if _has_new_errors(ctx, err_snap):
            return
        call_args = get_nodes(node, "args")
        is_builtin_call = (
            _is_type(call_func, ["Name"]) and get_str(call_func, "id") in _BUILTIN_FUNCS
        )
        j = 0
        while j < len(call_args):
            if is_builtin_call:
                _check_builtin_arg_optional(call_args[j], env, ctx, lineno)
                if _has_new_errors(ctx, err_snap):
                    return
            _validate_expr_access(call_args[j], env, ctx, lineno)
            if _has_new_errors(ctx, err_snap):
                return
            j += 1
        return
    if t == "BoolOp":
        err_snap = len(ctx.result._errors)
        op = get_node(node, "op")
        op_t = get_str(op, "_type")
        values = get_nodes(node, "values")
        if len(values) == 0:
            return
        _validate_expr_access(values[0], env, ctx, lineno)
        if _has_new_errors(ctx, err_snap):
            return
        if len(values) > 1:
            narrowed_env = env.copy()
            dummy_env = env.copy()
            if op_t == "And":
                _extract_narrowing(values[0], narrowed_env, dummy_env, ctx)
            elif op_t == "Or":
                _extract_narrowing(values[0], dummy_env, narrowed_env, ctx)
            j = 1
            while j < len(values):
                _validate_expr_access(values[j], narrowed_env, ctx, lineno)
                if _has_new_errors(ctx, err_snap):
                    return
                if j + 1 < len(values):
                    dummy2 = narrowed_env.copy()
                    if op_t == "And":
                        _extract_narrowing(values[j], narrowed_env, dummy2, ctx)
                    elif op_t == "Or":
                        _extract_narrowing(values[j], dummy2, narrowed_env, ctx)
                j += 1
        return
    if t == "IfExp":
        err_snap = len(ctx.result._errors)
        test = get_node(node, "test")
        ifbody = get_node(node, "body")
        orelse = get_node(node, "orelse")
        if len(test) > 0:
            _validate_expr_access(test, env, ctx, lineno)
            if _has_new_errors(ctx, err_snap):
                return
        then_env = env.copy()
        else_env = env.copy()
        if len(test) > 0:
            _extract_narrowing(test, then_env, else_env, ctx)
        if len(ifbody) > 0:
            _validate_expr_access(ifbody, then_env, ctx, lineno)
            if _has_new_errors(ctx, err_snap):
                return
        if len(orelse) > 0:
            _validate_expr_access(orelse, else_env, ctx, lineno)
        return


def _all_subclasses_have_attr(base_name: str, attr_name: str, ctx: _InferCtx) -> bool:
    """Check if all direct subclasses of base_name have the given attribute.

    Only returns True when there are 2+ subclasses, to avoid suppressing
    errors in simple base/child hierarchies where narrowing is expected.
    """
    all_classes = list(ctx.class_bases.keys())
    count = 0
    i = 0
    while i < len(all_classes):
        cls = all_classes[i]
        bases = ctx.class_bases.get(cls, [])
        j = 0
        while j < len(bases):
            if bases[j] == base_name:
                count += 1
                if not _class_has_attr(cls, attr_name, ctx):
                    return False
            j += 1
        i += 1
    return count >= 2


def _check_builtin_arg_optional(
    node: ASTNode, env: TypeEnv, ctx: _InferCtx, lineno: int
) -> None:
    """Check if a builtin function argument is optional (may be None)."""
    if _is_type(node, ["Name"]):
        name = get_str(node, "id")
        if name == "":
            return
        typ = env.get_type(name)
        if typ is not None and isinstance(typ, OptionalType):
            ctx.result.add_error(
                lineno,
                0,
                "cannot use optional type in arithmetic (may be None)",
            )


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
                cls_methods = ctx.tc_result.methods.get(cls)
                if cls_methods is not None and method_name in cls_methods:
                    return True
            j += 1
        i += 1
    return False


def _class_has_attr(class_name: str, attr_name: str, ctx: _InferCtx) -> bool:
    """Check if a class has the given attribute, including inherited ones."""
    current = class_name
    while current != "":
        cls = ctx.tc_result.classes.get(current)
        if cls is not None:
            if attr_name in cls.fields or attr_name in cls.const_fields:
                return True
        methods = ctx.tc_result.methods.get(current)
        if methods is not None and attr_name in methods:
            return True
        bases = ctx.class_bases.get(current)
        if bases is not None and len(bases) > 0:
            current = bases[0]
        else:
            current = ""
    return False


def _all_union_members_have_attr(typ: TypeNode, attr_name: str, ctx: _InferCtx) -> bool:
    """Check if all union variant struct/interface types have the given attribute."""
    if not isinstance(typ, UnionType):
        return False
    names = union_variant_names(typ)
    if len(names) == 0:
        return False
    i = 0
    while i < len(names):
        if not _class_has_attr(names[i], attr_name, ctx):
            return False
        i += 1
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_inference(
    tree: ASTNode,
    tc_result: TypeCollectResult,
    hier_result: HierarchyResult,
    known_classes: set[str],
    class_bases: dict[str, list[str]],
    flow_graphs: dict[str, FlowGraph],
) -> InferenceResult:
    """Run type inference and validation on the module AST."""
    result = InferenceResult()
    ctx = _InferCtx(
        tc_result, hier_result, known_classes, class_bases, result, flow_graphs
    )
    body = get_nodes(tree, "body")
    if len(body) == 0:
        return result
    i = 0
    while i < len(body):
        node = body[i]
        if get_str(node, "_type") == "AnnAssign":
            target = get_node(node, "target")
            annotation = get_node(node, "annotation")
            if (
                len(target) > 0
                and get_str(target, "_type") == "Name"
                and len(annotation) > 0
            ):
                var_name = get_str(target, "id")
                if var_name != "":
                    ann_str = annotation_to_str(annotation)
                    if ann_str != "":
                        sig_errors: list[TypeCollectError] = []
                        var_type = py_type_to_type_dict(
                            ann_str, known_classes, sig_errors, 0, 0
                        )
                        if len(sig_errors) == 0:
                            ctx.module_vars[var_name] = var_type
        i += 1
    i = 0
    while i < len(body):
        node = body[i]
        t = get_str(node, "_type")
        sf = get_str(node, "_source_file")
        if t == "FunctionDef":
            err_before = len(result._errors)
            _validate_func(node, ctx, "")
            ei = err_before
            while ei < len(result._errors):
                result._errors[ei].source_file = sf
                ei += 1
        elif t == "ClassDef":
            class_name = get_str(node, "name")
            class_body = get_nodes(node, "body")
            j = 0
            while j < len(class_body):
                stmt = class_body[j]
                if _is_type(stmt, ["FunctionDef"]):
                    stmt_sf = get_str(stmt, "_source_file")
                    if stmt_sf == "":
                        stmt_sf = sf
                    err_before = len(result._errors)
                    _validate_func(stmt, ctx, class_name)
                    ei = err_before
                    while ei < len(result._errors):
                        result._errors[ei].source_file = stmt_sf
                        ei += 1
                j += 1
        i += 1
    return result
