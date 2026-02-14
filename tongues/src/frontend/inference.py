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

# Type alias for AST dict nodes
ASTNode = dict[str, object]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class InferenceError:
    """An error found during inference."""

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
            + ": [inference] "
            + self.message
        )


class InferenceResult:
    """Result of inference analysis."""

    def __init__(self) -> None:
        self._errors: list[InferenceError] = []

    def add_error(self, lineno: int, col: int, message: str) -> None:
        self._errors.append(InferenceError(lineno, col, message))

    def errors(self) -> list[InferenceError]:
        return self._errors


# ---------------------------------------------------------------------------
# Type dict helpers
# ---------------------------------------------------------------------------

_ANY_TYPE: dict[str, object] = {"_type": "InterfaceRef", "name": "any"}
_INT_TYPE: dict[str, object] = {"kind": "int"}
_FLOAT_TYPE: dict[str, object] = {"kind": "float"}
_BOOL_TYPE: dict[str, object] = {"kind": "bool"}
_STR_TYPE: dict[str, object] = {"kind": "string"}
_BYTES_TYPE: dict[str, object] = {"_type": "Slice", "element": {"kind": "byte"}}
_VOID_TYPE: dict[str, object] = {"kind": "void"}


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


def _type_eq(a: dict[str, object], b: dict[str, object]) -> bool:
    """Check structural equality of two type dicts."""
    if a == b:
        return True
    a_kind = a.get("kind")
    b_kind = b.get("kind")
    if a_kind is not None or b_kind is not None:
        return a_kind == b_kind
    a_type = a.get("_type")
    b_type = b.get("_type")
    if a_type != b_type:
        return False
    if a_type == "Slice":
        ae = a.get("element")
        be = b.get("element")
        if isinstance(ae, dict) and isinstance(be, dict):
            return _type_eq(ae, be)
        return False
    if a_type == "Map":
        ak = a.get("key")
        bk = b.get("key")
        av = a.get("value")
        bv = b.get("value")
        if (
            isinstance(ak, dict)
            and isinstance(bk, dict)
            and isinstance(av, dict)
            and isinstance(bv, dict)
        ):
            return _type_eq(ak, bk) and _type_eq(av, bv)
        return False
    if a_type == "Set":
        ae = a.get("element")
        be = b.get("element")
        if isinstance(ae, dict) and isinstance(be, dict):
            return _type_eq(ae, be)
        return False
    if a_type == "Optional":
        ai = a.get("inner")
        bi = b.get("inner")
        if isinstance(ai, dict) and isinstance(bi, dict):
            return _type_eq(ai, bi)
        return False
    if a_type == "Tuple":
        a_elems = a.get("elements")
        b_elems = b.get("elements")
        a_var = a.get("variadic", False)
        b_var = b.get("variadic", False)
        if a_var != b_var:
            return False
        if not isinstance(a_elems, list) or not isinstance(b_elems, list):
            return False
        if len(a_elems) != len(b_elems):
            return False
        j = 0
        while j < len(a_elems):
            ae = a_elems[j]
            be = b_elems[j]
            if isinstance(ae, dict) and isinstance(be, dict):
                if not _type_eq(ae, be):
                    return False
            else:
                return False
            j += 1
        return True
    if a_type == "Pointer":
        at = a.get("target")
        bt = b.get("target")
        if isinstance(at, dict) and isinstance(bt, dict):
            return _type_eq(at, bt)
        return False
    if a_type == "StructRef":
        return a.get("name") == b.get("name")
    if a_type == "InterfaceRef":
        return a.get("name") == b.get("name")
    if a_type == "FuncType":
        ap = a.get("params")
        bp = b.get("params")
        ar = a.get("ret")
        br = b.get("ret")
        if not isinstance(ap, list) or not isinstance(bp, list):
            return False
        if len(ap) != len(bp):
            return False
        j = 0
        while j < len(ap):
            if isinstance(ap[j], dict) and isinstance(bp[j], dict):
                if not _type_eq(ap[j], bp[j]):
                    return False
            else:
                return False
            j += 1
        if isinstance(ar, dict) and isinstance(br, dict):
            return _type_eq(ar, br)
        return False
    return a == b


def _is_any(t: dict[str, object]) -> bool:
    return t.get("_type") == "InterfaceRef" and t.get("name") == "any"


def _is_optional(t: dict[str, object]) -> bool:
    return t.get("_type") == "Optional"


def _unwrap_optional(t: dict[str, object]) -> dict[str, object]:
    inner = t.get("inner")
    if isinstance(inner, dict):
        return inner
    return _ANY_TYPE


def _is_struct_pointer(t: dict[str, object]) -> bool:
    if t.get("_type") == "Pointer":
        target = t.get("target")
        if isinstance(target, dict) and target.get("_type") == "StructRef":
            return True
    return False


def _struct_name(t: dict[str, object]) -> str:
    """Extract struct name from Pointer(StructRef(name)) or StructRef(name)."""
    if t.get("_type") == "Pointer":
        target = t.get("target")
        if isinstance(target, dict) and target.get("_type") == "StructRef":
            name = target.get("name")
            if isinstance(name, str):
                return name
    if t.get("_type") == "StructRef":
        name = t.get("name")
        if isinstance(name, str):
            return name
    return ""


def _type_name(t: dict[str, object]) -> str:
    """Human-readable type name for error messages."""
    kind = t.get("kind")
    if isinstance(kind, str):
        if kind == "string":
            return "str"
        return kind
    tt = t.get("_type")
    if tt == "Slice":
        elem = t.get("element")
        if isinstance(elem, dict):
            return "list[" + _type_name(elem) + "]"
        return "list"
    if tt == "Map":
        k = t.get("key")
        v = t.get("value")
        if isinstance(k, dict) and isinstance(v, dict):
            return "dict[" + _type_name(k) + ", " + _type_name(v) + "]"
        return "dict"
    if tt == "Set":
        elem = t.get("element")
        if isinstance(elem, dict):
            return "set[" + _type_name(elem) + "]"
        return "set"
    if tt == "Tuple":
        elems = t.get("elements")
        if isinstance(elems, list) and len(elems) > 0:
            parts: list[str] = []
            j = 0
            while j < len(elems):
                e = elems[j]
                if isinstance(e, dict):
                    parts.append(_type_name(e))
                j += 1
            if t.get("variadic"):
                return "tuple[" + ", ".join(parts) + ", ...]"
            return "tuple[" + ", ".join(parts) + "]"
        return "tuple"
    if tt == "Optional":
        inner = t.get("inner")
        if isinstance(inner, dict):
            return _type_name(inner) + " | None"
        return "Optional"
    if tt == "Pointer":
        target = t.get("target")
        if isinstance(target, dict):
            return _type_name(target)
        return "Pointer"
    if tt == "StructRef":
        name = t.get("name")
        if isinstance(name, str):
            return name
        return "struct"
    if tt == "InterfaceRef":
        name = t.get("name")
        if name == "any":
            return "object"
        return "interface"
    if tt == "FuncType":
        return "Callable"
    return "unknown"


# ---------------------------------------------------------------------------
# Assignability
# ---------------------------------------------------------------------------


def _is_assignable(
    actual: dict[str, object],
    expected: dict[str, object],
    hier: HierarchyResult,
) -> bool:
    """Check if actual is assignable to expected."""
    if _type_eq(actual, expected):
        return True
    if _is_any(actual) or _is_any(expected):
        return True
    # void (None literal) assignable to Optional
    if actual.get("kind") == "void":
        if _is_optional(expected):
            return True
        # void assignable to interface (for hierarchy widening like Token = None)
        if expected.get("_type") == "InterfaceRef":
            return True
        return False
    # bool <: int <: float
    if actual.get("kind") == "bool" and expected.get("kind") == "int":
        return True
    if actual.get("kind") == "bool" and expected.get("kind") == "float":
        return True
    if actual.get("kind") == "int" and expected.get("kind") == "float":
        return True
    # T assignable to T?
    if _is_optional(expected):
        inner = _unwrap_optional(expected)
        if _is_assignable(actual, inner, hier):
            return True
    # Struct hierarchy: subclass assignable to base / interface
    if _is_struct_pointer(actual):
        a_name = _struct_name(actual)
        if _is_struct_pointer(expected):
            e_name = _struct_name(expected)
            if a_name == e_name:
                return True
            if hier.is_node(a_name) and hier.is_node(e_name):
                ancestors = hier.ancestors.get(a_name)
                if ancestors is not None:
                    j = 0
                    while j < len(ancestors):
                        if ancestors[j] == e_name:
                            return True
                        j += 1
                # Deep ancestor check
                return _is_ancestor(a_name, e_name, hier)
        # Struct assignable to InterfaceRef (hierarchy root)
        if expected.get("_type") == "InterfaceRef":
            if hier.is_node(a_name):
                return True
    # StructRef without Pointer wrapper
    if actual.get("_type") == "StructRef":
        a_name = _struct_name(actual)
        if expected.get("_type") == "InterfaceRef":
            if hier.is_node(a_name):
                return True
        if expected.get("_type") == "StructRef":
            return _struct_name(actual) == _struct_name(expected)
        if _is_struct_pointer(expected):
            return _struct_name(actual) == _struct_name(expected)
    # Collection element assignability (invariant but check same direction)
    if actual.get("_type") == "Slice" and expected.get("_type") == "Slice":
        ae = actual.get("element")
        be = expected.get("element")
        if isinstance(ae, dict) and isinstance(be, dict):
            return _is_assignable(ae, be, hier)
    if actual.get("_type") == "Map" and expected.get("_type") == "Map":
        ak = actual.get("key")
        bk = expected.get("key")
        av = actual.get("value")
        bv = expected.get("value")
        if (
            isinstance(ak, dict)
            and isinstance(bk, dict)
            and isinstance(av, dict)
            and isinstance(bv, dict)
        ):
            return _is_assignable(ak, bk, hier) and _is_assignable(av, bv, hier)
    if actual.get("_type") == "Set" and expected.get("_type") == "Set":
        ae = actual.get("element")
        be = expected.get("element")
        if isinstance(ae, dict) and isinstance(be, dict):
            return _is_assignable(ae, be, hier)
    # Tuple assignability
    if actual.get("_type") == "Tuple" and expected.get("_type") == "Tuple":
        a_elems = actual.get("elements")
        b_elems = expected.get("elements")
        a_var = actual.get("variadic", False)
        b_var = expected.get("variadic", False)
        if isinstance(a_elems, list) and isinstance(b_elems, list):
            if a_var and b_var:
                if len(a_elems) > 0 and len(b_elems) > 0:
                    ae = a_elems[0]
                    be = b_elems[0]
                    if isinstance(ae, dict) and isinstance(be, dict):
                        return _is_assignable(ae, be, hier)
            if not a_var and not b_var:
                if len(a_elems) != len(b_elems):
                    return False
                j = 0
                while j < len(a_elems):
                    ae = a_elems[j]
                    be = b_elems[j]
                    if isinstance(ae, dict) and isinstance(be, dict):
                        if not _is_assignable(ae, be, hier):
                            return False
                    j += 1
                return True
    # FuncType assignability
    if actual.get("_type") == "FuncType" and expected.get("_type") == "FuncType":
        ap = actual.get("params")
        bp = expected.get("params")
        ar = actual.get("ret")
        br = expected.get("ret")
        if isinstance(ap, list) and isinstance(bp, list):
            if len(ap) != len(bp):
                return False
            j = 0
            while j < len(ap):
                if isinstance(ap[j], dict) and isinstance(bp[j], dict):
                    if not _is_assignable(bp[j], ap[j], hier):
                        return False
                j += 1
        if isinstance(ar, dict) and isinstance(br, dict):
            return _is_assignable(ar, br, hier)
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
            current = []
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
        self.types: dict[str, dict[str, object]] = {}
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

    def set(self, name: str, typ: dict[str, object], source: str) -> None:
        self.types[name] = typ
        self.source_types[name] = source

    def get_type(self, name: str) -> dict[str, object] | None:
        return self.types.get(name)

    def get_source(self, name: str) -> str:
        return self.source_types.get(name, "")

    def narrow(self, name: str, typ: dict[str, object], source: str) -> None:
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
) -> dict[str, object]:
    """Synthesize the type of an expression node."""
    if not isinstance(node, dict):
        return _ANY_TYPE
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
        return _BOOL_TYPE
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
        return _ANY_TYPE
    if t == "JoinedStr":
        return _STR_TYPE
    if t == "FormattedValue":
        return _STR_TYPE
    if t == "NamedExpr":
        return _synth_namedexpr(node, env, ctx)
    if t == "Starred":
        return _ANY_TYPE
    return _ANY_TYPE


def _synth_constant(node: ASTNode) -> dict[str, object]:
    v = node.get("value")
    if v is None:
        return _VOID_TYPE
    if isinstance(v, bool):
        return _BOOL_TYPE
    if isinstance(v, int):
        return _INT_TYPE
    if isinstance(v, float):
        return _FLOAT_TYPE
    if isinstance(v, str):
        return _STR_TYPE
    if isinstance(v, bytes):
        return _BYTES_TYPE
    return _ANY_TYPE


def _synth_name(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    name = node.get("id")
    if not isinstance(name, str):
        return _ANY_TYPE
    typ = env.get_type(name)
    if typ is not None:
        return typ
    # User-defined function reference -> FuncType
    func_info = ctx.sig_result.functions.get(name)
    if func_info is not None:
        params: list[object] = []
        j = 0
        while j < len(func_info.params):
            params.append(func_info.params[j].typ)
            j += 1
        return {"_type": "FuncType", "params": params, "ret": func_info.return_type}
    # Builtin function references
    if name == "len":
        return {"_type": "FuncType", "params": [_ANY_TYPE], "ret": _INT_TYPE}
    if name == "str":
        return {"_type": "FuncType", "params": [_ANY_TYPE], "ret": _STR_TYPE}
    if name == "int":
        return {"_type": "FuncType", "params": [_ANY_TYPE], "ret": _INT_TYPE}
    if name == "bool":
        return {"_type": "FuncType", "params": [_ANY_TYPE], "ret": _BOOL_TYPE}
    return _ANY_TYPE


def _synth_attribute(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    value = node.get("value")
    attr = node.get("attr")
    if not isinstance(value, dict) or not isinstance(attr, str):
        return _ANY_TYPE
    obj_type = _synth_expr(value, env, ctx)
    return _resolve_attr(obj_type, attr, value, env, ctx)


def _resolve_attr(
    obj_type: dict[str, object],
    attr: str,
    value_node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
) -> dict[str, object]:
    """Resolve attribute access on a type."""
    # Unwrap Pointer to collection (not struct)
    if obj_type.get("_type") == "Pointer":
        target = obj_type.get("target")
        if isinstance(target, dict) and target.get("_type") != "StructRef":
            obj_type = target
    # String methods
    if obj_type.get("kind") == "string":
        if (
            attr == "upper"
            or attr == "lower"
            or attr == "strip"
            or attr == "lstrip"
            or attr == "rstrip"
        ):
            return {"_type": "FuncType", "params": [], "ret": _STR_TYPE}
        if attr == "split":
            return {
                "_type": "FuncType",
                "params": [],
                "ret": {"_type": "Slice", "element": _STR_TYPE},
            }
        if attr == "join":
            return {
                "_type": "FuncType",
                "params": [{"_type": "Slice", "element": _STR_TYPE}],
                "ret": _STR_TYPE,
            }
        if attr == "replace" or attr == "format":
            return {"_type": "FuncType", "params": [_STR_TYPE], "ret": _STR_TYPE}
        if attr == "startswith" or attr == "endswith":
            return {"_type": "FuncType", "params": [_STR_TYPE], "ret": _BOOL_TYPE}
        if attr == "find" or attr == "index" or attr == "count":
            return {"_type": "FuncType", "params": [_STR_TYPE], "ret": _INT_TYPE}
        return _ANY_TYPE
    # List methods
    if obj_type.get("_type") == "Slice":
        elem = obj_type.get("element")
        if not isinstance(elem, dict):
            elem = _ANY_TYPE
        if attr == "append":
            return {"_type": "FuncType", "params": [elem], "ret": _VOID_TYPE}
        if attr == "extend":
            return {
                "_type": "FuncType",
                "params": [{"_type": "Slice", "element": elem}],
                "ret": _VOID_TYPE,
            }
        if attr == "insert":
            return {"_type": "FuncType", "params": [_INT_TYPE, elem], "ret": _VOID_TYPE}
        if attr == "pop":
            return {"_type": "FuncType", "params": [], "ret": elem}
        if attr == "copy":
            return {"_type": "FuncType", "params": [], "ret": obj_type}
        if attr == "sort":
            return {"_type": "FuncType", "params": [], "ret": _VOID_TYPE}
        if attr == "reverse":
            return {"_type": "FuncType", "params": [], "ret": _VOID_TYPE}
        if attr == "clear":
            return {"_type": "FuncType", "params": [], "ret": _VOID_TYPE}
        if attr == "count":
            return {"_type": "FuncType", "params": [elem], "ret": _INT_TYPE}
        if attr == "index":
            return {"_type": "FuncType", "params": [elem], "ret": _INT_TYPE}
        if attr == "remove":
            return {"_type": "FuncType", "params": [elem], "ret": _VOID_TYPE}
        return _ANY_TYPE
    # Dict methods
    if obj_type.get("_type") == "Map":
        key_t = obj_type.get("key")
        val_t = obj_type.get("value")
        if not isinstance(key_t, dict):
            key_t = _ANY_TYPE
        if not isinstance(val_t, dict):
            val_t = _ANY_TYPE
        if attr == "get":
            return {
                "_type": "FuncType",
                "params": [key_t],
                "ret": {"_type": "Optional", "inner": val_t},
            }
        if attr == "keys":
            return {
                "_type": "FuncType",
                "params": [],
                "ret": {"_type": "Slice", "element": key_t},
            }
        if attr == "values":
            return {
                "_type": "FuncType",
                "params": [],
                "ret": {"_type": "Slice", "element": val_t},
            }
        if attr == "items":
            return {
                "_type": "FuncType",
                "params": [],
                "ret": {
                    "_type": "Slice",
                    "element": {
                        "_type": "Tuple",
                        "elements": [key_t, val_t],
                        "variadic": False,
                    },
                },
            }
        if attr == "pop":
            return {"_type": "FuncType", "params": [key_t], "ret": val_t}
        if attr == "update":
            return {"_type": "FuncType", "params": [obj_type], "ret": _VOID_TYPE}
        if attr == "copy":
            return {"_type": "FuncType", "params": [], "ret": obj_type}
        if attr == "clear":
            return {"_type": "FuncType", "params": [], "ret": _VOID_TYPE}
        if attr == "setdefault":
            return {"_type": "FuncType", "params": [key_t, val_t], "ret": val_t}
        return _ANY_TYPE
    # Set methods
    if obj_type.get("_type") == "Set":
        elem = obj_type.get("element")
        if not isinstance(elem, dict):
            elem = _ANY_TYPE
        if attr == "add":
            return {"_type": "FuncType", "params": [elem], "ret": _VOID_TYPE}
        if attr == "remove" or attr == "discard":
            return {"_type": "FuncType", "params": [elem], "ret": _VOID_TYPE}
        if attr == "union" or attr == "intersection" or attr == "difference":
            return {"_type": "FuncType", "params": [obj_type], "ret": obj_type}
        if attr == "copy":
            return {"_type": "FuncType", "params": [], "ret": obj_type}
        if attr == "clear":
            return {"_type": "FuncType", "params": [], "ret": _VOID_TYPE}
        return _ANY_TYPE
    # Struct field access
    sname = _struct_name(obj_type)
    if sname != "":
        return _resolve_struct_attr(sname, attr, ctx)
    # Optional: error handled by validator
    if _is_optional(obj_type):
        inner = _unwrap_optional(obj_type)
        return _resolve_attr(inner, attr, value_node, env, ctx)
    return _ANY_TYPE


def _resolve_struct_attr(sname: str, attr: str, ctx: _InferCtx) -> dict[str, object]:
    """Resolve attribute on a struct type."""
    # Check fields
    cls = ctx.field_result.classes.get(sname)
    if cls is not None:
        fld = cls.fields.get(attr)
        if fld is not None:
            return fld.typ
        # Check const fields (kind)
        if attr in cls.const_fields:
            return _STR_TYPE
    # Check methods — return FuncType (consistent with builtin method resolution)
    methods = ctx.sig_result.methods.get(sname)
    if methods is not None:
        method = methods.get(attr)
        if method is not None:
            params: list[object] = []
            j = 0
            while j < len(method.params):
                if method.params[j].name != "self":
                    params.append(method.params[j].typ)
                j += 1
            return {"_type": "FuncType", "params": params, "ret": method.return_type}
    return _ANY_TYPE


def _synth_call(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    """Synthesize return type of a call."""
    func = node.get("func")
    if not isinstance(func, dict):
        return _ANY_TYPE
    args = node.get("args", [])
    if not isinstance(args, list):
        args = []
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
    if func_type.get("_type") == "FuncType":
        ret = func_type.get("ret")
        if isinstance(ret, dict):
            return ret
    return _ANY_TYPE


def _synth_name_call(
    fname: str,
    args: list[object],
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
) -> dict[str, object]:
    """Synthesize return type for a direct function call."""
    # Builtin constructors/conversions
    if fname == "int":
        return _INT_TYPE
    if fname == "float":
        return _FLOAT_TYPE
    if fname == "str":
        return _STR_TYPE
    if fname == "bool":
        return _BOOL_TYPE
    if fname == "len":
        return _INT_TYPE
    if fname == "abs":
        return _INT_TYPE
    if fname == "ord":
        return _INT_TYPE
    if fname == "chr":
        return _STR_TYPE
    if fname == "repr":
        return _STR_TYPE
    if fname == "round":
        return _INT_TYPE
    if fname == "sum":
        return _INT_TYPE
    if fname == "min" or fname == "max":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                # min/max on a collection returns element type
                if ft.get("_type") == "Slice":
                    elem = ft.get("element")
                    if isinstance(elem, dict):
                        return elem
                # min/max on scalars: if mixed bool+int, return int
                if len(args) >= 2:
                    has_int = False
                    has_bool = False
                    j = 0
                    while j < len(args):
                        a = args[j]
                        if isinstance(a, dict):
                            at = _synth_expr(a, env, ctx)
                            if at.get("kind") == "int":
                                has_int = True
                            if at.get("kind") == "bool":
                                has_bool = True
                        j += 1
                    if has_int or has_bool:
                        return _INT_TYPE
                return ft
        return _INT_TYPE
    if fname == "isinstance":
        return _BOOL_TYPE
    if fname == "hash":
        return _INT_TYPE
    if fname == "range":
        return _ANY_TYPE  # range is special, not a true iterator
    if fname == "enumerate":
        # Returns iterator of (int, T)
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                elem = _element_type(ft)
                return {
                    "_type": "_Iterator",
                    "element": {
                        "_type": "Tuple",
                        "elements": [_INT_TYPE, elem],
                        "variadic": False,
                    },
                    "source": "enumerate",
                }
        return {
            "_type": "_Iterator",
            "element": {
                "_type": "Tuple",
                "elements": [_INT_TYPE, _ANY_TYPE],
                "variadic": False,
            },
            "source": "enumerate",
        }
    if fname == "zip":
        # Returns iterator of tuples
        elems: list[object] = []
        j = 0
        while j < len(args):
            a = args[j]
            if isinstance(a, dict):
                ft = _synth_expr(a, env, ctx)
                elems.append(_element_type(ft))
            j += 1
        return {
            "_type": "_Iterator",
            "element": {"_type": "Tuple", "elements": elems, "variadic": False},
            "source": "zip",
        }
    if fname == "reversed":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                elem = _element_type(ft)
                return {"_type": "_Iterator", "element": elem, "source": "reversed"}
        return {"_type": "_Iterator", "element": _ANY_TYPE, "source": "reversed"}
    if fname == "sorted":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                # If iterator, unwrap
                if ft.get("_type") == "_Iterator":
                    elem = ft.get("element")
                    if isinstance(elem, dict):
                        return {"_type": "Slice", "element": elem}
                elem = _element_type(ft)
                return {"_type": "Slice", "element": elem}
        return {"_type": "Slice", "element": _ANY_TYPE}
    if fname == "list":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                if ft.get("_type") == "_Iterator":
                    elem = ft.get("element")
                    if isinstance(elem, dict):
                        return {"_type": "Slice", "element": elem}
                elem = _element_type(ft)
                return {"_type": "Slice", "element": elem}
        return {"_type": "Slice", "element": _ANY_TYPE}
    if fname == "tuple":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                if ft.get("_type") == "_Iterator":
                    elem = ft.get("element")
                    if isinstance(elem, dict):
                        return {"_type": "Tuple", "elements": [elem], "variadic": True}
                elem = _element_type(ft)
                return {"_type": "Tuple", "elements": [elem], "variadic": True}
        return {"_type": "Tuple", "elements": [], "variadic": False}
    if fname == "set":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                elem = _element_type(ft)
                return {"_type": "Set", "element": elem}
        return {"_type": "Set", "element": _ANY_TYPE}
    if fname == "dict":
        if len(args) > 0:
            first = args[0]
            if isinstance(first, dict):
                ft = _synth_expr(first, env, ctx)
                if ft.get("_type") == "_Iterator":
                    elem = ft.get("element")
                    if isinstance(elem, dict) and elem.get("_type") == "Tuple":
                        telems = elem.get("elements")
                        if isinstance(telems, list) and len(telems) >= 2:
                            k = telems[0]
                            v = telems[1]
                            if isinstance(k, dict) and isinstance(v, dict):
                                return {"_type": "Map", "key": k, "value": v}
        return {"_type": "Map", "key": _ANY_TYPE, "value": _ANY_TYPE}
    if fname == "any" or fname == "all":
        return _BOOL_TYPE
    if fname == "divmod":
        return {"_type": "Tuple", "elements": [_INT_TYPE, _INT_TYPE], "variadic": False}
    if fname == "print":
        return _VOID_TYPE
    # User-defined function
    func_info = ctx.sig_result.functions.get(fname)
    if func_info is not None:
        return func_info.return_type
    # Class constructor
    if fname in ctx.known_classes:
        return {"_type": "Pointer", "target": {"_type": "StructRef", "name": fname}}
    # Callable variable
    typ = env.get_type(fname)
    if typ is not None and typ.get("_type") == "FuncType":
        ret = typ.get("ret")
        if isinstance(ret, dict):
            return ret
    return _ANY_TYPE


def _synth_method_call(
    func: ASTNode,
    args: list[object],
    node: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
) -> dict[str, object]:
    """Synthesize return type of a method call (obj.method(...))."""
    obj = func.get("value")
    attr = func.get("attr")
    if not isinstance(obj, dict) or not isinstance(attr, str):
        return _ANY_TYPE
    obj_type = _synth_expr(obj, env, ctx)
    # String join special case
    if obj_type.get("kind") == "string" and attr == "join":
        return _STR_TYPE
    attr_type = _resolve_attr(obj_type, attr, obj, env, ctx)
    if attr_type.get("_type") == "FuncType":
        ret = attr_type.get("ret")
        if isinstance(ret, dict):
            return ret
    # Direct method return type from sig table
    sname = _struct_name(obj_type)
    if sname != "":
        methods = ctx.sig_result.methods.get(sname)
        if methods is not None:
            method = methods.get(attr)
            if method is not None:
                return method.return_type
    return attr_type


def _element_type(t: dict[str, object]) -> dict[str, object]:
    """Get the element type of a collection."""
    if t.get("_type") == "Slice":
        elem = t.get("element")
        if isinstance(elem, dict):
            return elem
    if t.get("_type") == "Set":
        elem = t.get("element")
        if isinstance(elem, dict):
            return elem
    if t.get("_type") == "Map":
        key = t.get("key")
        if isinstance(key, dict):
            return key
    if t.get("_type") == "Tuple":
        elems = t.get("elements")
        if isinstance(elems, list) and len(elems) > 0:
            e = elems[0]
            if isinstance(e, dict):
                return e
    if t.get("kind") == "string":
        return _STR_TYPE
    return _ANY_TYPE


def _synth_subscript(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    value = node.get("value")
    slc = node.get("slice")
    if not isinstance(value, dict):
        return _ANY_TYPE
    obj_type = _synth_expr(value, env, ctx)
    # String indexing
    if obj_type.get("kind") == "string":
        if isinstance(slc, dict) and _is_type(slc, ["Slice"]):
            return _STR_TYPE
        return _STR_TYPE
    # List indexing
    if obj_type.get("_type") == "Slice":
        elem = obj_type.get("element")
        if isinstance(slc, dict) and _is_type(slc, ["Slice"]):
            return obj_type
        if isinstance(elem, dict):
            return elem
    # Dict indexing
    if obj_type.get("_type") == "Map":
        val = obj_type.get("value")
        if isinstance(val, dict):
            return val
    # Tuple indexing
    if obj_type.get("_type") == "Tuple":
        elems = obj_type.get("elements")
        variadic = obj_type.get("variadic", False)
        if isinstance(elems, list):
            if variadic and len(elems) > 0:
                e = elems[0]
                if isinstance(e, dict):
                    return e
            if isinstance(slc, dict) and _is_type(slc, ["Constant"]):
                idx = slc.get("value")
                if isinstance(idx, int) and not isinstance(idx, bool):
                    if 0 <= idx < len(elems):
                        e = elems[idx]
                        if isinstance(e, dict):
                            return e
                    elif idx < 0 and -idx <= len(elems):
                        e = elems[len(elems) + idx]
                        if isinstance(e, dict):
                            return e
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
                            + str(len(elems)),
                        )
    return _ANY_TYPE


def _synth_binop(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    left = node.get("left")
    right = node.get("right")
    op = node.get("op", {})
    if not isinstance(left, dict) or not isinstance(right, dict):
        return _ANY_TYPE
    lt = _synth_expr(left, env, ctx)
    rt = _synth_expr(right, env, ctx)
    op_type = ""
    if isinstance(op, dict):
        op_type = str(op.get("_type", ""))
    # String concatenation
    if lt.get("kind") == "string" and rt.get("kind") == "string":
        return _STR_TYPE
    # List concatenation
    if lt.get("_type") == "Slice" and rt.get("_type") == "Slice":
        le = lt.get("element")
        re = rt.get("element")
        if isinstance(le, dict) and isinstance(re, dict):
            if not _is_assignable(le, re, ctx.hier_result) and not _is_assignable(
                re, le, ctx.hier_result
            ):
                b_lineno = node.get("lineno", 0)
                if not isinstance(b_lineno, int):
                    b_lineno = 0
                ctx.result.add_error(
                    b_lineno,
                    0,
                    "cannot concatenate list["
                    + _type_name(le)
                    + "] and list["
                    + _type_name(re)
                    + "]",
                )
        return lt
    # Numeric
    lt_num = lt.get("kind") in ("int", "float", "bool")
    rt_num = rt.get("kind") in ("int", "float", "bool")
    if lt_num and rt_num:
        # Bitwise ops on bools
        if op_type in ("BitAnd", "BitOr", "BitXor"):
            if lt.get("kind") == "bool" and rt.get("kind") == "bool":
                return _BOOL_TYPE
            return _INT_TYPE
        if lt.get("kind") == "float" or rt.get("kind") == "float":
            return _FLOAT_TYPE
        return _INT_TYPE
    # String * int
    if lt.get("kind") == "string" and (
        rt.get("kind") == "int" or rt.get("kind") == "bool"
    ):
        return _STR_TYPE
    if (lt.get("kind") == "int" or lt.get("kind") == "bool") and rt.get(
        "kind"
    ) == "string":
        return _STR_TYPE
    return _ANY_TYPE


def _synth_unaryop(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    operand = node.get("operand")
    op = node.get("op", {})
    if not isinstance(operand, dict):
        return _ANY_TYPE
    ot = _synth_expr(operand, env, ctx)
    if isinstance(op, dict):
        op_type = op.get("_type", "")
        if op_type == "Not":
            return _BOOL_TYPE
        if op_type == "USub" or op_type == "UAdd":
            if ot.get("kind") == "bool":
                return _INT_TYPE
            return ot
        if op_type == "Invert":
            return _INT_TYPE
    return ot


def _synth_boolop(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    values = node.get("values", [])
    if not isinstance(values, list) or len(values) == 0:
        return _ANY_TYPE
    # Return type of last value (simplified)
    last = values[len(values) - 1]
    if isinstance(last, dict):
        return _synth_expr(last, env, ctx)
    return _ANY_TYPE


def _synth_ifexp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    body = node.get("body")
    if isinstance(body, dict):
        return _synth_expr(body, env, ctx)
    return _ANY_TYPE


def _synth_list(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    elts = node.get("elts", [])
    if not isinstance(elts, list) or len(elts) == 0:
        return {"_type": "Slice", "element": _ANY_TYPE}
    first = elts[0]
    if isinstance(first, dict):
        return {"_type": "Slice", "element": _synth_expr(first, env, ctx)}
    return {"_type": "Slice", "element": _ANY_TYPE}


def _synth_dict(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    keys = node.get("keys", [])
    values = node.get("values", [])
    if not isinstance(keys, list) or not isinstance(values, list):
        return {"_type": "Map", "key": _ANY_TYPE, "value": _ANY_TYPE}
    if len(keys) == 0:
        return {"_type": "Map", "key": _ANY_TYPE, "value": _ANY_TYPE}
    k = keys[0]
    v = values[0]
    kt = _ANY_TYPE
    vt = _ANY_TYPE
    if isinstance(k, dict):
        kt = _synth_expr(k, env, ctx)
    if isinstance(v, dict):
        vt = _synth_expr(v, env, ctx)
    return {"_type": "Map", "key": kt, "value": vt}


def _synth_set(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    elts = node.get("elts", [])
    if not isinstance(elts, list) or len(elts) == 0:
        return {"_type": "Set", "element": _ANY_TYPE}
    first = elts[0]
    if isinstance(first, dict):
        return {"_type": "Set", "element": _synth_expr(first, env, ctx)}
    return {"_type": "Set", "element": _ANY_TYPE}


def _synth_tuple(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    elts = node.get("elts", [])
    if not isinstance(elts, list):
        return {"_type": "Tuple", "elements": [], "variadic": False}
    elems: list[object] = []
    i = 0
    while i < len(elts):
        e = elts[i]
        if isinstance(e, dict):
            elems.append(_synth_expr(e, env, ctx))
        else:
            elems.append(_ANY_TYPE)
        i += 1
    return {"_type": "Tuple", "elements": elems, "variadic": False}


def _synth_listcomp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    elt = node.get("elt")
    generators = node.get("generators", [])
    comp_env = env.copy()
    if isinstance(generators, list):
        _bind_comprehension_vars(generators, comp_env, ctx)
    if isinstance(elt, dict):
        return {"_type": "Slice", "element": _synth_expr(elt, comp_env, ctx)}
    return {"_type": "Slice", "element": _ANY_TYPE}


def _synth_setcomp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    elt = node.get("elt")
    generators = node.get("generators", [])
    comp_env = env.copy()
    if isinstance(generators, list):
        _bind_comprehension_vars(generators, comp_env, ctx)
    if isinstance(elt, dict):
        return {"_type": "Set", "element": _synth_expr(elt, comp_env, ctx)}
    return {"_type": "Set", "element": _ANY_TYPE}


def _synth_dictcomp(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    key = node.get("key")
    value = node.get("value")
    generators = node.get("generators", [])
    comp_env = env.copy()
    if isinstance(generators, list):
        _bind_comprehension_vars(generators, comp_env, ctx)
    kt = _ANY_TYPE
    vt = _ANY_TYPE
    if isinstance(key, dict):
        kt = _synth_expr(key, comp_env, ctx)
    if isinstance(value, dict):
        vt = _synth_expr(value, comp_env, ctx)
    return {"_type": "Map", "key": kt, "value": vt}


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
            # Process ifs for narrowing (e.g. [x for x in xs if x is not None])
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


def _synth_namedexpr(node: ASTNode, env: TypeEnv, ctx: _InferCtx) -> dict[str, object]:
    target = node.get("target")
    value = node.get("value")
    if not isinstance(value, dict):
        return _ANY_TYPE
    vt = _synth_expr(value, env, ctx)
    if isinstance(target, dict) and _is_type(target, ["Name"]):
        name = target.get("id")
        if isinstance(name, str):
            env.set(name, vt, _type_name(vt))
    return vt


# ---------------------------------------------------------------------------
# Iteration element type
# ---------------------------------------------------------------------------


def _iteration_element(t: dict[str, object]) -> dict[str, object]:
    """Get the element type when iterating over a type."""
    if t.get("_type") == "_Iterator":
        elem = t.get("element")
        if isinstance(elem, dict):
            return elem
    if t.get("_type") == "Slice":
        elem = t.get("element")
        if isinstance(elem, dict):
            return elem
    if t.get("_type") == "Set":
        elem = t.get("element")
        if isinstance(elem, dict):
            return elem
    if t.get("_type") == "Map":
        key = t.get("key")
        if isinstance(key, dict):
            return key
    if t.get("_type") == "Tuple":
        variadic = t.get("variadic", False)
        elems = t.get("elements")
        if variadic and isinstance(elems, list) and len(elems) > 0:
            e = elems[0]
            if isinstance(e, dict):
                return e
    if t.get("kind") == "string":
        return _STR_TYPE
    return _ANY_TYPE


def _bind_target(target: object, typ: dict[str, object], env: TypeEnv) -> None:
    """Bind an assignment target (Name or Tuple) to a type."""
    if not isinstance(target, dict):
        return
    if _is_type(target, ["Name"]):
        name = target.get("id")
        if isinstance(name, str):
            env.set(name, typ, _type_name(typ))
    elif _is_type(target, ["Tuple", "List"]):
        elts = target.get("elts", [])
        if isinstance(elts, list) and typ.get("_type") == "Tuple":
            telems = typ.get("elements")
            if isinstance(telems, list):
                j = 0
                while j < len(elts) and j < len(telems):
                    e = telems[j]
                    if isinstance(e, dict):
                        _bind_target(elts[j], e, env)
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


# ---------------------------------------------------------------------------
# Statement validation
# ---------------------------------------------------------------------------


def _validate_func(func_node: ASTNode, ctx: _InferCtx, receiver: str) -> None:
    """Validate a single function/method body."""
    func_name = func_node.get("name", "")
    if not isinstance(func_name, str):
        func_name = ""
    # Look up signature
    func_info: FuncInfo | None = None
    if receiver != "":
        methods = ctx.sig_result.methods.get(receiver)
        if methods is not None:
            func_info = methods.get(func_name)
    else:
        func_info = ctx.sig_result.functions.get(func_name)
    if func_info is None:
        return
    # Build initial type environment from parameters
    env = TypeEnv()
    i = 0
    while i < len(func_info.params):
        p = func_info.params[i]
        env.set(p.name, p.typ, p.py_type)
        i += 1
    # If method, add self
    if receiver != "":
        self_type: dict[str, object] = {
            "_type": "Pointer",
            "target": {"_type": "StructRef", "name": receiver},
        }
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
        return False
    if t == "Raise":
        return True
    if t == "Try":
        _validate_try(stmt, env, func_info, ctx)
        return False
    if t == "Match":
        _validate_match(stmt, env, func_info, ctx)
        return False
    if t == "FunctionDef":
        # Nested function: error
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
    # Check iterator escape
    if _check_iterator_escape_return(value, env, ctx, lineno):
        return
    # Check generator escape
    if _check_generator_escape_return(value, env, ctx, lineno):
        return
    # Check for unguarded access on optional/union/object
    _validate_expr_access(value, env, ctx, lineno)
    if len(ctx.result._errors) > 0:
        return
    # Validate call arguments in return expression
    if _is_type(value, ["Call"]):
        _validate_call_args(value, env, ctx, lineno)
        if len(ctx.result._errors) > 0:
            return
    # Validate literal types
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
    # Check iterator escape on assignment
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
                        # Re-assignment: check assignability
                        if not _is_assignable(val_type, existing, ctx.hier_result):
                            ctx.result.add_error(
                                lineno,
                                0,
                                "cannot assign "
                                + _type_name(val_type)
                                + " to "
                                + _type_name(existing),
                            )
                            return
                    else:
                        # First assignment: infer type
                        # Empty collection without annotation is error
                        if _is_empty_collection(value) and _is_any(
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
                # Tuple unpacking
                _validate_unpack(tgt, val_type, value, env, ctx, lineno)
            elif _is_type(tgt, ["Subscript"]):
                # d[key] = value
                _validate_subscript_assign(tgt, val_type, env, ctx, lineno)
            elif _is_type(tgt, ["Attribute"]):
                # obj.attr = value (self.field = value)
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
    val_type: dict[str, object],
    value: ASTNode,
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Validate tuple unpacking."""
    elts = target.get("elts", [])
    if not isinstance(elts, list):
        return
    # Check for optional tuple
    if _is_optional(val_type):
        ctx.result.add_error(lineno, 0, "cannot unpack optional tuple without guard")
        return
    if val_type.get("_type") != "Tuple":
        # Could be unpacking from non-tuple (e.g. some other iterable)
        # Allow and bind with any
        j = 0
        while j < len(elts):
            _bind_target(elts[j], _ANY_TYPE, env)
            j += 1
        return
    telems = val_type.get("elements")
    variadic = val_type.get("variadic", False)
    if not isinstance(telems, list):
        return
    if variadic:
        # Variadic tuple: bind all targets with element type
        elem = _ANY_TYPE
        if len(telems) > 0:
            e = telems[0]
            if isinstance(e, dict):
                elem = e
        j = 0
        while j < len(elts):
            _bind_target(elts[j], elem, env)
            j += 1
        return
    if len(elts) != len(telems):
        ctx.result.add_error(
            lineno,
            0,
            "cannot unpack tuple of "
            + str(len(telems))
            + " elements into "
            + str(len(elts))
            + " targets",
        )
        return
    j = 0
    while j < len(elts):
        e = telems[j]
        if isinstance(e, dict):
            _bind_target(elts[j], e, env)
        j += 1


def _validate_subscript_assign(
    target: ASTNode,
    val_type: dict[str, object],
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
    if obj_type.get("_type") == "Map":
        key_t = obj_type.get("key")
        val_t = obj_type.get("value")
        if isinstance(key_t, dict) and isinstance(slc, dict):
            key_actual = _synth_expr(slc, env, ctx)
            if not _is_assignable(key_actual, key_t, ctx.hier_result):
                ctx.result.add_error(
                    lineno,
                    0,
                    "cannot assign "
                    + _type_name(key_actual)
                    + " key to "
                    + _type_name(key_t),
                )
                return
        if isinstance(val_t, dict):
            if not _is_assignable(val_type, val_t, ctx.hier_result):
                ctx.result.add_error(
                    lineno,
                    0,
                    "cannot assign "
                    + _type_name(val_type)
                    + " value to "
                    + _type_name(val_t),
                )
    elif obj_type.get("_type") == "Slice":
        elem = obj_type.get("element")
        if isinstance(elem, dict):
            if not _is_assignable(val_type, elem, ctx.hier_result):
                ctx.result.add_error(
                    lineno,
                    0,
                    "cannot assign "
                    + _type_name(val_type)
                    + " to list element "
                    + _type_name(elem),
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
    # For augmented assignment, just check that the value type is compatible
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
    # Check iterator escape in expression statement
    if _is_type(value, ["Call"]):
        func = value.get("func")
        if isinstance(func, dict) and _is_type(func, ["Name"]):
            fname = func.get("id")
            if (
                isinstance(fname, str)
                and fname not in _EAGER_CONSUMERS
                and fname not in _ITERATOR_FUNCS
            ):
                # Check args for iterator escape
                args = value.get("args", [])
                if isinstance(args, list):
                    j = 0
                    while j < len(args):
                        arg = args[j]
                        if isinstance(arg, dict):
                            _check_iterator_escape_arg(arg, fname, env, ctx, lineno)
                            _check_generator_escape_arg(arg, fname, env, ctx, lineno)
                        j += 1
        # Method call: check for generator in join
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
                                pass  # join is allowed for generators
                            else:
                                _check_generator_escape_arg(arg, attr, env, ctx, lineno)
                        j += 1
    # Validate the expression for type errors
    _synth_expr(value, env, ctx)
    # Check method call argument types
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
    # Direct function call
    if _is_type(func, ["Name"]):
        fname = func.get("id")
        if not isinstance(fname, str):
            return
        func_info = ctx.sig_result.functions.get(fname)
        if func_info is not None:
            _check_call_args(func_info, args, env, ctx, lineno)
            return
        # Constructor call
        if fname in ctx.known_classes:
            # Constructors checked by field types
            return
        # Callable variable
        ftype = env.get_type(fname)
        if ftype is not None and ftype.get("_type") == "FuncType":
            _check_func_type_args(ftype, args, env, ctx, lineno)
            return
        # Builtin len check
        if fname == "len":
            if len(args) > 0:
                a = args[0]
                if isinstance(a, dict):
                    at = _synth_expr(a, env, ctx)
                    if (
                        at.get("kind") == "int"
                        or at.get("kind") == "float"
                        or at.get("kind") == "bool"
                    ):
                        ctx.result.add_error(
                            lineno,
                            0,
                            "len() requires a sized type, got " + _type_name(at),
                        )
    # Method call (obj.method)
    if _is_type(func, ["Attribute"]):
        obj = func.get("value")
        attr = func.get("attr")
        if not isinstance(obj, dict) or not isinstance(attr, str):
            return
        obj_type = _synth_expr(obj, env, ctx)
        # Unbound method: ClassName.method()
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
        # Check method argument types
        sname = _struct_name(obj_type)
        if sname != "":
            methods = ctx.sig_result.methods.get(sname)
            method: FuncInfo | None = None
            if methods is not None:
                method = methods.get(attr)
            if method is not None:
                _check_call_args(method, args, env, ctx, lineno)
                return
            # Method not found on this struct — check if subclass has it
            if _subclass_has_method(sname, attr, ctx):
                ctx.result.add_error(
                    lineno, 0, "method '" + attr + "' not accessible on " + sname
                )
                return
        # Unwrap Pointer for collection method checking
        check_type = obj_type
        if obj_type.get("_type") == "Pointer":
            ptarget = obj_type.get("target")
            if isinstance(ptarget, dict) and ptarget.get("_type") != "StructRef":
                check_type = ptarget
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
    # Count required params
    n_required = 0
    j = 0
    while j < len(params):
        if not params[j].has_default:
            n_required += 1
        j += 1
    if len(args) < n_required or len(args) > len(params):
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
    ftype: dict[str, object],
    args: list[object],
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Check args against a FuncType."""
    params = ftype.get("params")
    if not isinstance(params, list):
        return
    if len(args) != len(params):
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
            expected = params[j]
            if isinstance(expected, dict):
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
    obj_type: dict[str, object],
    method: str,
    args: list[object],
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Validate collection method argument types."""
    if obj_type.get("_type") == "Slice":
        elem = obj_type.get("element")
        if not isinstance(elem, dict):
            return
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
    elif obj_type.get("_type") == "Set":
        elem = obj_type.get("element")
        if not isinstance(elem, dict):
            return
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
        body = []
    if not isinstance(orelse, list):
        orelse = []
    lineno = stmt.get("lineno", 0)
    if not isinstance(lineno, int):
        lineno = 0
    # Check truthiness
    if isinstance(test, dict):
        _check_truthiness(test, env, ctx, lineno)
    # Extract narrowing
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
    # If then branch returns, else narrowing flows to continuation
    if then_returns and not else_returns:
        # Propagate else env narrowings to outer env
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
    return then_returns and else_returns


def _validate_while(
    stmt: ASTNode, env: TypeEnv, func_info: FuncInfo, ctx: _InferCtx
) -> None:
    test = stmt.get("test")
    body = stmt.get("body", [])
    if not isinstance(body, list):
        body = []
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
        body = []
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
    # Assert narrows in place (not branching)
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
    # Comparisons are always bool, no truthiness check needed
    if t == "Compare":
        return
    # isinstance is always bool
    if t == "Call":
        func = test.get("func")
        if (
            isinstance(func, dict)
            and _is_type(func, ["Name"])
            and func.get("id") == "isinstance"
        ):
            return
    # BoolOp: check each value
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
    # UnaryOp Not: check operand
    if t == "UnaryOp":
        op = test.get("op", {})
        if isinstance(op, dict) and op.get("_type") == "Not":
            operand = test.get("operand")
            if isinstance(operand, dict):
                _check_truthiness(operand, env, ctx, lineno)
            return
    # NamedExpr: check the value's type
    if t == "NamedExpr":
        value = test.get("value")
        if isinstance(value, dict):
            vt = _synth_expr(value, env, ctx)
            _check_type_truthiness(vt, env, test, ctx, lineno)
        return
    expr_type = _synth_expr(test, env, ctx)
    _check_type_truthiness(expr_type, env, test, ctx, lineno)


def _check_type_truthiness(
    typ: dict[str, object],
    env: TypeEnv,
    node: ASTNode,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Check if a type has unambiguous truthiness."""
    # Bool is always fine
    if typ.get("kind") == "bool":
        return
    # int/float: rejected
    if typ.get("kind") == "int":
        ctx.result.add_error(
            lineno, 0, "truthiness of int not allowed (zero is valid data)"
        )
        return
    if typ.get("kind") == "float":
        ctx.result.add_error(
            lineno, 0, "truthiness of float not allowed (zero is valid data)"
        )
        return
    # Optional: check if inner is ambiguous
    if _is_optional(typ):
        inner = _unwrap_optional(typ)
        inner_kind = inner.get("kind")
        # Optional[int] is OK (means "is not None")
        if inner_kind == "int" or inner_kind == "float" or inner_kind == "bool":
            return
        # Optional[str|list|dict|set] is ambiguous
        if inner_kind == "string":
            ctx.result.add_error(lineno, 0, "ambiguous truthiness for optional str")
            return
        if inner.get("_type") == "Slice":
            ctx.result.add_error(lineno, 0, "ambiguous truthiness for optional list")
            return
        if inner.get("_type") == "Map":
            ctx.result.add_error(lineno, 0, "ambiguous truthiness for optional dict")
            return
        if inner.get("_type") == "Set":
            ctx.result.add_error(lineno, 0, "ambiguous truthiness for optional set")
            return
        return
    # Source type check for optionals that became InterfaceRef
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
    # list/str/dict/set: allowed
    if typ.get("kind") == "string":
        return
    if typ.get("_type") == "Slice":
        return
    if typ.get("_type") == "Map":
        return
    if typ.get("_type") == "Set":
        return
    # any/interface: allow (for walrus, etc.)
    if _is_any(typ):
        return
    if typ.get("_type") == "InterfaceRef":
        return
    # Struct types: generally not allowed in truthiness
    # But we allow it if it could be optional


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
    # isinstance(x, T)
    if t == "Call":
        func = test.get("func")
        if (
            isinstance(func, dict)
            and _is_type(func, ["Name"])
            and func.get("id") == "isinstance"
        ):
            _narrow_isinstance(test, then_env, else_env, ctx)
            return
    # x is None / x is not None
    if t == "Compare":
        _narrow_compare(test, then_env, else_env, ctx)
        return
    # not expr: swap then/else
    if t == "UnaryOp":
        op = test.get("op", {})
        if isinstance(op, dict) and op.get("_type") == "Not":
            operand = test.get("operand")
            if isinstance(operand, dict):
                _extract_narrowing(operand, else_env, then_env, ctx)
            return
    # BoolOp and/or
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
                # isinstance(x,A) or isinstance(x,B): extract union for then-branch
                values = test.get("values", [])
                if isinstance(values, list):
                    _narrow_or_isinstance(values, then_env, ctx)
                return
    # NamedExpr (walrus): if (val := func()): narrows val
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
                # If return type is optional, narrow to non-None in then branch
                if _is_optional(vt):
                    inner = _unwrap_optional(vt)
                    then_env.set(name, inner, _type_name(inner))
        return
    # Name by itself: truthiness narrowing for optionals
    if t == "Name":
        name = test.get("id")
        if isinstance(name, str):
            typ = then_env.get_type(name)
            source = then_env.get_source(name)
            if typ is not None and _is_optional(typ):
                inner = _unwrap_optional(typ)
                then_env.narrow(name, inner, _type_name(inner))
            elif source != "" and _is_optional_source(source):
                # Source-tracked optional
                non_none = _non_none_parts(source)
                if len(non_none) == 1:
                    sig_errors: list[SignatureError] = []
                    narrowed = py_type_to_type_dict(
                        non_none[0], ctx.known_classes, sig_errors, 0, 0
                    )
                    then_env.narrow(name, narrowed, non_none[0])
        return


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
    # Get the variable name
    name = ""
    if _is_type(target, ["Name"]):
        name = str(target.get("id", ""))
    if name == "":
        return
    # Get the narrowed type name
    narrow_name = ""
    if _is_type(type_arg, ["Name"]):
        narrow_name = str(type_arg.get("id", ""))
    if narrow_name == "":
        return
    # Build narrowed type
    sig_errors: list[SignatureError] = []
    narrowed = py_type_to_type_dict(narrow_name, ctx.known_classes, sig_errors, 0, 0)
    then_env.narrow(name, narrowed, narrow_name)
    # Else branch: narrow to remaining union members
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
            sig_errors2 = []
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
    # x is None
    if op_type == "Is" and _is_type(comp, ["Constant"]) and comp.get("value") is None:
        if _is_type(left, ["Name"]):
            name = str(left.get("id", ""))
            if name != "":
                # then: x is None (unchanged)
                # else: x is not None (narrow to non-None)
                _narrow_to_non_none(name, else_env, ctx)
        return
    # x is not None
    if (
        op_type == "IsNot"
        and _is_type(comp, ["Constant"])
        and comp.get("value") is None
    ):
        if _is_type(left, ["Name"]):
            name = str(left.get("id", ""))
            if name != "":
                _narrow_to_non_none(name, then_env, ctx)
        # x.attr is not None
        if _is_type(left, ["Attribute"]):
            value_node = left.get("value")
            attr = left.get("attr")
            if (
                isinstance(value_node, dict)
                and _is_type(value_node, ["Name"])
                and isinstance(attr, str)
            ):
                obj_name = str(value_node.get("id", ""))
                if obj_name != "":
                    then_env.guard_attr(obj_name + "." + attr)
        return
    # x.attr is None
    if op_type == "Is" and _is_type(comp, ["Constant"]) and comp.get("value") is None:
        if _is_type(left, ["Attribute"]):
            value_node = left.get("value")
            attr = left.get("attr")
            if (
                isinstance(value_node, dict)
                and _is_type(value_node, ["Name"])
                and isinstance(attr, str)
            ):
                obj_name = str(value_node.get("id", ""))
                if obj_name != "":
                    else_env.guard_attr(obj_name + "." + attr)
        return
    # x.kind == "value" (kind discrimination)
    if op_type == "Eq":
        if _is_type(left, ["Attribute"]):
            attr = left.get("attr")
            if isinstance(attr, str) and attr == "kind":
                # Validate kind value against known types
                comp_value = comp.get("value")
                if isinstance(comp_value, str):
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
    if typ is not None and _is_optional(typ):
        inner = _unwrap_optional(typ)
        env.narrow(name, inner, _type_name(inner))
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
            sig_errors = []
            narrowed = py_type_to_type_dict(
                new_source, ctx.known_classes, sig_errors, 0, 0
            )
            env.narrow(name, narrowed, new_source)


def _narrow_or_isinstance(
    values: list[object], then_env: TypeEnv, ctx: _InferCtx
) -> None:
    """Handle isinstance(x,A) or isinstance(x,B) in then branch."""
    # Just allow isinstance narrowing for individual values
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
    # Check if value is an iterator call
    fname = _is_iterator_call(value)
    if fname != "":
        ctx.result.add_error(lineno, 0, "cannot return " + fname + "()")
        return True
    # Check if value is wrapped in an eager consumer
    if _is_type(value, ["Call"]):
        func = value.get("func")
        if isinstance(func, dict) and _is_type(func, ["Name"]):
            wrapper = func.get("id")
            if isinstance(wrapper, str) and wrapper in _EAGER_CONSUMERS:
                return False  # Wrapped in consumer, OK
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
    # Check if it's a call wrapping a generator
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
                return False  # Eager consumer wrapping generator is OK
            # str.join wrapping
        if isinstance(func, dict) and _is_type(func, ["Attribute"]):
            attr = func.get("attr")
            if isinstance(attr, str) and attr == "join":
                return False  # join wrapping generator is OK
        # Check args for generator escape
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
    expected: dict[str, object] | None = None,
) -> None:
    """Check list literal for mixed types."""
    # Union-typed list context allows mixed literals
    if expected is not None and expected.get("_type") == "Slice":
        elem_t = expected.get("element")
        if isinstance(elem_t, dict) and _is_any(elem_t):
            return
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
            if not _is_assignable(
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
    node: ASTNode, env: TypeEnv, ctx: _InferCtx, lineno: int
) -> None:
    """Check dict literal for mixed key/value types."""
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
    expected: dict[str, object],
    env: TypeEnv,
    ctx: _InferCtx,
    lineno: int,
) -> None:
    """Extra validation on return expressions: literal checking, etc."""
    if _is_type(value, ["List"]):
        _validate_list_literal(value, env, ctx, lineno, expected)
    if _is_type(value, ["Dict"]):
        _validate_dict_literal(value, env, ctx, lineno)


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
    # Optional (IR type level)
    if _is_optional(typ):
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
    # Source-tracked optional
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
    # Source-tracked non-optional union
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
    # Explicit object type
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
    # BinOp: check operands
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
    # Attribute: check object (kind access is always allowed)
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
    # Subscript: check value
    if t == "Subscript":
        value = node.get("value")
        if isinstance(value, dict):
            _check_needs_narrowing(value, env, ctx, lineno, "subscript", "")
            _validate_expr_access(value, env, ctx, lineno)
        return
    # Call: recurse into func and args
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
        if not isinstance(node, dict):
            i += 1
            continue
        t = node.get("_type")
        if t == "FunctionDef":
            _validate_func(node, ctx, "")
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
                        _validate_func(stmt, ctx, class_name)
                        if len(result._errors) > 0:
                            return result
                    j += 1
        i += 1
    return result
