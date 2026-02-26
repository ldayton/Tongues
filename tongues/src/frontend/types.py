"""Type node hierarchy for the Tongues frontend.

Replaces dict[str, object] type dicts with a typed class hierarchy that taytsh
can represent as sealed interfaces with struct variants.

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# TYPE NODES
# ============================================================


@dataclass
class TypeNode:
    """Base for all type nodes."""


@dataclass
class PrimitiveType(TypeNode):
    """int, float, bool, byte, string, void, any, bytes."""

    kind: str


@dataclass
class SliceType(TypeNode):
    """list[T]."""

    element: TypeNode


@dataclass
class MapType(TypeNode):
    """dict[K, V]."""

    key: TypeNode
    value: TypeNode


@dataclass
class SetType(TypeNode):
    """set[T]."""

    element: TypeNode


@dataclass
class TupleType(TypeNode):
    """tuple[T, U, ...] or tuple[T, ...]."""

    elements: list[TypeNode]
    variadic: bool


@dataclass
class OptionalType(TypeNode):
    """T | None."""

    inner: TypeNode


@dataclass
class PointerType(TypeNode):
    """Pointer wrapper for mutated params."""

    target: TypeNode


@dataclass
class StructRef(TypeNode):
    """Reference to a struct type by name."""

    name: str


@dataclass
class InterfaceRef(TypeNode):
    """Reference to an interface type by name."""

    name: str


@dataclass
class FuncType(TypeNode):
    """Callable[[P1, P2], R]."""

    params: list[TypeNode]
    ret: TypeNode


@dataclass
class UnionType(TypeNode):
    """A | B (non-optional union)."""

    variants: list[TypeNode]


@dataclass
class IteratorType(TypeNode):
    """An iterator yielding elements of a given type."""

    element: TypeNode


# ============================================================
# LITERAL TYPE NODES (for default values)
# ============================================================


@dataclass
class NilLit(TypeNode):
    """None literal."""


@dataclass
class BoolLit(TypeNode):
    """Boolean literal default."""

    value: bool


@dataclass
class IntLit(TypeNode):
    """Integer literal default."""

    value: int


@dataclass
class FloatLit(TypeNode):
    """Float literal default."""

    value: float


@dataclass
class StringLit(TypeNode):
    """String literal default."""

    value: str


@dataclass
class ListLit(TypeNode):
    """Empty list literal default."""

    elements: list[TypeNode]


@dataclass
class MapLit(TypeNode):
    """Empty map literal default."""

    entries: list[TypeNode]


@dataclass
class SetLit(TypeNode):
    """Empty set literal default."""

    elements: list[TypeNode]


@dataclass
class TupleLit(TypeNode):
    """Empty tuple literal default."""

    elements: list[TypeNode]


# ============================================================
# JSON VALUE ADT (replaces `object` in AST dicts)
# ============================================================


@dataclass
class JsonValue:
    """Base for JSON-compatible values in AST dicts and serialization."""


@dataclass
class JStr(JsonValue):
    value: str


@dataclass
class JInt(JsonValue):
    value: int


@dataclass
class JFloat(JsonValue):
    value: float


@dataclass
class JBool(JsonValue):
    value: bool


@dataclass
class JNull(JsonValue):
    pass


@dataclass
class JList(JsonValue):
    items: list[JsonValue]


@dataclass
class JDict(JsonValue):
    entries: dict[str, JsonValue]


# Type alias for AST dict nodes
ASTNode = dict[str, JsonValue]


# ============================================================
# AST ACCESSOR HELPERS
# ============================================================


def get_str(node: dict[str, JsonValue], key: str) -> str:
    v = node.get(key)
    if isinstance(v, JStr):
        return v.value
    return ""


def get_int(node: dict[str, JsonValue], key: str) -> int:
    v = node.get(key)
    if isinstance(v, JInt):
        return v.value
    return 0


def get_float(node: dict[str, JsonValue], key: str) -> float:
    v = node.get(key)
    if isinstance(v, JFloat):
        return v.value
    return 0.0


def get_bool(node: dict[str, JsonValue], key: str) -> bool:
    v = node.get(key)
    if isinstance(v, JBool):
        return v.value
    return False


def get_node(node: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    """Get a nested AST node (unwraps JDict)."""
    v = node.get(key)
    if isinstance(v, JDict):
        return v.entries
    return {}


def get_nodes(node: dict[str, JsonValue], key: str) -> list[dict[str, JsonValue]]:
    """Get a list of nested AST nodes (unwraps JList of JDicts)."""
    v = node.get(key)
    if isinstance(v, JList):
        result: list[dict[str, JsonValue]] = []
        i = 0
        while i < len(v.items):
            item = v.items[i]
            if isinstance(item, JDict):
                result.append(item.entries)
            i += 1
        return result
    return []


def get_jlist(node: dict[str, JsonValue], key: str) -> list[JsonValue]:
    """Get raw JsonValue list (for mixed-type lists)."""
    v = node.get(key)
    if isinstance(v, JList):
        return v.items
    return []


def has_key(node: dict[str, JsonValue], key: str) -> bool:
    return node.get(key) is not None


# ============================================================
# HELPERS
# ============================================================


def is_any(t: TypeNode) -> bool:
    """Check if a type is the 'any' type."""
    if isinstance(t, InterfaceRef) and t.name == "any":
        return True
    if isinstance(t, PrimitiveType) and t.kind == "any":
        return True
    return False


def contains_any(t: TypeNode) -> bool:
    """Check if a type contains 'any' anywhere in its structure."""
    if is_any(t):
        return True
    if isinstance(t, SliceType):
        return contains_any(t.element)
    if isinstance(t, MapType):
        return contains_any(t.key) or contains_any(t.value)
    if isinstance(t, SetType):
        return contains_any(t.element)
    if isinstance(t, TupleType):
        i = 0
        while i < len(t.elements):
            if contains_any(t.elements[i]):
                return True
            i += 1
        return False
    if isinstance(t, OptionalType):
        return contains_any(t.inner)
    if isinstance(t, PointerType):
        return contains_any(t.target)
    if isinstance(t, FuncType):
        i = 0
        while i < len(t.params):
            if contains_any(t.params[i]):
                return True
            i += 1
        return contains_any(t.ret)
    if isinstance(t, UnionType):
        i = 0
        while i < len(t.variants):
            if contains_any(t.variants[i]):
                return True
            i += 1
        return False
    return False


def is_void(t: TypeNode) -> bool:
    """Check if a type is void."""
    return isinstance(t, PrimitiveType) and t.kind == "void"


def type_name(t: TypeNode) -> str:
    """Human-readable type name for error messages."""
    if isinstance(t, PrimitiveType):
        if t.kind == "string":
            return "str"
        if t.kind == "void":
            return "None"
        return t.kind
    if isinstance(t, SliceType):
        return "list[" + type_name(t.element) + "]"
    if isinstance(t, MapType):
        return "dict[" + type_name(t.key) + ", " + type_name(t.value) + "]"
    if isinstance(t, SetType):
        return "set[" + type_name(t.element) + "]"
    if isinstance(t, TupleType):
        if len(t.elements) > 0:
            parts: list[str] = []
            i = 0
            while i < len(t.elements):
                parts.append(type_name(t.elements[i]))
                i += 1
            if t.variadic:
                return "tuple[" + ", ".join(parts) + ", ...]"
            return "tuple[" + ", ".join(parts) + "]"
        return "tuple"
    if isinstance(t, OptionalType):
        return type_name(t.inner) + " | None"
    if isinstance(t, PointerType):
        return type_name(t.target)
    if isinstance(t, StructRef):
        return t.name
    if isinstance(t, InterfaceRef):
        if t.name == "any":
            return "object"
        return t.name
    if isinstance(t, FuncType):
        return "Callable"
    if isinstance(t, UnionType):
        parts: list[str] = []
        i = 0
        while i < len(t.variants):
            parts.append(type_name(t.variants[i]))
            i += 1
        return " | ".join(parts)
    if isinstance(t, IteratorType):
        return "Iterator[" + type_name(t.element) + "]"
    return "unknown"


def typenode_to_dict(t: TypeNode) -> JsonValue:
    """Convert a TypeNode to a JsonValue dict for serialization."""
    if isinstance(t, PrimitiveType):
        return JDict({"kind": JStr(t.kind)})
    if isinstance(t, SliceType):
        return JDict({"_type": JStr("Slice"), "element": typenode_to_dict(t.element)})
    if isinstance(t, MapType):
        return JDict(
            {
                "_type": JStr("Map"),
                "key": typenode_to_dict(t.key),
                "value": typenode_to_dict(t.value),
            }
        )
    if isinstance(t, SetType):
        return JDict({"_type": JStr("Set"), "element": typenode_to_dict(t.element)})
    if isinstance(t, TupleType):
        elems: list[JsonValue] = []
        i = 0
        while i < len(t.elements):
            elems.append(typenode_to_dict(t.elements[i]))
            i += 1
        return JDict(
            {
                "_type": JStr("Tuple"),
                "elements": JList(elems),
                "variadic": JBool(t.variadic),
            }
        )
    if isinstance(t, OptionalType):
        return JDict({"_type": JStr("Optional"), "inner": typenode_to_dict(t.inner)})
    if isinstance(t, PointerType):
        return JDict({"_type": JStr("Pointer"), "target": typenode_to_dict(t.target)})
    if isinstance(t, StructRef):
        return JDict({"_type": JStr("StructRef"), "name": JStr(t.name)})
    if isinstance(t, InterfaceRef):
        return JDict({"_type": JStr("InterfaceRef"), "name": JStr(t.name)})
    if isinstance(t, FuncType):
        params: list[JsonValue] = []
        i = 0
        while i < len(t.params):
            params.append(typenode_to_dict(t.params[i]))
            i += 1
        return JDict(
            {
                "_type": JStr("FuncType"),
                "params": JList(params),
                "ret": typenode_to_dict(t.ret),
            }
        )
    if isinstance(t, UnionType):
        variants: list[JsonValue] = []
        i = 0
        while i < len(t.variants):
            variants.append(typenode_to_dict(t.variants[i]))
            i += 1
        return JDict({"_type": JStr("Union"), "members": JList(variants)})
    if isinstance(t, NilLit):
        return JDict({"_type": JStr("NilLit")})
    if isinstance(t, BoolLit):
        return JDict({"_type": JStr("BoolLit"), "value": JBool(t.value)})
    if isinstance(t, IntLit):
        return JDict({"_type": JStr("IntLit"), "value": JInt(t.value)})
    if isinstance(t, FloatLit):
        return JDict({"_type": JStr("FloatLit"), "value": JFloat(t.value)})
    if isinstance(t, StringLit):
        return JDict({"_type": JStr("StringLit"), "value": JStr(t.value)})
    if isinstance(t, ListLit):
        return JDict({"_type": JStr("ListLit"), "elements": JList([])})
    if isinstance(t, MapLit):
        return JDict({"_type": JStr("MapLit"), "entries": JList([])})
    if isinstance(t, SetLit):
        return JDict({"_type": JStr("SetLit"), "elements": JList([])})
    if isinstance(t, TupleLit):
        return JDict({"_type": JStr("TupleLit"), "elements": JList([])})
    return JDict({"_type": JStr("InterfaceRef"), "name": JStr("any")})


# Commonly used singleton types
ANY_TYPE: TypeNode = InterfaceRef("any")
INT_TYPE: TypeNode = PrimitiveType("int")
FLOAT_TYPE: TypeNode = PrimitiveType("float")
BOOL_TYPE: TypeNode = PrimitiveType("bool")
STR_TYPE: TypeNode = PrimitiveType("string")
VOID_TYPE: TypeNode = PrimitiveType("void")
BYTE_TYPE: TypeNode = PrimitiveType("byte")
BYTES_TYPE: TypeNode = SliceType(PrimitiveType("byte"))
