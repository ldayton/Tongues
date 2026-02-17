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
# HELPERS
# ============================================================


def is_any(t: TypeNode) -> bool:
    """Check if a type is the 'any' type."""
    if isinstance(t, InterfaceRef) and t.name == "any":
        return True
    if isinstance(t, PrimitiveType) and t.kind == "any":
        return True
    return False


def is_void(t: TypeNode) -> bool:
    """Check if a type is void."""
    return isinstance(t, PrimitiveType) and t.kind == "void"


def type_name(t: TypeNode) -> str:
    """Human-readable type name for error messages."""
    if isinstance(t, PrimitiveType):
        if t.kind == "string":
            return "str"
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


def typenode_to_dict(t: TypeNode) -> dict[str, object]:
    """Convert a TypeNode to the legacy dict format for test serialization."""
    if isinstance(t, PrimitiveType):
        return {"kind": t.kind}
    if isinstance(t, SliceType):
        return {"_type": "Slice", "element": typenode_to_dict(t.element)}
    if isinstance(t, MapType):
        return {
            "_type": "Map",
            "key": typenode_to_dict(t.key),
            "value": typenode_to_dict(t.value),
        }
    if isinstance(t, SetType):
        return {"_type": "Set", "element": typenode_to_dict(t.element)}
    if isinstance(t, TupleType):
        elems: list[object] = []
        i = 0
        while i < len(t.elements):
            elems.append(typenode_to_dict(t.elements[i]))
            i += 1
        return {"_type": "Tuple", "elements": elems, "variadic": t.variadic}
    if isinstance(t, OptionalType):
        return {"_type": "Optional", "inner": typenode_to_dict(t.inner)}
    if isinstance(t, PointerType):
        return {"_type": "Pointer", "target": typenode_to_dict(t.target)}
    if isinstance(t, StructRef):
        return {"_type": "StructRef", "name": t.name}
    if isinstance(t, InterfaceRef):
        return {"_type": "InterfaceRef", "name": t.name}
    if isinstance(t, FuncType):
        params: list[object] = []
        i = 0
        while i < len(t.params):
            params.append(typenode_to_dict(t.params[i]))
            i += 1
        return {"_type": "FuncType", "params": params, "ret": typenode_to_dict(t.ret)}
    if isinstance(t, UnionType):
        variants: list[object] = []
        i = 0
        while i < len(t.variants):
            variants.append(typenode_to_dict(t.variants[i]))
            i += 1
        return {"_type": "Union", "members": variants}
    if isinstance(t, NilLit):
        return {"_type": "NilLit"}
    if isinstance(t, BoolLit):
        return {"_type": "BoolLit", "value": t.value}
    if isinstance(t, IntLit):
        return {"_type": "IntLit", "value": t.value}
    if isinstance(t, FloatLit):
        return {"_type": "FloatLit", "value": t.value}
    if isinstance(t, StringLit):
        return {"_type": "StringLit", "value": t.value}
    if isinstance(t, ListLit):
        return {"_type": "ListLit", "elements": []}
    if isinstance(t, MapLit):
        return {"_type": "MapLit", "entries": []}
    if isinstance(t, SetLit):
        return {"_type": "SetLit", "elements": []}
    if isinstance(t, TupleLit):
        return {"_type": "TupleLit", "elements": []}
    return {"_type": "InterfaceRef", "name": "any"}


# Commonly used singleton types
ANY_TYPE: TypeNode = InterfaceRef("any")
INT_TYPE: TypeNode = PrimitiveType("int")
FLOAT_TYPE: TypeNode = PrimitiveType("float")
BOOL_TYPE: TypeNode = PrimitiveType("bool")
STR_TYPE: TypeNode = PrimitiveType("string")
VOID_TYPE: TypeNode = PrimitiveType("void")
BYTE_TYPE: TypeNode = PrimitiveType("byte")
BYTES_TYPE: TypeNode = SliceType(PrimitiveType("byte"))
