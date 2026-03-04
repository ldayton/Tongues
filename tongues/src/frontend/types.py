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


@dataclass
class LiteralType(TypeNode):
    """A literal type: Literal["foo"], Literal[42], Literal[true]."""

    lit_value: str
    base: PrimitiveType


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
type ASTNode = dict[str, JsonValue]


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
        for item in v.items:
            if isinstance(item, JDict):
                result.append(item.entries)
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


def map_subtypes(t: TypeNode, replacements: list[TypeNode]) -> TypeNode:
    """Rebuild a union/optional by replacing leaf types with the given list.

    Extracts the structural shape of t (Union, Optional+Union, Optional, plain)
    and reconstructs it from replacements. Callers iterate get_subtypes(t),
    transform each element, and pass the results here.
    """
    if isinstance(t, UnionType):
        return combine_types(replacements)
    if isinstance(t, OptionalType):
        if isinstance(t.inner, UnionType):
            return combine_types(replacements)
        if len(replacements) == 1:
            return OptionalType(replacements[0])
        return combine_types(replacements)
    if len(replacements) == 1:
        return replacements[0]
    return combine_types(replacements)


def get_subtypes(t: TypeNode) -> list[TypeNode]:
    """Extract leaf variant types from a union/optional for iteration."""
    if isinstance(t, UnionType):
        return t.variants
    if isinstance(t, OptionalType):
        if isinstance(t.inner, UnionType):
            return t.inner.variants
        result: list[TypeNode] = [t.inner]
        return result
    result2: list[TypeNode] = [t]
    return result2


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
        for elem in t.elements:
            if contains_any(elem):
                return True
        return False
    if isinstance(t, OptionalType):
        return contains_any(t.inner)
    if isinstance(t, PointerType):
        return contains_any(t.target)
    if isinstance(t, FuncType):
        for param in t.params:
            if contains_any(param):
                return True
        return contains_any(t.ret)
    if isinstance(t, UnionType):
        for variant in t.variants:
            if contains_any(variant):
                return True
        return False
    if isinstance(t, LiteralType):
        return False
    return False


def type_name(t: TypeNode) -> str:
    """Human-readable type name for error messages."""
    if isinstance(t, PrimitiveType):
        if t.kind == "string":
            return "str"
        if t.kind == "void":
            return "None"
        if t.kind == "never":
            return "never"
        return t.kind
    if isinstance(t, SliceType):
        return "list[" + type_name(t.element) + "]"
    if isinstance(t, MapType):
        return "dict[" + type_name(t.key) + ", " + type_name(t.value) + "]"
    if isinstance(t, SetType):
        return "set[" + type_name(t.element) + "]"
    if isinstance(t, TupleType):
        if t.elements:
            parts: list[str] = []
            for elem in t.elements:
                parts.append(type_name(elem))
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
        for variant in t.variants:
            parts.append(type_name(variant))
        return " | ".join(parts)
    if isinstance(t, IteratorType):
        return "Iterator[" + type_name(t.element) + "]"
    if isinstance(t, LiteralType):
        if t.base.kind == "string":
            return 'Literal["' + t.lit_value + '"]'
        return "Literal[" + t.lit_value + "]"
    return "unknown"


def is_bytes_type(t: TypeNode) -> bool:
    """Check if t represents bytes (either PrimitiveType("bytes") or SliceType(byte))."""
    if isinstance(t, PrimitiveType) and t.kind == "bytes":
        return True
    if isinstance(t, SliceType):
        if isinstance(t.element, PrimitiveType) and t.element.kind == "byte":
            return True
    return False


def type_eq(a: TypeNode, b: TypeNode) -> bool:
    """Check structural equality of two TypeNodes."""
    if is_bytes_type(a) and is_bytes_type(b):
        return True
    if isinstance(a, PrimitiveType) and isinstance(b, PrimitiveType):
        return a.kind == b.kind
    if isinstance(a, PrimitiveType) or isinstance(b, PrimitiveType):
        return False
    if isinstance(a, SliceType) and isinstance(b, SliceType):
        return type_eq(a.element, b.element)
    if isinstance(a, MapType) and isinstance(b, MapType):
        return type_eq(a.key, b.key) and type_eq(a.value, b.value)
    if isinstance(a, SetType) and isinstance(b, SetType):
        return type_eq(a.element, b.element)
    if isinstance(a, OptionalType) and isinstance(b, OptionalType):
        return type_eq(a.inner, b.inner)
    if isinstance(a, TupleType) and isinstance(b, TupleType):
        if a.variadic != b.variadic:
            return False
        if len(a.elements) != len(b.elements):
            return False
        for j, elem_a in enumerate(a.elements):
            if not type_eq(elem_a, b.elements[j]):
                return False
        return True
    if isinstance(a, PointerType) and isinstance(b, PointerType):
        return type_eq(a.target, b.target)
    if isinstance(a, StructRef) and isinstance(b, StructRef):
        return a.name == b.name
    if isinstance(a, InterfaceRef) and isinstance(b, InterfaceRef):
        return a.name == b.name
    if isinstance(a, FuncType) and isinstance(b, FuncType):
        if len(a.params) != len(b.params):
            return False
        for j, param_a in enumerate(a.params):
            if not type_eq(param_a, b.params[j]):
                return False
        return type_eq(a.ret, b.ret)
    if isinstance(a, UnionType) and isinstance(b, UnionType):
        if len(a.variants) != len(b.variants):
            return False
        for var_a in a.variants:
            found = False
            for var_b in b.variants:
                if type_eq(var_a, var_b):
                    found = True
            if not found:
                return False
        return True
    if isinstance(a, LiteralType) and isinstance(b, LiteralType):
        return a.lit_value == b.lit_value and a.base.kind == b.base.kind
    return a == b


def combine_types(types: list[TypeNode]) -> TypeNode:
    """Flatten, deduplicate, and normalize a list of types into a single type."""
    # Flatten nested unions
    flat: list[TypeNode] = []
    for t in types:
        if isinstance(t, UnionType):
            for variant in t.variants:
                flat.append(variant)
        else:
            flat.append(t)
    # Filter out never
    filtered: list[TypeNode] = []
    for f in flat:
        if isinstance(f, PrimitiveType) and f.kind == "never":
            pass
        else:
            filtered.append(f)
    flat = filtered
    # Deduplicate via type_eq
    deduped: list[TypeNode] = []
    for f in flat:
        is_dup = False
        for d in deduped:
            if type_eq(f, d):
                is_dup = True
        if not is_dup:
            deduped.append(f)
    # Literal absorption: base type absorbs its literals
    base_kinds: set[str] = set()
    for di in deduped:
        if isinstance(di, PrimitiveType):
            base_kinds.add(di.kind)
    absorbed: list[TypeNode] = []
    has_lit_true = False
    has_lit_false = False
    for d in deduped:
        if isinstance(d, LiteralType):
            if d.base.kind not in base_kinds:
                if d.base.kind == "bool" and d.lit_value == "true":
                    has_lit_true = True
                if d.base.kind == "bool" and d.lit_value == "false":
                    has_lit_false = True
                absorbed.append(d)
        else:
            absorbed.append(d)
    if has_lit_true and has_lit_false:
        merged: list[TypeNode] = []
        for d2 in absorbed:
            if isinstance(d2, LiteralType) and d2.base.kind == "bool":
                pass
            else:
                merged.append(d2)
        merged.append(PrimitiveType("bool"))
        absorbed = merged
    deduped = absorbed
    if not deduped:
        return PrimitiveType("void")
    if len(deduped) == 1:
        return deduped[0]
    # Check for None/void among variants
    has_none = False
    others: list[TypeNode] = []
    for d in deduped:
        if isinstance(d, PrimitiveType) and d.kind == "void":
            has_none = True
        else:
            others.append(d)
    if has_none:
        if not others:
            return PrimitiveType("void")
        if len(others) == 1:
            return OptionalType(others[0])
        return OptionalType(UnionType(others))
    return UnionType(deduped)


def _removal_matches(variant: TypeNode, target: TypeNode) -> bool:
    """Check if variant matches target for union removal.

    Uses type_eq, but also matches generic containers with any-element against
    concrete containers (e.g. SliceType(any) matches SliceType(str)).
    """
    if type_eq(variant, target):
        return True
    if isinstance(variant, SliceType) and isinstance(target, SliceType):
        if is_any(target.element):
            return True
    if isinstance(variant, MapType) and isinstance(target, MapType):
        if is_any(target.key) and is_any(target.value):
            return True
    if isinstance(variant, SetType) and isinstance(target, SetType):
        if is_any(target.element):
            return True
    if isinstance(variant, TupleType) and isinstance(target, TupleType):
        if target.elements and is_any(target.elements[0]):
            return True
    return False


def remove_from_union(t: TypeNode, to_remove: list[TypeNode]) -> TypeNode:
    """Remove variants matching any type in to_remove."""
    if isinstance(t, UnionType):
        remaining: list[TypeNode] = []
        for variant in t.variants:
            matched = False
            for rm in to_remove:
                if _removal_matches(variant, rm):
                    matched = True
            if not matched:
                remaining.append(variant)
        if not remaining:
            return PrimitiveType("never")
        if len(remaining) == 1:
            return remaining[0]
        return UnionType(remaining)
    if isinstance(t, OptionalType) and isinstance(t.inner, UnionType):
        remaining2: list[TypeNode] = []
        for variant in t.inner.variants:
            matched = False
            for rm in to_remove:
                if _removal_matches(variant, rm):
                    matched = True
            if not matched:
                remaining2.append(variant)
        if not remaining2:
            return PrimitiveType("void")
        if len(remaining2) == 1:
            return OptionalType(remaining2[0])
        return OptionalType(UnionType(remaining2))
    if isinstance(t, OptionalType):
        for rm in to_remove:
            if _removal_matches(t.inner, rm):
                return PrimitiveType("void")
    for rm in to_remove:
        if _removal_matches(t, rm):
            return PrimitiveType("never")
    return t


def _variant_name(v: TypeNode) -> str:
    """Extract struct/interface name from a single variant."""
    if isinstance(v, StructRef):
        return v.name
    if isinstance(v, InterfaceRef):
        return v.name
    if isinstance(v, PointerType) and isinstance(v.target, StructRef):
        return v.target.name
    return ""


def union_variant_names(t: TypeNode) -> list[str]:
    """Extract struct/interface names from union variants."""
    result: list[str] = []
    inner = t
    if isinstance(t, OptionalType):
        inner = t.inner
    if isinstance(inner, UnionType):
        for variant in inner.variants:
            vn = _variant_name(variant)
            if vn:
                result.append(vn)
    else:
        vn = _variant_name(inner)
        if vn:
            result.append(vn)
    return result


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
        for elem in t.elements:
            elems.append(typenode_to_dict(elem))
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
        for param in t.params:
            params.append(typenode_to_dict(param))
        return JDict(
            {
                "_type": JStr("FuncType"),
                "params": JList(params),
                "ret": typenode_to_dict(t.ret),
            }
        )
    if isinstance(t, LiteralType):
        return JDict(
            {
                "_type": JStr("LiteralType"),
                "lit_value": JStr(t.lit_value),
                "base": typenode_to_dict(t.base),
            }
        )
    if isinstance(t, UnionType):
        variants: list[JsonValue] = []
        for variant in t.variants:
            variants.append(typenode_to_dict(variant))
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
NEVER_TYPE: TypeNode = PrimitiveType("never")
BYTES_TYPE: TypeNode = SliceType(PrimitiveType("byte"))
