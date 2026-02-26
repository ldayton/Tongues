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
    if isinstance(t, LiteralType):
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
        j = 0
        while j < len(a.elements):
            if not type_eq(a.elements[j], b.elements[j]):
                return False
            j += 1
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
        j = 0
        while j < len(a.params):
            if not type_eq(a.params[j], b.params[j]):
                return False
            j += 1
        return type_eq(a.ret, b.ret)
    if isinstance(a, UnionType) and isinstance(b, UnionType):
        if len(a.variants) != len(b.variants):
            return False
        i = 0
        while i < len(a.variants):
            found = False
            j = 0
            while j < len(b.variants):
                if type_eq(a.variants[i], b.variants[j]):
                    found = True
                j += 1
            if not found:
                return False
            i += 1
        return True
    if isinstance(a, LiteralType) and isinstance(b, LiteralType):
        return a.lit_value == b.lit_value and a.base.kind == b.base.kind
    return a == b


def combine_types(types: list[TypeNode]) -> TypeNode:
    """Flatten, deduplicate, and normalize a list of types into a single type."""
    # Flatten nested unions
    flat: list[TypeNode] = []
    i = 0
    while i < len(types):
        t = types[i]
        if isinstance(t, UnionType):
            j = 0
            while j < len(t.variants):
                flat.append(t.variants[j])
                j += 1
        else:
            flat.append(t)
        i += 1
    # Filter out never
    filtered: list[TypeNode] = []
    i = 0
    while i < len(flat):
        f = flat[i]
        if isinstance(f, PrimitiveType) and f.kind == "never":
            pass
        else:
            filtered.append(f)
        i += 1
    flat = filtered
    # Deduplicate via type_eq
    deduped: list[TypeNode] = []
    i = 0
    while i < len(flat):
        is_dup = False
        j = 0
        while j < len(deduped):
            if type_eq(flat[i], deduped[j]):
                is_dup = True
            j += 1
        if not is_dup:
            deduped.append(flat[i])
        i += 1
    # Literal absorption: base type absorbs its literals
    base_kinds: set[str] = set()
    i = 0
    while i < len(deduped):
        di = deduped[i]
        if isinstance(di, PrimitiveType):
            base_kinds.add(di.kind)
        i += 1
    absorbed: list[TypeNode] = []
    has_lit_true = False
    has_lit_false = False
    i = 0
    while i < len(deduped):
        d = deduped[i]
        if isinstance(d, LiteralType):
            if d.base.kind not in base_kinds:
                if d.base.kind == "bool" and d.lit_value == "true":
                    has_lit_true = True
                if d.base.kind == "bool" and d.lit_value == "false":
                    has_lit_false = True
                absorbed.append(d)
        else:
            absorbed.append(d)
        i += 1
    if has_lit_true and has_lit_false:
        merged: list[TypeNode] = []
        i = 0
        while i < len(absorbed):
            d2 = absorbed[i]
            if isinstance(d2, LiteralType) and d2.base.kind == "bool":
                pass
            else:
                merged.append(d2)
            i += 1
        merged.append(PrimitiveType("bool"))
        absorbed = merged
    deduped = absorbed
    if len(deduped) == 0:
        return PrimitiveType("void")
    if len(deduped) == 1:
        return deduped[0]
    # Check for None/void among variants
    has_none = False
    others: list[TypeNode] = []
    i = 0
    while i < len(deduped):
        d = deduped[i]
        if isinstance(d, PrimitiveType) and d.kind == "void":
            has_none = True
        else:
            others.append(d)
        i += 1
    if has_none:
        if len(others) == 0:
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
        if len(target.elements) > 0 and is_any(target.elements[0]):
            return True
    return False


def remove_from_union(t: TypeNode, to_remove: list[TypeNode]) -> TypeNode:
    """Remove variants matching any type in to_remove."""
    if isinstance(t, UnionType):
        remaining: list[TypeNode] = []
        i = 0
        while i < len(t.variants):
            matched = False
            j = 0
            while j < len(to_remove):
                if _removal_matches(t.variants[i], to_remove[j]):
                    matched = True
                j += 1
            if not matched:
                remaining.append(t.variants[i])
            i += 1
        if len(remaining) == 0:
            return PrimitiveType("never")
        if len(remaining) == 1:
            return remaining[0]
        return UnionType(remaining)
    if isinstance(t, OptionalType) and isinstance(t.inner, UnionType):
        remaining2: list[TypeNode] = []
        i = 0
        while i < len(t.inner.variants):
            matched = False
            j = 0
            while j < len(to_remove):
                if _removal_matches(t.inner.variants[i], to_remove[j]):
                    matched = True
                j += 1
            if not matched:
                remaining2.append(t.inner.variants[i])
            i += 1
        if len(remaining2) == 0:
            return PrimitiveType("void")
        if len(remaining2) == 1:
            return OptionalType(remaining2[0])
        return OptionalType(UnionType(remaining2))
    if isinstance(t, OptionalType):
        j = 0
        while j < len(to_remove):
            if _removal_matches(t.inner, to_remove[j]):
                return PrimitiveType("void")
            j += 1
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
        i = 0
        while i < len(inner.variants):
            vn = _variant_name(inner.variants[i])
            if vn != "":
                result.append(vn)
            i += 1
    else:
        vn = _variant_name(inner)
        if vn != "":
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
NEVER_TYPE: TypeNode = PrimitiveType("never")
