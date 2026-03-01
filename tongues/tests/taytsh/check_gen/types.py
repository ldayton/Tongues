"""TypePool — random type construction with constraint enforcement."""

from __future__ import annotations

from random import Random

from src.taytsh.ast import (
    Pos,
    TFuncType,
    TIdentType,
    TListType,
    TMapType,
    TOptionalType,
    TPrimitive,
    TSetType,
    TTupleType,
    TType,
    TUnionType,
)
from src.taytsh.check import (
    BOOL_T,
    BYTE_T,
    BYTES_T,
    FLOAT_T,
    INT_T,
    NIL_T,
    RUNE_T,
    STRING_T,
    VOID_T,
    EnumT,
    FnT,
    InterfaceT,
    ListT,
    MapT,
    SetT,
    StructT,
    TupleT,
    Type,
    UnionT,
    _has_zero_value,
    is_hashable,
    make_optional,
    normalize_union,
    type_eq,
)

from .features import FeatureVector
from .names import NameGen

P = Pos(1, 1)

ALL_PRIMITIVES: list[Type] = [INT_T, FLOAT_T, BOOL_T, BYTE_T, BYTES_T, STRING_T, RUNE_T]


class StructInfo:
    """Metadata for a generated struct."""

    def __init__(self, resolved: StructT, parent: str | None) -> None:
        self.resolved = resolved
        self.parent = parent


class InterfaceInfo:
    """Metadata for a generated interface."""

    def __init__(self, resolved: InterfaceT, variant_names: list[str]) -> None:
        self.resolved = resolved
        self.variant_names = variant_names


class EnumInfo:
    """Metadata for a generated enum."""

    def __init__(self, resolved: EnumT) -> None:
        self.resolved = resolved


class TypePool:
    def __init__(self, rng: Random, features: FeatureVector, names: NameGen) -> None:
        self.rng = rng
        self.features = features
        self.names = names
        self.pool: list[Type] = []
        self.structs: list[StructInfo] = []
        self.interfaces: list[InterfaceInfo] = []
        self.enums: list[EnumInfo] = []
        self.fn_types: list[FnT] = []

    def build(self) -> None:
        # 1. Always include all 7 primitives
        self.pool.extend(ALL_PRIMITIVES)

        # 2. 1-4 structs
        n_structs = self.rng.randint(1, 4)
        for _ in range(n_structs):
            self._make_struct(parent=None)

        # 3. 0-2 interfaces
        n_ifaces = self.rng.randint(0, 2)
        for _ in range(n_ifaces):
            self._make_interface()

        # 4. 0-2 enums
        n_enums = self.rng.randint(0, 2)
        for _ in range(n_enums):
            self._make_enum()

        # 5. 1-3 collection types (+ optional nested collections)
        n_collections = self.rng.randint(1, 3)
        for _ in range(n_collections):
            self._make_collection()
        if self.features.nested_collection and self.rng.random() < 0.6:
            self._make_nested_collection()

        # 6. 0-2 tuples
        n_tuples = self.rng.randint(0, 2)
        for _ in range(n_tuples):
            self._make_tuple()

        # 7. 0-1 fn types
        if self.rng.random() < 0.5:
            self._make_fn_type()

        # 8. Optional type if feature enabled
        if self.features.optional_type:
            inner = self._random_value_type()
            opt = make_optional(inner)
            if not self._pool_contains(opt):
                self.pool.append(opt)

        # 9. Union type if feature enabled
        if self.features.union_type:
            n_members = 3 if self.rng.random() < 0.3 else 2
            members: list[Type] = []
            for _ in range(n_members):
                t = self._random_value_type()
                while any(type_eq(t, m) for m in members):
                    t = self._random_value_type()
                members.append(t)
            union = normalize_union(members)
            if isinstance(union, UnionT) and not self._pool_contains(union):
                self.pool.append(union)

        # 10. Extra struct with rich field types (interfaces, collections, etc.)
        if self.interfaces and self.rng.random() < 0.4:
            self._make_struct(parent=None)

    def _make_struct(self, parent: str | None) -> StructT:
        name = self.names.struct_name()
        n_fields = self.rng.randint(1, 4)
        fields: dict[str, Type] = {}
        for i in range(n_fields):
            fname = self.names.var_name()
            ftype = self._random_value_type()
            fields[fname] = ftype
        st = StructT(
            kind="struct",
            name=name,
            fields=fields,
            methods={},
            parent=parent,
            field_order=list(fields.keys()),
        )
        self.structs.append(StructInfo(st, parent))
        self.pool.append(st)
        return st

    def _make_interface(self) -> InterfaceInfo:
        iname = self.names.interface_name()
        n_variants = self.rng.randint(2, 4)
        variant_names: list[str] = []
        for _ in range(n_variants):
            st = self._make_struct(parent=iname)
            variant_names.append(st.name)
        it = InterfaceT(kind="interface", name=iname, variants=variant_names)
        info = InterfaceInfo(it, variant_names)
        self.interfaces.append(info)
        self.pool.append(it)
        return info

    def _make_enum(self) -> EnumInfo:
        ename = self.names.enum_name()
        n_variants = self.rng.randint(2, 6)
        variants: list[str] = []
        for i in range(n_variants):
            variants.append(self.names.variant_name(i))
        et = EnumT(kind="enum", name=ename, variants=variants)
        info = EnumInfo(et)
        self.enums.append(info)
        self.pool.append(et)
        return info

    def _make_collection(self) -> None:
        kind = self.rng.choice(["list", "map", "set"])
        if kind == "list":
            elem = self._random_value_type()
            self.pool.append(ListT(kind="list", element=elem))
        elif kind == "map":
            key = self._random_hashable_type()
            value = self._random_value_type()
            self.pool.append(MapT(kind="map", key=key, value=value))
        else:
            elem = self._random_hashable_type()
            self.pool.append(SetT(kind="set", element=elem))

    def _make_nested_collection(self) -> None:
        """Wrap an existing collection in another collection."""
        inner_colls = [t for t in self.pool if isinstance(t, (ListT, MapT, SetT))]
        if not inner_colls:
            return
        inner = self.rng.choice(inner_colls)
        kind = self.rng.choice(["list", "map"])
        if kind == "list":
            self.pool.append(ListT(kind="list", element=inner))
        else:
            key = self._random_hashable_type()
            self.pool.append(MapT(kind="map", key=key, value=inner))

    def _make_tuple(self) -> None:
        arity = self.rng.randint(2, 3)
        elems = [self._random_value_type() for _ in range(arity)]
        self.pool.append(TupleT(kind="tuple", elements=elems))

    def _make_fn_type(self) -> None:
        n_params = self.rng.randint(0, 3)
        params = [self._random_value_type() for _ in range(n_params)]
        ret = self.random_type(exclude_void=False)
        ft = FnT(kind="fn", params=params, ret=ret)
        self.fn_types.append(ft)
        self.pool.append(ft)

    def _random_value_type(self) -> Type:
        """Pick a type from pool that is not void and not nil."""
        candidates = [
            t for t in self.pool if not type_eq(t, VOID_T) and not type_eq(t, NIL_T)
        ]
        if not candidates:
            return INT_T
        return self.rng.choice(candidates)

    def _random_hashable_type(self) -> Type:
        candidates = [t for t in self.pool if is_hashable(t) and not type_eq(t, NIL_T)]
        if not candidates:
            return INT_T
        return self.rng.choice(candidates)

    def _pool_contains(self, t: Type) -> bool:
        for p in self.pool:
            if type_eq(p, t):
                return True
        return False

    def random_type(
        self, *, exclude_void: bool = True, hashable_only: bool = False
    ) -> Type:
        candidates = list(self.pool)
        if exclude_void:
            candidates = [t for t in candidates if not type_eq(t, VOID_T)]
        if hashable_only:
            candidates = [t for t in candidates if is_hashable(t)]
        if not candidates:
            return INT_T
        return self.rng.choice(candidates)

    def random_value_type(self) -> Type:
        return self._random_value_type()

    def has_zero_value(self, t: Type) -> bool:
        return _has_zero_value(t)

    def matchable_types(self) -> list[Type]:
        result: list[Type] = []
        for t in self.pool:
            if isinstance(t, (InterfaceT, EnumT, UnionT)):
                result.append(t)
        return result

    def struct_types(self) -> list[StructT]:
        return [si.resolved for si in self.structs]

    def interface_info_for(self, t: InterfaceT) -> InterfaceInfo | None:
        for info in self.interfaces:
            if info.resolved.name == t.name:
                return info
        return None

    def struct_for_name(self, name: str) -> StructT | None:
        for si in self.structs:
            if si.resolved.name == name:
                return si.resolved
        return None

    def enum_for_name(self, name: str) -> EnumT | None:
        for ei in self.enums:
            if ei.resolved.name == name:
                return ei.resolved
        return None


def make_ttype(resolved: Type) -> TType:
    """Convert a resolved Type back into a parse-time TType AST node."""
    if isinstance(resolved, ListT):
        return TListType(pos=P, element=make_ttype(resolved.element))
    if isinstance(resolved, MapT):
        return TMapType(
            pos=P, key=make_ttype(resolved.key), value=make_ttype(resolved.value)
        )
    if isinstance(resolved, SetT):
        return TSetType(pos=P, element=make_ttype(resolved.element))
    if isinstance(resolved, TupleT):
        return TTupleType(pos=P, elements=[make_ttype(e) for e in resolved.elements])
    if isinstance(resolved, FnT):
        params = [make_ttype(p) for p in resolved.params]
        params.append(make_ttype(resolved.ret))
        return TFuncType(pos=P, params=params)
    if isinstance(resolved, StructT):
        return TIdentType(pos=P, name=resolved.name)
    if isinstance(resolved, InterfaceT):
        return TIdentType(pos=P, name=resolved.name)
    if isinstance(resolved, EnumT):
        return TIdentType(pos=P, name=resolved.name)
    if isinstance(resolved, UnionT):
        if len(resolved.members) == 2:
            non_nil = [m for m in resolved.members if not type_eq(m, NIL_T)]
            if len(non_nil) == 1:
                return TOptionalType(pos=P, inner=make_ttype(non_nil[0]))
        return TUnionType(pos=P, members=[make_ttype(m) for m in resolved.members])
    return TPrimitive(pos=P, kind=resolved.kind)
