"""Taytsh typechecker — validates a parsed TModule against the spec's type rules."""

from __future__ import annotations

from dataclasses import dataclass

from .ast import (
    Pos,
    TArg,
    TAssignStmt,
    TBinaryOp,
    TBoolLit,
    TBreakStmt,
    TByteLit,
    TBytesLit,
    TCall,
    TContinueStmt,
    TEnumDecl,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFnDecl,
    TFnLit,
    TFloatLit,
    TForStmt,
    TFuncType,
    TIdentType,
    TIfStmt,
    TIndex,
    TIntLit,
    TInterfaceDecl,
    TLetStmt,
    TListLit,
    TListType,
    TMapLit,
    TMapType,
    TMatchCase,
    TMatchStmt,
    TModule,
    TNilLit,
    TOpAssignStmt,
    TOptionalType,
    TPatternEnum,
    TPatternNil,
    TPatternType,
    TPrimitive,
    TRange,
    TReturnStmt,
    TRuneLit,
    TSetLit,
    TSetType,
    TSlice,
    TStmt,
    TStringLit,
    TStructDecl,
    TThrowStmt,
    TTernary,
    TTupleAccess,
    TTupleAssignStmt,
    TTupleLit,
    TTupleType,
    TTryStmt,
    TType,
    TUnaryOp,
    TUnionType,
    TVar,
    TWhileStmt,
)


# ============================================================
# RESOLVED TYPE REPRESENTATION
# ============================================================

TY_INT: str = "int"
TY_FLOAT: str = "float"
TY_BOOL: str = "bool"
TY_BYTE: str = "byte"
TY_BYTES: str = "bytes"
TY_STRING: str = "string"
TY_RUNE: str = "rune"
TY_NIL: str = "nil"
TY_VOID: str = "void"
TY_OBJ: str = "obj"


@dataclass
class Type:
    kind: str


@dataclass
class ListT(Type):
    element: Type


@dataclass
class MapT(Type):
    key: Type
    value: Type


@dataclass
class SetT(Type):
    element: Type


@dataclass
class TupleT(Type):
    elements: list[Type]


@dataclass
class FnT(Type):
    params: list[Type]
    ret: Type


@dataclass
class StructT(Type):
    name: str
    fields: dict[str, Type]
    methods: dict[str, FnT]
    parent: str | None


@dataclass
class InterfaceT(Type):
    name: str
    variants: list[str]


@dataclass
class EnumT(Type):
    name: str
    variants: list[str]


@dataclass
class UnionT(Type):
    members: list[Type]


# Primitive singletons
INT_T: Type = Type(kind=TY_INT)
FLOAT_T: Type = Type(kind=TY_FLOAT)
BOOL_T: Type = Type(kind=TY_BOOL)
BYTE_T: Type = Type(kind=TY_BYTE)
BYTES_T: Type = Type(kind=TY_BYTES)
STRING_T: Type = Type(kind=TY_STRING)
RUNE_T: Type = Type(kind=TY_RUNE)
NIL_T: Type = Type(kind=TY_NIL)
VOID_T: Type = Type(kind=TY_VOID)
OBJ_T: Type = Type(kind=TY_OBJ)

_PRIMITIVE_MAP: dict[str, Type] = {
    "int": INT_T,
    "float": FLOAT_T,
    "bool": BOOL_T,
    "byte": BYTE_T,
    "bytes": BYTES_T,
    "string": STRING_T,
    "rune": RUNE_T,
    "nil": NIL_T,
    "void": VOID_T,
}


# ============================================================
# TYPE EQUALITY
# ============================================================


def type_eq(a: Type, b: Type) -> bool:
    if a.kind != b.kind:
        return False
    if isinstance(a, ListT) and isinstance(b, ListT):
        return type_eq(a.element, b.element)
    if isinstance(a, MapT) and isinstance(b, MapT):
        return type_eq(a.key, b.key) and type_eq(a.value, b.value)
    if isinstance(a, SetT) and isinstance(b, SetT):
        return type_eq(a.element, b.element)
    if isinstance(a, TupleT) and isinstance(b, TupleT):
        if len(a.elements) != len(b.elements):
            return False
        i = 0
        while i < len(a.elements):
            if not type_eq(a.elements[i], b.elements[i]):
                return False
            i += 1
        return True
    if isinstance(a, FnT) and isinstance(b, FnT):
        if len(a.params) != len(b.params):
            return False
        i = 0
        while i < len(a.params):
            if not type_eq(a.params[i], b.params[i]):
                return False
            i += 1
        return type_eq(a.ret, b.ret)
    if isinstance(a, StructT) and isinstance(b, StructT):
        return a.name == b.name
    if isinstance(a, InterfaceT) and isinstance(b, InterfaceT):
        return a.name == b.name
    if isinstance(a, EnumT) and isinstance(b, EnumT):
        return a.name == b.name
    if isinstance(a, UnionT) and isinstance(b, UnionT):
        return _union_members_eq(a.members, b.members)
    return a.kind == b.kind


def _union_members_eq(a: list[Type], b: list[Type]) -> bool:
    if len(a) != len(b):
        return False
    for m in a:
        found = False
        for n in b:
            if type_eq(m, n):
                found = True
                break
        if not found:
            return False
    return True


# ============================================================
# TYPE KEYS
# ============================================================


def _type_key(t: Type) -> str:
    """Produce a string key for a resolved Type, for use in sets."""
    if isinstance(t, ListT):
        return "list[" + _type_key(t.element) + "]"
    if isinstance(t, MapT):
        return "map[" + _type_key(t.key) + "," + _type_key(t.value) + "]"
    if isinstance(t, SetT):
        return "set[" + _type_key(t.element) + "]"
    if isinstance(t, TupleT):
        parts: list[str] = []
        for e in t.elements:
            parts.append(_type_key(e))
        return "(" + ",".join(parts) + ")"
    if isinstance(t, FnT):
        parts2: list[str] = []
        for p in t.params:
            parts2.append(_type_key(p))
        return "fn[" + ",".join(parts2) + "->" + _type_key(t.ret) + "]"
    if isinstance(t, StructT):
        return "struct:" + t.name
    if isinstance(t, InterfaceT):
        return "interface:" + t.name
    if isinstance(t, EnumT):
        return "enum:" + t.name
    if isinstance(t, UnionT):
        keys: list[str] = []
        for m in t.members:
            keys.append(_type_key(m))
        keys.sort()
        return "union{" + "|".join(keys) + "}"
    return t.kind


def type_name(t: Type) -> str:
    """Human-readable name for a type, for error messages."""
    if isinstance(t, ListT):
        return "list[" + type_name(t.element) + "]"
    if isinstance(t, MapT):
        return "map[" + type_name(t.key) + ", " + type_name(t.value) + "]"
    if isinstance(t, SetT):
        return "set[" + type_name(t.element) + "]"
    if isinstance(t, TupleT):
        parts: list[str] = []
        for e in t.elements:
            parts.append(type_name(e))
        return "(" + ", ".join(parts) + ")"
    if isinstance(t, FnT):
        parts2: list[str] = []
        for p in t.params:
            parts2.append(type_name(p))
        parts2.append(type_name(t.ret))
        return "fn[" + ", ".join(parts2) + "]"
    if isinstance(t, StructT):
        return t.name
    if isinstance(t, InterfaceT):
        return t.name
    if isinstance(t, EnumT):
        return t.name
    if isinstance(t, UnionT):
        keyed: list[tuple[str, Type]] = []
        for m in t.members:
            keyed.append((_type_key(m), m))
        keyed.sort()
        parts3: list[str] = []
        for _k, m in keyed:
            parts3.append(type_name(m))
        return " | ".join(parts3)
    return t.kind


# ============================================================
# UNION NORMALIZATION
# ============================================================


def normalize_union(members: list[Type]) -> Type:
    flat: list[Type] = []
    for m in members:
        if isinstance(m, UnionT):
            for inner in m.members:
                flat.append(inner)
        else:
            flat.append(m)
    # Deduplicate
    deduped: list[Type] = []
    for m in flat:
        found = False
        for existing in deduped:
            if type_eq(m, existing):
                found = True
                break
        if not found:
            deduped.append(m)
    # Absorb obj
    for m in deduped:
        if m.kind == TY_OBJ and not isinstance(m, (StructT, InterfaceT, EnumT)):
            return OBJ_T
    if len(deduped) == 1:
        return deduped[0]
    return UnionT(kind="union", members=deduped)


def make_optional(inner: Type) -> Type:
    """Make T? — union of inner and nil."""
    if type_eq(inner, NIL_T):
        return NIL_T
    if isinstance(inner, UnionT):
        has_nil = False
        for m in inner.members:
            if type_eq(m, NIL_T):
                has_nil = True
                break
        if has_nil:
            return inner
        return normalize_union(list(inner.members) + [NIL_T])
    return normalize_union([inner, NIL_T])


def contains_nil(t: Type) -> bool:
    if type_eq(t, NIL_T):
        return True
    if isinstance(t, UnionT):
        for m in t.members:
            if type_eq(m, NIL_T):
                return True
    return False


def remove_nil(t: Type) -> Type:
    """Remove nil from a type (for narrowing)."""
    if type_eq(t, NIL_T):
        return NIL_T
    if isinstance(t, UnionT):
        remaining: list[Type] = []
        for m in t.members:
            if not type_eq(m, NIL_T):
                remaining.append(m)
        if len(remaining) == 0:
            return NIL_T
        if len(remaining) == 1:
            return remaining[0]
        return UnionT(kind="union", members=remaining)
    return t


# ============================================================
# ASSIGNABILITY
# ============================================================


def is_assignable(source: Type, target: Type) -> bool:
    """Can a value of type `source` be assigned to a slot of type `target`?"""
    if type_eq(source, target):
        return True
    # Source is struct, target is its interface
    if isinstance(source, StructT) and isinstance(target, InterfaceT):
        return source.parent == target.name
    # Target is obj (the universal supertype)
    if target.kind == TY_OBJ and not isinstance(target, (StructT, InterfaceT, EnumT)):
        return True
    # Source is nil, target contains nil
    if source.kind == TY_NIL and contains_nil(target):
        return True
    # Target is union, source is a member
    if isinstance(target, UnionT):
        for m in target.members:
            if is_assignable(source, m):
                return True
    # Source is union, all members assignable to target
    if isinstance(source, UnionT):
        all_ok = True
        for m in source.members:
            if not is_assignable(m, target):
                all_ok = False
                break
        if all_ok:
            return True
    return False


# ============================================================
# ZERO VALUES
# ============================================================


def is_hashable(t: Type) -> bool:
    """Check if a type is hashable (valid as map key or set element)."""
    if t.kind in (TY_INT, TY_FLOAT, TY_BOOL, TY_BYTE, TY_BYTES, TY_STRING, TY_RUNE):
        return True
    if isinstance(t, EnumT):
        return True
    if isinstance(t, TupleT):
        for e in t.elements:
            if not is_hashable(e):
                return False
        return True
    return False


def has_zero_value(t: Type) -> bool:
    if t.kind in (TY_INT, TY_FLOAT, TY_BOOL, TY_BYTE, TY_BYTES, TY_STRING, TY_RUNE):
        return True
    if t.kind == TY_OBJ and not isinstance(t, (StructT, InterfaceT, EnumT)):
        return True
    if isinstance(t, ListT):
        return True
    if isinstance(t, MapT):
        return True
    if isinstance(t, SetT):
        return True
    if isinstance(t, TupleT):
        for e in t.elements:
            if not has_zero_value(e):
                return False
        return True
    if isinstance(t, UnionT):
        return contains_nil(t)
    if t.kind == TY_NIL:
        return True
    return False


# ============================================================
# BUILT-IN NAMES (reserved)
# ============================================================

BUILTIN_NAMES: set[str] = {
    # Numeric
    "Abs",
    "Min",
    "Max",
    "Sum",
    "Pow",
    "Round",
    "DivMod",
    # Bytes
    "Encode",
    "Decode",
    # Strings
    "Len",
    "Concat",
    "RuneFromInt",
    "RuneToInt",
    "ParseInt",
    "ParseFloat",
    "FormatInt",
    "Upper",
    "Lower",
    "Trim",
    "TrimStart",
    "TrimEnd",
    "Split",
    "SplitN",
    "SplitWhitespace",
    "Join",
    "Find",
    "RFind",
    "Count",
    "Contains",
    "Replace",
    "Repeat",
    "StartsWith",
    "EndsWith",
    "IsDigit",
    "IsAlpha",
    "IsAlnum",
    "IsSpace",
    "IsUpper",
    "IsLower",
    "Format",
    # Lists
    "Append",
    "Insert",
    "Pop",
    "RemoveAt",
    "IndexOf",
    "Reversed",
    "Sorted",
    # Maps
    "Map",
    "Get",
    "Delete",
    "Keys",
    "Values",
    "Items",
    "Merge",
    # Sets
    "Set",
    "Add",
    "Remove",
    # Conversions
    "IntToFloat",
    "FloatToInt",
    "ByteToInt",
    "IntToByte",
    "ToString",
    # I/O
    "WriteOut",
    "WriteErr",
    "WritelnOut",
    "WritelnErr",
    "ReadLine",
    "ReadAll",
    "ReadBytes",
    "ReadBytesN",
    "Args",
    "GetEnv",
    "Exit",
    # Assert / Unwrap
    "Assert",
    "Unwrap",
    # Math
    "IsNaN",
    "IsInf",
}

# Names reserved for user bindings (top-level decls, locals, params, etc.).
# Most builtins are reserved; set-specific operations like Add can be shadowed.
RESERVED_NAMES: set[str] = BUILTIN_NAMES - {"Add"}

# Built-in error struct names
BUILTIN_STRUCTS: dict[str, dict[str, Type]] = {
    "KeyError": {"message": STRING_T},
    "IndexError": {"message": STRING_T},
    "ZeroDivisionError": {"message": STRING_T},
    "AssertError": {"message": STRING_T},
    "NilError": {"message": STRING_T},
    "ValueError": {"message": STRING_T},
}


# ============================================================
# CHECK ERROR
# ============================================================


class CheckError(Exception):
    def __init__(self, msg: str, line: int, col: int):
        self.msg: str = msg
        self.line: int = line
        self.col: int = col
        super().__init__(msg + " at line " + str(line) + " col " + str(col))


class _BuiltinCtx:
    """Helper for check_builtin_call — avoids nested function closures."""

    checker: Checker
    name: str
    arg_types: list[Type | None]
    n: int
    pos: Pos

    def __init__(
        self, checker: Checker, name: str, arg_types: list[Type | None], pos: Pos
    ) -> None:
        self.checker = checker
        self.name = name
        self.arg_types = arg_types
        self.n = len(arg_types)
        self.pos = pos

    def require(self, count: int) -> bool:
        if self.n != count:
            self.checker.error(
                self.name
                + " requires "
                + str(count)
                + " argument(s), got "
                + str(self.n),
                self.pos,
            )
            return False
        return True

    def require_range(self, lo: int, hi: int) -> bool:
        if self.n < lo or self.n > hi:
            self.checker.error(
                self.name
                + " requires "
                + str(lo)
                + "-"
                + str(hi)
                + " argument(s), got "
                + str(self.n),
                self.pos,
            )
            return False
        return True

    def arg(self, i: int) -> Type | None:
        if i < len(self.arg_types):
            return self.arg_types[i]
        return None


# ============================================================
# CHECKER
# ============================================================


class Checker:
    def __init__(self) -> None:
        self.errors: list[CheckError] = []
        self.types: dict[str, Type] = {}
        self.functions: dict[str, FnT] = {}
        self.fn_param_names: dict[str, list[str]] = {}
        self.scopes: list[dict[str, Type]] = []
        self.current_fn_ret: Type | None = None
        self.in_loop: bool = False
        self.current_struct: StructT | None = None
        # Register built-in error structs
        for name, fields in BUILTIN_STRUCTS.items():
            st = StructT(
                kind="struct", name=name, fields=dict(fields), methods={}, parent=None
            )
            self.types[name] = st

    def error(self, msg: str, pos: Pos) -> None:
        self.errors.append(CheckError(msg, pos.line, pos.col))

    # ── Scope management ──────────────────────────────────────

    def enter_scope(self) -> None:
        self.scopes.append({})

    def exit_scope(self) -> None:
        self.scopes.pop()

    def declare(self, name: str, typ: Type, pos: Pos) -> None:
        if name == "_":
            return
        if name in RESERVED_NAMES:
            self.error("cannot use reserved name '" + name + "'", pos)
            return
        # Check current scope for duplicate
        if len(self.scopes) > 0 and name in self.scopes[-1]:
            self.error("'" + name + "' already declared in this scope", pos)
            return
        # Check outer scopes for shadowing
        i = len(self.scopes) - 2
        while i >= 0:
            if name in self.scopes[i]:
                self.error("'" + name + "' shadows outer binding", pos)
                return
            i -= 1
        if len(self.scopes) > 0:
            self.scopes[-1][name] = typ

    def lookup(self, name: str, pos: Pos) -> Type | None:
        # Search scopes innermost-out
        i = len(self.scopes) - 1
        while i >= 0:
            if name in self.scopes[i]:
                return self.scopes[i][name]
            i -= 1
        # Check top-level functions
        if name in self.functions:
            fn = self.functions[name]
            return fn
        # Check type names (struct constructors, enum access)
        if name in self.types:
            return self.types[name]
        self.error("undefined name '" + name + "'", pos)
        return None

    # ── Type resolution ───────────────────────────────────────

    def resolve_type(self, t: TType) -> Type:
        """Resolve a parse-time TType node into a checked Type."""
        if isinstance(t, TPrimitive):
            result = _PRIMITIVE_MAP.get(t.kind)
            if result is None:
                self.error("unknown primitive type '" + t.kind + "'", t.pos)
                return OBJ_T
            return result
        if isinstance(t, TListType):
            elem = self.resolve_type(t.element)
            if type_eq(elem, VOID_T):
                self.error("void is not a value type", t.pos)
            return ListT(kind="list", element=elem)
        if isinstance(t, TMapType):
            key = self.resolve_type(t.key)
            value = self.resolve_type(t.value)
            if type_eq(key, VOID_T) or type_eq(value, VOID_T):
                self.error("void is not a value type", t.pos)
            if not type_eq(key, VOID_T) and not is_hashable(key):
                self.error(type_name(key) + " is not hashable", t.pos)
            return MapT(kind="map", key=key, value=value)
        if isinstance(t, TSetType):
            elem = self.resolve_type(t.element)
            if type_eq(elem, VOID_T):
                self.error("void is not a value type", t.pos)
            elif not is_hashable(elem):
                self.error(type_name(elem) + " is not hashable", t.pos)
            return SetT(kind="set", element=elem)
        if isinstance(t, TTupleType):
            elems: list[Type] = []
            for e in t.elements:
                resolved = self.resolve_type(e)
                if type_eq(resolved, VOID_T):
                    self.error("void is not a value type", t.pos)
                elems.append(resolved)
            return TupleT(kind="tuple", elements=elems)
        if isinstance(t, TFuncType):
            if len(t.params) < 1:
                self.error("fn type must have at least a return type", t.pos)
                return OBJ_T
            params: list[Type] = []
            i = 0
            while i < len(t.params) - 1:
                p = self.resolve_type(t.params[i])
                if type_eq(p, VOID_T):
                    self.error("void is not a value type", t.params[i].pos)
                params.append(p)
                i += 1
            ret = self.resolve_type(t.params[-1])
            return FnT(kind="fn", params=params, ret=ret)
        if isinstance(t, TIdentType):
            if t.name in self.types:
                return self.types[t.name]
            self.error("unknown type '" + t.name + "'", t.pos)
            return OBJ_T
        if isinstance(t, TUnionType):
            members: list[Type] = []
            for m in t.members:
                resolved = self.resolve_type(m)
                if type_eq(resolved, VOID_T):
                    self.error("void is not a value type", m.pos)
                members.append(resolved)
            return normalize_union(members)
        if isinstance(t, TOptionalType):
            inner = self.resolve_type(t.inner)
            return make_optional(inner)
        self.error("unhandled type node", t.pos)
        return OBJ_T

    # ── Pass 1: Collect declarations ──────────────────────────

    def collect_declarations(self, module: TModule) -> None:
        # First pass: register all type names (structs, interfaces, enums)
        # so they can reference each other
        for decl in module.decls:
            if isinstance(decl, TStructDecl):
                if decl.name in self.types:
                    self.error("duplicate type name '" + decl.name + "'", decl.pos)
                    continue
                if decl.name in RESERVED_NAMES:
                    self.error("cannot use reserved name '" + decl.name + "'", decl.pos)
                    continue
                # Placeholder — fields/methods filled in next loop
                st = StructT(
                    kind="struct",
                    name=decl.name,
                    fields={},
                    methods={},
                    parent=decl.parent,
                )
                self.types[decl.name] = st
            elif isinstance(decl, TInterfaceDecl):
                if decl.name in self.types:
                    self.error("duplicate type name '" + decl.name + "'", decl.pos)
                    continue
                if decl.name in RESERVED_NAMES:
                    self.error("cannot use reserved name '" + decl.name + "'", decl.pos)
                    continue
                it = InterfaceT(kind="interface", name=decl.name, variants=[])
                self.types[decl.name] = it
            elif isinstance(decl, TEnumDecl):
                if decl.name in self.types:
                    self.error("duplicate type name '" + decl.name + "'", decl.pos)
                    continue
                if decl.name in RESERVED_NAMES:
                    self.error("cannot use reserved name '" + decl.name + "'", decl.pos)
                    continue
                seen_variants: set[str] = set()
                for v in decl.variants:
                    if v in seen_variants:
                        self.error(
                            "duplicate variant '" + v + "' in " + decl.name, decl.pos
                        )
                    seen_variants.add(v)
                et = EnumT(kind="enum", name=decl.name, variants=list(decl.variants))
                self.types[decl.name] = et

        # Second pass: resolve struct fields, methods, and interface parents
        for decl in module.decls:
            if isinstance(decl, TStructDecl):
                if decl.name not in self.types:
                    continue
                st2 = self.types[decl.name]
                if not isinstance(st2, StructT):
                    continue
                # Resolve fields
                for f in decl.fields:
                    ft = self.resolve_type(f.typ)
                    st2.fields[f.name] = ft
                # Resolve methods
                for m in decl.methods:
                    mparams: list[Type] = []
                    for p in m.params:
                        if p.typ is not None:
                            mparams.append(self.resolve_type(p.typ))
                    mret = self.resolve_type(m.ret)
                    st2.methods[m.name] = FnT(kind="fn", params=mparams, ret=mret)
                # Register with parent interface
                if decl.parent is not None:
                    if decl.parent not in self.types:
                        self.error("unknown interface '" + decl.parent + "'", decl.pos)
                    else:
                        parent_type = self.types[decl.parent]
                        if not isinstance(parent_type, InterfaceT):
                            self.error(
                                "'" + decl.parent + "' is not an interface", decl.pos
                            )
                        else:
                            parent_type.variants.append(decl.name)

        # Third pass: register top-level functions
        for decl in module.decls:
            if isinstance(decl, TFnDecl):
                if decl.name in self.functions:
                    self.error("duplicate function name '" + decl.name + "'", decl.pos)
                    continue
                if decl.name in self.types:
                    self.error(
                        "'" + decl.name + "' already declared as a type", decl.pos
                    )
                    continue
                if decl.name in RESERVED_NAMES:
                    self.error("cannot use reserved name '" + decl.name + "'", decl.pos)
                    continue
                params2: list[Type] = []
                pnames: list[str] = []
                for p in decl.params:
                    pnames.append(p.name)
                    if p.typ is not None:
                        params2.append(self.resolve_type(p.typ))
                ret2 = self.resolve_type(decl.ret)
                self.functions[decl.name] = FnT(kind="fn", params=params2, ret=ret2)
                self.fn_param_names[decl.name] = pnames

    # ── Pass 2: Check bodies ──────────────────────────────────

    def check_bodies(self, module: TModule) -> None:
        for decl in module.decls:
            if isinstance(decl, TFnDecl):
                self.check_fn_decl(decl)
            elif isinstance(decl, TStructDecl):
                self.check_struct_methods(decl)

    def check_fn_decl(self, decl: TFnDecl) -> None:
        ret = self.resolve_type(decl.ret)
        self.current_fn_ret = ret
        self.enter_scope()
        for p in decl.params:
            if p.name == "self" and self.current_struct is None:
                self.error("self outside method", p.pos)
                continue
            if p.typ is not None:
                pt = self.resolve_type(p.typ)
                if type_eq(pt, VOID_T):
                    self.error("void is not a value type", p.pos)
                self.declare(p.name, pt, p.pos)
        self.check_stmts(decl.body)
        self.exit_scope()
        self.current_fn_ret = None

    def check_struct_methods(self, decl: TStructDecl) -> None:
        if decl.name not in self.types:
            return
        st = self.types[decl.name]
        if not isinstance(st, StructT):
            return
        old_struct = self.current_struct
        self.current_struct = st
        for method in decl.methods:
            ret = self.resolve_type(method.ret)
            self.current_fn_ret = ret
            self.enter_scope()
            # Bind self
            for p in method.params:
                if p.typ is None:
                    self.declare(p.name, st, p.pos)
                else:
                    pt = self.resolve_type(p.typ)
                    self.declare(p.name, pt, p.pos)
            self.check_stmts(method.body)
            self.exit_scope()
            self.current_fn_ret = None
        self.current_struct = old_struct

    # ── Assignment target validation ─────────────────────────

    def _is_valid_lvalue(self, expr: TExpr) -> bool:
        """Check if expr is a valid assignment target (variable, field, or index)."""
        if isinstance(expr, TVar):
            return True
        if isinstance(expr, TFieldAccess):
            return True
        if isinstance(expr, TIndex):
            return True
        if isinstance(expr, TTupleAccess):
            return True  # syntactically valid, immutability checked separately
        return False

    def _check_assign_target(self, target: TExpr, pos: Pos) -> str | None:
        """Check assignment target for immutability. Returns error msg or None."""
        if isinstance(target, TVar) and target.name == "self":
            return "cannot assign to self"
        if isinstance(target, TTupleAccess):
            return "tuple elements are immutable"
        if isinstance(target, TIndex):
            obj_type = self.check_expr(target.obj, None)
            if obj_type is not None and type_eq(obj_type, STRING_T):
                return "cannot assign to string index"
            if obj_type is not None and type_eq(obj_type, BYTES_T):
                return "cannot assign to bytes index"
        return None

    # ── Statement checking ────────────────────────────────────

    def check_stmts(self, stmts: list[TStmt]) -> None:
        for s in stmts:
            self.check_stmt(s)

    def check_stmt(self, stmt: TStmt) -> None:
        if isinstance(stmt, TLetStmt):
            self.check_let_stmt(stmt)
        elif isinstance(stmt, TAssignStmt):
            self.check_assign_stmt(stmt)
        elif isinstance(stmt, TOpAssignStmt):
            self.check_op_assign_stmt(stmt)
        elif isinstance(stmt, TTupleAssignStmt):
            self.check_tuple_assign_stmt(stmt)
        elif isinstance(stmt, TReturnStmt):
            self.check_return_stmt(stmt)
        elif isinstance(stmt, TBreakStmt):
            if not self.in_loop:
                self.error("break outside of loop", stmt.pos)
        elif isinstance(stmt, TContinueStmt):
            if not self.in_loop:
                self.error("continue outside of loop", stmt.pos)
        elif isinstance(stmt, TThrowStmt):
            self.check_expr(stmt.expr, None)
        elif isinstance(stmt, TExprStmt):
            self.check_expr(stmt.expr, None)
        elif isinstance(stmt, TIfStmt):
            self.check_if_stmt(stmt)
        elif isinstance(stmt, TWhileStmt):
            self.check_while_stmt(stmt)
        elif isinstance(stmt, TForStmt):
            self.check_for_stmt(stmt)
        elif isinstance(stmt, TMatchStmt):
            self.check_match_stmt(stmt)
        elif isinstance(stmt, TTryStmt):
            self.check_try_stmt(stmt)
        else:
            self.error("unhandled statement type", stmt.pos)

    def check_let_stmt(self, stmt: TLetStmt) -> None:
        declared_type = self.resolve_type(stmt.typ)
        if stmt.value is not None:
            val_type = self.check_expr(stmt.value, declared_type)
            if val_type is not None and not is_assignable(val_type, declared_type):
                self.error(
                    "cannot assign "
                    + type_name(val_type)
                    + " to "
                    + type_name(declared_type),
                    stmt.pos,
                )
        else:
            if not has_zero_value(declared_type):
                self.error(
                    type_name(declared_type)
                    + " has no zero value; initializer required",
                    stmt.pos,
                )
        self.declare(stmt.name, declared_type, stmt.pos)

    def check_assign_stmt(self, stmt: TAssignStmt) -> None:
        if not self._is_valid_lvalue(stmt.target):
            self.error("invalid assignment target", stmt.pos)
            return
        immut_err = self._check_assign_target(stmt.target, stmt.pos)
        if immut_err is not None:
            self.error(immut_err, stmt.pos)
            return
        target_type = self.check_expr(stmt.target, None)
        if target_type is not None:
            val_type = self.check_expr(stmt.value, target_type)
            if val_type is not None and not is_assignable(val_type, target_type):
                self.error(
                    "cannot assign "
                    + type_name(val_type)
                    + " to "
                    + type_name(target_type),
                    stmt.pos,
                )

    def check_op_assign_stmt(self, stmt: TOpAssignStmt) -> None:
        target_type = self.check_expr(stmt.target, None)
        val_type = self.check_expr(stmt.value, target_type)
        if target_type is not None and val_type is not None:
            # The operator is e.g. "+=" -> underlying op is "+"
            op = stmt.op
            if op.endswith("="):
                op = op[:-1]
            result = self.check_binary_op_types(op, target_type, val_type, stmt.pos)
            if result is not None and not is_assignable(result, target_type):
                self.error(
                    "operator result "
                    + type_name(result)
                    + " not assignable to "
                    + type_name(target_type),
                    stmt.pos,
                )

    def check_tuple_assign_stmt(self, stmt: TTupleAssignStmt) -> None:
        rhs_type = self.check_expr(stmt.value, None)
        if rhs_type is None:
            return
        if not isinstance(rhs_type, TupleT):
            self.error(
                "right side of tuple assignment must be a tuple, got "
                + type_name(rhs_type),
                stmt.pos,
            )
            return
        if len(rhs_type.elements) != len(stmt.targets):
            self.error(
                "tuple assignment arity mismatch: "
                + str(len(stmt.targets))
                + " targets, "
                + str(len(rhs_type.elements))
                + " values",
                stmt.pos,
            )
            return
        i = 0
        while i < len(stmt.targets):
            target_type = self.check_expr(stmt.targets[i], None)
            if target_type is not None and not is_assignable(
                rhs_type.elements[i], target_type
            ):
                self.error(
                    "cannot assign "
                    + type_name(rhs_type.elements[i])
                    + " to "
                    + type_name(target_type),
                    stmt.pos,
                )
            i += 1

    def check_return_stmt(self, stmt: TReturnStmt) -> None:
        if self.current_fn_ret is None:
            self.error("return outside of function", stmt.pos)
            return
        if stmt.value is None:
            if not type_eq(self.current_fn_ret, VOID_T):
                self.error("missing return value in non-void function", stmt.pos)
        else:
            val_type = self.check_expr(stmt.value, self.current_fn_ret)
            if val_type is not None and not is_assignable(
                val_type, self.current_fn_ret
            ):
                self.error(
                    "cannot return "
                    + type_name(val_type)
                    + " from function returning "
                    + type_name(self.current_fn_ret),
                    stmt.pos,
                )

    def check_if_stmt(self, stmt: TIfStmt) -> None:
        cond_type = self.check_expr(stmt.cond, BOOL_T)
        if cond_type is not None and not type_eq(cond_type, BOOL_T):
            self.error(
                "if condition must be bool, got " + type_name(cond_type), stmt.pos
            )
        # Nil narrowing in then-body
        narrowed_name: str | None = None
        narrowed_type: Type | None = None
        narrowed_else_type: Type | None = None
        if isinstance(stmt.cond, TBinaryOp):
            if (
                stmt.cond.op == "!="
                and isinstance(stmt.cond.right, TNilLit)
                and isinstance(stmt.cond.left, TVar)
            ):
                var_type = self.lookup(stmt.cond.left.name, stmt.cond.left.pos)
                if var_type is not None and contains_nil(var_type):
                    narrowed_name = stmt.cond.left.name
                    narrowed_type = remove_nil(var_type)
                    narrowed_else_type = NIL_T
            elif (
                stmt.cond.op == "=="
                and isinstance(stmt.cond.right, TNilLit)
                and isinstance(stmt.cond.left, TVar)
            ):
                var_type2 = self.lookup(stmt.cond.left.name, stmt.cond.left.pos)
                if var_type2 is not None and contains_nil(var_type2):
                    narrowed_name = stmt.cond.left.name
                    narrowed_type = NIL_T
                    narrowed_else_type = remove_nil(var_type2)
        # Check then-body with narrowing
        self.enter_scope()
        if narrowed_name is not None and narrowed_type is not None:
            self.scopes[-1][narrowed_name] = narrowed_type
        self.check_stmts(stmt.then_body)
        self.exit_scope()
        # Check else-body with reverse narrowing
        if stmt.else_body is not None:
            self.enter_scope()
            if narrowed_name is not None and narrowed_else_type is not None:
                self.scopes[-1][narrowed_name] = narrowed_else_type
            self.check_stmts(stmt.else_body)
            self.exit_scope()

    def check_while_stmt(self, stmt: TWhileStmt) -> None:
        cond_type = self.check_expr(stmt.cond, BOOL_T)
        if cond_type is not None and not type_eq(cond_type, BOOL_T):
            self.error(
                "while condition must be bool, got " + type_name(cond_type), stmt.pos
            )
        old_in_loop = self.in_loop
        self.in_loop = True
        self.enter_scope()
        self.check_stmts(stmt.body)
        self.exit_scope()
        self.in_loop = old_in_loop

    def check_for_stmt(self, stmt: TForStmt) -> None:
        old_in_loop = self.in_loop
        self.in_loop = True
        self.enter_scope()
        if isinstance(stmt.iterable, TRange):
            # range — all args must be int, loop var is int
            for arg in stmt.iterable.args:
                arg_type = self.check_expr(arg, INT_T)
                if arg_type is not None and not type_eq(arg_type, INT_T):
                    self.error(
                        "range argument must be int, got " + type_name(arg_type),
                        stmt.iterable.pos,
                    )
            if len(stmt.binding) == 1:
                self.declare(stmt.binding[0], INT_T, stmt.pos)
            elif len(stmt.binding) == 2:
                self.error("range does not support the two-variable form", stmt.pos)
                self.declare(stmt.binding[0], INT_T, stmt.pos)
                self.declare(stmt.binding[1], INT_T, stmt.pos)
        else:
            iter_type = self.check_expr(stmt.iterable, None)
            if iter_type is not None:
                self.bind_for_vars(stmt.binding, iter_type, stmt.pos)
        self.check_stmts(stmt.body)
        self.exit_scope()
        self.in_loop = old_in_loop

    def bind_for_vars(self, binding: list[str], iter_type: Type, pos: Pos) -> None:
        if isinstance(iter_type, ListT):
            if len(binding) == 1:
                self.declare(binding[0], iter_type.element, pos)
            elif len(binding) == 2:
                self.declare(binding[0], INT_T, pos)
                self.declare(binding[1], iter_type.element, pos)
        elif type_eq(iter_type, STRING_T):
            if len(binding) == 1:
                self.declare(binding[0], RUNE_T, pos)
            elif len(binding) == 2:
                self.declare(binding[0], INT_T, pos)
                self.declare(binding[1], RUNE_T, pos)
        elif type_eq(iter_type, BYTES_T):
            if len(binding) == 1:
                self.declare(binding[0], BYTE_T, pos)
            elif len(binding) == 2:
                self.declare(binding[0], INT_T, pos)
                self.declare(binding[1], BYTE_T, pos)
        elif isinstance(iter_type, MapT):
            if len(binding) == 1:
                self.declare(binding[0], iter_type.key, pos)
            elif len(binding) == 2:
                self.declare(binding[0], iter_type.key, pos)
                self.declare(binding[1], iter_type.value, pos)
        elif isinstance(iter_type, SetT):
            if len(binding) == 1:
                self.declare(binding[0], iter_type.element, pos)
            elif len(binding) == 2:
                self.error("sets do not support the two-variable for form", pos)
                self.declare(binding[0], iter_type.element, pos)
                self.declare(binding[1], iter_type.element, pos)
        else:
            self.error("cannot iterate over " + type_name(iter_type), pos)
            for b in binding:
                self.declare(b, OBJ_T, pos)

    def check_match_stmt(self, stmt: TMatchStmt) -> None:
        scrutinee_type = self.check_expr(stmt.expr, None)
        if scrutinee_type is None:
            return
        if not self._is_matchable(scrutinee_type):
            self.error("cannot match on " + type_name(scrutinee_type), stmt.pos)
            return
        covered: list[str] = []
        for case in stmt.cases:
            self.check_match_case(case, scrutinee_type, covered)
        has_default = stmt.default is not None
        if has_default:
            dflt = stmt.default
            assert dflt is not None
            self.enter_scope()
            if dflt.name is not None:
                self.declare(dflt.name, OBJ_T, dflt.pos)
            self.check_stmts(dflt.body)
            self.exit_scope()
        if not has_default:
            self.check_exhaustiveness(scrutinee_type, covered, stmt.pos)

    def _is_matchable(self, t: Type) -> bool:
        """Check if a type is valid as a match scrutinee."""
        if isinstance(t, (InterfaceT, EnumT)):
            return True
        if isinstance(t, UnionT):
            return True
        if t.kind == TY_OBJ and not isinstance(t, (StructT, InterfaceT, EnumT)):
            return True
        return False

    def _allowed_in_match(self, case_type: Type, scrutinee: Type) -> bool:
        """Check if case_type is a valid case for the given scrutinee."""
        if isinstance(scrutinee, InterfaceT):
            if isinstance(case_type, StructT) and case_type.parent == scrutinee.name:
                return True
            return False
        if isinstance(scrutinee, EnumT):
            return False  # enums use TPatternEnum, not TPatternType
        if isinstance(scrutinee, UnionT):
            for m in scrutinee.members:
                if type_eq(case_type, m):
                    return True
                if isinstance(m, InterfaceT) and isinstance(case_type, StructT):
                    if case_type.parent == m.name:
                        return True
                if isinstance(m, InterfaceT) and isinstance(case_type, InterfaceT):
                    if type_eq(case_type, m):
                        return True
            return False
        if scrutinee.kind == TY_OBJ:
            return True
        return False

    def check_match_case(
        self, case: TMatchCase, scrutinee: Type, covered: list[str]
    ) -> None:
        pat = case.pattern
        if isinstance(pat, TPatternNil):
            key = "nil"
            if key in covered:
                self.error("duplicate case: nil", pat.pos)
            covered.append(key)
            self.enter_scope()
            self.check_stmts(case.body)
            self.exit_scope()
        elif isinstance(pat, TPatternEnum):
            # Validate enum — check that this enum is valid for the scrutinee
            if pat.enum_name in self.types:
                enum_type = self.types[pat.enum_name]
                if isinstance(enum_type, EnumT):
                    # Check that this enum is the right one for the scrutinee
                    scrutinee_enum = self._get_scrutinee_enum(scrutinee)
                    if scrutinee_enum is not None and not type_eq(
                        enum_type, scrutinee_enum
                    ):
                        self.error(
                            "'"
                            + pat.enum_name
                            + "."
                            + pat.variant
                            + "' is not a variant of "
                            + scrutinee_enum.name,
                            pat.pos,
                        )
                    elif pat.variant not in enum_type.variants:
                        self.error(
                            "'"
                            + pat.variant
                            + "' is not a variant of "
                            + pat.enum_name,
                            pat.pos,
                        )
                    else:
                        key = pat.enum_name + "." + pat.variant
                        if key in covered:
                            self.error("duplicate match case: " + key, pat.pos)
                        covered.append(key)
                else:
                    self.error("'" + pat.enum_name + "' is not an enum", pat.pos)
            else:
                self.error("unknown type '" + pat.enum_name + "'", pat.pos)
            self.enter_scope()
            self.check_stmts(case.body)
            self.exit_scope()
        elif isinstance(pat, TPatternType):
            case_type = self.resolve_type(pat.type_name)
            if not self._allowed_in_match(case_type, scrutinee):
                self.error(
                    type_name(case_type)
                    + " is not a variant of "
                    + type_name(scrutinee),
                    pat.pos,
                )
            key2 = _type_key(case_type)
            if key2 in covered:
                self.error("duplicate match case: " + type_name(case_type), pat.pos)
            covered.append(key2)
            # For interface cases in a union, also mark all struct variants covered
            if isinstance(case_type, InterfaceT):
                for v in case_type.variants:
                    vkey = _type_key(self.types[v])
                    if vkey not in covered:
                        covered.append(vkey)
            self.enter_scope()
            self.declare(pat.name, case_type, pat.pos)
            self.check_stmts(case.body)
            self.exit_scope()

    def _get_scrutinee_enum(self, scrutinee: Type) -> EnumT | None:
        """Get the enum type from the scrutinee (direct or optional)."""
        if isinstance(scrutinee, EnumT):
            return scrutinee
        if isinstance(scrutinee, UnionT):
            for m in scrutinee.members:
                if isinstance(m, EnumT):
                    return m
        return None

    def check_exhaustiveness(
        self, scrutinee: Type, covered: list[str], pos: Pos
    ) -> None:
        required: list[str] = []
        if isinstance(scrutinee, InterfaceT):
            for v in scrutinee.variants:
                required.append(_type_key(self.types[v]))
        elif isinstance(scrutinee, EnumT):
            for v in scrutinee.variants:
                required.append(scrutinee.name + "." + v)
        elif isinstance(scrutinee, UnionT):
            for m in scrutinee.members:
                if type_eq(m, NIL_T):
                    required.append("nil")
                elif isinstance(m, InterfaceT):
                    for v in m.variants:
                        required.append(_type_key(self.types[v]))
                elif isinstance(m, EnumT):
                    for v in m.variants:
                        required.append(m.name + "." + v)
                else:
                    required.append(_type_key(m))
        elif type_eq(scrutinee, OBJ_T):
            self.error("non-exhaustive match on obj: default case required", pos)
            return
        else:
            return
        missing: list[str] = []
        for r in required:
            if r not in covered:
                missing.append(r)
        if len(missing) > 0:
            self.error("non-exhaustive match: missing cases", pos)

    def check_try_stmt(self, stmt: TTryStmt) -> None:
        self.enter_scope()
        self.check_stmts(stmt.body)
        self.exit_scope()
        seen_obj = False
        for catch in stmt.catches:
            self.enter_scope()
            if len(catch.types) == 0:
                catch_type = OBJ_T
            elif len(catch.types) == 1:
                catch_type = self.resolve_type(catch.types[0])
            else:
                members: list[Type] = []
                for ct in catch.types:
                    members.append(self.resolve_type(ct))
                catch_type = normalize_union(members)
            if seen_obj:
                self.error("unreachable catch after catch-all", catch.pos)
            if type_eq(catch_type, OBJ_T) and not isinstance(
                catch_type, (StructT, InterfaceT, EnumT)
            ):
                seen_obj = True
            self.declare(catch.name, catch_type, catch.pos)
            self.check_stmts(catch.body)
            self.exit_scope()
        if stmt.finally_body is not None:
            self.enter_scope()
            self.check_stmts(stmt.finally_body)
            self.exit_scope()

    # ── Expression checking ───────────────────────────────────

    def check_expr(self, expr: TExpr, expected: Type | None) -> Type | None:
        """Type-check an expression and return its type. Returns None on error."""
        if isinstance(expr, TIntLit):
            return INT_T
        if isinstance(expr, TFloatLit):
            return FLOAT_T
        if isinstance(expr, TBoolLit):
            return BOOL_T
        if isinstance(expr, TByteLit):
            if expected is not None and type_eq(expected, INT_T):
                return INT_T
            return BYTE_T
        if isinstance(expr, TStringLit):
            return STRING_T
        if isinstance(expr, TRuneLit):
            return RUNE_T
        if isinstance(expr, TBytesLit):
            return BYTES_T
        if isinstance(expr, TNilLit):
            return NIL_T
        if isinstance(expr, TVar):
            return self.check_var(expr)
        if isinstance(expr, TBinaryOp):
            return self.check_binary_op(expr)
        if isinstance(expr, TUnaryOp):
            return self.check_unary_op(expr)
        if isinstance(expr, TTernary):
            return self.check_ternary(expr, expected)
        if isinstance(expr, TFieldAccess):
            return self.check_field_access(expr)
        if isinstance(expr, TTupleAccess):
            return self.check_tuple_access(expr)
        if isinstance(expr, TIndex):
            return self.check_index(expr)
        if isinstance(expr, TSlice):
            return self.check_slice(expr)
        if isinstance(expr, TCall):
            return self.check_call(expr, expected)
        if isinstance(expr, TListLit):
            return self.check_list_lit(expr, expected)
        if isinstance(expr, TMapLit):
            return self.check_map_lit(expr, expected)
        if isinstance(expr, TSetLit):
            return self.check_set_lit(expr, expected)
        if isinstance(expr, TTupleLit):
            return self.check_tuple_lit(expr, expected)
        if isinstance(expr, TFnLit):
            return self.check_fn_lit(expr, expected)
        self.error("unhandled expression type", expr.pos)
        return None

    def check_var(self, expr: TVar) -> Type | None:
        return self.lookup(expr.name, expr.pos)

    def check_binary_op(self, expr: TBinaryOp) -> Type | None:
        left = self.check_expr(expr.left, None)
        right = self.check_expr(expr.right, None)
        if left is None or right is None:
            return None
        return self.check_binary_op_types(expr.op, left, right, expr.pos)

    def check_binary_op_types(
        self, op: str, left: Type, right: Type, pos: Pos
    ) -> Type | None:
        # Logical: both bool
        if op in ("&&", "||"):
            if not type_eq(left, BOOL_T):
                self.error(
                    "left operand of " + op + " must be bool, got " + type_name(left),
                    pos,
                )
                return None
            if not type_eq(right, BOOL_T):
                self.error(
                    "right operand of " + op + " must be bool, got " + type_name(right),
                    pos,
                )
                return None
            return BOOL_T
        # Equality: same type
        if op in ("==", "!="):
            if not type_eq(left, right) and not (
                is_assignable(left, right) or is_assignable(right, left)
            ):
                self.error(
                    "cannot compare " + type_name(left) + " and " + type_name(right),
                    pos,
                )
                return None
            return BOOL_T
        # Ordering: int, float, byte, rune, string — same type
        if op in ("<", "<=", ">", ">="):
            if not type_eq(left, right):
                self.error(
                    "cannot compare " + type_name(left) + " and " + type_name(right),
                    pos,
                )
                return None
            if left.kind not in (TY_INT, TY_FLOAT, TY_BYTE, TY_RUNE, TY_STRING):
                self.error("ordering not defined for " + type_name(left), pos)
                return None
            return BOOL_T
        # Arithmetic: +, -, *, /, % — int, float, byte; same type
        if op in ("+", "-", "*", "/", "%"):
            if not type_eq(left, right):
                self.error(
                    "operands of "
                    + op
                    + " must be same type, got "
                    + type_name(left)
                    + " and "
                    + type_name(right),
                    pos,
                )
                return None
            if left.kind not in (TY_INT, TY_FLOAT, TY_BYTE):
                self.error(op + " not defined for " + type_name(left), pos)
                return None
            return left
        # Bitwise: &, |, ^ — int, byte; same type
        if op in ("&", "|", "^"):
            if not type_eq(left, right):
                self.error(
                    "operands of "
                    + op
                    + " must be same type, got "
                    + type_name(left)
                    + " and "
                    + type_name(right),
                    pos,
                )
                return None
            if left.kind not in (TY_INT, TY_BYTE):
                self.error(op + " not defined for " + type_name(left), pos)
                return None
            return left
        # Shifts: <<, >> — left is int/byte, right is int
        if op in ("<<", ">>"):
            if left.kind not in (TY_INT, TY_BYTE):
                self.error(
                    "left operand of "
                    + op
                    + " must be int or byte, got "
                    + type_name(left),
                    pos,
                )
                return None
            if not type_eq(right, INT_T):
                self.error(
                    "right operand of " + op + " must be int, got " + type_name(right),
                    pos,
                )
                return None
            return left
        self.error("unknown binary operator: " + op, pos)
        return None

    def check_unary_op(self, expr: TUnaryOp) -> Type | None:
        operand = self.check_expr(expr.operand, None)
        if operand is None:
            return None
        if expr.op == "-":
            if operand.kind not in (TY_INT, TY_FLOAT, TY_BYTE):
                self.error("negation not defined for " + type_name(operand), expr.pos)
                return None
            return operand
        if expr.op == "!":
            if not type_eq(operand, BOOL_T):
                self.error(
                    "logical not requires bool, got " + type_name(operand), expr.pos
                )
                return None
            return BOOL_T
        if expr.op == "~":
            if operand.kind not in (TY_INT, TY_BYTE):
                self.error(
                    "bitwise complement not defined for " + type_name(operand), expr.pos
                )
                return None
            return operand
        self.error("unknown unary operator: " + expr.op, expr.pos)
        return None

    def check_ternary(self, expr: TTernary, expected: Type | None) -> Type | None:
        cond = self.check_expr(expr.cond, BOOL_T)
        if cond is not None and not type_eq(cond, BOOL_T):
            self.error(
                "ternary condition must be bool, got " + type_name(cond), expr.pos
            )
        then_type = self.check_expr(expr.then_expr, expected)
        else_type = self.check_expr(expr.else_expr, expected)
        if then_type is None or else_type is None:
            return then_type if then_type is not None else else_type
        if not type_eq(then_type, else_type):
            # Allow if one is assignable to the other
            if is_assignable(then_type, else_type):
                return else_type
            if is_assignable(else_type, then_type):
                return then_type
            self.error(
                "ternary branches must have same type, got "
                + type_name(then_type)
                + " and "
                + type_name(else_type),
                expr.pos,
            )
            return None
        return then_type

    def check_field_access(self, expr: TFieldAccess) -> Type | None:
        # Special case: Enum.Variant — TVar.field
        if isinstance(expr.obj, TVar):
            resolved = self.lookup(expr.obj.name, expr.obj.pos)
            if resolved is not None and isinstance(resolved, EnumT):
                if expr.field not in resolved.variants:
                    self.error(
                        "'" + expr.field + "' is not a variant of " + resolved.name,
                        expr.pos,
                    )
                    return None
                return resolved
        obj_type = self.check_expr(expr.obj, None)
        if obj_type is None:
            return None
        if isinstance(obj_type, StructT):
            if expr.field in obj_type.fields:
                return obj_type.fields[expr.field]
            if expr.field in obj_type.methods:
                self.error("cannot capture 'self'", expr.pos)
                return obj_type.methods[expr.field]
            self.error(
                "'" + obj_type.name + "' has no field or method '" + expr.field + "'",
                expr.pos,
            )
            return None
        if isinstance(obj_type, UnionT):
            field_type = self._union_field_type(obj_type, expr.field)
            if field_type is not None:
                return field_type
        self.error(
            "cannot access field '" + expr.field + "' on " + type_name(obj_type),
            expr.pos,
        )
        return None

    def _union_field_type(self, union: UnionT, field: str) -> Type | None:
        """If all members are structs with the same-typed field, return it.

        Unions containing nil (optionals) are excluded — field access
        requires narrowing first.
        """
        for m in union.members:
            if type_eq(m, NIL_T):
                return None
        result: Type | None = None
        for m in union.members:
            if not isinstance(m, StructT):
                return None
            if field not in m.fields:
                return None
            ft = m.fields[field]
            if result is None:
                result = ft
            elif not type_eq(result, ft):
                return None
        return result

    def check_tuple_access(self, expr: TTupleAccess) -> Type | None:
        obj_type = self.check_expr(expr.obj, None)
        if obj_type is None:
            return None
        if not isinstance(obj_type, TupleT):
            self.error(
                "tuple access on non-tuple type " + type_name(obj_type), expr.pos
            )
            return None
        if expr.index < 0 or expr.index >= len(obj_type.elements):
            self.error(
                "tuple index "
                + str(expr.index)
                + " out of range for "
                + type_name(obj_type),
                expr.pos,
            )
            return None
        return obj_type.elements[expr.index]

    def check_index(self, expr: TIndex) -> Type | None:
        obj_type = self.check_expr(expr.obj, None)
        if obj_type is None:
            return None
        idx_type = self.check_expr(expr.index, None)
        if isinstance(obj_type, ListT):
            if idx_type is not None and not type_eq(idx_type, INT_T):
                self.error(
                    "list index must be int, got " + type_name(idx_type), expr.pos
                )
            return obj_type.element
        if type_eq(obj_type, STRING_T):
            if idx_type is not None and not type_eq(idx_type, INT_T):
                self.error(
                    "string index must be int, got " + type_name(idx_type), expr.pos
                )
            return RUNE_T
        if type_eq(obj_type, BYTES_T):
            if idx_type is not None and not type_eq(idx_type, INT_T):
                self.error(
                    "bytes index must be int, got " + type_name(idx_type), expr.pos
                )
            return BYTE_T
        if isinstance(obj_type, MapT):
            if idx_type is not None and not type_eq(idx_type, obj_type.key):
                self.error(
                    "map key must be "
                    + type_name(obj_type.key)
                    + ", got "
                    + type_name(idx_type),
                    expr.pos,
                )
            return obj_type.value
        self.error("cannot index " + type_name(obj_type), expr.pos)
        return None

    def check_slice(self, expr: TSlice) -> Type | None:
        obj_type = self.check_expr(expr.obj, None)
        if obj_type is None:
            return None
        low_type = self.check_expr(expr.low, INT_T)
        high_type = self.check_expr(expr.high, INT_T)
        if low_type is not None and not type_eq(low_type, INT_T):
            self.error("slice bound must be int, got " + type_name(low_type), expr.pos)
        if high_type is not None and not type_eq(high_type, INT_T):
            self.error("slice bound must be int, got " + type_name(high_type), expr.pos)
        if isinstance(obj_type, ListT):
            return obj_type
        if type_eq(obj_type, STRING_T):
            return STRING_T
        if type_eq(obj_type, BYTES_T):
            return BYTES_T
        self.error("cannot slice " + type_name(obj_type), expr.pos)
        return None

    def check_call(self, expr: TCall, expected: Type | None) -> Type | None:
        # Built-in function call: TCall with TVar func
        # User-defined functions shadow builtins
        if (
            isinstance(expr.func, TVar)
            and expr.func.name in BUILTIN_NAMES
            and expr.func.name not in self.functions
        ):
            return self.check_builtin_call(
                expr.func.name, expr.args, expr.pos, expected
            )
        # Struct constructor: TCall with TVar func resolving to a struct type
        if isinstance(expr.func, TVar):
            resolved = self.lookup(expr.func.name, expr.func.pos)
            if resolved is not None and isinstance(resolved, StructT):
                return self.check_struct_constructor(resolved, expr.args, expr.pos)
            if resolved is not None and isinstance(resolved, FnT):
                pnames = self.fn_param_names.get(expr.func.name)
                return self.check_fn_call(resolved, expr.args, expr.pos, pnames)
            if resolved is not None:
                self.error("'" + expr.func.name + "' is not callable", expr.pos)
                return None
            return None
        # Method call: TCall with TFieldAccess func
        if isinstance(expr.func, TFieldAccess):
            return self.check_method_call(expr.func, expr.args, expr.pos)
        # Function value call
        func_type = self.check_expr(expr.func, None)
        if func_type is None:
            return None
        if isinstance(func_type, FnT):
            return self.check_fn_call(func_type, expr.args, expr.pos)
        self.error("cannot call " + type_name(func_type), expr.pos)
        return None

    def check_fn_call(
        self,
        fn: FnT,
        args: list[TArg],
        pos: Pos,
        param_names: list[str] | None = None,
    ) -> Type | None:
        if len(args) != len(fn.params):
            self.error(
                "expected " + str(len(fn.params)) + " arguments, got " + str(len(args)),
                pos,
            )
            return fn.ret
        # Check for named args
        has_named = len(args) > 0 and args[0].name is not None
        if has_named and param_names is not None:
            # Validate no mixing
            for a in args:
                if a.name is None:
                    self.error("cannot mix positional and named arguments", a.pos)
                    return fn.ret
            # Resolve named args to positional
            for a in args:
                assert a.name is not None
                if a.name not in param_names:
                    self.error("no parameter '" + a.name + "'", a.pos)
                    continue
                idx = param_names.index(a.name)
                arg_type = self.check_expr(a.value, fn.params[idx])
                if arg_type is not None and not is_assignable(arg_type, fn.params[idx]):
                    self.error(
                        "argument '"
                        + a.name
                        + "': cannot pass "
                        + type_name(arg_type)
                        + " as "
                        + type_name(fn.params[idx]),
                        a.pos,
                    )
        elif has_named:
            # Named args but no param names available — check for mixing
            for a in args:
                if a.name is None:
                    self.error("cannot mix positional and named arguments", a.pos)
                    return fn.ret
            # Just check types positionally
            i = 0
            while i < len(args):
                arg_type = self.check_expr(args[i].value, fn.params[i])
                if arg_type is not None and not is_assignable(arg_type, fn.params[i]):
                    self.error(
                        "argument "
                        + str(i + 1)
                        + ": cannot pass "
                        + type_name(arg_type)
                        + " as "
                        + type_name(fn.params[i]),
                        args[i].pos,
                    )
                i += 1
        else:
            # Check for positional args mixed with named
            i = 0
            while i < len(args):
                if args[i].name is not None:
                    self.error("cannot mix positional and named arguments", args[i].pos)
                    return fn.ret
                arg_type = self.check_expr(args[i].value, fn.params[i])
                if arg_type is not None and not is_assignable(arg_type, fn.params[i]):
                    self.error(
                        "argument "
                        + str(i + 1)
                        + ": cannot pass "
                        + type_name(arg_type)
                        + " as "
                        + type_name(fn.params[i]),
                        args[i].pos,
                    )
                i += 1
        return fn.ret

    def check_struct_constructor(
        self, st: StructT, args: list[TArg], pos: Pos
    ) -> Type | None:
        field_names = list(st.fields.keys())
        if len(args) == 0 and len(field_names) == 0:
            return st
        if len(args) != len(field_names):
            self.error(
                st.name
                + " has "
                + str(len(field_names))
                + " fields, got "
                + str(len(args))
                + " arguments",
                pos,
            )
            return st
        # Check if named or positional
        if args[0].name is not None:
            # Named construction — check for duplicate field names
            seen_names: set[str] = set()
            for arg in args:
                if arg.name is None:
                    self.error("cannot mix positional and named arguments", arg.pos)
                    return st
                if arg.name in seen_names:
                    self.error("duplicate field '" + arg.name + "'", arg.pos)
                    continue
                seen_names.add(arg.name)
                if arg.name not in st.fields:
                    self.error(
                        "'" + st.name + "' has no field '" + arg.name + "'", arg.pos
                    )
                    continue
                expected_type = st.fields[arg.name]
                arg_type = self.check_expr(arg.value, expected_type)
                if arg_type is not None and not is_assignable(arg_type, expected_type):
                    self.error(
                        "field '"
                        + arg.name
                        + "': cannot assign "
                        + type_name(arg_type)
                        + " to "
                        + type_name(expected_type),
                        arg.pos,
                    )
        else:
            # Positional construction
            i = 0
            while i < len(args):
                if args[i].name is not None:
                    self.error("cannot mix positional and named arguments", args[i].pos)
                    return st
                expected_type2 = st.fields[field_names[i]]
                arg_type2 = self.check_expr(args[i].value, expected_type2)
                if arg_type2 is not None and not is_assignable(
                    arg_type2, expected_type2
                ):
                    self.error(
                        "field '"
                        + field_names[i]
                        + "': cannot assign "
                        + type_name(arg_type2)
                        + " to "
                        + type_name(expected_type2),
                        args[i].pos,
                    )
                i += 1
        return st

    def check_method_call(
        self, access: TFieldAccess, args: list[TArg], pos: Pos
    ) -> Type | None:
        # Check for enum access used as call (shouldn't happen, but guard)
        if isinstance(access.obj, TVar):
            resolved = self.lookup(access.obj.name, access.obj.pos)
            if resolved is not None and isinstance(resolved, EnumT):
                self.error("enum variant is not callable", pos)
                return None
        obj_type = self.check_expr(access.obj, None)
        if obj_type is None:
            return None
        if isinstance(obj_type, StructT):
            if access.field in obj_type.methods:
                method = obj_type.methods[access.field]
                return self.check_fn_call(method, args, pos)
            if access.field in obj_type.fields:
                field_type = obj_type.fields[access.field]
                if isinstance(field_type, FnT):
                    return self.check_fn_call(field_type, args, pos)
                self.error(
                    "'" + access.field + "' is not a method of " + obj_type.name, pos
                )
                return None
            self.error(
                "'" + obj_type.name + "' has no method '" + access.field + "'", pos
            )
            return None
        self.error("cannot call method on " + type_name(obj_type), pos)
        return None

    def check_list_lit(self, expr: TListLit, expected: Type | None) -> Type | None:
        if len(expr.elements) == 0:
            if expected is not None and isinstance(expected, ListT):
                return expected
            self.error("cannot infer type of empty list literal", expr.pos)
            return None
        # Use expected element type if available
        elem_expected: Type | None = None
        if expected is not None and isinstance(expected, ListT):
            elem_expected = expected.element
        first = self.check_expr(expr.elements[0], elem_expected)
        if first is None:
            return None
        check_type = elem_expected if elem_expected is not None else first
        if not is_assignable(first, check_type):
            self.error(
                "list element: cannot assign "
                + type_name(first)
                + " to "
                + type_name(check_type),
                expr.elements[0].pos,
            )
        i = 1
        while i < len(expr.elements):
            elem = self.check_expr(expr.elements[i], check_type)
            if elem is not None and not is_assignable(elem, check_type):
                self.error(
                    "list elements must have same type, got "
                    + type_name(check_type)
                    + " and "
                    + type_name(elem),
                    expr.elements[i].pos,
                )
            i += 1
        return ListT(kind="list", element=check_type)

    def check_map_lit(self, expr: TMapLit, expected: Type | None) -> Type | None:
        if len(expr.entries) == 0:
            if expected is not None and isinstance(expected, MapT):
                return expected
            self.error("cannot infer type of empty map literal", expr.pos)
            return None
        key_expected: Type | None = None
        val_expected: Type | None = None
        if expected is not None and isinstance(expected, MapT):
            key_expected = expected.key
            val_expected = expected.value
        k0, v0 = expr.entries[0]
        key_type = self.check_expr(k0, key_expected)
        val_type = self.check_expr(v0, val_expected)
        if key_type is None or val_type is None:
            return None
        check_key = key_expected if key_expected is not None else key_type
        check_val = val_expected if val_expected is not None else val_type
        i = 1
        while i < len(expr.entries):
            ki, vi = expr.entries[i]
            kt = self.check_expr(ki, check_key)
            vt = self.check_expr(vi, check_val)
            if kt is not None and not is_assignable(kt, check_key):
                self.error("map keys must have same type", ki.pos)
            if vt is not None and not is_assignable(vt, check_val):
                self.error("map values must have same type", vi.pos)
            i += 1
        return MapT(kind="map", key=check_key, value=check_val)

    def check_set_lit(self, expr: TSetLit, expected: Type | None) -> Type | None:
        if len(expr.elements) == 0:
            if expected is not None and isinstance(expected, SetT):
                return expected
            self.error("cannot infer type of empty set literal", expr.pos)
            return None
        first = self.check_expr(expr.elements[0], None)
        if first is None:
            return None
        i = 1
        while i < len(expr.elements):
            elem = self.check_expr(expr.elements[i], first)
            if elem is not None and not type_eq(elem, first):
                self.error("set elements must have same type", expr.elements[i].pos)
            i += 1
        return SetT(kind="set", element=first)

    def check_tuple_lit(self, expr: TTupleLit, expected: Type | None) -> Type | None:
        elem_types: list[Type] = []
        i = 0
        while i < len(expr.elements):
            exp_elem: Type | None = None
            if (
                expected is not None
                and isinstance(expected, TupleT)
                and i < len(expected.elements)
            ):
                exp_elem = expected.elements[i]
            et = self.check_expr(expr.elements[i], exp_elem)
            if et is None:
                return None
            elem_types.append(et)
            i += 1
        return TupleT(kind="tuple", elements=elem_types)

    def check_fn_lit(self, expr: TFnLit, expected: Type | None) -> Type | None:
        params: list[Type] = []
        for p in expr.params:
            if p.typ is not None:
                params.append(self.resolve_type(p.typ))
            else:
                self.error("fn literal parameter must have a type", p.pos)
                params.append(OBJ_T)
        ret = self.resolve_type(expr.ret)
        # Check for captured variables (no closures allowed)
        param_names: set[str] = set()
        for p in expr.params:
            param_names.add(p.name)
        # Check body
        old_ret = self.current_fn_ret
        self.current_fn_ret = ret
        self.enter_scope()
        for p in expr.params:
            if p.typ is not None:
                pt = self.resolve_type(p.typ)
                self.declare(p.name, pt, p.pos)
        if isinstance(expr.body, list):
            # Check for captures before checking body
            self.check_closure_captures(expr.body, param_names, expr.pos)
            self.check_stmts(expr.body)
        else:
            self.check_closure_captures_expr(expr.body, param_names, expr.pos)
            self.check_expr(expr.body, ret)
        self.exit_scope()
        self.current_fn_ret = old_ret
        return FnT(kind="fn", params=params, ret=ret)

    def check_closure_captures(
        self, stmts: list[TStmt], param_names: set[str], pos: Pos
    ) -> None:
        """Check that fn literal body doesn't capture variables from enclosing scope."""
        for s in stmts:
            self._scan_stmt_for_captures(s, param_names, pos)

    def check_closure_captures_expr(
        self, expr: TExpr, param_names: set[str], pos: Pos
    ) -> None:
        self._scan_expr_for_captures(expr, param_names, pos)

    def _scan_expr_for_captures(
        self, expr: TExpr, param_names: set[str], pos: Pos
    ) -> None:
        if isinstance(expr, TVar):
            name = expr.name
            if name in param_names:
                return
            if name in self.functions:
                return
            if name in self.types:
                return
            if name in BUILTIN_NAMES:
                return
            # This is a capture
            self.error("cannot capture '" + name + "' in fn literal", expr.pos)
            return
        if isinstance(expr, TBinaryOp):
            self._scan_expr_for_captures(expr.left, param_names, pos)
            self._scan_expr_for_captures(expr.right, param_names, pos)
        elif isinstance(expr, TUnaryOp):
            self._scan_expr_for_captures(expr.operand, param_names, pos)
        elif isinstance(expr, TTernary):
            self._scan_expr_for_captures(expr.cond, param_names, pos)
            self._scan_expr_for_captures(expr.then_expr, param_names, pos)
            self._scan_expr_for_captures(expr.else_expr, param_names, pos)
        elif isinstance(expr, TFieldAccess):
            self._scan_expr_for_captures(expr.obj, param_names, pos)
        elif isinstance(expr, TTupleAccess):
            self._scan_expr_for_captures(expr.obj, param_names, pos)
        elif isinstance(expr, TIndex):
            self._scan_expr_for_captures(expr.obj, param_names, pos)
            self._scan_expr_for_captures(expr.index, param_names, pos)
        elif isinstance(expr, TSlice):
            self._scan_expr_for_captures(expr.obj, param_names, pos)
            self._scan_expr_for_captures(expr.low, param_names, pos)
            self._scan_expr_for_captures(expr.high, param_names, pos)
        elif isinstance(expr, TCall):
            self._scan_expr_for_captures(expr.func, param_names, pos)
            for a in expr.args:
                self._scan_expr_for_captures(a.value, param_names, pos)
        elif isinstance(expr, TListLit):
            for e in expr.elements:
                self._scan_expr_for_captures(e, param_names, pos)
        elif isinstance(expr, TMapLit):
            for k, v in expr.entries:
                self._scan_expr_for_captures(k, param_names, pos)
                self._scan_expr_for_captures(v, param_names, pos)
        elif isinstance(expr, TSetLit):
            for e in expr.elements:
                self._scan_expr_for_captures(e, param_names, pos)
        elif isinstance(expr, TTupleLit):
            for e in expr.elements:
                self._scan_expr_for_captures(e, param_names, pos)
        elif isinstance(expr, TFnLit):
            # Nested fn lits — the inner one can reference the outer's params
            inner_params: set[str] = set(param_names)
            for p in expr.params:
                inner_params.add(p.name)
            if isinstance(expr.body, list):
                for s in expr.body:
                    self._scan_stmt_for_captures(s, inner_params, pos)
            else:
                self._scan_expr_for_captures(expr.body, inner_params, pos)

    def _scan_stmt_for_captures(
        self, stmt: TStmt, param_names: set[str], pos: Pos
    ) -> None:
        if isinstance(stmt, TLetStmt):
            if stmt.value is not None:
                self._scan_expr_for_captures(stmt.value, param_names, pos)
            # The declared name becomes a local, not a capture
            param_names = set(param_names)
            param_names.add(stmt.name)
        elif isinstance(stmt, TAssignStmt):
            self._scan_expr_for_captures(stmt.target, param_names, pos)
            self._scan_expr_for_captures(stmt.value, param_names, pos)
        elif isinstance(stmt, TOpAssignStmt):
            self._scan_expr_for_captures(stmt.target, param_names, pos)
            self._scan_expr_for_captures(stmt.value, param_names, pos)
        elif isinstance(stmt, TTupleAssignStmt):
            for t in stmt.targets:
                self._scan_expr_for_captures(t, param_names, pos)
            self._scan_expr_for_captures(stmt.value, param_names, pos)
        elif isinstance(stmt, TReturnStmt):
            if stmt.value is not None:
                self._scan_expr_for_captures(stmt.value, param_names, pos)
        elif isinstance(stmt, TThrowStmt):
            self._scan_expr_for_captures(stmt.expr, param_names, pos)
        elif isinstance(stmt, TExprStmt):
            self._scan_expr_for_captures(stmt.expr, param_names, pos)
        elif isinstance(stmt, TIfStmt):
            self._scan_expr_for_captures(stmt.cond, param_names, pos)
            for s in stmt.then_body:
                self._scan_stmt_for_captures(s, param_names, pos)
            if stmt.else_body is not None:
                for s in stmt.else_body:
                    self._scan_stmt_for_captures(s, param_names, pos)
        elif isinstance(stmt, TWhileStmt):
            self._scan_expr_for_captures(stmt.cond, param_names, pos)
            for s in stmt.body:
                self._scan_stmt_for_captures(s, param_names, pos)
        elif isinstance(stmt, TForStmt):
            param_names = set(param_names)
            for b in stmt.binding:
                param_names.add(b)
            if isinstance(stmt.iterable, TRange):
                for a in stmt.iterable.args:
                    self._scan_expr_for_captures(a, param_names, pos)
            else:
                self._scan_expr_for_captures(stmt.iterable, param_names, pos)
            for s in stmt.body:
                self._scan_stmt_for_captures(s, param_names, pos)
        elif isinstance(stmt, TMatchStmt):
            self._scan_expr_for_captures(stmt.expr, param_names, pos)
            for case in stmt.cases:
                inner = set(param_names)
                if isinstance(case.pattern, TPatternType):
                    inner.add(case.pattern.name)
                for s in case.body:
                    self._scan_stmt_for_captures(s, inner, pos)
            if stmt.default is not None:
                inner2 = set(param_names)
                if stmt.default.name is not None:
                    inner2.add(stmt.default.name)
                for s in stmt.default.body:
                    self._scan_stmt_for_captures(s, inner2, pos)
        elif isinstance(stmt, TTryStmt):
            for s in stmt.body:
                self._scan_stmt_for_captures(s, param_names, pos)
            for catch in stmt.catches:
                inner3 = set(param_names)
                inner3.add(catch.name)
                for s in catch.body:
                    self._scan_stmt_for_captures(s, inner3, pos)
            if stmt.finally_body is not None:
                for s in stmt.finally_body:
                    self._scan_stmt_for_captures(s, param_names, pos)

    # ── Built-in function checking ────────────────────────────

    def check_builtin_call(
        self, name: str, args: list[TArg], pos: Pos, expected: Type | None
    ) -> Type | None:
        arg_types: list[Type | None] = []
        for a in args:
            arg_types.append(self.check_expr(a.value, None))
        n = len(args)
        ctx = _BuiltinCtx(self, name, arg_types, pos)
        require = ctx.require
        require_range = ctx.require_range
        arg = ctx.arg

        # ── Numeric ──
        if name == "Abs":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and t.kind not in (TY_INT, TY_FLOAT):
                self.error("Abs requires int or float", pos)
            return t
        if name == "Min" or name == "Max":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None and t2 is not None:
                if not type_eq(t1, t2):
                    self.error(
                        name
                        + " requires same type, got "
                        + type_name(t1)
                        + " and "
                        + type_name(t2),
                        pos,
                    )
                if t1.kind not in (TY_INT, TY_FLOAT, TY_BYTE):
                    self.error(name + " requires int, float, or byte", pos)
            return t1
        if name == "Sum":
            if not require(1):
                return None
            t = arg(0)
            if t is not None:
                if isinstance(t, ListT) and t.element.kind in (TY_INT, TY_FLOAT):
                    return t.element
                self.error("Sum requires list[int] or list[float]", pos)
            return None
        if name == "Pow":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None and t2 is not None:
                if not type_eq(t1, t2):
                    self.error("Pow requires same type", pos)
                if t1.kind not in (TY_INT, TY_FLOAT):
                    self.error("Pow requires int or float", pos)
            return t1
        if name == "Round":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, FLOAT_T):
                self.error("Round requires float", pos)
            return INT_T
        if name == "DivMod":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None and not type_eq(t1, INT_T):
                self.error("DivMod requires int", pos)
            if t2 is not None and not type_eq(t2, INT_T):
                self.error("DivMod requires int", pos)
            return TupleT(kind="tuple", elements=[INT_T, INT_T])

        # ── Bytes ──
        if name == "Encode":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, STRING_T):
                self.error("Encode requires string", pos)
            return BYTES_T
        if name == "Decode":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, BYTES_T):
                self.error("Decode requires bytes", pos)
            return STRING_T

        # ── Len ──
        if name == "Len":
            if not require(1):
                return None
            t = arg(0)
            if t is not None:
                if not (
                    isinstance(t, (ListT, MapT, SetT))
                    or t.kind in (TY_STRING, TY_BYTES)
                ):
                    self.error("Len requires string, bytes, list, map, or set", pos)
            return INT_T

        # ── Concat ──
        if name == "Concat":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None and t2 is not None:
                if type_eq(t1, STRING_T) and type_eq(t2, STRING_T):
                    return STRING_T
                if type_eq(t1, BYTES_T) and type_eq(t2, BYTES_T):
                    return BYTES_T
                self.error("Concat requires two strings or two bytes", pos)
            return STRING_T

        # ── Append ──
        if name == "Append":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None:
                if not isinstance(t1, ListT):
                    self.error("Append requires list as first argument", pos)
                elif t2 is not None and not is_assignable(t2, t1.element):
                    self.error(
                        "cannot append " + type_name(t2) + " to " + type_name(t1), pos
                    )
            return VOID_T

        # ── Insert ──
        if name == "Insert":
            if not require(3):
                return None
            t1 = arg(0)
            if t1 is not None and not isinstance(t1, ListT):
                self.error("Insert requires list as first argument", pos)
            t2 = arg(1)
            if t2 is not None and not type_eq(t2, INT_T):
                self.error("Insert index must be int", pos)
            t3 = arg(2)
            if t1 is not None and isinstance(t1, ListT) and t3 is not None:
                if not is_assignable(t3, t1.element):
                    self.error(
                        "cannot pass " + type_name(t3) + " as " + type_name(t1.element),
                        pos,
                    )
            return VOID_T

        # ── Pop ──
        if name == "Pop":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and isinstance(t, ListT):
                return t.element
            if t is not None:
                self.error("Pop requires list", pos)
            return None

        # ── RemoveAt ──
        if name == "RemoveAt":
            if not require(2):
                return None
            t1 = arg(0)
            if t1 is not None and not isinstance(t1, ListT):
                self.error("RemoveAt requires list", pos)
            t2 = arg(1)
            if t2 is not None and not type_eq(t2, INT_T):
                self.error("RemoveAt index must be int", pos)
            return VOID_T

        # ── IndexOf ──
        if name == "IndexOf":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None and not isinstance(t1, ListT):
                self.error("IndexOf requires list", pos)
            elif t1 is not None and isinstance(t1, ListT) and t2 is not None:
                if not is_assignable(t2, t1.element):
                    self.error(
                        "cannot pass " + type_name(t2) + " as " + type_name(t1.element),
                        pos,
                    )
            return INT_T

        # ── Contains ──
        if name == "Contains":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None:
                if isinstance(t1, ListT):
                    if t2 is not None and not is_assignable(t2, t1.element):
                        self.error(
                            "cannot pass "
                            + type_name(t2)
                            + " as "
                            + type_name(t1.element),
                            pos,
                        )
                elif isinstance(t1, SetT):
                    if t2 is not None and not is_assignable(t2, t1.element):
                        self.error(
                            "cannot pass "
                            + type_name(t2)
                            + " as "
                            + type_name(t1.element),
                            pos,
                        )
                elif isinstance(t1, MapT):
                    if t2 is not None and not is_assignable(t2, t1.key):
                        self.error(
                            "cannot pass " + type_name(t2) + " as " + type_name(t1.key),
                            pos,
                        )
                elif type_eq(t1, STRING_T):
                    pass
                else:
                    self.error("Contains requires list, set, map, or string", pos)
            return BOOL_T

        # ── Get ──
        if name == "Get":
            if not require_range(2, 3):
                return None
            t1 = arg(0)
            if t1 is not None and not isinstance(t1, MapT):
                self.error("Get requires map as first argument", pos)
                return None
            if t1 is not None and isinstance(t1, MapT):
                if n == 2:
                    return make_optional(t1.value)
                return t1.value
            return None

        # ── Delete ──
        if name == "Delete":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None and not isinstance(t1, MapT):
                self.error("Delete requires map", pos)
            elif t1 is not None and isinstance(t1, MapT) and t2 is not None:
                if not is_assignable(t2, t1.key):
                    self.error(
                        "cannot pass " + type_name(t2) + " as " + type_name(t1.key),
                        pos,
                    )
            return VOID_T

        # ── Keys / Values / Items ──
        if name == "Keys":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and isinstance(t, MapT):
                return ListT(kind="list", element=t.key)
            if t is not None:
                self.error("Keys requires map", pos)
            return None
        if name == "Values":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and isinstance(t, MapT):
                return ListT(kind="list", element=t.value)
            if t is not None:
                self.error("Values requires map", pos)
            return None
        if name == "Items":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and isinstance(t, MapT):
                return ListT(
                    kind="list", element=TupleT(kind="tuple", elements=[t.key, t.value])
                )
            if t is not None:
                self.error("Items requires map", pos)
            return None

        # ── Merge ──
        if name == "Merge":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None and not isinstance(t1, MapT):
                self.error("Merge requires map", pos)
                return None
            if t2 is not None and not isinstance(t2, MapT):
                self.error("Merge requires map", pos)
                return None
            if (
                t1 is not None
                and t2 is not None
                and isinstance(t1, MapT)
                and isinstance(t2, MapT)
            ):
                if not type_eq(t1, t2):
                    self.error(
                        "Merge maps must be same type, got "
                        + type_name(t1)
                        + " and "
                        + type_name(t2),
                        pos,
                    )
                return t1
            if t1 is not None and isinstance(t1, MapT):
                return t1
            return None

        # ── Map() / Set() ──
        if name == "Map":
            if not require(0):
                return None
            if expected is not None and isinstance(expected, MapT):
                return expected
            self.error("cannot infer type of Map()", pos)
            return None
        if name == "Set":
            if not require(0):
                return None
            if expected is not None and isinstance(expected, SetT):
                return expected
            self.error("cannot infer type of Set()", pos)
            return None

        # ── Add / Remove (set) ──
        if name == "Add":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None and not isinstance(t1, SetT):
                self.error("Add requires set as first argument", pos)
            elif t1 is not None and isinstance(t1, SetT) and t2 is not None:
                if not is_assignable(t2, t1.element):
                    self.error(
                        "cannot pass " + type_name(t2) + " as " + type_name(t1.element),
                        pos,
                    )
            return VOID_T
        if name == "Remove":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None and not isinstance(t1, SetT):
                self.error("Remove requires set as first argument", pos)
            elif t1 is not None and isinstance(t1, SetT) and t2 is not None:
                if not is_assignable(t2, t1.element):
                    self.error(
                        "cannot pass " + type_name(t2) + " as " + type_name(t1.element),
                        pos,
                    )
            return VOID_T

        # ── Repeat ──
        if name == "Repeat":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t2 is not None and not type_eq(t2, INT_T):
                self.error("Repeat count must be int", pos)
            if t1 is not None:
                if type_eq(t1, STRING_T):
                    return STRING_T
                if isinstance(t1, ListT):
                    return t1
                self.error("Repeat requires string or list", pos)
            return None

        # ── Reversed / Sorted ──
        if name == "Reversed":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and isinstance(t, ListT):
                return t
            if t is not None:
                self.error("Reversed requires list", pos)
            return None
        if name == "Sorted":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and isinstance(t, ListT):
                if t.element.kind not in (
                    TY_INT,
                    TY_FLOAT,
                    TY_BYTE,
                    TY_RUNE,
                    TY_STRING,
                ):
                    self.error("Sorted requires ordered type", pos)
                return t
            if t is not None:
                self.error("Sorted requires list", pos)
            return None

        # ── String functions ──
        if name in ("Upper", "Lower"):
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, STRING_T):
                self.error(name + " requires string", pos)
            return STRING_T
        if name in ("Trim", "TrimStart", "TrimEnd"):
            if not require(2):
                return None
            return STRING_T
        if name in ("Split", "SplitWhitespace"):
            if name == "Split":
                if not require(2):
                    return None
            else:
                if not require(1):
                    return None
            return ListT(kind="list", element=STRING_T)
        if name == "SplitN":
            if not require(3):
                return None
            return ListT(kind="list", element=STRING_T)
        if name == "Join":
            if not require(2):
                return None
            return STRING_T
        if name in ("Find", "RFind", "Count"):
            if not require(2):
                return None
            return INT_T
        if name == "Replace":
            if not require(3):
                return None
            return STRING_T
        if name in ("StartsWith", "EndsWith"):
            if not require(2):
                return None
            return BOOL_T
        if name in ("IsDigit", "IsAlpha", "IsAlnum", "IsSpace", "IsUpper", "IsLower"):
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not (type_eq(t, STRING_T) or type_eq(t, RUNE_T)):
                self.error(name + " requires string or rune", pos)
            return BOOL_T

        # ── RuneFromInt / RuneToInt ──
        if name == "RuneFromInt":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, INT_T):
                self.error("RuneFromInt requires int", pos)
            return RUNE_T
        if name == "RuneToInt":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, RUNE_T):
                self.error("RuneToInt requires rune", pos)
            return INT_T

        # ── ParseInt / ParseFloat ──
        if name == "ParseInt":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None and not type_eq(t1, STRING_T):
                self.error("ParseInt requires string as first argument", pos)
            if t2 is not None and not type_eq(t2, INT_T):
                self.error(
                    "cannot pass " + type_name(t2) + " as int",
                    pos,
                )
            return INT_T
        if name == "ParseFloat":
            if not require(1):
                return None
            return FLOAT_T
        if name == "FormatInt":
            if not require(2):
                return None
            t1 = arg(0)
            t2 = arg(1)
            if t1 is not None and not type_eq(t1, INT_T):
                self.error("FormatInt requires int as first argument", pos)
            if t2 is not None and not type_eq(t2, INT_T):
                self.error("FormatInt requires int as second argument", pos)
            return STRING_T

        # ── Conversions ──
        if name == "IntToFloat":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, INT_T):
                self.error("IntToFloat requires int", pos)
            return FLOAT_T
        if name == "FloatToInt":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, FLOAT_T):
                self.error("FloatToInt requires float", pos)
            return INT_T
        if name == "ByteToInt":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, BYTE_T):
                self.error("ByteToInt requires byte", pos)
            return INT_T
        if name == "IntToByte":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, INT_T):
                self.error("IntToByte requires int", pos)
            return BYTE_T
        if name == "ToString":
            if not require(1):
                return None
            return STRING_T

        # ── Format ──
        if name == "Format":
            if n < 1:
                self.error("Format requires at least 1 argument", pos)
                return None
            t = arg(0)
            if t is not None and not type_eq(t, STRING_T):
                self.error("Format template must be string", pos)
            # Check remaining args are all string
            i = 1
            while i < n:
                at = arg(i)
                if at is not None and not type_eq(at, STRING_T):
                    self.error("Format arguments must be string", args[i].pos)
                i += 1
            # Check placeholder count matches arg count
            if isinstance(args[0].value, TStringLit):
                placeholders = args[0].value.value.count("{}")
                arg_count = n - 1
                if placeholders != arg_count:
                    self.error(
                        "Format placeholder count mismatch: "
                        + str(placeholders)
                        + " placeholders, "
                        + str(arg_count)
                        + " arguments",
                        pos,
                    )
            return STRING_T

        # ── I/O ──
        if name in ("WriteOut", "WriteErr", "WritelnOut", "WritelnErr"):
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not (type_eq(t, STRING_T) or type_eq(t, BYTES_T)):
                self.error(name + " requires string or bytes", pos)
            return VOID_T
        if name == "ReadLine":
            if not require(0):
                return None
            return make_optional(STRING_T)
        if name == "ReadAll":
            if not require(0):
                return None
            return STRING_T
        if name == "ReadBytes":
            if not require(0):
                return None
            return BYTES_T
        if name == "ReadBytesN":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, INT_T):
                self.error("ReadBytesN requires int", pos)
            return BYTES_T
        if name == "Args":
            if not require(0):
                return None
            return ListT(kind="list", element=STRING_T)
        if name == "GetEnv":
            if not require(1):
                return None
            return make_optional(STRING_T)
        if name == "Exit":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, INT_T):
                self.error("Exit requires int", pos)
            return VOID_T

        # ── Assert ──
        if name == "Assert":
            if not require_range(1, 2):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, BOOL_T):
                self.error("Assert condition must be bool", pos)
            if n == 2:
                t2 = arg(1)
                if t2 is not None and not type_eq(t2, STRING_T):
                    self.error("Assert message must be string", pos)
            return VOID_T

        # ── Unwrap ──
        if name == "Unwrap":
            if not require(1):
                return None
            t = arg(0)
            if t is not None:
                if contains_nil(t):
                    return remove_nil(t)
                self.error("Unwrap requires optional type", pos)
            return None

        # ── Math extras ──
        if name == "IsNaN" or name == "IsInf":
            if not require(1):
                return None
            t = arg(0)
            if t is not None and not type_eq(t, FLOAT_T):
                self.error(name + " requires float", pos)
            return BOOL_T

        self.error("unknown built-in function: " + name, pos)
        return None


# ============================================================
# PUBLIC API
# ============================================================


def check(module: TModule) -> list[CheckError]:
    """Type-check a parsed TModule. Returns a list of errors (empty = ok)."""
    checker = Checker()
    checker.collect_declarations(module)
    if len(checker.errors) > 0:
        return checker.errors
    checker.check_bodies(module)
    _check_main(checker)
    return checker.errors


def check_with_info(module: TModule) -> tuple[list[CheckError], Checker]:
    """Type-check and return both errors and the Checker (for downstream passes)."""
    checker = Checker()
    checker.collect_declarations(module)
    if len(checker.errors) > 0:
        return (checker.errors, checker)
    checker.check_bodies(module)
    _check_main(checker)
    return (checker.errors, checker)


def _check_main(checker: Checker) -> None:
    if "Main" not in checker.functions:
        checker.error("missing Main", Pos(1, 1))
        return
    main = checker.functions["Main"]
    if len(main.params) > 0:
        checker.error("Main must take no parameters", Pos(1, 1))
    if not type_eq(main.ret, VOID_T):
        checker.error("Main must return void", Pos(1, 1))
