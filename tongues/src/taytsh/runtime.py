"""Taytsh runtime — typecheck and evaluate a Taytsh module.

This is a spec-faithful (as practical) interpreter for the Taytsh textual IR
defined in spec/taytsh.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence, cast

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
    TDecl,
    TEnumDecl,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFieldDecl,
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
# Diagnostics
# ============================================================


class TaytshError(Exception):
    """Base error for Taytsh typechecking/evaluation."""

    def __init__(self, msg: str, pos: Pos | None = None):
        if pos is None:
            super().__init__(msg)
        else:
            super().__init__(f"{msg} at line {pos.line} col {pos.col}")
        self.msg = msg
        self.pos = pos


class TaytshTypeError(TaytshError):
    """Static type error."""


class TaytshRuntimeFault(TaytshError):
    """Runtime fault (uncaught throw, invalid operation, etc.)."""


# ============================================================
# Types
# ============================================================


class Ty:
    """Base type for typechecker/runtime."""

    def display(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class TyPrim(Ty):
    kind: str

    def display(self) -> str:
        return self.kind


@dataclass(frozen=True)
class TyList(Ty):
    element: Ty

    def display(self) -> str:
        return f"list[{self.element.display()}]"


@dataclass(frozen=True)
class TyMap(Ty):
    key: Ty
    value: Ty

    def display(self) -> str:
        return f"map[{self.key.display()}, {self.value.display()}]"


@dataclass(frozen=True)
class TySet(Ty):
    element: Ty

    def display(self) -> str:
        return f"set[{self.element.display()}]"


@dataclass(frozen=True)
class TyTuple(Ty):
    elements: tuple[Ty, ...]

    def display(self) -> str:
        inner = ", ".join(t.display() for t in self.elements)
        return f"({inner})"


@dataclass(frozen=True)
class TyFunc(Ty):
    params: tuple[Ty, ...]
    ret: Ty

    def display(self) -> str:
        inner = ", ".join(t.display() for t in (*self.params, self.ret))
        return f"fn[{inner}]"


@dataclass(frozen=True)
class TyStruct(Ty):
    name: str

    def display(self) -> str:
        return self.name


@dataclass(frozen=True)
class TyInterface(Ty):
    name: str

    def display(self) -> str:
        return self.name


@dataclass(frozen=True)
class TyEnum(Ty):
    name: str

    def display(self) -> str:
        return self.name


@dataclass(frozen=True)
class TyUnion(Ty):
    members: tuple[Ty, ...]

    def display(self) -> str:
        return " | ".join(t.display() for t in self.members)


TY_INT = TyPrim("int")
TY_FLOAT = TyPrim("float")
TY_BOOL = TyPrim("bool")
TY_BYTE = TyPrim("byte")
TY_BYTES = TyPrim("bytes")
TY_STRING = TyPrim("string")
TY_RUNE = TyPrim("rune")
TY_VOID = TyPrim("void")  # return-only marker
TY_OBJ = TyPrim("obj")
TY_NIL = TyPrim("nil")


def _ty_key(t: Ty) -> tuple:
    if isinstance(t, TyPrim):
        return ("prim", t.kind)
    if isinstance(t, TyStruct):
        return ("struct", t.name)
    if isinstance(t, TyInterface):
        return ("iface", t.name)
    if isinstance(t, TyEnum):
        return ("enum", t.name)
    if isinstance(t, TyList):
        return ("list", _ty_key(t.element))
    if isinstance(t, TyMap):
        return ("map", _ty_key(t.key), _ty_key(t.value))
    if isinstance(t, TySet):
        return ("set", _ty_key(t.element))
    if isinstance(t, TyTuple):
        return ("tuple", tuple(_ty_key(e) for e in t.elements))
    if isinstance(t, TyFunc):
        return ("fn", tuple(_ty_key(p) for p in t.params), _ty_key(t.ret))
    if isinstance(t, TyUnion):
        return ("union", tuple(_ty_key(m) for m in t.members))
    return ("unknown", repr(t))


def ty_eq(a: Ty, b: Ty) -> bool:
    return _ty_key(a) == _ty_key(b)


def ty_is_nil(t: Ty) -> bool:
    return isinstance(t, TyPrim) and t.kind == "nil"


def ty_is_obj(t: Ty) -> bool:
    return isinstance(t, TyPrim) and t.kind == "obj"


def ty_union(members: Iterable[Ty]) -> Ty:
    flat: list[Ty] = []
    for m in members:
        if isinstance(m, TyUnion):
            flat.extend(m.members)
        else:
            flat.append(m)
    # absorb obj
    if any(ty_is_obj(m) for m in flat):
        return TY_OBJ
    # dedup
    uniq: dict[tuple, Ty] = {}
    for m in flat:
        uniq[_ty_key(m)] = m
    ordered = tuple(sorted(uniq.values(), key=_ty_key))
    if len(ordered) == 1:
        return ordered[0]
    return TyUnion(ordered)


def ty_without_nil(t: Ty) -> Ty:
    if isinstance(t, TyUnion):
        members = [m for m in t.members if not ty_is_nil(m)]
        return ty_union(members)
    return t


def ty_has_nil(t: Ty) -> bool:
    if ty_is_nil(t) or ty_is_obj(t):
        return True
    if isinstance(t, TyUnion):
        return any(ty_has_nil(m) for m in t.members)
    return False


# ============================================================
# Values
# ============================================================


class Value:
    """A runtime value with a concrete (non-union) type tag."""

    def ty(self) -> Ty:
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError


class HashableValue(Value):
    """A value that can be used as a map key / set element."""

    def __hash__(self) -> int:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


@dataclass
class VNil(Value):
    def ty(self) -> Ty:
        return TY_NIL

    def to_string(self) -> str:
        return "nil"


@dataclass(eq=False)
class VBool(HashableValue):
    value: bool

    def ty(self) -> Ty:
        return TY_BOOL

    def to_string(self) -> str:
        return "true" if self.value else "false"

    def __hash__(self) -> int:
        return hash(("bool", self.value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VBool) and self.value == other.value


@dataclass(eq=False)
class VInt(HashableValue):
    value: int

    def ty(self) -> Ty:
        return TY_INT

    def to_string(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(("int", self.value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VInt) and self.value == other.value


@dataclass(eq=False)
class VFloat(HashableValue):
    value: float

    def ty(self) -> Ty:
        return TY_FLOAT

    def to_string(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        # Include tag so float keys never collide with int keys of same value.
        return hash(("float", self.value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VFloat) and self.value == other.value


@dataclass(eq=False)
class VByte(HashableValue):
    value: int  # 0..255

    def __post_init__(self) -> None:
        self.value &= 0xFF

    def ty(self) -> Ty:
        return TY_BYTE

    def to_string(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(("byte", self.value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VByte) and self.value == other.value


@dataclass(eq=False)
class VBytes(HashableValue):
    value: bytes

    def ty(self) -> Ty:
        return TY_BYTES

    def to_string(self) -> str:
        # Best-effort debug representation.
        return self.value.hex()

    def __hash__(self) -> int:
        return hash(("bytes", self.value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VBytes) and self.value == other.value


@dataclass(eq=False)
class VString(HashableValue):
    value: str

    def ty(self) -> Ty:
        return TY_STRING

    def to_string(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(("string", self.value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VString) and self.value == other.value


@dataclass(eq=False)
class VRune(HashableValue):
    value: str  # len == 1

    def __post_init__(self) -> None:
        if len(self.value) != 1:
            raise ValueError("rune must be length 1")

    def ty(self) -> Ty:
        return TY_RUNE

    def to_string(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(("rune", self.value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VRune) and self.value == other.value


@dataclass(eq=False)
class VTuple(HashableValue):
    elements: tuple[Value, ...]
    typ: TyTuple

    def ty(self) -> Ty:
        return self.typ

    def to_string(self) -> str:
        inner = ", ".join(v.to_string() for v in self.elements)
        return f"({inner})"

    def __hash__(self) -> int:
        elems = []
        for e in self.elements:
            if not isinstance(e, HashableValue):
                raise TypeError("unhashable tuple element")
            elems.append(hash(e))
        return hash(("tuple", _ty_key(self.typ), tuple(elems)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VTuple):
            return False
        if not ty_eq(self.typ, other.typ):
            return False
        if len(self.elements) != len(other.elements):
            return False
        return all(_value_eq(a, b) for a, b in zip(self.elements, other.elements))


@dataclass
class VList(Value):
    elements: list[Value]
    typ: TyList

    def ty(self) -> Ty:
        return self.typ

    def to_string(self) -> str:
        inner = ", ".join(v.to_string() for v in self.elements)
        return f"[{inner}]"


@dataclass(eq=False)
class VEnum(HashableValue):
    enum_name: str
    variant: str

    def ty(self) -> Ty:
        return TyEnum(self.enum_name)

    def to_string(self) -> str:
        return f"{self.enum_name}.{self.variant}"

    def __hash__(self) -> int:
        return hash(("enum", self.enum_name, self.variant))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VEnum):
            return False
        return self.enum_name == other.enum_name and self.variant == other.variant


@dataclass(eq=False)
class VMap(Value):
    # Keys must be hashable Taytsh values.
    entries: dict[HashableValue, Value]
    typ: TyMap

    def ty(self) -> Ty:
        return self.typ

    def to_string(self) -> str:
        parts: list[str] = []
        for k, v in self.entries.items():
            parts.append(f"{k.to_string()}: {v.to_string()}")
        return "{" + ", ".join(parts) + "}"


@dataclass(eq=False)
class VSet(Value):
    elements: set[HashableValue]
    typ: TySet

    def ty(self) -> Ty:
        return self.typ

    def to_string(self) -> str:
        inner = ", ".join(v.to_string() for v in self.elements)
        return "{" + inner + "}"


@dataclass(eq=False)
class VStruct(Value):
    struct_name: str
    fields: dict[str, Value]

    def ty(self) -> Ty:
        return TyStruct(self.struct_name)

    def to_string(self) -> str:
        parts: list[str] = []
        for k, v in self.fields.items():
            parts.append(f"{k}: {v.to_string()}")
        inner = ", ".join(parts)
        return f"{self.struct_name}({inner})"


@dataclass
class VFunc(Value):
    typ: TyFunc
    name: str | None
    call: Callable[[list[Value]], Value]

    def ty(self) -> Ty:
        return self.typ

    def to_string(self) -> str:
        if self.name is None:
            return "<fn>"
        return f"<fn {self.name}>"


# ============================================================
# Control flow signals (internal)
# ============================================================


class _Signal(Exception):
    pass


@dataclass
class _Return(_Signal):
    value: Value | None


class _Break(_Signal):
    pass


class _Continue(_Signal):
    pass


@dataclass
class _Throw(_Signal):
    value: Value


@dataclass
class _Exit(_Signal):
    code: int


# ============================================================
# Runtime I/O
# ============================================================


class _Input:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read_all(self) -> bytes:
        out = self._data[self._pos :]
        self._pos = len(self._data)
        return out

    def read_n(self, n: int) -> bytes:
        if n <= 0:
            return b""
        out = self._data[self._pos : self._pos + n]
        self._pos += len(out)
        return out

    def read_line(self) -> bytes | None:
        if self._pos >= len(self._data):
            return None
        idx = self._data.find(b"\n", self._pos)
        if idx == -1:
            out = self._data[self._pos :]
            self._pos = len(self._data)
            return out
        out = self._data[self._pos : idx + 1]
        self._pos = idx + 1
        return out


@dataclass
class RunResult:
    exit_code: int
    stdout: bytes
    stderr: bytes


# ============================================================
# Placeholder: implementation continues in further steps
# ============================================================


def run(
    module: TModule,
    *,
    stdin: bytes = b"",
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> RunResult:
    """Typecheck and run a parsed Taytsh module."""
    idx = _build_index(module)
    tc = TypeChecker(idx)
    checked = tc.check_module(module)
    rt = Runtime(
        module,
        checked,
        tc,
        stdin=stdin,
        args=list(args) if args is not None else [],
        env=dict(env) if env is not None else {},
    )
    return rt.run_main()


# ============================================================
# Module indexing + typechecking
# ============================================================


_RESERVED_BINDINGS: set[str] = {
    # Built-in functions (spec: reserved names)
    "ToString",
    "Throw",
    "Catch",
    "Len",
    "Concat",
    "Abs",
    "Min",
    "Max",
    "Sum",
    "Pow",
    "Round",
    "DivMod",
    "IsNaN",
    "IsInf",
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
    "Append",
    "Insert",
    "Pop",
    "RemoveAt",
    "IndexOf",
    "Reversed",
    "Sorted",
    "Map",
    "Get",
    "Delete",
    "Keys",
    "Values",
    "Items",
    "Merge",
    "Set",
    "Add",
    "Remove",
    "Unwrap",
    "Encode",
    "Decode",
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
    "Assert",
    "IntToFloat",
    "FloatToInt",
    "ByteToInt",
    "IntToByte",
    # Built-in error struct names (treated as reserved for simplicity)
    "KeyError",
    "IndexError",
    "ZeroDivisionError",
    "AssertError",
    "NilError",
    "ValueError",
}


@dataclass(frozen=True)
class FnSig:
    params: tuple[Ty, ...]
    ret: Ty

    def ty(self) -> TyFunc:
        return TyFunc(self.params, self.ret)


@dataclass(frozen=True)
class FieldInfo:
    name: str
    ty: Ty
    decl: TFieldDecl


@dataclass(frozen=True)
class MethodInfo:
    name: str
    sig: FnSig  # includes self as first param
    decl: TFnDecl


@dataclass
class StructInfo:
    name: str
    implements: str | None
    fields: list[FieldInfo]
    field_map: dict[str, FieldInfo]
    methods: dict[str, MethodInfo]
    decl: TStructDecl


@dataclass
class EnumInfo:
    name: str
    variants: set[str]
    decl: TEnumDecl


@dataclass
class InterfaceInfo:
    name: str
    decl: TInterfaceDecl
    implementors: set[str]


@dataclass
class FnInfo:
    name: str
    sig: FnSig
    decl: TFnDecl


@dataclass
class ModuleIndex:
    funcs: dict[str, FnInfo]
    structs: dict[str, StructInfo]
    interfaces: dict[str, InterfaceInfo]
    enums: dict[str, EnumInfo]

    def has_type_name(self, name: str) -> bool:
        return name in self.structs or name in self.interfaces or name in self.enums


def _ensure_not_reserved(name: str, *, pos: Pos) -> None:
    if name in _RESERVED_BINDINGS:
        raise TaytshTypeError(f"reserved name '{name}'", pos)


def _build_index(module: TModule) -> ModuleIndex:
    funcs: dict[str, FnInfo] = {}
    structs: dict[str, StructInfo] = {}
    interfaces: dict[str, InterfaceInfo] = {}
    enums: dict[str, EnumInfo] = {}

    # Predeclare built-in error structs.
    def _builtin_err(name: str) -> StructInfo:
        decl = TStructDecl(
            Pos(0, 0),
            name,
            None,
            [TFieldDecl(Pos(0, 0), "message", TPrimitive(Pos(0, 0), "string"))],
            [],
        )
        # Types resolved later; placeholder for now.
        fi = FieldInfo("message", TY_STRING, decl.fields[0])
        return StructInfo(
            name=name,
            implements=None,
            fields=[fi],
            field_map={"message": fi},
            methods={},
            decl=decl,
        )

    for err in (
        "KeyError",
        "IndexError",
        "ZeroDivisionError",
        "AssertError",
        "NilError",
        "ValueError",
    ):
        structs[err] = _builtin_err(err)

    # First pass: collect decl kinds and detect duplicates across the flat namespace.
    seen_top: dict[str, TDecl] = {}
    for d in module.decls:
        name: str
        if isinstance(d, TFnDecl):
            name = d.name
        elif isinstance(d, TStructDecl):
            name = d.name
        elif isinstance(d, TInterfaceDecl):
            name = d.name
        elif isinstance(d, TEnumDecl):
            name = d.name
        else:
            continue
        _ensure_not_reserved(name, pos=d.pos)
        if name in seen_top or name in structs:
            raise TaytshTypeError(f"duplicate top-level name '{name}'", d.pos)
        seen_top[name] = d

    # Populate structs / interfaces / enums first so type resolution can refer to them.
    for name, d in seen_top.items():
        if isinstance(d, TInterfaceDecl):
            interfaces[name] = InterfaceInfo(name=name, decl=d, implementors=set())
        elif isinstance(d, TEnumDecl):
            enums[name] = EnumInfo(name=name, variants=set(d.variants), decl=d)
        elif isinstance(d, TStructDecl):
            structs[name] = StructInfo(
                name=name,
                implements=d.parent,
                fields=[],
                field_map={},
                methods={},
                decl=d,
            )

    # Second pass: resolve struct implements + field/method names (types later).
    for s in structs.values():
        decl = s.decl
        if decl.parent is not None:
            if decl.parent not in interfaces:
                raise TaytshTypeError(
                    f"struct '{s.name}' implements unknown interface '{decl.parent}'",
                    decl.pos,
                )
            interfaces[decl.parent].implementors.add(s.name)

        # Fields
        field_map: dict[str, FieldInfo] = {}
        fields: list[FieldInfo] = []
        for f in decl.fields:
            if f.name in field_map:
                raise TaytshTypeError(
                    f"duplicate field '{f.name}' in struct '{s.name}'", f.pos
                )
            fi = FieldInfo(f.name, TY_OBJ, f)  # placeholder ty resolved later
            field_map[f.name] = fi
            fields.append(fi)

        # Methods
        methods: dict[str, MethodInfo] = {}
        for m in decl.methods:
            if m.name in methods:
                raise TaytshTypeError(
                    f"duplicate method '{m.name}' in struct '{s.name}'", m.pos
                )
            if m.name in field_map:
                raise TaytshTypeError(
                    f"method '{m.name}' conflicts with field in struct '{s.name}'",
                    m.pos,
                )
            methods[m.name] = MethodInfo(m.name, FnSig((), TY_VOID), m)  # placeholder

        s.fields = fields
        s.field_map = field_map
        s.methods = methods

    # Functions (signatures resolved later).
    for name, d in seen_top.items():
        if isinstance(d, TFnDecl):
            funcs[name] = FnInfo(
                name=name, sig=FnSig((), TY_VOID), decl=d
            )  # placeholder

    return ModuleIndex(funcs=funcs, structs=structs, interfaces=interfaces, enums=enums)


class _TypeEnv:
    def __init__(self) -> None:
        self._scopes: list[dict[str, Ty]] = []
        self._overrides: list[dict[str, Ty]] = []

    def push_scope(self) -> None:
        self._scopes.append({})

    def pop_scope(self) -> None:
        self._scopes.pop()

    def push_override(self, mapping: dict[str, Ty]) -> None:
        self._overrides.append(mapping)

    def pop_override(self) -> None:
        self._overrides.pop()

    def is_bound_anywhere(self, name: str) -> bool:
        for scope in reversed(self._scopes):
            if name in scope:
                return True
        return False

    def bind(self, name: str, typ: Ty, *, pos: Pos) -> None:
        if name == "_":
            return
        _ensure_not_reserved(name, pos=pos)
        if not self._scopes:
            raise RuntimeError("no scope to bind into")
        if self.is_bound_anywhere(name):
            raise TaytshTypeError(f"name '{name}' already bound", pos)
        self._scopes[-1][name] = typ

    def get(self, name: str, *, pos: Pos) -> Ty:
        if name == "_":
            raise TaytshTypeError("'_' is discard-only and cannot be referenced", pos)
        for ovr in reversed(self._overrides):
            if name in ovr:
                return ovr[name]
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        raise TaytshTypeError(f"unknown name '{name}'", pos)


class TypeChecker:
    def __init__(self, index: ModuleIndex):
        self.index = index
        self.expr_types: dict[int, Ty] = {}

    def resolve_type(self, typ: TType, *, pos: Pos, allow_void: bool = False) -> Ty:
        if isinstance(typ, TPrimitive):
            if typ.kind == "int":
                return TY_INT
            if typ.kind == "float":
                return TY_FLOAT
            if typ.kind == "bool":
                return TY_BOOL
            if typ.kind == "byte":
                return TY_BYTE
            if typ.kind == "bytes":
                return TY_BYTES
            if typ.kind == "string":
                return TY_STRING
            if typ.kind == "rune":
                return TY_RUNE
            if typ.kind == "obj":
                return TY_OBJ
            if typ.kind == "nil":
                return TY_NIL
            if typ.kind == "void":
                if allow_void:
                    return TY_VOID
                raise TaytshTypeError("void is not a value type", pos)
            raise TaytshTypeError(f"unknown primitive type '{typ.kind}'", pos)

        if isinstance(typ, TListType):
            elem = self.resolve_type(typ.element, pos=typ.pos)
            return TyList(elem)
        if isinstance(typ, TMapType):
            key = self.resolve_type(typ.key, pos=typ.pos)
            val = self.resolve_type(typ.value, pos=typ.pos)
            if not self._is_hashable_type(key):
                raise TaytshTypeError(
                    f"map key type '{key.display()}' is not hashable", pos
                )
            return TyMap(key, val)
        if isinstance(typ, TSetType):
            elem = self.resolve_type(typ.element, pos=typ.pos)
            if not self._is_hashable_type(elem):
                raise TaytshTypeError(
                    f"set element type '{elem.display()}' is not hashable", pos
                )
            return TySet(elem)
        if isinstance(typ, TTupleType):
            if len(typ.elements) < 2:
                raise TaytshTypeError("tuple types require 2+ elements", pos)
            elems = tuple(self.resolve_type(e, pos=e.pos) for e in typ.elements)
            return TyTuple(elems)
        if isinstance(typ, TFuncType):
            if len(typ.params) < 1:
                raise TaytshTypeError("fn[...] requires at least a return type", pos)
            resolved = [
                self.resolve_type(t, pos=t.pos, allow_void=True) for t in typ.params
            ]
            ret = resolved[-1]
            params = tuple(resolved[:-1])
            if any(ty_eq(p, TY_VOID) for p in params):
                raise TaytshTypeError("void cannot be a parameter type", pos)
            return TyFunc(params, ret)
        if isinstance(typ, TIdentType):
            name = typ.name
            if name in self.index.structs:
                return TyStruct(name)
            if name in self.index.interfaces:
                return TyInterface(name)
            if name in self.index.enums:
                return TyEnum(name)
            raise TaytshTypeError(f"unknown type '{name}'", pos)
        if isinstance(typ, TUnionType):
            members = [self.resolve_type(m, pos=m.pos) for m in typ.members]
            if any(ty_eq(m, TY_VOID) for m in members):
                raise TaytshTypeError("void cannot appear in a union", pos)
            return ty_union(members)
        if isinstance(typ, TOptionalType):
            inner = self.resolve_type(typ.inner, pos=typ.pos)
            if ty_eq(inner, TY_VOID):
                raise TaytshTypeError("void cannot be optional", pos)
            return ty_union([inner, TY_NIL])

        raise TaytshTypeError("unsupported type syntax", pos)

    def _is_hashable_type(self, typ: Ty) -> bool:
        if isinstance(typ, TyPrim):
            return typ.kind in {
                "int",
                "float",
                "bool",
                "byte",
                "bytes",
                "string",
                "rune",
            }
        if isinstance(typ, TyEnum):
            return True
        if isinstance(typ, TyTuple):
            return all(self._is_hashable_type(t) for t in typ.elements)
        return False

    def _resolve_index_signatures(self) -> None:
        # Struct field types
        for s in self.index.structs.values():
            # Built-in error structs already have resolved placeholder types.
            if (
                s.name
                in {
                    "KeyError",
                    "IndexError",
                    "ZeroDivisionError",
                    "AssertError",
                    "NilError",
                    "ValueError",
                }
                and s.decl.pos.line == 0
            ):
                continue
            fields: list[FieldInfo] = []
            field_map: dict[str, FieldInfo] = {}
            for f in s.decl.fields:
                t = self.resolve_type(f.typ, pos=f.pos)
                if ty_eq(t, TY_VOID):
                    raise TaytshTypeError("void cannot be a field type", f.pos)
                fi = FieldInfo(f.name, t, f)
                fields.append(fi)
                field_map[f.name] = fi
            s.fields = fields
            s.field_map = field_map

            # Methods
            methods: dict[str, MethodInfo] = {}
            for m in s.decl.methods:
                sig = self._fn_decl_sig(m, method_self=TyStruct(s.name))
                methods[m.name] = MethodInfo(m.name, sig, m)
            s.methods = methods

        # Top-level function signatures
        for f in self.index.funcs.values():
            f.sig = self._fn_decl_sig(f.decl, method_self=None)

    def _fn_decl_sig(self, decl: TFnDecl, *, method_self: TyStruct | None) -> FnSig:
        params: list[Ty] = []
        seen = set()
        if method_self is not None:
            if not decl.params:
                raise TaytshTypeError("method must take self parameter", decl.pos)
            first = decl.params[0]
            if first.typ is not None or first.name != "self":
                raise TaytshTypeError(
                    "method must take self as first parameter", first.pos
                )
        for i, p in enumerate(decl.params):
            if p.name in seen and p.name != "_":
                raise TaytshTypeError(f"duplicate parameter '{p.name}'", p.pos)
            seen.add(p.name)
            if p.typ is None:
                if method_self is None:
                    raise TaytshTypeError("'self' only allowed in methods", p.pos)
                if i != 0 or p.name != "self":
                    raise TaytshTypeError("self must be the first parameter", p.pos)
                params.append(method_self)
            else:
                if p.name == "self":
                    raise TaytshTypeError("'self' parameter must omit type", p.pos)
                t = self.resolve_type(p.typ, pos=p.pos)
                if ty_eq(t, TY_VOID):
                    raise TaytshTypeError("void cannot be a parameter type", p.pos)
                params.append(t)
        ret = self.resolve_type(decl.ret, pos=decl.pos, allow_void=True)
        return FnSig(tuple(params), ret)

    def check_module(self, module: TModule) -> ModuleIndex:
        self._resolve_index_signatures()

        # Validate Main.
        if "Main" not in self.index.funcs:
            raise TaytshTypeError("missing Main() entrypoint", Pos(1, 1))
        main = self.index.funcs["Main"]
        if len(main.sig.params) != 0 or not ty_eq(main.sig.ret, TY_VOID):
            raise TaytshTypeError(
                "Main must have signature fn Main() -> void", main.decl.pos
            )

        # Typecheck all top-level functions and methods.
        for f in self.index.funcs.values():
            self._check_fn_body(f.decl, f.sig, allow_capture=False, method_self=None)
        for s in self.index.structs.values():
            for m in s.methods.values():
                self._check_fn_body(
                    m.decl, m.sig, allow_capture=False, method_self=TyStruct(s.name)
                )
        return self.index

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def _check_fn_body(
        self,
        decl: TFnDecl,
        sig: FnSig,
        *,
        allow_capture: bool,
        method_self: TyStruct | None,
    ) -> None:
        env = _TypeEnv()
        env.push_scope()
        # Bind params
        for i, p in enumerate(decl.params):
            if p.typ is None:
                if method_self is None:
                    raise TaytshTypeError("'self' only allowed in methods", p.pos)
                if i != 0:
                    raise TaytshTypeError("self must be first parameter", p.pos)
                env.bind("self", method_self, pos=p.pos)
            else:
                env.bind(p.name, sig.params[i], pos=p.pos)

        self._check_block(
            decl.body,
            env,
            fn_ret=sig.ret,
            in_loop=0,
            allow_capture=allow_capture,
        )

        # Non-void functions must return on all paths (best-effort).
        if not ty_eq(sig.ret, TY_VOID) and not self._block_always_returns(decl.body):
            raise TaytshTypeError(
                f"function '{decl.name}' may fall off without returning",
                decl.pos,
            )

    def _block_always_returns(self, stmts: list[TStmt]) -> bool:
        if not stmts:
            return False
        last = stmts[-1]
        if isinstance(last, TReturnStmt):
            return True
        if isinstance(last, TIfStmt):
            if last.else_body is None:
                return False
            return self._block_always_returns(
                last.then_body
            ) and self._block_always_returns(cast(list[TStmt], last.else_body))
        if isinstance(last, TMatchStmt):
            # Exhaustive match with all cases returning counts as always-return.
            for c in last.cases:
                if not self._block_always_returns(c.body):
                    return False
            if last.default is not None and not self._block_always_returns(
                last.default.body
            ):
                return False
            # If default absent, the match must be exhaustive (checked elsewhere) to count.
            return True
        if isinstance(last, TTryStmt):
            # Conservatively: try always returns only if finally doesn't exist and all bodies do.
            # (Full "completion" analysis is out of scope for the typechecker.)
            if last.finally_body is not None:
                return False
            if not self._block_always_returns(last.body):
                return False
            for c in last.catches:
                if not self._block_always_returns(c.body):
                    return False
            return bool(last.catches)
        return False

    def _check_block(
        self,
        stmts: list[TStmt],
        env: _TypeEnv,
        *,
        fn_ret: Ty,
        in_loop: int,
        allow_capture: bool,
    ) -> None:
        env.push_scope()
        try:
            for st in stmts:
                self._check_stmt(
                    st,
                    env,
                    fn_ret=fn_ret,
                    in_loop=in_loop,
                    allow_capture=allow_capture,
                )
        finally:
            env.pop_scope()

    def _check_stmt(
        self,
        st: TStmt,
        env: _TypeEnv,
        *,
        fn_ret: Ty,
        in_loop: int,
        allow_capture: bool,
    ) -> None:
        if isinstance(st, TLetStmt):
            t = self.resolve_type(st.typ, pos=st.pos)
            if ty_eq(t, TY_VOID):
                raise TaytshTypeError("void is not a valid variable type", st.pos)
            if st.value is None:
                if not self._has_zero_value(t):
                    raise TaytshTypeError(
                        f"type '{t.display()}' has no zero value; initializer required",
                        st.pos,
                    )
                env.bind(st.name, t, pos=st.pos)
                return
            vty = self._type_expr(
                st.value, env, expected=t, allow_capture=allow_capture
            )
            if not self._assignable(vty, t):
                raise TaytshTypeError(
                    f"cannot assign '{vty.display()}' to '{t.display()}'",
                    st.pos,
                )
            env.bind(st.name, t, pos=st.pos)
            return

        if isinstance(st, TAssignStmt):
            target_ty = self._type_lvalue(st.target, env, allow_capture=allow_capture)
            value_ty = self._type_expr(
                st.value, env, expected=target_ty, allow_capture=allow_capture
            )
            if not self._assignable(value_ty, target_ty):
                raise TaytshTypeError(
                    f"cannot assign '{value_ty.display()}' to '{target_ty.display()}'",
                    st.pos,
                )
            return

        if isinstance(st, TOpAssignStmt):
            target_ty = self._type_lvalue(st.target, env, allow_capture=allow_capture)
            value_ty = self._type_expr(
                st.value, env, expected=target_ty, allow_capture=allow_capture
            )
            if not self._assignable(value_ty, target_ty):
                raise TaytshTypeError(
                    f"cannot assign '{value_ty.display()}' to '{target_ty.display()}'",
                    st.pos,
                )
            base_op = st.op[:-1]
            _ = self._type_binary(base_op, target_ty, value_ty, pos=st.pos)
            return

        if isinstance(st, TTupleAssignStmt):
            if not isinstance(st.value, (TTupleLit, TCall)):
                raise TaytshTypeError(
                    "tuple assignment requires tuple literal or call expression", st.pos
                )
            rhs_ty = self._type_expr(st.value, env, allow_capture=allow_capture)
            if not isinstance(rhs_ty, TyTuple):
                raise TaytshTypeError("tuple assignment rhs must be a tuple", st.pos)
            if len(rhs_ty.elements) != len(st.targets):
                raise TaytshTypeError("tuple arity mismatch in assignment", st.pos)
            for i, tgt in enumerate(st.targets):
                t_ty = self._type_lvalue(tgt, env, allow_capture=allow_capture)
                if not self._assignable(rhs_ty.elements[i], t_ty):
                    raise TaytshTypeError(
                        f"cannot assign '{rhs_ty.elements[i].display()}' to '{t_ty.display()}'",
                        st.pos,
                    )
            return

        if isinstance(st, TReturnStmt):
            if ty_eq(fn_ret, TY_VOID):
                if st.value is not None:
                    raise TaytshTypeError("void function cannot return a value", st.pos)
                return
            if st.value is None:
                raise TaytshTypeError("non-void function must return a value", st.pos)
            vty = self._type_expr(
                st.value, env, expected=fn_ret, allow_capture=allow_capture
            )
            if not self._assignable(vty, fn_ret):
                raise TaytshTypeError(
                    f"cannot return '{vty.display()}' from function returning '{fn_ret.display()}'",
                    st.pos,
                )
            return

        if isinstance(st, TIfStmt):
            cty = self._type_expr(
                st.cond, env, expected=TY_BOOL, allow_capture=allow_capture
            )
            if not ty_eq(cty, TY_BOOL):
                raise TaytshTypeError("if condition must be bool", st.pos)
            then_ovr, else_ovr = self._nil_refinements(st.cond, env)
            if then_ovr:
                env.push_override(then_ovr)
            self._check_block(
                st.then_body,
                env,
                fn_ret=fn_ret,
                in_loop=in_loop,
                allow_capture=allow_capture,
            )
            if then_ovr:
                env.pop_override()
            if st.else_body is not None:
                if else_ovr:
                    env.push_override(else_ovr)
                self._check_block(
                    cast(list[TStmt], st.else_body),
                    env,
                    fn_ret=fn_ret,
                    in_loop=in_loop,
                    allow_capture=allow_capture,
                )
                if else_ovr:
                    env.pop_override()
            return

        if isinstance(st, TWhileStmt):
            cty = self._type_expr(
                st.cond, env, expected=TY_BOOL, allow_capture=allow_capture
            )
            if not ty_eq(cty, TY_BOOL):
                raise TaytshTypeError("while condition must be bool", st.pos)
            self._check_block(
                st.body,
                env,
                fn_ret=fn_ret,
                in_loop=in_loop + 1,
                allow_capture=allow_capture,
            )
            return

        if isinstance(st, TForStmt):
            self._check_for(
                st, env, fn_ret=fn_ret, in_loop=in_loop, allow_capture=allow_capture
            )
            return

        if isinstance(st, TBreakStmt):
            if in_loop <= 0:
                raise TaytshTypeError("break outside of loop", st.pos)
            return

        if isinstance(st, TContinueStmt):
            if in_loop <= 0:
                raise TaytshTypeError("continue outside of loop", st.pos)
            return

        if isinstance(st, TThrowStmt):
            ety = self._type_expr(st.expr, env, allow_capture=allow_capture)
            if ty_eq(ety, TY_VOID):
                raise TaytshTypeError("cannot throw void", st.pos)
            return

        if isinstance(st, TExprStmt):
            _ = self._type_expr(st.expr, env, allow_capture=allow_capture)
            return

        if isinstance(st, TMatchStmt):
            self._check_match(
                st, env, fn_ret=fn_ret, in_loop=in_loop, allow_capture=allow_capture
            )
            return

        if isinstance(st, TTryStmt):
            self._check_try(
                st, env, fn_ret=fn_ret, in_loop=in_loop, allow_capture=allow_capture
            )
            return

        raise TaytshTypeError("unsupported statement", st.pos)

    def _check_try(
        self,
        st: TTryStmt,
        env: _TypeEnv,
        *,
        fn_ret: Ty,
        in_loop: int,
        allow_capture: bool,
    ) -> None:
        self._check_block(
            st.body, env, fn_ret=fn_ret, in_loop=in_loop, allow_capture=allow_capture
        )
        for c in st.catches:
            types = [self.resolve_type(t, pos=t.pos) for t in c.types]
            if any(ty_eq(t, TY_VOID) for t in types):
                raise TaytshTypeError("void cannot be caught", c.pos)
            c_ty = ty_union(types)
            env.push_scope()
            try:
                env.bind(c.name, c_ty, pos=c.pos)
                self._check_block(
                    c.body,
                    env,
                    fn_ret=fn_ret,
                    in_loop=in_loop,
                    allow_capture=allow_capture,
                )
            finally:
                env.pop_scope()
        if st.finally_body is not None:
            self._check_block(
                st.finally_body,
                env,
                fn_ret=fn_ret,
                in_loop=in_loop,
                allow_capture=allow_capture,
            )

    def _check_for(
        self,
        st: TForStmt,
        env: _TypeEnv,
        *,
        fn_ret: Ty,
        in_loop: int,
        allow_capture: bool,
    ) -> None:
        if len(st.binding) not in (1, 2):
            raise TaytshTypeError("for must bind 1 or 2 variables", st.pos)

        # Range loop
        if isinstance(st.iterable, TRange):
            if len(st.binding) != 1:
                raise TaytshTypeError(
                    "range loops do not support two-variable form", st.pos
                )
            if not (1 <= len(st.iterable.args) <= 3):
                raise TaytshTypeError(
                    "range() expects 1 to 3 arguments", st.iterable.pos
                )
            for a in st.iterable.args:
                aty = self._type_expr(
                    a, env, expected=TY_INT, allow_capture=allow_capture
                )
                if not ty_eq(aty, TY_INT):
                    raise TaytshTypeError("range() arguments must be int", a.pos)
            loop_bindings = [(st.binding[0], TY_INT)]
            self._check_loop_body(
                st.body,
                env,
                loop_bindings,
                fn_ret=fn_ret,
                in_loop=in_loop,
                allow_capture=allow_capture,
            )
            return

        it_ty = self._type_expr(st.iterable, env, allow_capture=allow_capture)
        loop_bindings: list[tuple[str, Ty]] = []
        if isinstance(it_ty, TyList):
            if len(st.binding) == 1:
                loop_bindings.append((st.binding[0], it_ty.element))
            else:
                loop_bindings.append((st.binding[0], TY_INT))
                loop_bindings.append((st.binding[1], it_ty.element))
        elif ty_eq(it_ty, TY_STRING):
            if len(st.binding) == 1:
                loop_bindings.append((st.binding[0], TY_RUNE))
            else:
                loop_bindings.append((st.binding[0], TY_INT))
                loop_bindings.append((st.binding[1], TY_RUNE))
        elif ty_eq(it_ty, TY_BYTES):
            if len(st.binding) == 1:
                loop_bindings.append((st.binding[0], TY_BYTE))
            else:
                loop_bindings.append((st.binding[0], TY_INT))
                loop_bindings.append((st.binding[1], TY_BYTE))
        elif isinstance(it_ty, TyMap):
            if len(st.binding) == 1:
                loop_bindings.append((st.binding[0], it_ty.key))
            else:
                loop_bindings.append((st.binding[0], it_ty.key))
                loop_bindings.append((st.binding[1], it_ty.value))
        elif isinstance(it_ty, TySet):
            if len(st.binding) != 1:
                raise TaytshTypeError(
                    "set iteration does not support two-variable form", st.pos
                )
            loop_bindings.append((st.binding[0], it_ty.element))
        else:
            raise TaytshTypeError(f"type '{it_ty.display()}' is not iterable", st.pos)

        self._check_loop_body(
            st.body,
            env,
            loop_bindings,
            fn_ret=fn_ret,
            in_loop=in_loop,
            allow_capture=allow_capture,
        )

    def _check_loop_body(
        self,
        body: list[TStmt],
        env: _TypeEnv,
        loop_bindings: list[tuple[str, Ty]],
        *,
        fn_ret: Ty,
        in_loop: int,
        allow_capture: bool,
    ) -> None:
        env.push_scope()
        try:
            for name, t in loop_bindings:
                env.bind(name, t, pos=Pos(0, 0))
            self._check_block(
                body,
                env,
                fn_ret=fn_ret,
                in_loop=in_loop + 1,
                allow_capture=allow_capture,
            )
        finally:
            env.pop_scope()

    def _check_match(
        self,
        st: TMatchStmt,
        env: _TypeEnv,
        *,
        fn_ret: Ty,
        in_loop: int,
        allow_capture: bool,
    ) -> None:
        scrut_ty = self._type_expr(st.expr, env, allow_capture=allow_capture)

        # Typecheck cases and compute coverage.
        covered_types: set[tuple] = set()
        covered_enum: dict[str, set[str]] = {}
        covered_nil = False
        for c in st.cases:
            pat = c.pattern
            if isinstance(pat, TPatternNil):
                if not ty_has_nil(scrut_ty):
                    raise TaytshTypeError("case nil is unreachable", pat.pos)
                covered_nil = True
                self._check_block(
                    c.body,
                    env,
                    fn_ret=fn_ret,
                    in_loop=in_loop,
                    allow_capture=allow_capture,
                )
                continue
            if isinstance(pat, TPatternEnum):
                enum_ty = TyEnum(pat.enum_name)
                if not self._match_case_possible(scrut_ty, enum_ty):
                    raise TaytshTypeError(
                        "enum case does not match scrutinee type", pat.pos
                    )
                enum = self.index.enums.get(pat.enum_name)
                if enum is None or pat.variant not in enum.variants:
                    raise TaytshTypeError("unknown enum variant", pat.pos)
                seen = covered_enum.setdefault(pat.enum_name, set())
                if pat.variant in seen:
                    raise TaytshTypeError("duplicate enum case", pat.pos)
                seen.add(pat.variant)
                self._check_block(
                    c.body,
                    env,
                    fn_ret=fn_ret,
                    in_loop=in_loop,
                    allow_capture=allow_capture,
                )
                continue
            if isinstance(pat, TPatternType):
                case_ty = self.resolve_type(pat.type_name, pos=pat.pos)
                if not self._match_case_possible(scrut_ty, case_ty):
                    raise TaytshTypeError(
                        f"case '{case_ty.display()}' cannot match '{scrut_ty.display()}'",
                        pat.pos,
                    )
                key = _ty_key(case_ty)
                if key in covered_types:
                    raise TaytshTypeError("duplicate type case", pat.pos)
                covered_types.add(key)
                env.push_scope()
                try:
                    env.bind(pat.name, case_ty, pos=pat.pos)
                    self._check_block(
                        c.body,
                        env,
                        fn_ret=fn_ret,
                        in_loop=in_loop,
                        allow_capture=allow_capture,
                    )
                finally:
                    env.pop_scope()
                continue
            raise TaytshTypeError("unsupported match pattern", c.pos)

        # Default
        if st.default is not None:
            if st.default.name is None:
                self._check_block(
                    st.default.body,
                    env,
                    fn_ret=fn_ret,
                    in_loop=in_loop,
                    allow_capture=allow_capture,
                )
            else:
                env.push_scope()
                try:
                    env.bind(st.default.name, TY_OBJ, pos=st.default.pos)
                    self._check_block(
                        st.default.body,
                        env,
                        fn_ret=fn_ret,
                        in_loop=in_loop,
                        allow_capture=allow_capture,
                    )
                finally:
                    env.pop_scope()

        # Exhaustiveness
        if ty_eq(scrut_ty, TY_OBJ):
            if st.default is None:
                raise TaytshTypeError("match on obj requires default", st.pos)
            return
        if isinstance(scrut_ty, TyEnum):
            enum = self.index.enums[scrut_ty.name]
            if st.default is None and _ty_key(scrut_ty) not in covered_types:
                seen = covered_enum.get(scrut_ty.name, set())
                if seen != enum.variants:
                    missing = sorted(enum.variants - seen)
                    raise TaytshTypeError(
                        f"non-exhaustive enum match (missing {missing})", st.pos
                    )
            return
        if isinstance(scrut_ty, TyInterface):
            impls = self.index.interfaces[scrut_ty.name].implementors
            if st.default is not None:
                return
            # case naming the interface itself counts as exhaustive
            if _ty_key(scrut_ty) in covered_types:
                return
            covered_structs = {k[1] for k in covered_types if k[0] == "struct"}
            missing = impls - covered_structs
            if missing:
                raise TaytshTypeError(
                    f"non-exhaustive interface match (missing {sorted(missing)})",
                    st.pos,
                )
            return
        if isinstance(scrut_ty, TyUnion):
            if st.default is not None:
                return
            # Compute coverage per member.
            for m in scrut_ty.members:
                if ty_is_nil(m):
                    if not covered_nil:
                        raise TaytshTypeError(
                            "non-exhaustive match (missing nil)", st.pos
                        )
                    continue
                if isinstance(m, TyEnum):
                    if _ty_key(m) in covered_types:
                        continue
                    enum = self.index.enums.get(m.name)
                    if enum is None:
                        raise TaytshTypeError("unknown enum type", st.pos)
                    seen = covered_enum.get(m.name, set())
                    if seen != enum.variants:
                        raise TaytshTypeError(
                            f"non-exhaustive match (missing {m.name} variants)",
                            st.pos,
                        )
                    continue
                if isinstance(m, TyInterface):
                    if _ty_key(m) in covered_types:
                        continue
                    impls = self.index.interfaces[m.name].implementors
                    covered_structs = {
                        t[1]
                        for t in covered_types
                        if isinstance(t, tuple) and len(t) >= 2 and t[0] == "struct"
                    }
                    if not impls.issubset(covered_structs):
                        raise TaytshTypeError(
                            f"non-exhaustive match (missing {m.name} variants)",
                            st.pos,
                        )
                    continue
                if _ty_key(m) not in covered_types:
                    raise TaytshTypeError(
                        f"non-exhaustive match (missing {m.display()})", st.pos
                    )
            return
        # Concrete type: must have a case or default.
        if (
            st.default is None
            and _ty_key(scrut_ty) not in covered_types
            and not (ty_is_nil(scrut_ty) and covered_nil)
        ):
            raise TaytshTypeError("non-exhaustive match", st.pos)

    def _match_case_possible(self, scrutinee: Ty, case_ty: Ty) -> bool:
        if ty_eq(scrutinee, TY_OBJ):
            return True
        if ty_eq(scrutinee, case_ty):
            return True
        # Union scrutinee: any member compatible
        if isinstance(scrutinee, TyUnion):
            return any(self._match_case_possible(m, case_ty) for m in scrutinee.members)
        # Optional/nil via union
        if ty_is_nil(case_ty):
            return ty_has_nil(scrutinee)
        # Interface scrutinee: struct implementor can match
        if isinstance(scrutinee, TyInterface) and isinstance(case_ty, TyStruct):
            return case_ty.name in self.index.interfaces[scrutinee.name].implementors
        # Union member interface + case names interface itself
        if isinstance(scrutinee, TyInterface) and isinstance(case_ty, TyInterface):
            return scrutinee.name == case_ty.name
        return False

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------

    def _type_lvalue(self, expr: TExpr, env: _TypeEnv, *, allow_capture: bool) -> Ty:
        if isinstance(expr, TVar):
            return env.get(expr.name, pos=expr.pos)
        if isinstance(expr, TFieldAccess):
            obj_ty = self._type_expr(expr.obj, env, allow_capture=allow_capture)
            return self._field_type(obj_ty, expr.field, pos=expr.pos)
        if isinstance(expr, TIndex):
            obj_ty = self._type_expr(expr.obj, env, allow_capture=allow_capture)
            if isinstance(obj_ty, TyList):
                idx_ty = self._type_expr(
                    expr.index, env, expected=TY_INT, allow_capture=allow_capture
                )
                if not ty_eq(idx_ty, TY_INT):
                    raise TaytshTypeError("list index must be int", expr.pos)
                return obj_ty.element
            if isinstance(obj_ty, TyMap):
                key_ty = self._type_expr(
                    expr.index, env, expected=obj_ty.key, allow_capture=allow_capture
                )
                if not self._assignable(key_ty, obj_ty.key):
                    raise TaytshTypeError("map key type mismatch", expr.pos)
                return obj_ty.value
            raise TaytshTypeError(
                "index assignment only allowed on list or map", expr.pos
            )
        raise TaytshTypeError("invalid assignment target", expr.pos)

    def _type_expr(
        self,
        expr: TExpr,
        env: _TypeEnv,
        *,
        expected: Ty | None = None,
        allow_capture: bool,
    ) -> Ty:
        cached = self.expr_types.get(id(expr))
        if cached is not None:
            return cached
        if isinstance(expr, TIntLit):
            return TY_INT
        if isinstance(expr, TFloatLit):
            return TY_FLOAT
        if isinstance(expr, TByteLit):
            return TY_BYTE
        if isinstance(expr, TStringLit):
            return TY_STRING
        if isinstance(expr, TRuneLit):
            return TY_RUNE
        if isinstance(expr, TBytesLit):
            return TY_BYTES
        if isinstance(expr, TBoolLit):
            return TY_BOOL
        if isinstance(expr, TNilLit):
            return TY_NIL
        if isinstance(expr, TVar):
            name = expr.name
            # Locals
            if env.is_bound_anywhere(name):
                return env.get(name, pos=expr.pos)
            # Globals: functions and builtins only.
            if name in self.index.funcs:
                return self.index.funcs[name].sig.ty()
            if name in _RESERVED_BINDINGS:
                # A reserved name that is also a built-in function.
                if name in _BUILTIN_DISPATCH:
                    return _BUILTIN_DISPATCH[name].ty()
            if self.index.has_type_name(name):
                raise TaytshTypeError(f"'{name}' is a type name, not a value", expr.pos)
            raise TaytshTypeError(f"unknown name '{name}'", expr.pos)

        if isinstance(expr, TUnaryOp):
            oty = self._type_expr(expr.operand, env, allow_capture=allow_capture)
            return self._type_unary(expr.op, oty, pos=expr.pos)
        if isinstance(expr, TBinaryOp):
            lty = self._type_expr(expr.left, env, allow_capture=allow_capture)
            rty = self._type_expr(expr.right, env, allow_capture=allow_capture)
            return self._type_binary(expr.op, lty, rty, pos=expr.pos)
        if isinstance(expr, TTernary):
            cty = self._type_expr(
                expr.cond, env, expected=TY_BOOL, allow_capture=allow_capture
            )
            if not ty_eq(cty, TY_BOOL):
                raise TaytshTypeError("ternary condition must be bool", expr.pos)
            tty = self._type_expr(expr.then_expr, env, allow_capture=allow_capture)
            ety = self._type_expr(expr.else_expr, env, allow_capture=allow_capture)
            if not ty_eq(tty, ety):
                raise TaytshTypeError("ternary branches must have same type", expr.pos)
            return tty
        if isinstance(expr, TTupleAccess):
            obj_ty = self._type_expr(expr.obj, env, allow_capture=allow_capture)
            if not isinstance(obj_ty, TyTuple):
                raise TaytshTypeError("tuple access requires a tuple", expr.pos)
            if expr.index < 0 or expr.index >= len(obj_ty.elements):
                raise TaytshTypeError("tuple index out of range", expr.pos)
            return obj_ty.elements[expr.index]
        if isinstance(expr, TFieldAccess):
            # Enum value: EnumName.Variant
            if isinstance(expr.obj, TVar) and expr.obj.name in self.index.enums:
                enum = self.index.enums[expr.obj.name]
                if expr.field not in enum.variants:
                    raise TaytshTypeError("unknown enum variant", expr.pos)
                return TyEnum(enum.name)
            obj_ty = self._type_expr(expr.obj, env, allow_capture=allow_capture)
            return self._field_type(obj_ty, expr.field, pos=expr.pos)
        if isinstance(expr, TIndex):
            obj_ty = self._type_expr(expr.obj, env, allow_capture=allow_capture)
            if isinstance(obj_ty, TyList):
                idx_ty = self._type_expr(
                    expr.index, env, expected=TY_INT, allow_capture=allow_capture
                )
                if not ty_eq(idx_ty, TY_INT):
                    raise TaytshTypeError("list index must be int", expr.pos)
                return obj_ty.element
            if ty_eq(obj_ty, TY_STRING):
                idx_ty = self._type_expr(
                    expr.index, env, expected=TY_INT, allow_capture=allow_capture
                )
                if not ty_eq(idx_ty, TY_INT):
                    raise TaytshTypeError("string index must be int", expr.pos)
                return TY_RUNE
            if ty_eq(obj_ty, TY_BYTES):
                idx_ty = self._type_expr(
                    expr.index, env, expected=TY_INT, allow_capture=allow_capture
                )
                if not ty_eq(idx_ty, TY_INT):
                    raise TaytshTypeError("bytes index must be int", expr.pos)
                return TY_BYTE
            if isinstance(obj_ty, TyMap):
                kty = self._type_expr(
                    expr.index, env, expected=obj_ty.key, allow_capture=allow_capture
                )
                if not self._assignable(kty, obj_ty.key):
                    raise TaytshTypeError("map key type mismatch", expr.pos)
                return obj_ty.value
            raise TaytshTypeError("indexing not supported for this type", expr.pos)
        if isinstance(expr, TSlice):
            obj_ty = self._type_expr(expr.obj, env, allow_capture=allow_capture)
            low_ty = self._type_expr(
                expr.low, env, expected=TY_INT, allow_capture=allow_capture
            )
            high_ty = self._type_expr(
                expr.high, env, expected=TY_INT, allow_capture=allow_capture
            )
            if not ty_eq(low_ty, TY_INT) or not ty_eq(high_ty, TY_INT):
                raise TaytshTypeError("slice bounds must be int", expr.pos)
            if isinstance(obj_ty, TyList):
                return obj_ty
            if ty_eq(obj_ty, TY_STRING):
                return TY_STRING
            if ty_eq(obj_ty, TY_BYTES):
                return TY_BYTES
            raise TaytshTypeError("slicing not supported for this type", expr.pos)
        if isinstance(expr, TListLit):
            if not expr.elements:
                if isinstance(expected, TyList):
                    self.expr_types[id(expr)] = expected
                    return expected
                raise TaytshTypeError("cannot infer type for []", expr.pos)
            if isinstance(expected, TyList):
                for e in expr.elements:
                    ety = self._type_expr(
                        e, env, expected=expected.element, allow_capture=allow_capture
                    )
                    if not self._assignable(ety, expected.element):
                        raise TaytshTypeError("list element type mismatch", e.pos)
                self.expr_types[id(expr)] = expected
                return expected
            elem_ty = self._type_expr(
                expr.elements[0], env, allow_capture=allow_capture
            )
            for e in expr.elements[1:]:
                ety = self._type_expr(
                    e, env, expected=elem_ty, allow_capture=allow_capture
                )
                if not ty_eq(ety, elem_ty):
                    raise TaytshTypeError("list elements must have same type", e.pos)
            ty = TyList(elem_ty)
            self.expr_types[id(expr)] = ty
            return ty
        if isinstance(expr, TMapLit):
            if isinstance(expected, TyMap):
                if not self._is_hashable_type(expected.key):
                    raise TaytshTypeError(
                        f"map key type '{expected.key.display()}' is not hashable",
                        expr.pos,
                    )
                for k, v in expr.entries:
                    kt = self._type_expr(
                        k, env, expected=expected.key, allow_capture=allow_capture
                    )
                    vt = self._type_expr(
                        v, env, expected=expected.value, allow_capture=allow_capture
                    )
                    if not self._assignable(kt, expected.key) or not self._assignable(
                        vt, expected.value
                    ):
                        raise TaytshTypeError(
                            "map literal entry type mismatch", expr.pos
                        )
                self.expr_types[id(expr)] = expected
                return expected
            (k0, v0) = expr.entries[0]
            kty = self._type_expr(k0, env, allow_capture=allow_capture)
            vty = self._type_expr(v0, env, allow_capture=allow_capture)
            if not self._is_hashable_type(kty):
                raise TaytshTypeError(
                    f"map key type '{kty.display()}' is not hashable", expr.pos
                )
            for k, v in expr.entries[1:]:
                kt = self._type_expr(k, env, expected=kty, allow_capture=allow_capture)
                vt = self._type_expr(v, env, expected=vty, allow_capture=allow_capture)
                if not ty_eq(kt, kty) or not ty_eq(vt, vty):
                    raise TaytshTypeError(
                        "map literal entries must have uniform types", expr.pos
                    )
            ty = TyMap(kty, vty)
            self.expr_types[id(expr)] = ty
            return ty
        if isinstance(expr, TSetLit):
            if isinstance(expected, TySet):
                if not self._is_hashable_type(expected.element):
                    raise TaytshTypeError(
                        f"set element type '{expected.element.display()}' is not hashable",
                        expr.pos,
                    )
                for e in expr.elements:
                    ety = self._type_expr(
                        e, env, expected=expected.element, allow_capture=allow_capture
                    )
                    if not self._assignable(ety, expected.element):
                        raise TaytshTypeError("set element type mismatch", e.pos)
                self.expr_types[id(expr)] = expected
                return expected
            elem_ty = self._type_expr(
                expr.elements[0], env, allow_capture=allow_capture
            )
            if not self._is_hashable_type(elem_ty):
                raise TaytshTypeError(
                    f"set element type '{elem_ty.display()}' is not hashable", expr.pos
                )
            for e in expr.elements[1:]:
                ety = self._type_expr(
                    e, env, expected=elem_ty, allow_capture=allow_capture
                )
                if not ty_eq(ety, elem_ty):
                    raise TaytshTypeError("set elements must have same type", e.pos)
            ty = TySet(elem_ty)
            self.expr_types[id(expr)] = ty
            return ty
        if isinstance(expr, TTupleLit):
            if len(expr.elements) < 2:
                raise TaytshTypeError("tuple literals require 2+ elements", expr.pos)
            if isinstance(expected, TyTuple):
                if len(expected.elements) != len(expr.elements):
                    raise TaytshTypeError("tuple arity mismatch", expr.pos)
                for i, e in enumerate(expr.elements):
                    ety = self._type_expr(
                        e,
                        env,
                        expected=expected.elements[i],
                        allow_capture=allow_capture,
                    )
                    if not self._assignable(ety, expected.elements[i]):
                        raise TaytshTypeError("tuple element type mismatch", e.pos)
                self.expr_types[id(expr)] = expected
                return expected
            elems = tuple(
                self._type_expr(e, env, allow_capture=allow_capture)
                for e in expr.elements
            )
            ty = TyTuple(elems)
            self.expr_types[id(expr)] = ty
            return ty
        if isinstance(expr, TFnLit):
            sig = self._fn_lit_sig(expr, method_self=None)
            # No closures: function literal can reference only its own params + globals.
            self._check_fn_lit_body(expr, sig)
            ty = sig.ty()
            self.expr_types[id(expr)] = ty
            return ty
        if isinstance(expr, TCall):
            ty = self._type_call(
                expr, env, expected=expected, allow_capture=allow_capture
            )
            self.expr_types[id(expr)] = ty
            return ty

        raise TaytshTypeError("unsupported expression", expr.pos)

    def _fn_lit_sig(self, lit: TFnLit, *, method_self: TyStruct | None) -> FnSig:
        params: list[Ty] = []
        for p in lit.params:
            if p.typ is None:
                raise TaytshTypeError("'self' not allowed in function literals", p.pos)
            t = self.resolve_type(p.typ, pos=p.pos)
            if ty_eq(t, TY_VOID):
                raise TaytshTypeError("void cannot be a parameter type", p.pos)
            params.append(t)
        ret = self.resolve_type(lit.ret, pos=lit.pos, allow_void=True)
        return FnSig(tuple(params), ret)

    def _check_fn_lit_body(self, lit: TFnLit, sig: FnSig) -> None:
        env = _TypeEnv()
        env.push_scope()
        for i, p in enumerate(lit.params):
            env.bind(p.name, sig.params[i], pos=p.pos)
        # allow_capture=False because we only bound params; outer locals are not present.
        if isinstance(lit.body, list):
            self._check_block(
                lit.body, env, fn_ret=sig.ret, in_loop=0, allow_capture=False
            )
            if not ty_eq(sig.ret, TY_VOID) and not self._block_always_returns(lit.body):
                raise TaytshTypeError(
                    "function literal may fall off without returning", lit.pos
                )
        else:
            body_ty = self._type_expr(
                lit.body, env, expected=sig.ret, allow_capture=False
            )
            if not self._assignable(body_ty, sig.ret):
                raise TaytshTypeError(
                    f"cannot return '{body_ty.display()}' from function literal returning '{sig.ret.display()}'",
                    lit.pos,
                )

    def _type_call(
        self,
        call: TCall,
        env: _TypeEnv,
        *,
        expected: Ty | None,
        allow_capture: bool,
    ) -> Ty:
        # Struct constructor: Name(...)
        if isinstance(call.func, TVar) and call.func.name in self.index.structs:
            return self._type_struct_ctor(call, env, allow_capture=allow_capture)

        # Built-in call by name: Foo(...)
        if isinstance(call.func, TVar) and call.func.name in _BUILTIN_DISPATCH:
            return self._type_builtin_call(
                call.func.name,
                call.args,
                env,
                expected=expected,
                allow_capture=allow_capture,
                pos=call.pos,
            )

        # Method call: obj.Method(...)
        if isinstance(call.func, TFieldAccess):
            recv_ty = self._type_expr(call.func.obj, env, allow_capture=allow_capture)
            if isinstance(recv_ty, TyStruct):
                s = self.index.structs.get(recv_ty.name)
                if s is not None and call.func.field in s.methods:
                    mi = s.methods[call.func.field]
                    return self._type_user_call(
                        mi.sig,
                        call.args,
                        env,
                        allow_capture=allow_capture,
                        pos=call.pos,
                        method_receiver=recv_ty,
                    )
            # Otherwise treat as field access yielding a function value.
            fty = self._type_expr(call.func, env, allow_capture=allow_capture)
            if isinstance(fty, TyFunc):
                return self._type_user_call(
                    FnSig(fty.params, fty.ret),
                    call.args,
                    env,
                    allow_capture=allow_capture,
                    pos=call.pos,
                )
            raise TaytshTypeError("call target is not a function", call.pos)

        # Regular function value or named function.
        fty = self._type_expr(call.func, env, allow_capture=allow_capture)
        if isinstance(fty, TyFunc):
            return self._type_user_call(
                FnSig(fty.params, fty.ret),
                call.args,
                env,
                allow_capture=allow_capture,
                pos=call.pos,
            )
        raise TaytshTypeError("call target is not a function", call.pos)

    def _type_struct_ctor(
        self, call: TCall, env: _TypeEnv, *, allow_capture: bool
    ) -> Ty:
        assert isinstance(call.func, TVar)
        s = self.index.structs[call.func.name]
        field_order = [f.name for f in s.fields]
        # Named vs positional args
        has_named = any(a.name is not None for a in call.args)
        has_pos = any(a.name is None for a in call.args)
        if has_named and has_pos:
            raise TaytshTypeError("cannot mix named and positional args", call.pos)
        if not call.args:
            raise TaytshTypeError("struct construction requires all fields", call.pos)
        if has_pos:
            if len(call.args) != len(field_order):
                raise TaytshTypeError("wrong number of constructor args", call.pos)
            for i, arg in enumerate(call.args):
                f = s.field_map[field_order[i]]
                aty = self._type_expr(
                    arg.value, env, expected=f.ty, allow_capture=allow_capture
                )
                if not self._assignable(aty, f.ty):
                    raise TaytshTypeError(
                        f"cannot assign '{aty.display()}' to field '{f.name}: {f.ty.display()}'",
                        arg.pos,
                    )
        else:
            seen: set[str] = set()
            for arg in call.args:
                assert arg.name is not None
                if arg.name not in s.field_map:
                    raise TaytshTypeError(f"unknown field '{arg.name}'", arg.pos)
                if arg.name in seen:
                    raise TaytshTypeError(f"duplicate field '{arg.name}'", arg.pos)
                seen.add(arg.name)
                f = s.field_map[arg.name]
                aty = self._type_expr(
                    arg.value, env, expected=f.ty, allow_capture=allow_capture
                )
                if not self._assignable(aty, f.ty):
                    raise TaytshTypeError(
                        f"cannot assign '{aty.display()}' to field '{f.name}: {f.ty.display()}'",
                        arg.pos,
                    )
            if seen != set(field_order):
                missing = sorted(set(field_order) - seen)
                raise TaytshTypeError(f"missing fields {missing}", call.pos)
        return TyStruct(s.name)

    def _type_user_call(
        self,
        sig: FnSig,
        args: list[TArg],
        env: _TypeEnv,
        *,
        allow_capture: bool,
        pos: Pos,
        method_receiver: Ty | None = None,
    ) -> Ty:
        has_named = any(a.name is not None for a in args)
        if has_named:
            raise TaytshTypeError(
                "named args are only supported for struct construction", pos
            )
        params = list(sig.params)
        if method_receiver is not None:
            if not params:
                raise TaytshTypeError("invalid method signature", pos)
            # First param is self; receiver must be assignable.
            if not self._assignable(method_receiver, params[0]):
                raise TaytshTypeError("method receiver type mismatch", pos)
            params = params[1:]
        if len(args) != len(params):
            raise TaytshTypeError("wrong number of arguments", pos)
        for i, a in enumerate(args):
            aty = self._type_expr(
                a.value, env, expected=params[i], allow_capture=allow_capture
            )
            if not self._assignable(aty, params[i]):
                raise TaytshTypeError(
                    f"cannot pass '{aty.display()}' to param '{params[i].display()}'",
                    a.pos,
                )
        return sig.ret

    def _type_builtin_call(
        self,
        name: str,
        args: list[TArg],
        env: _TypeEnv,
        *,
        expected: Ty | None,
        allow_capture: bool,
        pos: Pos,
    ) -> Ty:
        fn = _BUILTIN_DISPATCH[name]
        return fn.typecheck(
            self, args, env, expected=expected, allow_capture=allow_capture, pos=pos
        )

    def _field_type(self, obj_ty: Ty, field: str, *, pos: Pos) -> Ty:
        if isinstance(obj_ty, TyStruct):
            s = self.index.structs.get(obj_ty.name)
            if s is None:
                raise TaytshTypeError(f"unknown struct type '{obj_ty.name}'", pos)
            if field in s.methods:
                raise TaytshTypeError("bound methods are not values", pos)
            if field not in s.field_map:
                raise TaytshTypeError(f"unknown field '{field}'", pos)
            return s.field_map[field].ty
        if isinstance(obj_ty, TyUnion):
            member_tys: list[Ty] = []
            for m in obj_ty.members:
                if isinstance(m, TyStruct):
                    s = self.index.structs.get(m.name)
                    if s is None or field not in s.field_map:
                        raise TaytshTypeError(
                            f"field '{field}' not present on all union members", pos
                        )
                    member_tys.append(s.field_map[field].ty)
                else:
                    raise TaytshTypeError(
                        f"field access not supported on '{m.display()}'", pos
                    )
            first = member_tys[0]
            if any(not ty_eq(t, first) for t in member_tys[1:]):
                raise TaytshTypeError(
                    f"field '{field}' has inconsistent type across union", pos
                )
            return first
        raise TaytshTypeError(
            f"field access not supported on '{obj_ty.display()}'", pos
        )

    def _type_unary(self, op: str, operand: Ty, *, pos: Pos) -> Ty:
        if op == "!":
            if not ty_eq(operand, TY_BOOL):
                raise TaytshTypeError("! requires bool", pos)
            return TY_BOOL
        if op == "-":
            if (
                ty_eq(operand, TY_INT)
                or ty_eq(operand, TY_FLOAT)
                or ty_eq(operand, TY_BYTE)
            ):
                return operand
            raise TaytshTypeError("- requires int/float/byte", pos)
        if op == "~":
            if ty_eq(operand, TY_INT) or ty_eq(operand, TY_BYTE):
                return operand
            raise TaytshTypeError("~ requires int/byte", pos)
        raise TaytshTypeError(f"unknown unary operator '{op}'", pos)

    def _type_binary(self, op: str, left: Ty, right: Ty, *, pos: Pos) -> Ty:
        if op in ("&&", "||"):
            if not ty_eq(left, TY_BOOL) or not ty_eq(right, TY_BOOL):
                raise TaytshTypeError(f"{op} requires bool operands", pos)
            return TY_BOOL

        if op in ("==", "!="):
            if ty_eq(left, right):
                return TY_BOOL
            # Allow nil checks: (T|nil) == nil
            if ty_is_nil(left) and ty_has_nil(right):
                return TY_BOOL
            if ty_is_nil(right) and ty_has_nil(left):
                return TY_BOOL
            raise TaytshTypeError("==/!= requires same type (or nil check)", pos)

        if op in ("<", "<=", ">", ">="):
            if not ty_eq(left, right):
                raise TaytshTypeError("comparison requires operands of same type", pos)
            if not (
                ty_eq(left, TY_INT)
                or ty_eq(left, TY_FLOAT)
                or ty_eq(left, TY_BYTE)
                or ty_eq(left, TY_RUNE)
                or ty_eq(left, TY_STRING)
            ):
                raise TaytshTypeError("type is not orderable", pos)
            return TY_BOOL

        if op in ("|", "^", "&"):
            if not ty_eq(left, right):
                raise TaytshTypeError("bitwise requires same operand types", pos)
            if not (ty_eq(left, TY_INT) or ty_eq(left, TY_BYTE)):
                raise TaytshTypeError("bitwise requires int/byte", pos)
            return left

        if op in ("<<", ">>"):
            if not (ty_eq(left, TY_INT) or ty_eq(left, TY_BYTE)):
                raise TaytshTypeError("shift requires int/byte on left", pos)
            if not ty_eq(right, TY_INT):
                raise TaytshTypeError("shift amount must be int", pos)
            return left

        if op in ("+", "-", "*", "/", "%"):
            if not ty_eq(left, right):
                raise TaytshTypeError("arithmetic requires operands of same type", pos)
            if not (
                ty_eq(left, TY_INT) or ty_eq(left, TY_FLOAT) or ty_eq(left, TY_BYTE)
            ):
                raise TaytshTypeError("arithmetic requires int/float/byte", pos)
            return left

        raise TaytshTypeError(f"unknown binary operator '{op}'", pos)

    def _nil_refinements(
        self, cond: TExpr, env: _TypeEnv
    ) -> tuple[dict[str, Ty], dict[str, Ty]]:
        if not isinstance(cond, TBinaryOp) or cond.op not in ("==", "!="):
            return ({}, {})

        def var_name(e: TExpr) -> str | None:
            if isinstance(e, TVar):
                return e.name
            return None

        def is_nil(e: TExpr) -> bool:
            return isinstance(e, TNilLit)

        left_var = var_name(cond.left)
        right_var = var_name(cond.right)
        if left_var is not None and is_nil(cond.right):
            name = left_var
        elif right_var is not None and is_nil(cond.left):
            name = right_var
        else:
            return ({}, {})

        try:
            vty = env.get(name, pos=cond.pos)
        except TaytshTypeError:
            return ({}, {})

        if not ty_has_nil(vty):
            return ({}, {})

        non_nil = ty_without_nil(vty)
        if cond.op == "!=":
            return ({name: non_nil}, {name: TY_NIL})
        return ({name: TY_NIL}, {name: non_nil})

    def _has_zero_value(self, typ: Ty) -> bool:
        if ty_eq(typ, TY_INT):
            return True
        if ty_eq(typ, TY_FLOAT):
            return True
        if ty_eq(typ, TY_BOOL):
            return True
        if ty_eq(typ, TY_BYTE):
            return True
        if ty_eq(typ, TY_BYTES):
            return True
        if ty_eq(typ, TY_STRING):
            return True
        if ty_eq(typ, TY_RUNE):
            return True
        if isinstance(typ, TyList):
            return True
        if isinstance(typ, TyMap):
            return True
        if isinstance(typ, TySet):
            return True
        if isinstance(typ, TyTuple):
            return all(self._has_zero_value(t) for t in typ.elements)
        if isinstance(typ, TyUnion):
            return any(ty_is_nil(m) for m in typ.members)
        if ty_eq(typ, TY_OBJ):
            return True
        return False

    def _assignable(self, src: Ty, dst: Ty) -> bool:
        if ty_eq(src, dst):
            return True
        if ty_is_obj(dst):
            return True
        if ty_is_nil(src):
            return ty_has_nil(dst)
        # Union destination: any member works.
        if isinstance(dst, TyUnion):
            return any(self._assignable(src, m) for m in dst.members)
        # Union source: all members must be assignable.
        if isinstance(src, TyUnion):
            return all(self._assignable(m, dst) for m in src.members)
        # Struct -> interface if implements.
        if isinstance(src, TyStruct) and isinstance(dst, TyInterface):
            return src.name in self.index.interfaces[dst.name].implementors
        return False


# ============================================================
# Built-in function dispatch (typechecking only for now)
# ============================================================


@dataclass(frozen=True)
class _Builtin:
    sig: FnSig
    typecheck: Callable[["TypeChecker", list[TArg], _TypeEnv, Ty | None, bool, Pos], Ty]

    def ty(self) -> TyFunc:
        return self.sig.ty()


def _builtin_simple(sig: FnSig) -> _Builtin:
    def _tc(
        tc: TypeChecker,
        args: list[TArg],
        env: _TypeEnv,
        expected: Ty | None,
        allow_capture: bool,
        pos: Pos,
    ) -> Ty:
        if any(a.name is not None for a in args):
            raise TaytshTypeError("named args not supported for this call", pos)
        if len(args) != len(sig.params):
            raise TaytshTypeError("wrong number of arguments", pos)
        for i, a in enumerate(args):
            aty = tc._type_expr(
                a.value, env, expected=sig.params[i], allow_capture=allow_capture
            )
            if not tc._assignable(aty, sig.params[i]):
                raise TaytshTypeError("argument type mismatch", a.pos)
        return sig.ret

    return _Builtin(sig, _tc)


def _tc_len(
    tc: TypeChecker,
    args: list[TArg],
    env: _TypeEnv,
    expected: Ty | None,
    allow_capture: bool,
    pos: Pos,
) -> Ty:
    if any(a.name is not None for a in args):
        raise TaytshTypeError("named args not supported for Len", pos)
    if len(args) != 1:
        raise TaytshTypeError("Len expects 1 argument", pos)
    aty = tc._type_expr(args[0].value, env, allow_capture=allow_capture)
    if (
        isinstance(aty, (TyList, TyMap, TySet))
        or ty_eq(aty, TY_STRING)
        or ty_eq(aty, TY_BYTES)
    ):
        return TY_INT
    raise TaytshTypeError("Len not supported for this type", pos)


def _tc_map_ctor(
    tc: TypeChecker,
    args: list[TArg],
    env: _TypeEnv,
    expected: Ty | None,
    allow_capture: bool,
    pos: Pos,
) -> Ty:
    if args:
        raise TaytshTypeError("Map() takes no arguments", pos)
    if isinstance(expected, TyMap):
        return expected
    raise TaytshTypeError("cannot infer type for Map()", pos)


def _tc_set_ctor(
    tc: TypeChecker,
    args: list[TArg],
    env: _TypeEnv,
    expected: Ty | None,
    allow_capture: bool,
    pos: Pos,
) -> Ty:
    if args:
        raise TaytshTypeError("Set() takes no arguments", pos)
    if isinstance(expected, TySet):
        return expected
    raise TaytshTypeError("cannot infer type for Set()", pos)


def _tc_get(
    tc: TypeChecker,
    args: list[TArg],
    env: _TypeEnv,
    expected: Ty | None,
    allow_capture: bool,
    pos: Pos,
) -> Ty:
    if any(a.name is not None for a in args):
        raise TaytshTypeError("named args not supported for Get", pos)
    if len(args) not in (2, 3):
        raise TaytshTypeError("Get expects 2 or 3 arguments", pos)
    mty = tc._type_expr(args[0].value, env, allow_capture=allow_capture)
    if not isinstance(mty, TyMap):
        raise TaytshTypeError("Get expects a map", pos)
    kty = tc._type_expr(
        args[1].value, env, expected=mty.key, allow_capture=allow_capture
    )
    if not tc._assignable(kty, mty.key):
        raise TaytshTypeError("Get key type mismatch", args[1].pos)
    if len(args) == 2:
        return ty_union([mty.value, TY_NIL])
    dty = tc._type_expr(
        args[2].value, env, expected=mty.value, allow_capture=allow_capture
    )
    if not tc._assignable(dty, mty.value):
        raise TaytshTypeError("Get default type mismatch", args[2].pos)
    return mty.value


def _tc_contains(
    tc: TypeChecker,
    args: list[TArg],
    env: _TypeEnv,
    expected: Ty | None,
    allow_capture: bool,
    pos: Pos,
) -> Ty:
    if any(a.name is not None for a in args):
        raise TaytshTypeError("named args not supported for Contains", pos)
    if len(args) != 2:
        raise TaytshTypeError("Contains expects 2 arguments", pos)
    aty = tc._type_expr(args[0].value, env, allow_capture=allow_capture)
    bty = tc._type_expr(args[1].value, env, allow_capture=allow_capture)
    if isinstance(aty, TyList) and tc._assignable(bty, aty.element):
        return TY_BOOL
    if isinstance(aty, TyMap) and tc._assignable(bty, aty.key):
        return TY_BOOL
    if isinstance(aty, TySet) and tc._assignable(bty, aty.element):
        return TY_BOOL
    if ty_eq(aty, TY_STRING) and ty_eq(bty, TY_STRING):
        return TY_BOOL
    raise TaytshTypeError("Contains argument types invalid", pos)


def _tc_tostring(
    tc: TypeChecker,
    args: list[TArg],
    env: _TypeEnv,
    expected: Ty | None,
    allow_capture: bool,
    pos: Pos,
) -> Ty:
    if any(a.name is not None for a in args):
        raise TaytshTypeError("named args not supported for ToString", pos)
    if len(args) != 1:
        raise TaytshTypeError("ToString expects 1 argument", pos)
    _ = tc._type_expr(args[0].value, env, allow_capture=allow_capture)
    return TY_STRING


def _tc_unwrap(
    tc: TypeChecker,
    args: list[TArg],
    env: _TypeEnv,
    expected: Ty | None,
    allow_capture: bool,
    pos: Pos,
) -> Ty:
    if any(a.name is not None for a in args):
        raise TaytshTypeError("named args not supported for Unwrap", pos)
    if len(args) != 1:
        raise TaytshTypeError("Unwrap expects 1 argument", pos)
    aty = tc._type_expr(args[0].value, env, allow_capture=allow_capture)
    if not ty_has_nil(aty):
        raise TaytshTypeError("Unwrap expects an optional/union-with-nil", pos)
    return ty_without_nil(aty)


def _tc_assert(
    tc: TypeChecker,
    args: list[TArg],
    env: _TypeEnv,
    expected: Ty | None,
    allow_capture: bool,
    pos: Pos,
) -> Ty:
    if any(a.name is not None for a in args):
        raise TaytshTypeError("named args not supported for Assert", pos)
    if len(args) not in (1, 2):
        raise TaytshTypeError("Assert expects 1 or 2 arguments", pos)
    cty = tc._type_expr(
        args[0].value, env, expected=TY_BOOL, allow_capture=allow_capture
    )
    if not ty_eq(cty, TY_BOOL):
        raise TaytshTypeError("Assert condition must be bool", args[0].pos)
    if len(args) == 2:
        mty = tc._type_expr(
            args[1].value, env, expected=TY_STRING, allow_capture=allow_capture
        )
        if not ty_eq(mty, TY_STRING):
            raise TaytshTypeError("Assert message must be string", args[1].pos)
    return TY_VOID


_BUILTIN_DISPATCH: dict[str, _Builtin] = {
    # Minimal builtins needed for typing most programs; runtime impl added later.
    "ToString": _Builtin(FnSig((TY_OBJ,), TY_STRING), _tc_tostring),
    "Len": _Builtin(FnSig((TY_OBJ,), TY_INT), _tc_len),
    "Map": _Builtin(FnSig((), TyMap(TY_OBJ, TY_OBJ)), _tc_map_ctor),
    "Set": _Builtin(FnSig((), TySet(TY_OBJ)), _tc_set_ctor),
    "Get": _Builtin(FnSig((TY_OBJ, TY_OBJ), TY_OBJ), _tc_get),
    "Contains": _Builtin(FnSig((TY_OBJ, TY_OBJ), TY_BOOL), _tc_contains),
    "Unwrap": _Builtin(FnSig((TY_OBJ,), TY_OBJ), _tc_unwrap),
    "Assert": _Builtin(FnSig((TY_BOOL,), TY_VOID), _tc_assert),
}


# ============================================================
# Evaluation
# ============================================================


def _value_eq(a: Value, b: Value) -> bool:
    # Concrete-type equality (union/obj equality is by held concrete type).
    if type(a) is not type(b):
        return False
    if isinstance(a, VNil):
        return True
    if isinstance(a, VBool):
        return a.value == cast(VBool, b).value
    if isinstance(a, VInt):
        return a.value == cast(VInt, b).value
    if isinstance(a, VFloat):
        return a.value == cast(VFloat, b).value
    if isinstance(a, VByte):
        return a.value == cast(VByte, b).value
    if isinstance(a, VBytes):
        return a.value == cast(VBytes, b).value
    if isinstance(a, VString):
        return a.value == cast(VString, b).value
    if isinstance(a, VRune):
        return a.value == cast(VRune, b).value
    if isinstance(a, VEnum):
        return a == b
    if isinstance(a, VTuple):
        other = cast(VTuple, b)
        if not ty_eq(a.typ, other.typ):
            return False
        if len(a.elements) != len(other.elements):
            return False
        return all(_value_eq(x, y) for x, y in zip(a.elements, other.elements))
    if isinstance(a, VList):
        other = cast(VList, b)
        if not ty_eq(a.typ, other.typ):
            return False
        if len(a.elements) != len(other.elements):
            return False
        return all(_value_eq(x, y) for x, y in zip(a.elements, other.elements))
    if isinstance(a, VMap):
        other = cast(VMap, b)
        if not ty_eq(a.typ, other.typ):
            return False
        if len(a.entries) != len(other.entries):
            return False
        for k, v in a.entries.items():
            if k not in other.entries:
                return False
            if not _value_eq(v, other.entries[k]):
                return False
        return True
    if isinstance(a, VSet):
        other = cast(VSet, b)
        if not ty_eq(a.typ, other.typ):
            return False
        if len(a.elements) != len(other.elements):
            return False
        return a.elements == other.elements
    if isinstance(a, VStruct):
        other = cast(VStruct, b)
        if a.struct_name != other.struct_name:
            return False
        if a.fields.keys() != other.fields.keys():
            return False
        for k in a.fields.keys():
            if not _value_eq(a.fields[k], other.fields[k]):
                return False
        return True
    if isinstance(a, VFunc):
        other = cast(VFunc, b)
        return ty_eq(a.typ, other.typ) and a.name == other.name and a.call is other.call
    raise TaytshRuntimeFault("unsupported equality", None)


def _as_hashable(v: Value) -> HashableValue:
    if isinstance(v, HashableValue):
        return v
    raise TaytshRuntimeFault("value is not hashable", None)


def _int_divmod_trunc(a: int, b: int) -> tuple[int, int]:
    if b == 0:
        raise ZeroDivisionError
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    r = a - q * b
    return (q, r)


@dataclass
class _Binding:
    ty: Ty
    value: Value


class _RuntimeEnv:
    def __init__(self) -> None:
        self._scopes: list[dict[str, _Binding]] = []

    def push_scope(self) -> None:
        self._scopes.append({})

    def pop_scope(self) -> None:
        self._scopes.pop()

    def bind(self, name: str, typ: Ty, value: Value) -> None:
        if name == "_":
            return
        if not self._scopes:
            raise RuntimeError("no scope")
        self._scopes[-1][name] = _Binding(typ, value)

    def lookup(self, name: str) -> _Binding:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        raise TaytshRuntimeFault(f"unknown name '{name}'", None)

    def get(self, name: str) -> Value:
        if name == "_":
            raise TaytshRuntimeFault("cannot read discard '_'", None)
        return self.lookup(name).value

    def get_ty(self, name: str) -> Ty:
        if name == "_":
            raise TaytshRuntimeFault("cannot read discard '_'", None)
        return self.lookup(name).ty

    def set(self, name: str, value: Value) -> None:
        if name == "_":
            return
        for scope in reversed(self._scopes):
            if name in scope:
                scope[name] = _Binding(scope[name].ty, value)
                return
        raise TaytshRuntimeFault(f"unknown name '{name}'", None)


class _LValueRef:
    def __init__(self, typ: Ty):
        self.typ = typ

    def get(self) -> Value:  # pragma: no cover
        raise NotImplementedError

    def set(self, value: Value) -> None:  # pragma: no cover
        raise NotImplementedError


class _VarRef(_LValueRef):
    def __init__(self, env: _RuntimeEnv, name: str, typ: Ty):
        super().__init__(typ)
        self._env = env
        self._name = name

    def get(self) -> Value:
        return self._env.get(self._name)

    def set(self, value: Value) -> None:
        self._env.set(self._name, value)


class _FieldRef(_LValueRef):
    def __init__(self, obj: VStruct, field: str, typ: Ty):
        super().__init__(typ)
        self._obj = obj
        self._field = field

    def get(self) -> Value:
        return self._obj.fields[self._field]

    def set(self, value: Value) -> None:
        self._obj.fields[self._field] = value


class _ListIndexRef(_LValueRef):
    def __init__(self, obj: VList, index: int, typ: Ty):
        super().__init__(typ)
        self._obj = obj
        self._index = index

    def get(self) -> Value:
        return self._obj.elements[self._index]

    def set(self, value: Value) -> None:
        self._obj.elements[self._index] = value


class _MapIndexRef(_LValueRef):
    def __init__(self, obj: VMap, key: HashableValue, typ: Ty):
        super().__init__(typ)
        self._obj = obj
        self._key = key

    def get(self) -> Value:
        if self._key not in self._obj.entries:
            raise KeyError
        return self._obj.entries[self._key]

    def set(self, value: Value) -> None:
        self._obj.entries[self._key] = value


class Runtime:
    def __init__(
        self,
        module: TModule,
        index: ModuleIndex,
        tc: TypeChecker,
        *,
        stdin: bytes,
        args: Sequence[str],
        env: Mapping[str, str],
    ):
        self.module = module
        self.index = index
        self.tc = tc
        self.stdin = _Input(stdin)
        self.args = list(args)
        self.env = dict(env)
        self.stdout = bytearray()
        self.stderr = bytearray()

        self._fn_values: dict[str, VFunc] = {}
        for name, info in self.index.funcs.items():
            self._fn_values[name] = self._make_user_fn(name, info.decl, info.sig)

        self._builtin_values: dict[str, VFunc] = {}
        for name, b in _BUILTIN_DISPATCH.items():
            if name in _BUILTIN_RUNTIME:
                self._builtin_values[name] = VFunc(
                    b.sig.ty(), name, lambda av, n=name: _BUILTIN_RUNTIME[n](self, av)
                )

    # ---- Errors / throwing -------------------------------------------------

    def _err(self, name: str, message: str) -> VStruct:
        return VStruct(name, {"message": VString(message)})

    def _throw_err(self, name: str, message: str) -> None:
        raise _Throw(self._err(name, message))

    # ---- Zero values -------------------------------------------------------

    def zero_value(self, typ: Ty) -> Value:
        if ty_eq(typ, TY_INT):
            return VInt(0)
        if ty_eq(typ, TY_FLOAT):
            return VFloat(0.0)
        if ty_eq(typ, TY_BOOL):
            return VBool(False)
        if ty_eq(typ, TY_BYTE):
            return VByte(0)
        if ty_eq(typ, TY_BYTES):
            return VBytes(b"")
        if ty_eq(typ, TY_STRING):
            return VString("")
        if ty_eq(typ, TY_RUNE):
            return VRune("\0")
        if ty_eq(typ, TY_OBJ) or ty_is_nil(typ):
            return VNil()
        if isinstance(typ, TyList):
            return VList([], typ)
        if isinstance(typ, TyMap):
            return VMap({}, typ)
        if isinstance(typ, TySet):
            return VSet(set(), typ)
        if isinstance(typ, TyTuple):
            elems = tuple(self.zero_value(t) for t in typ.elements)
            return VTuple(elems, typ)
        if isinstance(typ, TyUnion):
            if any(ty_is_nil(m) for m in typ.members):
                return VNil()
        raise TaytshRuntimeFault(f"type '{typ.display()}' has no zero value", None)

    # ---- Running -----------------------------------------------------------

    def run_main(self) -> RunResult:
        try:
            self._call_fn(
                self.index.funcs["Main"].decl, self.index.funcs["Main"].sig, []
            )
            return RunResult(0, bytes(self.stdout), bytes(self.stderr))
        except _Exit as e:
            return RunResult(e.code, bytes(self.stdout), bytes(self.stderr))
        except _Throw as t:
            # Uncaught exception: best-effort message to stderr.
            msg = t.value.to_string()
            self.stderr.extend((msg + "\n").encode("utf-8"))
            return RunResult(1, bytes(self.stdout), bytes(self.stderr))

    # ---- Functions ---------------------------------------------------------

    def _make_user_fn(self, name: str, decl: TFnDecl, sig: FnSig) -> VFunc:
        return VFunc(sig.ty(), name, lambda args: self._call_fn(decl, sig, args))

    def _call_fn(self, decl: TFnDecl, sig: FnSig, args: list[Value]) -> Value:
        env = _RuntimeEnv()
        env.push_scope()
        # Bind params
        for i, p in enumerate(decl.params):
            if p.typ is None:
                env.bind("self", sig.params[i], args[i])
            else:
                env.bind(p.name, sig.params[i], args[i])

        try:
            self._eval_block(decl.body, env, fn_ret=sig.ret)
        except _Return as r:
            if ty_eq(sig.ret, TY_VOID):
                return VNil()
            if r.value is None:
                raise TaytshRuntimeFault("missing return value", decl.pos)
            return r.value

        if ty_eq(sig.ret, TY_VOID):
            return VNil()
        raise TaytshRuntimeFault("function fell off without returning", decl.pos)

    # ---- Statements --------------------------------------------------------

    def _eval_block(self, stmts: list[TStmt], env: _RuntimeEnv, *, fn_ret: Ty) -> None:
        env.push_scope()
        try:
            for st in stmts:
                self._eval_stmt(st, env, fn_ret=fn_ret)
        finally:
            env.pop_scope()

    def _eval_stmt(self, st: TStmt, env: _RuntimeEnv, *, fn_ret: Ty) -> None:
        if isinstance(st, TLetStmt):
            vty = self.tc.resolve_type(st.typ, pos=st.pos)
            if st.value is None:
                env.bind(st.name, vty, self.zero_value(vty))
                return
            val = self._eval_expr(st.value, env, expected=vty)
            env.bind(st.name, vty, val)
            return

        if isinstance(st, TAssignStmt):
            ref = self._eval_lvalue_ref(st.target, env)
            val = self._eval_expr(st.value, env, expected=ref.typ)
            ref.set(val)
            return

        if isinstance(st, TOpAssignStmt):
            ref = self._eval_lvalue_ref(st.target, env)
            try:
                cur = ref.get()
            except KeyError:
                self._throw_err("KeyError", "missing key")
            rhs = self._eval_expr(st.value, env, expected=ref.typ)
            res = self._eval_binary(st.op[:-1], cur, rhs, pos=st.pos)
            ref.set(res)
            return

        if isinstance(st, TTupleAssignStmt):
            refs = [self._eval_lvalue_ref(t, env) for t in st.targets]
            rhs = self._eval_expr(
                st.value, env, expected=TyTuple(tuple(r.typ for r in refs))
            )
            if not isinstance(rhs, VTuple):
                raise TaytshRuntimeFault("tuple assignment rhs not a tuple", st.pos)
            if len(rhs.elements) != len(st.targets):
                raise TaytshRuntimeFault("tuple arity mismatch", st.pos)
            for i, ref in enumerate(refs):
                ref.set(rhs.elements[i])
            return

        if isinstance(st, TReturnStmt):
            if st.value is None:
                raise _Return(None)
            raise _Return(self._eval_expr(st.value, env, expected=fn_ret))

        if isinstance(st, TBreakStmt):
            raise _Break()
        if isinstance(st, TContinueStmt):
            raise _Continue()

        if isinstance(st, TThrowStmt):
            raise _Throw(self._eval_expr(st.expr, env))

        if isinstance(st, TExprStmt):
            _ = self._eval_expr(st.expr, env)
            return

        if isinstance(st, TIfStmt):
            cond = self._eval_expr(st.cond, env)
            if not isinstance(cond, VBool):
                raise TaytshRuntimeFault("if condition not bool", st.pos)
            if cond.value:
                self._eval_block(st.then_body, env, fn_ret=fn_ret)
            elif st.else_body is not None:
                self._eval_block(cast(list[TStmt], st.else_body), env, fn_ret=fn_ret)
            return

        if isinstance(st, TWhileStmt):
            while True:
                cond = self._eval_expr(st.cond, env)
                if not isinstance(cond, VBool):
                    raise TaytshRuntimeFault("while condition not bool", st.pos)
                if not cond.value:
                    return
                try:
                    self._eval_block(st.body, env, fn_ret=fn_ret)
                except _Continue:
                    continue
                except _Break:
                    return

        if isinstance(st, TForStmt):
            self._eval_for(st, env, fn_ret=fn_ret)
            return

        if isinstance(st, TMatchStmt):
            self._eval_match(st, env, fn_ret=fn_ret)
            return

        if isinstance(st, TTryStmt):
            self._eval_try(st, env, fn_ret=fn_ret)
            return

        raise TaytshRuntimeFault("unsupported statement", st.pos)

    def _eval_try(self, st: TTryStmt, env: _RuntimeEnv, *, fn_ret: Ty) -> None:
        pending: _Signal | None = None
        try:
            self._eval_block(st.body, env, fn_ret=fn_ret)
        except _Signal as s:
            pending = s
            if isinstance(s, _Throw):
                handled = False
                for c in st.catches:
                    if any(
                        self._matches_type(s.value, self.tc.resolve_type(t, pos=t.pos))
                        for t in c.types
                    ):
                        handled = True
                        pending = None
                        env.push_scope()
                        try:
                            # Catch binding static type is union of types, but runtime stores the concrete value.
                            c_ty = ty_union(
                                [self.tc.resolve_type(t, pos=t.pos) for t in c.types]
                            )
                            env.bind(c.name, c_ty, s.value)
                            self._eval_block(c.body, env, fn_ret=fn_ret)
                        except _Signal as inner:
                            pending = inner
                        finally:
                            env.pop_scope()
                        break
                if not handled:
                    pending = s
        finally:
            if st.finally_body is not None:
                try:
                    self._eval_block(st.finally_body, env, fn_ret=fn_ret)
                except _Signal as fin:
                    pending = fin
        if pending is not None:
            raise pending

    def _eval_match(self, st: TMatchStmt, env: _RuntimeEnv, *, fn_ret: Ty) -> None:
        v = self._eval_expr(st.expr, env)
        for c in st.cases:
            pat = c.pattern
            if isinstance(pat, TPatternNil):
                if isinstance(v, VNil):
                    self._eval_block(c.body, env, fn_ret=fn_ret)
                    return
            elif isinstance(pat, TPatternEnum):
                if (
                    isinstance(v, VEnum)
                    and v.enum_name == pat.enum_name
                    and v.variant == pat.variant
                ):
                    self._eval_block(c.body, env, fn_ret=fn_ret)
                    return
            elif isinstance(pat, TPatternType):
                case_ty = self.tc.resolve_type(pat.type_name, pos=pat.pos)
                if self._matches_type(v, case_ty):
                    env.push_scope()
                    try:
                        env.bind(pat.name, case_ty, v)
                        self._eval_block(c.body, env, fn_ret=fn_ret)
                    finally:
                        env.pop_scope()
                    return
        if st.default is not None:
            if st.default.name is None:
                self._eval_block(st.default.body, env, fn_ret=fn_ret)
                return
            env.push_scope()
            try:
                env.bind(st.default.name, TY_OBJ, v)
                self._eval_block(st.default.body, env, fn_ret=fn_ret)
            finally:
                env.pop_scope()
            return
        raise TaytshRuntimeFault("non-exhaustive match at runtime", st.pos)

    def _eval_for(self, st: TForStmt, env: _RuntimeEnv, *, fn_ret: Ty) -> None:
        if isinstance(st.iterable, TRange):
            ints = [self._eval_expr(a, env) for a in st.iterable.args]
            vals = []
            for iv in ints:
                if not isinstance(iv, VInt):
                    raise TaytshRuntimeFault(
                        "range() args must be int", st.iterable.pos
                    )
                vals.append(iv.value)
            if len(vals) == 1:
                start, end, step = 0, vals[0], 1
            elif len(vals) == 2:
                start, end = vals
                step = 1
            else:
                start, end, step = vals
            if step == 0:
                self._throw_err("ValueError", "range() step must be nonzero")
            i = start

            def cond(x: int) -> bool:
                return x < end if step > 0 else x > end

            while cond(i):
                env.push_scope()
                try:
                    env.bind(st.binding[0], TY_INT, VInt(i))
                    self._eval_block(st.body, env, fn_ret=fn_ret)
                except _Continue:
                    pass
                except _Break:
                    return
                finally:
                    env.pop_scope()
                i += step
            return

        it = self._eval_expr(st.iterable, env)

        if isinstance(it, VList):
            snapshot = list(it.elements)
            for idx, val in enumerate(snapshot):
                env.push_scope()
                try:
                    if len(st.binding) == 1:
                        env.bind(st.binding[0], it.typ.element, val)
                    else:
                        env.bind(st.binding[0], TY_INT, VInt(idx))
                        env.bind(st.binding[1], it.typ.element, val)
                    self._eval_block(st.body, env, fn_ret=fn_ret)
                except _Continue:
                    continue
                except _Break:
                    return
                finally:
                    env.pop_scope()
            return

        if isinstance(it, VString):
            snapshot = list(it.value)
            for idx, ch in enumerate(snapshot):
                env.push_scope()
                try:
                    if len(st.binding) == 1:
                        env.bind(st.binding[0], TY_RUNE, VRune(ch))
                    else:
                        env.bind(st.binding[0], TY_INT, VInt(idx))
                        env.bind(st.binding[1], TY_RUNE, VRune(ch))
                    self._eval_block(st.body, env, fn_ret=fn_ret)
                except _Continue:
                    continue
                except _Break:
                    return
                finally:
                    env.pop_scope()
            return

        if isinstance(it, VBytes):
            snapshot = list(it.value)
            for idx, b in enumerate(snapshot):
                env.push_scope()
                try:
                    if len(st.binding) == 1:
                        env.bind(st.binding[0], TY_BYTE, VByte(b))
                    else:
                        env.bind(st.binding[0], TY_INT, VInt(idx))
                        env.bind(st.binding[1], TY_BYTE, VByte(b))
                    self._eval_block(st.body, env, fn_ret=fn_ret)
                except _Continue:
                    continue
                except _Break:
                    return
                finally:
                    env.pop_scope()
            return

        if isinstance(it, VMap):
            snapshot = list(it.entries.items())
            for k, v in snapshot:
                env.push_scope()
                try:
                    if len(st.binding) == 1:
                        env.bind(st.binding[0], it.typ.key, k)
                    else:
                        env.bind(st.binding[0], it.typ.key, k)
                        env.bind(st.binding[1], it.typ.value, v)
                    self._eval_block(st.body, env, fn_ret=fn_ret)
                except _Continue:
                    continue
                except _Break:
                    return
                finally:
                    env.pop_scope()
            return

        if isinstance(it, VSet):
            if len(st.binding) != 1:
                raise TaytshRuntimeFault(
                    "set iteration does not support two bindings", st.pos
                )
            snapshot = list(it.elements)
            for v in snapshot:
                env.push_scope()
                try:
                    env.bind(st.binding[0], it.typ.element, v)
                    self._eval_block(st.body, env, fn_ret=fn_ret)
                except _Continue:
                    continue
                except _Break:
                    return
                finally:
                    env.pop_scope()
            return

        raise TaytshRuntimeFault("for over non-iterable", st.pos)

    # ---- Lvalues / expressions --------------------------------------------

    def _eval_lvalue_ref(self, expr: TExpr, env: _RuntimeEnv) -> _LValueRef:
        if isinstance(expr, TVar):
            return _VarRef(env, expr.name, env.get_ty(expr.name))
        if isinstance(expr, TFieldAccess):
            obj = self._eval_expr(expr.obj, env)
            if not isinstance(obj, VStruct):
                raise TaytshRuntimeFault("field access on non-struct", expr.pos)
            sinfo = self.index.structs.get(obj.struct_name)
            if sinfo is None or expr.field not in sinfo.field_map:
                raise TaytshRuntimeFault("unknown field", expr.pos)
            return _FieldRef(obj, expr.field, sinfo.field_map[expr.field].ty)
        if isinstance(expr, TIndex):
            obj = self._eval_expr(expr.obj, env)
            idx = self._eval_expr(expr.index, env)
            if isinstance(obj, VList):
                if not isinstance(idx, VInt):
                    raise TaytshRuntimeFault("list index not int", expr.pos)
                if idx.value < 0 or idx.value >= len(obj.elements):
                    self._throw_err("IndexError", "index out of bounds")
                return _ListIndexRef(obj, idx.value, obj.typ.element)
            if isinstance(obj, VMap):
                key = _as_hashable(idx)
                return _MapIndexRef(obj, key, obj.typ.value)
            raise TaytshRuntimeFault("index assignment only for list/map", expr.pos)
        raise TaytshRuntimeFault("invalid assignment target", expr.pos)

    def _eval_expr(
        self, expr: TExpr, env: _RuntimeEnv, *, expected: Ty | None = None
    ) -> Value:
        if isinstance(expr, TIntLit):
            return VInt(expr.value)
        if isinstance(expr, TFloatLit):
            return VFloat(expr.value)
        if isinstance(expr, TByteLit):
            return VByte(expr.value)
        if isinstance(expr, TStringLit):
            return VString(expr.value)
        if isinstance(expr, TRuneLit):
            return VRune(expr.value)
        if isinstance(expr, TBytesLit):
            return VBytes(expr.value)
        if isinstance(expr, TBoolLit):
            return VBool(expr.value)
        if isinstance(expr, TNilLit):
            return VNil()

        if isinstance(expr, TVar):
            if env._scopes and any(expr.name in s for s in env._scopes):
                return env.get(expr.name)
            if expr.name in self._fn_values:
                return self._fn_values[expr.name]
            if expr.name in self._builtin_values:
                return self._builtin_values[expr.name]
            raise TaytshRuntimeFault(f"unknown name '{expr.name}'", expr.pos)

        if isinstance(expr, TUnaryOp):
            operand = self._eval_expr(expr.operand, env)
            if expr.op == "!":
                if not isinstance(operand, VBool):
                    raise TaytshRuntimeFault("! operand not bool", expr.pos)
                return VBool(not operand.value)
            if expr.op == "-":
                if isinstance(operand, VInt):
                    return VInt(-operand.value)
                if isinstance(operand, VFloat):
                    return VFloat(-operand.value)
                if isinstance(operand, VByte):
                    return VByte((-operand.value) & 0xFF)
                raise TaytshRuntimeFault("- operand type", expr.pos)
            if expr.op == "~":
                if isinstance(operand, VInt):
                    return VInt(~operand.value)
                if isinstance(operand, VByte):
                    return VByte((~operand.value) & 0xFF)
                raise TaytshRuntimeFault("~ operand type", expr.pos)
            raise TaytshRuntimeFault("unknown unary op", expr.pos)

        if isinstance(expr, TBinaryOp):
            if expr.op == "&&":
                left = self._eval_expr(expr.left, env)
                if not isinstance(left, VBool):
                    raise TaytshRuntimeFault("&& left not bool", expr.pos)
                if not left.value:
                    return VBool(False)
                right = self._eval_expr(expr.right, env)
                if not isinstance(right, VBool):
                    raise TaytshRuntimeFault("&& right not bool", expr.pos)
                return VBool(right.value)
            if expr.op == "||":
                left = self._eval_expr(expr.left, env)
                if not isinstance(left, VBool):
                    raise TaytshRuntimeFault("|| left not bool", expr.pos)
                if left.value:
                    return VBool(True)
                right = self._eval_expr(expr.right, env)
                if not isinstance(right, VBool):
                    raise TaytshRuntimeFault("|| right not bool", expr.pos)
                return VBool(right.value)
            left = self._eval_expr(expr.left, env)
            right = self._eval_expr(expr.right, env)
            return self._eval_binary(expr.op, left, right, pos=expr.pos)

        if isinstance(expr, TTernary):
            cond = self._eval_expr(expr.cond, env)
            if not isinstance(cond, VBool):
                raise TaytshRuntimeFault("ternary condition not bool", expr.pos)
            return self._eval_expr(
                expr.then_expr if cond.value else expr.else_expr, env, expected=expected
            )

        if isinstance(expr, TTupleAccess):
            obj = self._eval_expr(expr.obj, env)
            if not isinstance(obj, VTuple):
                raise TaytshRuntimeFault("tuple access on non-tuple", expr.pos)
            return obj.elements[expr.index]

        if isinstance(expr, TFieldAccess):
            if isinstance(expr.obj, TVar) and expr.obj.name in self.index.enums:
                enum = self.index.enums[expr.obj.name]
                if expr.field not in enum.variants:
                    raise TaytshRuntimeFault("unknown enum variant", expr.pos)
                return VEnum(enum.name, expr.field)
            obj = self._eval_expr(expr.obj, env)
            if not isinstance(obj, VStruct):
                raise TaytshRuntimeFault("field access on non-struct", expr.pos)
            if expr.field not in obj.fields:
                raise TaytshRuntimeFault("unknown field", expr.pos)
            return obj.fields[expr.field]

        if isinstance(expr, TIndex):
            obj = self._eval_expr(expr.obj, env)
            idx = self._eval_expr(expr.index, env)
            return self._eval_index(obj, idx, pos=expr.pos)

        if isinstance(expr, TSlice):
            obj = self._eval_expr(expr.obj, env)
            low = self._eval_expr(expr.low, env)
            high = self._eval_expr(expr.high, env)
            if not isinstance(low, VInt) or not isinstance(high, VInt):
                raise TaytshRuntimeFault("slice bounds not int", expr.pos)
            return self._eval_slice(obj, low.value, high.value, pos=expr.pos)

        if isinstance(expr, TListLit):
            if expected is None:
                inferred = self.tc.expr_types.get(id(expr))
                if isinstance(inferred, TyList):
                    expected = inferred
            if isinstance(expected, TyList):
                elems = [
                    self._eval_expr(e, env, expected=expected.element)
                    for e in expr.elements
                ]
                typ = expected
            else:
                elems = [self._eval_expr(e, env) for e in expr.elements]
                if not elems:
                    raise TaytshRuntimeFault("cannot infer list type", expr.pos)
                typ = TyList(elems[0].ty())
            return VList(elems, typ)

        if isinstance(expr, TMapLit):
            if expected is None:
                inferred = self.tc.expr_types.get(id(expr))
                if isinstance(inferred, TyMap):
                    expected = inferred
            entries: dict[HashableValue, Value] = {}
            if isinstance(expected, TyMap):
                for k, v in expr.entries:
                    kk = _as_hashable(self._eval_expr(k, env, expected=expected.key))
                    vv = self._eval_expr(v, env, expected=expected.value)
                    entries[kk] = vv
                typ = expected
            else:
                for k, v in expr.entries:
                    kk = _as_hashable(self._eval_expr(k, env))
                    vv = self._eval_expr(v, env)
                    entries[kk] = vv
                first_k = next(iter(entries.keys()))
                first_v = next(iter(entries.values()))
                typ = TyMap(first_k.ty(), first_v.ty())
            return VMap(entries, typ)

        if isinstance(expr, TSetLit):
            if expected is None:
                inferred = self.tc.expr_types.get(id(expr))
                if isinstance(inferred, TySet):
                    expected = inferred
            if isinstance(expected, TySet):
                elems = {
                    _as_hashable(self._eval_expr(e, env, expected=expected.element))
                    for e in expr.elements
                }
                typ = expected
            else:
                elems = {_as_hashable(self._eval_expr(e, env)) for e in expr.elements}
                first = next(iter(elems))
                typ = TySet(first.ty())
            return VSet(elems, typ)

        if isinstance(expr, TTupleLit):
            if expected is None:
                inferred = self.tc.expr_types.get(id(expr))
                if isinstance(inferred, TyTuple):
                    expected = inferred
            if isinstance(expected, TyTuple):
                elems = tuple(
                    self._eval_expr(e, env, expected=expected.elements[i])
                    for i, e in enumerate(expr.elements)
                )
                typ = expected
            else:
                elems = tuple(self._eval_expr(e, env) for e in expr.elements)
                typ = TyTuple(tuple(e.ty() for e in elems))
            return VTuple(elems, typ)

        if isinstance(expr, TFnLit):
            sig = self.tc._fn_lit_sig(expr, method_self=None)
            return VFunc(sig.ty(), None, lambda av: self._call_fn_lit(expr, sig, av))

        if isinstance(expr, TCall):
            return self._eval_call(expr, env, expected=expected)

        raise TaytshRuntimeFault("unsupported expression", expr.pos)

    def _call_fn_lit(self, lit: TFnLit, sig: FnSig, args: list[Value]) -> Value:
        env = _RuntimeEnv()
        env.push_scope()
        for i, p in enumerate(lit.params):
            env.bind(p.name, sig.params[i], args[i])
        if isinstance(lit.body, list):
            try:
                self._eval_block(lit.body, env, fn_ret=sig.ret)
            except _Return as r:
                if ty_eq(sig.ret, TY_VOID):
                    return VNil()
                if r.value is None:
                    raise TaytshRuntimeFault("missing return value", lit.pos)
                return r.value
            if ty_eq(sig.ret, TY_VOID):
                return VNil()
            raise TaytshRuntimeFault("function literal fell off", lit.pos)
        return self._eval_expr(lit.body, env, expected=sig.ret)

    def _eval_call(
        self, call: TCall, env: _RuntimeEnv, *, expected: Ty | None
    ) -> Value:
        # Struct constructor
        if isinstance(call.func, TVar) and call.func.name in self.index.structs:
            return self._eval_struct_ctor(call, env)

        # Builtins Map/Set need expected type for tagging.
        if (
            isinstance(call.func, TVar)
            and call.func.name in ("Map", "Set")
            and not call.args
        ):
            if expected is None:
                inferred = self.tc.expr_types.get(id(call))
                expected = inferred
            if call.func.name == "Map":
                if not isinstance(expected, TyMap):
                    raise TaytshRuntimeFault("cannot infer Map() type", call.pos)
                return VMap({}, expected)
            if not isinstance(expected, TySet):
                raise TaytshRuntimeFault("cannot infer Set() type", call.pos)
            return VSet(set(), expected)

        # Built-in call by name
        if isinstance(call.func, TVar) and call.func.name in _BUILTIN_RUNTIME:
            args = [self._eval_expr(a.value, env) for a in call.args]
            return _BUILTIN_RUNTIME[call.func.name](self, args)

        # Method call: obj.Method(...)
        if isinstance(call.func, TFieldAccess):
            recv = self._eval_expr(call.func.obj, env)
            if isinstance(recv, VStruct):
                s = self.index.structs.get(recv.struct_name)
                if s is not None and call.func.field in s.methods:
                    mi = s.methods[call.func.field]
                    args = [recv] + [self._eval_expr(a.value, env) for a in call.args]
                    return self._call_fn(mi.decl, mi.sig, args)
            # Fall back: field value call
            fnv = self._eval_expr(call.func, env)
            if not isinstance(fnv, VFunc):
                raise TaytshRuntimeFault("call target not a function", call.pos)
            args = [self._eval_expr(a.value, env) for a in call.args]
            return fnv.call(args)

        fnv = self._eval_expr(call.func, env)
        if not isinstance(fnv, VFunc):
            raise TaytshRuntimeFault("call target not a function", call.pos)
        args = [self._eval_expr(a.value, env) for a in call.args]
        return fnv.call(args)

    def _eval_struct_ctor(self, call: TCall, env: _RuntimeEnv) -> VStruct:
        assert isinstance(call.func, TVar)
        s = self.index.structs[call.func.name]
        field_order = [f.name for f in s.fields]
        has_named = any(a.name is not None for a in call.args)
        has_pos = any(a.name is None for a in call.args)
        if has_named and has_pos:
            raise TaytshRuntimeFault("cannot mix named and positional args", call.pos)
        fields: dict[str, Value] = {}
        if has_pos:
            if len(call.args) != len(field_order):
                raise TaytshRuntimeFault("wrong number of constructor args", call.pos)
            for i, a in enumerate(call.args):
                fname = field_order[i]
                fty = s.field_map[fname].ty
                fields[fname] = self._eval_expr(a.value, env, expected=fty)
        else:
            for a in call.args:
                assert a.name is not None
                if a.name not in s.field_map:
                    raise TaytshRuntimeFault("unknown struct field", call.pos)
                fty = s.field_map[a.name].ty
                fields[a.name] = self._eval_expr(a.value, env, expected=fty)
            for f in field_order:
                if f not in fields:
                    raise TaytshRuntimeFault("missing struct field", call.pos)
        return VStruct(s.name, fields)

    def _matches_type(self, v: Value, typ: Ty) -> bool:
        if ty_eq(typ, TY_OBJ):
            return True
        if ty_is_nil(typ):
            return isinstance(v, VNil)
        if isinstance(typ, TyPrim):
            return ty_eq(v.ty(), typ)
        if isinstance(typ, TyStruct):
            return isinstance(v, VStruct) and v.struct_name == typ.name
        if isinstance(typ, TyInterface):
            return (
                isinstance(v, VStruct)
                and v.struct_name in self.index.interfaces[typ.name].implementors
            )
        if isinstance(typ, TyEnum):
            return isinstance(v, VEnum) and v.enum_name == typ.name
        if isinstance(typ, TyUnion):
            return any(self._matches_type(v, m) for m in typ.members)
        return ty_eq(v.ty(), typ)

    def _eval_index(self, obj: Value, idx: Value, *, pos: Pos) -> Value:
        if not isinstance(idx, VInt) and not isinstance(obj, VMap):
            raise TaytshRuntimeFault("index must be int", pos)
        if isinstance(obj, VList):
            i = cast(VInt, idx).value
            if i < 0 or i >= len(obj.elements):
                self._throw_err("IndexError", "index out of bounds")
            return obj.elements[i]
        if isinstance(obj, VString):
            i = cast(VInt, idx).value
            if i < 0 or i >= len(obj.value):
                self._throw_err("IndexError", "index out of bounds")
            return VRune(obj.value[i])
        if isinstance(obj, VBytes):
            i = cast(VInt, idx).value
            if i < 0 or i >= len(obj.value):
                self._throw_err("IndexError", "index out of bounds")
            return VByte(obj.value[i])
        if isinstance(obj, VMap):
            key = _as_hashable(idx)
            if key not in obj.entries:
                self._throw_err("KeyError", "missing key")
            return obj.entries[key]
        raise TaytshRuntimeFault("indexing not supported", pos)

    def _eval_slice(self, obj: Value, lo: int, hi: int, *, pos: Pos) -> Value:
        if lo < 0 or hi < 0:
            self._throw_err("IndexError", "slice bounds out of range")
        if lo > hi:
            self._throw_err("IndexError", "slice lo > hi")
        if isinstance(obj, VList):
            if hi > len(obj.elements):
                self._throw_err("IndexError", "slice bounds out of range")
            return VList(list(obj.elements[lo:hi]), obj.typ)
        if isinstance(obj, VString):
            if hi > len(obj.value):
                self._throw_err("IndexError", "slice bounds out of range")
            return VString(obj.value[lo:hi])
        if isinstance(obj, VBytes):
            if hi > len(obj.value):
                self._throw_err("IndexError", "slice bounds out of range")
            return VBytes(obj.value[lo:hi])
        raise TaytshRuntimeFault("slicing not supported", pos)

    def _eval_binary(self, op: str, left: Value, right: Value, *, pos: Pos) -> Value:
        if op == "==":
            return VBool(_value_eq(left, right))
        if op == "!=":
            return VBool(not _value_eq(left, right))

        # Numeric / ordered ops.
        if op in ("<", "<=", ">", ">="):
            if isinstance(left, VInt) and isinstance(right, VInt):
                return VBool(_cmp(op, left.value, right.value))
            if isinstance(left, VFloat) and isinstance(right, VFloat):
                return VBool(_cmp(op, left.value, right.value))
            if isinstance(left, VByte) and isinstance(right, VByte):
                return VBool(_cmp(op, left.value, right.value))
            if isinstance(left, VRune) and isinstance(right, VRune):
                return VBool(_cmp(op, left.value, right.value))
            if isinstance(left, VString) and isinstance(right, VString):
                return VBool(_cmp(op, left.value, right.value))
            raise TaytshRuntimeFault("invalid comparison operands", pos)

        if op in ("|", "^", "&"):
            if isinstance(left, VInt) and isinstance(right, VInt):
                if op == "|":
                    return VInt(left.value | right.value)
                if op == "^":
                    return VInt(left.value ^ right.value)
                return VInt(left.value & right.value)
            if isinstance(left, VByte) and isinstance(right, VByte):
                if op == "|":
                    return VByte(left.value | right.value)
                if op == "^":
                    return VByte(left.value ^ right.value)
                return VByte(left.value & right.value)
            raise TaytshRuntimeFault("invalid bitwise operands", pos)

        if op in ("<<", ">>"):
            if not isinstance(right, VInt):
                raise TaytshRuntimeFault("shift amount not int", pos)
            shift = right.value
            if shift < 0:
                self._throw_err("ValueError", "shift amount must be non-negative")
            if isinstance(left, VInt):
                return (
                    VInt(left.value << shift)
                    if op == "<<"
                    else VInt(left.value >> shift)
                )
            if isinstance(left, VByte):
                val = (left.value << shift) if op == "<<" else (left.value >> shift)
                return VByte(val & 0xFF)
            raise TaytshRuntimeFault("invalid shift operands", pos)

        if op in ("+", "-", "*", "/", "%"):
            if isinstance(left, VInt) and isinstance(right, VInt):
                if op == "+":
                    return VInt(left.value + right.value)
                if op == "-":
                    return VInt(left.value - right.value)
                if op == "*":
                    return VInt(left.value * right.value)
                if op == "/":
                    try:
                        q, _ = _int_divmod_trunc(left.value, right.value)
                    except ZeroDivisionError:
                        self._throw_err("ZeroDivisionError", "division by zero")
                    return VInt(q)
                try:
                    _, r = _int_divmod_trunc(left.value, right.value)
                except ZeroDivisionError:
                    self._throw_err("ZeroDivisionError", "division by zero")
                return VInt(r)
            if isinstance(left, VFloat) and isinstance(right, VFloat):
                if op == "+":
                    return VFloat(left.value + right.value)
                if op == "-":
                    return VFloat(left.value - right.value)
                if op == "*":
                    return VFloat(left.value * right.value)
                if op == "/":
                    return VFloat(left.value / right.value)
                return VFloat(math.fmod(left.value, right.value))
            if isinstance(left, VByte) and isinstance(right, VByte):
                if op == "+":
                    return VByte((left.value + right.value) & 0xFF)
                if op == "-":
                    return VByte((left.value - right.value) & 0xFF)
                if op == "*":
                    return VByte((left.value * right.value) & 0xFF)
                if op == "/":
                    try:
                        q, _ = _int_divmod_trunc(left.value, right.value)
                    except ZeroDivisionError:
                        self._throw_err("ZeroDivisionError", "division by zero")
                    return VByte(q & 0xFF)
                try:
                    _, r = _int_divmod_trunc(left.value, right.value)
                except ZeroDivisionError:
                    self._throw_err("ZeroDivisionError", "division by zero")
                return VByte(r & 0xFF)
            raise TaytshRuntimeFault("invalid arithmetic operands", pos)

        raise TaytshRuntimeFault(f"unknown operator '{op}'", pos)


def _cmp(op: str, a: object, b: object) -> bool:
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    raise AssertionError(op)


# ---- Minimal builtin runtime (expanded in step 4) --------------------------


def _bi_tostring(rt: Runtime, args: list[Value]) -> Value:
    return VString(args[0].to_string())


def _bi_len(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        return VInt(len(x.value))
    if isinstance(x, VBytes):
        return VInt(len(x.value))
    if isinstance(x, VList):
        return VInt(len(x.elements))
    if isinstance(x, VMap):
        return VInt(len(x.entries))
    if isinstance(x, VSet):
        return VInt(len(x.elements))
    raise TaytshRuntimeFault("Len unsupported", None)


def _bi_get(rt: Runtime, args: list[Value]) -> Value:
    m = args[0]
    if not isinstance(m, VMap):
        raise TaytshRuntimeFault("Get expects map", None)
    key = _as_hashable(args[1])
    if key in m.entries:
        return m.entries[key]
    if len(args) == 3:
        return args[2]
    return VNil()


def _bi_contains(rt: Runtime, args: list[Value]) -> Value:
    a, b = args
    if isinstance(a, VList):
        return VBool(any(_value_eq(x, b) for x in a.elements))
    if isinstance(a, VMap):
        return VBool(_as_hashable(b) in a.entries)
    if isinstance(a, VSet):
        return VBool(_as_hashable(b) in a.elements)
    if isinstance(a, VString) and isinstance(b, VString):
        return VBool(b.value in a.value)
    raise TaytshRuntimeFault("Contains unsupported", None)


def _bi_unwrap(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VNil):
        rt._throw_err("NilError", "Unwrap(nil)")
    return x


def _bi_assert(rt: Runtime, args: list[Value]) -> Value:
    cond = args[0]
    if not isinstance(cond, VBool):
        raise TaytshRuntimeFault("Assert expects bool", None)
    if cond.value:
        return VNil()
    msg = "assertion failed"
    if len(args) == 2 and isinstance(args[1], VString):
        msg = args[1].value
    rt._throw_err("AssertError", msg)
    return VNil()


_BUILTIN_RUNTIME: dict[str, Callable[[Runtime, list[Value]], Value]] = {
    "ToString": _bi_tostring,
    "Len": _bi_len,
    "Get": _bi_get,
    "Contains": _bi_contains,
    "Unwrap": _bi_unwrap,
    "Assert": _bi_assert,
}
