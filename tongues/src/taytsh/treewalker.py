"""Taytsh runtime — typecheck and evaluate a Taytsh module.

This is a spec-faithful (as practical) interpreter for the Taytsh textual IR
defined in spec/taytsh.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _isnan(x: float) -> bool:
    return x != x


def _isinf(x: float) -> bool:
    return x == float("inf") or x == float("-inf")


_INT64_MAX = (1 << 62) - 1 + (1 << 62)
_INT64_MIN = -_INT64_MAX - 1


def _copysign_inf(x: float) -> float:
    """Return +inf or -inf depending on sign of x."""
    if x < 0:
        return float("-inf")
    return float("inf")


def _floor(x: float) -> int:
    i = int(x)
    if x < 0 and x != i:
        return i - 1
    return i


def _ceil(x: float) -> int:
    i = int(x)
    if x > 0 and x != i:
        return i + 1
    return i


def _sqrt(x: float) -> float:
    return x**0.5


from .ast import (
    Pos,
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
    TFieldDecl,
    TFnDecl,
    TFnLit,
    TFloatLit,
    TForStmt,
    TIfStmt,
    TIndex,
    TIntLit,
    TInterfaceDecl,
    TLetStmt,
    TListLit,
    TMapLit,
    TMatchStmt,
    TModule,
    TModuleItem,
    TNilLit,
    TOpAssignStmt,
    TPatternEnum,
    TPatternNil,
    TPatternType,
    TPrimitive,
    TRange,
    TReturnStmt,
    TRuneLit,
    TSetLit,
    TSlice,
    TStmt,
    TStringLit,
    TStructDecl,
    TThrowStmt,
    TTernary,
    TTupleAccess,
    TTupleAssignStmt,
    TTupleLit,
    TTryStmt,
    TType,
    TUnaryOp,
    TVar,
    TWhileStmt,
)


# ============================================================
# Diagnostics
# ============================================================


class TaytshTypeError(Exception):
    """Static type error."""

    def __init__(self, msg: str, pos: Pos | None = None):
        if pos is None:
            super().__init__(msg)
        else:
            super().__init__(f"{msg} at line {pos.line} col {pos.col}")
        self.msg = msg
        self.pos = pos


class TaytshRuntimeFault(Exception):
    """Runtime fault (uncaught throw, invalid operation, etc.)."""

    def __init__(self, msg: str, pos: Pos | None = None):
        if pos is None:
            super().__init__(msg)
        else:
            super().__init__(f"{msg} at line {pos.line} col {pos.col}")
        self.msg = msg
        self.pos = pos


# ============================================================
# Checked types (imported from check.py)
# ============================================================

from .check import (
    BOOL_T,
    BYTE_T,
    BYTES_T,
    Checker,
    ERROR_T,
    EnumT,
    FLOAT_T,
    FnT,
    INT_T,
    InterfaceT,
    ListT,
    MapT,
    NIL_T,
    RUNE_T,
    STRING_T,
    SetT,
    StructT,
    TupleT,
    TY_ERROR,
    TY_NIL,
    Type,
    UnionT,
    VOID_T,
    check_with_info,
    normalize_union,
    type_eq,
    type_name,
)


def _is_nil(t: Type) -> bool:
    return t.kind == TY_NIL


def _is_error(t: Type) -> bool:
    return t.kind == TY_ERROR


def _builtin_arg_expected(name: str, expected: Type | None) -> Type | None:
    """Derive the expected arg type for a builtin from its expected return type."""
    if expected is None:
        return None
    if name == "Sum" and (type_eq(expected, INT_T) or type_eq(expected, FLOAT_T)):
        return ListT(kind="list", element=expected)
    if name == "Sorted" and isinstance(expected, ListT):
        return expected
    return None


# ============================================================
# Values
# ============================================================


@dataclass(unsafe_hash=True)
class Value:
    """A runtime value with a concrete (non-union) type tag."""

    def ty(self) -> Type:
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError


@dataclass(unsafe_hash=True)
class VNil(Value):
    def ty(self) -> Type:
        return NIL_T

    def to_string(self) -> str:
        return "nil"


@dataclass(unsafe_hash=True)
class VBool(Value):
    value: bool

    def ty(self) -> Type:
        return BOOL_T

    def to_string(self) -> str:
        return "true" if self.value else "false"


@dataclass(unsafe_hash=True)
class VInt(Value):
    value: int

    def ty(self) -> Type:
        return INT_T

    def to_string(self) -> str:
        return str(self.value)


@dataclass(unsafe_hash=True)
class VFloat(Value):
    value: float

    def ty(self) -> Type:
        return FLOAT_T

    def to_string(self) -> str:
        return str(self.value)


@dataclass(unsafe_hash=True)
class VByte(Value):
    value: int

    def ty(self) -> Type:
        return BYTE_T

    def to_string(self) -> str:
        return str(self.value)


@dataclass(unsafe_hash=True)
class VBytes(Value):
    value: bytes

    def ty(self) -> Type:
        return BYTES_T

    def to_string(self) -> str:
        return self.value.hex()


@dataclass(unsafe_hash=True)
class VString(Value):
    value: str

    def ty(self) -> Type:
        return STRING_T

    def to_string(self) -> str:
        return self.value


@dataclass(unsafe_hash=True)
class VRune(Value):
    value: str

    def ty(self) -> Type:
        return RUNE_T

    def to_string(self) -> str:
        return self.value


@dataclass(unsafe_hash=True)
class VTuple(Value):
    elements: list[Value]
    typ: TupleT = field(hash=False)

    def ty(self) -> Type:
        return self.typ

    def to_string(self) -> str:
        inner = ", ".join(v.to_string() for v in self.elements)
        return f"({inner})"


@dataclass
class VList(Value):
    elements: list[Value]
    typ: ListT

    def ty(self) -> Type:
        return self.typ

    def to_string(self) -> str:
        inner = ", ".join(v.to_string() for v in self.elements)
        return f"[{inner}]"


@dataclass(unsafe_hash=True)
class VEnum(Value):
    enum_name: str
    variant: str

    def ty(self) -> Type:
        return EnumT(kind="enum", name=self.enum_name, variants=[])

    def to_string(self) -> str:
        return self.enum_name + "." + self.variant


@dataclass
class VMap(Value):
    map_keys: list[Value]
    map_vals: list[Value]
    typ: MapT
    _shadow: dict[str, int] | None = field(default=None, init=False, repr=False)

    def ty(self) -> Type:
        return self.typ

    def to_string(self) -> str:
        parts: list[str] = []
        for mi, mk in enumerate(self.map_keys):
            parts.append(mk.to_string() + ": " + self.map_vals[mi].to_string())
        return "{" + ", ".join(parts) + "}"


@dataclass
class VSet(Value):
    elements: list[Value]
    typ: SetT
    _shadow: set[str] | None = field(default=None, init=False, repr=False)

    def ty(self) -> Type:
        return self.typ

    def to_string(self) -> str:
        inner = ", ".join(v.to_string() for v in self.elements)
        return "{" + inner + "}"


@dataclass
class VStruct(Value):
    struct_name: str
    fields: dict[str, Value]

    def ty(self) -> Type:
        return StructT(
            kind="struct",
            name=self.struct_name,
            fields={},
            methods={},
            parent=None,
            field_order=[],
        )

    def to_string(self) -> str:
        parts: list[str] = []
        for k, v in self.fields.items():
            parts.append(f"{k}: {v.to_string()}")
        inner = ", ".join(parts)
        return f"{self.struct_name}({inner})"


@dataclass
class VFunc(Value):
    typ: FnT
    name: str | None
    kind: str  # "user" | "builtin" | "fnlit"
    fn_key: str  # function/builtin name, or "" for fnlit
    fn_lit: TFnLit | None  # set for fnlit
    fn_sig: FnSig | None  # set for user and fnlit
    fn_decl: TFnDecl | None  # set for user

    def ty(self) -> Type:
        return self.typ

    def to_string(self) -> str:
        if self.name is None:
            return "<fn>"
        return f"<fn {self.name}>"


# ============================================================
# Control flow signals (internal)
# ============================================================


@dataclass
class _Return(Exception):
    value: Value | None


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


@dataclass
class _Throw(Exception):
    value: Value


@dataclass
class _Exit(Exception):
    code: int


# ============================================================
# Runtime I/O
# ============================================================


@dataclass
class _Input:
    data: bytes
    pos: int

    def read_all(self) -> bytes:
        out = self.data[self.pos :]
        self.pos = len(self.data)
        return out

    def read_n(self, n: int) -> bytes:
        if n <= 0:
            return b""
        out = self.data[self.pos : self.pos + n]
        self.pos += len(out)
        return out

    def read_line(self) -> bytes | None:
        if self.pos >= len(self.data):
            return None
        start: int = self.pos
        i: int = start
        while i < len(self.data):
            if self.data[i : i + 1] == b"\n":
                self.pos = i + 1
                return self.data[start : i + 1]
            i += 1
        self.pos = len(self.data)
        return self.data[start:]


@dataclass
class RunResult:
    exit_code: int
    stdout: bytes
    stderr: bytes


# ============================================================
# Placeholder: implementation continues in further steps
# ============================================================


def prepare(module: TModule) -> Runtime:
    """Typecheck and prepare a Runtime — expensive, do once per module."""
    check_result = check_with_info(module)
    checker: Checker = check_result[1]
    idx = _build_index(module)
    _resolve_index(idx, checker)
    fn_values: dict[str, VFunc] = {}
    for fn_name, fn_info in idx.funcs.items():
        fn_values[fn_name] = VFunc(
            fn_info.sig.ty(), fn_name, "user", fn_name, None, fn_info.sig, fn_info.decl
        )
    builtin_values: dict[str, VFunc] = {}
    for bi_name in _BUILTIN_NAMES_RT:
        fn_type = checker.functions.get(bi_name)
        if fn_type is not None:
            builtin_values[bi_name] = VFunc(
                fn_type, bi_name, "builtin", bi_name, None, None, None
            )
    return Runtime(
        module,
        idx,
        checker,
        checker.expr_types,
        _Input(b"", 0),
        [],
        {},
        b"",
        b"",
        fn_values,
        builtin_values,
    )


def run(
    module: TModule,
    stdin: bytes = b"",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> RunResult:
    """Typecheck and run a parsed Taytsh module."""
    rt = prepare(module)
    return rt.invoke(stdin, args, env)


# ============================================================
# Module indexing + typechecking
# ============================================================


_RESERVED_BINDINGS: set[str] = {
    # Built-in functions (spec: reserved names)
    "ToRepr",
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
    "WrappingAdd",
    "WrappingSub",
    "WrappingMul",
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
    "ReplaceCount",
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
    "Reverse",
    "Reversed",
    "Sorted",
    "ListFrom",
    "Map",
    "Get",
    "Delete",
    "Keys",
    "Values",
    "Items",
    "Merge",
    "Set",
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
    "Floor",
    "Ceil",
    "Sqrt",
    "ReadFile",
    "ReadFileBytes",
    "WriteFile",
    # Built-in error struct names (treated as reserved for simplicity)
    "KeyError",
    "IndexError",
    "ZeroDivisionError",
    "AssertError",
    "NilError",
    "ValueError",
    "IOError",
}


@dataclass
class FnSig:
    params: tuple[Type, ...]
    ret: Type

    def ty(self) -> FnT:
        return FnT(kind="fn", params=list(self.params), ret=self.ret)


@dataclass
class FieldInfo:
    name: str
    ty: Type
    decl: TFieldDecl


@dataclass
class MethodInfo:
    name: str
    sig: FnSig  # includes self as first param
    decl: TFnDecl


@dataclass
class StructInfo:
    name: str
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


def _ensure_not_reserved(name: str, *, pos: Pos) -> None:
    if name in _RESERVED_BINDINGS:
        raise TaytshTypeError(f"reserved name '{name}'", pos)


def _builtin_err(name: str) -> StructInfo:
    decl = TStructDecl(
        Pos(0, 0),
        name,
        None,
        [TFieldDecl(Pos(0, 0), "message", TPrimitive(Pos(0, 0), "string"))],
        [],
        {},
    )
    fi = FieldInfo("message", STRING_T, decl.fields[0])
    return StructInfo(
        name=name,
        fields=[fi],
        field_map={"message": fi},
        methods={},
        decl=decl,
    )


def _build_index(module: TModule) -> ModuleIndex:
    funcs: dict[str, FnInfo] = {}
    structs: dict[str, StructInfo] = {}
    interfaces: dict[str, InterfaceInfo] = {}
    enums: dict[str, EnumInfo] = {}

    for err in (
        "KeyError",
        "IndexError",
        "ZeroDivisionError",
        "AssertError",
        "NilError",
        "ValueError",
        "IOError",
    ):
        structs[err] = _builtin_err(err)

    # First pass: collect decl kinds and detect duplicates across the flat namespace.
    seen_top: dict[str, TModuleItem] = {}
    for d in module.decls:
        name: str = ""
        d_pos: Pos = Pos(0, 0)
        match d:
            case (
                TFnDecl() | TStructDecl() | TInterfaceDecl() | TEnumDecl() | TLetStmt()
            ):
                name = d.name
                d_pos = d.pos
            case _:
                continue
        _ensure_not_reserved(name, pos=d_pos)
        if name in seen_top or name in structs:
            raise TaytshTypeError("duplicate top-level name '" + name + "'", d_pos)
        seen_top[name] = d

    # Populate structs / interfaces / enums first so type resolution can refer to them.
    for sname, sd in seen_top.items():
        match sd:
            case TInterfaceDecl():
                interfaces[sname] = InterfaceInfo(
                    name=sname, decl=sd, implementors=set()
                )
            case TEnumDecl():
                enums[sname] = EnumInfo(name=sname, variants=set(sd.variants), decl=sd)
            case TStructDecl():
                structs[sname] = StructInfo(
                    name=sname,
                    fields=[],
                    field_map={},
                    methods={},
                    decl=sd,
                )

    # Second pass: resolve struct implements + field/method names (types later).
    _BUILTIN_ERR_NAMES = {
        "KeyError",
        "IndexError",
        "ZeroDivisionError",
        "AssertError",
        "NilError",
        "ValueError",
        "IOError",
    }
    for s in structs.values():
        if s.name in _BUILTIN_ERR_NAMES and s.decl.pos.line == 0:
            continue
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
            fi = FieldInfo(f.name, ERROR_T, f)  # placeholder ty resolved later
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
            methods[m.name] = MethodInfo(m.name, FnSig((), VOID_T), m)  # placeholder

        s.fields = fields
        s.field_map = field_map
        s.methods = methods

    # Functions (signatures resolved later).
    for fname, fd in seen_top.items():
        if isinstance(fd, TFnDecl):
            funcs[fname] = FnInfo(name=fname, sig=FnSig((), VOID_T), decl=fd)

    return ModuleIndex(funcs=funcs, structs=structs, interfaces=interfaces, enums=enums)


def _resolve_index(index: ModuleIndex, checker: Checker) -> None:
    """Resolve placeholder types in the ModuleIndex using the Checker."""
    for s in index.structs.values():
        if s.decl.pos.line == 0:
            continue
        for fi in s.fields:
            fi.ty = checker.resolve_type(fi.decl.typ)
        for mi in s.methods.values():
            params: list[Type] = []
            for p in mi.decl.params:
                if p.typ is None:
                    params.append(
                        StructT(
                            kind="struct",
                            name=s.name,
                            fields={},
                            methods={},
                            parent=None,
                            field_order=[],
                        )
                    )
                else:
                    params.append(checker.resolve_type(p.typ))
            ret = checker.resolve_type(mi.decl.ret)
            mi.sig = FnSig(tuple(params), ret)
    for fi in index.funcs.values():
        params: list[Type] = []
        for p in fi.decl.params:
            if p.typ is not None:
                params.append(checker.resolve_type(p.typ))
        ret = checker.resolve_type(fi.decl.ret)
        fi.sig = FnSig(tuple(params), ret)


def _expr_key(expr: TExpr) -> tuple[int, int]:
    return (expr.pos.line, expr.pos.col)


def _resolve_type(t: TType, checker: Checker) -> Type:
    """Resolve a parse-time TType to a checked Type using the Checker."""
    return checker.resolve_type(t)


def _fn_lit_sig(lit: TFnLit, checker: Checker) -> FnSig:
    """Resolve a function literal's signature."""
    params: list[Type] = []
    for p in lit.params:
        if p.typ is None:
            raise TaytshRuntimeFault("self not allowed in fn literals", p.pos)
        params.append(checker.resolve_type(p.typ))
    ret = checker.resolve_type(lit.ret)
    return FnSig(params, ret)


# ============================================================
# Evaluation
# ============================================================


def _same_value_class(a: Value, b: Value) -> bool:
    if isinstance(a, VNil):
        return isinstance(b, VNil)
    if isinstance(a, VBool):
        return isinstance(b, VBool)
    if isinstance(a, VInt):
        return isinstance(b, VInt)
    if isinstance(a, VFloat):
        return isinstance(b, VFloat)
    if isinstance(a, VByte):
        return isinstance(b, VByte)
    if isinstance(a, VBytes):
        return isinstance(b, VBytes)
    if isinstance(a, VString):
        return isinstance(b, VString)
    if isinstance(a, VRune):
        return isinstance(b, VRune)
    if isinstance(a, VEnum):
        return isinstance(b, VEnum)
    if isinstance(a, VTuple):
        return isinstance(b, VTuple)
    if isinstance(a, VList):
        return isinstance(b, VList)
    if isinstance(a, VMap):
        return isinstance(b, VMap)
    if isinstance(a, VSet):
        return isinstance(b, VSet)
    if isinstance(a, VStruct):
        return isinstance(b, VStruct)
    if isinstance(a, VFunc):
        return isinstance(b, VFunc)
    return False


def _value_eq(a: Value, b: Value) -> bool:
    if isinstance(a, VString):
        return isinstance(b, VString) and a.value == b.value
    if isinstance(a, VInt):
        if isinstance(b, VInt):
            return a.value == b.value
        return isinstance(b, VByte) and a.value == b.value
    if not _same_value_class(a, b):
        if isinstance(a, VByte) and isinstance(b, VInt):
            return a.value == b.value
        return False
    if isinstance(a, VNil):
        return True
    if isinstance(a, VBool) and isinstance(b, VBool):
        return a.value == b.value
    if isinstance(a, VFloat) and isinstance(b, VFloat):
        return a.value == b.value
    if isinstance(a, VByte) and isinstance(b, VByte):
        return a.value == b.value
    if isinstance(a, VBytes) and isinstance(b, VBytes):
        return a.value == b.value
    if isinstance(a, VString) and isinstance(b, VString):
        return a.value == b.value
    if isinstance(a, VRune) and isinstance(b, VRune):
        return a.value == b.value
    if isinstance(a, VEnum) and isinstance(b, VEnum):
        return a == b
    if isinstance(a, VTuple) and isinstance(b, VTuple):
        if not type_eq(a.typ, b.typ):
            return False
        if len(a.elements) != len(b.elements):
            return False
        for ei, ae in enumerate(a.elements):
            if not _value_eq(ae, b.elements[ei]):
                return False
        return True
    if isinstance(a, VList) and isinstance(b, VList):
        if not type_eq(a.typ, b.typ):
            return False
        if len(a.elements) != len(b.elements):
            return False
        for li, le in enumerate(a.elements):
            if not _value_eq(le, b.elements[li]):
                return False
        return True
    if isinstance(a, VMap):
        if not isinstance(b, VMap):
            return False
        if not type_eq(a.typ, b.typ):
            return False
        if len(a.map_keys) != len(b.map_keys):
            return False
        for mi, mk in enumerate(a.map_keys):
            if not _map_has(b, mk):
                return False
            if not _value_eq(a.map_vals[mi], _map_get(b, mk)):
                return False
        return True
    if isinstance(a, VSet):
        if not isinstance(b, VSet):
            return False
        if not type_eq(a.typ, b.typ):
            return False
        if len(a.elements) != len(b.elements):
            return False
        for e in a.elements:
            if not _set_has(b, e):
                return False
        return True
    if isinstance(a, VStruct):
        if not isinstance(b, VStruct):
            return False
        if a.struct_name != b.struct_name:
            return False
        if a.fields.keys() != b.fields.keys():
            return False
        for k in a.fields.keys():
            if not _value_eq(a.fields[k], b.fields[k]):
                return False
        return True
    if isinstance(a, VFunc):
        if not isinstance(b, VFunc):
            return False
        return type_eq(a.typ, b.typ) and a.name == b.name and a.kind == b.kind
    raise TaytshRuntimeFault("unsupported equality", None)


def _value_lt(a: Value, b: Value) -> bool:
    if isinstance(a, VInt) and isinstance(b, VInt):
        return a.value < b.value
    if isinstance(a, VFloat) and isinstance(b, VFloat):
        return a.value < b.value
    if isinstance(a, VString) and isinstance(b, VString):
        return a.value < b.value
    if isinstance(a, VByte) and isinstance(b, VByte):
        return a.value < b.value
    return False


def _is_hashable_value(v: Value) -> bool:
    if isinstance(v, VNil):
        return True
    if isinstance(v, VBool):
        return True
    if isinstance(v, VInt):
        return True
    if isinstance(v, VFloat):
        return True
    if isinstance(v, VByte):
        return True
    if isinstance(v, VBytes):
        return True
    if isinstance(v, VString):
        return True
    if isinstance(v, VRune):
        return True
    if isinstance(v, VTuple):
        return True
    if isinstance(v, VEnum):
        return True
    return False


def _as_hashable(v: Value) -> Value:
    if _is_hashable_value(v):
        return v
    raise TaytshRuntimeFault("value is not hashable", None)


def _hash_key(v: Value) -> str | None:
    """Convert a Value to a Python-hashable string key, or None if unhashable."""
    if isinstance(v, VString):
        return "s:" + v.value
    if isinstance(v, (VInt, VByte)):
        return "i:" + str(v.value)
    if isinstance(v, VBool):
        return "b:1" if v.value else "b:0"
    if isinstance(v, VNil):
        return "n"
    if isinstance(v, VFloat):
        return "f:" + str(v.value)
    if isinstance(v, VBytes):
        return "B:" + v.value.hex()
    if isinstance(v, VRune):
        return "r:" + v.value
    if isinstance(v, VEnum):
        return "e:" + v.enum_name + "." + v.variant
    if isinstance(v, VTuple):
        parts: list[str] = []
        for e in v.elements:
            hk = _hash_key(e)
            if hk is None:
                return None
            parts.append(hk)
        return "t:" + ",".join(parts)
    return None


# ---- Map helpers (list-based map avoids dict[Value, Value]) ----


def _build_map_shadow(m: VMap) -> None:
    shadow: dict[str, int] = {}
    for i, k in enumerate(m.map_keys):
        hk = _hash_key(k)
        if hk is not None:
            shadow[hk] = i
    m._shadow = shadow


def _map_find(m: VMap, key: Value) -> int:
    hk = _hash_key(key)
    if hk is not None:
        if m._shadow is None:
            _build_map_shadow(m)
        shadow = m._shadow
        if shadow is not None:
            idx = shadow.get(hk)
            if idx is not None:
                return idx
        return -1
    for mi, mk in enumerate(m.map_keys):
        if _value_eq(mk, key):
            return mi
    return -1


def _clamp_slice(lo: int, hi: int, n: int) -> tuple[int, int]:
    if hi > n:
        hi = n
    if lo > hi:
        lo = hi
    return (lo, hi)


def _map_get(m: VMap, key: Value) -> Value:
    idx = _map_find(m, key)
    if idx < 0:
        raise TaytshRuntimeFault("key not found", None)
    return m.map_vals[idx]


def _map_has(m: VMap, key: Value) -> bool:
    return _map_find(m, key) >= 0


def _map_set(m: VMap, key: Value, value: Value) -> None:
    idx = _map_find(m, key)
    if idx >= 0:
        m.map_vals[idx] = value
    else:
        m.map_keys.append(key)
        m.map_vals.append(value)
        hk = _hash_key(key)
        shadow = m._shadow
        if shadow is not None and hk is not None:
            shadow[hk] = len(m.map_keys) - 1


def _map_del(m: VMap, key: Value) -> None:
    idx = _map_find(m, key)
    if idx >= 0:
        m.map_keys.pop(idx)
        m.map_vals.pop(idx)
        m._shadow = None


# ---- Set helpers ----


def _list_set_has(elements: list[Value], val: Value) -> bool:
    for e in elements:
        if _value_eq(e, val):
            return True
    return False


def _list_set_add(elements: list[Value], val: Value) -> None:
    if not _list_set_has(elements, val):
        elements.append(val)


def _build_set_shadow(s: VSet) -> None:
    shadow: set[str] = set()
    for e in s.elements:
        hk = _hash_key(e)
        if hk is not None:
            shadow.add(hk)
    s._shadow = shadow


def _set_has(s: VSet, val: Value) -> bool:
    hk = _hash_key(val)
    if hk is not None:
        if s._shadow is None:
            _build_set_shadow(s)
        shadow = s._shadow
        if shadow is not None:
            return hk in shadow
        return False
    for e in s.elements:
        if _value_eq(e, val):
            return True
    return False


def _set_add(s: VSet, val: Value) -> None:
    if _set_has(s, val):
        return
    s.elements.append(val)
    hk = _hash_key(val)
    shadow = s._shadow
    if shadow is not None and hk is not None:
        shadow.add(hk)


def _set_discard(s: VSet, val: Value) -> None:
    i = 0
    while i < len(s.elements):
        if _value_eq(s.elements[i], val):
            s.elements.pop(i)
            s._shadow = None
            return
        i += 1


def _set_union(a: list[Value], b: list[Value]) -> list[Value]:
    result = list(a)
    for e in b:
        if not _list_set_has(result, e):
            result.append(e)
    return result


def _set_intersection(a: list[Value], b: list[Value]) -> list[Value]:
    result: list[Value] = []
    for e in a:
        if _list_set_has(b, e):
            result.append(e)
    return result


def _set_difference(a: list[Value], b: list[Value]) -> list[Value]:
    result: list[Value] = []
    for e in a:
        if not _list_set_has(b, e):
            result.append(e)
    return result


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
    ty: Type
    value: Value


class _RuntimeEnv:
    def __init__(self) -> None:
        self._scopes: list[dict[str, _Binding]] = []

    def push_scope(self) -> None:
        self._scopes.append({})

    def pop_scope(self) -> None:
        self._scopes.pop()

    def bind(self, name: str, typ: Type, value: Value) -> None:
        if name == "_":
            return
        if not self._scopes:
            raise RuntimeError("no scope")
        self._scopes[-1][name] = _Binding(typ, value)

    def lookup(self, name: str) -> _Binding:
        i = len(self._scopes) - 1
        while i >= 0:
            scope = self._scopes[i]
            if name in scope:
                return scope[name]
            i -= 1
        raise TaytshRuntimeFault(f"unknown name '{name}'", None)

    def has(self, name: str) -> bool:
        i = len(self._scopes) - 1
        while i >= 0:
            if name in self._scopes[i]:
                return True
            i -= 1
        return False

    def get(self, name: str) -> Value:
        if name == "_":
            raise TaytshRuntimeFault("cannot read discard '_'", None)
        return self.lookup(name).value

    def get_ty(self, name: str) -> Type:
        if name == "_":
            raise TaytshRuntimeFault("cannot read discard '_'", None)
        return self.lookup(name).ty

    def set(self, name: str, value: Value) -> None:
        if name == "_":
            return
        i = len(self._scopes) - 1
        while i >= 0:
            scope = self._scopes[i]
            if name in scope:
                scope[name] = _Binding(scope[name].ty, value)
                return
            i -= 1
        raise TaytshRuntimeFault(f"unknown name '{name}'", None)


class _LValueRef:
    def __init__(self, typ: Type):
        self.typ = typ

    def get(self) -> Value:  # pragma: no cover
        raise NotImplementedError

    def set(self, value: Value) -> None:  # pragma: no cover
        raise NotImplementedError


class _VarRef(_LValueRef):
    def __init__(self, typ: Type, env: _RuntimeEnv, name: str):
        super().__init__(typ)
        self._env = env
        self._name = name

    def get(self) -> Value:
        return self._env.get(self._name)

    def set(self, value: Value) -> None:
        self._env.set(self._name, value)


class _FieldRef(_LValueRef):
    def __init__(self, typ: Type, obj: VStruct, field: str):
        super().__init__(typ)
        self._obj = obj
        self._field = field

    def get(self) -> Value:
        return self._obj.fields[self._field]

    def set(self, value: Value) -> None:
        self._obj.fields[self._field] = value


class _DiscardRef(_LValueRef):
    def get(self) -> Value:
        raise TaytshRuntimeFault("cannot read discard '_'", None)

    def set(self, value: Value) -> None:
        pass


class _ListIndexRef(_LValueRef):
    def __init__(self, typ: Type, obj: VList, index: int):
        super().__init__(typ)
        self._obj = obj
        self._index = index

    def get(self) -> Value:
        return self._obj.elements[self._index]

    def set(self, value: Value) -> None:
        self._obj.elements[self._index] = value


class _MapIndexRef(_LValueRef):
    def __init__(self, typ: Type, obj: VMap, key: Value):
        super().__init__(typ)
        self._obj = obj
        self._key = key

    def get(self) -> Value:
        if not _map_has(self._obj, self._key):
            raise KeyError
        return _map_get(self._obj, self._key)

    def set(self, value: Value) -> None:
        _map_set(self._obj, self._key, value)


def _fmod(x: float, y: float) -> float:
    return x - float(int(x / y)) * y


def _range_cond(x: int, end: int, step: int) -> bool:
    return x < end if step > 0 else x > end


class Runtime:
    def __init__(
        self,
        module: TModule,
        index: ModuleIndex,
        checker: Checker,
        expr_types: dict[tuple[int, int], Type],
        stdin: _Input,
        args: list[str],
        env: dict[str, str],
        stdout: bytes,
        stderr: bytes,
        fn_values: dict[str, VFunc],
        builtin_values: dict[str, VFunc],
    ):
        self.module = module
        self.index = index
        self.checker = checker
        self.expr_types = expr_types
        self.stdin = stdin
        self.args = args
        self.env = env
        self.stdout = stdout
        self.stderr = stderr
        self._fn_values = fn_values
        self._builtin_values = builtin_values
        self._global_values: dict[str, Value] = {}

    # ---- Errors / throwing -------------------------------------------------

    def _err(self, name: str, message: str) -> VStruct:
        return VStruct(name, {"message": VString(message)})

    def _throw_err(self, name: str, message: str) -> None:
        raise _Throw(self._err(name, message))

    # ---- Zero values -------------------------------------------------------

    def zero_value(self, typ: Type) -> Value:
        if type_eq(typ, INT_T):
            return VInt(0)
        if type_eq(typ, FLOAT_T):
            return VFloat(0.0)
        if type_eq(typ, BOOL_T):
            return VBool(False)
        if type_eq(typ, BYTE_T):
            return VByte(0)
        if type_eq(typ, BYTES_T):
            return VBytes(b"")
        if type_eq(typ, STRING_T):
            return VString("")
        if type_eq(typ, RUNE_T):
            return VRune("\0")
        if type_eq(typ, ERROR_T) or _is_nil(typ):
            return VNil()
        if isinstance(typ, ListT):
            return VList([], typ)
        if isinstance(typ, MapT):
            return VMap([], [], typ)
        if isinstance(typ, SetT):
            return VSet([], typ)
        if isinstance(typ, TupleT):
            elems = [self.zero_value(t) for t in typ.elements]
            return VTuple(elems, typ)
        if isinstance(typ, StructT):
            fields = {name: self.zero_value(ft) for name, ft in typ.fields.items()}
            return VStruct(typ.name, fields)
        if isinstance(typ, EnumT):
            return VEnum(typ.name, typ.variants[0])
        if isinstance(typ, InterfaceT):
            return VNil()
        if isinstance(typ, UnionT):
            if any(_is_nil(m) for m in typ.members):
                return VNil()
        raise TaytshRuntimeFault(f"type '{type_name(typ)}' has no zero value", None)

    # ---- Running -----------------------------------------------------------

    def invoke(
        self,
        stdin: bytes = b"",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> RunResult:
        """Reset per-invocation state and run Main()."""
        self.stdin = _Input(stdin, 0)
        self.args = list(args) if args is not None else []
        self.env = env if env is not None else {}
        self.stdout = b""
        self.stderr = b""
        self._global_values = {}
        return self.run_main()

    def run_main(self) -> RunResult:
        try:
            self._init_top_level_lets()
            self._call_fn(
                self.index.funcs["Main"].decl, self.index.funcs["Main"].sig, []
            )
            return RunResult(0, self.stdout, self.stderr)
        except _Exit as e:
            return RunResult(e.code, self.stdout, self.stderr)
        except _Throw as t:
            # Uncaught exception: best-effort message to stderr.
            msg: str = t.value.to_string()
            err_line: str = msg + "\n"
            self.stderr += err_line.encode("utf-8")
            return RunResult(1, self.stdout, self.stderr)

    def _init_top_level_lets(self) -> None:
        """Evaluate top-level let statements in source order."""
        env = _RuntimeEnv()
        env.push_scope()
        for decl in self.module.decls:
            if isinstance(decl, TLetStmt):
                vty = _resolve_type(decl.typ, self.checker)
                if decl.value is None:
                    val = self.zero_value(vty)
                else:
                    val = self._eval_expr(decl.value, env, expected=vty)
                self._global_values[decl.name] = val

    # ---- Functions ---------------------------------------------------------

    def _call_fn(self, decl: TFnDecl, sig: FnSig, args: list[Value]) -> Value:
        intrinsic = _call_intrinsic(decl.name, args)
        if intrinsic is not None:
            return intrinsic
        env = _RuntimeEnv()
        env.push_scope()
        # Bind params
        for i, p in enumerate(decl.params):
            if p.typ is None:
                env.bind("this", sig.params[i], args[i])
            elif i < len(args):
                env.bind(p.name, sig.params[i], args[i])
            else:
                env.bind(p.name, sig.params[i], self.zero_value(sig.params[i]))

        try:
            self._eval_block(decl.body, env, fn_ret=sig.ret)
        except _Return as r:
            if type_eq(sig.ret, VOID_T):
                return VNil()
            if r.value is None:
                raise TaytshRuntimeFault("missing return value", decl.pos)
            return r.value

        if type_eq(sig.ret, VOID_T):
            return VNil()
        raise TaytshRuntimeFault("function fell off without returning", decl.pos)

    # ---- Statements --------------------------------------------------------

    def _eval_block(
        self, stmts: list[TStmt], env: _RuntimeEnv, *, fn_ret: Type
    ) -> None:
        env.push_scope()
        try:
            for st in stmts:
                self._eval_stmt(st, env, fn_ret=fn_ret)
        finally:
            env.pop_scope()

    def _eval_stmt(self, st: TStmt, env: _RuntimeEnv, *, fn_ret: Type) -> None:
        if isinstance(st, TIfStmt):
            self._stmt_if(st, env, fn_ret)
            return
        if isinstance(st, TLetStmt):
            self._stmt_let(st, env, fn_ret)
            return
        if isinstance(st, TAssignStmt):
            self._stmt_assign(st, env, fn_ret)
            return
        if isinstance(st, TExprStmt):
            if (
                isinstance(st.expr, TCall)
                and isinstance(st.expr.func, TVar)
                and st.expr.func.name == "reveal_type"
            ):
                return
            self._eval_expr(st.expr, env)
            return
        if isinstance(st, TReturnStmt):
            if st.value is None:
                raise _Return(None)
            raise _Return(self._eval_expr(st.value, env, expected=fn_ret))
        if isinstance(st, TWhileStmt):
            self._stmt_while(st, env, fn_ret)
            return
        if isinstance(st, TForStmt):
            self._eval_for(st, env, fn_ret=fn_ret)
            return
        if isinstance(st, TOpAssignStmt):
            self._stmt_opassign(st, env, fn_ret)
            return
        if isinstance(st, TTupleAssignStmt):
            self._stmt_tuple_assign(st, env, fn_ret)
            return
        if isinstance(st, TMatchStmt):
            self._eval_match(st, env, fn_ret=fn_ret)
            return
        if isinstance(st, TTryStmt):
            self._eval_try(st, env, fn_ret=fn_ret)
            return
        if isinstance(st, TBreakStmt):
            raise _Break("")
        if isinstance(st, TContinueStmt):
            raise _Continue("")
        if isinstance(st, TThrowStmt):
            raise _Throw(self._eval_expr(st.expr, env))
        raise TaytshRuntimeFault("unsupported statement", st.pos)

    def _stmt_let(self, st: TLetStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        vty = _resolve_type(st.typ, self.checker)
        if st.value is None:
            env.bind(st.name, vty, self.zero_value(vty))
            return
        val = self._eval_expr(st.value, env, expected=vty)
        env.bind(st.name, vty, val)

    def _stmt_assign(self, st: TAssignStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        ref = self._eval_lvalue_ref(st.target, env)
        val = self._eval_expr(st.value, env, expected=ref.typ)
        ref.set(val)

    def _stmt_opassign(self, st: TOpAssignStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        oref = self._eval_lvalue_ref(st.target, env)
        cur: Value = VNil()
        try:
            cur = oref.get()
        except KeyError:
            self._throw_err("KeyError", "missing key")
        rhs = self._eval_expr(st.value, env, expected=oref.typ)
        res = self._eval_binary(st.op[:-1], cur, rhs, pos=st.pos)
        oref.set(res)

    def _stmt_tuple_assign(
        self, st: TTupleAssignStmt, env: _RuntimeEnv, fn_ret: Type
    ) -> None:
        trefs = [self._eval_lvalue_ref(t, env) for t in st.targets]
        tref_types: list[Type] = []
        for tref in trefs:
            tref_types.append(tref.typ)
        rhs = self._eval_expr(
            st.value,
            env,
            expected=TupleT(kind="tuple", elements=tref_types),
        )
        if not isinstance(rhs, VTuple):
            raise TaytshRuntimeFault("tuple assignment rhs not a tuple", st.pos)
        if len(rhs.elements) != len(st.targets):
            raise TaytshRuntimeFault("tuple arity mismatch", st.pos)
        for ti, tr in enumerate(trefs):
            tr.set(rhs.elements[ti])

    def _stmt_return(self, st: TReturnStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        if st.value is None:
            raise _Return(None)
        raise _Return(self._eval_expr(st.value, env, expected=fn_ret))

    def _stmt_break(self, st: TBreakStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        raise _Break("")

    def _stmt_continue(self, st: TContinueStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        raise _Continue("")

    def _stmt_throw(self, st: TThrowStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        raise _Throw(self._eval_expr(st.expr, env))

    def _stmt_expr(self, st: TExprStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        self._eval_expr(st.expr, env)

    def _stmt_if(self, st: TIfStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        while True:
            cond = self._eval_expr(st.cond, env)
            if not isinstance(cond, VBool):
                raise TaytshRuntimeFault("if condition not bool", st.pos)
            if cond.value:
                self._eval_block(st.then_body, env, fn_ret=fn_ret)
                return
            if st.else_body is None:
                return
            el0 = st.else_body[0]
            if len(st.else_body) == 1 and isinstance(el0, TIfStmt):
                st = el0
                continue
            self._eval_block(st.else_body, env, fn_ret=fn_ret)
            return

    def _stmt_while(self, st: TWhileStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
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

    def _stmt_for(self, st: TForStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        self._eval_for(st, env, fn_ret=fn_ret)

    def _stmt_match(self, st: TMatchStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        self._eval_match(st, env, fn_ret=fn_ret)

    def _stmt_try(self, st: TTryStmt, env: _RuntimeEnv, fn_ret: Type) -> None:
        self._eval_try(st, env, fn_ret=fn_ret)

    def _eval_try(self, st: TTryStmt, env: _RuntimeEnv, *, fn_ret: Type) -> None:
        try:
            self._eval_try_body(st, env, fn_ret=fn_ret)
        finally:
            if st.finally_body is not None:
                self._eval_block(st.finally_body, env, fn_ret=fn_ret)

    def _eval_try_body(self, st: TTryStmt, env: _RuntimeEnv, *, fn_ret: Type) -> None:
        try:
            self._eval_block(st.body, env, fn_ret=fn_ret)
        except _Throw as s:
            for c in st.catches:
                if not c.types or any(
                    self._matches_type(s.value, _resolve_type(t, self.checker))
                    for t in c.types
                ):
                    env.push_scope()
                    try:
                        if c.types:
                            c_ty = normalize_union(
                                [_resolve_type(t, self.checker) for t in c.types]
                            )
                        else:
                            c_ty = ERROR_T
                        env.bind(c.name, c_ty, s.value)
                        self._eval_block(c.body, env, fn_ret=fn_ret)
                    finally:
                        env.pop_scope()
                    return
            raise s

    def _eval_match(self, st: TMatchStmt, env: _RuntimeEnv, *, fn_ret: Type) -> None:
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
                case_ty = _resolve_type(pat.type_name, self.checker)
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
                env.bind(st.default.name, ERROR_T, v)
                self._eval_block(st.default.body, env, fn_ret=fn_ret)
            finally:
                env.pop_scope()
            return
        raise TaytshRuntimeFault("non-exhaustive match at runtime", st.pos)

    def _eval_for(self, st: TForStmt, env: _RuntimeEnv, *, fn_ret: Type) -> None:
        if isinstance(st.iterable, TRange):
            ints = [self._eval_expr(a, env) for a in st.iterable.args]
            vals: list[int] = []
            for iv in ints:
                if not isinstance(iv, VInt):
                    raise TaytshRuntimeFault(
                        "range() args must be int", st.iterable.pos
                    )
                vals.append(iv.value)
            if len(vals) == 1:
                start, end, step = 0, vals[0], 1
            elif len(vals) == 2:
                start = vals[0]
                end = vals[1]
                step = 1
            else:
                start = vals[0]
                end = vals[1]
                step = vals[2]
            if step == 0:
                self._throw_err("ValueError", "range() step must be nonzero")
            i = start
            while _range_cond(i, end, step):
                env.push_scope()
                try:
                    env.bind(st.binding[0], INT_T, VInt(i))
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
            list_snap: list[Value] = list(it.elements)
            for li, lval in enumerate(list_snap):
                env.push_scope()
                try:
                    if len(st.binding) == 1:
                        env.bind(st.binding[0], it.typ.element, lval)
                    elif st.annotations.get(
                        "iter_kind"
                    ) == "tuple_unpack" and isinstance(lval, VTuple):
                        ti = 0
                        while ti < len(st.binding) and ti < len(lval.elements):
                            env.bind(
                                st.binding[ti], lval.typ.elements[ti], lval.elements[ti]
                            )
                            ti += 1
                    else:
                        env.bind(st.binding[0], INT_T, VInt(li))
                        env.bind(st.binding[1], it.typ.element, lval)
                    self._eval_block(st.body, env, fn_ret=fn_ret)
                except _Continue:
                    pass
                except _Break:
                    return
                finally:
                    env.pop_scope()
            return

        if isinstance(it, VString):
            str_val: str = it.value
            si = 0
            while si < len(str_val):
                ch: str = str_val[si]
                env.push_scope()
                try:
                    if len(st.binding) == 1:
                        env.bind(st.binding[0], RUNE_T, VRune(ch))
                    else:
                        env.bind(st.binding[0], INT_T, VInt(si))
                        env.bind(st.binding[1], RUNE_T, VRune(ch))
                    self._eval_block(st.body, env, fn_ret=fn_ret)
                except _Continue:
                    pass
                except _Break:
                    return
                finally:
                    env.pop_scope()
                si += 1
            return

        if isinstance(it, VBytes):
            bi = 0
            while bi < len(it.value):
                bv = int(it.value[bi])
                env.push_scope()
                try:
                    if len(st.binding) == 1:
                        env.bind(st.binding[0], BYTE_T, VByte(bv))
                    else:
                        env.bind(st.binding[0], INT_T, VInt(bi))
                        env.bind(st.binding[1], BYTE_T, VByte(bv))
                    self._eval_block(st.body, env, fn_ret=fn_ret)
                except _Continue:
                    pass
                except _Break:
                    return
                finally:
                    env.pop_scope()
                bi += 1
            return

        if isinstance(it, VMap):
            snap_keys = list(it.map_keys)
            snap_vals = list(it.map_vals)
            for k, v in zip(snap_keys, snap_vals):
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
            if expr.name == "_":
                return _DiscardRef(NIL_T)
            return _VarRef(env.get_ty(expr.name), env, expr.name)
        if isinstance(expr, TFieldAccess):
            obj = self._eval_expr(expr.obj, env)
            if not isinstance(obj, VStruct):
                raise TaytshRuntimeFault("field access on non-struct", expr.pos)
            sinfo = self.index.structs.get(obj.struct_name)
            if sinfo is None or expr.field not in sinfo.field_map:
                raise TaytshRuntimeFault("unknown field", expr.pos)
            return _FieldRef(sinfo.field_map[expr.field].ty, obj, expr.field)
        if isinstance(expr, TIndex):
            obj = self._eval_expr(expr.obj, env)
            idx = self._eval_expr(expr.index, env)
            if isinstance(obj, VList):
                if not isinstance(idx, VInt):
                    raise TaytshRuntimeFault("list index not int", expr.pos)
                if idx.value < 0 or idx.value >= len(obj.elements):
                    self._throw_err("IndexError", "index out of bounds")
                return _ListIndexRef(obj.typ.element, obj, idx.value)
            if isinstance(obj, VMap):
                key = _as_hashable(idx)
                return _MapIndexRef(obj.typ.value, obj, key)
            raise TaytshRuntimeFault("index assignment only for list/map", expr.pos)
        raise TaytshRuntimeFault("invalid assignment target", expr.pos)

    def _eval_expr(
        self, expr: TExpr, env: _RuntimeEnv, *, expected: Type | None = None
    ) -> Value:
        if isinstance(expr, TVar):
            return self._expr_var(expr, env, expected)
        if isinstance(expr, TCall):
            return self._eval_call(expr, env, expected=expected)
        if isinstance(expr, TBinaryOp):
            return self._expr_binary(expr, env, expected)
        if isinstance(expr, TStringLit):
            return VString(expr.value)
        if isinstance(expr, TIntLit):
            return VInt(expr.value)
        if isinstance(expr, TBoolLit):
            return VBool(expr.value)
        if isinstance(expr, TFieldAccess):
            return self._expr_field_access(expr, env, expected)
        if isinstance(expr, TFloatLit):
            return VFloat(expr.value)
        if isinstance(expr, TNilLit):
            return VNil()
        if isinstance(expr, TUnaryOp):
            return self._expr_unary(expr, env, expected)
        if isinstance(expr, TTernary):
            return self._expr_ternary(expr, env, expected)
        if isinstance(expr, TIndex):
            return self._expr_index(expr, env, expected)
        if isinstance(expr, TFnLit):
            return self._expr_fnlit(expr, env, expected)
        if isinstance(expr, TByteLit):
            return self._expr_byte(expr, env, expected)
        if isinstance(expr, TRuneLit):
            return VRune(expr.value)
        if isinstance(expr, TBytesLit):
            return VBytes(expr.value)
        if isinstance(expr, TTupleAccess):
            return self._expr_tuple_access(expr, env, expected)
        if isinstance(expr, TSlice):
            return self._expr_slice(expr, env, expected)
        if isinstance(expr, TListLit):
            return self._expr_list(expr, env, expected)
        if isinstance(expr, TMapLit):
            return self._expr_map(expr, env, expected)
        if isinstance(expr, TSetLit):
            return self._expr_set(expr, env, expected)
        if isinstance(expr, TTupleLit):
            return self._expr_tuple(expr, env, expected)
        raise TaytshRuntimeFault("unsupported expression", expr.pos)

    def _expr_int(
        self, expr: TIntLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        return VInt(expr.value)

    def _expr_float(
        self, expr: TFloatLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        return VFloat(expr.value)

    def _expr_byte(
        self, expr: TByteLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        if expected is not None and expected.kind == "int":
            return VInt(expr.value)
        return VByte(expr.value)

    def _expr_string(
        self, expr: TStringLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        return VString(expr.value)

    def _expr_rune(
        self, expr: TRuneLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        return VRune(expr.value)

    def _expr_bytes(
        self, expr: TBytesLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        return VBytes(expr.value)

    def _expr_bool(
        self, expr: TBoolLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        return VBool(expr.value)

    def _expr_nil(
        self, expr: TNilLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        return VNil()

    def _expr_var(self, expr: TVar, env: _RuntimeEnv, expected: Type | None) -> Value:
        if env.has(expr.name):
            return env.get(expr.name)
        if expr.name in self._global_values:
            return self._global_values[expr.name]
        if expr.name in self._fn_values:
            return self._fn_values[expr.name]
        if expr.name in self._builtin_values:
            return self._builtin_values[expr.name]
        raise TaytshRuntimeFault(f"unknown name '{expr.name}'", expr.pos)

    def _expr_unary(
        self, expr: TUnaryOp, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        operand = self._eval_expr(expr.operand, env)
        if expr.op == "!":
            if not isinstance(operand, VBool):
                raise TaytshRuntimeFault("! operand not bool", expr.pos)
            return VBool(not operand.value)
        if expr.op == "-":
            if isinstance(operand, VInt):
                result = -operand.value
                if self.module.strict_math and (
                    result < _INT64_MIN or result > _INT64_MAX
                ):
                    self._throw_err("ValueError", "integer overflow")
                return VInt(result)
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

    def _expr_binary(
        self, expr: TBinaryOp, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
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
        if expr.op == "==" or expr.op == "!=":
            right = self._eval_expr(expr.right, env)
            left = self._eval_expr(expr.left, env, expected=right.ty())
        else:
            left = self._eval_expr(expr.left, env)
            right = self._eval_expr(expr.right, env)
        return self._eval_binary(expr.op, left, right, pos=expr.pos)

    def _expr_ternary(
        self, expr: TTernary, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        cond = self._eval_expr(expr.cond, env)
        if not isinstance(cond, VBool):
            raise TaytshRuntimeFault("ternary condition not bool", expr.pos)
        return self._eval_expr(
            expr.then_expr if cond.value else expr.else_expr, env, expected=expected
        )

    def _expr_tuple_access(
        self, expr: TTupleAccess, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        obj = self._eval_expr(expr.obj, env)
        if not isinstance(obj, VTuple):
            raise TaytshRuntimeFault("tuple access on non-tuple", expr.pos)
        return obj.elements[expr.index]

    def _expr_field_access(
        self, expr: TFieldAccess, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
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

    def _expr_index(
        self, expr: TIndex, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        obj = self._eval_expr(expr.obj, env)
        idx = self._eval_expr(expr.index, env)
        return self._eval_index(obj, idx, pos=expr.pos)

    def _expr_slice(
        self, expr: TSlice, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        obj = self._eval_expr(expr.obj, env)
        low = self._eval_expr(expr.low, env)
        high = self._eval_expr(expr.high, env)
        if not isinstance(low, VInt) or not isinstance(high, VInt):
            raise TaytshRuntimeFault("slice bounds not int", expr.pos)
        return self._eval_slice(obj, low.value, high.value, pos=expr.pos)

    def _expr_list(
        self, expr: TListLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        if expected is None:
            inferred = self.expr_types.get(_expr_key(expr))
            if isinstance(inferred, ListT):
                expected = inferred
        list_elems: list[Value] = []
        if isinstance(expected, ListT):
            for e in expr.elements:
                list_elems.append(self._eval_expr(e, env, expected=expected.element))
            list_typ: ListT = expected
        else:
            for e in expr.elements:
                list_elems.append(self._eval_expr(e, env))
            if not list_elems:
                raise TaytshRuntimeFault("cannot infer list type", expr.pos)
            list_typ = ListT(kind="list", element=list_elems[0].ty())
        return VList(list_elems, list_typ)

    def _expr_map(
        self, expr: TMapLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        if expected is None:
            inferred = self.expr_types.get(_expr_key(expr))
            if isinstance(inferred, MapT):
                expected = inferred
        mk: list[Value] = []
        mv: list[Value] = []
        if isinstance(expected, MapT):
            for k, v in expr.entries:
                kk = _as_hashable(self._eval_expr(k, env, expected=expected.key))
                vv = self._eval_expr(v, env, expected=expected.value)
                mk.append(kk)
                mv.append(vv)
            map_typ: MapT = expected
        else:
            for k, v in expr.entries:
                kk = _as_hashable(self._eval_expr(k, env))
                vv = self._eval_expr(v, env)
                mk.append(kk)
                mv.append(vv)
            map_typ = MapT(kind="map", key=mk[0].ty(), value=mv[0].ty())
        return VMap(mk, mv, map_typ)

    def _expr_set(
        self, expr: TSetLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        if expected is None:
            inferred = self.expr_types.get(_expr_key(expr))
            if isinstance(inferred, SetT):
                expected = inferred
        if isinstance(expected, SetT):
            set_elems: list[Value] = []
            for e in expr.elements:
                _list_set_add(
                    set_elems,
                    _as_hashable(self._eval_expr(e, env, expected=expected.element)),
                )
            set_typ: SetT = expected
        else:
            set_elems2: list[Value] = []
            for e in expr.elements:
                _list_set_add(set_elems2, _as_hashable(self._eval_expr(e, env)))
            set_typ = SetT(kind="set", element=set_elems2[0].ty())
            set_elems = set_elems2
        return VSet(set_elems, set_typ)

    def _expr_tuple(
        self, expr: TTupleLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        if expected is None:
            inferred = self.expr_types.get(_expr_key(expr))
            if isinstance(inferred, TupleT):
                expected = inferred
        tup_elems: list[Value] = []
        if isinstance(expected, TupleT) and len(expected.elements) >= len(
            expr.elements
        ):
            for ti, te in enumerate(expr.elements):
                tup_elems.append(
                    self._eval_expr(te, env, expected=expected.elements[ti])
                )
            tup_typ: TupleT = expected
        else:
            for e in expr.elements:
                tup_elems.append(self._eval_expr(e, env))
            tup_typ = TupleT(kind="tuple", elements=[e.ty() for e in tup_elems])
        return VTuple(tup_elems, tup_typ)

    def _expr_fnlit(
        self, expr: TFnLit, env: _RuntimeEnv, expected: Type | None
    ) -> Value:
        sig = _fn_lit_sig(expr, self.checker)
        return VFunc(sig.ty(), None, "fnlit", "", expr, sig, None)

    def _expr_call(self, expr: TCall, env: _RuntimeEnv, expected: Type | None) -> Value:
        return self._eval_call(expr, env, expected=expected)

    def _call_fn_lit(self, lit: TFnLit, sig: FnSig, args: list[Value]) -> Value:
        env = _RuntimeEnv()
        env.push_scope()
        for i, p in enumerate(lit.params):
            env.bind(p.name, sig.params[i], args[i])
        is_arrow = lit.annotations.get("fn_lit.arrow") == "true"
        if is_arrow:
            first_stmt = lit.body[0]
            if isinstance(first_stmt, TExprStmt):
                return self._eval_expr(first_stmt.expr, env, expected=sig.ret)
        try:
            self._eval_block(lit.body, env, fn_ret=sig.ret)
        except _Return as r:
            if type_eq(sig.ret, VOID_T):
                return VNil()
            if r.value is None:
                raise TaytshRuntimeFault("missing return value", lit.pos)
            return r.value
        if type_eq(sig.ret, VOID_T):
            return VNil()
        raise TaytshRuntimeFault("function literal fell off", lit.pos)

    def _invoke_vfunc(self, fnv: VFunc, args: list[Value]) -> Value:
        if fnv.kind == "user" and fnv.fn_decl is not None and fnv.fn_sig is not None:
            return self._call_fn(fnv.fn_decl, fnv.fn_sig, args)
        if fnv.kind == "builtin":
            return _dispatch_builtin(self, fnv.fn_key, args)
        if fnv.kind == "fnlit" and fnv.fn_lit is not None and fnv.fn_sig is not None:
            return self._call_fn_lit(fnv.fn_lit, fnv.fn_sig, args)
        raise TaytshRuntimeFault("unknown VFunc kind: " + fnv.kind, None)

    def _eval_call(
        self, call: TCall, env: _RuntimeEnv, *, expected: Type | None
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
                inferred = self.expr_types.get(_expr_key(call))
                expected = inferred
            if call.func.name == "Map":
                if not isinstance(expected, MapT):
                    raise TaytshRuntimeFault("cannot infer Map() type", call.pos)
                return VMap([], [], expected)
            if not isinstance(expected, SetT):
                raise TaytshRuntimeFault("cannot infer Set() type", call.pos)
            return VSet([], expected)

        # Built-in call by name
        if isinstance(call.func, TVar) and call.func.name in _BUILTIN_NAMES_RT:
            bi_expected = _builtin_arg_expected(call.func.name, expected)
            args: list[Value] = []
            for a in call.args:
                arg_expected: Type | None = self.expr_types.get(_expr_key(a.value))
                if bi_expected is not None:
                    arg_expected = bi_expected
                args.append(self._eval_expr(a.value, env, expected=arg_expected))
            return _dispatch_builtin(self, call.func.name, args)

        # Method call: obj.Method(...)
        if isinstance(call.func, TFieldAccess):
            recv = self._eval_expr(call.func.obj, env)
            if isinstance(recv, VMap) and call.func.field == "get":
                if len(call.args) != 1:
                    raise TaytshRuntimeFault("map.get expects 1 arg", call.pos)
                key = _as_hashable(self._eval_expr(call.args[0].value, env))
                idx = _map_find(recv, key)
                if idx < 0:
                    return VNil()
                return recv.map_vals[idx]
            if isinstance(recv, VStruct):
                s = self.index.structs.get(recv.struct_name)
                if s is not None and call.func.field in s.methods:
                    mi = s.methods[call.func.field]
                    margs: list[Value] = [recv]
                    for a in call.args:
                        margs.append(self._eval_expr(a.value, env))
                    args = margs
                    return self._call_fn(mi.decl, mi.sig, args)
            # Fall back: field value call
            fnv = self._eval_expr(call.func, env)
            if not isinstance(fnv, VFunc):
                raise TaytshRuntimeFault("call target not a function", call.pos)
            args = [self._eval_expr(a.value, env) for a in call.args]
            return self._invoke_vfunc(fnv, args)

        fnv = self._eval_expr(call.func, env)
        if not isinstance(fnv, VFunc):
            raise TaytshRuntimeFault("call target not a function", call.pos)
        args = [self._eval_expr(a.value, env) for a in call.args]
        return self._invoke_vfunc(fnv, args)

    def _eval_struct_ctor(self, call: TCall, env: _RuntimeEnv) -> VStruct:
        assert isinstance(call.func, TVar)
        s = self.index.structs[call.func.name]
        field_order = [f.name for f in s.fields]
        has_named = any(a.name is not None for a in call.args)
        has_pos = any(a.name is None for a in call.args)
        if has_named and has_pos:
            raise TaytshRuntimeFault("cannot mix named and positional args", call.pos)
        min_required = sum(1 for f in s.fields if not f.decl.has_default)
        fields: dict[str, Value] = {}
        if has_pos:
            if len(call.args) < min_required or len(call.args) > len(field_order):
                raise TaytshRuntimeFault("wrong number of constructor args", call.pos)
            for i, a in enumerate(call.args):
                fname = field_order[i]
                fty = s.field_map[fname].ty
                fields[fname] = self._eval_expr(a.value, env, expected=fty)
            for dname in field_order[len(call.args) :]:
                fields[dname] = self.zero_value(s.field_map[dname].ty)
        else:
            for a in call.args:
                assert a.name is not None
                if a.name not in s.field_map:
                    raise TaytshRuntimeFault("unknown struct field", call.pos)
                fty = s.field_map[a.name].ty
                fields[a.name] = self._eval_expr(a.value, env, expected=fty)
            for f in s.fields:
                if f.name not in fields:
                    if not f.decl.has_default:
                        raise TaytshRuntimeFault("missing struct field", call.pos)
                    fields[f.name] = self.zero_value(f.ty)
        return VStruct(s.name, fields)

    def _matches_type(self, v: Value, typ: Type) -> bool:
        if _is_error(typ):
            return True
        if _is_nil(typ):
            return isinstance(v, VNil)
        if isinstance(typ, StructT):
            return isinstance(v, VStruct) and v.struct_name == typ.name
        if isinstance(typ, InterfaceT):
            return (
                isinstance(v, VStruct)
                and v.struct_name in self.index.interfaces[typ.name].implementors
            )
        if isinstance(typ, EnumT):
            return isinstance(v, VEnum) and v.enum_name == typ.name
        if isinstance(typ, UnionT):
            return any(self._matches_type(v, m) for m in typ.members)
        return type_eq(v.ty(), typ)

    def _eval_index(self, obj: Value, idx: Value, *, pos: Pos) -> Value:
        if not isinstance(idx, VInt) and not isinstance(obj, VMap):
            raise TaytshRuntimeFault("index must be int", pos)
        if isinstance(obj, VList) and isinstance(idx, VInt):
            if idx.value < 0 or idx.value >= len(obj.elements):
                self._throw_err("IndexError", "index out of bounds")
            return obj.elements[idx.value]
        if isinstance(obj, VString) and isinstance(idx, VInt):
            if idx.value < 0 or idx.value >= len(obj.value):
                self._throw_err("IndexError", "index out of bounds")
            return VRune(obj.value[idx.value])
        if isinstance(obj, VBytes) and isinstance(idx, VInt):
            if idx.value < 0 or idx.value >= len(obj.value):
                self._throw_err("IndexError", "index out of bounds")
            bval = int(obj.value[idx.value])
            return VByte(bval)
        if isinstance(obj, VMap):
            key = _as_hashable(idx)
            if not _map_has(obj, key):
                self._throw_err("KeyError", "missing key")
            return _map_get(obj, key)
        raise TaytshRuntimeFault("indexing not supported", pos)

    def _eval_slice(self, obj: Value, lo: int, hi: int, *, pos: Pos) -> Value:
        if lo < 0:
            self._throw_err("IndexError", "slice out of range")
        if isinstance(obj, VList):
            n = len(obj.elements)
            lo, hi = _clamp_slice(lo, hi, n)
            return VList(list(obj.elements[lo:hi]), obj.typ)
        if isinstance(obj, VString):
            n = len(obj.value)
            lo, hi = _clamp_slice(lo, hi, n)
            return VString(obj.value[lo:hi])
        if isinstance(obj, VBytes):
            n = len(obj.value)
            lo, hi = _clamp_slice(lo, hi, n)
            return VBytes(obj.value[lo:hi])
        if isinstance(obj, VTuple):
            n = len(obj.elements)
            lo, hi = _clamp_slice(lo, hi, n)
            elems = list(obj.elements[lo:hi])
            typ = TupleT(kind="tuple", elements=list(obj.typ.elements[lo:hi]))
            return VTuple(elems, typ)
        raise TaytshRuntimeFault("slicing not supported", pos)

    def _eval_binary(self, op: str, left: Value, right: Value, *, pos: Pos) -> Value:
        if op == "==":
            return VBool(_value_eq(left, right))
        if op == "!=":
            return VBool(not _value_eq(left, right))

        # Numeric / ordered ops.
        if op in ("<", "<=", ">", ">="):
            if isinstance(left, VInt) and isinstance(right, VInt):
                return VBool(_cmp_int(op, left.value, right.value))
            if isinstance(left, VFloat) and isinstance(right, VFloat):
                return VBool(_cmp_float(op, left.value, right.value))
            if isinstance(left, VByte) and isinstance(right, VByte):
                return VBool(_cmp_int(op, left.value, right.value))
            if isinstance(left, VInt) and isinstance(right, VByte):
                return VBool(_cmp_int(op, left.value, right.value))
            if isinstance(left, VByte) and isinstance(right, VInt):
                return VBool(_cmp_int(op, left.value, right.value))
            if isinstance(left, VRune) and isinstance(right, VRune):
                return VBool(_cmp_str(op, left.value, right.value))
            if isinstance(left, VString) and isinstance(right, VString):
                return VBool(_cmp_str(op, left.value, right.value))
            if isinstance(left, VBytes) and isinstance(right, VBytes):
                return VBool(_cmp_bytes(op, left.value, right.value))
            if isinstance(left, VTuple) and isinstance(right, VTuple):
                return VBool(_cmp_tuples(op, left, right))
            if isinstance(left, VList) and isinstance(right, VList):
                return VBool(_cmp_lists(op, left, right))
            if isinstance(left, (VTuple, VList)) and isinstance(right, (VTuple, VList)):
                le: list[Value] = []
                re: list[Value] = []
                if isinstance(left, VTuple):
                    le = left.elements
                elif isinstance(left, VList):
                    le = left.elements
                if isinstance(right, VTuple):
                    re = right.elements
                elif isinstance(right, VList):
                    re = right.elements
                return VBool(_cmp_seqs(op, le, re))
            raise TaytshRuntimeFault("invalid comparison operands", pos)

        if op in ("|", "^", "&"):
            lv: int | None = None
            rv: int | None = None
            is_byte_only = isinstance(left, VByte) and isinstance(right, VByte)
            if isinstance(left, (VInt, VByte)):
                lv = left.value
            if isinstance(right, (VInt, VByte)):
                rv = right.value
            if lv is not None and rv is not None:
                if op == "|":
                    result = lv | rv
                elif op == "^":
                    result = lv ^ rv
                else:
                    result = lv & rv
                return VByte(result) if is_byte_only else VInt(result)
            raise TaytshRuntimeFault("invalid bitwise operands", pos)

        if op in ("<<", ">>", ">>>"):
            if not isinstance(right, VInt):
                raise TaytshRuntimeFault("shift amount not int", pos)
            shift = right.value
            if shift < 0:
                self._throw_err("ValueError", "shift amount must be non-negative")
            if self.module.strict_math and isinstance(left, VInt) and shift >= 64:
                self._throw_err("ValueError", "shift amount >= 64")
            if isinstance(left, VInt):
                if op == "<<":
                    result = left.value << shift
                    if self.module.strict_math and (
                        result < _INT64_MIN or result > _INT64_MAX
                    ):
                        self._throw_err("ValueError", "integer overflow")
                    return VInt(result)
                if op == ">>>":
                    unsigned = left.value & 0xFFFFFFFFFFFFFFFF
                    result = unsigned >> shift
                    if result > _INT64_MAX:
                        result -= 1 << 64
                    return VInt(result)
                return VInt(left.value >> shift)
            if isinstance(left, VByte):
                if op == "<<":
                    val = left.value << shift
                else:
                    val = left.value >> shift
                return VByte(val & 0xFF)
            raise TaytshRuntimeFault("invalid shift operands", pos)

        if op in ("+", "-", "*", "/", "%"):
            if isinstance(left, VInt) and isinstance(right, VInt):
                if op in ("+", "-", "*"):
                    if op == "+":
                        result = left.value + right.value
                    elif op == "-":
                        result = left.value - right.value
                    else:
                        result = left.value * right.value
                    if self.module.strict_math and (
                        result < _INT64_MIN or result > _INT64_MAX
                    ):
                        self._throw_err("ValueError", "integer overflow")
                    return VInt(result)
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
            if isinstance(left, VFloat):
                if isinstance(right, VFloat):
                    if op == "+":
                        return VFloat(left.value + right.value)
                    if op == "-":
                        return VFloat(left.value - right.value)
                    if op == "*":
                        return VFloat(left.value * right.value)
                    if op == "/":
                        if right.value == 0.0:
                            if left.value == 0.0:
                                return VFloat(float("nan"))
                            return VFloat(_copysign_inf(left.value))
                        return VFloat(left.value / right.value)
                    if right.value == 0.0:
                        if self.module.strict_math:
                            self._throw_err("ValueError", "float modulo by zero")
                        return VFloat(float("nan"))
                    return VFloat(_fmod(left.value, right.value))
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


def _cmp_int(op: str, a: int, b: int) -> bool:
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    return a >= b


def _cmp_float(op: str, a: float, b: float) -> bool:
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    return a >= b


def _cmp_str(op: str, a: str, b: str) -> bool:
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    return a >= b


def _cmp_bytes(op: str, a: bytes, b: bytes) -> bool:
    i = 0
    while i < len(a) and i < len(b):
        ai = a[i]
        bi = b[i]
        if ai < bi:
            return op in ("<", "<=")
        if ai > bi:
            return op in (">", ">=")
        i += 1
    if len(a) < len(b):
        return op in ("<", "<=")
    if len(a) > len(b):
        return op in (">", ">=")
    return op in ("<=", ">=")


def _cmp_seqs(op: str, a: list[Value], b: list[Value]) -> bool:
    i = 0
    while i < len(a) and i < len(b):
        if _value_lt(a[i], b[i]):
            return op in ("<", "<=")
        if _value_lt(b[i], a[i]):
            return op in (">", ">=")
        i += 1
    la = len(a)
    lb = len(b)
    if op == "<":
        return la < lb
    if op == "<=":
        return la <= lb
    if op == ">":
        return la > lb
    return la >= lb


def _cmp_tuples(op: str, a: VTuple, b: VTuple) -> bool:
    return _cmp_seqs(op, a.elements, b.elements)


def _cmp_lists(op: str, a: VList, b: VList) -> bool:
    return _cmp_seqs(op, a.elements, b.elements)


# ---- Minimal builtin runtime (expanded in step 4) --------------------------


def _strict_tostring(v: Value, rt: Runtime, *, in_composite: bool = False) -> str:
    """Strict ToString: canonical format for cross-target consistency."""
    if isinstance(v, VNil):
        return "nil"
    if isinstance(v, VBool):
        return "true" if v.value else "false"
    if isinstance(v, VInt):
        return str(v.value)
    if isinstance(v, VFloat):
        if _isnan(v.value):
            return "NaN"
        if v.value == float("inf"):
            return "Inf"
        if v.value == float("-inf"):
            return "-Inf"
        return repr(v.value)
    if isinstance(v, VByte):
        return str(v.value)
    if isinstance(v, VRune):
        return f"'{v.value}'" if in_composite else v.value
    if isinstance(v, VString):
        return f'"{v.value}"' if in_composite else v.value
    if isinstance(v, VBytes):
        hex_chars = "0123456789abcdef"
        hex_list: list[str] = []
        bi = 0
        while bi < len(v.value):
            bvi = int(v.value[bi])
            hex_list.append("\\x" + hex_chars[bvi >> 4] + hex_chars[bvi & 0x0F])
            bi += 1
        hex_parts = "".join(hex_list)
        return 'b"' + hex_parts + '"'
    if isinstance(v, VList):
        inner = ", ".join(
            _strict_tostring(e, rt, in_composite=True) for e in v.elements
        )
        return f"[{inner}]"
    if isinstance(v, VTuple):
        inner = ", ".join(
            _strict_tostring(e, rt, in_composite=True) for e in v.elements
        )
        return f"({inner})"
    if isinstance(v, VMap):
        map_keys_sk: list[tuple[int, float, str]] = []
        map_indices: list[int] = []
        for map_di, mk in enumerate(v.map_keys):
            map_keys_sk.append(_sort_key(mk))
            map_indices.append(map_di)
        _sort_decorated(map_keys_sk, map_indices)
        map_parts: list[str] = []
        for map_idx in map_indices:
            map_parts.append(
                _strict_tostring(v.map_keys[map_idx], rt, in_composite=True)
                + ": "
                + _strict_tostring(v.map_vals[map_idx], rt, in_composite=True)
            )
        return "{" + ", ".join(map_parts) + "}"
    if isinstance(v, VSet):
        set_keys_sk: list[tuple[int, float, str]] = []
        set_indices: list[int] = []
        for set_di, se in enumerate(v.elements):
            set_keys_sk.append(_sort_key(se))
            set_indices.append(set_di)
        _sort_decorated(set_keys_sk, set_indices)
        set_parts: list[str] = []
        for set_idx in set_indices:
            set_parts.append(
                _strict_tostring(v.elements[set_idx], rt, in_composite=True)
            )
        set_inner = ", ".join(set_parts)
        return "{" + set_inner + "}"
    if isinstance(v, VEnum):
        return f"{v.enum_name}.{v.variant}"
    if isinstance(v, VStruct):
        si = rt.index.structs.get(v.struct_name)
        if si is not None:
            for ts_name in ("ToString", "__repr__", "to_string"):
                if ts_name in si.methods:
                    mi = si.methods[ts_name]
                    result = rt._call_fn(mi.decl, mi.sig, [v])
                    if isinstance(result, VString):
                        return result.value
                    break
        struct_parts: list[str] = []
        for fname in v.fields:
            struct_parts.append(
                fname + ": " + _strict_tostring(v.fields[fname], rt, in_composite=True)
            )
        return v.struct_name + "{" + ", ".join(struct_parts) + "}"
    if isinstance(v, VFunc):
        return type_name(v.ty())
    return v.to_string()


def _bi_tostring(rt: Runtime, args: list[Value]) -> Value:
    if rt.module.strict_tostring:
        return VString(_strict_tostring(args[0], rt))
    v = args[0]
    if isinstance(v, VStruct):
        si = rt.index.structs.get(v.struct_name)
        if si is not None:
            for ts_name in ("ToString", "__repr__", "to_string"):
                if ts_name in si.methods:
                    mi = si.methods[ts_name]
                    result = rt._call_fn(mi.decl, mi.sig, [v])
                    if isinstance(result, VString):
                        return result
                    break
    return VString(v.to_string())


def _bi_len(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        return VInt(len(x.value))
    if isinstance(x, VBytes):
        return VInt(len(x.value))
    if isinstance(x, VList):
        return VInt(len(x.elements))
    if isinstance(x, VMap):
        return VInt(len(x.map_keys))
    if isinstance(x, VSet):
        return VInt(len(x.elements))
    raise TaytshRuntimeFault("Len unsupported", None)


def _bi_get(rt: Runtime, args: list[Value]) -> Value:
    m = args[0]
    if not isinstance(m, VMap):
        raise TaytshRuntimeFault("Get expects map", None)
    key = _as_hashable(args[1])
    if _map_has(m, key):
        return _map_get(m, key)
    if len(args) == 3:
        return args[2]
    return VNil()


def _bi_contains(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if isinstance(a, VList):
        return VBool(any(_value_eq(x, b) for x in a.elements))
    if isinstance(a, VMap):
        return VBool(_map_has(a, _as_hashable(b)))
    if isinstance(a, VSet):
        return VBool(_set_has(a, _as_hashable(b)))
    if isinstance(a, VString) and isinstance(b, VString):
        return VBool(b.value in a.value)
    if isinstance(a, VString) and isinstance(b, VRune):
        return VBool(b.value in a.value)
    if isinstance(a, VBytes) and isinstance(b, VBytes):
        return VBool(b.value in a.value)
    if isinstance(a, VBytes) and isinstance(b, VInt):
        return VBool(bytes([b.value & 0xFF]) in a.value)
    if isinstance(a, VTuple):
        return VBool(any(_value_eq(x, b) for x in a.elements))
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


# ---------------------------------------------------------------------------
# Numeric builtins
# ---------------------------------------------------------------------------


def _bi_round(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VFloat):
        raise TaytshRuntimeFault("Round expects float", None)

    if _isnan(x.value) or _isinf(x.value):
        rt._throw_err("ValueError", "Round on non-finite float")
    # half-away-from-zero
    if x.value >= 0.0:
        return VInt(_floor(x.value + 0.5))
    return VInt(_ceil(x.value - 0.5))


def _bi_floor(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VFloat):
        raise TaytshRuntimeFault("Floor expects float", None)

    if _isnan(x.value) or _isinf(x.value):
        rt._throw_err("ValueError", "Floor on non-finite float")
    return VInt(_floor(x.value))


def _bi_ceil(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VFloat):
        raise TaytshRuntimeFault("Ceil expects float", None)

    if _isnan(x.value) or _isinf(x.value):
        rt._throw_err("ValueError", "Ceil on non-finite float")
    return VInt(_ceil(x.value))


def _bi_sqrt(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VFloat):
        raise TaytshRuntimeFault("Sqrt expects float", None)

    return VFloat(_sqrt(x.value))


def _bi_isnan(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VFloat):
        raise TaytshRuntimeFault("IsNaN expects float", None)

    return VBool(_isnan(x.value))


def _bi_isinf(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VFloat):
        raise TaytshRuntimeFault("IsInf expects float", None)

    return VBool(_isinf(x.value))


def _bi_divmod(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if not isinstance(a, VInt) or not isinstance(b, VInt):
        raise TaytshRuntimeFault("DivMod expects int, int", None)
    if b.value == 0:
        rt._throw_err("ZeroDivisionError", "division by zero")
    q, r = _int_divmod_trunc(a.value, b.value)
    return VTuple([VInt(q), VInt(r)], TupleT(kind="tuple", elements=[INT_T, INT_T]))


def _bi_floor_div(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if isinstance(a, VFloat) or isinstance(b, VFloat):
        fa: float = 0.0
        fb: float = 0.0
        if isinstance(a, VFloat):
            fa = a.value
        elif isinstance(a, VInt):
            fa = float(a.value)
        else:
            raise TaytshRuntimeFault("FloorDiv expects numeric types", None)
        if isinstance(b, VFloat):
            fb = b.value
        elif isinstance(b, VInt):
            fb = float(b.value)
        else:
            raise TaytshRuntimeFault("FloorDiv expects numeric types", None)
        if fb == 0.0:
            rt._throw_err("ZeroDivisionError", "division by zero")
        return VFloat(fa // fb)
    if not isinstance(a, VInt) or not isinstance(b, VInt):
        raise TaytshRuntimeFault("FloorDiv expects numeric types", None)
    if b.value == 0:
        rt._throw_err("ZeroDivisionError", "division by zero")
    return VInt(a.value // b.value)


def _bi_python_mod(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if isinstance(a, VFloat) or isinstance(b, VFloat):
        fa: float = 0.0
        fb: float = 0.0
        if isinstance(a, VFloat):
            fa = a.value
        elif isinstance(a, VInt):
            fa = float(a.value)
        else:
            raise TaytshRuntimeFault("PythonMod expects numeric types", None)
        if isinstance(b, VFloat):
            fb = b.value
        elif isinstance(b, VInt):
            fb = float(b.value)
        else:
            raise TaytshRuntimeFault("PythonMod expects numeric types", None)
        if fb == 0.0:
            rt._throw_err("ZeroDivisionError", "division by zero")
        return VFloat(fa % fb)
    if not isinstance(a, VInt) or not isinstance(b, VInt):
        raise TaytshRuntimeFault("PythonMod expects numeric types", None)
    if b.value == 0:
        rt._throw_err("ZeroDivisionError", "division by zero")
    return VInt(a.value % b.value)


def _wrap_i64(val: int) -> int:
    """Wrap an arbitrary Python int to signed int64 range."""
    val = val & 0xFFFFFFFFFFFFFFFF
    if val > _INT64_MAX:
        val -= 1 << 64
    return val


def _bi_wrapping_add(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if not isinstance(a, VInt) or not isinstance(b, VInt):
        raise TaytshRuntimeFault("WrappingAdd expects int, int", None)
    return VInt(_wrap_i64(a.value + b.value))


def _bi_wrapping_sub(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if not isinstance(a, VInt) or not isinstance(b, VInt):
        raise TaytshRuntimeFault("WrappingSub expects int, int", None)
    return VInt(_wrap_i64(a.value - b.value))


def _bi_wrapping_mul(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if not isinstance(a, VInt) or not isinstance(b, VInt):
        raise TaytshRuntimeFault("WrappingMul expects int, int", None)
    return VInt(_wrap_i64(a.value * b.value))


def _bi_abs(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VInt):
        return VInt(abs(x.value))
    if isinstance(x, VFloat):
        return VFloat(abs(x.value))
    raise TaytshRuntimeFault("Abs expects int or float", None)


def _bi_min(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    if len(args) == 1:
        items: list[Value] | None = None
        if isinstance(a, VList):
            items = a.elements
        elif isinstance(a, VTuple):
            items = a.elements
        if items is not None:
            if not items:
                rt._throw_err("ValueError", "min() arg is an empty sequence")
            best = items[0]
            for item in items[1:]:
                if _value_lt(item, best):
                    best = item
            return best
    b = args[1]
    if isinstance(a, VInt) and isinstance(b, VInt):
        return VInt(min(a.value, b.value))
    if isinstance(a, VFloat) and isinstance(b, VFloat):
        if _isnan(a.value) or _isnan(b.value):
            return VFloat(float("nan"))
        return VFloat(min(a.value, b.value))
    if isinstance(a, VByte) and isinstance(b, VByte):
        return VByte(min(a.value, b.value))
    raise TaytshRuntimeFault("Min expects matching numeric types", None)


def _bi_max(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    if len(args) == 1:
        items: list[Value] | None = None
        if isinstance(a, VList):
            items = a.elements
        elif isinstance(a, VTuple):
            items = a.elements
        if items is not None:
            if not items:
                rt._throw_err("ValueError", "max() arg is an empty sequence")
            best = items[0]
            for item in items[1:]:
                if _value_lt(best, item):
                    best = item
            return best
    b = args[1]
    if isinstance(a, VInt) and isinstance(b, VInt):
        return VInt(max(a.value, b.value))
    if isinstance(a, VFloat) and isinstance(b, VFloat):
        if _isnan(a.value) or _isnan(b.value):
            return VFloat(float("nan"))
        return VFloat(max(a.value, b.value))
    if isinstance(a, VByte) and isinstance(b, VByte):
        return VByte(max(a.value, b.value))
    raise TaytshRuntimeFault("Max expects matching numeric types", None)


def _bi_sum(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    if isinstance(xs, (VSet, VTuple)):
        itotal: int = 0
        for e in xs.elements:
            if not isinstance(e, VInt):
                raise TaytshRuntimeFault("Sum elements must be numeric", None)
            itotal += e.value
        return VInt(itotal)
    if not isinstance(xs, VList):
        raise TaytshRuntimeFault("Sum expects list, set, or tuple", None)
    if not xs.elements:
        if type_eq(xs.typ.element, FLOAT_T):
            return VFloat(0.0)
        return VInt(0)
    if isinstance(xs.elements[0], VFloat):
        ftotal: float = 0.0
        for e in xs.elements:
            if not isinstance(e, VFloat):
                raise TaytshRuntimeFault("Sum elements must be numeric", None)
            ftotal += e.value
        return VFloat(ftotal)
    itotal: int = 0
    for e in xs.elements:
        if not isinstance(e, VInt):
            raise TaytshRuntimeFault("Sum elements must be numeric", None)
        itotal += e.value
    return VInt(itotal)


def _bi_pow(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if isinstance(a, VInt) and isinstance(b, VInt):
        if rt.module.strict_math:
            if b.value < 0:
                rt._throw_err("ValueError", "Pow(int, int) with negative exponent")
            result = a.value**b.value
            if result < _INT64_MIN or result > _INT64_MAX:
                rt._throw_err("ValueError", "integer overflow")
            return VInt(result)
        return VInt(a.value**b.value)
    if isinstance(a, VFloat) and isinstance(b, VFloat):
        return VFloat(a.value**b.value)
    raise TaytshRuntimeFault("Pow expects matching numeric types", None)


# ---------------------------------------------------------------------------
# Conversion builtins
# ---------------------------------------------------------------------------


def _bi_int_to_float(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VInt):
        raise TaytshRuntimeFault("IntToFloat expects int", None)
    val: int = x.value
    return VFloat(float(val))


def _bi_float_to_int(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VFloat):
        raise TaytshRuntimeFault("FloatToInt expects float", None)
    if _isnan(x.value) or _isinf(x.value):
        rt._throw_err("ValueError", "FloatToInt on non-finite float")
    val: float = x.value
    return VInt(int(val))


def _bi_byte_to_int(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VByte):
        raise TaytshRuntimeFault("ByteToInt expects byte", None)
    return VInt(x.value)


def _bi_int_to_byte(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VInt):
        raise TaytshRuntimeFault("IntToByte expects int", None)
    return VByte(x.value & 0xFF)


def _bi_rune_from_int(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VInt):
        raise TaytshRuntimeFault("RuneFromInt expects int", None)
    return VRune(chr(x.value))


def _bi_rune_to_int(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if not isinstance(x, VRune):
        raise TaytshRuntimeFault("RuneToInt expects rune", None)
    ch: str = x.value
    return VInt(ord(ch))


# ---------------------------------------------------------------------------
# String builtins
# ---------------------------------------------------------------------------


def _bi_parse_int(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    base = args[1]
    if not isinstance(s, VString) or not isinstance(base, VInt):
        raise TaytshRuntimeFault("ParseInt expects string, int", None)
    try:
        return VInt(int(s.value, base.value))
    except (ValueError, OverflowError):
        rt._throw_err("ValueError", "invalid integer: " + repr(s.value))
    return VNil()  # unreachable


def _bi_parse_float(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    if not isinstance(s, VString):
        raise TaytshRuntimeFault("ParseFloat expects string", None)
    try:
        return VFloat(float(s.value))
    except ValueError:
        rt._throw_err("ValueError", "invalid float: " + repr(s.value))
    return VNil()


def _bi_format_int(rt: Runtime, args: list[Value]) -> Value:
    n = args[0]
    base = args[1]
    if not isinstance(n, VInt) or not isinstance(base, VInt):
        raise TaytshRuntimeFault("FormatInt expects int, int", None)
    b = base.value
    if b == 10:
        return VString(str(n.value))
    if b == 16:
        return VString(hex(n.value)[2:])
    if n.value == 0:
        return VString("0")
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    neg = n.value < 0
    val = abs(n.value)
    chars: list[str] = []
    while val > 0:
        chars.append(digits[val % b])
        val = val // b
    if neg:
        chars.append("-")
    chars.reverse()
    return VString("".join(chars))


def _bi_upper(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    if isinstance(s, VBytes):
        return VBytes(s.value.upper())
    if not isinstance(s, VString):
        raise TaytshRuntimeFault("Upper expects string or bytes", None)
    return VString(s.value.upper())


def _bi_lower(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    if isinstance(s, VBytes):
        return VBytes(s.value.lower())
    if not isinstance(s, VString):
        raise TaytshRuntimeFault("Lower expects string or bytes", None)
    return VString(s.value.lower())


def _bi_trim(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    chars = args[1]
    if isinstance(s, VBytes):
        if isinstance(chars, VBytes):
            return VBytes(s.value.strip(chars.value))
        if isinstance(chars, VString):
            return VBytes(s.value.strip(chars.value.encode()))
        raise TaytshRuntimeFault("Trim expects compatible types", None)
    if not isinstance(s, VString) or not isinstance(chars, VString):
        raise TaytshRuntimeFault("Trim expects string/string or bytes/bytes", None)
    return VString(s.value.strip(chars.value))


def _bi_trim_start(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    chars = args[1]
    if isinstance(s, VBytes):
        if isinstance(chars, VBytes):
            return VBytes(s.value.lstrip(chars.value))
        if isinstance(chars, VString):
            return VBytes(s.value.lstrip(chars.value.encode()))
        raise TaytshRuntimeFault("TrimStart expects compatible types", None)
    if not isinstance(s, VString) or not isinstance(chars, VString):
        raise TaytshRuntimeFault("TrimStart expects string/string or bytes/bytes", None)
    return VString(s.value.lstrip(chars.value))


def _bi_trim_end(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    chars = args[1]
    if isinstance(s, VBytes):
        if isinstance(chars, VBytes):
            return VBytes(s.value.rstrip(chars.value))
        if isinstance(chars, VString):
            return VBytes(s.value.rstrip(chars.value.encode()))
        raise TaytshRuntimeFault("TrimEnd expects compatible types", None)
    if not isinstance(s, VString) or not isinstance(chars, VString):
        raise TaytshRuntimeFault("TrimEnd expects string/string or bytes/bytes", None)
    return VString(s.value.rstrip(chars.value))


def _bi_split(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    sep = args[1]
    if isinstance(s, VBytes) and isinstance(sep, VBytes):
        if sep.value == b"":
            rt._throw_err("ValueError", "Split separator must not be empty")
        parts_b = s.value.split(sep.value)
        elems_b: list[Value] = []
        for p in parts_b:
            elems_b.append(VBytes(p))
        return VList(elems_b, ListT(kind="list", element=BYTES_T))
    if not isinstance(s, VString) or not isinstance(sep, VString):
        raise TaytshRuntimeFault("Split expects string/string or bytes/bytes", None)
    if not sep.value:
        rt._throw_err("ValueError", "Split separator must not be empty")
    parts = s.value.split(sep.value)
    elems: list[Value] = []
    for p in parts:
        elems.append(VString(p))
    return VList(elems, ListT(kind="list", element=STRING_T))


def _bi_split_n(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    sep = args[1]
    n = args[2]
    if (
        not isinstance(s, VString)
        or not isinstance(sep, VString)
        or not isinstance(n, VInt)
    ):
        raise TaytshRuntimeFault("SplitN expects string, string, int", None)
    if n.value <= 0:
        rt._throw_err("ValueError", "SplitN max must be > 0")
    parts = s.value.split(sep.value, n.value - 1)
    elems: list[Value] = []
    for p in parts:
        elems.append(VString(p))
    return VList(elems, ListT(kind="list", element=STRING_T))


def _bi_split_whitespace(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    if not isinstance(s, VString):
        raise TaytshRuntimeFault("SplitWhitespace expects string", None)
    parts = s.value.split()
    elems: list[Value] = []
    for p in parts:
        elems.append(VString(p))
    return VList(elems, ListT(kind="list", element=STRING_T))


def _bi_join(rt: Runtime, args: list[Value]) -> Value:
    sep = args[0]
    parts = args[1]
    if isinstance(sep, VBytes) and isinstance(parts, VList):
        bs: list[bytes] = []
        for e in parts.elements:
            if isinstance(e, VBytes):
                bs.append(e.value)
            else:
                raise TaytshRuntimeFault("Join list element not bytes", None)
        return VBytes(sep.value.join(bs))
    if not isinstance(sep, VString) or not isinstance(parts, VList):
        raise TaytshRuntimeFault("Join expects string, list[string]", None)
    strs: list[str] = []
    for e in parts.elements:
        if isinstance(e, VString):
            strs.append(e.value)
        else:
            raise TaytshRuntimeFault("Join list element not string", None)
    return VString(sep.value.join(strs))


def _bi_find(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    sub = args[1]
    if isinstance(s, VBytes) and isinstance(sub, VBytes):
        return VInt(s.value.find(sub.value))
    if not isinstance(s, VString) or not isinstance(sub, VString):
        raise TaytshRuntimeFault("Find expects string/string or bytes/bytes", None)
    return VInt(s.value.find(sub.value))


def _bi_rfind(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    sub = args[1]
    if not isinstance(s, VString) or not isinstance(sub, VString):
        raise TaytshRuntimeFault("RFind expects string, string", None)
    return VInt(s.value.rfind(sub.value))


def _bi_count(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    sub = args[1]
    if isinstance(s, VList):
        n = 0
        for el in s.elements:
            if _value_eq(el, sub):
                n += 1
        return VInt(n)
    if isinstance(s, VBytes) and isinstance(sub, VBytes):
        return VInt(s.value.count(sub.value))
    if not isinstance(s, VString) or not isinstance(sub, VString):
        raise TaytshRuntimeFault(
            "Count expects string/string, bytes/bytes, or list/value", None
        )
    return VInt(s.value.count(sub.value))


def _bi_replace(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    old = args[1]
    new = args[2]
    if isinstance(s, VBytes) and isinstance(old, VBytes) and isinstance(new, VBytes):
        return VBytes(s.value.replace(old.value, new.value))
    if (
        not isinstance(s, VString)
        or not isinstance(old, VString)
        or not isinstance(new, VString)
    ):
        raise TaytshRuntimeFault(
            "Replace expects string/string/string or bytes/bytes/bytes", None
        )
    return VString(s.value.replace(old.value, new.value))


def _bi_replace_count(rt: Runtime, args: list[Value]) -> Value:
    s: Value = args[0]
    old: Value = args[1]
    repl: Value = args[2]
    count: Value = args[3]
    if (
        not isinstance(s, VString)
        or not isinstance(old, VString)
        or not isinstance(repl, VString)
        or not isinstance(count, VInt)
    ):
        raise TaytshRuntimeFault(
            "ReplaceCount expects string, string, string, int", None
        )
    return VString(s.value.replace(old.value, repl.value, count.value))


def _bi_starts_with(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    pre = args[1]
    if isinstance(s, VBytes) and isinstance(pre, VBytes):
        return VBool(s.value.startswith(pre.value))
    if not isinstance(s, VString) or not isinstance(pre, VString):
        raise TaytshRuntimeFault("StartsWith expects string, string", None)
    return VBool(s.value.startswith(pre.value))


def _bi_ends_with(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    suf = args[1]
    if isinstance(s, VBytes) and isinstance(suf, VBytes):
        return VBool(s.value.endswith(suf.value))
    if not isinstance(s, VString) or not isinstance(suf, VString):
        raise TaytshRuntimeFault("EndsWith expects string, string", None)
    return VBool(s.value.endswith(suf.value))


def _bi_encode(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    if not isinstance(s, VString):
        raise TaytshRuntimeFault("Encode expects string", None)
    return VBytes(s.value.encode("utf-8"))


def _decode_utf8(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _Throw(VStruct("ValueError", {"message": VString(str(e))})) from e


def _bi_decode(rt: Runtime, args: list[Value]) -> Value:
    b = args[0]
    if not isinstance(b, VBytes):
        raise TaytshRuntimeFault("Decode expects bytes", None)
    return VString(_decode_utf8(b.value))


def _bi_concat(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if isinstance(a, VString) and isinstance(b, VString):
        return VString(a.value + b.value)
    if isinstance(a, VBytes) and isinstance(b, VBytes):
        return VBytes(a.value + b.value)
    if isinstance(a, VList) and isinstance(b, VList):
        merged: list[Value] = list(a.elements) + list(b.elements)
        return VList(merged, a.typ)
    if isinstance(a, VList) and isinstance(b, VTuple):
        merged = list(a.elements) + list(b.elements)
        return VList(merged, a.typ)
    if isinstance(a, VTuple) and isinstance(b, VList):
        merged = list(a.elements) + list(b.elements)
        return VList(merged, b.typ)
    if isinstance(a, VTuple) and isinstance(b, VTuple):
        elem_ty: Type = ERROR_T
        if a.typ.elements:
            elem_ty = a.typ.elements[0]
        merged = list(a.elements) + list(b.elements)
        return VList(merged, ListT(kind="list", element=elem_ty))
    raise TaytshRuntimeFault("Concat expects matching string, bytes, or list", None)


def _bi_repeat(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    n = args[1]
    if not isinstance(n, VInt):
        raise TaytshRuntimeFault("Repeat expects int count", None)
    if isinstance(a, VString):
        return VString(a.value * max(0, n.value))
    if isinstance(a, VList):
        return VList(list(a.elements) * max(0, n.value), a.typ)
    if isinstance(a, VTuple):
        elem_ty = a.typ.elements[0] if a.typ.elements else ERROR_T
        return VList(
            list(a.elements) * max(0, n.value), ListT(kind="list", element=elem_ty)
        )
    if isinstance(a, VBytes):
        return VBytes(a.value * max(0, n.value))
    raise TaytshRuntimeFault("Repeat expects string or list", None)


def _bi_format(rt: Runtime, args: list[Value]) -> Value:
    template = args[0]
    if not isinstance(template, VString):
        raise TaytshRuntimeFault("Format expects string template", None)
    parts = template.value.split("{}")
    if len(parts) - 1 != len(args) - 1:
        raise TaytshRuntimeFault("Format: placeholder count mismatch", None)
    result: list[str] = [parts[0]]
    for fi, arg in enumerate(args[1:], 1):
        if not isinstance(arg, VString):
            raise TaytshRuntimeFault("Format arguments must be string", None)
        result.append(arg.value)
        result.append(parts[fi])
    return VString("".join(result))


# ---------------------------------------------------------------------------
# Character classifier builtins
# ---------------------------------------------------------------------------


def _bi_is_digit(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        return VBool(len(x.value) > 0 and all(str(c).isdigit() for c in x.value))
    if isinstance(x, VRune):
        return VBool(str(x.value).isdigit())
    raise TaytshRuntimeFault("IsDigit expects string or rune", None)


def _bi_is_alpha(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        return VBool(len(x.value) > 0 and all(str(c).isalpha() for c in x.value))
    if isinstance(x, VRune):
        return VBool(str(x.value).isalpha())
    raise TaytshRuntimeFault("IsAlpha expects string or rune", None)


def _bi_is_alnum(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        return VBool(len(x.value) > 0 and all(str(c).isalnum() for c in x.value))
    if isinstance(x, VRune):
        return VBool(str(x.value).isalnum())
    raise TaytshRuntimeFault("IsAlnum expects string or rune", None)


def _bi_is_space(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        return VBool(len(x.value) > 0 and all(str(c).isspace() for c in x.value))
    if isinstance(x, VRune):
        return VBool(str(x.value).isspace())
    raise TaytshRuntimeFault("IsSpace expects string or rune", None)


def _bi_is_upper(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        return VBool(x.value.isupper())
    if isinstance(x, VRune):
        return VBool(str(x.value).isupper())
    raise TaytshRuntimeFault("IsUpper expects string or rune", None)


def _bi_is_lower(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        return VBool(x.value.islower())
    if isinstance(x, VRune):
        return VBool(str(x.value).islower())
    raise TaytshRuntimeFault("IsLower expects string or rune", None)


# ---------------------------------------------------------------------------
# List builtins
# ---------------------------------------------------------------------------


def _bi_append(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    v = args[1]
    if not isinstance(xs, VList):
        raise TaytshRuntimeFault("Append expects list", None)
    xs.elements.append(v)
    return VNil()


def _bi_insert(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    i = args[1]
    v = args[2]
    if not isinstance(xs, VList) or not isinstance(i, VInt):
        raise TaytshRuntimeFault("Insert expects list, int, value", None)
    if i.value < 0 or i.value > len(xs.elements):
        rt._throw_err("IndexError", "Insert index out of bounds")
    xs.elements.insert(i.value, v)
    return VNil()


def _bi_pop(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    if isinstance(xs, VSet):
        if not xs.elements:
            rt._throw_err("KeyError", "Pop on empty set")
        return xs.elements.pop()
    if not isinstance(xs, VList):
        raise TaytshRuntimeFault("Pop expects list or set", None)
    if not xs.elements:
        rt._throw_err("IndexError", "Pop on empty list")
    return xs.elements.pop()


def _bi_is_type(rt: Runtime, args: list[Value]) -> Value:
    v = args[0]
    tn = args[1]
    if not isinstance(tn, VString):
        return VBool(False)
    type_name = tn.value
    if isinstance(v, VStruct):
        if v.struct_name == type_name:
            return VBool(True)
        for decl in rt.module.decls:
            if isinstance(decl, TStructDecl) and decl.name == v.struct_name:
                if decl.parent == type_name:
                    return VBool(True)
                break
        return VBool(False)
    if isinstance(v, VEnum):
        return VBool(v.enum_name == type_name)
    if isinstance(v, VInt):
        return VBool(type_name == "int")
    if isinstance(v, VFloat):
        return VBool(type_name == "float")
    if isinstance(v, VBool):
        return VBool(type_name == "bool")
    if isinstance(v, VString):
        return VBool(type_name == "string")
    if isinstance(v, VByte):
        return VBool(type_name == "byte")
    if isinstance(v, VBytes):
        return VBool(type_name == "bytes")
    if isinstance(v, VRune):
        return VBool(type_name == "rune")
    if isinstance(v, VNil):
        return VBool(type_name == "nil")
    if isinstance(v, VList):
        return VBool(type_name == "list")
    if isinstance(v, VMap):
        return VBool(type_name == "map" or type_name == "dict")
    if isinstance(v, VSet):
        return VBool(type_name == "set")
    if isinstance(v, VTuple):
        return VBool(type_name == "tuple")
    return VBool(False)


def _bi_replace_slice(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    lo = args[1]
    hi = args[2]
    vals = args[3]
    if (
        not isinstance(xs, VList)
        or not isinstance(lo, VInt)
        or not isinstance(hi, VInt)
        or not isinstance(vals, VList)
    ):
        raise TaytshRuntimeFault("ReplaceSlice expects list, int, int, list", None)
    xs.elements[lo.value : hi.value] = vals.elements
    return VNil()


def _bi_remove_at(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    i = args[1]
    if not isinstance(xs, VList) or not isinstance(i, VInt):
        raise TaytshRuntimeFault("RemoveAt expects list, int", None)
    if i.value < 0 or i.value >= len(xs.elements):
        rt._throw_err("IndexError", "RemoveAt index out of bounds")
    xs.elements.pop(i.value)
    return VNil()


def _bi_index_of(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    v = args[1]
    if not isinstance(xs, VList):
        raise TaytshRuntimeFault("IndexOf expects list", None)
    for i, e in enumerate(xs.elements):
        if _value_eq(e, v):
            return VInt(i)
    return VInt(-1)


def _bi_reverse(rt: Runtime, args: list[Value]) -> Value:
    """Reverse a string."""
    s = args[0]
    if not isinstance(s, VString):
        raise TaytshRuntimeFault("Reverse expects string", None)
    return VString(s.value[::-1])


def _bi_reversed(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    if isinstance(xs, VBytes):
        return VBytes(bytes(reversed(xs.value)))
    if not isinstance(xs, VList):
        raise TaytshRuntimeFault("Reversed expects list or bytes", None)
    rev: list[Value] = []
    i = len(xs.elements) - 1
    while i >= 0:
        rev.append(xs.elements[i])
        i -= 1
    return VList(rev, xs.typ)


def _sort_key(v: Value) -> tuple[int, float, str]:
    if isinstance(v, VInt):
        return (0, v.value + 0.0, "")
    if isinstance(v, VFloat):
        return (1, v.value, "")
    if isinstance(v, VByte):
        return (2, v.value + 0.0, "")
    if isinstance(v, VRune):
        return (3, 0.0, v.value)
    if isinstance(v, VString):
        return (4, 0.0, v.value)
    raise TaytshRuntimeFault("Sorted: unsupported element type", None)


def _cmp_sort_key(a: tuple[int, float, str], b: tuple[int, float, str]) -> int:
    """Compare two sort keys. Returns -1, 0, or 1."""
    if a[0] < b[0]:
        return -1
    if a[0] > b[0]:
        return 1
    if a[1] < b[1]:
        return -1
    if a[1] > b[1]:
        return 1
    if a[2] < b[2]:
        return -1
    if a[2] > b[2]:
        return 1
    return 0


def _sort_decorated(keys: list[tuple[int, float, str]], indices: list[int]) -> None:
    """Insertion sort on parallel key/index arrays."""
    i = 1
    while i < len(keys):
        k = keys[i]
        idx = indices[i]
        j = i - 1
        while j >= 0 and _cmp_sort_key(keys[j], k) > 0:
            keys[j + 1] = keys[j]
            indices[j + 1] = indices[j]
            j -= 1
        keys[j + 1] = k
        indices[j + 1] = idx
        i += 1


def _bi_sorted(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    if isinstance(xs, VSet):
        s_keys: list[tuple[int, float, str]] = []
        s_idx: list[int] = []
        for sort_si, se in enumerate(xs.elements):
            s_keys.append(_sort_key(se))
            s_idx.append(sort_si)
        _sort_decorated(s_keys, s_idx)
        sresult: list[Value] = []
        for si in s_idx:
            sresult.append(xs.elements[si])
        return VList(sresult, ListT(kind="list", element=xs.typ.element))
    if isinstance(xs, VTuple):
        if not xs.elements:
            return VList([], ListT(kind="list", element=INT_T))
        t_keys: list[tuple[int, float, str]] = []
        t_idx: list[int] = []
        for sort_ti, te in enumerate(xs.elements):
            t_keys.append(_sort_key(te))
            t_idx.append(sort_ti)
        _sort_decorated(t_keys, t_idx)
        tresult: list[Value] = []
        for ti in t_idx:
            tresult.append(xs.elements[ti])
        elem_t = xs.typ.elements[0] if xs.typ.elements else INT_T
        return VList(tresult, ListT(kind="list", element=elem_t))
    if not isinstance(xs, VList):
        raise TaytshRuntimeFault("Sorted expects list, set, or tuple", None)
    if rt.module.strict_math:
        for e in xs.elements:
            if isinstance(e, VFloat) and _isnan(e.value):
                rt._throw_err("ValueError", "Sorted: list contains NaN")
    l_keys: list[tuple[int, float, str]] = []
    l_idx: list[int] = []
    for sort_li, le in enumerate(xs.elements):
        l_keys.append(_sort_key(le))
        l_idx.append(sort_li)
    _sort_decorated(l_keys, l_idx)
    lresult: list[Value] = []
    for li in l_idx:
        lresult.append(xs.elements[li])
    return VList(lresult, xs.typ)


# ---------------------------------------------------------------------------
# Map builtins
# ---------------------------------------------------------------------------


def _bi_delete(rt: Runtime, args: list[Value]) -> Value:
    m = args[0]
    k = args[1]
    if not isinstance(m, VMap):
        raise TaytshRuntimeFault("Delete expects map", None)
    key = _as_hashable(k)
    _map_del(m, key)
    return VNil()


def _bi_keys(rt: Runtime, args: list[Value]) -> Value:
    m = args[0]
    if not isinstance(m, VMap):
        raise TaytshRuntimeFault("Keys expects map", None)
    return VList(list(m.map_keys), ListT(kind="list", element=m.typ.key))


def _bi_values(rt: Runtime, args: list[Value]) -> Value:
    m = args[0]
    if not isinstance(m, VMap):
        raise TaytshRuntimeFault("Values expects map", None)
    return VList(list(m.map_vals), ListT(kind="list", element=m.typ.value))


def _bi_items(rt: Runtime, args: list[Value]) -> Value:
    m = args[0]
    if not isinstance(m, VMap):
        raise TaytshRuntimeFault("Items expects map", None)
    pair_ty = TupleT(kind="tuple", elements=[m.typ.key, m.typ.value])
    elems: list[Value] = []
    for mi, mk in enumerate(m.map_keys):
        elems.append(VTuple([mk, m.map_vals[mi]], pair_ty))
    return VList(elems, ListT(kind="list", element=pair_ty))


def _bi_map_from_keys(rt: Runtime, args: list[Value]) -> Value:
    keys_val = args[0]
    default_val = args[1]
    if not isinstance(keys_val, VList):
        raise TaytshRuntimeFault("MapFromKeys expects list", None)
    mk: list[Value] = []
    mv: list[Value] = []
    for k in keys_val.elements:
        mk.append(k)
        mv.append(default_val)
    key_ty = keys_val.typ.element if isinstance(keys_val.typ, ListT) else ERROR_T
    val_ty = default_val.ty() if not isinstance(default_val, VNil) else ERROR_T
    return VMap(mk, mv, MapT(kind="map", key=key_ty, value=val_ty))


def _bi_pop_item(rt: Runtime, args: list[Value]) -> Value:
    m = args[0]
    if not isinstance(m, VMap):
        raise TaytshRuntimeFault("PopItem expects map", None)
    if not m.map_keys:
        raise TaytshRuntimeFault("PopItem on empty map", None)
    last_key = m.map_keys.pop()
    last_val = m.map_vals.pop()
    pair_ty = TupleT(kind="tuple", elements=[m.typ.key, m.typ.value])
    return VTuple([last_key, last_val], pair_ty)


def _bi_merge(rt: Runtime, args: list[Value]) -> Value:
    m1 = args[0]
    m2 = args[1]
    if not isinstance(m1, VMap) or not isinstance(m2, VMap):
        raise TaytshRuntimeFault("Merge expects map, map", None)
    mk = list(m1.map_keys)
    mv = list(m1.map_vals)
    result = VMap(mk, mv, m1.typ)
    for mi, mk2 in enumerate(m2.map_keys):
        _map_set(result, mk2, m2.map_vals[mi])
    return result


# ---------------------------------------------------------------------------
# Set builtins
# ---------------------------------------------------------------------------


def _bi_add(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    v = args[1]
    if not isinstance(s, VSet):
        raise TaytshRuntimeFault("Add expects set", None)
    _set_add(s, _as_hashable(v))
    return VNil()


def _bi_remove(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    v = args[1]
    if not isinstance(s, VSet):
        raise TaytshRuntimeFault("Remove expects set", None)
    _set_discard(s, _as_hashable(v))
    return VNil()


def _bi_union(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if not isinstance(a, VSet) or not isinstance(b, VSet):
        raise TaytshRuntimeFault("Union expects two sets", None)
    return VSet(_set_union(a.elements, b.elements), a.typ)


def _bi_intersection(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if not isinstance(a, VSet) or not isinstance(b, VSet):
        raise TaytshRuntimeFault("Intersection expects two sets", None)
    return VSet(_set_intersection(a.elements, b.elements), a.typ)


def _bi_difference(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if not isinstance(a, VSet) or not isinstance(b, VSet):
        raise TaytshRuntimeFault("Difference expects two sets", None)
    return VSet(_set_difference(a.elements, b.elements), a.typ)


def _bi_bytes_ctor(rt: Runtime, args: list[Value]) -> Value:
    n = args[0]
    if not isinstance(n, VInt):
        raise TaytshRuntimeFault("Bytes expects int", None)
    return VBytes(b"\x00" * n.value)


def _bi_bytes_from(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    if not isinstance(xs, VList):
        raise TaytshRuntimeFault("BytesFrom expects list", None)
    result: list[int] = []
    for v in xs.elements:
        if isinstance(v, VByte):
            result.append(v.value)
        elif isinstance(v, VInt):
            result.append(v.value & 0xFF)
        else:
            raise TaytshRuntimeFault("BytesFrom element must be byte or int", None)
    return VBytes(bytes(result))


def _bi_range_list(rt: Runtime, args: list[Value]) -> Value:
    start = args[0]
    end = args[1]
    step = args[2]
    if (
        not isinstance(start, VInt)
        or not isinstance(end, VInt)
        or not isinstance(step, VInt)
    ):
        raise TaytshRuntimeFault("RangeList expects int, int, int", None)
    elements: list[Value] = []
    ri = start.value
    while (step.value > 0 and ri < end.value) or (step.value < 0 and ri > end.value):
        elements.append(VInt(ri))
        ri += step.value
    return VList(elements, ListT(kind="list", element=INT_T))


def _bi_map_from_pairs(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    if not isinstance(xs, VList):
        raise TaytshRuntimeFault("MapFromPairs expects list", None)
    mk: list[Value] = []
    mv: list[Value] = []
    key_typ = ERROR_T
    val_typ = ERROR_T
    for v in xs.elements:
        if not isinstance(v, VTuple) or len(v.elements) != 2:
            raise TaytshRuntimeFault("MapFromPairs elements must be 2-tuples", None)
        k = v.elements[0]
        val = v.elements[1]
        mk.append(_as_hashable(k))
        mv.append(val)
    if isinstance(xs.typ, ListT) and isinstance(xs.typ.element, TupleT):
        elems = xs.typ.element.elements
        if len(elems) == 2:
            key_typ = elems[0]
            val_typ = elems[1]
    return VMap(mk, mv, MapT(kind="map", key=key_typ, value=val_typ))


def _bi_list_compare(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if not isinstance(a, VList) or not isinstance(b, VList):
        raise TaytshRuntimeFault("ListCompare expects two lists", None)
    min_len = min(len(a.elements), len(b.elements))
    i = 0
    while i < min_len:
        av = a.elements[i]
        bv = b.elements[i]
        c = _compare_values(av, bv)
        if c != 0:
            return VInt(c)
        i += 1
    if len(a.elements) < len(b.elements):
        return VInt(-1)
    if len(a.elements) > len(b.elements):
        return VInt(1)
    return VInt(0)


def _bi_set_from_list(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    if isinstance(xs, VSet):
        return xs
    if not isinstance(xs, VList):
        raise TaytshRuntimeFault("SetFromList expects list or set", None)
    elems: list[Value] = []
    for v in xs.elements:
        _list_set_add(elems, _as_hashable(v))
    return VSet(elems, SetT(kind="set", element=xs.typ.element))


def _bi_chars(rt: Runtime, args: list[Value]) -> Value:
    s = args[0]
    if not isinstance(s, VString):
        raise TaytshRuntimeFault("Chars expects string", None)
    elems: list[Value] = []
    for c in s.value:
        elems.append(VString(str(c)))
    return VList(elems, ListT(kind="list", element=STRING_T))


def _bi_list_from(rt: Runtime, args: list[Value]) -> Value:
    xs = args[0]
    if isinstance(xs, VList):
        return VList(list(xs.elements), xs.typ)
    if isinstance(xs, VSet):
        return VList(list(xs.elements), ListT(kind="list", element=xs.typ.element))
    raise TaytshRuntimeFault("ListFrom expects list or set", None)


def _bi_zip(rt: Runtime, args: list[Value]) -> Value:
    a = args[0]
    b = args[1]
    if not isinstance(a, VList) or not isinstance(b, VList):
        raise TaytshRuntimeFault("Zip expects two lists", None)
    min_len = min(len(a.elements), len(b.elements))
    elem_ty = TupleT(kind="tuple", elements=[a.typ.element, b.typ.element])
    result: list[Value] = []
    i = 0
    while i < min_len:
        result.append(VTuple([a.elements[i], b.elements[i]], elem_ty))
        i += 1
    return VList(result, ListT(kind="list", element=elem_ty))


def _compare_values(a: Value, b: Value) -> int:
    """Compare two values, returning -1, 0, or 1."""
    if isinstance(a, VInt) and isinstance(b, VInt):
        return -1 if a.value < b.value else (1 if a.value > b.value else 0)
    if isinstance(a, VFloat) and isinstance(b, VFloat):
        return -1 if a.value < b.value else (1 if a.value > b.value else 0)
    if isinstance(a, VString) and isinstance(b, VString):
        return -1 if a.value < b.value else (1 if a.value > b.value else 0)
    if isinstance(a, VByte) and isinstance(b, VByte):
        return -1 if a.value < b.value else (1 if a.value > b.value else 0)
    if isinstance(a, VBool) and isinstance(b, VBool):
        ai = 1 if a.value else 0
        bi = 1 if b.value else 0
        return -1 if ai < bi else (1 if ai > bi else 0)
    return 0


# ---------------------------------------------------------------------------
# I/O builtins
# ---------------------------------------------------------------------------


def _bi_write_out(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        rt.stdout = rt.stdout + x.value.encode("utf-8")
    elif isinstance(x, VBytes):
        rt.stdout = rt.stdout + x.value
    else:
        raise TaytshRuntimeFault("WriteOut expects string or bytes", None)
    return VNil()


def _bi_write_err(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        rt.stderr = rt.stderr + x.value.encode("utf-8")
    elif isinstance(x, VBytes):
        rt.stderr = rt.stderr + x.value
    else:
        raise TaytshRuntimeFault("WriteErr expects string or bytes", None)
    return VNil()


def _bi_writeln_out(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        rt.stdout = rt.stdout + (x.value + "\n").encode("utf-8")
    elif isinstance(x, VBytes):
        rt.stdout = rt.stdout + x.value + b"\n"
    else:
        raise TaytshRuntimeFault("WritelnOut expects string or bytes", None)
    return VNil()


def _bi_writeln_err(rt: Runtime, args: list[Value]) -> Value:
    x = args[0]
    if isinstance(x, VString):
        rt.stderr = rt.stderr + (x.value + "\n").encode("utf-8")
    elif isinstance(x, VBytes):
        rt.stderr = rt.stderr + x.value + b"\n"
    else:
        raise TaytshRuntimeFault("WritelnErr expects string or bytes", None)
    return VNil()


def _bi_read_line(rt: Runtime, args: list[Value]) -> Value:
    line = rt.stdin.read_line()
    if line is None:
        return VNil()
    text = _decode_utf8(line)
    if text.endswith("\r\n"):
        text = text[:-2]
    elif text.endswith("\n"):
        text = text[:-1]
    return VString(text)


def _bi_read_all(rt: Runtime, args: list[Value]) -> Value:
    return VString(_decode_utf8(rt.stdin.read_all()))


def _bi_read_bytes(rt: Runtime, args: list[Value]) -> Value:
    return VBytes(rt.stdin.read_all())


def _bi_read_bytes_n(rt: Runtime, args: list[Value]) -> Value:
    n = args[0]
    if not isinstance(n, VInt):
        raise TaytshRuntimeFault("ReadBytesN expects int", None)
    return VBytes(rt.stdin.read_n(n.value))


def _bi_read_file(rt: Runtime, args: list[Value]) -> Value:
    path = args[0]
    if not isinstance(path, VString):
        raise TaytshRuntimeFault("ReadFile expects string", None)
    try:
        with open(path.value, "rb") as f:
            data = f.read()
        try:
            return VString(data.decode("utf-8"))
        except UnicodeDecodeError as e:
            rt._throw_err("ValueError", str(e))
            return VNil()
    except OSError as e:
        rt._throw_err("IOError", str(e))
        return VNil()


def _bi_read_file_bytes(rt: Runtime, args: list[Value]) -> Value:
    path = args[0]
    if not isinstance(path, VString):
        raise TaytshRuntimeFault("ReadFileBytes expects string", None)
    try:
        with open(path.value, "rb") as f:
            data = f.read()
        return VBytes(data)
    except OSError as e:
        rt._throw_err("IOError", str(e))
        return VNil()


def _bi_write_file(rt: Runtime, args: list[Value]) -> Value:
    path = args[0]
    data = args[1]
    if not isinstance(path, VString):
        raise TaytshRuntimeFault("WriteFile expects string path", None)
    try:
        if isinstance(data, VString):
            with open(path.value, "w") as f:
                f.write(data.value)
        elif isinstance(data, VBytes):
            with open(path.value, "wb") as f:
                f.write(data.value)
        else:
            raise TaytshRuntimeFault("WriteFile expects string or bytes", None)
    except OSError as e:
        rt._throw_err("IOError", str(e))
    return VNil()


def _bi_args(rt: Runtime, args: list[Value]) -> Value:
    elems: list[Value] = []
    for a in rt.args[1:]:
        elems.append(VString(a))
    return VList(elems, ListT(kind="list", element=STRING_T))


def _bi_get_env(rt: Runtime, args: list[Value]) -> Value:
    name = args[0]
    if not isinstance(name, VString):
        raise TaytshRuntimeFault("GetEnv expects string", None)
    val = rt.env.get(name.value)
    if val is None:
        return VNil()
    return VString(val)


def _bi_exit(rt: Runtime, args: list[Value]) -> Value:
    code = args[0]
    if not isinstance(code, VInt):
        raise TaytshRuntimeFault("Exit expects int", None)
    raise _Exit(code.value)


def _dispatch_builtin(rt: Runtime, name: str, args: list[Value]) -> Value:
    """Dispatch a builtin function call by name."""
    if name in ("ToString", "ToRepr"):
        return _bi_tostring(rt, args)
    if name == "Len":
        return _bi_len(rt, args)
    if name == "Get":
        return _bi_get(rt, args)
    if name == "Contains":
        return _bi_contains(rt, args)
    if name == "Unwrap":
        return _bi_unwrap(rt, args)
    if name == "Assert":
        return _bi_assert(rt, args)
    if name == "Round":
        return _bi_round(rt, args)
    if name == "Floor":
        return _bi_floor(rt, args)
    if name == "Ceil":
        return _bi_ceil(rt, args)
    if name == "Sqrt":
        return _bi_sqrt(rt, args)
    if name == "IsNaN":
        return _bi_isnan(rt, args)
    if name == "IsInf":
        return _bi_isinf(rt, args)
    if name == "DivMod":
        return _bi_divmod(rt, args)
    if name == "WrappingAdd":
        return _bi_wrapping_add(rt, args)
    if name == "WrappingSub":
        return _bi_wrapping_sub(rt, args)
    if name == "WrappingMul":
        return _bi_wrapping_mul(rt, args)
    if name == "Abs":
        return _bi_abs(rt, args)
    if name == "Min":
        return _bi_min(rt, args)
    if name == "Max":
        return _bi_max(rt, args)
    if name == "Sum":
        return _bi_sum(rt, args)
    if name == "Pow":
        return _bi_pow(rt, args)
    if name == "IntToFloat":
        return _bi_int_to_float(rt, args)
    if name == "FloatToInt":
        return _bi_float_to_int(rt, args)
    if name == "ByteToInt":
        return _bi_byte_to_int(rt, args)
    if name == "IntToByte":
        return _bi_int_to_byte(rt, args)
    if name == "RuneFromInt":
        return _bi_rune_from_int(rt, args)
    if name == "RuneToInt":
        return _bi_rune_to_int(rt, args)
    if name == "ParseInt":
        return _bi_parse_int(rt, args)
    if name == "ParseFloat":
        return _bi_parse_float(rt, args)
    if name == "FormatInt":
        return _bi_format_int(rt, args)
    if name == "Upper":
        return _bi_upper(rt, args)
    if name == "Lower":
        return _bi_lower(rt, args)
    if name == "Trim":
        return _bi_trim(rt, args)
    if name == "TrimStart":
        return _bi_trim_start(rt, args)
    if name == "TrimEnd":
        return _bi_trim_end(rt, args)
    if name == "Split":
        return _bi_split(rt, args)
    if name == "SplitN":
        return _bi_split_n(rt, args)
    if name == "SplitWhitespace":
        return _bi_split_whitespace(rt, args)
    if name == "Join":
        return _bi_join(rt, args)
    if name == "Find":
        return _bi_find(rt, args)
    if name == "RFind":
        return _bi_rfind(rt, args)
    if name == "Count":
        return _bi_count(rt, args)
    if name == "Replace":
        return _bi_replace(rt, args)
    if name == "ReplaceCount":
        return _bi_replace_count(rt, args)
    if name == "StartsWith":
        return _bi_starts_with(rt, args)
    if name == "EndsWith":
        return _bi_ends_with(rt, args)
    if name == "Encode":
        return _bi_encode(rt, args)
    if name == "Decode":
        return _bi_decode(rt, args)
    if name == "Concat":
        return _bi_concat(rt, args)
    if name == "Repeat":
        return _bi_repeat(rt, args)
    if name == "Format":
        return _bi_format(rt, args)
    if name == "IsDigit":
        return _bi_is_digit(rt, args)
    if name == "IsAlpha":
        return _bi_is_alpha(rt, args)
    if name == "IsAlnum":
        return _bi_is_alnum(rt, args)
    if name == "IsSpace":
        return _bi_is_space(rt, args)
    if name == "IsUpper":
        return _bi_is_upper(rt, args)
    if name == "IsLower":
        return _bi_is_lower(rt, args)
    if name == "Append":
        return _bi_append(rt, args)
    if name == "Insert":
        return _bi_insert(rt, args)
    if name == "Pop":
        return _bi_pop(rt, args)
    if name == "RemoveAt":
        return _bi_remove_at(rt, args)
    if name == "IndexOf":
        return _bi_index_of(rt, args)
    if name == "Reverse":
        return _bi_reverse(rt, args)
    if name == "Reversed":
        return _bi_reversed(rt, args)
    if name == "Sorted":
        return _bi_sorted(rt, args)
    if name == "Delete":
        return _bi_delete(rt, args)
    if name == "Keys":
        return _bi_keys(rt, args)
    if name == "Values":
        return _bi_values(rt, args)
    if name == "Items":
        return _bi_items(rt, args)
    if name == "Merge":
        return _bi_merge(rt, args)
    if name == "PopItem":
        return _bi_pop_item(rt, args)
    if name == "MapFromKeys":
        return _bi_map_from_keys(rt, args)
    if name == "Add":
        return _bi_add(rt, args)
    if name == "Remove":
        return _bi_remove(rt, args)
    if name == "Union":
        return _bi_union(rt, args)
    if name == "Intersection":
        return _bi_intersection(rt, args)
    if name == "Difference":
        return _bi_difference(rt, args)
    if name == "Bytes":
        return _bi_bytes_ctor(rt, args)
    if name == "BytesFrom":
        return _bi_bytes_from(rt, args)
    if name == "RangeList":
        return _bi_range_list(rt, args)
    if name == "MapFromPairs":
        return _bi_map_from_pairs(rt, args)
    if name == "ListCompare":
        return _bi_list_compare(rt, args)
    if name == "Zip":
        return _bi_zip(rt, args)
    if name == "SetFromList":
        return _bi_set_from_list(rt, args)
    if name == "Chars":
        return _bi_chars(rt, args)
    if name == "ListFrom":
        return _bi_list_from(rt, args)
    if name == "WriteOut":
        return _bi_write_out(rt, args)
    if name == "WriteErr":
        return _bi_write_err(rt, args)
    if name == "WritelnOut":
        return _bi_writeln_out(rt, args)
    if name == "WritelnErr":
        return _bi_writeln_err(rt, args)
    if name == "ReadLine":
        return _bi_read_line(rt, args)
    if name == "ReadAll":
        return _bi_read_all(rt, args)
    if name == "ReadBytes":
        return _bi_read_bytes(rt, args)
    if name == "ReadBytesN":
        return _bi_read_bytes_n(rt, args)
    if name == "ReadFile":
        return _bi_read_file(rt, args)
    if name == "ReadFileBytes":
        return _bi_read_file_bytes(rt, args)
    if name == "WriteFile":
        return _bi_write_file(rt, args)
    if name == "Args":
        return _bi_args(rt, args)
    if name == "GetEnv":
        return _bi_get_env(rt, args)
    if name == "Exit":
        return _bi_exit(rt, args)
    if name == "FloorDiv":
        return _bi_floor_div(rt, args)
    if name == "PythonMod":
        return _bi_python_mod(rt, args)
    if name == "IsNil":
        return VBool(isinstance(args[0], VNil))
    if name == "IsType":
        return _bi_is_type(rt, args)
    if name == "ReplaceSlice":
        return _bi_replace_slice(rt, args)
    raise TaytshRuntimeFault("unknown builtin: " + name, None)


_BUILTIN_NAMES_RT: set[str] = {
    "ToRepr",
    "ToString",
    "Len",
    "Get",
    "Contains",
    "Unwrap",
    "Assert",
    "Round",
    "Floor",
    "Ceil",
    "Sqrt",
    "IsNaN",
    "IsInf",
    "DivMod",
    "WrappingAdd",
    "WrappingSub",
    "WrappingMul",
    "Abs",
    "Min",
    "Max",
    "Sum",
    "Pow",
    "IntToFloat",
    "FloatToInt",
    "ByteToInt",
    "IntToByte",
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
    "Replace",
    "ReplaceCount",
    "StartsWith",
    "EndsWith",
    "Encode",
    "Decode",
    "Concat",
    "Repeat",
    "Format",
    "IsDigit",
    "IsAlpha",
    "IsAlnum",
    "IsSpace",
    "IsUpper",
    "IsLower",
    "Append",
    "Insert",
    "Pop",
    "RemoveAt",
    "IndexOf",
    "Reverse",
    "Reversed",
    "Sorted",
    "Delete",
    "Keys",
    "Values",
    "Items",
    "Merge",
    "PopItem",
    "MapFromKeys",
    "Add",
    "Remove",
    "Union",
    "Intersection",
    "Difference",
    "Bytes",
    "BytesFrom",
    "RangeList",
    "MapFromPairs",
    "ListCompare",
    "Zip",
    "SetFromList",
    "Chars",
    "ListFrom",
    "WriteOut",
    "WriteErr",
    "WritelnOut",
    "WritelnErr",
    "ReadLine",
    "ReadAll",
    "ReadBytes",
    "ReadBytesN",
    "ReadFile",
    "ReadFileBytes",
    "WriteFile",
    "Args",
    "GetEnv",
    "Exit",
    "FloorDiv",
    "PythonMod",
    "IsNil",
    "IsType",
    "ReplaceSlice",
}


def _call_intrinsic(name: str, args: list[Value]) -> Value | None:
    """Native implementations of hot user functions. Returns None if not an intrinsic."""
    if name == "is_whitespace":
        c = args[0]
        if isinstance(c, VString):
            return VBool(c.value == " " or c.value == "\t")
        return VBool(False)
    if name == "is_alpha":
        c = args[0]
        if isinstance(c, VString) and len(c.value) == 1:
            v = c.value
            return VBool(("a" <= v <= "z") or ("A" <= v <= "Z") or v == "_")
        return VBool(False)
    if name == "is_alnum":
        c = args[0]
        if isinstance(c, VString) and len(c.value) == 1:
            v = c.value
            return VBool(
                ("a" <= v <= "z") or ("A" <= v <= "Z") or v == "_" or ("0" <= v <= "9")
            )
        return VBool(False)
    if name == "is_digit":
        c = args[0]
        if isinstance(c, VString) and len(c.value) == 1:
            return VBool("0" <= c.value <= "9")
        return VBool(False)
    return None
