"""Taytsh AST — parse-time node definitions."""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# POSITION
# ============================================================


@dataclass
class Pos:
    """Source position, 1-indexed."""

    line: int
    col: int


# ============================================================
# TYPES (parse-time, unresolved)
# ============================================================


@dataclass
class TType:
    """Base for all type nodes."""

    pos: Pos


@dataclass
class TPrimitive(TType):
    """int, float, bool, byte, bytes, string, rune, void, obj, nil."""

    kind: str


@dataclass
class TListType(TType):
    """list[T]."""

    element: TType


@dataclass
class TMapType(TType):
    """map[K, V]."""

    key: TType
    value: TType


@dataclass
class TSetType(TType):
    """set[T]."""

    element: TType


@dataclass
class TTupleType(TType):
    """(T, U, ...) — 2+ elements."""

    elements: list[TType]


@dataclass
class TFuncType(TType):
    """fn[T..., R] — last element is return type."""

    params: list[TType]


@dataclass
class TIdentType(TType):
    """User-defined struct/interface/enum name."""

    name: str


@dataclass
class TUnionType(TType):
    """A | B — 2+ members."""

    members: list[TType]


@dataclass
class TOptionalType(TType):
    """T? — sugar for T | nil."""

    inner: TType


# ============================================================
# DECLARATIONS
# ============================================================


@dataclass
class TDecl:
    """Base for all declarations."""

    pos: Pos


@dataclass
class TParam:
    """Function parameter. typ is None for 'self'."""

    pos: Pos
    name: str
    typ: TType | None


@dataclass
class TFnDecl(TDecl):
    """fn Name(params) -> RetType { body }."""

    name: str
    params: list[TParam]
    ret: TType
    body: list[TStmt]


@dataclass
class TFieldDecl:
    """Struct field: name: Type."""

    pos: Pos
    name: str
    typ: TType


@dataclass
class TStructDecl(TDecl):
    """struct Name : Parent { fields and methods }."""

    name: str
    parent: str | None
    fields: list[TFieldDecl]
    methods: list[TFnDecl]


@dataclass
class TInterfaceDecl(TDecl):
    """interface Name { }."""

    name: str


@dataclass
class TEnumDecl(TDecl):
    """enum Name { Variant1 Variant2 ... }."""

    name: str
    variants: list[str]


@dataclass
class TModule:
    """Top-level module — list of declarations."""

    decls: list[TDecl]


# ============================================================
# STATEMENTS
# ============================================================


@dataclass
class TStmt:
    """Base for all statements."""

    pos: Pos


@dataclass
class TLetStmt(TStmt):
    """let name: Type = expr."""

    name: str
    typ: TType
    value: TExpr | None


@dataclass
class TAssignStmt(TStmt):
    """target = value."""

    target: TExpr
    value: TExpr


@dataclass
class TOpAssignStmt(TStmt):
    """target op= value."""

    target: TExpr
    op: str
    value: TExpr


@dataclass
class TTupleAssignStmt(TStmt):
    """a, b = value."""

    targets: list[TExpr]
    value: TExpr


@dataclass
class TReturnStmt(TStmt):
    """return expr?."""

    value: TExpr | None


@dataclass
class TBreakStmt(TStmt):
    """break."""


@dataclass
class TContinueStmt(TStmt):
    """continue."""


@dataclass
class TThrowStmt(TStmt):
    """throw expr."""

    expr: TExpr


@dataclass
class TExprStmt(TStmt):
    """Bare expression as statement."""

    expr: TExpr


@dataclass
class TIfStmt(TStmt):
    """if cond { ... } else { ... }."""

    cond: TExpr
    then_body: list[TStmt]
    else_body: list[TStmt] | None


@dataclass
class TWhileStmt(TStmt):
    """while cond { ... }."""

    cond: TExpr
    body: list[TStmt]


@dataclass
class TRange:
    """range(args) — 1 to 3 args."""

    pos: Pos
    args: list[TExpr]


@dataclass
class TForStmt(TStmt):
    """for binding in iterable/range { ... }."""

    binding: list[str]
    iterable: TExpr | TRange
    body: list[TStmt]


@dataclass
class TPatternType:
    """case name: TypeName."""

    pos: Pos
    name: str
    type_name: TType


@dataclass
class TPatternEnum:
    """case EnumName.Variant."""

    pos: Pos
    enum_name: str
    variant: str


@dataclass
class TPatternNil:
    """case nil."""

    pos: Pos


@dataclass
class TMatchCase:
    """case Pattern { ... }."""

    pos: Pos
    pattern: TPatternType | TPatternEnum | TPatternNil
    body: list[TStmt]


@dataclass
class TDefault:
    """default (name: obj)? { ... }."""

    pos: Pos
    name: str | None
    body: list[TStmt]


@dataclass
class TMatchStmt(TStmt):
    """match expr { cases default? }."""

    expr: TExpr
    cases: list[TMatchCase]
    default: TDefault | None


@dataclass
class TCatch:
    """catch name: Type1 | Type2 { ... }."""

    pos: Pos
    name: str
    types: list[TType]
    body: list[TStmt]


@dataclass
class TTryStmt(TStmt):
    """try { ... } catch ... finally { ... }."""

    body: list[TStmt]
    catches: list[TCatch]
    finally_body: list[TStmt] | None


# ============================================================
# EXPRESSIONS
# ============================================================


@dataclass
class TExpr:
    """Base for all expressions."""

    pos: Pos


@dataclass
class TIntLit(TExpr):
    """Integer literal."""

    value: int
    raw: str


@dataclass
class TFloatLit(TExpr):
    """Float literal."""

    value: float
    raw: str


@dataclass
class TByteLit(TExpr):
    """Byte literal (0xff)."""

    value: int
    raw: str


@dataclass
class TStringLit(TExpr):
    """String literal with escapes resolved."""

    value: str


@dataclass
class TRuneLit(TExpr):
    """Rune literal with escapes resolved."""

    value: str


@dataclass
class TBytesLit(TExpr):
    """Bytes literal with escapes resolved."""

    value: bytes


@dataclass
class TBoolLit(TExpr):
    """true or false."""

    value: bool


@dataclass
class TNilLit(TExpr):
    """nil."""


@dataclass
class TVar(TExpr):
    """Variable reference."""

    name: str


@dataclass
class TBinaryOp(TExpr):
    """left op right."""

    op: str
    left: TExpr
    right: TExpr


@dataclass
class TUnaryOp(TExpr):
    """op operand."""

    op: str
    operand: TExpr


@dataclass
class TTernary(TExpr):
    """cond ? then_expr : else_expr."""

    cond: TExpr
    then_expr: TExpr
    else_expr: TExpr


@dataclass
class TFieldAccess(TExpr):
    """obj.field."""

    obj: TExpr
    field: str


@dataclass
class TTupleAccess(TExpr):
    """obj.0, obj.1."""

    obj: TExpr
    index: int


@dataclass
class TIndex(TExpr):
    """obj[index]."""

    obj: TExpr
    index: TExpr


@dataclass
class TSlice(TExpr):
    """obj[low:high]."""

    obj: TExpr
    low: TExpr
    high: TExpr


@dataclass
class TCall(TExpr):
    """func(args)."""

    func: TExpr
    args: list[TArg]


@dataclass
class TArg:
    """Call argument. name is None for positional."""

    pos: Pos
    name: str | None
    value: TExpr


@dataclass
class TListLit(TExpr):
    """[elements]."""

    elements: list[TExpr]


@dataclass
class TMapLit(TExpr):
    """{ k: v, ... }."""

    entries: list[tuple[TExpr, TExpr]]


@dataclass
class TSetLit(TExpr):
    """{ elements }."""

    elements: list[TExpr]


@dataclass
class TTupleLit(TExpr):
    """(a, b, ...) — 2+ elements."""

    elements: list[TExpr]


@dataclass
class TFnLit(TExpr):
    """(params) -> RetType { body } or (params) -> RetType => expr."""

    params: list[TParam]
    ret: TType
    body: list[TStmt] | TExpr
