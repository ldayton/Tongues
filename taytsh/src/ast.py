"""Taytsh AST — parse-time node definitions."""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# Annotation type alias (not a runtime construct, just for brevity)
# ============================================================

Ann = dict[str, bool | int | str | tuple[int, int]]


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
    """int, float, bool, byte, bytes, string, rune, void, nil."""

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
    annotations: Ann


@dataclass
class TFnDecl(TDecl):
    """fn Name(params) -> RetType { body }."""

    name: str
    params: list[TParam]
    ret: TType
    body: list[TStmt]
    annotations: Ann


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
    annotations: Ann


@dataclass
class TInterfaceDecl(TDecl):
    """interface Name { }."""

    name: str
    annotations: Ann


@dataclass
class TEnumDecl(TDecl):
    """enum Name { Variant1 Variant2 ... }."""

    name: str
    variants: list[str]
    annotations: Ann


@dataclass
class TModule:
    """Top-level module — list of declarations."""

    decls: list[TDecl]
    strict_math: bool = False
    strict_tostring: bool = False


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
    annotations: Ann


@dataclass
class TAssignStmt(TStmt):
    """target = value."""

    target: TExpr
    value: TExpr
    annotations: Ann


@dataclass
class TOpAssignStmt(TStmt):
    """target op= value."""

    target: TExpr
    op: str
    value: TExpr
    annotations: Ann


@dataclass
class TTupleAssignStmt(TStmt):
    """a, b = value."""

    targets: list[TExpr]
    value: TExpr
    annotations: Ann


@dataclass
class TReturnStmt(TStmt):
    """return expr?."""

    value: TExpr | None
    annotations: Ann


@dataclass
class TBreakStmt(TStmt):
    """break."""

    annotations: Ann


@dataclass
class TContinueStmt(TStmt):
    """continue."""

    annotations: Ann


@dataclass
class TThrowStmt(TStmt):
    """throw expr."""

    expr: TExpr
    annotations: Ann


@dataclass
class TExprStmt(TStmt):
    """Bare expression as statement."""

    expr: TExpr
    annotations: Ann


@dataclass
class TIfStmt(TStmt):
    """if cond { ... } else { ... }."""

    cond: TExpr
    then_body: list[TStmt]
    else_body: list[TStmt] | None
    annotations: Ann


@dataclass
class TWhileStmt(TStmt):
    """while cond { ... }."""

    cond: TExpr
    body: list[TStmt]
    annotations: Ann


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
    annotations: Ann


@dataclass
class TPatternType:
    """case name: TypeName."""

    pos: Pos
    name: str
    type_name: TType
    annotations: Ann


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
    annotations: Ann


@dataclass
class TDefault:
    """default name? { ... }."""

    pos: Pos
    name: str | None
    body: list[TStmt]
    annotations: Ann


@dataclass
class TMatchStmt(TStmt):
    """match expr { cases default? }."""

    expr: TExpr
    cases: list[TMatchCase]
    default: TDefault | None
    annotations: Ann


@dataclass
class TCatch:
    """catch name: Type1 | Type2 { ... }."""

    pos: Pos
    name: str
    types: list[TType]
    body: list[TStmt]
    annotations: Ann


@dataclass
class TTryStmt(TStmt):
    """try { ... } catch ... finally { ... }."""

    body: list[TStmt]
    catches: list[TCatch]
    finally_body: list[TStmt] | None
    annotations: Ann


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
    annotations: Ann


@dataclass
class TFloatLit(TExpr):
    """Float literal."""

    value: float
    raw: str
    annotations: Ann


@dataclass
class TByteLit(TExpr):
    """Byte literal (0xff)."""

    value: int
    raw: str
    annotations: Ann


@dataclass
class TStringLit(TExpr):
    """String literal with escapes resolved."""

    value: str
    annotations: Ann


@dataclass
class TRuneLit(TExpr):
    """Rune literal with escapes resolved."""

    value: str
    annotations: Ann


@dataclass
class TBytesLit(TExpr):
    """Bytes literal with escapes resolved."""

    value: bytes
    annotations: Ann


@dataclass
class TBoolLit(TExpr):
    """true or false."""

    value: bool
    annotations: Ann


@dataclass
class TNilLit(TExpr):
    """nil."""

    annotations: Ann


@dataclass
class TVar(TExpr):
    """Variable reference."""

    name: str
    annotations: Ann


@dataclass
class TBinaryOp(TExpr):
    """left op right."""

    op: str
    left: TExpr
    right: TExpr
    annotations: Ann


@dataclass
class TUnaryOp(TExpr):
    """op operand."""

    op: str
    operand: TExpr
    annotations: Ann


@dataclass
class TTernary(TExpr):
    """cond ? then_expr : else_expr."""

    cond: TExpr
    then_expr: TExpr
    else_expr: TExpr
    annotations: Ann


@dataclass
class TFieldAccess(TExpr):
    """obj.field."""

    obj: TExpr
    field: str
    annotations: Ann


@dataclass
class TTupleAccess(TExpr):
    """obj.0, obj.1."""

    obj: TExpr
    index: int
    annotations: Ann


@dataclass
class TIndex(TExpr):
    """obj[index]."""

    obj: TExpr
    index: TExpr
    annotations: Ann


@dataclass
class TSlice(TExpr):
    """obj[low:high]."""

    obj: TExpr
    low: TExpr
    high: TExpr
    annotations: Ann


@dataclass
class TCall(TExpr):
    """func(args)."""

    func: TExpr
    args: list[TArg]
    annotations: Ann


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
    annotations: Ann


@dataclass
class TMapLit(TExpr):
    """{ k: v, ... }."""

    entries: list[tuple[TExpr, TExpr]]
    annotations: Ann


@dataclass
class TSetLit(TExpr):
    """{ elements }."""

    elements: list[TExpr]
    annotations: Ann


@dataclass
class TTupleLit(TExpr):
    """(a, b, ...) — 2+ elements."""

    elements: list[TExpr]
    annotations: Ann


@dataclass
class TFnLit(TExpr):
    """(params) -> RetType { body } or (params) -> RetType => expr."""

    params: list[TParam]
    ret: TType
    body: list[TStmt] | TExpr
    annotations: Ann


# ============================================================
# GENERIC ANNOTATION SERIALIZER
# ============================================================


def serialize_annotations(module: TModule, prefix: str) -> dict[str, dict]:
    """Serialize all annotations matching prefix from every function into nested dicts.

    Returns {fn_name: fn_dict} where fn_dict contains params, lets, body, vars,
    escapes, and any function-level annotations — all with the prefix stripped.
    """
    pfx = prefix + "."
    plen = len(pfx)

    def _strip(ann: Ann) -> dict:
        return {k[plen:]: v for k, v in ann.items() if k.startswith(pfx)}

    def _collect_lets(stmts: list[TStmt], lets: dict) -> None:
        for stmt in stmts:
            if isinstance(stmt, TLetStmt):
                a = _strip(stmt.annotations)
                if a:
                    lets[stmt.name] = a
            if isinstance(stmt, TIfStmt):
                _collect_lets(stmt.then_body, lets)
                if stmt.else_body is not None:
                    _collect_lets(stmt.else_body, lets)
            elif isinstance(stmt, TWhileStmt):
                _collect_lets(stmt.body, lets)
            elif isinstance(stmt, TForStmt):
                _collect_lets(stmt.body, lets)
            elif isinstance(stmt, TMatchStmt):
                for case in stmt.cases:
                    _collect_lets(case.body, lets)
                if stmt.default is not None:
                    _collect_lets(stmt.default.body, lets)
            elif isinstance(stmt, TTryStmt):
                _collect_lets(stmt.body, lets)
                for catch in stmt.catches:
                    _collect_lets(catch.body, lets)
                if stmt.finally_body is not None:
                    _collect_lets(stmt.finally_body, lets)

    def _collect_vars_expr(expr: TExpr, result: dict) -> None:
        if isinstance(expr, TVar):
            a = _strip(expr.annotations)
            if a:
                result.setdefault(expr.name, {}).update(a)
        elif isinstance(expr, TBinaryOp):
            _collect_vars_expr(expr.left, result)
            _collect_vars_expr(expr.right, result)
        elif isinstance(expr, TUnaryOp):
            _collect_vars_expr(expr.operand, result)
        elif isinstance(expr, TCall):
            _collect_vars_expr(expr.func, result)
            for a in expr.args:
                _collect_vars_expr(a.value, result)
        elif isinstance(expr, TFieldAccess):
            _collect_vars_expr(expr.obj, result)
        elif isinstance(expr, TTupleAccess):
            _collect_vars_expr(expr.obj, result)
        elif isinstance(expr, TIndex):
            _collect_vars_expr(expr.obj, result)
            _collect_vars_expr(expr.index, result)
        elif isinstance(expr, TTernary):
            _collect_vars_expr(expr.cond, result)
            _collect_vars_expr(expr.then_expr, result)
            _collect_vars_expr(expr.else_expr, result)
        elif isinstance(expr, TSlice):
            _collect_vars_expr(expr.obj, result)
            _collect_vars_expr(expr.low, result)
            _collect_vars_expr(expr.high, result)
        elif isinstance(expr, TListLit):
            for e in expr.elements:
                _collect_vars_expr(e, result)
        elif isinstance(expr, TMapLit):
            for k, v in expr.entries:
                _collect_vars_expr(k, result)
                _collect_vars_expr(v, result)
        elif isinstance(expr, TSetLit):
            for e in expr.elements:
                _collect_vars_expr(e, result)
        elif isinstance(expr, TTupleLit):
            for e in expr.elements:
                _collect_vars_expr(e, result)
        elif isinstance(expr, TFnLit):
            if isinstance(expr.body, list):
                _collect_vars_stmts(expr.body, result)
            else:
                _collect_vars_expr(expr.body, result)

    def _collect_vars_stmt(stmt: TStmt, result: dict) -> None:
        if isinstance(stmt, TExprStmt):
            _collect_vars_expr(stmt.expr, result)
        elif isinstance(stmt, TReturnStmt) and stmt.value is not None:
            _collect_vars_expr(stmt.value, result)
        elif isinstance(stmt, TThrowStmt):
            _collect_vars_expr(stmt.expr, result)
        elif isinstance(stmt, TLetStmt) and stmt.value is not None:
            _collect_vars_expr(stmt.value, result)
        elif isinstance(stmt, TAssignStmt):
            _collect_vars_expr(stmt.target, result)
            _collect_vars_expr(stmt.value, result)
        elif isinstance(stmt, TOpAssignStmt):
            _collect_vars_expr(stmt.target, result)
            _collect_vars_expr(stmt.value, result)
        elif isinstance(stmt, TTupleAssignStmt):
            for t in stmt.targets:
                _collect_vars_expr(t, result)
            _collect_vars_expr(stmt.value, result)
        elif isinstance(stmt, TIfStmt):
            _collect_vars_expr(stmt.cond, result)
            _collect_vars_stmts(stmt.then_body, result)
            if stmt.else_body is not None:
                _collect_vars_stmts(stmt.else_body, result)
        elif isinstance(stmt, TWhileStmt):
            _collect_vars_expr(stmt.cond, result)
            _collect_vars_stmts(stmt.body, result)
        elif isinstance(stmt, TForStmt):
            if isinstance(stmt.iterable, TRange):
                for a in stmt.iterable.args:
                    _collect_vars_expr(a, result)
            else:
                _collect_vars_expr(stmt.iterable, result)
            _collect_vars_stmts(stmt.body, result)
        elif isinstance(stmt, TMatchStmt):
            _collect_vars_expr(stmt.expr, result)
            for case in stmt.cases:
                _collect_vars_stmts(case.body, result)
            if stmt.default is not None:
                _collect_vars_stmts(stmt.default.body, result)
        elif isinstance(stmt, TTryStmt):
            _collect_vars_stmts(stmt.body, result)
            for catch in stmt.catches:
                _collect_vars_stmts(catch.body, result)
            if stmt.finally_body is not None:
                _collect_vars_stmts(stmt.finally_body, result)

    def _collect_vars_stmts(stmts: list[TStmt], result: dict) -> None:
        for stmt in stmts:
            _collect_vars_stmt(stmt, result)

    def _serialize_stmt(stmt: TStmt) -> dict:
        d: dict = {"type": type(stmt).__name__}
        ann = _strip(stmt.annotations)
        binder: dict = {}
        for k, v in ann.items():
            if k.startswith("binder."):
                rest = k[7:]
                dot = rest.find(".")
                if dot != -1:
                    binder.setdefault(rest[:dot], {})[rest[dot + 1 :]] = v
                else:
                    d[k] = v
            else:
                d[k] = v
        if binder:
            d["binder"] = binder
        if isinstance(stmt, TMatchStmt):
            cases = []
            for case in stmt.cases:
                cd: dict = _strip(case.annotations)
                if isinstance(case.pattern, TPatternType):
                    pat = _strip(case.pattern.annotations)
                    cd.update(pat)
                    for ka, va in pat.items():
                        cd["pattern." + ka] = va
                cases.append(cd)
            d["cases"] = cases
            if stmt.default is not None:
                d["default"] = _strip(stmt.default.annotations)
        elif isinstance(stmt, TTryStmt):
            d["catches"] = [_strip(c.annotations) for c in stmt.catches]
        return d

    def _serialize_fn(fn: TFnDecl) -> dict:
        d: dict = _strip(fn.annotations)
        params: dict = {}
        for p in fn.params:
            a = _strip(p.annotations)
            if a:
                params[p.name] = a
        if params:
            d["params"] = params
        lets: dict = {}
        _collect_lets(fn.body, lets)
        if lets:
            d["lets"] = lets
        d["body"] = [_serialize_stmt(s) for s in fn.body]
        vars_dict: dict = {}
        _collect_vars_stmts(fn.body, vars_dict)
        if vars_dict:
            d["vars"] = vars_dict
            escapes = {n: True for n, a in vars_dict.items() if a.get("escapes")}
            if escapes:
                d["escapes"] = escapes
        return d

    result: dict = {}
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            result[decl.name] = _serialize_fn(decl)
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                result[f"{decl.name}.{method.name}"] = _serialize_fn(method)
    return result
