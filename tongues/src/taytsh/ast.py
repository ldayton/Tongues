"""Taytsh AST — parse-time node definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..frontend.types import JsonValue, JStr, JInt, JFloat, JBool, JNull, JList, JDict


# ============================================================
# Annotation type alias (not a runtime construct, just for brevity)
# ============================================================

Ann = dict[str, str]


# ============================================================
# POSITION
# ============================================================


@dataclass
class Pos:
    """Source position, 1-indexed."""

    line: int
    col: int
    source_file: str = ""


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
# MODULE ITEM BASE
# ============================================================


@dataclass
class TModuleItem:
    """Base for top-level module items (declarations and statements)."""

    pos: Pos


# ============================================================
# DECLARATIONS
# ============================================================


@dataclass
class TDecl(TModuleItem):
    """Base for all declarations."""


@dataclass
class TParam:
    """Function parameter. typ is None for 'self'."""

    pos: Pos
    name: str
    typ: TType | None
    annotations: Ann
    has_default: bool = False


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
    has_default: bool = False


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

    decls: list[TModuleItem]
    strict_math: bool = False
    strict_tostring: bool = False


# ============================================================
# STATEMENTS
# ============================================================


@dataclass
class TStmt(TModuleItem):
    """Base for all statements."""


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
class TForStmt(TStmt):
    """for binding in iterable/range { ... }."""

    binding: list[str]
    iterable: TExpr
    body: list[TStmt]
    annotations: Ann


@dataclass
class TPattern:
    """Base for all match patterns."""

    pos: Pos


@dataclass
class TPatternType(TPattern):
    """case name: TypeName."""

    name: str
    type_name: TType
    annotations: Ann


@dataclass
class TPatternEnum(TPattern):
    """case EnumName.Variant."""

    enum_name: str
    variant: str


@dataclass
class TPatternNil(TPattern):
    """case nil."""


@dataclass
class TMatchCase:
    """case Pattern { ... }."""

    pos: Pos
    pattern: TPattern
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
class TRange(TExpr):
    """range(args) — 1 to 3 args."""

    args: list[TExpr]
    annotations: Ann


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
    """(params) -> RetType { body }."""

    params: list[TParam]
    ret: TType
    body: list[TStmt]
    annotations: Ann


# ============================================================
# GENERIC ANNOTATION SERIALIZER
# ============================================================


def _sa_strip(ann: Ann, pfx: str, plen: int) -> Ann:
    result: Ann = {}
    for k, v in ann.items():
        if k.startswith(pfx):
            result[k[plen:]] = v
    return result


def _sa_collect_lets(
    stmts: list[TStmt], lets: dict[str, Ann], pfx: str, plen: int
) -> None:
    for stmt in stmts:
        if isinstance(stmt, TLetStmt):
            a = _sa_strip(stmt.annotations, pfx, plen)
            if a:
                lets[stmt.name] = a
        if isinstance(stmt, TIfStmt):
            _sa_collect_lets(stmt.then_body, lets, pfx, plen)
            if stmt.else_body is not None:
                _sa_collect_lets(stmt.else_body, lets, pfx, plen)
        elif isinstance(stmt, TWhileStmt):
            _sa_collect_lets(stmt.body, lets, pfx, plen)
        elif isinstance(stmt, TForStmt):
            _sa_collect_lets(stmt.body, lets, pfx, plen)
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                _sa_collect_lets(case.body, lets, pfx, plen)
            if stmt.default is not None:
                _sa_collect_lets(stmt.default.body, lets, pfx, plen)
        elif isinstance(stmt, TTryStmt):
            _sa_collect_lets(stmt.body, lets, pfx, plen)
            for catch in stmt.catches:
                _sa_collect_lets(catch.body, lets, pfx, plen)
            if stmt.finally_body is not None:
                _sa_collect_lets(stmt.finally_body, lets, pfx, plen)


def _sa_collect_vars_expr(
    expr: TExpr, result: dict[str, Ann], pfx: str, plen: int
) -> None:
    if isinstance(expr, TVar):
        a = _sa_strip(expr.annotations, pfx, plen)
        if a:
            result.setdefault(expr.name, {}).update(a)
    elif isinstance(expr, TBinaryOp):
        _sa_collect_vars_expr(expr.left, result, pfx, plen)
        _sa_collect_vars_expr(expr.right, result, pfx, plen)
    elif isinstance(expr, TUnaryOp):
        _sa_collect_vars_expr(expr.operand, result, pfx, plen)
    elif isinstance(expr, TCall):
        _sa_collect_vars_expr(expr.func, result, pfx, plen)
        for arg in expr.args:
            _sa_collect_vars_expr(arg.value, result, pfx, plen)
    elif isinstance(expr, TFieldAccess):
        _sa_collect_vars_expr(expr.obj, result, pfx, plen)
    elif isinstance(expr, TTupleAccess):
        _sa_collect_vars_expr(expr.obj, result, pfx, plen)
    elif isinstance(expr, TIndex):
        _sa_collect_vars_expr(expr.obj, result, pfx, plen)
        _sa_collect_vars_expr(expr.index, result, pfx, plen)
    elif isinstance(expr, TTernary):
        _sa_collect_vars_expr(expr.cond, result, pfx, plen)
        _sa_collect_vars_expr(expr.then_expr, result, pfx, plen)
        _sa_collect_vars_expr(expr.else_expr, result, pfx, plen)
    elif isinstance(expr, TSlice):
        _sa_collect_vars_expr(expr.obj, result, pfx, plen)
        _sa_collect_vars_expr(expr.low, result, pfx, plen)
        _sa_collect_vars_expr(expr.high, result, pfx, plen)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _sa_collect_vars_expr(e, result, pfx, plen)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _sa_collect_vars_expr(k, result, pfx, plen)
            _sa_collect_vars_expr(v, result, pfx, plen)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _sa_collect_vars_expr(e, result, pfx, plen)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _sa_collect_vars_expr(e, result, pfx, plen)
    elif isinstance(expr, TFnLit):
        _sa_collect_vars_stmts(expr.body, result, pfx, plen)


def _sa_collect_vars_stmt(
    stmt: TStmt, result: dict[str, Ann], pfx: str, plen: int
) -> None:
    if isinstance(stmt, TExprStmt):
        _sa_collect_vars_expr(stmt.expr, result, pfx, plen)
    elif isinstance(stmt, TReturnStmt) and stmt.value is not None:
        _sa_collect_vars_expr(stmt.value, result, pfx, plen)
    elif isinstance(stmt, TThrowStmt):
        _sa_collect_vars_expr(stmt.expr, result, pfx, plen)
    elif isinstance(stmt, TLetStmt) and stmt.value is not None:
        _sa_collect_vars_expr(stmt.value, result, pfx, plen)
    elif isinstance(stmt, TAssignStmt):
        _sa_collect_vars_expr(stmt.target, result, pfx, plen)
        _sa_collect_vars_expr(stmt.value, result, pfx, plen)
    elif isinstance(stmt, TOpAssignStmt):
        _sa_collect_vars_expr(stmt.target, result, pfx, plen)
        _sa_collect_vars_expr(stmt.value, result, pfx, plen)
    elif isinstance(stmt, TTupleAssignStmt):
        for t in stmt.targets:
            _sa_collect_vars_expr(t, result, pfx, plen)
        _sa_collect_vars_expr(stmt.value, result, pfx, plen)
    elif isinstance(stmt, TIfStmt):
        _sa_collect_vars_expr(stmt.cond, result, pfx, plen)
        _sa_collect_vars_stmts(stmt.then_body, result, pfx, plen)
        if stmt.else_body is not None:
            _sa_collect_vars_stmts(stmt.else_body, result, pfx, plen)
    elif isinstance(stmt, TWhileStmt):
        _sa_collect_vars_expr(stmt.cond, result, pfx, plen)
        _sa_collect_vars_stmts(stmt.body, result, pfx, plen)
    elif isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                _sa_collect_vars_expr(a, result, pfx, plen)
        else:
            _sa_collect_vars_expr(stmt.iterable, result, pfx, plen)
        _sa_collect_vars_stmts(stmt.body, result, pfx, plen)
    elif isinstance(stmt, TMatchStmt):
        _sa_collect_vars_expr(stmt.expr, result, pfx, plen)
        for case in stmt.cases:
            _sa_collect_vars_stmts(case.body, result, pfx, plen)
        if stmt.default is not None:
            _sa_collect_vars_stmts(stmt.default.body, result, pfx, plen)
    elif isinstance(stmt, TTryStmt):
        _sa_collect_vars_stmts(stmt.body, result, pfx, plen)
        for catch in stmt.catches:
            _sa_collect_vars_stmts(catch.body, result, pfx, plen)
        if stmt.finally_body is not None:
            _sa_collect_vars_stmts(stmt.finally_body, result, pfx, plen)


def _sa_collect_vars_stmts(
    stmts: list[TStmt], result: dict[str, Ann], pfx: str, plen: int
) -> None:
    for stmt in stmts:
        _sa_collect_vars_stmt(stmt, result, pfx, plen)


def _sa_stmt_type_name(stmt: TStmt) -> str:
    if isinstance(stmt, TLetStmt):
        return "TLetStmt"
    if isinstance(stmt, TAssignStmt):
        return "TAssignStmt"
    if isinstance(stmt, TOpAssignStmt):
        return "TOpAssignStmt"
    if isinstance(stmt, TTupleAssignStmt):
        return "TTupleAssignStmt"
    if isinstance(stmt, TReturnStmt):
        return "TReturnStmt"
    if isinstance(stmt, TBreakStmt):
        return "TBreakStmt"
    if isinstance(stmt, TContinueStmt):
        return "TContinueStmt"
    if isinstance(stmt, TThrowStmt):
        return "TThrowStmt"
    if isinstance(stmt, TExprStmt):
        return "TExprStmt"
    if isinstance(stmt, TIfStmt):
        return "TIfStmt"
    if isinstance(stmt, TWhileStmt):
        return "TWhileStmt"
    if isinstance(stmt, TForStmt):
        return "TForStmt"
    if isinstance(stmt, TMatchStmt):
        return "TMatchStmt"
    if isinstance(stmt, TTryStmt):
        return "TTryStmt"
    return "TStmt"


def _wrap_value(v: str) -> JsonValue:
    """Wrap an annotation string value as a JsonValue."""
    return JStr(v)


def _wrap_ann(ann: Ann) -> JsonValue:
    """Wrap an Ann dict as a JDict."""
    d: dict[str, JsonValue] = {}
    keys = list(ann.keys())
    i = 0
    while i < len(keys):
        d[keys[i]] = JStr(ann[keys[i]])
        i += 1
    return JDict(d)


def _wrap_raw(obj: JsonValue) -> JsonValue:
    """Recursively wrap a raw Python value tree (from asdict) as JsonValue."""
    if obj is None:
        return JNull()
    if isinstance(obj, bool):
        return JBool(obj)
    if isinstance(obj, int):
        return JInt(obj)
    if isinstance(obj, float):
        return JFloat(obj)
    if isinstance(obj, str):
        return JStr(obj)
    if isinstance(obj, bytes):
        items: list[JsonValue] = []
        i = 0
        while i < len(obj):
            items.append(JInt(obj[i]))
            i += 1
        return JList(items)
    if isinstance(obj, tuple):
        items: list[JsonValue] = []
        i = 0
        while i < len(obj):
            items.append(_wrap_raw(obj[i]))
            i += 1
        return JList(items)
    if isinstance(obj, list):
        items: list[JsonValue] = []
        i = 0
        while i < len(obj):
            items.append(_wrap_raw(obj[i]))
            i += 1
        return JList(items)
    if isinstance(obj, dict):
        d: dict[str, JsonValue] = {}
        keys = list(obj.keys())
        i = 0
        while i < len(keys):
            d[keys[i]] = _wrap_raw(obj[keys[i]])
            i += 1
        return JDict(d)
    return JNull()


def to_dict(module: TModule) -> JsonValue:
    """Serialize a TModule to a JsonValue tree via dataclasses.asdict."""
    return _wrap_raw(asdict(module))


def _sa_serialize_stmt(stmt: TStmt, pfx: str, plen: int) -> dict[str, JsonValue]:
    d: dict[str, JsonValue] = {"type": JStr(_sa_stmt_type_name(stmt))}
    ann = _sa_strip(stmt.annotations, pfx, plen)
    binder: dict[str, dict[str, JsonValue]] = {}
    for k, v in ann.items():
        if k.startswith("binder."):
            rest = k[7:]
            dot = rest.find(".")
            if dot != -1:
                binder.setdefault(rest[:dot], {})[rest[dot + 1 :]] = _wrap_value(v)
            else:
                d[k] = _wrap_value(v)
        else:
            d[k] = _wrap_value(v)
    if binder:
        bd: dict[str, JsonValue] = {}
        for bk, bv in binder.items():
            bd[bk] = JDict(bv)
        d["binder"] = JDict(bd)
    if isinstance(stmt, TMatchStmt):
        cases: list[JsonValue] = []
        for case in stmt.cases:
            cd: Ann = _sa_strip(case.annotations, pfx, plen)
            if isinstance(case.pattern, TPatternType):
                pat = _sa_strip(case.pattern.annotations, pfx, plen)
                cd.update(pat)
                for ka, va in pat.items():
                    cd["pattern." + ka] = va
            cases.append(_wrap_ann(cd))
        d["cases"] = JList(cases)
        if stmt.default is not None:
            d["default"] = _wrap_ann(_sa_strip(stmt.default.annotations, pfx, plen))
    elif isinstance(stmt, TTryStmt):
        catches: list[JsonValue] = [
            _wrap_ann(_sa_strip(c.annotations, pfx, plen)) for c in stmt.catches
        ]
        d["catches"] = JList(catches)
    return d


def _sa_serialize_fn(fn: TFnDecl, pfx: str, plen: int) -> dict[str, JsonValue]:
    raw_ann = _sa_strip(fn.annotations, pfx, plen)
    d: dict[str, JsonValue] = {}
    for k, v in raw_ann.items():
        d[k] = _wrap_value(v)
    params: dict[str, Ann] = {}
    for p in fn.params:
        a = _sa_strip(p.annotations, pfx, plen)
        if a:
            params[p.name] = a
    if params:
        pd: dict[str, JsonValue] = {}
        for pk, pv in params.items():
            pd[pk] = _wrap_ann(pv)
        d["params"] = JDict(pd)
    lets: dict[str, Ann] = {}
    _sa_collect_lets(fn.body, lets, pfx, plen)
    if lets:
        ld: dict[str, JsonValue] = {}
        for lk, lv in lets.items():
            ld[lk] = _wrap_ann(lv)
        d["lets"] = JDict(ld)
    body_items: list[JsonValue] = [
        JDict(_sa_serialize_stmt(s, pfx, plen)) for s in fn.body
    ]
    d["body"] = JList(body_items)
    vars_dict: dict[str, Ann] = {}
    _sa_collect_vars_stmts(fn.body, vars_dict, pfx, plen)
    if vars_dict:
        vd: dict[str, JsonValue] = {}
        for vk, vv in vars_dict.items():
            vd[vk] = _wrap_ann(vv)
        d["vars"] = JDict(vd)
        escapes: dict[str, JsonValue] = {}
        for n, va in vars_dict.items():
            if va.get("escapes") is not None:
                escapes[n] = JBool(True)
        if escapes:
            d["escapes"] = JDict(escapes)
    return d


def serialize_annotations(module: TModule, prefix: str) -> dict[str, JsonValue]:
    """Serialize all annotations matching prefix from every function into nested dicts."""
    pfx = prefix + "."
    plen = len(pfx)
    result: dict[str, JsonValue] = {}
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            result[decl.name] = JDict(_sa_serialize_fn(decl, pfx, plen))
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                result[f"{decl.name}.{method.name}"] = JDict(
                    _sa_serialize_fn(method, pfx, plen)
                )
    return result
