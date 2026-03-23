"""Taytsh AST — parse-time node definitions."""

from __future__ import annotations
from typing import assert_never

from dataclasses import dataclass

from ..frontend.types import JsonValue, JStr, JInt, JFloat, JBool, JNull, JList, JDict


# ============================================================
# Annotation type alias (not a runtime construct, just for brevity)
# ============================================================

type Ann = dict[str, str]


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
    annotations: Ann


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


@dataclass
class TFieldDecl:
    """Struct field: name: Type."""

    pos: Pos
    name: str
    typ: TType
    has_default: bool = False
    self_ref: bool = False
    default_expr: "TExpr | None" = None


@dataclass
class TStructDecl(TDecl):
    """struct Name : Parent { fields and methods }."""

    name: str
    parent: str | None
    fields: list[TFieldDecl]
    methods: list[TFnDecl]


@dataclass
class TInterfaceDecl(TDecl):
    """interface Name { fields }."""

    name: str
    fields: list[TFieldDecl]


@dataclass
class TEnumDecl(TDecl):
    """enum Name { Variant1 Variant2 ... }."""

    name: str
    variants: list[str]


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

    pass


@dataclass
class TContinueStmt(TStmt):
    """continue."""

    pass


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
class TForStmt(TStmt):
    """for binding in iterable/range { ... }."""

    binding: list[str]
    iterable: TExpr
    body: list[TStmt]


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


# ============================================================
# EXPRESSIONS
# ============================================================


@dataclass
class TExpr:
    """Base for all expressions."""

    pos: Pos
    annotations: Ann


@dataclass
class TRange(TExpr):
    """range(args) — 1 to 3 args."""

    args: list[TExpr]


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

    pass


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
    """(params) -> RetType { body }."""

    params: list[TParam]
    ret: TType
    body: list[TStmt]


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
        match stmt:
            case TIfStmt():
                _sa_collect_lets(stmt.then_body, lets, pfx, plen)
                if stmt.else_body is not None:
                    _sa_collect_lets(stmt.else_body, lets, pfx, plen)
            case TWhileStmt() | TForStmt():
                _sa_collect_lets(stmt.body, lets, pfx, plen)
            case TMatchStmt():
                for case in stmt.cases:
                    _sa_collect_lets(case.body, lets, pfx, plen)
                if stmt.default is not None:
                    _sa_collect_lets(stmt.default.body, lets, pfx, plen)
            case TTryStmt():
                _sa_collect_lets(stmt.body, lets, pfx, plen)
                for catch in stmt.catches:
                    _sa_collect_lets(catch.body, lets, pfx, plen)
                if stmt.finally_body is not None:
                    _sa_collect_lets(stmt.finally_body, lets, pfx, plen)


def _sa_collect_vars_expr(
    expr: TExpr, result: dict[str, Ann], pfx: str, plen: int
) -> None:
    match expr:
        case TVar():
            a = _sa_strip(expr.annotations, pfx, plen)
            if a:
                if expr.name not in result:
                    result[expr.name] = {}
                for ak in a:
                    result[expr.name][ak] = a[ak]
        case TBinaryOp():
            _sa_collect_vars_expr(expr.left, result, pfx, plen)
            _sa_collect_vars_expr(expr.right, result, pfx, plen)
        case TUnaryOp():
            _sa_collect_vars_expr(expr.operand, result, pfx, plen)
        case TCall():
            _sa_collect_vars_expr(expr.func, result, pfx, plen)
            for arg in expr.args:
                _sa_collect_vars_expr(arg.value, result, pfx, plen)
        case TFieldAccess() | TTupleAccess():
            _sa_collect_vars_expr(expr.obj, result, pfx, plen)
        case TIndex():
            _sa_collect_vars_expr(expr.obj, result, pfx, plen)
            _sa_collect_vars_expr(expr.index, result, pfx, plen)
        case TTernary():
            _sa_collect_vars_expr(expr.cond, result, pfx, plen)
            _sa_collect_vars_expr(expr.then_expr, result, pfx, plen)
            _sa_collect_vars_expr(expr.else_expr, result, pfx, plen)
        case TSlice():
            _sa_collect_vars_expr(expr.obj, result, pfx, plen)
            _sa_collect_vars_expr(expr.low, result, pfx, plen)
            _sa_collect_vars_expr(expr.high, result, pfx, plen)
        case TListLit() | TSetLit() | TTupleLit():
            for e in expr.elements:
                _sa_collect_vars_expr(e, result, pfx, plen)
        case TMapLit():
            for k, v in expr.entries:
                _sa_collect_vars_expr(k, result, pfx, plen)
                _sa_collect_vars_expr(v, result, pfx, plen)
        case TFnLit():
            _sa_collect_vars_stmts(expr.body, result, pfx, plen)


def _sa_collect_vars_stmt(
    stmt: TStmt, result: dict[str, Ann], pfx: str, plen: int
) -> None:
    match stmt:
        case TExprStmt():
            _sa_collect_vars_expr(stmt.expr, result, pfx, plen)
        case TReturnStmt():
            if stmt.value is not None:
                _sa_collect_vars_expr(stmt.value, result, pfx, plen)
        case TThrowStmt():
            _sa_collect_vars_expr(stmt.expr, result, pfx, plen)
        case TLetStmt():
            if stmt.value is not None:
                _sa_collect_vars_expr(stmt.value, result, pfx, plen)
        case TAssignStmt():
            _sa_collect_vars_expr(stmt.target, result, pfx, plen)
            _sa_collect_vars_expr(stmt.value, result, pfx, plen)
        case TOpAssignStmt():
            _sa_collect_vars_expr(stmt.target, result, pfx, plen)
            _sa_collect_vars_expr(stmt.value, result, pfx, plen)
        case TTupleAssignStmt():
            for t in stmt.targets:
                _sa_collect_vars_expr(t, result, pfx, plen)
            _sa_collect_vars_expr(stmt.value, result, pfx, plen)
        case TIfStmt():
            _sa_collect_vars_expr(stmt.cond, result, pfx, plen)
            _sa_collect_vars_stmts(stmt.then_body, result, pfx, plen)
            if stmt.else_body is not None:
                _sa_collect_vars_stmts(stmt.else_body, result, pfx, plen)
        case TWhileStmt():
            _sa_collect_vars_expr(stmt.cond, result, pfx, plen)
            _sa_collect_vars_stmts(stmt.body, result, pfx, plen)
        case TForStmt():
            if isinstance(stmt.iterable, TRange):
                for a in stmt.iterable.args:
                    _sa_collect_vars_expr(a, result, pfx, plen)
            else:
                _sa_collect_vars_expr(stmt.iterable, result, pfx, plen)
            _sa_collect_vars_stmts(stmt.body, result, pfx, plen)
        case TMatchStmt():
            _sa_collect_vars_expr(stmt.expr, result, pfx, plen)
            for case in stmt.cases:
                _sa_collect_vars_stmts(case.body, result, pfx, plen)
            if stmt.default is not None:
                _sa_collect_vars_stmts(stmt.default.body, result, pfx, plen)
        case TTryStmt():
            _sa_collect_vars_stmts(stmt.body, result, pfx, plen)
            for catch in stmt.catches:
                _sa_collect_vars_stmts(catch.body, result, pfx, plen)
            if stmt.finally_body is not None:
                _sa_collect_vars_stmts(stmt.finally_body, result, pfx, plen)
        case TBreakStmt() | TContinueStmt():
            pass
        case _:
            assert_never(stmt)


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
    for k in ann:
        d[k] = JStr(ann[k])
    return JDict(d)


def _pos_json(p: Pos) -> JsonValue:
    return JDict({"line": JInt(p.line), "col": JInt(p.col)})


def _ann_json(a: Ann) -> JsonValue:
    d: dict[str, JsonValue] = {}
    for k in a:
        d[k] = JStr(a[k])
    return JDict(d)


def _types_json(items: list[TType]) -> JsonValue:
    result: list[JsonValue] = []
    for item in items:
        result.append(_type_json(item))
    return JList(result)


def _exprs_json(items: list[TExpr]) -> JsonValue:
    result: list[JsonValue] = []
    for item in items:
        result.append(_expr_json(item))
    return JList(result)


def _stmts_json(items: list[TStmt]) -> JsonValue:
    result: list[JsonValue] = []
    for item in items:
        result.append(_stmt_json(item))
    return JList(result)


def _type_json(t: TType) -> JsonValue:
    d: dict[str, JsonValue] = {"pos": _pos_json(t.pos)}
    match t:
        case TPrimitive():
            d["kind"] = JStr(t.kind)
        case TListType() | TSetType():
            d["element"] = _type_json(t.element)
        case TMapType():
            d["key"] = _type_json(t.key)
            d["value"] = _type_json(t.value)
        case TTupleType():
            d["elements"] = _types_json(t.elements)
        case TFuncType():
            d["params"] = _types_json(t.params)
        case TIdentType():
            d["name"] = JStr(t.name)
        case TUnionType():
            d["members"] = _types_json(t.members)
        case TOptionalType():
            d["inner"] = _type_json(t.inner)
    return JDict(d)


def _param_json(p: TParam) -> JsonValue:
    d: dict[str, JsonValue] = {
        "pos": _pos_json(p.pos),
        "name": JStr(p.name),
        "typ": _type_json(p.typ) if p.typ is not None else JNull(),
        "annotations": _ann_json(p.annotations),
        "has_default": JBool(p.has_default),
    }
    return JDict(d)


def _field_decl_json(f: TFieldDecl) -> JsonValue:
    return JDict(
        {
            "pos": _pos_json(f.pos),
            "name": JStr(f.name),
            "typ": _type_json(f.typ),
            "has_default": JBool(f.has_default),
            "self_ref": JBool(f.self_ref),
        }
    )


def _arg_json(a: TArg) -> JsonValue:
    return JDict(
        {
            "pos": _pos_json(a.pos),
            "name": JStr(a.name) if a.name is not None else JNull(),
            "value": _expr_json(a.value),
        }
    )


def _pattern_json(p: TPattern) -> JsonValue:
    d: dict[str, JsonValue] = {"pos": _pos_json(p.pos)}
    match p:
        case TPatternType():
            d["name"] = JStr(p.name)
            d["type_name"] = _type_json(p.type_name)
            d["annotations"] = _ann_json(p.annotations)
        case TPatternEnum():
            d["enum_name"] = JStr(p.enum_name)
            d["variant"] = JStr(p.variant)
    return JDict(d)


def _match_case_json(c: TMatchCase) -> JsonValue:
    return JDict(
        {
            "pos": _pos_json(c.pos),
            "pattern": _pattern_json(c.pattern),
            "body": _stmts_json(c.body),
            "annotations": _ann_json(c.annotations),
        }
    )


def _default_json(d: TDefault) -> JsonValue:
    return JDict(
        {
            "pos": _pos_json(d.pos),
            "name": JStr(d.name) if d.name is not None else JNull(),
            "body": _stmts_json(d.body),
            "annotations": _ann_json(d.annotations),
        }
    )


def _catch_json(c: TCatch) -> JsonValue:
    return JDict(
        {
            "pos": _pos_json(c.pos),
            "name": JStr(c.name),
            "types": _types_json(c.types),
            "body": _stmts_json(c.body),
            "annotations": _ann_json(c.annotations),
        }
    )


def _expr_json(e: TExpr) -> JsonValue:
    d: dict[str, JsonValue] = {"pos": _pos_json(e.pos)}
    match e:
        case TRange():
            d["args"] = _exprs_json(e.args)
        case TIntLit():
            d["value"] = JInt(e.value)
            d["raw"] = JStr(e.raw)
        case TFloatLit():
            d["value"] = JFloat(e.value)
            d["raw"] = JStr(e.raw)
        case TByteLit():
            d["value"] = JInt(e.value)
            d["raw"] = JStr(e.raw)
        case TStringLit():
            d["value"] = JStr(e.value)
        case TRuneLit():
            d["value"] = JStr(e.value)
        case TBytesLit():
            blist: list[JsonValue] = []
            for b in e.value:
                blist.append(JInt(b))
            d["value"] = JList(blist)
        case TBoolLit():
            d["value"] = JBool(e.value)
        case TNilLit():
            pass
        case TVar():
            d["name"] = JStr(e.name)
        case TBinaryOp():
            d["op"] = JStr(e.op)
            d["left"] = _expr_json(e.left)
            d["right"] = _expr_json(e.right)
        case TUnaryOp():
            d["op"] = JStr(e.op)
            d["operand"] = _expr_json(e.operand)
        case TTernary():
            d["cond"] = _expr_json(e.cond)
            d["then_expr"] = _expr_json(e.then_expr)
            d["else_expr"] = _expr_json(e.else_expr)
        case TFieldAccess():
            d["obj"] = _expr_json(e.obj)
            d["field"] = JStr(e.field)
        case TTupleAccess():
            d["obj"] = _expr_json(e.obj)
            d["index"] = JInt(e.index)
        case TIndex():
            d["obj"] = _expr_json(e.obj)
            d["index"] = _expr_json(e.index)
        case TSlice():
            d["obj"] = _expr_json(e.obj)
            d["low"] = _expr_json(e.low)
            d["high"] = _expr_json(e.high)
        case TCall():
            d["func"] = _expr_json(e.func)
            alist: list[JsonValue] = []
            for arg in e.args:
                alist.append(_arg_json(arg))
            d["args"] = JList(alist)
        case TListLit() | TSetLit() | TTupleLit():
            d["elements"] = _exprs_json(e.elements)
        case TMapLit():
            elist: list[JsonValue] = []
            for ek, ev in e.entries:
                elist.append(JList([_expr_json(ek), _expr_json(ev)]))
            d["entries"] = JList(elist)
        case TFnLit():
            plist: list[JsonValue] = []
            for par in e.params:
                plist.append(_param_json(par))
            d["params"] = JList(plist)
            d["ret"] = _type_json(e.ret)
            d["body"] = _stmts_json(e.body)
        case _:
            assert_never(e)
    d["annotations"] = _ann_json(e.annotations)
    return JDict(d)


def _stmt_json(s: TStmt) -> JsonValue:
    d: dict[str, JsonValue] = {"pos": _pos_json(s.pos)}
    match s:
        case TLetStmt():
            d["name"] = JStr(s.name)
            d["typ"] = _type_json(s.typ)
            d["value"] = _expr_json(s.value) if s.value is not None else JNull()
        case TAssignStmt():
            d["target"] = _expr_json(s.target)
            d["value"] = _expr_json(s.value)
        case TOpAssignStmt():
            d["target"] = _expr_json(s.target)
            d["op"] = JStr(s.op)
            d["value"] = _expr_json(s.value)
        case TTupleAssignStmt():
            d["targets"] = _exprs_json(s.targets)
            d["value"] = _expr_json(s.value)
        case TReturnStmt():
            d["value"] = _expr_json(s.value) if s.value is not None else JNull()
        case TBreakStmt() | TContinueStmt():
            pass
        case TThrowStmt():
            d["expr"] = _expr_json(s.expr)
        case TExprStmt():
            d["expr"] = _expr_json(s.expr)
        case TIfStmt():
            d["cond"] = _expr_json(s.cond)
            d["then_body"] = _stmts_json(s.then_body)
            d["else_body"] = (
                _stmts_json(s.else_body) if s.else_body is not None else JNull()
            )
        case TWhileStmt():
            d["cond"] = _expr_json(s.cond)
            d["body"] = _stmts_json(s.body)
        case TForStmt():
            blist: list[JsonValue] = []
            for bnd in s.binding:
                blist.append(JStr(bnd))
            d["binding"] = JList(blist)
            d["iterable"] = _expr_json(s.iterable)
            d["body"] = _stmts_json(s.body)
        case TMatchStmt():
            d["expr"] = _expr_json(s.expr)
            clist: list[JsonValue] = []
            for mc in s.cases:
                clist.append(_match_case_json(mc))
            d["cases"] = JList(clist)
            d["default"] = (
                _default_json(s.default) if s.default is not None else JNull()
            )
        case TTryStmt():
            d["body"] = _stmts_json(s.body)
            calist: list[JsonValue] = []
            for ct in s.catches:
                calist.append(_catch_json(ct))
            d["catches"] = JList(calist)
            d["finally_body"] = (
                _stmts_json(s.finally_body) if s.finally_body is not None else JNull()
            )
        case _:
            assert_never(s)
    d["annotations"] = _ann_json(s.annotations)
    return JDict(d)


def _decl_json(decl: TModuleItem) -> JsonValue:
    if isinstance(decl, TFnDecl):
        plist: list[JsonValue] = []
        for par in decl.params:
            plist.append(_param_json(par))
        return JDict(
            {
                "pos": _pos_json(decl.pos),
                "name": JStr(decl.name),
                "params": JList(plist),
                "ret": _type_json(decl.ret),
                "body": _stmts_json(decl.body),
                "annotations": _ann_json(decl.annotations),
            }
        )
    if isinstance(decl, TStructDecl):
        flist: list[JsonValue] = []
        for fld in decl.fields:
            flist.append(_field_decl_json(fld))
        mlist: list[JsonValue] = []
        for meth in decl.methods:
            mlist.append(_decl_json(meth))
        return JDict(
            {
                "pos": _pos_json(decl.pos),
                "name": JStr(decl.name),
                "parent": JStr(decl.parent) if decl.parent is not None else JNull(),
                "fields": JList(flist),
                "methods": JList(mlist),
                "annotations": _ann_json(decl.annotations),
            }
        )
    if isinstance(decl, TInterfaceDecl):
        flist: list[JsonValue] = []
        for fld in decl.fields:
            flist.append(_field_decl_json(fld))
        return JDict(
            {
                "pos": _pos_json(decl.pos),
                "name": JStr(decl.name),
                "fields": JList(flist),
                "annotations": _ann_json(decl.annotations),
            }
        )
    if isinstance(decl, TEnumDecl):
        vlist: list[JsonValue] = []
        for vnt in decl.variants:
            vlist.append(JStr(vnt))
        return JDict(
            {
                "pos": _pos_json(decl.pos),
                "name": JStr(decl.name),
                "variants": JList(vlist),
                "annotations": _ann_json(decl.annotations),
            }
        )
    if isinstance(decl, TLetStmt):
        return _stmt_json(decl)
    return JNull()


def to_dict(module: TModule) -> JsonValue:
    """Serialize a TModule to a JsonValue tree."""
    dlist: list[JsonValue] = []
    for decl in module.decls:
        dlist.append(_decl_json(decl))
    d: dict[str, JsonValue] = {
        "decls": JList(dlist),
        "strict_math": JBool(module.strict_math),
        "strict_tostring": JBool(module.strict_tostring),
    }
    return JDict(d)


def _sa_serialize_stmt(stmt: TStmt, pfx: str, plen: int) -> dict[str, JsonValue]:
    d: dict[str, JsonValue] = {"type": JStr(_sa_stmt_type_name(stmt))}
    ann = _sa_strip(stmt.annotations, pfx, plen)
    binder: dict[str, dict[str, JsonValue]] = {}
    for k, v in ann.items():
        if k.startswith("binder."):
            rest = k[7:]
            dot = rest.find(".")
            if dot != -1:
                bkey = rest[:dot]
                if bkey not in binder:
                    binder[bkey] = {}
                binder[bkey][rest[dot + 1 :]] = _wrap_value(v)
            else:
                d[k] = _wrap_value(v)
        else:
            d[k] = _wrap_value(v)
    if binder:
        bd: dict[str, JsonValue] = {}
        for bk, bv in binder.items():
            bd[bk] = JDict(bv)
        d["binder"] = JDict(bd)
    match stmt:
        case TMatchStmt():
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
        case TTryStmt():
            catches: list[JsonValue] = [
                _wrap_ann(_sa_strip(c.annotations, pfx, plen)) for c in stmt.catches
            ]
            d["catches"] = JList(catches)
        case TWhileStmt():
            body_items: list[JsonValue] = []
            for s in stmt.body:
                body_items.append(JDict(_sa_serialize_stmt(s, pfx, plen)))
            d["body"] = JList(body_items)
        case TForStmt():
            body_items2: list[JsonValue] = []
            for s in stmt.body:
                body_items2.append(JDict(_sa_serialize_stmt(s, pfx, plen)))
            d["body"] = JList(body_items2)
        case TIfStmt():
            then_items: list[JsonValue] = []
            for s in stmt.then_body:
                then_items.append(JDict(_sa_serialize_stmt(s, pfx, plen)))
            d["then_body"] = JList(then_items)
            if stmt.else_body is not None:
                else_items: list[JsonValue] = []
                for s in stmt.else_body:
                    else_items.append(JDict(_sa_serialize_stmt(s, pfx, plen)))
                d["else_body"] = JList(else_items)
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
    body_items: list[JsonValue] = []
    for s in fn.body:
        body_items.append(JDict(_sa_serialize_stmt(s, pfx, plen)))
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


# ============================================================
# Concrete union type aliases (closed unions for exhaustiveness checking)
# ============================================================

type ConcreteStmt = (
    TLetStmt
    | TAssignStmt
    | TOpAssignStmt
    | TTupleAssignStmt
    | TReturnStmt
    | TBreakStmt
    | TContinueStmt
    | TThrowStmt
    | TExprStmt
    | TIfStmt
    | TWhileStmt
    | TForStmt
    | TMatchStmt
    | TTryStmt
)

type ConcreteExpr = (
    TRange
    | TIntLit
    | TFloatLit
    | TByteLit
    | TStringLit
    | TRuneLit
    | TBytesLit
    | TBoolLit
    | TNilLit
    | TVar
    | TBinaryOp
    | TUnaryOp
    | TTernary
    | TFieldAccess
    | TTupleAccess
    | TIndex
    | TSlice
    | TCall
    | TListLit
    | TMapLit
    | TSetLit
    | TTupleLit
    | TFnLit
)

type ConcreteDecl = TFnDecl | TStructDecl | TInterfaceDecl | TEnumDecl

type ConcreteType = (
    TPrimitive
    | TListType
    | TMapType
    | TSetType
    | TTupleType
    | TFuncType
    | TIdentType
    | TUnionType
    | TOptionalType
)

type ConcretePattern = TPatternType | TPatternEnum | TPatternNil


def serialize_annotations(module: TModule, prefix: str) -> dict[str, JsonValue]:
    """Serialize all annotations matching prefix from every function into nested dicts."""
    pfx = prefix + "."
    plen = len(pfx)
    result: dict[str, JsonValue] = {}
    for decl in module.decls:
        match decl:
            case TFnDecl():
                result[decl.name] = JDict(_sa_serialize_fn(decl, pfx, plen))
            case TStructDecl():
                for method in decl.methods:
                    result[f"{decl.name}.{method.name}"] = JDict(
                        _sa_serialize_fn(method, pfx, plen)
                    )
    return result


def _cea_add(ann: Ann, line: int, result: dict[int, dict[str, str]]) -> None:
    """Add non-internal annotations to result."""
    if ann:
        if line not in result:
            result[line] = {}
        for k, v in ann.items():
            if not k.startswith("_"):
                result[line][k] = v


def _cea_expr(expr: TExpr, result: dict[int, dict[str, str]]) -> None:
    """Collect expression annotations by line number."""
    if isinstance(expr, TIntLit):
        e: TIntLit = expr
        _cea_add(e.annotations, e.pos.line, result)
    elif isinstance(expr, TFloatLit):
        e2: TFloatLit = expr
        _cea_add(e2.annotations, e2.pos.line, result)
    elif isinstance(expr, TBoolLit):
        e3: TBoolLit = expr
        _cea_add(e3.annotations, e3.pos.line, result)
    elif isinstance(expr, TByteLit):
        e4: TByteLit = expr
        _cea_add(e4.annotations, e4.pos.line, result)
    elif isinstance(expr, TStringLit):
        e5: TStringLit = expr
        _cea_add(e5.annotations, e5.pos.line, result)
    elif isinstance(expr, TRuneLit):
        e6: TRuneLit = expr
        _cea_add(e6.annotations, e6.pos.line, result)
    elif isinstance(expr, TBytesLit):
        e7: TBytesLit = expr
        _cea_add(e7.annotations, e7.pos.line, result)
    elif isinstance(expr, TNilLit):
        e8: TNilLit = expr
        _cea_add(e8.annotations, e8.pos.line, result)
    elif isinstance(expr, TVar):
        e9: TVar = expr
        _cea_add(e9.annotations, e9.pos.line, result)
    elif isinstance(expr, TBinaryOp):
        e10: TBinaryOp = expr
        _cea_add(e10.annotations, e10.pos.line, result)
        _cea_expr(e10.left, result)
        _cea_expr(e10.right, result)
    elif isinstance(expr, TUnaryOp):
        e11: TUnaryOp = expr
        _cea_add(e11.annotations, e11.pos.line, result)
        _cea_expr(e11.operand, result)
    elif isinstance(expr, TCall):
        e12: TCall = expr
        _cea_add(e12.annotations, e12.pos.line, result)
        _cea_expr(e12.func, result)
        for arg in e12.args:
            _cea_expr(arg.value, result)
    elif isinstance(expr, TFieldAccess):
        e13: TFieldAccess = expr
        _cea_add(e13.annotations, e13.pos.line, result)
        _cea_expr(e13.obj, result)
    elif isinstance(expr, TTupleAccess):
        e14: TTupleAccess = expr
        _cea_add(e14.annotations, e14.pos.line, result)
        _cea_expr(e14.obj, result)
    elif isinstance(expr, TIndex):
        e15: TIndex = expr
        _cea_add(e15.annotations, e15.pos.line, result)
        _cea_expr(e15.obj, result)
        _cea_expr(e15.index, result)
    elif isinstance(expr, TTernary):
        e16: TTernary = expr
        _cea_add(e16.annotations, e16.pos.line, result)
        _cea_expr(e16.cond, result)
        _cea_expr(e16.then_expr, result)
        _cea_expr(e16.else_expr, result)
    elif isinstance(expr, TSlice):
        e17: TSlice = expr
        _cea_add(e17.annotations, e17.pos.line, result)
        _cea_expr(e17.obj, result)
        _cea_expr(e17.low, result)
        _cea_expr(e17.high, result)
    elif isinstance(expr, TListLit):
        e18: TListLit = expr
        _cea_add(e18.annotations, e18.pos.line, result)
        for el in e18.elements:
            _cea_expr(el, result)
    elif isinstance(expr, TSetLit):
        e19: TSetLit = expr
        _cea_add(e19.annotations, e19.pos.line, result)
        for el in e19.elements:
            _cea_expr(el, result)
    elif isinstance(expr, TTupleLit):
        e20: TTupleLit = expr
        _cea_add(e20.annotations, e20.pos.line, result)
        for el in e20.elements:
            _cea_expr(el, result)
    elif isinstance(expr, TMapLit):
        e21: TMapLit = expr
        _cea_add(e21.annotations, e21.pos.line, result)
        for k, v in e21.entries:
            _cea_expr(k, result)
            _cea_expr(v, result)
    elif isinstance(expr, TFnLit):
        e22: TFnLit = expr
        _cea_add(e22.annotations, e22.pos.line, result)
        _cea_stmts(e22.body, result)
    elif isinstance(expr, TRange):
        e23: TRange = expr
        _cea_add(e23.annotations, e23.pos.line, result)
        for a in e23.args:
            _cea_expr(a, result)


def _cea_stmt(stmt: TStmt, result: dict[int, dict[str, str]]) -> None:
    """Collect expression annotations from a statement."""
    match stmt:
        case TExprStmt():
            _cea_expr(stmt.expr, result)
        case TReturnStmt():
            if stmt.value is not None:
                _cea_expr(stmt.value, result)
        case TThrowStmt():
            _cea_expr(stmt.expr, result)
        case TLetStmt():
            if stmt.value is not None:
                _cea_expr(stmt.value, result)
        case TAssignStmt():
            _cea_expr(stmt.target, result)
            _cea_expr(stmt.value, result)
        case TOpAssignStmt():
            _cea_expr(stmt.target, result)
            _cea_expr(stmt.value, result)
        case TTupleAssignStmt():
            for t in stmt.targets:
                _cea_expr(t, result)
            _cea_expr(stmt.value, result)
        case TIfStmt():
            _cea_expr(stmt.cond, result)
            _cea_stmts(stmt.then_body, result)
            if stmt.else_body is not None:
                _cea_stmts(stmt.else_body, result)
        case TWhileStmt():
            _cea_expr(stmt.cond, result)
            _cea_stmts(stmt.body, result)
        case TForStmt():
            if isinstance(stmt.iterable, TRange):
                for a in stmt.iterable.args:
                    _cea_expr(a, result)
            else:
                _cea_expr(stmt.iterable, result)
            _cea_stmts(stmt.body, result)
        case TMatchStmt():
            _cea_expr(stmt.expr, result)
            for case in stmt.cases:
                _cea_stmts(case.body, result)
            if stmt.default is not None:
                _cea_stmts(stmt.default.body, result)
        case TTryStmt():
            _cea_stmts(stmt.body, result)
            for catch in stmt.catches:
                _cea_stmts(catch.body, result)
            if stmt.finally_body is not None:
                _cea_stmts(stmt.finally_body, result)
        case TBreakStmt() | TContinueStmt():
            pass


def _cea_stmts(stmts: list[TStmt], result: dict[int, dict[str, str]]) -> None:
    for stmt in stmts:
        _cea_stmt(stmt, result)


def collect_expr_annotations(module: TModule) -> dict[int, dict[str, str]]:
    """Collect all expression annotations from a module, keyed by line number."""
    result: dict[int, dict[str, str]] = {}
    for decl in module.decls:
        match decl:
            case TFnDecl():
                _cea_stmts(decl.body, result)
            case TStructDecl():
                for method in decl.methods:
                    _cea_stmts(method.body, result)
    return result
