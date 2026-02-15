"""Strings analysis pass for Taytsh IR.

Classifies string-typed bindings by content (ascii/bmp/unknown), records
string usage patterns (indexed/iterated/len_called), and detects string
builder loops.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..taytsh.ast import (
    TAssignStmt,
    TBinaryOp,
    TBoolLit,
    TByteLit,
    TBytesLit,
    TCall,
    TDefault,
    TEnumDecl,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFloatLit,
    TFnDecl,
    TFnLit,
    TForStmt,
    TIfStmt,
    TIndex,
    TIntLit,
    TLetStmt,
    TListLit,
    TMapLit,
    TMatchStmt,
    TModule,
    TNilLit,
    TOpAssignStmt,
    TPatternEnum,
    TPatternNil,
    TPatternType,
    TParam,
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
    TUnaryOp,
    TVar,
    TWhileStmt,
)
from ..taytsh.check import (
    BOOL_T,
    BYTE_T,
    BYTES_T,
    Checker,
    FLOAT_T,
    INT_T,
    InterfaceT,
    ListT,
    MapT,
    NIL_T,
    ERROR_T,
    RUNE_T,
    STRING_T,
    SetT,
    StructT,
    TupleT,
    Type,
    UnionT,
    normalize_union,
    type_eq,
)

_C_ASCII = "ascii"
_C_BMP = "bmp"
_C_UNKNOWN = "unknown"
_CONTENT_ORDER: dict[str, int] = {_C_ASCII: 0, _C_BMP: 1, _C_UNKNOWN: 2}


@dataclass
class _BindingInfo:
    node: TLetStmt | TForStmt | TPatternType | TDefault | TParam
    name: str
    declared_type: Type
    order: int
    binder_name: str | None = None
    base_unknown: bool = False
    content: str = _C_UNKNOWN
    indexed: bool = False
    iterated: bool = False
    len_called: bool = False


@dataclass
class _Source:
    kind: str  # "expr" | "zero" | "unknown"
    expr: TExpr | None = None


@dataclass
class _StringsCtx:
    checker: Checker
    var_types: dict[str, Type]
    string_bindings: dict[str, _BindingInfo]
    string_sources: dict[str, list[_Source]]
    list_string_types: dict[str, Type]
    list_string_sources: dict[str, list[_Source]]
    decl_order: list[str]
    loop_nodes: list[TForStmt | TWhileStmt]


def _join_content(a: str | None, b: str) -> str:
    if a is None:
        return b
    return b if _CONTENT_ORDER[b] > _CONTENT_ORDER[a] else a


def _contains_string_type(t: Type) -> bool:
    if type_eq(t, STRING_T):
        return True
    if isinstance(t, UnionT):
        for m in t.members:
            if _contains_string_type(m):
                return True
    return False


def _is_list_of_string_type(t: Type) -> bool:
    return isinstance(t, ListT) and _contains_string_type(t.element)


def _literal_content(value: str) -> str:
    content = _C_ASCII
    for ch in value:
        cp = ord(ch)
        if cp > 0xFFFF:
            return _C_UNKNOWN
        if cp > 0x7F:
            content = _C_BMP
    return content


def _register_string_binding(
    ctx: _StringsCtx,
    name: str,
    node: TLetStmt | TForStmt | TPatternType | TDefault | TParam,
    declared_type: Type,
    *,
    binder_name: str | None = None,
    base_unknown: bool = False,
) -> None:
    if name in ctx.string_bindings:
        info = ctx.string_bindings[name]
        info.base_unknown = info.base_unknown or base_unknown
        return
    info = _BindingInfo(
        node=node,
        name=name,
        declared_type=declared_type,
        order=len(ctx.decl_order),
        binder_name=binder_name,
        base_unknown=base_unknown,
    )
    ctx.string_bindings[name] = info
    ctx.string_sources[name] = []
    ctx.decl_order.append(name)


def _add_string_source(ctx: _StringsCtx, name: str, source: _Source) -> None:
    if name not in ctx.string_sources:
        return
    ctx.string_sources[name].append(source)


def _add_list_source(ctx: _StringsCtx, name: str, source: _Source) -> None:
    if name not in ctx.list_string_sources:
        return
    ctx.list_string_sources[name].append(source)


def _resolve_expr_type(expr: TExpr, ctx: _StringsCtx) -> Type | None:
    if isinstance(expr, TVar):
        if expr.name in ctx.var_types:
            return ctx.var_types[expr.name]
        if expr.name in ctx.checker.functions:
            return ctx.checker.functions[expr.name]
        if expr.name in ctx.checker.types:
            return ctx.checker.types[expr.name]
        return None
    if isinstance(expr, TIntLit):
        return INT_T
    if isinstance(expr, TFloatLit):
        return FLOAT_T
    if isinstance(expr, TBoolLit):
        return BOOL_T
    if isinstance(expr, TByteLit):
        return BYTE_T
    if isinstance(expr, TStringLit):
        return STRING_T
    if isinstance(expr, TRuneLit):
        return RUNE_T
    if isinstance(expr, TBytesLit):
        return BYTES_T
    if isinstance(expr, TNilLit):
        return NIL_T
    if isinstance(expr, TListLit):
        if len(expr.elements) == 0:
            return None
        elem_t = _resolve_expr_type(expr.elements[0], ctx)
        if elem_t is not None:
            return ListT(kind="list", element=elem_t)
        return None
    if isinstance(expr, TMapLit):
        if len(expr.entries) == 0:
            return None
        key_t = _resolve_expr_type(expr.entries[0][0], ctx)
        val_t = _resolve_expr_type(expr.entries[0][1], ctx)
        if key_t is not None and val_t is not None:
            return MapT(kind="map", key=key_t, value=val_t)
        return None
    if isinstance(expr, TSetLit):
        if len(expr.elements) == 0:
            return None
        elem_t = _resolve_expr_type(expr.elements[0], ctx)
        if elem_t is not None:
            return SetT(kind="set", element=elem_t)
        return None
    if isinstance(expr, TTupleLit):
        elems: list[Type] = []
        for e in expr.elements:
            t = _resolve_expr_type(e, ctx)
            if t is None:
                return None
            elems.append(t)
        return TupleT(kind="tuple", elements=elems)
    if isinstance(expr, TFieldAccess):
        obj_t = _resolve_expr_type(expr.obj, ctx)
        if isinstance(obj_t, StructT):
            if expr.field in obj_t.fields:
                return obj_t.fields[expr.field]
        return None
    if isinstance(expr, TIndex):
        obj_t = _resolve_expr_type(expr.obj, ctx)
        if isinstance(obj_t, ListT):
            return obj_t.element
        if isinstance(obj_t, MapT):
            return obj_t.value
        if obj_t is not None and type_eq(obj_t, STRING_T):
            return RUNE_T
        if obj_t is not None and type_eq(obj_t, BYTES_T):
            return BYTE_T
        return None
    if isinstance(expr, TSlice):
        obj_t = _resolve_expr_type(expr.obj, ctx)
        if isinstance(obj_t, ListT):
            return obj_t
        if obj_t is not None and type_eq(obj_t, STRING_T):
            return STRING_T
        if obj_t is not None and type_eq(obj_t, BYTES_T):
            return BYTES_T
        return None
    if isinstance(expr, TCall):
        return _resolve_call_type(expr, ctx)
    return None


def _resolve_call_type(expr: TCall, ctx: _StringsCtx) -> Type | None:
    if isinstance(expr.func, TVar):
        name = expr.func.name
        if name in ctx.checker.functions:
            return ctx.checker.functions[name].ret
        if name in ctx.checker.types:
            return ctx.checker.types[name]
        if name in (
            "ToString",
            "Concat",
            "FormatInt",
            "Lower",
            "Upper",
            "Trim",
            "TrimStart",
            "TrimEnd",
            "Replace",
            "Repeat",
            "Reverse",
            "Join",
            "Format",
            "ReadAll",
            "Decode",
        ):
            return STRING_T
        if name == "ReadLine":
            return normalize_union([STRING_T, NIL_T])
        if name in ("Split", "SplitN", "SplitWhitespace"):
            return ListT(kind="list", element=STRING_T)
        if name == "Len":
            return INT_T
        if name == "Args":
            return ListT(kind="list", element=STRING_T)
        if name == "Keys":
            if len(expr.args) > 0:
                t = _resolve_expr_type(expr.args[0].value, ctx)
                if isinstance(t, MapT):
                    return ListT(kind="list", element=t.key)
        if name == "Values":
            if len(expr.args) > 0:
                t = _resolve_expr_type(expr.args[0].value, ctx)
                if isinstance(t, MapT):
                    return ListT(kind="list", element=t.value)
        if name == "Items":
            if len(expr.args) > 0:
                t = _resolve_expr_type(expr.args[0].value, ctx)
                if isinstance(t, MapT):
                    tup = TupleT(kind="tuple", elements=[t.key, t.value])
                    return ListT(kind="list", element=tup)
        if name == "Get":
            if len(expr.args) > 0:
                t = _resolve_expr_type(expr.args[0].value, ctx)
                if isinstance(t, MapT):
                    return normalize_union([t.value, NIL_T])
        if name == "Map":
            return None
        if name == "Set":
            return None
        return None
    if isinstance(expr.func, TFieldAccess):
        obj_t = _resolve_expr_type(expr.func.obj, ctx)
        if isinstance(obj_t, StructT) and expr.func.field in obj_t.methods:
            return obj_t.methods[expr.func.field].ret
    return None


def _resolve_for_binder_types(
    stmt: TForStmt, ctx: _StringsCtx
) -> dict[str, Type] | None:
    if isinstance(stmt.iterable, TRange):
        result: dict[str, Type] = {}
        for b in stmt.binding:
            result[b] = INT_T
        return result
    iter_t = _resolve_expr_type(stmt.iterable, ctx)
    if iter_t is None:
        return None
    result2: dict[str, Type] = {}
    if isinstance(iter_t, ListT):
        if len(stmt.binding) == 1:
            result2[stmt.binding[0]] = iter_t.element
        elif len(stmt.binding) == 2:
            result2[stmt.binding[0]] = INT_T
            result2[stmt.binding[1]] = iter_t.element
    elif isinstance(iter_t, MapT):
        if len(stmt.binding) == 1:
            result2[stmt.binding[0]] = iter_t.key
        elif len(stmt.binding) == 2:
            result2[stmt.binding[0]] = iter_t.key
            result2[stmt.binding[1]] = iter_t.value
    elif isinstance(iter_t, SetT):
        if len(stmt.binding) == 1:
            result2[stmt.binding[0]] = iter_t.element
    elif type_eq(iter_t, STRING_T):
        if len(stmt.binding) == 1:
            result2[stmt.binding[0]] = RUNE_T
        elif len(stmt.binding) == 2:
            result2[stmt.binding[0]] = INT_T
            result2[stmt.binding[1]] = RUNE_T
    elif type_eq(iter_t, BYTES_T):
        if len(stmt.binding) == 1:
            result2[stmt.binding[0]] = BYTE_T
        elif len(stmt.binding) == 2:
            result2[stmt.binding[0]] = INT_T
            result2[stmt.binding[1]] = BYTE_T
    return result2 if len(result2) > 0 else None


def _compute_residual_type(scrutinee: Type | None, covered: list[Type]) -> Type:
    if scrutinee is None:
        return ERROR_T
    if isinstance(scrutinee, UnionT):
        remaining: list[Type] = []
        for m in scrutinee.members:
            is_covered = False
            for c in covered:
                if type_eq(m, c):
                    is_covered = True
                    break
            if not is_covered:
                remaining.append(m)
        if len(remaining) == 0:
            return ERROR_T
        if len(remaining) == 1:
            return remaining[0]
        return normalize_union(remaining)
    for c in covered:
        if type_eq(scrutinee, c):
            return ERROR_T
    return scrutinee


def _base_var(expr: TExpr) -> str | None:
    if isinstance(expr, TVar):
        return expr.name
    if isinstance(expr, TFieldAccess):
        return _base_var(expr.obj)
    if isinstance(expr, TIndex):
        return _base_var(expr.obj)
    if isinstance(expr, TTupleAccess):
        return _base_var(expr.obj)
    if isinstance(expr, TSlice):
        return _base_var(expr.obj)
    return None


def _expr_reads(name: str, expr: TExpr) -> bool:
    if isinstance(expr, TVar):
        return expr.name == name
    if isinstance(expr, TBinaryOp):
        return _expr_reads(name, expr.left) or _expr_reads(name, expr.right)
    if isinstance(expr, TUnaryOp):
        return _expr_reads(name, expr.operand)
    if isinstance(expr, TTernary):
        return (
            _expr_reads(name, expr.cond)
            or _expr_reads(name, expr.then_expr)
            or _expr_reads(name, expr.else_expr)
        )
    if isinstance(expr, TFieldAccess):
        return _expr_reads(name, expr.obj)
    if isinstance(expr, TTupleAccess):
        return _expr_reads(name, expr.obj)
    if isinstance(expr, TIndex):
        return _expr_reads(name, expr.obj) or _expr_reads(name, expr.index)
    if isinstance(expr, TSlice):
        return (
            _expr_reads(name, expr.obj)
            or _expr_reads(name, expr.low)
            or _expr_reads(name, expr.high)
        )
    if isinstance(expr, TCall):
        if _expr_reads(name, expr.func):
            return True
        for a in expr.args:
            if _expr_reads(name, a.value):
                return True
        return False
    if isinstance(expr, TListLit):
        for e in expr.elements:
            if _expr_reads(name, e):
                return True
        return False
    if isinstance(expr, TTupleLit):
        for e in expr.elements:
            if _expr_reads(name, e):
                return True
        return False
    if isinstance(expr, TSetLit):
        for e in expr.elements:
            if _expr_reads(name, e):
                return True
        return False
    if isinstance(expr, TMapLit):
        for k, v in expr.entries:
            if _expr_reads(name, k) or _expr_reads(name, v):
                return True
        return False
    return False


def _target_reads(name: str, target: TExpr) -> bool:
    if isinstance(target, TVar):
        return False
    if isinstance(target, TIndex):
        return _expr_reads(name, target.obj) or _expr_reads(name, target.index)
    if isinstance(target, TFieldAccess):
        return _expr_reads(name, target.obj)
    if isinstance(target, TTupleAccess):
        return _expr_reads(name, target.obj)
    if isinstance(target, TSlice):
        return (
            _expr_reads(name, target.obj)
            or _expr_reads(name, target.low)
            or _expr_reads(name, target.high)
        )
    return False


def _is_accum_concat(value: TExpr, name: str) -> bool:
    if not isinstance(value, TCall):
        return False
    if not isinstance(value.func, TVar):
        return False
    if value.func.name != "Concat":
        return False
    if len(value.args) < 2:
        return False
    first = value.args[0].value
    if not (isinstance(first, TVar) and first.name == name):
        return False
    i = 1
    while i < len(value.args):
        if _expr_reads(name, value.args[i].value):
            return False
        i += 1
    return True


def _builder_walk_stmt(name: str, stmt: TStmt, found: list[bool]) -> bool:
    if isinstance(stmt, TLetStmt):
        if stmt.value is not None and _expr_reads(name, stmt.value):
            return False
        return True
    if isinstance(stmt, TAssignStmt):
        if isinstance(stmt.target, TVar) and stmt.target.name == name:
            if _is_accum_concat(stmt.value, name):
                found[0] = True
                return True
            return False
        if _expr_reads(name, stmt.value) or _target_reads(name, stmt.target):
            return False
        return True
    if isinstance(stmt, TOpAssignStmt):
        if isinstance(stmt.target, TVar) and stmt.target.name == name:
            return False
        if _expr_reads(name, stmt.value) or _expr_reads(name, stmt.target):
            return False
        return True
    if isinstance(stmt, TTupleAssignStmt):
        if _expr_reads(name, stmt.value):
            return False
        for t in stmt.targets:
            if isinstance(t, TVar) and t.name == name:
                return False
            if _target_reads(name, t):
                return False
        return True
    if isinstance(stmt, TExprStmt):
        return not _expr_reads(name, stmt.expr)
    if isinstance(stmt, TReturnStmt):
        return stmt.value is None or (not _expr_reads(name, stmt.value))
    if isinstance(stmt, TThrowStmt):
        return not _expr_reads(name, stmt.expr)
    if isinstance(stmt, TIfStmt):
        if _expr_reads(name, stmt.cond):
            return False
        for s in stmt.then_body:
            if not _builder_walk_stmt(name, s, found):
                return False
        if stmt.else_body is not None:
            for s in stmt.else_body:
                if not _builder_walk_stmt(name, s, found):
                    return False
        return True
    if isinstance(stmt, TWhileStmt):
        if _expr_reads(name, stmt.cond):
            return False
        for s in stmt.body:
            if not _builder_walk_stmt(name, s, found):
                return False
        return True
    if isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                if _expr_reads(name, a):
                    return False
        else:
            if _expr_reads(name, stmt.iterable):
                return False
        for s in stmt.body:
            if not _builder_walk_stmt(name, s, found):
                return False
        return True
    if isinstance(stmt, TTryStmt):
        for s in stmt.body:
            if not _builder_walk_stmt(name, s, found):
                return False
        for catch in stmt.catches:
            for s in catch.body:
                if not _builder_walk_stmt(name, s, found):
                    return False
        if stmt.finally_body is not None:
            for s in stmt.finally_body:
                if not _builder_walk_stmt(name, s, found):
                    return False
        return True
    if isinstance(stmt, TMatchStmt):
        if _expr_reads(name, stmt.expr):
            return False
        for case in stmt.cases:
            for s in case.body:
                if not _builder_walk_stmt(name, s, found):
                    return False
        if stmt.default is not None:
            for s in stmt.default.body:
                if not _builder_walk_stmt(name, s, found):
                    return False
        return True
    return True


def _candidate_builder_in_stmts(name: str, stmts: list[TStmt]) -> bool:
    found: list[bool] = [False]
    for stmt in stmts:
        if not _builder_walk_stmt(name, stmt, found):
            return False
    return found[0]


def _classify_list_source(
    source: _Source,
    string_content: dict[str, str],
    list_content: dict[str, str],
    ctx: _StringsCtx,
) -> str:
    if source.kind == "unknown":
        return _C_UNKNOWN
    if source.expr is None:
        return _C_UNKNOWN
    expr = source.expr
    if isinstance(expr, TListLit):
        level: str | None = _C_ASCII
        for e in expr.elements:
            c = _classify_string_expr(e, string_content, list_content, ctx)
            level = _join_content(level, c)
        return level if level is not None else _C_ASCII
    if isinstance(expr, TVar):
        if expr.name in list_content:
            return list_content[expr.name]
    return _C_UNKNOWN


def _classify_string_expr(
    expr: TExpr,
    string_content: dict[str, str],
    list_content: dict[str, str],
    ctx: _StringsCtx,
) -> str:
    if isinstance(expr, TStringLit):
        return _literal_content(expr.value)
    if isinstance(expr, TVar):
        if expr.name in string_content:
            return string_content[expr.name]
        return _C_UNKNOWN
    if isinstance(expr, TSlice):
        return _classify_string_expr(expr.obj, string_content, list_content, ctx)
    if isinstance(expr, TTernary):
        a = _classify_string_expr(expr.then_expr, string_content, list_content, ctx)
        b = _classify_string_expr(expr.else_expr, string_content, list_content, ctx)
        return _join_content(a, b)
    if isinstance(expr, TCall):
        if isinstance(expr.func, TVar):
            name = expr.func.name
            if name in ("ReadAll", "ReadLine", "ReadFile", "Decode"):
                return _C_UNKNOWN
            if name == "ToString":
                if len(expr.args) == 0:
                    return _C_UNKNOWN
                t = _resolve_expr_type(expr.args[0].value, ctx)
                if t is not None and (
                    type_eq(t, INT_T)
                    or type_eq(t, FLOAT_T)
                    or type_eq(t, BOOL_T)
                    or type_eq(t, BYTE_T)
                ):
                    return _C_ASCII
                return _C_UNKNOWN
            if name == "FormatInt":
                return _C_ASCII
            if name == "Concat":
                if len(expr.args) < 2:
                    return _C_UNKNOWN
                a = _classify_string_expr(
                    expr.args[0].value, string_content, list_content, ctx
                )
                b = _classify_string_expr(
                    expr.args[1].value, string_content, list_content, ctx
                )
                return _join_content(a, b)
            if name == "Format":
                if len(expr.args) == 0:
                    return _C_UNKNOWN
                level: str | None = None
                for a in expr.args:
                    level = _join_content(
                        level,
                        _classify_string_expr(
                            a.value, string_content, list_content, ctx
                        ),
                    )
                return level if level is not None else _C_UNKNOWN
            if name in (
                "Lower",
                "Upper",
                "Trim",
                "TrimStart",
                "TrimEnd",
                "Repeat",
                "Reverse",
            ):
                if len(expr.args) == 0:
                    return _C_UNKNOWN
                return _classify_string_expr(
                    expr.args[0].value, string_content, list_content, ctx
                )
            if name == "Replace":
                if len(expr.args) < 3:
                    return _C_UNKNOWN
                level: str | None = None
                for i in range(3):
                    level = _join_content(
                        level,
                        _classify_string_expr(
                            expr.args[i].value, string_content, list_content, ctx
                        ),
                    )
                return level if level is not None else _C_UNKNOWN
            if name == "Join":
                if len(expr.args) < 2:
                    return _C_UNKNOWN
                sep_c = _classify_string_expr(
                    expr.args[0].value, string_content, list_content, ctx
                )
                parts_expr = expr.args[1].value
                parts_c = _C_UNKNOWN
                if isinstance(parts_expr, TListLit):
                    level: str | None = _C_ASCII
                    for e in parts_expr.elements:
                        level = _join_content(
                            level,
                            _classify_string_expr(e, string_content, list_content, ctx),
                        )
                    parts_c = level if level is not None else _C_ASCII
                elif isinstance(parts_expr, TVar) and parts_expr.name in list_content:
                    parts_c = list_content[parts_expr.name]
                return _join_content(sep_c, parts_c)
            return _C_UNKNOWN
        return _C_UNKNOWN
    if isinstance(expr, (TFieldAccess, TIndex)):
        return _C_UNKNOWN
    return _C_UNKNOWN


def _mark_len_called_from_call(expr: TCall, ctx: _StringsCtx) -> None:
    if isinstance(expr.func, TVar) and expr.func.name == "Len" and len(expr.args) > 0:
        base = _base_var(expr.args[0].value)
        if base is not None and base in ctx.string_bindings:
            ctx.string_bindings[base].len_called = True


def _walk_expr_usage(expr: TExpr, ctx: _StringsCtx) -> None:
    if isinstance(expr, TBinaryOp):
        _walk_expr_usage(expr.left, ctx)
        _walk_expr_usage(expr.right, ctx)
    elif isinstance(expr, TUnaryOp):
        _walk_expr_usage(expr.operand, ctx)
    elif isinstance(expr, TTernary):
        _walk_expr_usage(expr.cond, ctx)
        _walk_expr_usage(expr.then_expr, ctx)
        _walk_expr_usage(expr.else_expr, ctx)
    elif isinstance(expr, TFieldAccess):
        _walk_expr_usage(expr.obj, ctx)
    elif isinstance(expr, TTupleAccess):
        _walk_expr_usage(expr.obj, ctx)
    elif isinstance(expr, TIndex):
        base = _base_var(expr.obj)
        if base is not None and base in ctx.string_bindings:
            ctx.string_bindings[base].indexed = True
        _walk_expr_usage(expr.obj, ctx)
        _walk_expr_usage(expr.index, ctx)
    elif isinstance(expr, TSlice):
        base = _base_var(expr.obj)
        if base is not None and base in ctx.string_bindings:
            ctx.string_bindings[base].indexed = True
        _walk_expr_usage(expr.obj, ctx)
        _walk_expr_usage(expr.low, ctx)
        _walk_expr_usage(expr.high, ctx)
    elif isinstance(expr, TCall):
        _mark_len_called_from_call(expr, ctx)
        _walk_expr_usage(expr.func, ctx)
        for a in expr.args:
            _walk_expr_usage(a.value, ctx)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _walk_expr_usage(e, ctx)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _walk_expr_usage(k, ctx)
            _walk_expr_usage(v, ctx)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _walk_expr_usage(e, ctx)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _walk_expr_usage(e, ctx)
    elif isinstance(expr, TFnLit):
        if isinstance(expr.body, list):
            _walk_stmts(expr.body, ctx, set())
        else:
            _walk_expr_usage(expr.body, ctx)


def _compute_builder(
    loop: TForStmt | TWhileStmt, ctx: _StringsCtx, declared_before: set[str]
) -> str:
    candidates: list[str] = []
    for name in ctx.decl_order:
        if name in declared_before:
            candidates.append(name)
    builders: list[str] = []
    for name in candidates:
        if _candidate_builder_in_stmts(name, loop.body):
            builders.append(name)
    return ",".join(builders)


def _walk_stmts(stmts: list[TStmt], ctx: _StringsCtx, declared: set[str]) -> None:
    for stmt in stmts:
        _walk_stmt(stmt, ctx, declared)


def _walk_stmt(stmt: TStmt, ctx: _StringsCtx, declared: set[str]) -> None:
    if isinstance(stmt, TLetStmt):
        if stmt.value is not None:
            _walk_expr_usage(stmt.value, ctx)
        declared_t = ctx.checker.resolve_type(stmt.typ)
        ctx.var_types[stmt.name] = declared_t
        if _contains_string_type(declared_t):
            _register_string_binding(
                ctx, stmt.name, stmt, declared_t, base_unknown=False
            )
            dead = bool(stmt.annotations.get("liveness.initial_value_unused", False))
            if not dead:
                if stmt.value is None:
                    _add_string_source(ctx, stmt.name, _Source(kind="zero"))
                else:
                    _add_string_source(
                        ctx, stmt.name, _Source(kind="expr", expr=stmt.value)
                    )
        if _is_list_of_string_type(declared_t):
            ctx.list_string_types[stmt.name] = declared_t
            if stmt.name not in ctx.list_string_sources:
                ctx.list_string_sources[stmt.name] = []
            if stmt.value is not None:
                _add_list_source(ctx, stmt.name, _Source(kind="expr", expr=stmt.value))
        declared.add(stmt.name)
        return

    if isinstance(stmt, TAssignStmt):
        _walk_expr_usage(stmt.target, ctx)
        _walk_expr_usage(stmt.value, ctx)
        if isinstance(stmt.target, TVar):
            name = stmt.target.name
            if name in ctx.string_bindings:
                _add_string_source(ctx, name, _Source(kind="expr", expr=stmt.value))
            if name in ctx.list_string_sources:
                _add_list_source(ctx, name, _Source(kind="expr", expr=stmt.value))
        return

    if isinstance(stmt, TOpAssignStmt):
        _walk_expr_usage(stmt.target, ctx)
        _walk_expr_usage(stmt.value, ctx)
        if isinstance(stmt.target, TVar):
            name = stmt.target.name
            if name in ctx.string_bindings:
                _add_string_source(ctx, name, _Source(kind="unknown"))
        return

    if isinstance(stmt, TTupleAssignStmt):
        for t in stmt.targets:
            _walk_expr_usage(t, ctx)
        _walk_expr_usage(stmt.value, ctx)
        for t in stmt.targets:
            if isinstance(t, TVar) and t.name in ctx.string_bindings:
                _add_string_source(ctx, t.name, _Source(kind="unknown"))
        return

    if isinstance(stmt, TExprStmt):
        _walk_expr_usage(stmt.expr, ctx)
        return

    if isinstance(stmt, TReturnStmt):
        if stmt.value is not None:
            _walk_expr_usage(stmt.value, ctx)
        return

    if isinstance(stmt, TThrowStmt):
        _walk_expr_usage(stmt.expr, ctx)
        return

    if isinstance(stmt, TIfStmt):
        _walk_expr_usage(stmt.cond, ctx)
        _walk_stmts(stmt.then_body, ctx, set(declared))
        if stmt.else_body is not None:
            _walk_stmts(stmt.else_body, ctx, set(declared))
        return

    if isinstance(stmt, TWhileStmt):
        _walk_expr_usage(stmt.cond, ctx)
        stmt.annotations["strings.builder"] = _compute_builder(stmt, ctx, declared)
        ctx.loop_nodes.append(stmt)
        _walk_stmts(stmt.body, ctx, set(declared))
        return

    if isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                _walk_expr_usage(a, ctx)
        else:
            _walk_expr_usage(stmt.iterable, ctx)
            iter_base = _base_var(stmt.iterable)
            if iter_base is not None and iter_base in ctx.string_bindings:
                ctx.string_bindings[iter_base].iterated = True

        stmt.annotations["strings.builder"] = _compute_builder(stmt, ctx, declared)
        ctx.loop_nodes.append(stmt)

        binder_types = _resolve_for_binder_types(stmt, ctx)
        child_declared = set(declared)
        for bname in stmt.binding:
            btype = binder_types.get(bname) if binder_types is not None else None
            if btype is None:
                btype = ERROR_T
            ctx.var_types[bname] = btype
            if _contains_string_type(btype):
                _register_string_binding(
                    ctx,
                    bname,
                    stmt,
                    btype,
                    binder_name=bname,
                    base_unknown=True,
                )
            child_declared.add(bname)

        _walk_stmts(stmt.body, ctx, child_declared)
        return

    if isinstance(stmt, TTryStmt):
        _walk_stmts(stmt.body, ctx, set(declared))
        for catch in stmt.catches:
            _walk_stmts(catch.body, ctx, set(declared))
        if stmt.finally_body is not None:
            _walk_stmts(stmt.finally_body, ctx, set(declared))
        return

    if isinstance(stmt, TMatchStmt):
        _walk_expr_usage(stmt.expr, ctx)
        scrutinee_t = _resolve_expr_type(stmt.expr, ctx)
        covered: list[Type] = []
        for case in stmt.cases:
            case_declared = set(declared)
            pat = case.pattern
            if isinstance(pat, TPatternType):
                case_t = ctx.checker.resolve_type(pat.type_name)
                covered.append(case_t)
                ctx.var_types[pat.name] = case_t
                if _contains_string_type(case_t):
                    _register_string_binding(
                        ctx,
                        pat.name,
                        pat,
                        case_t,
                        base_unknown=True,
                    )
                case_declared.add(pat.name)
            elif isinstance(pat, TPatternEnum):
                enum_t = ctx.checker.types.get(pat.enum_name)
                if enum_t is not None and isinstance(enum_t, (InterfaceT, StructT)):
                    covered.append(enum_t)
                elif enum_t is not None and isinstance(enum_t, Type):
                    covered.append(enum_t)
            elif isinstance(pat, TPatternNil):
                covered.append(NIL_T)
            _walk_stmts(case.body, ctx, case_declared)
        if stmt.default is not None:
            dflt_declared = set(declared)
            if stmt.default.name is not None:
                residual = _compute_residual_type(scrutinee_t, covered)
                ctx.var_types[stmt.default.name] = residual
                if _contains_string_type(residual):
                    _register_string_binding(
                        ctx,
                        stmt.default.name,
                        stmt.default,
                        residual,
                        base_unknown=True,
                    )
                dflt_declared.add(stmt.default.name)
            _walk_stmts(stmt.default.body, ctx, dflt_declared)
        return


def _compute_contents(ctx: _StringsCtx) -> None:
    string_content: dict[str, str] = {}
    list_content: dict[str, str] = {}

    for name, info in ctx.string_bindings.items():
        if info.base_unknown:
            string_content[name] = _C_UNKNOWN
        else:
            string_content[name] = _C_ASCII
            has_non_dead_source = False
            for src in ctx.string_sources.get(name, []):
                if src.kind == "expr":
                    string_content[name] = _classify_string_expr(
                        src.expr, string_content, list_content, ctx
                    )
                    has_non_dead_source = True
                    break
                if src.kind == "zero":
                    string_content[name] = _C_ASCII
                    has_non_dead_source = True
                    break
                if src.kind == "unknown":
                    string_content[name] = _C_UNKNOWN
                    has_non_dead_source = True
                    break
            if not has_non_dead_source:
                string_content[name] = _C_UNKNOWN

    for name in ctx.list_string_sources:
        list_content[name] = _C_UNKNOWN

    limit = len(ctx.string_bindings) + len(ctx.list_string_sources) + 4
    i = 0
    while i < limit:
        changed = False

        for name, sources in ctx.list_string_sources.items():
            level: str | None = None
            for src in sources:
                c = _classify_list_source(src, string_content, list_content, ctx)
                level = _join_content(level, c)
            new_level = level if level is not None else _C_UNKNOWN
            if list_content.get(name) != new_level:
                list_content[name] = new_level
                changed = True

        for name, info in ctx.string_bindings.items():
            level: str | None = _C_UNKNOWN if info.base_unknown else None
            for src in ctx.string_sources.get(name, []):
                if src.kind == "zero":
                    level = _join_content(level, _C_ASCII)
                elif src.kind == "unknown":
                    level = _join_content(level, _C_UNKNOWN)
                elif src.expr is not None:
                    c = _classify_string_expr(
                        src.expr, string_content, list_content, ctx
                    )
                    level = _join_content(level, c)
            new_level = level if level is not None else _C_UNKNOWN
            if string_content.get(name) != new_level:
                string_content[name] = new_level
                changed = True

        if not changed:
            break
        i += 1

    for name, info in ctx.string_bindings.items():
        info.content = string_content.get(name, _C_UNKNOWN)


def _stamp_bindings(ctx: _StringsCtx) -> None:
    for name in ctx.decl_order:
        info = ctx.string_bindings[name]
        if info.binder_name is not None:
            b = info.binder_name
            info.node.annotations[f"strings.binder.{b}.content"] = info.content
            info.node.annotations[f"strings.binder.{b}.indexed"] = info.indexed
            info.node.annotations[f"strings.binder.{b}.iterated"] = info.iterated
            info.node.annotations[f"strings.binder.{b}.len_called"] = info.len_called
            continue
        info.node.annotations["strings.content"] = info.content
        info.node.annotations["strings.indexed"] = info.indexed
        info.node.annotations["strings.iterated"] = info.iterated
        info.node.annotations["strings.len_called"] = info.len_called


def _analyze_fn(decl: TFnDecl, checker: Checker, self_type: Type | None = None) -> None:
    ctx = _StringsCtx(
        checker=checker,
        var_types={},
        string_bindings={},
        string_sources={},
        list_string_types={},
        list_string_sources={},
        decl_order=[],
        loop_nodes=[],
    )

    declared: set[str] = set()
    for p in decl.params:
        if p.typ is not None:
            pt = checker.resolve_type(p.typ)
        elif p.name == "self" and self_type is not None:
            pt = self_type
        else:
            continue
        ctx.var_types[p.name] = pt
        declared.add(p.name)
        if _contains_string_type(pt):
            _register_string_binding(ctx, p.name, p, pt, base_unknown=True)
        if _is_list_of_string_type(pt):
            ctx.list_string_types[p.name] = pt
            ctx.list_string_sources[p.name] = []

    _walk_stmts(decl.body, ctx, declared)
    _compute_contents(ctx)
    _stamp_bindings(ctx)

    for stmt in ctx.loop_nodes:
        if "strings.builder" not in stmt.annotations:
            stmt.annotations["strings.builder"] = ""


def analyze_strings(module: TModule, checker: Checker) -> None:
    """Run strings analysis on all functions in the module."""
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            _analyze_fn(decl, checker)
        elif isinstance(decl, TStructDecl):
            st = checker.types.get(decl.name)
            for method in decl.methods:
                _analyze_fn(method, checker, self_type=st)
        elif isinstance(decl, TEnumDecl):
            continue
