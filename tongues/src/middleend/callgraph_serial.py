"""Callgraph serialization — JSON output for already-annotated AST nodes."""

from __future__ import annotations

from ..frontend.types import JDict, JStr, JsonValue
from ..taytsh.ast import (
    TAssignStmt,
    TBinaryOp,
    TCall,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFnDecl,
    TFnLit,
    TForStmt,
    TIfStmt,
    TIndex,
    TLetStmt,
    TListLit,
    TMapLit,
    TMatchStmt,
    TModule,
    TOpAssignStmt,
    TRange,
    TReturnStmt,
    TSetLit,
    TSlice,
    TStmt,
    TStructDecl,
    TTernary,
    TThrowStmt,
    TTryStmt,
    TTupleAssignStmt,
    TTupleLit,
    TUnaryOp,
    TVar,
    TWhileStmt,
)
from ..taytsh.check import Checker, StructT


def _sc_collect_calls_stmts(
    stmts: list[TStmt],
    calls: dict[str, dict[str, JsonValue]],
    checker: Checker,
) -> None:
    for stmt in stmts:
        _sc_collect_calls_stmt(stmt, calls, checker)


def _sc_collect_calls_stmt(
    stmt: TStmt,
    calls: dict[str, dict[str, JsonValue]],
    checker: Checker,
) -> None:
    if isinstance(stmt, TExprStmt):
        _sc_collect_calls_expr(stmt.expr, calls, checker)
    elif isinstance(stmt, TReturnStmt) and stmt.value is not None:
        _sc_collect_calls_expr(stmt.value, calls, checker)
    elif isinstance(stmt, TThrowStmt):
        _sc_collect_calls_expr(stmt.expr, calls, checker)
    elif isinstance(stmt, TLetStmt) and stmt.value is not None:
        _sc_collect_calls_expr(stmt.value, calls, checker)
    elif isinstance(stmt, TAssignStmt):
        _sc_collect_calls_expr(stmt.target, calls, checker)
        _sc_collect_calls_expr(stmt.value, calls, checker)
    elif isinstance(stmt, TOpAssignStmt):
        _sc_collect_calls_expr(stmt.target, calls, checker)
        _sc_collect_calls_expr(stmt.value, calls, checker)
    elif isinstance(stmt, TTupleAssignStmt):
        for t in stmt.targets:
            _sc_collect_calls_expr(t, calls, checker)
        _sc_collect_calls_expr(stmt.value, calls, checker)
    elif isinstance(stmt, TIfStmt):
        _sc_collect_calls_expr(stmt.cond, calls, checker)
        _sc_collect_calls_stmts(stmt.then_body, calls, checker)
        if stmt.else_body is not None:
            _sc_collect_calls_stmts(stmt.else_body, calls, checker)
    elif isinstance(stmt, TWhileStmt):
        _sc_collect_calls_expr(stmt.cond, calls, checker)
        _sc_collect_calls_stmts(stmt.body, calls, checker)
    elif isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                _sc_collect_calls_expr(a, calls, checker)
        else:
            _sc_collect_calls_expr(stmt.iterable, calls, checker)
        _sc_collect_calls_stmts(stmt.body, calls, checker)
    elif isinstance(stmt, TMatchStmt):
        _sc_collect_calls_expr(stmt.expr, calls, checker)
        for case in stmt.cases:
            _sc_collect_calls_stmts(case.body, calls, checker)
        if stmt.default is not None:
            _sc_collect_calls_stmts(stmt.default.body, calls, checker)
    elif isinstance(stmt, TTryStmt):
        _sc_collect_calls_stmts(stmt.body, calls, checker)
        for catch in stmt.catches:
            _sc_collect_calls_stmts(catch.body, calls, checker)
        if stmt.finally_body is not None:
            _sc_collect_calls_stmts(stmt.finally_body, calls, checker)


def _sc_collect_calls_expr(
    expr: TExpr,
    calls: dict[str, dict[str, JsonValue]],
    checker: Checker,
) -> None:
    if isinstance(expr, TCall):
        name: str | None = None
        if isinstance(expr.func, TVar):
            n = expr.func.name
            t = checker.types.get(n)
            if t is not None and isinstance(t, StructT):
                name = None
            elif n in checker.functions:
                name = n
            else:
                name = None
        elif isinstance(expr.func, TFieldAccess):
            name = expr.func.field
        if name is not None:
            ann = expr.annotations.get("callgraph.is_tail_call")
            if ann is not None:
                is_tail = JStr(str(ann))
            else:
                is_tail = JStr("false")
            if name not in calls:
                calls[name] = {}
            calls[name]["is_tail_call"] = is_tail
        if isinstance(expr.func, TFieldAccess):
            _sc_collect_calls_expr(expr.func.obj, calls, checker)
        elif not isinstance(expr.func, TVar):
            _sc_collect_calls_expr(expr.func, calls, checker)
        for arg in expr.args:
            _sc_collect_calls_expr(arg.value, calls, checker)
    elif isinstance(expr, TBinaryOp):
        _sc_collect_calls_expr(expr.left, calls, checker)
        _sc_collect_calls_expr(expr.right, calls, checker)
    elif isinstance(expr, TUnaryOp):
        _sc_collect_calls_expr(expr.operand, calls, checker)
    elif isinstance(expr, TTernary):
        _sc_collect_calls_expr(expr.cond, calls, checker)
        _sc_collect_calls_expr(expr.then_expr, calls, checker)
        _sc_collect_calls_expr(expr.else_expr, calls, checker)
    elif isinstance(expr, TFieldAccess):
        _sc_collect_calls_expr(expr.obj, calls, checker)
    elif isinstance(expr, TIndex):
        _sc_collect_calls_expr(expr.obj, calls, checker)
        _sc_collect_calls_expr(expr.index, calls, checker)
    elif isinstance(expr, TSlice):
        _sc_collect_calls_expr(expr.obj, calls, checker)
        _sc_collect_calls_expr(expr.low, calls, checker)
        _sc_collect_calls_expr(expr.high, calls, checker)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _sc_collect_calls_expr(e, calls, checker)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _sc_collect_calls_expr(k, calls, checker)
            _sc_collect_calls_expr(v, calls, checker)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _sc_collect_calls_expr(e, calls, checker)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _sc_collect_calls_expr(e, calls, checker)
    elif isinstance(expr, TFnLit):
        _sc_collect_calls_stmts(expr.body, calls, checker)


def _sc_serialize_fn(decl: TFnDecl, checker: Checker) -> dict[str, JsonValue]:
    """Serialize one function's callgraph data."""
    d: dict[str, JsonValue] = {}
    pfx = "callgraph."
    plen = len(pfx)
    for k in decl.annotations:
        if k.startswith(pfx):
            d[k[plen:]] = JStr(str(decl.annotations[k]))
    calls: dict[str, dict[str, JsonValue]] = {}
    _sc_collect_calls_stmts(decl.body, calls, checker)
    if len(calls) > 0:
        call_entries: dict[str, JsonValue] = {}
        for ck in calls:
            call_entries[ck] = JDict(calls[ck])
        d["calls"] = JDict(call_entries)
    return d


def serialize_callgraph(module: TModule, checker: Checker) -> dict[str, JsonValue]:
    """Serialize callgraph annotations and call info for all functions."""
    result: dict[str, JsonValue] = {}
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            result[decl.name] = JDict(_sc_serialize_fn(decl, checker))
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                key = decl.name + "." + method.name
                result[key] = JDict(_sc_serialize_fn(method, checker))
    return result
