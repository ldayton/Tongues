"""Type propagation pass: write checker-computed types onto AST annotations."""

from __future__ import annotations

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
    TTupleAccess,
    TTupleAssignStmt,
    TTupleLit,
    TTryStmt,
    TUnaryOp,
    TVar,
    TWhileStmt,
)
from ..taytsh.check import Checker, Type, type_name


def _annotate_expr(expr: TExpr, expr_types: dict[tuple[int, int], Type]) -> None:
    key = (expr.pos.line, expr.pos.col)
    typ = expr_types.get(key)
    if typ is not None:
        expr.annotations["type"] = type_name(typ)
    if isinstance(expr, TBinaryOp):
        _annotate_expr(expr.left, expr_types)
        _annotate_expr(expr.right, expr_types)
    elif isinstance(expr, TUnaryOp):
        _annotate_expr(expr.operand, expr_types)
    elif isinstance(expr, TTernary):
        _annotate_expr(expr.cond, expr_types)
        _annotate_expr(expr.then_expr, expr_types)
        _annotate_expr(expr.else_expr, expr_types)
    elif isinstance(expr, TFieldAccess):
        _annotate_expr(expr.obj, expr_types)
    elif isinstance(expr, TTupleAccess):
        _annotate_expr(expr.obj, expr_types)
    elif isinstance(expr, TIndex):
        _annotate_expr(expr.obj, expr_types)
        _annotate_expr(expr.index, expr_types)
    elif isinstance(expr, TSlice):
        _annotate_expr(expr.obj, expr_types)
        _annotate_expr(expr.low, expr_types)
        _annotate_expr(expr.high, expr_types)
    elif isinstance(expr, TCall):
        _annotate_expr(expr.func, expr_types)
        for arg in expr.args:
            _annotate_expr(arg.value, expr_types)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _annotate_expr(e, expr_types)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _annotate_expr(e, expr_types)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _annotate_expr(e, expr_types)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _annotate_expr(k, expr_types)
            _annotate_expr(v, expr_types)
    elif isinstance(expr, TFnLit):
        _annotate_stmts(expr.body, expr_types)


def _annotate_target(target: TExpr, expr_types: dict[tuple[int, int], Type]) -> None:
    """Annotate expressions inside assignment targets."""
    if isinstance(target, TVar):
        return
    if isinstance(target, TIndex):
        _annotate_expr(target.obj, expr_types)
        _annotate_expr(target.index, expr_types)
    elif isinstance(target, TFieldAccess):
        _annotate_expr(target.obj, expr_types)
    elif isinstance(target, TTupleAccess):
        _annotate_expr(target.obj, expr_types)


def _annotate_stmts(
    stmts: list[TStmt], expr_types: dict[tuple[int, int], Type]
) -> None:
    for stmt in stmts:
        if isinstance(stmt, TLetStmt):
            if stmt.value is not None:
                _annotate_expr(stmt.value, expr_types)
        elif isinstance(stmt, TAssignStmt):
            _annotate_target(stmt.target, expr_types)
            _annotate_expr(stmt.value, expr_types)
        elif isinstance(stmt, TOpAssignStmt):
            _annotate_target(stmt.target, expr_types)
            _annotate_expr(stmt.value, expr_types)
        elif isinstance(stmt, TTupleAssignStmt):
            for t in stmt.targets:
                _annotate_target(t, expr_types)
            _annotate_expr(stmt.value, expr_types)
        elif isinstance(stmt, TExprStmt):
            _annotate_expr(stmt.expr, expr_types)
        elif isinstance(stmt, TReturnStmt):
            if stmt.value is not None:
                _annotate_expr(stmt.value, expr_types)
        elif isinstance(stmt, TThrowStmt):
            _annotate_expr(stmt.expr, expr_types)
        elif isinstance(stmt, TIfStmt):
            _annotate_expr(stmt.cond, expr_types)
            _annotate_stmts(stmt.then_body, expr_types)
            if stmt.else_body is not None:
                _annotate_stmts(stmt.else_body, expr_types)
        elif isinstance(stmt, TWhileStmt):
            _annotate_expr(stmt.cond, expr_types)
            _annotate_stmts(stmt.body, expr_types)
        elif isinstance(stmt, TForStmt):
            if isinstance(stmt.iterable, TRange):
                for a in stmt.iterable.args:
                    _annotate_expr(a, expr_types)
            else:
                _annotate_expr(stmt.iterable, expr_types)
            _annotate_stmts(stmt.body, expr_types)
        elif isinstance(stmt, TTryStmt):
            _annotate_stmts(stmt.body, expr_types)
            for catch in stmt.catches:
                _annotate_stmts(catch.body, expr_types)
            if stmt.finally_body is not None:
                _annotate_stmts(stmt.finally_body, expr_types)
        elif isinstance(stmt, TMatchStmt):
            _annotate_expr(stmt.expr, expr_types)
            for case in stmt.cases:
                _annotate_stmts(case.body, expr_types)
            if stmt.default is not None:
                _annotate_stmts(stmt.default.body, expr_types)


def _annotate_fn(decl: TFnDecl, expr_types: dict[tuple[int, int], Type]) -> None:
    _annotate_stmts(decl.body, expr_types)


def propagate_types(module: TModule, checker: Checker) -> None:
    """Write checker-computed types onto expr.annotations['type']."""
    et = checker.expr_types
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            _annotate_fn(decl, et)
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                _annotate_fn(method, et)
        elif isinstance(decl, TLetStmt):
            if decl.value is not None:
                _annotate_expr(decl.value, et)
