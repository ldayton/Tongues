"""Ownership analysis pass for Taytsh IR.

Infers ownership classifications (owned/borrowed/shared) and detects escapes
for each binding, writing annotations onto AST nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    TPatternType,
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
from ..taytsh.check import Checker, StructT


# ============================================================
# OWNERSHIP CONTEXT
# ============================================================


@dataclass
class _OwnershipCtx:
    checker: Checker
    fn_name: str
    region: str
    var_ownership: dict[str, str]
    params: set[str]
    escaping: set[str]
    let_stmts: dict[str, TLetStmt]


# ============================================================
# OWNERSHIP INFERENCE
# ============================================================


def _infer_ownership(expr: TExpr, ctx: _OwnershipCtx) -> str:
    if isinstance(expr, TCall):
        if isinstance(expr.func, TVar):
            name = expr.func.name
            t = ctx.checker.types.get(name)
            if isinstance(t, StructT):
                return "owned"
            if name == "Get":
                return "borrowed"
        return "owned"
    if isinstance(expr, (TListLit, TMapLit, TSetLit, TTupleLit)):
        return "owned"
    if isinstance(expr, TVar):
        return ctx.var_ownership.get(expr.name, "borrowed")
    if isinstance(expr, TFieldAccess):
        return "borrowed"
    if isinstance(expr, TIndex):
        return "borrowed"
    if isinstance(expr, TSlice):
        return "borrowed"
    if isinstance(expr, TTernary):
        a = _infer_ownership(expr.then_expr, ctx)
        b = _infer_ownership(expr.else_expr, ctx)
        return _join_ownership(a, b)
    return "owned"


def _join_ownership(a: str, b: str) -> str:
    if a == b:
        return a
    return "shared"


# ============================================================
# EXPRESSION WALKER
# ============================================================


def _walk_expr(expr: TExpr, ctx: _OwnershipCtx, escaping: bool) -> None:
    if escaping and isinstance(expr, TVar):
        ownership = _infer_ownership(expr, ctx)
        if ownership == "borrowed":
            expr.annotations["ownership.escapes"] = True
            ctx.escaping.add(expr.name)
    if isinstance(expr, TVar):
        return
    if isinstance(expr, TFieldAccess):
        _walk_expr(expr.obj, ctx, False)
    elif isinstance(expr, TTupleAccess):
        _walk_expr(expr.obj, ctx, False)
    elif isinstance(expr, TIndex):
        _walk_expr(expr.obj, ctx, False)
        _walk_expr(expr.index, ctx, False)
    elif isinstance(expr, TSlice):
        _walk_expr(expr.obj, ctx, False)
        _walk_expr(expr.low, ctx, False)
        _walk_expr(expr.high, ctx, False)
    elif isinstance(expr, TBinaryOp):
        _walk_expr(expr.left, ctx, False)
        _walk_expr(expr.right, ctx, False)
    elif isinstance(expr, TUnaryOp):
        _walk_expr(expr.operand, ctx, False)
    elif isinstance(expr, TTernary):
        _walk_expr(expr.cond, ctx, False)
        _walk_expr(expr.then_expr, ctx, escaping)
        _walk_expr(expr.else_expr, ctx, escaping)
    elif isinstance(expr, TCall):
        _walk_call(expr, ctx)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _walk_expr(e, ctx, True)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _walk_expr(k, ctx, True)
            _walk_expr(v, ctx, True)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _walk_expr(e, ctx, True)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _walk_expr(e, ctx, True)
    elif isinstance(expr, TFnLit):
        _analyze_fn_lit(expr, ctx)


def _walk_call(expr: TCall, ctx: _OwnershipCtx) -> None:
    if isinstance(expr.func, TVar):
        name = expr.func.name
        t = ctx.checker.types.get(name)
        if isinstance(t, StructT):
            for a in expr.args:
                _walk_expr(a.value, ctx, True)
            return
        if name == "Append" or name == "Add":
            if len(expr.args) > 0:
                _walk_expr(expr.args[0].value, ctx, False)
            if len(expr.args) > 1:
                _walk_expr(expr.args[1].value, ctx, True)
            for i in range(2, len(expr.args)):
                _walk_expr(expr.args[i].value, ctx, False)
            return
        if name == "Insert":
            if len(expr.args) > 0:
                _walk_expr(expr.args[0].value, ctx, False)
            if len(expr.args) > 1:
                _walk_expr(expr.args[1].value, ctx, False)
            if len(expr.args) > 2:
                _walk_expr(expr.args[2].value, ctx, True)
            return
        if name == "ReplaceSlice":
            if len(expr.args) > 0:
                _walk_expr(expr.args[0].value, ctx, False)
            for i in range(1, len(expr.args)):
                _walk_expr(expr.args[i].value, ctx, True)
            return
    if isinstance(expr.func, TFieldAccess):
        _walk_expr(expr.func.obj, ctx, False)
    else:
        _walk_expr(expr.func, ctx, False)
    for a in expr.args:
        _walk_expr(a.value, ctx, False)


# ============================================================
# ESCAPE CHECKS
# ============================================================


def _check_field_escape(value: TExpr, ctx: _OwnershipCtx) -> None:
    ownership = _infer_ownership(value, ctx)
    if ownership == "borrowed" and isinstance(value, TVar):
        value.annotations["ownership.escapes"] = True
        ctx.escaping.add(value.name)


def _check_collection_escape(value: TExpr, ctx: _OwnershipCtx) -> None:
    ownership = _infer_ownership(value, ctx)
    if ownership == "borrowed" and isinstance(value, TVar):
        value.annotations["ownership.escapes"] = True
        ctx.escaping.add(value.name)


def _check_return_escape(value: TExpr, ctx: _OwnershipCtx) -> None:
    if isinstance(value, TVar):
        value.annotations["ownership.escapes"] = True
        ctx.escaping.add(value.name)
    elif isinstance(value, TFieldAccess):
        if isinstance(value.obj, TVar):
            value.obj.annotations["ownership.escapes"] = True


def _check_throw_escape(expr: TExpr, ctx: _OwnershipCtx) -> None:
    if isinstance(expr, TVar):
        expr.annotations["ownership.escapes"] = True
        ctx.escaping.add(expr.name)


# ============================================================
# STATEMENT WALKER
# ============================================================


def _walk_stmts(stmts: list[TStmt], ctx: _OwnershipCtx) -> None:
    for s in stmts:
        _walk_stmt(s, ctx)


def _walk_stmt(stmt: TStmt, ctx: _OwnershipCtx) -> None:
    if isinstance(stmt, TLetStmt):
        dead = stmt.annotations.get("liveness.initial_value_unused", False)
        if stmt.value is not None:
            if not dead:
                _walk_expr(stmt.value, ctx, False)
            ownership = _infer_ownership(stmt.value, ctx)
        else:
            ownership = "owned"
        stmt.annotations["ownership.kind"] = ownership
        stmt.annotations["ownership.region"] = ctx.region
        ctx.var_ownership[stmt.name] = ownership
        ctx.let_stmts[stmt.name] = stmt
    elif isinstance(stmt, TAssignStmt):
        _walk_expr(stmt.value, ctx, False)
        if isinstance(stmt.target, TVar):
            ownership = _infer_ownership(stmt.value, ctx)
            prev = ctx.var_ownership.get(stmt.target.name)
            if prev is not None and prev != ownership:
                ctx.var_ownership[stmt.target.name] = _join_ownership(prev, ownership)
            else:
                ctx.var_ownership[stmt.target.name] = ownership
        elif isinstance(stmt.target, TFieldAccess):
            _walk_expr(stmt.target.obj, ctx, False)
            _check_field_escape(stmt.value, ctx)
        elif isinstance(stmt.target, TIndex):
            _walk_expr(stmt.target.obj, ctx, False)
            _walk_expr(stmt.target.index, ctx, True)
            _check_collection_escape(stmt.value, ctx)
        elif isinstance(stmt.target, TTupleAccess):
            _walk_expr(stmt.target.obj, ctx, False)
    elif isinstance(stmt, TTupleAssignStmt):
        _walk_expr(stmt.value, ctx, False)
        for t in stmt.targets:
            if isinstance(t, TVar):
                ctx.var_ownership[t.name] = "owned"
    elif isinstance(stmt, TOpAssignStmt):
        _walk_expr(stmt.value, ctx, False)
        if isinstance(stmt.target, TFieldAccess):
            _walk_expr(stmt.target.obj, ctx, False)
        elif isinstance(stmt.target, TIndex):
            _walk_expr(stmt.target.obj, ctx, False)
            _walk_expr(stmt.target.index, ctx, False)
    elif isinstance(stmt, TExprStmt):
        _walk_expr(stmt.expr, ctx, False)
    elif isinstance(stmt, TReturnStmt):
        if stmt.value is not None:
            _walk_expr(stmt.value, ctx, False)
            _check_return_escape(stmt.value, ctx)
    elif isinstance(stmt, TThrowStmt):
        _walk_expr(stmt.expr, ctx, False)
        _check_throw_escape(stmt.expr, ctx)
    elif isinstance(stmt, TIfStmt):
        _walk_expr(stmt.cond, ctx, False)
        _walk_stmts(stmt.then_body, ctx)
        if stmt.else_body is not None:
            _walk_stmts(stmt.else_body, ctx)
    elif isinstance(stmt, TWhileStmt):
        _walk_expr(stmt.cond, ctx, False)
        _walk_stmts(stmt.body, ctx)
    elif isinstance(stmt, TForStmt):
        _walk_for_stmt(stmt, ctx)
    elif isinstance(stmt, TTryStmt):
        _walk_try_stmt(stmt, ctx)
    elif isinstance(stmt, TMatchStmt):
        _walk_match_stmt(stmt, ctx)


# ============================================================
# FOR STATEMENT
# ============================================================


def _walk_for_stmt(stmt: TForStmt, ctx: _OwnershipCtx) -> None:
    if isinstance(stmt.iterable, TRange):
        for a in stmt.iterable.args:
            _walk_expr(a, ctx, False)
    else:
        _walk_expr(stmt.iterable, ctx, False)
    for i, bname in enumerate(stmt.binding):
        kind = "borrowed"
        stmt.annotations["ownership.binder." + bname + ".kind"] = kind
        stmt.annotations["ownership.binder." + bname + ".region"] = ctx.region
        ctx.var_ownership[bname] = kind
    _walk_stmts(stmt.body, ctx)


# ============================================================
# TRY STATEMENT
# ============================================================


def _walk_try_stmt(stmt: TTryStmt, ctx: _OwnershipCtx) -> None:
    _walk_stmts(stmt.body, ctx)
    for catch in stmt.catches:
        ctx.var_ownership[catch.name] = "borrowed"
        catch.annotations["ownership.kind"] = "borrowed"
        catch.annotations["ownership.region"] = ctx.region
        _walk_stmts(catch.body, ctx)
    if stmt.finally_body is not None:
        _walk_stmts(stmt.finally_body, ctx)


# ============================================================
# MATCH STATEMENT
# ============================================================


def _walk_match_stmt(stmt: TMatchStmt, ctx: _OwnershipCtx) -> None:
    _walk_expr(stmt.expr, ctx, False)
    for case in stmt.cases:
        pat = case.pattern
        if isinstance(pat, TPatternType):
            ctx.var_ownership[pat.name] = "borrowed"
            pat.annotations["ownership.kind"] = "borrowed"
            pat.annotations["ownership.region"] = ctx.region
        _walk_stmts(case.body, ctx)
    if stmt.default is not None:
        if stmt.default.name is not None:
            ctx.var_ownership[stmt.default.name] = "borrowed"
            stmt.default.annotations["ownership.kind"] = "borrowed"
            stmt.default.annotations["ownership.region"] = ctx.region
        _walk_stmts(stmt.default.body, ctx)


# ============================================================
# FUNCTION LITERAL
# ============================================================


def _analyze_fn_lit(expr: TFnLit, parent_ctx: _OwnershipCtx) -> None:
    ctx = _OwnershipCtx(
        checker=parent_ctx.checker,
        fn_name="<lambda>",
        region="fn:<lambda>",
        var_ownership={},
        params=set(),
        escaping=set(),
        let_stmts={},
    )
    for p in expr.params:
        ctx.params.add(p.name)
        ctx.var_ownership[p.name] = "borrowed"
        p.annotations["ownership.kind"] = "borrowed"
        p.annotations["ownership.region"] = ctx.region
    if isinstance(expr.body, list):
        _walk_stmts(expr.body, ctx)
    else:
        _walk_expr(expr.body, ctx, False)
    for name, let_stmt in ctx.let_stmts.items():
        final = ctx.var_ownership.get(name)
        if final is not None:
            let_stmt.annotations["ownership.kind"] = final


# ============================================================
# PER-FUNCTION ANALYSIS
# ============================================================


def _analyze_fn(decl: TFnDecl, checker: Checker) -> None:
    region = "fn:" + decl.name
    ctx = _OwnershipCtx(
        checker=checker,
        fn_name=decl.name,
        region=region,
        var_ownership={},
        params=set(),
        escaping=set(),
        let_stmts={},
    )
    for p in decl.params:
        ctx.params.add(p.name)
        ctx.var_ownership[p.name] = "borrowed"
        p.annotations["ownership.kind"] = "borrowed"
        p.annotations["ownership.region"] = region
    _walk_stmts(decl.body, ctx)
    for name, let_stmt in ctx.let_stmts.items():
        final = ctx.var_ownership.get(name)
        if final is not None:
            let_stmt.annotations["ownership.kind"] = final


# ============================================================
# PUBLIC API
# ============================================================


def analyze_ownership(module: TModule, checker: Checker) -> None:
    """Run ownership analysis on all functions in the module."""
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            _analyze_fn(decl, checker)
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                _analyze_fn(method, checker)
