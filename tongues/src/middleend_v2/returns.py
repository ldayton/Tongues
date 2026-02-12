"""Returns analysis pass for Taytsh IR.

Analyzes each function body independently, writing annotations onto AST nodes
for always-returns, needs-named-returns, may-return-nil, and try-body-has-return.
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
    TForStmt,
    TIfStmt,
    TLetStmt,
    TMatchStmt,
    TModule,
    TNilLit,
    TReturnStmt,
    TStmt,
    TStructDecl,
    TThrowStmt,
    TTernary,
    TTryStmt,
    TVar,
    TWhileStmt,
)
from ..taytsh.check import (
    Checker,
    NIL_T,
    StructT,
    Type,
    contains_nil,
    remove_nil,
)


# ============================================================
# ACCUMULATORS AND CONTEXT
# ============================================================


@dataclass
class _FnResults:
    """Mutable accumulator for function-level facts."""

    needs_named_returns: bool = False
    may_return_nil: bool = False


class _ReturnsCtx:
    def __init__(
        self,
        checker: Checker,
        locals: dict[str, Type] | None = None,
        narrowings: dict[str, Type] | None = None,
        fn_results: _FnResults | None = None,
    ) -> None:
        self.checker = checker
        self.locals: dict[str, Type] = locals if locals is not None else {}
        self.narrowings: dict[str, Type] = narrowings if narrowings is not None else {}
        self.fn_results: _FnResults = (
            fn_results if fn_results is not None else _FnResults()
        )


def _fork_ctx(
    ctx: _ReturnsCtx, extra_narrowings: dict[str, Type] | None = None
) -> _ReturnsCtx:
    new_narrowings = dict(ctx.narrowings)
    if extra_narrowings is not None:
        new_narrowings.update(extra_narrowings)
    return _ReturnsCtx(
        checker=ctx.checker,
        locals=ctx.locals,
        narrowings=new_narrowings,
        fn_results=ctx.fn_results,
    )


# ============================================================
# HELPERS
# ============================================================


def _is_exit_call(stmt: TStmt) -> bool:
    """Check if stmt is Exit() call."""
    if isinstance(stmt, TExprStmt) and isinstance(stmt.expr, TCall):
        if isinstance(stmt.expr.func, TVar) and stmt.expr.func.name == "Exit":
            return True
    return False


def _resolve_expr_type(expr: TExpr, ctx: _ReturnsCtx) -> Type | None:
    """Minimal type resolver — only enough for may_return_nil."""
    if isinstance(expr, TNilLit):
        return NIL_T
    if isinstance(expr, TVar):
        if expr.name in ctx.narrowings:
            return ctx.narrowings[expr.name]
        if expr.name in ctx.locals:
            return ctx.locals[expr.name]
        if expr.name in ctx.checker.functions:
            return ctx.checker.functions[expr.name]
        return None
    if isinstance(expr, TCall):
        if isinstance(expr.func, TVar):
            name = expr.func.name
            if name == "Get":
                return NIL_T
            if name in ctx.checker.functions:
                return ctx.checker.functions[name].ret
        if isinstance(expr.func, TFieldAccess):
            obj_t = _resolve_expr_type(expr.func.obj, ctx)
            if obj_t is not None and isinstance(obj_t, StructT):
                if expr.func.field in obj_t.methods:
                    return obj_t.methods[expr.func.field].ret
        return None
    if isinstance(expr, TFieldAccess):
        obj_t = _resolve_expr_type(expr.obj, ctx)
        if obj_t is not None and isinstance(obj_t, StructT):
            if expr.field in obj_t.fields:
                return obj_t.fields[expr.field]
        return None
    if isinstance(expr, TTernary):
        t_then = _resolve_expr_type(expr.then_expr, ctx)
        t_else = _resolve_expr_type(expr.else_expr, ctx)
        if t_then is not None and t_else is not None:
            if contains_nil(t_then) or contains_nil(t_else):
                return NIL_T
        return t_then
    return None


def _extract_nil_narrowing(
    cond: TExpr, ctx: _ReturnsCtx
) -> tuple[dict[str, Type], dict[str, Type]]:
    """Extract nil narrowings from a condition. Returns (then_narrows, else_narrows)."""
    then_narrows: dict[str, Type] = {}
    else_narrows: dict[str, Type] = {}
    if not isinstance(cond, TBinaryOp) or cond.op not in ("!=", "=="):
        return then_narrows, else_narrows
    var_node: TVar | None = None
    if isinstance(cond.left, TVar) and isinstance(cond.right, TNilLit):
        var_node = cond.left
    elif isinstance(cond.right, TVar) and isinstance(cond.left, TNilLit):
        var_node = cond.right
    if var_node is None:
        return then_narrows, else_narrows
    name = var_node.name
    typ = ctx.narrowings.get(name) or ctx.locals.get(name)
    if typ is None or not contains_nil(typ):
        return then_narrows, else_narrows
    non_nil = remove_nil(typ)
    if cond.op == "!=":
        then_narrows[name] = non_nil
        else_narrows[name] = NIL_T
    else:
        then_narrows[name] = NIL_T
        else_narrows[name] = non_nil
    return then_narrows, else_narrows


def _may_be_nil(expr: TExpr, ctx: _ReturnsCtx) -> bool:
    """Check if an expression might evaluate to nil."""
    if isinstance(expr, TNilLit):
        return True
    t = _resolve_expr_type(expr, ctx)
    return t is not None and contains_nil(t)


def _merge_branch_narrowings(
    ctx: _ReturnsCtx, branches: list[_ReturnsCtx]
) -> None:
    """Merge narrowings back to parent: non-nil only if ALL branches agree."""
    if not branches:
        return
    common = set(branches[0].narrowings.keys())
    for bctx in branches[1:]:
        common &= set(bctx.narrowings.keys())
    for name in common:
        if all(not contains_nil(bctx.narrowings[name]) for bctx in branches):
            declared = ctx.locals.get(name)
            if declared is not None and contains_nil(declared):
                ctx.narrowings[name] = remove_nil(declared)


# ============================================================
# CONTAINS RETURN (structural, no context)
# ============================================================


def _contains_return(stmts: list[TStmt]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, TReturnStmt):
            return True
        if isinstance(stmt, TIfStmt):
            if _contains_return(stmt.then_body):
                return True
            if stmt.else_body is not None and _contains_return(stmt.else_body):
                return True
        elif isinstance(stmt, (TWhileStmt, TForStmt)):
            if _contains_return(stmt.body):
                return True
        elif isinstance(stmt, TTryStmt):
            if _contains_return(stmt.body):
                return True
            for c in stmt.catches:
                if _contains_return(c.body):
                    return True
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                if _contains_return(case.body):
                    return True
            if stmt.default is not None and _contains_return(stmt.default.body):
                return True
    return False


# ============================================================
# NEEDS NAMED RETURNS (structural)
# ============================================================


def _check_needs_named_returns(stmts: list[TStmt]) -> bool:
    """Check if any try/catch contains returns anywhere inside."""
    for stmt in stmts:
        if isinstance(stmt, TTryStmt):
            if _contains_return(stmt.body):
                return True
            for c in stmt.catches:
                if _contains_return(c.body):
                    return True
        if isinstance(stmt, TIfStmt):
            if _check_needs_named_returns(stmt.then_body):
                return True
            if stmt.else_body is not None and _check_needs_named_returns(
                stmt.else_body
            ):
                return True
        elif isinstance(stmt, (TWhileStmt, TForStmt)):
            if _check_needs_named_returns(stmt.body):
                return True
        elif isinstance(stmt, TTryStmt):
            if _check_needs_named_returns(stmt.body):
                return True
            for c in stmt.catches:
                if _check_needs_named_returns(c.body):
                    return True
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                if _check_needs_named_returns(case.body):
                    return True
            if stmt.default is not None and _check_needs_named_returns(
                stmt.default.body
            ):
                return True
    return False


# ============================================================
# MAIN WALKER
# ============================================================


def _walk_block(stmts: list[TStmt], ctx: _ReturnsCtx) -> bool:
    """Walk a statement list. Stamps annotations, returns whether block always-returns."""
    for stmt in stmts:
        # Terminators
        if isinstance(stmt, TReturnStmt):
            if stmt.value is not None:
                ret_type = _resolve_expr_type(stmt.value, ctx)
                if ret_type is not None and contains_nil(ret_type):
                    ctx.fn_results.may_return_nil = True
            return True
        if isinstance(stmt, TThrowStmt):
            return True
        if _is_exit_call(stmt):
            return True

        # Compound statements
        if isinstance(stmt, TIfStmt):
            then_narrows, else_narrows = _extract_nil_narrowing(stmt.cond, ctx)
            then_ctx = _fork_ctx(ctx, then_narrows)
            then_returns = _walk_block(stmt.then_body, then_ctx)
            else_ctx = _fork_ctx(ctx, else_narrows)
            if stmt.else_body is not None:
                else_returns = _walk_block(stmt.else_body, else_ctx)
                combined = then_returns and else_returns
            else:
                else_returns = False
                combined = False
            stmt.annotations["returns.always_returns"] = combined
            if combined:
                return True
            survivors = []
            if not then_returns:
                survivors.append(then_ctx)
            if not else_returns:
                survivors.append(else_ctx)
            _merge_branch_narrowings(ctx, survivors)

        elif isinstance(stmt, TWhileStmt):
            _walk_block(stmt.body, ctx)
            stmt.annotations["returns.always_returns"] = False

        elif isinstance(stmt, TForStmt):
            _walk_block(stmt.body, ctx)
            stmt.annotations["returns.always_returns"] = False

        elif isinstance(stmt, TMatchStmt):
            all_return = True
            survivors: list[_ReturnsCtx] = []
            for case in stmt.cases:
                case_ctx = _fork_ctx(ctx)
                case_returns = _walk_block(case.body, case_ctx)
                case.annotations["returns.always_returns"] = case_returns
                if not case_returns:
                    all_return = False
                    survivors.append(case_ctx)
            if stmt.default is not None:
                dflt_ctx = _fork_ctx(ctx)
                dflt_returns = _walk_block(stmt.default.body, dflt_ctx)
                stmt.default.annotations["returns.always_returns"] = dflt_returns
                if not dflt_returns:
                    all_return = False
                    survivors.append(dflt_ctx)
            stmt.annotations["returns.always_returns"] = all_return
            if all_return:
                return True
            _merge_branch_narrowings(ctx, survivors)

        elif isinstance(stmt, TTryStmt):
            body_returns = _walk_block(stmt.body, ctx)
            all_catches_return = True
            for catch in stmt.catches:
                catch_returns = _walk_block(catch.body, ctx)
                catch.annotations["returns.always_returns"] = catch_returns
                if not catch_returns:
                    all_catches_return = False
            combined = body_returns and all_catches_return
            stmt.annotations["returns.always_returns"] = combined
            stmt.annotations["returns.body_has_return"] = _contains_return(stmt.body)
            if _contains_return(stmt.body) or any(
                _contains_return(c.body) for c in stmt.catches
            ):
                ctx.fn_results.needs_named_returns = True
            if combined:
                return True

        elif isinstance(stmt, TLetStmt):
            declared_type = ctx.checker.resolve_type(stmt.typ)
            ctx.locals[stmt.name] = declared_type

        elif isinstance(stmt, TAssignStmt):
            if isinstance(stmt.target, TVar):
                name = stmt.target.name
                declared = ctx.locals.get(name)
                if declared is not None and contains_nil(declared):
                    if _may_be_nil(stmt.value, ctx):
                        ctx.narrowings.pop(name, None)
                    else:
                        ctx.narrowings[name] = remove_nil(declared)

    return False


# ============================================================
# PER-FUNCTION ENTRY
# ============================================================


def _analyze_fn(decl: TFnDecl, checker: Checker, self_type: Type | None = None) -> None:
    fn_results = _FnResults()
    locals_: dict[str, Type] = {}
    for p in decl.params:
        if p.typ is not None:
            locals_[p.name] = checker.resolve_type(p.typ)
        elif p.name == "self" and self_type is not None:
            locals_[p.name] = self_type
    ctx = _ReturnsCtx(
        checker=checker,
        locals=locals_,
        fn_results=fn_results,
    )
    always = _walk_block(decl.body, ctx)
    decl.annotations["returns.always_returns"] = always
    decl.annotations["returns.needs_named_returns"] = fn_results.needs_named_returns
    decl.annotations["returns.may_return_nil"] = fn_results.may_return_nil


# ============================================================
# PUBLIC API
# ============================================================


def analyze_returns(module: TModule, checker: Checker) -> None:
    """Run returns analysis on all functions in the module."""
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            _analyze_fn(decl, checker)
        elif isinstance(decl, TStructDecl):
            st = checker.types.get(decl.name)
            for method in decl.methods:
                _analyze_fn(method, checker, self_type=st)


def contains_return(stmts: list[TStmt]) -> bool:
    """Public utility: check if statement list contains any return (recursive)."""
    return _contains_return(stmts)
