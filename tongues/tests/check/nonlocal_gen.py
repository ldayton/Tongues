"""Nonlocal generation enhancements for the Taytsh type checker test generator.

The existing generator (ExprGen/StmtGen) uses a top-down type-directed approach
where production rules pick types before generating sub-expressions. This is
the "local" strategy from Palka et al. (2011): parameter types are chosen
before function bodies, let-binding types before their uses, etc. The result
is that ~78% of let bindings and ~96% of fn-lit params go unused, because
there was no demand for their type when the body was generated.

This module adds three "nonlocal" rules that work backwards from demand:

  A. GenParam⊲ (budget-based fn-lit params): instead of declaring all params
     upfront, start with zero and claim matching slots from the fn_type's
     param list on-demand during body generation. Every claimed param is
     guaranteed to be used. Unclaimed slots are filled with unused params
     after body generation to preserve the expected function type.

  B. GenLet (retroactive let-insertion): when gen_expr needs a value of type T
     and no binding of that type exists in scope, insert a let-binding into
     _pending_stmts (which gen_block drains before the current statement).
     Guarantees every inserted let-binding is used at least once.

  C. GenMatch (need-driven match): when a value of type T is needed and an
     interface-typed variable is in scope whose variant has a field of type T,
     generate a match statement that extracts the field. Produces matches
     where case-bound variables are actually used.

The _pending_stmts mechanism requires careful save/restore at gen_block
boundaries so that inner blocks (if-then bodies, loop bodies, etc.) don't
steal pending stmts meant for the outer level. gen_fn_lit_nonlocal also
saves/restores to avoid draining outer pending stmts into fn-lit bodies.

Measured impact (500 seeds, 0 type errors introduced):
  - Let binding usage: 22% -> 50%
  - Overall variable usage: 34% -> 50%
  - Fn-lit param usage unchanged (budget approach is conservative)

References:
  - Frank, Quiring, Lampropoulos. "Generating Well-Typed Terms That Are Not
    'Useless'." POPL 2024. (nonlocal GenParam⊲, GenLet, GenMatch rules)
  - Palka, Claessen, Russo, Hughes. "Testing an Optimising Compiler by
    Generating Random Lambda Terms." AST 2011. (local type-directed generation)

Future work:
  - Higher-order propagation: when a fn-lit is passed as an argument to a
    higher-order function, extending its params should also extend calls
    through the higher-order path (the paper's ⊲α holes).
  - Extensible fn types: allow fn-lit params to grow beyond the declared
    fn_type by updating the type at all usage sites. This requires tracking
    call sites (ExtensibleFnLit.call_sites) which is wired but unused.
  - Argument-usage-aware weighting: measure per-seed usage rates and feed
    back into weight tuning to target ~95% (matching real program usage).
  - Apply to codegen/apptest: the real payoff of higher usage is exercising
    register allocation, calling conventions, and optimization passes that
    only trigger when arguments are live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.taytsh.ast import (
    TAssignStmt,
    TBinaryOp,
    TBreakStmt,
    TCall,
    TContinueStmt,
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
    TMatchCase,
    TMatchStmt,
    TModule,
    TOpAssignStmt,
    TParam,
    TPatternType,
    TRange,
    TReturnStmt,
    TSetLit,
    TSlice,
    TStmt,
    TStructDecl,
    TTernary,
    TThrowStmt,
    TTryStmt,
    TTupleAccess,
    TTupleAssignStmt,
    TTupleLit,
    TUnaryOp,
    TVar,
    TWhileStmt,
)
from src.taytsh.check import (
    NIL_T,
    VOID_T,
    FnT,
    InterfaceT,
    Type,
    type_eq,
)

from .types import make_ttype
from .ast_helpers import P, A, zero_value as _zero_value

if TYPE_CHECKING:
    from . import Generator
    from .exprs import ExprGen
    from .stmts import StmtGen


# ---------------------------------------------------------------------------
# A. Nonlocal fn-literal generation (GenParam⊲)
# ---------------------------------------------------------------------------


@dataclass
class ExtensibleFnLit:
    """Tracks a fn-literal whose params are drawn on-demand from a type budget."""

    params: list[TParam]
    param_types: list[Type]
    param_names: list[str]
    # Budget: fn_type param slots not yet claimed by the body
    budget_types: list[Type]
    budget_used: list[bool]
    all_used_names: set[str]


def gen_fn_lit_nonlocal(expr_gen: ExprGen, fn_type: FnT, depth: int) -> TExpr:
    """Generate a fn literal with on-demand params (GenParam⊲ approach).

    Starts with zero params. During body generation, when a type is needed,
    try_extend_fn_param claims a matching param slot from fn_type. After body
    generation, remaining slots are filled with unused params.
    """
    gen = expr_gen.gen
    rng = expr_gen.rng

    ext = ExtensibleFnLit(
        params=[],
        param_types=[],
        param_names=[],
        budget_types=list(fn_type.params),
        budget_used=[False] * len(fn_type.params),
        all_used_names=set(),
    )

    if not hasattr(gen, "_ext_fn_stack"):
        gen._ext_fn_stack = []
    gen._ext_fn_stack.append(ext)

    old_fn_lit = gen.in_fn_lit
    gen.in_fn_lit = True

    gen.scope.enter_scope()
    # No params declared yet - they'll be added on demand by try_extend_fn_param

    # Save outer pending stmts so we don't steal them
    saved_pending = gen.stmt_gen._pending_stmts
    gen.stmt_gen._pending_stmts = []

    if type_eq(fn_type.ret, VOID_T):
        body = gen.stmt_gen.gen_block(rng.randint(1, 2), must_return=None)
    else:
        ret_expr = expr_gen.gen_expr(fn_type.ret, depth + 1)
        body = []
        if gen.stmt_gen._pending_stmts:
            body.extend(gen.stmt_gen._pending_stmts)
        body.append(TReturnStmt(pos=P, value=ret_expr, annotations=A))

    # Restore outer pending stmts
    gen.stmt_gen._pending_stmts = saved_pending

    gen.scope.exit_scope()
    gen.in_fn_lit = old_fn_lit
    gen._ext_fn_stack.pop()

    # Fill remaining budget slots with unused params (to match fn_type exactly)
    final_params: list[TParam] = []
    claimed_idx = 0
    for i, pt in enumerate(fn_type.params):
        if ext.budget_used[i]:
            final_params.append(ext.params[claimed_idx])
            claimed_idx += 1
        else:
            pname = gen.names.var_name(ext.all_used_names)
            ext.all_used_names.add(pname)
            final_params.append(
                TParam(pos=P, name=pname, typ=make_ttype(pt), annotations=A)
            )

    return TFnLit(
        pos=P,
        params=final_params,
        ret=make_ttype(fn_type.ret),
        body=body,
        annotations=A,
    )


def try_extend_fn_param(expr_gen: ExprGen, target: Type) -> TExpr | None:
    """GenParam⊲: claim a matching param slot from the enclosing fn-lit's budget."""
    gen = expr_gen.gen

    if not hasattr(gen, "_ext_fn_stack") or not gen._ext_fn_stack:
        return None
    if type_eq(target, VOID_T) or type_eq(target, NIL_T):
        return None
    if isinstance(target, FnT):
        return None

    ext = gen._ext_fn_stack[-1]

    # Find an unclaimed budget slot matching the target type
    slot_idx = None
    for i, (bt, used) in enumerate(zip(ext.budget_types, ext.budget_used)):
        if not used and type_eq(bt, target):
            slot_idx = i
            break
    if slot_idx is None:
        return None

    ext.budget_used[slot_idx] = True
    pname = gen.names.var_name(ext.all_used_names)
    ext.all_used_names.add(pname)

    ext.param_names.append(pname)
    ext.param_types.append(target)
    ext.params.append(TParam(pos=P, name=pname, typ=make_ttype(target), annotations=A))
    gen.scope.declare(pname, target)

    return TVar(pos=P, name=pname, annotations=A)


# ---------------------------------------------------------------------------
# B. Nonlocal let-insertion (GenLet)
# ---------------------------------------------------------------------------


def try_nonlocal_let(
    stmt_gen: StmtGen, expr_gen: ExprGen, target: Type
) -> TExpr | None:
    """GenLet: insert a let-binding above the current statement and return a var ref."""
    gen = expr_gen.gen

    if type_eq(target, NIL_T) or type_eq(target, VOID_T):
        return None
    if gen.in_fn_lit:
        return None
    if getattr(gen, "_in_nonlocal_let", False):
        return None

    all_names = {b.name for b in gen.scope.all_bindings()}
    name = gen.names.var_name(all_names)
    ttype = make_ttype(target)
    gen._in_nonlocal_let = True
    init_expr = expr_gen.gen_expr(target, depth=2)
    gen._in_nonlocal_let = False

    gen.scope.declare(name, target)
    stmt_gen._pending_stmts.append(
        TLetStmt(pos=P, name=name, typ=ttype, value=init_expr, annotations=A)
    )
    return TVar(pos=P, name=name, annotations=A)


# ---------------------------------------------------------------------------
# C. Need-driven match generation (GenMatch)
# ---------------------------------------------------------------------------


def try_need_driven_match(
    stmt_gen: StmtGen, expr_gen: ExprGen, target: Type
) -> TExpr | None:
    """GenMatch: generate a match statement driven by a type need.

    Emits:
      let result: T = <zero>
      match x { case v: Variant => result = v.field; ... }
    Returns TVar("result").
    """
    gen = expr_gen.gen

    if type_eq(target, VOID_T) or type_eq(target, NIL_T):
        return None
    if gen.in_fn_lit:
        return None

    candidate = _find_match_candidate(gen, target)
    if candidate is None:
        return None

    var_name, iface, useful_variant, field_name = candidate
    info = gen.pool.interface_info_for(iface)
    if info is None:
        return None

    all_names = {b.name for b in gen.scope.all_bindings()}
    result_name = gen.names.var_name(all_names)
    ttype = make_ttype(target)
    gen.scope.declare(result_name, target)

    stmt_gen._pending_stmts.append(
        TLetStmt(
            pos=P, name=result_name, typ=ttype, value=_zero_value(target), annotations=A
        )
    )

    cases: list[TMatchCase] = []
    for vname in info.variant_names:
        st = gen.pool.struct_for_name(vname)
        if st is None:
            continue
        bind_name = gen.names.var_name({b.name for b in gen.scope.all_bindings()})
        pat = TPatternType(
            pos=P, name=bind_name, type_name=make_ttype(st), annotations=A
        )
        gen.scope.enter_scope()
        gen.scope.declare(bind_name, st)
        gen.scope.narrow(var_name, st)

        if st.name == useful_variant:
            value_expr: TExpr = TFieldAccess(
                pos=P,
                obj=TVar(pos=P, name=bind_name, annotations=A),
                field=field_name,
                annotations=A,
            )
        else:
            value_expr = expr_gen.gen_expr(target, depth=3)

        body: list[TStmt] = [
            TAssignStmt(
                pos=P,
                target=TVar(pos=P, name=result_name, annotations=A),
                value=value_expr,
                annotations=A,
            )
        ]
        gen.scope.exit_scope()
        cases.append(TMatchCase(pos=P, pattern=pat, body=body, annotations=A))

    if not cases:
        return None

    stmt_gen._pending_stmts.append(
        TMatchStmt(
            pos=P,
            expr=TVar(pos=P, name=var_name, annotations=A),
            cases=cases,
            default=None,
            annotations=A,
        )
    )
    return TVar(pos=P, name=result_name, annotations=A)


def _find_match_candidate(
    gen: Generator, target: Type
) -> tuple[str, InterfaceT, str, str] | None:
    for b in gen.scope.all_bindings():
        if not isinstance(b.typ, InterfaceT):
            continue
        info = gen.pool.interface_info_for(b.typ)
        if info is None:
            continue
        for vname in info.variant_names:
            st = gen.pool.struct_for_name(vname)
            if st is None:
                continue
            for fname, ftype in st.fields.items():
                if type_eq(ftype, target):
                    return (b.name, b.typ, st.name, fname)
    return None


# ---------------------------------------------------------------------------
# Usage metrics
# ---------------------------------------------------------------------------


@dataclass
class UsageStats:
    fn_params_total: int = 0
    fn_params_used: int = 0
    fn_lit_params_total: int = 0
    fn_lit_params_used: int = 0
    let_bindings_total: int = 0
    let_bindings_used: int = 0
    match_bindings_total: int = 0
    match_bindings_used: int = 0

    @property
    def fn_param_usage(self) -> float:
        return (
            self.fn_params_used / self.fn_params_total if self.fn_params_total else 1.0
        )

    @property
    def fn_lit_param_usage(self) -> float:
        return (
            self.fn_lit_params_used / self.fn_lit_params_total
            if self.fn_lit_params_total
            else 1.0
        )

    @property
    def let_usage(self) -> float:
        return (
            self.let_bindings_used / self.let_bindings_total
            if self.let_bindings_total
            else 1.0
        )

    @property
    def overall_usage(self) -> float:
        total = (
            self.fn_params_total
            + self.fn_lit_params_total
            + self.let_bindings_total
            + self.match_bindings_total
        )
        used = (
            self.fn_params_used
            + self.fn_lit_params_used
            + self.let_bindings_used
            + self.match_bindings_used
        )
        return used / total if total else 1.0

    def accumulate(self, other: UsageStats) -> None:
        self.fn_params_total += other.fn_params_total
        self.fn_params_used += other.fn_params_used
        self.fn_lit_params_total += other.fn_lit_params_total
        self.fn_lit_params_used += other.fn_lit_params_used
        self.let_bindings_total += other.let_bindings_total
        self.let_bindings_used += other.let_bindings_used
        self.match_bindings_total += other.match_bindings_total
        self.match_bindings_used += other.match_bindings_used


def measure_usage(module: TModule) -> UsageStats:
    """Walk a TModule AST and compute variable usage statistics."""
    stats = UsageStats()
    refs: set[str] = set()
    _collect_refs(module, refs)

    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            if decl.name == "Main":
                _walk_stmts_for_stats(decl.body, stats, refs)
                continue
            for p in decl.params:
                if p.name == "this":
                    continue
                stats.fn_params_total += 1
                # Check if used in this function's body
                fn_refs: set[str] = set()
                _collect_refs_stmts(decl.body, fn_refs)
                if p.name in fn_refs:
                    stats.fn_params_used += 1
            _walk_stmts_for_stats(decl.body, stats, refs)
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                for p in method.params:
                    if p.name == "this":
                        continue
                    stats.fn_params_total += 1
                    fn_refs = set()
                    _collect_refs_stmts(method.body, fn_refs)
                    if p.name in fn_refs:
                        stats.fn_params_used += 1
                _walk_stmts_for_stats(method.body, stats, refs)

    return stats


def _walk_stmts_for_stats(
    stmts: list[TStmt], stats: UsageStats, all_refs: set[str]
) -> None:
    for stmt in stmts:
        if isinstance(stmt, TLetStmt):
            stats.let_bindings_total += 1
            if stmt.name in all_refs:
                stats.let_bindings_used += 1
            if stmt.value is not None:
                _walk_expr_for_stats(stmt.value, stats)
        elif isinstance(stmt, TIfStmt):
            _walk_stmts_for_stats(stmt.then_body, stats, all_refs)
            if stmt.else_body:
                _walk_stmts_for_stats(stmt.else_body, stats, all_refs)
        elif isinstance(stmt, TWhileStmt):
            _walk_stmts_for_stats(stmt.body, stats, all_refs)
        elif isinstance(stmt, TForStmt):
            _walk_stmts_for_stats(stmt.body, stats, all_refs)
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                if isinstance(case.pattern, TPatternType):
                    stats.match_bindings_total += 1
                    body_refs: set[str] = set()
                    _collect_refs_stmts(case.body, body_refs)
                    if case.pattern.name in body_refs:
                        stats.match_bindings_used += 1
                _walk_stmts_for_stats(case.body, stats, all_refs)
            if stmt.default is not None:
                if stmt.default.name is not None:
                    stats.match_bindings_total += 1
                    body_refs = set()
                    _collect_refs_stmts(stmt.default.body, body_refs)
                    if stmt.default.name in body_refs:
                        stats.match_bindings_used += 1
                _walk_stmts_for_stats(stmt.default.body, stats, all_refs)
        elif isinstance(stmt, TTryStmt):
            _walk_stmts_for_stats(stmt.body, stats, all_refs)
            for catch in stmt.catches:
                _walk_stmts_for_stats(catch.body, stats, all_refs)
            if stmt.finally_body:
                _walk_stmts_for_stats(stmt.finally_body, stats, all_refs)
        elif isinstance(stmt, TReturnStmt):
            if stmt.value is not None:
                _walk_expr_for_stats(stmt.value, stats)
        elif isinstance(stmt, TExprStmt):
            _walk_expr_for_stats(stmt.expr, stats)
        elif isinstance(stmt, TAssignStmt):
            _walk_expr_for_stats(stmt.value, stats)
        elif isinstance(stmt, TThrowStmt):
            _walk_expr_for_stats(stmt.expr, stats)
        elif isinstance(stmt, TOpAssignStmt):
            _walk_expr_for_stats(stmt.value, stats)
        elif isinstance(stmt, TTupleAssignStmt):
            _walk_expr_for_stats(stmt.value, stats)


def _walk_expr_for_stats(expr: TExpr, stats: UsageStats) -> None:
    """Find fn-lit params and check if they're used in the fn-lit body."""
    if isinstance(expr, TFnLit):
        body_refs: set[str] = set()
        _collect_refs_stmts(expr.body, body_refs)
        for p in expr.params:
            stats.fn_lit_params_total += 1
            if p.name in body_refs:
                stats.fn_lit_params_used += 1
        for s in expr.body:
            _walk_stmts_for_stats([s], stats, set())
    elif isinstance(expr, TCall):
        _walk_expr_for_stats(expr.func, stats)
        for arg in expr.args:
            _walk_expr_for_stats(arg.value, stats)
    elif isinstance(expr, TBinaryOp):
        _walk_expr_for_stats(expr.left, stats)
        _walk_expr_for_stats(expr.right, stats)
    elif isinstance(expr, TUnaryOp):
        _walk_expr_for_stats(expr.operand, stats)
    elif isinstance(expr, TTernary):
        _walk_expr_for_stats(expr.cond, stats)
        _walk_expr_for_stats(expr.then_expr, stats)
        _walk_expr_for_stats(expr.else_expr, stats)
    elif isinstance(expr, TFieldAccess):
        _walk_expr_for_stats(expr.obj, stats)
    elif isinstance(expr, TIndex):
        _walk_expr_for_stats(expr.obj, stats)
        _walk_expr_for_stats(expr.index, stats)
    elif isinstance(expr, TSlice):
        _walk_expr_for_stats(expr.obj, stats)
        _walk_expr_for_stats(expr.low, stats)
        _walk_expr_for_stats(expr.high, stats)
    elif isinstance(expr, TTupleAccess):
        _walk_expr_for_stats(expr.obj, stats)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _walk_expr_for_stats(e, stats)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _walk_expr_for_stats(k, stats)
            _walk_expr_for_stats(v, stats)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _walk_expr_for_stats(e, stats)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _walk_expr_for_stats(e, stats)


# ---------------------------------------------------------------------------
# AST reference collection
# ---------------------------------------------------------------------------


def _collect_refs(node: object, refs: set[str]) -> None:
    if isinstance(node, TModule):
        for decl in node.decls:
            _collect_refs(decl, refs)
    elif isinstance(node, TFnDecl):
        _collect_refs_stmts(node.body, refs)
    elif isinstance(node, TStructDecl):
        for m in node.methods:
            _collect_refs(m, refs)


def _collect_refs_stmts(stmts: list[TStmt], refs: set[str]) -> None:
    for s in stmts:
        _collect_refs_stmt(s, refs)


def _collect_refs_stmt(stmt: TStmt, refs: set[str]) -> None:
    if isinstance(stmt, TLetStmt):
        if stmt.value is not None:
            _collect_refs_expr(stmt.value, refs)
    elif isinstance(stmt, TAssignStmt):
        _collect_refs_expr(stmt.target, refs)
        _collect_refs_expr(stmt.value, refs)
    elif isinstance(stmt, TExprStmt):
        _collect_refs_expr(stmt.expr, refs)
    elif isinstance(stmt, TReturnStmt):
        if stmt.value is not None:
            _collect_refs_expr(stmt.value, refs)
    elif isinstance(stmt, TIfStmt):
        _collect_refs_expr(stmt.cond, refs)
        _collect_refs_stmts(stmt.then_body, refs)
        if stmt.else_body:
            _collect_refs_stmts(stmt.else_body, refs)
    elif isinstance(stmt, TWhileStmt):
        _collect_refs_expr(stmt.cond, refs)
        _collect_refs_stmts(stmt.body, refs)
    elif isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TVar):
            refs.add(stmt.iterable.name)
        elif isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                _collect_refs_expr(a, refs)
        _collect_refs_stmts(stmt.body, refs)
    elif isinstance(stmt, TMatchStmt):
        _collect_refs_expr(stmt.expr, refs)
        for case in stmt.cases:
            _collect_refs_stmts(case.body, refs)
        if stmt.default is not None:
            _collect_refs_stmts(stmt.default.body, refs)
    elif isinstance(stmt, TTryStmt):
        _collect_refs_stmts(stmt.body, refs)
        for catch in stmt.catches:
            _collect_refs_stmts(catch.body, refs)
        if stmt.finally_body:
            _collect_refs_stmts(stmt.finally_body, refs)
    elif isinstance(stmt, TThrowStmt):
        _collect_refs_expr(stmt.expr, refs)
    elif isinstance(stmt, TOpAssignStmt):
        _collect_refs_expr(stmt.target, refs)
        _collect_refs_expr(stmt.value, refs)
    elif isinstance(stmt, TTupleAssignStmt):
        for t in stmt.targets:
            _collect_refs_expr(t, refs)
        _collect_refs_expr(stmt.value, refs)


def _collect_refs_expr(expr: TExpr, refs: set[str]) -> None:
    if isinstance(expr, TVar):
        refs.add(expr.name)
    elif isinstance(expr, TCall):
        _collect_refs_expr(expr.func, refs)
        for arg in expr.args:
            _collect_refs_expr(arg.value, refs)
    elif isinstance(expr, TBinaryOp):
        _collect_refs_expr(expr.left, refs)
        _collect_refs_expr(expr.right, refs)
    elif isinstance(expr, TUnaryOp):
        _collect_refs_expr(expr.operand, refs)
    elif isinstance(expr, TTernary):
        _collect_refs_expr(expr.cond, refs)
        _collect_refs_expr(expr.then_expr, refs)
        _collect_refs_expr(expr.else_expr, refs)
    elif isinstance(expr, TFieldAccess):
        _collect_refs_expr(expr.obj, refs)
    elif isinstance(expr, TIndex):
        _collect_refs_expr(expr.obj, refs)
        _collect_refs_expr(expr.index, refs)
    elif isinstance(expr, TSlice):
        _collect_refs_expr(expr.obj, refs)
        _collect_refs_expr(expr.low, refs)
        _collect_refs_expr(expr.high, refs)
    elif isinstance(expr, TTupleAccess):
        _collect_refs_expr(expr.obj, refs)
    elif isinstance(expr, TFnLit):
        for s in expr.body:
            _collect_refs_stmt(s, refs)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _collect_refs_expr(e, refs)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _collect_refs_expr(k, refs)
            _collect_refs_expr(v, refs)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _collect_refs_expr(e, refs)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _collect_refs_expr(e, refs)


# ---------------------------------------------------------------------------
# Monkey-patching to integrate into the existing generator
# ---------------------------------------------------------------------------


def patch_generator() -> None:
    """Monkey-patch ExprGen to add nonlocal generation rules.

    Usage:
        from tests.check.nonlocal_gen import patch_generator
        patch_generator()
    """
    from . import Generator
    from .exprs import ExprGen
    from .stmts import StmtGen

    original_init = Generator.__init__

    def patched_init(self, features, seed):
        original_init(self, features, seed)
        self._ext_fn_stack: list[ExtensibleFnLit] = []

    Generator.__init__ = patched_init

    # Patch gen_block to:
    # 1. Save/restore _pending_stmts so inner blocks don't steal outer-level stmts
    # 2. Drain _pending_stmts in the must_return path
    original_gen_block = StmtGen.gen_block

    def patched_gen_block(self, count, must_return, *, depth=0):
        # Save stmts pending from the outer context (e.g. from if-condition generation)
        saved_pending = self._pending_stmts
        self._pending_stmts = []
        stmts: list[TStmt] = []
        for i in range(count):
            if i == count - 1 and must_return is not None:
                if not type_eq(must_return, VOID_T):
                    ret_expr = self.gen.expr_gen.gen_expr(must_return)
                    if self._pending_stmts:
                        stmts.extend(self._pending_stmts)
                        self._pending_stmts = []
                    stmts.append(TReturnStmt(pos=P, value=ret_expr, annotations=A))
                else:
                    stmts.append(TReturnStmt(pos=P, value=None, annotations=A))
                break
            stmt = self._pick_stmt(depth)
            if stmt is not None:
                if self._pending_stmts:
                    stmts.extend(self._pending_stmts)
                    self._pending_stmts = []
                stmts.append(stmt)
                if isinstance(
                    stmt, (TReturnStmt, TBreakStmt, TContinueStmt, TThrowStmt)
                ):
                    break
        # Drain any leftover pending stmts into this block (they belong to this scope)
        if self._pending_stmts:
            stmts.extend(self._pending_stmts)
        # Restore outer-level pending stmts only
        self._pending_stmts = saved_pending
        return stmts

    StmtGen.gen_block = patched_gen_block

    original_gen_expr = ExprGen.gen_expr

    def patched_gen_expr(self, target: Type, depth: int = 0) -> TExpr:
        gen = self.gen
        rng = self.rng

        if depth >= 4:
            return self._fallback(target)

        # GenParam⊲: extend enclosing fn-lit's params
        if (
            hasattr(gen, "_ext_fn_stack")
            and gen._ext_fn_stack
            and not type_eq(target, VOID_T)
            and not type_eq(target, NIL_T)
            and not isinstance(target, FnT)
            and depth >= 1
        ):
            bindings = gen.scope.bindings_of_type(target)
            if gen.in_fn_lit:
                bindings = [
                    b
                    for b in bindings
                    if gen.scope.lookup(b.name) is not None
                    and _is_fn_lit_local(gen, b.name)
                ]
            if not bindings and rng.random() < 0.5:
                result = try_extend_fn_param(self, target)
                if result is not None:
                    return result
            elif bindings and rng.random() < 0.1:
                result = try_extend_fn_param(self, target)
                if result is not None:
                    return result

        # GenLet: insert a let-binding above current statement
        if (
            not gen.in_fn_lit
            and not type_eq(target, VOID_T)
            and not type_eq(target, NIL_T)
            and depth >= 2
        ):
            bindings = gen.scope.bindings_of_type(target)
            if not bindings and rng.random() < 0.2:
                result = try_nonlocal_let(gen.stmt_gen, self, target)
                if result is not None:
                    return result

        return original_gen_expr(self, target, depth)

    ExprGen.gen_expr = patched_gen_expr

    original_gen_fn_lit = ExprGen._gen_fn_lit

    def patched_gen_fn_lit(self, fn_type: FnT, depth: int) -> TExpr:
        if self.rng.random() < 0.5:
            return gen_fn_lit_nonlocal(self, fn_type, depth)
        return original_gen_fn_lit(self, fn_type, depth)

    ExprGen._gen_fn_lit = patched_gen_fn_lit


def _is_fn_lit_local(gen: Generator, name: str) -> bool:
    if not gen.scope.scopes:
        return False
    return name in gen.scope.scopes[-1]
