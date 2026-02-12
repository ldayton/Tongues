"""Liveness analysis pass for Taytsh IR.

Analyzes each function body independently, writing annotations onto AST nodes
for dead stores (initial_value_unused), unused catch/match bindings, and
unused tuple assignment targets.
"""

from __future__ import annotations

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
from ..taytsh.check import Checker


# ============================================================
# EXPRESSION READS
# ============================================================


def _expr_reads(name: str, expr: TExpr) -> bool:
    """Check if expression reads the variable."""
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
    # Literals and TFnLit (can't capture) don't read outer vars
    return False


# ============================================================
# TARGET READS
# ============================================================


def _target_reads(name: str, target: TExpr) -> bool:
    """Check if assignment target reads the variable (index/field chains)."""
    if isinstance(target, TVar):
        return False
    if isinstance(target, TIndex):
        return _expr_reads(name, target.obj) or _expr_reads(name, target.index)
    if isinstance(target, TFieldAccess):
        return _expr_reads(name, target.obj)
    if isinstance(target, TTupleAccess):
        return _expr_reads(name, target.obj)
    return False


# ============================================================
# FIRST ACCESS TYPE
# ============================================================


def _first_access_in_stmts(name: str, stmts: list[TStmt]) -> str | None:
    """Find first access type in a list of statements."""
    for stmt in stmts:
        result = _first_access_type(name, stmt)
        if result is not None:
            return result
    return None


def _first_access_type(name: str, stmt: TStmt) -> str | None:
    """Determine if first access to `name` in stmt is 'read', 'write', or None."""
    if isinstance(stmt, TLetStmt):
        if stmt.value is not None and _expr_reads(name, stmt.value):
            return "read"
        return None
    if isinstance(stmt, TAssignStmt):
        if _expr_reads(name, stmt.value):
            return "read"
        if isinstance(stmt.target, TVar) and stmt.target.name == name:
            return "write"
        if _target_reads(name, stmt.target):
            return "read"
        return None
    if isinstance(stmt, TOpAssignStmt):
        # OpAssign reads before writing (x += 1 reads x)
        if isinstance(stmt.target, TVar) and stmt.target.name == name:
            return "read"
        if _expr_reads(name, stmt.value):
            return "read"
        if _target_reads(name, stmt.target):
            return "read"
        return None
    if isinstance(stmt, TTupleAssignStmt):
        if _expr_reads(name, stmt.value):
            return "read"
        for target in stmt.targets:
            if isinstance(target, TVar) and target.name == name:
                return "write"
        return None
    if isinstance(stmt, TExprStmt):
        if _expr_reads(name, stmt.expr):
            return "read"
        return None
    if isinstance(stmt, TReturnStmt):
        if stmt.value is not None and _expr_reads(name, stmt.value):
            return "read"
        return None
    if isinstance(stmt, TThrowStmt):
        if _expr_reads(name, stmt.expr):
            return "read"
        return None
    if isinstance(stmt, TIfStmt):
        if _expr_reads(name, stmt.cond):
            return "read"
        then_result = _first_access_in_stmts(name, stmt.then_body)
        else_result = (
            _first_access_in_stmts(name, stmt.else_body)
            if stmt.else_body is not None
            else None
        )
        if then_result == "read" or else_result == "read":
            return "read"
        if then_result == "write" and else_result == "write":
            return "write"
        return None
    if isinstance(stmt, TWhileStmt):
        if _expr_reads(name, stmt.cond):
            return "read"
        result = _first_access_in_stmts(name, stmt.body)
        if result == "read":
            return "read"
        return None
    if isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                if _expr_reads(name, a):
                    return "read"
        elif _expr_reads(name, stmt.iterable):
            return "read"
        result = _first_access_in_stmts(name, stmt.body)
        if result == "read":
            return "read"
        return None
    if isinstance(stmt, TTryStmt):
        try_result = _first_access_in_stmts(name, stmt.body)
        if try_result == "read":
            return "read"
        catch_results = []
        for catch in stmt.catches:
            catch_result = _first_access_in_stmts(name, catch.body)
            if catch_result == "read":
                return "read"
            catch_results.append(catch_result)
        if try_result == "write" and (
            not catch_results or all(r == "write" for r in catch_results)
        ):
            return "write"
        if stmt.finally_body is not None:
            finally_result = _first_access_in_stmts(name, stmt.finally_body)
            if finally_result is not None:
                return finally_result
        return None
    if isinstance(stmt, TMatchStmt):
        if _expr_reads(name, stmt.expr):
            return "read"
        results = []
        for case in stmt.cases:
            case_result = _first_access_in_stmts(name, case.body)
            if case_result == "read":
                return "read"
            results.append(case_result)
        if stmt.default is not None:
            dflt_result = _first_access_in_stmts(name, stmt.default.body)
            if dflt_result == "read":
                return "read"
            results.append(dflt_result)
        if results and all(r == "write" for r in results):
            return "write"
        return None
    return None


# ============================================================
# WRITTEN-BEFORE-READ CHECK
# ============================================================


def _is_written_before_read(name: str, stmts: list[TStmt]) -> bool:
    """Check if variable is assigned before any read in statement sequence."""
    for stmt in stmts:
        result = _first_access_type(name, stmt)
        if result == "read":
            return False
        if result == "write":
            return True
    return False


# ============================================================
# INITIAL VALUE ANALYSIS
# ============================================================


def _analyze_initial_value_in_stmts(stmts: list[TStmt]) -> None:
    """Analyze statements for TLetStmt with unused initial values."""
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, TLetStmt):
            stmt.annotations["liveness.initial_value_unused"] = _is_written_before_read(
                stmt.name, stmts[i + 1 :]
            )
        if isinstance(stmt, TIfStmt):
            _analyze_initial_value_in_stmts(stmt.then_body)
            if stmt.else_body is not None:
                _analyze_initial_value_in_stmts(stmt.else_body)
        elif isinstance(stmt, TWhileStmt):
            _analyze_initial_value_in_stmts(stmt.body)
        elif isinstance(stmt, TForStmt):
            _analyze_initial_value_in_stmts(stmt.body)
        elif isinstance(stmt, TTryStmt):
            _analyze_initial_value_in_stmts(stmt.body)
            for catch in stmt.catches:
                _analyze_initial_value_in_stmts(catch.body)
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                _analyze_initial_value_in_stmts(case.body)
            if stmt.default is not None:
                _analyze_initial_value_in_stmts(stmt.default.body)


# ============================================================
# CATCH / MATCH UNUSED BINDING ANALYSIS
# ============================================================


def _analyze_catch_and_match_bindings(stmts: list[TStmt]) -> None:
    """Analyze catch and match bindings for unused variables."""
    for stmt in stmts:
        if isinstance(stmt, TTryStmt):
            for catch in stmt.catches:
                catch.annotations[
                    "liveness.catch_var_unused"
                ] = not _binding_used_in_stmts(catch.name, catch.body)
            _analyze_catch_and_match_bindings(stmt.body)
            for catch in stmt.catches:
                _analyze_catch_and_match_bindings(catch.body)
            if stmt.finally_body is not None:
                _analyze_catch_and_match_bindings(stmt.finally_body)
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                pat = case.pattern
                if isinstance(pat, TPatternType):
                    pat.annotations[
                        "liveness.match_var_unused"
                    ] = not _binding_used_in_stmts(pat.name, case.body)
                _analyze_catch_and_match_bindings(case.body)
            if stmt.default is not None:
                if stmt.default.name is not None:
                    stmt.default.annotations[
                        "liveness.match_var_unused"
                    ] = not _binding_used_in_stmts(stmt.default.name, stmt.default.body)
                _analyze_catch_and_match_bindings(stmt.default.body)
        elif isinstance(stmt, TIfStmt):
            _analyze_catch_and_match_bindings(stmt.then_body)
            if stmt.else_body is not None:
                _analyze_catch_and_match_bindings(stmt.else_body)
        elif isinstance(stmt, TWhileStmt):
            _analyze_catch_and_match_bindings(stmt.body)
        elif isinstance(stmt, TForStmt):
            _analyze_catch_and_match_bindings(stmt.body)


def _binding_read_in_expr(name: str, expr: TExpr, shadowed: bool) -> bool:
    """Return True if expr reads `name` from the tracked outer binding."""
    if isinstance(expr, TVar):
        return not shadowed and expr.name == name
    if isinstance(expr, TBinaryOp):
        return _binding_read_in_expr(
            name, expr.left, shadowed
        ) or _binding_read_in_expr(name, expr.right, shadowed)
    if isinstance(expr, TUnaryOp):
        return _binding_read_in_expr(name, expr.operand, shadowed)
    if isinstance(expr, TTernary):
        return (
            _binding_read_in_expr(name, expr.cond, shadowed)
            or _binding_read_in_expr(name, expr.then_expr, shadowed)
            or _binding_read_in_expr(name, expr.else_expr, shadowed)
        )
    if isinstance(expr, TFieldAccess):
        return _binding_read_in_expr(name, expr.obj, shadowed)
    if isinstance(expr, TTupleAccess):
        return _binding_read_in_expr(name, expr.obj, shadowed)
    if isinstance(expr, TIndex):
        return _binding_read_in_expr(name, expr.obj, shadowed) or _binding_read_in_expr(
            name, expr.index, shadowed
        )
    if isinstance(expr, TSlice):
        return (
            _binding_read_in_expr(name, expr.obj, shadowed)
            or _binding_read_in_expr(name, expr.low, shadowed)
            or _binding_read_in_expr(name, expr.high, shadowed)
        )
    if isinstance(expr, TCall):
        if _binding_read_in_expr(name, expr.func, shadowed):
            return True
        return any(_binding_read_in_expr(name, a.value, shadowed) for a in expr.args)
    if isinstance(expr, TListLit):
        return any(_binding_read_in_expr(name, e, shadowed) for e in expr.elements)
    if isinstance(expr, TTupleLit):
        return any(_binding_read_in_expr(name, e, shadowed) for e in expr.elements)
    if isinstance(expr, TSetLit):
        return any(_binding_read_in_expr(name, e, shadowed) for e in expr.elements)
    if isinstance(expr, TMapLit):
        return any(
            _binding_read_in_expr(name, k, shadowed)
            or _binding_read_in_expr(name, v, shadowed)
            for k, v in expr.entries
        )
    return False


def _binding_read_in_target(name: str, target: TExpr, shadowed: bool) -> bool:
    """Return True if assignment target expression reads `name`."""
    if isinstance(target, TVar):
        return False
    if isinstance(target, TIndex):
        return _binding_read_in_expr(
            name, target.obj, shadowed
        ) or _binding_read_in_expr(name, target.index, shadowed)
    if isinstance(target, TFieldAccess):
        return _binding_read_in_expr(name, target.obj, shadowed)
    if isinstance(target, TTupleAccess):
        return _binding_read_in_expr(name, target.obj, shadowed)
    return False


def _binding_used_in_stmt(name: str, stmt: TStmt, shadowed: bool) -> bool:
    """Return True if stmt reads tracked binding `name`."""
    if isinstance(stmt, TAssignStmt):
        return _binding_read_in_expr(
            name, stmt.value, shadowed
        ) or _binding_read_in_target(name, stmt.target, shadowed)
    if isinstance(stmt, TOpAssignStmt):
        return _binding_read_in_expr(
            name, stmt.value, shadowed
        ) or _binding_read_in_expr(name, stmt.target, shadowed)
    if isinstance(stmt, TTupleAssignStmt):
        if _binding_read_in_expr(name, stmt.value, shadowed):
            return True
        return any(_binding_read_in_target(name, t, shadowed) for t in stmt.targets)
    if isinstance(stmt, TExprStmt):
        return _binding_read_in_expr(name, stmt.expr, shadowed)
    if isinstance(stmt, TReturnStmt):
        return stmt.value is not None and _binding_read_in_expr(
            name, stmt.value, shadowed
        )
    if isinstance(stmt, TThrowStmt):
        return _binding_read_in_expr(name, stmt.expr, shadowed)
    if isinstance(stmt, TIfStmt):
        if _binding_read_in_expr(name, stmt.cond, shadowed):
            return True
        if _binding_used_in_stmts(name, stmt.then_body, shadowed):
            return True
        return stmt.else_body is not None and _binding_used_in_stmts(
            name, stmt.else_body, shadowed
        )
    if isinstance(stmt, TWhileStmt):
        if _binding_read_in_expr(name, stmt.cond, shadowed):
            return True
        return _binding_used_in_stmts(name, stmt.body, shadowed)
    if isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                if _binding_read_in_expr(name, a, shadowed):
                    return True
        elif _binding_read_in_expr(name, stmt.iterable, shadowed):
            return True
        body_shadowed = shadowed or name in stmt.binding
        return _binding_used_in_stmts(name, stmt.body, body_shadowed)
    if isinstance(stmt, TTryStmt):
        if _binding_used_in_stmts(name, stmt.body, shadowed):
            return True
        for catch in stmt.catches:
            catch_shadowed = shadowed or catch.name == name
            if _binding_used_in_stmts(name, catch.body, catch_shadowed):
                return True
        if stmt.finally_body is not None:
            return _binding_used_in_stmts(name, stmt.finally_body, shadowed)
        return False
    if isinstance(stmt, TMatchStmt):
        if _binding_read_in_expr(name, stmt.expr, shadowed):
            return True
        for case in stmt.cases:
            case_shadowed = shadowed
            pat = case.pattern
            if isinstance(pat, TPatternType) and pat.name == name:
                case_shadowed = True
            if _binding_used_in_stmts(name, case.body, case_shadowed):
                return True
        if stmt.default is not None:
            default_shadowed = shadowed
            if stmt.default.name == name:
                default_shadowed = True
            return _binding_used_in_stmts(name, stmt.default.body, default_shadowed)
        return False
    return False


def _binding_used_in_stmts(
    name: str, stmts: list[TStmt], shadowed: bool = False
) -> bool:
    """Return True if tracked binding `name` is read within stmts."""
    block_shadowed = shadowed
    for stmt in stmts:
        if isinstance(stmt, TLetStmt):
            if stmt.value is not None and _binding_read_in_expr(
                name, stmt.value, block_shadowed
            ):
                return True
            if stmt.name == name:
                # Lexical shadowing applies to following statements in this block.
                block_shadowed = True
            continue
        if _binding_used_in_stmt(name, stmt, block_shadowed):
            return True
    return False


# ============================================================
# TUPLE UNUSED INDICES
# ============================================================


def _analyze_tuple_targets_in_stmts(stmts: list[TStmt]) -> None:
    """Mark unused tuple target indices in statements."""
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, TTupleAssignStmt):
            remaining = stmts[i + 1 :]
            unused: list[str] = []
            for j, t in enumerate(stmt.targets):
                if isinstance(t, TVar) and t.name != "_":
                    first = _first_access_in_stmts(t.name, remaining)
                    if first != "read":
                        unused.append(str(j))
            stmt.annotations["liveness.tuple_unused_indices"] = ",".join(unused)
        if isinstance(stmt, TIfStmt):
            _analyze_tuple_targets_in_stmts(stmt.then_body)
            if stmt.else_body is not None:
                _analyze_tuple_targets_in_stmts(stmt.else_body)
        elif isinstance(stmt, TWhileStmt):
            _analyze_tuple_targets_in_stmts(stmt.body)
        elif isinstance(stmt, TForStmt):
            _analyze_tuple_targets_in_stmts(stmt.body)
        elif isinstance(stmt, TTryStmt):
            _analyze_tuple_targets_in_stmts(stmt.body)
            for catch in stmt.catches:
                _analyze_tuple_targets_in_stmts(catch.body)
            if stmt.finally_body is not None:
                _analyze_tuple_targets_in_stmts(stmt.finally_body)
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                _analyze_tuple_targets_in_stmts(case.body)
            if stmt.default is not None:
                _analyze_tuple_targets_in_stmts(stmt.default.body)


# ============================================================
# PER-FUNCTION ANALYSIS
# ============================================================


def _analyze_fn(decl: TFnDecl) -> None:
    """Run liveness analysis on a single function."""
    _analyze_initial_value_in_stmts(decl.body)
    _analyze_catch_and_match_bindings(decl.body)
    _analyze_tuple_targets_in_stmts(decl.body)


# ============================================================
# PUBLIC API
# ============================================================


def analyze_liveness(module: TModule, checker: Checker) -> None:
    """Run liveness analysis on all functions in the module."""
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            _analyze_fn(decl)
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                _analyze_fn(method)
