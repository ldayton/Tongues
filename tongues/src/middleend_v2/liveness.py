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
        if try_result == "write" and catch_results and all(
            r == "write" for r in catch_results
        ):
            return "write"
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
# COLLECT USED VAR NAMES (read positions only)
# ============================================================


def _collect_expr_var_names(expr: TExpr, out: set[str]) -> None:
    """Collect all TVar.name references in an expression."""
    if isinstance(expr, TVar):
        out.add(expr.name)
        return
    if isinstance(expr, TBinaryOp):
        _collect_expr_var_names(expr.left, out)
        _collect_expr_var_names(expr.right, out)
    elif isinstance(expr, TUnaryOp):
        _collect_expr_var_names(expr.operand, out)
    elif isinstance(expr, TTernary):
        _collect_expr_var_names(expr.cond, out)
        _collect_expr_var_names(expr.then_expr, out)
        _collect_expr_var_names(expr.else_expr, out)
    elif isinstance(expr, TFieldAccess):
        _collect_expr_var_names(expr.obj, out)
    elif isinstance(expr, TTupleAccess):
        _collect_expr_var_names(expr.obj, out)
    elif isinstance(expr, TIndex):
        _collect_expr_var_names(expr.obj, out)
        _collect_expr_var_names(expr.index, out)
    elif isinstance(expr, TSlice):
        _collect_expr_var_names(expr.obj, out)
        _collect_expr_var_names(expr.low, out)
        _collect_expr_var_names(expr.high, out)
    elif isinstance(expr, TCall):
        _collect_expr_var_names(expr.func, out)
        for a in expr.args:
            _collect_expr_var_names(a.value, out)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _collect_expr_var_names(e, out)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _collect_expr_var_names(e, out)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _collect_expr_var_names(e, out)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _collect_expr_var_names(k, out)
            _collect_expr_var_names(v, out)


def _collect_target_read_names(target: TExpr, out: set[str]) -> None:
    """Collect var names read by an assignment target (not the assigned var)."""
    if isinstance(target, TVar):
        return
    if isinstance(target, TIndex):
        _collect_expr_var_names(target.obj, out)
        _collect_expr_var_names(target.index, out)
    elif isinstance(target, TFieldAccess):
        _collect_expr_var_names(target.obj, out)
    elif isinstance(target, TTupleAccess):
        _collect_expr_var_names(target.obj, out)


def _collect_stmt_var_names(stmt: TStmt, out: set[str]) -> None:
    """Collect variable names in read positions within a statement."""
    if isinstance(stmt, TLetStmt):
        if stmt.value is not None:
            _collect_expr_var_names(stmt.value, out)
    elif isinstance(stmt, TAssignStmt):
        _collect_expr_var_names(stmt.value, out)
        _collect_target_read_names(stmt.target, out)
    elif isinstance(stmt, TOpAssignStmt):
        _collect_expr_var_names(stmt.value, out)
        _collect_expr_var_names(stmt.target, out)
    elif isinstance(stmt, TTupleAssignStmt):
        _collect_expr_var_names(stmt.value, out)
        for t in stmt.targets:
            _collect_target_read_names(t, out)
    elif isinstance(stmt, TExprStmt):
        _collect_expr_var_names(stmt.expr, out)
    elif isinstance(stmt, TReturnStmt):
        if stmt.value is not None:
            _collect_expr_var_names(stmt.value, out)
    elif isinstance(stmt, TThrowStmt):
        _collect_expr_var_names(stmt.expr, out)
    elif isinstance(stmt, TIfStmt):
        _collect_expr_var_names(stmt.cond, out)
        for s in stmt.then_body:
            _collect_stmt_var_names(s, out)
        if stmt.else_body is not None:
            for s in stmt.else_body:
                _collect_stmt_var_names(s, out)
    elif isinstance(stmt, TWhileStmt):
        _collect_expr_var_names(stmt.cond, out)
        for s in stmt.body:
            _collect_stmt_var_names(s, out)
    elif isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                _collect_expr_var_names(a, out)
        else:
            _collect_expr_var_names(stmt.iterable, out)
        for s in stmt.body:
            _collect_stmt_var_names(s, out)
    elif isinstance(stmt, TTryStmt):
        for s in stmt.body:
            _collect_stmt_var_names(s, out)
        for catch in stmt.catches:
            for s in catch.body:
                _collect_stmt_var_names(s, out)
        if stmt.finally_body is not None:
            for s in stmt.finally_body:
                _collect_stmt_var_names(s, out)
    elif isinstance(stmt, TMatchStmt):
        _collect_expr_var_names(stmt.expr, out)
        for case in stmt.cases:
            for s in case.body:
                _collect_stmt_var_names(s, out)
        if stmt.default is not None:
            for s in stmt.default.body:
                _collect_stmt_var_names(s, out)


def _collect_used_var_names(stmts: list[TStmt]) -> set[str]:
    """Recursively collect all variable names in read positions."""
    out: set[str] = set()
    for stmt in stmts:
        _collect_stmt_var_names(stmt, out)
    return out


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
                used = _collect_used_var_names(catch.body)
                catch.annotations["liveness.catch_var_unused"] = catch.name not in used
            _analyze_catch_and_match_bindings(stmt.body)
            for catch in stmt.catches:
                _analyze_catch_and_match_bindings(catch.body)
            if stmt.finally_body is not None:
                _analyze_catch_and_match_bindings(stmt.finally_body)
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                pat = case.pattern
                if isinstance(pat, TPatternType):
                    used = _collect_used_var_names(case.body)
                    pat.annotations["liveness.match_var_unused"] = pat.name not in used
                _analyze_catch_and_match_bindings(case.body)
            if stmt.default is not None:
                if stmt.default.name is not None:
                    used = _collect_used_var_names(stmt.default.body)
                    stmt.default.annotations["liveness.match_var_unused"] = (
                        stmt.default.name not in used
                    )
                _analyze_catch_and_match_bindings(stmt.default.body)
        elif isinstance(stmt, TIfStmt):
            _analyze_catch_and_match_bindings(stmt.then_body)
            if stmt.else_body is not None:
                _analyze_catch_and_match_bindings(stmt.else_body)
        elif isinstance(stmt, TWhileStmt):
            _analyze_catch_and_match_bindings(stmt.body)
        elif isinstance(stmt, TForStmt):
            _analyze_catch_and_match_bindings(stmt.body)


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
