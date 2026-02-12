"""Hoisting analysis pass for Taytsh IR.

Analyzes each function body independently, writing annotations for variable
hoisting (Go pre-declaration), continue detection (Lua workarounds), and
rune variable collection (Go string indexing).
"""

from __future__ import annotations

from ..taytsh.ast import (
    TAssignStmt,
    TBinaryOp,
    TCall,
    TContinueStmt,
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
from ..taytsh.check import Checker, Type, type_name, STRING_T, type_eq


# ============================================================
# RUNE VARIABLE COLLECTION
# ============================================================


def _collect_rune_expr(expr: TExpr, bindings: dict[str, Type], out: set[str]) -> None:
    """Find string-typed vars used as base of TIndex/TSlice."""
    if isinstance(expr, TIndex):
        if isinstance(expr.obj, TVar):
            t = bindings.get(expr.obj.name)
            if t is not None and type_eq(t, STRING_T):
                out.add(expr.obj.name)
        _collect_rune_expr(expr.obj, bindings, out)
        _collect_rune_expr(expr.index, bindings, out)
        return
    if isinstance(expr, TSlice):
        if isinstance(expr.obj, TVar):
            t = bindings.get(expr.obj.name)
            if t is not None and type_eq(t, STRING_T):
                out.add(expr.obj.name)
        _collect_rune_expr(expr.obj, bindings, out)
        _collect_rune_expr(expr.low, bindings, out)
        _collect_rune_expr(expr.high, bindings, out)
        return
    if isinstance(expr, TBinaryOp):
        _collect_rune_expr(expr.left, bindings, out)
        _collect_rune_expr(expr.right, bindings, out)
    elif isinstance(expr, TUnaryOp):
        _collect_rune_expr(expr.operand, bindings, out)
    elif isinstance(expr, TTernary):
        _collect_rune_expr(expr.cond, bindings, out)
        _collect_rune_expr(expr.then_expr, bindings, out)
        _collect_rune_expr(expr.else_expr, bindings, out)
    elif isinstance(expr, TFieldAccess):
        _collect_rune_expr(expr.obj, bindings, out)
    elif isinstance(expr, TTupleAccess):
        _collect_rune_expr(expr.obj, bindings, out)
    elif isinstance(expr, TCall):
        _collect_rune_expr(expr.func, bindings, out)
        for a in expr.args:
            _collect_rune_expr(a.value, bindings, out)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _collect_rune_expr(e, bindings, out)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _collect_rune_expr(e, bindings, out)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _collect_rune_expr(e, bindings, out)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _collect_rune_expr(k, bindings, out)
            _collect_rune_expr(v, bindings, out)


def _collect_rune_stmt(stmt: TStmt, bindings: dict[str, Type], out: set[str]) -> None:
    """Walk a statement for rune variable uses."""
    if isinstance(stmt, TLetStmt):
        if stmt.value is not None:
            _collect_rune_expr(stmt.value, bindings, out)
    elif isinstance(stmt, TAssignStmt):
        _collect_rune_expr(stmt.value, bindings, out)
        _collect_rune_expr(stmt.target, bindings, out)
    elif isinstance(stmt, TOpAssignStmt):
        _collect_rune_expr(stmt.value, bindings, out)
        _collect_rune_expr(stmt.target, bindings, out)
    elif isinstance(stmt, TTupleAssignStmt):
        _collect_rune_expr(stmt.value, bindings, out)
        for t in stmt.targets:
            _collect_rune_expr(t, bindings, out)
    elif isinstance(stmt, TExprStmt):
        _collect_rune_expr(stmt.expr, bindings, out)
    elif isinstance(stmt, TReturnStmt):
        if stmt.value is not None:
            _collect_rune_expr(stmt.value, bindings, out)
    elif isinstance(stmt, TThrowStmt):
        _collect_rune_expr(stmt.expr, bindings, out)
    elif isinstance(stmt, TIfStmt):
        _collect_rune_expr(stmt.cond, bindings, out)
        _collect_rune_stmts(stmt.then_body, bindings, out)
        if stmt.else_body is not None:
            _collect_rune_stmts(stmt.else_body, bindings, out)
    elif isinstance(stmt, TWhileStmt):
        _collect_rune_expr(stmt.cond, bindings, out)
        _collect_rune_stmts(stmt.body, bindings, out)
    elif isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                _collect_rune_expr(a, bindings, out)
        else:
            _collect_rune_expr(stmt.iterable, bindings, out)
        _collect_rune_stmts(stmt.body, bindings, out)
    elif isinstance(stmt, TTryStmt):
        _collect_rune_stmts(stmt.body, bindings, out)
        for catch in stmt.catches:
            _collect_rune_stmts(catch.body, bindings, out)
        if stmt.finally_body is not None:
            _collect_rune_stmts(stmt.finally_body, bindings, out)
    elif isinstance(stmt, TMatchStmt):
        _collect_rune_expr(stmt.expr, bindings, out)
        for case in stmt.cases:
            _collect_rune_stmts(case.body, bindings, out)
        if stmt.default is not None:
            _collect_rune_stmts(stmt.default.body, bindings, out)


def _collect_rune_stmts(
    stmts: list[TStmt], bindings: dict[str, Type], out: set[str]
) -> None:
    for stmt in stmts:
        _collect_rune_stmt(stmt, bindings, out)


def _collect_rune_vars(stmts: list[TStmt], bindings: dict[str, Type]) -> set[str]:
    out: set[str] = set()
    _collect_rune_stmts(stmts, bindings, out)
    return out


# ============================================================
# CONTINUE DETECTION
# ============================================================


def _has_continue(stmts: list[TStmt]) -> bool:
    """Check for TContinueStmt, stopping at nested loops."""
    for stmt in stmts:
        if isinstance(stmt, TContinueStmt):
            return True
        if isinstance(stmt, TIfStmt):
            if _has_continue(stmt.then_body):
                return True
            if stmt.else_body is not None and _has_continue(stmt.else_body):
                return True
        elif isinstance(stmt, TTryStmt):
            if _has_continue(stmt.body):
                return True
            for catch in stmt.catches:
                if _has_continue(catch.body):
                    return True
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                if _has_continue(case.body):
                    return True
            if stmt.default is not None and _has_continue(stmt.default.body):
                return True
        # TWhileStmt/TForStmt: don't recurse — nested loops own their continues
    return False


# ============================================================
# COLLECT LET DECLARATIONS INSIDE CONTROL STRUCTURES
# ============================================================


def _collect_let_decls(
    stmts: list[TStmt], declared: set[str], checker: Checker
) -> dict[str, str]:
    """Recursively find TLetStmt inside stmts, returning {name: type_string} for undeclared names."""
    result: dict[str, str] = {}
    for stmt in stmts:
        if isinstance(stmt, TLetStmt):
            if stmt.name not in declared:
                resolved = checker.resolve_type(stmt.typ)
                result[stmt.name] = type_name(resolved)
        if isinstance(stmt, TIfStmt):
            result.update(_collect_let_decls(stmt.then_body, declared, checker))
            if stmt.else_body is not None:
                result.update(_collect_let_decls(stmt.else_body, declared, checker))
        elif isinstance(stmt, TWhileStmt):
            result.update(_collect_let_decls(stmt.body, declared, checker))
        elif isinstance(stmt, TForStmt):
            result.update(_collect_let_decls(stmt.body, declared, checker))
        elif isinstance(stmt, TTryStmt):
            result.update(_collect_let_decls(stmt.body, declared, checker))
            for catch in stmt.catches:
                result.update(_collect_let_decls(catch.body, declared, checker))
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                result.update(_collect_let_decls(case.body, declared, checker))
            if stmt.default is not None:
                result.update(_collect_let_decls(stmt.default.body, declared, checker))
    return result


# ============================================================
# COLLECT USED VARIABLE NAMES
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
    """Collect var names read by an assignment target."""
    if isinstance(target, TVar):
        out.add(target.name)
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


def _collect_used_vars(stmts: list[TStmt]) -> set[str]:
    """Recursively collect all variable names referenced in statements."""
    out: set[str] = set()
    for stmt in stmts:
        _collect_stmt_var_names(stmt, out)
    return out


# ============================================================
# HOISTED VARS SERIALIZATION
# ============================================================


def _serialize_hoisted(pairs: list[tuple[str, str]]) -> str:
    """Serialize [(name, type_str)] to "x:int;y:string"."""
    parts: list[str] = []
    for name, typ in pairs:
        parts.append(name + ":" + typ)
    return ";".join(parts)


# ============================================================
# MAIN STATEMENT WALKER
# ============================================================


def _analyze_stmts(stmts: list[TStmt], declared: set[str], checker: Checker) -> None:
    """Walk statements, annotating control structures with hoisted_vars and has_continue."""
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, TLetStmt):
            declared.add(stmt.name)
            continue

        is_control = isinstance(
            stmt, (TIfStmt, TTryStmt, TWhileStmt, TForStmt, TMatchStmt)
        )
        if not is_control:
            continue

        # For while/for: annotate has_continue
        if isinstance(stmt, TWhileStmt):
            stmt.annotations["hoisting.has_continue"] = _has_continue(stmt.body)
        elif isinstance(stmt, TForStmt):
            stmt.annotations["hoisting.has_continue"] = _has_continue(stmt.body)

        # Collect let decls inside this control structure
        inner_decls = _collect_let_decls(_get_control_bodies(stmt), declared, checker)

        # Collect vars used after this structure
        after_used = _collect_used_vars(stmts[i + 1 :])

        # Intersection: variables declared inside but used after
        hoisted: list[tuple[str, str]] = []
        for name in sorted(inner_decls):
            if name in after_used:
                hoisted.append((name, inner_decls[name]))

        stmt.annotations["hoisting.hoisted_vars"] = _serialize_hoisted(hoisted)

        # Add hoisted names to declared set so they aren't re-hoisted at outer levels
        for name, _ in hoisted:
            declared.add(name)

        # Recurse into children
        _recurse_control_children(stmt, declared, checker)


def _get_control_bodies(stmt: TStmt) -> list[TStmt]:
    """Gather all inner statements of a control structure into a flat list."""
    result: list[TStmt] = []
    if isinstance(stmt, TIfStmt):
        result.extend(stmt.then_body)
        if stmt.else_body is not None:
            result.extend(stmt.else_body)
    elif isinstance(stmt, TTryStmt):
        result.extend(stmt.body)
        for catch in stmt.catches:
            result.extend(catch.body)
    elif isinstance(stmt, TWhileStmt):
        result.extend(stmt.body)
    elif isinstance(stmt, TForStmt):
        result.extend(stmt.body)
    elif isinstance(stmt, TMatchStmt):
        for case in stmt.cases:
            result.extend(case.body)
        if stmt.default is not None:
            result.extend(stmt.default.body)
    return result


def _recurse_control_children(
    stmt: TStmt, declared: set[str], checker: Checker
) -> None:
    """Recurse into the bodies of a control structure."""
    if isinstance(stmt, TIfStmt):
        _analyze_stmts(stmt.then_body, set(declared), checker)
        if stmt.else_body is not None:
            _analyze_stmts(stmt.else_body, set(declared), checker)
    elif isinstance(stmt, TTryStmt):
        _analyze_stmts(stmt.body, set(declared), checker)
        for catch in stmt.catches:
            _analyze_stmts(catch.body, set(declared), checker)
    elif isinstance(stmt, TWhileStmt):
        _analyze_stmts(stmt.body, set(declared), checker)
    elif isinstance(stmt, TForStmt):
        child_declared = set(declared)
        for b in stmt.binding:
            child_declared.add(b)
        _analyze_stmts(stmt.body, child_declared, checker)
    elif isinstance(stmt, TMatchStmt):
        for case in stmt.cases:
            _analyze_stmts(case.body, set(declared), checker)
        if stmt.default is not None:
            _analyze_stmts(stmt.default.body, set(declared), checker)


# ============================================================
# PER-FUNCTION ANALYSIS
# ============================================================


def _analyze_fn(decl: TFnDecl, checker: Checker, self_type: Type | None = None) -> None:
    """Run hoisting analysis on a single function."""
    # Build bindings map from params and let statements
    bindings: dict[str, Type] = {}
    for p in decl.params:
        if p.typ is not None:
            bindings[p.name] = checker.resolve_type(p.typ)
        elif p.name == "self" and self_type is not None:
            bindings[p.name] = self_type
    _collect_fn_let_bindings(decl.body, bindings, checker)

    # Rune vars
    rune_vars = _collect_rune_vars(decl.body, bindings)
    names = sorted(rune_vars)
    decl.annotations["hoisting.rune_vars"] = ",".join(names)

    # Hoisted vars and has_continue
    declared: set[str] = set()
    for p in decl.params:
        declared.add(p.name)
    _analyze_stmts(decl.body, declared, checker)


def _collect_fn_let_bindings(
    stmts: list[TStmt], bindings: dict[str, Type], checker: Checker
) -> None:
    """Recursively collect all let bindings in a function to build the type map."""
    for stmt in stmts:
        if isinstance(stmt, TLetStmt):
            bindings[stmt.name] = checker.resolve_type(stmt.typ)
        if isinstance(stmt, TIfStmt):
            _collect_fn_let_bindings(stmt.then_body, bindings, checker)
            if stmt.else_body is not None:
                _collect_fn_let_bindings(stmt.else_body, bindings, checker)
        elif isinstance(stmt, TWhileStmt):
            _collect_fn_let_bindings(stmt.body, bindings, checker)
        elif isinstance(stmt, TForStmt):
            _collect_fn_let_bindings(stmt.body, bindings, checker)
        elif isinstance(stmt, TTryStmt):
            _collect_fn_let_bindings(stmt.body, bindings, checker)
            for catch in stmt.catches:
                _collect_fn_let_bindings(catch.body, bindings, checker)
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                _collect_fn_let_bindings(case.body, bindings, checker)
            if stmt.default is not None:
                _collect_fn_let_bindings(stmt.default.body, bindings, checker)


# ============================================================
# PUBLIC API
# ============================================================


def analyze_hoisting(module: TModule, checker: Checker) -> None:
    """Run hoisting analysis on all functions in the module."""
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            _analyze_fn(decl, checker)
        elif isinstance(decl, TStructDecl):
            st = checker.types.get(decl.name)
            for method in decl.methods:
                _analyze_fn(method, checker, self_type=st)
