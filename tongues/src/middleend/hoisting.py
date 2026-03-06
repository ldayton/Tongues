"""Hoisting analysis pass for Taytsh IR.

Analyzes each function body independently, writing annotations for variable
hoisting (Go pre-declaration), continue detection (Lua workarounds), and
rune variable collection (Go string indexing).
"""

from __future__ import annotations

from ..taytsh.ast import (
    TAssignStmt,
    TBinaryOp,
    TBreakStmt,
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
from ..taytsh.check import Checker, StructT, Type, type_name, STRING_T, type_eq


# ============================================================
# RUNE VARIABLE COLLECTION
# ============================================================


def _check_self_field_rune(
    fa: TFieldAccess, bindings: dict[str, Type], out: set[str]
) -> None:
    """If fa is self.field and the field is string-typed, add field name to rune vars."""
    if isinstance(fa.obj, TVar) and fa.obj.name == "this":
        self_t = bindings.get("this")
        if self_t is not None and isinstance(self_t, StructT):
            field_t = self_t.fields.get(fa.field)
            if field_t is not None and type_eq(field_t, STRING_T):
                out.add(fa.field)


def _collect_rune_expr(expr: TExpr, bindings: dict[str, Type], out: set[str]) -> None:
    """Find string-typed vars used as base of TIndex/TSlice."""
    if isinstance(expr, TIndex):
        if isinstance(expr.obj, TVar):
            t = bindings.get(expr.obj.name)
            if t is not None and type_eq(t, STRING_T):
                out.add(expr.obj.name)
        elif isinstance(expr.obj, TFieldAccess):
            _check_self_field_rune(expr.obj, bindings, out)
        _collect_rune_expr(expr.obj, bindings, out)
        _collect_rune_expr(expr.index, bindings, out)
        return
    if isinstance(expr, TSlice):
        if isinstance(expr.obj, TVar):
            t = bindings.get(expr.obj.name)
            if t is not None and type_eq(t, STRING_T):
                out.add(expr.obj.name)
        elif isinstance(expr.obj, TFieldAccess):
            _check_self_field_rune(expr.obj, bindings, out)
        _collect_rune_expr(expr.obj, bindings, out)
        _collect_rune_expr(expr.low, bindings, out)
        _collect_rune_expr(expr.high, bindings, out)
        return
    match expr:
        case TBinaryOp():
            _collect_rune_expr(expr.left, bindings, out)
            _collect_rune_expr(expr.right, bindings, out)
        case TUnaryOp():
            _collect_rune_expr(expr.operand, bindings, out)
        case TTernary():
            _collect_rune_expr(expr.cond, bindings, out)
            _collect_rune_expr(expr.then_expr, bindings, out)
            _collect_rune_expr(expr.else_expr, bindings, out)
        case TFieldAccess() | TTupleAccess():
            _collect_rune_expr(expr.obj, bindings, out)
        case TCall():
            _collect_rune_expr(expr.func, bindings, out)
            for a in expr.args:
                _collect_rune_expr(a.value, bindings, out)
        case TListLit() | TTupleLit() | TSetLit():
            for e in expr.elements:
                _collect_rune_expr(e, bindings, out)
        case TMapLit():
            for k, v in expr.entries:
                _collect_rune_expr(k, bindings, out)
                _collect_rune_expr(v, bindings, out)


def _collect_rune_stmt(stmt: TStmt, bindings: dict[str, Type], out: set[str]) -> None:
    """Walk a statement for rune variable uses."""
    match stmt:
        case TLetStmt():
            if stmt.value is not None:
                _collect_rune_expr(stmt.value, bindings, out)
        case TAssignStmt():
            _collect_rune_expr(stmt.value, bindings, out)
            _collect_rune_expr(stmt.target, bindings, out)
        case TOpAssignStmt():
            _collect_rune_expr(stmt.value, bindings, out)
            _collect_rune_expr(stmt.target, bindings, out)
        case TTupleAssignStmt():
            _collect_rune_expr(stmt.value, bindings, out)
            for t in stmt.targets:
                _collect_rune_expr(t, bindings, out)
        case TExprStmt():
            _collect_rune_expr(stmt.expr, bindings, out)
        case TReturnStmt():
            if stmt.value is not None:
                _collect_rune_expr(stmt.value, bindings, out)
        case TThrowStmt():
            _collect_rune_expr(stmt.expr, bindings, out)
        case TIfStmt():
            _collect_rune_expr(stmt.cond, bindings, out)
            _collect_rune_stmts(stmt.then_body, bindings, out)
            if stmt.else_body is not None:
                _collect_rune_stmts(stmt.else_body, bindings, out)
        case TWhileStmt():
            _collect_rune_expr(stmt.cond, bindings, out)
            _collect_rune_stmts(stmt.body, bindings, out)
        case TForStmt():
            if isinstance(stmt.iterable, TRange):
                for a in stmt.iterable.args:
                    _collect_rune_expr(a, bindings, out)
            else:
                _collect_rune_expr(stmt.iterable, bindings, out)
            _collect_rune_stmts(stmt.body, bindings, out)
        case TTryStmt():
            _collect_rune_stmts(stmt.body, bindings, out)
            for catch in stmt.catches:
                _collect_rune_stmts(catch.body, bindings, out)
            if stmt.finally_body is not None:
                _collect_rune_stmts(stmt.finally_body, bindings, out)
        case TMatchStmt():
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
# CONTINUE / BREAK DETECTION
# ============================================================


def _has_flow_stmt(stmts: list[TStmt], check_continue: bool) -> bool:
    """Check for continue/break, stopping at nested loops."""
    for stmt in stmts:
        if check_continue and isinstance(stmt, TContinueStmt):
            return True
        if not check_continue and isinstance(stmt, TBreakStmt):
            return True
        if isinstance(stmt, TIfStmt):
            if _has_flow_stmt(stmt.then_body, check_continue):
                return True
            if stmt.else_body is not None and _has_flow_stmt(
                stmt.else_body, check_continue
            ):
                return True
        elif isinstance(stmt, TTryStmt):
            if _has_flow_stmt(stmt.body, check_continue):
                return True
            for catch in stmt.catches:
                if _has_flow_stmt(catch.body, check_continue):
                    return True
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                if _has_flow_stmt(case.body, check_continue):
                    return True
            if stmt.default is not None and _has_flow_stmt(
                stmt.default.body, check_continue
            ):
                return True
    return False


# ============================================================
# COLLECT LET DECLARATIONS INSIDE CONTROL STRUCTURES
# ============================================================


def _find_let_stmts(stmts: list[TStmt]) -> list[TLetStmt]:
    """Recursively find all TLetStmt nodes in control flow bodies."""
    result: list[TLetStmt] = []
    for stmt in stmts:
        if isinstance(stmt, TLetStmt):
            result.append(stmt)
        if isinstance(stmt, TIfStmt):
            result.extend(_find_let_stmts(stmt.then_body))
            if stmt.else_body is not None:
                result.extend(_find_let_stmts(stmt.else_body))
        elif isinstance(stmt, TWhileStmt):
            result.extend(_find_let_stmts(stmt.body))
        elif isinstance(stmt, TForStmt):
            result.extend(_find_let_stmts(stmt.body))
        elif isinstance(stmt, TTryStmt):
            result.extend(_find_let_stmts(stmt.body))
            for catch in stmt.catches:
                result.extend(_find_let_stmts(catch.body))
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                result.extend(_find_let_stmts(case.body))
            if stmt.default is not None:
                result.extend(_find_let_stmts(stmt.default.body))
    return result


def _collect_let_decls(
    stmts: list[TStmt], declared: set[str], checker: Checker
) -> dict[str, str]:
    """Return {name: type_string} for let bindings not already in declared."""
    result: dict[str, str] = {}
    for let in _find_let_stmts(stmts):
        if let.name not in declared:
            resolved = checker.resolve_type(let.typ)
            result[let.name] = type_name(resolved)
    return result


# ============================================================
# COLLECT USED VARIABLE NAMES
# ============================================================


def _collect_expr_var_names(expr: TExpr, out: set[str]) -> None:
    """Collect all TVar.name references in an expression."""
    if isinstance(expr, TVar):
        out.add(expr.name)
        return
    match expr:
        case TBinaryOp():
            _collect_expr_var_names(expr.left, out)
            _collect_expr_var_names(expr.right, out)
        case TUnaryOp():
            _collect_expr_var_names(expr.operand, out)
        case TTernary():
            _collect_expr_var_names(expr.cond, out)
            _collect_expr_var_names(expr.then_expr, out)
            _collect_expr_var_names(expr.else_expr, out)
        case TFieldAccess() | TTupleAccess():
            _collect_expr_var_names(expr.obj, out)
        case TIndex():
            _collect_expr_var_names(expr.obj, out)
            _collect_expr_var_names(expr.index, out)
        case TSlice():
            _collect_expr_var_names(expr.obj, out)
            _collect_expr_var_names(expr.low, out)
            _collect_expr_var_names(expr.high, out)
        case TCall():
            _collect_expr_var_names(expr.func, out)
            for a in expr.args:
                _collect_expr_var_names(a.value, out)
        case TListLit() | TTupleLit() | TSetLit():
            for e in expr.elements:
                _collect_expr_var_names(e, out)
        case TMapLit():
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


def _collect_stmts_var_names(stmts: list[TStmt], out: set[str]) -> None:
    for stmt in stmts:
        _collect_stmt_var_names(stmt, out)


def _collect_stmt_var_names(stmt: TStmt, out: set[str]) -> None:
    """Collect variable names in read positions within a statement."""
    match stmt:
        case TLetStmt():
            if stmt.value is not None:
                _collect_expr_var_names(stmt.value, out)
        case TAssignStmt():
            _collect_expr_var_names(stmt.value, out)
            _collect_target_read_names(stmt.target, out)
        case TOpAssignStmt():
            _collect_expr_var_names(stmt.value, out)
            _collect_expr_var_names(stmt.target, out)
        case TTupleAssignStmt():
            _collect_expr_var_names(stmt.value, out)
            for t in stmt.targets:
                _collect_target_read_names(t, out)
        case TExprStmt():
            _collect_expr_var_names(stmt.expr, out)
        case TReturnStmt():
            if stmt.value is not None:
                _collect_expr_var_names(stmt.value, out)
        case TThrowStmt():
            _collect_expr_var_names(stmt.expr, out)
        case TIfStmt():
            _collect_expr_var_names(stmt.cond, out)
            _collect_stmts_var_names(stmt.then_body, out)
            if stmt.else_body is not None:
                _collect_stmts_var_names(stmt.else_body, out)
        case TWhileStmt():
            _collect_expr_var_names(stmt.cond, out)
            _collect_stmts_var_names(stmt.body, out)
        case TForStmt():
            if isinstance(stmt.iterable, TRange):
                for a in stmt.iterable.args:
                    _collect_expr_var_names(a, out)
            else:
                _collect_expr_var_names(stmt.iterable, out)
            _collect_stmts_var_names(stmt.body, out)
        case TTryStmt():
            _collect_stmts_var_names(stmt.body, out)
            for catch in stmt.catches:
                _collect_stmts_var_names(catch.body, out)
            if stmt.finally_body is not None:
                _collect_stmts_var_names(stmt.finally_body, out)
        case TMatchStmt():
            _collect_expr_var_names(stmt.expr, out)
            for case in stmt.cases:
                _collect_stmts_var_names(case.body, out)
            if stmt.default is not None:
                _collect_stmts_var_names(stmt.default.body, out)


def _collect_used_vars(stmts: list[TStmt]) -> set[str]:
    """Collect all variable names referenced in statements."""
    out: set[str] = set()
    _collect_stmts_var_names(stmts, out)
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

        if isinstance(stmt, TWhileStmt):
            stmt.annotations["hoisting.has_continue"] = (
                "true" if _has_flow_stmt(stmt.body, True) else "false"
            )
        elif isinstance(stmt, TForStmt):
            stmt.annotations["hoisting.has_continue"] = (
                "true" if _has_flow_stmt(stmt.body, True) else "false"
            )
        elif isinstance(stmt, TMatchStmt):
            all_case_stmts: list[TStmt] = []
            for case in stmt.cases:
                all_case_stmts.extend(case.body)
            if stmt.default is not None:
                all_case_stmts.extend(stmt.default.body)
            stmt.annotations["hoisting.has_break"] = (
                "true" if _has_flow_stmt(all_case_stmts, False) else "false"
            )

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
    match stmt:
        case TIfStmt():
            result.extend(stmt.then_body)
            if stmt.else_body is not None:
                result.extend(stmt.else_body)
        case TTryStmt():
            result.extend(stmt.body)
            for catch in stmt.catches:
                result.extend(catch.body)
        case TWhileStmt() | TForStmt():
            result.extend(stmt.body)
        case TMatchStmt():
            for case in stmt.cases:
                result.extend(case.body)
            if stmt.default is not None:
                result.extend(stmt.default.body)
    return result


def _recurse_control_children(
    stmt: TStmt, declared: set[str], checker: Checker
) -> None:
    """Recurse into the bodies of a control structure."""
    match stmt:
        case TIfStmt():
            _analyze_stmts(stmt.then_body, set(declared), checker)
            if stmt.else_body is not None:
                _analyze_stmts(stmt.else_body, set(declared), checker)
        case TTryStmt():
            _analyze_stmts(stmt.body, set(declared), checker)
            for catch in stmt.catches:
                _analyze_stmts(catch.body, set(declared), checker)
        case TWhileStmt():
            _analyze_stmts(stmt.body, set(declared), checker)
        case TForStmt():
            child_declared = set(declared)
            for b in stmt.binding:
                child_declared.add(b)
            _analyze_stmts(stmt.body, child_declared, checker)
        case TMatchStmt():
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
        elif p.name == "this" and self_type is not None:
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
    """Collect all let bindings in a function to build the type map."""
    for let in _find_let_stmts(stmts):
        bindings[let.name] = checker.resolve_type(let.typ)


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
