"""Scope analysis pass for Taytsh IR.

Analyzes each function body independently, writing annotations onto AST nodes
for reassignment/constness, parameter modification/unused, narrowed types,
interface detection, and function reference detection.
"""

from __future__ import annotations

from dataclasses import dataclass

from .type_resolve import ScopeResolver
from ..taytsh.ast import (
    Ann,
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
    TNilLit,
    TOpAssignStmt,
    TPatternEnum,
    TPatternNil,
    TPatternType,
    TRange,
    TReturnStmt,
    TSetLit,
    TSlice,
    TStmt,
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
    BUILTIN_NAMES,
    Checker,
    InterfaceT,
    NIL_T,
    ERROR_T,
    StructT,
    Type,
    UnionT,
    VOID_T,
    contains_nil,
    normalize_union,
    remove_nil,
    type_eq,
    type_name,
)

# Built-in functions whose first argument is mutated in-place.
_MUTATING_BUILTINS: set[str] = {
    "Append",
    "Insert",
    "Pop",
    "RemoveAt",
    "Delete",
    "Add",
    "Remove",
    "ReplaceSlice",
}


# ============================================================
# BINDING INFO
# ============================================================


@dataclass
class _BindingInfo:
    """One binding tracked during the walk."""

    annotations: Ann
    declared_type: Type
    is_param: bool
    binder_name: str | None = None
    reassigned: bool = False
    modified: bool = False
    used: bool = False


# ============================================================
# SCOPE CONTEXT
# ============================================================


@dataclass
class _ScopeCtx:
    checker: Checker
    top_level_fns: set[str]
    bindings: dict[str, _BindingInfo]
    narrowings: dict[str, Type]


def _fork_ctx(
    ctx: _ScopeCtx, extra_narrowings: dict[str, Type] | None = None
) -> _ScopeCtx:
    """Fork context with independent narrowings but shared bindings."""
    new_narrowings = ctx.narrowings.copy()
    if extra_narrowings is not None:
        new_narrowings.update(extra_narrowings)
    return _ScopeCtx(
        checker=ctx.checker,
        top_level_fns=ctx.top_level_fns,
        bindings=ctx.bindings,
        narrowings=new_narrowings,
    )


# ============================================================
# SCOPE TYPE RESOLVER
# ============================================================


def _make_scope_resolver(ctx: _ScopeCtx) -> ScopeResolver:
    locals: dict[str, Type] = {}
    for name, info in ctx.bindings.items():
        locals[name] = info.declared_type
    locals.update(ctx.narrowings)
    return ScopeResolver(locals, ctx.checker)


# ============================================================
# GET BASE VARIABLE
# ============================================================


def _get_base_var(expr: TExpr) -> str | None:
    """Extract the root variable name from x, x.f, x[i], x.f.g[i] chains."""
    if isinstance(expr, TVar):
        return expr.name
    if isinstance(expr, TFieldAccess):
        return _get_base_var(expr.obj)
    if isinstance(expr, TIndex):
        return _get_base_var(expr.obj)
    if isinstance(expr, TTupleAccess):
        return _get_base_var(expr.obj)
    return None


# ============================================================
# ASSIGNMENT TARGET ANALYSIS
# ============================================================


def _check_assign_target(target: TExpr, ctx: _ScopeCtx) -> None:
    """Process an assignment target for reassignment/mutation tracking."""
    if isinstance(target, TVar):
        name = target.name
        if name in ctx.bindings:
            info = ctx.bindings[name]
            info.reassigned = True
            if info.is_param:
                info.modified = True
    elif isinstance(target, (TFieldAccess, TIndex, TTupleAccess)):
        base = _get_base_var(target)
        if base is not None and base in ctx.bindings:
            info = ctx.bindings[base]
            if info.is_param:
                info.modified = True


def _check_call_mutation(expr: TCall, ctx: _ScopeCtx) -> None:
    """Check if a call mutates a parameter (mutating builtins or void methods)."""
    # Mutating builtin: first arg is the mutated collection
    if isinstance(expr.func, TVar) and expr.func.name in _MUTATING_BUILTINS:
        if expr.args:
            base = _get_base_var(expr.args[0].value)
            if base is not None and base in ctx.bindings:
                info = ctx.bindings[base]
                if info.is_param:
                    info.modified = True
    # Void-returning method on a parameter: p.Method(...)
    if isinstance(expr.func, TFieldAccess):
        base = _get_base_var(expr.func.obj)
        if base is not None and base in ctx.bindings:
            info = ctx.bindings[base]
            if info.is_param:
                obj_type = info.declared_type
                if isinstance(obj_type, StructT):
                    method_name = expr.func.field
                    if method_name in obj_type.methods:
                        method_fn = obj_type.methods[method_name]
                        if type_eq(method_fn.ret, VOID_T):
                            info.modified = True


# ============================================================
# WALK EXPRESSIONS
# ============================================================


def _walk_expr(expr: TExpr, ctx: _ScopeCtx) -> None:
    """Walk an expression, recording uses and writing use-site annotations."""
    if isinstance(expr, TVar):
        name = expr.name
        if name in ctx.bindings:
            info = ctx.bindings[name]
            if info.is_param:
                info.used = True
            # Determine effective type at this use site
            effective_type = ctx.narrowings.get(name, info.declared_type)
            if isinstance(effective_type, InterfaceT):
                expr.annotations["scope.is_interface"] = "true"
            if not type_eq(effective_type, info.declared_type):
                expr.annotations["scope.narrowed_type"] = type_name(effective_type)
        elif name in ctx.top_level_fns and name not in BUILTIN_NAMES:
            expr.annotations["scope.is_function_ref"] = "true"
        return
    match expr:
        case TBinaryOp():
            _walk_expr(expr.left, ctx)
            _walk_expr(expr.right, ctx)
        case TUnaryOp():
            _walk_expr(expr.operand, ctx)
        case TTernary():
            _walk_expr(expr.cond, ctx)
            _walk_expr(expr.then_expr, ctx)
            _walk_expr(expr.else_expr, ctx)
        case TFieldAccess() | TTupleAccess():
            _walk_expr(expr.obj, ctx)
        case TIndex():
            _walk_expr(expr.obj, ctx)
            _walk_expr(expr.index, ctx)
        case TSlice():
            _walk_expr(expr.obj, ctx)
            _walk_expr(expr.low, ctx)
            _walk_expr(expr.high, ctx)
        case TCall():
            _check_call_mutation(expr, ctx)
            _walk_expr(expr.func, ctx)
            for a in expr.args:
                _walk_expr(a.value, ctx)
        case TListLit() | TSetLit() | TTupleLit():
            for e in expr.elements:
                _walk_expr(e, ctx)
        case TMapLit():
            for k, v in expr.entries:
                _walk_expr(k, ctx)
                _walk_expr(v, ctx)
        case TFnLit():
            _analyze_fn_lit(expr, ctx)


def _analyze_fn_lit(expr: TFnLit, parent_ctx: _ScopeCtx) -> None:
    """Analyze a function literal with its own independent scope."""
    ctx = _ScopeCtx(
        checker=parent_ctx.checker,
        top_level_fns=parent_ctx.top_level_fns,
        bindings={},
        narrowings={},
    )
    for p in expr.params:
        if p.typ is not None:
            pt = parent_ctx.checker.resolve_type(p.typ)
            ctx.bindings[p.name] = _BindingInfo(
                annotations=p.annotations, declared_type=pt, is_param=True
            )
    _walk_stmts(expr.body, ctx)
    _stamp_bindings(ctx)


# ============================================================
# WALK STATEMENTS
# ============================================================


def _walk_stmts(stmts: list[TStmt], ctx: _ScopeCtx) -> None:
    for s in stmts:
        _walk_stmt(s, ctx)


def _walk_stmt(stmt: TStmt, ctx: _ScopeCtx) -> None:
    match stmt:
        case TLetStmt():
            if stmt.value is not None:
                _walk_expr(stmt.value, ctx)
            declared_type = ctx.checker.resolve_type(stmt.typ)
            ctx.bindings[stmt.name] = _BindingInfo(
                annotations=stmt.annotations,
                declared_type=declared_type,
                is_param=False,
            )
        case TAssignStmt():
            _walk_expr(stmt.value, ctx)
            _check_assign_target(stmt.target, ctx)
            _walk_assign_target_uses(stmt.target, ctx)
            if isinstance(stmt.target, TIndex):
                base = _get_base_var(stmt.target.obj)
                if base is not None and base in ctx.bindings:
                    info = ctx.bindings[base]
                    if info.is_param:
                        info.modified = True
        case TOpAssignStmt():
            _walk_expr(stmt.value, ctx)
            if isinstance(stmt.target, TVar):
                name = stmt.target.name
                if name in ctx.bindings:
                    info = ctx.bindings[name]
                    info.reassigned = True
                    if info.is_param:
                        info.modified = True
            else:
                base = _get_base_var(stmt.target)
                if base is not None and base in ctx.bindings:
                    info = ctx.bindings[base]
                    if info.is_param:
                        info.modified = True
            _walk_assign_target_uses(stmt.target, ctx)
        case TTupleAssignStmt():
            _walk_expr(stmt.value, ctx)
            for t in stmt.targets:
                _check_assign_target(t, ctx)
        case TReturnStmt():
            if stmt.value is not None:
                _walk_expr(stmt.value, ctx)
        case TThrowStmt() | TExprStmt():
            _walk_expr(stmt.expr, ctx)
        case TIfStmt():
            _walk_expr(stmt.cond, ctx)
            _walk_if_stmt(stmt, ctx)
        case TWhileStmt():
            _walk_expr(stmt.cond, ctx)
            _walk_stmts(stmt.body, ctx)
        case TForStmt():
            _walk_for_stmt(stmt, ctx)
        case TMatchStmt():
            _walk_match_stmt(stmt, ctx)
        case TTryStmt():
            _walk_try_stmt(stmt, ctx)


def _walk_assign_target_uses(target: TExpr, ctx: _ScopeCtx) -> None:
    """Walk assignment target sub-expressions for use tracking (not the top-level var)."""
    match target:
        case TFieldAccess() | TTupleAccess():
            _walk_expr(target.obj, ctx)
        case TIndex():
            _walk_expr(target.obj, ctx)
            _walk_expr(target.index, ctx)
        # TVar targets: don't count the target itself as a "use" for unused tracking


# ============================================================
# IF STATEMENT — NARROWING
# ============================================================


def _walk_if_stmt(stmt: TIfStmt, ctx: _ScopeCtx) -> None:
    """Handle if-stmt with potential nil narrowing."""
    then_narrowings: dict[str, Type] = {}
    else_narrowings: dict[str, Type] = {}

    if isinstance(stmt.cond, TBinaryOp):
        var_node: TVar | None = None
        is_nil_check = False
        is_neq = False

        if stmt.cond.op in ("!=", "=="):
            if isinstance(stmt.cond.left, TVar) and isinstance(
                stmt.cond.right, TNilLit
            ):
                var_node = stmt.cond.left
                is_nil_check = True
                is_neq = stmt.cond.op == "!="
            elif isinstance(stmt.cond.right, TVar) and isinstance(
                stmt.cond.left, TNilLit
            ):
                var_node = stmt.cond.right
                is_nil_check = True
                is_neq = stmt.cond.op == "!="

        if is_nil_check and var_node is not None:
            name = var_node.name
            if name in ctx.bindings:
                declared = ctx.bindings[name].declared_type
                if contains_nil(declared):
                    non_nil = remove_nil(declared)
                    if is_neq:
                        then_narrowings[name] = non_nil
                        else_narrowings[name] = NIL_T
                    else:
                        then_narrowings[name] = NIL_T
                        else_narrowings[name] = non_nil

    then_ctx = _fork_ctx(ctx, then_narrowings)
    _walk_stmts(stmt.then_body, then_ctx)

    if stmt.else_body is not None:
        else_ctx = _fork_ctx(ctx, else_narrowings)
        _walk_stmts(stmt.else_body, else_ctx)


# ============================================================
# FOR STATEMENT
# ============================================================


def _walk_for_stmt(stmt: TForStmt, ctx: _ScopeCtx) -> None:
    """Handle for-stmt: register binders, walk iterable and body."""
    # Walk the iterable expression first
    if isinstance(stmt.iterable, TRange):
        for a in stmt.iterable.args:
            _walk_expr(a, ctx)
    else:
        _walk_expr(stmt.iterable, ctx)

    # Resolve binder types and register them
    binder_types = _make_scope_resolver(ctx).resolve_for_binder_types(stmt)
    for bname in stmt.binding:
        btype = ERROR_T
        if binder_types is not None:
            bt = binder_types.get(bname)
            if bt is not None:
                btype = bt
        ctx.bindings[bname] = _BindingInfo(
            annotations=stmt.annotations,
            declared_type=btype,
            is_param=False,
            binder_name=bname,
        )

    _walk_stmts(stmt.body, ctx)


# ============================================================
# MATCH STATEMENT
# ============================================================


def _walk_match_stmt(stmt: TMatchStmt, ctx: _ScopeCtx) -> None:
    """Handle match-stmt: walk scrutinee, then each case with its binding."""
    _walk_expr(stmt.expr, ctx)

    scrutinee_type = _make_scope_resolver(ctx).resolve(stmt.expr)
    covered_types: list[Type] = []

    for case in stmt.cases:
        pat = case.pattern
        case_ctx = _fork_ctx(ctx)

        if isinstance(pat, TPatternType):
            case_type = ctx.checker.resolve_type(pat.type_name)
            covered_types.append(case_type)
            case_ctx.bindings[pat.name] = _BindingInfo(
                annotations=pat.annotations, declared_type=case_type, is_param=False
            )
        elif isinstance(pat, TPatternEnum):
            enum_type = ctx.checker.types.get(pat.enum_name)
            if enum_type is not None:
                covered_types.append(enum_type)
        elif isinstance(pat, TPatternNil):
            covered_types.append(NIL_T)

        _walk_stmts(case.body, case_ctx)

        if isinstance(pat, TPatternType):
            iface = _detect_case_interface(pat.name, case.body, case_ctx)
            pat.annotations["scope.case_interface"] = iface

    if stmt.default is not None:
        dflt_ctx = _fork_ctx(ctx)
        if stmt.default.name is not None:
            residual = _compute_residual_type(scrutinee_type, covered_types, ctx)
            dflt_ctx.bindings[stmt.default.name] = _BindingInfo(
                annotations=stmt.default.annotations,
                declared_type=residual,
                is_param=False,
            )
        _walk_stmts(stmt.default.body, dflt_ctx)
        if stmt.default.name is not None:
            iface = _detect_case_interface(
                stmt.default.name, stmt.default.body, dflt_ctx
            )
            stmt.default.annotations["scope.case_interface"] = iface


def _detect_case_interface(binding_name: str, body: list[TStmt], ctx: _ScopeCtx) -> str:
    """Detect if a case binding is used through an interface in the body.

    Returns the interface name or "" if none.
    """
    for stmt in body:
        result = _scan_stmt_for_interface_use(binding_name, stmt, ctx)
        if result is not None:
            return result
    return ""


def _scan_stmt_for_interface_use(name: str, stmt: TStmt, ctx: _ScopeCtx) -> str | None:
    if isinstance(stmt, TExprStmt):
        return _scan_expr_for_interface_use(name, stmt.expr, ctx)
    if isinstance(stmt, TReturnStmt) and stmt.value is not None:
        return _scan_expr_for_interface_use(name, stmt.value, ctx)
    if isinstance(stmt, TThrowStmt):
        return _scan_expr_for_interface_use(name, stmt.expr, ctx)
    if isinstance(stmt, TLetStmt) and stmt.value is not None:
        return _scan_expr_for_interface_use(name, stmt.value, ctx)
    if isinstance(stmt, TAssignStmt):
        r = _scan_expr_for_interface_use(name, stmt.value, ctx)
        if r is not None:
            return r
        return _scan_expr_for_interface_use(name, stmt.target, ctx)
    if isinstance(stmt, TOpAssignStmt):
        r = _scan_expr_for_interface_use(name, stmt.value, ctx)
        if r is not None:
            return r
        return _scan_expr_for_interface_use(name, stmt.target, ctx)
    if isinstance(stmt, TTupleAssignStmt):
        r = _scan_expr_for_interface_use(name, stmt.value, ctx)
        if r is not None:
            return r
        for t in stmt.targets:
            r = _scan_expr_for_interface_use(name, t, ctx)
            if r is not None:
                return r
    if isinstance(stmt, TIfStmt):
        r = _scan_expr_for_interface_use(name, stmt.cond, ctx)
        if r is not None:
            return r
        for s in stmt.then_body:
            r = _scan_stmt_for_interface_use(name, s, ctx)
            if r is not None:
                return r
        if stmt.else_body is not None:
            for s in stmt.else_body:
                r = _scan_stmt_for_interface_use(name, s, ctx)
                if r is not None:
                    return r
    if isinstance(stmt, TWhileStmt):
        r = _scan_expr_for_interface_use(name, stmt.cond, ctx)
        if r is not None:
            return r
        for s in stmt.body:
            r = _scan_stmt_for_interface_use(name, s, ctx)
            if r is not None:
                return r
    if isinstance(stmt, TForStmt):
        for s in stmt.body:
            r = _scan_stmt_for_interface_use(name, s, ctx)
            if r is not None:
                return r
    if isinstance(stmt, TTryStmt):
        for s in stmt.body:
            r = _scan_stmt_for_interface_use(name, s, ctx)
            if r is not None:
                return r
        for catch in stmt.catches:
            for s in catch.body:
                r = _scan_stmt_for_interface_use(name, s, ctx)
                if r is not None:
                    return r
    return None


def _scan_expr_for_interface_use(name: str, expr: TExpr, ctx: _ScopeCtx) -> str | None:
    """Check if `name` is passed to a function parameter typed as an interface."""
    if isinstance(expr, TCall):
        # Check each argument: is it `name` passed to an interface-typed param?
        result = _check_call_interface_arg(name, expr, ctx)
        if result is not None:
            return result
        # Recurse into sub-expressions
        r = _scan_expr_for_interface_use(name, expr.func, ctx)
        if r is not None:
            return r
        for a in expr.args:
            r = _scan_expr_for_interface_use(name, a.value, ctx)
            if r is not None:
                return r
        return None
    if isinstance(expr, TBinaryOp):
        r = _scan_expr_for_interface_use(name, expr.left, ctx)
        if r is not None:
            return r
        return _scan_expr_for_interface_use(name, expr.right, ctx)
    if isinstance(expr, TUnaryOp):
        return _scan_expr_for_interface_use(name, expr.operand, ctx)
    if isinstance(expr, TTernary):
        r = _scan_expr_for_interface_use(name, expr.cond, ctx)
        if r is not None:
            return r
        r = _scan_expr_for_interface_use(name, expr.then_expr, ctx)
        if r is not None:
            return r
        return _scan_expr_for_interface_use(name, expr.else_expr, ctx)
    if isinstance(expr, TFieldAccess):
        return _scan_expr_for_interface_use(name, expr.obj, ctx)
    if isinstance(expr, TIndex):
        r = _scan_expr_for_interface_use(name, expr.obj, ctx)
        if r is not None:
            return r
        return _scan_expr_for_interface_use(name, expr.index, ctx)
    if isinstance(expr, TSlice):
        r = _scan_expr_for_interface_use(name, expr.obj, ctx)
        if r is not None:
            return r
        r = _scan_expr_for_interface_use(name, expr.low, ctx)
        if r is not None:
            return r
        return _scan_expr_for_interface_use(name, expr.high, ctx)
    if isinstance(expr, TListLit):
        for e in expr.elements:
            r = _scan_expr_for_interface_use(name, e, ctx)
            if r is not None:
                return r
    if isinstance(expr, TTupleLit):
        for e in expr.elements:
            r = _scan_expr_for_interface_use(name, e, ctx)
            if r is not None:
                return r
    if isinstance(expr, TMapLit):
        for k, v in expr.entries:
            r = _scan_expr_for_interface_use(name, k, ctx)
            if r is not None:
                return r
            r = _scan_expr_for_interface_use(name, v, ctx)
            if r is not None:
                return r
    if isinstance(expr, TSetLit):
        for e in expr.elements:
            r = _scan_expr_for_interface_use(name, e, ctx)
            if r is not None:
                return r
    return None


def _check_call_interface_arg(name: str, call: TCall, ctx: _ScopeCtx) -> str | None:
    """If `name` is passed as an argument to an interface-typed parameter, return the interface name."""
    # Resolve param types for the called function
    param_types: list[Type] | None = None
    if isinstance(call.func, TVar):
        fname = call.func.name
        if fname in ctx.checker.functions:
            param_types = ctx.checker.functions[fname].params
        elif fname in ctx.checker.types:
            t = ctx.checker.types[fname]
            if isinstance(t, StructT):
                param_types = list(t.fields.values())
    elif isinstance(call.func, TFieldAccess):
        obj_t = _make_scope_resolver(ctx).resolve(call.func.obj)
        if obj_t is not None and isinstance(obj_t, StructT):
            mname = call.func.field
            if mname in obj_t.methods:
                # Skip self param
                param_types = obj_t.methods[mname].params[1:]
    if param_types is None:
        return None
    for i, arg in enumerate(call.args):
        if isinstance(arg.value, TVar) and arg.value.name == name:
            if i < len(param_types):
                pt = param_types[i]
                if isinstance(pt, InterfaceT):
                    return pt.name
    return None


def _compute_residual_type(
    scrutinee: Type | None, covered: list[Type], ctx: _ScopeCtx
) -> Type:
    """Compute the residual type for a default arm (scrutinee minus covered)."""
    if scrutinee is None:
        return ERROR_T

    if isinstance(scrutinee, InterfaceT):
        remaining: list[Type] = []
        for variant_name in scrutinee.variants:
            vt = ctx.checker.types.get(variant_name)
            if vt is None:
                continue
            is_covered = any(type_eq(vt, c) for c in covered)
            if not is_covered:
                remaining.append(vt)
        if not remaining:
            return ERROR_T
        if len(remaining) == 1:
            return remaining[0]
        return normalize_union(remaining)

    if isinstance(scrutinee, UnionT):
        remaining2: list[Type] = []
        for m in scrutinee.members:
            is_covered = False
            for c in covered:
                if type_eq(m, c):
                    is_covered = True
                    break
                if isinstance(m, StructT) and isinstance(c, InterfaceT):
                    if m.parent == c.name:
                        is_covered = True
                        break
            if not is_covered:
                remaining2.append(m)
        if not remaining2:
            return ERROR_T
        if len(remaining2) == 1:
            return remaining2[0]
        return normalize_union(remaining2)

    return ERROR_T


# ============================================================
# TRY STATEMENT
# ============================================================


def _walk_try_stmt(stmt: TTryStmt, ctx: _ScopeCtx) -> None:
    _walk_stmts(stmt.body, ctx)
    for catch in stmt.catches:
        catch_ctx = _fork_ctx(ctx)
        if len(catch.types) == 1:
            catch_type = ctx.checker.resolve_type(catch.types[0])
        else:
            members: list[Type] = []
            for ct in catch.types:
                members.append(ctx.checker.resolve_type(ct))
            catch_type = normalize_union(members)
        catch_ctx.bindings[catch.name] = _BindingInfo(
            annotations=catch.annotations, declared_type=catch_type, is_param=False
        )
        _walk_stmts(catch.body, catch_ctx)
    if stmt.finally_body is not None:
        _walk_stmts(stmt.finally_body, ctx)


# ============================================================
# STAMP ANNOTATIONS
# ============================================================


def _stamp_bindings(ctx: _ScopeCtx) -> None:
    """Write final annotations onto binding declaration nodes."""
    for name, info in ctx.bindings.items():
        ann = info.annotations
        if info.binder_name is not None:
            bname = info.binder_name
            ann[f"scope.binder.{bname}.is_reassigned"] = (
                "true" if info.reassigned else "false"
            )
            ann[f"scope.binder.{bname}.is_const"] = (
                "false" if info.reassigned else "true"
            )
        else:
            ann["scope.is_reassigned"] = "true" if info.reassigned else "false"
            ann["scope.is_const"] = "false" if info.reassigned else "true"
        if info.is_param:
            ann["scope.is_modified"] = "true" if info.modified else "false"
            ann["scope.is_unused"] = "false" if info.used else "true"


# ============================================================
# FUNCTION ANALYSIS
# ============================================================


def _analyze_fn(decl: TFnDecl, ctx: _ScopeCtx, self_type: Type | None = None) -> None:
    """Analyze a single function declaration."""
    fn_ctx = _ScopeCtx(
        checker=ctx.checker,
        top_level_fns=ctx.top_level_fns,
        bindings={},
        narrowings={},
    )
    for p in decl.params:
        if p.typ is not None:
            fn_ctx.bindings[p.name] = _BindingInfo(
                annotations=p.annotations,
                declared_type=ctx.checker.resolve_type(p.typ),
                is_param=True,
            )
        elif p.name == "this" and self_type is not None:
            fn_ctx.bindings[p.name] = _BindingInfo(
                annotations=p.annotations, declared_type=self_type, is_param=True
            )
    _walk_stmts(decl.body, fn_ctx)
    _stamp_bindings(fn_ctx)


# ============================================================
# PUBLIC API
# ============================================================


def analyze_scope(module: TModule, checker: Checker) -> None:
    """Run scope analysis on all functions in the module."""
    top_level_fns: set[str] = set(checker.functions.keys())

    base_ctx = _ScopeCtx(
        checker=checker,
        top_level_fns=top_level_fns,
        bindings={},
        narrowings={},
    )

    for decl in module.decls:
        match decl:
            case TFnDecl():
                _analyze_fn(decl, base_ctx)
            case TStructDecl():
                st = checker.types.get(decl.name)
                for method in decl.methods:
                    _analyze_fn(method, base_ctx, self_type=st)
