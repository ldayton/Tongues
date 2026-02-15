"""Scope analysis pass for Taytsh IR.

Analyzes each function body independently, writing annotations onto AST nodes
for reassignment/constness, parameter modification/unused, narrowed types,
interface detection, and function reference detection.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..taytsh.ast import (
    TAssignStmt,
    TBinaryOp,
    TBoolLit,
    TByteLit,
    TBytesLit,
    TCall,
    TCatch,
    TDefault,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFloatLit,
    TFnDecl,
    TFnLit,
    TForStmt,
    TIfStmt,
    TIndex,
    TIntLit,
    TLetStmt,
    TListLit,
    TMapLit,
    TMatchStmt,
    TModule,
    TNilLit,
    TOpAssignStmt,
    TParam,
    TPatternEnum,
    TPatternNil,
    TPatternType,
    TRange,
    TReturnStmt,
    TRuneLit,
    TSetLit,
    TSlice,
    TStmt,
    TStringLit,
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
    BOOL_T,
    BUILTIN_NAMES,
    BYTE_T,
    BYTES_T,
    Checker,
    FLOAT_T,
    INT_T,
    InterfaceT,
    ListT,
    MapT,
    NIL_T,
    ERROR_T,
    RUNE_T,
    STRING_T,
    SetT,
    StructT,
    TupleT,
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

    node: TParam | TLetStmt | TForStmt | TPatternType | TDefault | TCatch
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
    new_narrowings = dict(ctx.narrowings)
    if extra_narrowings is not None:
        new_narrowings.update(extra_narrowings)
    return _ScopeCtx(
        checker=ctx.checker,
        top_level_fns=ctx.top_level_fns,
        bindings=ctx.bindings,
        narrowings=new_narrowings,
    )


# ============================================================
# TYPE RESOLUTION FOR EXPRESSIONS
# ============================================================


def _resolve_expr_type(expr: TExpr, ctx: _ScopeCtx) -> Type | None:
    """Resolve the type of an expression for iterable/call target analysis."""
    if isinstance(expr, TVar):
        if expr.name in ctx.narrowings:
            return ctx.narrowings[expr.name]
        if expr.name in ctx.bindings:
            return ctx.bindings[expr.name].declared_type
        if expr.name in ctx.checker.functions:
            return ctx.checker.functions[expr.name]
        if expr.name in ctx.checker.types:
            return ctx.checker.types[expr.name]
        return None
    if isinstance(expr, TIntLit):
        return INT_T
    if isinstance(expr, TFloatLit):
        return FLOAT_T
    if isinstance(expr, TBoolLit):
        return BOOL_T
    if isinstance(expr, TByteLit):
        return BYTE_T
    if isinstance(expr, TStringLit):
        return STRING_T
    if isinstance(expr, TRuneLit):
        return RUNE_T
    if isinstance(expr, TBytesLit):
        return BYTES_T
    if isinstance(expr, TNilLit):
        return NIL_T
    if isinstance(expr, TListLit):
        if len(expr.elements) > 0:
            elem_t = _resolve_expr_type(expr.elements[0], ctx)
            if elem_t is not None:
                return ListT(kind="list", element=elem_t)
        return None
    if isinstance(expr, TMapLit):
        if len(expr.entries) > 0:
            kt = _resolve_expr_type(expr.entries[0][0], ctx)
            vt = _resolve_expr_type(expr.entries[0][1], ctx)
            if kt is not None and vt is not None:
                return MapT(kind="map", key=kt, value=vt)
        return None
    if isinstance(expr, TSetLit):
        if len(expr.elements) > 0:
            elem_t = _resolve_expr_type(expr.elements[0], ctx)
            if elem_t is not None:
                return SetT(kind="set", element=elem_t)
        return None
    if isinstance(expr, TTupleLit):
        elems: list[Type] = []
        for e in expr.elements:
            t = _resolve_expr_type(e, ctx)
            if t is None:
                return None
            elems.append(t)

        return TupleT(kind="tuple", elements=elems)
    if isinstance(expr, TCall):
        return _resolve_call_return_type(expr, ctx)
    if isinstance(expr, TFieldAccess):
        obj_t = _resolve_expr_type(expr.obj, ctx)
        if obj_t is not None and isinstance(obj_t, StructT):
            if expr.field in obj_t.fields:
                return obj_t.fields[expr.field]
        return None
    if isinstance(expr, TIndex):
        obj_t = _resolve_expr_type(expr.obj, ctx)
        if obj_t is not None:
            if isinstance(obj_t, ListT):
                return obj_t.element
            if isinstance(obj_t, MapT):
                return obj_t.value
            if type_eq(obj_t, STRING_T):
                return RUNE_T
            if type_eq(obj_t, BYTES_T):
                return BYTE_T
        return None
    if isinstance(expr, TSlice):
        obj_t = _resolve_expr_type(expr.obj, ctx)
        if obj_t is not None:
            if isinstance(obj_t, ListT):
                return obj_t
            if type_eq(obj_t, STRING_T):
                return STRING_T
            if type_eq(obj_t, BYTES_T):
                return BYTES_T
        return None
    return None


def _resolve_call_return_type(expr: TCall, ctx: _ScopeCtx) -> Type | None:
    """Resolve return type of a call expression."""
    if isinstance(expr.func, TVar):
        name = expr.func.name
        if name in ctx.checker.functions:
            return ctx.checker.functions[name].ret
        if name in ctx.checker.types:
            return ctx.checker.types[name]
        # Check builtins — simplified, just handle the common ones
        if name in BUILTIN_NAMES:
            return _resolve_builtin_return(name, expr, ctx)
        return None
    if isinstance(expr.func, TFieldAccess):
        obj_t = _resolve_expr_type(expr.func.obj, ctx)
        if obj_t is not None and isinstance(obj_t, StructT):
            if expr.func.field in obj_t.methods:
                return obj_t.methods[expr.func.field].ret
        return None
    return None


def _resolve_builtin_return(name: str, expr: TCall, ctx: _ScopeCtx) -> Type | None:
    """Resolve return type for common built-in calls."""
    if name == "Len":
        return INT_T
    if name in (
        "Append",
        "Insert",
        "RemoveAt",
        "Delete",
        "Add",
        "Remove",
        "ReplaceSlice",
    ):
        return VOID_T
    if name == "Pop":
        if len(expr.args) > 0:
            t = _resolve_expr_type(expr.args[0].value, ctx)
            if t is not None and isinstance(t, ListT):
                return t.element
        return None
    if name in ("FloorDiv", "PythonMod"):
        if len(expr.args) > 0:
            return _resolve_expr_type(expr.args[0].value, ctx)
        return INT_T
    if name == "ToString":
        return STRING_T
    if name in ("Keys", "Values"):
        if len(expr.args) > 0:
            t = _resolve_expr_type(expr.args[0].value, ctx)
            if t is not None and isinstance(t, MapT):
                if name == "Keys":
                    return ListT(kind="list", element=t.key)
                return ListT(kind="list", element=t.value)
        return None
    if name == "Sorted" or name == "Reversed":
        if len(expr.args) > 0:
            return _resolve_expr_type(expr.args[0].value, ctx)
        return None
    if name in (
        "Concat",
        "Upper",
        "Lower",
        "Join",
        "Replace",
        "Trim",
        "TrimStart",
        "TrimEnd",
    ):
        return STRING_T
    if name in ("Split", "SplitN", "SplitWhitespace"):
        return ListT(kind="list", element=STRING_T)
    if name == "Args":
        return ListT(kind="list", element=STRING_T)
    return None


# ============================================================
# FOR-BINDER TYPE RESOLUTION
# ============================================================


def _resolve_for_binder_types(stmt: TForStmt, ctx: _ScopeCtx) -> dict[str, Type] | None:
    """Resolve types for for-loop binder variables. Returns name->type map or None."""
    if isinstance(stmt.iterable, TRange):
        result: dict[str, Type] = {}
        for b in stmt.binding:
            result[b] = INT_T
        return result
    iter_type = _resolve_expr_type(stmt.iterable, ctx)
    if iter_type is None:
        return None
    result2: dict[str, Type] = {}
    if isinstance(iter_type, ListT):
        if len(stmt.binding) == 1:
            result2[stmt.binding[0]] = iter_type.element
        elif len(stmt.binding) == 2:
            result2[stmt.binding[0]] = INT_T
            result2[stmt.binding[1]] = iter_type.element
    elif type_eq(iter_type, STRING_T):
        if len(stmt.binding) == 1:
            result2[stmt.binding[0]] = RUNE_T
        elif len(stmt.binding) == 2:
            result2[stmt.binding[0]] = INT_T
            result2[stmt.binding[1]] = RUNE_T
    elif type_eq(iter_type, BYTES_T):
        if len(stmt.binding) == 1:
            result2[stmt.binding[0]] = BYTE_T
        elif len(stmt.binding) == 2:
            result2[stmt.binding[0]] = INT_T
            result2[stmt.binding[1]] = BYTE_T
    elif isinstance(iter_type, MapT):
        if len(stmt.binding) == 1:
            result2[stmt.binding[0]] = iter_type.key
        elif len(stmt.binding) == 2:
            result2[stmt.binding[0]] = iter_type.key
            result2[stmt.binding[1]] = iter_type.value
    elif isinstance(iter_type, SetT):
        if len(stmt.binding) == 1:
            result2[stmt.binding[0]] = iter_type.element
    else:
        return None
    return result2 if len(result2) > 0 else None


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
        if len(expr.args) > 0:
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
                expr.annotations["scope.is_interface"] = True
            if not type_eq(effective_type, info.declared_type):
                expr.annotations["scope.narrowed_type"] = type_name(effective_type)
        elif name in ctx.top_level_fns and name not in BUILTIN_NAMES:
            expr.annotations["scope.is_function_ref"] = True
        return
    if isinstance(expr, TBinaryOp):
        _walk_expr(expr.left, ctx)
        _walk_expr(expr.right, ctx)
    elif isinstance(expr, TUnaryOp):
        _walk_expr(expr.operand, ctx)
    elif isinstance(expr, TTernary):
        _walk_expr(expr.cond, ctx)
        _walk_expr(expr.then_expr, ctx)
        _walk_expr(expr.else_expr, ctx)
    elif isinstance(expr, TFieldAccess):
        _walk_expr(expr.obj, ctx)
    elif isinstance(expr, TTupleAccess):
        _walk_expr(expr.obj, ctx)
    elif isinstance(expr, TIndex):
        _walk_expr(expr.obj, ctx)
        _walk_expr(expr.index, ctx)
    elif isinstance(expr, TSlice):
        _walk_expr(expr.obj, ctx)
        _walk_expr(expr.low, ctx)
        _walk_expr(expr.high, ctx)
    elif isinstance(expr, TCall):
        _check_call_mutation(expr, ctx)
        _walk_expr(expr.func, ctx)
        for a in expr.args:
            _walk_expr(a.value, ctx)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _walk_expr(e, ctx)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _walk_expr(k, ctx)
            _walk_expr(v, ctx)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _walk_expr(e, ctx)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _walk_expr(e, ctx)
    elif isinstance(expr, TFnLit):
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
            ctx.bindings[p.name] = _BindingInfo(node=p, declared_type=pt, is_param=True)
    if isinstance(expr.body, list):
        _walk_stmts(expr.body, ctx)
    else:
        _walk_expr(expr.body, ctx)
    _stamp_bindings(ctx)


# ============================================================
# WALK STATEMENTS
# ============================================================


def _walk_stmts(stmts: list[TStmt], ctx: _ScopeCtx) -> None:
    for s in stmts:
        _walk_stmt(s, ctx)


def _walk_stmt(stmt: TStmt, ctx: _ScopeCtx) -> None:
    if isinstance(stmt, TLetStmt):
        if stmt.value is not None:
            _walk_expr(stmt.value, ctx)
        declared_type = ctx.checker.resolve_type(stmt.typ)
        ctx.bindings[stmt.name] = _BindingInfo(
            node=stmt, declared_type=declared_type, is_param=False
        )

    elif isinstance(stmt, TAssignStmt):
        _walk_expr(stmt.value, ctx)
        _check_assign_target(stmt.target, ctx)
        # Walk the target for variable uses (field/index chains)
        _walk_assign_target_uses(stmt.target, ctx)
        # Map index assignment: m[k] = v mutates m if it's a param
        if isinstance(stmt.target, TIndex):
            base = _get_base_var(stmt.target.obj)
            if base is not None and base in ctx.bindings:
                info = ctx.bindings[base]
                if info.is_param:
                    info.modified = True

    elif isinstance(stmt, TOpAssignStmt):
        _walk_expr(stmt.value, ctx)
        if isinstance(stmt.target, TVar):
            name = stmt.target.name
            if name in ctx.bindings:
                info = ctx.bindings[name]
                info.reassigned = True
                if info.is_param:
                    info.modified = True
        else:
            # For field/index op-assign, it's mutation not reassignment
            base = _get_base_var(stmt.target)
            if base is not None and base in ctx.bindings:
                info = ctx.bindings[base]
                if info.is_param:
                    info.modified = True
        _walk_assign_target_uses(stmt.target, ctx)

    elif isinstance(stmt, TTupleAssignStmt):
        _walk_expr(stmt.value, ctx)
        for t in stmt.targets:
            _check_assign_target(t, ctx)

    elif isinstance(stmt, TReturnStmt):
        if stmt.value is not None:
            _walk_expr(stmt.value, ctx)

    elif isinstance(stmt, TThrowStmt):
        _walk_expr(stmt.expr, ctx)

    elif isinstance(stmt, TExprStmt):
        _walk_expr(stmt.expr, ctx)

    elif isinstance(stmt, TIfStmt):
        _walk_expr(stmt.cond, ctx)
        _walk_if_stmt(stmt, ctx)

    elif isinstance(stmt, TWhileStmt):
        _walk_expr(stmt.cond, ctx)
        _walk_stmts(stmt.body, ctx)

    elif isinstance(stmt, TForStmt):
        _walk_for_stmt(stmt, ctx)

    elif isinstance(stmt, TMatchStmt):
        _walk_match_stmt(stmt, ctx)

    elif isinstance(stmt, TTryStmt):
        _walk_try_stmt(stmt, ctx)


def _walk_assign_target_uses(target: TExpr, ctx: _ScopeCtx) -> None:
    """Walk assignment target sub-expressions for use tracking (not the top-level var)."""
    if isinstance(target, TFieldAccess):
        _walk_expr(target.obj, ctx)
    elif isinstance(target, TIndex):
        _walk_expr(target.obj, ctx)
        _walk_expr(target.index, ctx)
    elif isinstance(target, TTupleAccess):
        _walk_expr(target.obj, ctx)
    # TVar targets: don't count the target itself as a "use" for unused tracking


# ============================================================
# IF STATEMENT — NARROWING
# ============================================================


def _walk_if_stmt(stmt: TIfStmt, ctx: _ScopeCtx) -> None:
    """Handle if-stmt with potential nil narrowing."""
    narrowed_name: str | None = None
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
                    narrowed_name = name
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
    binder_types = _resolve_for_binder_types(stmt, ctx)
    for bname in stmt.binding:
        btype = binder_types.get(bname) if binder_types is not None else None
        if btype is None:
            btype = ERROR_T
        ctx.bindings[bname] = _BindingInfo(
            node=stmt,
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

    scrutinee_type = _resolve_expr_type(stmt.expr, ctx)
    covered_types: list[Type] = []

    for case in stmt.cases:
        pat = case.pattern
        case_ctx = _fork_ctx(ctx)

        if isinstance(pat, TPatternType):
            case_type = ctx.checker.resolve_type(pat.type_name)
            covered_types.append(case_type)
            case_ctx.bindings[pat.name] = _BindingInfo(
                node=pat, declared_type=case_type, is_param=False
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
        dflt = stmt.default
        dflt_ctx = _fork_ctx(ctx)
        if dflt.name is not None:
            residual = _compute_residual_type(scrutinee_type, covered_types, ctx)
            dflt_ctx.bindings[dflt.name] = _BindingInfo(
                node=dflt, declared_type=residual, is_param=False
            )
        _walk_stmts(dflt.body, dflt_ctx)
        if dflt.name is not None:
            iface = _detect_case_interface(dflt.name, dflt.body, dflt_ctx)
            dflt.annotations["scope.case_interface"] = iface


def _detect_case_interface(binding_name: str, body: list[TStmt], ctx: _ScopeCtx) -> str:
    """Detect if a case binding is used through an interface in the body.

    Returns the interface name or "" if none.
    """
    for stmt in body:
        result = _scan_stmt_for_interface_use(binding_name, stmt, ctx)
        if result:
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
        if r:
            return r
        return _scan_expr_for_interface_use(name, stmt.target, ctx)
    if isinstance(stmt, TOpAssignStmt):
        r = _scan_expr_for_interface_use(name, stmt.value, ctx)
        if r:
            return r
        return _scan_expr_for_interface_use(name, stmt.target, ctx)
    if isinstance(stmt, TTupleAssignStmt):
        r = _scan_expr_for_interface_use(name, stmt.value, ctx)
        if r:
            return r
        for t in stmt.targets:
            r = _scan_expr_for_interface_use(name, t, ctx)
            if r:
                return r
    if isinstance(stmt, TIfStmt):
        r = _scan_expr_for_interface_use(name, stmt.cond, ctx)
        if r:
            return r
        for s in stmt.then_body:
            r = _scan_stmt_for_interface_use(name, s, ctx)
            if r:
                return r
        if stmt.else_body is not None:
            for s in stmt.else_body:
                r = _scan_stmt_for_interface_use(name, s, ctx)
                if r:
                    return r
    if isinstance(stmt, TWhileStmt):
        r = _scan_expr_for_interface_use(name, stmt.cond, ctx)
        if r:
            return r
        for s in stmt.body:
            r = _scan_stmt_for_interface_use(name, s, ctx)
            if r:
                return r
    if isinstance(stmt, TForStmt):
        for s in stmt.body:
            r = _scan_stmt_for_interface_use(name, s, ctx)
            if r:
                return r
    if isinstance(stmt, TTryStmt):
        for s in stmt.body:
            r = _scan_stmt_for_interface_use(name, s, ctx)
            if r:
                return r
        for catch in stmt.catches:
            for s in catch.body:
                r = _scan_stmt_for_interface_use(name, s, ctx)
                if r:
                    return r
    return None


def _scan_expr_for_interface_use(name: str, expr: TExpr, ctx: _ScopeCtx) -> str | None:
    """Check if `name` is passed to a function parameter typed as an interface."""
    if isinstance(expr, TCall):
        # Check each argument: is it `name` passed to an interface-typed param?
        result = _check_call_interface_arg(name, expr, ctx)
        if result:
            return result
        # Recurse into sub-expressions
        r = _scan_expr_for_interface_use(name, expr.func, ctx)
        if r:
            return r
        for a in expr.args:
            r = _scan_expr_for_interface_use(name, a.value, ctx)
            if r:
                return r
        return None
    if isinstance(expr, TBinaryOp):
        r = _scan_expr_for_interface_use(name, expr.left, ctx)
        if r:
            return r
        return _scan_expr_for_interface_use(name, expr.right, ctx)
    if isinstance(expr, TUnaryOp):
        return _scan_expr_for_interface_use(name, expr.operand, ctx)
    if isinstance(expr, TTernary):
        r = _scan_expr_for_interface_use(name, expr.cond, ctx)
        if r:
            return r
        r = _scan_expr_for_interface_use(name, expr.then_expr, ctx)
        if r:
            return r
        return _scan_expr_for_interface_use(name, expr.else_expr, ctx)
    if isinstance(expr, TFieldAccess):
        return _scan_expr_for_interface_use(name, expr.obj, ctx)
    if isinstance(expr, TIndex):
        r = _scan_expr_for_interface_use(name, expr.obj, ctx)
        if r:
            return r
        return _scan_expr_for_interface_use(name, expr.index, ctx)
    if isinstance(expr, TSlice):
        r = _scan_expr_for_interface_use(name, expr.obj, ctx)
        if r:
            return r
        r = _scan_expr_for_interface_use(name, expr.low, ctx)
        if r:
            return r
        return _scan_expr_for_interface_use(name, expr.high, ctx)
    if isinstance(expr, TListLit):
        for e in expr.elements:
            r = _scan_expr_for_interface_use(name, e, ctx)
            if r:
                return r
    if isinstance(expr, TTupleLit):
        for e in expr.elements:
            r = _scan_expr_for_interface_use(name, e, ctx)
            if r:
                return r
    if isinstance(expr, TMapLit):
        for k, v in expr.entries:
            r = _scan_expr_for_interface_use(name, k, ctx)
            if r:
                return r
            r = _scan_expr_for_interface_use(name, v, ctx)
            if r:
                return r
    if isinstance(expr, TSetLit):
        for e in expr.elements:
            r = _scan_expr_for_interface_use(name, e, ctx)
            if r:
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
        obj_t = _resolve_expr_type(call.func.obj, ctx)
        if obj_t is not None and isinstance(obj_t, StructT):
            mname = call.func.field
            if mname in obj_t.methods:
                # Skip self param
                param_types = obj_t.methods[mname].params[1:]
    if param_types is None:
        return None
    for i, arg in enumerate(call.args):
        if isinstance(arg.value, TVar) and arg.value.name == name:
            if i < len(param_types) and isinstance(param_types[i], InterfaceT):
                return param_types[i].name
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
        if len(remaining) == 0:
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
        if len(remaining2) == 0:
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
            node=catch, declared_type=catch_type, is_param=False
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
        node = info.node
        if info.binder_name is not None:
            # For-binder: composite keys on the TForStmt node
            bname = info.binder_name
            node.annotations[f"scope.binder.{bname}.is_reassigned"] = info.reassigned
            node.annotations[f"scope.binder.{bname}.is_const"] = not info.reassigned
        else:
            node.annotations["scope.is_reassigned"] = info.reassigned
            node.annotations["scope.is_const"] = not info.reassigned
        if info.is_param:
            node.annotations["scope.is_modified"] = info.modified
            node.annotations["scope.is_unused"] = not info.used


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
            pt = ctx.checker.resolve_type(p.typ)
        elif p.name == "self" and self_type is not None:
            pt = self_type
        else:
            continue
        fn_ctx.bindings[p.name] = _BindingInfo(node=p, declared_type=pt, is_param=True)
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
        if isinstance(decl, TFnDecl):
            _analyze_fn(decl, base_ctx)
        elif isinstance(decl, TStructDecl):
            st = checker.types.get(decl.name)
            for method in decl.methods:
                _analyze_fn(method, base_ctx, self_type=st)
