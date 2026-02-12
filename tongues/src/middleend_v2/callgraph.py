"""Callgraph analysis pass for Taytsh IR.

Inter-procedural pass that computes throw sets, recursion detection (SCCs),
and tail call identification for every function and method in the module.
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
    TTryStmt,
    TTupleAssignStmt,
    TTupleLit,
    TUnaryOp,
    TVar,
    TWhileStmt,
)
from ..taytsh.check import (
    BUILTIN_NAMES,
    Checker,
    InterfaceT,
    MapT,
    StructT,
    Type,
    type_eq,
    BYTE_T,
    INT_T,
)

# Additional runtime builtins not in BUILTIN_NAMES
_EXTRA_BUILTINS = {"Floor", "Ceil", "ReadFile", "WriteFile", "Sorted", "Reverse"}

# Combined set for call target resolution
_ALL_BUILTINS = BUILTIN_NAMES | _EXTRA_BUILTINS


# ============================================================
# BUILT-IN THROW TABLES
# ============================================================

# Built-in functions with known throw behaviour
BUILTIN_THROWS: dict[str, set[str]] = {
    "ParseInt": {"ValueError"},
    "ParseFloat": {"ValueError"},
    "FloatToInt": {"ValueError"},
    "Round": {"ValueError"},
    "Floor": {"ValueError"},
    "Ceil": {"ValueError"},
    "Unwrap": {"NilError"},
    "Assert": {"AssertError"},
    "Pop": {"IndexError"},
    "ReadFile": {"IOError"},
    "WriteFile": {"IOError"},
}

# Operators that throw ZeroDivisionError on int/byte division/modulo
_DIV_OPS = {"/", "%"}

# Strict-math operators that can throw ValueError on int overflow
_STRICT_INT_OPS = {"+", "-", "*"}


def _add_throws(
    types: set[str], throws: set[str], caught_filter: set[str] | None
) -> None:
    if caught_filter is not None:
        throws.update(types - caught_filter)
    else:
        throws.update(types)


# ============================================================
# TYPE RESOLUTION FOR EXPRESSIONS
# ============================================================


class _TypeResolver:
    """Minimal expression type resolver for callgraph's needs.

    We need to know the type of an expression to determine:
    - Whether an index is on a map (KeyError) vs list/string/bytes (IndexError)
    - The receiver type for method calls (to resolve struct methods and interface dispatch)
    - Whether division operands are int/byte (ZeroDivisionError) vs float (no throw)
    """

    def __init__(self, checker: Checker, fn: TFnDecl, struct_type: StructT | None):
        self.checker = checker
        self.locals: dict[str, Type] = {}
        self.catch_vars: dict[str, set[str]] = {}
        for p in fn.params:
            if p.typ is not None:
                self.locals[p.name] = checker.resolve_type(p.typ)
            elif p.name == "self" and struct_type is not None:
                self.locals[p.name] = struct_type

    def resolve(self, expr: TExpr) -> Type | None:
        if isinstance(expr, TVar):
            if expr.name in self.locals:
                return self.locals[expr.name]
            if expr.name in self.checker.functions:
                return self.checker.functions[expr.name]
            if expr.name in self.checker.types:
                return self.checker.types[expr.name]
            return None
        if isinstance(expr, TCall):
            # Struct constructor: Foo(...) returns Foo
            if isinstance(expr.func, TVar):
                t = self.checker.types.get(expr.func.name)
                if t is not None and isinstance(t, StructT):
                    return t
            return None
        if isinstance(expr, TFieldAccess):
            obj_t = self.resolve(expr.obj)
            if obj_t is not None and isinstance(obj_t, StructT):
                if expr.field in obj_t.fields:
                    return obj_t.fields[expr.field]
            return None
        if isinstance(expr, TIndex):
            obj_t = self.resolve(expr.obj)
            if obj_t is not None and isinstance(obj_t, MapT):
                return obj_t.value
            return None
        return None

    def register_let(self, name: str, typ: Type) -> None:
        self.locals[name] = typ


# ============================================================
# STEP 1: BUILD CALL GRAPH
# ============================================================


def _build_call_graph(
    module: TModule, checker: Checker
) -> tuple[
    dict[str, TFnDecl],
    dict[str, set[str]],
    dict[str, StructT | None],
]:
    """Build call graph edges.

    Returns:
        fn_decls: key -> TFnDecl
        edges: key -> set of callee keys
        fn_structs: key -> StructT or None (for methods)
    """
    fn_decls: dict[str, TFnDecl] = {}
    edges: dict[str, set[str]] = {}
    fn_structs: dict[str, StructT | None] = {}

    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            key = decl.name
            fn_decls[key] = decl
            fn_structs[key] = None
            edges[key] = set()
        elif isinstance(decl, TStructDecl):
            st = checker.types.get(decl.name)
            for method in decl.methods:
                key = f"{decl.name}.{method.name}"
                fn_decls[key] = method
                fn_structs[key] = st if isinstance(st, StructT) else None
                edges[key] = set()

    for key, decl in fn_decls.items():
        resolver = _TypeResolver(checker, decl, fn_structs[key])
        _collect_edges(decl.body, key, edges, fn_decls, checker, resolver)

    return fn_decls, edges, fn_structs


def _collect_edges(
    stmts: list[TStmt],
    caller: str,
    edges: dict[str, set[str]],
    fn_decls: dict[str, TFnDecl],
    checker: Checker,
    resolver: _TypeResolver,
) -> None:
    for stmt in stmts:
        _collect_edges_stmt(stmt, caller, edges, fn_decls, checker, resolver)


def _collect_edges_stmt(
    stmt: TStmt,
    caller: str,
    edges: dict[str, set[str]],
    fn_decls: dict[str, TFnDecl],
    checker: Checker,
    resolver: _TypeResolver,
) -> None:
    if isinstance(stmt, TExprStmt):
        _collect_edges_expr(stmt.expr, caller, edges, fn_decls, checker, resolver)
    elif isinstance(stmt, TReturnStmt) and stmt.value is not None:
        _collect_edges_expr(stmt.value, caller, edges, fn_decls, checker, resolver)
    elif isinstance(stmt, TThrowStmt):
        _collect_edges_expr(stmt.expr, caller, edges, fn_decls, checker, resolver)
    elif isinstance(stmt, TLetStmt):
        if stmt.value is not None:
            _collect_edges_expr(stmt.value, caller, edges, fn_decls, checker, resolver)
        resolver.register_let(stmt.name, checker.resolve_type(stmt.typ))
    elif isinstance(stmt, TAssignStmt):
        _collect_edges_expr(stmt.target, caller, edges, fn_decls, checker, resolver)
        _collect_edges_expr(stmt.value, caller, edges, fn_decls, checker, resolver)
    elif isinstance(stmt, TOpAssignStmt):
        _collect_edges_expr(stmt.target, caller, edges, fn_decls, checker, resolver)
        _collect_edges_expr(stmt.value, caller, edges, fn_decls, checker, resolver)
    elif isinstance(stmt, TTupleAssignStmt):
        for t in stmt.targets:
            _collect_edges_expr(t, caller, edges, fn_decls, checker, resolver)
        _collect_edges_expr(stmt.value, caller, edges, fn_decls, checker, resolver)
    elif isinstance(stmt, TIfStmt):
        _collect_edges_expr(stmt.cond, caller, edges, fn_decls, checker, resolver)
        _collect_edges(stmt.then_body, caller, edges, fn_decls, checker, resolver)
        if stmt.else_body is not None:
            _collect_edges(stmt.else_body, caller, edges, fn_decls, checker, resolver)
    elif isinstance(stmt, TWhileStmt):
        _collect_edges_expr(stmt.cond, caller, edges, fn_decls, checker, resolver)
        _collect_edges(stmt.body, caller, edges, fn_decls, checker, resolver)
    elif isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                _collect_edges_expr(a, caller, edges, fn_decls, checker, resolver)
        else:
            _collect_edges_expr(
                stmt.iterable, caller, edges, fn_decls, checker, resolver
            )
        _collect_edges(stmt.body, caller, edges, fn_decls, checker, resolver)
    elif isinstance(stmt, TMatchStmt):
        _collect_edges_expr(stmt.expr, caller, edges, fn_decls, checker, resolver)
        for case in stmt.cases:
            _collect_edges(case.body, caller, edges, fn_decls, checker, resolver)
        if stmt.default is not None:
            _collect_edges(
                stmt.default.body, caller, edges, fn_decls, checker, resolver
            )
    elif isinstance(stmt, TTryStmt):
        _collect_edges(stmt.body, caller, edges, fn_decls, checker, resolver)
        for catch in stmt.catches:
            _collect_edges(catch.body, caller, edges, fn_decls, checker, resolver)
        if stmt.finally_body is not None:
            _collect_edges(
                stmt.finally_body, caller, edges, fn_decls, checker, resolver
            )


def _collect_edges_expr(
    expr: TExpr,
    caller: str,
    edges: dict[str, set[str]],
    fn_decls: dict[str, TFnDecl],
    checker: Checker,
    resolver: _TypeResolver,
) -> None:
    if isinstance(expr, TCall):
        for callee_key in _resolve_all_call_targets(expr, fn_decls, checker, resolver):
            edges[caller].add(callee_key)
        # Also walk args
        _collect_edges_expr(expr.func, caller, edges, fn_decls, checker, resolver)
        for arg in expr.args:
            _collect_edges_expr(arg.value, caller, edges, fn_decls, checker, resolver)
    elif isinstance(expr, TBinaryOp):
        _collect_edges_expr(expr.left, caller, edges, fn_decls, checker, resolver)
        _collect_edges_expr(expr.right, caller, edges, fn_decls, checker, resolver)
    elif isinstance(expr, TUnaryOp):
        _collect_edges_expr(expr.operand, caller, edges, fn_decls, checker, resolver)
    elif isinstance(expr, TTernary):
        _collect_edges_expr(expr.cond, caller, edges, fn_decls, checker, resolver)
        _collect_edges_expr(expr.then_expr, caller, edges, fn_decls, checker, resolver)
        _collect_edges_expr(expr.else_expr, caller, edges, fn_decls, checker, resolver)
    elif isinstance(expr, TFieldAccess):
        _collect_edges_expr(expr.obj, caller, edges, fn_decls, checker, resolver)
    elif isinstance(expr, TIndex):
        _collect_edges_expr(expr.obj, caller, edges, fn_decls, checker, resolver)
        _collect_edges_expr(expr.index, caller, edges, fn_decls, checker, resolver)
    elif isinstance(expr, TSlice):
        _collect_edges_expr(expr.obj, caller, edges, fn_decls, checker, resolver)
        _collect_edges_expr(expr.low, caller, edges, fn_decls, checker, resolver)
        _collect_edges_expr(expr.high, caller, edges, fn_decls, checker, resolver)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _collect_edges_expr(e, caller, edges, fn_decls, checker, resolver)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _collect_edges_expr(k, caller, edges, fn_decls, checker, resolver)
            _collect_edges_expr(v, caller, edges, fn_decls, checker, resolver)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _collect_edges_expr(e, caller, edges, fn_decls, checker, resolver)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _collect_edges_expr(e, caller, edges, fn_decls, checker, resolver)
    elif isinstance(expr, TFnLit):
        if isinstance(expr.body, list):
            _collect_edges(expr.body, caller, edges, fn_decls, checker, resolver)
        else:
            _collect_edges_expr(expr.body, caller, edges, fn_decls, checker, resolver)


def _resolve_all_call_targets(
    call: TCall,
    fn_decls: dict[str, TFnDecl],
    checker: Checker,
    resolver: _TypeResolver,
) -> list[str]:
    """Resolve a call to all possible function keys (handles interface dispatch)."""
    func = call.func
    if isinstance(func, TVar):
        name = func.name
        if name in _ALL_BUILTINS and name not in checker.functions:
            return []
        t = checker.types.get(name)
        if t is not None and isinstance(t, StructT):
            return []
        if name in fn_decls:
            return [name]
        return []
    if isinstance(func, TFieldAccess):
        obj_t = resolver.resolve(func.obj)
        if obj_t is not None and isinstance(obj_t, StructT):
            method_key = f"{obj_t.name}.{func.field}"
            if method_key in fn_decls:
                return [method_key]
            if obj_t.parent is not None:
                iface = checker.types.get(obj_t.parent)
                if iface is not None and isinstance(iface, InterfaceT):
                    targets = []
                    for variant_name in iface.variants:
                        vkey = f"{variant_name}.{func.field}"
                        if vkey in fn_decls:
                            targets.append(vkey)
                    if targets:
                        return targets
        if obj_t is not None and isinstance(obj_t, InterfaceT):
            targets = []
            for variant_name in obj_t.variants:
                vkey = f"{variant_name}.{func.field}"
                if vkey in fn_decls:
                    targets.append(vkey)
            return targets
        return []
    return []


# ============================================================
# STEP 2: DETECT RECURSION (SCCs via Tarjan's)
# ============================================================


class _TarjanState:
    """Mutable state for Tarjan's SCC algorithm."""

    def __init__(self, edges: dict[str, set[str]]) -> None:
        self.edges = edges
        self.index: int = 0
        self.stack: list[str] = []
        self.on_stack: set[str] = set()
        self.indices: dict[str, int] = {}
        self.lowlinks: dict[str, int] = {}
        self.result: list[list[str]] = []


def _strongconnect(v: str, st: _TarjanState) -> None:
    st.indices[v] = st.index
    st.lowlinks[v] = st.index
    st.index += 1
    st.stack.append(v)
    st.on_stack.add(v)
    for w in st.edges.get(v, set()):
        if w not in st.indices:
            if w in st.edges:
                _strongconnect(w, st)
                st.lowlinks[v] = min(st.lowlinks[v], st.lowlinks[w])
        elif w in st.on_stack:
            st.lowlinks[v] = min(st.lowlinks[v], st.indices[w])
    if st.lowlinks[v] == st.indices[v]:
        scc: list[str] = []
        while True:
            w = st.stack.pop()
            st.on_stack.discard(w)
            scc.append(w)
            if w == v:
                break
        st.result.append(scc)


def _compute_sccs(keys: list[str], edges: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's SCC algorithm. Returns SCCs in reverse topological order."""
    st = _TarjanState(edges)
    for v in keys:
        if v not in st.indices:
            _strongconnect(v, st)
    return st.result


def _detect_recursion(
    fn_decls: dict[str, TFnDecl],
    edges: dict[str, set[str]],
) -> tuple[list[list[str]], dict[str, int]]:
    """Detect SCCs and write recursion annotations.

    Returns SCCs in reverse topo order and a mapping from fn key to SCC index.
    """
    keys = list(fn_decls.keys())
    sccs = _compute_sccs(keys, edges)
    key_to_scc: dict[str, int] = {}
    for i, scc in enumerate(sccs):
        for key in scc:
            key_to_scc[key] = i

    scc_counter = 0
    for i, scc in enumerate(sccs):
        is_recursive = False
        if len(scc) > 1:
            is_recursive = True
        elif len(scc) == 1:
            key = scc[0]
            if key in edges.get(key, set()):
                is_recursive = True

        if is_recursive:
            group_id = f"scc:{scc_counter}"
            scc_counter += 1
        else:
            group_id = ""

        for key in scc:
            decl = fn_decls[key]
            decl.annotations["callgraph.is_recursive"] = is_recursive
            decl.annotations["callgraph.recursive_group"] = group_id

    return sccs, key_to_scc


# ============================================================
# STEP 3: PROPAGATE THROW TYPES
# ============================================================


def _check_op_throws(
    op: str,
    left_expr: TExpr,
    throws: set[str],
    resolver: _TypeResolver,
    strict_math: bool,
    caught_filter: set[str] | None,
) -> None:
    """Check if an operator can throw based on operand types."""
    left_t = resolver.resolve(left_expr)
    if op in _DIV_OPS:
        if left_t is not None and (type_eq(left_t, INT_T) or type_eq(left_t, BYTE_T)):
            _add_throws({"ZeroDivisionError"}, throws, caught_filter)
    if strict_math and op == "%" and left_t is not None:
        if not type_eq(left_t, INT_T) and not type_eq(left_t, BYTE_T):
            _add_throws({"ValueError"}, throws, caught_filter)
    if strict_math and op in _STRICT_INT_OPS:
        if left_t is None or type_eq(left_t, INT_T):
            _add_throws({"ValueError"}, throws, caught_filter)
    if strict_math and op == "<<":
        if left_t is None or type_eq(left_t, INT_T):
            _add_throws({"ValueError"}, throws, caught_filter)


def _propagate_throws(
    sccs: list[list[str]],
    fn_decls: dict[str, TFnDecl],
    fn_structs: dict[str, StructT | None],
    edges: dict[str, set[str]],
    checker: Checker,
    strict_math: bool,
) -> dict[str, set[str]]:
    """Propagate throw types through the call graph in reverse topo order."""
    throw_sets: dict[str, set[str]] = {}

    for scc in sccs:
        scc_set = set(scc)
        for key in scc:
            decl = fn_decls[key]
            resolver = _TypeResolver(checker, decl, fn_structs[key])
            throws: set[str] = set()
            _collect_fn_throws(
                decl.body,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                throw_sets,
                None,
            )
            throw_sets[key] = throws

        # Fixed point for SCC
        if len(scc) > 1 or (len(scc) == 1 and scc[0] in edges.get(scc[0], set())):
            changed = True
            while changed:
                changed = False
                for key in scc:
                    before = len(throw_sets[key])
                    for callee in edges.get(key, set()):
                        if callee in scc_set and callee in throw_sets:
                            throw_sets[key] |= throw_sets[callee]
                    if len(throw_sets[key]) > before:
                        changed = True

    # Write annotations
    for key, fn_throws in throw_sets.items():
        decl = fn_decls[key]
        decl.annotations["callgraph.throws"] = ";".join(sorted(fn_throws))

    return throw_sets


def _collect_fn_throws(
    stmts: list[TStmt],
    throws: set[str],
    checker: Checker,
    resolver: _TypeResolver,
    fn_decls: dict[str, TFnDecl],
    strict_math: bool,
    callee_throws: dict[str, set[str]],
    caught_filter: set[str] | None,
) -> None:
    """Collect throws including transitive callee throws, with try/catch filtering."""
    for stmt in stmts:
        _collect_fn_throws_stmt(
            stmt,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )


def _collect_fn_throws_stmt(
    stmt: TStmt,
    throws: set[str],
    checker: Checker,
    resolver: _TypeResolver,
    fn_decls: dict[str, TFnDecl],
    strict_math: bool,
    callee_throws: dict[str, set[str]],
    caught_filter: set[str] | None,
) -> None:
    if isinstance(stmt, TThrowStmt):
        if isinstance(stmt.expr, TCall) and isinstance(stmt.expr.func, TVar):
            _add_throws({stmt.expr.func.name}, throws, caught_filter)
        elif isinstance(stmt.expr, TVar):
            name = stmt.expr.name
            if name in resolver.catch_vars:
                _add_throws(resolver.catch_vars[name], throws, caught_filter)
            else:
                _add_throws({name}, throws, caught_filter)
        _collect_fn_throws_expr(
            stmt.expr,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
    elif isinstance(stmt, TExprStmt):
        _collect_fn_throws_expr(
            stmt.expr,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
    elif isinstance(stmt, TReturnStmt) and stmt.value is not None:
        _collect_fn_throws_expr(
            stmt.value,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
    elif isinstance(stmt, TLetStmt):
        if stmt.value is not None:
            _collect_fn_throws_expr(
                stmt.value,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
        resolver.register_let(stmt.name, checker.resolve_type(stmt.typ))
    elif isinstance(stmt, TAssignStmt):
        _collect_fn_throws_expr(
            stmt.target,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _collect_fn_throws_expr(
            stmt.value,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
    elif isinstance(stmt, TOpAssignStmt):
        _collect_fn_throws_expr(
            stmt.target,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _collect_fn_throws_expr(
            stmt.value,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _check_op_throws(
            stmt.op.rstrip("="),
            stmt.target,
            throws,
            resolver,
            strict_math,
            caught_filter,
        )
    elif isinstance(stmt, TTupleAssignStmt):
        for t in stmt.targets:
            _collect_fn_throws_expr(
                t,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
        _collect_fn_throws_expr(
            stmt.value,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
    elif isinstance(stmt, TIfStmt):
        _collect_fn_throws_expr(
            stmt.cond,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _collect_fn_throws(
            stmt.then_body,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        if stmt.else_body is not None:
            _collect_fn_throws(
                stmt.else_body,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
    elif isinstance(stmt, TWhileStmt):
        _collect_fn_throws_expr(
            stmt.cond,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _collect_fn_throws(
            stmt.body,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
    elif isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                _collect_fn_throws_expr(
                    a,
                    throws,
                    checker,
                    resolver,
                    fn_decls,
                    strict_math,
                    callee_throws,
                    caught_filter,
                )
        else:
            _collect_fn_throws_expr(
                stmt.iterable,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
        _collect_fn_throws(
            stmt.body,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
    elif isinstance(stmt, TMatchStmt):
        _collect_fn_throws_expr(
            stmt.expr,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        for case in stmt.cases:
            _collect_fn_throws(
                case.body,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
        if stmt.default is not None:
            _collect_fn_throws(
                stmt.default.body,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
    elif isinstance(stmt, TTryStmt):
        _collect_fn_throws_try(
            stmt,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )


def _collect_fn_throws_try(
    stmt: TTryStmt,
    throws: set[str],
    checker: Checker,
    resolver: _TypeResolver,
    fn_decls: dict[str, TFnDecl],
    strict_math: bool,
    callee_throws: dict[str, set[str]],
    outer_filter: set[str] | None,
) -> None:
    has_catch_all = False
    caught_types: set[str] = set()
    for catch in stmt.catches:
        if not catch.types:
            has_catch_all = True
        else:
            for ct in catch.types:
                resolved = checker.resolve_type(ct)
                if isinstance(resolved, StructT):
                    caught_types.add(resolved.name)

    # Collect throws from try body
    try_throws: set[str] = set()
    _collect_fn_throws(
        stmt.body,
        try_throws,
        checker,
        resolver,
        fn_decls,
        strict_math,
        callee_throws,
        None,
    )

    if has_catch_all:
        residual = set[str]()
    else:
        residual = try_throws - caught_types

    if outer_filter is not None:
        throws.update(residual - outer_filter)
    else:
        throws.update(residual)

    # Process catch bodies
    for i, catch in enumerate(stmt.catches):
        if not catch.types:
            preceding_caught: set[str] = set()
            for j in range(i):
                for ct in stmt.catches[j].types:
                    resolved = checker.resolve_type(ct)
                    if isinstance(resolved, StructT):
                        preceding_caught.add(resolved.name)
            catch_handles = try_throws - preceding_caught
        else:
            catch_handles = set()
            for ct in catch.types:
                resolved = checker.resolve_type(ct)
                if isinstance(resolved, StructT):
                    catch_handles.add(resolved.name)

        prev = resolver.catch_vars.get(catch.name)
        resolver.catch_vars[catch.name] = catch_handles
        _collect_fn_throws(
            catch.body,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            outer_filter,
        )
        if prev is not None:
            resolver.catch_vars[catch.name] = prev
        else:
            resolver.catch_vars.pop(catch.name, None)

    if stmt.finally_body is not None:
        _collect_fn_throws(
            stmt.finally_body,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            outer_filter,
        )


def _collect_fn_throws_expr(
    expr: TExpr,
    throws: set[str],
    checker: Checker,
    resolver: _TypeResolver,
    fn_decls: dict[str, TFnDecl],
    strict_math: bool,
    callee_throws: dict[str, set[str]],
    caught_filter: set[str] | None,
) -> None:
    if isinstance(expr, TCall):
        if isinstance(expr.func, TVar):
            name = expr.func.name
            if name in BUILTIN_THROWS and name not in checker.functions:
                _add_throws(BUILTIN_THROWS[name], throws, caught_filter)
            if strict_math and name == "Sorted" and name not in checker.functions:
                _add_throws({"ValueError"}, throws, caught_filter)
            if strict_math and name == "Pow" and name not in checker.functions:
                _add_throws({"ValueError"}, throws, caught_filter)
            # Transitive throws from user-defined callee
            targets = _resolve_all_call_targets(expr, fn_decls, checker, resolver)
            for target in targets:
                if target in callee_throws:
                    _add_throws(callee_throws[target], throws, caught_filter)
        elif isinstance(expr.func, TFieldAccess):
            targets = _resolve_all_call_targets(expr, fn_decls, checker, resolver)
            for target in targets:
                if target in callee_throws:
                    _add_throws(callee_throws[target], throws, caught_filter)
            _collect_fn_throws_expr(
                expr.func.obj,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
        else:
            # Function-value call — conservative: union all throw sets
            all_throws: set[str] = set()
            for t in callee_throws.values():
                all_throws |= t
            _add_throws(all_throws, throws, caught_filter)
        for arg in expr.args:
            _collect_fn_throws_expr(
                arg.value,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
    elif isinstance(expr, TBinaryOp):
        _collect_fn_throws_expr(
            expr.left,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _collect_fn_throws_expr(
            expr.right,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _check_op_throws(
            expr.op, expr.left, throws, resolver, strict_math, caught_filter
        )
    elif isinstance(expr, TUnaryOp):
        _collect_fn_throws_expr(
            expr.operand,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        if strict_math and expr.op == "-":
            op_t = resolver.resolve(expr.operand)
            if op_t is not None and type_eq(op_t, INT_T):
                _add_throws({"ValueError"}, throws, caught_filter)
    elif isinstance(expr, TTernary):
        _collect_fn_throws_expr(
            expr.cond,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _collect_fn_throws_expr(
            expr.then_expr,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _collect_fn_throws_expr(
            expr.else_expr,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
    elif isinstance(expr, TIndex):
        _collect_fn_throws_expr(
            expr.obj,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _collect_fn_throws_expr(
            expr.index,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        obj_t = resolver.resolve(expr.obj)
        if obj_t is not None and isinstance(obj_t, MapT):
            _add_throws({"KeyError"}, throws, caught_filter)
        else:
            _add_throws({"IndexError"}, throws, caught_filter)
    elif isinstance(expr, TSlice):
        _collect_fn_throws_expr(
            expr.obj,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _collect_fn_throws_expr(
            expr.low,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _collect_fn_throws_expr(
            expr.high,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
        _add_throws({"IndexError"}, throws, caught_filter)
    elif isinstance(expr, TFieldAccess):
        _collect_fn_throws_expr(
            expr.obj,
            throws,
            checker,
            resolver,
            fn_decls,
            strict_math,
            callee_throws,
            caught_filter,
        )
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _collect_fn_throws_expr(
                e,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _collect_fn_throws_expr(
                k,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
            _collect_fn_throws_expr(
                v,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _collect_fn_throws_expr(
                e,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _collect_fn_throws_expr(
                e,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
    elif isinstance(expr, TFnLit):
        if isinstance(expr.body, list):
            _collect_fn_throws(
                expr.body,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )
        else:
            _collect_fn_throws_expr(
                expr.body,
                throws,
                checker,
                resolver,
                fn_decls,
                strict_math,
                callee_throws,
                caught_filter,
            )


# ============================================================
# STEP 4: DETECT TAIL CALLS
# ============================================================


def _detect_tail_calls(fn_decls: dict[str, TFnDecl]) -> None:
    """Walk each function body marking calls in tail position."""
    for decl in fn_decls.values():
        _walk_tail_stmts(decl.body, tail=True)


def _walk_tail_stmts(stmts: list[TStmt], *, tail: bool) -> None:
    """Walk statements; only the last statement can be in tail position."""
    for i, stmt in enumerate(stmts):
        is_last = i == len(stmts) - 1
        _walk_tail_stmt(stmt, tail=tail and is_last)


def _walk_tail_stmt(stmt: TStmt, *, tail: bool) -> None:
    if isinstance(stmt, TReturnStmt):
        if stmt.value is not None:
            _walk_tail_expr(stmt.value, tail=tail)
    elif isinstance(stmt, TExprStmt):
        # A bare expression statement at the end of a function body
        # (void function) — the call is in tail position
        _walk_tail_expr(stmt.expr, tail=tail)
    elif isinstance(stmt, TIfStmt):
        _walk_tail_expr(stmt.cond, tail=False)
        _walk_tail_stmts(stmt.then_body, tail=tail)
        if stmt.else_body is not None:
            _walk_tail_stmts(stmt.else_body, tail=tail)
    elif isinstance(stmt, TMatchStmt):
        _walk_tail_expr(stmt.expr, tail=False)
        for case in stmt.cases:
            _walk_tail_stmts(case.body, tail=tail)
        if stmt.default is not None:
            _walk_tail_stmts(stmt.default.body, tail=tail)
    elif isinstance(stmt, TTryStmt):
        # try body: NEVER in tail position
        _walk_tail_stmts(stmt.body, tail=False)
        if stmt.finally_body is not None:
            # With finally: catch bodies NOT in tail position, finally IS
            for catch in stmt.catches:
                _walk_tail_stmts(catch.body, tail=False)
            _walk_tail_stmts(stmt.finally_body, tail=tail)
        else:
            # Without finally: catch bodies inherit tail position
            for catch in stmt.catches:
                _walk_tail_stmts(catch.body, tail=tail)
    elif isinstance(stmt, (TWhileStmt, TForStmt)):
        # Loop bodies are NEVER in tail position
        if isinstance(stmt, TWhileStmt):
            _walk_tail_expr(stmt.cond, tail=False)
        elif isinstance(stmt, TForStmt):
            if isinstance(stmt.iterable, TRange):
                for a in stmt.iterable.args:
                    _walk_tail_expr(a, tail=False)
            else:
                _walk_tail_expr(stmt.iterable, tail=False)
        _walk_tail_stmts(stmt.body, tail=False)
    elif isinstance(stmt, TLetStmt) and stmt.value is not None:
        _walk_tail_expr(stmt.value, tail=False)
    elif isinstance(stmt, TAssignStmt):
        _walk_tail_expr(stmt.target, tail=False)
        _walk_tail_expr(stmt.value, tail=False)
    elif isinstance(stmt, TOpAssignStmt):
        _walk_tail_expr(stmt.target, tail=False)
        _walk_tail_expr(stmt.value, tail=False)
    elif isinstance(stmt, TTupleAssignStmt):
        for t in stmt.targets:
            _walk_tail_expr(t, tail=False)
        _walk_tail_expr(stmt.value, tail=False)
    elif isinstance(stmt, TThrowStmt):
        _walk_tail_expr(stmt.expr, tail=False)


def _walk_tail_expr(expr: TExpr, *, tail: bool) -> None:
    if isinstance(expr, TCall):
        if tail:
            expr.annotations["callgraph.is_tail_call"] = True
        # Walk subexpressions — never in tail position
        if isinstance(expr.func, TFieldAccess):
            _walk_tail_expr(expr.func.obj, tail=False)
        elif not isinstance(expr.func, TVar):
            _walk_tail_expr(expr.func, tail=False)
        for arg in expr.args:
            _walk_tail_expr(arg.value, tail=False)
    elif isinstance(expr, TTernary):
        _walk_tail_expr(expr.cond, tail=False)
        _walk_tail_expr(expr.then_expr, tail=tail)
        _walk_tail_expr(expr.else_expr, tail=tail)
    elif isinstance(expr, TBinaryOp):
        _walk_tail_expr(expr.left, tail=False)
        _walk_tail_expr(expr.right, tail=False)
    elif isinstance(expr, TUnaryOp):
        _walk_tail_expr(expr.operand, tail=False)
    elif isinstance(expr, TFieldAccess):
        _walk_tail_expr(expr.obj, tail=False)
    elif isinstance(expr, TIndex):
        _walk_tail_expr(expr.obj, tail=False)
        _walk_tail_expr(expr.index, tail=False)
    elif isinstance(expr, TSlice):
        _walk_tail_expr(expr.obj, tail=False)
        _walk_tail_expr(expr.low, tail=False)
        _walk_tail_expr(expr.high, tail=False)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _walk_tail_expr(e, tail=False)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _walk_tail_expr(k, tail=False)
            _walk_tail_expr(v, tail=False)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _walk_tail_expr(e, tail=False)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _walk_tail_expr(e, tail=False)
    elif isinstance(expr, TFnLit):
        if isinstance(expr.body, list):
            _walk_tail_stmts(expr.body, tail=False)
        else:
            _walk_tail_expr(expr.body, tail=False)


# ============================================================
# PUBLIC API
# ============================================================


def analyze_callgraph(module: TModule, checker: Checker) -> None:
    """Run callgraph analysis on all functions in the module."""
    fn_decls, edges, fn_structs = _build_call_graph(module, checker)
    sccs, _ = _detect_recursion(fn_decls, edges)
    _propagate_throws(sccs, fn_decls, fn_structs, edges, checker, module.strict_math)
    _detect_tail_calls(fn_decls)
