"""Tests for the v2 scope analysis pass."""

from src.taytsh import parse as taytsh_parse
from src.taytsh.ast import (
    TCatch,
    TDefault,
    TFnDecl,
    TForStmt,
    TLetStmt,
    TParam,
    TPatternType,
    TStructDecl,
    TVar,
)
from src.taytsh.check import check_with_info
from src.middleend_v2.scope import analyze_scope


def _run(source: str):
    """Parse, check, and run scope analysis. Returns the module."""
    module = taytsh_parse(source)
    errors, checker = check_with_info(module)
    assert errors == [], [str(e) for e in errors]
    analyze_scope(module, checker)
    return module


def _find_fn(module, name: str) -> TFnDecl:
    for d in module.decls:
        if isinstance(d, TFnDecl) and d.name == name:
            return d
    raise ValueError(f"no fn {name}")


def _find_method(module, struct_name: str, method_name: str) -> TFnDecl:
    for d in module.decls:
        if isinstance(d, TStructDecl) and d.name == struct_name:
            for m in d.methods:
                if m.name == method_name:
                    return m
    raise ValueError(f"no method {struct_name}.{method_name}")


def _find_param(fn: TFnDecl, name: str) -> TParam:
    for p in fn.params:
        if p.name == name:
            return p
    raise ValueError(f"no param {name}")


def _find_let(stmts, name: str) -> TLetStmt:
    for s in stmts:
        if isinstance(s, TLetStmt) and s.name == name:
            return s
    raise ValueError(f"no let {name}")


def _find_var(stmts, name: str) -> TVar:
    """Find the first TVar with given name in a flat list of stmts (searches expr stmts)."""
    from src.taytsh.ast import TExprStmt, TReturnStmt, TCall, TBinaryOp
    for s in stmts:
        if isinstance(s, TExprStmt):
            expr = s.expr
            if isinstance(expr, TVar) and expr.name == name:
                return expr
            if isinstance(expr, TCall):
                for a in expr.args:
                    if isinstance(a.value, TVar) and a.value.name == name:
                        return a.value
                if isinstance(expr.func, TVar) and expr.func.name == name:
                    return expr.func
        if isinstance(s, TReturnStmt) and s.value is not None:
            if isinstance(s.value, TVar) and s.value.name == name:
                return s.value
            if isinstance(s.value, TBinaryOp):
                if isinstance(s.value.left, TVar) and s.value.left.name == name:
                    return s.value.left
                if isinstance(s.value.right, TVar) and s.value.right.name == name:
                    return s.value.right
    raise ValueError(f"no var {name}")


# ── Reassignment / const ──


def test_let_const():
    m = _run("""
fn Main() -> void {
    let x: int = 1
    let y: int = 2
    x = 3
    WritelnOut(ToString(y))
}
""")
    fn = _find_fn(m, "Main")
    let_x = _find_let(fn.body, "x")
    let_y = _find_let(fn.body, "y")
    assert let_x.annotations["scope.is_reassigned"] is True
    assert let_x.annotations["scope.is_const"] is False
    assert let_y.annotations["scope.is_reassigned"] is False
    assert let_y.annotations["scope.is_const"] is True


def test_op_assign_reassigns():
    m = _run("""
fn Main() -> void {
    let x: int = 1
    x += 2
    WritelnOut(ToString(x))
}
""")
    fn = _find_fn(m, "Main")
    let_x = _find_let(fn.body, "x")
    assert let_x.annotations["scope.is_reassigned"] is True


def test_field_assign_not_reassign():
    m = _run("""
struct Pt {
    x: int
    y: int
}
fn Main() -> void {
    let p: Pt = Pt(x: 1, y: 2)
    p.x = 3
    WritelnOut(ToString(p.x))
}
""")
    fn = _find_fn(m, "Main")
    let_p = _find_let(fn.body, "p")
    assert let_p.annotations["scope.is_reassigned"] is False
    assert let_p.annotations["scope.is_const"] is True


# ── Parameter modification ──


def test_param_reassigned():
    m = _run("""
fn Foo(x: int) -> int {
    x = x + 1
    return x
}
fn Main() -> void { WritelnOut(ToString(Foo(1))) }
""")
    fn = _find_fn(m, "Foo")
    px = _find_param(fn, "x")
    assert px.annotations["scope.is_reassigned"] is True
    assert px.annotations["scope.is_modified"] is True


def test_param_field_mutation():
    m = _run("""
struct Pt {
    x: int
    y: int
}
fn Mutate(p: Pt) -> void {
    p.x = 99
}
fn Main() -> void { Mutate(Pt(x: 1, y: 2)) }
""")
    fn = _find_fn(m, "Mutate")
    pp = _find_param(fn, "p")
    assert pp.annotations["scope.is_reassigned"] is False
    assert pp.annotations["scope.is_modified"] is True


def test_param_mutating_builtin():
    m = _run("""
fn Push(xs: list[int]) -> void {
    Append(xs, 1)
}
fn Main() -> void {
    let a: list[int] = [1, 2]
    Push(a)
}
""")
    fn = _find_fn(m, "Push")
    pxs = _find_param(fn, "xs")
    assert pxs.annotations["scope.is_reassigned"] is False
    assert pxs.annotations["scope.is_modified"] is True


def test_param_index_mutation():
    m = _run("""
fn SetFirst(xs: list[int]) -> void {
    xs[0] = 99
}
fn Main() -> void {
    let a: list[int] = [1, 2]
    SetFirst(a)
}
""")
    fn = _find_fn(m, "SetFirst")
    pxs = _find_param(fn, "xs")
    assert pxs.annotations["scope.is_reassigned"] is False
    assert pxs.annotations["scope.is_modified"] is True


# ── Parameter unused ──


def test_param_unused():
    m = _run("""
fn Ignore(x: int) -> int {
    return 0
}
fn Main() -> void { WritelnOut(ToString(Ignore(1))) }
""")
    fn = _find_fn(m, "Ignore")
    px = _find_param(fn, "x")
    assert px.annotations["scope.is_unused"] is True
    assert px.annotations["scope.is_modified"] is False


def test_param_used():
    m = _run("""
fn Identity(x: int) -> int {
    return x
}
fn Main() -> void { WritelnOut(ToString(Identity(1))) }
""")
    fn = _find_fn(m, "Identity")
    px = _find_param(fn, "x")
    assert px.annotations["scope.is_unused"] is False


# ── Nil narrowing ──


def test_nil_narrowing_neq():
    m = _run("""
fn Narrow(x: int?) -> int {
    if x != nil {
        return x
    }
    return 0
}
fn Main() -> void { WritelnOut(ToString(Narrow(1))) }
""")
    fn = _find_fn(m, "Narrow")
    # The return x inside the if body
    from src.taytsh.ast import TIfStmt
    if_stmt = fn.body[0]
    assert isinstance(if_stmt, TIfStmt)
    ret_var = _find_var(if_stmt.then_body, "x")
    assert ret_var.annotations["scope.narrowed_type"] == "int"


def test_nil_narrowing_eq():
    m = _run("""
fn Narrow(x: int?) -> int {
    if x == nil {
        return 0
    } else {
        return x
    }
}
fn Main() -> void { WritelnOut(ToString(Narrow(1))) }
""")
    fn = _find_fn(m, "Narrow")
    from src.taytsh.ast import TIfStmt
    if_stmt = fn.body[0]
    assert isinstance(if_stmt, TIfStmt)
    assert if_stmt.else_body is not None
    ret_var = _find_var(if_stmt.else_body, "x")
    assert ret_var.annotations["scope.narrowed_type"] == "int"


# ── Match case binding ──


def test_match_case_binding():
    m = _run("""
interface Shape {}
struct Circle : Shape {
    radius: int
}
struct Rect : Shape {
    w: int
    h: int
}
fn Area(s: Shape) -> int {
    match s {
        case c: Circle {
            return c.radius
        }
        case r: Rect {
            return r.w
        }
    }
}
fn Main() -> void {
    WritelnOut(ToString(Area(Circle(radius: 5))))
}
""")
    fn = _find_fn(m, "Area")
    from src.taytsh.ast import TMatchStmt
    match_stmt = fn.body[0]
    assert isinstance(match_stmt, TMatchStmt)
    pat0 = match_stmt.cases[0].pattern
    assert isinstance(pat0, TPatternType)
    assert pat0.annotations["scope.is_reassigned"] is False
    assert pat0.annotations["scope.is_const"] is True


# ── For-binder annotations ──


def test_for_binder_const():
    m = _run("""
fn Main() -> void {
    let xs: list[int] = [1, 2, 3]
    for x in xs {
        WritelnOut(ToString(x))
    }
}
""")
    fn = _find_fn(m, "Main")
    from src.taytsh.ast import TForStmt
    for_stmt = fn.body[1]
    assert isinstance(for_stmt, TForStmt)
    assert for_stmt.annotations["scope.binder.x.is_reassigned"] is False
    assert for_stmt.annotations["scope.binder.x.is_const"] is True


def test_for_binder_reassigned():
    m = _run("""
fn Main() -> void {
    let xs: list[int] = [1, 2, 3]
    for x in xs {
        x = x + 1
        WritelnOut(ToString(x))
    }
}
""")
    fn = _find_fn(m, "Main")
    from src.taytsh.ast import TForStmt
    for_stmt = fn.body[1]
    assert isinstance(for_stmt, TForStmt)
    assert for_stmt.annotations["scope.binder.x.is_reassigned"] is True
    assert for_stmt.annotations["scope.binder.x.is_const"] is False


def test_for_two_binders():
    m = _run("""
fn Main() -> void {
    let xs: list[int] = [1, 2, 3]
    for i, x in xs {
        WritelnOut(ToString(i))
        WritelnOut(ToString(x))
    }
}
""")
    fn = _find_fn(m, "Main")
    from src.taytsh.ast import TForStmt
    for_stmt = fn.body[1]
    assert isinstance(for_stmt, TForStmt)
    assert for_stmt.annotations["scope.binder.i.is_const"] is True
    assert for_stmt.annotations["scope.binder.x.is_const"] is True


# ── Function ref detection ──


def test_function_ref():
    m = _run("""
fn AddOne(x: int) -> int { return x + 1 }
fn Apply(f: fn[int, int], x: int) -> int { return f(x) }
fn Main() -> void {
    WritelnOut(ToString(Apply(AddOne, 1)))
}
""")
    fn = _find_fn(m, "Main")
    # The call is Apply(AddOne, 1) — the AddOne is a TVar in the args
    from src.taytsh.ast import TExprStmt, TCall
    expr_stmt = fn.body[0]
    assert isinstance(expr_stmt, TExprStmt)
    outer_call = expr_stmt.expr
    assert isinstance(outer_call, TCall)
    # WritelnOut(ToString(Apply(AddOne, 1)))
    # outer_call.func is WritelnOut, args[0] is ToString(Apply(AddOne, 1))
    tostring_call = outer_call.args[0].value
    assert isinstance(tostring_call, TCall)
    apply_call = tostring_call.args[0].value
    assert isinstance(apply_call, TCall)
    addone_var = apply_call.args[0].value
    assert isinstance(addone_var, TVar)
    assert addone_var.name == "AddOne"
    assert addone_var.annotations.get("scope.is_function_ref") is True


def test_function_ref_as_callee():
    m = _run("""
fn Helper() -> void {}
fn Main() -> void {
    Helper()
}
""")
    fn = _find_fn(m, "Main")
    from src.taytsh.ast import TExprStmt, TCall
    expr_stmt = fn.body[0]
    assert isinstance(expr_stmt, TExprStmt)
    call = expr_stmt.expr
    assert isinstance(call, TCall)
    func_var = call.func
    assert isinstance(func_var, TVar)
    assert func_var.annotations.get("scope.is_function_ref") is True


def test_param_not_function_ref():
    m = _run("""
fn Apply(f: fn[int, int], x: int) -> int { return f(x) }
fn Main() -> void { WritelnOut(ToString(Apply((x: int) -> int => x, 1))) }
""")
    fn = _find_fn(m, "Apply")
    from src.taytsh.ast import TReturnStmt, TCall
    ret = fn.body[0]
    assert isinstance(ret, TReturnStmt)
    call = ret.value
    assert isinstance(call, TCall)
    f_var = call.func
    assert isinstance(f_var, TVar)
    assert f_var.annotations.get("scope.is_function_ref") is not True


# ── Interface type detection ──


def test_interface_typed():
    m = _run("""
interface Shape {}
struct Circle : Shape {
    radius: int
}
fn Use(s: Shape) -> int {
    match s {
        case c: Circle {
            return c.radius
        }
        default {
            return 0
        }
    }
}
fn Main() -> void {
    WritelnOut(ToString(Use(Circle(radius: 5))))
}
""")
    fn = _find_fn(m, "Use")
    # s is used in `match s` — it's a TVar
    from src.taytsh.ast import TMatchStmt
    match_stmt = fn.body[0]
    assert isinstance(match_stmt, TMatchStmt)
    s_var = match_stmt.expr
    assert isinstance(s_var, TVar)
    assert s_var.annotations.get("scope.is_interface") is True


# ── Catch binding ──


def test_catch_binding():
    m = _run("""
fn Main() -> void {
    try {
        WritelnOut("hi")
    } catch e: KeyError {
        WritelnOut(e.message)
    }
}
""")
    fn = _find_fn(m, "Main")
    from src.taytsh.ast import TTryStmt
    try_stmt = fn.body[0]
    assert isinstance(try_stmt, TTryStmt)
    catch = try_stmt.catches[0]
    assert isinstance(catch, TCatch)
    assert catch.annotations["scope.is_reassigned"] is False
    assert catch.annotations["scope.is_const"] is True


# ── Void method mutation ──


def test_void_method_mutation():
    m = _run("""
struct Counter {
    n: int
    fn Inc(self) -> void {
        self.n = self.n + 1
    }
}
fn Bump(c: Counter) -> void {
    c.Inc()
}
fn Main() -> void {
    let c: Counter = Counter(n: 0)
    Bump(c)
}
""")
    fn = _find_fn(m, "Bump")
    pc = _find_param(fn, "c")
    assert pc.annotations["scope.is_modified"] is True


# ── Tuple assignment reassigns ──


def test_tuple_assign_reassigns():
    m = _run("""
fn Main() -> void {
    let a: int = 1
    let b: int = 2
    a, b = DivMod(10, 3)
    WritelnOut(ToString(a))
}
""")
    fn = _find_fn(m, "Main")
    let_a = _find_let(fn.body, "a")
    let_b = _find_let(fn.body, "b")
    assert let_a.annotations["scope.is_reassigned"] is True
    assert let_b.annotations["scope.is_reassigned"] is True
