"""Tests for the v2 returns analysis pass."""

from src.taytsh import parse as taytsh_parse
from src.taytsh.ast import (
    TCatch,
    TDefault,
    TFnDecl,
    TForStmt,
    TIfStmt,
    TMatchCase,
    TMatchStmt,
    TStructDecl,
    TTryStmt,
    TWhileStmt,
)
from src.taytsh.check import check_with_info
from src.middleend_v2.returns import analyze_returns


def _run(source: str):
    """Parse, check, and run returns analysis. Returns the module."""
    module = taytsh_parse(source)
    errors, checker = check_with_info(module)
    assert errors == [], [str(e) for e in errors]
    analyze_returns(module, checker)
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


# ============================================================
# always_returns
# ============================================================


def test_simple_return():
    m = _run("""
fn Foo() -> int {
    return 1
}
fn Main() -> void { WritelnOut(ToString(Foo())) }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.always_returns"] is True


def test_no_return():
    m = _run("""
fn Foo() -> void {
    WritelnOut("hi")
}
fn Main() -> void { Foo() }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.always_returns"] is False


def test_if_else_both_return():
    m = _run("""
fn Foo(x: int) -> int {
    if x > 0 {
        return 1
    } else {
        return 0
    }
}
fn Main() -> void { WritelnOut(ToString(Foo(1))) }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.always_returns"] is True
    if_stmt = fn.body[0]
    assert isinstance(if_stmt, TIfStmt)
    assert if_stmt.annotations["returns.always_returns"] is True


def test_if_no_else():
    m = _run("""
fn Foo(x: int) -> int {
    if x > 0 {
        return 1
    }
    return 0
}
fn Main() -> void { WritelnOut(ToString(Foo(1))) }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.always_returns"] is True
    if_stmt = fn.body[0]
    assert isinstance(if_stmt, TIfStmt)
    assert if_stmt.annotations["returns.always_returns"] is False


def test_while_always_false():
    m = _run("""
fn Foo() -> void {
    while true {
        return
    }
}
fn Main() -> void { Foo() }
""")
    fn = _find_fn(m, "Foo")
    while_stmt = fn.body[0]
    assert isinstance(while_stmt, TWhileStmt)
    assert while_stmt.annotations["returns.always_returns"] is False


def test_for_always_false():
    m = _run("""
fn Foo() -> void {
    let xs: list[int] = [1, 2, 3]
    for x in xs {
        return
    }
}
fn Main() -> void { Foo() }
""")
    fn = _find_fn(m, "Foo")
    for_stmt = fn.body[1]
    assert isinstance(for_stmt, TForStmt)
    assert for_stmt.annotations["returns.always_returns"] is False


def test_match_all_arms_return():
    m = _run("""
interface Shape {}
struct Circle : Shape { radius: int }
struct Rect : Shape { w: int h: int }
fn Describe(s: Shape) -> string {
    match s {
        case c: Circle {
            return "circle"
        }
        case r: Rect {
            return "rect"
        }
    }
}
fn Main() -> void { WritelnOut(Describe(Circle(radius: 1))) }
""")
    fn = _find_fn(m, "Describe")
    assert fn.annotations["returns.always_returns"] is True
    match_stmt = fn.body[0]
    assert isinstance(match_stmt, TMatchStmt)
    assert match_stmt.annotations["returns.always_returns"] is True
    for case in match_stmt.cases:
        assert case.annotations["returns.always_returns"] is True


def test_match_missing_arm():
    m = _run("""
interface Shape {}
struct Circle : Shape { radius: int }
struct Rect : Shape { w: int h: int }
fn Describe(s: Shape) -> string {
    match s {
        case c: Circle {
            return "circle"
        }
        case r: Rect {
            WritelnOut("rect")
        }
    }
    return "unknown"
}
fn Main() -> void { WritelnOut(Describe(Circle(radius: 1))) }
""")
    fn = _find_fn(m, "Describe")
    match_stmt = fn.body[0]
    assert isinstance(match_stmt, TMatchStmt)
    assert match_stmt.annotations["returns.always_returns"] is False


def test_try_catch_both_return():
    m = _run("""
fn Foo(s: string) -> int {
    try {
        return ParseInt(s, 10)
    } catch e: ValueError {
        return 0
    }
}
fn Main() -> void { WritelnOut(ToString(Foo("42"))) }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.always_returns"] is True
    try_stmt = fn.body[0]
    assert isinstance(try_stmt, TTryStmt)
    assert try_stmt.annotations["returns.always_returns"] is True


def test_throw_terminator():
    m = _run("""
fn Foo() -> int {
    throw ValueError(message: "nope")
}
fn Main() -> void {
    try {
        WritelnOut(ToString(Foo()))
    } catch e: ValueError {
        WritelnOut("caught")
    }
}
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.always_returns"] is True


def test_exit_terminator():
    m = _run("""
fn Foo() -> void {
    Exit(1)
}
fn Main() -> void { Foo() }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.always_returns"] is True


def test_nested_compound():
    m = _run("""
fn Foo(x: int) -> int {
    if x > 0 {
        if x > 10 {
            return 100
        } else {
            return 1
        }
    } else {
        return 0
    }
}
fn Main() -> void { WritelnOut(ToString(Foo(5))) }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.always_returns"] is True


# ============================================================
# needs_named_returns
# ============================================================


def test_catch_body_has_return():
    m = _run("""
fn Foo(s: string) -> int {
    try {
        return ParseInt(s, 10)
    } catch e: ValueError {
        return 0
    }
}
fn Main() -> void { WritelnOut(ToString(Foo("42"))) }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.needs_named_returns"] is True


def test_no_try_catch():
    m = _run("""
fn Foo() -> int {
    return 1
}
fn Main() -> void { WritelnOut(ToString(Foo())) }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.needs_named_returns"] is False


def test_try_catch_without_returns():
    m = _run("""
fn Foo() -> void {
    try {
        WritelnOut("hi")
    } catch e: ValueError {
        WritelnOut("err")
    }
}
fn Main() -> void { Foo() }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.needs_named_returns"] is False


def test_nested_try_in_if():
    m = _run("""
fn Foo(x: int, s: string) -> int {
    if x > 0 {
        try {
            return ParseInt(s, 10)
        } catch e: ValueError {
            return 0
        }
    }
    return x
}
fn Main() -> void { WritelnOut(ToString(Foo(1, "42"))) }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.needs_named_returns"] is True


# ============================================================
# may_return_nil
# ============================================================


def test_return_nil():
    m = _run("""
fn Foo() -> int? {
    return nil
}
fn Main() -> void {
    let x: int? = Foo()
    WritelnOut(ToString(0))
}
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.may_return_nil"] is True


def test_return_optional_var_no_narrowing():
    m = _run("""
fn Foo(x: int?) -> int? {
    return x
}
fn Main() -> void {
    let r: int? = Foo(nil)
    WritelnOut(ToString(0))
}
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.may_return_nil"] is True


def test_return_narrowed_var():
    m = _run("""
fn Foo(x: int?) -> int {
    if x != nil {
        return x
    }
    return 0
}
fn Main() -> void { WritelnOut(ToString(Foo(1))) }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.may_return_nil"] is False


def test_return_int_literal():
    m = _run("""
fn Foo() -> int {
    return 42
}
fn Main() -> void { WritelnOut(ToString(Foo())) }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.may_return_nil"] is False


def test_return_call_with_optional_return():
    m = _run("""
fn MaybeNil() -> int? {
    return nil
}
fn Foo() -> int? {
    return MaybeNil()
}
fn Main() -> void {
    let r: int? = Foo()
    WritelnOut(ToString(0))
}
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.may_return_nil"] is True


def test_bare_return():
    m = _run("""
fn Foo() -> void {
    return
}
fn Main() -> void { Foo() }
""")
    fn = _find_fn(m, "Foo")
    assert fn.annotations["returns.may_return_nil"] is False


# ============================================================
# body_has_return
# ============================================================


def test_return_in_try_body():
    m = _run("""
fn Foo(s: string) -> int {
    try {
        return ParseInt(s, 10)
    } catch e: ValueError {
        return 0
    }
}
fn Main() -> void { WritelnOut(ToString(Foo("42"))) }
""")
    fn = _find_fn(m, "Foo")
    try_stmt = fn.body[0]
    assert isinstance(try_stmt, TTryStmt)
    assert try_stmt.annotations["returns.body_has_return"] is True


def test_no_return_in_try_body():
    m = _run("""
fn Foo() -> void {
    try {
        WritelnOut("hi")
    } catch e: ValueError {
        WritelnOut("err")
    }
}
fn Main() -> void { Foo() }
""")
    fn = _find_fn(m, "Foo")
    try_stmt = fn.body[0]
    assert isinstance(try_stmt, TTryStmt)
    assert try_stmt.annotations["returns.body_has_return"] is False


def test_nested_return_in_if_inside_try():
    m = _run("""
fn Foo(x: int, s: string) -> int {
    try {
        if x > 0 {
            return ParseInt(s, 10)
        }
    } catch e: ValueError {
        WritelnOut("err")
    }
    return 0
}
fn Main() -> void { WritelnOut(ToString(Foo(1, "42"))) }
""")
    fn = _find_fn(m, "Foo")
    try_stmt = fn.body[0]
    assert isinstance(try_stmt, TTryStmt)
    assert try_stmt.annotations["returns.body_has_return"] is True


# ============================================================
# Struct methods
# ============================================================


def test_method_annotations():
    m = _run("""
struct Calc {
    value: int
    fn Get(self) -> int {
        return self.value
    }
}
fn Main() -> void {
    let c: Calc = Calc(value: 42)
    WritelnOut(ToString(c.Get()))
}
""")
    method = _find_method(m, "Calc", "Get")
    assert method.annotations["returns.always_returns"] is True
    assert method.annotations["returns.needs_named_returns"] is False
    assert method.annotations["returns.may_return_nil"] is False


# ============================================================
# Match with default
# ============================================================


def test_match_with_default_all_return():
    m = _run("""
interface Shape {}
struct Circle : Shape { radius: int }
struct Rect : Shape { w: int h: int }
fn Describe(s: Shape) -> string {
    match s {
        case c: Circle {
            return "circle"
        }
        default {
            return "other"
        }
    }
}
fn Main() -> void { WritelnOut(Describe(Circle(radius: 1))) }
""")
    fn = _find_fn(m, "Describe")
    assert fn.annotations["returns.always_returns"] is True
    match_stmt = fn.body[0]
    assert isinstance(match_stmt, TMatchStmt)
    assert match_stmt.annotations["returns.always_returns"] is True
    assert match_stmt.default is not None
    assert match_stmt.default.annotations["returns.always_returns"] is True


def test_catch_always_returns():
    m = _run("""
fn Foo(s: string) -> int {
    try {
        return ParseInt(s, 10)
    } catch e: ValueError {
        return 0
    }
}
fn Main() -> void { WritelnOut(ToString(Foo("42"))) }
""")
    fn = _find_fn(m, "Foo")
    try_stmt = fn.body[0]
    assert isinstance(try_stmt, TTryStmt)
    catch = try_stmt.catches[0]
    assert isinstance(catch, TCatch)
    assert catch.annotations["returns.always_returns"] is True
