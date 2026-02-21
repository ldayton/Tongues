"""Tests for the Taytsh bytecode VM."""

from pathlib import Path

import pytest

from src.taytsh import parse
from src.taytsh.vm import vm_run

APP_DIR = Path(__file__).parent / "23_taytsh_app"


def _run_ty(source: str) -> tuple[int, str, str]:
    module = parse(source)
    result = vm_run(module)
    return result.exit_code, result.stdout, result.stderr


class TestBasicInt:
    def test_int_add(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    let a: int = 2
    let b: int = 3
    Assert(a + b == 5)
}
""")
        assert code == 0, err

    def test_int_sub(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    Assert(10 - 7 == 3)
}
""")
        assert code == 0, err

    def test_int_mul(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    Assert(6 * 7 == 42)
}
""")
        assert code == 0, err

    def test_int_div(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    Assert(10 / 3 == 3)
    Assert(-10 / 3 == -3)
}
""")
        assert code == 0, err

    def test_int_mod(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    Assert(10 % 3 == 1)
    Assert(-7 % 2 == -1)
}
""")
        assert code == 0, err

    def test_int_comparison(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    Assert(1 < 2)
    Assert(2 > 1)
    Assert(1 <= 1)
    Assert(1 >= 1)
    Assert(1 == 1)
    Assert(1 != 2)
}
""")
        assert code == 0, err

    def test_int_bitwise(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    Assert((0xf0 & 0x0f) == 0)
    Assert((0xf0 | 0x0f) == 255)
    Assert((0xf0 ^ 0x0f) == 255)
    Assert(~0 == -1)
}
""")
        assert code == 0, err

    def test_int_shifts(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    Assert(1 << 10 == 1024)
    Assert(1024 >> 10 == 1)
    Assert(-8 >> 1 == -4)
}
""")
        assert code == 0, err


class TestBool:
    def test_bool_and_or(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    Assert(true && true)
    Assert(!(true && false))
    Assert(true || false)
    Assert(!(false || false))
}
""")
        assert code == 0, err

    def test_bool_not(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    Assert(!false)
    Assert(!!true)
}
""")
        assert code == 0, err

    def test_short_circuit(self) -> None:
        code, out, err = _run_ty("""
fn Boom() -> bool {
    throw AssertError("should not be called")
}
fn Main() -> void {
    let result: bool = true || Boom()
    Assert(result)
    result = false && Boom()
    Assert(!result)
}
""")
        assert code == 0, err


class TestFunctions:
    def test_direct_call(self) -> None:
        code, out, err = _run_ty("""
fn Add(a: int, b: int) -> int {
    return a + b
}
fn Main() -> void {
    Assert(Add(2, 3) == 5)
}
""")
        assert code == 0, err

    def test_recursive_call(self) -> None:
        code, out, err = _run_ty("""
fn Fib(n: int) -> int {
    if n <= 1 {
        return n
    }
    return Fib(n - 1) + Fib(n - 2)
}
fn Main() -> void {
    Assert(Fib(10) == 55)
}
""")
        assert code == 0, err


class TestControlFlow:
    def test_if_else(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    let x: int = 5
    let r: string = ""
    if x > 0 {
        r = "pos"
    } else {
        r = "neg"
    }
    Assert(r == "pos")
}
""")
        assert code == 0, err

    def test_while(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    let n: int = 0
    while n < 10 {
        n += 1
    }
    Assert(n == 10)
}
""")
        assert code == 0, err

    def test_for_range(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    let total: int = 0
    for i in range(5) {
        total += i
    }
    Assert(total == 10)
}
""")
        assert code == 0, err

    def test_break(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    let count: int = 0
    for i in range(100) {
        if i == 5 {
            break
        }
        count += 1
    }
    Assert(count == 5)
}
""")
        assert code == 0, err

    def test_continue(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    let total: int = 0
    for i in range(10) {
        if i % 2 == 0 {
            continue
        }
        total += i
    }
    Assert(total == 25)
}
""")
        assert code == 0, err


class TestOutput:
    def test_writeln_out(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    WritelnOut("hello")
}
""")
        assert code == 0
        assert out == "hello\n"

    def test_tostring_int(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    WritelnOut(ToString(42))
}
""")
        assert code == 0
        assert out == "42\n"


class TestExceptions:
    def test_try_catch(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    let caught: bool = false
    try {
        throw ValueError("test")
    } catch e: ValueError {
        caught = true
        Assert(e.message == "test")
    }
    Assert(caught)
}
""")
        assert code == 0, err

    def test_div_zero(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    let caught: bool = false
    try {
        let x: int = 42 / 0
    } catch e: ZeroDivisionError {
        caught = true
    }
    Assert(caught)
}
""")
        assert code == 0, err


class TestZeroValues:
    def test_zero_int(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    let z: int
    Assert(z == 0)
}
""")
        assert code == 0, err

    def test_zero_bool(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    let z: bool
    Assert(!z)
}
""")
        assert code == 0, err

    def test_zero_string(self) -> None:
        code, out, err = _run_ty("""
fn Main() -> void {
    let z: string
    Assert(z == "")
}
""")
        assert code == 0, err


# ============================================================
# App test integration — run each .ty file through the VM
# ============================================================

_PHASE1_APPS: list[str] = []


def discover_vm_apps() -> list[pytest.param]:
    if not APP_DIR.exists():
        return []
    params: list[pytest.param] = []
    for path in sorted(APP_DIR.glob("*.ty")):
        if _PHASE1_APPS and path.stem not in _PHASE1_APPS:
            continue
        params.append(pytest.param(path, id=path.stem))
    return params


@pytest.mark.parametrize("app_path", discover_vm_apps())
def test_vm_app(app_path: Path) -> None:
    source = app_path.read_text()
    module = parse(source)
    result = vm_run(module)
    if result.exit_code != 0:
        output = (result.stdout + result.stderr).strip()
        pytest.fail(f"Exit code {result.exit_code}:\n{output}")
