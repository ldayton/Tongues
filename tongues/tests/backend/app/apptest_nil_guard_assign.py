"""Nil guard with assignment should narrow to non-nil."""

import sys
from dataclasses import dataclass


@dataclass
class Foo:
    x: int


def ensure_foo(v: Foo | None) -> Foo:
    if v is None:
        v = Foo(0)
    return v


def test_nil_guard_assign() -> None:
    """Assignment inside nil guard narrows to non-nil."""
    result: Foo = ensure_foo(None)
    assert result.x == 0
    result = ensure_foo(Foo(42))
    assert result.x == 42


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_nil_guard_assign", test_nil_guard_assign),
    ]
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print("  PASS " + name)
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {name}: {e!s}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {name}: {e!s}")
    print(f"{passed!s} passed, {failed!s} failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
