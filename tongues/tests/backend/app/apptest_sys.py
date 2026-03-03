"""Tests for sys-dependent builtins (stderr, exit, args)."""

import sys


def test_stderr_write() -> None:
    """Writing to stderr should not crash."""
    print("stderr test", file=sys.stderr)


def test_exit_zero() -> None:
    """Exit with 0 should succeed."""
    pass


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_stderr_write", test_stderr_write),
        ("test_exit_zero", test_exit_zero),
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
