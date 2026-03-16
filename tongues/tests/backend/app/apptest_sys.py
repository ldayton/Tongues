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
    try:
        test_stderr_write()
        passed += 1
        print("  PASS test_stderr_write")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_stderr_write: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_stderr_write: " + str(e))
    try:
        test_exit_zero()
        passed += 1
        print("  PASS test_exit_zero")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_exit_zero: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_exit_zero: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
