"""Cross-lib import tests — a lib module importing from another lib module."""

import sys

from lib.crosstest import edit_distance


def test_basic() -> None:
    assert edit_distance("", "") == 0


def test_insert() -> None:
    assert edit_distance("ac", "abc") == 1


def test_substitute() -> None:
    assert edit_distance("abc", "axc") == 1


def test_kitten_sitting() -> None:
    assert edit_distance("kitten", "sitting") == 3


def test_symmetry() -> None:
    assert edit_distance("abc", "xyz") == edit_distance("xyz", "abc")


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_basic", test_basic),
        ("test_insert", test_insert),
        ("test_substitute", test_substitute),
        ("test_kitten_sitting", test_kitten_sitting),
        ("test_symmetry", test_symmetry),
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
