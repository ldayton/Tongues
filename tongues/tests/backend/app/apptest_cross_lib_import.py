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
    try:
        test_basic()
        passed += 1
        print("  PASS test_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_basic: " + str(e))
    try:
        test_insert()
        passed += 1
        print("  PASS test_insert")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_insert: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_insert: " + str(e))
    try:
        test_substitute()
        passed += 1
        print("  PASS test_substitute")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_substitute: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_substitute: " + str(e))
    try:
        test_kitten_sitting()
        passed += 1
        print("  PASS test_kitten_sitting")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_kitten_sitting: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_kitten_sitting: " + str(e))
    try:
        test_symmetry()
        passed += 1
        print("  PASS test_symmetry")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_symmetry: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_symmetry: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
