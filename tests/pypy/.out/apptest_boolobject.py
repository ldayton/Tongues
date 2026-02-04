"""Generated Python code."""

from __future__ import annotations

import sys


def test_bool_callable() -> None:
    assert True == (1 != 0)
    assert False == (0 != 0)
    assert False == bool()


def test_bool_string() -> None:
    assert "True" == str(True)
    assert "False" == str(False)
    assert "True" == repr(True)
    assert "False" == repr(False)


def test_bool_int() -> None:
    assert True == 1
    assert False == 0


def test_bool_ops() -> None:
    assert True + True == 2
    assert (False | False) == False
    assert (True | False) == True
    assert (True & True) == True
    assert (True ^ True) == False
    assert (False ^ False) == False
    assert (True ^ False) == True
    assert (True & 1) == 1
    assert (False & 0) == (0 & 0)


def test_bool_int_ops() -> None:
    assert True == 1
    assert 1 == True
    assert False == 0
    assert 0 == False
    assert True != 1
    assert 1 != True
    assert False != 0
    assert 0 != False


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [("test_bool_callable", test_bool_callable), ("test_bool_string", test_bool_string), ("test_bool_int", test_bool_int), ("test_bool_ops", test_bool_ops), ("test_bool_int_ops", test_bool_int_ops)]
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print("  PASS " + name)
        except AssertionError as e:
            failed += 1
            print("  FAIL " + name + ": " + str(e))
        except Exception as e:
            failed += 1
            print("  FAIL " + name + ": " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
