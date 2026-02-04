"""Bool object tests, adapted from PyPy's apptest_boolobject.py.

Dropped tests:
  test_bool_long - Python 2 long type
  test_new - bool.__new__() reflection
  test_cant_subclass_bool - subclassing builtins
  test_bool_int_ops (partial) - identity comparisons between literals
"""

import sys


def test_bool_callable() -> None:
    assert True == bool(1)
    assert False == bool(0)
    assert False == bool()


def test_bool_string() -> None:
    assert "True" == str(True)
    assert "False" == str(False)
    assert "True" == repr(True)
    assert "False" == repr(False)


def test_bool_int() -> None:
    assert int(True) is 1
    assert int(False) is 0


def test_bool_ops() -> None:
    assert True + True == 2
    assert False | False is False
    assert True | False is True
    assert True & True is True
    assert True ^ True is False
    assert False ^ False is False
    assert True ^ False is True
    assert True & 1 == 1
    assert False & 0 == 0 & 0


def test_bool_int_ops() -> None:
    assert True == 1
    assert 1 == True
    assert False == 0
    assert 0 == False



def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_bool_callable", test_bool_callable),
        ("test_bool_string", test_bool_string),
        ("test_bool_int", test_bool_int),
        ("test_bool_ops", test_bool_ops),
        ("test_bool_int_ops", test_bool_int_ops),
    ]
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
