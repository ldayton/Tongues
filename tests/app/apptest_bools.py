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


def test_bool_arithmetic() -> None:
    # Subtraction
    assert True - True == 0
    assert True - False == 1
    assert False - True == -1
    # Multiplication
    assert True * True == 1
    assert True * False == 0
    assert True * 10 == 10
    assert False * 10 == 0
    # Negation
    assert -True == -1
    assert -False == 0


def test_bool_comparisons() -> None:
    # Bool comparisons
    assert True > False
    assert False < True
    assert True >= True
    assert False <= False
    # Mixed bool/int comparisons
    assert True >= 1
    assert True <= 1
    assert False >= 0
    assert False < 1


def test_bool_from_negative() -> None:
    # Negative numbers are truthy
    assert bool(-1) == True
    assert bool(-100) == True
    # Negative zero is still falsy
    assert bool(-0) == False


def test_bool_floor_division() -> None:
    # Floor division with bools
    assert True // True == 1
    assert True // 1 == 1
    assert False // True == 0
    assert True // 2 == 0
    assert 5 // True == 5


def test_bool_modulo() -> None:
    # Modulo with bools
    assert True % True == 0
    assert True % 2 == 1
    assert False % True == 0
    assert 5 % True == 0
    assert 7 % True == 0


def test_bool_divmod() -> None:
    # divmod with bools
    q, r = divmod(True, True)
    assert q == 1
    assert r == 0
    q, r = divmod(True, 2)
    assert q == 0
    assert r == 1
    q, r = divmod(5, True)
    assert q == 5
    assert r == 0


def test_bool_pow() -> None:
    # pow with bools
    assert pow(True, True) == 1
    assert pow(True, False) == 1
    assert pow(False, True) == 0
    assert pow(False, False) == 1
    assert pow(True, 10) == 1
    assert pow(2, True) == 2
    assert pow(2, False) == 1


def test_bool_min_max() -> None:
    # min/max with bools
    assert min(True, False) == False
    assert max(True, False) == True
    assert min(True, True) == True
    assert max(False, False) == False
    # Mixed with ints
    assert min(True, 0) == 0
    assert min(True, 2) == 1
    assert max(False, -1) == 0
    assert max(True, 2) == 2


def test_bool_abs() -> None:
    # abs of bools
    assert abs(True) == 1
    assert abs(False) == 0


def test_bool_bitwise_with_ints() -> None:
    # Bitwise operations between bools and ints
    assert True & 3 == 1
    assert False & 3 == 0
    assert True | 2 == 3
    assert False | 2 == 2
    assert True ^ 3 == 2
    assert False ^ 3 == 3
    # Reverse order
    assert 3 & True == 1
    assert 3 | False == 3
    assert 3 ^ True == 2


def test_bool_shifts() -> None:
    # Shift operations with bools
    assert True << 1 == 2
    assert True << 3 == 8
    assert False << 5 == 0
    assert True >> 1 == 0
    assert 4 >> True == 2
    assert 4 << True == 8


def test_bool_invert() -> None:
    # Bitwise invert of bools
    assert ~True == -2
    assert ~False == -1


def test_bool_not_vs_invert() -> None:
    # Logical not vs bitwise invert
    assert not True == False
    assert not False == True
    assert ~True == -2
    assert ~False == -1
    # They're very different!
    assert (not True) != (~True)


def test_bool_in_expressions() -> None:
    # Bools in complex expressions
    assert True + True + True == 3
    assert True * True * True == 1
    assert (True + False) * 2 == 2
    assert True + 1 + True == 3


def test_bool_from_strings() -> None:
    # bool() on strings
    assert bool("True") == True
    assert bool("False") == True  # non-empty string is truthy!
    assert bool("") == False
    assert bool("0") == True  # non-empty string is truthy!
    assert bool(" ") == True


def test_bool_from_collections() -> None:
    # bool() on collections
    assert bool([]) == False
    assert bool([0]) == True  # non-empty list
    assert bool([False]) == True
    assert bool({}) == False
    assert bool({0: 0}) == True


def test_bool_identity_elements() -> None:
    # Identity elements for bool arithmetic
    # Additive identity
    assert True + False == True
    assert False + False == False
    # Multiplicative identity
    assert True * True == True
    assert False * True == False


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_bool_callable", test_bool_callable),
        ("test_bool_string", test_bool_string),
        ("test_bool_int", test_bool_int),
        ("test_bool_ops", test_bool_ops),
        ("test_bool_int_ops", test_bool_int_ops),
        ("test_bool_arithmetic", test_bool_arithmetic),
        ("test_bool_comparisons", test_bool_comparisons),
        ("test_bool_from_negative", test_bool_from_negative),
        ("test_bool_floor_division", test_bool_floor_division),
        ("test_bool_modulo", test_bool_modulo),
        ("test_bool_divmod", test_bool_divmod),
        ("test_bool_pow", test_bool_pow),
        ("test_bool_min_max", test_bool_min_max),
        ("test_bool_abs", test_bool_abs),
        ("test_bool_bitwise_with_ints", test_bool_bitwise_with_ints),
        ("test_bool_shifts", test_bool_shifts),
        ("test_bool_invert", test_bool_invert),
        ("test_bool_not_vs_invert", test_bool_not_vs_invert),
        ("test_bool_in_expressions", test_bool_in_expressions),
        ("test_bool_from_strings", test_bool_from_strings),
        ("test_bool_from_collections", test_bool_from_collections),
        ("test_bool_identity_elements", test_bool_identity_elements),
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
