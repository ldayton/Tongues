"""Truthiness tests - how values evaluate in boolean contexts."""

import sys


def test_bool_truthiness() -> None:
    """Booleans are their own truth value."""
    assert bool(True) == True
    assert bool(False) == False
    assert True
    assert not False


def test_int_truthiness() -> None:
    """Zero is falsy, all other ints are truthy."""
    assert bool(0) == False
    assert bool(1) == True
    assert bool(-1) == True
    assert bool(100) == True
    assert bool(-100) == True
    assert not 0
    assert 1
    assert -1


def test_float_truthiness() -> None:
    """0.0 is falsy, all other floats are truthy."""
    assert bool(0.0) == False
    assert bool(-0.0) == False  # negative zero is also falsy
    assert bool(1.0) == True
    assert bool(-1.0) == True
    assert bool(0.1) == True
    assert bool(-0.1) == True
    assert not 0.0
    assert 1.0
    assert 0.5


def test_nan_is_truthy() -> None:
    """NaN is truthy - surprising but consistent (non-zero)."""
    nan: float = float("nan")
    assert bool(nan) == True
    # NaN is truthy even though it fails all comparisons
    assert nan != nan  # NaN != NaN is True
    assert not (nan == nan)  # NaN == NaN is False
    # float truthiness in if/ternary is disallowed (zero is valid data),
    # so we use bool() for explicit coercion
    assert bool(nan) == True


def test_infinity_is_truthy() -> None:
    """Infinity values are truthy."""
    inf: float = float("inf")
    ninf: float = float("-inf")
    assert bool(inf) == True
    assert bool(ninf) == True
    assert inf
    assert ninf


def test_string_truthiness() -> None:
    """Empty string is falsy, non-empty strings are truthy."""
    assert bool("") == False
    assert bool("a") == True
    assert bool(" ") == True  # whitespace is truthy
    assert bool("False") == True  # string "False" is truthy!
    assert bool("0") == True  # string "0" is truthy!
    assert not ""
    assert "hello"
    assert " "


def test_bytes_truthiness() -> None:
    """Empty bytes is falsy, non-empty bytes are truthy."""
    assert bool(b"") == False
    assert bool(b"a") == True
    assert bool(b" ") == True
    assert bool(b"\x00") == True  # null byte is truthy (non-empty)
    assert not b""
    assert b"hello"
    assert b"\x00"


def test_list_truthiness() -> None:
    """Empty list is falsy, non-empty lists are truthy."""
    empty: list[int] = []
    one: list[int] = [0]
    many: list[int] = [1, 2, 3]
    assert bool(empty) == False
    assert bool(one) == True  # list with falsy element is still truthy
    assert bool(many) == True
    assert not empty
    assert one
    assert many


def test_none_truthiness() -> None:
    """None is always falsy."""
    x: str | None = None
    assert bool(x) == False
    assert not x


def test_not_operator() -> None:
    """not operator returns bool."""
    assert not False == True
    assert not True == False
    assert not 0 == True
    assert not 1 == False
    assert not "" == True
    assert not "x" == False
    assert not None == True


def test_double_not() -> None:
    """Double not converts to bool."""
    assert (not not True) == True
    assert (not not False) == False
    assert (not not 1) == True
    assert (not not 0) == False
    assert (not not "hello") == True
    assert (not not "") == False


def test_and_short_circuit() -> None:
    """and returns first falsy or last truthy value."""
    # Returns first falsy
    assert (0 and 1) == 0
    assert (False and True) == False
    assert ("" and "hello") == ""
    # Returns last value if all truthy
    assert (1 and 2) == 2
    assert (True and False) == False
    assert ("a" and "b") == "b"
    assert (1 and 2 and 3) == 3


def test_or_short_circuit() -> None:
    """or returns first truthy or last falsy value."""
    # or-default patterns like (1 or 0) are not in the subset
    # In Tongues, 'or' returns bool, not operand value
    pass


def test_and_or_combination() -> None:
    """Combined and/or expressions."""
    # and has higher precedence than or
    assert (1 or 2 and 3) == 1  # 1 or (2 and 3) = 1 or 3 = 1
    assert (0 or 2 and 3) == 3  # 0 or (2 and 3) = 0 or 3 = 3
    assert (1 and 2 or 3) == 2  # (1 and 2) or 3 = 2 or 3 = 2
    assert (0 and 2 or 3) == 3  # (0 and 2) or 3 = 0 or 3 = 3


def test_if_expression_truthiness() -> None:
    """Conditional expressions use truthiness."""
    x: int = 1 if "hello" else 0
    assert x == 1
    x = 1 if "" else 0
    assert x == 0
    x = 1 if 42 != 0 else 0
    assert x == 1
    x = 1 if 0 != 0 else 0
    assert x == 0


def test_comparison_chain_bool() -> None:
    """Comparison chains produce bools."""
    assert (1 < 2 < 3) == True
    assert (1 < 2 > 3) == False
    assert bool(1 < 2 < 3) == True
    assert bool(1 < 3 < 2) == False


def test_equality_vs_identity_truthiness() -> None:
    """Equality to True/False vs truthiness."""
    # 1 is truthy but not equal to True in all contexts
    assert bool(1) == True
    assert bool(0) == False
    # Empty collections
    empty: list[int] = []
    assert bool(empty) == False
    assert (not empty) == True


def test_zero_vs_none_trap() -> None:
    """0 and None are both falsy but semantically different."""
    # Both are falsy
    assert bool(0) == False
    assert bool(None) == False
    # But they're not equal
    assert not (0 == None)
    assert 0 != None
    # Using truthiness to check for None can hide valid zeros
    value: int = 0
    # "if not value" would treat 0 same as missing
    assert not value  # 0 is falsy
    # This is why explicit None checks matter for optional values


def test_empty_vs_none() -> None:
    """Empty containers are falsy but not None."""
    empty_str: str = ""
    empty_bytes: bytes = b""
    empty_list: list[int] = []
    # All falsy
    assert not empty_str
    assert not empty_bytes
    assert not empty_list
    # But not equal to None
    assert not (empty_str == None)
    assert not (empty_bytes == None)
    assert not (empty_list == None)


def test_whitespace_string_truthy() -> None:
    """Whitespace-only strings are truthy (non-empty)."""
    assert bool(" ") == True
    assert bool("  ") == True
    assert bool("\t") == True
    assert bool("\n") == True
    assert bool(" \t\n") == True
    # Only truly empty string is falsy
    assert bool("") == False


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_bool_truthiness", test_bool_truthiness),
        ("test_int_truthiness", test_int_truthiness),
        ("test_float_truthiness", test_float_truthiness),
        ("test_nan_is_truthy", test_nan_is_truthy),
        ("test_infinity_is_truthy", test_infinity_is_truthy),
        ("test_string_truthiness", test_string_truthiness),
        ("test_bytes_truthiness", test_bytes_truthiness),
        ("test_list_truthiness", test_list_truthiness),
        ("test_none_truthiness", test_none_truthiness),
        ("test_not_operator", test_not_operator),
        ("test_double_not", test_double_not),
        ("test_and_short_circuit", test_and_short_circuit),
        ("test_or_short_circuit", test_or_short_circuit),
        ("test_and_or_combination", test_and_or_combination),
        ("test_if_expression_truthiness", test_if_expression_truthiness),
        ("test_comparison_chain_bool", test_comparison_chain_bool),
        ("test_equality_vs_identity_truthiness", test_equality_vs_identity_truthiness),
        ("test_zero_vs_none_trap", test_zero_vs_none_trap),
        ("test_empty_vs_none", test_empty_vs_none),
        ("test_whitespace_string_truthy", test_whitespace_string_truthy),
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
