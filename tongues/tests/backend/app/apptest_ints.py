"""Integer object tests.

Comprehensive edge cases for integer operations, avoiding width-dependent tests.
"""

import sys


def test_int_arithmetic_basic() -> None:
    # Addition
    assert 1 + 1 == 2
    assert 0 + 0 == 0
    assert -1 + 1 == 0
    assert 100 + 200 == 300
    # Subtraction
    assert 5 - 3 == 2
    assert 3 - 5 == -2
    assert 0 - 0 == 0
    assert -10 - -5 == -5
    # Multiplication
    assert 3 * 4 == 12
    assert -3 * 4 == -12
    assert -3 * -4 == 12
    assert 0 * 1000 == 0
    assert 1 * 1 == 1


def test_int_arithmetic_identity() -> None:
    # Additive identity (x + 0 = x)
    assert 42 + 0 == 42
    assert -42 + 0 == -42
    assert 0 + 0 == 0
    # Subtractive identity (x - 0 = x)
    assert 42 - 0 == 42
    assert -42 - 0 == -42
    # Multiplicative identity (x * 1 = x)
    assert 42 * 1 == 42
    assert -42 * 1 == -42
    assert 0 * 1 == 0
    # Multiplication by zero
    assert 42 * 0 == 0
    assert -42 * 0 == 0
    assert 0 * 0 == 0


def test_int_arithmetic_associative() -> None:
    # Addition is associative: (a + b) + c == a + (b + c)
    assert (1 + 2) + 3 == 1 + (2 + 3)
    assert (-5 + 10) + 3 == -5 + (10 + 3)
    # Multiplication is associative: (a * b) * c == a * (b * c)
    assert (2 * 3) * 4 == 2 * (3 * 4)
    assert (-2 * 3) * 4 == -2 * (3 * 4)


def test_int_arithmetic_commutative() -> None:
    # Addition is commutative: a + b == b + a
    assert 3 + 5 == 5 + 3
    assert -3 + 5 == 5 + -3
    # Multiplication is commutative: a * b == b * a
    assert 3 * 5 == 5 * 3
    assert -3 * 5 == 5 * -3


def test_int_floor_division_positive() -> None:
    # Exact division
    assert 10 // 2 == 5
    assert 9 // 3 == 3
    assert 100 // 10 == 10
    # Division with remainder
    assert 10 // 3 == 3
    assert 7 // 2 == 3
    assert 1 // 2 == 0
    # Division by 1 (identity)
    assert 42 // 1 == 42
    assert 0 // 1 == 0
    # Zero divided by anything
    assert 0 // 5 == 0
    assert 0 // 100 == 0


def test_int_floor_division_negative() -> None:
    # Python floors toward negative infinity, not toward zero
    assert -10 // 3 == -4
    assert 10 // -3 == -4
    assert -10 // -3 == 3
    assert -7 // 2 == -4
    assert 7 // -2 == -4
    assert -7 // -2 == 3
    # Edge: -1 division
    assert -1 // 2 == -1
    assert 1 // -2 == -1
    assert -1 // -2 == 0


def test_int_modulo_positive() -> None:
    assert 10 % 3 == 1
    assert 9 % 3 == 0
    assert 7 % 2 == 1
    assert 1 % 5 == 1
    assert 0 % 5 == 0
    # Modulo by 1 always 0
    assert 42 % 1 == 0
    assert 0 % 1 == 0


def test_int_modulo_negative() -> None:
    # Result has same sign as divisor in Python
    assert -10 % 3 == 2
    assert 10 % -3 == -2
    assert -10 % -3 == -1
    assert -7 % 2 == 1
    assert 7 % -2 == -1
    assert -7 % -2 == -1
    assert -1 % 3 == 2
    assert 1 % -3 == -2


def test_int_division_identity() -> None:
    # Verify: x == (x // y) * y + (x % y)
    x: int = 17
    y: int = 5
    assert x == (x // y) * y + (x % y)
    x = -17
    y = 5
    assert x == (x // y) * y + (x % y)
    x = 17
    y = -5
    assert x == (x // y) * y + (x % y)
    x = -17
    y = -5
    assert x == (x // y) * y + (x % y)


def test_int_negation() -> None:
    assert -0 == 0
    assert -1 == 0 - 1
    assert -(-1) == 1
    assert -(-(-5)) == -5
    assert -100 + 100 == 0
    # Double negation
    assert -(-42) == 42
    assert -(-(-(-42))) == 42


def test_int_comparisons_equality() -> None:
    assert 5 == 5
    assert 0 == 0
    assert -1 == -1
    assert -0 == 0
    # Inequality
    assert 5 != 6
    assert 0 != 1
    assert -1 != 1
    assert 1 != -1


def test_int_comparisons_ordering() -> None:
    # Less than
    assert 1 < 2
    assert -1 < 0
    assert -5 < -4
    assert 0 < 1
    assert -100 < 100
    # Greater than
    assert 2 > 1
    assert 0 > -1
    assert -4 > -5
    assert 1 > 0
    assert 100 > -100
    # Less than or equal
    assert 1 <= 2
    assert 1 <= 1
    assert -1 <= 0
    assert -1 <= -1
    # Greater than or equal
    assert 2 >= 1
    assert 2 >= 2
    assert 0 >= -1
    assert 0 >= 0


def test_int_comparisons_transitive() -> None:
    # If a < b and b < c, then a < c
    a: int = 1
    b: int = 5
    c: int = 10
    assert a < b
    assert b < c
    assert a < c
    # With negatives
    a = -10
    b = -5
    c = 0
    assert a < b
    assert b < c
    assert a < c


def test_int_conversion_from_int() -> None:
    assert int(42) == 42
    assert int(-42) == -42
    assert int(0) == 0


def test_int_conversion_from_bool() -> None:
    assert int(True) == 1
    assert int(False) == 0


def test_int_conversion_from_string() -> None:
    assert int("42") == 42
    assert int("-42") == -42
    assert int("0") == 0
    # Leading zeros
    assert int("00123") == 123
    assert int("007") == 7
    assert int("000") == 0
    # Explicit positive sign
    assert int("+42") == 42
    assert int("+0") == 0


def test_int_conversion_whitespace() -> None:
    # Leading/trailing whitespace is stripped
    assert int(" 42") == 42
    assert int("42 ") == 42
    assert int(" 42 ") == 42
    assert int("  -42  ") == -42
    assert int(" 0 ") == 0


def test_int_string_repr() -> None:
    assert str(0) == "0"
    assert str(42) == "42"
    assert str(-42) == "-42"
    assert str(1000) == "1000"
    assert str(-1000) == "-1000"
    assert repr(0) == "0"
    assert repr(42) == "42"
    assert repr(-42) == "-42"


def test_int_abs() -> None:
    assert abs(0) == 0
    assert abs(5) == 5
    assert abs(-5) == 5
    assert abs(100) == 100
    assert abs(-100) == 100
    # abs of abs
    assert abs(abs(-5)) == 5


def test_int_min_two_args() -> None:
    assert min(1, 2) == 1
    assert min(2, 1) == 1
    assert min(-1, 1) == -1
    assert min(1, -1) == -1
    assert min(0, 0) == 0
    assert min(-5, -10) == -10
    assert min(-10, -5) == -10


def test_int_min_multiple_args() -> None:
    assert min(3, 1, 2) == 1
    assert min(1, 2, 3) == 1
    assert min(3, 2, 1) == 1
    assert min(-1, -2, -3) == -3
    assert min(0, -1, 1) == -1
    assert min(5, 5, 5) == 5


def test_int_max_two_args() -> None:
    assert max(1, 2) == 2
    assert max(2, 1) == 2
    assert max(-1, 1) == 1
    assert max(1, -1) == 1
    assert max(0, 0) == 0
    assert max(-5, -10) == -5
    assert max(-10, -5) == -5


def test_int_max_multiple_args() -> None:
    assert max(3, 1, 2) == 3
    assert max(1, 2, 3) == 3
    assert max(3, 2, 1) == 3
    assert max(-1, -2, -3) == -1
    assert max(0, -1, 1) == 1
    assert max(5, 5, 5) == 5


def test_int_divmod_positive() -> None:
    q, r = divmod(10, 3)
    assert q == 3
    assert r == 1
    q, r = divmod(9, 3)
    assert q == 3
    assert r == 0
    q, r = divmod(1, 5)
    assert q == 0
    assert r == 1
    q, r = divmod(0, 5)
    assert q == 0
    assert r == 0


def test_int_divmod_negative() -> None:
    q, r = divmod(-10, 3)
    assert q == -4
    assert r == 2
    q, r = divmod(10, -3)
    assert q == -4
    assert r == -2
    q, r = divmod(-10, -3)
    assert q == 3
    assert r == -1


def test_int_divmod_identity() -> None:
    # Verify divmod(x, y) == (x // y, x % y)
    x: int = 17
    y: int = 5
    q, r = divmod(x, y)
    assert q == x // y
    assert r == x % y
    x = -17
    q, r = divmod(x, y)
    assert q == x // y
    assert r == x % y


def test_int_pow_basic() -> None:
    assert pow(2, 0) == 1
    assert pow(2, 1) == 2
    assert pow(2, 2) == 4
    assert pow(2, 3) == 8
    assert pow(2, 10) == 1024
    assert pow(3, 3) == 27
    assert pow(10, 3) == 1000


def test_int_pow_edge_cases() -> None:
    # Anything to power 0 is 1
    assert pow(0, 0) == 1
    assert pow(1, 0) == 1
    assert pow(100, 0) == 1
    assert pow(-5, 0) == 1
    # 0 to any positive power is 0
    assert pow(0, 1) == 0
    assert pow(0, 5) == 0
    assert pow(0, 100) == 0
    # 1 to any power is 1
    assert pow(1, 1) == 1
    assert pow(1, 100) == 1
    assert pow(1, 1000) == 1


def test_hex_oct_bin() -> None:
    """hex/oct/bin with [2:] slice returns correct string."""
    assert hex(255)[2:] == "ff"
    assert oct(255)[2:] == "377"
    assert bin(255)[2:] == "11111111"
    assert hex(0)[2:] == "0"


def test_int_pow_negative_base() -> None:
    # Negative base, even exponent -> positive
    assert pow(-2, 2) == 4
    assert pow(-3, 2) == 9
    assert pow(-2, 4) == 16
    # Negative base, odd exponent -> negative
    assert pow(-2, 1) == -2
    assert pow(-2, 3) == -8
    assert pow(-3, 3) == -27
    # -1 to even/odd powers
    assert pow(-1, 0) == 1
    assert pow(-1, 1) == -1
    assert pow(-1, 2) == 1
    assert pow(-1, 3) == -1
    assert pow(-1, 100) == 1
    assert pow(-1, 101) == -1


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_int_arithmetic_basic()
        passed += 1
        print("  PASS test_int_arithmetic_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_arithmetic_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_arithmetic_basic: " + str(e))
    try:
        test_int_arithmetic_identity()
        passed += 1
        print("  PASS test_int_arithmetic_identity")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_arithmetic_identity: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_arithmetic_identity: " + str(e))
    try:
        test_int_arithmetic_associative()
        passed += 1
        print("  PASS test_int_arithmetic_associative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_arithmetic_associative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_arithmetic_associative: " + str(e))
    try:
        test_int_arithmetic_commutative()
        passed += 1
        print("  PASS test_int_arithmetic_commutative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_arithmetic_commutative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_arithmetic_commutative: " + str(e))
    try:
        test_int_floor_division_positive()
        passed += 1
        print("  PASS test_int_floor_division_positive")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_floor_division_positive: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_floor_division_positive: " + str(e))
    try:
        test_int_floor_division_negative()
        passed += 1
        print("  PASS test_int_floor_division_negative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_floor_division_negative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_floor_division_negative: " + str(e))
    try:
        test_int_modulo_positive()
        passed += 1
        print("  PASS test_int_modulo_positive")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_modulo_positive: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_modulo_positive: " + str(e))
    try:
        test_int_modulo_negative()
        passed += 1
        print("  PASS test_int_modulo_negative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_modulo_negative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_modulo_negative: " + str(e))
    try:
        test_int_division_identity()
        passed += 1
        print("  PASS test_int_division_identity")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_division_identity: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_division_identity: " + str(e))
    try:
        test_int_negation()
        passed += 1
        print("  PASS test_int_negation")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_negation: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_negation: " + str(e))
    try:
        test_int_comparisons_equality()
        passed += 1
        print("  PASS test_int_comparisons_equality")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_comparisons_equality: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_comparisons_equality: " + str(e))
    try:
        test_int_comparisons_ordering()
        passed += 1
        print("  PASS test_int_comparisons_ordering")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_comparisons_ordering: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_comparisons_ordering: " + str(e))
    try:
        test_int_comparisons_transitive()
        passed += 1
        print("  PASS test_int_comparisons_transitive")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_comparisons_transitive: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_comparisons_transitive: " + str(e))
    try:
        test_int_conversion_from_int()
        passed += 1
        print("  PASS test_int_conversion_from_int")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_conversion_from_int: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_conversion_from_int: " + str(e))
    try:
        test_int_conversion_from_bool()
        passed += 1
        print("  PASS test_int_conversion_from_bool")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_conversion_from_bool: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_conversion_from_bool: " + str(e))
    try:
        test_int_conversion_from_string()
        passed += 1
        print("  PASS test_int_conversion_from_string")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_conversion_from_string: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_conversion_from_string: " + str(e))
    try:
        test_int_conversion_whitespace()
        passed += 1
        print("  PASS test_int_conversion_whitespace")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_conversion_whitespace: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_conversion_whitespace: " + str(e))
    try:
        test_int_string_repr()
        passed += 1
        print("  PASS test_int_string_repr")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_string_repr: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_string_repr: " + str(e))
    try:
        test_int_abs()
        passed += 1
        print("  PASS test_int_abs")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_abs: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_abs: " + str(e))
    try:
        test_int_min_two_args()
        passed += 1
        print("  PASS test_int_min_two_args")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_min_two_args: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_min_two_args: " + str(e))
    try:
        test_int_min_multiple_args()
        passed += 1
        print("  PASS test_int_min_multiple_args")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_min_multiple_args: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_min_multiple_args: " + str(e))
    try:
        test_int_max_two_args()
        passed += 1
        print("  PASS test_int_max_two_args")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_max_two_args: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_max_two_args: " + str(e))
    try:
        test_int_max_multiple_args()
        passed += 1
        print("  PASS test_int_max_multiple_args")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_max_multiple_args: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_max_multiple_args: " + str(e))
    try:
        test_int_divmod_positive()
        passed += 1
        print("  PASS test_int_divmod_positive")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_divmod_positive: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_divmod_positive: " + str(e))
    try:
        test_int_divmod_negative()
        passed += 1
        print("  PASS test_int_divmod_negative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_divmod_negative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_divmod_negative: " + str(e))
    try:
        test_int_divmod_identity()
        passed += 1
        print("  PASS test_int_divmod_identity")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_divmod_identity: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_divmod_identity: " + str(e))
    try:
        test_int_pow_basic()
        passed += 1
        print("  PASS test_int_pow_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_pow_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_pow_basic: " + str(e))
    try:
        test_int_pow_edge_cases()
        passed += 1
        print("  PASS test_int_pow_edge_cases")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_pow_edge_cases: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_pow_edge_cases: " + str(e))
    try:
        test_hex_oct_bin()
        passed += 1
        print("  PASS test_hex_oct_bin")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_hex_oct_bin: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_hex_oct_bin: " + str(e))
    try:
        test_int_pow_negative_base()
        passed += 1
        print("  PASS test_int_pow_negative_base")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_int_pow_negative_base: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_int_pow_negative_base: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
