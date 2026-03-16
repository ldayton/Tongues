"""Softfloat IEEE 754 double-precision tests."""

import sys

from lib.softfloat import (
    F64_ZERO,
    F64_NEG_ZERO,
    F64_INF,
    F64_NEG_INF,
    DEFAULT_NAN,
    sign_f64,
    exp_f64,
    frac_f64,
    pack_f64,
    is_nan_f64,
    is_inf_f64,
    f64_add,
    f64_sub,
    f64_mul,
    f64_div,
    f64_sqrt,
    f64_fmod,
    f64_neg,
    f64_abs,
    f64_eq,
    f64_lt,
    f64_le,
    f64_min,
    f64_max,
    i64_to_f64,
    f64_to_i64,
    f64_floor,
    f64_ceil,
    f64_round,
    f64_to_str,
    str_to_f64,
)

# -- IEEE 754 bit patterns for common values --

ONE: int = 0x3FF0000000000000
TWO: int = 0x4000000000000000
THREE: int = 0x4008000000000000
FOUR: int = 0x4010000000000000
FIVE: int = 0x4014000000000000
SIX: int = 0x4018000000000000
SEVEN: int = 0x401C000000000000
TEN: int = 0x4024000000000000
FORTY_TWO: int = 0x4045000000000000
HUNDRED: int = 0x4059000000000000
NEG_ONE: int = 0xBFF0000000000000
NEG_SEVEN: int = 0xC01C000000000000
NEG_FORTY_TWO: int = 0xC045000000000000
HALF: int = 0x3FE0000000000000
MILLION: int = 0x412E848000000000


def test_i64_to_f64() -> None:
    assert i64_to_f64(0) == F64_ZERO
    assert i64_to_f64(1) == ONE
    assert i64_to_f64(2) == TWO
    assert i64_to_f64(3) == THREE
    assert i64_to_f64(4) == FOUR
    assert i64_to_f64(5) == FIVE
    assert i64_to_f64(6) == SIX
    assert i64_to_f64(7) == SEVEN
    assert i64_to_f64(10) == TEN
    assert i64_to_f64(42) == FORTY_TWO
    assert i64_to_f64(100) == HUNDRED
    assert i64_to_f64(-1) == NEG_ONE
    assert i64_to_f64(-42) == NEG_FORTY_TWO
    assert i64_to_f64(1000000) == MILLION


def test_bit_decomposition() -> None:
    assert sign_f64(ONE) == 0
    assert sign_f64(NEG_ONE) == 1
    assert sign_f64(F64_ZERO) == 0
    assert sign_f64(F64_NEG_ZERO) == 1
    assert exp_f64(ONE) == 0x3FF
    assert exp_f64(TWO) == 0x400
    assert exp_f64(F64_ZERO) == 0
    assert exp_f64(F64_INF) == 0x7FF
    assert frac_f64(ONE) == 0
    assert frac_f64(THREE) == 0x8000000000000
    assert frac_f64(F64_ZERO) == 0
    # pack round-trip
    s: int = sign_f64(FORTY_TWO)
    e: int = exp_f64(FORTY_TWO)
    f: int = frac_f64(FORTY_TWO)
    assert pack_f64(s, e, f) == FORTY_TWO
    s = sign_f64(NEG_SEVEN)
    e = exp_f64(NEG_SEVEN)
    f = frac_f64(NEG_SEVEN)
    assert pack_f64(s, e, f) == NEG_SEVEN
    # NaN / Inf predicates
    assert is_nan_f64(DEFAULT_NAN) == True
    assert is_nan_f64(ONE) == False
    assert is_nan_f64(F64_INF) == False
    assert is_nan_f64(F64_ZERO) == False
    assert is_inf_f64(F64_INF) == True
    assert is_inf_f64(F64_NEG_INF) == True
    assert is_inf_f64(ONE) == False
    assert is_inf_f64(DEFAULT_NAN) == False


def test_add_sub() -> None:
    # 1 + 2 = 3
    assert f64_add(ONE, TWO) == THREE
    # 3 + 4 = 7
    assert f64_add(THREE, FOUR) == SEVEN
    # 42 + 0 = 42
    assert f64_add(FORTY_TWO, F64_ZERO) == FORTY_TWO
    # 1 + (-1) = 0
    assert f64_eq(f64_add(ONE, NEG_ONE), F64_ZERO)
    # 3 - 1 = 2
    assert f64_sub(THREE, ONE) == TWO
    # 7 - 4 = 3
    assert f64_sub(SEVEN, FOUR) == THREE
    # 1 - 3 = -2
    r: int = f64_sub(ONE, THREE)
    assert sign_f64(r) == 1
    assert f64_abs(r) == TWO
    # -1 + -1 = -2
    neg_two: int = f64_add(NEG_ONE, NEG_ONE)
    assert sign_f64(neg_two) == 1
    assert f64_abs(neg_two) == TWO
    # 5 - 5 = 0
    assert f64_eq(f64_sub(FIVE, FIVE), F64_ZERO)


def test_mul_div() -> None:
    # 2 * 3 = 6
    assert f64_mul(TWO, THREE) == SIX
    # 6 * 7 = 42
    assert f64_mul(SIX, SEVEN) == FORTY_TWO
    # 1 * x = x
    assert f64_mul(ONE, FORTY_TWO) == FORTY_TWO
    # 0 * x = 0
    assert f64_eq(f64_mul(F64_ZERO, SEVEN), F64_ZERO)
    # sign rules: neg * pos = neg
    r: int = f64_mul(NEG_ONE, SEVEN)
    assert r == NEG_SEVEN
    # neg * neg = pos
    assert f64_mul(NEG_ONE, NEG_ONE) == ONE
    # 6 / 2 = 3
    assert f64_div(SIX, TWO) == THREE
    # 42 / 7 = 6
    assert f64_div(FORTY_TWO, SEVEN) == SIX
    # 42 / 1 = 42
    assert f64_div(FORTY_TWO, ONE) == FORTY_TWO
    # non-integer: 1/3 via string
    third: int = f64_div(ONE, THREE)
    s: str = f64_to_str(third)
    assert s[0] == "3"
    assert s[1] == "."


def test_unary_ops() -> None:
    # neg flips sign
    assert f64_neg(ONE) == NEG_ONE
    assert f64_neg(NEG_ONE) == ONE
    assert f64_neg(F64_ZERO) == F64_NEG_ZERO
    assert f64_neg(F64_NEG_ZERO) == F64_ZERO
    # abs removes sign
    assert f64_abs(NEG_ONE) == ONE
    assert f64_abs(ONE) == ONE
    assert f64_abs(NEG_FORTY_TWO) == FORTY_TWO
    assert f64_abs(F64_NEG_ZERO) == F64_ZERO
    # sqrt(4) = 2, sqrt(100) = 10
    assert f64_sqrt(FOUR) == TWO
    assert f64_sqrt(HUNDRED) == TEN
    assert f64_sqrt(ONE) == ONE
    assert f64_sqrt(F64_ZERO) == F64_ZERO


def test_fmod() -> None:
    # 7 mod 3 = 1
    assert f64_fmod(SEVEN, THREE) == ONE
    # 10 mod 3 = 1
    assert f64_fmod(TEN, THREE) == ONE
    # 5 mod 2 = 1
    assert f64_fmod(FIVE, TWO) == ONE
    # 6 mod 3 = 0
    assert f64_eq(f64_fmod(SIX, THREE), F64_ZERO)
    # sign follows dividend: -7 mod 3 = -1
    r: int = f64_fmod(NEG_SEVEN, THREE)
    assert sign_f64(r) == 1
    assert f64_abs(r) == ONE
    # 7 mod -3 = 1 (positive)
    r = f64_fmod(SEVEN, f64_neg(THREE))
    assert sign_f64(r) == 0
    assert r == ONE
    # fmod(x, inf) = x
    assert f64_fmod(SEVEN, F64_INF) == SEVEN
    # fmod(inf, x) = NaN
    assert is_nan_f64(f64_fmod(F64_INF, THREE))
    # fmod(x, 0) = NaN
    assert is_nan_f64(f64_fmod(SEVEN, F64_ZERO))


def test_comparisons() -> None:
    # eq
    assert f64_eq(ONE, ONE) == True
    assert f64_eq(ONE, TWO) == False
    assert f64_eq(F64_ZERO, F64_NEG_ZERO) == True
    # lt
    assert f64_lt(ONE, TWO) == True
    assert f64_lt(TWO, ONE) == False
    assert f64_lt(ONE, ONE) == False
    assert f64_lt(NEG_ONE, ONE) == True
    assert f64_lt(ONE, NEG_ONE) == False
    # le
    assert f64_le(ONE, TWO) == True
    assert f64_le(ONE, ONE) == True
    assert f64_le(TWO, ONE) == False
    # min / max
    assert f64_min(ONE, TWO) == ONE
    assert f64_min(TWO, ONE) == ONE
    assert f64_max(ONE, TWO) == TWO
    assert f64_max(TWO, ONE) == TWO
    # infinity in comparisons
    assert f64_lt(HUNDRED, F64_INF) == True
    assert f64_lt(F64_NEG_INF, NEG_ONE) == True
    assert f64_min(F64_INF, ONE) == ONE
    assert f64_max(F64_NEG_INF, NEG_ONE) == NEG_ONE


def test_nan_behavior() -> None:
    nan: int = DEFAULT_NAN
    # NaN != NaN
    assert f64_eq(nan, nan) == False
    assert f64_eq(nan, ONE) == False
    # NaN comparisons always false
    assert f64_lt(nan, ONE) == False
    assert f64_lt(ONE, nan) == False
    assert f64_le(nan, ONE) == False
    assert f64_le(ONE, nan) == False
    # NaN propagation in arithmetic
    assert is_nan_f64(f64_add(nan, ONE))
    assert is_nan_f64(f64_sub(nan, ONE))
    assert is_nan_f64(f64_mul(nan, ONE))
    assert is_nan_f64(f64_div(nan, ONE))
    assert is_nan_f64(f64_add(ONE, nan))
    assert is_nan_f64(f64_div(ONE, nan))
    # min/max propagate NaN
    assert is_nan_f64(f64_min(nan, ONE))
    assert is_nan_f64(f64_max(nan, ONE))
    assert is_nan_f64(f64_min(ONE, nan))
    assert is_nan_f64(f64_max(ONE, nan))
    # sqrt of negative = NaN
    assert is_nan_f64(f64_sqrt(NEG_ONE))


def test_infinity_and_zero_edge_cases() -> None:
    # inf + inf = inf
    assert f64_add(F64_INF, F64_INF) == F64_INF
    # inf - inf = NaN
    assert is_nan_f64(f64_sub(F64_INF, F64_INF))
    # inf * 0 = NaN
    assert is_nan_f64(f64_mul(F64_INF, F64_ZERO))
    # 0 / 0 = NaN
    assert is_nan_f64(f64_div(F64_ZERO, F64_ZERO))
    # 1 / 0 = inf
    assert f64_div(ONE, F64_ZERO) == F64_INF
    # -1 / 0 = -inf
    assert f64_div(NEG_ONE, F64_ZERO) == F64_NEG_INF
    # sqrt(inf) = inf
    assert f64_sqrt(F64_INF) == F64_INF
    # sqrt(-0) = -0
    assert f64_sqrt(F64_NEG_ZERO) == F64_NEG_ZERO
    # inf / inf = NaN
    assert is_nan_f64(f64_div(F64_INF, F64_INF))
    # 0 + 0 = 0
    assert f64_eq(f64_add(F64_ZERO, F64_ZERO), F64_ZERO)
    # neg_inf + neg_inf = neg_inf
    assert f64_add(F64_NEG_INF, F64_NEG_INF) == F64_NEG_INF


def test_conversions_and_rounding() -> None:
    # f64_to_i64 round-trips for integers
    assert f64_to_i64(ONE) == 1
    assert f64_to_i64(FORTY_TWO) == 42
    assert f64_to_i64(NEG_SEVEN) == -7
    assert f64_to_i64(F64_ZERO) == 0
    assert f64_to_i64(HUNDRED) == 100
    assert f64_to_i64(MILLION) == 1000000
    # floor, ceil, round on integers are identity
    assert f64_floor(SEVEN) == 7
    assert f64_ceil(SEVEN) == 7
    assert f64_round(SEVEN) == 7
    # fractional via str_to_f64: 2.7
    v: int = str_to_f64("2.7")
    assert f64_floor(v) == 2
    assert f64_ceil(v) == 3
    assert f64_round(v) == 3
    # -2.7
    v = str_to_f64("-2.7")
    assert f64_floor(v) == -3
    assert f64_ceil(v) == -2
    assert f64_round(v) == -3
    # 0.5 rounds to 1
    assert f64_round(HALF) == 1
    # -0.5 rounds to -1
    assert f64_round(f64_neg(HALF)) == -1
    # 2.3
    v = str_to_f64("2.3")
    assert f64_floor(v) == 2
    assert f64_ceil(v) == 3
    assert f64_round(v) == 2


def test_string_conversion() -> None:
    # special values
    assert f64_to_str(F64_INF) == "Inf"
    assert f64_to_str(F64_NEG_INF) == "-Inf"
    assert f64_to_str(DEFAULT_NAN) == "NaN"
    assert f64_to_str(F64_ZERO) == "0.0"
    assert f64_to_str(F64_NEG_ZERO) == "-0.0"
    # str_to_f64 specials
    assert str_to_f64("Inf") == F64_INF
    assert str_to_f64("-Inf") == F64_NEG_INF
    assert is_nan_f64(str_to_f64("NaN"))
    # round-trip integer values through string
    s1: str = f64_to_str(ONE)
    assert str_to_f64(s1) == ONE
    s42: str = f64_to_str(FORTY_TWO)
    assert str_to_f64(s42) == FORTY_TWO
    s100: str = f64_to_str(HUNDRED)
    assert str_to_f64(s100) == HUNDRED
    # round-trip negative
    sn7: str = f64_to_str(NEG_SEVEN)
    assert str_to_f64(sn7) == NEG_SEVEN
    # invalid input -> NaN
    assert is_nan_f64(str_to_f64(""))
    assert is_nan_f64(str_to_f64("abc"))
    assert is_nan_f64(str_to_f64("1.2.3"))
    # str_to_f64 for simple values
    assert str_to_f64("0") == F64_ZERO
    assert str_to_f64("1") == ONE
    assert str_to_f64("-1") == NEG_ONE
    assert str_to_f64("42") == FORTY_TWO


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_i64_to_f64()
        passed += 1
        print("  PASS test_i64_to_f64")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_i64_to_f64: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_i64_to_f64: " + str(e))
    try:
        test_bit_decomposition()
        passed += 1
        print("  PASS test_bit_decomposition")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bit_decomposition: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bit_decomposition: " + str(e))
    try:
        test_add_sub()
        passed += 1
        print("  PASS test_add_sub")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_add_sub: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_add_sub: " + str(e))
    try:
        test_mul_div()
        passed += 1
        print("  PASS test_mul_div")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_mul_div: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_mul_div: " + str(e))
    try:
        test_unary_ops()
        passed += 1
        print("  PASS test_unary_ops")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_unary_ops: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_unary_ops: " + str(e))
    try:
        test_fmod()
        passed += 1
        print("  PASS test_fmod")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_fmod: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_fmod: " + str(e))
    try:
        test_comparisons()
        passed += 1
        print("  PASS test_comparisons")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_comparisons: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_comparisons: " + str(e))
    try:
        test_nan_behavior()
        passed += 1
        print("  PASS test_nan_behavior")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_nan_behavior: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_nan_behavior: " + str(e))
    try:
        test_infinity_and_zero_edge_cases()
        passed += 1
        print("  PASS test_infinity_and_zero_edge_cases")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_infinity_and_zero_edge_cases: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_infinity_and_zero_edge_cases: " + str(e))
    try:
        test_conversions_and_rounding()
        passed += 1
        print("  PASS test_conversions_and_rounding")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_conversions_and_rounding: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_conversions_and_rounding: " + str(e))
    try:
        test_string_conversion()
        passed += 1
        print("  PASS test_string_conversion")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_conversion: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_conversion: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
