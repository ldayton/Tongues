"""Float object tests.

Avoids tests that assume a particular float width (32-bit vs 64-bit).
"""

import sys


def test_float_equality() -> None:
    """Float equality comparisons."""
    assert 1.0 == 1.0
    assert 0.0 == 0.0
    assert -1.0 == -1.0
    assert not (1.0 == 2.0)
    assert 1.0 != 2.0


def test_float_ordering() -> None:
    """Float ordering comparisons."""
    assert 1.0 < 2.0
    assert 2.0 > 1.0
    assert -1.0 < 0.0
    assert 0.0 < 1.0
    assert 1.0 <= 1.0
    assert 1.0 <= 2.0
    assert 2.0 >= 1.0
    assert 2.0 >= 2.0


def test_float_arithmetic() -> None:
    """Basic float arithmetic."""
    assert 1.0 + 2.0 == 3.0
    assert 5.0 - 3.0 == 2.0
    assert 2.0 * 3.0 == 6.0
    assert 6.0 / 2.0 == 3.0
    assert -1.0 + 1.0 == 0.0
    assert 0.0 * 100.0 == 0.0


def test_float_negation() -> None:
    """Float negation."""
    x: float = 1.5
    assert -x == -1.5
    assert -(-x) == 1.5
    assert -0.0 == 0.0  # negative zero equals zero


def test_float_division() -> None:
    """Float division always returns float."""
    assert 7.0 / 2.0 == 3.5
    assert 1.0 / 4.0 == 0.25
    assert -1.0 / 2.0 == -0.5


def test_int_to_float() -> None:
    """Int to float conversion."""
    assert float(0) == 0.0
    assert float(1) == 1.0
    assert float(-1) == -1.0
    assert float(100) == 100.0


def test_float_to_int() -> None:
    """Float to int truncates toward zero."""
    assert int(1.0) == 1
    assert int(1.9) == 1
    assert int(-1.9) == -1  # truncates toward zero, not floor
    assert int(0.0) == 0
    assert int(-0.0) == 0


def test_float_bool() -> None:
    """Float truthiness - 0.0 is falsy."""
    assert bool(1.0) == True
    assert bool(-1.0) == True
    assert bool(0.5) == True
    assert bool(0.0) == False
    assert bool(-0.0) == False
    assert not 0.0
    assert 1.0


def test_float_abs() -> None:
    """Absolute value of floats."""
    assert abs(1.5) == 1.5
    assert abs(-1.5) == 1.5
    assert abs(0.0) == 0.0
    assert abs(-0.0) == 0.0


def test_float_int_mixed() -> None:
    """Mixed float/int arithmetic promotes to float."""
    x: float = 1.5 + 1
    assert x == 2.5
    x = 3.0 * 2
    assert x == 6.0
    x = 5.0 - 3
    assert x == 2.0


def test_float_comparison_with_int() -> None:
    """Floats can be compared with ints."""
    assert 1.0 == 1
    assert 2.0 > 1
    assert 0.5 < 1
    assert 1 < 1.5
    assert 2 == 2.0


def test_positive_infinity() -> None:
    """Positive infinity behavior."""
    inf: float = float("inf")
    assert inf > 0.0
    assert inf > 1000000.0
    assert inf == inf
    assert inf + 1.0 == inf
    assert inf * 2.0 == inf
    assert 1.0 / inf == 0.0


def test_negative_infinity() -> None:
    """Negative infinity behavior."""
    ninf: float = float("-inf")
    assert ninf < 0.0
    assert ninf < -1000000.0
    assert ninf == ninf
    assert ninf - 1.0 == ninf
    assert ninf * 2.0 == ninf


def test_infinity_comparison() -> None:
    """Infinity comparisons."""
    inf: float = float("inf")
    ninf: float = float("-inf")
    assert inf > ninf
    assert ninf < inf
    assert inf != ninf


def test_nan_basics() -> None:
    """NaN (Not a Number) basics."""
    nan: float = float("nan")
    # NaN is not equal to anything, including itself
    assert not (nan == nan)
    assert nan != nan
    assert not (nan < 0.0)
    assert not (nan > 0.0)
    assert not (nan == 0.0)


def test_nan_comparisons() -> None:
    """All NaN comparisons return False (except !=)."""
    nan: float = float("nan")
    assert not (nan < nan)
    assert not (nan > nan)
    assert not (nan <= nan)
    assert not (nan >= nan)
    assert not (nan < 1.0)
    assert not (nan > 1.0)
    assert not (nan <= 1.0)
    assert not (nan >= 1.0)
    # != is the only True comparison
    assert nan != 1.0
    assert 1.0 != nan


def test_nan_arithmetic() -> None:
    """NaN propagates through arithmetic."""
    nan: float = float("nan")
    x: float = nan + 1.0
    assert x != x  # result is NaN
    x = nan - 1.0
    assert x != x
    x = nan * 2.0
    assert x != x
    x = nan / 2.0
    assert x != x
    x = 1.0 + nan
    assert x != x


def test_inf_produces_nan() -> None:
    """Operations that produce NaN from infinity."""
    inf: float = float("inf")
    ninf: float = float("-inf")
    # inf - inf = nan
    x: float = inf - inf
    assert x != x
    # inf + (-inf) = nan
    x = inf + ninf
    assert x != x
    # inf * 0 = nan
    x = inf * 0.0
    assert x != x
    # inf / inf = nan
    x = inf / inf
    assert x != x


def test_inf_division() -> None:
    """Division involving infinity."""
    inf: float = float("inf")
    # finite / inf = 0
    assert 1.0 / inf == 0.0
    assert 0.0 / inf == 0.0
    assert 1000000.0 / inf == 0.0
    assert -1.0 / inf == 0.0
    # inf / finite = inf
    assert inf / 1.0 == inf
    assert inf / 1000000.0 == inf


def test_zero_signs() -> None:
    """Positive and negative zero are equal."""
    pz: float = 0.0
    nz: float = -0.0
    assert pz == nz
    assert not (pz < nz)
    assert not (pz > nz)


def test_float_floor_division() -> None:
    """Floor division with floats."""
    assert 7.0 // 2.0 == 3.0
    assert 7.5 // 2.0 == 3.0
    assert 7.0 // 2.5 == 2.0
    assert -7.0 // 2.0 == -4.0  # floors toward -inf
    assert 7.0 // -2.0 == -4.0
    assert -7.0 // -2.0 == 3.0
    # Mixed int/float
    assert 7.0 // 2 == 3.0
    assert 7 // 2.0 == 3.0


def test_float_modulo() -> None:
    """Modulo with floats."""
    assert 7.0 % 3.0 == 1.0
    assert 7.5 % 2.0 == 1.5
    assert -7.0 % 3.0 == 2.0  # result has sign of divisor
    assert 7.0 % -3.0 == -2.0
    assert -7.0 % -3.0 == -1.0
    # Mixed int/float
    assert 7.0 % 3 == 1.0
    assert 7 % 3.0 == 1.0


def test_float_divmod() -> None:
    """divmod with floats."""
    q, r = divmod(7.0, 3.0)
    assert q == 2.0
    assert r == 1.0
    q, r = divmod(-7.0, 3.0)
    assert q == -3.0
    assert r == 2.0
    q, r = divmod(7.5, 2.0)
    assert q == 3.0
    assert r == 1.5


def test_float_divmod_identity() -> None:
    """Verify: x == (x // y) * y + (x % y) for exact values."""
    x: float = 8.0
    y: float = 2.0
    assert x == (x // y) * y + (x % y)
    x = -8.0
    assert x == (x // y) * y + (x % y)


def test_float_pow() -> None:
    """pow() with floats."""
    assert pow(2.0, 3.0) == 8.0
    assert pow(4.0, 0.5) == 2.0  # square root
    assert pow(8.0, 1.0 / 3.0) - 2.0 < 0.0001  # cube root (approx)
    assert pow(2.0, -1.0) == 0.5
    assert pow(2.0, 0.0) == 1.0
    assert pow(0.0, 0.0) == 1.0


def test_float_pow_special() -> None:
    """pow() with special values."""
    inf: float = float("inf")
    assert pow(2.0, inf) == inf
    assert pow(0.5, inf) == 0.0
    assert pow(2.0, -inf) == 0.0
    assert pow(0.5, -inf) == inf
    assert pow(inf, 0.0) == 1.0


def test_float_round_basic() -> None:
    """round() basics."""
    assert round(1.4) == 1
    assert round(1.5) == 2
    assert round(1.6) == 2
    assert round(-1.4) == -1
    assert round(-1.5) == -2
    assert round(-1.6) == -2


def test_float_round_half_even() -> None:
    """round() uses banker's rounding (round half to even)."""
    assert round(0.5) == 0
    assert round(1.5) == 2
    assert round(2.5) == 2
    assert round(3.5) == 4
    assert round(4.5) == 4


def test_float_round_ndigits() -> None:
    """round() with ndigits."""
    # Use values that round clearly (not .5)
    assert round(1.234, 1) == 1.2
    assert round(1.267, 1) == 1.3
    assert round(1.234, 0) == 1.0
    assert round(1.789, 0) == 2.0


def test_float_round_integers() -> None:
    """round() on values that are clearly one side or the other."""
    assert round(0.1) == 0
    assert round(0.9) == 1
    assert round(1.1) == 1
    assert round(1.9) == 2
    assert round(-0.1) == 0
    assert round(-0.9) == -1


def test_float_min_max() -> None:
    """min/max with floats."""
    assert min(1.5, 2.5) == 1.5
    assert max(1.5, 2.5) == 2.5
    assert min(-1.5, 1.5) == -1.5
    assert max(-1.5, 1.5) == 1.5
    assert min(1.0, 2.0, 3.0) == 1.0
    assert max(1.0, 2.0, 3.0) == 3.0


def test_float_min_max_special() -> None:
    """min/max with infinity and NaN."""
    inf: float = float("inf")
    ninf: float = float("-inf")
    assert min(inf, 1.0) == 1.0
    assert max(ninf, 1.0) == 1.0
    assert min(inf, ninf) == ninf
    assert max(inf, ninf) == inf


def test_float_from_string() -> None:
    """Float from string conversion."""
    assert float("1.5") == 1.5
    assert float("-1.5") == -1.5
    assert float("1e10") == 1e10
    assert float("1E10") == 1e10
    assert float("1.5e2") == 150.0
    assert float("1.5e-2") == 0.015
    assert float("+1.5") == 1.5
    assert float("  1.5  ") == 1.5  # whitespace stripped


def test_float_from_string_special() -> None:
    """Float from string - special values."""
    assert float("inf") == float("inf")
    assert float("-inf") == float("-inf")
    assert float("Infinity") == float("inf")
    nan: float = float("nan")
    assert nan != nan  # NaN check


def test_float_str_repr() -> None:
    """str/repr of floats."""
    assert str(1.5) == "1.5"
    assert str(-1.5) == "-1.5"
    assert str(1.0) == "1.0"
    assert repr(1.5) == "1.5"


def test_float_exponent_notation() -> None:
    """Very large/small floats use exponent notation."""
    big: float = 1e20
    small: float = 1e-20
    assert big > 0
    assert small > 0
    assert small < 1
    assert big * small == 1.0


def test_float_not_exact() -> None:
    """Some float operations are not exact."""
    # 0.1 + 0.2 may not equal 0.3 exactly in binary float
    a: float = 0.1 + 0.2
    b: float = 0.3
    # Don't test equality - it's implementation-dependent
    # Just verify they're close
    diff: float = a - b
    assert diff < 0.01
    assert diff > -0.01


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_float_equality", test_float_equality),
        ("test_float_ordering", test_float_ordering),
        ("test_float_arithmetic", test_float_arithmetic),
        ("test_float_negation", test_float_negation),
        ("test_float_division", test_float_division),
        ("test_int_to_float", test_int_to_float),
        ("test_float_to_int", test_float_to_int),
        ("test_float_bool", test_float_bool),
        ("test_float_abs", test_float_abs),
        ("test_float_int_mixed", test_float_int_mixed),
        ("test_float_comparison_with_int", test_float_comparison_with_int),
        ("test_positive_infinity", test_positive_infinity),
        ("test_negative_infinity", test_negative_infinity),
        ("test_infinity_comparison", test_infinity_comparison),
        ("test_nan_basics", test_nan_basics),
        ("test_nan_comparisons", test_nan_comparisons),
        ("test_nan_arithmetic", test_nan_arithmetic),
        ("test_inf_produces_nan", test_inf_produces_nan),
        ("test_inf_division", test_inf_division),
        ("test_zero_signs", test_zero_signs),
        ("test_float_floor_division", test_float_floor_division),
        ("test_float_modulo", test_float_modulo),
        ("test_float_divmod", test_float_divmod),
        ("test_float_divmod_identity", test_float_divmod_identity),
        ("test_float_pow", test_float_pow),
        ("test_float_pow_special", test_float_pow_special),
        ("test_float_round_basic", test_float_round_basic),
        ("test_float_round_half_even", test_float_round_half_even),
        ("test_float_round_ndigits", test_float_round_ndigits),
        ("test_float_round_integers", test_float_round_integers),
        ("test_float_min_max", test_float_min_max),
        ("test_float_min_max_special", test_float_min_max_special),
        ("test_float_from_string", test_float_from_string),
        ("test_float_from_string_special", test_float_from_string_special),
        ("test_float_str_repr", test_float_str_repr),
        ("test_float_exponent_notation", test_float_exponent_notation),
        ("test_float_not_exact", test_float_not_exact),
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
