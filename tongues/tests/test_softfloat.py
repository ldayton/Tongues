"""TestFloat-style tests for the softfloat library.

Uses weighted random generation to hit boundary cases with higher probability:
- Special exponents: 0 (subnormal), 1, 0x3FE-0x400 (near 1.0), 0x7FE-0x7FF (inf/NaN)
- Special significands: 0, 1, all-ones, single-bit, high-bit patterns
- Two signs mixed uniformly

500,000 rounds per operation by default.
"""

import math
import random
import struct

import pytest

from src.backend.softfloat import (
    F64_SIGN,
    f64_abs,
    f64_add,
    f64_ceil,
    f64_div,
    f64_eq,
    f64_floor,
    f64_fmod,
    f64_le,
    f64_lt,
    f64_max,
    f64_min,
    f64_mul,
    f64_neg,
    f64_round,
    f64_sqrt,
    f64_sub,
    f64_to_i64,
    f64_to_str,
    i64_to_f64,
    is_nan_f64,
    str_to_f64,
)

ROUNDS = 500_000
SEED = 0xF64


def f2i(f: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", f))[0]


def i2f(i: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", i))[0]


# ---------------------------------------------------------------------------
# Weighted random generation (TestFloat-style)
# ---------------------------------------------------------------------------

# Exponents likely to trigger edge cases
SPECIAL_EXPS = [
    0x000,  # subnormal / zero
    0x001,  # smallest normal
    0x002,
    0x3FD,  # near 0.5
    0x3FE,  # 0.5 .. 1.0
    0x3FF,  # 1.0 .. 2.0
    0x400,  # 2.0 .. 4.0
    0x401,
    0x432,  # near int53 boundary
    0x433,  # 2^52 (ULP = 1)
    0x434,  # 2^53 (ULP = 2)
    0x7FD,  # near overflow
    0x7FE,  # largest finite
    0x7FF,  # inf / NaN
]

# Significands likely to trigger edge cases
SPECIAL_SIGS = [
    0x0000000000000,  # zero
    0x0000000000001,  # smallest
    0x0000000000002,
    0x4000000000000,  # mid-range single bit
    0x8000000000000,  # half (0.5 in the fraction)
    0xFFFFFFFFFFFFC,  # near max, round-bit patterns
    0xFFFFFFFFFFFFE,
    0xFFFFFFFFFFFFF,  # max
    0x0000000000010,  # low bits
    0x0010000000000,  # isolated middle bit
]


def weighted_f64(rng: random.Random) -> int:
    """Generate a float64 bit pattern weighted toward boundary cases."""
    r: int = rng.randint(0, 99)
    if r < 30:
        # 30%: special exponent + random significand
        exp = rng.choice(SPECIAL_EXPS)
        sig = rng.randint(0, 0xFFFFFFFFFFFFF)
    elif r < 50:
        # 20%: random exponent + special significand
        exp = rng.randint(0, 0x7FF)
        sig = rng.choice(SPECIAL_SIGS)
    elif r < 60:
        # 10%: special exponent + special significand
        exp = rng.choice(SPECIAL_EXPS)
        sig = rng.choice(SPECIAL_SIGS)
    else:
        # 40%: fully random
        exp = rng.randint(0, 0x7FF)
        sig = rng.randint(0, 0xFFFFFFFFFFFFF)
    sign = rng.randint(0, 1)
    return (sign << 63) | (exp << 52) | sig


def check_bits(got: int, expected: int, *, zero_sign_sensitive: bool = True) -> bool:
    """Compare two float64 bit patterns, treating all NaNs as equal."""
    if is_nan_f64(got) and is_nan_f64(expected):
        return True
    if not zero_sign_sensitive:
        if (got | expected) == F64_SIGN:
            return True
    return got == expected


def check_bool(got: bool, expected: bool) -> bool:
    return got == expected


# ---------------------------------------------------------------------------
# Binary operations
# ---------------------------------------------------------------------------


def ref_binary(op: str, a: float, b: float) -> float | None:
    """Python reference for binary float ops. Returns None to skip."""
    try:
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "div":
            return a / b
        if op == "fmod":
            if b == 0.0 or math.isinf(a) or math.isnan(a) or math.isnan(b):
                return None
            return math.fmod(a, b)
    except (ZeroDivisionError, ValueError, OverflowError):
        return None
    return None


BINARY_OPS = {
    "add": f64_add,
    "sub": f64_sub,
    "mul": f64_mul,
    "div": f64_div,
    "fmod": f64_fmod,
}


@pytest.mark.parametrize("op", BINARY_OPS)
def test_binary(op: str):
    fn = BINARY_OPS[op]
    rng = random.Random(SEED)
    fails = 0
    tested = 0
    first_failure = ""
    for _ in range(ROUNDS):
        a_bits = weighted_f64(rng)
        b_bits = weighted_f64(rng)
        a_f = i2f(a_bits)
        b_f = i2f(b_bits)
        ref = ref_binary(op, a_f, b_f)
        if ref is None:
            continue
        tested += 1
        expected = f2i(ref)
        got = fn(a_bits, b_bits)
        if not check_bits(got, expected):
            fails += 1
            if fails == 1:
                first_failure = (
                    f"{op}({a_f}, {b_f}): got {i2f(got)} ({got:#018x}), "
                    f"expected {ref} ({expected:#018x})"
                )
    assert fails == 0, f"{fails}/{tested} failures. First: {first_failure}"


# ---------------------------------------------------------------------------
# Sqrt
# ---------------------------------------------------------------------------


def test_sqrt():
    rng = random.Random(SEED)
    fails = 0
    tested = 0
    first_failure = ""
    for _ in range(ROUNDS):
        a_bits = weighted_f64(rng)
        a_f = i2f(a_bits)
        if a_f < 0 or math.isnan(a_f):
            # Negative or NaN — just check we don't crash
            got = f64_sqrt(a_bits)
            if a_bits == F64_SIGN:
                # sqrt(-0) = -0 per IEEE 754
                assert got == F64_SIGN
            elif not math.isnan(a_f):
                assert is_nan_f64(got)
            continue
        tested += 1
        expected = f2i(math.sqrt(a_f))
        got = f64_sqrt(a_bits)
        if not check_bits(got, expected):
            fails += 1
            if fails == 1:
                first_failure = (
                    f"sqrt({a_f}): got {i2f(got)} ({got:#018x}), "
                    f"expected {math.sqrt(a_f)} ({expected:#018x})"
                )
    assert fails == 0, f"{fails}/{tested} failures. First: {first_failure}"


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["eq", "lt", "le"])
def test_comparison(op: str):
    fn = {"eq": f64_eq, "lt": f64_lt, "le": f64_le}[op]
    py = {
        "eq": lambda a, b: a == b,
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
    }[op]
    rng = random.Random(SEED)
    fails = 0
    first_failure = ""
    for _ in range(ROUNDS):
        a_bits = weighted_f64(rng)
        b_bits = weighted_f64(rng)
        a_f = i2f(a_bits)
        b_f = i2f(b_bits)
        expected = py(a_f, b_f)
        got = fn(a_bits, b_bits)
        if not check_bool(got, expected):
            fails += 1
            if fails == 1:
                first_failure = f"{op}({a_f}, {b_f}): got {got}, expected {expected}"
    assert fails == 0, f"{fails}/{ROUNDS} failures. First: {first_failure}"


# ---------------------------------------------------------------------------
# Unary: neg, abs
# ---------------------------------------------------------------------------


def test_neg():
    rng = random.Random(SEED)
    for _ in range(ROUNDS):
        a = weighted_f64(rng)
        got = f64_neg(a)
        assert got == (a ^ F64_SIGN)


def test_abs():
    rng = random.Random(SEED)
    for _ in range(ROUNDS):
        a = weighted_f64(rng)
        got = f64_abs(a)
        assert got == (a & 0x7FFFFFFFFFFFFFFF)


# ---------------------------------------------------------------------------
# Min / Max
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["min", "max"])
def test_minmax(op: str):
    fn = {"min": f64_min, "max": f64_max}[op]
    rng = random.Random(SEED)
    fails = 0
    first_failure = ""
    for _ in range(ROUNDS):
        a_bits = weighted_f64(rng)
        b_bits = weighted_f64(rng)
        a_f = i2f(a_bits)
        b_f = i2f(b_bits)
        got = fn(a_bits, b_bits)
        if is_nan_f64(a_bits) or is_nan_f64(b_bits):
            if not is_nan_f64(got):
                fails += 1
                if fails == 1:
                    first_failure = f"{op}(NaN, ...) should be NaN"
            continue
        py_ref = min(a_f, b_f) if op == "min" else max(a_f, b_f)
        expected = f2i(py_ref)
        if not check_bits(got, expected, zero_sign_sensitive=False):
            fails += 1
            if fails == 1:
                first_failure = f"{op}({a_f}, {b_f}): got {i2f(got)}, expected {py_ref}"
    assert fails == 0, f"{fails}/{ROUNDS} failures. First: {first_failure}"


# ---------------------------------------------------------------------------
# Conversions: i64_to_f64, f64_to_i64
# ---------------------------------------------------------------------------


def test_i64_to_f64():
    rng = random.Random(SEED)
    fails = 0
    first_failure = ""
    for _ in range(ROUNDS):
        n = rng.randint(-(2**63), 2**63 - 1)
        got = i64_to_f64(n)
        expected = f2i(float(n))
        if got != expected:
            fails += 1
            if fails == 1:
                first_failure = (
                    f"i64_to_f64({n}): got {i2f(got)} ({got:#018x}), "
                    f"expected {float(n)} ({expected:#018x})"
                )
    assert fails == 0, f"{fails}/{ROUNDS} failures. First: {first_failure}"


def test_f64_to_i64():
    rng = random.Random(SEED)
    fails = 0
    tested = 0
    first_failure = ""
    for _ in range(ROUNDS):
        a_bits = weighted_f64(rng)
        a_f = i2f(a_bits)
        if math.isnan(a_f) or math.isinf(a_f) or abs(a_f) >= 2.0**63:
            continue
        tested += 1
        got = f64_to_i64(a_bits)
        expected = int(a_f)
        if got != expected:
            fails += 1
            if fails == 1:
                first_failure = f"f64_to_i64({a_f}): got {got}, expected {expected}"
    assert fails == 0, f"{fails}/{tested} failures. First: {first_failure}"


# ---------------------------------------------------------------------------
# Floor, Ceil, Round
# ---------------------------------------------------------------------------


def ref_round_haz(f: float) -> int:
    """Round half-away-from-zero (not Python's half-to-even)."""
    t = int(f)
    frac = f - t
    if frac >= 0.5:
        return t + 1
    elif frac <= -0.5:
        return t - 1
    return t


@pytest.mark.parametrize(
    "op,ref",
    [
        ("floor", math.floor),
        ("ceil", math.ceil),
        ("round", ref_round_haz),
    ],
)
def test_float_to_int(op: str, ref):
    fn = {"floor": f64_floor, "ceil": f64_ceil, "round": f64_round}[op]
    rng = random.Random(SEED)
    fails = 0
    tested = 0
    first_failure = ""
    for _ in range(ROUNDS):
        a_bits = weighted_f64(rng)
        a_f = i2f(a_bits)
        if math.isnan(a_f) or math.isinf(a_f) or abs(a_f) >= 2.0**63:
            continue
        tested += 1
        got = fn(a_bits)
        expected = ref(a_f)
        if got != expected:
            fails += 1
            if fails == 1:
                first_failure = f"{op}({a_f}): got {got}, expected {expected}"
    assert fails == 0, f"{fails}/{tested} failures. First: {first_failure}"


# ---------------------------------------------------------------------------
# ToString round-trip
# ---------------------------------------------------------------------------


def test_to_str():
    rng = random.Random(SEED)
    fails = 0
    first_failure = ""
    for _ in range(ROUNDS):
        a_bits = weighted_f64(rng)
        a_f = i2f(a_bits)
        got = f64_to_str(a_bits)
        if math.isnan(a_f):
            assert got == "NaN"
            continue
        if math.isinf(a_f):
            expected = "-Inf" if a_f < 0 else "Inf"
            assert got == expected
            continue
        if a_f == 0.0:
            expected = "-0.0" if (a_bits >> 63) else "0.0"
            assert got == expected
            continue
        expected = "%.16e" % a_f
        if got != expected:
            fails += 1
            if fails == 1:
                first_failure = f"to_str({a_f}): got {got!r}, expected {expected!r}"
    assert fails == 0, f"{fails}/{ROUNDS} failures. First: {first_failure}"


def test_round_trip():
    """f64_to_str -> str_to_f64 must recover the original bits."""
    rng = random.Random(SEED)
    fails = 0
    first_failure = ""
    for _ in range(ROUNDS):
        a_bits = weighted_f64(rng)
        s = f64_to_str(a_bits)
        back = str_to_f64(s)
        if is_nan_f64(a_bits):
            if not is_nan_f64(back):
                fails += 1
                if fails == 1:
                    first_failure = f"NaN round-trip: {s!r} -> {back:#018x}"
            continue
        if back != a_bits:
            fails += 1
            if fails == 1:
                first_failure = f"round-trip {a_bits:#018x} -> {s!r} -> {back:#018x}"
    assert fails == 0, f"{fails}/{ROUNDS} failures. First: {first_failure}"
