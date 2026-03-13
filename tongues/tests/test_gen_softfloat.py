"""TestFloat-style tests for the softfloat library.

Uses weighted random generation to hit boundary cases with higher probability:
- Special exponents: 0 (subnormal), 1, 0x3FE-0x400 (near 1.0), 0x7FE-0x7FF (inf/NaN)
- Special significands: 0, 1, all-ones, single-bit, high-bit patterns
- Two signs mixed uniformly

500,000 rounds per operation by default.
"""

from contextlib import contextmanager
import math
import random
import struct

import pytest

pytestmark = pytest.mark.timeout(30)

from src.lib.softfloat import (
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

import src.lib.softfloat as _sf

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

# Sweep exponents: TestFloat-informed, chosen for exp-diff coverage when paired
SWEEP_EXPS = [
    0x000,  # subnormal/zero
    0x001,  # smallest normal
    0x002,
    0x003,  # near-min normal range
    0x3CA,  # 2^(-53): shift = significand width
    0x3CB,  # 2^(-52): ULP at 1.0
    0x3FC,  # 2^(-3)
    0x3FD,  # near 0.5
    0x3FE,  # 0.5..1.0
    0x3FF,  # 1.0..2.0
    0x400,  # 2.0..4.0
    0x401,
    0x402,  # 2^3 above unity
    0x432,  # near int53
    0x433,  # 2^52
    0x434,  # 2^53
    0x435,  # one past int53
    0x43C,  # near int64 boundary
    0x43E,  # INT64_MIN special exponent
    0x7FC,  # two below max
    0x7FD,  # near overflow
    0x7FE,  # largest finite
    0x7FF,  # inf/NaN
]

# TestFloat Level-1 core significands
SWEEP_SIGS = [0x0000000000000, 0x4000000000000, 0x8000000000000, 0xFFFFFFFFFFFFF]

# Precomputed sweep: 23 exps × 4 sigs × 2 signs = 184 values
SWEEP_VALUES = [
    (sign << 63) | (exp << 52) | sig
    for exp in SWEEP_EXPS
    for sig in SWEEP_SIGS
    for sign in (0, 1)
]

# Pruned P2 table covering all four TestFloat significand families
P2_SIGS = [
    # Single-bit-set (7 positions spanning LSB to MSB)
    0x0000000000001,
    0x0000000000004,
    0x0000000001000,
    0x0000001000000,
    0x0001000000000,
    0x4000000000000,
    0x8000000000000,
    # All-ones-with-gap (3 positions)
    0xFFFFFFFFFFFFE,
    0xFFFFBFFFFFFFF,
    0x7FFFFFFFFFFFF,
    # Contiguous-ones-from-LSB (3 lengths)
    0x000000000FFFF,
    0x00000FFFFFFFF,
    0xFFFFFFFFFFFFF,
    # Boundaries
    0x0000000000000,
    0xFFFFFFFFFFFFC,
    0x8000000000001,
]

# Richer sweep for mutation testing — P2 sigs have low-bit entropy that
# creates rounding ties after the << 9 alignment shift in add_mags_f64.
MUTATION_SWEEP_VALUES = [
    (sign << 63) | (exp << 52) | sig
    for exp in SWEEP_EXPS
    for sig in P2_SIGS
    for sign in (0, 1)
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


ENHANCED_SEED = 0xF64_E
ENHANCED_ROUNDS = 200_000


def weighted_f64_v2(rng: random.Random) -> int:
    """Enhanced generator using sweep exponents and sum-of-two-P2 significands."""
    r = rng.randint(0, 99)
    if r < 25:
        exp = rng.choice(SWEEP_EXPS)
        sig = (rng.choice(P2_SIGS) + rng.choice(P2_SIGS)) & 0xFFFFFFFFFFFFF
    elif r < 50:
        exp = rng.randint(0, 0x7FF)
        sig = (rng.choice(P2_SIGS) + rng.choice(P2_SIGS)) & 0xFFFFFFFFFFFFF
    elif r < 65:
        exp = rng.choice(SWEEP_EXPS)
        sig = rng.choice(P2_SIGS)
    elif r < 75:
        exp = rng.choice(SWEEP_EXPS)
        sig = rng.choice(SWEEP_SIGS)
    else:
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


# ---------------------------------------------------------------------------
# Structured sweep tests (TestFloat-style Cartesian product)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", BINARY_OPS)
def test_binary_sweep(op: str):
    fn = BINARY_OPS[op]
    fails = 0
    tested = 0
    first_failure = ""
    for a_bits in SWEEP_VALUES:
        for b_bits in SWEEP_VALUES:
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


def test_sqrt_sweep():
    fails = 0
    tested = 0
    first_failure = ""
    for a_bits in SWEEP_VALUES:
        a_f = i2f(a_bits)
        if a_f < 0 or math.isnan(a_f):
            got = f64_sqrt(a_bits)
            if a_bits == F64_SIGN:
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


@pytest.mark.parametrize("op", ["eq", "lt", "le"])
def test_comparison_sweep(op: str):
    fn = {"eq": f64_eq, "lt": f64_lt, "le": f64_le}[op]
    py = {
        "eq": lambda a, b: a == b,
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
    }[op]
    fails = 0
    first_failure = ""
    for a_bits in SWEEP_VALUES:
        for b_bits in SWEEP_VALUES:
            a_f = i2f(a_bits)
            b_f = i2f(b_bits)
            expected = py(a_f, b_f)
            got = fn(a_bits, b_bits)
            if not check_bool(got, expected):
                fails += 1
                if fails == 1:
                    first_failure = (
                        f"{op}({a_f}, {b_f}): got {got}, expected {expected}"
                    )
    total = len(SWEEP_VALUES) ** 2
    assert fails == 0, f"{fails}/{total} failures. First: {first_failure}"


@pytest.mark.parametrize("op", ["min", "max"])
def test_minmax_sweep(op: str):
    fn = {"min": f64_min, "max": f64_max}[op]
    fails = 0
    first_failure = ""
    for a_bits in SWEEP_VALUES:
        for b_bits in SWEEP_VALUES:
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
                    first_failure = (
                        f"{op}({a_f}, {b_f}): got {i2f(got)}, expected {py_ref}"
                    )
    total = len(SWEEP_VALUES) ** 2
    assert fails == 0, f"{fails}/{total} failures. First: {first_failure}"


def test_neg_sweep():
    for a in SWEEP_VALUES:
        got = f64_neg(a)
        assert got == (a ^ F64_SIGN)


def test_abs_sweep():
    for a in SWEEP_VALUES:
        got = f64_abs(a)
        assert got == (a & 0x7FFFFFFFFFFFFFFF)


def test_i64_to_f64_sweep():
    fails = 0
    first_failure = ""
    for a_bits in SWEEP_VALUES:
        a_f = i2f(a_bits)
        if math.isnan(a_f) or math.isinf(a_f):
            continue
        n = int(a_f)
        if not (-(2**63) <= n <= 2**63 - 1):
            continue
        got = i64_to_f64(n)
        expected = f2i(float(n))
        if got != expected:
            fails += 1
            if fails == 1:
                first_failure = (
                    f"i64_to_f64({n}): got {i2f(got)} ({got:#018x}), "
                    f"expected {float(n)} ({expected:#018x})"
                )
    assert fails == 0, f"i64_to_f64 sweep failures. First: {first_failure}"


def test_f64_to_i64_sweep():
    fails = 0
    first_failure = ""
    tested = 0
    for a_bits in SWEEP_VALUES:
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


@pytest.mark.parametrize(
    "op,ref",
    [
        ("floor", math.floor),
        ("ceil", math.ceil),
        ("round", ref_round_haz),
    ],
)
def test_floor_ceil_round_sweep(op: str, ref):
    fn = {"floor": f64_floor, "ceil": f64_ceil, "round": f64_round}[op]
    fails = 0
    tested = 0
    first_failure = ""
    for a_bits in SWEEP_VALUES:
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


def test_to_str_sweep():
    fails = 0
    first_failure = ""
    for a_bits in SWEEP_VALUES:
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
    assert fails == 0, f"{fails}/{len(SWEEP_VALUES)} failures. First: {first_failure}"


def test_round_trip_sweep():
    """f64_to_str -> str_to_f64 must recover the original bits."""
    fails = 0
    first_failure = ""
    for a_bits in SWEEP_VALUES:
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
    assert fails == 0, f"{fails}/{len(SWEEP_VALUES)} failures. First: {first_failure}"


# ---------------------------------------------------------------------------
# Enhanced random tests (weighted_f64_v2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", BINARY_OPS)
def test_binary_enhanced(op: str):
    fn = BINARY_OPS[op]
    rng = random.Random(ENHANCED_SEED)
    fails = 0
    tested = 0
    first_failure = ""
    for _ in range(ENHANCED_ROUNDS):
        a_bits = weighted_f64_v2(rng)
        b_bits = weighted_f64_v2(rng)
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


def test_sqrt_enhanced():
    rng = random.Random(ENHANCED_SEED)
    fails = 0
    tested = 0
    first_failure = ""
    for _ in range(ENHANCED_ROUNDS):
        a_bits = weighted_f64_v2(rng)
        a_f = i2f(a_bits)
        if a_f < 0 or math.isnan(a_f):
            got = f64_sqrt(a_bits)
            if a_bits == F64_SIGN:
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
# Conversion boundary tests
# ---------------------------------------------------------------------------


def test_i64_to_f64_boundaries():
    """Deterministic boundary tests at rounding thresholds."""
    values = []
    base = 2**53
    for delta in range(-4, 5):
        values.append(base + delta)
        values.append(-(base + delta))
    values.extend([2**62, 2**63 - 1, -(2**63), 0, 1, -1, 2, -2])
    fails = 0
    first_failure = ""
    for n in values:
        got = i64_to_f64(n)
        expected = f2i(float(n))
        if got != expected:
            fails += 1
            if fails == 1:
                first_failure = (
                    f"i64_to_f64({n}): got {i2f(got)} ({got:#018x}), "
                    f"expected {float(n)} ({expected:#018x})"
                )
    assert fails == 0, f"{fails}/{len(values)} failures. First: {first_failure}"


def test_f64_to_i64_boundaries():
    """Deterministic boundary tests at integer boundaries."""
    test_cases = []
    boundary_sigs = [
        0x0000000000000,
        0x0000000000001,
        0xFFFFFFFFFFFFF,
        0x8000000000000,
    ]
    for exp in range(0x432, 0x43F):
        for sig in boundary_sigs:
            for sign in (0, 1):
                bits = (sign << 63) | (exp << 52) | sig
                f = i2f(bits)
                if math.isnan(f) or math.isinf(f) or abs(f) >= 2.0**63:
                    continue
                test_cases.append((bits, int(f)))
    for bits in [
        0x0000000000000000,
        0x8000000000000000,
        0x0000000000000001,
        0x8000000000000001,
    ]:
        f = i2f(bits)
        if not (math.isnan(f) or math.isinf(f) or abs(f) >= 2.0**63):
            test_cases.append((bits, int(f)))
    fails = 0
    first_failure = ""
    for bits, expected in test_cases:
        got = f64_to_i64(bits)
        if got != expected:
            fails += 1
            if fails == 1:
                first_failure = (
                    f"f64_to_i64({i2f(bits)}): got {got}, expected {expected}"
                )
    assert fails == 0, f"{fails}/{len(test_cases)} failures. First: {first_failure}"


# ---------------------------------------------------------------------------
# Mutation testing — verify the harness catches known bugs
# ---------------------------------------------------------------------------


@contextmanager
def _patched(attr, replacement):
    """Temporarily replace a softfloat function for mutation testing."""
    original = getattr(_sf, attr)
    setattr(_sf, attr, replacement)
    try:
        yield
    finally:
        setattr(_sf, attr, original)


def _run_binary_against_ref(fn, op_name, rounds=ROUNDS, seed=SEED):
    """Run a binary op against Python reference, return failure count."""
    rng = random.Random(seed)
    fails = 0
    for _ in range(rounds):
        a = weighted_f64(rng)
        b = weighted_f64(rng)
        ref = ref_binary(op_name, i2f(a), i2f(b))
        if ref is None:
            continue
        if not check_bits(fn(a, b), f2i(ref)):
            fails += 1
    return fails


def _run_sqrt_against_ref(fn, rounds=ROUNDS, seed=SEED):
    """Run sqrt against Python math.sqrt, return failure count."""
    rng = random.Random(seed)
    fails = 0
    for _ in range(rounds):
        a = weighted_f64(rng)
        a_f = i2f(a)
        if a_f < 0 or math.isnan(a_f):
            continue
        if not check_bits(fn(a), f2i(math.sqrt(a_f))):
            fails += 1
    return fails


def _run_comparison_against_ref(fn, py_fn, rounds=ROUNDS, seed=SEED):
    """Run comparison against Python reference, return failure count."""
    rng = random.Random(seed)
    fails = 0
    for _ in range(rounds):
        a = weighted_f64(rng)
        b = weighted_f64(rng)
        if fn(a, b) != py_fn(i2f(a), i2f(b)):
            fails += 1
    return fails


def _run_binary_sweep(fn, op_name):
    """Run a binary op against Python reference over all mutation sweep pairs."""
    fails = 0
    for a in MUTATION_SWEEP_VALUES:
        for b in MUTATION_SWEEP_VALUES:
            ref = ref_binary(op_name, i2f(a), i2f(b))
            if ref is None:
                continue
            if not check_bits(fn(a, b), f2i(ref)):
                fails += 1
    return fails


# --- Mutated variants (each embeds a single known bug) ---


def _f64_add_sign_flip(a, b):
    """f64_add with result sign flipped for same-sign addition."""
    sign_a = _sf.sign_f64(a)
    sign_b = _sf.sign_f64(b)
    if sign_a == sign_b:
        return _sf.add_mags_f64(a, b, sign_a) ^ F64_SIGN
    return _sf.sub_mags_f64(a, b, sign_a)


def _f64_mul_exp_bias(a, b):
    """f64_mul with exponent bias 0x400 instead of 0x3FF."""
    sign_a = _sf.sign_f64(a)
    exp_a = _sf.exp_f64(a)
    sig_a = _sf.frac_f64(a)
    sign_b = _sf.sign_f64(b)
    exp_b = _sf.exp_f64(b)
    sig_b = _sf.frac_f64(b)
    sign_z = sign_a ^ sign_b
    if exp_a == 0x7FF:
        if sig_a != 0 or (exp_b == 0x7FF and sig_b != 0):
            return _sf.propagate_nan_f64(a, b)
        if (exp_b | sig_b) == 0:
            return _sf.DEFAULT_NAN
        return _sf.pack_f64(sign_z, 0x7FF, 0)
    if exp_b == 0x7FF:
        if sig_b != 0:
            return _sf.propagate_nan_f64(a, b)
        if (exp_a | sig_a) == 0:
            return _sf.DEFAULT_NAN
        return _sf.pack_f64(sign_z, 0x7FF, 0)
    if exp_a == 0:
        if sig_a == 0:
            return _sf.pack_f64(sign_z, 0, 0)
        norm = _sf.norm_subnormal_f64_sig(sig_a)
        exp_a = norm[0]
        sig_a = norm[1]
    if exp_b == 0:
        if sig_b == 0:
            return _sf.pack_f64(sign_z, 0, 0)
        norm = _sf.norm_subnormal_f64_sig(sig_b)
        exp_b = norm[0]
        sig_b = norm[1]
    exp_z = exp_a + exp_b - 0x400  # BUG: 0x400 instead of 0x3FF
    sig_a = (sig_a | 0x0010000000000000) << 10
    sig_b = (sig_b | 0x0010000000000000) << 11
    prod = _sf.mul64_to_128(sig_a, sig_b)
    sig_z = prod[0] | (1 if prod[1] != 0 else 0)
    if sig_z < 0x4000000000000000:
        exp_z = exp_z - 1
        sig_z = sig_z << 1
    return _sf.round_pack_to_f64(sign_z, exp_z, sig_z)


def _f64_div_no_remainder_sticky(a, b):
    """f64_div with remainder sticky bit dropped."""
    sign_a = _sf.sign_f64(a)
    exp_a = _sf.exp_f64(a)
    sig_a = _sf.frac_f64(a)
    sign_b = _sf.sign_f64(b)
    exp_b = _sf.exp_f64(b)
    sig_b = _sf.frac_f64(b)
    sign_z = sign_a ^ sign_b
    if exp_a == 0x7FF:
        if sig_a != 0:
            return _sf.propagate_nan_f64(a, b)
        if exp_b == 0x7FF:
            if sig_b != 0:
                return _sf.propagate_nan_f64(a, b)
            return _sf.DEFAULT_NAN
        return _sf.pack_f64(sign_z, 0x7FF, 0)
    if exp_b == 0x7FF:
        if sig_b != 0:
            return _sf.propagate_nan_f64(a, b)
        return _sf.pack_f64(sign_z, 0, 0)
    if exp_b == 0:
        if sig_b == 0:
            if (exp_a | sig_a) == 0:
                return _sf.DEFAULT_NAN
            return _sf.pack_f64(sign_z, 0x7FF, 0)
        norm = _sf.norm_subnormal_f64_sig(sig_b)
        exp_b = norm[0]
        sig_b = norm[1]
    if exp_a == 0:
        if sig_a == 0:
            return _sf.pack_f64(sign_z, 0, 0)
        norm = _sf.norm_subnormal_f64_sig(sig_a)
        exp_a = norm[0]
        sig_a = norm[1]
    exp_z = exp_a - exp_b + 0x3FE
    sig_a = sig_a | 0x0010000000000000
    sig_b = sig_b | 0x0010000000000000
    if sig_a < sig_b:
        exp_z = exp_z - 1
        dividend = sig_a << 63
    else:
        dividend = sig_a << 62
    q = dividend // sig_b
    sig_z = q  # BUG: dropped remainder sticky
    if sig_z < 0x4000000000000000:
        exp_z = exp_z - 1
        sig_z = sig_z << 1
    return _sf.round_pack_to_f64(sign_z, exp_z, sig_z)


def _f64_sqrt_no_remainder_sticky(a):
    """f64_sqrt with remainder sticky bit dropped."""
    sign_a = _sf.sign_f64(a)
    exp_a = _sf.exp_f64(a)
    sig_a = _sf.frac_f64(a)
    if exp_a == 0x7FF:
        if sig_a != 0:
            return a | 0x0008000000000000
        if sign_a == 0:
            return a
        return _sf.DEFAULT_NAN
    if sign_a != 0:
        if (exp_a | sig_a) == 0:
            return a
        return _sf.DEFAULT_NAN
    if exp_a == 0:
        if sig_a == 0:
            return a
        norm = _sf.norm_subnormal_f64_sig(sig_a)
        exp_a = norm[0]
        sig_a = norm[1]
    exp_z = ((exp_a - 0x3FF) >> 1) + 0x3FE
    sig_a = sig_a | 0x0010000000000000
    if (exp_a & 1) == 0:
        sig_a = sig_a << 1
    n = sig_a << 72
    q = _sf._isqrt_125(n)
    rem = n - q * q
    if rem < 0:
        q = q - 1
        rem = n - q * q
    sig_z = q  # BUG: dropped remainder sticky
    if sig_z < 0x4000000000000000:
        exp_z = exp_z - 1
        sig_z = sig_z << 1
    return _sf.round_pack_to_f64(0, exp_z, sig_z)


def _f64_lt_sign_ignore(a, b):
    """f64_lt that ignores signs, comparing magnitudes only."""
    if is_nan_f64(a) or is_nan_f64(b):
        return False
    ua = a & 0x7FFFFFFFFFFFFFFF
    ub = b & 0x7FFFFFFFFFFFFFFF
    if ua == 0 and ub == 0:
        return False
    return ua < ub  # BUG: ignores signs


class TestMutations:
    """Verify the test harness catches known bugs within ROUNDS iterations."""

    def test_no_sticky_bit(self):
        """shift_right_jam64 without sticky bit must be caught."""

        def broken(a, dist):
            if dist < 64:
                return a >> dist
            return 0

        with _patched("shift_right_jam64", broken):
            fails = _run_binary_against_ref(f64_add, "add")
        assert fails > 0, "Mutation 'no_sticky_bit' was not detected"

    def test_no_tie_to_even(self):
        """round_pack_to_f64 without tie-to-even must be caught."""

        def broken(sign, exp, sig):
            round_increment = 0x200
            round_bits = sig & 0x3FF
            if exp < 0 or exp >= 0x7FD:
                if exp < 0:
                    sig = _sf.shift_right_jam64(sig, 0 - exp)
                    exp = 0
                    round_bits = sig & 0x3FF
                elif exp > 0x7FD or (sig + round_increment) >= 0x8000000000000000:
                    return _sf.pack_f64(sign, 0x7FF, 0)
            sig = (sig + round_increment) >> 10
            # BUG: removed tie-to-even
            if sig == 0:
                exp = 0
            return _sf.pack_f64(sign, exp, sig)

        with _patched("round_pack_to_f64", broken):
            fails = _run_binary_against_ref(f64_add, "add")
        assert fails > 0, "Mutation 'no_tie_to_even' was not detected"

    def test_wrong_round_increment(self):
        """round_pack_to_f64 with wrong round increment must be caught."""

        def broken(sign, exp, sig):
            round_increment = 0x100  # BUG: 0x100 instead of 0x200
            round_bits = sig & 0x3FF
            if exp < 0 or exp >= 0x7FD:
                if exp < 0:
                    sig = _sf.shift_right_jam64(sig, 0 - exp)
                    exp = 0
                    round_bits = sig & 0x3FF
                elif exp > 0x7FD or (sig + round_increment) >= 0x8000000000000000:
                    return _sf.pack_f64(sign, 0x7FF, 0)
            sig = (sig + round_increment) >> 10
            if round_bits == 0x200:
                sig = sig - (sig & 1)
            if sig == 0:
                exp = 0
            return _sf.pack_f64(sign, exp, sig)

        with _patched("round_pack_to_f64", broken):
            fails = _run_binary_against_ref(f64_add, "add")
        assert fails > 0, "Mutation 'wrong_round_increment' was not detected"

    def test_sign_flip_add(self):
        """Flipped result sign for same-sign addition must be caught."""
        fails = _run_binary_against_ref(_f64_add_sign_flip, "add")
        assert fails > 0, "Mutation 'sign_flip_add' was not detected"

    def test_exp_bias_mul(self):
        """Off-by-one exponent bias in mul must be caught."""
        fails = _run_binary_against_ref(_f64_mul_exp_bias, "mul")
        assert fails > 0, "Mutation 'exp_bias_mul' was not detected"

    def test_div_no_remainder_sticky(self):
        """Dropped remainder sticky in div must be caught."""
        fails = _run_binary_against_ref(_f64_div_no_remainder_sticky, "div")
        assert fails > 0, "Mutation 'div_no_remainder_sticky' was not detected"

    def test_sqrt_no_remainder_sticky(self):
        """Dropped remainder sticky in sqrt must be caught."""
        fails = _run_sqrt_against_ref(_f64_sqrt_no_remainder_sticky)
        assert fails > 0, "Mutation 'sqrt_no_remainder_sticky' was not detected"

    def test_comparison_sign_ignore(self):
        """Sign-ignoring f64_lt must be caught."""
        fails = _run_comparison_against_ref(_f64_lt_sign_ignore, lambda a, b: a < b)
        assert fails > 0, "Mutation 'comparison_sign_ignore' was not detected"

    def test_no_sticky_bit_sweep(self):
        """Sweep proves sticky-bit mutation is structurally covered."""

        def broken(a, dist):
            if dist < 64:
                return a >> dist
            return 0

        with _patched("shift_right_jam64", broken):
            fails = _run_binary_sweep(f64_add, "add")
        assert fails > 0, "Mutation 'no_sticky_bit' not detected by sweep"

    def test_no_tie_to_even_sweep(self):
        """Sweep proves tie-to-even mutation is structurally covered."""

        def broken(sign, exp, sig):
            round_increment = 0x200
            round_bits = sig & 0x3FF
            if exp < 0 or exp >= 0x7FD:
                if exp < 0:
                    sig = _sf.shift_right_jam64(sig, 0 - exp)
                    exp = 0
                    round_bits = sig & 0x3FF
                elif exp > 0x7FD or (sig + round_increment) >= 0x8000000000000000:
                    return _sf.pack_f64(sign, 0x7FF, 0)
            sig = (sig + round_increment) >> 10
            if sig == 0:
                exp = 0
            return _sf.pack_f64(sign, exp, sig)

        with _patched("round_pack_to_f64", broken):
            fails = _run_binary_sweep(f64_add, "add")
        assert fails > 0, "Mutation 'no_tie_to_even' not detected by sweep"
