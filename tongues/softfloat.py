"""Software float64 — IEEE 754 double precision using only integer operations."""

# ---------------------------------------------------------------------------
# Layer 1: Constants and bit manipulation
# ---------------------------------------------------------------------------

MASK64: int = 0xFFFFFFFFFFFFFFFF
F64_SIGN: int = 0x8000000000000000
F64_INF: int = 0x7FF0000000000000
F64_NEG_INF: int = F64_SIGN | F64_INF
DEFAULT_NAN: int = 0x7FF8000000000000
F64_ZERO: int = 0
F64_NEG_ZERO: int = F64_SIGN

ROUND_NEAR_EVEN: int = 0


def sign_f64(ui: int) -> int:
    return (ui >> 63) & 1


def exp_f64(ui: int) -> int:
    return (ui >> 52) & 0x7FF


def frac_f64(ui: int) -> int:
    return ui & 0x000FFFFFFFFFFFFF


def pack_f64(sign: int, exp: int, sig: int) -> int:
    """Pack using addition so carry from sig bit 52 increments the exponent."""
    return ((sign << 63) + (exp << 52) + sig) & MASK64


def is_nan_f64(ui: int) -> bool:
    return (ui & 0x7FFFFFFFFFFFFFFF) > F64_INF


def wrapping_add(a: int, b: int) -> int:
    return (a + b) & MASK64


def wrapping_sub(a: int, b: int) -> int:
    return (a - b) & MASK64


def wrapping_mul(a: int, b: int) -> int:
    return (a * b) & MASK64


def lsr64(a: int, n: int) -> int:
    """Logical shift right, unsigned 64-bit."""
    return (a & MASK64) >> n


# ---------------------------------------------------------------------------
# Layer 2: Core internal helpers
# ---------------------------------------------------------------------------


def shift_right_jam64(a: int, dist: int) -> int:
    """Shift right with sticky bit — any bits shifted out set the LSB."""
    if dist < 64:
        remainder: int = a & ((1 << dist) - 1)
        if remainder != 0:
            return (a >> dist) | 1
        return a >> dist
    if a != 0:
        return 1
    return 0


def count_leading_zeros64(a: int) -> int:
    """CLZ for 64-bit unsigned value via binary search."""
    a = a & MASK64
    if a == 0:
        return 64
    n: int = 0
    if (a & 0xFFFFFFFF00000000) == 0:
        n += 32
        a = a << 32
    if (a & 0xFFFF000000000000) == 0:
        n += 16
        a = a << 16
    if (a & 0xFF00000000000000) == 0:
        n += 8
        a = a << 8
    if (a & 0xF000000000000000) == 0:
        n += 4
        a = a << 4
    if (a & 0xC000000000000000) == 0:
        n += 2
        a = a << 2
    if (a & 0x8000000000000000) == 0:
        n += 1
    return n


def norm_subnormal_f64_sig(sig: int) -> tuple[int, int]:
    """Normalize a subnormal significand. Returns (exp, sig)."""
    shift_count: int = count_leading_zeros64(sig) - 11
    return (1 - shift_count, sig << shift_count)


def round_pack_to_f64(sign: int, exp: int, sig: int) -> int:
    """Round and pack into float64. sig has implicit 1 at bit 62."""
    round_increment: int = 0x200
    round_bits: int = sig & 0x3FF
    if exp < 0 or exp >= 0x7FD:
        if exp < 0:
            sig = shift_right_jam64(sig, 0 - exp)
            exp = 0
            round_bits = sig & 0x3FF
        elif exp > 0x7FD or (sig + round_increment) >= 0x8000000000000000:
            return pack_f64(sign, 0x7FF, 0)
    sig = (sig + round_increment) >> 10
    if round_bits == 0x200:
        sig = sig - (sig & 1)
    if sig == 0:
        exp = 0
    return pack_f64(sign, exp, sig)


def norm_round_pack_to_f64(sign: int, exp: int, sig: int) -> int:
    """Normalize, then round and pack."""
    shift_dist: int = count_leading_zeros64(sig) - 1
    exp = exp - shift_dist
    if shift_dist >= 10 and exp >= 0 and exp < 0x7FD:
        if sig == 0:
            return pack_f64(sign, 0, 0)
        return pack_f64(sign, exp, sig << (shift_dist - 10))
    return round_pack_to_f64(sign, exp, sig << shift_dist)


def mul64_to_128(a: int, b: int) -> tuple[int, int]:
    """Multiply two 64-bit unsigned values, return (high64, low64)."""
    a_lo: int = a & 0xFFFFFFFF
    a_hi: int = lsr64(a, 32)
    b_lo: int = b & 0xFFFFFFFF
    b_hi: int = lsr64(b, 32)
    p0: int = a_lo * b_lo
    p1: int = a_hi * b_lo
    p2: int = a_lo * b_hi
    p3: int = a_hi * b_hi
    mid: int = p1 + p2
    lo: int = p0 + ((mid & 0xFFFFFFFF) << 32)
    hi: int = p3 + (mid >> 32) + (lo >> 64)
    return (hi & MASK64, lo & MASK64)


def propagate_nan_f64(a: int, b: int) -> int:
    """NaN propagation — prefer first NaN, made quiet."""
    qa: int = a | 0x0008000000000000
    qb: int = b | 0x0008000000000000
    if is_nan_f64(a):
        if is_nan_f64(b):
            if qa < qb:
                return qb
            return qa
        return qa
    return qb


# ---------------------------------------------------------------------------
# Layer 3: Addition and subtraction
# ---------------------------------------------------------------------------


def add_mags_f64(ui_a: int, ui_b: int, sign_z: int) -> int:
    """Add magnitudes of two float64 values."""
    exp_a: int = exp_f64(ui_a)
    sig_a: int = frac_f64(ui_a)
    exp_b: int = exp_f64(ui_b)
    sig_b: int = frac_f64(ui_b)
    exp_diff: int = exp_a - exp_b
    if exp_diff == 0:
        if exp_a == 0:
            return pack_f64(sign_z, 0, sig_a + sig_b)
        if exp_a == 0x7FF:
            if (sig_a | sig_b) != 0:
                return propagate_nan_f64(ui_a, ui_b)
            return ui_a
        exp_z: int = exp_a
        sig_z: int = 0x0020000000000000 + sig_a + sig_b
        sig_z = sig_z << 9
    elif exp_diff > 0:
        sig_a = sig_a << 9
        sig_b = sig_b << 9
        if exp_a == 0x7FF:
            if sig_a != 0:
                return propagate_nan_f64(ui_a, ui_b)
            return ui_a
        if exp_b != 0:
            sig_b = sig_b + 0x2000000000000000
        else:
            exp_diff = exp_diff - 1
        sig_b = shift_right_jam64(sig_b, exp_diff)
        exp_z = exp_a
        sig_a = sig_a + 0x2000000000000000
        sig_z = sig_a + sig_b
    else:
        sig_a = sig_a << 9
        sig_b = sig_b << 9
        if exp_b == 0x7FF:
            if sig_b != 0:
                return propagate_nan_f64(ui_a, ui_b)
            return pack_f64(sign_z, 0x7FF, 0)
        if exp_a != 0:
            sig_a = sig_a + 0x2000000000000000
        else:
            exp_diff = exp_diff + 1
        sig_a = shift_right_jam64(sig_a, 0 - exp_diff)
        exp_z = exp_b
        sig_b = sig_b + 0x2000000000000000
        sig_z = sig_a + sig_b
    if sig_z < 0x4000000000000000:
        exp_z = exp_z - 1
        sig_z = sig_z << 1
    return round_pack_to_f64(sign_z, exp_z, sig_z)


def sub_mags_f64(ui_a: int, ui_b: int, sign_z: int) -> int:
    """Subtract magnitudes of two float64 values."""
    exp_a: int = exp_f64(ui_a)
    sig_a: int = frac_f64(ui_a)
    exp_b: int = exp_f64(ui_b)
    sig_b: int = frac_f64(ui_b)
    exp_diff: int = exp_a - exp_b
    if exp_diff == 0:
        if exp_a == 0x7FF:
            if (sig_a | sig_b) != 0:
                return propagate_nan_f64(ui_a, ui_b)
            return DEFAULT_NAN
        sig_diff: int = sig_a - sig_b
        if sig_diff == 0:
            return F64_ZERO
        if exp_a != 0:
            exp_a = exp_a - 1
        if sig_diff < 0:
            sign_z = 1 - sign_z
            sig_diff = 0 - sig_diff
        shift_dist: int = count_leading_zeros64(sig_diff) - 11
        exp_z: int = exp_a - shift_dist
        if exp_z >= 0:
            return pack_f64(sign_z, exp_z, sig_diff << shift_dist)
        if exp_a == 0:
            return pack_f64(sign_z, 0, sig_diff)
        return pack_f64(sign_z, 0, sig_diff << exp_a)
    sig_a = sig_a << 10
    sig_b = sig_b << 10
    sig_z: int = 0
    if exp_diff > 0:
        if exp_a == 0x7FF:
            if sig_a != 0:
                return propagate_nan_f64(ui_a, ui_b)
            return ui_a
        if exp_b != 0:
            sig_b = sig_b | 0x4000000000000000
        else:
            exp_diff = exp_diff - 1
        sig_b = shift_right_jam64(sig_b, exp_diff)
        sig_a = sig_a | 0x4000000000000000
        exp_z = exp_a
        sig_z = sig_a - sig_b
    else:
        if exp_b == 0x7FF:
            if sig_b != 0:
                return propagate_nan_f64(ui_a, ui_b)
            return pack_f64(1 - sign_z, 0x7FF, 0)
        if exp_a != 0:
            sig_a = sig_a | 0x4000000000000000
        else:
            exp_diff = exp_diff + 1
        sig_a = shift_right_jam64(sig_a, 0 - exp_diff)
        sig_b = sig_b | 0x4000000000000000
        exp_z = exp_b
        sig_z = sig_b - sig_a
        sign_z = 1 - sign_z
    return norm_round_pack_to_f64(sign_z, exp_z - 1, sig_z)


def f64_add(a: int, b: int) -> int:
    sign_a: int = sign_f64(a)
    sign_b: int = sign_f64(b)
    if sign_a == sign_b:
        return add_mags_f64(a, b, sign_a)
    return sub_mags_f64(a, b, sign_a)


def f64_sub(a: int, b: int) -> int:
    sign_a: int = sign_f64(a)
    sign_b: int = sign_f64(b)
    if sign_a == sign_b:
        return sub_mags_f64(a, b, sign_a)
    return add_mags_f64(a, b, sign_a)


# ---------------------------------------------------------------------------
# Layer 4: Multiplication
# ---------------------------------------------------------------------------


def f64_mul(a: int, b: int) -> int:
    sign_a: int = sign_f64(a)
    exp_a: int = exp_f64(a)
    sig_a: int = frac_f64(a)
    sign_b: int = sign_f64(b)
    exp_b: int = exp_f64(b)
    sig_b: int = frac_f64(b)
    sign_z: int = sign_a ^ sign_b
    if exp_a == 0x7FF:
        if sig_a != 0 or (exp_b == 0x7FF and sig_b != 0):
            return propagate_nan_f64(a, b)
        if (exp_b | sig_b) == 0:
            return DEFAULT_NAN
        return pack_f64(sign_z, 0x7FF, 0)
    if exp_b == 0x7FF:
        if sig_b != 0:
            return propagate_nan_f64(a, b)
        if (exp_a | sig_a) == 0:
            return DEFAULT_NAN
        return pack_f64(sign_z, 0x7FF, 0)
    if exp_a == 0:
        if sig_a == 0:
            return pack_f64(sign_z, 0, 0)
        norm: tuple[int, int] = norm_subnormal_f64_sig(sig_a)
        exp_a = norm[0]
        sig_a = norm[1]
    if exp_b == 0:
        if sig_b == 0:
            return pack_f64(sign_z, 0, 0)
        norm = norm_subnormal_f64_sig(sig_b)
        exp_b = norm[0]
        sig_b = norm[1]
    exp_z: int = exp_a + exp_b - 0x3FF
    sig_a = (sig_a | 0x0010000000000000) << 10
    sig_b = (sig_b | 0x0010000000000000) << 11
    prod: tuple[int, int] = mul64_to_128(sig_a, sig_b)
    sig_z: int = prod[0] | (1 if prod[1] != 0 else 0)
    if sig_z < 0x4000000000000000:
        exp_z = exp_z - 1
        sig_z = sig_z << 1
    return round_pack_to_f64(sign_z, exp_z, sig_z)


# ---------------------------------------------------------------------------
# Layer 5: Division
# ---------------------------------------------------------------------------


def f64_div(a: int, b: int) -> int:
    sign_a: int = sign_f64(a)
    exp_a: int = exp_f64(a)
    sig_a: int = frac_f64(a)
    sign_b: int = sign_f64(b)
    exp_b: int = exp_f64(b)
    sig_b: int = frac_f64(b)
    sign_z: int = sign_a ^ sign_b
    if exp_a == 0x7FF:
        if sig_a != 0:
            return propagate_nan_f64(a, b)
        if exp_b == 0x7FF:
            if sig_b != 0:
                return propagate_nan_f64(a, b)
            return DEFAULT_NAN
        return pack_f64(sign_z, 0x7FF, 0)
    if exp_b == 0x7FF:
        if sig_b != 0:
            return propagate_nan_f64(a, b)
        return pack_f64(sign_z, 0, 0)
    if exp_b == 0:
        if sig_b == 0:
            if (exp_a | sig_a) == 0:
                return DEFAULT_NAN
            return pack_f64(sign_z, 0x7FF, 0)
        norm: tuple[int, int] = norm_subnormal_f64_sig(sig_b)
        exp_b = norm[0]
        sig_b = norm[1]
    if exp_a == 0:
        if sig_a == 0:
            return pack_f64(sign_z, 0, 0)
        norm = norm_subnormal_f64_sig(sig_a)
        exp_a = norm[0]
        sig_a = norm[1]
    exp_z: int = exp_a - exp_b + 0x3FE
    sig_a = sig_a | 0x0010000000000000
    sig_b = sig_b | 0x0010000000000000
    if sig_a < sig_b:
        exp_z = exp_z - 1
        dividend: int = sig_a << 63
    else:
        dividend = sig_a << 62
    q: int = dividend // sig_b
    r: int = dividend - q * sig_b
    sig_z: int = q | (1 if r != 0 else 0)
    if sig_z < 0x4000000000000000:
        exp_z = exp_z - 1
        sig_z = sig_z << 1
    return round_pack_to_f64(sign_z, exp_z, sig_z)


# ---------------------------------------------------------------------------
# Layer 6: Square root
# ---------------------------------------------------------------------------


def _isqrt_125(n: int) -> int:
    """Integer square root of a 125–126 bit value via Newton's method."""
    x: int = 1 << 63
    while True:
        x1: int = (x + n // x) >> 1
        if x1 >= x:
            return x
        x = x1


def f64_sqrt(a: int) -> int:
    sign_a: int = sign_f64(a)
    exp_a: int = exp_f64(a)
    sig_a: int = frac_f64(a)
    if exp_a == 0x7FF:
        if sig_a != 0:
            return a | 0x0008000000000000
        if sign_a == 0:
            return a
        return DEFAULT_NAN
    if sign_a != 0:
        if (exp_a | sig_a) == 0:
            return a
        return DEFAULT_NAN
    if exp_a == 0:
        if sig_a == 0:
            return a
        norm: tuple[int, int] = norm_subnormal_f64_sig(sig_a)
        exp_a = norm[0]
        sig_a = norm[1]
    exp_z: int = ((exp_a - 0x3FF) >> 1) + 0x3FE
    sig_a = sig_a | 0x0010000000000000
    if (exp_a & 1) == 0:
        sig_a = sig_a << 1
    n: int = sig_a << 72
    q: int = _isqrt_125(n)
    rem: int = n - q * q
    if rem < 0:
        q = q - 1
        rem = n - q * q
    sig_z: int = q | (1 if rem != 0 else 0)
    if sig_z < 0x4000000000000000:
        exp_z = exp_z - 1
        sig_z = sig_z << 1
    return round_pack_to_f64(0, exp_z, sig_z)


# ---------------------------------------------------------------------------
# Layer 7: Remainder
# ---------------------------------------------------------------------------


def f64_fmod(a: int, b: int) -> int:
    """Truncating remainder: a - trunc(a/b) * b. Sign follows dividend."""
    sign_a: int = sign_f64(a)
    exp_a: int = exp_f64(a)
    sig_a: int = frac_f64(a)
    exp_b: int = exp_f64(b)
    sig_b: int = frac_f64(b)
    if exp_a == 0x7FF:
        if sig_a != 0 or (exp_b == 0x7FF and sig_b != 0):
            return propagate_nan_f64(a, b)
        return DEFAULT_NAN
    if exp_b == 0x7FF:
        if sig_b != 0:
            return propagate_nan_f64(a, b)
        return a
    if exp_b == 0:
        if sig_b == 0:
            return DEFAULT_NAN
        norm: tuple[int, int] = norm_subnormal_f64_sig(sig_b)
        exp_b = norm[0]
        sig_b = norm[1]
    if exp_a == 0:
        if sig_a == 0:
            return a
        norm = norm_subnormal_f64_sig(sig_a)
        exp_a = norm[0]
        sig_a = norm[1]
    sig_a = sig_a | 0x0010000000000000
    sig_b = sig_b | 0x0010000000000000
    exp_diff: int = exp_a - exp_b
    if exp_diff < 0:
        return a
    r: int = (sig_a << exp_diff) % sig_b
    if r == 0:
        return pack_f64(sign_a, 0, 0)
    return norm_round_pack_to_f64(sign_a, exp_b - 1, r << 10)


# ---------------------------------------------------------------------------
# Layer 8: Sign operations, comparisons, conversions
# ---------------------------------------------------------------------------


def f64_neg(a: int) -> int:
    return a ^ F64_SIGN


def f64_abs(a: int) -> int:
    return a & 0x7FFFFFFFFFFFFFFF


def is_inf_f64(ui: int) -> bool:
    return (ui & 0x7FFFFFFFFFFFFFFF) == F64_INF


def f64_eq(a: int, b: int) -> bool:
    if is_nan_f64(a) or is_nan_f64(b):
        return False
    ua: int = a & 0x7FFFFFFFFFFFFFFF
    ub: int = b & 0x7FFFFFFFFFFFFFFF
    if ua == 0 and ub == 0:
        return True
    return a == b


def f64_lt(a: int, b: int) -> bool:
    if is_nan_f64(a) or is_nan_f64(b):
        return False
    sign_a: int = sign_f64(a)
    sign_b: int = sign_f64(b)
    ua: int = a & 0x7FFFFFFFFFFFFFFF
    ub: int = b & 0x7FFFFFFFFFFFFFFF
    if ua == 0 and ub == 0:
        return False
    if sign_a != sign_b:
        return sign_a == 1
    if sign_a == 0:
        return ua < ub
    return ua > ub


def f64_le(a: int, b: int) -> bool:
    if is_nan_f64(a) or is_nan_f64(b):
        return False
    sign_a: int = sign_f64(a)
    sign_b: int = sign_f64(b)
    ua: int = a & 0x7FFFFFFFFFFFFFFF
    ub: int = b & 0x7FFFFFFFFFFFFFFF
    if ua == 0 and ub == 0:
        return True
    if sign_a != sign_b:
        return sign_a == 1
    if sign_a == 0:
        return ua <= ub
    return ua >= ub


def f64_min(a: int, b: int) -> int:
    """Min with NaN propagation for strict mode."""
    if is_nan_f64(a):
        return a | 0x0008000000000000
    if is_nan_f64(b):
        return b | 0x0008000000000000
    if f64_lt(a, b):
        return a
    return b


def f64_max(a: int, b: int) -> int:
    """Max with NaN propagation for strict mode."""
    if is_nan_f64(a):
        return a | 0x0008000000000000
    if is_nan_f64(b):
        return b | 0x0008000000000000
    if f64_lt(b, a):
        return a
    return b


def i64_to_f64(n: int) -> int:
    """Convert a signed 64-bit integer to float64."""
    if n == 0:
        return F64_ZERO
    sign_z: int = 0
    if n < 0:
        sign_z = 1
        n = 0 - n
    if n >= 0x8000000000000000:
        return pack_f64(sign_z, 0x43E, 0)
    return norm_round_pack_to_f64(sign_z, 0x43C, n)


def f64_to_i64(a: int) -> int:
    """Convert float64 to signed 64-bit integer, truncating toward zero.

    Returns INT64_MAX/MIN for NaN/inf/overflow — caller should check first.
    """
    sign_a: int = sign_f64(a)
    exp_a: int = exp_f64(a)
    sig_a: int = frac_f64(a)
    if exp_a == 0x7FF:
        if sign_a == 0:
            return 0x7FFFFFFFFFFFFFFF
        return -(1 << 63)
    if exp_a == 0:
        return 0
    sig_a = sig_a | 0x0010000000000000
    shift: int = exp_a - 0x433
    if shift >= 0:
        if shift > 10:
            if shift == 11 and sig_a == 0x0010000000000000 and sign_a != 0:
                return -(1 << 63)
            if sign_a == 0:
                return 0x7FFFFFFFFFFFFFFF
            return -(1 << 63)
        result: int = sig_a << shift
    else:
        if (0 - shift) >= 53:
            return 0
        result = sig_a >> (0 - shift)
    if sign_a != 0:
        result = 0 - result
    return result
