"""Bitwise operation tests.

Comprehensive edge cases for bitwise operations, avoiding width-dependent tests.
"""

import sys


def test_bitwise_and_basic() -> None:
    assert 0b1111 & 0b1010 == 0b1010
    assert 0b1100 & 0b0011 == 0b0000
    assert 0b1010 & 0b1010 == 0b1010
    assert 0xFF & 0x0F == 0x0F
    assert 0xFF & 0xFF == 0xFF
    assert 0xFF & 0x00 == 0x00


def test_bitwise_and_identity() -> None:
    # x & x == x
    assert 42 & 42 == 42
    assert 0 & 0 == 0
    assert 255 & 255 == 255
    # x & 0 == 0
    assert 42 & 0 == 0
    assert 255 & 0 == 0
    assert 0 & 0 == 0


def test_bitwise_and_commutative() -> None:
    assert (0b1100 & 0b1010) == (0b1010 & 0b1100)
    assert (255 & 15) == (15 & 255)
    assert (0 & 42) == (42 & 0)


def test_bitwise_or_basic() -> None:
    assert 0b1100 | 0b0011 == 0b1111
    assert 0b1010 | 0b0101 == 0b1111
    assert 0b1010 | 0b1010 == 0b1010
    assert 0x0F | 0xF0 == 0xFF
    assert 0x00 | 0xFF == 0xFF
    assert 0x00 | 0x00 == 0x00


def test_bitwise_or_identity() -> None:
    # x | x == x
    assert 42 | 42 == 42
    assert 0 | 0 == 0
    assert 255 | 255 == 255
    # x | 0 == x
    assert 42 | 0 == 42
    assert 255 | 0 == 255
    assert 0 | 0 == 0


def test_bitwise_or_commutative() -> None:
    assert (0b1100 | 0b0011) == (0b0011 | 0b1100)
    assert (255 | 15) == (15 | 255)
    assert (0 | 42) == (42 | 0)


def test_bitwise_xor_basic() -> None:
    assert 0b1100 ^ 0b1010 == 0b0110
    assert 0b1111 ^ 0b1111 == 0b0000
    assert 0b1111 ^ 0b0000 == 0b1111
    assert 0xFF ^ 0x00 == 0xFF
    assert 0xFF ^ 0xFF == 0x00
    assert 0x0F ^ 0xF0 == 0xFF


def test_bitwise_xor_identity() -> None:
    # x ^ x == 0
    assert 42 ^ 42 == 0
    assert 255 ^ 255 == 0
    assert 0 ^ 0 == 0
    # x ^ 0 == x
    assert 42 ^ 0 == 42
    assert 255 ^ 0 == 255
    assert 0 ^ 0 == 0


def test_bitwise_xor_commutative() -> None:
    assert (0b1100 ^ 0b1010) == (0b1010 ^ 0b1100)
    assert (255 ^ 15) == (15 ^ 255)
    assert (0 ^ 42) == (42 ^ 0)


def test_bitwise_xor_self_inverse() -> None:
    # (x ^ y) ^ y == x
    x: int = 42
    y: int = 17
    assert (x ^ y) ^ y == x
    assert (x ^ y) ^ x == y
    x = 255
    y = 128
    assert (x ^ y) ^ y == x


def test_bitwise_not_basic() -> None:
    # ~x == -(x+1) for any integer
    assert ~0 == -1
    assert ~1 == -2
    assert ~(-1) == 0
    assert ~(-2) == 1
    assert ~42 == -43
    assert ~(-43) == 42


def test_bitwise_not_double() -> None:
    # ~~x == x
    assert ~~0 == 0
    assert ~~1 == 1
    assert ~~42 == 42
    assert ~~(-1) == -1
    assert ~~(-42) == -42


def test_left_shift_basic() -> None:
    assert 1 << 0 == 1
    assert 1 << 1 == 2
    assert 1 << 2 == 4
    assert 1 << 3 == 8
    assert 1 << 4 == 16
    assert 1 << 8 == 256
    assert 1 << 10 == 1024


def test_left_shift_multiplies() -> None:
    # x << n == x * 2^n
    assert 5 << 1 == 10
    assert 5 << 2 == 20
    assert 5 << 3 == 40
    assert 3 << 4 == 48
    assert 7 << 3 == 56


def test_left_shift_zero() -> None:
    # 0 << n == 0
    assert 0 << 0 == 0
    assert 0 << 1 == 0
    assert 0 << 10 == 0
    # x << 0 == x
    assert 42 << 0 == 42
    assert 255 << 0 == 255


def test_right_shift_basic() -> None:
    assert 1 >> 0 == 1
    assert 1 >> 1 == 0
    assert 2 >> 1 == 1
    assert 4 >> 1 == 2
    assert 4 >> 2 == 1
    assert 256 >> 8 == 1
    assert 1024 >> 10 == 1


def test_right_shift_divides() -> None:
    # x >> n == x // 2^n (for positive x)
    assert 10 >> 1 == 5
    assert 20 >> 2 == 5
    assert 40 >> 3 == 5
    assert 48 >> 4 == 3
    assert 56 >> 3 == 7


def test_right_shift_rounds_down() -> None:
    # Right shift rounds toward negative infinity
    assert 7 >> 1 == 3
    assert 7 >> 2 == 1
    assert 7 >> 3 == 0
    assert 15 >> 1 == 7
    assert 15 >> 2 == 3
    assert 15 >> 3 == 1


def test_right_shift_zero() -> None:
    # 0 >> n == 0
    assert 0 >> 0 == 0
    assert 0 >> 1 == 0
    assert 0 >> 10 == 0
    # x >> 0 == x
    assert 42 >> 0 == 42
    assert 255 >> 0 == 255


def test_shift_negative_values() -> None:
    # Python uses arithmetic right shift (sign-extending)
    assert -1 >> 1 == -1
    assert -2 >> 1 == -1
    assert -4 >> 1 == -2
    assert -4 >> 2 == -1
    assert -8 >> 3 == -1
    # Left shift of negative
    assert -1 << 1 == -2
    assert -1 << 2 == -4
    assert -2 << 1 == -4


def test_demorgan_and() -> None:
    # ~(x & y) == ~x | ~y
    x: int = 0b1100
    y: int = 0b1010
    assert ~(x & y) == (~x | ~y)
    x = 255
    y = 15
    assert ~(x & y) == (~x | ~y)


def test_demorgan_or() -> None:
    # ~(x | y) == ~x & ~y
    x: int = 0b1100
    y: int = 0b1010
    assert ~(x | y) == (~x & ~y)
    x = 255
    y = 15
    assert ~(x | y) == (~x & ~y)


def test_bitwise_associative() -> None:
    a: int = 0b1100
    b: int = 0b1010
    c: int = 0b0110
    # AND is associative
    assert (a & b) & c == a & (b & c)
    # OR is associative
    assert (a | b) | c == a | (b | c)
    # XOR is associative
    assert (a ^ b) ^ c == a ^ (b ^ c)


def test_bitwise_distributive() -> None:
    a: int = 0b1100
    b: int = 0b1010
    c: int = 0b0110
    # AND distributes over OR
    assert a & (b | c) == (a & b) | (a & c)
    # OR distributes over AND
    assert a | (b & c) == (a | b) & (a | c)


def test_mixed_operations() -> None:
    # Combining multiple bitwise ops
    assert (0xFF & 0x0F) | 0xF0 == 0xFF
    assert (0xFF ^ 0x0F) & 0xF0 == 0xF0
    assert ~(0xFF & 0x0F) & 0xFF == 0xF0
    # Shifts with masks
    assert (1 << 4) | (1 << 0) == 17
    assert (0xFF >> 4) & 0x0F == 0x0F


def test_bit_extraction() -> None:
    # Extract specific bits using AND mask
    value: int = 0b11010110
    assert value & 0b00001111 == 0b00000110
    assert value & 0b11110000 == 0b11010000
    assert (value >> 4) & 0b00001111 == 0b00001101


def test_bit_setting() -> None:
    # Set specific bits using OR
    value: int = 0b00000000
    value = value | 0b00000001
    assert value == 0b00000001
    value = value | 0b00000100
    assert value == 0b00000101
    value = value | 0b00010000
    assert value == 0b00010101


def test_bit_clearing() -> None:
    # Clear specific bits using AND with inverted mask
    value: int = 0b11111111
    value = value & 0b11111110
    assert value == 0b11111110
    value = value & 0b11111011
    assert value == 0b11111010
    value = value & 0b11101111
    assert value == 0b11101010


def test_bit_toggling() -> None:
    # Toggle specific bits using XOR
    value: int = 0b10101010
    value = value ^ 0b00001111
    assert value == 0b10100101
    value = value ^ 0b00001111
    assert value == 0b10101010


def test_power_of_two_check() -> None:
    # x & (x - 1) == 0 iff x is a power of 2 (for x > 0)
    assert 1 & (1 - 1) == 0
    assert 2 & (2 - 1) == 0
    assert 4 & (4 - 1) == 0
    assert 8 & (8 - 1) == 0
    assert 16 & (16 - 1) == 0
    # Non-powers of two
    assert 3 & (3 - 1) != 0
    assert 5 & (5 - 1) != 0
    assert 6 & (6 - 1) != 0
    assert 7 & (7 - 1) != 0


def test_isolate_lowest_set_bit() -> None:
    # n & (-n) isolates the lowest set bit
    assert 0b1010 & (-0b1010) == 0b0010
    assert 0b1100 & (-0b1100) == 0b0100
    assert 0b1000 & (-0b1000) == 0b1000
    assert 12 & (-12) == 4
    assert 10 & (-10) == 2
    assert 8 & (-8) == 8
    assert 1 & (-1) == 1
    # Edge: zero has no set bits
    assert 0 & (-0) == 0


def test_clear_lowest_set_bit() -> None:
    # n & (n - 1) clears the lowest set bit
    assert 0b1010 & (0b1010 - 1) == 0b1000
    assert 0b1100 & (0b1100 - 1) == 0b1000
    assert 0b1110 & (0b1110 - 1) == 0b1100
    assert 12 & (12 - 1) == 8
    assert 10 & (10 - 1) == 8
    assert 7 & (7 - 1) == 6
    assert 1 & (1 - 1) == 0


def test_xor_swap() -> None:
    # XOR swap algorithm: swap two values without temp variable
    a: int = 42
    b: int = 17
    a = a ^ b
    b = a ^ b
    a = a ^ b
    assert a == 17
    assert b == 42
    # Works with negative numbers too
    a = -5
    b = 10
    a = a ^ b
    b = a ^ b
    a = a ^ b
    assert a == 10
    assert b == -5


def test_xor_with_minus_one() -> None:
    # x ^ -1 == ~x (XOR with all 1s flips all bits)
    assert 0 ^ -1 == ~0
    assert 1 ^ -1 == ~1
    assert 42 ^ -1 == ~42
    assert -1 ^ -1 == ~(-1)
    assert 255 ^ -1 == ~255
    assert -42 ^ -1 == ~(-42)


def test_negative_bitwise_and() -> None:
    # AND with negative numbers (two's complement behavior)
    assert -1 & 0xFF == 0xFF
    assert -1 & 0x0F == 0x0F
    assert -2 & 0xFF == 0xFE
    assert -256 & 0xFF == 0
    assert -5 & -3 == -7
    assert -8 & 7 == 0


def test_negative_bitwise_or() -> None:
    # OR with negative numbers
    assert -1 | 0 == -1
    assert -1 | 0xFF == -1
    assert -2 | 1 == -1
    assert -8 | 7 == -1
    assert -5 | -3 == -1
    assert -4 | 3 == -1


def test_negative_bitwise_xor() -> None:
    # XOR with negative numbers
    assert -1 ^ 0 == -1
    assert -1 ^ -1 == 0
    assert -1 ^ 0xFF == -256
    assert -2 ^ -1 == 1
    assert -5 ^ -3 == 6


def test_absorption_laws() -> None:
    # a & (a | b) == a
    a: int = 0b1100
    b: int = 0b1010
    assert a & (a | b) == a
    # a | (a & b) == a
    assert a | (a & b) == a
    # With different values
    a = 42
    b = 17
    assert a & (a | b) == a
    assert a | (a & b) == a


def test_complement_laws() -> None:
    # a & ~a == 0
    assert 42 & ~42 == 0
    assert 255 & ~255 == 0
    assert 0 & ~0 == 0
    # a | ~a == -1 (all bits set in two's complement)
    assert 42 | ~42 == -1
    assert 255 | ~255 == -1
    assert 0 | ~0 == -1


def test_shift_large_amounts() -> None:
    # Shifting by large amounts
    assert 1 << 20 == 1048576
    assert 1 << 30 == 1073741824
    # Right shift large amounts toward zero
    assert 0xFF >> 8 == 0
    assert 0xFF >> 10 == 0
    assert 1000000 >> 20 == 0


def test_shift_equals_multiply_divide() -> None:
    # Left shift equals multiply by power of 2
    x: int = 13
    assert x << 1 == x * 2
    assert x << 2 == x * 4
    assert x << 3 == x * 8
    assert x << 4 == x * 16
    # Right shift equals floor divide by power of 2
    x = 100
    assert x >> 1 == x // 2
    assert x >> 2 == x // 4
    assert x >> 3 == x // 8
    assert x >> 4 == x // 16


def test_check_bit_at_position() -> None:
    # Check if bit at position i is set: (n >> i) & 1
    n: int = 0b10110
    assert (n >> 0) & 1 == 0
    assert (n >> 1) & 1 == 1
    assert (n >> 2) & 1 == 1
    assert (n >> 3) & 1 == 0
    assert (n >> 4) & 1 == 1


def test_set_bit_at_position() -> None:
    # Set bit at position i: n | (1 << i)
    n: int = 0b10000
    assert n | (1 << 0) == 0b10001
    assert n | (1 << 1) == 0b10010
    assert n | (1 << 2) == 0b10100
    # Setting already-set bit is idempotent
    assert n | (1 << 4) == n


def test_clear_bit_at_position() -> None:
    # Clear bit at position i: n & ~(1 << i)
    n: int = 0b11111
    assert n & ~(1 << 0) == 0b11110
    assert n & ~(1 << 1) == 0b11101
    assert n & ~(1 << 2) == 0b11011
    assert n & ~(1 << 3) == 0b10111
    assert n & ~(1 << 4) == 0b01111


def test_toggle_bit_at_position() -> None:
    # Toggle bit at position i: n ^ (1 << i)
    n: int = 0b10101
    assert n ^ (1 << 0) == 0b10100
    assert n ^ (1 << 1) == 0b10111
    assert n ^ (1 << 2) == 0b10001
    # Toggle twice returns original
    assert (n ^ (1 << 0)) ^ (1 << 0) == n


def test_count_set_bits_kernighan() -> None:
    # Brian Kernighan's algorithm: count iterations of n &= n - 1
    # Each iteration clears lowest set bit
    n: int = 0b1011
    count: int = 0
    while n != 0:
        n = n & (n - 1)
        count = count + 1
    assert count == 3
    # Test another value
    n = 0b11111111
    count = 0
    while n != 0:
        n = n & (n - 1)
        count = count + 1
    assert count == 8
    # Power of 2 has exactly 1 bit
    n = 64
    count = 0
    while n != 0:
        n = n & (n - 1)
        count = count + 1
    assert count == 1


def test_sign_extension_right_shift() -> None:
    # Python right shift preserves sign (arithmetic shift)
    assert -1 >> 31 == -1
    assert -2 >> 1 == -1
    assert -3 >> 1 == -2
    assert -4 >> 1 == -2
    assert -5 >> 1 == -3
    assert -100 >> 3 == -13


def test_idempotent_operations() -> None:
    # AND and OR are idempotent: x & x == x, x | x == x
    assert 0 & 0 == 0
    assert 0 | 0 == 0
    assert 42 & 42 == 42
    assert 42 | 42 == 42
    assert -1 & -1 == -1
    assert -1 | -1 == -1
    assert -42 & -42 == -42
    assert -42 | -42 == -42


def test_xor_find_unique() -> None:
    # XOR can find unique element when others appear twice
    # a ^ a == 0, so pairs cancel out
    result: int = 5 ^ 3 ^ 7 ^ 3 ^ 5
    assert result == 7
    result = 1 ^ 2 ^ 1
    assert result == 2
    result = 100 ^ 200 ^ 300 ^ 200 ^ 100
    assert result == 300


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_bitwise_and_basic()
        passed += 1
        print("  PASS test_bitwise_and_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_and_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_and_basic: " + str(e))
    try:
        test_bitwise_and_identity()
        passed += 1
        print("  PASS test_bitwise_and_identity")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_and_identity: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_and_identity: " + str(e))
    try:
        test_bitwise_and_commutative()
        passed += 1
        print("  PASS test_bitwise_and_commutative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_and_commutative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_and_commutative: " + str(e))
    try:
        test_bitwise_or_basic()
        passed += 1
        print("  PASS test_bitwise_or_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_or_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_or_basic: " + str(e))
    try:
        test_bitwise_or_identity()
        passed += 1
        print("  PASS test_bitwise_or_identity")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_or_identity: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_or_identity: " + str(e))
    try:
        test_bitwise_or_commutative()
        passed += 1
        print("  PASS test_bitwise_or_commutative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_or_commutative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_or_commutative: " + str(e))
    try:
        test_bitwise_xor_basic()
        passed += 1
        print("  PASS test_bitwise_xor_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_xor_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_xor_basic: " + str(e))
    try:
        test_bitwise_xor_identity()
        passed += 1
        print("  PASS test_bitwise_xor_identity")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_xor_identity: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_xor_identity: " + str(e))
    try:
        test_bitwise_xor_commutative()
        passed += 1
        print("  PASS test_bitwise_xor_commutative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_xor_commutative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_xor_commutative: " + str(e))
    try:
        test_bitwise_xor_self_inverse()
        passed += 1
        print("  PASS test_bitwise_xor_self_inverse")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_xor_self_inverse: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_xor_self_inverse: " + str(e))
    try:
        test_bitwise_not_basic()
        passed += 1
        print("  PASS test_bitwise_not_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_not_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_not_basic: " + str(e))
    try:
        test_bitwise_not_double()
        passed += 1
        print("  PASS test_bitwise_not_double")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_not_double: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_not_double: " + str(e))
    try:
        test_left_shift_basic()
        passed += 1
        print("  PASS test_left_shift_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_left_shift_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_left_shift_basic: " + str(e))
    try:
        test_left_shift_multiplies()
        passed += 1
        print("  PASS test_left_shift_multiplies")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_left_shift_multiplies: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_left_shift_multiplies: " + str(e))
    try:
        test_left_shift_zero()
        passed += 1
        print("  PASS test_left_shift_zero")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_left_shift_zero: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_left_shift_zero: " + str(e))
    try:
        test_right_shift_basic()
        passed += 1
        print("  PASS test_right_shift_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_right_shift_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_right_shift_basic: " + str(e))
    try:
        test_right_shift_divides()
        passed += 1
        print("  PASS test_right_shift_divides")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_right_shift_divides: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_right_shift_divides: " + str(e))
    try:
        test_right_shift_rounds_down()
        passed += 1
        print("  PASS test_right_shift_rounds_down")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_right_shift_rounds_down: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_right_shift_rounds_down: " + str(e))
    try:
        test_right_shift_zero()
        passed += 1
        print("  PASS test_right_shift_zero")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_right_shift_zero: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_right_shift_zero: " + str(e))
    try:
        test_shift_negative_values()
        passed += 1
        print("  PASS test_shift_negative_values")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_shift_negative_values: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_shift_negative_values: " + str(e))
    try:
        test_demorgan_and()
        passed += 1
        print("  PASS test_demorgan_and")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_demorgan_and: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_demorgan_and: " + str(e))
    try:
        test_demorgan_or()
        passed += 1
        print("  PASS test_demorgan_or")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_demorgan_or: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_demorgan_or: " + str(e))
    try:
        test_bitwise_associative()
        passed += 1
        print("  PASS test_bitwise_associative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_associative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_associative: " + str(e))
    try:
        test_bitwise_distributive()
        passed += 1
        print("  PASS test_bitwise_distributive")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bitwise_distributive: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bitwise_distributive: " + str(e))
    try:
        test_mixed_operations()
        passed += 1
        print("  PASS test_mixed_operations")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_mixed_operations: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_mixed_operations: " + str(e))
    try:
        test_bit_extraction()
        passed += 1
        print("  PASS test_bit_extraction")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bit_extraction: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bit_extraction: " + str(e))
    try:
        test_bit_setting()
        passed += 1
        print("  PASS test_bit_setting")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bit_setting: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bit_setting: " + str(e))
    try:
        test_bit_clearing()
        passed += 1
        print("  PASS test_bit_clearing")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bit_clearing: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bit_clearing: " + str(e))
    try:
        test_bit_toggling()
        passed += 1
        print("  PASS test_bit_toggling")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bit_toggling: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bit_toggling: " + str(e))
    try:
        test_power_of_two_check()
        passed += 1
        print("  PASS test_power_of_two_check")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_power_of_two_check: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_power_of_two_check: " + str(e))
    try:
        test_isolate_lowest_set_bit()
        passed += 1
        print("  PASS test_isolate_lowest_set_bit")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_isolate_lowest_set_bit: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_isolate_lowest_set_bit: " + str(e))
    try:
        test_clear_lowest_set_bit()
        passed += 1
        print("  PASS test_clear_lowest_set_bit")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_clear_lowest_set_bit: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_clear_lowest_set_bit: " + str(e))
    try:
        test_xor_swap()
        passed += 1
        print("  PASS test_xor_swap")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_xor_swap: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_xor_swap: " + str(e))
    try:
        test_xor_with_minus_one()
        passed += 1
        print("  PASS test_xor_with_minus_one")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_xor_with_minus_one: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_xor_with_minus_one: " + str(e))
    try:
        test_negative_bitwise_and()
        passed += 1
        print("  PASS test_negative_bitwise_and")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_negative_bitwise_and: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_negative_bitwise_and: " + str(e))
    try:
        test_negative_bitwise_or()
        passed += 1
        print("  PASS test_negative_bitwise_or")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_negative_bitwise_or: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_negative_bitwise_or: " + str(e))
    try:
        test_negative_bitwise_xor()
        passed += 1
        print("  PASS test_negative_bitwise_xor")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_negative_bitwise_xor: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_negative_bitwise_xor: " + str(e))
    try:
        test_absorption_laws()
        passed += 1
        print("  PASS test_absorption_laws")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_absorption_laws: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_absorption_laws: " + str(e))
    try:
        test_complement_laws()
        passed += 1
        print("  PASS test_complement_laws")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_complement_laws: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_complement_laws: " + str(e))
    try:
        test_shift_large_amounts()
        passed += 1
        print("  PASS test_shift_large_amounts")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_shift_large_amounts: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_shift_large_amounts: " + str(e))
    try:
        test_shift_equals_multiply_divide()
        passed += 1
        print("  PASS test_shift_equals_multiply_divide")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_shift_equals_multiply_divide: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_shift_equals_multiply_divide: " + str(e))
    try:
        test_check_bit_at_position()
        passed += 1
        print("  PASS test_check_bit_at_position")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_check_bit_at_position: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_check_bit_at_position: " + str(e))
    try:
        test_set_bit_at_position()
        passed += 1
        print("  PASS test_set_bit_at_position")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_bit_at_position: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_bit_at_position: " + str(e))
    try:
        test_clear_bit_at_position()
        passed += 1
        print("  PASS test_clear_bit_at_position")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_clear_bit_at_position: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_clear_bit_at_position: " + str(e))
    try:
        test_toggle_bit_at_position()
        passed += 1
        print("  PASS test_toggle_bit_at_position")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_toggle_bit_at_position: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_toggle_bit_at_position: " + str(e))
    try:
        test_count_set_bits_kernighan()
        passed += 1
        print("  PASS test_count_set_bits_kernighan")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_count_set_bits_kernighan: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_count_set_bits_kernighan: " + str(e))
    try:
        test_sign_extension_right_shift()
        passed += 1
        print("  PASS test_sign_extension_right_shift")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_sign_extension_right_shift: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_sign_extension_right_shift: " + str(e))
    try:
        test_idempotent_operations()
        passed += 1
        print("  PASS test_idempotent_operations")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_idempotent_operations: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_idempotent_operations: " + str(e))
    try:
        test_xor_find_unique()
        passed += 1
        print("  PASS test_xor_find_unique")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_xor_find_unique: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_xor_find_unique: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
