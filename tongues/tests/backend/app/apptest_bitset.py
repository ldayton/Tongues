"""Bitset tests — bit operations, set algebra, popcount, and edge cases."""

import sys

from lib.bitset import bitset_clear
from lib.bitset import bitset_difference
from lib.bitset import bitset_intersection
from lib.bitset import bitset_new
from lib.bitset import bitset_popcount
from lib.bitset import bitset_set
from lib.bitset import bitset_test
from lib.bitset import bitset_to_list
from lib.bitset import bitset_toggle
from lib.bitset import bitset_union


# -- Creation --


def test_new_all_zero() -> None:
    bs: list[int] = bitset_new(64)
    assert bitset_popcount(bs) == 0


def test_new_size() -> None:
    bs: list[int] = bitset_new(100)
    assert bs[0] == 100


# -- Set / Test --


def test_set_and_test() -> None:
    bs: list[int] = bitset_new(64)
    bitset_set(bs, 0)
    assert bitset_test(bs, 0)
    assert not bitset_test(bs, 1)


def test_set_high_bit() -> None:
    bs: list[int] = bitset_new(64)
    bitset_set(bs, 63)
    assert bitset_test(bs, 63)
    assert not bitset_test(bs, 62)


def test_set_multiple() -> None:
    bs: list[int] = bitset_new(64)
    bitset_set(bs, 0)
    bitset_set(bs, 31)
    bitset_set(bs, 32)
    bitset_set(bs, 63)
    assert bitset_test(bs, 0)
    assert bitset_test(bs, 31)
    assert bitset_test(bs, 32)
    assert bitset_test(bs, 63)
    assert not bitset_test(bs, 1)
    assert not bitset_test(bs, 33)


def test_set_word_boundary() -> None:
    """Bits 31 and 32 are in different words."""
    bs: list[int] = bitset_new(64)
    bitset_set(bs, 31)
    bitset_set(bs, 32)
    assert bitset_test(bs, 31)
    assert bitset_test(bs, 32)
    assert not bitset_test(bs, 30)
    assert not bitset_test(bs, 33)


# -- Clear --


def test_clear() -> None:
    bs: list[int] = bitset_new(64)
    bitset_set(bs, 5)
    assert bitset_test(bs, 5)
    bitset_clear(bs, 5)
    assert not bitset_test(bs, 5)


def test_clear_already_clear() -> None:
    bs: list[int] = bitset_new(64)
    bitset_clear(bs, 5)
    assert not bitset_test(bs, 5)


def test_clear_preserves_others() -> None:
    bs: list[int] = bitset_new(64)
    bitset_set(bs, 4)
    bitset_set(bs, 5)
    bitset_set(bs, 6)
    bitset_clear(bs, 5)
    assert bitset_test(bs, 4)
    assert not bitset_test(bs, 5)
    assert bitset_test(bs, 6)


# -- Toggle --


def test_toggle_on() -> None:
    bs: list[int] = bitset_new(64)
    bitset_toggle(bs, 10)
    assert bitset_test(bs, 10)


def test_toggle_off() -> None:
    bs: list[int] = bitset_new(64)
    bitset_set(bs, 10)
    bitset_toggle(bs, 10)
    assert not bitset_test(bs, 10)


def test_toggle_double() -> None:
    bs: list[int] = bitset_new(64)
    bitset_toggle(bs, 10)
    bitset_toggle(bs, 10)
    assert not bitset_test(bs, 10)


# -- Popcount --


def test_popcount_empty() -> None:
    assert bitset_popcount(bitset_new(64)) == 0


def test_popcount_one() -> None:
    bs: list[int] = bitset_new(64)
    bitset_set(bs, 42)
    assert bitset_popcount(bs) == 1


def test_popcount_several() -> None:
    bs: list[int] = bitset_new(128)
    bitset_set(bs, 0)
    bitset_set(bs, 31)
    bitset_set(bs, 32)
    bitset_set(bs, 63)
    bitset_set(bs, 64)
    bitset_set(bs, 127)
    assert bitset_popcount(bs) == 6


def test_popcount_full_word() -> None:
    bs: list[int] = bitset_new(32)
    i: int = 0
    while i < 32:
        bitset_set(bs, i)
        i += 1
    assert bitset_popcount(bs) == 32


# -- Union --


def test_union_basic() -> None:
    a: list[int] = bitset_new(64)
    b: list[int] = bitset_new(64)
    bitset_set(a, 0)
    bitset_set(a, 2)
    bitset_set(b, 1)
    bitset_set(b, 2)
    result: list[int] = bitset_union(a, b)
    assert bitset_test(result, 0)
    assert bitset_test(result, 1)
    assert bitset_test(result, 2)
    assert not bitset_test(result, 3)


def test_union_different_sizes() -> None:
    a: list[int] = bitset_new(32)
    b: list[int] = bitset_new(96)
    bitset_set(a, 0)
    bitset_set(b, 64)
    result: list[int] = bitset_union(a, b)
    assert result[0] == 96
    assert bitset_test(result, 0)
    assert bitset_test(result, 64)


# -- Intersection --


def test_intersection_basic() -> None:
    a: list[int] = bitset_new(64)
    b: list[int] = bitset_new(64)
    bitset_set(a, 0)
    bitset_set(a, 1)
    bitset_set(b, 1)
    bitset_set(b, 2)
    result: list[int] = bitset_intersection(a, b)
    assert not bitset_test(result, 0)
    assert bitset_test(result, 1)
    assert not bitset_test(result, 2)


def test_intersection_disjoint() -> None:
    a: list[int] = bitset_new(64)
    b: list[int] = bitset_new(64)
    bitset_set(a, 0)
    bitset_set(b, 1)
    result: list[int] = bitset_intersection(a, b)
    assert bitset_popcount(result) == 0


def test_intersection_different_sizes() -> None:
    a: list[int] = bitset_new(96)
    b: list[int] = bitset_new(32)
    bitset_set(a, 0)
    bitset_set(a, 64)
    bitset_set(b, 0)
    result: list[int] = bitset_intersection(a, b)
    assert result[0] == 32
    assert bitset_test(result, 0)


# -- Difference --


def test_difference_basic() -> None:
    a: list[int] = bitset_new(64)
    b: list[int] = bitset_new(64)
    bitset_set(a, 0)
    bitset_set(a, 1)
    bitset_set(a, 2)
    bitset_set(b, 1)
    result: list[int] = bitset_difference(a, b)
    assert bitset_test(result, 0)
    assert not bitset_test(result, 1)
    assert bitset_test(result, 2)


def test_difference_no_overlap() -> None:
    a: list[int] = bitset_new(64)
    b: list[int] = bitset_new(64)
    bitset_set(a, 0)
    bitset_set(b, 1)
    result: list[int] = bitset_difference(a, b)
    assert bitset_test(result, 0)
    assert bitset_popcount(result) == 1


# -- to_list --


def test_to_list_empty() -> None:
    assert bitset_to_list(bitset_new(64)) == []


def test_to_list_several() -> None:
    bs: list[int] = bitset_new(64)
    bitset_set(bs, 3)
    bitset_set(bs, 7)
    bitset_set(bs, 42)
    assert bitset_to_list(bs) == [3, 7, 42]


def test_to_list_word_boundary() -> None:
    bs: list[int] = bitset_new(96)
    bitset_set(bs, 31)
    bitset_set(bs, 32)
    bitset_set(bs, 63)
    bitset_set(bs, 64)
    assert bitset_to_list(bs) == [31, 32, 63, 64]


# -- Larger bitsets --


def test_large_set_all() -> None:
    bs: list[int] = bitset_new(256)
    i: int = 0
    while i < 256:
        bitset_set(bs, i)
        i += 1
    assert bitset_popcount(bs) == 256


def test_large_even_bits() -> None:
    bs: list[int] = bitset_new(128)
    i: int = 0
    while i < 128:
        bitset_set(bs, i)
        i += 2
    assert bitset_popcount(bs) == 64
    assert bitset_test(bs, 0)
    assert not bitset_test(bs, 1)
    assert bitset_test(bs, 2)
    assert not bitset_test(bs, 3)


# -- Edge cases --


def test_single_bit_bitset() -> None:
    bs: list[int] = bitset_new(1)
    assert not bitset_test(bs, 0)
    bitset_set(bs, 0)
    assert bitset_test(bs, 0)
    assert bitset_popcount(bs) == 1


def test_set_idempotent() -> None:
    bs: list[int] = bitset_new(64)
    bitset_set(bs, 5)
    bitset_set(bs, 5)
    assert bitset_popcount(bs) == 1


def test_union_with_empty() -> None:
    a: list[int] = bitset_new(64)
    b: list[int] = bitset_new(64)
    bitset_set(a, 10)
    result: list[int] = bitset_union(a, b)
    assert bitset_popcount(result) == 1
    assert bitset_test(result, 10)


def test_intersection_with_empty() -> None:
    a: list[int] = bitset_new(64)
    b: list[int] = bitset_new(64)
    bitset_set(a, 10)
    result: list[int] = bitset_intersection(a, b)
    assert bitset_popcount(result) == 0


def test_non_aligned_size() -> None:
    """Size 33 requires 2 words but only 1 bit in the second."""
    bs: list[int] = bitset_new(33)
    bitset_set(bs, 32)
    assert bitset_test(bs, 32)
    assert bitset_popcount(bs) == 1


def test_non_aligned_size_50() -> None:
    bs: list[int] = bitset_new(50)
    bitset_set(bs, 0)
    bitset_set(bs, 31)
    bitset_set(bs, 32)
    bitset_set(bs, 49)
    assert bitset_to_list(bs) == [0, 31, 32, 49]
    assert bitset_popcount(bs) == 4


def test_popcount_after_clear() -> None:
    bs: list[int] = bitset_new(64)
    bitset_set(bs, 0)
    bitset_set(bs, 10)
    bitset_set(bs, 20)
    assert bitset_popcount(bs) == 3
    bitset_clear(bs, 10)
    assert bitset_popcount(bs) == 2


def test_toggle_word_boundary() -> None:
    bs: list[int] = bitset_new(96)
    bitset_toggle(bs, 31)
    bitset_toggle(bs, 32)
    assert bitset_test(bs, 31)
    assert bitset_test(bs, 32)
    bitset_toggle(bs, 31)
    assert not bitset_test(bs, 31)
    assert bitset_test(bs, 32)


def test_union_via_to_list() -> None:
    a: list[int] = bitset_new(64)
    b: list[int] = bitset_new(64)
    bitset_set(a, 1)
    bitset_set(a, 3)
    bitset_set(b, 2)
    bitset_set(b, 3)
    assert bitset_to_list(bitset_union(a, b)) == [1, 2, 3]


def test_intersection_via_to_list() -> None:
    a: list[int] = bitset_new(64)
    b: list[int] = bitset_new(64)
    bitset_set(a, 1)
    bitset_set(a, 3)
    bitset_set(a, 5)
    bitset_set(b, 3)
    bitset_set(b, 5)
    bitset_set(b, 7)
    assert bitset_to_list(bitset_intersection(a, b)) == [3, 5]


def test_difference_via_to_list() -> None:
    a: list[int] = bitset_new(64)
    b: list[int] = bitset_new(64)
    bitset_set(a, 1)
    bitset_set(a, 3)
    bitset_set(a, 5)
    bitset_set(b, 3)
    bitset_set(b, 7)
    assert bitset_to_list(bitset_difference(a, b)) == [1, 5]


def test_difference_b_extra_bits_ignored() -> None:
    """Bits in b but not in a should not appear in result."""
    a: list[int] = bitset_new(64)
    b: list[int] = bitset_new(64)
    bitset_set(a, 0)
    bitset_set(b, 0)
    bitset_set(b, 1)
    bitset_set(b, 2)
    result: list[int] = bitset_difference(a, b)
    assert bitset_to_list(result) == []


def test_to_list_three_words() -> None:
    bs: list[int] = bitset_new(96)
    bitset_set(bs, 0)
    bitset_set(bs, 33)
    bitset_set(bs, 65)
    bitset_set(bs, 95)
    assert bitset_to_list(bs) == [0, 33, 65, 95]


def test_set_ops_single_bit() -> None:
    a: list[int] = bitset_new(1)
    b: list[int] = bitset_new(1)
    bitset_set(a, 0)
    assert bitset_to_list(bitset_union(a, b)) == [0]
    assert bitset_to_list(bitset_intersection(a, b)) == []
    assert bitset_to_list(bitset_difference(a, b)) == [0]


def test_difference_with_self() -> None:
    a: list[int] = bitset_new(64)
    bitset_set(a, 0)
    bitset_set(a, 31)
    bitset_set(a, 63)
    result: list[int] = bitset_difference(a, a)
    assert bitset_popcount(result) == 0


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_new_all_zero()
        passed += 1
        print("  PASS test_new_all_zero")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_new_all_zero: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_new_all_zero: " + str(e))
    try:
        test_new_size()
        passed += 1
        print("  PASS test_new_size")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_new_size: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_new_size: " + str(e))
    try:
        test_set_and_test()
        passed += 1
        print("  PASS test_set_and_test")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_and_test: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_and_test: " + str(e))
    try:
        test_set_high_bit()
        passed += 1
        print("  PASS test_set_high_bit")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_high_bit: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_high_bit: " + str(e))
    try:
        test_set_multiple()
        passed += 1
        print("  PASS test_set_multiple")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_multiple: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_multiple: " + str(e))
    try:
        test_set_word_boundary()
        passed += 1
        print("  PASS test_set_word_boundary")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_word_boundary: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_word_boundary: " + str(e))
    try:
        test_clear()
        passed += 1
        print("  PASS test_clear")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_clear: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_clear: " + str(e))
    try:
        test_clear_already_clear()
        passed += 1
        print("  PASS test_clear_already_clear")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_clear_already_clear: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_clear_already_clear: " + str(e))
    try:
        test_clear_preserves_others()
        passed += 1
        print("  PASS test_clear_preserves_others")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_clear_preserves_others: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_clear_preserves_others: " + str(e))
    try:
        test_toggle_on()
        passed += 1
        print("  PASS test_toggle_on")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_toggle_on: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_toggle_on: " + str(e))
    try:
        test_toggle_off()
        passed += 1
        print("  PASS test_toggle_off")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_toggle_off: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_toggle_off: " + str(e))
    try:
        test_toggle_double()
        passed += 1
        print("  PASS test_toggle_double")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_toggle_double: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_toggle_double: " + str(e))
    try:
        test_popcount_empty()
        passed += 1
        print("  PASS test_popcount_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_popcount_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_popcount_empty: " + str(e))
    try:
        test_popcount_one()
        passed += 1
        print("  PASS test_popcount_one")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_popcount_one: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_popcount_one: " + str(e))
    try:
        test_popcount_several()
        passed += 1
        print("  PASS test_popcount_several")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_popcount_several: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_popcount_several: " + str(e))
    try:
        test_popcount_full_word()
        passed += 1
        print("  PASS test_popcount_full_word")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_popcount_full_word: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_popcount_full_word: " + str(e))
    try:
        test_union_basic()
        passed += 1
        print("  PASS test_union_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_union_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_union_basic: " + str(e))
    try:
        test_union_different_sizes()
        passed += 1
        print("  PASS test_union_different_sizes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_union_different_sizes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_union_different_sizes: " + str(e))
    try:
        test_intersection_basic()
        passed += 1
        print("  PASS test_intersection_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_intersection_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_intersection_basic: " + str(e))
    try:
        test_intersection_disjoint()
        passed += 1
        print("  PASS test_intersection_disjoint")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_intersection_disjoint: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_intersection_disjoint: " + str(e))
    try:
        test_intersection_different_sizes()
        passed += 1
        print("  PASS test_intersection_different_sizes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_intersection_different_sizes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_intersection_different_sizes: " + str(e))
    try:
        test_difference_basic()
        passed += 1
        print("  PASS test_difference_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_difference_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_difference_basic: " + str(e))
    try:
        test_difference_no_overlap()
        passed += 1
        print("  PASS test_difference_no_overlap")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_difference_no_overlap: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_difference_no_overlap: " + str(e))
    try:
        test_to_list_empty()
        passed += 1
        print("  PASS test_to_list_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_to_list_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_to_list_empty: " + str(e))
    try:
        test_to_list_several()
        passed += 1
        print("  PASS test_to_list_several")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_to_list_several: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_to_list_several: " + str(e))
    try:
        test_to_list_word_boundary()
        passed += 1
        print("  PASS test_to_list_word_boundary")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_to_list_word_boundary: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_to_list_word_boundary: " + str(e))
    try:
        test_large_set_all()
        passed += 1
        print("  PASS test_large_set_all")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_large_set_all: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_large_set_all: " + str(e))
    try:
        test_large_even_bits()
        passed += 1
        print("  PASS test_large_even_bits")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_large_even_bits: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_large_even_bits: " + str(e))
    try:
        test_single_bit_bitset()
        passed += 1
        print("  PASS test_single_bit_bitset")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_single_bit_bitset: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_single_bit_bitset: " + str(e))
    try:
        test_set_idempotent()
        passed += 1
        print("  PASS test_set_idempotent")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_idempotent: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_idempotent: " + str(e))
    try:
        test_union_with_empty()
        passed += 1
        print("  PASS test_union_with_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_union_with_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_union_with_empty: " + str(e))
    try:
        test_intersection_with_empty()
        passed += 1
        print("  PASS test_intersection_with_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_intersection_with_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_intersection_with_empty: " + str(e))
    try:
        test_non_aligned_size()
        passed += 1
        print("  PASS test_non_aligned_size")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_non_aligned_size: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_non_aligned_size: " + str(e))
    try:
        test_non_aligned_size_50()
        passed += 1
        print("  PASS test_non_aligned_size_50")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_non_aligned_size_50: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_non_aligned_size_50: " + str(e))
    try:
        test_popcount_after_clear()
        passed += 1
        print("  PASS test_popcount_after_clear")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_popcount_after_clear: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_popcount_after_clear: " + str(e))
    try:
        test_toggle_word_boundary()
        passed += 1
        print("  PASS test_toggle_word_boundary")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_toggle_word_boundary: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_toggle_word_boundary: " + str(e))
    try:
        test_union_via_to_list()
        passed += 1
        print("  PASS test_union_via_to_list")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_union_via_to_list: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_union_via_to_list: " + str(e))
    try:
        test_intersection_via_to_list()
        passed += 1
        print("  PASS test_intersection_via_to_list")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_intersection_via_to_list: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_intersection_via_to_list: " + str(e))
    try:
        test_difference_via_to_list()
        passed += 1
        print("  PASS test_difference_via_to_list")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_difference_via_to_list: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_difference_via_to_list: " + str(e))
    try:
        test_difference_b_extra_bits_ignored()
        passed += 1
        print("  PASS test_difference_b_extra_bits_ignored")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_difference_b_extra_bits_ignored: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_difference_b_extra_bits_ignored: " + str(e))
    try:
        test_to_list_three_words()
        passed += 1
        print("  PASS test_to_list_three_words")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_to_list_three_words: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_to_list_three_words: " + str(e))
    try:
        test_set_ops_single_bit()
        passed += 1
        print("  PASS test_set_ops_single_bit")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_ops_single_bit: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_ops_single_bit: " + str(e))
    try:
        test_difference_with_self()
        passed += 1
        print("  PASS test_difference_with_self")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_difference_with_self: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_difference_with_self: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
