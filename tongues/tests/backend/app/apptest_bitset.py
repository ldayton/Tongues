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
    tests = [
        ("test_new_all_zero", test_new_all_zero),
        ("test_new_size", test_new_size),
        ("test_set_and_test", test_set_and_test),
        ("test_set_high_bit", test_set_high_bit),
        ("test_set_multiple", test_set_multiple),
        ("test_set_word_boundary", test_set_word_boundary),
        ("test_clear", test_clear),
        ("test_clear_already_clear", test_clear_already_clear),
        ("test_clear_preserves_others", test_clear_preserves_others),
        ("test_toggle_on", test_toggle_on),
        ("test_toggle_off", test_toggle_off),
        ("test_toggle_double", test_toggle_double),
        ("test_popcount_empty", test_popcount_empty),
        ("test_popcount_one", test_popcount_one),
        ("test_popcount_several", test_popcount_several),
        ("test_popcount_full_word", test_popcount_full_word),
        ("test_union_basic", test_union_basic),
        ("test_union_different_sizes", test_union_different_sizes),
        ("test_intersection_basic", test_intersection_basic),
        ("test_intersection_disjoint", test_intersection_disjoint),
        ("test_intersection_different_sizes", test_intersection_different_sizes),
        ("test_difference_basic", test_difference_basic),
        ("test_difference_no_overlap", test_difference_no_overlap),
        ("test_to_list_empty", test_to_list_empty),
        ("test_to_list_several", test_to_list_several),
        ("test_to_list_word_boundary", test_to_list_word_boundary),
        ("test_large_set_all", test_large_set_all),
        ("test_large_even_bits", test_large_even_bits),
        ("test_single_bit_bitset", test_single_bit_bitset),
        ("test_set_idempotent", test_set_idempotent),
        ("test_union_with_empty", test_union_with_empty),
        ("test_intersection_with_empty", test_intersection_with_empty),
        ("test_difference_with_self", test_difference_with_self),
    ]
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print("  PASS " + name)
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {name}: {e!s}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {name}: {e!s}")
    print(f"{passed!s} passed, {failed!s} failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
