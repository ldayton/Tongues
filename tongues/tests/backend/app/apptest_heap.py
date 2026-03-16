"""Min-heap tests — invariants, operations, sorting, and edge cases."""

import sys

from lib.heap import heap_peek
from lib.heap import heap_pop
from lib.heap import heap_push
from lib.heap import heap_push_pop
from lib.heap import heap_replace
from lib.heap import heap_size
from lib.heap import heap_sort
from lib.heap import heapify


# -- Helpers --


def _is_min_heap(h: list[int]) -> bool:
    """Verify the min-heap invariant: parent <= both children."""
    i: int = 0
    n: int = len(h)
    while i < n:
        left: int = 2 * i + 1
        right: int = 2 * i + 2
        if left < n and h[i] > h[left]:
            return False
        if right < n and h[i] > h[right]:
            return False
        i += 1
    return True


# -- Push / Peek / Pop basics --


def test_push_single() -> None:
    h: list[int] = []
    heap_push(h, 42)
    assert heap_size(h) == 1
    assert heap_peek(h) == 42


def test_push_ascending() -> None:
    h: list[int] = []
    heap_push(h, 1)
    heap_push(h, 2)
    heap_push(h, 3)
    assert heap_peek(h) == 1
    assert _is_min_heap(h)


def test_push_descending() -> None:
    h: list[int] = []
    heap_push(h, 3)
    heap_push(h, 2)
    heap_push(h, 1)
    assert heap_peek(h) == 1
    assert _is_min_heap(h)


def test_pop_order() -> None:
    """Pop should return elements in ascending order."""
    h: list[int] = []
    heap_push(h, 5)
    heap_push(h, 3)
    heap_push(h, 7)
    heap_push(h, 1)
    heap_push(h, 4)
    assert heap_pop(h) == 1
    assert heap_pop(h) == 3
    assert heap_pop(h) == 4
    assert heap_pop(h) == 5
    assert heap_pop(h) == 7
    assert heap_size(h) == 0


def test_pop_empty() -> None:
    h: list[int] = []
    try:
        heap_pop(h)
        assert False, "expected IndexError"
    except IndexError:
        pass


def test_peek_empty() -> None:
    h: list[int] = []
    try:
        heap_peek(h)
        assert False, "expected IndexError"
    except IndexError:
        pass


def test_push_pop_single() -> None:
    h: list[int] = []
    heap_push(h, 10)
    assert heap_pop(h) == 10
    assert heap_size(h) == 0


# -- Duplicates --


def test_duplicates() -> None:
    h: list[int] = []
    heap_push(h, 5)
    heap_push(h, 5)
    heap_push(h, 5)
    assert heap_pop(h) == 5
    assert heap_pop(h) == 5
    assert heap_pop(h) == 5
    assert heap_size(h) == 0


def test_duplicates_mixed() -> None:
    h: list[int] = []
    heap_push(h, 3)
    heap_push(h, 1)
    heap_push(h, 3)
    heap_push(h, 1)
    heap_push(h, 2)
    assert heap_pop(h) == 1
    assert heap_pop(h) == 1
    assert heap_pop(h) == 2
    assert heap_pop(h) == 3
    assert heap_pop(h) == 3


# -- Negative values --


def test_negative_values() -> None:
    h: list[int] = []
    heap_push(h, -1)
    heap_push(h, -5)
    heap_push(h, -3)
    heap_push(h, 0)
    assert heap_pop(h) == -5
    assert heap_pop(h) == -3
    assert heap_pop(h) == -1
    assert heap_pop(h) == 0


# -- Heapify --


def test_heapify_empty() -> None:
    h: list[int] = []
    heapify(h)
    assert h == []


def test_heapify_single() -> None:
    h: list[int] = [42]
    heapify(h)
    assert h == [42]


def test_heapify_sorted() -> None:
    h: list[int] = [1, 2, 3, 4, 5]
    heapify(h)
    assert _is_min_heap(h)
    assert h[0] == 1


def test_heapify_reverse() -> None:
    h: list[int] = [5, 4, 3, 2, 1]
    heapify(h)
    assert _is_min_heap(h)
    assert h[0] == 1


def test_heapify_random() -> None:
    h: list[int] = [7, 2, 9, 1, 5, 8, 3, 6, 4]
    heapify(h)
    assert _is_min_heap(h)
    assert h[0] == 1


def test_heapify_then_pop_all() -> None:
    h: list[int] = [10, 4, 15, 20, 0, 8, 3]
    heapify(h)
    prev: int = heap_pop(h)
    while len(h) > 0:
        cur: int = heap_pop(h)
        assert cur >= prev
        prev = cur


# -- heap_sort --


def test_sort_empty() -> None:
    assert heap_sort([]) == []


def test_sort_single() -> None:
    assert heap_sort([1]) == [1]


def test_sort_already_sorted() -> None:
    assert heap_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_sort_reverse() -> None:
    assert heap_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_sort_duplicates() -> None:
    assert heap_sort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]) == [
        1,
        1,
        2,
        3,
        3,
        4,
        5,
        5,
        5,
        6,
        9,
    ]


def test_sort_negative() -> None:
    assert heap_sort([-3, 0, -1, 5, -2]) == [-3, -2, -1, 0, 5]


def test_sort_preserves_input() -> None:
    data: list[int] = [3, 1, 2]
    result: list[int] = heap_sort(data)
    assert result == [1, 2, 3]
    assert data == [3, 1, 2]


# -- push_pop --


def test_push_pop_smaller() -> None:
    """Push a value smaller than min — returned immediately."""
    h: list[int] = [1, 3, 5]
    heapify(h)
    assert heap_push_pop(h, 0) == 0
    assert heap_peek(h) == 1


def test_push_pop_larger() -> None:
    """Push a value larger than min — min is returned, new value enters heap."""
    h: list[int] = [1, 3, 5]
    heapify(h)
    assert heap_push_pop(h, 2) == 1
    assert heap_peek(h) == 2
    assert _is_min_heap(h)


def test_push_pop_equal() -> None:
    h: list[int] = [1, 3, 5]
    heapify(h)
    assert heap_push_pop(h, 1) == 1
    assert heap_peek(h) == 1


def test_push_pop_empty() -> None:
    """On empty heap, just returns the value."""
    h: list[int] = []
    assert heap_push_pop(h, 42) == 42
    assert heap_size(h) == 0


# -- replace --


def test_replace_basic() -> None:
    h: list[int] = [1, 3, 5]
    heapify(h)
    assert heap_replace(h, 4) == 1
    assert heap_peek(h) == 3
    assert _is_min_heap(h)


def test_replace_smaller() -> None:
    h: list[int] = [1, 3, 5]
    heapify(h)
    assert heap_replace(h, 0) == 1
    assert heap_peek(h) == 0
    assert _is_min_heap(h)


def test_replace_empty() -> None:
    h: list[int] = []
    try:
        heap_replace(h, 1)
        assert False, "expected IndexError"
    except IndexError:
        pass


# -- Size invariants --


def test_size_after_operations() -> None:
    h: list[int] = []
    assert heap_size(h) == 0
    heap_push(h, 1)
    assert heap_size(h) == 1
    heap_push(h, 2)
    assert heap_size(h) == 2
    heap_pop(h)
    assert heap_size(h) == 1
    heap_pop(h)
    assert heap_size(h) == 0


# -- Larger dataset --


def test_100_elements() -> None:
    h: list[int] = []
    i: int = 100
    while i > 0:
        heap_push(h, i)
        i -= 1
    assert _is_min_heap(h)
    assert heap_peek(h) == 1
    prev: int = heap_pop(h)
    while len(h) > 0:
        cur: int = heap_pop(h)
        assert cur >= prev
        prev = cur


def test_heapify_100() -> None:
    data: list[int] = []
    i: int = 100
    while i > 0:
        data.append(i)
        i -= 1
    heapify(data)
    assert _is_min_heap(data)
    assert data[0] == 1


# -- Two-element heap (single-child sift_down path) --


def test_two_elements() -> None:
    h: list[int] = []
    heap_push(h, 2)
    heap_push(h, 1)
    assert _is_min_heap(h)
    assert heap_pop(h) == 1
    assert _is_min_heap(h)
    assert heap_pop(h) == 2


def test_two_elements_already_ordered() -> None:
    h: list[int] = []
    heap_push(h, 1)
    heap_push(h, 2)
    assert heap_pop(h) == 1
    assert _is_min_heap(h)
    assert heap_pop(h) == 2


# -- Invariant holds after every intermediate pop --


def test_invariant_after_each_pop() -> None:
    h: list[int] = [9, 1, 7, 3, 5, 2, 8, 4, 6, 0]
    heapify(h)
    while len(h) > 0:
        assert _is_min_heap(h)
        heap_pop(h)
    assert _is_min_heap(h)


# -- Interleaved push/pop --


def test_interleaved() -> None:
    h: list[int] = []
    heap_push(h, 5)
    heap_push(h, 3)
    assert heap_pop(h) == 3
    heap_push(h, 7)
    heap_push(h, 1)
    assert heap_pop(h) == 1
    assert heap_pop(h) == 5
    heap_push(h, 2)
    heap_push(h, 4)
    assert heap_pop(h) == 2
    assert heap_pop(h) == 4
    assert heap_pop(h) == 7
    assert heap_size(h) == 0


# -- All-same values --


def test_all_same() -> None:
    h: list[int] = []
    i: int = 0
    while i < 10:
        heap_push(h, 42)
        i += 1
    assert _is_min_heap(h)
    while len(h) > 0:
        assert heap_pop(h) == 42


# -- Extreme values --


def test_extreme_values() -> None:
    h: list[int] = []
    heap_push(h, 2147483647)
    heap_push(h, -2147483648)
    heap_push(h, 0)
    assert heap_pop(h) == -2147483648
    assert heap_pop(h) == 0
    assert heap_pop(h) == 2147483647


# -- push_pop / replace size invariants --


def test_push_pop_size_unchanged() -> None:
    h: list[int] = [1, 3, 5]
    heapify(h)
    heap_push_pop(h, 2)
    assert heap_size(h) == 3
    assert _is_min_heap(h)


def test_replace_size_unchanged() -> None:
    h: list[int] = [1, 3, 5]
    heapify(h)
    heap_replace(h, 2)
    assert heap_size(h) == 3
    assert _is_min_heap(h)


# -- heapify idempotent --


def test_heapify_idempotent() -> None:
    h: list[int] = [5, 3, 7, 1, 9, 2]
    heapify(h)
    copy: list[int] = []
    i: int = 0
    while i < len(h):
        copy.append(h[i])
        i += 1
    heapify(h)
    assert h == copy


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_push_single()
        passed += 1
        print("  PASS test_push_single")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_push_single: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_push_single: " + str(e))
    try:
        test_push_ascending()
        passed += 1
        print("  PASS test_push_ascending")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_push_ascending: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_push_ascending: " + str(e))
    try:
        test_push_descending()
        passed += 1
        print("  PASS test_push_descending")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_push_descending: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_push_descending: " + str(e))
    try:
        test_pop_order()
        passed += 1
        print("  PASS test_pop_order")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_pop_order: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_pop_order: " + str(e))
    try:
        test_pop_empty()
        passed += 1
        print("  PASS test_pop_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_pop_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_pop_empty: " + str(e))
    try:
        test_peek_empty()
        passed += 1
        print("  PASS test_peek_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_peek_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_peek_empty: " + str(e))
    try:
        test_push_pop_single()
        passed += 1
        print("  PASS test_push_pop_single")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_push_pop_single: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_push_pop_single: " + str(e))
    try:
        test_duplicates()
        passed += 1
        print("  PASS test_duplicates")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_duplicates: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_duplicates: " + str(e))
    try:
        test_duplicates_mixed()
        passed += 1
        print("  PASS test_duplicates_mixed")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_duplicates_mixed: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_duplicates_mixed: " + str(e))
    try:
        test_negative_values()
        passed += 1
        print("  PASS test_negative_values")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_negative_values: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_negative_values: " + str(e))
    try:
        test_heapify_empty()
        passed += 1
        print("  PASS test_heapify_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_heapify_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_heapify_empty: " + str(e))
    try:
        test_heapify_single()
        passed += 1
        print("  PASS test_heapify_single")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_heapify_single: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_heapify_single: " + str(e))
    try:
        test_heapify_sorted()
        passed += 1
        print("  PASS test_heapify_sorted")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_heapify_sorted: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_heapify_sorted: " + str(e))
    try:
        test_heapify_reverse()
        passed += 1
        print("  PASS test_heapify_reverse")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_heapify_reverse: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_heapify_reverse: " + str(e))
    try:
        test_heapify_random()
        passed += 1
        print("  PASS test_heapify_random")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_heapify_random: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_heapify_random: " + str(e))
    try:
        test_heapify_then_pop_all()
        passed += 1
        print("  PASS test_heapify_then_pop_all")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_heapify_then_pop_all: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_heapify_then_pop_all: " + str(e))
    try:
        test_sort_empty()
        passed += 1
        print("  PASS test_sort_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_sort_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_sort_empty: " + str(e))
    try:
        test_sort_single()
        passed += 1
        print("  PASS test_sort_single")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_sort_single: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_sort_single: " + str(e))
    try:
        test_sort_already_sorted()
        passed += 1
        print("  PASS test_sort_already_sorted")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_sort_already_sorted: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_sort_already_sorted: " + str(e))
    try:
        test_sort_reverse()
        passed += 1
        print("  PASS test_sort_reverse")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_sort_reverse: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_sort_reverse: " + str(e))
    try:
        test_sort_duplicates()
        passed += 1
        print("  PASS test_sort_duplicates")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_sort_duplicates: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_sort_duplicates: " + str(e))
    try:
        test_sort_negative()
        passed += 1
        print("  PASS test_sort_negative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_sort_negative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_sort_negative: " + str(e))
    try:
        test_sort_preserves_input()
        passed += 1
        print("  PASS test_sort_preserves_input")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_sort_preserves_input: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_sort_preserves_input: " + str(e))
    try:
        test_push_pop_smaller()
        passed += 1
        print("  PASS test_push_pop_smaller")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_push_pop_smaller: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_push_pop_smaller: " + str(e))
    try:
        test_push_pop_larger()
        passed += 1
        print("  PASS test_push_pop_larger")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_push_pop_larger: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_push_pop_larger: " + str(e))
    try:
        test_push_pop_equal()
        passed += 1
        print("  PASS test_push_pop_equal")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_push_pop_equal: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_push_pop_equal: " + str(e))
    try:
        test_push_pop_empty()
        passed += 1
        print("  PASS test_push_pop_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_push_pop_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_push_pop_empty: " + str(e))
    try:
        test_replace_basic()
        passed += 1
        print("  PASS test_replace_basic")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_replace_basic: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_replace_basic: " + str(e))
    try:
        test_replace_smaller()
        passed += 1
        print("  PASS test_replace_smaller")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_replace_smaller: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_replace_smaller: " + str(e))
    try:
        test_replace_empty()
        passed += 1
        print("  PASS test_replace_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_replace_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_replace_empty: " + str(e))
    try:
        test_size_after_operations()
        passed += 1
        print("  PASS test_size_after_operations")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_size_after_operations: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_size_after_operations: " + str(e))
    try:
        test_100_elements()
        passed += 1
        print("  PASS test_100_elements")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_100_elements: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_100_elements: " + str(e))
    try:
        test_heapify_100()
        passed += 1
        print("  PASS test_heapify_100")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_heapify_100: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_heapify_100: " + str(e))
    try:
        test_two_elements()
        passed += 1
        print("  PASS test_two_elements")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_two_elements: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_two_elements: " + str(e))
    try:
        test_two_elements_already_ordered()
        passed += 1
        print("  PASS test_two_elements_already_ordered")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_two_elements_already_ordered: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_two_elements_already_ordered: " + str(e))
    try:
        test_invariant_after_each_pop()
        passed += 1
        print("  PASS test_invariant_after_each_pop")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invariant_after_each_pop: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invariant_after_each_pop: " + str(e))
    try:
        test_interleaved()
        passed += 1
        print("  PASS test_interleaved")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_interleaved: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_interleaved: " + str(e))
    try:
        test_all_same()
        passed += 1
        print("  PASS test_all_same")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_all_same: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_all_same: " + str(e))
    try:
        test_extreme_values()
        passed += 1
        print("  PASS test_extreme_values")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_extreme_values: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_extreme_values: " + str(e))
    try:
        test_push_pop_size_unchanged()
        passed += 1
        print("  PASS test_push_pop_size_unchanged")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_push_pop_size_unchanged: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_push_pop_size_unchanged: " + str(e))
    try:
        test_replace_size_unchanged()
        passed += 1
        print("  PASS test_replace_size_unchanged")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_replace_size_unchanged: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_replace_size_unchanged: " + str(e))
    try:
        test_heapify_idempotent()
        passed += 1
        print("  PASS test_heapify_idempotent")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_heapify_idempotent: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_heapify_idempotent: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
