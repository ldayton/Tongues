"""Min-heap tests — invariants, operations, sorting, and edge cases."""

import sys

from lib.heap import heap_sort
from lib.heap import heapify
from lib.heap import peek
from lib.heap import pop
from lib.heap import push
from lib.heap import push_pop
from lib.heap import replace
from lib.heap import size


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
    push(h, 42)
    assert size(h) == 1
    assert peek(h) == 42


def test_push_ascending() -> None:
    h: list[int] = []
    push(h, 1)
    push(h, 2)
    push(h, 3)
    assert peek(h) == 1
    assert _is_min_heap(h)


def test_push_descending() -> None:
    h: list[int] = []
    push(h, 3)
    push(h, 2)
    push(h, 1)
    assert peek(h) == 1
    assert _is_min_heap(h)


def test_pop_order() -> None:
    """Pop should return elements in ascending order."""
    h: list[int] = []
    push(h, 5)
    push(h, 3)
    push(h, 7)
    push(h, 1)
    push(h, 4)
    assert pop(h) == 1
    assert pop(h) == 3
    assert pop(h) == 4
    assert pop(h) == 5
    assert pop(h) == 7
    assert size(h) == 0


def test_pop_empty() -> None:
    h: list[int] = []
    try:
        pop(h)
        assert False, "expected IndexError"
    except IndexError:
        pass


def test_peek_empty() -> None:
    h: list[int] = []
    try:
        peek(h)
        assert False, "expected IndexError"
    except IndexError:
        pass


def test_push_pop_single() -> None:
    h: list[int] = []
    push(h, 10)
    assert pop(h) == 10
    assert size(h) == 0


# -- Duplicates --


def test_duplicates() -> None:
    h: list[int] = []
    push(h, 5)
    push(h, 5)
    push(h, 5)
    assert pop(h) == 5
    assert pop(h) == 5
    assert pop(h) == 5
    assert size(h) == 0


def test_duplicates_mixed() -> None:
    h: list[int] = []
    push(h, 3)
    push(h, 1)
    push(h, 3)
    push(h, 1)
    push(h, 2)
    assert pop(h) == 1
    assert pop(h) == 1
    assert pop(h) == 2
    assert pop(h) == 3
    assert pop(h) == 3


# -- Negative values --


def test_negative_values() -> None:
    h: list[int] = []
    push(h, -1)
    push(h, -5)
    push(h, -3)
    push(h, 0)
    assert pop(h) == -5
    assert pop(h) == -3
    assert pop(h) == -1
    assert pop(h) == 0


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
    prev: int = pop(h)
    while len(h) > 0:
        cur: int = pop(h)
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
    assert push_pop(h, 0) == 0
    assert peek(h) == 1


def test_push_pop_larger() -> None:
    """Push a value larger than min — min is returned, new value enters heap."""
    h: list[int] = [1, 3, 5]
    heapify(h)
    assert push_pop(h, 2) == 1
    assert peek(h) == 2
    assert _is_min_heap(h)


def test_push_pop_equal() -> None:
    h: list[int] = [1, 3, 5]
    heapify(h)
    assert push_pop(h, 1) == 1
    assert peek(h) == 1


def test_push_pop_empty() -> None:
    """On empty heap, just returns the value."""
    h: list[int] = []
    assert push_pop(h, 42) == 42
    assert size(h) == 0


# -- replace --


def test_replace_basic() -> None:
    h: list[int] = [1, 3, 5]
    heapify(h)
    assert replace(h, 4) == 1
    assert peek(h) == 3
    assert _is_min_heap(h)


def test_replace_smaller() -> None:
    h: list[int] = [1, 3, 5]
    heapify(h)
    assert replace(h, 0) == 1
    assert peek(h) == 0
    assert _is_min_heap(h)


def test_replace_empty() -> None:
    h: list[int] = []
    try:
        replace(h, 1)
        assert False, "expected IndexError"
    except IndexError:
        pass


# -- Size invariants --


def test_size_after_operations() -> None:
    h: list[int] = []
    assert size(h) == 0
    push(h, 1)
    assert size(h) == 1
    push(h, 2)
    assert size(h) == 2
    pop(h)
    assert size(h) == 1
    pop(h)
    assert size(h) == 0


# -- Larger dataset --


def test_100_elements() -> None:
    h: list[int] = []
    i: int = 100
    while i > 0:
        push(h, i)
        i -= 1
    assert _is_min_heap(h)
    assert peek(h) == 1
    prev: int = pop(h)
    while len(h) > 0:
        cur: int = pop(h)
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
    push(h, 2)
    push(h, 1)
    assert _is_min_heap(h)
    assert pop(h) == 1
    assert _is_min_heap(h)
    assert pop(h) == 2


def test_two_elements_already_ordered() -> None:
    h: list[int] = []
    push(h, 1)
    push(h, 2)
    assert pop(h) == 1
    assert _is_min_heap(h)
    assert pop(h) == 2


# -- Invariant holds after every intermediate pop --


def test_invariant_after_each_pop() -> None:
    h: list[int] = [9, 1, 7, 3, 5, 2, 8, 4, 6, 0]
    heapify(h)
    while len(h) > 0:
        assert _is_min_heap(h)
        pop(h)
    assert _is_min_heap(h)


# -- Interleaved push/pop --


def test_interleaved() -> None:
    h: list[int] = []
    push(h, 5)
    push(h, 3)
    assert pop(h) == 3
    push(h, 7)
    push(h, 1)
    assert pop(h) == 1
    assert pop(h) == 5
    push(h, 2)
    push(h, 4)
    assert pop(h) == 2
    assert pop(h) == 4
    assert pop(h) == 7
    assert size(h) == 0


# -- All-same values --


def test_all_same() -> None:
    h: list[int] = []
    i: int = 0
    while i < 10:
        push(h, 42)
        i += 1
    assert _is_min_heap(h)
    while len(h) > 0:
        assert pop(h) == 42


# -- Extreme values --


def test_extreme_values() -> None:
    h: list[int] = []
    push(h, 2147483647)
    push(h, -2147483648)
    push(h, 0)
    assert pop(h) == -2147483648
    assert pop(h) == 0
    assert pop(h) == 2147483647


# -- push_pop / replace size invariants --


def test_push_pop_size_unchanged() -> None:
    h: list[int] = [1, 3, 5]
    heapify(h)
    push_pop(h, 2)
    assert size(h) == 3
    assert _is_min_heap(h)


def test_replace_size_unchanged() -> None:
    h: list[int] = [1, 3, 5]
    heapify(h)
    replace(h, 2)
    assert size(h) == 3
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
    tests = [
        ("test_push_single", test_push_single),
        ("test_push_ascending", test_push_ascending),
        ("test_push_descending", test_push_descending),
        ("test_pop_order", test_pop_order),
        ("test_pop_empty", test_pop_empty),
        ("test_peek_empty", test_peek_empty),
        ("test_push_pop_single", test_push_pop_single),
        ("test_duplicates", test_duplicates),
        ("test_duplicates_mixed", test_duplicates_mixed),
        ("test_negative_values", test_negative_values),
        ("test_heapify_empty", test_heapify_empty),
        ("test_heapify_single", test_heapify_single),
        ("test_heapify_sorted", test_heapify_sorted),
        ("test_heapify_reverse", test_heapify_reverse),
        ("test_heapify_random", test_heapify_random),
        ("test_heapify_then_pop_all", test_heapify_then_pop_all),
        ("test_sort_empty", test_sort_empty),
        ("test_sort_single", test_sort_single),
        ("test_sort_already_sorted", test_sort_already_sorted),
        ("test_sort_reverse", test_sort_reverse),
        ("test_sort_duplicates", test_sort_duplicates),
        ("test_sort_negative", test_sort_negative),
        ("test_sort_preserves_input", test_sort_preserves_input),
        ("test_push_pop_smaller", test_push_pop_smaller),
        ("test_push_pop_larger", test_push_pop_larger),
        ("test_push_pop_equal", test_push_pop_equal),
        ("test_push_pop_empty", test_push_pop_empty),
        ("test_replace_basic", test_replace_basic),
        ("test_replace_smaller", test_replace_smaller),
        ("test_replace_empty", test_replace_empty),
        ("test_size_after_operations", test_size_after_operations),
        ("test_100_elements", test_100_elements),
        ("test_heapify_100", test_heapify_100),
        ("test_two_elements", test_two_elements),
        ("test_two_elements_already_ordered", test_two_elements_already_ordered),
        ("test_invariant_after_each_pop", test_invariant_after_each_pop),
        ("test_interleaved", test_interleaved),
        ("test_all_same", test_all_same),
        ("test_extreme_values", test_extreme_values),
        ("test_push_pop_size_unchanged", test_push_pop_size_unchanged),
        ("test_replace_size_unchanged", test_replace_size_unchanged),
        ("test_heapify_idempotent", test_heapify_idempotent),
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
