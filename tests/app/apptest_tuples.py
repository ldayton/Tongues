"""Tuple object tests."""

import sys


def minmax(items: list[int]) -> tuple[int, int]:
    """Return min and max of items."""
    return (min(items), max(items))


def make_pair() -> tuple[int, int]:
    """Return (1, 2)."""
    return (1, 2)


def make_pair_for_ignore() -> tuple[int, int]:
    """Return (1, 4) for ignore test."""
    return (1, 4)


def make_packing_pair() -> tuple[int, int]:
    """Return (10, 20)."""
    return (10, 20)


def swap(x: int, y: int) -> tuple[int, int]:
    """Swap two values."""
    return (y, x)


def test_tuple_equality() -> None:
    """Tuple equality comparisons."""
    assert (1, 2, 3) == (1, 2, 3)
    assert () == ()
    assert (1,) == (1,)
    assert not ((1, 2) == (1, 3))
    assert (1, 2) != (1, 3)
    assert (1, 2) != (1, 2, 3)
    assert (1, 2, 3) != (1, 2)


def test_tuple_ordering() -> None:
    """Tuple ordering is lexicographic."""
    assert (1, 2) < (1, 3)
    assert (1, 2) < (1, 2, 3)
    assert () < (1,)
    assert (1,) < (2,)
    assert (1, 2) <= (1, 2)
    assert (1, 2) <= (1, 3)
    assert (1, 3) > (1, 2)
    assert (1, 2, 3) > (1, 2)
    assert (1, 2) >= (1, 2)
    assert (1, 3) >= (1, 2)


def test_tuple_length() -> None:
    """len() returns element count."""
    assert len(()) == 0
    assert len((1,)) == 1
    assert len((1, 2, 3)) == 3
    assert len((1, 2, 3, 4, 5)) == 5


def test_tuple_indexing() -> None:
    """Indexing returns element."""
    t: tuple[int, int, int, int, int] = (10, 20, 30, 40, 50)
    assert t[0] == 10
    assert t[1] == 20
    assert t[4] == 50
    # Negative indexing
    assert t[-1] == 50
    assert t[-2] == 40
    assert t[-5] == 10


def test_tuple_slicing() -> None:
    """Tuple slicing returns new tuple."""
    t: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5)
    assert t[0:2] == (1, 2)
    assert t[1:4] == (2, 3, 4)
    assert t[:3] == (1, 2, 3)
    assert t[2:] == (3, 4, 5)
    assert t[:] == (1, 2, 3, 4, 5)
    assert t[::2] == (1, 3, 5)
    assert t[::-1] == (5, 4, 3, 2, 1)
    # Empty slices
    assert t[2:2] == ()
    assert t[5:10] == ()


def test_tuple_slice_step() -> None:
    """Slicing with step."""
    t: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert t[::2] == (0, 2, 4, 6, 8)
    assert t[1::2] == (1, 3, 5, 7, 9)
    assert t[::3] == (0, 3, 6, 9)
    assert t[::-1] == (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
    assert t[::-2] == (9, 7, 5, 3, 1)


def test_tuple_concatenation() -> None:
    """Tuple concatenation with +."""
    assert (1, 2) + (3, 4) == (1, 2, 3, 4)
    assert () + (1,) == (1,)
    assert (1,) + () == (1,)
    assert () + () == ()
    assert (1,) + (2,) + (3,) == (1, 2, 3)


def test_tuple_repetition() -> None:
    """Tuple repetition with *."""
    assert (1,) * 3 == (1, 1, 1)
    assert (1, 2) * 2 == (1, 2, 1, 2)
    assert (1,) * 0 == ()
    assert (1,) * 1 == (1,)
    assert 3 * (1,) == (1, 1, 1)
    assert () * 5 == ()


def test_tuple_repetition_negative() -> None:
    """Negative multiplier gives empty tuple."""
    assert (1, 2, 3) * -1 == ()
    assert (1, 2, 3) * -100 == ()
    assert -5 * (1, 2) == ()


def test_tuple_contains() -> None:
    """Membership testing with in."""
    t: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5)
    assert 1 in t
    assert 3 in t
    assert 5 in t
    assert not (0 in t)
    assert not (6 in t)
    assert 0 not in t
    assert 6 not in t


def test_tuple_contains_empty() -> None:
    """Membership in empty tuple."""
    t: tuple[int, ...] = ()
    assert not (1 in t)
    assert 1 not in t


def test_tuple_bool() -> None:
    """Tuple truthiness - empty is falsy."""
    assert bool((1, 2, 3)) == True
    assert bool((0,)) == True
    assert bool(()) == False
    assert not ()
    assert (1,)


def test_tuple_count() -> None:
    """count() returns number of occurrences."""
    t: tuple[int, ...] = (1, 2, 2, 3, 2, 4)
    assert t.count(1) == 1
    assert t.count(2) == 3
    assert t.count(3) == 1
    assert t.count(5) == 0
    assert ().count(1) == 0


def test_tuple_index() -> None:
    """index() returns first index of value."""
    t: tuple[int, ...] = (10, 20, 30, 20, 40)
    assert t.index(10) == 0
    assert t.index(20) == 1
    assert t.index(30) == 2
    assert t.index(40) == 4


def test_tuple_iteration() -> None:
    """Iterating over tuple."""
    t: tuple[int, int, int] = (1, 2, 3)
    result: list[int] = []
    for x in t:
        result.append(x)
    assert result == [1, 2, 3]


def test_tuple_iteration_empty() -> None:
    """Iterating over empty tuple."""
    t: tuple[int, ...] = ()
    count: int = 0
    for x in t:
        count = count + 1
    assert count == 0


def test_tuple_enumerate() -> None:
    """enumerate() with tuple."""
    t: tuple[str, str, str] = ("a", "b", "c")
    indices: list[int] = []
    values: list[str] = []
    for i, v in enumerate(t):
        indices.append(i)
        values.append(v)
    assert indices == [0, 1, 2]
    assert values == ["a", "b", "c"]


def test_tuple_unpacking() -> None:
    """Tuple unpacking."""
    a, b = make_pair()
    assert a == 1
    assert b == 2


def test_tuple_unpacking_two() -> None:
    """Two-element tuple unpacking."""
    s, n = ("hello", 42)
    assert s == "hello"
    assert n == 42


def test_tuple_unpacking_nested() -> None:
    """Nested tuple unpacking."""
    a, inner = (1, (2, 3))
    assert a == 1
    assert inner == (2, 3)
    b, c = (inner[0], inner[1])
    assert b == 2
    assert c == 3


def test_tuple_swap() -> None:
    """Tuple swap idiom."""
    a: int = 1
    b: int = 2
    a, b = swap(a, b)
    assert a == 2
    assert b == 1


def test_tuple_sum() -> None:
    """sum() of tuple."""
    assert sum(()) == 0
    assert sum((1,)) == 1
    assert sum((1, 2, 3)) == 6
    assert sum((1, 2, 3, 4, 5)) == 15
    assert sum((-1, 0, 1)) == 0


def test_tuple_min_max() -> None:
    """min() and max() of tuple."""
    t: tuple[int, ...] = (3, 1, 4, 1, 5, 9, 2, 6)
    assert min(t) == 1
    assert max(t) == 9
    assert min((42,)) == 42
    assert max((42,)) == 42
    assert min((-5, -1, -10)) == -10
    assert max((-5, -1, -10)) == -1


def test_tuple_sorted() -> None:
    """sorted() returns list from tuple."""
    t: tuple[int, ...] = (3, 1, 4, 1, 5)
    result: list[int] = sorted(t)
    assert result == [1, 1, 3, 4, 5]
    assert t == (3, 1, 4, 1, 5)


def test_tuple_sorted_reverse() -> None:
    """sorted(reverse=True) on tuple."""
    t: tuple[int, ...] = (3, 1, 4, 1, 5)
    result: list[int] = sorted(t, reverse=True)
    assert result == [5, 4, 3, 1, 1]


def test_tuple_all_any() -> None:
    """all() and any() on tuples."""
    assert all((True, True, True)) == True
    assert all((True, False, True)) == False
    assert all(()) == True
    assert any((False, False, True)) == True
    assert any((False, False, False)) == False
    assert any(()) == False


def test_tuple_all_any_truthy() -> None:
    """all() and any() use truthiness."""
    assert all((1, 2, 3)) == True
    assert all((1, 0, 3)) == False
    assert any((0, 0, 1)) == True
    assert any((0, 0, 0)) == False
    assert any(("", "", "x")) == True
    assert any(("", "", "")) == False


def test_tuple_heterogeneous() -> None:
    """Tuples can hold different types."""
    t: tuple[int, str, float] = (1, "hello", 3.14)
    assert t[0] == 1
    assert t[1] == "hello"
    assert t[2] == 3.14


def test_tuple_nested() -> None:
    """Nested tuples."""
    t: tuple[tuple[int, int], tuple[int, int]] = ((1, 2), (3, 4))
    assert t[0] == (1, 2)
    assert t[1] == (3, 4)
    assert t[0][0] == 1
    assert t[1][1] == 4


def test_tuple_as_dict_key() -> None:
    """Tuples can be dict keys."""
    d: dict[tuple[int, int], str] = {(0, 0): "origin", (1, 0): "right", (0, 1): "up"}
    assert d[(0, 0)] == "origin"
    assert d[(1, 0)] == "right"
    assert (0, 0) in d
    assert (2, 2) not in d


def test_tuple_from_list() -> None:
    """tuple() from list."""
    items: list[int] = [1, 2, 3, 4, 5]
    t: tuple[int, ...] = tuple(items)
    assert t == (1, 2, 3, 4, 5)


def test_tuple_from_string() -> None:
    """tuple() from string."""
    t: tuple[str, ...] = tuple("hello")
    assert t == ("h", "e", "l", "l", "o")
    assert tuple("") == ()


def test_tuple_from_range() -> None:
    """tuple() from range."""
    t: tuple[int, ...] = tuple(range(5))
    assert t == (0, 1, 2, 3, 4)
    t = tuple(range(2, 6))
    assert t == (2, 3, 4, 5)


def test_tuple_from_set() -> None:
    """tuple() from set (order unspecified)."""
    s: set[int] = {1, 2, 3}
    t: tuple[int, ...] = tuple(s)
    assert len(t) == 3
    assert 1 in t
    assert 2 in t
    assert 3 in t


def test_tuple_single_element() -> None:
    """Single-element tuple requires trailing comma."""
    t: tuple[int] = (42,)
    assert len(t) == 1
    assert t[0] == 42
    # Without comma, it's just grouping
    n: int = (42)
    assert n == 42


def test_tuple_immutable() -> None:
    """Tuples are immutable - no modification methods."""
    t: tuple[int, int, int] = (1, 2, 3)
    # Can't do t[0] = 10 - would be a type error
    # But we can create new tuples
    t2: tuple[int, ...] = t + (4,)
    assert t == (1, 2, 3)
    assert t2 == (1, 2, 3, 4)


def test_tuple_zip() -> None:
    """zip() produces tuples."""
    a: list[int] = [1, 2, 3]
    b: list[str] = ["a", "b", "c"]
    pairs: list[tuple[int, str]] = list(zip(a, b))
    assert pairs == [(1, "a"), (2, "b"), (3, "c")]


def test_tuple_zip_unequal() -> None:
    """zip() stops at shortest."""
    a: list[int] = [1, 2, 3, 4, 5]
    b: list[str] = ["a", "b", "c"]
    pairs: list[tuple[int, str]] = list(zip(a, b))
    assert len(pairs) == 3
    assert pairs == [(1, "a"), (2, "b"), (3, "c")]


def test_tuple_comparison_mixed_length() -> None:
    """Comparison of different length tuples."""
    assert (1, 2) < (1, 2, 0)
    assert (1, 2, 0) > (1, 2)
    assert (1, 2) != (1, 2, 0)


def test_tuple_comparison_empty() -> None:
    """Empty tuple comparisons."""
    assert () == ()
    assert () < (1,)
    assert () < (0,)
    assert not (() > (1,))
    assert () <= ()
    assert () >= ()


def test_tuple_with_none() -> None:
    """Tuples containing None."""
    t: tuple[int | None, int | None, int | None] = (1, None, 3)
    assert t[0] == 1
    assert t[1] is None
    assert t[2] == 3
    assert None in t


def test_tuple_return_multiple() -> None:
    """Functions returning tuples."""
    result: tuple[int, int] = minmax([3, 1, 4, 1, 5, 9])
    assert result == (1, 9)
    lo, hi = minmax([3, 1, 4, 1, 5, 9])
    assert lo == 1
    assert hi == 9


def test_tuple_divmod() -> None:
    """divmod() returns tuple."""
    q, r = divmod(17, 5)
    assert q == 3
    assert r == 2
    result: tuple[int, int] = divmod(17, 5)
    assert result == (3, 2)


def test_tuple_enumerate_unpack() -> None:
    """Unpacking enumerate results."""
    items: list[str] = ["a", "b", "c"]
    for i, v in enumerate(items):
        if i == 0:
            assert v == "a"
        elif i == 1:
            assert v == "b"
        elif i == 2:
            assert v == "c"


def test_tuple_items_unpack() -> None:
    """Unpacking dict items."""
    d: dict[str, int] = {"a": 1, "b": 2}
    keys: list[str] = []
    vals: list[int] = []
    for k, v in d.items():
        keys.append(k)
        vals.append(v)
    assert len(keys) == 2
    assert "a" in keys
    assert "b" in keys


def test_tuple_packing_no_parens() -> None:
    """Tuple packing without parentheses."""
    t = 1, 2, 3
    assert t == (1, 2, 3)
    assert len(t) == 3
    a, b = make_packing_pair()
    assert a == 10
    assert b == 20


def test_tuple_trailing_comma() -> None:
    """Trailing comma is allowed in multi-element tuples."""
    t1: tuple[int, int, int] = (1, 2, 3,)
    t2: tuple[int, int, int] = (1, 2, 3)
    assert t1 == t2
    assert t1 == (1, 2, 3)


def test_tuple_in_set() -> None:
    """Tuples can be set elements."""
    s: set[tuple[int, int]] = {(1, 2), (3, 4), (1, 2)}
    assert len(s) == 2
    assert (1, 2) in s
    assert (3, 4) in s
    assert (5, 6) not in s


def test_tuple_mutable_contents() -> None:
    """Mutable objects inside tuple can be modified."""
    inner: list[int] = [1, 2, 3]
    t: tuple[list[int], int] = (inner, 42)
    assert t[0] == [1, 2, 3]
    inner.append(4)
    assert t[0] == [1, 2, 3, 4]
    t[0].append(5)
    assert t[0] == [1, 2, 3, 4, 5]
    assert inner == [1, 2, 3, 4, 5]


def test_tuple_augmented_assignment() -> None:
    """Augmented assignment creates new tuple."""
    t1: tuple[int, int] = (1, 2)
    t2: tuple[int, int] = t1
    t1 += (3, 4)
    assert t1 == (1, 2, 3, 4)
    assert t2 == (1, 2)


def test_tuple_multiply_augmented() -> None:
    """Augmented multiplication creates new tuple."""
    t1: tuple[int, int] = (1, 2)
    t2: tuple[int, int] = t1
    t1 *= 2
    assert t1 == (1, 2, 1, 2)
    assert t2 == (1, 2)


def test_tuple_comparison_first_diff() -> None:
    """Comparison stops at first difference."""
    assert (1, 100) < (2, 0)
    assert (1, 100, 100) < (2,)
    assert (5, 1) > (4, 999)
    assert (0, 0, 0, 1) > (0, 0, 0, 0)


def test_tuple_comparison_prefix() -> None:
    """Shorter tuple is less if it's a prefix."""
    assert (1, 2) < (1, 2, 3)
    assert (1, 2) < (1, 2, 0)
    assert (1, 2) < (1, 2, -100)
    assert () < (0,)
    assert () < (-1,)


def test_tuple_equal_elements() -> None:
    """Tuple with all same elements."""
    t: tuple[int, ...] = (5, 5, 5, 5, 5)
    assert len(t) == 5
    assert t.count(5) == 5
    assert min(t) == 5
    assert max(t) == 5
    assert sum(t) == 25
    assert t[0] == t[-1]


def test_tuple_generator_expression() -> None:
    """tuple() from generator expression."""
    t: tuple[int, ...] = tuple(x * x for x in range(5))
    assert t == (0, 1, 4, 9, 16)
    t2: tuple[int, ...] = tuple(x for x in [1, 2, 3] if x > 1)
    assert t2 == (2, 3)


def test_tuple_empty_variations() -> None:
    """Various ways to create empty tuple."""
    t1: tuple[int, ...] = ()
    t2: tuple[int, ...] = tuple()
    t3: tuple[int, ...] = tuple([])
    assert t1 == t2
    assert t2 == t3
    assert len(t1) == 0
    assert not t1


def test_tuple_single_vs_parens() -> None:
    """Single element: comma makes tuple, parens alone don't."""
    t: tuple[int] = (1,)
    n: int = (1)
    assert isinstance(t, tuple)
    assert isinstance(n, int)
    assert len(t) == 1
    assert t[0] == n
    assert t != n


def test_tuple_identity_vs_equality() -> None:
    """Same value tuples may be different objects."""
    t1: tuple[int, int, int] = (1, 2, 3)
    t2: tuple[int, int, int] = (1, 2, 3)
    t3: tuple[int, int, int] = t1
    assert t1 == t2
    assert t1 == t3
    t1 += (4,)
    assert t1 == (1, 2, 3, 4)
    assert t3 == (1, 2, 3)


def test_tuple_nested_empty() -> None:
    """Nested empty tuples."""
    t: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    assert len(t) == 2
    assert t[0] == ()
    assert t[1] == ()
    assert len(t[0]) == 0


def test_tuple_slice_creates_copy() -> None:
    """Slicing creates a new tuple."""
    t1: tuple[int, int, int] = (1, 2, 3)
    t2: tuple[int, ...] = t1[:]
    assert t1 == t2
    t3: tuple[int, ...] = t1[0:2]
    assert t3 == (1, 2)


def test_tuple_concat_empty() -> None:
    """Concatenating with empty tuple."""
    t: tuple[int, int, int] = (1, 2, 3)
    assert t + () == t
    assert () + t == t
    assert t + () == (1, 2, 3)


def test_tuple_multiply_zero_one() -> None:
    """Multiplication edge cases."""
    t: tuple[int, int] = (1, 2)
    assert t * 0 == ()
    assert t * 1 == (1, 2)
    assert t * 1 == t
    assert () * 100 == ()


def test_tuple_index_start_stop() -> None:
    """index() with start and stop arguments."""
    t: tuple[int, ...] = (1, 2, 3, 2, 4, 2, 5)
    assert t.index(2) == 1
    assert t.index(2, 2) == 3
    assert t.index(2, 4) == 5
    assert t.index(2, 2, 4) == 3


def test_tuple_count_none() -> None:
    """count() with None elements."""
    t: tuple[int | None, ...] = (1, None, 2, None, None, 3)
    assert t.count(None) == 3
    assert t.count(1) == 1
    assert t.count(4) == 0


def test_tuple_negative_index_slice() -> None:
    """Negative indices in slicing."""
    t: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    assert t[-3:] == (3, 4, 5)
    assert t[:-3] == (0, 1, 2)
    assert t[-4:-1] == (2, 3, 4)
    assert t[-1:-4:-1] == (5, 4, 3)


def test_tuple_bool_single_falsy() -> None:
    """Tuple with single falsy element is truthy."""
    assert bool((0,)) == True
    assert bool(("",)) == True
    assert bool((False,)) == True
    assert bool((None,)) == True
    assert (0,)
    assert (False,)


def test_tuple_comparison_heterogeneous_equality() -> None:
    """Equality comparison works with heterogeneous tuples."""
    t1: tuple[int, str, float] = (1, "a", 2.5)
    t2: tuple[int, str, float] = (1, "a", 2.5)
    t3: tuple[int, str, float] = (1, "b", 2.5)
    assert t1 == t2
    assert t1 != t3


def test_tuple_unpack_ignore() -> None:
    """Unpacking with _ to ignore values."""
    first, last = make_pair_for_ignore()
    assert first == 1
    assert last == 4


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_tuple_equality", test_tuple_equality),
        ("test_tuple_ordering", test_tuple_ordering),
        ("test_tuple_length", test_tuple_length),
        ("test_tuple_indexing", test_tuple_indexing),
        ("test_tuple_slicing", test_tuple_slicing),
        ("test_tuple_slice_step", test_tuple_slice_step),
        ("test_tuple_concatenation", test_tuple_concatenation),
        ("test_tuple_repetition", test_tuple_repetition),
        ("test_tuple_repetition_negative", test_tuple_repetition_negative),
        ("test_tuple_contains", test_tuple_contains),
        ("test_tuple_contains_empty", test_tuple_contains_empty),
        ("test_tuple_bool", test_tuple_bool),
        ("test_tuple_count", test_tuple_count),
        ("test_tuple_index", test_tuple_index),
        ("test_tuple_iteration", test_tuple_iteration),
        ("test_tuple_iteration_empty", test_tuple_iteration_empty),
        ("test_tuple_enumerate", test_tuple_enumerate),
        ("test_tuple_unpacking", test_tuple_unpacking),
        ("test_tuple_unpacking_two", test_tuple_unpacking_two),
        ("test_tuple_unpacking_nested", test_tuple_unpacking_nested),
        ("test_tuple_swap", test_tuple_swap),
        ("test_tuple_sum", test_tuple_sum),
        ("test_tuple_min_max", test_tuple_min_max),
        ("test_tuple_sorted", test_tuple_sorted),
        ("test_tuple_sorted_reverse", test_tuple_sorted_reverse),
        ("test_tuple_all_any", test_tuple_all_any),
        ("test_tuple_all_any_truthy", test_tuple_all_any_truthy),
        ("test_tuple_heterogeneous", test_tuple_heterogeneous),
        ("test_tuple_nested", test_tuple_nested),
        ("test_tuple_as_dict_key", test_tuple_as_dict_key),
        ("test_tuple_from_list", test_tuple_from_list),
        ("test_tuple_from_string", test_tuple_from_string),
        ("test_tuple_from_range", test_tuple_from_range),
        ("test_tuple_from_set", test_tuple_from_set),
        ("test_tuple_single_element", test_tuple_single_element),
        ("test_tuple_immutable", test_tuple_immutable),
        ("test_tuple_zip", test_tuple_zip),
        ("test_tuple_zip_unequal", test_tuple_zip_unequal),
        ("test_tuple_comparison_mixed_length", test_tuple_comparison_mixed_length),
        ("test_tuple_comparison_empty", test_tuple_comparison_empty),
        ("test_tuple_with_none", test_tuple_with_none),
        ("test_tuple_return_multiple", test_tuple_return_multiple),
        ("test_tuple_divmod", test_tuple_divmod),
        ("test_tuple_enumerate_unpack", test_tuple_enumerate_unpack),
        ("test_tuple_items_unpack", test_tuple_items_unpack),
        ("test_tuple_packing_no_parens", test_tuple_packing_no_parens),
        ("test_tuple_trailing_comma", test_tuple_trailing_comma),
        ("test_tuple_in_set", test_tuple_in_set),
        ("test_tuple_mutable_contents", test_tuple_mutable_contents),
        ("test_tuple_augmented_assignment", test_tuple_augmented_assignment),
        ("test_tuple_multiply_augmented", test_tuple_multiply_augmented),
        ("test_tuple_comparison_first_diff", test_tuple_comparison_first_diff),
        ("test_tuple_comparison_prefix", test_tuple_comparison_prefix),
        ("test_tuple_equal_elements", test_tuple_equal_elements),
        ("test_tuple_generator_expression", test_tuple_generator_expression),
        ("test_tuple_empty_variations", test_tuple_empty_variations),
        ("test_tuple_single_vs_parens", test_tuple_single_vs_parens),
        ("test_tuple_identity_vs_equality", test_tuple_identity_vs_equality),
        ("test_tuple_nested_empty", test_tuple_nested_empty),
        ("test_tuple_slice_creates_copy", test_tuple_slice_creates_copy),
        ("test_tuple_concat_empty", test_tuple_concat_empty),
        ("test_tuple_multiply_zero_one", test_tuple_multiply_zero_one),
        ("test_tuple_index_start_stop", test_tuple_index_start_stop),
        ("test_tuple_count_none", test_tuple_count_none),
        ("test_tuple_negative_index_slice", test_tuple_negative_index_slice),
        ("test_tuple_bool_single_falsy", test_tuple_bool_single_falsy),
        ("test_tuple_comparison_heterogeneous_equality", test_tuple_comparison_heterogeneous_equality),
        ("test_tuple_unpack_ignore", test_tuple_unpack_ignore),
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
