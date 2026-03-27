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


def test_tuple_concatenation() -> None:
    """Tuple concatenation with +."""
    assert (1, 2) + (3, 4) == (1, 2, 3, 4)
    assert () + (1,) == (1,)
    assert (1,) + () == (1,)
    assert () + () == ()
    assert (1,) + (2,) + (3,) == (1, 2, 3)


def test_tuple_contains() -> None:
    """Membership testing with in."""
    t: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5)
    assert 1 in t
    assert 3 in t
    assert 5 in t
    assert 0 not in t
    assert 6 not in t
    assert 0 not in t
    assert 6 not in t


def test_tuple_bool() -> None:
    """Tuple truthiness - empty is falsy."""
    assert bool((1, 2, 3)) == True
    assert bool((0,)) == True
    assert bool(()) == False
    assert not ()
    assert (1,)


def test_tuple_iteration() -> None:
    """Iterating over tuple."""
    t: tuple[int, int, int] = (1, 2, 3)
    result: list[int] = []
    for x in t:
        result.append(x)
    assert result == [1, 2, 3]


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


def test_tuple_single_element() -> None:
    """Single-element tuple requires trailing comma."""
    t: tuple[int] = (42,)
    assert len(t) == 1
    assert t[0] == 42
    # Without comma, it's just grouping
    n: int = 42
    assert n == 42


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
    t1: tuple[int, int, int] = (
        1,
        2,
        3,
    )
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


def test_tuple_single_vs_parens() -> None:
    """Single element: comma makes tuple, parens alone don't."""
    t: tuple[int] = (1,)
    n: int = 1
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


def test_tuple_concat_empty() -> None:
    """Concatenating with empty tuple."""
    t: tuple[int, int, int] = (1, 2, 3)
    assert t + () == t
    assert () + t == t
    assert t + () == (1, 2, 3)


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


def test_tuple_unpack_discard() -> None:
    """Unpacking with _ discard variable."""
    x, _ = make_pair_for_ignore()
    assert x == 1
    _, y = make_pair_for_ignore()
    assert y == 4


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_tuple_equality()
        passed += 1
        print("  PASS test_tuple_equality")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_equality: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_equality: " + str(e))
    try:
        test_tuple_ordering()
        passed += 1
        print("  PASS test_tuple_ordering")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_ordering: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_ordering: " + str(e))
    try:
        test_tuple_length()
        passed += 1
        print("  PASS test_tuple_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_length: " + str(e))
    try:
        test_tuple_indexing()
        passed += 1
        print("  PASS test_tuple_indexing")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_indexing: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_indexing: " + str(e))
    try:
        test_tuple_concatenation()
        passed += 1
        print("  PASS test_tuple_concatenation")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_concatenation: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_concatenation: " + str(e))
    try:
        test_tuple_contains()
        passed += 1
        print("  PASS test_tuple_contains")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_contains: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_contains: " + str(e))
    try:
        test_tuple_bool()
        passed += 1
        print("  PASS test_tuple_bool")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_bool: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_bool: " + str(e))
    try:
        test_tuple_iteration()
        passed += 1
        print("  PASS test_tuple_iteration")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_iteration: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_iteration: " + str(e))
    try:
        test_tuple_enumerate()
        passed += 1
        print("  PASS test_tuple_enumerate")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_enumerate: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_enumerate: " + str(e))
    try:
        test_tuple_unpacking()
        passed += 1
        print("  PASS test_tuple_unpacking")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_unpacking: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_unpacking: " + str(e))
    try:
        test_tuple_unpacking_two()
        passed += 1
        print("  PASS test_tuple_unpacking_two")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_unpacking_two: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_unpacking_two: " + str(e))
    try:
        test_tuple_unpacking_nested()
        passed += 1
        print("  PASS test_tuple_unpacking_nested")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_unpacking_nested: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_unpacking_nested: " + str(e))
    try:
        test_tuple_swap()
        passed += 1
        print("  PASS test_tuple_swap")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_swap: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_swap: " + str(e))
    try:
        test_tuple_sum()
        passed += 1
        print("  PASS test_tuple_sum")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_sum: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_sum: " + str(e))
    try:
        test_tuple_all_any()
        passed += 1
        print("  PASS test_tuple_all_any")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_all_any: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_all_any: " + str(e))
    try:
        test_tuple_all_any_truthy()
        passed += 1
        print("  PASS test_tuple_all_any_truthy")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_all_any_truthy: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_all_any_truthy: " + str(e))
    try:
        test_tuple_heterogeneous()
        passed += 1
        print("  PASS test_tuple_heterogeneous")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_heterogeneous: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_heterogeneous: " + str(e))
    try:
        test_tuple_nested()
        passed += 1
        print("  PASS test_tuple_nested")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_nested: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_nested: " + str(e))
    try:
        test_tuple_as_dict_key()
        passed += 1
        print("  PASS test_tuple_as_dict_key")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_as_dict_key: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_as_dict_key: " + str(e))
    try:
        test_tuple_single_element()
        passed += 1
        print("  PASS test_tuple_single_element")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_single_element: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_single_element: " + str(e))
    try:
        test_tuple_zip()
        passed += 1
        print("  PASS test_tuple_zip")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_zip: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_zip: " + str(e))
    try:
        test_tuple_zip_unequal()
        passed += 1
        print("  PASS test_tuple_zip_unequal")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_zip_unequal: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_zip_unequal: " + str(e))
    try:
        test_tuple_comparison_mixed_length()
        passed += 1
        print("  PASS test_tuple_comparison_mixed_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_comparison_mixed_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_comparison_mixed_length: " + str(e))
    try:
        test_tuple_comparison_empty()
        passed += 1
        print("  PASS test_tuple_comparison_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_comparison_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_comparison_empty: " + str(e))
    try:
        test_tuple_with_none()
        passed += 1
        print("  PASS test_tuple_with_none")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_with_none: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_with_none: " + str(e))
    try:
        test_tuple_return_multiple()
        passed += 1
        print("  PASS test_tuple_return_multiple")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_return_multiple: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_return_multiple: " + str(e))
    try:
        test_tuple_divmod()
        passed += 1
        print("  PASS test_tuple_divmod")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_divmod: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_divmod: " + str(e))
    try:
        test_tuple_enumerate_unpack()
        passed += 1
        print("  PASS test_tuple_enumerate_unpack")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_enumerate_unpack: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_enumerate_unpack: " + str(e))
    try:
        test_tuple_items_unpack()
        passed += 1
        print("  PASS test_tuple_items_unpack")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_items_unpack: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_items_unpack: " + str(e))
    try:
        test_tuple_packing_no_parens()
        passed += 1
        print("  PASS test_tuple_packing_no_parens")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_packing_no_parens: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_packing_no_parens: " + str(e))
    try:
        test_tuple_trailing_comma()
        passed += 1
        print("  PASS test_tuple_trailing_comma")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_trailing_comma: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_trailing_comma: " + str(e))
    try:
        test_tuple_in_set()
        passed += 1
        print("  PASS test_tuple_in_set")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_in_set: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_in_set: " + str(e))
    try:
        test_tuple_mutable_contents()
        passed += 1
        print("  PASS test_tuple_mutable_contents")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_mutable_contents: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_mutable_contents: " + str(e))
    try:
        test_tuple_augmented_assignment()
        passed += 1
        print("  PASS test_tuple_augmented_assignment")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_augmented_assignment: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_augmented_assignment: " + str(e))
    try:
        test_tuple_multiply_augmented()
        passed += 1
        print("  PASS test_tuple_multiply_augmented")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_multiply_augmented: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_multiply_augmented: " + str(e))
    try:
        test_tuple_comparison_first_diff()
        passed += 1
        print("  PASS test_tuple_comparison_first_diff")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_comparison_first_diff: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_comparison_first_diff: " + str(e))
    try:
        test_tuple_comparison_prefix()
        passed += 1
        print("  PASS test_tuple_comparison_prefix")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_comparison_prefix: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_comparison_prefix: " + str(e))
    try:
        test_tuple_single_vs_parens()
        passed += 1
        print("  PASS test_tuple_single_vs_parens")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_single_vs_parens: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_single_vs_parens: " + str(e))
    try:
        test_tuple_identity_vs_equality()
        passed += 1
        print("  PASS test_tuple_identity_vs_equality")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_identity_vs_equality: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_identity_vs_equality: " + str(e))
    try:
        test_tuple_concat_empty()
        passed += 1
        print("  PASS test_tuple_concat_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_concat_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_concat_empty: " + str(e))
    try:
        test_tuple_bool_single_falsy()
        passed += 1
        print("  PASS test_tuple_bool_single_falsy")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_bool_single_falsy: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_bool_single_falsy: " + str(e))
    try:
        test_tuple_comparison_heterogeneous_equality()
        passed += 1
        print("  PASS test_tuple_comparison_heterogeneous_equality")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_comparison_heterogeneous_equality: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_comparison_heterogeneous_equality: " + str(e))
    try:
        test_tuple_unpack_ignore()
        passed += 1
        print("  PASS test_tuple_unpack_ignore")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_unpack_ignore: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_unpack_ignore: " + str(e))
    try:
        test_tuple_unpack_discard()
        passed += 1
        print("  PASS test_tuple_unpack_discard")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tuple_unpack_discard: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tuple_unpack_discard: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
