"""Set object tests."""

import sys


def test_set_equality() -> None:
    """Set equality comparisons."""
    assert {1, 2, 3} == {1, 2, 3}
    assert {1, 2, 3} == {3, 2, 1}
    assert {1, 2} != {1, 3}
    assert {1, 2} != {1, 2, 3}
    assert set() == set()


def test_set_length() -> None:
    """len() returns element count."""
    assert len(set()) == 0
    assert len({1}) == 1
    assert len({1, 2, 3}) == 3
    assert len({1, 2, 3, 4, 5}) == 5


def test_set_no_duplicates() -> None:
    """Sets eliminate duplicates."""
    s: set[int] = {1, 1, 2, 2, 3, 3}
    assert len(s) == 3
    assert s == {1, 2, 3}


def test_set_contains() -> None:
    """Membership testing with in."""
    s: set[int] = {1, 2, 3, 4, 5}
    assert 1 in s
    assert 3 in s
    assert 5 in s
    assert 0 not in s
    assert 6 not in s
    assert 0 not in s
    assert 6 not in s


def test_set_contains_empty() -> None:
    """Membership in empty set."""
    s: set[int] = set()
    assert 1 not in s
    assert 1 not in s


def test_set_bool() -> None:
    """Set truthiness - empty is falsy."""
    assert bool({1, 2, 3}) == True
    assert bool({0}) == True
    assert bool(set()) == False
    assert not set()
    assert {1}


def test_set_add() -> None:
    """add() inserts element."""
    s: set[int] = {1, 2}
    s.add(3)
    assert s == {1, 2, 3}
    s.add(4)
    assert s == {1, 2, 3, 4}
    # Adding existing element has no effect
    s.add(2)
    assert s == {1, 2, 3, 4}
    assert len(s) == 4


def test_set_add_empty() -> None:
    """add() to empty set."""
    s: set[int] = set()
    s.add(1)
    assert s == {1}
    s.add(2)
    assert s == {1, 2}


def test_set_remove() -> None:
    """remove() deletes element."""
    s: set[int] = {1, 2, 3, 4, 5}
    s.remove(3)
    assert s == {1, 2, 4, 5}
    s.remove(1)
    assert s == {2, 4, 5}
    s.remove(5)
    assert s == {2, 4}


def test_set_discard() -> None:
    """discard() removes if present, no error if absent."""
    s: set[int] = {1, 2, 3}
    s.discard(2)
    assert s == {1, 3}
    # Discarding non-existent element is fine
    s.discard(99)
    assert s == {1, 3}
    s.discard(1)
    s.discard(3)
    assert s == set()


def test_set_pop() -> None:
    """pop() removes and returns arbitrary element."""
    s: set[int] = {1, 2, 3}
    x: int = s.pop()
    assert x in {1, 2, 3}
    assert len(s) == 2
    assert x not in s
    y: int = s.pop()
    assert len(s) == 1
    z: int = s.pop()
    assert len(s) == 0


def test_set_clear() -> None:
    """clear() removes all elements."""
    s: set[int] = {1, 2, 3}
    s.clear()
    assert s == set()
    assert len(s) == 0
    s.clear()
    assert s == set()


def test_set_copy() -> None:
    """copy() creates shallow copy."""
    original: set[int] = {1, 2, 3}
    copied: set[int] = original.copy()
    assert copied == original
    copied.add(4)
    assert copied == {1, 2, 3, 4}
    assert original == {1, 2, 3}
    original.add(5)
    assert original == {1, 2, 3, 5}
    assert copied == {1, 2, 3, 4}


def test_set_union() -> None:
    """union() returns combined set."""
    a: set[int] = {1, 2, 3}
    b: set[int] = {3, 4, 5}
    c: set[int] = a.union(b)
    assert c == {1, 2, 3, 4, 5}
    # Originals unchanged
    assert a == {1, 2, 3}
    assert b == {3, 4, 5}


def test_set_union_empty() -> None:
    """union() with empty sets."""
    a: set[int] = {1, 2, 3}
    b: set[int] = set()
    assert a.union(b) == {1, 2, 3}
    assert b.union(a) == {1, 2, 3}
    assert b.union(b) == set()


def test_set_union_operator() -> None:
    """Union with | operator."""
    a: set[int] = {1, 2, 3}
    b: set[int] = {3, 4, 5}
    assert a | b == {1, 2, 3, 4, 5}
    assert b | a == {1, 2, 3, 4, 5}


def test_set_intersection() -> None:
    """intersection() returns common elements."""
    a: set[int] = {1, 2, 3, 4}
    b: set[int] = {3, 4, 5, 6}
    c: set[int] = a.intersection(b)
    assert c == {3, 4}
    assert a == {1, 2, 3, 4}
    assert b == {3, 4, 5, 6}


def test_set_intersection_empty() -> None:
    """intersection() with empty or disjoint sets."""
    a: set[int] = {1, 2, 3}
    b: set[int] = set()
    assert a.intersection(b) == set()
    assert b.intersection(a) == set()
    c: set[int] = {4, 5, 6}
    assert a.intersection(c) == set()


def test_set_intersection_operator() -> None:
    """Intersection with & operator."""
    a: set[int] = {1, 2, 3, 4}
    b: set[int] = {3, 4, 5, 6}
    assert a & b == {3, 4}
    assert b & a == {3, 4}


def test_set_difference() -> None:
    """difference() returns elements in self but not other."""
    a: set[int] = {1, 2, 3, 4}
    b: set[int] = {3, 4, 5, 6}
    assert a.difference(b) == {1, 2}
    assert b.difference(a) == {5, 6}


def test_set_difference_empty() -> None:
    """difference() with empty sets."""
    a: set[int] = {1, 2, 3}
    b: set[int] = set()
    assert a.difference(b) == {1, 2, 3}
    assert b.difference(a) == set()
    assert a.difference(a) == set()


def test_set_difference_operator() -> None:
    """Difference with - operator."""
    a: set[int] = {1, 2, 3, 4}
    b: set[int] = {3, 4, 5, 6}
    assert a - b == {1, 2}
    assert b - a == {5, 6}


def test_set_symmetric_difference() -> None:
    """Symmetric difference with ^ operator."""
    a: set[int] = {1, 2, 3, 4}
    b: set[int] = {3, 4, 5, 6}
    assert a ^ b == {1, 2, 5, 6}
    assert b ^ a == {1, 2, 5, 6}


def test_set_symmetric_difference_empty() -> None:
    """Symmetric difference edge cases."""
    a: set[int] = {1, 2, 3}
    b: set[int] = set()
    assert a ^ b == {1, 2, 3}
    assert b ^ a == {1, 2, 3}
    assert a ^ a == set()


def test_set_issubset() -> None:
    """issubset() checks if all elements in other."""
    a: set[int] = {1, 2}
    b: set[int] = {1, 2, 3, 4}
    assert a.issubset(b) == True
    assert b.issubset(a) == False
    assert a.issubset(a) == True
    c: set[int] = {1, 2}
    assert a.issubset(c) == True


def test_set_issubset_operator() -> None:
    """Subset with <= operator."""
    a: set[int] = {1, 2}
    b: set[int] = {1, 2, 3, 4}
    assert a <= b
    assert not (b <= a)
    assert a <= a


def test_set_proper_subset() -> None:
    """Proper subset with < operator."""
    a: set[int] = {1, 2}
    b: set[int] = {1, 2, 3, 4}
    assert a < b
    assert not (b < a)
    assert not (a < a)


def test_set_issuperset() -> None:
    """issuperset() checks if contains all elements of other."""
    a: set[int] = {1, 2, 3, 4}
    b: set[int] = {1, 2}
    assert a.issuperset(b) == True
    assert b.issuperset(a) == False
    assert a.issuperset(a) == True


def test_set_issuperset_operator() -> None:
    """Superset with >= operator."""
    a: set[int] = {1, 2, 3, 4}
    b: set[int] = {1, 2}
    assert a >= b
    assert not (b >= a)
    assert a >= a


def test_set_proper_superset() -> None:
    """Proper superset with > operator."""
    a: set[int] = {1, 2, 3, 4}
    b: set[int] = {1, 2}
    assert a > b
    assert not (b > a)
    assert not (a > a)


def test_set_disjoint() -> None:
    """isdisjoint() checks no common elements."""
    a: set[int] = {1, 2, 3}
    b: set[int] = {4, 5, 6}
    assert a.isdisjoint(b) == True
    assert b.isdisjoint(a) == True
    c: set[int] = {3, 4, 5}
    assert a.isdisjoint(c) == False


def test_set_disjoint_empty() -> None:
    """Empty set is disjoint with everything."""
    a: set[int] = {1, 2, 3}
    b: set[int] = set()
    assert a.isdisjoint(b) == True
    assert b.isdisjoint(a) == True
    assert b.isdisjoint(b) == True


def test_set_iteration() -> None:
    """Iterating over set."""
    s: set[int] = {1, 2, 3}
    result: list[int] = []
    for x in s:
        result.append(x)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_set_iteration_empty() -> None:
    """Iterating over empty set."""
    s: set[int] = set()
    count: int = 0
    for x in s:
        count = count + 1
    assert count == 0


def test_set_comprehension() -> None:
    """Set comprehension."""
    squares: set[int] = {x * x for x in [1, 2, 3, 4, 5]}
    assert squares == {1, 4, 9, 16, 25}


def test_set_comprehension_condition() -> None:
    """Set comprehension with condition."""
    evens: set[int] = {x for x in [1, 2, 3, 4, 5, 6] if x % 2 == 0}
    assert evens == {2, 4, 6}


def test_set_comprehension_dedup() -> None:
    """Set comprehension eliminates duplicates."""
    s: set[int] = {x % 3 for x in [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    assert s == {0, 1, 2}


def test_set_from_list() -> None:
    """set() from list removes duplicates."""
    items: list[int] = [1, 2, 2, 3, 3, 3, 4]
    s: set[int] = set(items)
    assert s == {1, 2, 3, 4}


def test_set_from_string() -> None:
    """set() from string."""
    s: set[str] = set("hello")
    assert s == {"h", "e", "l", "o"}
    assert len(s) == 4


def test_set_from_range() -> None:
    """set() from range."""
    s: set[int] = set(range(5))
    assert s == {0, 1, 2, 3, 4}


def test_set_string_elements() -> None:
    """Set of strings."""
    s: set[str] = {"apple", "banana", "cherry"}
    assert "apple" in s
    assert "grape" not in s
    assert len(s) == 3
    s.add("date")
    assert s == {"apple", "banana", "cherry", "date"}


def test_set_min_max() -> None:
    """min() and max() of set."""
    s: set[int] = {3, 1, 4, 1, 5, 9, 2, 6}
    assert min(s) == 1
    assert max(s) == 9


def test_set_sum() -> None:
    """sum() of set."""
    s: set[int] = {1, 2, 3, 4, 5}
    assert sum(s) == 15


def test_set_sorted() -> None:
    """sorted() on set."""
    s: set[int] = {3, 1, 4, 1, 5, 9, 2, 6}
    result: list[int] = sorted(s)
    assert result == [1, 2, 3, 4, 5, 6, 9]


def test_set_len_after_operations() -> None:
    """Length tracking through operations."""
    s: set[int] = set()
    assert len(s) == 0
    s.add(1)
    assert len(s) == 1
    s.add(2)
    assert len(s) == 2
    s.add(2)
    assert len(s) == 2
    s.remove(1)
    assert len(s) == 1
    s.clear()
    assert len(s) == 0


def test_set_update() -> None:
    """Update set with |= operator."""
    s: set[int] = {1, 2, 3}
    s |= {3, 4, 5}
    assert s == {1, 2, 3, 4, 5}


def test_set_intersection_update() -> None:
    """Intersection update with &= operator."""
    s: set[int] = {1, 2, 3, 4}
    s &= {2, 3, 4, 5}
    assert s == {2, 3, 4}


def test_set_difference_update() -> None:
    """Difference update with -= operator."""
    s: set[int] = {1, 2, 3, 4, 5}
    s -= {2, 4}
    assert s == {1, 3, 5}


def test_set_symmetric_difference_update() -> None:
    """Symmetric difference update with ^= operator."""
    s: set[int] = {1, 2, 3, 4}
    s ^= {3, 4, 5, 6}
    assert s == {1, 2, 5, 6}


def test_set_identity() -> None:
    """Set identity vs equality."""
    a: set[int] = {1, 2, 3}
    b: set[int] = {1, 2, 3}
    c: set[int] = a
    assert a == b
    assert a == c
    a.add(4)
    assert c == {1, 2, 3, 4}
    assert b == {1, 2, 3}


def test_set_all_any() -> None:
    """all() and any() on sets."""
    assert all({True, True, True}) == True
    assert all({True, False, True}) == False
    assert any({False, False, True}) == True
    assert any({False, False, False}) == False


def test_set_all_any_truthy() -> None:
    """all() and any() use truthiness."""
    assert all({1, 2, 3}) == True
    assert all({1, 0, 3}) == False
    assert any({0, 0, 1}) == True
    assert any({0}) == False


def test_empty_set_literal_gotcha() -> None:
    """Empty {} creates dict, not set."""
    empty_dict: dict[str, int] = {}
    empty_set: set[int] = set()
    assert isinstance(empty_dict, dict)
    assert isinstance(empty_set, set)
    assert empty_dict != empty_set
    assert len(empty_dict) == 0
    assert len(empty_set) == 0


def test_set_update_method() -> None:
    """update() method adds from iterable."""
    s: set[int] = {1, 2}
    s.update([3, 4])
    assert s == {1, 2, 3, 4}
    s.update([4, 5], [6, 7])
    assert s == {1, 2, 3, 4, 5, 6, 7}
    s.update([])
    assert s == {1, 2, 3, 4, 5, 6, 7}


def test_set_update_with_string() -> None:
    """update() with string adds each character."""
    s: set[str] = {"a", "b"}
    s.update("cd")
    assert s == {"a", "b", "c", "d"}
    s.update("ab")
    assert s == {"a", "b", "c", "d"}


def test_set_add_vs_update() -> None:
    """add() takes single element, update() takes iterable."""
    s1: set[str] = set()
    s1.add("hello")
    assert s1 == {"hello"}
    s2: set[str] = set()
    s2.update("hello")
    assert s2 == {"h", "e", "l", "o"}


def test_set_subset_with_self() -> None:
    """Set is subset of itself but not proper subset."""
    s: set[int] = {1, 2, 3}
    assert s <= s
    assert s.issubset(s)
    assert not (s < s)
    assert s >= s
    assert s.issuperset(s)
    assert not (s > s)


def test_set_empty_subset_of_all() -> None:
    """Empty set is subset of every set."""
    empty: set[int] = set()
    s: set[int] = {1, 2, 3}
    assert empty <= s
    assert empty.issubset(s)
    assert empty < s
    assert empty <= empty
    assert not (empty < empty)


def test_set_symmetric_difference_self() -> None:
    """Symmetric difference with self is empty."""
    s: set[int] = {1, 2, 3}
    assert s ^ s == set()
    assert s.symmetric_difference(s) == set()


def test_set_difference_multiple() -> None:
    """difference() with multiple arguments."""
    s: set[int] = {1, 2, 3, 4, 5}
    result: set[int] = s.difference({1, 2}, {4, 5})
    assert result == {3}


def test_set_intersection_multiple() -> None:
    """intersection() with multiple arguments."""
    a: set[int] = {1, 2, 3, 4, 5}
    b: set[int] = {2, 3, 4, 5, 6}
    c: set[int] = {3, 4, 5, 6, 7}
    result: set[int] = a.intersection(b, c)
    assert result == {3, 4, 5}


def test_set_union_multiple() -> None:
    """union() with multiple arguments."""
    a: set[int] = {1, 2}
    b: set[int] = {3, 4}
    c: set[int] = {5, 6}
    result: set[int] = a.union(b, c)
    assert result == {1, 2, 3, 4, 5, 6}


def test_set_pop_arbitrary() -> None:
    """pop() returns arbitrary element (not specified which)."""
    s: set[int] = {1}
    x: int = s.pop()
    assert x == 1
    assert s == set()


def test_set_pop_reduces_length() -> None:
    """Each pop() reduces length by 1."""
    s: set[int] = {1, 2, 3, 4, 5}
    for i in range(5):
        assert len(s) == 5 - i
        s.pop()
    assert len(s) == 0


def test_set_equal_different_order() -> None:
    """Sets with same elements are equal regardless of creation order."""
    s1: set[int] = set()
    s1.add(1)
    s1.add(2)
    s1.add(3)
    s2: set[int] = set()
    s2.add(3)
    s2.add(1)
    s2.add(2)
    assert s1 == s2


def test_set_disjoint_with_self() -> None:
    """Non-empty set is not disjoint with itself."""
    s: set[int] = {1, 2, 3}
    assert not s.isdisjoint(s)
    empty: set[int] = set()
    assert empty.isdisjoint(empty)


def test_set_single_element() -> None:
    """Single element set."""
    s: set[int] = {42}
    assert len(s) == 1
    assert 42 in s
    assert s == {42}
    assert min(s) == 42
    assert max(s) == 42
    assert sum(s) == 42


def test_set_bool_with_zero() -> None:
    """Set containing zero is truthy."""
    s: set[int] = {0}
    assert bool(s) == True
    assert s
    s.add(0)
    assert len(s) == 1


def test_set_update_with_dict() -> None:
    """update() with dict adds keys, not values."""
    s: set[str] = {"a"}
    d: dict[str, int] = {"b": 1, "c": 2}
    s.update(d)
    assert s == {"a", "b", "c"}


def test_set_remove_vs_discard() -> None:
    """remove() raises KeyError, discard() doesn't."""
    s: set[int] = {1, 2, 3}
    s.remove(1)
    assert s == {2, 3}
    s.discard(2)
    assert s == {3}
    s.discard(99)
    assert s == {3}


def test_set_clear_and_reuse() -> None:
    """Set can be reused after clear()."""
    s: set[int] = {1, 2, 3}
    s.clear()
    assert s == set()
    s.add(4)
    s.add(5)
    assert s == {4, 5}


def test_set_copy_independent() -> None:
    """Copied set is independent."""
    original: set[int] = {1, 2, 3}
    copied: set[int] = original.copy()
    original.clear()
    assert copied == {1, 2, 3}
    assert original == set()


def test_set_generator_expression() -> None:
    """set() from generator expression."""
    s: set[int] = set(x * x for x in range(5))
    assert s == {0, 1, 4, 9, 16}
    s2: set[int] = set(x for x in [1, 2, 2, 3, 3, 3])
    assert s2 == {1, 2, 3}


def test_set_tuple_elements() -> None:
    """Set of tuples."""
    s: set[tuple[int, int]] = {(1, 2), (3, 4), (1, 2)}
    assert len(s) == 2
    assert (1, 2) in s
    assert (3, 4) in s
    assert (5, 6) not in s


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_set_equality()
        passed += 1
        print("  PASS test_set_equality")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_equality: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_equality: " + str(e))
    try:
        test_set_length()
        passed += 1
        print("  PASS test_set_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_length: " + str(e))
    try:
        test_set_no_duplicates()
        passed += 1
        print("  PASS test_set_no_duplicates")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_no_duplicates: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_no_duplicates: " + str(e))
    try:
        test_set_contains()
        passed += 1
        print("  PASS test_set_contains")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_contains: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_contains: " + str(e))
    try:
        test_set_contains_empty()
        passed += 1
        print("  PASS test_set_contains_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_contains_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_contains_empty: " + str(e))
    try:
        test_set_bool()
        passed += 1
        print("  PASS test_set_bool")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_bool: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_bool: " + str(e))
    try:
        test_set_add()
        passed += 1
        print("  PASS test_set_add")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_add: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_add: " + str(e))
    try:
        test_set_add_empty()
        passed += 1
        print("  PASS test_set_add_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_add_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_add_empty: " + str(e))
    try:
        test_set_remove()
        passed += 1
        print("  PASS test_set_remove")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_remove: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_remove: " + str(e))
    try:
        test_set_discard()
        passed += 1
        print("  PASS test_set_discard")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_discard: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_discard: " + str(e))
    try:
        test_set_pop()
        passed += 1
        print("  PASS test_set_pop")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_pop: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_pop: " + str(e))
    try:
        test_set_clear()
        passed += 1
        print("  PASS test_set_clear")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_clear: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_clear: " + str(e))
    try:
        test_set_copy()
        passed += 1
        print("  PASS test_set_copy")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_copy: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_copy: " + str(e))
    try:
        test_set_union()
        passed += 1
        print("  PASS test_set_union")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_union: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_union: " + str(e))
    try:
        test_set_union_empty()
        passed += 1
        print("  PASS test_set_union_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_union_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_union_empty: " + str(e))
    try:
        test_set_union_operator()
        passed += 1
        print("  PASS test_set_union_operator")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_union_operator: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_union_operator: " + str(e))
    try:
        test_set_intersection()
        passed += 1
        print("  PASS test_set_intersection")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_intersection: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_intersection: " + str(e))
    try:
        test_set_intersection_empty()
        passed += 1
        print("  PASS test_set_intersection_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_intersection_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_intersection_empty: " + str(e))
    try:
        test_set_intersection_operator()
        passed += 1
        print("  PASS test_set_intersection_operator")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_intersection_operator: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_intersection_operator: " + str(e))
    try:
        test_set_difference()
        passed += 1
        print("  PASS test_set_difference")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_difference: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_difference: " + str(e))
    try:
        test_set_difference_empty()
        passed += 1
        print("  PASS test_set_difference_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_difference_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_difference_empty: " + str(e))
    try:
        test_set_difference_operator()
        passed += 1
        print("  PASS test_set_difference_operator")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_difference_operator: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_difference_operator: " + str(e))
    try:
        test_set_symmetric_difference()
        passed += 1
        print("  PASS test_set_symmetric_difference")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_symmetric_difference: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_symmetric_difference: " + str(e))
    try:
        test_set_symmetric_difference_empty()
        passed += 1
        print("  PASS test_set_symmetric_difference_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_symmetric_difference_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_symmetric_difference_empty: " + str(e))
    try:
        test_set_issubset()
        passed += 1
        print("  PASS test_set_issubset")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_issubset: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_issubset: " + str(e))
    try:
        test_set_issubset_operator()
        passed += 1
        print("  PASS test_set_issubset_operator")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_issubset_operator: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_issubset_operator: " + str(e))
    try:
        test_set_proper_subset()
        passed += 1
        print("  PASS test_set_proper_subset")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_proper_subset: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_proper_subset: " + str(e))
    try:
        test_set_issuperset()
        passed += 1
        print("  PASS test_set_issuperset")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_issuperset: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_issuperset: " + str(e))
    try:
        test_set_issuperset_operator()
        passed += 1
        print("  PASS test_set_issuperset_operator")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_issuperset_operator: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_issuperset_operator: " + str(e))
    try:
        test_set_proper_superset()
        passed += 1
        print("  PASS test_set_proper_superset")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_proper_superset: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_proper_superset: " + str(e))
    try:
        test_set_disjoint()
        passed += 1
        print("  PASS test_set_disjoint")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_disjoint: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_disjoint: " + str(e))
    try:
        test_set_disjoint_empty()
        passed += 1
        print("  PASS test_set_disjoint_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_disjoint_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_disjoint_empty: " + str(e))
    try:
        test_set_iteration()
        passed += 1
        print("  PASS test_set_iteration")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_iteration: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_iteration: " + str(e))
    try:
        test_set_iteration_empty()
        passed += 1
        print("  PASS test_set_iteration_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_iteration_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_iteration_empty: " + str(e))
    try:
        test_set_comprehension()
        passed += 1
        print("  PASS test_set_comprehension")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_comprehension: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_comprehension: " + str(e))
    try:
        test_set_comprehension_condition()
        passed += 1
        print("  PASS test_set_comprehension_condition")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_comprehension_condition: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_comprehension_condition: " + str(e))
    try:
        test_set_comprehension_dedup()
        passed += 1
        print("  PASS test_set_comprehension_dedup")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_comprehension_dedup: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_comprehension_dedup: " + str(e))
    try:
        test_set_from_list()
        passed += 1
        print("  PASS test_set_from_list")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_from_list: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_from_list: " + str(e))
    try:
        test_set_from_string()
        passed += 1
        print("  PASS test_set_from_string")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_from_string: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_from_string: " + str(e))
    try:
        test_set_from_range()
        passed += 1
        print("  PASS test_set_from_range")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_from_range: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_from_range: " + str(e))
    try:
        test_set_string_elements()
        passed += 1
        print("  PASS test_set_string_elements")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_string_elements: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_string_elements: " + str(e))
    try:
        test_set_min_max()
        passed += 1
        print("  PASS test_set_min_max")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_min_max: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_min_max: " + str(e))
    try:
        test_set_sum()
        passed += 1
        print("  PASS test_set_sum")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_sum: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_sum: " + str(e))
    try:
        test_set_sorted()
        passed += 1
        print("  PASS test_set_sorted")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_sorted: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_sorted: " + str(e))
    try:
        test_set_len_after_operations()
        passed += 1
        print("  PASS test_set_len_after_operations")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_len_after_operations: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_len_after_operations: " + str(e))
    try:
        test_set_update()
        passed += 1
        print("  PASS test_set_update")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_update: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_update: " + str(e))
    try:
        test_set_intersection_update()
        passed += 1
        print("  PASS test_set_intersection_update")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_intersection_update: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_intersection_update: " + str(e))
    try:
        test_set_difference_update()
        passed += 1
        print("  PASS test_set_difference_update")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_difference_update: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_difference_update: " + str(e))
    try:
        test_set_symmetric_difference_update()
        passed += 1
        print("  PASS test_set_symmetric_difference_update")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_symmetric_difference_update: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_symmetric_difference_update: " + str(e))
    try:
        test_set_identity()
        passed += 1
        print("  PASS test_set_identity")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_identity: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_identity: " + str(e))
    try:
        test_set_all_any()
        passed += 1
        print("  PASS test_set_all_any")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_all_any: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_all_any: " + str(e))
    try:
        test_set_all_any_truthy()
        passed += 1
        print("  PASS test_set_all_any_truthy")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_all_any_truthy: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_all_any_truthy: " + str(e))
    try:
        test_empty_set_literal_gotcha()
        passed += 1
        print("  PASS test_empty_set_literal_gotcha")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_empty_set_literal_gotcha: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_empty_set_literal_gotcha: " + str(e))
    try:
        test_set_update_method()
        passed += 1
        print("  PASS test_set_update_method")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_update_method: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_update_method: " + str(e))
    try:
        test_set_update_with_string()
        passed += 1
        print("  PASS test_set_update_with_string")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_update_with_string: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_update_with_string: " + str(e))
    try:
        test_set_add_vs_update()
        passed += 1
        print("  PASS test_set_add_vs_update")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_add_vs_update: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_add_vs_update: " + str(e))
    try:
        test_set_subset_with_self()
        passed += 1
        print("  PASS test_set_subset_with_self")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_subset_with_self: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_subset_with_self: " + str(e))
    try:
        test_set_empty_subset_of_all()
        passed += 1
        print("  PASS test_set_empty_subset_of_all")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_empty_subset_of_all: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_empty_subset_of_all: " + str(e))
    try:
        test_set_symmetric_difference_self()
        passed += 1
        print("  PASS test_set_symmetric_difference_self")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_symmetric_difference_self: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_symmetric_difference_self: " + str(e))
    try:
        test_set_difference_multiple()
        passed += 1
        print("  PASS test_set_difference_multiple")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_difference_multiple: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_difference_multiple: " + str(e))
    try:
        test_set_intersection_multiple()
        passed += 1
        print("  PASS test_set_intersection_multiple")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_intersection_multiple: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_intersection_multiple: " + str(e))
    try:
        test_set_union_multiple()
        passed += 1
        print("  PASS test_set_union_multiple")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_union_multiple: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_union_multiple: " + str(e))
    try:
        test_set_pop_arbitrary()
        passed += 1
        print("  PASS test_set_pop_arbitrary")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_pop_arbitrary: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_pop_arbitrary: " + str(e))
    try:
        test_set_pop_reduces_length()
        passed += 1
        print("  PASS test_set_pop_reduces_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_pop_reduces_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_pop_reduces_length: " + str(e))
    try:
        test_set_equal_different_order()
        passed += 1
        print("  PASS test_set_equal_different_order")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_equal_different_order: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_equal_different_order: " + str(e))
    try:
        test_set_disjoint_with_self()
        passed += 1
        print("  PASS test_set_disjoint_with_self")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_disjoint_with_self: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_disjoint_with_self: " + str(e))
    try:
        test_set_single_element()
        passed += 1
        print("  PASS test_set_single_element")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_single_element: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_single_element: " + str(e))
    try:
        test_set_bool_with_zero()
        passed += 1
        print("  PASS test_set_bool_with_zero")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_bool_with_zero: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_bool_with_zero: " + str(e))
    try:
        test_set_update_with_dict()
        passed += 1
        print("  PASS test_set_update_with_dict")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_update_with_dict: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_update_with_dict: " + str(e))
    try:
        test_set_remove_vs_discard()
        passed += 1
        print("  PASS test_set_remove_vs_discard")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_remove_vs_discard: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_remove_vs_discard: " + str(e))
    try:
        test_set_clear_and_reuse()
        passed += 1
        print("  PASS test_set_clear_and_reuse")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_clear_and_reuse: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_clear_and_reuse: " + str(e))
    try:
        test_set_copy_independent()
        passed += 1
        print("  PASS test_set_copy_independent")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_copy_independent: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_copy_independent: " + str(e))
    try:
        test_set_generator_expression()
        passed += 1
        print("  PASS test_set_generator_expression")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_generator_expression: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_generator_expression: " + str(e))
    try:
        test_set_tuple_elements()
        passed += 1
        print("  PASS test_set_tuple_elements")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_set_tuple_elements: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_set_tuple_elements: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
