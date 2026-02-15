"""List object tests."""

import sys


def test_list_equality() -> None:
    """List equality comparisons."""
    assert [1, 2, 3] == [1, 2, 3]
    assert [] == []
    assert [1] == [1]
    assert not ([1, 2] == [1, 3])
    assert [1, 2] != [1, 3]
    assert [1, 2] != [1, 2, 3]
    assert [1, 2, 3] != [1, 2]


def test_list_ordering() -> None:
    """List ordering is lexicographic."""
    assert [1, 2] < [1, 3]
    assert [1, 2] < [1, 2, 3]
    assert [] < [1]
    assert [1] < [2]
    assert [1, 2] <= [1, 2]
    assert [1, 2] <= [1, 3]
    assert [1, 3] > [1, 2]
    assert [1, 2, 3] > [1, 2]
    assert [1, 2] >= [1, 2]
    assert [1, 3] >= [1, 2]


def test_list_length() -> None:
    """len() returns element count."""
    assert len([]) == 0
    assert len([1]) == 1
    assert len([1, 2, 3]) == 3
    assert len([1, 2, 3, 4, 5]) == 5


def test_list_indexing() -> None:
    """Indexing returns element."""
    items: list[int] = [10, 20, 30, 40, 50]
    assert items[0] == 10
    assert items[1] == 20
    assert items[4] == 50
    # Negative indexing
    assert items[-1] == 50
    assert items[-2] == 40
    assert items[-5] == 10


def test_list_index_assignment() -> None:
    """Index assignment modifies element."""
    items: list[int] = [1, 2, 3]
    items[0] = 10
    assert items[0] == 10
    items[2] = 30
    assert items[2] == 30
    items[-1] = 300
    assert items[-1] == 300
    assert items == [10, 2, 300]


def test_list_slicing() -> None:
    """List slicing returns new list."""
    items: list[int] = [1, 2, 3, 4, 5]
    assert items[0:2] == [1, 2]
    assert items[1:4] == [2, 3, 4]
    assert items[:3] == [1, 2, 3]
    assert items[2:] == [3, 4, 5]
    assert items[:] == [1, 2, 3, 4, 5]
    assert items[::2] == [1, 3, 5]
    assert items[::-1] == [5, 4, 3, 2, 1]
    # Empty slices
    assert items[2:2] == []
    assert items[5:10] == []


def test_list_slice_step() -> None:
    """Slicing with step."""
    items: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert items[::2] == [0, 2, 4, 6, 8]
    assert items[1::2] == [1, 3, 5, 7, 9]
    assert items[::3] == [0, 3, 6, 9]
    assert items[::-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert items[::-2] == [9, 7, 5, 3, 1]
    assert items[8:2:-1] == [8, 7, 6, 5, 4, 3]


def test_list_concatenation() -> None:
    """List concatenation with +."""
    assert [1, 2] + [3, 4] == [1, 2, 3, 4]
    assert [] + [1] == [1]
    assert [1] + [] == [1]
    assert [] + [] == []
    assert [1] + [2] + [3] == [1, 2, 3]


def test_list_repetition() -> None:
    """List repetition with *."""
    assert [1] * 3 == [1, 1, 1]
    assert [1, 2] * 2 == [1, 2, 1, 2]
    assert [1] * 0 == []
    assert [1] * 1 == [1]
    assert 3 * [1] == [1, 1, 1]
    assert [] * 5 == []


def test_list_repetition_negative() -> None:
    """Negative multiplier gives empty list."""
    assert [1, 2, 3] * -1 == []
    assert [1, 2, 3] * -100 == []
    assert -5 * [1, 2] == []


def test_list_contains() -> None:
    """Membership testing with in."""
    items: list[int] = [1, 2, 3, 4, 5]
    assert 1 in items
    assert 3 in items
    assert 5 in items
    assert 0 not in items
    assert 6 not in items
    assert 0 not in items
    assert 6 not in items


def test_list_contains_empty() -> None:
    """Membership in empty list."""
    empty: list[int] = []
    assert 1 not in empty
    assert 1 not in empty


def test_list_bool() -> None:
    """List truthiness - empty is falsy."""
    assert bool([1, 2, 3]) == True
    assert bool([0]) == True
    assert bool([]) == False
    assert not []
    assert [1]


def test_list_append() -> None:
    """append() adds element to end."""
    items: list[int] = [1, 2]
    items.append(3)
    assert items == [1, 2, 3]
    items.append(4)
    assert items == [1, 2, 3, 4]
    # Append to empty
    empty: list[int] = []
    empty.append(1)
    assert empty == [1]


def test_list_extend() -> None:
    """extend() adds multiple elements."""
    items: list[int] = [1, 2]
    items.extend([3, 4])
    assert items == [1, 2, 3, 4]
    items.extend([5])
    assert items == [1, 2, 3, 4, 5]
    items.extend([])
    assert items == [1, 2, 3, 4, 5]


def test_list_insert() -> None:
    """insert() adds element at index."""
    items: list[int] = [1, 3]
    items.insert(1, 2)
    assert items == [1, 2, 3]
    items.insert(0, 0)
    assert items == [0, 1, 2, 3]
    items.insert(4, 4)
    assert items == [0, 1, 2, 3, 4]
    # Insert beyond end
    items.insert(100, 5)
    assert items == [0, 1, 2, 3, 4, 5]
    # Negative index
    items.insert(-1, 99)
    assert items[-2] == 99


def test_list_pop() -> None:
    """pop() removes and returns last element."""
    items: list[int] = [1, 2, 3]
    x: int = items.pop()
    assert x == 3
    assert items == [1, 2]
    x = items.pop()
    assert x == 2
    assert items == [1]


def test_list_pop_index() -> None:
    """pop(i) removes and returns element at index."""
    items: list[int] = [1, 2, 3, 4, 5]
    x: int = items.pop(0)
    assert x == 1
    assert items == [2, 3, 4, 5]
    x = items.pop(2)
    assert x == 4
    assert items == [2, 3, 5]
    x = items.pop(-1)
    assert x == 5
    assert items == [2, 3]


def test_list_remove() -> None:
    """remove() removes first occurrence of value."""
    items: list[int] = [1, 2, 3, 2, 4]
    items.remove(2)
    assert items == [1, 3, 2, 4]
    items.remove(2)
    assert items == [1, 3, 4]
    items.remove(1)
    assert items == [3, 4]


def test_list_clear() -> None:
    """clear() removes all elements."""
    items: list[int] = [1, 2, 3]
    items.clear()
    assert items == []
    assert len(items) == 0
    # Clear empty list is fine
    items.clear()
    assert items == []


def test_list_copy() -> None:
    """copy() creates shallow copy."""
    original: list[int] = [1, 2, 3]
    copied: list[int] = original.copy()
    assert copied == original
    # Modifying copy doesn't affect original
    copied.append(4)
    assert copied == [1, 2, 3, 4]
    assert original == [1, 2, 3]
    # Modifying original doesn't affect copy
    original.append(5)
    assert original == [1, 2, 3, 5]
    assert copied == [1, 2, 3, 4]


def test_list_index() -> None:
    """index() returns first index of value."""
    items: list[int] = [10, 20, 30, 20, 40]
    assert items.index(10) == 0
    assert items.index(20) == 1
    assert items.index(30) == 2
    assert items.index(40) == 4


def test_list_count() -> None:
    """count() returns number of occurrences."""
    items: list[int] = [1, 2, 2, 3, 2, 4]
    assert items.count(1) == 1
    assert items.count(2) == 3
    assert items.count(3) == 1
    assert items.count(5) == 0
    assert [].count(1) == 0


def test_list_reverse() -> None:
    """reverse() reverses in place."""
    items: list[int] = [1, 2, 3, 4, 5]
    items.reverse()
    assert items == [5, 4, 3, 2, 1]
    # Reverse again
    items.reverse()
    assert items == [1, 2, 3, 4, 5]
    # Reverse empty
    empty: list[int] = []
    empty.reverse()
    assert empty == []
    # Reverse single element
    single: list[int] = [1]
    single.reverse()
    assert single == [1]


def test_list_sort() -> None:
    """sort() sorts in place."""
    items: list[int] = [3, 1, 4, 1, 5, 9, 2, 6]
    items.sort()
    assert items == [1, 1, 2, 3, 4, 5, 6, 9]
    # Already sorted
    items.sort()
    assert items == [1, 1, 2, 3, 4, 5, 6, 9]


def test_list_sort_reverse() -> None:
    """sort(reverse=True) sorts descending."""
    items: list[int] = [3, 1, 4, 1, 5]
    items.sort(reverse=True)
    assert items == [5, 4, 3, 1, 1]


def test_list_sort_strings() -> None:
    """sort() on strings."""
    words: list[str] = ["banana", "apple", "cherry"]
    words.sort()
    assert words == ["apple", "banana", "cherry"]
    words.sort(reverse=True)
    assert words == ["cherry", "banana", "apple"]


def test_list_iteration() -> None:
    """Iterating over list."""
    items: list[int] = [1, 2, 3]
    result: list[int] = []
    for x in items:
        result.append(x)
    assert result == [1, 2, 3]


def test_list_iteration_empty() -> None:
    """Iterating over empty list."""
    items: list[int] = []
    count: int = 0
    for x in items:
        count = count + 1
    assert count == 0


def test_list_enumerate() -> None:
    """enumerate() with list."""
    items: list[str] = ["a", "b", "c"]
    indices: list[int] = []
    values: list[str] = []
    for i, v in enumerate(items):
        indices.append(i)
        values.append(v)
    assert indices == [0, 1, 2]
    assert values == ["a", "b", "c"]


def test_list_comprehension() -> None:
    """List comprehension creates new list."""
    squares: list[int] = [x * x for x in [1, 2, 3, 4, 5]]
    assert squares == [1, 4, 9, 16, 25]
    # With condition
    evens: list[int] = [x for x in [1, 2, 3, 4, 5, 6] if x % 2 == 0]
    assert evens == [2, 4, 6]


def test_list_comprehension_nested() -> None:
    """Nested list comprehension."""
    matrix: list[list[int]] = [[1, 2], [3, 4], [5, 6]]
    flat: list[int] = [x for row in matrix for x in row]
    assert flat == [1, 2, 3, 4, 5, 6]


def test_list_sum() -> None:
    """sum() of list."""
    assert sum([]) == 0
    assert sum([1]) == 1
    assert sum([1, 2, 3]) == 6
    assert sum([1, 2, 3, 4, 5]) == 15
    assert sum([-1, 0, 1]) == 0


def test_list_min_max() -> None:
    """min() and max() of list."""
    items: list[int] = [3, 1, 4, 1, 5, 9, 2, 6]
    assert min(items) == 1
    assert max(items) == 9
    # Single element
    assert min([42]) == 42
    assert max([42]) == 42
    # Negative numbers
    assert min([-5, -1, -10]) == -10
    assert max([-5, -1, -10]) == -1


def test_list_sorted() -> None:
    """sorted() returns new sorted list."""
    items: list[int] = [3, 1, 4, 1, 5]
    result: list[int] = sorted(items)
    assert result == [1, 1, 3, 4, 5]
    # Original unchanged
    assert items == [3, 1, 4, 1, 5]


def test_list_sorted_reverse() -> None:
    """sorted(reverse=True) returns descending."""
    items: list[int] = [3, 1, 4, 1, 5]
    result: list[int] = sorted(items, reverse=True)
    assert result == [5, 4, 3, 1, 1]
    assert items == [3, 1, 4, 1, 5]


def test_list_all_any() -> None:
    """all() and any() on lists."""
    assert all([True, True, True]) == True
    assert all([True, False, True]) == False
    assert all([]) == True
    assert any([False, False, True]) == True
    assert any([False, False, False]) == False
    assert any([]) == False


def test_list_all_any_truthy() -> None:
    """all() and any() use truthiness."""
    assert all([1, 2, 3]) == True
    assert all([1, 0, 3]) == False
    assert any([0, 0, 1]) == True
    assert any([0, 0, 0]) == False
    assert any(["", "", "x"]) == True
    assert any(["", "", ""]) == False


def test_list_nested() -> None:
    """Nested lists."""
    matrix: list[list[int]] = [[1, 2], [3, 4], [5, 6]]
    assert len(matrix) == 3
    assert matrix[0] == [1, 2]
    assert matrix[1][0] == 3
    assert matrix[2][1] == 6


def test_list_nested_modification() -> None:
    """Modifying nested lists."""
    matrix: list[list[int]] = [[1, 2], [3, 4]]
    matrix[0][0] = 10
    assert matrix == [[10, 2], [3, 4]]
    matrix[1].append(5)
    assert matrix == [[10, 2], [3, 4, 5]]


def test_list_constructor_from_range() -> None:
    """list() from range."""
    items: list[int] = list(range(5))
    assert items == [0, 1, 2, 3, 4]
    items = list(range(2, 6))
    assert items == [2, 3, 4, 5]
    items = list(range(0, 10, 2))
    assert items == [0, 2, 4, 6, 8]


def test_list_constructor_from_string() -> None:
    """list() from string."""
    chars: list[str] = list("hello")
    assert chars == ["h", "e", "l", "l", "o"]
    assert list("") == []


def test_list_zip() -> None:
    """zip() with lists."""
    a: list[int] = [1, 2, 3]
    b: list[str] = ["a", "b", "c"]
    pairs: list[tuple[int, str]] = list(zip(a, b))
    assert pairs == [(1, "a"), (2, "b"), (3, "c")]


def test_list_zip_unequal() -> None:
    """zip() stops at shortest."""
    a: list[int] = [1, 2, 3, 4, 5]
    b: list[str] = ["a", "b", "c"]
    pairs: list[tuple[int, str]] = list(zip(a, b))
    assert len(pairs) == 3
    assert pairs == [(1, "a"), (2, "b"), (3, "c")]


def test_list_identity() -> None:
    """List identity vs equality."""
    a: list[int] = [1, 2, 3]
    b: list[int] = [1, 2, 3]
    c: list[int] = a
    # Equal
    assert a == b
    assert a == c
    # c is same object as a
    a.append(4)
    assert c == [1, 2, 3, 4]
    # b is different object
    assert b == [1, 2, 3]


def test_list_slice_assignment() -> None:
    """Slice assignment replaces section."""
    items: list[int] = [1, 2, 3, 4, 5]
    items[1:3] = [20, 30]
    assert items == [1, 20, 30, 4, 5]
    # Different length replacement
    items[1:3] = [100]
    assert items == [1, 100, 4, 5]
    # Insert via slice
    items[2:2] = [200, 300]
    assert items == [1, 100, 200, 300, 4, 5]


def test_list_multiplication_reference() -> None:
    """Multiplication creates references for nested lists."""
    # For simple types, multiplication works as expected
    nums: list[int] = [0] * 3
    nums[0] = 1
    assert nums == [1, 0, 0]


def test_list_string_elements() -> None:
    """List of strings."""
    words: list[str] = ["hello", "world"]
    assert words[0] == "hello"
    assert "hello" in words
    assert "foo" not in words
    words.append("!")
    assert words == ["hello", "world", "!"]


def test_list_multiplication_nested_reference() -> None:
    """Gotcha: multiplication creates shared references for nested lists."""
    rows: list[list[int]] = [[0]] * 3
    rows[0].append(1)
    # All three inner lists are the same object
    assert rows[0] == [0, 1]
    assert rows[1] == [0, 1]
    assert rows[2] == [0, 1]
    # Correct approach: list comprehension creates independent lists
    rows2: list[list[int]] = [[0] for _ in range(3)]
    rows2[0].append(1)
    assert rows2[0] == [0, 1]
    assert rows2[1] == [0]
    assert rows2[2] == [0]


def test_list_append_vs_extend() -> None:
    """append() adds single element, extend() unpacks iterable."""
    items1: list[int] = [1, 2]
    items1.append(3)
    assert items1 == [1, 2, 3]
    items2: list[int] = [1, 2]
    items2.extend([3])
    assert items2 == [1, 2, 3]


def test_list_extend_unpacks() -> None:
    """extend() unpacks any iterable."""
    items: list[int] = [1, 2]
    items.extend([3, 4, 5])
    assert items == [1, 2, 3, 4, 5]
    items.extend(range(6, 9))
    assert items == [1, 2, 3, 4, 5, 6, 7, 8]


def test_list_extend_string() -> None:
    """Gotcha: extend() with string adds each character."""
    chars: list[str] = ["a", "b"]
    chars.extend("cd")
    assert chars == ["a", "b", "c", "d"]
    assert len(chars) == 4


def test_list_iadd_like_extend() -> None:
    """+= behaves like extend, not append."""
    items: list[int] = [1, 2]
    items += [3, 4]
    assert items == [1, 2, 3, 4]
    # += with range
    items += range(5, 7)
    assert items == [1, 2, 3, 4, 5, 6]


def test_list_iadd_string() -> None:
    """Gotcha: += with string adds each character."""
    chars: list[str] = ["a"]
    chars += "bc"
    assert chars == ["a", "b", "c"]


def test_list_sort_returns_none() -> None:
    """Gotcha: sort() returns None (in-place modification)."""
    items: list[int] = [3, 1, 2]
    result: None = items.sort()
    assert result is None
    assert items == [1, 2, 3]


def test_list_reverse_returns_none() -> None:
    """reverse() returns None (in-place modification)."""
    items: list[int] = [1, 2, 3]
    result: None = items.reverse()
    assert result is None
    assert items == [3, 2, 1]


def test_list_slice_out_of_bounds() -> None:
    """Slicing out of bounds is forgiving."""
    items: list[int] = [1, 2, 3]
    # Start beyond end
    assert items[10:20] == []
    # End beyond end
    assert items[1:100] == [2, 3]
    # Negative beyond start
    assert items[-100:2] == [1, 2]
    # Both beyond
    assert items[-100:100] == [1, 2, 3]


def test_list_insert_negative() -> None:
    """insert() with negative index inserts before that position."""
    items: list[int] = [1, 2, 3]
    items.insert(-1, 99)
    # -1 refers to last element, so 99 inserted before it
    assert items == [1, 2, 99, 3]
    items2: list[int] = [1, 2, 3]
    items2.insert(-100, 0)
    # Beyond start clips to beginning
    assert items2 == [0, 1, 2, 3]


def test_list_slice_delete() -> None:
    """Slice assignment with empty list deletes."""
    items: list[int] = [1, 2, 3, 4, 5]
    items[1:4] = []
    assert items == [1, 5]
    # Delete all
    items2: list[int] = [1, 2, 3]
    items2[:] = []
    assert items2 == []


def test_list_slice_insert_via_zero_length() -> None:
    """Zero-length slice assignment inserts."""
    items: list[int] = [1, 4]
    items[1:1] = [2, 3]
    assert items == [1, 2, 3, 4]
    # Insert at beginning
    items[0:0] = [0]
    assert items == [0, 1, 2, 3, 4]
    # Insert at end
    items[5:5] = [5, 6]
    assert items == [0, 1, 2, 3, 4, 5, 6]


def test_list_copy_shallow() -> None:
    """copy() is shallow - nested objects are shared."""
    original: list[list[int]] = [[1, 2], [3, 4]]
    copied: list[list[int]] = original.copy()
    # Outer list is different
    copied.append([5, 6])
    assert len(original) == 2
    assert len(copied) == 3
    # But inner lists are same objects
    original[0].append(99)
    assert copied[0] == [1, 2, 99]


def test_list_slice_copy_shallow() -> None:
    """Slicing [:] creates shallow copy."""
    original: list[list[int]] = [[1], [2]]
    copied: list[list[int]] = original[:]
    assert copied == original
    # Different outer list
    copied.append([3])
    assert len(original) == 2
    # Same inner lists
    original[0].append(99)
    assert copied[0] == [1, 99]


def test_list_index_with_start() -> None:
    """index() with start parameter."""
    items: list[int] = [1, 2, 1, 2, 1]
    assert items.index(1) == 0
    assert items.index(1, 1) == 2
    assert items.index(1, 3) == 4
    assert items.index(2, 2) == 3


def test_list_index_with_start_end() -> None:
    """index() with start and end parameters."""
    items: list[int] = [0, 1, 2, 1, 0]
    assert items.index(1, 0, 3) == 1
    assert items.index(1, 2, 4) == 3


def test_list_pop_empty_default() -> None:
    """pop() on single element list."""
    items: list[int] = [42]
    x: int = items.pop()
    assert x == 42
    assert items == []


def test_list_equality_different_types() -> None:
    """Lists of different element types."""
    ints: list[int] = [1, 2, 3]
    floats: list[float] = [1.0, 2.0, 3.0]
    assert ints == floats


def test_list_multiply_zero() -> None:
    """Multiplying by zero gives empty list."""
    items: list[int] = [1, 2, 3]
    result: list[int] = items * 0
    assert result == []
    assert items == [1, 2, 3]


def test_list_concatenation_creates_new() -> None:
    """Concatenation creates new list."""
    a: list[int] = [1, 2]
    b: list[int] = [3, 4]
    c: list[int] = a + b
    c.append(5)
    assert a == [1, 2]
    assert b == [3, 4]
    assert c == [1, 2, 3, 4, 5]


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_list_equality", test_list_equality),
        ("test_list_ordering", test_list_ordering),
        ("test_list_length", test_list_length),
        ("test_list_indexing", test_list_indexing),
        ("test_list_index_assignment", test_list_index_assignment),
        ("test_list_slicing", test_list_slicing),
        ("test_list_slice_step", test_list_slice_step),
        ("test_list_concatenation", test_list_concatenation),
        ("test_list_repetition", test_list_repetition),
        ("test_list_repetition_negative", test_list_repetition_negative),
        ("test_list_contains", test_list_contains),
        ("test_list_contains_empty", test_list_contains_empty),
        ("test_list_bool", test_list_bool),
        ("test_list_append", test_list_append),
        ("test_list_extend", test_list_extend),
        ("test_list_insert", test_list_insert),
        ("test_list_pop", test_list_pop),
        ("test_list_pop_index", test_list_pop_index),
        ("test_list_remove", test_list_remove),
        ("test_list_clear", test_list_clear),
        ("test_list_copy", test_list_copy),
        ("test_list_index", test_list_index),
        ("test_list_count", test_list_count),
        ("test_list_reverse", test_list_reverse),
        ("test_list_sort", test_list_sort),
        ("test_list_sort_reverse", test_list_sort_reverse),
        ("test_list_sort_strings", test_list_sort_strings),
        ("test_list_iteration", test_list_iteration),
        ("test_list_iteration_empty", test_list_iteration_empty),
        ("test_list_enumerate", test_list_enumerate),
        ("test_list_comprehension", test_list_comprehension),
        ("test_list_comprehension_nested", test_list_comprehension_nested),
        ("test_list_sum", test_list_sum),
        ("test_list_min_max", test_list_min_max),
        ("test_list_sorted", test_list_sorted),
        ("test_list_sorted_reverse", test_list_sorted_reverse),
        ("test_list_all_any", test_list_all_any),
        ("test_list_all_any_truthy", test_list_all_any_truthy),
        ("test_list_nested", test_list_nested),
        ("test_list_nested_modification", test_list_nested_modification),
        ("test_list_constructor_from_range", test_list_constructor_from_range),
        ("test_list_constructor_from_string", test_list_constructor_from_string),
        ("test_list_zip", test_list_zip),
        ("test_list_zip_unequal", test_list_zip_unequal),
        ("test_list_identity", test_list_identity),
        ("test_list_slice_assignment", test_list_slice_assignment),
        ("test_list_multiplication_reference", test_list_multiplication_reference),
        ("test_list_string_elements", test_list_string_elements),
        (
            "test_list_multiplication_nested_reference",
            test_list_multiplication_nested_reference,
        ),
        ("test_list_append_vs_extend", test_list_append_vs_extend),
        ("test_list_extend_unpacks", test_list_extend_unpacks),
        ("test_list_extend_string", test_list_extend_string),
        ("test_list_iadd_like_extend", test_list_iadd_like_extend),
        ("test_list_iadd_string", test_list_iadd_string),
        ("test_list_sort_returns_none", test_list_sort_returns_none),
        ("test_list_reverse_returns_none", test_list_reverse_returns_none),
        ("test_list_slice_out_of_bounds", test_list_slice_out_of_bounds),
        ("test_list_insert_negative", test_list_insert_negative),
        ("test_list_slice_delete", test_list_slice_delete),
        (
            "test_list_slice_insert_via_zero_length",
            test_list_slice_insert_via_zero_length,
        ),
        ("test_list_copy_shallow", test_list_copy_shallow),
        ("test_list_slice_copy_shallow", test_list_slice_copy_shallow),
        ("test_list_index_with_start", test_list_index_with_start),
        ("test_list_index_with_start_end", test_list_index_with_start_end),
        ("test_list_pop_empty_default", test_list_pop_empty_default),
        ("test_list_equality_different_types", test_list_equality_different_types),
        ("test_list_multiply_zero", test_list_multiply_zero),
        ("test_list_concatenation_creates_new", test_list_concatenation_creates_new),
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
