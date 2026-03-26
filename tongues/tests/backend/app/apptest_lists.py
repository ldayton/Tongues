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


def test_list_constructor_from_bytes() -> None:
    """list() from bytes produces list of int byte values."""
    items: list[int] = list(b"\x01\x02\x03")
    assert items == [1, 2, 3]
    assert list(b"\x00\xff") == [0, 255]
    assert list(b"") == []


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


def test_reversed_list() -> None:
    """reversed() iterates in reverse order."""
    items: list[int] = [1, 2, 3, 4, 5]
    result: list[int] = []
    for x in reversed(items):
        result.append(x)
    assert result == [5, 4, 3, 2, 1]


def test_reversed_range() -> None:
    """reversed(range()) iterates range in reverse."""
    result: list[int] = []
    for i in reversed(range(5)):
        result.append(i)
    assert result == [4, 3, 2, 1, 0]


def test_reversed_range_start_stop() -> None:
    """reversed(range(a, b)) iterates in reverse."""
    result: list[int] = []
    for i in reversed(range(2, 7)):
        result.append(i)
    assert result == [6, 5, 4, 3, 2]


def test_string_iteration_yields_str() -> None:
    """for ch in string yields str elements."""
    s: str = "abc"
    result: list[str] = []
    for ch in s:
        result.append(ch)
    assert result == ["a", "b", "c"]


def test_zip_bytes() -> None:
    """zip() over bytes sequences."""
    a: bytes = b"\x01\x02\x03"
    b: bytes = b"\x04\x05\x06"
    sums: list[int] = []
    for x, y in zip(a, b):
        sums.append(x + y)
    assert sums == [5, 7, 9]


def test_zip_three_lists() -> None:
    """zip() with three arguments."""
    a: list[int] = [1, 2, 3]
    b: list[str] = ["a", "b", "c"]
    c: list[int] = [10, 20, 30]
    result: list[str] = []
    for x, y, z in zip(a, b, c):
        result.append(str(x) + y + str(z))
    assert result == ["1a10", "2b20", "3c30"]


def test_list_repetition_with_annotation() -> None:
    """[expr] * n respects annotation type."""
    slots: list[int] = [0] * 5
    assert len(slots) == 5
    assert slots == [0, 0, 0, 0, 0]
    slots[2] = 42
    assert slots == [0, 0, 42, 0, 0]


def test_list_comprehension_with_annotation() -> None:
    """Comprehension result respects annotation type."""
    squares: list[int] = [x * x for x in range(5)]
    assert squares == [0, 1, 4, 9, 16]


def test_any_genexpr() -> None:
    """any() with generator expression."""
    assert any(x > 0 for x in [1, -2, -3]) == True
    assert any(x > 0 for x in [-1, -2, -3]) == False
    assert any(x > 0 for x in []) == False
    assert any(x == 2 for x in [1, 2, 3]) == True
    assert any(x == 5 for x in [1, 2, 3]) == False


def test_all_genexpr() -> None:
    """all() with generator expression."""
    assert all(x > 0 for x in [1, 2, 3]) == True
    assert all(x > 0 for x in [1, -2, 3]) == False
    assert all(x > 0 for x in []) == True


def test_any_genexpr_with_filter() -> None:
    """any() with generator expression and if-filter."""
    assert any(x > 0 for x in [-2, -4, 6] if x % 2 == 0) == True
    assert any(x > 0 for x in [-2, -4, -6] if x % 2 == 0) == False


def test_all_genexpr_with_filter() -> None:
    """all() with generator expression and if-filter."""
    assert all(x > 0 for x in [2, 4, -3] if x % 2 == 0) == True
    assert all(x > 0 for x in [-2, 4, -3] if x % 2 == 0) == False


def test_any_all_combined() -> None:
    """any() and all() with generator expressions on same data."""
    xs: list[int] = [1, -2, 3, -4, 5]
    assert any(x > 0 for x in xs) == True
    assert all(x > 0 for x in xs) == False
    assert any(x < 0 for x in xs) == True
    assert all(x != 0 for x in xs) == True


def test_reversed_range_arithmetic_start() -> None:
    """range(len(xs) - 1, -1, -1) with arithmetic start expression."""
    xs: list[int] = [10, 20, 30, 40, 50]
    result: list[int] = []
    for i in range(len(xs) - 1, -1, -1):
        result.append(xs[i])
    assert result == [50, 40, 30, 20, 10]


def test_reversed_range_arithmetic_step() -> None:
    """range(len(xs) - 1, 0, -2) with arithmetic start and step != -1."""
    xs: list[str] = ["a", "b", "c", "d", "e"]
    result: list[str] = []
    for i in range(len(xs) - 1, 0, -2):
        result.append(xs[i])
    assert result == ["e", "c"]


def test_reversed_range_with_subtraction() -> None:
    """range(n - 3, 0, -1) where start involves subtraction."""
    n: int = 10
    result: list[int] = []
    for i in range(n - 3, 0, -1):
        result.append(i)
    assert result == [7, 6, 5, 4, 3, 2, 1]


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_list_equality()
        passed += 1
        print("  PASS test_list_equality")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_equality: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_equality: " + str(e))
    try:
        test_list_ordering()
        passed += 1
        print("  PASS test_list_ordering")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_ordering: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_ordering: " + str(e))
    try:
        test_list_length()
        passed += 1
        print("  PASS test_list_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_length: " + str(e))
    try:
        test_list_indexing()
        passed += 1
        print("  PASS test_list_indexing")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_indexing: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_indexing: " + str(e))
    try:
        test_list_index_assignment()
        passed += 1
        print("  PASS test_list_index_assignment")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_index_assignment: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_index_assignment: " + str(e))
    try:
        test_list_slicing()
        passed += 1
        print("  PASS test_list_slicing")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_slicing: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_slicing: " + str(e))
    try:
        test_list_slice_step()
        passed += 1
        print("  PASS test_list_slice_step")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_slice_step: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_slice_step: " + str(e))
    try:
        test_list_concatenation()
        passed += 1
        print("  PASS test_list_concatenation")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_concatenation: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_concatenation: " + str(e))
    try:
        test_list_repetition()
        passed += 1
        print("  PASS test_list_repetition")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_repetition: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_repetition: " + str(e))
    try:
        test_list_repetition_negative()
        passed += 1
        print("  PASS test_list_repetition_negative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_repetition_negative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_repetition_negative: " + str(e))
    try:
        test_list_contains()
        passed += 1
        print("  PASS test_list_contains")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_contains: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_contains: " + str(e))
    try:
        test_list_contains_empty()
        passed += 1
        print("  PASS test_list_contains_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_contains_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_contains_empty: " + str(e))
    try:
        test_list_bool()
        passed += 1
        print("  PASS test_list_bool")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_bool: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_bool: " + str(e))
    try:
        test_list_append()
        passed += 1
        print("  PASS test_list_append")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_append: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_append: " + str(e))
    try:
        test_list_extend()
        passed += 1
        print("  PASS test_list_extend")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_extend: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_extend: " + str(e))
    try:
        test_list_insert()
        passed += 1
        print("  PASS test_list_insert")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_insert: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_insert: " + str(e))
    try:
        test_list_pop()
        passed += 1
        print("  PASS test_list_pop")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_pop: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_pop: " + str(e))
    try:
        test_list_pop_index()
        passed += 1
        print("  PASS test_list_pop_index")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_pop_index: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_pop_index: " + str(e))
    try:
        test_list_remove()
        passed += 1
        print("  PASS test_list_remove")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_remove: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_remove: " + str(e))
    try:
        test_list_clear()
        passed += 1
        print("  PASS test_list_clear")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_clear: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_clear: " + str(e))
    try:
        test_list_copy()
        passed += 1
        print("  PASS test_list_copy")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_copy: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_copy: " + str(e))
    try:
        test_list_index()
        passed += 1
        print("  PASS test_list_index")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_index: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_index: " + str(e))
    try:
        test_list_count()
        passed += 1
        print("  PASS test_list_count")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_count: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_count: " + str(e))
    try:
        test_list_reverse()
        passed += 1
        print("  PASS test_list_reverse")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_reverse: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_reverse: " + str(e))
    try:
        test_list_sort()
        passed += 1
        print("  PASS test_list_sort")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_sort: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_sort: " + str(e))
    try:
        test_list_sort_reverse()
        passed += 1
        print("  PASS test_list_sort_reverse")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_sort_reverse: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_sort_reverse: " + str(e))
    try:
        test_list_sort_strings()
        passed += 1
        print("  PASS test_list_sort_strings")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_sort_strings: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_sort_strings: " + str(e))
    try:
        test_list_iteration()
        passed += 1
        print("  PASS test_list_iteration")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_iteration: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_iteration: " + str(e))
    try:
        test_list_iteration_empty()
        passed += 1
        print("  PASS test_list_iteration_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_iteration_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_iteration_empty: " + str(e))
    try:
        test_list_enumerate()
        passed += 1
        print("  PASS test_list_enumerate")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_enumerate: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_enumerate: " + str(e))
    try:
        test_list_comprehension()
        passed += 1
        print("  PASS test_list_comprehension")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_comprehension: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_comprehension: " + str(e))
    try:
        test_list_comprehension_nested()
        passed += 1
        print("  PASS test_list_comprehension_nested")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_comprehension_nested: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_comprehension_nested: " + str(e))
    try:
        test_list_sum()
        passed += 1
        print("  PASS test_list_sum")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_sum: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_sum: " + str(e))
    try:
        test_list_min_max()
        passed += 1
        print("  PASS test_list_min_max")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_min_max: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_min_max: " + str(e))
    try:
        test_list_sorted()
        passed += 1
        print("  PASS test_list_sorted")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_sorted: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_sorted: " + str(e))
    try:
        test_list_sorted_reverse()
        passed += 1
        print("  PASS test_list_sorted_reverse")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_sorted_reverse: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_sorted_reverse: " + str(e))
    try:
        test_list_all_any()
        passed += 1
        print("  PASS test_list_all_any")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_all_any: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_all_any: " + str(e))
    try:
        test_list_all_any_truthy()
        passed += 1
        print("  PASS test_list_all_any_truthy")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_all_any_truthy: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_all_any_truthy: " + str(e))
    try:
        test_list_nested()
        passed += 1
        print("  PASS test_list_nested")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_nested: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_nested: " + str(e))
    try:
        test_list_nested_modification()
        passed += 1
        print("  PASS test_list_nested_modification")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_nested_modification: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_nested_modification: " + str(e))
    try:
        test_list_constructor_from_bytes()
        passed += 1
        print("  PASS test_list_constructor_from_bytes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_constructor_from_bytes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_constructor_from_bytes: " + str(e))
    try:
        test_list_constructor_from_range()
        passed += 1
        print("  PASS test_list_constructor_from_range")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_constructor_from_range: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_constructor_from_range: " + str(e))
    try:
        test_list_constructor_from_string()
        passed += 1
        print("  PASS test_list_constructor_from_string")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_constructor_from_string: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_constructor_from_string: " + str(e))
    try:
        test_list_zip()
        passed += 1
        print("  PASS test_list_zip")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_zip: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_zip: " + str(e))
    try:
        test_list_zip_unequal()
        passed += 1
        print("  PASS test_list_zip_unequal")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_zip_unequal: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_zip_unequal: " + str(e))
    try:
        test_list_identity()
        passed += 1
        print("  PASS test_list_identity")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_identity: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_identity: " + str(e))
    try:
        test_list_slice_assignment()
        passed += 1
        print("  PASS test_list_slice_assignment")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_slice_assignment: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_slice_assignment: " + str(e))
    try:
        test_list_multiplication_reference()
        passed += 1
        print("  PASS test_list_multiplication_reference")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_multiplication_reference: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_multiplication_reference: " + str(e))
    try:
        test_list_string_elements()
        passed += 1
        print("  PASS test_list_string_elements")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_string_elements: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_string_elements: " + str(e))
    try:
        test_list_append_vs_extend()
        passed += 1
        print("  PASS test_list_append_vs_extend")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_append_vs_extend: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_append_vs_extend: " + str(e))
    try:
        test_list_extend_unpacks()
        passed += 1
        print("  PASS test_list_extend_unpacks")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_extend_unpacks: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_extend_unpacks: " + str(e))
    try:
        test_list_extend_string()
        passed += 1
        print("  PASS test_list_extend_string")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_extend_string: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_extend_string: " + str(e))
    try:
        test_list_iadd_like_extend()
        passed += 1
        print("  PASS test_list_iadd_like_extend")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_iadd_like_extend: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_iadd_like_extend: " + str(e))
    try:
        test_list_iadd_string()
        passed += 1
        print("  PASS test_list_iadd_string")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_iadd_string: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_iadd_string: " + str(e))
    try:
        test_list_sort_returns_none()
        passed += 1
        print("  PASS test_list_sort_returns_none")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_sort_returns_none: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_sort_returns_none: " + str(e))
    try:
        test_list_reverse_returns_none()
        passed += 1
        print("  PASS test_list_reverse_returns_none")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_reverse_returns_none: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_reverse_returns_none: " + str(e))
    try:
        test_list_slice_out_of_bounds()
        passed += 1
        print("  PASS test_list_slice_out_of_bounds")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_slice_out_of_bounds: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_slice_out_of_bounds: " + str(e))
    try:
        test_list_insert_negative()
        passed += 1
        print("  PASS test_list_insert_negative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_insert_negative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_insert_negative: " + str(e))
    try:
        test_list_slice_delete()
        passed += 1
        print("  PASS test_list_slice_delete")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_slice_delete: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_slice_delete: " + str(e))
    try:
        test_list_copy_shallow()
        passed += 1
        print("  PASS test_list_copy_shallow")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_copy_shallow: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_copy_shallow: " + str(e))
    try:
        test_list_slice_copy_shallow()
        passed += 1
        print("  PASS test_list_slice_copy_shallow")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_slice_copy_shallow: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_slice_copy_shallow: " + str(e))
    try:
        test_list_index_with_start()
        passed += 1
        print("  PASS test_list_index_with_start")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_index_with_start: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_index_with_start: " + str(e))
    try:
        test_list_index_with_start_end()
        passed += 1
        print("  PASS test_list_index_with_start_end")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_index_with_start_end: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_index_with_start_end: " + str(e))
    try:
        test_list_pop_empty_default()
        passed += 1
        print("  PASS test_list_pop_empty_default")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_pop_empty_default: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_pop_empty_default: " + str(e))
    try:
        test_list_equality_different_types()
        passed += 1
        print("  PASS test_list_equality_different_types")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_equality_different_types: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_equality_different_types: " + str(e))
    try:
        test_list_multiply_zero()
        passed += 1
        print("  PASS test_list_multiply_zero")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_multiply_zero: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_multiply_zero: " + str(e))
    try:
        test_list_concatenation_creates_new()
        passed += 1
        print("  PASS test_list_concatenation_creates_new")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_concatenation_creates_new: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_concatenation_creates_new: " + str(e))
    try:
        test_reversed_list()
        passed += 1
        print("  PASS test_reversed_list")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_reversed_list: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_reversed_list: " + str(e))
    try:
        test_reversed_range()
        passed += 1
        print("  PASS test_reversed_range")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_reversed_range: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_reversed_range: " + str(e))
    try:
        test_reversed_range_start_stop()
        passed += 1
        print("  PASS test_reversed_range_start_stop")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_reversed_range_start_stop: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_reversed_range_start_stop: " + str(e))
    try:
        test_string_iteration_yields_str()
        passed += 1
        print("  PASS test_string_iteration_yields_str")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_iteration_yields_str: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_iteration_yields_str: " + str(e))
    try:
        test_zip_bytes()
        passed += 1
        print("  PASS test_zip_bytes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_zip_bytes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_zip_bytes: " + str(e))
    try:
        test_zip_three_lists()
        passed += 1
        print("  PASS test_zip_three_lists")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_zip_three_lists: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_zip_three_lists: " + str(e))
    try:
        test_list_repetition_with_annotation()
        passed += 1
        print("  PASS test_list_repetition_with_annotation")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_list_repetition_with_annotation: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_list_repetition_with_annotation: " + str(e))
    try:
        test_any_genexpr()
        passed += 1
        print("  PASS test_any_genexpr")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_any_genexpr: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_any_genexpr: " + str(e))
    try:
        test_all_genexpr()
        passed += 1
        print("  PASS test_all_genexpr")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_all_genexpr: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_all_genexpr: " + str(e))
    try:
        test_any_genexpr_with_filter()
        passed += 1
        print("  PASS test_any_genexpr_with_filter")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_any_genexpr_with_filter: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_any_genexpr_with_filter: " + str(e))
    try:
        test_all_genexpr_with_filter()
        passed += 1
        print("  PASS test_all_genexpr_with_filter")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_all_genexpr_with_filter: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_all_genexpr_with_filter: " + str(e))
    try:
        test_any_all_combined()
        passed += 1
        print("  PASS test_any_all_combined")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_any_all_combined: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_any_all_combined: " + str(e))
    try:
        test_reversed_range_arithmetic_start()
        passed += 1
        print("  PASS test_reversed_range_arithmetic_start")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_reversed_range_arithmetic_start: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_reversed_range_arithmetic_start: " + str(e))
    try:
        test_reversed_range_arithmetic_step()
        passed += 1
        print("  PASS test_reversed_range_arithmetic_step")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_reversed_range_arithmetic_step: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_reversed_range_arithmetic_step: " + str(e))
    try:
        test_reversed_range_with_subtraction()
        passed += 1
        print("  PASS test_reversed_range_with_subtraction")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_reversed_range_with_subtraction: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_reversed_range_with_subtraction: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
