"""Bytearray backend codegen tests."""

import sys


def test_bytearray_empty_constructor() -> None:
    """bytearray() creates empty list."""
    ba: bytearray = bytearray()
    assert len(ba) == 0
    assert list(ba) == []


def test_bytearray_from_bytes() -> None:
    """bytearray(b"hi") creates list of byte values."""
    ba: bytearray = bytearray(b"hi")
    assert list(ba) == [104, 105]


def test_bytearray_length() -> None:
    """len() works on bytearray."""
    ba0: bytearray = bytearray()
    assert len(ba0) == 0
    ba2: bytearray = bytearray(b"ab")
    assert len(ba2) == 2
    ba5: bytearray = bytearray(b"hello")
    assert len(ba5) == 5


def test_bytearray_indexing() -> None:
    """Indexing returns int."""
    ba: bytearray = bytearray(b"abc")
    assert ba[0] == 97
    assert ba[1] == 98
    assert ba[2] == 99
    assert ba[-1] == 99
    assert ba[-3] == 97


def test_bytearray_append() -> None:
    """append() adds element to end."""
    ba: bytearray = bytearray()
    ba.append(65)
    assert ba[0] == 65
    ba.append(66)
    assert len(ba) == 2
    assert ba[1] == 66


def test_bytearray_pop() -> None:
    """pop() removes and returns last element."""
    ba: bytearray = bytearray(b"abc")
    x: int = ba.pop()
    assert x == 99
    assert len(ba) == 2
    x = ba.pop()
    assert x == 98
    assert len(ba) == 1


def test_bytearray_insert() -> None:
    """insert() adds element at index."""
    ba: bytearray = bytearray(b"ac")
    ba.insert(1, 98)
    assert ba[0] == 97
    assert ba[1] == 98
    assert ba[2] == 99


def test_bytearray_extend() -> None:
    """extend() adds elements from another bytearray."""
    ba: bytearray = bytearray(b"ab")
    ba.extend(bytearray(b"cd"))
    assert len(ba) == 4
    assert ba[2] == 99
    assert ba[3] == 100


def test_bytearray_copy() -> None:
    """copy() creates independent copy."""
    original: bytearray = bytearray(b"abc")
    copied: bytearray = original.copy()
    copied.append(100)
    assert len(original) == 3
    assert len(copied) == 4


def test_bytearray_sort() -> None:
    """sort() sorts in place."""
    ba: bytearray = bytearray(b"cab")
    ba.sort()
    assert ba[0] == 97
    assert ba[1] == 98
    assert ba[2] == 99


def test_bytearray_reverse() -> None:
    """reverse() reverses in place."""
    ba: bytearray = bytearray(b"abc")
    ba.reverse()
    assert ba[0] == 99
    assert ba[1] == 98
    assert ba[2] == 97


def test_bytearray_clear() -> None:
    """clear() empties the bytearray."""
    ba: bytearray = bytearray(b"abc")
    ba.clear()
    assert len(ba) == 0


def test_bytearray_remove() -> None:
    """remove() removes first occurrence."""
    ba: bytearray = bytearray(b"abcba")
    ba.remove(98)
    assert len(ba) == 4
    assert ba[0] == 97
    assert ba[1] == 99


def test_bytearray_index() -> None:
    """index() finds index of element."""
    ba: bytearray = bytearray(b"abcde")
    assert ba.index(97) == 0
    assert ba.index(99) == 2
    assert ba.index(101) == 4


def test_bytearray_iteration() -> None:
    """for loop yields ints."""
    ba: bytearray = bytearray(b"abc")
    result: list[int] = []
    for x in ba:
        result.append(x)
    assert result == [97, 98, 99]


def test_bytearray_list_conversion() -> None:
    """list(ba) returns list[int]."""
    ba: bytearray = bytearray(b"hi")
    items: list[int] = list(ba)
    assert items == [104, 105]


def test_bytearray_decode() -> None:
    """decode() converts bytearray to string."""
    ba: bytearray = bytearray(b"hello")
    s: str = ba.decode()
    assert s == "hello"


def test_bytearray_decode_empty() -> None:
    """decode() on empty bytearray."""
    ba: bytearray = bytearray()
    assert ba.decode() == ""


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_bytearray_empty_constructor()
        passed += 1
        print("  PASS test_bytearray_empty_constructor")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_empty_constructor: {e!s}")
    try:
        test_bytearray_from_bytes()
        passed += 1
        print("  PASS test_bytearray_from_bytes")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_from_bytes: {e!s}")
    try:
        test_bytearray_length()
        passed += 1
        print("  PASS test_bytearray_length")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_length: {e!s}")
    try:
        test_bytearray_indexing()
        passed += 1
        print("  PASS test_bytearray_indexing")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_indexing: {e!s}")
    try:
        test_bytearray_append()
        passed += 1
        print("  PASS test_bytearray_append")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_append: {e!s}")
    try:
        test_bytearray_pop()
        passed += 1
        print("  PASS test_bytearray_pop")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_pop: {e!s}")
    try:
        test_bytearray_insert()
        passed += 1
        print("  PASS test_bytearray_insert")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_insert: {e!s}")
    try:
        test_bytearray_extend()
        passed += 1
        print("  PASS test_bytearray_extend")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_extend: {e!s}")
    try:
        test_bytearray_copy()
        passed += 1
        print("  PASS test_bytearray_copy")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_copy: {e!s}")
    try:
        test_bytearray_sort()
        passed += 1
        print("  PASS test_bytearray_sort")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_sort: {e!s}")
    try:
        test_bytearray_reverse()
        passed += 1
        print("  PASS test_bytearray_reverse")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_reverse: {e!s}")
    try:
        test_bytearray_clear()
        passed += 1
        print("  PASS test_bytearray_clear")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_clear: {e!s}")
    try:
        test_bytearray_remove()
        passed += 1
        print("  PASS test_bytearray_remove")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_remove: {e!s}")
    try:
        test_bytearray_index()
        passed += 1
        print("  PASS test_bytearray_index")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_index: {e!s}")
    try:
        test_bytearray_iteration()
        passed += 1
        print("  PASS test_bytearray_iteration")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_iteration: {e!s}")
    try:
        test_bytearray_list_conversion()
        passed += 1
        print("  PASS test_bytearray_list_conversion")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_list_conversion: {e!s}")
    try:
        test_bytearray_decode()
        passed += 1
        print("  PASS test_bytearray_decode")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_decode: {e!s}")
    try:
        test_bytearray_decode_empty()
        passed += 1
        print("  PASS test_bytearray_decode_empty")
    except Exception as e:
        failed += 1
        print(f"  FAIL test_bytearray_decode_empty: {e!s}")
    print(f"{passed!s} passed, {failed!s} failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
