"""None object tests.

Corner cases from:
- https://realpython.com/null-in-python/
- https://www.pythontutorial.net/advanced-python/python-none/
"""

import sys


def test_none_equality() -> None:
    """None compared with == to itself."""
    assert None == None
    assert not (None != None)


def test_none_str() -> None:
    """String representations of None."""
    assert str(None) == "None"
    assert repr(None) == "None"


def test_none_bool() -> None:
    """None is falsy - bool(None) returns False."""
    assert bool(None) == False
    assert not None


def test_none_not() -> None:
    """not None is True."""
    assert not None
    x: bool = not None
    assert x == True


def _no_return() -> None:
    """Helper: function with no return statement."""
    pass


def _returns_none() -> None:
    """Helper: function that explicitly returns None."""
    return None


def test_implicit_return() -> None:
    """Functions without return statement return None."""
    result: None = _no_return()
    assert result is None


def test_explicit_return_none() -> None:
    """Functions can explicitly return None."""
    result: None = _returns_none()
    assert result is None


def test_none_is_singleton() -> None:
    """None is a singleton - all Nones are the same object."""
    a: None = None
    b: None = None
    # assert a is b  # subset requires literal on one side of 'is'
    assert a is None
    assert b is None


def test_none_is_vs_equals() -> None:
    """is None vs == None - both work but is is preferred."""
    x: str | None = None
    assert x is None
    assert x == None
    y: str | None = "hello"
    assert y is not None
    assert y != None


def test_none_in_list() -> None:
    """None can be stored in lists."""
    items: list[int | None] = [1, None, 2, None, 3]
    assert items[0] == 1
    assert items[1] is None
    assert items[2] == 2
    assert items[3] is None
    assert len(items) == 5


def test_none_count_in_list() -> None:
    """Count None occurrences in list."""
    items: list[int | None] = [None, 1, None, 2, None]
    count: int = 0
    for item in items:
        if item is None:
            count = count + 1
    assert count == 3


def test_none_as_dict_value() -> None:
    """None can be a dict value."""
    d: dict[str, int | None] = {"a": 1, "b": None, "c": 3}
    assert d["a"] == 1
    assert d["b"] is None
    assert d["c"] == 3


def test_none_vs_missing_key() -> None:
    """None value vs missing key - different concepts."""
    d: dict[str, int | None] = {"a": None}
    assert "a" in d
    assert d["a"] is None
    assert "b" not in d
    assert d.get("b") is None
    assert d.get("b", -1) == -1
    assert d.get("a", -1) is None


def test_none_conditional_assignment() -> None:
    """Conditional assignment based on None check."""
    x: int | None = None
    if x is None:
        x = 42
    assert x == 42
    if x is None:
        x = 100
    assert x == 42


def test_none_ternary() -> None:
    """Ternary expression with None."""
    x: int | None = None
    result: int = 0 if x is None else x
    assert result == 0
    x = 5
    result = 0 if x is None else x
    assert result == 5


def test_none_not_equal_to_falsy() -> None:
    """None is not equal to other falsy values."""
    assert None != 0
    assert None != ""
    assert None != False
    assert not None
    assert not 0
    assert not ""
    assert not False


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_none_equality", test_none_equality),
        ("test_none_str", test_none_str),
        ("test_none_bool", test_none_bool),
        ("test_none_not", test_none_not),
        ("test_implicit_return", test_implicit_return),
        ("test_explicit_return_none", test_explicit_return_none),
        ("test_none_is_singleton", test_none_is_singleton),
        ("test_none_is_vs_equals", test_none_is_vs_equals),
        ("test_none_in_list", test_none_in_list),
        ("test_none_count_in_list", test_none_count_in_list),
        ("test_none_as_dict_value", test_none_as_dict_value),
        ("test_none_vs_missing_key", test_none_vs_missing_key),
        ("test_none_conditional_assignment", test_none_conditional_assignment),
        ("test_none_ternary", test_none_ternary),
        ("test_none_not_equal_to_falsy", test_none_not_equal_to_falsy),
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
