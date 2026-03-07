"""Enum tests for StrEnum and IntEnum."""

import sys

from enum import IntEnum, StrEnum


class Color(StrEnum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Priority(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


def test_strenum_equality() -> None:
    assert Color.RED == Color.RED
    assert Color.RED != Color.GREEN


def test_strenum_tostring() -> None:
    assert str(Color.RED) == "Color.RED"


def describe_color(c: Color) -> str:
    if c == Color.RED:
        return "red"
    if c == Color.GREEN:
        return "green"
    return "blue"


def test_strenum_as_parameter() -> None:
    assert describe_color(Color.RED) == "red"
    assert describe_color(Color.BLUE) == "blue"


def test_intenum_equality() -> None:
    assert Priority.LOW == Priority.LOW
    assert Priority.LOW != Priority.HIGH


def test_intenum_tostring() -> None:
    assert str(Priority.LOW) == "Priority.LOW"


def priority_level(p: Priority) -> str:
    if p == Priority.HIGH:
        return "high"
    return "low"


def test_intenum_as_parameter() -> None:
    assert priority_level(Priority.HIGH) == "high"
    assert priority_level(Priority.LOW) == "low"


def test_strenum_in_match() -> None:
    c: Color = Color.RED
    result: str = ""
    if c == Color.RED:
        result = "red"
    elif c == Color.GREEN:
        result = "green"
    else:
        result = "blue"
    assert result == "red"


def test_intenum_in_match() -> None:
    p: Priority = Priority.MEDIUM
    result: str = ""
    if p == Priority.LOW:
        result = "low"
    elif p == Priority.MEDIUM:
        result = "mid"
    else:
        result = "high"
    assert result == "mid"


def test_strenum_in_list() -> None:
    colors: list[Color] = [Color.RED, Color.GREEN, Color.BLUE]
    assert len(colors) == 3
    assert colors[0] == Color.RED


def test_strenum_as_dict_key() -> None:
    names: dict[Color, str] = {Color.RED: "red", Color.GREEN: "green"}
    assert names[Color.RED] == "red"
    assert names[Color.GREEN] == "green"


def test_strenum_in_set() -> None:
    s: set[Color] = {Color.RED, Color.GREEN, Color.RED}
    assert len(s) == 2


def test_optional_strenum() -> None:
    c: Color | None = None
    assert c is None
    c = Color.RED
    assert c is not None


def test_optional_intenum() -> None:
    p: Priority | None = None
    assert p is None
    p = Priority.HIGH
    assert p is not None


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_strenum_equality", test_strenum_equality),
        ("test_strenum_tostring", test_strenum_tostring),
        ("test_strenum_as_parameter", test_strenum_as_parameter),
        ("test_intenum_equality", test_intenum_equality),
        ("test_intenum_tostring", test_intenum_tostring),
        ("test_intenum_as_parameter", test_intenum_as_parameter),
        ("test_strenum_in_match", test_strenum_in_match),
        ("test_intenum_in_match", test_intenum_in_match),
        ("test_strenum_in_list", test_strenum_in_list),
        ("test_strenum_as_dict_key", test_strenum_as_dict_key),
        ("test_strenum_in_set", test_strenum_in_set),
        ("test_optional_strenum", test_optional_strenum),
        ("test_optional_intenum", test_optional_intenum),
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
