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
    try:
        test_strenum_equality()
        passed += 1
        print("  PASS test_strenum_equality")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_strenum_equality: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_strenum_equality: " + str(e))
    try:
        test_strenum_tostring()
        passed += 1
        print("  PASS test_strenum_tostring")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_strenum_tostring: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_strenum_tostring: " + str(e))
    try:
        test_strenum_as_parameter()
        passed += 1
        print("  PASS test_strenum_as_parameter")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_strenum_as_parameter: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_strenum_as_parameter: " + str(e))
    try:
        test_intenum_equality()
        passed += 1
        print("  PASS test_intenum_equality")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_intenum_equality: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_intenum_equality: " + str(e))
    try:
        test_intenum_tostring()
        passed += 1
        print("  PASS test_intenum_tostring")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_intenum_tostring: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_intenum_tostring: " + str(e))
    try:
        test_intenum_as_parameter()
        passed += 1
        print("  PASS test_intenum_as_parameter")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_intenum_as_parameter: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_intenum_as_parameter: " + str(e))
    try:
        test_strenum_in_match()
        passed += 1
        print("  PASS test_strenum_in_match")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_strenum_in_match: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_strenum_in_match: " + str(e))
    try:
        test_intenum_in_match()
        passed += 1
        print("  PASS test_intenum_in_match")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_intenum_in_match: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_intenum_in_match: " + str(e))
    try:
        test_strenum_in_list()
        passed += 1
        print("  PASS test_strenum_in_list")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_strenum_in_list: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_strenum_in_list: " + str(e))
    try:
        test_strenum_as_dict_key()
        passed += 1
        print("  PASS test_strenum_as_dict_key")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_strenum_as_dict_key: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_strenum_as_dict_key: " + str(e))
    try:
        test_strenum_in_set()
        passed += 1
        print("  PASS test_strenum_in_set")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_strenum_in_set: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_strenum_in_set: " + str(e))
    try:
        test_optional_strenum()
        passed += 1
        print("  PASS test_optional_strenum")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_optional_strenum: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_optional_strenum: " + str(e))
    try:
        test_optional_intenum()
        passed += 1
        print("  PASS test_optional_intenum")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_optional_intenum: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_optional_intenum: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
