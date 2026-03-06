"""Enum tests for StrEnum and IntEnum."""

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


def test_strenum_as_parameter() -> None:
    def describe(c: Color) -> str:
        if c == Color.RED:
            return "red"
        if c == Color.GREEN:
            return "green"
        return "blue"
    assert describe(Color.RED) == "red"
    assert describe(Color.BLUE) == "blue"


def test_intenum_equality() -> None:
    assert Priority.LOW == Priority.LOW
    assert Priority.LOW != Priority.HIGH


def test_intenum_tostring() -> None:
    assert str(Priority.LOW) == "Priority.LOW"


def test_intenum_as_parameter() -> None:
    def level(p: Priority) -> str:
        if p == Priority.HIGH:
            return "high"
        return "low"
    assert level(Priority.HIGH) == "high"
    assert level(Priority.LOW) == "low"


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
    assert c == Color.RED


def test_optional_intenum() -> None:
    p: Priority | None = None
    assert p is None
    p = Priority.HIGH
    assert p == Priority.HIGH
