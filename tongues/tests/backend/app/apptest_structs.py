"""Struct equality tests."""

import sys
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


def test_struct_eq_same() -> None:
    """Two structs with same fields compare equal."""
    a: Point = Point(1, 2)
    b: Point = Point(1, 2)
    assert a == b


def test_struct_eq_different() -> None:
    """Different field values compare not-equal."""
    a: Point = Point(1, 2)
    b: Point = Point(1, 3)
    assert not (a == b)


def test_struct_ne() -> None:
    """!= operator on structs."""
    a: Point = Point(1, 2)
    b: Point = Point(3, 4)
    assert a != b
    assert not (Point(5, 6) != Point(5, 6))


def test_struct_eq_in_list() -> None:
    """Struct equality works with 'in' operator."""
    pts: list[Point] = [Point(1, 2), Point(3, 4)]
    assert Point(1, 2) in pts
    assert Point(3, 4) in pts
    assert Point(5, 6) not in pts


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_struct_eq_same", test_struct_eq_same),
        ("test_struct_eq_different", test_struct_eq_different),
        ("test_struct_ne", test_struct_ne),
        ("test_struct_eq_in_list", test_struct_eq_in_list),
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
