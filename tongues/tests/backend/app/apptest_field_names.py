"""Tests for struct fields whose names collide with Python imports."""

import sys
from dataclasses import dataclass


@dataclass
class Node:
    field: str
    children: list[str]


def test_field_name_collision() -> None:
    """A struct field named 'field' must not shadow dataclasses.field."""
    n: Node = Node("hello", ["a", "b"])
    assert n.field == "hello"
    assert len(n.children) == 2


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_field_name_collision", test_field_name_collision),
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
