"""Hierarchy struct with inherited kind field — #272.

Node declares kind: str as a class-level annotation. Subclasses set
self.kind = "..." in __init__, making it a const field with a default.
The lowering must mark the inherited kind field as has_default=True
so the checker doesn't require it as a constructor argument.
"""

import sys


class Node:
    """Base class with kind annotation."""

    kind: str

    def to_str(self) -> str:
        return self.kind


class Word(Node):
    value: str

    def __init__(self, value: str) -> None:
        self.kind = "word"
        self.value = value

    def to_str(self) -> str:
        return self.value


class Pair(Node):
    left: Node
    right: Node

    def __init__(self, left: Node, right: Node) -> None:
        self.kind = "pair"
        self.left = left
        self.right = right

    def to_str(self) -> str:
        return "(" + self.left.to_str() + " " + self.right.to_str() + ")"


class Empty(Node):
    def __init__(self) -> None:
        self.kind = "empty"

    def to_str(self) -> str:
        return ""


def test_construct_kind_omitted() -> None:
    """Constructor should not require kind as an argument."""
    w: Word = Word("hello")
    assert w.value == "hello"


def test_construct_zero_args() -> None:
    """Empty node has only a kind const field — zero user args."""
    e: Empty = Empty()
    assert isinstance(e, Empty)


def test_construct_nested() -> None:
    """Nested hierarchy construction with kind fields."""
    p: Pair = Pair(Word("a"), Word("b"))
    assert isinstance(p.left, Word)


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_construct_kind_omitted()
        passed += 1
        print("  PASS test_construct_kind_omitted")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_construct_kind_omitted: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_construct_kind_omitted: " + str(e))
    try:
        test_construct_zero_args()
        passed += 1
        print("  PASS test_construct_zero_args")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_construct_zero_args: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_construct_zero_args: " + str(e))
    try:
        test_construct_nested()
        passed += 1
        print("  PASS test_construct_nested")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_construct_nested: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_construct_nested: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
