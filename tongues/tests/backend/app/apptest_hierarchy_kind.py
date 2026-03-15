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
    assert w.kind == "word"


def test_construct_zero_args() -> None:
    """Empty node has only a kind const field — zero user args."""
    e: Empty = Empty()
    assert e.kind == "empty"


def test_construct_nested() -> None:
    """Nested hierarchy construction with kind fields."""
    p: Pair = Pair(Word("a"), Word("b"))
    assert p.kind == "pair"
    assert isinstance(p.left, Word)


def main() -> None:
    test_construct_kind_omitted()
    test_construct_zero_args()
    test_construct_nested()
    print("ok")


if __name__ == "__main__":
    main()
