"""Hierarchy struct with kind const field — #272."""

import sys


class Node:
    def __init__(self) -> None:
        pass


class Word(Node):
    def __init__(self, value: str) -> None:
        self.kind = "word"
        self.value = value


class Pair(Node):
    def __init__(self, left: Node, right: Node) -> None:
        self.kind = "pair"
        self.left = left
        self.right = right


def test_construct_kind_omitted() -> None:
    """Constructor should not require kind as an argument."""
    w: Word = Word("hello")
    assert w.value == "hello"


def test_construct_nested() -> None:
    """Nested hierarchy construction with kind fields."""
    p: Pair = Pair(Word("a"), Word("b"))
    assert isinstance(p.left, Word)


def main() -> None:
    test_construct_kind_omitted()
    test_construct_nested()
    print("ok")


if __name__ == "__main__":
    main()
