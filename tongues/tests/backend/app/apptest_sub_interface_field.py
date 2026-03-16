"""Common field access on interface with sub-interface variants."""

import sys


class Node:
    kind: str

    def __init__(self) -> None:
        pass


class ArithNode(Node):
    def __init__(self) -> None:
        pass


class ArithNumber(ArithNode):
    value: int

    def __init__(self, value: int) -> None:
        self.kind = "number"
        self.value = value


class Word(Node):
    text: str

    def __init__(self, text: str) -> None:
        self.kind = "word"
        self.text = text


def get_kind(n: Node) -> str:
    return n.kind


def test_common_field_through_sub_interface() -> None:
    """Access kind field on Node when ArithNode is a sub-interface."""
    w: Node = Word("hello")
    assert get_kind(w) == "word"
    a: Node = ArithNumber(42)
    assert get_kind(a) == "number"


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_common_field_through_sub_interface()
        passed += 1
        print("  PASS test_common_field_through_sub_interface")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_common_field_through_sub_interface: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_common_field_through_sub_interface: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
