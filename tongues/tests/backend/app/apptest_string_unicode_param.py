"""Codepoint-correct string ops when astral strings flow through parameters and locals.

These tests exercise the case where a string containing non-BMP characters
(surrogate pairs in UTF-16) is passed as an argument or assigned to a local
variable — not embedded as a literal.  All string operations must use
codepoint semantics consistently, regardless of how the string arrived.
"""

import sys


def _len_via_param(s: str) -> int:
    return len(s)


def _index_via_param(s: str, i: int) -> str:
    return s[i]


def _slice_via_param(s: str, lo: int, hi: int) -> str:
    return s[lo:hi]


def _find_via_param(s: str, sub: str) -> int:
    return s.find(sub)


def _rfind_via_param(s: str, sub: str) -> int:
    return s.rfind(sub)


def _startswith_via_param(s: str, prefix: str) -> bool:
    return s.startswith(prefix)


def _startswith_at_via_param(s: str, prefix: str, start: int) -> bool:
    return s.startswith(prefix, start)


def test_len_param_astral() -> None:
    """len() on a parameter holding an astral string counts codepoints."""
    assert _len_via_param("\U0001f600") == 1
    assert _len_via_param("a\U0001f600b") == 3
    assert _len_via_param("\U0001f600\U0001f601\U0001f602") == 3


def test_index_param_astral() -> None:
    """Indexing a parameter holding an astral string is codepoint-based."""
    assert _index_via_param("a\U0001f600b", 0) == "a"
    assert _index_via_param("a\U0001f600b", 1) == "\U0001f600"
    assert _index_via_param("a\U0001f600b", 2) == "b"


def test_slice_param_astral() -> None:
    """Slicing a parameter with astral characters uses codepoint offsets."""
    assert _slice_via_param("a\U0001f600b\U0001f601c", 1, 4) == "\U0001f600b\U0001f601"
    assert _slice_via_param("a\U0001f600b", 0, 2) == "a\U0001f600"
    assert (
        _slice_via_param("\U0001f600\U0001f601\U0001f602", 1, 3)
        == "\U0001f601\U0001f602"
    )


def test_find_param_astral() -> None:
    """find() on a parameter with astral characters returns codepoint index."""
    assert _find_via_param("a\U0001f600bc", "b") == 2
    assert _find_via_param("a\U0001f600bc", "\U0001f600") == 1
    assert _find_via_param("a\U0001f600bc", "c") == 3
    assert _find_via_param("a\U0001f600bc", "z") == -1


def test_rfind_param_astral() -> None:
    """rfind() on a parameter with astral characters returns codepoint index."""
    assert _rfind_via_param("a\U0001f600b\U0001f600c", "\U0001f600") == 3
    assert _rfind_via_param("a\U0001f600b\U0001f600c", "a") == 0
    assert _rfind_via_param("a\U0001f600b\U0001f600c", "c") == 4


def test_startswith_param_astral() -> None:
    """startswith() on a parameter with astral characters."""
    assert _startswith_via_param("\U0001f600abc", "\U0001f600")
    assert not _startswith_via_param("a\U0001f600bc", "\U0001f600")
    assert _startswith_via_param("a\U0001f600bc", "a")


def test_startswith_at_param_astral() -> None:
    """startswith(prefix, start) with codepoint offset on an astral string param."""
    assert _startswith_at_via_param("a\U0001f600bc", "\U0001f600", 1)
    assert _startswith_at_via_param("a\U0001f600bc", "b", 2)
    assert not _startswith_at_via_param("a\U0001f600bc", "a", 1)


def _scan_loop(source: str) -> list[str]:
    """Simulate a lexer-style character scan loop over an astral string."""
    chars: list[str] = []
    pos: int = 0
    n: int = len(source)
    while pos < n:
        chars.append(source[pos])
        pos += 1
    return chars


def test_scan_loop_astral() -> None:
    """A len/index scan loop over a parameter is codepoint-consistent."""
    chars: list[str] = _scan_loop("a\U0001f618b")
    assert len(chars) == 3
    assert chars[0] == "a"
    assert chars[1] == "\U0001f618"
    assert chars[2] == "b"


def test_scan_loop_multiple_astral() -> None:
    """Scan loop with multiple adjacent astral characters."""
    chars: list[str] = _scan_loop("\U0001f600\U0001f601\U0001f602")
    assert len(chars) == 3
    assert chars[0] == "\U0001f600"
    assert chars[1] == "\U0001f601"
    assert chars[2] == "\U0001f602"


def _local_reassign(s: str) -> int:
    """len() on a local that was assigned from a parameter."""
    t: str = s
    return len(t)


def test_local_reassigned_from_param() -> None:
    """A local variable assigned from a parameter inherits codepoint semantics."""
    assert _local_reassign("a\U0001f600b") == 3


class Lexer:
    """Minimal lexer to test field/parameter consistency."""

    def __init__(self, source: str) -> None:
        self.source: str = source
        self.length: int = len(source)
        self.pos: int = 0

    def peek(self) -> str:
        return self.source[self.pos]

    def advance(self) -> str:
        ch: str = self.source[self.pos]
        self.pos += 1
        return ch

    def remaining(self) -> int:
        return self.length - self.pos

    def rest(self) -> str:
        return self.source[self.pos :]

    def collect_all(self) -> list[str]:
        result: list[str] = []
        while self.pos < self.length:
            result.append(self.advance())
        return result


def test_lexer_astral() -> None:
    """Lexer-style class with astral input: field and param ops agree."""
    lex: Lexer = Lexer("a\U0001f618b")
    assert lex.length == 3
    assert lex.remaining() == 3
    assert lex.peek() == "a"
    assert lex.advance() == "a"
    assert lex.remaining() == 2
    assert lex.peek() == "\U0001f618"
    assert lex.advance() == "\U0001f618"
    assert lex.remaining() == 1
    assert lex.rest() == "b"
    assert lex.advance() == "b"
    assert lex.remaining() == 0


def test_lexer_collect_all_astral() -> None:
    """Lexer collecting all characters from an astral string."""
    chars: list[str] = Lexer("\U0001f600x\U0001f601").collect_all()
    assert len(chars) == 3
    assert chars[0] == "\U0001f600"
    assert chars[1] == "x"
    assert chars[2] == "\U0001f601"


def test_lexer_only_astral() -> None:
    """Lexer with a string that is entirely astral characters."""
    lex: Lexer = Lexer("\U0001f618\U0001f60d")
    assert lex.length == 2
    chars: list[str] = lex.collect_all()
    assert len(chars) == 2
    assert chars[0] == "\U0001f618"
    assert chars[1] == "\U0001f60d"


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_len_param_astral()
        passed += 1
        print("  PASS test_len_param_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_len_param_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_len_param_astral: " + str(e))
    try:
        test_index_param_astral()
        passed += 1
        print("  PASS test_index_param_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_index_param_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_index_param_astral: " + str(e))
    try:
        test_slice_param_astral()
        passed += 1
        print("  PASS test_slice_param_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_slice_param_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_slice_param_astral: " + str(e))
    try:
        test_find_param_astral()
        passed += 1
        print("  PASS test_find_param_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_find_param_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_find_param_astral: " + str(e))
    try:
        test_rfind_param_astral()
        passed += 1
        print("  PASS test_rfind_param_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_rfind_param_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_rfind_param_astral: " + str(e))
    try:
        test_startswith_param_astral()
        passed += 1
        print("  PASS test_startswith_param_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_startswith_param_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_startswith_param_astral: " + str(e))
    try:
        test_startswith_at_param_astral()
        passed += 1
        print("  PASS test_startswith_at_param_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_startswith_at_param_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_startswith_at_param_astral: " + str(e))
    try:
        test_scan_loop_astral()
        passed += 1
        print("  PASS test_scan_loop_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_scan_loop_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_scan_loop_astral: " + str(e))
    try:
        test_scan_loop_multiple_astral()
        passed += 1
        print("  PASS test_scan_loop_multiple_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_scan_loop_multiple_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_scan_loop_multiple_astral: " + str(e))
    try:
        test_local_reassigned_from_param()
        passed += 1
        print("  PASS test_local_reassigned_from_param")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_local_reassigned_from_param: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_local_reassigned_from_param: " + str(e))
    try:
        test_lexer_astral()
        passed += 1
        print("  PASS test_lexer_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_lexer_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_lexer_astral: " + str(e))
    try:
        test_lexer_collect_all_astral()
        passed += 1
        print("  PASS test_lexer_collect_all_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_lexer_collect_all_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_lexer_collect_all_astral: " + str(e))
    try:
        test_lexer_only_astral()
        passed += 1
        print("  PASS test_lexer_only_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_lexer_only_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_lexer_only_astral: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
