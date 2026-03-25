"""Unicode string operations — indexing, length, slicing, iteration with non-ASCII."""

import sys


def test_len_bmp() -> None:
    """len() counts codepoints, not bytes or UTF-16 code units."""
    assert len("café") == 4
    assert len("héllo") == 5
    assert len("\u00e9") == 1
    assert len("\u03b1\u03b2\u03b3") == 3


def test_len_astral() -> None:
    """len() with characters outside BMP (surrogate pairs in UTF-16)."""
    assert len("\U0001f600") == 1
    assert len("a\U0001f600b") == 3
    assert len("\U0001f600\U0001f601") == 2


def test_index_bmp() -> None:
    """Indexing into string with BMP multibyte characters."""
    s: str = "caf\u00e9"
    assert s[0] == "c"
    assert s[3] == "\u00e9"
    assert s[2] == "f"


def test_index_astral() -> None:
    """Indexing into string with astral plane characters."""
    s: str = "a\U0001f600b"
    assert s[0] == "a"
    assert s[1] == "\U0001f600"
    assert s[2] == "b"


def test_slice_bmp() -> None:
    """Slicing string with BMP multibyte characters."""
    s: str = "\u03b1\u03b2\u03b3\u03b4"
    assert s[1:3] == "\u03b2\u03b3"
    assert s[:2] == "\u03b1\u03b2"
    assert s[2:] == "\u03b3\u03b4"


def test_slice_astral() -> None:
    """Slicing string with astral plane characters."""
    s: str = "x\U0001f600y\U0001f601z"
    assert s[1:4] == "\U0001f600y\U0001f601"
    assert s[:2] == "x\U0001f600"
    assert s[3:] == "\U0001f601z"


def test_iter_bmp() -> None:
    """Iterating over string with BMP multibyte characters yields codepoints."""
    chars: list[str] = []
    for ch in "caf\u00e9":
        chars.append(ch)
    assert len(chars) == 4
    assert chars[3] == "\u00e9"


def test_iter_astral() -> None:
    """Iterating over string with astral plane characters yields whole codepoints."""
    chars: list[str] = []
    for ch in "a\U0001f600b":
        chars.append(ch)
    assert len(chars) == 3
    assert chars[0] == "a"
    assert chars[1] == "\U0001f600"
    assert chars[2] == "b"


def test_find_bmp() -> None:
    """find() returns codepoint index, not byte/code-unit index."""
    s: str = "caf\u00e9s"
    assert s.find("s") == 4
    assert s.find("\u00e9") == 3


def test_find_astral() -> None:
    """find() with astral plane characters."""
    s: str = "a\U0001f600bc"
    assert s.find("b") == 2
    assert s.find("c") == 3
    assert s.find("\U0001f600") == 1


def test_startswith_bmp() -> None:
    """startswith() with BMP multibyte prefix."""
    assert "caf\u00e9".startswith("caf\u00e9")
    assert "caf\u00e9".startswith("caf")
    assert not "caf\u00e9".startswith("\u00e9")


def test_count_bmp() -> None:
    """count() with BMP multibyte characters."""
    s: str = "\u00e9t\u00e9"
    assert s.count("\u00e9") == 2
    assert s.count("t") == 1


def test_concat_preserves_unicode() -> None:
    """String concatenation preserves multibyte characters."""
    a: str = "caf"
    b: str = "\u00e9"
    c: str = a + b
    assert c == "caf\u00e9"
    assert len(c) == 4


def test_builder_bmp() -> None:
    """String building in a loop with BMP characters."""
    parts: list[str] = ["\u03b1", "\u03b2", "\u03b3"]
    result: str = ""
    for p in parts:
        result = result + p
    assert result == "\u03b1\u03b2\u03b3"
    assert len(result) == 3


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_len_bmp()
        passed += 1
        print("  PASS test_len_bmp")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_len_bmp: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_len_bmp: " + str(e))
    try:
        test_len_astral()
        passed += 1
        print("  PASS test_len_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_len_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_len_astral: " + str(e))
    try:
        test_index_bmp()
        passed += 1
        print("  PASS test_index_bmp")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_index_bmp: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_index_bmp: " + str(e))
    try:
        test_index_astral()
        passed += 1
        print("  PASS test_index_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_index_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_index_astral: " + str(e))
    try:
        test_slice_bmp()
        passed += 1
        print("  PASS test_slice_bmp")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_slice_bmp: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_slice_bmp: " + str(e))
    try:
        test_slice_astral()
        passed += 1
        print("  PASS test_slice_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_slice_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_slice_astral: " + str(e))
    try:
        test_iter_bmp()
        passed += 1
        print("  PASS test_iter_bmp")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_iter_bmp: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_iter_bmp: " + str(e))
    try:
        test_iter_astral()
        passed += 1
        print("  PASS test_iter_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_iter_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_iter_astral: " + str(e))
    try:
        test_find_bmp()
        passed += 1
        print("  PASS test_find_bmp")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_find_bmp: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_find_bmp: " + str(e))
    try:
        test_find_astral()
        passed += 1
        print("  PASS test_find_astral")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_find_astral: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_find_astral: " + str(e))
    try:
        test_startswith_bmp()
        passed += 1
        print("  PASS test_startswith_bmp")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_startswith_bmp: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_startswith_bmp: " + str(e))
    try:
        test_count_bmp()
        passed += 1
        print("  PASS test_count_bmp")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_count_bmp: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_count_bmp: " + str(e))
    try:
        test_concat_preserves_unicode()
        passed += 1
        print("  PASS test_concat_preserves_unicode")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_concat_preserves_unicode: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_concat_preserves_unicode: " + str(e))
    try:
        test_builder_bmp()
        passed += 1
        print("  PASS test_builder_bmp")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_builder_bmp: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_builder_bmp: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
