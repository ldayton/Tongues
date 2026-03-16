"""String iteration: for ch in s should work with append."""

import sys


def test_string_iter_append() -> None:
    """Iterating a string and appending chars to a list."""
    s: str = "hello"
    chars: list[str] = []
    for ch in s:
        chars.append(ch)
    assert len(chars) == 5
    assert chars[0] == "h"
    assert chars[4] == "o"


def test_string_iter_nested() -> None:
    """Nested: for over list of strings, inner for over chars."""
    lines: list[str] = ["ab", "cd"]
    all_chars: list[str] = []
    for line in lines:
        for ch in line:
            all_chars.append(ch)
    assert len(all_chars) == 4
    assert all_chars[0] == "a"
    assert all_chars[3] == "d"


def test_string_iter_while_nested() -> None:
    """While loop containing for-in-string."""
    source: str = "abc"
    pos: int = 0
    chars: list[str] = []
    while pos < len(source):
        line: str = source[pos : pos + 1]
        for ch in line:
            chars.append(ch)
        pos += 1
    assert len(chars) == 3


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_string_iter_append()
        passed += 1
        print("  PASS test_string_iter_append")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_iter_append: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_iter_append: " + str(e))
    try:
        test_string_iter_nested()
        passed += 1
        print("  PASS test_string_iter_nested")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_iter_nested: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_iter_nested: " + str(e))
    try:
        test_string_iter_while_nested()
        passed += 1
        print("  PASS test_string_iter_while_nested")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_iter_while_nested: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_iter_while_nested: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
