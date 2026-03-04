"""Glob pattern matching tests — wildcards, classes, escaping, and edge cases."""

import sys

from lib.glob import glob_match
from lib.glob import GlobError


# -- Literal matching --


def test_literal_exact() -> None:
    assert glob_match("abc", "abc")


def test_literal_mismatch() -> None:
    assert not glob_match("abc", "abd")


def test_literal_shorter_text() -> None:
    assert not glob_match("abc", "ab")


def test_literal_longer_text() -> None:
    assert not glob_match("ab", "abc")


def test_empty_both() -> None:
    assert glob_match("", "")


def test_empty_pattern() -> None:
    assert not glob_match("", "a")


def test_empty_text() -> None:
    assert not glob_match("a", "")


# -- ? wildcard --


def test_question_single() -> None:
    assert glob_match("?", "a")


def test_question_empty() -> None:
    assert not glob_match("?", "")


def test_question_two_chars() -> None:
    assert not glob_match("?", "ab")


def test_question_middle() -> None:
    assert glob_match("a?c", "abc")
    assert glob_match("a?c", "axc")


def test_question_no_match_length() -> None:
    assert not glob_match("a?c", "abbc")


def test_multiple_questions() -> None:
    assert glob_match("???", "abc")
    assert not glob_match("???", "ab")
    assert not glob_match("???", "abcd")


# -- * wildcard --


def test_star_alone() -> None:
    assert glob_match("*", "")
    assert glob_match("*", "abc")
    assert glob_match("*", "anything at all")


def test_star_prefix() -> None:
    assert glob_match("*bc", "abc")
    assert glob_match("*bc", "bc")
    assert glob_match("*bc", "xxbc")


def test_star_suffix() -> None:
    assert glob_match("ab*", "ab")
    assert glob_match("ab*", "abc")
    assert glob_match("ab*", "abxyz")


def test_star_middle() -> None:
    assert glob_match("a*c", "ac")
    assert glob_match("a*c", "abc")
    assert glob_match("a*c", "aXYZc")


def test_star_no_match() -> None:
    assert not glob_match("a*c", "ab")
    assert not glob_match("*c", "ab")
    assert not glob_match("a*", "ba")


def test_double_star() -> None:
    """** behaves same as * (no recursive directory matching)."""
    assert glob_match("**", "anything")
    assert glob_match("a**b", "aXXb")


def test_multiple_stars() -> None:
    assert glob_match("a*b*c", "abc")
    assert glob_match("a*b*c", "aXbYc")
    assert glob_match("a*b*c", "aXXbYYc")


def test_star_backtrack() -> None:
    """Star must backtrack to find the right match."""
    assert glob_match("*a", "aa")
    assert glob_match("*ab", "aab")
    assert glob_match("a*a", "aba")
    assert glob_match("a*a", "abba")


# -- Character classes --


def test_class_single() -> None:
    assert glob_match("[abc]", "a")
    assert glob_match("[abc]", "b")
    assert glob_match("[abc]", "c")
    assert not glob_match("[abc]", "d")


def test_class_in_pattern() -> None:
    assert glob_match("h[ae]llo", "hello")
    assert glob_match("h[ae]llo", "hallo")
    assert not glob_match("h[ae]llo", "hillo")


def test_class_range() -> None:
    assert glob_match("[a-z]", "m")
    assert glob_match("[a-z]", "a")
    assert glob_match("[a-z]", "z")
    assert not glob_match("[a-z]", "A")
    assert not glob_match("[a-z]", "0")


def test_class_range_digits() -> None:
    assert glob_match("[0-9]", "5")
    assert not glob_match("[0-9]", "a")


def test_class_multiple_ranges() -> None:
    assert glob_match("[a-zA-Z]", "m")
    assert glob_match("[a-zA-Z]", "M")
    assert not glob_match("[a-zA-Z]", "5")


def test_class_negated_caret() -> None:
    assert glob_match("[^abc]", "d")
    assert not glob_match("[^abc]", "a")


def test_class_negated_bang() -> None:
    assert glob_match("[!abc]", "d")
    assert not glob_match("[!abc]", "a")


def test_class_negated_range() -> None:
    assert glob_match("[^a-z]", "A")
    assert not glob_match("[^a-z]", "m")


def test_class_bracket_literal() -> None:
    """']' as first char in class is literal."""
    assert glob_match("[]a]", "]")
    assert glob_match("[]a]", "a")
    assert not glob_match("[]a]", "b")


def test_class_dash_start() -> None:
    """'-' at start of class is literal."""
    assert glob_match("[-a]", "-")
    assert glob_match("[-a]", "a")


def test_class_dash_end() -> None:
    """'-' before ']' is literal."""
    assert glob_match("[a-]", "a")
    assert glob_match("[a-]", "-")
    assert not glob_match("[a-]", "b")


# -- Escape sequences --


def test_escape_star() -> None:
    assert glob_match("\\*", "*")
    assert not glob_match("\\*", "a")


def test_escape_question() -> None:
    assert glob_match("\\?", "?")
    assert not glob_match("\\?", "a")


def test_escape_bracket() -> None:
    assert glob_match("\\[", "[")
    assert not glob_match("\\[", "a")


def test_escape_backslash() -> None:
    assert glob_match("\\\\", "\\")


def test_escape_in_class() -> None:
    assert glob_match("[\\*a]", "*")
    assert glob_match("[\\*a]", "a")
    assert not glob_match("[\\*a]", "b")


# -- Combined patterns --


def test_star_question() -> None:
    assert glob_match("*?", "a")
    assert glob_match("*?", "abc")
    assert not glob_match("*?", "")


def test_star_class() -> None:
    assert glob_match("*[0-9]", "abc5")
    assert not glob_match("*[0-9]", "abcd")


def test_complex_combined() -> None:
    assert glob_match("h?l*[!0-9]", "hello")
    assert glob_match("h?l*[!0-9]", "helloworld")
    assert not glob_match("h?l*[!0-9]", "hello5")


def test_file_extension() -> None:
    assert glob_match("*.txt", "readme.txt")
    assert glob_match("*.txt", ".txt")
    assert not glob_match("*.txt", "readme.md")


def test_prefix_suffix() -> None:
    assert glob_match("test_*_spec", "test_foo_spec")
    assert glob_match("test_*_spec", "test__spec")
    assert not glob_match("test_*_spec", "test_foo")


def test_class_star_class() -> None:
    assert glob_match("[abc]*[xyz]", "ax")
    assert glob_match("[abc]*[xyz]", "bMMMz")
    assert not glob_match("[abc]*[xyz]", "dMx")


# -- Error cases --


def test_unclosed_bracket() -> None:
    try:
        glob_match("[abc", "a")
        assert False, "expected GlobError"
    except GlobError:
        pass


def test_trailing_backslash() -> None:
    try:
        glob_match("\\", "x")
        assert False, "expected GlobError"
    except GlobError:
        pass


def test_backslash_eof_in_class() -> None:
    try:
        glob_match("[\\", "a")
        assert False, "expected GlobError"
    except GlobError:
        pass


# -- Edge cases --


def test_only_stars() -> None:
    assert glob_match("***", "anything")
    assert glob_match("***", "")


def test_only_questions() -> None:
    assert glob_match("?????", "abcde")
    assert not glob_match("?????", "abcd")


def test_star_empty_text() -> None:
    assert glob_match("*", "")


def test_question_star_combo() -> None:
    assert glob_match("?*", "a")
    assert glob_match("?*", "abc")
    assert not glob_match("?*", "")


def test_star_question_star() -> None:
    assert glob_match("*?*", "x")
    assert glob_match("*?*", "abc")
    assert not glob_match("*?*", "")


def test_repeated_pattern() -> None:
    assert glob_match("abab", "abab")
    assert not glob_match("abab", "abba")


def test_class_with_star() -> None:
    """'*' inside class is literal."""
    assert glob_match("[*]", "*")
    assert not glob_match("[*]", "a")


def test_class_with_question() -> None:
    """'?' inside class is literal."""
    assert glob_match("[?]", "?")
    assert not glob_match("[?]", "a")


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_literal_exact", test_literal_exact),
        ("test_literal_mismatch", test_literal_mismatch),
        ("test_literal_shorter_text", test_literal_shorter_text),
        ("test_literal_longer_text", test_literal_longer_text),
        ("test_empty_both", test_empty_both),
        ("test_empty_pattern", test_empty_pattern),
        ("test_empty_text", test_empty_text),
        ("test_question_single", test_question_single),
        ("test_question_empty", test_question_empty),
        ("test_question_two_chars", test_question_two_chars),
        ("test_question_middle", test_question_middle),
        ("test_question_no_match_length", test_question_no_match_length),
        ("test_multiple_questions", test_multiple_questions),
        ("test_star_alone", test_star_alone),
        ("test_star_prefix", test_star_prefix),
        ("test_star_suffix", test_star_suffix),
        ("test_star_middle", test_star_middle),
        ("test_star_no_match", test_star_no_match),
        ("test_double_star", test_double_star),
        ("test_multiple_stars", test_multiple_stars),
        ("test_star_backtrack", test_star_backtrack),
        ("test_class_single", test_class_single),
        ("test_class_in_pattern", test_class_in_pattern),
        ("test_class_range", test_class_range),
        ("test_class_range_digits", test_class_range_digits),
        ("test_class_multiple_ranges", test_class_multiple_ranges),
        ("test_class_negated_caret", test_class_negated_caret),
        ("test_class_negated_bang", test_class_negated_bang),
        ("test_class_negated_range", test_class_negated_range),
        ("test_class_bracket_literal", test_class_bracket_literal),
        ("test_class_dash_start", test_class_dash_start),
        ("test_class_dash_end", test_class_dash_end),
        ("test_escape_star", test_escape_star),
        ("test_escape_question", test_escape_question),
        ("test_escape_bracket", test_escape_bracket),
        ("test_escape_backslash", test_escape_backslash),
        ("test_escape_in_class", test_escape_in_class),
        ("test_star_question", test_star_question),
        ("test_star_class", test_star_class),
        ("test_complex_combined", test_complex_combined),
        ("test_file_extension", test_file_extension),
        ("test_prefix_suffix", test_prefix_suffix),
        ("test_class_star_class", test_class_star_class),
        ("test_unclosed_bracket", test_unclosed_bracket),
        ("test_trailing_backslash", test_trailing_backslash),
        ("test_backslash_eof_in_class", test_backslash_eof_in_class),
        ("test_only_stars", test_only_stars),
        ("test_only_questions", test_only_questions),
        ("test_star_empty_text", test_star_empty_text),
        ("test_question_star_combo", test_question_star_combo),
        ("test_star_question_star", test_star_question_star),
        ("test_repeated_pattern", test_repeated_pattern),
        ("test_class_with_star", test_class_with_star),
        ("test_class_with_question", test_class_with_question),
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
