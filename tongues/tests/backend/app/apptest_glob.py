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


def test_adjacent_classes() -> None:
    assert glob_match("[ab][cd]", "ac")
    assert glob_match("[ab][cd]", "bd")
    assert not glob_match("[ab][cd]", "ab")
    assert not glob_match("[ab][cd]", "cc")


def test_backslash_in_class() -> None:
    assert glob_match("[\\\\]", "\\")
    assert not glob_match("[\\\\]", "a")


def test_escape_in_middle() -> None:
    assert glob_match("a\\*b", "a*b")
    assert not glob_match("a\\*b", "aXb")


def test_mixed_ranges_in_class() -> None:
    assert glob_match("[a-z0-9_]", "m")
    assert glob_match("[a-z0-9_]", "5")
    assert glob_match("[a-z0-9_]", "_")
    assert not glob_match("[a-z0-9_]", "A")
    assert not glob_match("[a-z0-9_]", "-")


def test_negated_range_with_star() -> None:
    assert glob_match("*[!0-9]", "abc")
    assert not glob_match("*[!0-9]", "abc5")
    assert glob_match("*[!0-9]", "5x")


def test_class_then_literal() -> None:
    assert glob_match("[a-z]bc", "xbc")
    assert not glob_match("[a-z]bc", "Xbc")
    assert not glob_match("[a-z]bc", "xbd")


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
    try:
        test_literal_exact()
        passed += 1
        print("  PASS test_literal_exact")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_literal_exact: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_literal_exact: " + str(e))
    try:
        test_literal_mismatch()
        passed += 1
        print("  PASS test_literal_mismatch")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_literal_mismatch: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_literal_mismatch: " + str(e))
    try:
        test_literal_shorter_text()
        passed += 1
        print("  PASS test_literal_shorter_text")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_literal_shorter_text: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_literal_shorter_text: " + str(e))
    try:
        test_literal_longer_text()
        passed += 1
        print("  PASS test_literal_longer_text")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_literal_longer_text: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_literal_longer_text: " + str(e))
    try:
        test_empty_both()
        passed += 1
        print("  PASS test_empty_both")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_empty_both: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_empty_both: " + str(e))
    try:
        test_empty_pattern()
        passed += 1
        print("  PASS test_empty_pattern")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_empty_pattern: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_empty_pattern: " + str(e))
    try:
        test_empty_text()
        passed += 1
        print("  PASS test_empty_text")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_empty_text: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_empty_text: " + str(e))
    try:
        test_question_single()
        passed += 1
        print("  PASS test_question_single")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_question_single: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_question_single: " + str(e))
    try:
        test_question_empty()
        passed += 1
        print("  PASS test_question_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_question_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_question_empty: " + str(e))
    try:
        test_question_two_chars()
        passed += 1
        print("  PASS test_question_two_chars")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_question_two_chars: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_question_two_chars: " + str(e))
    try:
        test_question_middle()
        passed += 1
        print("  PASS test_question_middle")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_question_middle: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_question_middle: " + str(e))
    try:
        test_question_no_match_length()
        passed += 1
        print("  PASS test_question_no_match_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_question_no_match_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_question_no_match_length: " + str(e))
    try:
        test_multiple_questions()
        passed += 1
        print("  PASS test_multiple_questions")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_multiple_questions: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_multiple_questions: " + str(e))
    try:
        test_star_alone()
        passed += 1
        print("  PASS test_star_alone")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_star_alone: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_star_alone: " + str(e))
    try:
        test_star_prefix()
        passed += 1
        print("  PASS test_star_prefix")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_star_prefix: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_star_prefix: " + str(e))
    try:
        test_star_suffix()
        passed += 1
        print("  PASS test_star_suffix")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_star_suffix: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_star_suffix: " + str(e))
    try:
        test_star_middle()
        passed += 1
        print("  PASS test_star_middle")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_star_middle: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_star_middle: " + str(e))
    try:
        test_star_no_match()
        passed += 1
        print("  PASS test_star_no_match")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_star_no_match: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_star_no_match: " + str(e))
    try:
        test_double_star()
        passed += 1
        print("  PASS test_double_star")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_double_star: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_double_star: " + str(e))
    try:
        test_multiple_stars()
        passed += 1
        print("  PASS test_multiple_stars")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_multiple_stars: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_multiple_stars: " + str(e))
    try:
        test_star_backtrack()
        passed += 1
        print("  PASS test_star_backtrack")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_star_backtrack: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_star_backtrack: " + str(e))
    try:
        test_class_single()
        passed += 1
        print("  PASS test_class_single")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_single: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_single: " + str(e))
    try:
        test_class_in_pattern()
        passed += 1
        print("  PASS test_class_in_pattern")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_in_pattern: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_in_pattern: " + str(e))
    try:
        test_class_range()
        passed += 1
        print("  PASS test_class_range")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_range: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_range: " + str(e))
    try:
        test_class_range_digits()
        passed += 1
        print("  PASS test_class_range_digits")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_range_digits: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_range_digits: " + str(e))
    try:
        test_class_multiple_ranges()
        passed += 1
        print("  PASS test_class_multiple_ranges")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_multiple_ranges: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_multiple_ranges: " + str(e))
    try:
        test_class_negated_caret()
        passed += 1
        print("  PASS test_class_negated_caret")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_negated_caret: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_negated_caret: " + str(e))
    try:
        test_class_negated_bang()
        passed += 1
        print("  PASS test_class_negated_bang")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_negated_bang: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_negated_bang: " + str(e))
    try:
        test_class_negated_range()
        passed += 1
        print("  PASS test_class_negated_range")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_negated_range: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_negated_range: " + str(e))
    try:
        test_class_bracket_literal()
        passed += 1
        print("  PASS test_class_bracket_literal")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_bracket_literal: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_bracket_literal: " + str(e))
    try:
        test_class_dash_start()
        passed += 1
        print("  PASS test_class_dash_start")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_dash_start: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_dash_start: " + str(e))
    try:
        test_class_dash_end()
        passed += 1
        print("  PASS test_class_dash_end")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_dash_end: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_dash_end: " + str(e))
    try:
        test_escape_star()
        passed += 1
        print("  PASS test_escape_star")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_escape_star: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_escape_star: " + str(e))
    try:
        test_escape_question()
        passed += 1
        print("  PASS test_escape_question")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_escape_question: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_escape_question: " + str(e))
    try:
        test_escape_bracket()
        passed += 1
        print("  PASS test_escape_bracket")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_escape_bracket: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_escape_bracket: " + str(e))
    try:
        test_escape_backslash()
        passed += 1
        print("  PASS test_escape_backslash")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_escape_backslash: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_escape_backslash: " + str(e))
    try:
        test_escape_in_class()
        passed += 1
        print("  PASS test_escape_in_class")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_escape_in_class: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_escape_in_class: " + str(e))
    try:
        test_star_question()
        passed += 1
        print("  PASS test_star_question")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_star_question: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_star_question: " + str(e))
    try:
        test_star_class()
        passed += 1
        print("  PASS test_star_class")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_star_class: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_star_class: " + str(e))
    try:
        test_complex_combined()
        passed += 1
        print("  PASS test_complex_combined")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_complex_combined: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_complex_combined: " + str(e))
    try:
        test_file_extension()
        passed += 1
        print("  PASS test_file_extension")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_file_extension: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_file_extension: " + str(e))
    try:
        test_prefix_suffix()
        passed += 1
        print("  PASS test_prefix_suffix")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_prefix_suffix: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_prefix_suffix: " + str(e))
    try:
        test_class_star_class()
        passed += 1
        print("  PASS test_class_star_class")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_star_class: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_star_class: " + str(e))
    try:
        test_adjacent_classes()
        passed += 1
        print("  PASS test_adjacent_classes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_adjacent_classes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_adjacent_classes: " + str(e))
    try:
        test_backslash_in_class()
        passed += 1
        print("  PASS test_backslash_in_class")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_backslash_in_class: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_backslash_in_class: " + str(e))
    try:
        test_escape_in_middle()
        passed += 1
        print("  PASS test_escape_in_middle")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_escape_in_middle: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_escape_in_middle: " + str(e))
    try:
        test_mixed_ranges_in_class()
        passed += 1
        print("  PASS test_mixed_ranges_in_class")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_mixed_ranges_in_class: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_mixed_ranges_in_class: " + str(e))
    try:
        test_negated_range_with_star()
        passed += 1
        print("  PASS test_negated_range_with_star")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_negated_range_with_star: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_negated_range_with_star: " + str(e))
    try:
        test_class_then_literal()
        passed += 1
        print("  PASS test_class_then_literal")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_then_literal: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_then_literal: " + str(e))
    try:
        test_unclosed_bracket()
        passed += 1
        print("  PASS test_unclosed_bracket")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_unclosed_bracket: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_unclosed_bracket: " + str(e))
    try:
        test_trailing_backslash()
        passed += 1
        print("  PASS test_trailing_backslash")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_trailing_backslash: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_trailing_backslash: " + str(e))
    try:
        test_backslash_eof_in_class()
        passed += 1
        print("  PASS test_backslash_eof_in_class")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_backslash_eof_in_class: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_backslash_eof_in_class: " + str(e))
    try:
        test_only_stars()
        passed += 1
        print("  PASS test_only_stars")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_only_stars: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_only_stars: " + str(e))
    try:
        test_only_questions()
        passed += 1
        print("  PASS test_only_questions")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_only_questions: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_only_questions: " + str(e))
    try:
        test_star_empty_text()
        passed += 1
        print("  PASS test_star_empty_text")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_star_empty_text: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_star_empty_text: " + str(e))
    try:
        test_question_star_combo()
        passed += 1
        print("  PASS test_question_star_combo")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_question_star_combo: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_question_star_combo: " + str(e))
    try:
        test_star_question_star()
        passed += 1
        print("  PASS test_star_question_star")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_star_question_star: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_star_question_star: " + str(e))
    try:
        test_repeated_pattern()
        passed += 1
        print("  PASS test_repeated_pattern")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_repeated_pattern: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_repeated_pattern: " + str(e))
    try:
        test_class_with_star()
        passed += 1
        print("  PASS test_class_with_star")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_with_star: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_with_star: " + str(e))
    try:
        test_class_with_question()
        passed += 1
        print("  PASS test_class_with_question")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_class_with_question: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_class_with_question: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
