"""Levenshtein edit distance tests — known vectors and properties."""

import sys

from lib.levenshtein import levenshtein


# -- Identity and empty --


def test_both_empty() -> None:
    assert levenshtein("", "") == 0


def test_first_empty() -> None:
    assert levenshtein("", "abc") == 3


def test_second_empty() -> None:
    assert levenshtein("abc", "") == 3


def test_equal_strings() -> None:
    assert levenshtein("hello", "hello") == 0


def test_single_equal() -> None:
    assert levenshtein("a", "a") == 0


# -- Single operations --


def test_single_insert() -> None:
    assert levenshtein("ac", "abc") == 1


def test_single_delete() -> None:
    assert levenshtein("abc", "ac") == 1


def test_single_substitute() -> None:
    assert levenshtein("abc", "axc") == 1


def test_single_char_different() -> None:
    assert levenshtein("a", "b") == 1


# -- Classic examples --


def test_kitten_sitting() -> None:
    assert levenshtein("kitten", "sitting") == 3


def test_saturday_sunday() -> None:
    assert levenshtein("saturday", "sunday") == 3


def test_rosettacode() -> None:
    assert levenshtein("rosettacode", "raisethysword") == 8


# -- Swap (no transposition in pure Levenshtein) --


def test_swap_ab_ba() -> None:
    """Pure Levenshtein: swap costs 2 (sub+sub), not 1."""
    assert levenshtein("ab", "ba") == 2


def test_swap_abc_bac() -> None:
    assert levenshtein("abc", "bac") == 2


# -- Prefix / suffix --


def test_prefix() -> None:
    assert levenshtein("abc", "abcdef") == 3


def test_suffix() -> None:
    assert levenshtein("def", "abcdef") == 3


def test_common_prefix_different_suffix() -> None:
    assert levenshtein("abcX", "abcY") == 1


def test_different_prefix_common_suffix() -> None:
    assert levenshtein("Xabc", "Yabc") == 1


# -- Lengths --


def test_single_vs_long() -> None:
    assert levenshtein("a", "abcde") == 4


def test_long_vs_single() -> None:
    assert levenshtein("abcde", "a") == 4


def test_completely_different() -> None:
    assert levenshtein("abc", "xyz") == 3


def test_longer_completely_different() -> None:
    assert levenshtein("abcd", "efgh") == 4


# -- Symmetry --


def test_symmetry_kitten() -> None:
    assert levenshtein("kitten", "sitting") == levenshtein("sitting", "kitten")


def test_symmetry_empty() -> None:
    assert levenshtein("", "test") == levenshtein("test", "")


def test_symmetry_different() -> None:
    assert levenshtein("abc", "xyz") == levenshtein("xyz", "abc")


# -- Repeated characters --


def test_repeated_a() -> None:
    assert levenshtein("aaa", "a") == 2


def test_repeated_vs_empty() -> None:
    assert levenshtein("aaaa", "") == 4


def test_repeated_insert_middle() -> None:
    assert levenshtein("aaa", "aba") == 1


# -- Case sensitivity --


def test_case_sensitive() -> None:
    assert levenshtein("ABC", "abc") == 3


def test_case_one_char() -> None:
    assert levenshtein("A", "a") == 1


# -- Longer strings --


def test_longer_strings() -> None:
    assert levenshtein("intention", "execution") == 5


def test_insert_at_start() -> None:
    assert levenshtein("abc", "xabc") == 1


def test_insert_at_end() -> None:
    assert levenshtein("abc", "abcx") == 1


def test_delete_at_start() -> None:
    assert levenshtein("xabc", "abc") == 1


def test_delete_at_end() -> None:
    assert levenshtein("abcx", "abc") == 1


# -- Unicode --


def test_unicode_equal() -> None:
    assert levenshtein("caf\u00e9", "caf\u00e9") == 0


def test_unicode_one_sub() -> None:
    assert levenshtein("caf\u00e9", "cafe") == 1


def test_unicode_cjk() -> None:
    assert levenshtein("\u4e2d\u6587", "\u4e2d\u56fd") == 1


def test_unicode_vs_empty() -> None:
    assert levenshtein("", "\u00e9\u00e8") == 2


# -- Triangle inequality --


def test_triangle_inequality() -> None:
    """d(a,c) <= d(a,b) + d(b,c)."""
    ab: int = levenshtein("abc", "axc")
    bc: int = levenshtein("axc", "xyz")
    ac: int = levenshtein("abc", "xyz")
    assert ac <= ab + bc


def test_longer_strings2() -> None:
    assert (
        levenshtein(
            "pneumonoultramicroscopicsilicovolcanoconiosis", "ultramicroscopically"
        )
        == 27
    )


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_both_empty", test_both_empty),
        ("test_first_empty", test_first_empty),
        ("test_second_empty", test_second_empty),
        ("test_equal_strings", test_equal_strings),
        ("test_single_equal", test_single_equal),
        ("test_single_insert", test_single_insert),
        ("test_single_delete", test_single_delete),
        ("test_single_substitute", test_single_substitute),
        ("test_single_char_different", test_single_char_different),
        ("test_kitten_sitting", test_kitten_sitting),
        ("test_saturday_sunday", test_saturday_sunday),
        ("test_rosettacode", test_rosettacode),
        ("test_swap_ab_ba", test_swap_ab_ba),
        ("test_swap_abc_bac", test_swap_abc_bac),
        ("test_prefix", test_prefix),
        ("test_suffix", test_suffix),
        ("test_common_prefix_different_suffix", test_common_prefix_different_suffix),
        ("test_different_prefix_common_suffix", test_different_prefix_common_suffix),
        ("test_single_vs_long", test_single_vs_long),
        ("test_long_vs_single", test_long_vs_single),
        ("test_completely_different", test_completely_different),
        ("test_longer_completely_different", test_longer_completely_different),
        ("test_symmetry_kitten", test_symmetry_kitten),
        ("test_symmetry_empty", test_symmetry_empty),
        ("test_symmetry_different", test_symmetry_different),
        ("test_repeated_a", test_repeated_a),
        ("test_repeated_vs_empty", test_repeated_vs_empty),
        ("test_repeated_insert_middle", test_repeated_insert_middle),
        ("test_case_sensitive", test_case_sensitive),
        ("test_case_one_char", test_case_one_char),
        ("test_insert_at_start", test_insert_at_start),
        ("test_insert_at_end", test_insert_at_end),
        ("test_delete_at_start", test_delete_at_start),
        ("test_delete_at_end", test_delete_at_end),
        ("test_unicode_equal", test_unicode_equal),
        ("test_unicode_one_sub", test_unicode_one_sub),
        ("test_unicode_cjk", test_unicode_cjk),
        ("test_unicode_vs_empty", test_unicode_vs_empty),
        ("test_triangle_inequality", test_triangle_inequality),
        ("test_longer_strings", test_longer_strings),
        ("test_longer_strings2", test_longer_strings2),
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
