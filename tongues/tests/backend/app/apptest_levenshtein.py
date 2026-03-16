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
    try:
        test_both_empty()
        passed += 1
        print("  PASS test_both_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_both_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_both_empty: " + str(e))
    try:
        test_first_empty()
        passed += 1
        print("  PASS test_first_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_first_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_first_empty: " + str(e))
    try:
        test_second_empty()
        passed += 1
        print("  PASS test_second_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_second_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_second_empty: " + str(e))
    try:
        test_equal_strings()
        passed += 1
        print("  PASS test_equal_strings")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_equal_strings: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_equal_strings: " + str(e))
    try:
        test_single_equal()
        passed += 1
        print("  PASS test_single_equal")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_single_equal: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_single_equal: " + str(e))
    try:
        test_single_insert()
        passed += 1
        print("  PASS test_single_insert")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_single_insert: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_single_insert: " + str(e))
    try:
        test_single_delete()
        passed += 1
        print("  PASS test_single_delete")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_single_delete: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_single_delete: " + str(e))
    try:
        test_single_substitute()
        passed += 1
        print("  PASS test_single_substitute")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_single_substitute: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_single_substitute: " + str(e))
    try:
        test_single_char_different()
        passed += 1
        print("  PASS test_single_char_different")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_single_char_different: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_single_char_different: " + str(e))
    try:
        test_kitten_sitting()
        passed += 1
        print("  PASS test_kitten_sitting")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_kitten_sitting: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_kitten_sitting: " + str(e))
    try:
        test_saturday_sunday()
        passed += 1
        print("  PASS test_saturday_sunday")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_saturday_sunday: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_saturday_sunday: " + str(e))
    try:
        test_rosettacode()
        passed += 1
        print("  PASS test_rosettacode")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_rosettacode: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_rosettacode: " + str(e))
    try:
        test_swap_ab_ba()
        passed += 1
        print("  PASS test_swap_ab_ba")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_swap_ab_ba: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_swap_ab_ba: " + str(e))
    try:
        test_swap_abc_bac()
        passed += 1
        print("  PASS test_swap_abc_bac")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_swap_abc_bac: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_swap_abc_bac: " + str(e))
    try:
        test_prefix()
        passed += 1
        print("  PASS test_prefix")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_prefix: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_prefix: " + str(e))
    try:
        test_suffix()
        passed += 1
        print("  PASS test_suffix")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_suffix: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_suffix: " + str(e))
    try:
        test_common_prefix_different_suffix()
        passed += 1
        print("  PASS test_common_prefix_different_suffix")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_common_prefix_different_suffix: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_common_prefix_different_suffix: " + str(e))
    try:
        test_different_prefix_common_suffix()
        passed += 1
        print("  PASS test_different_prefix_common_suffix")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_different_prefix_common_suffix: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_different_prefix_common_suffix: " + str(e))
    try:
        test_single_vs_long()
        passed += 1
        print("  PASS test_single_vs_long")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_single_vs_long: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_single_vs_long: " + str(e))
    try:
        test_long_vs_single()
        passed += 1
        print("  PASS test_long_vs_single")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_long_vs_single: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_long_vs_single: " + str(e))
    try:
        test_completely_different()
        passed += 1
        print("  PASS test_completely_different")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_completely_different: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_completely_different: " + str(e))
    try:
        test_longer_completely_different()
        passed += 1
        print("  PASS test_longer_completely_different")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_longer_completely_different: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_longer_completely_different: " + str(e))
    try:
        test_symmetry_kitten()
        passed += 1
        print("  PASS test_symmetry_kitten")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_symmetry_kitten: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_symmetry_kitten: " + str(e))
    try:
        test_symmetry_empty()
        passed += 1
        print("  PASS test_symmetry_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_symmetry_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_symmetry_empty: " + str(e))
    try:
        test_symmetry_different()
        passed += 1
        print("  PASS test_symmetry_different")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_symmetry_different: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_symmetry_different: " + str(e))
    try:
        test_repeated_a()
        passed += 1
        print("  PASS test_repeated_a")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_repeated_a: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_repeated_a: " + str(e))
    try:
        test_repeated_vs_empty()
        passed += 1
        print("  PASS test_repeated_vs_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_repeated_vs_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_repeated_vs_empty: " + str(e))
    try:
        test_repeated_insert_middle()
        passed += 1
        print("  PASS test_repeated_insert_middle")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_repeated_insert_middle: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_repeated_insert_middle: " + str(e))
    try:
        test_case_sensitive()
        passed += 1
        print("  PASS test_case_sensitive")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_case_sensitive: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_case_sensitive: " + str(e))
    try:
        test_case_one_char()
        passed += 1
        print("  PASS test_case_one_char")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_case_one_char: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_case_one_char: " + str(e))
    try:
        test_insert_at_start()
        passed += 1
        print("  PASS test_insert_at_start")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_insert_at_start: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_insert_at_start: " + str(e))
    try:
        test_insert_at_end()
        passed += 1
        print("  PASS test_insert_at_end")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_insert_at_end: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_insert_at_end: " + str(e))
    try:
        test_delete_at_start()
        passed += 1
        print("  PASS test_delete_at_start")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_delete_at_start: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_delete_at_start: " + str(e))
    try:
        test_delete_at_end()
        passed += 1
        print("  PASS test_delete_at_end")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_delete_at_end: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_delete_at_end: " + str(e))
    try:
        test_unicode_equal()
        passed += 1
        print("  PASS test_unicode_equal")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_unicode_equal: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_unicode_equal: " + str(e))
    try:
        test_unicode_one_sub()
        passed += 1
        print("  PASS test_unicode_one_sub")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_unicode_one_sub: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_unicode_one_sub: " + str(e))
    try:
        test_unicode_cjk()
        passed += 1
        print("  PASS test_unicode_cjk")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_unicode_cjk: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_unicode_cjk: " + str(e))
    try:
        test_unicode_vs_empty()
        passed += 1
        print("  PASS test_unicode_vs_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_unicode_vs_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_unicode_vs_empty: " + str(e))
    try:
        test_triangle_inequality()
        passed += 1
        print("  PASS test_triangle_inequality")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_triangle_inequality: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_triangle_inequality: " + str(e))
    try:
        test_longer_strings()
        passed += 1
        print("  PASS test_longer_strings")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_longer_strings: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_longer_strings: " + str(e))
    try:
        test_longer_strings2()
        passed += 1
        print("  PASS test_longer_strings2")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_longer_strings2: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_longer_strings2: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
