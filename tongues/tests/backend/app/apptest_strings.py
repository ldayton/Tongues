"""String object tests."""

import sys


def test_string_equality() -> None:
    """String equality comparisons."""
    assert "hello" == "hello"
    assert "" == ""
    assert " " == " "
    assert not ("hello" == "world")
    assert "hello" != "world"
    assert "Hello" != "hello"  # case sensitive


def test_string_ordering() -> None:
    """String ordering is lexicographic."""
    assert "a" < "b"
    assert "abc" < "abd"
    assert "abc" < "abcd"
    assert "" < "a"
    assert "A" < "a"  # uppercase < lowercase in ASCII
    assert "abc" <= "abc"
    assert "abc" >= "abc"
    assert "z" > "a"
    assert "10" < "9"  # lexicographic, not numeric


def test_string_length() -> None:
    """len() returns character count."""
    assert len("") == 0
    assert len("a") == 1
    assert len("hello") == 5
    assert len(" ") == 1
    assert len("  ") == 2


def test_string_indexing() -> None:
    """Indexing returns single character string."""
    s: str = "hello"
    assert s[0] == "h"
    assert s[1] == "e"
    assert s[4] == "o"
    # Negative indexing
    assert s[-1] == "o"
    assert s[-5] == "h"


def test_string_slicing() -> None:
    """String slicing returns string."""
    s: str = "hello"
    assert s[0:2] == "he"
    assert s[1:4] == "ell"
    assert s[:3] == "hel"
    assert s[2:] == "llo"
    assert s[:] == "hello"
    assert s[::2] == "hlo"
    assert s[::-1] == "olleh"
    # Empty slices
    assert s[2:2] == ""
    assert s[5:10] == ""


def test_string_concatenation() -> None:
    """String concatenation with +."""
    assert "hello" + " " + "world" == "hello world"
    assert "" + "a" == "a"
    assert "a" + "" == "a"
    assert "" + "" == ""


def test_string_repetition() -> None:
    """String repetition with *."""
    assert "a" * 3 == "aaa"
    assert "ab" * 2 == "abab"
    assert "x" * 0 == ""
    assert "x" * 1 == "x"
    assert 3 * "a" == "aaa"  # reverse order


def test_string_contains() -> None:
    """Membership testing with in."""
    assert "ell" in "hello"
    assert "h" in "hello"
    assert "o" in "hello"
    assert "" in "hello"  # empty string in any string
    assert "" in ""
    assert "x" not in "hello"
    assert "Hello" not in "hello"  # case sensitive


def test_string_upper_lower() -> None:
    """upper() and lower() methods."""
    assert "Hello".upper() == "HELLO"
    assert "Hello".lower() == "hello"
    assert "HELLO".lower() == "hello"
    assert "hello".upper() == "HELLO"
    assert "123".upper() == "123"
    assert "123".lower() == "123"
    assert "".upper() == ""
    assert "".lower() == ""


def test_string_strip() -> None:
    """strip(), lstrip(), rstrip() methods."""
    assert "  hello  ".strip() == "hello"
    assert "  hello  ".lstrip() == "hello  "
    assert "  hello  ".rstrip() == "  hello"
    assert "hello".strip() == "hello"
    assert "".strip() == ""
    assert "   ".strip() == ""
    # Strip with argument
    assert "xxhelloxx".strip("x") == "hello"
    assert "xyzhellozyx".strip("xyz") == "hello"


def test_string_split() -> None:
    """split() method."""
    assert "a,b,c".split(",") == ["a", "b", "c"]
    assert "hello".split("l") == ["he", "", "o"]
    assert "hello".split("x") == ["hello"]
    assert "a  b".split(" ") == ["a", "", "b"]
    assert "".split(",") == [""]


def test_string_join() -> None:
    """join() method."""
    assert ",".join(["a", "b", "c"]) == "a,b,c"
    assert "".join(["a", "b"]) == "ab"
    assert "-".join(["x"]) == "x"
    empty: list[str] = []
    assert "-".join(empty) == ""
    assert "::".join(["a", "b"]) == "a::b"


def test_string_replace() -> None:
    """replace() method."""
    assert "hello".replace("l", "L") == "heLLo"
    assert "hello".replace("ll", "LL") == "heLLo"
    assert "hello".replace("x", "y") == "hello"
    assert "aaa".replace("a", "b") == "bbb"
    assert "hello".replace("", "-") == "-h-e-l-l-o-"


def test_string_find() -> None:
    """find() returns index or -1."""
    assert "hello".find("l") == 2
    assert "hello".find("ll") == 2
    assert "hello".find("o") == 4
    assert "hello".find("x") == -1
    assert "hello".find("") == 0
    assert "hello".find("hello") == 0
    assert "hello".find("hello world") == -1


def test_string_rfind() -> None:
    """rfind() finds last occurrence."""
    assert "hello".rfind("l") == 3
    assert "hello".rfind("o") == 4
    assert "hello".rfind("x") == -1
    assert "abcabc".rfind("abc") == 3
    assert "abcabc".rfind("") == 6


def test_string_count() -> None:
    """count() method."""
    assert "hello".count("l") == 2
    assert "hello".count("ll") == 1
    assert "hello".count("x") == 0
    assert "aaa".count("a") == 3
    assert "aaa".count("aa") == 1  # non-overlapping
    assert "".count("a") == 0
    assert "hello".count("") == 6  # between each char + ends


def test_string_startswith_endswith() -> None:
    """startswith() and endswith() methods."""
    assert "hello".startswith("he")
    assert "hello".startswith("")
    assert "hello".startswith("hello")
    assert not "hello".startswith("lo")
    assert "hello".endswith("lo")
    assert "hello".endswith("")
    assert "hello".endswith("hello")
    assert not "hello".endswith("he")


def test_string_isalpha() -> None:
    """isalpha() for alphabetic strings."""
    assert "hello".isalpha()
    assert "Hello".isalpha()
    assert not "hello1".isalpha()
    assert not "hello world".isalpha()
    assert not "".isalpha()
    assert "a".isalpha()


def test_string_isdigit() -> None:
    """isdigit() for digit strings."""
    assert "123".isdigit()
    assert "0".isdigit()
    assert not "12.3".isdigit()
    assert not "12a".isdigit()
    assert not "".isdigit()
    assert not "-1".isdigit()


def test_string_isalnum() -> None:
    """isalnum() for alphanumeric strings."""
    assert "hello".isalnum()
    assert "hello123".isalnum()
    assert "123".isalnum()
    assert not "hello world".isalnum()
    assert not "hello!".isalnum()
    assert not "".isalnum()


def test_string_isspace() -> None:
    """isspace() for whitespace strings."""
    assert " ".isspace()
    assert "   ".isspace()
    assert "\t".isspace()
    assert "\n".isspace()
    assert " \t\n".isspace()
    assert not "".isspace()
    assert not " a ".isspace()


def test_string_isupper_islower() -> None:
    """isupper() and islower() methods."""
    assert "HELLO".isupper()
    assert not "Hello".isupper()
    assert not "hello".isupper()
    assert not "".isupper()
    assert "hello".islower()
    assert not "Hello".islower()
    assert not "HELLO".islower()
    assert not "".islower()
    # Strings with non-letters
    assert "HELLO123".isupper()
    assert "hello123".islower()


def test_string_bool() -> None:
    """String truthiness - empty is falsy."""
    assert bool("hello") == True
    assert bool("") == False
    assert bool(" ") == True
    assert not ""
    assert "x"


def test_string_str() -> None:
    """str() identity on strings."""
    assert str("hello") == "hello"
    assert str("") == ""
    assert str(" ") == " "


def test_string_escape_sequences() -> None:
    """Common escape sequences."""
    assert len("\n") == 1
    assert len("\t") == 1
    assert len("\\") == 1
    assert len('"') == 1
    assert len("'") == 1
    assert "\n" != "n"
    assert "\t" != "t"


def test_string_multiplication_edge() -> None:
    """Edge cases for string multiplication."""
    assert "" * 100 == ""
    assert "a" * 0 == ""
    assert 0 * "hello" == ""
    assert "ab" * 1 == "ab"


def test_string_comparison_empty() -> None:
    """Empty string comparisons."""
    assert "" == ""
    assert "" < "a"
    assert "" < " "
    assert not ("" > "a")
    assert "" <= ""
    assert "" >= ""


def test_string_split_maxsplit() -> None:
    """split() with maxsplit parameter."""
    assert "a,b,c,d".split(",", 1) == ["a", "b,c,d"]
    assert "a,b,c,d".split(",", 2) == ["a", "b", "c,d"]
    assert "a,b,c".split(",", 10) == ["a", "b", "c"]  # maxsplit > occurrences
    assert "a,b,c".split(",", 0) == ["a,b,c"]  # no splits


def test_string_split_consecutive_delimiters() -> None:
    """Consecutive delimiters create empty strings."""
    assert "a,,b".split(",") == ["a", "", "b"]
    assert ",,a,,".split(",") == ["", "", "a", "", ""]
    assert "a--b".split("-") == ["a", "", "b"]


def test_string_split_whitespace() -> None:
    """split() with None collapses whitespace."""
    # None separator splits on any whitespace and removes empty strings
    assert "a b  c".split() == ["a", "b", "c"]
    assert "  a  b  ".split() == ["a", "b"]
    assert "a\tb\nc".split() == ["a", "b", "c"]
    empty_parts: list[str] = []
    assert "   ".split() == empty_parts
    # Compare to explicit space separator
    assert "a  b".split(" ") == ["a", "", "b"]


def test_string_split_leading_trailing() -> None:
    """Leading/trailing delimiters create empty strings."""
    assert "/a/b/".split("/") == ["", "a", "b", ""]
    assert "/home/user".split("/", 1) == ["", "home/user"]


def test_unicode_length() -> None:
    """len() counts code points, not bytes."""
    # ASCII
    assert len("hello") == 5
    # Multi-byte UTF-8 characters are still 1 code point
    assert len("\u00e9") == 1  # é (e with acute)
    assert len("\u4e2d") == 1  # 中 (Chinese character)
    # Emoji (astral plane) is 1 code point
    assert len("\U0001f600") == 1  # 😀


def test_unicode_indexing() -> None:
    """Indexing works on code points."""
    s: str = "a\u4e2db"  # a中b
    assert len(s) == 3
    assert s[0] == "a"
    assert s[1] == "\u4e2d"
    assert s[2] == "b"


def test_string_multiplication_negative() -> None:
    """Negative multiplier gives empty string."""
    assert "hello" * -1 == ""
    assert "hello" * -100 == ""
    assert -5 * "abc" == ""


def test_string_partition() -> None:
    """partition() splits on first occurrence."""
    assert "a,b,c".partition(",") == ("a", ",", "b,c")
    assert "hello".partition(",") == ("hello", "", "")
    assert "hello,world".partition(",") == ("hello", ",", "world")
    assert ",hello".partition(",") == ("", ",", "hello")
    assert "hello,".partition(",") == ("hello", ",", "")
    assert "a::b::c".partition("::") == ("a", "::", "b::c")


def test_string_rpartition() -> None:
    """rpartition() splits on last occurrence."""
    assert "a,b,c".rpartition(",") == ("a,b", ",", "c")
    assert "hello".rpartition(",") == ("", "", "hello")
    assert "hello,world".rpartition(",") == ("hello", ",", "world")
    assert ",hello".rpartition(",") == ("", ",", "hello")
    assert "hello,".rpartition(",") == ("hello", ",", "")
    assert "a::b::c".rpartition("::") == ("a::b", "::", "c")


def test_string_removeprefix() -> None:
    """removeprefix() removes a leading substring."""
    assert "hello world".removeprefix("hello ") == "world"
    assert "hello".removeprefix("xyz") == "hello"
    assert "hello".removeprefix("") == "hello"
    assert "hello".removeprefix("hello") == ""
    assert "hello".removeprefix("hello world") == "hello"
    assert "aaa".removeprefix("a") == "aa"


def test_string_removesuffix() -> None:
    """removesuffix() removes a trailing substring."""
    assert "hello world".removesuffix(" world") == "hello"
    assert "hello".removesuffix("xyz") == "hello"
    assert "hello".removesuffix("") == "hello"
    assert "hello".removesuffix("hello") == ""
    assert "hello".removesuffix("hello world") == "hello"
    assert "aaa".removesuffix("a") == "aa"


def test_strip_regex_special_chars() -> None:
    """strip() with regex-special characters."""
    assert "]-hello-]".strip("]-") == "hello"
    assert "\\data\\".strip("\\") == "data"


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_string_equality()
        passed += 1
        print("  PASS test_string_equality")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_equality: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_equality: " + str(e))
    try:
        test_string_ordering()
        passed += 1
        print("  PASS test_string_ordering")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_ordering: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_ordering: " + str(e))
    try:
        test_string_length()
        passed += 1
        print("  PASS test_string_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_length: " + str(e))
    try:
        test_string_indexing()
        passed += 1
        print("  PASS test_string_indexing")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_indexing: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_indexing: " + str(e))
    try:
        test_string_slicing()
        passed += 1
        print("  PASS test_string_slicing")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_slicing: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_slicing: " + str(e))
    try:
        test_string_concatenation()
        passed += 1
        print("  PASS test_string_concatenation")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_concatenation: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_concatenation: " + str(e))
    try:
        test_string_repetition()
        passed += 1
        print("  PASS test_string_repetition")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_repetition: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_repetition: " + str(e))
    try:
        test_string_contains()
        passed += 1
        print("  PASS test_string_contains")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_contains: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_contains: " + str(e))
    try:
        test_string_upper_lower()
        passed += 1
        print("  PASS test_string_upper_lower")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_upper_lower: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_upper_lower: " + str(e))
    try:
        test_string_strip()
        passed += 1
        print("  PASS test_string_strip")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_strip: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_strip: " + str(e))
    try:
        test_string_split()
        passed += 1
        print("  PASS test_string_split")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_split: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_split: " + str(e))
    try:
        test_string_join()
        passed += 1
        print("  PASS test_string_join")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_join: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_join: " + str(e))
    try:
        test_string_replace()
        passed += 1
        print("  PASS test_string_replace")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_replace: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_replace: " + str(e))
    try:
        test_string_find()
        passed += 1
        print("  PASS test_string_find")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_find: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_find: " + str(e))
    try:
        test_string_rfind()
        passed += 1
        print("  PASS test_string_rfind")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_rfind: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_rfind: " + str(e))
    try:
        test_string_count()
        passed += 1
        print("  PASS test_string_count")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_count: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_count: " + str(e))
    try:
        test_string_startswith_endswith()
        passed += 1
        print("  PASS test_string_startswith_endswith")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_startswith_endswith: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_startswith_endswith: " + str(e))
    try:
        test_string_isalpha()
        passed += 1
        print("  PASS test_string_isalpha")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_isalpha: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_isalpha: " + str(e))
    try:
        test_string_isdigit()
        passed += 1
        print("  PASS test_string_isdigit")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_isdigit: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_isdigit: " + str(e))
    try:
        test_string_isalnum()
        passed += 1
        print("  PASS test_string_isalnum")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_isalnum: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_isalnum: " + str(e))
    try:
        test_string_isspace()
        passed += 1
        print("  PASS test_string_isspace")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_isspace: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_isspace: " + str(e))
    try:
        test_string_isupper_islower()
        passed += 1
        print("  PASS test_string_isupper_islower")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_isupper_islower: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_isupper_islower: " + str(e))
    try:
        test_string_bool()
        passed += 1
        print("  PASS test_string_bool")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_bool: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_bool: " + str(e))
    try:
        test_string_str()
        passed += 1
        print("  PASS test_string_str")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_str: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_str: " + str(e))
    try:
        test_string_escape_sequences()
        passed += 1
        print("  PASS test_string_escape_sequences")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_escape_sequences: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_escape_sequences: " + str(e))
    try:
        test_string_multiplication_edge()
        passed += 1
        print("  PASS test_string_multiplication_edge")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_multiplication_edge: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_multiplication_edge: " + str(e))
    try:
        test_string_comparison_empty()
        passed += 1
        print("  PASS test_string_comparison_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_comparison_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_comparison_empty: " + str(e))
    try:
        test_string_split_maxsplit()
        passed += 1
        print("  PASS test_string_split_maxsplit")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_split_maxsplit: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_split_maxsplit: " + str(e))
    try:
        test_string_split_whitespace()
        passed += 1
        print("  PASS test_string_split_whitespace")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_split_whitespace: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_split_whitespace: " + str(e))
    try:
        test_string_split_leading_trailing()
        passed += 1
        print("  PASS test_string_split_leading_trailing")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_split_leading_trailing: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_split_leading_trailing: " + str(e))
    try:
        test_unicode_length()
        passed += 1
        print("  PASS test_unicode_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_unicode_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_unicode_length: " + str(e))
    try:
        test_unicode_indexing()
        passed += 1
        print("  PASS test_unicode_indexing")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_unicode_indexing: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_unicode_indexing: " + str(e))
    try:
        test_string_multiplication_negative()
        passed += 1
        print("  PASS test_string_multiplication_negative")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_multiplication_negative: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_multiplication_negative: " + str(e))
    try:
        test_string_partition()
        passed += 1
        print("  PASS test_string_partition")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_partition: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_partition: " + str(e))
    try:
        test_string_rpartition()
        passed += 1
        print("  PASS test_string_rpartition")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_rpartition: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_rpartition: " + str(e))
    try:
        test_string_removeprefix()
        passed += 1
        print("  PASS test_string_removeprefix")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_removeprefix: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_removeprefix: " + str(e))
    try:
        test_string_removesuffix()
        passed += 1
        print("  PASS test_string_removesuffix")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_string_removesuffix: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_string_removesuffix: " + str(e))
    try:
        test_strip_regex_special_chars()
        passed += 1
        print("  PASS test_strip_regex_special_chars")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_strip_regex_special_chars: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_strip_regex_special_chars: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
