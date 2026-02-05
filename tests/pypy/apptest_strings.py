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
    assert not ("x" in "hello")
    assert not ("Hello" in "hello")  # case sensitive


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
    assert "-".join([]) == ""
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


def test_string_capitalize_title() -> None:
    """capitalize() and title() methods."""
    assert "hello".capitalize() == "Hello"
    assert "HELLO".capitalize() == "Hello"
    assert "hello world".capitalize() == "Hello world"
    assert "".capitalize() == ""
    assert "hello world".title() == "Hello World"
    assert "HELLO WORLD".title() == "Hello World"


def test_string_center_ljust_rjust() -> None:
    """center(), ljust(), rjust() padding methods."""
    assert "hi".center(6) == "  hi  "
    assert "hi".ljust(6) == "hi    "
    assert "hi".rjust(6) == "    hi"
    # With fill character
    assert "hi".center(6, "-") == "--hi--"
    assert "hi".ljust(6, "-") == "hi----"
    assert "hi".rjust(6, "-") == "----hi"
    # Width less than string length
    assert "hello".center(3) == "hello"


def test_string_zfill() -> None:
    """zfill() zero-pads numbers."""
    assert "42".zfill(5) == "00042"
    assert "42".zfill(1) == "42"
    assert "-42".zfill(5) == "-0042"
    assert "+42".zfill(5) == "+0042"
    assert "".zfill(3) == "000"


def test_string_partition() -> None:
    """partition() splits on first occurrence."""
    assert "hello".partition("l") == ("he", "l", "lo")
    assert "hello".partition("x") == ("hello", "", "")
    assert "a:b:c".partition(":") == ("a", ":", "b:c")


def test_string_rpartition() -> None:
    """rpartition() splits on last occurrence."""
    assert "hello".rpartition("l") == ("hel", "l", "o")
    assert "hello".rpartition("x") == ("", "", "hello")


def test_string_splitlines() -> None:
    """splitlines() splits on line boundaries."""
    assert "a\nb\nc".splitlines() == ["a", "b", "c"]
    assert "a\nb\n".splitlines() == ["a", "b"]
    assert "".splitlines() == []
    assert "no newline".splitlines() == ["no newline"]


def test_string_expandtabs() -> None:
    """expandtabs() replaces tabs with spaces."""
    assert "a\tb".expandtabs(4) == "a   b"
    assert "\t".expandtabs(4) == "    "
    assert "no tabs".expandtabs() == "no tabs"


def test_string_bool() -> None:
    """String truthiness - empty is falsy."""
    assert bool("hello") == True
    assert bool("") == False
    assert bool(" ") == True
    assert not ""
    assert "x"


def test_string_str_repr() -> None:
    """str() and repr() of strings."""
    assert str("hello") == "hello"
    assert repr("hello") == "'hello'"
    assert repr("it's") == "\"it's\""


def test_string_escape_sequences() -> None:
    """Common escape sequences."""
    assert len("\n") == 1
    assert len("\t") == 1
    assert len("\\") == 1
    assert len("\"") == 1
    assert len("\'") == 1
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


def test_string_rsplit_maxsplit() -> None:
    """rsplit() with maxsplit splits from right."""
    assert "a,b,c,d".rsplit(",", 1) == ["a,b,c", "d"]
    assert "a,b,c,d".rsplit(",", 2) == ["a,b", "c", "d"]
    assert "a,b,c".rsplit(",", 10) == ["a", "b", "c"]
    assert "a,b,c".rsplit(",", 0) == ["a,b,c"]


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
    assert "   ".split() == []
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
    assert len("\U0001F600") == 1  # 😀


def test_unicode_indexing() -> None:
    """Indexing works on code points."""
    s: str = "a\u4e2db"  # a中b
    assert len(s) == 3
    assert s[0] == "a"
    assert s[1] == "\u4e2d"
    assert s[2] == "b"


def test_casefold() -> None:
    """casefold() for case-insensitive comparison."""
    assert "Hello".casefold() == "hello"
    assert "HELLO".casefold() == "hello"
    assert "HeLLo WoRLd".casefold() == "hello world"
    assert "".casefold() == ""


def test_string_multiplication_negative() -> None:
    """Negative multiplier gives empty string."""
    assert "hello" * -1 == ""
    assert "hello" * -100 == ""
    assert -5 * "abc" == ""


def test_string_format_basic() -> None:
    """Basic string formatting with format()."""
    assert "Hello, {}!".format("world") == "Hello, world!"
    assert "{} + {} = {}".format(1, 2, 3) == "1 + 2 = 3"
    assert "{0} {1} {0}".format("a", "b") == "a b a"


def test_string_swapcase() -> None:
    """swapcase() inverts case."""
    assert "Hello".swapcase() == "hELLO"
    assert "hELLO".swapcase() == "Hello"
    assert "123".swapcase() == "123"
    assert "".swapcase() == ""


def test_string_removeprefix_removesuffix() -> None:
    """removeprefix() and removesuffix() methods."""
    assert "hello_world".removeprefix("hello_") == "world"
    assert "hello_world".removeprefix("xxx") == "hello_world"
    assert "hello_world".removesuffix("_world") == "hello"
    assert "hello_world".removesuffix("xxx") == "hello_world"
    assert "aaa".removeprefix("a") == "aa"  # only removes once


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_string_equality", test_string_equality),
        ("test_string_ordering", test_string_ordering),
        ("test_string_length", test_string_length),
        ("test_string_indexing", test_string_indexing),
        ("test_string_slicing", test_string_slicing),
        ("test_string_concatenation", test_string_concatenation),
        ("test_string_repetition", test_string_repetition),
        ("test_string_contains", test_string_contains),
        ("test_string_upper_lower", test_string_upper_lower),
        ("test_string_strip", test_string_strip),
        ("test_string_split", test_string_split),
        ("test_string_join", test_string_join),
        ("test_string_replace", test_string_replace),
        ("test_string_find", test_string_find),
        ("test_string_rfind", test_string_rfind),
        ("test_string_count", test_string_count),
        ("test_string_startswith_endswith", test_string_startswith_endswith),
        ("test_string_isalpha", test_string_isalpha),
        ("test_string_isdigit", test_string_isdigit),
        ("test_string_isalnum", test_string_isalnum),
        ("test_string_isspace", test_string_isspace),
        ("test_string_isupper_islower", test_string_isupper_islower),
        ("test_string_capitalize_title", test_string_capitalize_title),
        ("test_string_center_ljust_rjust", test_string_center_ljust_rjust),
        ("test_string_zfill", test_string_zfill),
        ("test_string_partition", test_string_partition),
        ("test_string_rpartition", test_string_rpartition),
        ("test_string_splitlines", test_string_splitlines),
        ("test_string_expandtabs", test_string_expandtabs),
        ("test_string_bool", test_string_bool),
        ("test_string_str_repr", test_string_str_repr),
        ("test_string_escape_sequences", test_string_escape_sequences),
        ("test_string_multiplication_edge", test_string_multiplication_edge),
        ("test_string_comparison_empty", test_string_comparison_empty),
        ("test_string_split_maxsplit", test_string_split_maxsplit),
        ("test_string_rsplit_maxsplit", test_string_rsplit_maxsplit),
        ("test_string_split_consecutive_delimiters", test_string_split_consecutive_delimiters),
        ("test_string_split_whitespace", test_string_split_whitespace),
        ("test_string_split_leading_trailing", test_string_split_leading_trailing),
        ("test_unicode_length", test_unicode_length),
        ("test_unicode_indexing", test_unicode_indexing),
        ("test_casefold", test_casefold),
        ("test_string_multiplication_negative", test_string_multiplication_negative),
        ("test_string_format_basic", test_string_format_basic),
        ("test_string_swapcase", test_string_swapcase),
        ("test_string_removeprefix_removesuffix", test_string_removeprefix_removesuffix),
    ]
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print("  PASS " + name)
        except AssertionError as e:
            failed += 1
            print("  FAIL " + name + ": " + str(e))
        except Exception as e:
            failed += 1
            print("  FAIL " + name + ": " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
