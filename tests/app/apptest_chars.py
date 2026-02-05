"""Single character string tests."""

import sys


def test_char_equality() -> None:
    """Character equality comparisons."""
    assert "a" == "a"
    assert "A" == "A"
    assert "0" == "0"
    assert " " == " "
    assert not ("a" == "b")
    assert "a" != "b"


def test_char_ordering() -> None:
    """Character ordering comparisons."""
    assert "a" < "b"
    assert "A" < "B"
    assert "a" > "A"  # lowercase > uppercase in ASCII
    assert "0" < "9"
    assert "9" > "0"
    assert "a" <= "a"
    assert "a" <= "b"
    assert "b" >= "a"
    assert "b" >= "b"


def test_ord() -> None:
    """ord() returns Unicode code point."""
    assert ord("a") == 97
    assert ord("A") == 65
    assert ord("0") == 48
    assert ord(" ") == 32
    assert ord("\n") == 10
    assert ord("\t") == 9


def test_ord_boundaries() -> None:
    """ord() at ASCII boundaries."""
    assert ord("\x00") == 0  # null
    assert ord("\x7f") == 127  # DEL (last ASCII)
    assert ord("\x80") == 128  # first non-ASCII


def test_chr() -> None:
    """chr() returns character from code point."""
    assert chr(97) == "a"
    assert chr(65) == "A"
    assert chr(48) == "0"
    assert chr(32) == " "
    assert chr(10) == "\n"
    assert chr(9) == "\t"


def test_chr_boundaries() -> None:
    """chr() at ASCII boundaries."""
    assert chr(0) == "\x00"  # null
    assert chr(127) == "\x7f"  # DEL
    assert chr(128) == "\x80"  # first non-ASCII


def test_chr_unicode_boundaries() -> None:
    """chr() at Unicode plane boundaries."""
    assert ord(chr(0xFFFF)) == 0xFFFF  # last BMP character
    assert ord(chr(0x10000)) == 0x10000  # first astral (SMP)
    assert ord(chr(0x10FFFF)) == 0x10FFFF  # max Unicode code point


def test_surrogate_range() -> None:
    """Surrogate code points (U+D800-U+DFFF) can be created but not encoded."""
    # Python allows creating strings with surrogate code points
    s: str = chr(0xD800)  # first surrogate
    assert ord(s) == 0xD800
    s = chr(0xDFFF)  # last surrogate
    assert ord(s) == 0xDFFF


def test_ord_chr_roundtrip() -> None:
    """ord and chr are inverses."""
    assert chr(ord("x")) == "x"
    assert ord(chr(100)) == 100
    assert chr(ord("Z")) == "Z"
    assert ord(chr(0)) == 0
    assert ord(chr(127)) == 127
    assert ord(chr(255)) == 255


def test_isalpha() -> None:
    """isalpha() for letters."""
    assert "a".isalpha()
    assert "Z".isalpha()
    assert not "0".isalpha()
    assert not " ".isalpha()
    assert not "\n".isalpha()


def test_isdigit() -> None:
    """isdigit() for digits."""
    assert "0".isdigit()
    assert "9".isdigit()
    assert not "a".isdigit()
    assert not " ".isdigit()


def test_isalnum() -> None:
    """isalnum() for alphanumeric."""
    assert "a".isalnum()
    assert "Z".isalnum()
    assert "0".isalnum()
    assert "9".isalnum()
    assert not " ".isalnum()
    assert not "\n".isalnum()


def test_isspace() -> None:
    """isspace() for whitespace."""
    assert " ".isspace()
    assert "\t".isspace()
    assert "\n".isspace()
    assert "\r".isspace()  # carriage return
    assert "\f".isspace()  # form feed
    assert "\v".isspace()  # vertical tab
    assert not "a".isspace()
    assert not "0".isspace()
    assert not "\x00".isspace()  # null is not whitespace


def test_isupper() -> None:
    """isupper() for uppercase letters."""
    assert "A".isupper()
    assert "Z".isupper()
    assert not "a".isupper()
    assert not "0".isupper()
    assert not " ".isupper()


def test_islower() -> None:
    """islower() for lowercase letters."""
    assert "a".islower()
    assert "z".islower()
    assert not "A".islower()
    assert not "0".islower()
    assert not " ".islower()


def test_upper() -> None:
    """upper() converts to uppercase."""
    assert "a".upper() == "A"
    assert "z".upper() == "Z"
    assert "A".upper() == "A"
    assert "0".upper() == "0"
    assert " ".upper() == " "


def test_lower() -> None:
    """lower() converts to lowercase."""
    assert "A".lower() == "a"
    assert "Z".lower() == "z"
    assert "a".lower() == "a"
    assert "0".lower() == "0"
    assert " ".lower() == " "


def test_char_str_repr() -> None:
    """str() and repr() of characters."""
    assert str("a") == "a"
    assert str(" ") == " "
    assert repr("a") == "'a'"
    assert repr(" ") == "' '"


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_char_equality", test_char_equality),
        ("test_char_ordering", test_char_ordering),
        ("test_ord", test_ord),
        ("test_ord_boundaries", test_ord_boundaries),
        ("test_chr", test_chr),
        ("test_chr_boundaries", test_chr_boundaries),
        ("test_chr_unicode_boundaries", test_chr_unicode_boundaries),
        ("test_surrogate_range", test_surrogate_range),
        ("test_ord_chr_roundtrip", test_ord_chr_roundtrip),
        ("test_isalpha", test_isalpha),
        ("test_isdigit", test_isdigit),
        ("test_isalnum", test_isalnum),
        ("test_isspace", test_isspace),
        ("test_isupper", test_isupper),
        ("test_islower", test_islower),
        ("test_upper", test_upper),
        ("test_lower", test_lower),
        ("test_char_str_repr", test_char_str_repr),
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
