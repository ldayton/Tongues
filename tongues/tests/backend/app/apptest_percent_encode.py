"""Percent-encoding tests — RFC 3986 unreserved set, encode/decode roundtrip."""

import sys

from lib.percent_encode import percent_decode
from lib.percent_encode import percent_encode


# -- Encode: unreserved pass-through --


def test_encode_empty() -> None:
    assert percent_encode("") == ""


def test_encode_alpha_lower() -> None:
    assert percent_encode("abcxyz") == "abcxyz"


def test_encode_alpha_upper() -> None:
    assert percent_encode("ABCXYZ") == "ABCXYZ"


def test_encode_digits() -> None:
    assert percent_encode("0123456789") == "0123456789"


def test_encode_unreserved_symbols() -> None:
    assert percent_encode("-_.~") == "-_.~"


def test_encode_all_unreserved() -> None:
    assert percent_encode("Hello-World_2024.v1~draft") == "Hello-World_2024.v1~draft"


# -- Encode: reserved and special characters --


def test_encode_space() -> None:
    assert percent_encode(" ") == "%20"


def test_encode_exclamation() -> None:
    assert percent_encode("!") == "%21"


def test_encode_hash() -> None:
    assert percent_encode("#") == "%23"


def test_encode_percent() -> None:
    assert percent_encode("%") == "%25"


def test_encode_ampersand() -> None:
    assert percent_encode("&") == "%26"


def test_encode_plus() -> None:
    assert percent_encode("+") == "%2B"


def test_encode_slash() -> None:
    assert percent_encode("/") == "%2F"


def test_encode_colon() -> None:
    assert percent_encode(":") == "%3A"


def test_encode_equals() -> None:
    assert percent_encode("=") == "%3D"


def test_encode_question() -> None:
    assert percent_encode("?") == "%3F"


def test_encode_at() -> None:
    assert percent_encode("@") == "%40"


def test_encode_brackets() -> None:
    assert percent_encode("[]") == "%5B%5D"


# -- Encode: mixed --


def test_encode_hello_world() -> None:
    assert percent_encode("hello world") == "hello%20world"


def test_encode_query_string() -> None:
    assert percent_encode("key=value&foo=bar") == "key%3Dvalue%26foo%3Dbar"


def test_encode_url_path() -> None:
    assert percent_encode("/path/to/file") == "%2Fpath%2Fto%2Ffile"


def test_encode_email() -> None:
    assert percent_encode("user@example.com") == "user%40example.com"


# -- Encode: bytes above 127 (multi-byte UTF-8) --


def test_encode_umlaut() -> None:
    """U+00E9 (e-acute) = UTF-8 C3 A9."""
    assert percent_encode("\u00e9") == "%C3%A9"


def test_encode_euro_sign() -> None:
    """U+20AC (euro) = UTF-8 E2 82 AC."""
    assert percent_encode("\u20ac") == "%E2%82%AC"


def test_encode_cjk() -> None:
    """U+4E2D = UTF-8 E4 B8 AD."""
    assert percent_encode("\u4e2d") == "%E4%B8%AD"


# -- Decode: basics --


def test_decode_empty() -> None:
    assert percent_decode("") == ""


def test_decode_no_encoding() -> None:
    assert percent_decode("hello") == "hello"


def test_decode_space() -> None:
    assert percent_decode("%20") == " "


def test_decode_percent() -> None:
    assert percent_decode("%25") == "%"


def test_decode_hello_world() -> None:
    assert percent_decode("hello%20world") == "hello world"


def test_decode_mixed() -> None:
    assert percent_decode("key%3Dvalue%26foo%3Dbar") == "key=value&foo=bar"


def test_decode_lowercase_hex() -> None:
    assert percent_decode("%2f") == "/"


def test_decode_uppercase_hex() -> None:
    assert percent_decode("%2F") == "/"


def test_decode_multibyte() -> None:
    """Decode UTF-8 multi-byte sequence."""
    assert percent_decode("%C3%A9") == "\u00e9"


def test_decode_consecutive() -> None:
    assert percent_decode("%20%20%20") == "   "


# -- Decode: pass-through of unencoded characters --


def test_decode_unreserved_passthrough() -> None:
    assert percent_decode("abc-._~123") == "abc-._~123"


def test_decode_reserved_passthrough() -> None:
    """Characters not preceded by % pass through as-is."""
    assert percent_decode("a+b") == "a+b"


# -- Roundtrip --


def test_roundtrip_simple() -> None:
    s: str = "hello world"
    assert percent_decode(percent_encode(s)) == s


def test_roundtrip_special() -> None:
    s: str = "foo=bar&baz=qux"
    assert percent_decode(percent_encode(s)) == s


def test_roundtrip_all_unreserved() -> None:
    s: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~"
    assert percent_encode(s) == s
    assert percent_decode(s) == s


def test_roundtrip_percent_literal() -> None:
    s: str = "100%"
    assert percent_decode(percent_encode(s)) == s


def test_roundtrip_unicode() -> None:
    s: str = "caf\u00e9"
    assert percent_decode(percent_encode(s)) == s


# -- Edge cases --


def test_encode_single_unreserved() -> None:
    assert percent_encode("a") == "a"


def test_encode_single_reserved() -> None:
    assert percent_encode("!") == "%21"


def test_encode_tab() -> None:
    assert percent_encode("\t") == "%09"


def test_encode_newline() -> None:
    assert percent_encode("\n") == "%0A"


def test_encode_null() -> None:
    assert percent_encode(chr(0)) == "%00"


def test_encode_del() -> None:
    """0x7F is not unreserved."""
    assert percent_encode(chr(127)) == "%7F"


def test_encode_4byte_emoji() -> None:
    """U+1F600 (grinning face) = UTF-8 F0 9F 98 80."""
    assert percent_encode("\U0001f600") == "%F0%9F%98%80"


def test_encode_mixed_ascii_and_multibyte() -> None:
    assert percent_encode("hi\u00e9!") == "hi%C3%A9%21"


def test_encode_double_encoding() -> None:
    """Encoding an already-encoded string escapes the percent signs."""
    assert percent_encode("%20") == "%2520"


def test_decode_null_byte() -> None:
    assert percent_decode("%00") == chr(0)


def test_decode_4byte_emoji() -> None:
    assert percent_decode("%F0%9F%98%80") == "\U0001f600"


def test_roundtrip_4byte() -> None:
    s: str = "\U0001f600"
    assert percent_decode(percent_encode(s)) == s


def test_roundtrip_mixed_multibyte() -> None:
    s: str = "a\u00e9\u4e2d\U0001f600z"
    assert percent_decode(percent_encode(s)) == s


def test_decode_trailing_percent() -> None:
    """Incomplete percent sequence at end passes through."""
    assert percent_decode("abc%") == "abc%"


def test_decode_incomplete_hex() -> None:
    """Only one hex digit after percent passes through."""
    assert percent_decode("abc%2") == "abc%2"


def test_decode_invalid_hex() -> None:
    """Non-hex after percent passes through."""
    assert percent_decode("abc%GH") == "abc%GH"


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_encode_empty", test_encode_empty),
        ("test_encode_alpha_lower", test_encode_alpha_lower),
        ("test_encode_alpha_upper", test_encode_alpha_upper),
        ("test_encode_digits", test_encode_digits),
        ("test_encode_unreserved_symbols", test_encode_unreserved_symbols),
        ("test_encode_all_unreserved", test_encode_all_unreserved),
        ("test_encode_space", test_encode_space),
        ("test_encode_exclamation", test_encode_exclamation),
        ("test_encode_hash", test_encode_hash),
        ("test_encode_percent", test_encode_percent),
        ("test_encode_ampersand", test_encode_ampersand),
        ("test_encode_plus", test_encode_plus),
        ("test_encode_slash", test_encode_slash),
        ("test_encode_colon", test_encode_colon),
        ("test_encode_equals", test_encode_equals),
        ("test_encode_question", test_encode_question),
        ("test_encode_at", test_encode_at),
        ("test_encode_brackets", test_encode_brackets),
        ("test_encode_hello_world", test_encode_hello_world),
        ("test_encode_query_string", test_encode_query_string),
        ("test_encode_url_path", test_encode_url_path),
        ("test_encode_email", test_encode_email),
        ("test_encode_umlaut", test_encode_umlaut),
        ("test_encode_euro_sign", test_encode_euro_sign),
        ("test_encode_cjk", test_encode_cjk),
        ("test_decode_empty", test_decode_empty),
        ("test_decode_no_encoding", test_decode_no_encoding),
        ("test_decode_space", test_decode_space),
        ("test_decode_percent", test_decode_percent),
        ("test_decode_hello_world", test_decode_hello_world),
        ("test_decode_mixed", test_decode_mixed),
        ("test_decode_lowercase_hex", test_decode_lowercase_hex),
        ("test_decode_uppercase_hex", test_decode_uppercase_hex),
        ("test_decode_multibyte", test_decode_multibyte),
        ("test_decode_consecutive", test_decode_consecutive),
        ("test_decode_unreserved_passthrough", test_decode_unreserved_passthrough),
        ("test_decode_reserved_passthrough", test_decode_reserved_passthrough),
        ("test_roundtrip_simple", test_roundtrip_simple),
        ("test_roundtrip_special", test_roundtrip_special),
        ("test_roundtrip_all_unreserved", test_roundtrip_all_unreserved),
        ("test_roundtrip_percent_literal", test_roundtrip_percent_literal),
        ("test_roundtrip_unicode", test_roundtrip_unicode),
        ("test_encode_single_unreserved", test_encode_single_unreserved),
        ("test_encode_single_reserved", test_encode_single_reserved),
        ("test_encode_tab", test_encode_tab),
        ("test_encode_newline", test_encode_newline),
        ("test_encode_null", test_encode_null),
        ("test_encode_del", test_encode_del),
        ("test_encode_4byte_emoji", test_encode_4byte_emoji),
        (
            "test_encode_mixed_ascii_and_multibyte",
            test_encode_mixed_ascii_and_multibyte,
        ),
        ("test_encode_double_encoding", test_encode_double_encoding),
        ("test_decode_null_byte", test_decode_null_byte),
        ("test_decode_4byte_emoji", test_decode_4byte_emoji),
        ("test_roundtrip_4byte", test_roundtrip_4byte),
        ("test_roundtrip_mixed_multibyte", test_roundtrip_mixed_multibyte),
        ("test_decode_trailing_percent", test_decode_trailing_percent),
        ("test_decode_incomplete_hex", test_decode_incomplete_hex),
        ("test_decode_invalid_hex", test_decode_invalid_hex),
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
