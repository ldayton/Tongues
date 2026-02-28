"""Base64 encode/decode tests."""

import sys

from lib.base64 import b64decode
from lib.base64 import b64encode


# -- RFC 4648 test vectors --


def test_rfc4648_empty() -> None:
    assert b64encode(b"") == ""
    assert b64decode("") == b""


def test_rfc4648_f() -> None:
    assert b64encode(b"f") == "Zg=="
    assert b64decode("Zg==") == b"f"


def test_rfc4648_fo() -> None:
    assert b64encode(b"fo") == "Zm8="
    assert b64decode("Zm8=") == b"fo"


def test_rfc4648_foo() -> None:
    assert b64encode(b"foo") == "Zm9v"
    assert b64decode("Zm9v") == b"foo"


def test_rfc4648_foob() -> None:
    assert b64encode(b"foob") == "Zm9vYg=="
    assert b64decode("Zm9vYg==") == b"foob"


def test_rfc4648_fooba() -> None:
    assert b64encode(b"fooba") == "Zm9vYmE="
    assert b64decode("Zm9vYmE=") == b"fooba"


def test_rfc4648_foobar() -> None:
    assert b64encode(b"foobar") == "Zm9vYmFy"
    assert b64decode("Zm9vYmFy") == b"foobar"


# -- Padding boundary cases (0, 1, 2 remainder bytes) --


def test_pad_0_remainder() -> None:
    """3 bytes -> 4 chars, no padding."""
    assert b64encode(b"abc") == "YWJj"
    assert b64encode(b"abcdef") == "YWJjZGVm"


def test_pad_1_remainder() -> None:
    """1 byte -> 4 chars with ==."""
    assert b64encode(b"a") == "YQ=="
    assert b64encode(b"z") == "eg=="
    assert b64encode(b"\x00") == "AA=="
    assert b64encode(b"\xff") == "/w=="


def test_pad_2_remainder() -> None:
    """2 bytes -> 4 chars with =."""
    assert b64encode(b"ab") == "YWI="
    assert b64encode(b"\x00\x00") == "AAA="
    assert b64encode(b"\xff\xff") == "//8="


# -- Byte boundary values --


def test_null_bytes() -> None:
    assert b64encode(b"\x00") == "AA=="
    assert b64encode(b"\x00\x00\x00") == "AAAA"
    assert b64decode("AAAA") == b"\x00\x00\x00"


def test_max_bytes() -> None:
    assert b64encode(b"\xff\xff\xff") == "////"
    assert b64decode("////") == b"\xff\xff\xff"


def test_ascending_bytes() -> None:
    assert b64encode(b"\x00\x01\x02") == "AAEC"
    assert b64decode("AAEC") == b"\x00\x01\x02"


def test_descending_bytes() -> None:
    assert b64encode(b"\xfe\xfd\xfc") == "/v38"
    assert b64decode("/v38") == b"\xfe\xfd\xfc"


# -- Roundtrip properties --


def test_roundtrip_empty() -> None:
    assert b64decode(b64encode(b"")) == b""


def test_roundtrip_single_bytes() -> None:
    i: int = 0
    while i < 256:
        b: bytes = bytes([i])
        assert b64decode(b64encode(b)) == b
        i += 1


def test_roundtrip_two_bytes() -> None:
    pairs: list[list[int]] = [
        [0, 0], [0, 255], [255, 0], [255, 255],
        [127, 128], [1, 2], [254, 253], [0, 1],
    ]
    for pair in pairs:
        b: bytes = bytes(pair)
        assert b64decode(b64encode(b)) == b


def test_roundtrip_three_bytes() -> None:
    triples: list[list[int]] = [
        [0, 0, 0], [255, 255, 255], [1, 2, 3],
        [0, 127, 255], [128, 0, 128], [10, 20, 30],
    ]
    for triple in triples:
        b: bytes = bytes(triple)
        assert b64decode(b64encode(b)) == b


def test_roundtrip_longer() -> None:
    data: bytes = b"The quick brown fox jumps over the lazy dog"
    assert b64decode(b64encode(data)) == data


def test_roundtrip_binary_pattern() -> None:
    """All 256 byte values in sequence."""
    vals: list[int] = []
    i: int = 0
    while i < 256:
        vals.append(i)
        i += 1
    data: bytes = bytes(vals)
    assert b64decode(b64encode(data)) == data


# -- Known encode values --


def test_encode_hello_world() -> None:
    assert b64encode(b"Hello, World!") == "SGVsbG8sIFdvcmxkIQ=="


def test_encode_digits() -> None:
    assert b64encode(b"0123456789") == "MDEyMzQ1Njc4OQ=="


def test_encode_binary_data() -> None:
    assert b64encode(b"\x00\x10\x83\x10\x51\x87\x20\x92\x8b") == "ABCDEFGHIJKL"


# -- Known decode values --


def test_decode_hello_world() -> None:
    assert b64decode("SGVsbG8sIFdvcmxkIQ==") == b"Hello, World!"


def test_decode_digits() -> None:
    assert b64decode("MDEyMzQ1Njc4OQ==") == b"0123456789"


# -- Alphabet coverage --


def test_encode_produces_all_chars() -> None:
    """Encoding 0x00..0xBF (192 bytes = 64 triples) covers the full alphabet."""
    alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    vals: list[int] = []
    i: int = 0
    while i < 192:
        vals.append(i)
        i += 1
    encoded: str = b64encode(bytes(vals))
    for ch in alphabet:
        assert ch in encoded


def test_decode_each_alphabet_char() -> None:
    """Encoding bytes([i<<2, 0, 0]) puts index i in the first 6-bit slot."""
    alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    i: int = 0
    while i < 64:
        data: bytes = bytes([i << 2, 0, 0])
        encoded: str = b64encode(data)
        assert encoded[0] == alphabet[i]
        i += 1


# -- Specific 6-bit packing --


def test_6bit_all_zeros() -> None:
    assert b64encode(b"\x00\x00\x00") == "AAAA"


def test_6bit_all_ones() -> None:
    assert b64encode(b"\xff\xff\xff") == "////"


def test_6bit_boundary() -> None:
    """0x00 0x00 0x3F -> 6-bit groups: 0, 0, 0, 63."""
    assert b64encode(b"\x00\x00\x3f") == "AAA/"


def test_6bit_high() -> None:
    """0xFC 0x00 0x00 -> 6-bit groups: 63, 0, 0, 0."""
    assert b64encode(b"\xfc\x00\x00") == "/AAA"


# -- Longer messages --


def test_encode_pangram() -> None:
    data: bytes = b"Pack my box with five dozen liquor jugs."
    expected: str = "UGFjayBteSBib3ggd2l0aCBmaXZlIGRvemVuIGxpcXVvciBqdWdzLg=="
    assert b64encode(data) == expected
    assert b64decode(expected) == data


def test_encode_repeated() -> None:
    data: bytes = b"AAAA" * 10
    encoded: str = b64encode(data)
    assert b64decode(encoded) == data


def test_encode_long_binary() -> None:
    """512 bytes of patterned data."""
    vals: list[int] = []
    i: int = 0
    while i < 512:
        vals.append(i & 0xFF)
        i += 1
    data: bytes = bytes(vals)
    encoded: str = b64encode(data)
    assert b64decode(encoded) == data
    assert len(encoded) == 684


# -- Output length invariants --


def test_encoded_length() -> None:
    """Padded base64 output is always a multiple of 4."""
    i: int = 0
    while i < 20:
        vals: list[int] = []
        j: int = 0
        while j < i:
            vals.append(j & 0xFF)
            j += 1
        data: bytes = bytes(vals)
        encoded: str = b64encode(data)
        assert len(encoded) % 4 == 0
        i += 1


def test_decoded_length() -> None:
    """n input bytes -> ceil(n/3)*4 output chars."""
    cases: list[list[int]] = [
        [0, 0], [1, 4], [2, 4], [3, 4],
        [4, 8], [5, 8], [6, 8],
        [7, 12], [8, 12], [9, 12],
    ]
    for case in cases:
        n: int = case[0]
        expected_len: int = case[1]
        vals: list[int] = []
        j: int = 0
        while j < n:
            vals.append(0)
            j += 1
        data: bytes = bytes(vals)
        assert len(b64encode(data)) == expected_len


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_rfc4648_empty", test_rfc4648_empty),
        ("test_rfc4648_f", test_rfc4648_f),
        ("test_rfc4648_fo", test_rfc4648_fo),
        ("test_rfc4648_foo", test_rfc4648_foo),
        ("test_rfc4648_foob", test_rfc4648_foob),
        ("test_rfc4648_fooba", test_rfc4648_fooba),
        ("test_rfc4648_foobar", test_rfc4648_foobar),
        ("test_pad_0_remainder", test_pad_0_remainder),
        ("test_pad_1_remainder", test_pad_1_remainder),
        ("test_pad_2_remainder", test_pad_2_remainder),
        ("test_null_bytes", test_null_bytes),
        ("test_max_bytes", test_max_bytes),
        ("test_ascending_bytes", test_ascending_bytes),
        ("test_descending_bytes", test_descending_bytes),
        ("test_roundtrip_empty", test_roundtrip_empty),
        ("test_roundtrip_single_bytes", test_roundtrip_single_bytes),
        ("test_roundtrip_two_bytes", test_roundtrip_two_bytes),
        ("test_roundtrip_three_bytes", test_roundtrip_three_bytes),
        ("test_roundtrip_longer", test_roundtrip_longer),
        ("test_roundtrip_binary_pattern", test_roundtrip_binary_pattern),
        ("test_encode_hello_world", test_encode_hello_world),
        ("test_encode_digits", test_encode_digits),
        ("test_encode_binary_data", test_encode_binary_data),
        ("test_decode_hello_world", test_decode_hello_world),
        ("test_decode_digits", test_decode_digits),
        ("test_encode_produces_all_chars", test_encode_produces_all_chars),
        ("test_decode_each_alphabet_char", test_decode_each_alphabet_char),
        ("test_6bit_all_zeros", test_6bit_all_zeros),
        ("test_6bit_all_ones", test_6bit_all_ones),
        ("test_6bit_boundary", test_6bit_boundary),
        ("test_6bit_high", test_6bit_high),
        ("test_encode_pangram", test_encode_pangram),
        ("test_encode_repeated", test_encode_repeated),
        ("test_encode_long_binary", test_encode_long_binary),
        ("test_encoded_length", test_encoded_length),
        ("test_decoded_length", test_decoded_length),
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
