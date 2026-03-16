"""Base64 encode/decode tests."""

import sys

from lib.base64 import b64decode
from lib.base64 import b64decode_strict
from lib.base64 import b64encode
from lib.base64 import Base64Error


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
    b: bytes = b""
    while i < 256:
        b = bytes([i])
        assert b64decode(b64encode(b)) == b
        i += 1


def test_roundtrip_two_bytes() -> None:
    pairs: list[list[int]] = [
        [0, 0],
        [0, 255],
        [255, 0],
        [255, 255],
        [127, 128],
        [1, 2],
        [254, 253],
        [0, 1],
    ]
    b: bytes = b""
    for pair in pairs:
        b = bytes(pair)
        assert b64decode(b64encode(b)) == b


def test_roundtrip_three_bytes() -> None:
    triples: list[list[int]] = [
        [0, 0, 0],
        [255, 255, 255],
        [1, 2, 3],
        [0, 127, 255],
        [128, 0, 128],
        [10, 20, 30],
    ]
    b: bytes = b""
    for triple in triples:
        b = bytes(triple)
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
    data: bytes = b""
    encoded: str = ""
    while i < 64:
        data = bytes([i << 2, 0, 0])
        encoded = b64encode(data)
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
    j: int = 0
    vals: list[int] = []
    data: bytes = b""
    encoded: str = ""
    while i < 20:
        vals = []
        j = 0
        while j < i:
            vals.append(j & 0xFF)
            j += 1
        data = bytes(vals)
        encoded = b64encode(data)
        assert len(encoded) % 4 == 0
        i += 1


def test_decoded_length() -> None:
    """n input bytes -> ceil(n/3)*4 output chars."""
    cases: list[list[int]] = [
        [0, 0],
        [1, 4],
        [2, 4],
        [3, 4],
        [4, 8],
        [5, 8],
        [6, 8],
        [7, 12],
        [8, 12],
        [9, 12],
    ]
    n: int = 0
    expected_len: int = 0
    vals: list[int] = []
    j: int = 0
    data: bytes = b""
    for case in cases:
        n = case[0]
        expected_len = case[1]
        vals = []
        j = 0
        while j < n:
            vals.append(0)
            j += 1
        data = bytes(vals)
        assert len(b64encode(data)) == expected_len


# -- Invalid input handling (Go stdlib behavior) --


def test_invalid_char_position_0() -> None:
    """Invalid character at position 0."""
    try:
        b64decode("!ABC")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 0


def test_invalid_char_position_1() -> None:
    """Invalid character at position 1."""
    try:
        b64decode("A!BC")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 1


def test_invalid_char_position_2() -> None:
    """Invalid character at position 2."""
    try:
        b64decode("AB!C")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 2


def test_invalid_char_position_3() -> None:
    """Invalid character at position 3."""
    try:
        b64decode("ABC!")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 3


def test_invalid_char_in_second_group() -> None:
    """Invalid character in second 4-char group."""
    try:
        b64decode("ABCDAB!D")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 6


def test_invalid_space() -> None:
    """Space is not valid base64."""
    try:
        b64decode("AB CD")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 2


def test_invalid_newline() -> None:
    """Newline is not valid base64."""
    try:
        b64decode("ABCD\nEFGH")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 4


def test_invalid_length_1() -> None:
    """Length 1 is invalid (need at least 2 chars for 1 byte)."""
    try:
        b64decode("A")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 1


def test_invalid_length_2() -> None:
    """Length 2 without padding is invalid."""
    try:
        b64decode("AB")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 2


def test_invalid_length_3() -> None:
    """Length 3 without padding is invalid."""
    try:
        b64decode("ABC")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 3


def test_invalid_length_5() -> None:
    """Length 5 is invalid (not multiple of 4)."""
    try:
        b64decode("ABCDE")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 5


def test_invalid_padding_middle() -> None:
    """Padding in wrong position."""
    try:
        b64decode("A=CD")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 1


def test_invalid_padding_first() -> None:
    """Padding at start."""
    try:
        b64decode("=BCD")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 0


def test_various_invalid_chars() -> None:
    """Various invalid ASCII characters."""
    invalid_chars: str = "!@#$%^&*()[]{}|;:',.<>?`~\"\\ "
    i: int = 0
    ch: str = ""
    while i < len(invalid_chars):
        ch = invalid_chars[i]
        try:
            b64decode("ABC" + ch)
            assert False, "expected Base64Error for char: " + ch
        except Base64Error as e:
            assert e.position == 3
        i += 1


# -- Strict mode: non-zero padding bits (RFC 4648 §3.5) --


def test_strict_rejects_nonzero_pad_bits_2pad() -> None:
    """With ==, lower 4 bits of char at n-3 must be zero."""
    # 'A' = 0 (0b000000) - canonical
    assert b64decode_strict("YQ==") == b"a"
    # 'B' = 1 (0b000001) - non-zero lower 4 bits
    try:
        b64decode_strict("YR==")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 1
    # 'D' = 3 (0b000011) - non-zero lower 4 bits
    try:
        b64decode_strict("YT==")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 1


def test_strict_rejects_nonzero_pad_bits_1pad() -> None:
    """With =, lower 2 bits of char at n-2 must be zero."""
    # 'I' = 8 (0b001000) - canonical (lower 2 bits = 00)
    assert b64decode_strict("YWI=") == b"ab"
    # 'J' = 9 (0b001001) - non-zero lower 2 bits
    try:
        b64decode_strict("YWJ=")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 2
    # 'K' = 10 (0b001010) - non-zero lower 2 bits
    try:
        b64decode_strict("YWK=")
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 2


def test_strict_accepts_canonical_padding() -> None:
    """Strict mode accepts properly zero-padded input."""
    # All RFC test vectors should pass strict mode
    assert b64decode_strict("") == b""
    assert b64decode_strict("Zg==") == b"f"
    assert b64decode_strict("Zm8=") == b"fo"
    assert b64decode_strict("Zm9v") == b"foo"
    assert b64decode_strict("Zm9vYg==") == b"foob"
    assert b64decode_strict("Zm9vYmE=") == b"fooba"
    assert b64decode_strict("Zm9vYmFy") == b"foobar"


def test_strict_github_issue_example() -> None:
    """The exact example from Go issue #15656."""
    good: str = "WvLTlMrX9NpYDQlEIFlnDA=="
    bad: str = "WvLTlMrX9NpYDQlEIFlnDB=="
    # Good should decode fine
    b64decode_strict(good)
    # Bad should fail in strict mode
    try:
        b64decode_strict(bad)
        assert False, "expected Base64Error"
    except Base64Error as e:
        assert e.position == 21  # position of 'B'


def test_lenient_accepts_nonzero_pad_bits() -> None:
    """Non-strict mode (default) accepts non-canonical input."""
    # These should all succeed without strict=True
    assert b64decode("YR==") == b"a"  # same as YQ==
    assert b64decode("YWJ=") == b"ab"  # same as YWI=
    assert b64decode("WvLTlMrX9NpYDQlEIFlnDB==") == b64decode(
        "WvLTlMrX9NpYDQlEIFlnDA=="
    )


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_rfc4648_empty()
        passed += 1
        print("  PASS test_rfc4648_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_rfc4648_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_rfc4648_empty: " + str(e))
    try:
        test_rfc4648_f()
        passed += 1
        print("  PASS test_rfc4648_f")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_rfc4648_f: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_rfc4648_f: " + str(e))
    try:
        test_rfc4648_fo()
        passed += 1
        print("  PASS test_rfc4648_fo")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_rfc4648_fo: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_rfc4648_fo: " + str(e))
    try:
        test_rfc4648_foo()
        passed += 1
        print("  PASS test_rfc4648_foo")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_rfc4648_foo: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_rfc4648_foo: " + str(e))
    try:
        test_rfc4648_foob()
        passed += 1
        print("  PASS test_rfc4648_foob")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_rfc4648_foob: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_rfc4648_foob: " + str(e))
    try:
        test_rfc4648_fooba()
        passed += 1
        print("  PASS test_rfc4648_fooba")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_rfc4648_fooba: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_rfc4648_fooba: " + str(e))
    try:
        test_rfc4648_foobar()
        passed += 1
        print("  PASS test_rfc4648_foobar")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_rfc4648_foobar: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_rfc4648_foobar: " + str(e))
    try:
        test_pad_0_remainder()
        passed += 1
        print("  PASS test_pad_0_remainder")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_pad_0_remainder: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_pad_0_remainder: " + str(e))
    try:
        test_pad_1_remainder()
        passed += 1
        print("  PASS test_pad_1_remainder")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_pad_1_remainder: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_pad_1_remainder: " + str(e))
    try:
        test_pad_2_remainder()
        passed += 1
        print("  PASS test_pad_2_remainder")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_pad_2_remainder: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_pad_2_remainder: " + str(e))
    try:
        test_null_bytes()
        passed += 1
        print("  PASS test_null_bytes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_null_bytes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_null_bytes: " + str(e))
    try:
        test_max_bytes()
        passed += 1
        print("  PASS test_max_bytes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_max_bytes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_max_bytes: " + str(e))
    try:
        test_ascending_bytes()
        passed += 1
        print("  PASS test_ascending_bytes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_ascending_bytes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_ascending_bytes: " + str(e))
    try:
        test_descending_bytes()
        passed += 1
        print("  PASS test_descending_bytes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_descending_bytes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_descending_bytes: " + str(e))
    try:
        test_roundtrip_empty()
        passed += 1
        print("  PASS test_roundtrip_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_empty: " + str(e))
    try:
        test_roundtrip_single_bytes()
        passed += 1
        print("  PASS test_roundtrip_single_bytes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_single_bytes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_single_bytes: " + str(e))
    try:
        test_roundtrip_two_bytes()
        passed += 1
        print("  PASS test_roundtrip_two_bytes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_two_bytes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_two_bytes: " + str(e))
    try:
        test_roundtrip_three_bytes()
        passed += 1
        print("  PASS test_roundtrip_three_bytes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_three_bytes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_three_bytes: " + str(e))
    try:
        test_roundtrip_longer()
        passed += 1
        print("  PASS test_roundtrip_longer")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_longer: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_longer: " + str(e))
    try:
        test_roundtrip_binary_pattern()
        passed += 1
        print("  PASS test_roundtrip_binary_pattern")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_binary_pattern: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_binary_pattern: " + str(e))
    try:
        test_encode_hello_world()
        passed += 1
        print("  PASS test_encode_hello_world")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_hello_world: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_hello_world: " + str(e))
    try:
        test_encode_digits()
        passed += 1
        print("  PASS test_encode_digits")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_digits: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_digits: " + str(e))
    try:
        test_encode_binary_data()
        passed += 1
        print("  PASS test_encode_binary_data")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_binary_data: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_binary_data: " + str(e))
    try:
        test_decode_hello_world()
        passed += 1
        print("  PASS test_decode_hello_world")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decode_hello_world: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decode_hello_world: " + str(e))
    try:
        test_decode_digits()
        passed += 1
        print("  PASS test_decode_digits")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decode_digits: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decode_digits: " + str(e))
    try:
        test_encode_produces_all_chars()
        passed += 1
        print("  PASS test_encode_produces_all_chars")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_produces_all_chars: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_produces_all_chars: " + str(e))
    try:
        test_decode_each_alphabet_char()
        passed += 1
        print("  PASS test_decode_each_alphabet_char")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decode_each_alphabet_char: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decode_each_alphabet_char: " + str(e))
    try:
        test_6bit_all_zeros()
        passed += 1
        print("  PASS test_6bit_all_zeros")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_6bit_all_zeros: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_6bit_all_zeros: " + str(e))
    try:
        test_6bit_all_ones()
        passed += 1
        print("  PASS test_6bit_all_ones")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_6bit_all_ones: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_6bit_all_ones: " + str(e))
    try:
        test_6bit_boundary()
        passed += 1
        print("  PASS test_6bit_boundary")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_6bit_boundary: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_6bit_boundary: " + str(e))
    try:
        test_6bit_high()
        passed += 1
        print("  PASS test_6bit_high")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_6bit_high: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_6bit_high: " + str(e))
    try:
        test_encode_pangram()
        passed += 1
        print("  PASS test_encode_pangram")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_pangram: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_pangram: " + str(e))
    try:
        test_encode_repeated()
        passed += 1
        print("  PASS test_encode_repeated")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_repeated: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_repeated: " + str(e))
    try:
        test_encode_long_binary()
        passed += 1
        print("  PASS test_encode_long_binary")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_long_binary: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_long_binary: " + str(e))
    try:
        test_encoded_length()
        passed += 1
        print("  PASS test_encoded_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encoded_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encoded_length: " + str(e))
    try:
        test_decoded_length()
        passed += 1
        print("  PASS test_decoded_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decoded_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decoded_length: " + str(e))
    try:
        test_invalid_char_position_0()
        passed += 1
        print("  PASS test_invalid_char_position_0")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_char_position_0: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_char_position_0: " + str(e))
    try:
        test_invalid_char_position_1()
        passed += 1
        print("  PASS test_invalid_char_position_1")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_char_position_1: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_char_position_1: " + str(e))
    try:
        test_invalid_char_position_2()
        passed += 1
        print("  PASS test_invalid_char_position_2")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_char_position_2: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_char_position_2: " + str(e))
    try:
        test_invalid_char_position_3()
        passed += 1
        print("  PASS test_invalid_char_position_3")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_char_position_3: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_char_position_3: " + str(e))
    try:
        test_invalid_char_in_second_group()
        passed += 1
        print("  PASS test_invalid_char_in_second_group")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_char_in_second_group: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_char_in_second_group: " + str(e))
    try:
        test_invalid_space()
        passed += 1
        print("  PASS test_invalid_space")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_space: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_space: " + str(e))
    try:
        test_invalid_newline()
        passed += 1
        print("  PASS test_invalid_newline")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_newline: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_newline: " + str(e))
    try:
        test_invalid_length_1()
        passed += 1
        print("  PASS test_invalid_length_1")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_length_1: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_length_1: " + str(e))
    try:
        test_invalid_length_2()
        passed += 1
        print("  PASS test_invalid_length_2")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_length_2: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_length_2: " + str(e))
    try:
        test_invalid_length_3()
        passed += 1
        print("  PASS test_invalid_length_3")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_length_3: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_length_3: " + str(e))
    try:
        test_invalid_length_5()
        passed += 1
        print("  PASS test_invalid_length_5")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_length_5: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_length_5: " + str(e))
    try:
        test_invalid_padding_middle()
        passed += 1
        print("  PASS test_invalid_padding_middle")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_padding_middle: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_padding_middle: " + str(e))
    try:
        test_invalid_padding_first()
        passed += 1
        print("  PASS test_invalid_padding_first")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_padding_first: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_padding_first: " + str(e))
    try:
        test_various_invalid_chars()
        passed += 1
        print("  PASS test_various_invalid_chars")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_various_invalid_chars: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_various_invalid_chars: " + str(e))
    try:
        test_strict_github_issue_example()
        passed += 1
        print("  PASS test_strict_github_issue_example")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_strict_github_issue_example: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_strict_github_issue_example: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
