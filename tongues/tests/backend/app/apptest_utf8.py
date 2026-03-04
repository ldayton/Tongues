"""UTF-8 codec tests — encoding, decoding, validation, and error detection."""

import sys

from lib.utf8 import codepoint_len
from lib.utf8 import decode
from lib.utf8 import decode_codepoint
from lib.utf8 import encode
from lib.utf8 import encode_codepoint
from lib.utf8 import is_valid
from lib.utf8 import Utf8Error


# -- Single codepoint encode --


def test_encode_ascii() -> None:
    assert encode_codepoint(0) == b"\x00"
    assert encode_codepoint(0x41) == b"A"
    assert encode_codepoint(0x7F) == b"\x7f"


def test_encode_2byte() -> None:
    assert encode_codepoint(0x80) == b"\xc2\x80"
    assert encode_codepoint(0xA9) == b"\xc2\xa9"
    assert encode_codepoint(0x7FF) == b"\xdf\xbf"


def test_encode_3byte() -> None:
    assert encode_codepoint(0x800) == b"\xe0\xa0\x80"
    assert encode_codepoint(0x20AC) == b"\xe2\x82\xac"
    assert encode_codepoint(0xFFFF) == b"\xef\xbf\xbf"


def test_encode_4byte() -> None:
    assert encode_codepoint(0x10000) == b"\xf0\x90\x80\x80"
    assert encode_codepoint(0x1F600) == b"\xf0\x9f\x98\x80"
    assert encode_codepoint(0x10FFFF) == b"\xf4\x8f\xbf\xbf"


def test_encode_surrogate_replaced() -> None:
    """Surrogates (U+D800..U+DFFF) produce replacement character."""
    replacement: bytes = encode_codepoint(0xFFFD)
    assert encode_codepoint(0xD800) == replacement
    assert encode_codepoint(0xDBFF) == replacement
    assert encode_codepoint(0xDC00) == replacement
    assert encode_codepoint(0xDFFF) == replacement


def test_encode_out_of_range_replaced() -> None:
    replacement: bytes = encode_codepoint(0xFFFD)
    assert encode_codepoint(0x110000) == replacement
    assert encode_codepoint(-1) == replacement


# -- Encode list --


def test_encode_empty() -> None:
    assert encode([]) == b""


def test_encode_ascii_string() -> None:
    assert encode([0x48, 0x65, 0x6C, 0x6C, 0x6F]) == b"Hello"


def test_encode_mixed() -> None:
    cps: list[int] = [0x41, 0xE9, 0x20AC, 0x1F600]
    result: bytes = encode(cps)
    assert result == b"\x41\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80"


# -- Single codepoint decode --


def test_decode_cp_ascii() -> None:
    result: tuple[int, int] = decode_codepoint(b"A", 0)
    assert result[0] == 0x41
    assert result[1] == 1


def test_decode_cp_2byte() -> None:
    result: tuple[int, int] = decode_codepoint(b"\xc3\xa9", 0)
    assert result[0] == 0xE9
    assert result[1] == 2


def test_decode_cp_3byte() -> None:
    result: tuple[int, int] = decode_codepoint(b"\xe2\x82\xac", 0)
    assert result[0] == 0x20AC
    assert result[1] == 3


def test_decode_cp_4byte() -> None:
    result: tuple[int, int] = decode_codepoint(b"\xf0\x9f\x98\x80", 0)
    assert result[0] == 0x1F600
    assert result[1] == 4


def test_decode_cp_offset() -> None:
    """Decode from a non-zero position."""
    data: bytes = b"A\xc3\xa9"
    result: tuple[int, int] = decode_codepoint(data, 1)
    assert result[0] == 0xE9
    assert result[1] == 3


# -- Decode full sequence --


def test_decode_empty() -> None:
    assert decode(b"") == []


def test_decode_ascii_string() -> None:
    assert decode(b"Hello") == [0x48, 0x65, 0x6C, 0x6C, 0x6F]


def test_decode_mixed() -> None:
    data: bytes = b"\x41\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80"
    assert decode(data) == [0x41, 0xE9, 0x20AC, 0x1F600]


# -- Roundtrip --


def test_roundtrip_ascii() -> None:
    cps: list[int] = [0x48, 0x65, 0x6C, 0x6C, 0x6F]
    assert decode(encode(cps)) == cps


def test_roundtrip_mixed() -> None:
    cps: list[int] = [0x41, 0xE9, 0x20AC, 0x1F600]
    assert decode(encode(cps)) == cps


def test_roundtrip_boundaries() -> None:
    """Boundary codepoints for each byte length."""
    cps: list[int] = [0x00, 0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x10FFFF]
    assert decode(encode(cps)) == cps


def test_roundtrip_all_ascii() -> None:
    cps: list[int] = []
    i: int = 0
    while i < 128:
        cps.append(i)
        i += 1
    assert decode(encode(cps)) == cps


# -- Invalid decode: truncated sequences --


def test_truncated_2byte() -> None:
    try:
        decode(b"\xc3")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_truncated_3byte_1() -> None:
    try:
        decode(b"\xe2")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_truncated_3byte_2() -> None:
    try:
        decode(b"\xe2\x82")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_truncated_4byte() -> None:
    try:
        decode(b"\xf0\x9f\x98")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


# -- Invalid decode: bad continuation bytes --


def test_bad_continuation_2byte() -> None:
    try:
        decode(b"\xc3\x00")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 1


def test_bad_continuation_3byte() -> None:
    try:
        decode(b"\xe2\x82\x00")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 2


def test_bad_continuation_4byte() -> None:
    try:
        decode(b"\xf0\x9f\x98\x00")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 3


# -- Invalid decode: overlong encodings --


def test_overlong_2byte() -> None:
    """C0 80 is overlong encoding of U+0000."""
    try:
        decode(b"\xc0\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_overlong_3byte() -> None:
    """E0 80 80 is overlong encoding of U+0000."""
    try:
        decode(b"\xe0\x80\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_overlong_4byte() -> None:
    """F0 80 80 80 is overlong encoding of U+0000."""
    try:
        decode(b"\xf0\x80\x80\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


# -- Invalid decode: surrogates --


def test_encoded_surrogate() -> None:
    """ED A0 80 = U+D800 (lead surrogate) — must reject."""
    try:
        decode(b"\xed\xa0\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_encoded_surrogate_trail() -> None:
    """ED BF BF = U+DFFF (trail surrogate) — must reject."""
    try:
        decode(b"\xed\xbf\xbf")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


# -- Invalid decode: out of range --


def test_above_max() -> None:
    """F4 90 80 80 = U+110000 — above max codepoint."""
    try:
        decode(b"\xf4\x90\x80\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_fe_byte() -> None:
    """0xFE is never valid in UTF-8."""
    try:
        decode(b"\xfe")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_ff_byte() -> None:
    """0xFF is never valid in UTF-8."""
    try:
        decode(b"\xff")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_bare_continuation() -> None:
    """A continuation byte (0x80-0xBF) without a lead byte."""
    try:
        decode(b"\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


# -- Invalid byte mid-sequence --


def test_error_position_after_valid() -> None:
    """Error position is reported relative to the bad byte, not the start."""
    try:
        decode(b"AB\xff")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 2


def test_error_position_mid_multibyte() -> None:
    try:
        decode(b"A\xc3\x00")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 2


# -- is_valid --


def test_valid_empty() -> None:
    assert is_valid(b"")


def test_valid_ascii() -> None:
    assert is_valid(b"Hello, world!")


def test_valid_multibyte() -> None:
    assert is_valid(b"\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80")


def test_invalid_detected() -> None:
    assert not is_valid(b"\xff")
    assert not is_valid(b"\xc0\x80")
    assert not is_valid(b"abc\xfe")


# -- codepoint_len --


def test_len_empty() -> None:
    assert codepoint_len(b"") == 0


def test_len_ascii() -> None:
    assert codepoint_len(b"Hello") == 5


def test_len_multibyte() -> None:
    data: bytes = b"\x41\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80"
    assert codepoint_len(data) == 4


def test_len_all_4byte() -> None:
    data: bytes = encode([0x1F600, 0x1F601, 0x1F602])
    assert codepoint_len(data) == 3
    assert len(data) == 12


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_encode_ascii", test_encode_ascii),
        ("test_encode_2byte", test_encode_2byte),
        ("test_encode_3byte", test_encode_3byte),
        ("test_encode_4byte", test_encode_4byte),
        ("test_encode_surrogate_replaced", test_encode_surrogate_replaced),
        ("test_encode_out_of_range_replaced", test_encode_out_of_range_replaced),
        ("test_encode_empty", test_encode_empty),
        ("test_encode_ascii_string", test_encode_ascii_string),
        ("test_encode_mixed", test_encode_mixed),
        ("test_decode_cp_ascii", test_decode_cp_ascii),
        ("test_decode_cp_2byte", test_decode_cp_2byte),
        ("test_decode_cp_3byte", test_decode_cp_3byte),
        ("test_decode_cp_4byte", test_decode_cp_4byte),
        ("test_decode_cp_offset", test_decode_cp_offset),
        ("test_decode_empty", test_decode_empty),
        ("test_decode_ascii_string", test_decode_ascii_string),
        ("test_decode_mixed", test_decode_mixed),
        ("test_roundtrip_ascii", test_roundtrip_ascii),
        ("test_roundtrip_mixed", test_roundtrip_mixed),
        ("test_roundtrip_boundaries", test_roundtrip_boundaries),
        ("test_roundtrip_all_ascii", test_roundtrip_all_ascii),
        ("test_truncated_2byte", test_truncated_2byte),
        ("test_truncated_3byte_1", test_truncated_3byte_1),
        ("test_truncated_3byte_2", test_truncated_3byte_2),
        ("test_truncated_4byte", test_truncated_4byte),
        ("test_bad_continuation_2byte", test_bad_continuation_2byte),
        ("test_bad_continuation_3byte", test_bad_continuation_3byte),
        ("test_bad_continuation_4byte", test_bad_continuation_4byte),
        ("test_overlong_2byte", test_overlong_2byte),
        ("test_overlong_3byte", test_overlong_3byte),
        ("test_overlong_4byte", test_overlong_4byte),
        ("test_encoded_surrogate", test_encoded_surrogate),
        ("test_encoded_surrogate_trail", test_encoded_surrogate_trail),
        ("test_above_max", test_above_max),
        ("test_fe_byte", test_fe_byte),
        ("test_ff_byte", test_ff_byte),
        ("test_bare_continuation", test_bare_continuation),
        ("test_error_position_after_valid", test_error_position_after_valid),
        ("test_error_position_mid_multibyte", test_error_position_mid_multibyte),
        ("test_valid_empty", test_valid_empty),
        ("test_valid_ascii", test_valid_ascii),
        ("test_valid_multibyte", test_valid_multibyte),
        ("test_invalid_detected", test_invalid_detected),
        ("test_len_empty", test_len_empty),
        ("test_len_ascii", test_len_ascii),
        ("test_len_multibyte", test_len_multibyte),
        ("test_len_all_4byte", test_len_all_4byte),
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
