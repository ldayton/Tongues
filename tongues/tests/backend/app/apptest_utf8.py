"""UTF-8 codec tests — encoding, decoding, validation, and error detection."""

import sys

from lib.utf8 import utf8_codepoint_len
from lib.utf8 import utf8_decode
from lib.utf8 import utf8_decode_codepoint
from lib.utf8 import utf8_encode
from lib.utf8 import utf8_encode_codepoint
from lib.utf8 import utf8_is_valid
from lib.utf8 import Utf8Error


# -- Single codepoint encode --


def test_encode_ascii() -> None:
    assert utf8_encode_codepoint(0) == b"\x00"
    assert utf8_encode_codepoint(0x41) == b"A"
    assert utf8_encode_codepoint(0x7F) == b"\x7f"


def test_encode_2byte() -> None:
    assert utf8_encode_codepoint(0x80) == b"\xc2\x80"
    assert utf8_encode_codepoint(0xA9) == b"\xc2\xa9"
    assert utf8_encode_codepoint(0x7FF) == b"\xdf\xbf"


def test_encode_3byte() -> None:
    assert utf8_encode_codepoint(0x800) == b"\xe0\xa0\x80"
    assert utf8_encode_codepoint(0x20AC) == b"\xe2\x82\xac"
    assert utf8_encode_codepoint(0xFFFF) == b"\xef\xbf\xbf"


def test_encode_4byte() -> None:
    assert utf8_encode_codepoint(0x10000) == b"\xf0\x90\x80\x80"
    assert utf8_encode_codepoint(0x1F600) == b"\xf0\x9f\x98\x80"
    assert utf8_encode_codepoint(0x10FFFF) == b"\xf4\x8f\xbf\xbf"


def test_encode_surrogate_replaced() -> None:
    """Surrogates (U+D800..U+DFFF) produce replacement character."""
    replacement: bytes = utf8_encode_codepoint(0xFFFD)
    assert utf8_encode_codepoint(0xD800) == replacement
    assert utf8_encode_codepoint(0xDBFF) == replacement
    assert utf8_encode_codepoint(0xDC00) == replacement
    assert utf8_encode_codepoint(0xDFFF) == replacement


def test_encode_out_of_range_replaced() -> None:
    replacement: bytes = utf8_encode_codepoint(0xFFFD)
    assert utf8_encode_codepoint(0x110000) == replacement
    assert utf8_encode_codepoint(-1) == replacement


# -- Encode list --


def test_encode_empty() -> None:
    assert utf8_encode([]) == b""


def test_encode_ascii_string() -> None:
    assert utf8_encode([0x48, 0x65, 0x6C, 0x6C, 0x6F]) == b"Hello"


def test_encode_mixed() -> None:
    cps: list[int] = [0x41, 0xE9, 0x20AC, 0x1F600]
    result: bytes = utf8_encode(cps)
    assert result == b"\x41\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80"


# -- Single codepoint decode --


def test_decode_cp_ascii() -> None:
    result: tuple[int, int] = utf8_decode_codepoint(b"A", 0)
    assert result[0] == 0x41
    assert result[1] == 1


def test_decode_cp_2byte() -> None:
    result: tuple[int, int] = utf8_decode_codepoint(b"\xc3\xa9", 0)
    assert result[0] == 0xE9
    assert result[1] == 2


def test_decode_cp_3byte() -> None:
    result: tuple[int, int] = utf8_decode_codepoint(b"\xe2\x82\xac", 0)
    assert result[0] == 0x20AC
    assert result[1] == 3


def test_decode_cp_4byte() -> None:
    result: tuple[int, int] = utf8_decode_codepoint(b"\xf0\x9f\x98\x80", 0)
    assert result[0] == 0x1F600
    assert result[1] == 4


def test_decode_cp_offset() -> None:
    """Decode from a non-zero position."""
    data: bytes = b"A\xc3\xa9"
    result: tuple[int, int] = utf8_decode_codepoint(data, 1)
    assert result[0] == 0xE9
    assert result[1] == 3


# -- Decode full sequence --


def test_decode_empty() -> None:
    assert utf8_decode(b"") == []


def test_decode_ascii_string() -> None:
    assert utf8_decode(b"Hello") == [0x48, 0x65, 0x6C, 0x6C, 0x6F]


def test_decode_mixed() -> None:
    data: bytes = b"\x41\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80"
    assert utf8_decode(data) == [0x41, 0xE9, 0x20AC, 0x1F600]


# -- Roundtrip --


def test_roundtrip_ascii() -> None:
    cps: list[int] = [0x48, 0x65, 0x6C, 0x6C, 0x6F]
    assert utf8_decode(utf8_encode(cps)) == cps


def test_roundtrip_mixed() -> None:
    cps: list[int] = [0x41, 0xE9, 0x20AC, 0x1F600]
    assert utf8_decode(utf8_encode(cps)) == cps


def test_roundtrip_boundaries() -> None:
    """Boundary codepoints for each byte length."""
    cps: list[int] = [0x00, 0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x10FFFF]
    assert utf8_decode(utf8_encode(cps)) == cps


def test_roundtrip_all_ascii() -> None:
    cps: list[int] = []
    i: int = 0
    while i < 128:
        cps.append(i)
        i += 1
    assert utf8_decode(utf8_encode(cps)) == cps


# -- Invalid decode: truncated sequences --


def test_truncated_2byte() -> None:
    try:
        utf8_decode(b"\xc3")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_truncated_3byte_1() -> None:
    try:
        utf8_decode(b"\xe2")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_truncated_3byte_2() -> None:
    try:
        utf8_decode(b"\xe2\x82")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_truncated_4byte() -> None:
    try:
        utf8_decode(b"\xf0\x9f\x98")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


# -- Invalid decode: bad continuation bytes --


def test_bad_continuation_2byte() -> None:
    try:
        utf8_decode(b"\xc3\x00")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 1


def test_bad_continuation_3byte() -> None:
    try:
        utf8_decode(b"\xe2\x82\x00")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 2


def test_bad_continuation_4byte() -> None:
    try:
        utf8_decode(b"\xf0\x9f\x98\x00")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 3


# -- Invalid decode: overlong encodings --


def test_overlong_2byte() -> None:
    """C0 80 is overlong encoding of U+0000."""
    try:
        utf8_decode(b"\xc0\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_overlong_3byte() -> None:
    """E0 80 80 is overlong encoding of U+0000."""
    try:
        utf8_decode(b"\xe0\x80\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_overlong_3byte_max() -> None:
    """E0 9F BF encodes U+07FF — valid codepoint, but must use 2 bytes."""
    try:
        utf8_decode(b"\xe0\x9f\xbf")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_overlong_3byte_boundary() -> None:
    """E0 A0 80 = U+0800 — first valid 3-byte encoding."""
    result: tuple[int, int] = utf8_decode_codepoint(b"\xe0\xa0\x80", 0)
    assert result[0] == 0x800
    assert result[1] == 3


def test_overlong_4byte() -> None:
    """F0 80 80 80 is overlong encoding of U+0000."""
    try:
        utf8_decode(b"\xf0\x80\x80\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_overlong_4byte_max() -> None:
    """F0 8F BF BF encodes U+FFFF — valid codepoint, but must use 3 bytes."""
    try:
        utf8_decode(b"\xf0\x8f\xbf\xbf")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_overlong_4byte_boundary() -> None:
    """F0 90 80 80 = U+10000 — first valid 4-byte encoding."""
    result: tuple[int, int] = utf8_decode_codepoint(b"\xf0\x90\x80\x80", 0)
    assert result[0] == 0x10000
    assert result[1] == 4


# -- Invalid decode: surrogates --


def test_encoded_surrogate() -> None:
    """ED A0 80 = U+D800 (lead surrogate) — must reject."""
    try:
        utf8_decode(b"\xed\xa0\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_encoded_surrogate_trail() -> None:
    """ED BF BF = U+DFFF (trail surrogate) — must reject."""
    try:
        utf8_decode(b"\xed\xbf\xbf")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_surrogate_boundary_before() -> None:
    """U+D7FF — last codepoint before surrogate range, must be valid."""
    result: tuple[int, int] = utf8_decode_codepoint(b"\xed\x9f\xbf", 0)
    assert result[0] == 0xD7FF


def test_surrogate_boundary_after() -> None:
    """U+E000 — first codepoint after surrogate range, must be valid."""
    result: tuple[int, int] = utf8_decode_codepoint(b"\xee\x80\x80", 0)
    assert result[0] == 0xE000


# -- Invalid decode: out of range --


def test_above_max() -> None:
    """F4 90 80 80 = U+110000 — above max codepoint."""
    try:
        utf8_decode(b"\xf4\x90\x80\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_f5_byte() -> None:
    """F5 would start a sequence for U+140000+ — always invalid."""
    try:
        utf8_decode(b"\xf5\x80\x80\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_fe_byte() -> None:
    """0xFE is never valid in UTF-8."""
    try:
        utf8_decode(b"\xfe")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_ff_byte() -> None:
    """0xFF is never valid in UTF-8."""
    try:
        utf8_decode(b"\xff")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_bare_continuation() -> None:
    """A continuation byte (0x80-0xBF) without a lead byte."""
    try:
        utf8_decode(b"\x80")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


def test_bare_continuation_bf() -> None:
    """0xBF — highest continuation byte, still invalid as lead."""
    try:
        utf8_decode(b"\xbf")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 0


# -- Invalid byte mid-sequence --


def test_error_position_after_valid() -> None:
    """Error position is reported relative to the bad byte, not the start."""
    try:
        utf8_decode(b"AB\xff")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 2


def test_error_position_mid_multibyte() -> None:
    try:
        utf8_decode(b"A\xc3\x00")
        assert False, "expected Utf8Error"
    except Utf8Error as e:
        assert e.position == 2


# -- is_valid --


def test_valid_empty() -> None:
    assert utf8_is_valid(b"")


def test_valid_ascii() -> None:
    assert utf8_is_valid(b"Hello, world!")


def test_valid_multibyte() -> None:
    assert utf8_is_valid(b"\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80")


def test_invalid_detected() -> None:
    assert not utf8_is_valid(b"\xff")
    assert not utf8_is_valid(b"\xc0\x80")
    assert not utf8_is_valid(b"abc\xfe")


# -- Noncharacters (valid codepoints, must not be rejected) --


def test_noncharacter_fffe() -> None:
    """U+FFFE is a noncharacter but valid in UTF-8."""
    cps: list[int] = utf8_decode(b"\xef\xbf\xbe")
    assert cps == [0xFFFE]


def test_noncharacter_ffff() -> None:
    """U+FFFF is a noncharacter but valid in UTF-8."""
    cps: list[int] = utf8_decode(b"\xef\xbf\xbf")
    assert cps == [0xFFFF]


def test_roundtrip_noncharacters() -> None:
    cps: list[int] = [0xFFFE, 0xFFFF]
    assert utf8_decode(utf8_encode(cps)) == cps


# -- Encode surrogate/OOR boundary roundtrips --


def test_encode_surrogate_boundaries() -> None:
    """U+D7FF and U+E000 should roundtrip cleanly."""
    cps: list[int] = [0xD7FF, 0xE000]
    assert utf8_decode(utf8_encode(cps)) == cps


def test_encode_max_codepoint_roundtrip() -> None:
    cps: list[int] = [0x10FFFF]
    assert utf8_decode(utf8_encode(cps)) == cps


# -- codepoint_len --


def test_len_empty() -> None:
    assert utf8_codepoint_len(b"") == 0


def test_len_ascii() -> None:
    assert utf8_codepoint_len(b"Hello") == 5


def test_len_multibyte() -> None:
    data: bytes = b"\x41\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80"
    assert utf8_codepoint_len(data) == 4


def test_len_all_4byte() -> None:
    data: bytes = utf8_encode([0x1F600, 0x1F601, 0x1F602])
    assert utf8_codepoint_len(data) == 3
    assert len(data) == 12


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_encode_ascii()
        passed += 1
        print("  PASS test_encode_ascii")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_ascii: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_ascii: " + str(e))
    try:
        test_encode_2byte()
        passed += 1
        print("  PASS test_encode_2byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_2byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_2byte: " + str(e))
    try:
        test_encode_3byte()
        passed += 1
        print("  PASS test_encode_3byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_3byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_3byte: " + str(e))
    try:
        test_encode_4byte()
        passed += 1
        print("  PASS test_encode_4byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_4byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_4byte: " + str(e))
    try:
        test_encode_surrogate_replaced()
        passed += 1
        print("  PASS test_encode_surrogate_replaced")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_surrogate_replaced: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_surrogate_replaced: " + str(e))
    try:
        test_encode_out_of_range_replaced()
        passed += 1
        print("  PASS test_encode_out_of_range_replaced")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_out_of_range_replaced: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_out_of_range_replaced: " + str(e))
    try:
        test_encode_empty()
        passed += 1
        print("  PASS test_encode_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_empty: " + str(e))
    try:
        test_encode_ascii_string()
        passed += 1
        print("  PASS test_encode_ascii_string")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_ascii_string: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_ascii_string: " + str(e))
    try:
        test_encode_mixed()
        passed += 1
        print("  PASS test_encode_mixed")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_mixed: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_mixed: " + str(e))
    try:
        test_decode_cp_ascii()
        passed += 1
        print("  PASS test_decode_cp_ascii")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decode_cp_ascii: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decode_cp_ascii: " + str(e))
    try:
        test_decode_cp_2byte()
        passed += 1
        print("  PASS test_decode_cp_2byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decode_cp_2byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decode_cp_2byte: " + str(e))
    try:
        test_decode_cp_3byte()
        passed += 1
        print("  PASS test_decode_cp_3byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decode_cp_3byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decode_cp_3byte: " + str(e))
    try:
        test_decode_cp_4byte()
        passed += 1
        print("  PASS test_decode_cp_4byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decode_cp_4byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decode_cp_4byte: " + str(e))
    try:
        test_decode_cp_offset()
        passed += 1
        print("  PASS test_decode_cp_offset")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decode_cp_offset: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decode_cp_offset: " + str(e))
    try:
        test_decode_empty()
        passed += 1
        print("  PASS test_decode_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decode_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decode_empty: " + str(e))
    try:
        test_decode_ascii_string()
        passed += 1
        print("  PASS test_decode_ascii_string")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decode_ascii_string: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decode_ascii_string: " + str(e))
    try:
        test_decode_mixed()
        passed += 1
        print("  PASS test_decode_mixed")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_decode_mixed: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_decode_mixed: " + str(e))
    try:
        test_roundtrip_ascii()
        passed += 1
        print("  PASS test_roundtrip_ascii")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_ascii: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_ascii: " + str(e))
    try:
        test_roundtrip_mixed()
        passed += 1
        print("  PASS test_roundtrip_mixed")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_mixed: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_mixed: " + str(e))
    try:
        test_roundtrip_boundaries()
        passed += 1
        print("  PASS test_roundtrip_boundaries")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_boundaries: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_boundaries: " + str(e))
    try:
        test_roundtrip_all_ascii()
        passed += 1
        print("  PASS test_roundtrip_all_ascii")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_all_ascii: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_all_ascii: " + str(e))
    try:
        test_truncated_2byte()
        passed += 1
        print("  PASS test_truncated_2byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_truncated_2byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_truncated_2byte: " + str(e))
    try:
        test_truncated_3byte_1()
        passed += 1
        print("  PASS test_truncated_3byte_1")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_truncated_3byte_1: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_truncated_3byte_1: " + str(e))
    try:
        test_truncated_3byte_2()
        passed += 1
        print("  PASS test_truncated_3byte_2")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_truncated_3byte_2: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_truncated_3byte_2: " + str(e))
    try:
        test_truncated_4byte()
        passed += 1
        print("  PASS test_truncated_4byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_truncated_4byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_truncated_4byte: " + str(e))
    try:
        test_bad_continuation_2byte()
        passed += 1
        print("  PASS test_bad_continuation_2byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bad_continuation_2byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bad_continuation_2byte: " + str(e))
    try:
        test_bad_continuation_3byte()
        passed += 1
        print("  PASS test_bad_continuation_3byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bad_continuation_3byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bad_continuation_3byte: " + str(e))
    try:
        test_bad_continuation_4byte()
        passed += 1
        print("  PASS test_bad_continuation_4byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bad_continuation_4byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bad_continuation_4byte: " + str(e))
    try:
        test_overlong_2byte()
        passed += 1
        print("  PASS test_overlong_2byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_overlong_2byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_overlong_2byte: " + str(e))
    try:
        test_overlong_3byte()
        passed += 1
        print("  PASS test_overlong_3byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_overlong_3byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_overlong_3byte: " + str(e))
    try:
        test_overlong_3byte_max()
        passed += 1
        print("  PASS test_overlong_3byte_max")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_overlong_3byte_max: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_overlong_3byte_max: " + str(e))
    try:
        test_overlong_3byte_boundary()
        passed += 1
        print("  PASS test_overlong_3byte_boundary")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_overlong_3byte_boundary: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_overlong_3byte_boundary: " + str(e))
    try:
        test_overlong_4byte()
        passed += 1
        print("  PASS test_overlong_4byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_overlong_4byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_overlong_4byte: " + str(e))
    try:
        test_overlong_4byte_max()
        passed += 1
        print("  PASS test_overlong_4byte_max")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_overlong_4byte_max: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_overlong_4byte_max: " + str(e))
    try:
        test_overlong_4byte_boundary()
        passed += 1
        print("  PASS test_overlong_4byte_boundary")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_overlong_4byte_boundary: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_overlong_4byte_boundary: " + str(e))
    try:
        test_encoded_surrogate()
        passed += 1
        print("  PASS test_encoded_surrogate")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encoded_surrogate: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encoded_surrogate: " + str(e))
    try:
        test_encoded_surrogate_trail()
        passed += 1
        print("  PASS test_encoded_surrogate_trail")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encoded_surrogate_trail: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encoded_surrogate_trail: " + str(e))
    try:
        test_surrogate_boundary_before()
        passed += 1
        print("  PASS test_surrogate_boundary_before")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_surrogate_boundary_before: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_surrogate_boundary_before: " + str(e))
    try:
        test_surrogate_boundary_after()
        passed += 1
        print("  PASS test_surrogate_boundary_after")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_surrogate_boundary_after: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_surrogate_boundary_after: " + str(e))
    try:
        test_above_max()
        passed += 1
        print("  PASS test_above_max")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_above_max: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_above_max: " + str(e))
    try:
        test_f5_byte()
        passed += 1
        print("  PASS test_f5_byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_f5_byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_f5_byte: " + str(e))
    try:
        test_fe_byte()
        passed += 1
        print("  PASS test_fe_byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_fe_byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_fe_byte: " + str(e))
    try:
        test_ff_byte()
        passed += 1
        print("  PASS test_ff_byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_ff_byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_ff_byte: " + str(e))
    try:
        test_bare_continuation()
        passed += 1
        print("  PASS test_bare_continuation")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bare_continuation: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bare_continuation: " + str(e))
    try:
        test_bare_continuation_bf()
        passed += 1
        print("  PASS test_bare_continuation_bf")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bare_continuation_bf: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bare_continuation_bf: " + str(e))
    try:
        test_error_position_after_valid()
        passed += 1
        print("  PASS test_error_position_after_valid")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_error_position_after_valid: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_error_position_after_valid: " + str(e))
    try:
        test_error_position_mid_multibyte()
        passed += 1
        print("  PASS test_error_position_mid_multibyte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_error_position_mid_multibyte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_error_position_mid_multibyte: " + str(e))
    try:
        test_valid_empty()
        passed += 1
        print("  PASS test_valid_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_valid_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_valid_empty: " + str(e))
    try:
        test_valid_ascii()
        passed += 1
        print("  PASS test_valid_ascii")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_valid_ascii: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_valid_ascii: " + str(e))
    try:
        test_valid_multibyte()
        passed += 1
        print("  PASS test_valid_multibyte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_valid_multibyte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_valid_multibyte: " + str(e))
    try:
        test_invalid_detected()
        passed += 1
        print("  PASS test_invalid_detected")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_invalid_detected: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_invalid_detected: " + str(e))
    try:
        test_noncharacter_fffe()
        passed += 1
        print("  PASS test_noncharacter_fffe")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_noncharacter_fffe: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_noncharacter_fffe: " + str(e))
    try:
        test_noncharacter_ffff()
        passed += 1
        print("  PASS test_noncharacter_ffff")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_noncharacter_ffff: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_noncharacter_ffff: " + str(e))
    try:
        test_roundtrip_noncharacters()
        passed += 1
        print("  PASS test_roundtrip_noncharacters")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_noncharacters: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_noncharacters: " + str(e))
    try:
        test_encode_surrogate_boundaries()
        passed += 1
        print("  PASS test_encode_surrogate_boundaries")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_surrogate_boundaries: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_surrogate_boundaries: " + str(e))
    try:
        test_encode_max_codepoint_roundtrip()
        passed += 1
        print("  PASS test_encode_max_codepoint_roundtrip")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_encode_max_codepoint_roundtrip: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_encode_max_codepoint_roundtrip: " + str(e))
    try:
        test_len_empty()
        passed += 1
        print("  PASS test_len_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_len_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_len_empty: " + str(e))
    try:
        test_len_ascii()
        passed += 1
        print("  PASS test_len_ascii")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_len_ascii: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_len_ascii: " + str(e))
    try:
        test_len_multibyte()
        passed += 1
        print("  PASS test_len_multibyte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_len_multibyte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_len_multibyte: " + str(e))
    try:
        test_len_all_4byte()
        passed += 1
        print("  PASS test_len_all_4byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_len_all_4byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_len_all_4byte: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
