"""CRC-32 tests — known vectors, incremental update, hex formatting."""

import sys

from lib.crc32 import crc32
from lib.crc32 import crc32_hex
from lib.crc32 import crc32_update


# -- Known vectors --


def test_empty() -> None:
    assert crc32(b"") == 0


def test_123456789() -> None:
    """The canonical CRC-32 check value."""
    assert crc32(b"123456789") == 0xCBF43926


def test_single_a() -> None:
    assert crc32(b"a") == 0xE8B7BE43


def test_abc() -> None:
    assert crc32(b"abc") == 0x352441C2


def test_hello_world() -> None:
    assert crc32(b"hello world") == 0x0D4A1185


def test_single_zero_byte() -> None:
    assert crc32(b"\x00") == 0xD202EF8D


def test_single_ff_byte() -> None:
    assert crc32(bytes([255])) == 0xFF000000


def test_all_zeros_4() -> None:
    assert crc32(bytes([0, 0, 0, 0])) == 0x2144DF1C


def test_sequential_bytes() -> None:
    """0x00 through 0x03."""
    assert crc32(bytes([0, 1, 2, 3])) == 0x8BB98613


def test_alphabet() -> None:
    assert crc32(b"abcdefghijklmnopqrstuvwxyz") == 0x4C2750BD


# -- Incremental update --


def test_update_empty_then_data() -> None:
    """Incremental: start from 0, feed data."""
    c: int = crc32_update(0, b"1234")
    c = crc32_update(c, b"56789")
    assert c == 0xCBF43926


def test_update_byte_at_a_time() -> None:
    """Feed one byte at a time."""
    c: int = 0
    c = crc32_update(c, b"a")
    c = crc32_update(c, b"b")
    c = crc32_update(c, b"c")
    assert c == 0x352441C2


def test_update_matches_oneshot() -> None:
    """Incremental split matches one-shot for longer data."""
    full: bytes = b"hello world"
    c: int = crc32_update(0, b"hello")
    c = crc32_update(c, b" ")
    c = crc32_update(c, b"world")
    assert c == crc32(full)


# -- Hex output --


def test_hex_empty() -> None:
    assert crc32_hex(b"") == "00000000"


def test_hex_123456789() -> None:
    assert crc32_hex(b"123456789") == "cbf43926"


def test_hex_abc() -> None:
    assert crc32_hex(b"abc") == "352441c2"


def test_hex_single_a() -> None:
    assert crc32_hex(b"a") == "e8b7be43"


def test_hex_length() -> None:
    """Hex output is always 8 characters."""
    result: str = crc32_hex(b"test")
    assert len(result) == 8


def test_hex_lowercase() -> None:
    """Hex output uses lowercase."""
    result: str = crc32_hex(b"123456789")
    i: int = 0
    while i < len(result):
        c: str = result[i]
        if c >= "A" and c <= "F":
            assert False, "expected lowercase hex"
        i += 1


# -- Edge cases --


def test_long_repeated() -> None:
    """256 bytes of 0x41 ('A')."""
    data: list[int] = []
    i: int = 0
    while i < 256:
        data.append(65)
        i += 1
    assert crc32(bytes(data)) == 0x49975B13


def test_256_sequential() -> None:
    """Bytes 0x00 through 0xFF."""
    data: list[int] = []
    i: int = 0
    while i < 256:
        data.append(i)
        i += 1
    assert crc32(bytes(data)) == 0x29058C73


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_empty()
        passed += 1
        print("  PASS test_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_empty: " + str(e))
    try:
        test_123456789()
        passed += 1
        print("  PASS test_123456789")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_123456789: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_123456789: " + str(e))
    try:
        test_single_a()
        passed += 1
        print("  PASS test_single_a")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_single_a: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_single_a: " + str(e))
    try:
        test_abc()
        passed += 1
        print("  PASS test_abc")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_abc: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_abc: " + str(e))
    try:
        test_hello_world()
        passed += 1
        print("  PASS test_hello_world")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_hello_world: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_hello_world: " + str(e))
    try:
        test_single_zero_byte()
        passed += 1
        print("  PASS test_single_zero_byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_single_zero_byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_single_zero_byte: " + str(e))
    try:
        test_single_ff_byte()
        passed += 1
        print("  PASS test_single_ff_byte")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_single_ff_byte: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_single_ff_byte: " + str(e))
    try:
        test_all_zeros_4()
        passed += 1
        print("  PASS test_all_zeros_4")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_all_zeros_4: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_all_zeros_4: " + str(e))
    try:
        test_sequential_bytes()
        passed += 1
        print("  PASS test_sequential_bytes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_sequential_bytes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_sequential_bytes: " + str(e))
    try:
        test_alphabet()
        passed += 1
        print("  PASS test_alphabet")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_alphabet: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_alphabet: " + str(e))
    try:
        test_update_empty_then_data()
        passed += 1
        print("  PASS test_update_empty_then_data")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_update_empty_then_data: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_update_empty_then_data: " + str(e))
    try:
        test_update_byte_at_a_time()
        passed += 1
        print("  PASS test_update_byte_at_a_time")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_update_byte_at_a_time: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_update_byte_at_a_time: " + str(e))
    try:
        test_update_matches_oneshot()
        passed += 1
        print("  PASS test_update_matches_oneshot")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_update_matches_oneshot: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_update_matches_oneshot: " + str(e))
    try:
        test_hex_empty()
        passed += 1
        print("  PASS test_hex_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_hex_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_hex_empty: " + str(e))
    try:
        test_hex_123456789()
        passed += 1
        print("  PASS test_hex_123456789")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_hex_123456789: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_hex_123456789: " + str(e))
    try:
        test_hex_abc()
        passed += 1
        print("  PASS test_hex_abc")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_hex_abc: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_hex_abc: " + str(e))
    try:
        test_hex_single_a()
        passed += 1
        print("  PASS test_hex_single_a")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_hex_single_a: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_hex_single_a: " + str(e))
    try:
        test_hex_length()
        passed += 1
        print("  PASS test_hex_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_hex_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_hex_length: " + str(e))
    try:
        test_hex_lowercase()
        passed += 1
        print("  PASS test_hex_lowercase")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_hex_lowercase: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_hex_lowercase: " + str(e))
    try:
        test_long_repeated()
        passed += 1
        print("  PASS test_long_repeated")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_long_repeated: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_long_repeated: " + str(e))
    try:
        test_256_sequential()
        passed += 1
        print("  PASS test_256_sequential")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_256_sequential: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_256_sequential: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
