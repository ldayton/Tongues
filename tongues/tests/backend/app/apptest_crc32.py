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
    data: bytes = b"abc"
    c: int = 0
    i: int = 0
    while i < len(data):
        c = crc32_update(c, bytes([data[i]]))
        i += 1
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
    tests = [
        ("test_empty", test_empty),
        ("test_123456789", test_123456789),
        ("test_single_a", test_single_a),
        ("test_abc", test_abc),
        ("test_hello_world", test_hello_world),
        ("test_single_zero_byte", test_single_zero_byte),
        ("test_single_ff_byte", test_single_ff_byte),
        ("test_all_zeros_4", test_all_zeros_4),
        ("test_sequential_bytes", test_sequential_bytes),
        ("test_alphabet", test_alphabet),
        ("test_update_empty_then_data", test_update_empty_then_data),
        ("test_update_byte_at_a_time", test_update_byte_at_a_time),
        ("test_update_matches_oneshot", test_update_matches_oneshot),
        ("test_hex_empty", test_hex_empty),
        ("test_hex_123456789", test_hex_123456789),
        ("test_hex_abc", test_hex_abc),
        ("test_hex_single_a", test_hex_single_a),
        ("test_hex_length", test_hex_length),
        ("test_hex_lowercase", test_hex_lowercase),
        ("test_long_repeated", test_long_repeated),
        ("test_256_sequential", test_256_sequential),
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
