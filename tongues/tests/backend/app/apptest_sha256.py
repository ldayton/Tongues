"""SHA-256 tests — NIST FIPS 180-4 vectors and properties."""

import sys

from lib.sha256 import sha256
from lib.sha256 import sha256_bytes


# -- NIST test vectors --


def test_empty() -> None:
    assert (
        sha256(b"")
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_abc() -> None:
    assert (
        sha256(b"abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_two_block_448bit() -> None:
    assert (
        sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
        == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    )


def test_single_a() -> None:
    assert (
        sha256(b"a")
        == "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
    )


def test_two_block_896bit() -> None:
    assert (
        sha256(
            b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
        )
        == "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
    )


# -- sha256_bytes properties --


def test_bytes_length() -> None:
    assert len(sha256_bytes(b"")) == 32
    assert len(sha256_bytes(b"abc")) == 32
    assert len(sha256_bytes(b"hello world")) == 32


def test_bytes_hex_roundtrip() -> None:
    """sha256_bytes → manual hex should match sha256."""
    hex_chars: str = "0123456789abcdef"
    data: bytes = b"test roundtrip"
    raw: bytes = sha256_bytes(data)
    parts: list[str] = []
    i: int = 0
    while i < len(raw):
        b: int = raw[i]
        parts.append(hex_chars[b >> 4])
        parts.append(hex_chars[b & 0x0F])
        i += 1
    assert "".join(parts) == sha256(data)


# -- Padding boundary cases --


def test_55_bytes() -> None:
    """55 bytes — last single-block message (55 + 1 + 8 = 64)."""
    data: bytes = b"abcdefghijklmnopqrstuvwxyz01234567890123456789012345678"
    assert len(data) == 55
    assert (
        sha256(data)
        == "ca77d20b968bf7c2ae0d7898c10f6ef677b6138e6943f11c823cfa02a696c39e"
    )


def test_56_bytes() -> None:
    """56 bytes — first length that spills padding into a second block."""
    data: bytes = b"abcdefghijklmnopqrstuvwxyz012345678901234567890123456789"
    assert len(data) == 56
    assert (
        sha256(data)
        == "b3866906e306798f445559eda8c5f9da978fdbdfc972e660a0cd79147311ad22"
    )


def test_64_bytes() -> None:
    """Exactly one full block of input."""
    data: bytes = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!@"
    assert len(data) == 64
    assert (
        sha256(data)
        == "8bd8b71acf927db5f94100ae137bfb5769ee57d60b95dbbab294173ef073c01a"
    )


def test_128_bytes() -> None:
    """Multi-block: 128 bytes = 3 blocks after padding."""
    data: bytes = b"a" * 128
    assert (
        sha256(data)
        == "6836cf13bac400e9105071cd6af47084dfacad4e5e302c94bfed24e013afb73e"
    )


# -- Byte-value edge cases --


def test_null_byte() -> None:
    assert (
        sha256(b"\x00")
        == "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"
    )


def test_ff_byte() -> None:
    assert (
        sha256(b"\xff")
        == "a8100ae6aa1940d0b663bb31cd466142ebbdbd5187131b92d93818987832eb89"
    )


def test_null_block() -> None:
    """64 zero bytes."""
    data: bytes = bytes(64)
    assert (
        sha256(data)
        == "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
    )


def test_ff_block() -> None:
    """64 0xFF bytes."""
    vals: list[int] = []
    i: int = 0
    while i < 64:
        vals.append(255)
        i += 1
    assert (
        sha256(bytes(vals))
        == "8667e718294e9e0df1d30600ba3eeb201f764aad2dad72748643e4a285e1d1f7"
    )


# -- Additional known vectors --


def test_hello_world() -> None:
    assert (
        sha256(b"hello world")
        == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


def test_digits() -> None:
    assert (
        sha256(b"0123456789")
        == "84d89877f0d4041efb6bf91a16f0248f2fd573e6af05c19f96bedb9f882f7882"
    )


def test_space() -> None:
    assert (
        sha256(b" ")
        == "36a9e7f1c95b82ffb99743e0c5c4ce95d83c9a430aac59f84ef3cbfab6145068"
    )


def test_newline() -> None:
    assert (
        sha256(b"\n")
        == "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
    )


# -- Determinism --


def test_same_input_same_output() -> None:
    data: bytes = b"determinism check"
    assert sha256(data) == sha256(data)
    assert sha256_bytes(data) == sha256_bytes(data)


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_empty", test_empty),
        ("test_abc", test_abc),
        ("test_two_block_448bit", test_two_block_448bit),
        ("test_single_a", test_single_a),
        ("test_two_block_896bit", test_two_block_896bit),
        ("test_bytes_length", test_bytes_length),
        ("test_bytes_hex_roundtrip", test_bytes_hex_roundtrip),
        ("test_55_bytes", test_55_bytes),
        ("test_56_bytes", test_56_bytes),
        ("test_64_bytes", test_64_bytes),
        ("test_128_bytes", test_128_bytes),
        ("test_null_byte", test_null_byte),
        ("test_ff_byte", test_ff_byte),
        ("test_null_block", test_null_block),
        ("test_ff_block", test_ff_block),
        ("test_hello_world", test_hello_world),
        ("test_digits", test_digits),
        ("test_space", test_space),
        ("test_newline", test_newline),
        ("test_same_input_same_output", test_same_input_same_output),
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
