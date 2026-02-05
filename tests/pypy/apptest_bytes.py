"""Bytes object tests."""

import sys


def test_bytes_equality() -> None:
    """Bytes equality comparisons."""
    assert b"hello" == b"hello"
    assert b"" == b""
    assert b"\x00" == b"\x00"
    assert not (b"hello" == b"world")
    assert b"hello" != b"world"
    assert b"Hello" != b"hello"  # case sensitive


def test_bytes_ordering() -> None:
    """Bytes ordering is lexicographic."""
    assert b"a" < b"b"
    assert b"abc" < b"abd"
    assert b"abc" < b"abcd"
    assert b"" < b"a"
    assert b"A" < b"a"  # uppercase < lowercase in ASCII
    assert b"abc" <= b"abc"
    assert b"abc" >= b"abc"


def test_bytes_length() -> None:
    """len() returns byte count."""
    assert len(b"") == 0
    assert len(b"a") == 1
    assert len(b"hello") == 5
    assert len(b"\x00") == 1
    assert len(b"\x00\x01\x02") == 3


def test_bytes_indexing() -> None:
    """Indexing returns int (byte value)."""
    b: bytes = b"hello"
    assert b[0] == 104  # ord('h')
    assert b[1] == 101  # ord('e')
    assert b[4] == 111  # ord('o')
    # Negative indexing
    assert b[-1] == 111  # ord('o')
    assert b[-5] == 104  # ord('h')


def test_bytes_indexing_special() -> None:
    """Indexing special byte values."""
    b: bytes = b"\x00\xff\x7f\x80"
    assert b[0] == 0    # null
    assert b[1] == 255  # max byte
    assert b[2] == 127  # max signed byte
    assert b[3] == 128  # min negative as unsigned


def test_bytes_concatenation() -> None:
    """Bytes concatenation with +."""
    assert b"hello" + b" " + b"world" == b"hello world"
    assert b"" + b"a" == b"a"
    assert b"a" + b"" == b"a"
    assert b"" + b"" == b""


def test_bytes_repetition() -> None:
    """Bytes repetition with *."""
    assert b"a" * 3 == b"aaa"
    assert b"ab" * 2 == b"abab"
    assert b"x" * 0 == b""
    assert b"x" * 1 == b"x"
    assert 3 * b"a" == b"aaa"


def test_bytes_contains() -> None:
    """Membership testing with in."""
    assert b"ell" in b"hello"
    assert b"h" in b"hello"
    assert b"o" in b"hello"
    assert b"" in b"hello"
    assert not (b"x" in b"hello")
    assert not (b"Hello" in b"hello")


def test_bytes_slicing() -> None:
    """Bytes slicing returns bytes."""
    b: bytes = b"hello"
    assert b[0:2] == b"he"
    assert b[1:4] == b"ell"
    assert b[:3] == b"hel"
    assert b[2:] == b"llo"
    assert b[:] == b"hello"
    assert b[::2] == b"hlo"
    assert b[::-1] == b"olleh"


def test_bytes_from_int_list() -> None:
    """Create bytes from list of ints."""
    assert bytes([104, 105]) == b"hi"
    assert bytes([0, 255]) == b"\x00\xff"
    assert bytes([]) == b""


def test_bytes_from_int() -> None:
    """bytes(n) creates n null bytes."""
    assert bytes(0) == b""
    assert bytes(1) == b"\x00"
    assert bytes(3) == b"\x00\x00\x00"
    assert len(bytes(5)) == 5
    # All bytes are zero
    b: bytes = bytes(4)
    assert b[0] == 0
    assert b[1] == 0
    assert b[2] == 0
    assert b[3] == 0


def test_bytes_null_handling() -> None:
    """Null bytes are valid data, not terminators."""
    b: bytes = b"a\x00b\x00c"
    assert len(b) == 5
    assert b[0] == 97   # 'a'
    assert b[1] == 0    # null
    assert b[2] == 98   # 'b'
    assert b[3] == 0    # null
    assert b[4] == 99   # 'c'
    # Slicing preserves nulls
    assert b[1:4] == b"\x00b\x00"
    # Contains works with nulls
    assert b"\x00" in b
    assert b"\x00b" in b


def test_bytes_index_vs_slice() -> None:
    """Indexing returns int, slicing returns bytes."""
    b: bytes = b"abc"
    # Index returns int
    x: int = b[0]
    assert x == 97
    # Slice returns bytes (even single element)
    s: bytes = b[0:1]
    assert s == b"a"
    assert len(s) == 1
    # Empty slice returns empty bytes
    assert b[1:1] == b""


def test_bytes_bool() -> None:
    """Bytes truthiness - empty is falsy."""
    assert bool(b"hello") == True
    assert bool(b"") == False
    assert not b""
    assert b"x"


def test_bytes_iteration() -> None:
    """Iterating yields ints."""
    result: list[int] = []
    for byte in b"hi":
        result.append(byte)
    assert result[0] == 104
    assert result[1] == 105
    assert len(result) == 2


def test_bytes_count() -> None:
    """count() method."""
    assert b"hello".count(b"l") == 2
    assert b"hello".count(b"ll") == 1
    assert b"hello".count(b"x") == 0
    assert b"aaa".count(b"a") == 3
    assert b"aaa".count(b"aa") == 1


def test_bytes_find() -> None:
    """find() returns index or -1."""
    assert b"hello".find(b"l") == 2
    assert b"hello".find(b"ll") == 2
    assert b"hello".find(b"o") == 4
    assert b"hello".find(b"x") == -1
    assert b"hello".find(b"") == 0


def test_bytes_startswith_endswith() -> None:
    """startswith() and endswith() methods."""
    assert b"hello".startswith(b"he")
    assert b"hello".startswith(b"")
    assert b"hello".startswith(b"hello")
    assert not b"hello".startswith(b"lo")
    assert b"hello".endswith(b"lo")
    assert b"hello".endswith(b"")
    assert b"hello".endswith(b"hello")
    assert not b"hello".endswith(b"he")


def test_bytes_upper_lower() -> None:
    """upper() and lower() methods."""
    assert b"Hello".upper() == b"HELLO"
    assert b"Hello".lower() == b"hello"
    assert b"HELLO".lower() == b"hello"
    assert b"hello".upper() == b"HELLO"
    assert b"123".upper() == b"123"
    assert b"123".lower() == b"123"


def test_bytes_strip() -> None:
    """strip(), lstrip(), rstrip() methods."""
    assert b"  hello  ".strip() == b"hello"
    assert b"  hello  ".lstrip() == b"hello  "
    assert b"  hello  ".rstrip() == b"  hello"
    assert b"hello".strip() == b"hello"
    assert b"xxhelloxx".strip(b"x") == b"hello"


def test_bytes_split() -> None:
    """split() method."""
    assert b"a,b,c".split(b",") == [b"a", b"b", b"c"]
    assert b"hello".split(b"l") == [b"he", b"", b"o"]
    assert b"hello".split(b"x") == [b"hello"]


def test_bytes_join() -> None:
    """join() method."""
    assert b",".join([b"a", b"b", b"c"]) == b"a,b,c"
    assert b"".join([b"a", b"b"]) == b"ab"
    assert b"-".join([b"x"]) == b"x"
    assert b"-".join([]) == b""


def test_bytes_replace() -> None:
    """replace() method."""
    assert b"hello".replace(b"l", b"L") == b"heLLo"
    assert b"hello".replace(b"ll", b"LL") == b"heLLo"
    assert b"hello".replace(b"x", b"y") == b"hello"
    assert b"aaa".replace(b"a", b"b") == b"bbb"


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_bytes_equality", test_bytes_equality),
        ("test_bytes_ordering", test_bytes_ordering),
        ("test_bytes_length", test_bytes_length),
        ("test_bytes_indexing", test_bytes_indexing),
        ("test_bytes_indexing_special", test_bytes_indexing_special),
        ("test_bytes_concatenation", test_bytes_concatenation),
        ("test_bytes_repetition", test_bytes_repetition),
        ("test_bytes_contains", test_bytes_contains),
        ("test_bytes_slicing", test_bytes_slicing),
        ("test_bytes_from_int_list", test_bytes_from_int_list),
        ("test_bytes_from_int", test_bytes_from_int),
        ("test_bytes_null_handling", test_bytes_null_handling),
        ("test_bytes_index_vs_slice", test_bytes_index_vs_slice),
        ("test_bytes_bool", test_bytes_bool),
        ("test_bytes_iteration", test_bytes_iteration),
        ("test_bytes_count", test_bytes_count),
        ("test_bytes_find", test_bytes_find),
        ("test_bytes_startswith_endswith", test_bytes_startswith_endswith),
        ("test_bytes_upper_lower", test_bytes_upper_lower),
        ("test_bytes_strip", test_bytes_strip),
        ("test_bytes_split", test_bytes_split),
        ("test_bytes_join", test_bytes_join),
        ("test_bytes_replace", test_bytes_replace),
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
