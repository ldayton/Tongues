"""UTF-8 codec — encode/decode between Unicode codepoints and bytes."""

from dataclasses import dataclass


@dataclass
class Utf8Error(Exception):
    """Raised on invalid UTF-8 input."""

    position: int


_REPLACEMENT: int = 0xFFFD


def encode_codepoint(cp: int) -> bytes:
    """Encode a single Unicode codepoint to UTF-8 bytes."""
    if cp < 0:
        return encode_codepoint(_REPLACEMENT)
    if cp < 0x80:
        return bytes([cp])
    if cp < 0x800:
        return bytes(
            [
                0xC0 | (cp >> 6),
                0x80 | (cp & 0x3F),
            ]
        )
    if cp < 0x10000:
        if cp >= 0xD800 and cp <= 0xDFFF:
            return encode_codepoint(_REPLACEMENT)
        return bytes(
            [
                0xE0 | (cp >> 12),
                0x80 | ((cp >> 6) & 0x3F),
                0x80 | (cp & 0x3F),
            ]
        )
    if cp <= 0x10FFFF:
        return bytes(
            [
                0xF0 | (cp >> 18),
                0x80 | ((cp >> 12) & 0x3F),
                0x80 | ((cp >> 6) & 0x3F),
                0x80 | (cp & 0x3F),
            ]
        )
    return encode_codepoint(_REPLACEMENT)


def encode(codepoints: list[int]) -> bytes:
    """Encode a list of Unicode codepoints to UTF-8 bytes."""
    out: list[int] = []
    i: int = 0
    while i < len(codepoints):
        cp: int = codepoints[i]
        chunk: bytes = encode_codepoint(cp)
        j: int = 0
        while j < len(chunk):
            out.append(chunk[j])
            j += 1
        i += 1
    return bytes(out)


def _is_cont(b: int) -> bool:
    return (b & 0xC0) == 0x80


def decode_codepoint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode one codepoint starting at pos. Returns (codepoint, next_pos).

    Raises Utf8Error on invalid sequences.
    """
    n: int = len(data)
    if pos >= n:
        raise Utf8Error(pos)
    b0: int = data[pos]
    if b0 < 0x80:
        return (b0, pos + 1)
    if b0 < 0xC2:
        raise Utf8Error(pos)
    if b0 < 0xE0:
        if pos + 1 >= n:
            raise Utf8Error(pos)
        b1: int = data[pos + 1]
        if not _is_cont(b1):
            raise Utf8Error(pos + 1)
        cp: int = ((b0 & 0x1F) << 6) | (b1 & 0x3F)
        return (cp, pos + 2)
    if b0 < 0xF0:
        if pos + 2 >= n:
            raise Utf8Error(pos)
        b1 = data[pos + 1]
        b2: int = data[pos + 2]
        if not _is_cont(b1):
            raise Utf8Error(pos + 1)
        if not _is_cont(b2):
            raise Utf8Error(pos + 2)
        cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
        if cp < 0x800:
            raise Utf8Error(pos)
        if cp >= 0xD800 and cp <= 0xDFFF:
            raise Utf8Error(pos)
        return (cp, pos + 3)
    if b0 < 0xF5:
        if pos + 3 >= n:
            raise Utf8Error(pos)
        b1 = data[pos + 1]
        b2 = data[pos + 2]
        b3: int = data[pos + 3]
        if not _is_cont(b1):
            raise Utf8Error(pos + 1)
        if not _is_cont(b2):
            raise Utf8Error(pos + 2)
        if not _is_cont(b3):
            raise Utf8Error(pos + 3)
        cp = (
            ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F)
        )
        if cp < 0x10000:
            raise Utf8Error(pos)
        if cp > 0x10FFFF:
            raise Utf8Error(pos)
        return (cp, pos + 4)
    raise Utf8Error(pos)


def decode(data: bytes) -> list[int]:
    """Decode UTF-8 bytes to a list of Unicode codepoints.

    Raises Utf8Error on the first invalid byte.
    """
    out: list[int] = []
    pos: int = 0
    while pos < len(data):
        result: tuple[int, int] = decode_codepoint(data, pos)
        out.append(result[0])
        pos = result[1]
    return out


def is_valid(data: bytes) -> bool:
    """Return True if data is valid UTF-8."""
    pos: int = 0
    while pos < len(data):
        try:
            result: tuple[int, int] = decode_codepoint(data, pos)
            pos = result[1]
        except Utf8Error:
            return False
    return True


def codepoint_len(data: bytes) -> int:
    """Count the number of Unicode codepoints in valid UTF-8 data."""
    count: int = 0
    pos: int = 0
    while pos < len(data):
        result: tuple[int, int] = decode_codepoint(data, pos)
        pos = result[1]
        count += 1
    return count
