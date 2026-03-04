"""Percent-encoding (RFC 3986 unreserved set)."""

_HEX: str = "0123456789ABCDEF"


def _is_unreserved(b: int) -> bool:
    """Return True if byte is an RFC 3986 unreserved character."""
    if b >= 65 and b <= 90:
        return True
    if b >= 97 and b <= 122:
        return True
    if b >= 48 and b <= 57:
        return True
    if b == 45 or b == 95 or b == 46 or b == 126:
        return True
    return False


def _hex_val(c: int) -> int:
    """Return 0-15 for an ASCII hex digit ordinal, or -1 if invalid."""
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 65 and c <= 70:
        return c - 55
    if c >= 97 and c <= 102:
        return c - 87
    return -1


def _encode_cp(cp: int) -> list[int]:
    """Encode a single codepoint to UTF-8 bytes."""
    out: list[int] = []
    if cp < 0x80:
        out.append(cp)
    elif cp < 0x800:
        out.append(0xC0 | (cp >> 6))
        out.append(0x80 | (cp & 0x3F))
    elif cp < 0x10000:
        out.append(0xE0 | (cp >> 12))
        out.append(0x80 | ((cp >> 6) & 0x3F))
        out.append(0x80 | (cp & 0x3F))
    else:
        out.append(0xF0 | (cp >> 18))
        out.append(0x80 | ((cp >> 12) & 0x3F))
        out.append(0x80 | ((cp >> 6) & 0x3F))
        out.append(0x80 | (cp & 0x3F))
    return out


def _decode_cp(data: bytes, pos: int) -> tuple[int, int]:
    """Decode one UTF-8 codepoint starting at pos. Returns (codepoint, next_pos)."""
    b0: int = data[pos]
    if b0 < 0x80:
        return (b0, pos + 1)
    if b0 < 0xE0:
        return ((b0 & 0x1F) << 6 | (data[pos + 1] & 0x3F), pos + 2)
    if b0 < 0xF0:
        return (
            (b0 & 0x0F) << 12 | (data[pos + 1] & 0x3F) << 6 | (data[pos + 2] & 0x3F),
            pos + 3,
        )
    return (
        (b0 & 0x07) << 18
        | (data[pos + 1] & 0x3F) << 12
        | (data[pos + 2] & 0x3F) << 6
        | (data[pos + 3] & 0x3F),
        pos + 4,
    )


def percent_encode(s: str) -> str:
    """Percent-encode a string using RFC 3986 unreserved set."""
    out: list[str] = []
    for ch in s:
        utf8: list[int] = _encode_cp(ord(ch))
        for b in utf8:
            if _is_unreserved(b):
                out.append(chr(b))
            else:
                out.append("%")
                out.append(_HEX[b >> 4])
                out.append(_HEX[b & 15])
    return "".join(out)


def percent_decode(s: str) -> str:
    """Decode a percent-encoded string."""
    raw: list[int] = []
    i: int = 0
    n: int = len(s)
    hi: int = 0
    lo: int = 0
    while i < n:
        if s[i] == "%" and i + 2 < n:
            hi = _hex_val(ord(s[i + 1]))
            lo = _hex_val(ord(s[i + 2]))
            if hi >= 0 and lo >= 0:
                raw.append(hi * 16 + lo)
                i += 3
                continue
        raw.append(ord(s[i]))
        i += 1
    data: bytes = bytes(raw)
    out: list[str] = []
    j: int = 0
    result: tuple[int, int] = (0, 0)
    while j < len(data):
        result = _decode_cp(data, j)
        out.append(chr(result[0]))
        j = result[1]
    return "".join(out)
