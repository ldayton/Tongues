"""Base64 encoding and decoding (RFC 4648)."""

_ENCODE_TABLE: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def _make_decode_table() -> list[int]:
    table: list[int] = []
    i: int = 0
    while i < 256:
        table.append(-1)
        i += 1
    i = 0
    while i < 64:
        table[ord(_ENCODE_TABLE[i])] = i
        i += 1
    return table


_DECODE_TABLE: list[int] = _make_decode_table()


def b64encode(data: bytes) -> str:
    """Encode bytes to a base64 string."""
    n: int = len(data)
    if n == 0:
        return ""
    parts: list[str] = []
    i: int = 0
    full: int = n - (n % 3)
    while i < full:
        val: int = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
        parts.append(_ENCODE_TABLE[(val >> 18) & 0x3F])
        parts.append(_ENCODE_TABLE[(val >> 12) & 0x3F])
        parts.append(_ENCODE_TABLE[(val >> 6) & 0x3F])
        parts.append(_ENCODE_TABLE[val & 0x3F])
        i += 3
    remain: int = n - full
    if remain == 1:
        val = data[i] << 16
        parts.append(_ENCODE_TABLE[(val >> 18) & 0x3F])
        parts.append(_ENCODE_TABLE[(val >> 12) & 0x3F])
        parts.append("=")
        parts.append("=")
    elif remain == 2:
        val = (data[i] << 16) | (data[i + 1] << 8)
        parts.append(_ENCODE_TABLE[(val >> 18) & 0x3F])
        parts.append(_ENCODE_TABLE[(val >> 12) & 0x3F])
        parts.append(_ENCODE_TABLE[(val >> 6) & 0x3F])
        parts.append("=")
    return "".join(parts)


def b64decode(s: str) -> bytes:
    """Decode a base64 string to bytes."""
    n: int = len(s)
    if n == 0:
        return b""
    pad: int = 0
    if s[n - 1] == "=":
        pad = 1
        if s[n - 2] == "=":
            pad = 2
    out: list[int] = []
    i: int = 0
    while i < n:
        a: int = _DECODE_TABLE[ord(s[i])]
        b: int = _DECODE_TABLE[ord(s[i + 1])]
        if i + 2 < n and s[i + 2] != "=":
            c: int = _DECODE_TABLE[ord(s[i + 2])]
        else:
            c = 0
        if i + 3 < n and s[i + 3] != "=":
            d: int = _DECODE_TABLE[ord(s[i + 3])]
        else:
            d = 0
        val: int = (a << 18) | (b << 12) | (c << 6) | d
        out.append((val >> 16) & 0xFF)
        out.append((val >> 8) & 0xFF)
        out.append(val & 0xFF)
        i += 4
    trim: int = len(out) - pad
    return bytes(out[:trim])
