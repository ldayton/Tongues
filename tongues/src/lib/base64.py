"""Base64 encoding and decoding (RFC 4648)."""

from dataclasses import dataclass


@dataclass
class Base64Error(Exception):
    """Raised when base64 decoding encounters invalid input."""

    position: int


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
    """Decode a base64 string to bytes. Raises Base64Error on invalid input."""
    return _b64decode_impl(s, False)


def b64decode_strict(s: str) -> bytes:
    """Decode with strict validation of padding bits (RFC 4648 §3.5)."""
    return _b64decode_impl(s, True)


def _b64decode_impl(s: str, strict: bool) -> bytes:
    """Decode a base64 string to bytes."""
    n: int = len(s)
    if n == 0:
        return b""
    # Validate all characters first, then check length
    pad: int = 0
    i: int = 0
    while i < n:
        ch: str = s[i]
        if ch == "=":
            # Padding only valid at positions n-1 or n-2
            if i < n - 2:
                raise Base64Error(i)
            if i == n - 2:
                if n % 4 != 0:
                    raise Base64Error(i)
                if s[n - 1] != "=":
                    raise Base64Error(i)
                pad = 2
            elif i == n - 1:
                if n % 4 != 0:
                    raise Base64Error(i)
                if pad == 0:
                    pad = 1
        elif _DECODE_TABLE[ord(ch)] == -1:
            raise Base64Error(i)
        i += 1
    if n % 4 != 0:
        raise Base64Error(n)
    # Strict mode: check that unused padding bits are zero
    if strict and pad > 0:
        if pad == 2:
            # With ==, char at n-3 has only top 2 bits used, lower 4 must be 0
            val: int = _DECODE_TABLE[ord(s[n - 3])]
            if val & 0x0F != 0:
                raise Base64Error(n - 3)
        elif pad == 1:
            # With =, char at n-2 has only top 4 bits used, lower 2 must be 0
            val = _DECODE_TABLE[ord(s[n - 2])]
            if val & 0x03 != 0:
                raise Base64Error(n - 2)
    # Decode
    out: list[int] = []
    i = 0
    while i < n:
        a: int = _DECODE_TABLE[ord(s[i])]
        b: int = _DECODE_TABLE[ord(s[i + 1])]
        if s[i + 2] != "=":
            c: int = _DECODE_TABLE[ord(s[i + 2])]
        else:
            c = 0
        if s[i + 3] != "=":
            d: int = _DECODE_TABLE[ord(s[i + 3])]
        else:
            d = 0
        val = (a << 18) | (b << 12) | (c << 6) | d
        out.append((val >> 16) & 0xFF)
        out.append((val >> 8) & 0xFF)
        out.append(val & 0xFF)
        i += 4
    trim: int = len(out) - pad
    return bytes(out[:trim])
