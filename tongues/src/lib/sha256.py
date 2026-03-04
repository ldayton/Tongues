"""SHA-256 (FIPS 180-4) — pure subset-Python implementation."""

_MASK: int = 0xFFFFFFFF

_K: list[int] = [
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
]

_H: list[int] = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]

_HEX: str = "0123456789abcdef"


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _MASK


def _ch(x: int, y: int, z: int) -> int:
    return (x & y) ^ ((x ^ _MASK) & z)


def _maj(x: int, y: int, z: int) -> int:
    return (x & y) ^ (x & z) ^ (y & z)


def _sigma0(x: int) -> int:
    return _rotr(x, 2) ^ _rotr(x, 13) ^ _rotr(x, 22)


def _sigma1(x: int) -> int:
    return _rotr(x, 6) ^ _rotr(x, 11) ^ _rotr(x, 25)


def _lsig0(x: int) -> int:
    return _rotr(x, 7) ^ _rotr(x, 18) ^ (x >> 3)


def _lsig1(x: int) -> int:
    return _rotr(x, 17) ^ _rotr(x, 19) ^ (x >> 10)


def _compress(state: list[int], block: bytes) -> list[int]:
    w: list[int] = []
    for i in range(16):
        j: int = i * 4
        w.append(
            (block[j] << 24) | (block[j + 1] << 16) | (block[j + 2] << 8) | block[j + 3]
        )
    for i in range(16, 64):
        w.append((_lsig1(w[i - 2]) + w[i - 7] + _lsig0(w[i - 15]) + w[i - 16]) & _MASK)
    a: int = state[0]
    b: int = state[1]
    c: int = state[2]
    d: int = state[3]
    e: int = state[4]
    f: int = state[5]
    g: int = state[6]
    h: int = state[7]
    for i in range(64):
        t1: int = (h + _sigma1(e) + _ch(e, f, g) + _K[i] + w[i]) & _MASK
        t2: int = (_sigma0(a) + _maj(a, b, c)) & _MASK
        h = g
        g = f
        f = e
        e = (d + t1) & _MASK
        d = c
        c = b
        b = a
        a = (t1 + t2) & _MASK
    return [
        (state[0] + a) & _MASK,
        (state[1] + b) & _MASK,
        (state[2] + c) & _MASK,
        (state[3] + d) & _MASK,
        (state[4] + e) & _MASK,
        (state[5] + f) & _MASK,
        (state[6] + g) & _MASK,
        (state[7] + h) & _MASK,
    ]


def _pad(data: bytes) -> bytes:
    n: int = len(data)
    bit_len: int = n * 8
    buf: list[int] = []
    for i in range(n):
        buf.append(data[i])
    buf.append(0x80)
    while len(buf) % 64 != 56:
        buf.append(0)
    for i in range(56, -1, -8):
        buf.append((bit_len >> i) & 0xFF)
    return bytes(buf)


def sha256_bytes(data: bytes) -> bytes:
    """Return the SHA-256 digest as 32 raw bytes."""
    padded: bytes = _pad(data)
    state: list[int] = [_H[0], _H[1], _H[2], _H[3], _H[4], _H[5], _H[6], _H[7]]
    for i in range(0, len(padded), 64):
        state = _compress(state, padded[i : i + 64])
    out: list[int] = []
    for val in state:
        out.append((val >> 24) & 0xFF)
        out.append((val >> 16) & 0xFF)
        out.append((val >> 8) & 0xFF)
        out.append(val & 0xFF)
    return bytes(out)


def sha256(data: bytes) -> str:
    """Return the SHA-256 hex digest of data."""
    raw: bytes = sha256_bytes(data)
    parts: list[str] = []
    for i in range(len(raw)):
        b: int = raw[i]
        parts.append(_HEX[b >> 4])
        parts.append(_HEX[b & 0x0F])
    return "".join(parts)
