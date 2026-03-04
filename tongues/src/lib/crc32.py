"""CRC-32 checksum (IEEE 802.3)."""

_POLY: int = 0xEDB88320
_MASK: int = 0xFFFFFFFF
_HEX: str = "0123456789abcdef"


def _make_table() -> list[int]:
    """Build the 256-entry CRC lookup table from the reflected polynomial."""
    table: list[int] = []
    for i in range(256):
        crc: int = i
        for j in range(8):
            if (crc & 1) == 1:
                crc = (crc >> 1) ^ _POLY
            else:
                crc = crc >> 1
        table.append(crc)
    return table


_TABLE: list[int] = _make_table()


def crc32_update(crc: int, data: bytes) -> int:
    """Update a running CRC with new data. Initial crc should be 0."""
    c: int = (crc ^ _MASK) & _MASK
    for i in range(len(data)):
        c = _TABLE[(c ^ data[i]) & 0xFF] ^ (c >> 8)
    return (c ^ _MASK) & _MASK


def crc32(data: bytes) -> int:
    """Return CRC-32 checksum as unsigned 32-bit integer."""
    return crc32_update(0, data)


def crc32_hex(data: bytes) -> str:
    """Return CRC-32 as 8-char lowercase hex string."""
    val: int = crc32(data)
    out: list[str] = []
    for i in range(7, -1, -1):
        out.append(_HEX[(val >> (i * 4)) & 0xF])
    return "".join(out)
