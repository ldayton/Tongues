"""Bitset — fixed-size bit array with set operations."""

_BITS: int = 32
_WORD_MASK: int = 0xFFFFFFFF


def _popcount_word(x: int) -> int:
    """Count set bits in a 32-bit word."""
    count: int = 0
    while x > 0:
        count += x & 1
        x = x >> 1
    return count


def bitset_new(size: int) -> list[int]:
    """Create a zeroed bitset for 'size' bits. Element 0 is the size."""
    words: int = (size + _BITS - 1) // _BITS
    bs: list[int] = [size]
    i: int = 0
    while i < words:
        bs.append(0)
        i += 1
    return bs


def bitset_set(bs: list[int], i: int) -> None:
    """Set bit i."""
    bs[1 + (i >> 5)] = (bs[1 + (i >> 5)] | (1 << (i & 31))) & _WORD_MASK


def bitset_clear(bs: list[int], i: int) -> None:
    """Clear bit i."""
    bs[1 + (i >> 5)] = bs[1 + (i >> 5)] & (_WORD_MASK ^ (1 << (i & 31)))


def bitset_test(bs: list[int], i: int) -> bool:
    """Return True if bit i is set."""
    return (bs[1 + (i >> 5)] & (1 << (i & 31))) != 0


def bitset_toggle(bs: list[int], i: int) -> None:
    """Toggle bit i."""
    bs[1 + (i >> 5)] = (bs[1 + (i >> 5)] ^ (1 << (i & 31))) & _WORD_MASK


def bitset_popcount(bs: list[int]) -> int:
    """Count the number of set bits."""
    count: int = 0
    i: int = 1
    while i < len(bs):
        count += _popcount_word(bs[i])
        i += 1
    return count


def bitset_union(a: list[int], b: list[int]) -> list[int]:
    """Return a new bitset that is the union (OR) of a and b."""
    sa: int = a[0]
    sb: int = b[0]
    size: int = sa
    if sb > sa:
        size = sb
    result: list[int] = bitset_new(size)
    i: int = 1
    wa: int = 0
    wb: int = 0
    while i < len(result):
        wa = 0
        wb = 0
        if i < len(a):
            wa = a[i]
        if i < len(b):
            wb = b[i]
        result[i] = (wa | wb) & _WORD_MASK
        i += 1
    return result


def bitset_intersection(a: list[int], b: list[int]) -> list[int]:
    """Return a new bitset that is the intersection (AND) of a and b."""
    sa: int = a[0]
    sb: int = b[0]
    size: int = sa
    if sb < sa:
        size = sb
    result: list[int] = bitset_new(size)
    i: int = 1
    wa: int = 0
    wb: int = 0
    while i < len(result):
        wa = 0
        wb = 0
        if i < len(a):
            wa = a[i]
        if i < len(b):
            wb = b[i]
        result[i] = wa & wb
        i += 1
    return result


def bitset_difference(a: list[int], b: list[int]) -> list[int]:
    """Return a new bitset with bits in a but not in b (AND-NOT)."""
    result: list[int] = bitset_new(a[0])
    i: int = 1
    wb: int = 0
    while i < len(result):
        wb = 0
        if i < len(b):
            wb = b[i]
        result[i] = a[i] & (wb ^ _WORD_MASK)
        i += 1
    return result


def bitset_to_list(bs: list[int]) -> list[int]:
    """Return sorted list of set bit indices."""
    out: list[int] = []
    i: int = 0
    while i < bs[0]:
        if bitset_test(bs, i):
            out.append(i)
        i += 1
    return out
