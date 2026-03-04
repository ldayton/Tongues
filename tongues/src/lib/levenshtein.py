"""Levenshtein edit distance (Wagner-Fischer, two-row)."""


def levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between two strings."""
    la: int = len(a)
    lb: int = len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    short: str = a
    long: str = b
    sn: int = la
    ln: int = lb
    if la > lb:
        short = b
        long = a
        sn = lb
        ln = la
    prev: list[int] = []
    curr: list[int] = []
    j: int = 0
    while j <= sn:
        prev.append(j)
        curr.append(0)
        j += 1
    i: int = 0
    cost: int = 0
    ins: int = 0
    dele: int = 0
    sub: int = 0
    best: int = 0
    while i < ln:
        curr[0] = i + 1
        j = 0
        while j < sn:
            if long[i] == short[j]:
                cost = 0
            else:
                cost = 1
            ins = curr[j] + 1
            dele = prev[j + 1] + 1
            sub = prev[j] + cost
            best = ins
            if dele < best:
                best = dele
            if sub < best:
                best = sub
            curr[j + 1] = best
            j += 1
        j = 0
        while j <= sn:
            prev[j] = curr[j]
            j += 1
        i += 1
    return prev[sn]
