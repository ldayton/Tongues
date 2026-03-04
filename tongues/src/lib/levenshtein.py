"""Levenshtein edit distance (Wagner-Fischer, two-row)."""


def levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between two strings."""
    len_a: int = len(a)
    len_b: int = len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a
    short: str = a
    long: str = b
    short_len: int = len_a
    long_len: int = len_b
    if len_a > len_b:
        short = b
        long = a
        short_len = len_b
        long_len = len_a
    prev: list[int] = list(range(short_len + 1))
    curr: list[int] = [0] * (short_len + 1)
    i: int = 0
    cost: int = 0
    ins_cost: int = 0
    del_cost: int = 0
    sub_cost: int = 0
    best: int = 0
    while i < long_len:
        curr[0] = i + 1
        j: int = 0
        while j < short_len:
            if long[i] == short[j]:
                cost = 0
            else:
                cost = 1
            ins_cost = curr[j] + 1
            del_cost = prev[j + 1] + 1
            sub_cost = prev[j] + cost
            best = ins_cost
            if del_cost < best:
                best = del_cost
            if sub_cost < best:
                best = sub_cost
            curr[j + 1] = best
            j += 1
        prev, curr = curr, prev
        i += 1
    return prev[short_len]
