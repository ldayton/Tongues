"""Glob pattern matching — Unix-style wildcards."""

from dataclasses import dataclass


@dataclass
class GlobError(Exception):
    """Raised on malformed glob patterns."""

    position: int


def _scan_chunk(pattern: str, pos: int) -> tuple[bool, int, int]:
    """Split at next unbracketed *. Returns (star, chunk_start, chunk_end)."""
    n: int = len(pattern)
    star: bool = False
    while pos < n and pattern[pos] == "*":
        star = True
        pos += 1
    start: int = pos
    in_bracket: bool = False
    ch: str = ""
    while pos < n:
        ch = pattern[pos]
        if ch == "\\":
            if pos + 1 < n:
                pos += 1
        elif ch == "[":
            in_bracket = True
        elif ch == "]":
            in_bracket = False
        elif ch == "*" and not in_bracket:
            break
        pos += 1
    return (star, start, pos)


def _match_class(pattern: str, pos: int, ch: str) -> tuple[bool, int]:
    """Match ch against bracket expression after '['. Returns (matched, end_pos)."""
    n: int = len(pattern)
    negated: bool = False
    if pos < n and (pattern[pos] == "^" or pattern[pos] == "!"):
        negated = True
        pos += 1
    matched: bool = False
    count: int = 0
    lo: str = ""
    hi: str = ""
    while pos < n:
        if pattern[pos] == "]" and count > 0:
            if negated:
                return (not matched, pos + 1)
            return (matched, pos + 1)
        lo = pattern[pos]
        if lo == "\\":
            pos += 1
            if pos >= n:
                raise GlobError(pos - 1)
            lo = pattern[pos]
        pos += 1
        hi = lo
        if pos + 1 < n and pattern[pos] == "-" and pattern[pos + 1] != "]":
            pos += 1
            hi = pattern[pos]
            if hi == "\\":
                pos += 1
                if pos >= n:
                    raise GlobError(pos - 1)
                hi = pattern[pos]
            pos += 1
        if lo <= ch and ch <= hi:
            matched = True
        count += 1
    raise GlobError(pos)


def _match_chunk(
    pattern: str, ps: int, pe: int, text: str, tp: int
) -> tuple[bool, int]:
    """Match pattern[ps:pe] against text at tp. Returns (ok, next_tp)."""
    tn: int = len(text)
    ch: str = ""
    result: tuple[bool, int] = (False, 0)
    while ps < pe:
        if tp >= tn:
            return (False, tp)
        ch = pattern[ps]
        if ch == "?":
            ps += 1
            tp += 1
        elif ch == "[":
            ps += 1
            result = _match_class(pattern, ps, text[tp])
            if not result[0]:
                return (False, tp)
            ps = result[1]
            tp += 1
        elif ch == "\\":
            ps += 1
            if ps >= pe:
                raise GlobError(ps - 1)
            if text[tp] != pattern[ps]:
                return (False, tp)
            ps += 1
            tp += 1
        else:
            if text[tp] != ch:
                return (False, tp)
            ps += 1
            tp += 1
    return (True, tp)


def glob_match(pattern: str, text: str) -> bool:
    """Match text against a glob pattern with *, ?, [abc], [a-z], [^a], and \\\\ escaping."""
    pp: int = 0
    tp: int = 0
    pn: int = len(pattern)
    tn: int = len(text)
    scan: tuple[bool, int, int] = (False, 0, 0)
    star: bool = False
    cs: int = 0
    ce: int = 0
    result: tuple[bool, int] = (False, 0)
    found: bool = False
    i: int = 0
    while pp < pn:
        scan = _scan_chunk(pattern, pp)
        star = scan[0]
        cs = scan[1]
        ce = scan[2]
        if star and cs == ce:
            return True
        result = _match_chunk(pattern, cs, ce, text, tp)
        if result[0] and (result[1] == tn or ce < pn):
            tp = result[1]
            pp = ce
            continue
        if star:
            found = False
            i = tp + 1
            while i <= tn:
                result = _match_chunk(pattern, cs, ce, text, i)
                if result[0] and (result[1] == tn or ce < pn):
                    tp = result[1]
                    pp = ce
                    found = True
                    break
                i += 1
            if found:
                continue
        return False
    return tp == tn
