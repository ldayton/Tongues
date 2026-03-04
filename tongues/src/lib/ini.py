"""INI file parser and writer."""

from dataclasses import dataclass


@dataclass
class IniError(Exception):
    """Raised on malformed INI input."""

    line: int


def _trim(s: str) -> str:
    """Strip leading and trailing spaces/tabs."""
    start: int = 0
    end: int = len(s)
    while start < end and (s[start] == " " or s[start] == "\t"):
        start += 1
    while end > start and (s[end - 1] == " " or s[end - 1] == "\t"):
        end -= 1
    return s[start:end]


def _skip_to_eol(text: str, pos: int) -> tuple[int, int]:
    """Advance past current line. Returns (next_pos, lines_consumed)."""
    n: int = len(text)
    while pos < n and text[pos] != "\n" and text[pos] != "\r":
        pos += 1
    if pos < n and text[pos] == "\r":
        pos += 1
        if pos < n and text[pos] == "\n":
            pos += 1
        return (pos, 1)
    if pos < n and text[pos] == "\n":
        pos += 1
        return (pos, 1)
    return (pos, 0)


def _find_section(sections: list[list[str]], name: str) -> int:
    """Return index of section with given name, or -1."""
    i: int = 0
    while i < len(sections):
        if sections[i][0] == name:
            return i
        i += 1
    return -1


def _set_key(section: list[str], key: str, val: str) -> None:
    """Set key in section. Updates existing key or appends."""
    i: int = 1
    while i < len(section):
        if section[i] == key:
            section[i + 1] = val
            return
        i += 2
    section.append(key)
    section.append(val)


def ini_parse(text: str) -> list[list[str]]:
    """Parse INI text into sections.

    Each section is [name, key1, val1, key2, val2, ...].
    Keys before any [section] header go into section named "".
    """
    n: int = len(text)
    sections: list[list[str]] = [[""]]
    current: list[str] = sections[0]
    pos: int = 0
    line_num: int = 1
    eol: tuple[int, int] = (0, 0)
    while pos < n:
        while pos < n and (text[pos] == " " or text[pos] == "\t"):
            pos += 1
        if pos >= n:
            break
        if text[pos] == "\n" or text[pos] == "\r":
            eol = _skip_to_eol(text, pos)
            pos = eol[0]
            line_num += eol[1]
            continue
        if text[pos] == "#" or text[pos] == ";":
            eol = _skip_to_eol(text, pos)
            pos = eol[0]
            line_num += eol[1]
            continue
        if text[pos] == "[":
            pos += 1
            start: int = pos
            while (
                pos < n and text[pos] != "]" and text[pos] != "\n" and text[pos] != "\r"
            ):
                pos += 1
            if pos >= n or text[pos] != "]":
                raise IniError(line_num)
            name: str = _trim(text[start:pos])
            pos += 1
            eol = _skip_to_eol(text, pos)
            pos = eol[0]
            line_num += eol[1]
            idx: int = _find_section(sections, name)
            if idx >= 0:
                current = sections[idx]
            else:
                current = [name]
                sections.append(current)
            continue
        start = pos
        while pos < n and text[pos] != "=" and text[pos] != "\n" and text[pos] != "\r":
            pos += 1
        if pos >= n or text[pos] != "=":
            raise IniError(line_num)
        key: str = _trim(text[start:pos])
        pos += 1
        start = pos
        while pos < n and text[pos] != "\n" and text[pos] != "\r":
            pos += 1
        val: str = _trim(text[start:pos])
        eol = _skip_to_eol(text, pos)
        pos = eol[0]
        line_num += eol[1]
        _set_key(current, key, val)
    if len(sections[0]) == 1:
        if len(sections) == 1:
            return []
        result: list[list[str]] = []
        i: int = 1
        while i < len(sections):
            result.append(sections[i])
            i += 1
        return result
    return sections


def ini_get(sections: list[list[str]], section: str, key: str) -> str:
    """Find value by section and key. Returns "" if not found."""
    i: int = 0
    j: int = 0
    while i < len(sections):
        if sections[i][0] == section:
            j = 1
            while j < len(sections[i]):
                if sections[i][j] == key:
                    return sections[i][j + 1]
                j += 2
        i += 1
    return ""


def ini_write(sections: list[list[str]]) -> str:
    """Serialize sections to INI format."""
    parts: list[str] = []
    i: int = 0
    j: int = 0
    while i < len(sections):
        if i > 0:
            parts.append("\n")
        if sections[i][0] != "":
            parts.append("[")
            parts.append(sections[i][0])
            parts.append("]\n")
        j = 1
        while j < len(sections[i]):
            parts.append(sections[i][j])
            parts.append(" = ")
            parts.append(sections[i][j + 1])
            parts.append("\n")
            j += 2
        i += 1
    return "".join(parts)
