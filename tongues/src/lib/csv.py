"""CSV parser and writer (RFC 4180)."""

from dataclasses import dataclass


@dataclass
class CsvError(Exception):
    """Raised on malformed CSV input."""

    line: int
    column: int


def parse(text: str) -> list[list[str]]:
    """Parse CSV text into a list of records (list of fields).

    Follows RFC 4180: comma-delimited, double-quote escaping via "",
    quoted fields may contain newlines and commas. CRLF normalized to LF.
    Blank lines are skipped.
    """
    return _parse(text, ",")


def parse_tsv(text: str) -> list[list[str]]:
    """Parse tab-separated text into a list of records."""
    return _parse(text, "\t")


def _parse(text: str, sep: str) -> list[list[str]]:
    n: int = len(text)
    records: list[list[str]] = []
    pos: int = 0
    line: int = 1
    while pos < n:
        if text[pos] == "\n":
            line += 1
            pos += 1
            continue
        if text[pos] == "\r":
            pos += 1
            if pos < n and text[pos] == "\n":
                pos += 1
            line += 1
            continue
        record, pos, line = _parse_record(text, pos, line, sep)
        records.append(record)
    return records


def _parse_record(
    text: str, pos: int, line: int, sep: str
) -> tuple[list[str], int, int]:
    """Parse one record. Returns (fields, next_pos, next_line)."""
    n: int = len(text)
    fields: list[str] = []
    while True:
        field, pos, line = _parse_field(text, pos, line, sep)
        fields.append(field)
        if pos >= n:
            break
        if text[pos] == sep:
            pos += 1
            if pos >= n:
                fields.append("")
                break
            continue
        if text[pos] == "\r":
            pos += 1
            if pos < n and text[pos] == "\n":
                pos += 1
            line += 1
            break
        if text[pos] == "\n":
            pos += 1
            line += 1
            break
        raise CsvError(line, pos + 1)
    return (fields, pos, line)


def _parse_field(text: str, pos: int, line: int, sep: str) -> tuple[str, int, int]:
    """Parse one field. Returns (value, next_pos, next_line)."""
    n: int = len(text)
    if pos >= n:
        return ("", pos, line)
    if text[pos] == '"':
        return _parse_quoted(text, pos, line)
    buf: list[str] = []
    while pos < n:
        ch: str = text[pos]
        if ch == sep or ch == "\n" or ch == "\r":
            break
        if ch == '"':
            raise CsvError(line, pos + 1)
        buf.append(ch)
        pos += 1
    return ("".join(buf), pos, line)


def _parse_quoted(text: str, pos: int, line: int) -> tuple[str, int, int]:
    """Parse a quoted field starting at the opening quote."""
    n: int = len(text)
    pos += 1
    buf: list[str] = []
    while pos < n:
        ch: str = text[pos]
        if ch == '"':
            pos += 1
            if pos < n and text[pos] == '"':
                buf.append('"')
                pos += 1
                continue
            return ("".join(buf), pos, line)
        if ch == "\r":
            pos += 1
            if pos < n and text[pos] == "\n":
                pos += 1
            buf.append("\n")
            line += 1
            continue
        if ch == "\n":
            buf.append("\n")
            line += 1
            pos += 1
            continue
        buf.append(ch)
        pos += 1
    raise CsvError(line, pos + 1)


def write(records: list[list[str]]) -> str:
    """Write records to CSV text (RFC 4180, LF line endings)."""
    return _write(records, ",")


def write_tsv(records: list[list[str]]) -> str:
    """Write records to tab-separated text."""
    return _write(records, "\t")


def _write(records: list[list[str]], sep: str) -> str:
    parts: list[str] = []
    i: int = 0
    while i < len(records):
        record: list[str] = records[i]
        j: int = 0
        while j < len(record):
            if j > 0:
                parts.append(sep)
            parts.append(_quote_field(record[j], sep))
            j += 1
        parts.append("\n")
        i += 1
    return "".join(parts)


def _needs_quoting(field: str, sep: str) -> bool:
    i: int = 0
    while i < len(field):
        ch: str = field[i]
        if ch == sep or ch == '"' or ch == "\n" or ch == "\r":
            return True
        i += 1
    return False


def _quote_field(field: str, sep: str) -> str:
    if not _needs_quoting(field, sep):
        return field
    buf: list[str] = ['"']
    i: int = 0
    while i < len(field):
        ch: str = field[i]
        if ch == '"':
            buf.append('""')
        else:
            buf.append(ch)
        i += 1
    buf.append('"')
    return "".join(buf)
