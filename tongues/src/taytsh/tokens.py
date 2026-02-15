"""Taytsh tokenizer — lexes source into a flat token list."""

from __future__ import annotations


# Token type constants
TK_INT = "INT"
TK_FLOAT = "FLOAT"
TK_BYTE = "BYTE"
TK_STRING = "STRING"
TK_RUNE = "RUNE"
TK_BYTES = "BYTES"
TK_IDENT = "IDENT"
TK_OP = "OP"
TK_EOF = "EOF"

KEYWORDS: set[str] = {
    "bool",
    "break",
    "byte",
    "bytes",
    "case",
    "catch",
    "continue",
    "default",
    "else",
    "enum",
    "false",
    "finally",
    "float",
    "fn",
    "for",
    "if",
    "in",
    "int",
    "interface",
    "let",
    "list",
    "map",
    "match",
    "nil",
    "range",
    "return",
    "rune",
    "self",
    "set",
    "string",
    "struct",
    "throw",
    "true",
    "try",
    "void",
    "while",
}

# Multi-character operators, sorted by length descending for greedy matching
MULTI_OPS: list[str] = [
    "<<=",
    ">>=",
    ">>>",
    "->",
    "=>",
    "&&",
    "||",
    "<=",
    ">=",
    "==",
    "!=",
    "<<",
    ">>",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
]

SINGLE_OPS: set[str] = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "&",
    "|",
    "^",
    "~",
    "!",
    "<",
    ">",
    "=",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ",",
    ":",
    ".",
    "?",
    "@",
}

ESCAPE_MAP: dict[str, str] = {
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "\\": "\\",
    '"': '"',
    "'": "'",
    "0": "\0",
}


class TokenizeError(Exception):
    """Error during tokenization."""

    def __init__(self, msg: str, line: int, col: int):
        self.msg: str = msg
        self.line: int = line
        self.col: int = col
        super().__init__(msg + " at line " + str(line) + " col " + str(col))


class Token:
    """A token with type, value, and position."""

    def __init__(self, type_: str, value: str, line: int, col: int):
        self.type: str = type_
        self.value: str = value
        self.line: int = line
        self.col: int = col
        self.bytes_value: bytes = b""

    def __repr__(self) -> str:
        return (
            "Token("
            + self.type
            + ", "
            + repr(self.value)
            + ", "
            + str(self.line)
            + ", "
            + str(self.col)
            + ")"
        )


def _is_digit(c: str) -> bool:
    return c >= "0" and c <= "9"


def _is_hex(c: str) -> bool:
    return (c >= "0" and c <= "9") or (c >= "a" and c <= "f") or (c >= "A" and c <= "F")


def _is_alpha(c: str) -> bool:
    return (c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or c == "_"


def _is_alnum(c: str) -> bool:
    return _is_alpha(c) or _is_digit(c)


def _hex_val(c: str) -> int:
    if c >= "0" and c <= "9":
        return ord(c) - ord("0")
    if c >= "a" and c <= "f":
        return ord(c) - ord("a") + 10
    return ord(c) - ord("A") + 10


def _process_escape(src: str, pos: int, line: int, col: int) -> tuple[str, int]:
    """Process escape after backslash. Returns (resolved_char, new_pos)."""
    if pos >= len(src):
        raise TokenizeError("unexpected end of string in escape", line, col)
    c = src[pos]
    if c in ESCAPE_MAP:
        return ESCAPE_MAP[c], pos + 1
    if c == "x":
        if pos + 2 >= len(src):
            raise TokenizeError("incomplete \\x escape", line, col)
        h1 = src[pos + 1]
        h2 = src[pos + 2]
        if not _is_hex(h1) or not _is_hex(h2):
            raise TokenizeError("invalid hex escape", line, col)
        val = _hex_val(h1) * 16 + _hex_val(h2)
        return chr(val), pos + 3
    raise TokenizeError("invalid escape: \\" + c, line, col)


def _process_escape_byte(src: str, pos: int, line: int, col: int) -> tuple[int, int]:
    """Process escape for bytes literal. Returns (byte_value, new_pos)."""
    if pos >= len(src):
        raise TokenizeError("unexpected end of bytes in escape", line, col)
    c = src[pos]
    if c == "n":
        return ord("\n"), pos + 1
    if c == "r":
        return ord("\r"), pos + 1
    if c == "t":
        return ord("\t"), pos + 1
    if c == "\\":
        return ord("\\"), pos + 1
    if c == '"':
        return ord('"'), pos + 1
    if c == "0":
        return 0, pos + 1
    if c == "x":
        if pos + 2 >= len(src):
            raise TokenizeError("incomplete \\x escape in bytes", line, col)
        h1 = src[pos + 1]
        h2 = src[pos + 2]
        if not _is_hex(h1) or not _is_hex(h2):
            raise TokenizeError("invalid hex escape in bytes", line, col)
        val = _hex_val(h1) * 16 + _hex_val(h2)
        return val, pos + 3
    raise TokenizeError("invalid escape in bytes: \\" + c, line, col)


def tokenize(source: str) -> list[Token]:
    """Tokenize Taytsh source into a flat list ending with TK_EOF."""
    tokens: list[Token] = []
    pos = 0
    line = 1
    col = 1
    length = len(source)

    while pos < length:
        c = source[pos]

        # Newlines
        if c == "\n":
            pos += 1
            line += 1
            col = 1
            continue

        # Whitespace
        if c == " " or c == "\t" or c == "\r":
            pos += 1
            col += 1
            continue

        # Line comment: --
        if c == "-" and pos + 1 < length and source[pos + 1] == "-":
            while pos < length and source[pos] != "\n":
                pos += 1
            continue

        start_pos = pos
        start_line = line
        start_col = col

        # Bytes literal: b"..."
        if c == "b" and pos + 1 < length and source[pos + 1] == '"':
            pos += 2
            col += 2
            byte_vals: list[int] = []
            while pos < length and source[pos] != '"':
                if source[pos] == "\n":
                    raise TokenizeError(
                        "unterminated bytes literal", start_line, start_col
                    )
                if source[pos] == "\\":
                    pos += 1
                    col += 1
                    bval, pos = _process_escape_byte(source, pos, start_line, col)
                    byte_vals.append(bval)
                else:
                    byte_vals.append(ord(source[pos]))
                    pos += 1
                col += 1
            if pos >= length:
                raise TokenizeError("unterminated bytes literal", start_line, start_col)
            pos += 1  # skip closing "
            col += 1
            raw = source[start_pos:pos]
            tok = Token(TK_BYTES, raw, start_line, start_col)
            tok.bytes_value = bytes(byte_vals)
            tokens.append(tok)
            continue

        # Number: 0x byte, int, or float
        if _is_digit(c):
            if (
                c == "0"
                and pos + 1 < length
                and (source[pos + 1] == "x" or source[pos + 1] == "X")
            ):
                pos += 2
                col += 2
                hex_start = pos
                while pos < length and _is_hex(source[pos]):
                    pos += 1
                    col += 1
                hex_len = pos - hex_start
                if hex_len != 2:
                    raise TokenizeError(
                        "byte literal must have exactly 2 hex digits",
                        start_line,
                        start_col,
                    )
                raw = source[start_pos:pos]
                tokens.append(Token(TK_BYTE, raw, start_line, start_col))
                continue

            while pos < length and _is_digit(source[pos]):
                pos += 1
                col += 1
            is_float = False
            after_dot = tokens and tokens[-1].type == TK_OP and tokens[-1].value == "."
            if not after_dot and pos < length and source[pos] == ".":
                if pos + 1 < length and _is_digit(source[pos + 1]):
                    is_float = True
                    pos += 1
                    col += 1
                    while pos < length and _is_digit(source[pos]):
                        pos += 1
                        col += 1
            if pos < length and (source[pos] == "e" or source[pos] == "E"):
                is_float = True
                pos += 1
                col += 1
                if pos < length and (source[pos] == "+" or source[pos] == "-"):
                    pos += 1
                    col += 1
                if pos >= length or not _is_digit(source[pos]):
                    raise TokenizeError("invalid float exponent", start_line, start_col)
                while pos < length and _is_digit(source[pos]):
                    pos += 1
                    col += 1
            raw = source[start_pos:pos]
            if is_float:
                tokens.append(Token(TK_FLOAT, raw, start_line, start_col))
            else:
                tokens.append(Token(TK_INT, raw, start_line, start_col))
            continue

        # String literal: "..."
        if c == '"':
            pos += 1
            col += 1
            chars: list[str] = []
            while pos < length and source[pos] != '"':
                if source[pos] == "\n":
                    raise TokenizeError(
                        "unterminated string literal", start_line, start_col
                    )
                if source[pos] == "\\":
                    pos += 1
                    col += 1
                    ch, pos = _process_escape(source, pos, start_line, col)
                    chars.append(ch)
                else:
                    chars.append(source[pos])
                    pos += 1
                col += 1
            if pos >= length:
                raise TokenizeError(
                    "unterminated string literal", start_line, start_col
                )
            pos += 1  # skip closing "
            col += 1
            value = ""
            for ch in chars:
                value += ch
            tokens.append(Token(TK_STRING, value, start_line, start_col))
            continue

        # Rune literal: '...'
        if c == "'":
            pos += 1
            col += 1
            if pos >= length or source[pos] == "\n":
                raise TokenizeError("unterminated rune literal", start_line, start_col)
            if source[pos] == "\\":
                pos += 1
                col += 1
                rune_ch, pos = _process_escape(source, pos, start_line, col)
            elif source[pos] == "'":
                raise TokenizeError("empty rune literal", start_line, start_col)
            else:
                rune_ch = source[pos]
                pos += 1
            col += 1
            if pos >= length or source[pos] != "'":
                raise TokenizeError("unterminated rune literal", start_line, start_col)
            pos += 1  # skip closing '
            col += 1
            tokens.append(Token(TK_RUNE, rune_ch, start_line, start_col))
            continue

        # Identifier or keyword
        if _is_alpha(c):
            while pos < length and _is_alnum(source[pos]):
                pos += 1
                col += 1
            word = source[start_pos:pos]
            if word in KEYWORDS:
                tokens.append(Token(word, word, start_line, start_col))
            else:
                tokens.append(Token(TK_IDENT, word, start_line, start_col))
            continue

        # Multi-character operators
        matched = False
        for op in MULTI_OPS:
            op_len = len(op)
            if pos + op_len <= length and source[pos : pos + op_len] == op:
                tokens.append(Token(TK_OP, op, start_line, start_col))
                pos += op_len
                col += op_len
                matched = True
                break
        if matched:
            continue

        # Single-character operators
        if c in SINGLE_OPS:
            tokens.append(Token(TK_OP, c, start_line, start_col))
            pos += 1
            col += 1
            continue

        raise TokenizeError("unexpected character: " + repr(c), line, col)

    tokens.append(Token(TK_EOF, "", line, col))
    return tokens
