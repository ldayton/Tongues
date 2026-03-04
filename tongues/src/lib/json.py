"""JSON parser using static union types."""

from __future__ import annotations

import sys
from dataclasses import dataclass


class JsonError(Exception):
    pass


_HEX: str = "0123456789abcdef"


# -- JSON value types (discriminated union via isinstance) --


@dataclass
class JsonNull:
    pass


@dataclass
class JsonBool:
    value: bool


@dataclass
class JsonNumber:
    value: float


@dataclass
class JsonString:
    value: str


@dataclass
class JsonArray:
    items: list[JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject]


@dataclass
class JsonObject:
    entries: list[
        tuple[
            str, JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject
        ]
    ]


# -- Parser state --


@dataclass
class ParseResult:
    value: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject
    pos: int


def _skip_ws(s: str, pos: int) -> int:
    while pos < len(s):
        ch: str = s[pos]
        if ch == " " or ch == "\t" or ch == "\n" or ch == "\r":
            pos += 1
        else:
            break
    return pos


def _expect(s: str, pos: int, ch: str) -> int:
    if pos >= len(s):
        raise JsonError(f"unexpected end of input, expected '{ch}'")
    if s[pos] != ch:
        raise JsonError(f"expected '{ch}' at position {str(pos)}, got '{s[pos]}'")
    return pos + 1


# -- Primitive parsers --


def _parse_null(s: str, pos: int) -> ParseResult:
    if s[pos : pos + 4] == "null":
        return ParseResult(JsonNull(), pos + 4)
    raise JsonError(f"expected 'null' at position {str(pos)}")


def _parse_true(s: str, pos: int) -> ParseResult:
    if s[pos : pos + 4] == "true":
        return ParseResult(JsonBool(True), pos + 4)
    raise JsonError(f"expected 'true' at position {str(pos)}")


def _parse_false(s: str, pos: int) -> ParseResult:
    if s[pos : pos + 5] == "false":
        return ParseResult(JsonBool(False), pos + 5)
    raise JsonError(f"expected 'false' at position {str(pos)}")


# -- Number parser --


def _is_digit(ch: str) -> bool:
    return ch >= "0" and ch <= "9"


def _parse_number(s: str, pos: int) -> ParseResult:
    start: int = pos
    if pos < len(s) and s[pos] == "-":
        pos += 1
    if pos >= len(s) or not _is_digit(s[pos]):
        raise JsonError(f"expected digit at position {str(pos)}")
    if s[pos] == "0":
        pos += 1
    else:
        while pos < len(s) and _is_digit(s[pos]):
            pos += 1
    if pos < len(s) and s[pos] == ".":
        pos += 1
        if pos >= len(s) or not _is_digit(s[pos]):
            raise JsonError(f"expected digit after '.' at position {str(pos)}")
        while pos < len(s) and _is_digit(s[pos]):
            pos += 1
    if pos < len(s) and (s[pos] == "e" or s[pos] == "E"):
        pos += 1
        if pos < len(s) and (s[pos] == "+" or s[pos] == "-"):
            pos += 1
        if pos >= len(s) or not _is_digit(s[pos]):
            raise JsonError(f"expected digit in exponent at position {str(pos)}")
        while pos < len(s) and _is_digit(s[pos]):
            pos += 1
    num_str: str = s[start:pos]
    return ParseResult(JsonNumber(float(num_str)), pos)


# -- String parser --


def _hex_value(ch: str) -> int:
    if ch >= "0" and ch <= "9":
        return ord(ch) - ord("0")
    if ch >= "a" and ch <= "f":
        return ord(ch) - ord("a") + 10
    if ch >= "A" and ch <= "F":
        return ord(ch) - ord("A") + 10
    return -1


def _parse_hex4(s: str, pos: int) -> tuple[int, int]:
    if pos + 4 > len(s):
        raise JsonError(f"incomplete unicode escape at position {str(pos)}")
    result: int = 0
    for i in range(4):
        v: int = _hex_value(s[pos + i])
        if v < 0:
            raise JsonError(f"invalid hex digit at position {str(pos + i)}")
        result = result * 16 + v
    return result, pos + 4


def _parse_string_body(s: str, pos: int) -> tuple[str, int]:
    parts: list[str] = []
    while pos < len(s):
        ch: str = s[pos]
        if ch == '"':
            return "".join(parts), pos + 1
        if ch == "\\":
            pos += 1
            if pos >= len(s):
                raise JsonError("unexpected end of input in string escape")
            esc: str = s[pos]
            if esc == '"':
                parts.append('"')
                pos += 1
            elif esc == "\\":
                parts.append("\\")
                pos += 1
            elif esc == "/":
                parts.append("/")
                pos += 1
            elif esc == "b":
                parts.append("\b")
                pos += 1
            elif esc == "f":
                parts.append("\f")
                pos += 1
            elif esc == "n":
                parts.append("\n")
                pos += 1
            elif esc == "r":
                parts.append("\r")
                pos += 1
            elif esc == "t":
                parts.append("\t")
                pos += 1
            elif esc == "u":
                code, pos = _parse_hex4(s, pos + 1)
                if code >= 0xD800 and code <= 0xDBFF:
                    if pos + 1 < len(s) and s[pos] == "\\" and s[pos + 1] == "u":
                        low, pos = _parse_hex4(s, pos + 2)
                        if low >= 0xDC00 and low <= 0xDFFF:
                            code = 0x10000 + (code - 0xD800) * 0x400 + (low - 0xDC00)
                        else:
                            raise JsonError(
                                f"invalid low surrogate at position {str(pos)}"
                            )
                    else:
                        raise JsonError(f"missing low surrogate at position {str(pos)}")
                parts.append(chr(code))
            else:
                raise JsonError(
                    "invalid escape '\\" + esc + "' at position " + str(pos)
                )
        else:
            parts.append(ch)
            pos += 1
    raise JsonError("unterminated string")


def _parse_string(s: str, pos: int) -> ParseResult:
    pos = _expect(s, pos, '"')
    text, pos = _parse_string_body(s, pos)
    return ParseResult(JsonString(text), pos)


def _parse_string_raw(s: str, pos: int) -> tuple[str, int]:
    pos = _expect(s, pos, '"')
    return _parse_string_body(s, pos)


# -- Compound parsers --


def _parse_array(s: str, pos: int) -> ParseResult:
    pos = _expect(s, pos, "[")
    pos = _skip_ws(s, pos)
    items: list[
        JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject
    ] = []
    if pos < len(s) and s[pos] == "]":
        return ParseResult(JsonArray(items), pos + 1)
    result: ParseResult = _parse_value(s, pos)
    items.append(result.value)
    pos = _skip_ws(s, result.pos)
    while pos < len(s) and s[pos] == ",":
        pos = _skip_ws(s, pos + 1)
        result = _parse_value(s, pos)
        items.append(result.value)
        pos = _skip_ws(s, result.pos)
    pos = _expect(s, pos, "]")
    return ParseResult(JsonArray(items), pos)


def _parse_object(s: str, pos: int) -> ParseResult:
    pos = _expect(s, pos, "{")
    pos = _skip_ws(s, pos)
    entries: list[
        tuple[
            str, JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject
        ]
    ] = []
    if pos < len(s) and s[pos] == "}":
        return ParseResult(JsonObject(entries), pos + 1)
    key, pos = _parse_string_raw(s, _skip_ws(s, pos))
    pos = _skip_ws(s, pos)
    pos = _expect(s, pos, ":")
    result: ParseResult = _parse_value(s, pos)
    entries.append((key, result.value))
    pos = _skip_ws(s, result.pos)
    while pos < len(s) and s[pos] == ",":
        pos = _skip_ws(s, pos + 1)
        key, pos = _parse_string_raw(s, _skip_ws(s, pos))
        pos = _skip_ws(s, pos)
        pos = _expect(s, pos, ":")
        result = _parse_value(s, pos)
        entries.append((key, result.value))
        pos = _skip_ws(s, result.pos)
    pos = _expect(s, pos, "}")
    return ParseResult(JsonObject(entries), pos)


# -- Top-level dispatch --


def _parse_value(s: str, pos: int) -> ParseResult:
    pos = _skip_ws(s, pos)
    if pos >= len(s):
        raise JsonError("unexpected end of input")
    ch: str = s[pos]
    if ch == "n":
        return _parse_null(s, pos)
    elif ch == "t":
        return _parse_true(s, pos)
    elif ch == "f":
        return _parse_false(s, pos)
    elif ch == '"':
        return _parse_string(s, pos)
    elif ch == "[":
        return _parse_array(s, pos)
    elif ch == "{":
        return _parse_object(s, pos)
    elif ch == "-" or _is_digit(ch):
        return _parse_number(s, pos)
    else:
        raise JsonError(f"unexpected character '{ch}' at position {str(pos)}")


def json_parse(
    s: str,
) -> JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject:
    """Parse a JSON string into a JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject."""
    result: ParseResult = _parse_value(s, 0)
    pos: int = _skip_ws(s, result.pos)
    if pos < len(s):
        raise JsonError(f"trailing content at position {str(pos)}")
    return result.value


# -- Serialization --


def json_stringify(
    value: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject,
) -> str:
    """Serialize a JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject to a JSON string."""
    if isinstance(value, JsonNull):
        return "null"
    elif isinstance(value, JsonBool):
        if value.value:
            return "true"
        else:
            return "false"
    elif isinstance(value, JsonNumber):
        n: float = value.value
        if n != n or n == float("inf") or n == float("-inf"):
            return str(n)
        i: int = int(n)
        if float(i) == n and abs(n) < 1e15:
            return str(i)
        return str(n)
    elif isinstance(value, JsonString):
        return _escape_string(value.value)
    elif isinstance(value, JsonArray):
        parts: list[str] = []
        for item in value.items:
            parts.append(json_stringify(item))
        return "[" + ",".join(parts) + "]"
    elif isinstance(value, JsonObject):
        parts: list[str] = []
        for key, val in value.entries:
            parts.append(_escape_string(key) + ":" + json_stringify(val))
        return "{" + ",".join(parts) + "}"
    raise JsonError("invalid json value")


def _escape_string(s: str) -> str:
    parts: list[str] = ['"']
    for ch in s:
        if ch == '"':
            parts.append('\\"')
        elif ch == "\\":
            parts.append("\\\\")
        elif ch == "\n":
            parts.append("\\n")
        elif ch == "\r":
            parts.append("\\r")
        elif ch == "\t":
            parts.append("\\t")
        elif ch == "\b":
            parts.append("\\b")
        elif ch == "\f":
            parts.append("\\f")
        elif ord(ch) < 0x20:
            code: int = ord(ch)
            h3: str = _HEX[code % 16]
            h2: str = _HEX[(code // 16) % 16]
            parts.append("\\u00" + h2 + h3)
        else:
            parts.append(ch)
    parts.append('"')
    return "".join(parts)


# -- Accessors --


def json_get_string(
    value: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject,
) -> str:
    if isinstance(value, JsonString):
        return value.value
    raise JsonError("expected JsonString")


def json_get_number(
    value: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject,
) -> float:
    if isinstance(value, JsonNumber):
        return value.value
    raise JsonError("expected JsonNumber")


def json_get_bool(
    value: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject,
) -> bool:
    if isinstance(value, JsonBool):
        return value.value
    raise JsonError("expected JsonBool")


def json_get_items(
    value: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject,
) -> list[JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject]:
    if isinstance(value, JsonArray):
        return value.items
    raise JsonError("expected JsonArray")


def json_get_field(
    value: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject,
    key: str,
) -> JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject:
    if isinstance(value, JsonObject):
        for k, v in value.entries:
            if k == key:
                return v
        raise JsonError(f"key '{key}' not found")
    raise JsonError("expected JsonObject")


def json_is_null(
    value: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject,
) -> bool:
    return isinstance(value, JsonNull)


# -- Main (self-test) --


def _check(name: str, input: str, expected: str) -> bool:
    try:
        v: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
            json_parse(input)
        )
        actual: str = json_stringify(v)
        if actual == expected:
            print("  PASS " + name)
            return True
        else:
            print("  FAIL " + name + ": got " + actual + ", want " + expected)
            return False
    except JsonError as e:
        print("  FAIL " + name + ": " + str(e))
        return False


def _self_test() -> int:
    passed: int = 0
    failed: int = 0
    cases: list[tuple[str, str, str]] = [
        ("null", "null", "null"),
        ("true", "true", "true"),
        ("false", "false", "false"),
        ("zero", "0", "0"),
        ("int", "42", "42"),
        ("neg", "-7", "-7"),
        ("float", "3.14", "3.14"),
        ("exp", "1e10", "10000000000"),
        ("string", '"hello"', '"hello"'),
        ("escape", '"a\\nb"', '"a\\nb"'),
        ("unicode", '"\\u0041"', '"A"'),
        ("empty_array", "[]", "[]"),
        ("array", "[1,2,3]", "[1,2,3]"),
        ("nested", "[[1],[2]]", "[[1],[2]]"),
        ("empty_object", "{}", "{}"),
        ("object", '{"a":1,"b":2}', '{"a":1,"b":2}'),
        ("whitespace", '  { "x" : [ 1 , 2 ] }  ', '{"x":[1,2]}'),
        (
            "mixed",
            '{"n":null,"b":true,"a":[1,"two"]}',
            '{"n":null,"b":true,"a":[1,"two"]}',
        ),
    ]
    i: int = 0
    while i < len(cases):
        if _check(cases[i][0], cases[i][1], cases[i][2]):
            passed += 1
        else:
            failed += 1
        i += 1

    # round-trip test
    try:
        src: str = '{"name":"test","values":[1,2.5,true,null],"nested":{"a":"b"}}'
        rt: str = json_stringify(json_parse(src))
        rt2: str = json_stringify(json_parse(rt))
        if rt == rt2:
            print("  PASS round_trip")
            passed += 1
        else:
            print("  FAIL round_trip: mismatch after double round-trip")
            failed += 1
    except JsonError as e:
        print("  FAIL round_trip: " + str(e))
        failed += 1

    # accessor test
    try:
        doc: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
            json_parse('{"name":"alice","age":30,"scores":[95,87]}')
        )
        assert json_get_string(json_get_field(doc, "name")) == "alice"
        assert json_get_number(json_get_field(doc, "age")) == 30.0
        scores: list[
            JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject
        ] = json_get_items(json_get_field(doc, "scores"))
        assert len(scores) == 2
        assert json_get_number(scores[0]) == 95.0
        print("  PASS accessors")
        passed += 1
    except JsonError as e:
        print("  FAIL accessors: " + str(e))
        failed += 1
    except AssertionError as e:
        print("  FAIL accessors: " + str(e))
        failed += 1

    # error test
    try:
        json_parse("{invalid}")
        print("  FAIL error_handling: should have raised")
        failed += 1
    except JsonError:
        print("  PASS error_handling")
        passed += 1

    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
