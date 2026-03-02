"""JSON parse/stringify tests."""

import sys

from lib.json import (
    JsonNull,
    JsonBool,
    JsonNumber,
    JsonString,
    JsonArray,
    JsonObject,
    JsonError,
    parse,
    stringify,
    get_string,
    get_number,
    get_bool,
    get_items,
    get_field,
    is_null,
)


def test_parse_primitives() -> None:
    assert stringify(parse("null")) == "null"
    assert stringify(parse("true")) == "true"
    assert stringify(parse("false")) == "false"
    assert stringify(parse("0")) == "0"
    assert stringify(parse("42")) == "42"
    assert stringify(parse("-7")) == "-7"
    assert stringify(parse("3.14")) == "3.14"
    assert stringify(parse("1e10")) == "10000000000"
    assert stringify(parse("-0.5")) == "-0.5"
    assert stringify(parse("1E2")) == "100"
    assert stringify(parse("1e+2")) == "100"
    assert stringify(parse("1e-1")) == "0.1"


def test_parse_strings() -> None:
    assert stringify(parse('"hello"')) == '"hello"'
    assert stringify(parse('""')) == '""'
    assert stringify(parse('"a\\nb"')) == '"a\\nb"'
    assert stringify(parse('"a\\tb"')) == '"a\\tb"'
    assert stringify(parse('"a\\\\b"')) == '"a\\\\b"'
    assert stringify(parse('"a\\"b"')) == '"a\\"b"'
    assert stringify(parse('"a\\/b"')) == '"a/b"'
    assert stringify(parse('"\\b\\f"')) == '"\\b\\f"'


def test_parse_unicode() -> None:
    assert stringify(parse('"\\u0041"')) == '"A"'
    assert stringify(parse('"\\u0048\\u0069"')) == '"Hi"'
    assert stringify(parse('"\\u0000"')) == '"\\u0000"'
    # Surrogate pair for U+1F600 (grinning face)
    emoji: str = get_string(parse('"\\uD83D\\uDE00"'))
    assert len(emoji) == 1
    assert ord(emoji) == 0x1F600


def test_parse_arrays() -> None:
    assert stringify(parse("[]")) == "[]"
    assert stringify(parse("[1,2,3]")) == "[1,2,3]"
    assert stringify(parse("[[1],[2]]")) == "[[1],[2]]"
    assert stringify(parse('[1,"two",true,null]')) == '[1,"two",true,null]'
    assert stringify(parse("[  1 , 2 , 3  ]")) == "[1,2,3]"


def test_parse_objects() -> None:
    assert stringify(parse("{}")) == "{}"
    assert stringify(parse('{"a":1,"b":2}')) == '{"a":1,"b":2}'
    assert stringify(parse('{"a":{"b":{"c":3}}}')) == '{"a":{"b":{"c":3}}}'
    assert (
        stringify(parse('{"n":null,"b":true,"a":[1,"two"]}'))
        == '{"n":null,"b":true,"a":[1,"two"]}'
    )


def test_stringify() -> None:
    assert stringify(parse("null")) == "null"
    assert stringify(parse("true")) == "true"
    assert stringify(parse("false")) == "false"
    assert stringify(parse("42")) == "42"
    assert stringify(parse("3.14")) == "3.14"
    assert stringify(parse('"hello"')) == '"hello"'
    assert stringify(parse("[1,2,3]")) == "[1,2,3]"
    assert stringify(parse('{"a":1}')) == '{"a":1}'
    # Integers render without decimal
    assert stringify(parse("100")) == "100"
    assert stringify(parse("-0")) == "0"


def test_round_trip() -> None:
    doc: str = '{"name":"test","values":[1,2.5,true,null],"nested":{"a":"b"}}'
    rt1: str = stringify(parse(doc))
    rt2: str = stringify(parse(rt1))
    assert rt1 == rt2
    complex_doc: str = (
        '{"arr":[[1,2],[3,4]],"obj":{"x":{"y":true}},"s":"hello\\nworld"}'
    )
    c1: str = stringify(parse(complex_doc))
    c2: str = stringify(parse(c1))
    assert c1 == c2


def test_accessors() -> None:
    doc: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = parse(
        '{"name":"alice","age":30,"active":true,"scores":[95,87],"extra":null}'
    )
    assert get_string(get_field(doc, "name")) == "alice"
    assert get_number(get_field(doc, "age")) == 30.0
    assert get_bool(get_field(doc, "active")) == True
    assert is_null(get_field(doc, "extra")) == True
    scores: list[
        JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject
    ] = get_items(get_field(doc, "scores"))
    assert len(scores) == 2
    assert get_number(scores[0]) == 95.0
    assert get_number(scores[1]) == 87.0


def test_accessor_errors() -> None:
    num: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = parse(
        "42"
    )
    str_val: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        parse('"hi"')
    )
    arr: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = parse(
        "[1]"
    )
    null_val: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        parse("null")
    )
    # get_string on non-string
    try:
        get_string(num)
        assert False, "expected JsonError"
    except JsonError:
        pass
    # get_number on non-number
    try:
        get_number(str_val)
        assert False, "expected JsonError"
    except JsonError:
        pass
    # get_bool on non-bool
    try:
        get_bool(num)
        assert False, "expected JsonError"
    except JsonError:
        pass
    # get_items on non-array
    try:
        get_items(num)
        assert False, "expected JsonError"
    except JsonError:
        pass
    # get_field on non-object
    try:
        get_field(arr, "x")
        assert False, "expected JsonError"
    except JsonError:
        pass
    # get_field with missing key
    obj: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = parse(
        '{"a":1}'
    )
    try:
        get_field(obj, "z")
        assert False, "expected JsonError"
    except JsonError:
        pass
    # is_null on non-null
    assert is_null(num) == False


def test_parse_errors() -> None:
    bad_inputs: list[str] = [
        "",
        "42 extra",
        '"unterminated',
        "[1,2",
        '{"a":1',
        '"bad\\qescape"',
        "tru",
        "fals",
        "nul",
        "{invalid}",
        "[,]",
    ]
    i: int = 0
    while i < len(bad_inputs):
        try:
            parse(bad_inputs[i])
            assert False, "expected JsonError for: " + bad_inputs[i]
        except JsonError:
            pass
        i += 1


def test_whitespace_handling() -> None:
    assert stringify(parse("  null  ")) == "null"
    assert stringify(parse("\t\n\r 42 \t\n\r")) == "42"
    assert stringify(parse('  { "x" : [ 1 , 2 ] }  ')) == '{"x":[1,2]}'
    assert stringify(parse(" [\n  1,\n  2,\n  3\n] ")) == "[1,2,3]"
    assert stringify(parse('{\r\n\t"a"\t:\t1\t}')) == '{"a":1}'


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_parse_primitives", test_parse_primitives),
        ("test_parse_strings", test_parse_strings),
        ("test_parse_unicode", test_parse_unicode),
        ("test_parse_arrays", test_parse_arrays),
        ("test_parse_objects", test_parse_objects),
        ("test_stringify", test_stringify),
        ("test_round_trip", test_round_trip),
        ("test_accessors", test_accessors),
        ("test_accessor_errors", test_accessor_errors),
        ("test_parse_errors", test_parse_errors),
        ("test_whitespace_handling", test_whitespace_handling),
    ]
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print("  PASS " + name)
        except AssertionError as e:
            failed += 1
            print("  FAIL " + name + ": " + str(e))
        except Exception as e:
            failed += 1
            print("  FAIL " + name + ": " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
