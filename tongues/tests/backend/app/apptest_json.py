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
    json_parse,
    json_stringify,
    json_get_string,
    json_get_number,
    json_get_bool,
    json_get_items,
    json_get_field,
    json_is_null,
)


def test_parse_primitives() -> None:
    assert json_stringify(json_parse("null")) == "null"
    assert json_stringify(json_parse("true")) == "true"
    assert json_stringify(json_parse("false")) == "false"
    assert json_stringify(json_parse("0")) == "0"
    assert json_stringify(json_parse("42")) == "42"
    assert json_stringify(json_parse("-7")) == "-7"
    assert json_stringify(json_parse("3.14")) == "3.14"
    assert json_stringify(json_parse("1e10")) == "10000000000"
    assert json_stringify(json_parse("-0.5")) == "-0.5"
    assert json_stringify(json_parse("1E2")) == "100"
    assert json_stringify(json_parse("1e+2")) == "100"
    assert json_stringify(json_parse("1e-1")) == "0.1"


def test_parse_strings() -> None:
    assert json_stringify(json_parse('"hello"')) == '"hello"'
    assert json_stringify(json_parse('""')) == '""'
    assert json_stringify(json_parse('"a\\nb"')) == '"a\\nb"'
    assert json_stringify(json_parse('"a\\tb"')) == '"a\\tb"'
    assert json_stringify(json_parse('"a\\\\b"')) == '"a\\\\b"'
    assert json_stringify(json_parse('"a\\"b"')) == '"a\\"b"'
    assert json_stringify(json_parse('"a\\/b"')) == '"a/b"'
    assert json_stringify(json_parse('"\\b\\f"')) == '"\\b\\f"'


def test_parse_unicode() -> None:
    assert json_stringify(json_parse('"\\u0041"')) == '"A"'
    assert json_stringify(json_parse('"\\u0048\\u0069"')) == '"Hi"'
    assert json_stringify(json_parse('"\\u0000"')) == '"\\u0000"'
    # Surrogate pair for U+1F600 (grinning face)
    emoji: str = json_get_string(json_parse('"\\uD83D\\uDE00"'))
    assert len(emoji) == 1
    assert ord(emoji) == 0x1F600


def test_parse_arrays() -> None:
    assert json_stringify(json_parse("[]")) == "[]"
    assert json_stringify(json_parse("[1,2,3]")) == "[1,2,3]"
    assert json_stringify(json_parse("[[1],[2]]")) == "[[1],[2]]"
    assert json_stringify(json_parse('[1,"two",true,null]')) == '[1,"two",true,null]'
    assert json_stringify(json_parse("[  1 , 2 , 3  ]")) == "[1,2,3]"


def test_parse_objects() -> None:
    assert json_stringify(json_parse("{}")) == "{}"
    assert json_stringify(json_parse('{"a":1,"b":2}')) == '{"a":1,"b":2}'
    assert json_stringify(json_parse('{"a":{"b":{"c":3}}}')) == '{"a":{"b":{"c":3}}}'
    assert (
        json_stringify(json_parse('{"n":null,"b":true,"a":[1,"two"]}'))
        == '{"n":null,"b":true,"a":[1,"two"]}'
    )


def test_stringify() -> None:
    assert json_stringify(json_parse("null")) == "null"
    assert json_stringify(json_parse("true")) == "true"
    assert json_stringify(json_parse("false")) == "false"
    assert json_stringify(json_parse("42")) == "42"
    assert json_stringify(json_parse("3.14")) == "3.14"
    assert json_stringify(json_parse('"hello"')) == '"hello"'
    assert json_stringify(json_parse("[1,2,3]")) == "[1,2,3]"
    assert json_stringify(json_parse('{"a":1}')) == '{"a":1}'
    # Integers render without decimal
    assert json_stringify(json_parse("100")) == "100"
    assert json_stringify(json_parse("-0")) == "0"


def test_round_trip() -> None:
    doc: str = '{"name":"test","values":[1,2.5,true,null],"nested":{"a":"b"}}'
    rt1: str = json_stringify(json_parse(doc))
    rt2: str = json_stringify(json_parse(rt1))
    assert rt1 == rt2
    complex_doc: str = (
        '{"arr":[[1,2],[3,4]],"obj":{"x":{"y":true}},"s":"hello\\nworld"}'
    )
    c1: str = json_stringify(json_parse(complex_doc))
    c2: str = json_stringify(json_parse(c1))
    assert c1 == c2


def test_accessors() -> None:
    doc: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        json_parse(
            '{"name":"alice","age":30,"active":true,"scores":[95,87],"extra":null}'
        )
    )
    assert json_get_string(json_get_field(doc, "name")) == "alice"
    assert json_get_number(json_get_field(doc, "age")) == 30.0
    assert json_get_bool(json_get_field(doc, "active")) == True
    assert json_is_null(json_get_field(doc, "extra")) == True
    scores: list[
        JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject
    ] = json_get_items(json_get_field(doc, "scores"))
    assert len(scores) == 2
    assert json_get_number(scores[0]) == 95.0
    assert json_get_number(scores[1]) == 87.0


def test_accessor_errors() -> None:
    num: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        json_parse("42")
    )
    str_val: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        json_parse('"hi"')
    )
    arr: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        json_parse("[1]")
    )
    null_val: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        json_parse("null")
    )
    # get_string on non-string
    try:
        json_get_string(num)
        assert False, "expected JsonError"
    except JsonError:
        pass
    # get_number on non-number
    try:
        json_get_number(str_val)
        assert False, "expected JsonError"
    except JsonError:
        pass
    # get_bool on non-bool
    try:
        json_get_bool(num)
        assert False, "expected JsonError"
    except JsonError:
        pass
    # get_items on non-array
    try:
        json_get_items(num)
        assert False, "expected JsonError"
    except JsonError:
        pass
    # get_field on non-object
    try:
        json_get_field(arr, "x")
        assert False, "expected JsonError"
    except JsonError:
        pass
    # get_field with missing key
    obj: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        json_parse('{"a":1}')
    )
    try:
        json_get_field(obj, "z")
        assert False, "expected JsonError"
    except JsonError:
        pass
    # is_null on non-null
    assert json_is_null(num) == False


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
            json_parse(bad_inputs[i])
            assert False, "expected JsonError for: " + bad_inputs[i]
        except JsonError:
            pass
        i += 1


def test_whitespace_handling() -> None:
    assert json_stringify(json_parse("  null  ")) == "null"
    assert json_stringify(json_parse("\t\n\r 42 \t\n\r")) == "42"
    assert json_stringify(json_parse('  { "x" : [ 1 , 2 ] }  ')) == '{"x":[1,2]}'
    assert json_stringify(json_parse(" [\n  1,\n  2,\n  3\n] ")) == "[1,2,3]"
    assert json_stringify(json_parse('{\r\n\t"a"\t:\t1\t}')) == '{"a":1}'


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_parse_primitives()
        passed += 1
        print("  PASS test_parse_primitives")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_primitives: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_primitives: " + str(e))
    try:
        test_parse_strings()
        passed += 1
        print("  PASS test_parse_strings")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_strings: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_strings: " + str(e))
    try:
        test_parse_unicode()
        passed += 1
        print("  PASS test_parse_unicode")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_unicode: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_unicode: " + str(e))
    try:
        test_parse_arrays()
        passed += 1
        print("  PASS test_parse_arrays")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_arrays: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_arrays: " + str(e))
    try:
        test_parse_objects()
        passed += 1
        print("  PASS test_parse_objects")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_objects: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_objects: " + str(e))
    try:
        test_stringify()
        passed += 1
        print("  PASS test_stringify")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_stringify: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_stringify: " + str(e))
    try:
        test_round_trip()
        passed += 1
        print("  PASS test_round_trip")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_round_trip: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_round_trip: " + str(e))
    try:
        test_accessors()
        passed += 1
        print("  PASS test_accessors")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_accessors: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_accessors: " + str(e))
    try:
        test_accessor_errors()
        passed += 1
        print("  PASS test_accessor_errors")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_accessor_errors: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_accessor_errors: " + str(e))
    try:
        test_parse_errors()
        passed += 1
        print("  PASS test_parse_errors")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_errors: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_errors: " + str(e))
    try:
        test_whitespace_handling()
        passed += 1
        print("  PASS test_whitespace_handling")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_whitespace_handling: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_whitespace_handling: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
