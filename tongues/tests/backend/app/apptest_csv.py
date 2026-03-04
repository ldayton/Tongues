"""CSV parser/writer tests — RFC 4180, edge cases, and roundtrips."""

import sys

from lib.csv import CsvError
from lib.csv import parse
from lib.csv import parse_tsv
from lib.csv import write
from lib.csv import write_tsv


# -- Basic parsing --


def test_parse_empty() -> None:
    assert parse("") == []


def test_parse_single_field() -> None:
    assert parse("hello\n") == [["hello"]]


def test_parse_single_field_no_newline() -> None:
    assert parse("hello") == [["hello"]]


def test_parse_single_record() -> None:
    assert parse("a,b,c\n") == [["a", "b", "c"]]


def test_parse_multiple_records() -> None:
    result: list[list[str]] = parse("a,b\nc,d\n")
    assert result == [["a", "b"], ["c", "d"]]


def test_parse_no_trailing_newline() -> None:
    assert parse("a,b\nc,d") == [["a", "b"], ["c", "d"]]


# -- Quoted fields --


def test_quoted_simple() -> None:
    assert parse('"hello"\n') == [["hello"]]


def test_quoted_with_comma() -> None:
    assert parse('"a,b",c\n') == [["a,b", "c"]]


def test_quoted_with_newline() -> None:
    assert parse('"a\nb",c\n') == [["a\nb", "c"]]


def test_quoted_with_escaped_quote() -> None:
    assert parse('"a""b"\n') == [['a"b']]


def test_quoted_empty() -> None:
    assert parse('""\n') == [[""]]


def test_quoted_only_quotes() -> None:
    assert parse('""""\n') == [['"']]


def test_quoted_multiple_escaped() -> None:
    assert parse('"he said ""hi"" and ""bye"""\n') == [['he said "hi" and "bye"']]


# -- Empty fields --


def test_empty_fields() -> None:
    assert parse(",\n") == [["", ""]]


def test_empty_middle() -> None:
    assert parse("a,,b\n") == [["a", "", "b"]]


def test_trailing_comma() -> None:
    assert parse("a,b,\n") == [["a", "b", ""]]


def test_leading_comma() -> None:
    assert parse(",a,b\n") == [["", "a", "b"]]


def test_all_empty() -> None:
    assert parse(",,\n") == [["", "", ""]]


# -- CRLF handling --


def test_crlf_line_ending() -> None:
    assert parse("a,b\r\nc,d\r\n") == [["a", "b"], ["c", "d"]]


def test_cr_only_line_ending() -> None:
    assert parse("a,b\rc,d\r") == [["a", "b"], ["c", "d"]]


def test_crlf_in_quoted_field() -> None:
    assert parse('"a\r\nb"\n') == [["a\nb"]]


def test_mixed_line_endings() -> None:
    assert parse("a\nb\r\nc\rd") == [["a"], ["b"], ["c"], ["d"]]


# -- Blank lines skipped --


def test_blank_lines_skipped() -> None:
    assert parse("a\n\nb\n") == [["a"], ["b"]]


def test_multiple_blank_lines() -> None:
    assert parse("\n\na,b\n\n\nc,d\n\n") == [["a", "b"], ["c", "d"]]


def test_only_blank_lines() -> None:
    assert parse("\n\n\n") == []


# -- Bare quote errors --


def test_bare_quote_in_unquoted() -> None:
    try:
        parse('a"b\n')
        assert False, "expected CsvError"
    except CsvError as e:
        assert e.line == 1


def test_unterminated_quote() -> None:
    try:
        parse('"abc\n')
        assert False, "expected CsvError"
    except CsvError as e:
        assert e.line == 2


def test_unterminated_quote_eof() -> None:
    try:
        parse('"abc')
        assert False, "expected CsvError"
    except CsvError as e:
        assert e.line == 1


# -- Writer --


def test_write_empty() -> None:
    assert write([]) == ""


def test_write_single_record() -> None:
    assert write([["a", "b", "c"]]) == "a,b,c\n"


def test_write_multiple_records() -> None:
    assert write([["a", "b"], ["c", "d"]]) == "a,b\nc,d\n"


def test_write_quotes_comma() -> None:
    assert write([["a,b", "c"]]) == '"a,b",c\n'


def test_write_quotes_newline() -> None:
    assert write([["a\nb", "c"]]) == '"a\nb",c\n'


def test_write_quotes_quote() -> None:
    assert write([['a"b', "c"]]) == '"a""b",c\n'


def test_write_empty_field() -> None:
    assert write([["", "a"]]) == ",a\n"


def test_write_all_empty() -> None:
    assert write([["", "", ""]]) == ",,\n"


# -- Roundtrip --


def test_roundtrip_simple() -> None:
    records: list[list[str]] = [["a", "b", "c"], ["d", "e", "f"]]
    assert parse(write(records)) == records


def test_roundtrip_quoted() -> None:
    records: list[list[str]] = [["hello, world", 'say "hi"'], ["a\nb", ""]]
    assert parse(write(records)) == records


def test_roundtrip_empty_fields() -> None:
    records: list[list[str]] = [["", "", ""], ["a", "", "b"]]
    assert parse(write(records)) == records


def test_roundtrip_single_field() -> None:
    records: list[list[str]] = [["only"]]
    assert parse(write(records)) == records


# -- TSV --


def test_tsv_parse() -> None:
    assert parse_tsv("a\tb\tc\n") == [["a", "b", "c"]]


def test_tsv_write() -> None:
    assert write_tsv([["a", "b", "c"]]) == "a\tb\tc\n"


def test_tsv_roundtrip() -> None:
    records: list[list[str]] = [["hello\tworld", "b"], ["c", "d"]]
    assert parse_tsv(write_tsv(records)) == records


def test_tsv_comma_not_special() -> None:
    """Commas are literal in TSV."""
    assert parse_tsv("a,b\tc\n") == [["a,b", "c"]]


# -- Realistic data --


def test_header_and_data() -> None:
    text: str = 'name,age,city\nAlice,30,"New York"\nBob,25,London\n'
    result: list[list[str]] = parse(text)
    assert len(result) == 3
    assert result[0] == ["name", "age", "city"]
    assert result[1] == ["Alice", "30", "New York"]
    assert result[2] == ["Bob", "25", "London"]


def test_multiline_quoted_field() -> None:
    text: str = 'id,notes\n1,"line one\nline two\nline three"\n2,simple\n'
    result: list[list[str]] = parse(text)
    assert len(result) == 3
    assert result[1] == ["1", "line one\nline two\nline three"]
    assert result[2] == ["2", "simple"]


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_parse_empty", test_parse_empty),
        ("test_parse_single_field", test_parse_single_field),
        ("test_parse_single_field_no_newline", test_parse_single_field_no_newline),
        ("test_parse_single_record", test_parse_single_record),
        ("test_parse_multiple_records", test_parse_multiple_records),
        ("test_parse_no_trailing_newline", test_parse_no_trailing_newline),
        ("test_quoted_simple", test_quoted_simple),
        ("test_quoted_with_comma", test_quoted_with_comma),
        ("test_quoted_with_newline", test_quoted_with_newline),
        ("test_quoted_with_escaped_quote", test_quoted_with_escaped_quote),
        ("test_quoted_empty", test_quoted_empty),
        ("test_quoted_only_quotes", test_quoted_only_quotes),
        ("test_quoted_multiple_escaped", test_quoted_multiple_escaped),
        ("test_empty_fields", test_empty_fields),
        ("test_empty_middle", test_empty_middle),
        ("test_trailing_comma", test_trailing_comma),
        ("test_leading_comma", test_leading_comma),
        ("test_all_empty", test_all_empty),
        ("test_crlf_line_ending", test_crlf_line_ending),
        ("test_cr_only_line_ending", test_cr_only_line_ending),
        ("test_crlf_in_quoted_field", test_crlf_in_quoted_field),
        ("test_mixed_line_endings", test_mixed_line_endings),
        ("test_blank_lines_skipped", test_blank_lines_skipped),
        ("test_multiple_blank_lines", test_multiple_blank_lines),
        ("test_only_blank_lines", test_only_blank_lines),
        ("test_bare_quote_in_unquoted", test_bare_quote_in_unquoted),
        ("test_unterminated_quote", test_unterminated_quote),
        ("test_unterminated_quote_eof", test_unterminated_quote_eof),
        ("test_write_empty", test_write_empty),
        ("test_write_single_record", test_write_single_record),
        ("test_write_multiple_records", test_write_multiple_records),
        ("test_write_quotes_comma", test_write_quotes_comma),
        ("test_write_quotes_newline", test_write_quotes_newline),
        ("test_write_quotes_quote", test_write_quotes_quote),
        ("test_write_empty_field", test_write_empty_field),
        ("test_write_all_empty", test_write_all_empty),
        ("test_roundtrip_simple", test_roundtrip_simple),
        ("test_roundtrip_quoted", test_roundtrip_quoted),
        ("test_roundtrip_empty_fields", test_roundtrip_empty_fields),
        ("test_roundtrip_single_field", test_roundtrip_single_field),
        ("test_tsv_parse", test_tsv_parse),
        ("test_tsv_write", test_tsv_write),
        ("test_tsv_roundtrip", test_tsv_roundtrip),
        ("test_tsv_comma_not_special", test_tsv_comma_not_special),
        ("test_header_and_data", test_header_and_data),
        ("test_multiline_quoted_field", test_multiline_quoted_field),
    ]
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print("  PASS " + name)
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {name}: {e!s}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {name}: {e!s}")
    print(f"{passed!s} passed, {failed!s} failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
