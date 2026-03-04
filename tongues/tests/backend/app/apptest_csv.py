"""CSV parser/writer tests — RFC 4180, edge cases, and roundtrips."""

import sys

from lib.csv import CsvError
from lib.csv import csv_parse
from lib.csv import csv_parse_tsv
from lib.csv import csv_write
from lib.csv import csv_write_tsv


# -- Basic parsing --


def test_parse_empty() -> None:
    assert csv_parse("") == []


def test_parse_single_field() -> None:
    assert csv_parse("hello\n") == [["hello"]]


def test_parse_single_field_no_newline() -> None:
    assert csv_parse("hello") == [["hello"]]


def test_parse_single_record() -> None:
    assert csv_parse("a,b,c\n") == [["a", "b", "c"]]


def test_parse_multiple_records() -> None:
    result: list[list[str]] = csv_parse("a,b\nc,d\n")
    assert result == [["a", "b"], ["c", "d"]]


def test_parse_no_trailing_newline() -> None:
    assert csv_parse("a,b\nc,d") == [["a", "b"], ["c", "d"]]


# -- Quoted fields --


def test_quoted_simple() -> None:
    assert csv_parse('"hello"\n') == [["hello"]]


def test_quoted_with_comma() -> None:
    assert csv_parse('"a,b",c\n') == [["a,b", "c"]]


def test_quoted_with_newline() -> None:
    assert csv_parse('"a\nb",c\n') == [["a\nb", "c"]]


def test_quoted_with_escaped_quote() -> None:
    assert csv_parse('"a""b"\n') == [['a"b']]


def test_quoted_empty() -> None:
    assert csv_parse('""\n') == [[""]]


def test_quoted_only_quotes() -> None:
    assert csv_parse('""""\n') == [['"']]


def test_quoted_multiple_escaped() -> None:
    assert csv_parse('"he said ""hi"" and ""bye"""\n') == [['he said "hi" and "bye"']]


# -- Empty fields --


def test_empty_fields() -> None:
    assert csv_parse(",\n") == [["", ""]]


def test_empty_middle() -> None:
    assert csv_parse("a,,b\n") == [["a", "", "b"]]


def test_trailing_comma() -> None:
    assert csv_parse("a,b,\n") == [["a", "b", ""]]


def test_leading_comma() -> None:
    assert csv_parse(",a,b\n") == [["", "a", "b"]]


def test_all_empty() -> None:
    assert csv_parse(",,\n") == [["", "", ""]]


# -- CRLF handling --


def test_crlf_line_ending() -> None:
    assert csv_parse("a,b\r\nc,d\r\n") == [["a", "b"], ["c", "d"]]


def test_cr_only_line_ending() -> None:
    assert csv_parse("a,b\rc,d\r") == [["a", "b"], ["c", "d"]]


def test_crlf_in_quoted_field() -> None:
    assert csv_parse('"a\r\nb"\n') == [["a\nb"]]


def test_mixed_line_endings() -> None:
    assert csv_parse("a\nb\r\nc\rd") == [["a"], ["b"], ["c"], ["d"]]


# -- Blank lines skipped --


def test_blank_lines_skipped() -> None:
    assert csv_parse("a\n\nb\n") == [["a"], ["b"]]


def test_multiple_blank_lines() -> None:
    assert csv_parse("\n\na,b\n\n\nc,d\n\n") == [["a", "b"], ["c", "d"]]


def test_only_blank_lines() -> None:
    assert csv_parse("\n\n\n") == []


# -- Bare quote errors --


def test_bare_quote_in_unquoted() -> None:
    try:
        csv_parse('a"b\n')
        assert False, "expected CsvError"
    except CsvError as e:
        assert e.line == 1


def test_unterminated_quote() -> None:
    try:
        csv_parse('"abc\n')
        assert False, "expected CsvError"
    except CsvError as e:
        assert e.line == 2


def test_unterminated_quote_eof() -> None:
    try:
        csv_parse('"abc')
        assert False, "expected CsvError"
    except CsvError as e:
        assert e.line == 1


# -- Writer --


def test_write_empty() -> None:
    assert csv_write([]) == ""


def test_write_single_record() -> None:
    assert csv_write([["a", "b", "c"]]) == "a,b,c\n"


def test_write_multiple_records() -> None:
    assert csv_write([["a", "b"], ["c", "d"]]) == "a,b\nc,d\n"


def test_write_quotes_comma() -> None:
    assert csv_write([["a,b", "c"]]) == '"a,b",c\n'


def test_write_quotes_newline() -> None:
    assert csv_write([["a\nb", "c"]]) == '"a\nb",c\n'


def test_write_quotes_quote() -> None:
    assert csv_write([['a"b', "c"]]) == '"a""b",c\n'


def test_write_empty_field() -> None:
    assert csv_write([["", "a"]]) == ",a\n"


def test_write_all_empty() -> None:
    assert csv_write([["", "", ""]]) == ",,\n"


# -- Roundtrip --


def test_roundtrip_simple() -> None:
    records: list[list[str]] = [["a", "b", "c"], ["d", "e", "f"]]
    assert csv_parse(csv_write(records)) == records


def test_roundtrip_quoted() -> None:
    records: list[list[str]] = [["hello, world", 'say "hi"'], ["a\nb", ""]]
    assert csv_parse(csv_write(records)) == records


def test_roundtrip_empty_fields() -> None:
    records: list[list[str]] = [["", "", ""], ["a", "", "b"]]
    assert csv_parse(csv_write(records)) == records


def test_roundtrip_single_field() -> None:
    records: list[list[str]] = [["only"]]
    assert csv_parse(csv_write(records)) == records


# -- TSV --


def test_tsv_parse() -> None:
    assert csv_parse_tsv("a\tb\tc\n") == [["a", "b", "c"]]


def test_tsv_write() -> None:
    assert csv_write_tsv([["a", "b", "c"]]) == "a\tb\tc\n"


def test_tsv_roundtrip() -> None:
    records: list[list[str]] = [["hello\tworld", "b"], ["c", "d"]]
    assert csv_parse_tsv(csv_write_tsv(records)) == records


def test_tsv_comma_not_special() -> None:
    """Commas are literal in TSV."""
    assert csv_parse_tsv("a,b\tc\n") == [["a,b", "c"]]


# -- Realistic data --


def test_header_and_data() -> None:
    text: str = 'name,age,city\nAlice,30,"New York"\nBob,25,London\n'
    result: list[list[str]] = csv_parse(text)
    assert len(result) == 3
    assert result[0] == ["name", "age", "city"]
    assert result[1] == ["Alice", "30", "New York"]
    assert result[2] == ["Bob", "25", "London"]


def test_all_quoted_fields() -> None:
    assert csv_parse('"a","b","c"\n') == [["a", "b", "c"]]


def test_quoted_at_middle() -> None:
    assert csv_parse('a,"b,c",d\n') == [["a", "b,c", "d"]]


def test_mixed_quoted_unquoted() -> None:
    assert csv_parse('plain,"has,comma",plain2\n') == [["plain", "has,comma", "plain2"]]


def test_field_just_quote() -> None:
    """A field whose value is a single double-quote character."""
    assert csv_parse('""""\n') == [['"']]


def test_single_column() -> None:
    assert csv_parse("a\nb\nc\n") == [["a"], ["b"], ["c"]]


def test_write_cr_in_field() -> None:
    assert csv_write([["a\rb", "c"]]) == '"a\rb",c\n'


def test_parse_cr_in_quoted_normalized() -> None:
    """Bare CR inside quoted field is normalized to LF."""
    assert csv_parse('"a\rb"\n') == [["a\nb"]]


def test_multiline_quoted_field() -> None:
    text: str = 'id,notes\n1,"line one\nline two\nline three"\n2,simple\n'
    result: list[list[str]] = csv_parse(text)
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
        ("test_all_quoted_fields", test_all_quoted_fields),
        ("test_quoted_at_middle", test_quoted_at_middle),
        ("test_mixed_quoted_unquoted", test_mixed_quoted_unquoted),
        ("test_field_just_quote", test_field_just_quote),
        ("test_single_column", test_single_column),
        ("test_write_cr_in_field", test_write_cr_in_field),
        ("test_parse_cr_in_quoted_normalized", test_parse_cr_in_quoted_normalized),
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
