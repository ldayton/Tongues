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
    try:
        test_parse_empty()
        passed += 1
        print("  PASS test_parse_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_empty: " + str(e))
    try:
        test_parse_single_field()
        passed += 1
        print("  PASS test_parse_single_field")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_single_field: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_single_field: " + str(e))
    try:
        test_parse_single_field_no_newline()
        passed += 1
        print("  PASS test_parse_single_field_no_newline")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_single_field_no_newline: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_single_field_no_newline: " + str(e))
    try:
        test_parse_single_record()
        passed += 1
        print("  PASS test_parse_single_record")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_single_record: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_single_record: " + str(e))
    try:
        test_parse_multiple_records()
        passed += 1
        print("  PASS test_parse_multiple_records")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_multiple_records: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_multiple_records: " + str(e))
    try:
        test_parse_no_trailing_newline()
        passed += 1
        print("  PASS test_parse_no_trailing_newline")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_no_trailing_newline: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_no_trailing_newline: " + str(e))
    try:
        test_quoted_simple()
        passed += 1
        print("  PASS test_quoted_simple")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_quoted_simple: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_quoted_simple: " + str(e))
    try:
        test_quoted_with_comma()
        passed += 1
        print("  PASS test_quoted_with_comma")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_quoted_with_comma: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_quoted_with_comma: " + str(e))
    try:
        test_quoted_with_newline()
        passed += 1
        print("  PASS test_quoted_with_newline")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_quoted_with_newline: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_quoted_with_newline: " + str(e))
    try:
        test_quoted_with_escaped_quote()
        passed += 1
        print("  PASS test_quoted_with_escaped_quote")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_quoted_with_escaped_quote: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_quoted_with_escaped_quote: " + str(e))
    try:
        test_quoted_empty()
        passed += 1
        print("  PASS test_quoted_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_quoted_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_quoted_empty: " + str(e))
    try:
        test_quoted_only_quotes()
        passed += 1
        print("  PASS test_quoted_only_quotes")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_quoted_only_quotes: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_quoted_only_quotes: " + str(e))
    try:
        test_quoted_multiple_escaped()
        passed += 1
        print("  PASS test_quoted_multiple_escaped")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_quoted_multiple_escaped: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_quoted_multiple_escaped: " + str(e))
    try:
        test_empty_fields()
        passed += 1
        print("  PASS test_empty_fields")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_empty_fields: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_empty_fields: " + str(e))
    try:
        test_empty_middle()
        passed += 1
        print("  PASS test_empty_middle")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_empty_middle: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_empty_middle: " + str(e))
    try:
        test_trailing_comma()
        passed += 1
        print("  PASS test_trailing_comma")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_trailing_comma: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_trailing_comma: " + str(e))
    try:
        test_leading_comma()
        passed += 1
        print("  PASS test_leading_comma")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_leading_comma: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_leading_comma: " + str(e))
    try:
        test_all_empty()
        passed += 1
        print("  PASS test_all_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_all_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_all_empty: " + str(e))
    try:
        test_crlf_line_ending()
        passed += 1
        print("  PASS test_crlf_line_ending")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_crlf_line_ending: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_crlf_line_ending: " + str(e))
    try:
        test_cr_only_line_ending()
        passed += 1
        print("  PASS test_cr_only_line_ending")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_cr_only_line_ending: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_cr_only_line_ending: " + str(e))
    try:
        test_crlf_in_quoted_field()
        passed += 1
        print("  PASS test_crlf_in_quoted_field")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_crlf_in_quoted_field: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_crlf_in_quoted_field: " + str(e))
    try:
        test_mixed_line_endings()
        passed += 1
        print("  PASS test_mixed_line_endings")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_mixed_line_endings: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_mixed_line_endings: " + str(e))
    try:
        test_blank_lines_skipped()
        passed += 1
        print("  PASS test_blank_lines_skipped")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_blank_lines_skipped: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_blank_lines_skipped: " + str(e))
    try:
        test_multiple_blank_lines()
        passed += 1
        print("  PASS test_multiple_blank_lines")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_multiple_blank_lines: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_multiple_blank_lines: " + str(e))
    try:
        test_only_blank_lines()
        passed += 1
        print("  PASS test_only_blank_lines")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_only_blank_lines: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_only_blank_lines: " + str(e))
    try:
        test_bare_quote_in_unquoted()
        passed += 1
        print("  PASS test_bare_quote_in_unquoted")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_bare_quote_in_unquoted: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_bare_quote_in_unquoted: " + str(e))
    try:
        test_unterminated_quote()
        passed += 1
        print("  PASS test_unterminated_quote")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_unterminated_quote: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_unterminated_quote: " + str(e))
    try:
        test_unterminated_quote_eof()
        passed += 1
        print("  PASS test_unterminated_quote_eof")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_unterminated_quote_eof: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_unterminated_quote_eof: " + str(e))
    try:
        test_write_empty()
        passed += 1
        print("  PASS test_write_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_write_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_write_empty: " + str(e))
    try:
        test_write_single_record()
        passed += 1
        print("  PASS test_write_single_record")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_write_single_record: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_write_single_record: " + str(e))
    try:
        test_write_multiple_records()
        passed += 1
        print("  PASS test_write_multiple_records")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_write_multiple_records: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_write_multiple_records: " + str(e))
    try:
        test_write_quotes_comma()
        passed += 1
        print("  PASS test_write_quotes_comma")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_write_quotes_comma: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_write_quotes_comma: " + str(e))
    try:
        test_write_quotes_newline()
        passed += 1
        print("  PASS test_write_quotes_newline")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_write_quotes_newline: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_write_quotes_newline: " + str(e))
    try:
        test_write_quotes_quote()
        passed += 1
        print("  PASS test_write_quotes_quote")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_write_quotes_quote: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_write_quotes_quote: " + str(e))
    try:
        test_write_empty_field()
        passed += 1
        print("  PASS test_write_empty_field")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_write_empty_field: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_write_empty_field: " + str(e))
    try:
        test_write_all_empty()
        passed += 1
        print("  PASS test_write_all_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_write_all_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_write_all_empty: " + str(e))
    try:
        test_roundtrip_simple()
        passed += 1
        print("  PASS test_roundtrip_simple")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_simple: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_simple: " + str(e))
    try:
        test_roundtrip_quoted()
        passed += 1
        print("  PASS test_roundtrip_quoted")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_quoted: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_quoted: " + str(e))
    try:
        test_roundtrip_empty_fields()
        passed += 1
        print("  PASS test_roundtrip_empty_fields")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_empty_fields: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_empty_fields: " + str(e))
    try:
        test_roundtrip_single_field()
        passed += 1
        print("  PASS test_roundtrip_single_field")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_roundtrip_single_field: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_roundtrip_single_field: " + str(e))
    try:
        test_tsv_parse()
        passed += 1
        print("  PASS test_tsv_parse")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tsv_parse: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tsv_parse: " + str(e))
    try:
        test_tsv_write()
        passed += 1
        print("  PASS test_tsv_write")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tsv_write: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tsv_write: " + str(e))
    try:
        test_tsv_roundtrip()
        passed += 1
        print("  PASS test_tsv_roundtrip")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tsv_roundtrip: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tsv_roundtrip: " + str(e))
    try:
        test_tsv_comma_not_special()
        passed += 1
        print("  PASS test_tsv_comma_not_special")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_tsv_comma_not_special: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_tsv_comma_not_special: " + str(e))
    try:
        test_header_and_data()
        passed += 1
        print("  PASS test_header_and_data")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_header_and_data: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_header_and_data: " + str(e))
    try:
        test_all_quoted_fields()
        passed += 1
        print("  PASS test_all_quoted_fields")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_all_quoted_fields: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_all_quoted_fields: " + str(e))
    try:
        test_quoted_at_middle()
        passed += 1
        print("  PASS test_quoted_at_middle")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_quoted_at_middle: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_quoted_at_middle: " + str(e))
    try:
        test_mixed_quoted_unquoted()
        passed += 1
        print("  PASS test_mixed_quoted_unquoted")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_mixed_quoted_unquoted: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_mixed_quoted_unquoted: " + str(e))
    try:
        test_field_just_quote()
        passed += 1
        print("  PASS test_field_just_quote")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_field_just_quote: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_field_just_quote: " + str(e))
    try:
        test_single_column()
        passed += 1
        print("  PASS test_single_column")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_single_column: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_single_column: " + str(e))
    try:
        test_write_cr_in_field()
        passed += 1
        print("  PASS test_write_cr_in_field")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_write_cr_in_field: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_write_cr_in_field: " + str(e))
    try:
        test_parse_cr_in_quoted_normalized()
        passed += 1
        print("  PASS test_parse_cr_in_quoted_normalized")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_parse_cr_in_quoted_normalized: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_parse_cr_in_quoted_normalized: " + str(e))
    try:
        test_multiline_quoted_field()
        passed += 1
        print("  PASS test_multiline_quoted_field")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_multiline_quoted_field: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_multiline_quoted_field: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
