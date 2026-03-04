"""INI parser/writer tests — sections, keys, comments, roundtrips, and errors."""

import sys

from lib.ini import IniError
from lib.ini import ini_get
from lib.ini import ini_parse
from lib.ini import ini_write


# -- Basic parsing --


def test_parse_empty() -> None:
    assert ini_parse("") == []


def test_parse_single_section() -> None:
    result: list[list[str]] = ini_parse("[section]\nkey = value\n")
    assert len(result) == 1
    assert result[0][0] == "section"
    assert result[0][1] == "key"
    assert result[0][2] == "value"


def test_parse_multiple_sections() -> None:
    text: str = "[a]\nx = 1\n[b]\ny = 2\n"
    result: list[list[str]] = ini_parse(text)
    assert len(result) == 2
    assert result[0][0] == "a"
    assert result[1][0] == "b"


def test_parse_multiple_keys() -> None:
    text: str = "[db]\nhost = localhost\nport = 5432\nname = mydb\n"
    result: list[list[str]] = ini_parse(text)
    assert len(result) == 1
    assert ini_get(result, "db", "host") == "localhost"
    assert ini_get(result, "db", "port") == "5432"
    assert ini_get(result, "db", "name") == "mydb"


# -- Default section --


def test_default_section() -> None:
    """Keys before any [section] go into section named ''."""
    result: list[list[str]] = ini_parse("key = value\n")
    assert len(result) == 1
    assert result[0][0] == ""
    assert ini_get(result, "", "key") == "value"


def test_default_and_named() -> None:
    text: str = "global = yes\n[local]\nfoo = bar\n"
    result: list[list[str]] = ini_parse(text)
    assert len(result) == 2
    assert ini_get(result, "", "global") == "yes"
    assert ini_get(result, "local", "foo") == "bar"


# -- Comments --


def test_hash_comment() -> None:
    text: str = "# comment\n[s]\nk = v\n"
    result: list[list[str]] = ini_parse(text)
    assert len(result) == 1
    assert ini_get(result, "s", "k") == "v"


def test_semicolon_comment() -> None:
    text: str = "; comment\n[s]\nk = v\n"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "k") == "v"


def test_comment_between_keys() -> None:
    text: str = "[s]\na = 1\n# skip\nb = 2\n"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "a") == "1"
    assert ini_get(result, "s", "b") == "2"


def test_only_comments() -> None:
    assert ini_parse("# nothing\n; here\n") == []


# -- Whitespace --


def test_trim_key_value() -> None:
    text: str = "[s]\n  key  =  value  \n"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "key") == "value"


def test_trim_section_name() -> None:
    text: str = "[  spaced  ]\nk = v\n"
    result: list[list[str]] = ini_parse(text)
    assert result[0][0] == "spaced"


def test_blank_lines_skipped() -> None:
    text: str = "\n\n[s]\n\nk = v\n\n"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "k") == "v"


def test_tabs_in_whitespace() -> None:
    text: str = "[s]\n\tkey\t=\tvalue\t\n"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "key") == "value"


# -- Empty values --


def test_empty_value() -> None:
    text: str = "[s]\nkey =\n"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "key") == ""


def test_empty_value_trimmed() -> None:
    text: str = "[s]\nkey =   \n"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "key") == ""


# -- Duplicate handling --


def test_duplicate_key_last_wins() -> None:
    text: str = "[s]\nk = first\nk = second\n"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "k") == "second"


def test_duplicate_section_merges() -> None:
    text: str = "[s]\na = 1\n[s]\nb = 2\n"
    result: list[list[str]] = ini_parse(text)
    assert len(result) == 1
    assert ini_get(result, "s", "a") == "1"
    assert ini_get(result, "s", "b") == "2"


def test_duplicate_section_key_last_wins() -> None:
    text: str = "[s]\nk = first\n[s]\nk = second\n"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "k") == "second"


# -- CRLF --


def test_crlf_line_endings() -> None:
    text: str = "[s]\r\nk = v\r\n"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "k") == "v"


def test_cr_only_line_endings() -> None:
    text: str = "[s]\rk = v\r"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "k") == "v"


# -- No trailing newline --


def test_no_trailing_newline() -> None:
    text: str = "[s]\nk = v"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "k") == "v"


# -- ini_get --


def test_get_missing_section() -> None:
    result: list[list[str]] = ini_parse("[s]\nk = v\n")
    assert ini_get(result, "other", "k") == ""


def test_get_missing_key() -> None:
    result: list[list[str]] = ini_parse("[s]\nk = v\n")
    assert ini_get(result, "s", "other") == ""


# -- Error cases --


def test_unclosed_section() -> None:
    try:
        ini_parse("[bad\nk = v\n")
        assert False, "expected IniError"
    except IniError:
        pass


def test_missing_equals() -> None:
    try:
        ini_parse("[s]\nnoequals\n")
        assert False, "expected IniError"
    except IniError:
        pass


# -- Writer --


def test_write_empty() -> None:
    assert ini_write([]) == ""


def test_write_single_section() -> None:
    sections: list[list[str]] = [["db", "host", "localhost", "port", "5432"]]
    result: str = ini_write(sections)
    assert result == "[db]\nhost = localhost\nport = 5432\n"


def test_write_multiple_sections() -> None:
    sections: list[list[str]] = [["a", "x", "1"], ["b", "y", "2"]]
    result: str = ini_write(sections)
    assert result == "[a]\nx = 1\n\n[b]\ny = 2\n"


def test_write_default_section() -> None:
    sections: list[list[str]] = [["", "key", "val"], ["s", "k", "v"]]
    result: str = ini_write(sections)
    assert result == "key = val\n\n[s]\nk = v\n"


# -- Roundtrip --


def test_roundtrip_simple() -> None:
    text: str = "[db]\nhost = localhost\nport = 5432\n"
    assert ini_write(ini_parse(text)) == text


def test_roundtrip_multiple() -> None:
    text: str = "[a]\nx = 1\n\n[b]\ny = 2\n"
    assert ini_write(ini_parse(text)) == text


def test_roundtrip_default_and_named() -> None:
    text: str = "global = yes\n\n[local]\nfoo = bar\n"
    assert ini_write(ini_parse(text)) == text


# -- Realistic --


def test_realistic_config() -> None:
    text: str = (
        "# Database config\n"
        "[database]\n"
        "host = 127.0.0.1\n"
        "port = 3306\n"
        "name = production\n"
        "\n"
        "; Server settings\n"
        "[server]\n"
        "bind = 0.0.0.0\n"
        "workers = 4\n"
    )
    result: list[list[str]] = ini_parse(text)
    assert len(result) == 2
    assert ini_get(result, "database", "host") == "127.0.0.1"
    assert ini_get(result, "database", "port") == "3306"
    assert ini_get(result, "server", "workers") == "4"


def test_section_no_keys() -> None:
    text: str = "[empty]\n[full]\nk = v\n"
    result: list[list[str]] = ini_parse(text)
    assert len(result) == 2
    assert result[0][0] == "empty"
    assert len(result[0]) == 1
    assert ini_get(result, "full", "k") == "v"


def test_value_with_equals() -> None:
    """Value containing = is preserved (only first = is the delimiter)."""
    text: str = "[s]\nk = a=b=c\n"
    result: list[list[str]] = ini_parse(text)
    assert ini_get(result, "s", "k") == "a=b=c"


def main() -> int:
    passed: int = 0
    failed: int = 0
    tests = [
        ("test_parse_empty", test_parse_empty),
        ("test_parse_single_section", test_parse_single_section),
        ("test_parse_multiple_sections", test_parse_multiple_sections),
        ("test_parse_multiple_keys", test_parse_multiple_keys),
        ("test_default_section", test_default_section),
        ("test_default_and_named", test_default_and_named),
        ("test_hash_comment", test_hash_comment),
        ("test_semicolon_comment", test_semicolon_comment),
        ("test_comment_between_keys", test_comment_between_keys),
        ("test_only_comments", test_only_comments),
        ("test_trim_key_value", test_trim_key_value),
        ("test_trim_section_name", test_trim_section_name),
        ("test_blank_lines_skipped", test_blank_lines_skipped),
        ("test_tabs_in_whitespace", test_tabs_in_whitespace),
        ("test_empty_value", test_empty_value),
        ("test_empty_value_trimmed", test_empty_value_trimmed),
        ("test_duplicate_key_last_wins", test_duplicate_key_last_wins),
        ("test_duplicate_section_merges", test_duplicate_section_merges),
        ("test_duplicate_section_key_last_wins", test_duplicate_section_key_last_wins),
        ("test_crlf_line_endings", test_crlf_line_endings),
        ("test_cr_only_line_endings", test_cr_only_line_endings),
        ("test_no_trailing_newline", test_no_trailing_newline),
        ("test_get_missing_section", test_get_missing_section),
        ("test_get_missing_key", test_get_missing_key),
        ("test_unclosed_section", test_unclosed_section),
        ("test_missing_equals", test_missing_equals),
        ("test_write_empty", test_write_empty),
        ("test_write_single_section", test_write_single_section),
        ("test_write_multiple_sections", test_write_multiple_sections),
        ("test_write_default_section", test_write_default_section),
        ("test_roundtrip_simple", test_roundtrip_simple),
        ("test_roundtrip_multiple", test_roundtrip_multiple),
        ("test_roundtrip_default_and_named", test_roundtrip_default_and_named),
        ("test_realistic_config", test_realistic_config),
        ("test_section_no_keys", test_section_no_keys),
        ("test_value_with_equals", test_value_with_equals),
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
