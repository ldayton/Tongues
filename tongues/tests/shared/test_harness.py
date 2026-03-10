"""Shared test harness: .tests file parsing, dotpath resolution, assertion checking.

Tongues-subset module — transpiles to Ruby/Perl via the compiler.
All functions are pure: no file I/O, no subprocess, no regex.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from lib.json import (
    JsonNull,
    JsonBool,
    JsonNumber,
    JsonString,
    JsonArray,
    JsonObject,
    json_parse,
    json_get_string,
    json_get_number,
    json_get_items,
    json_get_field,
    json_is_null,
    json_stringify,
)


# -- Dataclasses --


@dataclass
class SpecEntry:
    name: str
    input: str
    expected: str


@dataclass
class SimpleEntry:
    name: str
    content: str


@dataclass
class CliAssertion:
    kind: str
    value: str


@dataclass
class CliSpec:
    args: list[str]
    stdin: str
    stdin_hex: str
    assertions: list[CliAssertion]


@dataclass
class LinkerFile:
    path: str
    source: str


@dataclass
class LinkerSpec:
    files: list[LinkerFile]
    args: list[str]
    assertions: list[CliAssertion]


@dataclass
class RevealAssertion:
    lineno: int
    expected_type: str


@dataclass
class AnnotationAssertion:
    lineno: int
    key: str
    expected_value: str


# -- Helpers --


def _trim_blank_lines(text: str) -> str:
    """Trim leading/trailing blank lines without touching inner whitespace."""
    lines: list[str] = text.split("\n")
    start: int = 0
    while start < len(lines) and lines[start] == "":
        start += 1
    end: int = len(lines)
    while end > start and lines[end - 1] == "":
        end -= 1
    return "\n".join(lines[start:end])


# -- .tests file parsing --


def parse_spec_file(text: str) -> list[SpecEntry]:
    """Parse a .tests file into SpecEntry list. Takes file contents, not a path."""
    lines: list[str] = text.split("\n")
    result: list[SpecEntry] = []
    i: int = 0
    while i < len(lines):
        if lines[i].startswith("=== "):
            test_name: str = lines[i][4:].strip()
            i += 1
            input_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                input_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            expected_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                expected_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            result.append(
                SpecEntry(
                    test_name,
                    "\n".join(input_lines),
                    _trim_blank_lines("\n".join(expected_lines)),
                )
            )
        else:
            i += 1
    return result


def _parse_cli_assertions(expected_lines: list[str]) -> list[CliAssertion]:
    """Parse assertion lines into CliAssertion list."""
    assertions: list[CliAssertion] = []
    for raw_line in expected_lines:
        stripped: str = raw_line.strip()
        if stripped == "":
            continue
        if stripped.startswith("exit:"):
            assertions.append(CliAssertion("exit", stripped[5:].strip()))
        elif stripped.startswith("exit-not:"):
            assertions.append(CliAssertion("exit-not", stripped[9:].strip()))
        elif stripped.startswith("stderr:"):
            assertions.append(CliAssertion("stderr", stripped[7:].strip()))
        elif stripped.startswith("stderr-contains:"):
            assertions.append(CliAssertion("stderr-contains", stripped[16:].strip()))
        elif stripped.startswith("stderr-empty:"):
            assertions.append(CliAssertion("stderr-empty", ""))
        elif stripped.startswith("stdout-contains:"):
            assertions.append(CliAssertion("stdout-contains", stripped[16:].strip()))
        elif stripped.startswith("stdout-empty:"):
            assertions.append(CliAssertion("stdout-empty", ""))
    return assertions


def parse_cli_test_file(text: str) -> list[tuple[str, CliSpec]]:
    """Parse a CLI .tests file. Takes file contents, not a path."""
    lines: list[str] = text.split("\n")
    result: list[tuple[str, CliSpec]] = []
    i: int = 0
    while i < len(lines):
        if lines[i].startswith("=== "):
            test_name: str = lines[i][4:].strip()
            i += 1
            input_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                input_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            expected_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                expected_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            spec: CliSpec = _parse_cli_spec(input_lines, expected_lines)
            result.append((test_name, spec))
        else:
            i += 1
    return result


def _parse_cli_spec(input_lines: list[str], expected_lines: list[str]) -> CliSpec:
    """Parse input + expected lines into a CliSpec."""
    args: list[str] = []
    stdin: str = ""
    stdin_hex: str = ""
    body_start: int = 0
    if len(input_lines) > 0 and input_lines[0].startswith("args:"):
        args_str: str = input_lines[0][5:].strip()
        if args_str != "":
            args = args_str.split()
        body_start = 1
    remaining: list[str] = input_lines[body_start:]
    if len(remaining) > 0 and remaining[0].startswith("stdin-bytes:"):
        stdin_hex = remaining[0][12:].strip()
    else:
        stdin = "\n".join(remaining)
    assertions: list[CliAssertion] = _parse_cli_assertions(expected_lines)
    return CliSpec(args, stdin, stdin_hex, assertions)


def parse_linker_test_file(text: str) -> list[tuple[str, LinkerSpec]]:
    """Parse a linker .tests file. Takes file contents, not a path."""
    lines: list[str] = text.split("\n")
    result: list[tuple[str, LinkerSpec]] = []
    i: int = 0
    while i < len(lines):
        if lines[i].startswith("=== "):
            test_name: str = lines[i][4:].strip()
            i += 1
            input_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                input_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            expected_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                expected_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            spec: LinkerSpec = _parse_linker_spec(input_lines, expected_lines)
            result.append((test_name, spec))
        else:
            i += 1
    return result


def _parse_linker_spec(input_lines: list[str], expected_lines: list[str]) -> LinkerSpec:
    """Parse input with file: directives + expected into a LinkerSpec."""
    files: list[LinkerFile] = []
    args: list[str] = []
    current_path: str = ""
    has_current: bool = False
    current_lines: list[str] = []
    for line in input_lines:
        if line.startswith("file: "):
            if has_current:
                files.append(LinkerFile(current_path, "\n".join(current_lines)))
            current_path = line[6:].strip()
            has_current = True
            current_lines = []
        elif line.startswith("args: "):
            if has_current:
                files.append(LinkerFile(current_path, "\n".join(current_lines)))
                has_current = False
                current_lines = []
            args = line[6:].strip().split()
        else:
            current_lines.append(line)
    if has_current:
        files.append(LinkerFile(current_path, "\n".join(current_lines)))
    assertions: list[CliAssertion] = _parse_cli_assertions(expected_lines)
    return LinkerSpec(files, args, assertions)


def parse_simple_tests(text: str) -> list[SimpleEntry]:
    """Parse '=== name' + content blocks. Takes file contents, not a path."""
    lines: list[str] = text.split("\n")
    result: list[SimpleEntry] = []
    i: int = 0
    while i < len(lines):
        if lines[i].startswith("=== "):
            name: str = lines[i][4:].strip()
            i += 1
            content_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("=== "):
                content_lines.append(lines[i])
                i += 1
            result.append(
                SimpleEntry(name, _trim_blank_lines("\n".join(content_lines)))
            )
        else:
            i += 1
    return result


# -- Dotpath resolution on JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject --


def resolve_dotpath(
    obj: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject,
    path: str,
) -> JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject:
    """Resolve a dot-separated path against a JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject tree."""
    parts: list[str] = path.split(".")
    current: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        obj
    )
    i: int = 0
    while i < len(parts):
        part: str = parts[i]
        if part == "length":
            if isinstance(current, JsonArray):
                return JsonNumber(float(len(current.items)))
            if isinstance(current, JsonString):
                return JsonNumber(float(len(current.value)))
            if isinstance(current, JsonObject):
                return JsonNumber(float(len(current.entries)))
            raise Exception("length on non-array/string/object")
        if isinstance(current, JsonArray):
            idx: int = int(part)
            current = current.items[idx]
            i += 1
        elif isinstance(current, JsonObject):
            found_key: bool = False
            for k, v in current.entries:
                if k == part:
                    current = v
                    found_key = True
                    break
            if found_key:
                i += 1
            else:
                found_composite: bool = False
                j: int = i + 1
                while j < len(parts):
                    composite_parts: list[str] = []
                    idx2: int = i
                    while idx2 <= j:
                        composite_parts.append(parts[idx2])
                        idx2 += 1
                    composite: str = ".".join(composite_parts)
                    for k2, v2 in current.entries:
                        if k2 == composite:
                            current = v2
                            i = j + 1
                            found_composite = True
                            break
                    if found_composite:
                        break
                    j += 1
                if not found_composite:
                    raise Exception("key not found: " + part)
        else:
            raise Exception("cannot traverse with key: " + part)
    return current


# -- Assertion checking --


def to_comparable(
    value: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject,
) -> str:
    """Convert a JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject to its string form for comparison."""
    if isinstance(value, JsonNull):
        return "null"
    if isinstance(value, JsonBool):
        if value.value:
            return "true"
        return "false"
    if isinstance(value, JsonNumber):
        n: float = value.value
        i: int = int(n)
        if float(i) == n:
            return str(i)
        return str(n)
    if isinstance(value, JsonString):
        return value.value
    return json_stringify(value)


def check_reveals(
    assertions: list[RevealAssertion],
    actuals: list[tuple[int, str]],
) -> str:
    """Check reveal_type assertions. Returns '' on pass, error message on failure."""
    for ra in assertions:
        found: bool = False
        for actual_line, actual_type in actuals:
            if actual_line == ra.lineno:
                if actual_type != ra.expected_type:
                    return (
                        "reveal_type at line "
                        + str(ra.lineno)
                        + ": expected '"
                        + ra.expected_type
                        + "', got '"
                        + actual_type
                        + "'"
                    )
                found = True
                break
        if not found:
            return "No reveal_type found at line " + str(ra.lineno)
    return ""


def check_annotations(
    assertions: list[AnnotationAssertion],
    actuals: dict[int, dict[str, str]],
) -> str:
    """Check annotation assertions. Returns '' on pass, error message on failure."""
    for aa in assertions:
        if aa.lineno not in actuals:
            return "No annotations found at line " + str(aa.lineno)
        line_anns: dict[str, str] = actuals[aa.lineno]
        if aa.key not in line_anns:
            keys: list[str] = []
            for k in line_anns:
                keys.append(k)
            return (
                "Annotation '"
                + aa.key
                + "' not found at line "
                + str(aa.lineno)
                + ", have: "
                + str(keys)
            )
        actual_val: str = line_anns[aa.key]
        if actual_val != aa.expected_value:
            return (
                "Annotation at line "
                + str(aa.lineno)
                + ": expected "
                + aa.key
                + "='"
                + aa.expected_value
                + "', got '"
                + actual_val
                + "'"
            )
    return ""


def check_expected(
    expected: str,
    errors: list[str],
    warnings: list[str],
    data: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject | None,
    reveals: list[tuple[int, str]],
    annotations: dict[int, dict[str, str]],
    phase: str,
    lenient_errors: bool,
) -> str:
    """Check phase result against expected. Returns '' on pass, error message on failure."""
    reveal_assertions: list[RevealAssertion] = []
    annotation_assertions: list[AnnotationAssertion] = []
    verdict_lines: list[str] = []
    for line in expected.split("\n"):
        stripped: str = line.strip()
        if stripped.startswith("reveal:"):
            rest: str = stripped[7:]
            eq_pos: int = rest.index("=")
            lineno: int = int(rest[:eq_pos].strip())
            expected_type: str = rest[eq_pos + 1 :].strip()
            reveal_assertions.append(RevealAssertion(lineno, expected_type))
        elif stripped.startswith("annotation:"):
            rest = stripped[11:]
            first_eq: int = rest.index("=")
            lineno = int(rest[:first_eq].strip())
            after_first: str = rest[first_eq + 1 :]
            second_eq_rel: int = after_first.index("=")
            key: str = after_first[:second_eq_rel].strip()
            value: str = after_first[second_eq_rel + 1 :].strip()
            annotation_assertions.append(AnnotationAssertion(lineno, key, value))
        else:
            verdict_lines.append(line)
    expected = _trim_blank_lines("\n".join(verdict_lines))
    if expected == "":
        expected = "ok"
    if expected == "ok":
        if len(errors) > 0:
            return "Expected ok, got error: " + errors[0]
        err: str = check_reveals(reveal_assertions, reveals)
        if err != "":
            return err
        err = check_annotations(annotation_assertions, annotations)
        if err != "":
            return err
        return ""
    if expected.startswith("error:"):
        expected_msg: str = expected[6:].strip()
        if len(errors) == 0:
            return "Expected error containing '" + expected_msg + "', got ok"
        if not lenient_errors and expected_msg != "":
            found: bool = False
            for e in errors:
                if expected_msg.lower() in e.lower():
                    found = True
                    break
            if not found:
                return (
                    "Expected error containing '"
                    + expected_msg
                    + "', got: "
                    + str(errors)
                )
        return ""
    if expected.startswith("warning:"):
        expected_msg = expected[8:].strip()
        if len(warnings) == 0:
            return "Expected warning containing '" + expected_msg + "', got none"
        found = False
        for w in warnings:
            if expected_msg.lower() in w.lower():
                found = True
                break
        if not found:
            return (
                "Expected warning containing '"
                + expected_msg
                + "', got: "
                + str(warnings)
            )
        return ""
    if len(errors) > 0:
        return phase + " failed: " + errors[0]
    if data is None:
        return "No data returned from " + phase
    for assert_line in expected.split("\n"):
        stripped_line: str = assert_line.strip()
        if stripped_line == "":
            continue
        if "=" not in stripped_line:
            return "Bad assertion (no '='): " + stripped_line
        eq_idx: int = stripped_line.index("=")
        path: str = stripped_line[:eq_idx].strip()
        expected_val: str = stripped_line[eq_idx + 1 :].strip()
        actual: (
            JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject
        ) = JsonNull()
        try:
            actual = resolve_dotpath(data, path)
        except Exception as exc:
            return "Path '" + path + "' not found in result: " + str(exc)
        actual_str: str = to_comparable(actual)
        if "." in expected_val and " " not in expected_val:
            try:
                ref_val: (
                    JsonNull
                    | JsonBool
                    | JsonNumber
                    | JsonString
                    | JsonArray
                    | JsonObject
                ) = resolve_dotpath(data, expected_val)
                expected_val = to_comparable(ref_val)
            except Exception:
                pass
        if actual_str != expected_val:
            return (
                "Assertion failed: "
                + path
                + "\n  expected: "
                + repr(expected_val)
                + "\n  actual:   "
                + repr(actual_str)
            )
    return ""


def contains_normalized(haystack: str, needle: str) -> bool:
    """Check if needle appears in haystack with line-by-line normalized matching."""
    needle_stripped: list[str] = []
    for line in needle.strip().split("\n"):
        s: str = line.strip()
        if s != "":
            needle_stripped.append(s)
    haystack_stripped: list[str] = []
    for line in haystack.split("\n"):
        s = line.strip()
        if s != "":
            haystack_stripped.append(s)
    if len(needle_stripped) == 0:
        return True
    i: int = 0
    while i < len(haystack_stripped):
        if needle_stripped[0] in haystack_stripped[i]:
            match: bool = True
            j: int = 1
            while j < len(needle_stripped):
                if (
                    i + j >= len(haystack_stripped)
                    or needle_stripped[j] not in haystack_stripped[i + j]
                ):
                    match = False
                    break
                j += 1
            if match:
                return True
        i += 1
    return False


def check_cli_assertions(
    exit_code: int, stdout: str, stderr: str, assertions: list[CliAssertion]
) -> str:
    """Check CLI assertions. Returns '' on pass, error message on failure."""
    for a in assertions:
        if a.kind == "exit":
            expected_exit: int = int(a.value)
            if exit_code != expected_exit:
                return (
                    "expected exit "
                    + a.value
                    + ", got "
                    + str(exit_code)
                    + "\nstderr: "
                    + stderr
                )
        elif a.kind == "exit-not":
            not_exit: int = int(a.value)
            if exit_code == not_exit:
                return "expected exit != " + a.value + ", got " + str(exit_code)
        elif a.kind == "stderr":
            actual_stderr: str = stderr.rstrip()
            if actual_stderr != a.value:
                return (
                    "expected stderr " + repr(a.value) + ", got " + repr(actual_stderr)
                )
        elif a.kind == "stderr-contains":
            if a.value not in stderr:
                return (
                    "expected stderr to contain "
                    + repr(a.value)
                    + ", got "
                    + repr(stderr)
                )
        elif a.kind == "stderr-empty":
            if stderr != "":
                return "expected empty stderr, got " + repr(stderr)
        elif a.kind == "stdout-contains":
            if a.value not in stdout:
                return (
                    "expected stdout to contain "
                    + repr(a.value)
                    + ", got "
                    + repr(stdout)
                )
        elif a.kind == "stdout-empty":
            if stdout != "":
                return "expected empty stdout, got " + repr(stdout[:200])
    return ""


def find_lib_imports(source: str) -> list[str]:
    """Extract unique lib module names from 'from lib.X import' statements.

    Uses string ops instead of regex for subset compliance.
    """
    seen: list[str] = []
    for line in source.split("\n"):
        if line.startswith("from lib."):
            rest: str = line[9:]
            space_idx: int = rest.index(" ")
            module_name: str = rest[:space_idx]
            already: bool = False
            for s in seen:
                if s == module_name:
                    already = True
                    break
            if not already:
                seen.append(module_name)
    return seen


def build_project_input(
    app_path: str, app_source: str, lib_sources: list[tuple[str, str]]
) -> str:
    """Build NUL-delimited project input string."""
    parts: list[str] = [app_path, app_source]
    for import_path, src in lib_sources:
        parts.append(import_path)
        parts.append(src)
    return "\0".join(parts)


def cli_needs_backend(
    args: list[str], assertions: list[CliAssertion], emitter_langs: list[str]
) -> bool:
    """Check if a CLI test needs a non-emitter backend (should be skipped)."""
    has_stop_at: bool = False
    for a in args:
        if a == "--stop-at":
            has_stop_at = True
            break
    if has_stop_at:
        return False
    expects_success: bool = False
    for a in assertions:
        if a.kind == "exit" and a.value == "0":
            expects_success = True
            break
    if not expects_success:
        return False
    has_target: bool = False
    target: str = ""
    i: int = 0
    while i < len(args):
        if args[i] == "--target" and i + 1 < len(args):
            has_target = True
            target = args[i + 1]
            break
        i += 1
    if not has_target:
        return False
    for lang in emitter_langs:
        if lang == target:
            return False
    return True


# -- Self-test --


def _self_test() -> int:
    passed: int = 0
    failed: int = 0

    # parse_spec_file
    spec_text: str = "=== test one\ninput line\n---\nexpected line\n---\n=== test two\ninput2\n---\nerror: bad\n---\n"
    specs: list[SpecEntry] = parse_spec_file(spec_text)
    if (
        len(specs) == 2
        and specs[0].name == "test one"
        and specs[0].input == "input line"
        and specs[0].expected == "expected line"
    ):
        print("  PASS parse_spec_file")
        passed += 1
    else:
        print("  FAIL parse_spec_file")
        failed += 1

    # parse_simple_tests
    simple_text: str = "=== alpha\ncontent a\n=== beta\ncontent b\n"
    simples: list[SimpleEntry] = parse_simple_tests(simple_text)
    if (
        len(simples) == 2
        and simples[0].name == "alpha"
        and simples[0].content == "content a"
    ):
        print("  PASS parse_simple_tests")
        passed += 1
    else:
        print("  FAIL parse_simple_tests")
        failed += 1

    # contains_normalized
    if contains_normalized("  foo bar  \n  baz  ", "foo bar\nbaz"):
        print("  PASS contains_normalized")
        passed += 1
    else:
        print("  FAIL contains_normalized")
        failed += 1

    if not contains_normalized("foo\nbar", "baz"):
        print("  PASS contains_normalized_negative")
        passed += 1
    else:
        print("  FAIL contains_normalized_negative")
        failed += 1

    # find_lib_imports
    imports: list[str] = find_lib_imports(
        "from lib.json import parse\nfrom lib.base64 import encode\nfrom lib.json import stringify\n"
    )
    if len(imports) == 2 and imports[0] == "json" and imports[1] == "base64":
        print("  PASS find_lib_imports")
        passed += 1
    else:
        print("  FAIL find_lib_imports: " + str(imports))
        failed += 1

    # build_project_input
    proj: str = build_project_input("main.py", "code", [("lib/json.py", "lib code")])
    if proj == "main.py\0code\0lib/json.py\0lib code":
        print("  PASS build_project_input")
        passed += 1
    else:
        print("  FAIL build_project_input")
        failed += 1

    # resolve_dotpath
    doc: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        json_parse('{"a":{"b":[10,20]},"x.y":3}')
    )
    r1: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        resolve_dotpath(doc, "a.b.1")
    )
    if isinstance(r1, JsonNumber) and int(r1.value) == 20:
        print("  PASS resolve_dotpath")
        passed += 1
    else:
        print("  FAIL resolve_dotpath")
        failed += 1

    # resolve_dotpath composite key
    r2: JsonNull | JsonBool | JsonNumber | JsonString | JsonArray | JsonObject = (
        resolve_dotpath(doc, "x.y")
    )
    if isinstance(r2, JsonNumber) and int(r2.value) == 3:
        print("  PASS resolve_dotpath_composite")
        passed += 1
    else:
        print("  FAIL resolve_dotpath_composite")
        failed += 1

    # to_comparable
    if (
        to_comparable(JsonNull()) == "null"
        and to_comparable(JsonBool(True)) == "true"
        and to_comparable(JsonNumber(42.0)) == "42"
    ):
        print("  PASS to_comparable")
        passed += 1
    else:
        print("  FAIL to_comparable")
        failed += 1

    # check_cli_assertions
    r3: str = check_cli_assertions(
        0,
        "hello",
        "",
        [CliAssertion("exit", "0"), CliAssertion("stdout-contains", "hello")],
    )
    if r3 == "":
        print("  PASS check_cli_assertions")
        passed += 1
    else:
        print("  FAIL check_cli_assertions: " + r3)
        failed += 1

    # cli_needs_backend
    if cli_needs_backend(
        ["--target", "c"], [CliAssertion("exit", "0")], ["python", "ruby", "perl"]
    ):
        print("  PASS cli_needs_backend")
        passed += 1
    else:
        print("  FAIL cli_needs_backend")
        failed += 1

    if not cli_needs_backend(
        ["--target", "ruby"], [CliAssertion("exit", "0")], ["python", "ruby", "perl"]
    ):
        print("  PASS cli_needs_backend_negative")
        passed += 1
    else:
        print("  FAIL cli_needs_backend_negative")
        failed += 1

    # parse_cli_test_file
    cli_text: str = (
        "=== basic\nargs: --help\n---\nexit: 0\nstdout-contains: usage\n---\n"
    )
    cli_tests: list[tuple[str, CliSpec]] = parse_cli_test_file(cli_text)
    if (
        len(cli_tests) == 1
        and cli_tests[0][0] == "basic"
        and len(cli_tests[0][1].args) == 1
        and cli_tests[0][1].args[0] == "--help"
    ):
        print("  PASS parse_cli_test_file")
        passed += 1
    else:
        print("  FAIL parse_cli_test_file")
        failed += 1

    # parse_linker_test_file
    linker_text: str = "=== link test\nfile: a.py\nprint(1)\nfile: b.py\nprint(2)\nargs: --project\n---\nexit: 0\n---\n"
    linker_tests: list[tuple[str, LinkerSpec]] = parse_linker_test_file(linker_text)
    if (
        len(linker_tests) == 1
        and len(linker_tests[0][1].files) == 2
        and linker_tests[0][1].files[0].path == "a.py"
    ):
        print("  PASS parse_linker_test_file")
        passed += 1
    else:
        print("  FAIL parse_linker_test_file")
        failed += 1

    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
