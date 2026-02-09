"""Unified test runner for all Tongues test phases."""

import ast
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.frontend import Frontend, compile, ParseError
from src.frontend.hierarchy import build_hierarchy
from src.frontend.names import resolve_names
from src.frontend.parse import parse
from src.frontend.subset import verify as verify_subset
from src.serialize import fields_to_dict, hierarchy_to_dict, signatures_to_dict

from conftest import TARGETS, _SKIP_LANGS, _check_version

PARSE_TIMEOUT = 5
TESTS_DIR = Path(__file__).parent
TONGUES_DIR = TESTS_DIR.parent

# fmt: off
TESTS = {
    "frontend": {
        "cli":       {"dir": "01_cli",        "run": "cli"},
        "parse":     {"dir": "02_parse",      "run": "phase"},
        "subset":    {"dir": "03_subset",     "run": "phase"},
        "names":     {"dir": "04_names",      "run": "phase"},
        "sigs":      {"dir": "05_signatures", "run": "phase"},
        "fields":    {"dir": "06_fields",     "run": "phase"},
        "hierarchy": {"dir": "07_hierarchy",  "run": "phase"},
        "typecheck": {"dir": "08_inference",  "run": "phase"},
    },
    "backend": {
        "codegen":   {"dir": "15_codegen",    "run": "codegen"},
        "apptest":   {"dir": "15_app",        "run": "apptest"},
    },
}
# fmt: on


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _timeout_handler(signum, frame):
    raise TimeoutError("parse() timed out")


signal.signal(signal.SIGALRM, _timeout_handler)


def parse_spec_file(path: Path) -> list[tuple[str, str, str]]:
    """Parse a .tests file into (name, input, expected) tuples."""
    lines = path.read_text().split("\n")
    result: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("=== "):
            test_name = line[4:].strip()
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
            test_input = "\n".join(input_lines)
            expected = "\n".join(expected_lines).strip()
            result.append((test_name, test_input, expected))
        else:
            i += 1
    return result


def discover_specs(test_dir: Path) -> list[tuple[str, str, str]]:
    """Glob *.tests in test_dir, return (test_id, input, expected) tuples."""
    results = []
    for test_file in sorted(test_dir.glob("*.tests")):
        for name, input_code, expected in parse_spec_file(test_file):
            results.append((f"{test_file.stem}/{name}", input_code, expected))
    return results


def parse_cli_test_file(path: Path) -> list[tuple[str, dict]]:
    """Parse a CLI .tests file into (name, spec) tuples."""
    lines = path.read_text().split("\n")
    result: list[tuple[str, dict]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("=== "):
            test_name = line[4:].strip()
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
            spec = _parse_cli_spec(input_lines, expected_lines)
            result.append((test_name, spec))
        else:
            i += 1
    return result


def _parse_cli_spec(input_lines: list[str], expected_lines: list[str]) -> dict:
    """Parse input + expected lines into a CLI test spec dict."""
    spec: dict = {"args": [], "stdin": None, "stdin_bytes": None, "assertions": []}
    body_start = 0
    if input_lines and input_lines[0].startswith("args:"):
        args_str = input_lines[0][5:].strip()
        spec["args"] = args_str.split() if args_str else []
        body_start = 1
    remaining = input_lines[body_start:]
    if remaining and remaining[0].startswith("stdin-bytes:"):
        hex_str = remaining[0][len("stdin-bytes:") :].strip()
        spec["stdin_bytes"] = bytes.fromhex(hex_str)
    else:
        spec["stdin"] = "\n".join(remaining)
    for line in expected_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("exit:"):
            spec["assertions"].append(("exit", int(line[5:].strip())))
        elif line.startswith("exit-not:"):
            spec["assertions"].append(("exit-not", int(line[9:].strip())))
        elif line.startswith("stderr:"):
            spec["assertions"].append(("stderr", line[7:].strip()))
        elif line.startswith("stderr-contains:"):
            spec["assertions"].append(("stderr-contains", line[16:].strip()))
        elif line.startswith("stderr-empty:"):
            spec["assertions"].append(("stderr-empty", None))
        elif line.startswith("stdout-contains:"):
            spec["assertions"].append(("stdout-contains", line[16:].strip()))
        elif line.startswith("stdout-empty:"):
            spec["assertions"].append(("stdout-empty", None))
    return spec


def discover_cli_tests(test_dir: Path) -> list[tuple[str, dict]]:
    """Find all CLI tests across .tests files."""
    results = []
    for test_file in sorted(test_dir.glob("*.tests")):
        for name, spec in parse_cli_test_file(test_file):
            results.append((f"{test_file.stem}/{name}", spec))
    return results


def parse_codegen_file(path: Path) -> list[tuple[str, str, dict[str, str]]]:
    """Parse .tests file into (name, input, {lang: expected}) tuples."""
    lines = path.read_text().split("\n")
    result: list[tuple[str, str, dict[str, str]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("=== "):
            test_name = line[4:].strip()
            i += 1
            input_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                input_lines.append(lines[i])
                i += 1
            expected_by_lang: dict[str, str] = {}
            while i < len(lines) and lines[i].startswith("--- "):
                lang = lines[i][4:].strip()
                i += 1
                expected_lines: list[str] = []
                while i < len(lines) and not lines[i].startswith("---"):
                    expected_lines.append(lines[i])
                    i += 1
                expected_by_lang[lang] = "\n".join(expected_lines)
            if i < len(lines) and lines[i] == "---":
                i += 1
            test_input = "\n".join(input_lines)
            result.append((test_name, test_input, expected_by_lang))
        else:
            i += 1
    return result


def discover_codegen_tests(test_dir: Path) -> list[tuple[str, str, str, str, bool]]:
    """Find all codegen tests, returns (test_id, input, lang, expected, has_explicit)."""
    results = []
    for test_file in sorted(test_dir.glob("*.tests")):
        tests = parse_codegen_file(test_file)
        for name, input_code, expected_by_lang in tests:
            langs = list(expected_by_lang.keys())
            if "python" not in langs:
                langs.append("python")
            for lang in langs:
                has_explicit = lang in expected_by_lang
                expected = expected_by_lang.get(lang, input_code)
                test_id = f"{test_file.stem}/{name}[{lang}]"
                results.append((test_id, input_code, lang, expected, has_explicit))
    return results


def discover_apptests(test_dir: Path) -> list[Path]:
    """Find all apptest_*.py files."""
    return sorted(test_dir.glob("apptest_*.py"))


def run_cli(spec: dict) -> subprocess.CompletedProcess[bytes]:
    """Run tongues CLI from a test spec."""
    cmd = [sys.executable, "-m", "src.tongues", *spec["args"]]
    if spec["stdin_bytes"] is not None:
        stdin_data = spec["stdin_bytes"]
    elif spec["stdin"] is not None:
        stdin_data = spec["stdin"].encode()
    else:
        stdin_data = b""
    return subprocess.run(cmd, input=stdin_data, capture_output=True, cwd=TONGUES_DIR)


def check_cli_assertions(
    result: subprocess.CompletedProcess[bytes], assertions: list[tuple]
) -> None:
    """Check all assertions against a CLI result."""
    for kind, value in assertions:
        if kind == "exit":
            assert result.returncode == value, (
                f"expected exit {value}, got {result.returncode}"
                f"\nstderr: {result.stderr.decode(errors='replace')}"
            )
        elif kind == "exit-not":
            assert result.returncode != value, (
                f"expected exit != {value}, got {result.returncode}"
            )
        elif kind == "stderr":
            actual = result.stderr.decode(errors="replace").rstrip("\n")
            assert actual == value, f"expected stderr {value!r}, got {actual!r}"
        elif kind == "stderr-contains":
            actual = result.stderr.decode(errors="replace")
            assert value in actual, (
                f"expected stderr to contain {value!r}, got {actual!r}"
            )
        elif kind == "stderr-empty":
            assert result.stderr == b"", f"expected empty stderr, got {result.stderr!r}"
        elif kind == "stdout-contains":
            actual = result.stdout.decode(errors="replace")
            assert value in actual, (
                f"expected stdout to contain {value!r}, got {actual!r}"
            )
        elif kind == "stdout-empty":
            assert result.stdout == b"", (
                f"expected empty stdout, got {result.stdout[:200]!r}"
            )


def resolve_dotpath(obj: object, path: str) -> object:
    """Resolve a dot-separated path against a nested dict/list structure.

    Supports dict keys (including composite keys like 'Foo.bar'),
    integer list indices, and '.length' for len().
    """
    parts = path.split(".")
    current = obj
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "length":
            return len(current)
        if isinstance(current, list):
            current = current[int(part)]
            i += 1
        elif isinstance(current, dict):
            if part in current:
                current = current[part]
                i += 1
            else:
                found = False
                for j in range(i + 1, len(parts)):
                    composite = ".".join(parts[i : j + 1])
                    if composite in current:
                        current = current[composite]
                        i = j + 1
                        found = True
                        break
                if not found:
                    raise KeyError(part)
        else:
            raise KeyError(
                f"cannot traverse {type(current).__name__} with key {part!r}"
            )
    return current


def to_comparable(value: object) -> str:
    """Convert a value to its string form for comparison."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Phase result + runners
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict | None = None


def run_parse(source: str) -> PhaseResult:
    try:
        signal.alarm(PARSE_TIMEOUT)
        parse(source)
        return PhaseResult()
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    finally:
        signal.alarm(0)


def run_subset(source: str) -> PhaseResult:
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    result = verify_subset(ast_dict)
    return PhaseResult(
        errors=[e.message for e in result.errors()],
        warnings=[w.message for w in result.warnings()],
    )


def run_names(source: str) -> PhaseResult:
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    result = resolve_names(ast_dict)
    return PhaseResult(
        errors=[e.message for e in result.errors()],
        warnings=[w.message for w in result.warnings],
    )


def _run_frontend_through_sigs(source: str) -> tuple[Frontend, dict]:
    """Shared pipeline: parse → names → Frontend → sigs. Raises on error."""
    ast_dict = parse(source)
    name_result = resolve_names(ast_dict)
    errors = name_result.errors()
    if errors:
        raise RuntimeError(errors[0].message)
    fe = Frontend()
    fe.init_from_names(source, name_result)
    fe.collect_sigs(ast_dict)
    return fe, ast_dict


def run_sigs(source: str) -> PhaseResult:
    try:
        fe, _ = _run_frontend_through_sigs(source)
        return PhaseResult(data=signatures_to_dict(fe.symbols))
    except Exception as e:
        return PhaseResult(errors=[str(e)])


def run_fields(source: str) -> PhaseResult:
    try:
        fe, ast_dict = _run_frontend_through_sigs(source)
        fe.collect_flds(ast_dict)
        result = fields_to_dict(fe.symbols)
        for sname, struct in fe.symbols.structs.items():
            entry = result["classes"][sname]
            entry["param_to_field"] = dict(struct.param_to_field)
            entry["const_fields"] = dict(struct.const_fields)
            entry["needs_constructor"] = struct.needs_constructor
        return PhaseResult(data=result)
    except Exception as e:
        return PhaseResult(errors=[str(e)])


def run_hierarchy(source: str) -> PhaseResult:
    try:
        fe, ast_dict = _run_frontend_through_sigs(source)
        fe.collect_flds(ast_dict)
        rel = build_hierarchy(fe.symbols)
        result = hierarchy_to_dict(fe.symbols, rel.hierarchy_root)
        result["node_types"] = sorted(rel.node_types)
        result["exception_types"] = sorted(rel.exception_types)
        return PhaseResult(data=result)
    except Exception as e:
        return PhaseResult(errors=[str(e)])


def run_typecheck(source: str) -> PhaseResult:
    try:
        compile(source)
        return PhaseResult()
    except Exception as e:
        return PhaseResult(errors=[str(e)])


RUNNERS = {
    "parse": run_parse,
    "subset": run_subset,
    "names": run_names,
    "sigs": run_sigs,
    "fields": run_fields,
    "hierarchy": run_hierarchy,
    "typecheck": run_typecheck,
}


# ---------------------------------------------------------------------------
# Unified assertion checker
# ---------------------------------------------------------------------------


def check_expected(
    expected: str, result: PhaseResult, phase: str, *, lenient_errors: bool = False
) -> None:
    if expected == "ok":
        if result.errors:
            pytest.fail(f"Expected ok, got error: {result.errors[0]}")
        return
    if expected.startswith("error:"):
        expected_msg = expected[6:].strip()
        if not result.errors:
            pytest.fail(f"Expected error containing '{expected_msg}', got ok")
        if not lenient_errors and expected_msg:
            found = any(expected_msg.lower() in e.lower() for e in result.errors)
            if not found:
                pytest.fail(
                    f"Expected error containing '{expected_msg}', got: {result.errors}"
                )
        return
    if expected.startswith("warning:"):
        expected_msg = expected[8:].strip()
        if not result.warnings:
            pytest.fail(f"Expected warning containing '{expected_msg}', got none")
        found = any(expected_msg.lower() in w.lower() for w in result.warnings)
        if not found:
            pytest.fail(
                f"Expected warning containing '{expected_msg}', got: {result.warnings}"
            )
        return
    # Dotpath assertions
    if result.errors:
        pytest.fail(f"{phase} failed: {result.errors[0]}")
    assert result.data is not None, f"No data returned from {phase}"
    for line in expected.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            pytest.fail(f"Bad assertion (no '='): {line}")
        path, expected_val = line.split("=", 1)
        path = path.strip()
        expected_val = expected_val.strip()
        try:
            actual = resolve_dotpath(result.data, path)
        except (KeyError, IndexError, TypeError) as e:
            pytest.fail(f"Path '{path}' not found in result: {e}")
        actual_str = to_comparable(actual)
        if actual_str != expected_val:
            pytest.fail(
                f"Assertion failed: {path}\n"
                f"  expected: {expected_val!r}\n"
                f"  actual:   {actual_str!r}"
            )


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    ignore_version = metafunc.config.getoption("ignore_version")
    ignore_skips = metafunc.config.getoption("ignore_skips")
    target_filter = metafunc.config.getoption("target")
    for section in TESTS.values():
        for name, cfg in section.items():
            test_dir = TESTS_DIR / cfg["dir"]
            run = cfg["run"]
            if run == "cli" and "cli_spec" in metafunc.fixturenames:
                tests = discover_cli_tests(test_dir)
                params = [pytest.param(spec, id=tid) for tid, spec in tests]
                metafunc.parametrize("cli_spec", params)
            elif run == "phase":
                fixture = f"{name}_input"
                if fixture in metafunc.fixturenames:
                    specs = discover_specs(test_dir)
                    params = [
                        pytest.param(inp, exp, id=tid) for tid, inp, exp in specs
                    ]
                    metafunc.parametrize(f"{fixture},{name}_expected", params)
            elif run == "codegen" and "codegen_input" in metafunc.fixturenames:
                tests = discover_codegen_tests(test_dir)
                if target_filter:
                    tests = [t for t in tests if t[2] in target_filter]
                params = [
                    pytest.param(inp, lang, exp, has_exp, id=tid)
                    for tid, inp, lang, exp, has_exp in tests
                ]
                metafunc.parametrize(
                    "codegen_input,codegen_lang,codegen_expected,codegen_has_explicit",
                    params,
                )
            elif run == "apptest" and "apptest" in metafunc.fixturenames:
                apptests = discover_apptests(test_dir)
                targets = (
                    [TARGETS[t] for t in target_filter if t in TARGETS]
                    if target_filter
                    else list(TARGETS.values())
                )
                params = []
                for apptest in apptests:
                    for target in targets:
                        test_id = f"{target.name}/{apptest.stem}"
                        version_ok, version_msg = _check_version(target.name)
                        if not version_ok and not ignore_version:
                            params.append(
                                pytest.param(
                                    apptest,
                                    target,
                                    id=test_id,
                                    marks=pytest.mark.skip(
                                        reason=f"wrong version: {version_msg}"
                                    ),
                                )
                            )
                        elif not ignore_skips and target.name in _SKIP_LANGS.get(
                            apptest.stem, set()
                        ):
                            params.append(
                                pytest.param(
                                    apptest,
                                    target,
                                    id=test_id,
                                    marks=pytest.mark.skip(reason="known failure"),
                                )
                            )
                        else:
                            params.append(pytest.param(apptest, target, id=test_id))
                metafunc.parametrize("apptest,target", params)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_cli(cli_spec: dict) -> None:
    """Run a single CLI test case from .tests file."""
    result = run_cli(cli_spec)
    check_cli_assertions(result, cli_spec["assertions"])


def test_parse(parse_input, parse_expected):
    check_expected(parse_expected, run_parse(parse_input), "parse", lenient_errors=True)


def test_subset(subset_input, subset_expected):
    check_expected(subset_expected, run_subset(subset_input), "subset")


def test_names(names_input, names_expected):
    check_expected(names_expected, run_names(names_input), "names")


def test_sigs(sigs_input, sigs_expected):
    check_expected(sigs_expected, run_sigs(sigs_input), "sigs")


def test_fields(fields_input, fields_expected):
    check_expected(fields_expected, run_fields(fields_input), "fields")


def test_hierarchy(hierarchy_input, hierarchy_expected):
    check_expected(hierarchy_expected, run_hierarchy(hierarchy_input), "hierarchy")


def test_typecheck(typecheck_input, typecheck_expected):
    check_expected(typecheck_expected, run_typecheck(typecheck_input), "typecheck")


# ---------------------------------------------------------------------------
# Codegen tests
# ---------------------------------------------------------------------------


def contains_normalized(haystack: str, needle: str) -> bool:
    """Check if needle appears in haystack, normalizing line-by-line whitespace."""
    needle_lines = [line.strip() for line in needle.strip().split("\n") if line.strip()]
    haystack_lines = [line.strip() for line in haystack.split("\n") if line.strip()]
    if not needle_lines:
        return True
    for i in range(len(haystack_lines)):
        if haystack_lines[i] == needle_lines[0]:
            match = True
            for j in range(1, len(needle_lines)):
                if (
                    i + j >= len(haystack_lines)
                    or haystack_lines[i + j] != needle_lines[j]
                ):
                    match = False
                    break
            if match:
                return True
    return False


def is_expression(code: str) -> bool:
    """Check if code is a valid Python expression (not a statement)."""
    try:
        ast.parse(code, mode="eval")
        return True
    except SyntaxError:
        return False


def test_codegen(
    codegen_input: str,
    codegen_lang: str,
    codegen_expected: str,
    codegen_has_explicit: bool,
    transpiled_output: str,
):
    """Verify transpiler output matches expected code."""
    if not contains_normalized(transpiled_output, codegen_expected):
        pytest.fail(
            f"Expected not found in output:\n--- expected ---\n{codegen_expected}\n--- got ---\n{transpiled_output}"
        )
    if codegen_lang == "python" and codegen_has_explicit:
        input_stripped = codegen_input.strip()
        output_stripped = transpiled_output.strip()
        if is_expression(input_stripped) and is_expression(output_stripped):
            input_result = eval(input_stripped)
            output_result = eval(output_stripped)
            if input_result != output_result:
                pytest.fail(
                    f"Semantic mismatch:\n  input  {input_stripped!r} = {input_result!r}\n  output {output_stripped!r} = {output_result!r}"
                )


# ---------------------------------------------------------------------------
# App tests
# ---------------------------------------------------------------------------


def test_apptest(apptest: Path, target, executable: list[str]):
    """Run a transpiled apptest and verify it executes successfully."""
    try:
        result = subprocess.run(executable, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        pytest.fail("Test timed out after 10 seconds")
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        pytest.fail(f"Test failed with exit code {result.returncode}:\n{output}")
