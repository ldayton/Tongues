"""Test runner for Taytsh IR language"""

import signal
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src import check as taytsh_check, parse as taytsh_parse
from src.runtime import run as taytsh_run

PARSE_TIMEOUT = 5
TESTS_DIR = Path(__file__).parent

TESTS = {
    "taytsh_parse": {"dir": "parser", "run": "phase"},
    "taytsh_check": {"dir": "checker", "run": "phase"},
    "taytsh_app": {"dir": "apps", "run": "taytsh_app"},
}


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def _timeout_handler(signum, frame):
    raise TimeoutError("parse() timed out")


signal.signal(signal.SIGALRM, _timeout_handler)


# ---------------------------------------------------------------------------
# Spec file parsing
# ---------------------------------------------------------------------------


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


def discover_taytsh_apps(test_dir: Path) -> list[Path]:
    """Find all .ty files in a directory."""
    return sorted(test_dir.glob("*.ty"))


# ---------------------------------------------------------------------------
# Phase result + assertion checker
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict | None = None


def resolve_dotpath(obj: object, path: str) -> object:
    """Resolve a dot-separated path against a nested dict/list structure."""
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
# Phase runners
# ---------------------------------------------------------------------------


def run_taytsh_parse(source: str) -> PhaseResult:
    try:
        signal.alarm(PARSE_TIMEOUT)
        module = taytsh_parse(source)
        return PhaseResult(
            data={
                "strict_math": module.strict_math,
                "strict_tostring": module.strict_tostring,
            }
        )
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    finally:
        signal.alarm(0)


def run_taytsh_check(source: str) -> PhaseResult:
    try:
        signal.alarm(PARSE_TIMEOUT)
        errors = taytsh_check(source)
        if errors:
            return PhaseResult(errors=[str(e) for e in errors])
        return PhaseResult()
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    finally:
        signal.alarm(0)


RUNNERS = {
    "taytsh_parse": run_taytsh_parse,
    "taytsh_check": run_taytsh_check,
}


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    for name, cfg in TESTS.items():
        test_dir = TESTS_DIR / cfg["dir"]
        run = cfg["run"]
        if run == "phase":
            fixture = f"{name}_input"
            if fixture in metafunc.fixturenames:
                specs = discover_specs(test_dir)
                params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in specs]
                metafunc.parametrize(f"{fixture},{name}_expected", params)
        elif run == "taytsh_app" and "taytsh_app" in metafunc.fixturenames:
            apps = discover_taytsh_apps(test_dir)
            params = [pytest.param(p, id=p.stem) for p in apps]
            metafunc.parametrize("taytsh_app", params)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_taytsh_parse(taytsh_parse_input, taytsh_parse_expected):
    check_expected(
        taytsh_parse_expected,
        run_taytsh_parse(taytsh_parse_input),
        "taytsh_parse",
        lenient_errors=True,
    )


def test_taytsh_check(taytsh_check_input, taytsh_check_expected):
    check_expected(
        taytsh_check_expected,
        run_taytsh_check(taytsh_check_input),
        "taytsh_check",
        lenient_errors=True,
    )


def test_taytsh_app(taytsh_app: Path):
    """Parse and run a .ty program in-process. Exit code 0 = pass."""
    source = taytsh_app.read_text()
    module = taytsh_parse(source)
    result = taytsh_run(module)
    if result.exit_code != 0:
        output = (result.stdout + result.stderr).decode(errors="replace").strip()
        pytest.fail(f"Exit code {result.exit_code}:\n{output}")
