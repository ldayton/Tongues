"""Pytest-based name resolution tests."""

from pathlib import Path

import pytest

from src.frontend.parse import parse
from src.frontend.names import resolve_names

NAMES_DIR = Path(__file__).parent / "04_names"


def parse_names_file(path: Path) -> list[tuple[str, str, str]]:
    """Parse .tests file into (name, input, expected) tuples.

    Expected is one of: 'ok', 'error: <message>', 'warning: <message>'
    """
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


def discover_names_tests() -> list[tuple[str, str, str, str]]:
    """Find all names tests, returns (test_id, input, expected, file_stem)."""
    results = []
    for test_file in sorted(NAMES_DIR.glob("*.tests")):
        tests = parse_names_file(test_file)
        for name, input_code, expected in tests:
            test_id = f"{test_file.stem}/{name}"
            results.append((test_id, input_code, expected, test_file.stem))
    return results


def pytest_generate_tests(metafunc):
    """Parametrize tests over names test files."""
    if "names_input" in metafunc.fixturenames:
        tests = discover_names_tests()
        params = [
            pytest.param(input_code, expected, id=test_id)
            for test_id, input_code, expected, _ in tests
        ]
        metafunc.parametrize("names_input,names_expected", params)


def test_names(names_input: str, names_expected: str):
    """Verify name resolution produces expected result."""
    try:
        ast_dict = parse(names_input)
    except Exception as e:
        if names_expected.startswith("error:"):
            return
        pytest.fail(f"Parse error: {e}")

    result = resolve_names(ast_dict)
    errors = result.errors()
    warnings = result.warnings()

    if names_expected == "ok":
        if errors:
            pytest.fail(f"Expected ok, got error: {errors[0].message}")
    elif names_expected.startswith("error:"):
        expected_msg = names_expected[6:].strip()
        if not errors:
            pytest.fail(f"Expected error containing '{expected_msg}', got ok")
        found = False
        for err in errors:
            if expected_msg.lower() in err.message.lower():
                found = True
                break
        if not found:
            actual_msgs = [e.message for e in errors]
            pytest.fail(
                f"Expected error containing '{expected_msg}', got: {actual_msgs}"
            )
    elif names_expected.startswith("warning:"):
        expected_msg = names_expected[8:].strip()
        if not warnings:
            pytest.fail(f"Expected warning containing '{expected_msg}', got none")
        found = False
        for warn in warnings:
            if expected_msg.lower() in warn.message.lower():
                found = True
                break
        if not found:
            actual_msgs = [w.message for w in warnings]
            pytest.fail(
                f"Expected warning containing '{expected_msg}', got: {actual_msgs}"
            )
    else:
        pytest.fail(f"Unknown expected format: {names_expected}")
