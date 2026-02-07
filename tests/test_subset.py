"""Pytest-based subset checker tests."""

from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tongues"))

from src.frontend.parse import parse
from src.frontend.subset import verify as verify_subset

SUBSET_DIR = Path(__file__).parent / "subset"


def parse_subset_file(path: Path) -> list[tuple[str, str, str]]:
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


def discover_subset_tests() -> list[tuple[str, str, str, str]]:
    """Find all subset tests, returns (test_id, input, expected, file_stem)."""
    results = []
    for test_file in sorted(SUBSET_DIR.glob("*.tests")):
        tests = parse_subset_file(test_file)
        for name, input_code, expected in tests:
            test_id = f"{test_file.stem}/{name}"
            results.append((test_id, input_code, expected, test_file.stem))
    return results


def pytest_generate_tests(metafunc):
    """Parametrize tests over subset test files."""
    if "subset_input" in metafunc.fixturenames:
        tests = discover_subset_tests()
        params = [
            pytest.param(input_code, expected, id=test_id)
            for test_id, input_code, expected, _ in tests
        ]
        metafunc.parametrize("subset_input,subset_expected", params)


def test_subset(subset_input: str, subset_expected: str):
    """Verify subset checker produces expected result."""
    try:
        ast_dict = parse(subset_input)
    except Exception as e:
        if subset_expected.startswith("error:"):
            # Parse error might be expected for some tests
            return
        pytest.fail(f"Parse error: {e}")

    result = verify_subset(ast_dict)
    errors = result.errors()
    warnings = result.warnings()

    if subset_expected == "ok":
        if errors:
            pytest.fail(f"Expected ok, got error: {errors[0].message}")
    elif subset_expected.startswith("error:"):
        expected_msg = subset_expected[6:].strip()
        if not errors:
            pytest.fail(f"Expected error containing '{expected_msg}', got ok")
        # Check if any error message contains the expected substring
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
    elif subset_expected.startswith("warning:"):
        expected_msg = subset_expected[8:].strip()
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
        pytest.fail(f"Unknown expected format: {subset_expected}")
