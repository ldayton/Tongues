"""Pytest-based type checking tests."""

from pathlib import Path

import pytest

from src.frontend import compile, ParseError

TYPECHECK_DIR = Path(__file__).parent / "09_inference"


def parse_typecheck_file(path: Path) -> list[tuple[str, str, str]]:
    """Parse .tests file into (name, input, expected) tuples.

    Expected is one of: 'ok', 'error: <message>'
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


def discover_typecheck_tests() -> list[tuple[str, str, str, str]]:
    """Find all typecheck tests, returns (test_id, input, expected, file_stem)."""
    results = []
    for test_file in sorted(TYPECHECK_DIR.glob("*.tests")):
        tests = parse_typecheck_file(test_file)
        for name, input_code, expected in tests:
            test_id = f"{test_file.stem}/{name}"
            results.append((test_id, input_code, expected, test_file.stem))
    return results


def pytest_generate_tests(metafunc):
    """Parametrize tests over typecheck test files."""
    if "typecheck_input" in metafunc.fixturenames:
        tests = discover_typecheck_tests()
        params = [
            pytest.param(input_code, expected, id=test_id)
            for test_id, input_code, expected, _ in tests
        ]
        metafunc.parametrize("typecheck_input,typecheck_expected", params)


def test_typecheck(typecheck_input: str, typecheck_expected: str):
    """Verify type checker produces expected result."""
    try:
        compile(typecheck_input)
        result_ok = True
        error_msg = ""
    except ParseError as e:
        result_ok = False
        error_msg = str(e)
    except Exception as e:
        result_ok = False
        error_msg = str(e)

    if typecheck_expected == "ok":
        if not result_ok:
            pytest.fail(f"Expected ok, got error: {error_msg}")
    elif typecheck_expected.startswith("error:"):
        expected_msg = typecheck_expected[6:].strip()
        if result_ok:
            pytest.fail(f"Expected error containing '{expected_msg}', got ok")
        if expected_msg.lower() not in error_msg.lower():
            pytest.fail(f"Expected error containing '{expected_msg}', got: {error_msg}")
    else:
        pytest.fail(f"Unknown expected format: {typecheck_expected}")
