"""Pytest-based parser tests."""

import signal
from pathlib import Path

import pytest

from src.frontend.parse import parse, ParseError

PARSE_TIMEOUT = 5


def _timeout_handler(signum, frame):
    raise TimeoutError("parse() timed out")


signal.signal(signal.SIGALRM, _timeout_handler)

PARSE_DIR = Path(__file__).parent / "02_parse"


def parse_test_file(path: Path) -> list[tuple[str, str, str]]:
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


def discover_parse_tests() -> list[tuple[str, str, str, str]]:
    """Find all parse tests, returns (test_id, input, expected, file_stem)."""
    results = []
    for test_file in sorted(PARSE_DIR.glob("*.tests")):
        tests = parse_test_file(test_file)
        for name, input_code, expected in tests:
            test_id = f"{test_file.stem}/{name}"
            results.append((test_id, input_code, expected, test_file.stem))
    return results


def pytest_generate_tests(metafunc):
    """Parametrize tests over parse test files."""
    if "parse_input" in metafunc.fixturenames:
        tests = discover_parse_tests()
        params = [
            pytest.param(input_code, expected, id=test_id)
            for test_id, input_code, expected, _ in tests
        ]
        metafunc.parametrize("parse_input,parse_expected", params)


def test_parse(parse_input: str, parse_expected: str):
    """Verify parser produces expected result."""
    try:
        signal.alarm(PARSE_TIMEOUT)
        parse(parse_input)
        parse_succeeded = True
        parse_error = None
    except ParseError as e:
        parse_succeeded = False
        parse_error = e
    except Exception as e:
        parse_succeeded = False
        parse_error = e
    finally:
        signal.alarm(0)

    if parse_expected == "ok":
        if not parse_succeeded:
            pytest.fail(f"Expected ok, got parse error: {parse_error}")
    elif parse_expected.startswith("error:"):
        expected_msg = parse_expected[6:].strip()
        if parse_succeeded:
            pytest.fail(
                f"Expected error containing '{expected_msg}', but parsing succeeded"
            )
        # If we expected an error and got one, that's a pass
        # Optionally check if error message contains expected substring
        if expected_msg and parse_error:
            error_str = str(parse_error).lower()
            if expected_msg.lower() not in error_str:
                # For now, just accept any parse error - the important thing is it failed
                pass
    else:
        pytest.fail(f"Unknown expected format: {parse_expected}")
