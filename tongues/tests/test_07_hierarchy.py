"""Pytest-based hierarchy tests."""

from pathlib import Path

import pytest

from src.frontend.parse import parse
from src.frontend.names import resolve_names
from src.frontend.hierarchy import build_hierarchy
from src.frontend import Frontend
from src.serialize import hierarchy_to_dict

HIERARCHY_DIR = Path(__file__).parent / "07_hierarchy"


def parse_hierarchy_file(path: Path) -> list[tuple[str, str, str]]:
    """Parse .tests file into (name, input, expected) tuples."""
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


def discover_hierarchy_tests() -> list[tuple[str, str, str, str]]:
    """Find all hierarchy tests, returns (test_id, input, expected, file_stem)."""
    results = []
    for test_file in sorted(HIERARCHY_DIR.glob("*.tests")):
        tests = parse_hierarchy_file(test_file)
        for name, input_code, expected in tests:
            test_id = f"{test_file.stem}/{name}"
            results.append((test_id, input_code, expected, test_file.stem))
    return results


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


def run_hierarchy(source: str) -> dict[str, object]:
    """Run pipeline through phase 7 and return the hierarchy dict."""
    ast_dict = parse(source)
    name_result = resolve_names(ast_dict)
    errors = name_result.errors()
    if errors:
        raise RuntimeError(errors[0].message)
    fe = Frontend()
    fe.init_from_names(source, name_result)
    fe.collect_sigs(ast_dict)
    fe.collect_flds(ast_dict)
    rel = build_hierarchy(fe.symbols)
    result = hierarchy_to_dict(fe.symbols, rel.hierarchy_root)
    result["node_types"] = sorted(rel.node_types)
    result["exception_types"] = sorted(rel.exception_types)
    return result


def pytest_generate_tests(metafunc):
    """Parametrize tests over hierarchy test files."""
    if "hierarchy_input" in metafunc.fixturenames:
        tests = discover_hierarchy_tests()
        params = [
            pytest.param(input_code, expected, id=test_id)
            for test_id, input_code, expected, _ in tests
        ]
        metafunc.parametrize("hierarchy_input,hierarchy_expected", params)


def test_hierarchy(hierarchy_input: str, hierarchy_expected: str):
    """Verify hierarchy analysis produces expected result."""
    if hierarchy_expected == "ok":
        try:
            run_hierarchy(hierarchy_input)
        except Exception as e:
            pytest.fail(f"Expected ok, got error: {e}")
        return

    if hierarchy_expected.startswith("error:"):
        expected_msg = hierarchy_expected[6:].strip()
        try:
            run_hierarchy(hierarchy_input)
            pytest.fail(f"Expected error containing '{expected_msg}', got ok")
        except Exception as e:
            if expected_msg.lower() not in str(e).lower():
                pytest.fail(f"Expected error containing '{expected_msg}', got: {e}")
        return

    try:
        result = run_hierarchy(hierarchy_input)
    except Exception as e:
        pytest.fail(f"Hierarchy failed: {e}")

    for line in hierarchy_expected.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            pytest.fail(f"Bad assertion (no '='): {line}")
        path, expected_val = line.split("=", 1)
        path = path.strip()
        expected_val = expected_val.strip()
        try:
            actual = resolve_dotpath(result, path)
        except (KeyError, IndexError, TypeError) as e:
            pytest.fail(f"Path '{path}' not found in result: {e}")
        actual_str = _to_comparable(actual)
        if actual_str != expected_val:
            pytest.fail(
                f"Assertion failed: {path}\n"
                f"  expected: {expected_val!r}\n"
                f"  actual:   {actual_str!r}"
            )


def _to_comparable(value: object) -> str:
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
