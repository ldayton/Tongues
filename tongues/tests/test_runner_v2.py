"""Test runner for Tongues v2 test phases."""

import signal
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.backend_v2.python import emit_python as emit_python_v2
from src.middleend_v2.callgraph import analyze_callgraph
from src.middleend_v2.hoisting import analyze_hoisting
from src.middleend_v2.liveness import analyze_liveness
from src.middleend_v2.ownership import analyze_ownership
from src.middleend_v2.returns import analyze_returns
from src.middleend_v2.scope import analyze_scope
from src.middleend_v2.strings import analyze_strings
from src.taytsh import parse as taytsh_parse
from src.taytsh.ast import (
    TCall,
    TFieldAccess,
    TFnDecl,
    TStructDecl,
    TVar,
    serialize_annotations,
)
from src.taytsh.check import Checker, StructT, check_with_info

PARSE_TIMEOUT = 5
TESTS_DIR = Path(__file__).parent

# fmt: off
TESTS = {
    "middleend": {
        "scope_v2":     {"dir": "13_v2_scope",     "run": "phase"},
        "returns_v2":   {"dir": "14_v2_returns",   "run": "phase"},
        "liveness_v2":  {"dir": "15_v2_liveness",  "run": "phase"},
        "strings_v2":   {"dir": "16_v2_strings",   "run": "phase"},
        "hoisting_v2":  {"dir": "17_v2_hoisting",  "run": "phase"},
        "ownership_v2": {"dir": "18_v2_ownership", "run": "phase"},
        "callgraph_v2": {"dir": "19_v2_callgraph", "run": "phase"},
    },
    "backend": {
        "codegen_v2_python": {"dir": "20_v2_codegen", "run": "codegen_v2_python"},
    },
}
# fmt: on


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def _timeout_handler(signum, frame):
    raise TimeoutError("parse() timed out")


signal.signal(signal.SIGALRM, _timeout_handler)


# ---------------------------------------------------------------------------
# Discovery + parsing
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


def discover_codegen_v2_python_tests(test_dir: Path) -> list[tuple[str, str, str]]:
    """Find v2 Python codegen tests, returns (test_id, input, expected)."""
    results = []
    for test_file in sorted(test_dir.glob("*.tests")):
        tests = parse_codegen_file(test_file)
        for name, input_code, expected_by_lang in tests:
            expected = expected_by_lang.get("python")
            if expected is None:
                pytest.fail(
                    f"{test_file.name}:{name} missing '--- python' expected block"
                )
            test_id = f"{test_file.stem}/{name}[python-v2]"
            results.append((test_id, input_code, expected))
    return results


# ---------------------------------------------------------------------------
# Result + assertion checker
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
        # Cross-reference: if RHS looks like a dotpath, resolve it too
        if "." in expected_val and " " not in expected_val:
            try:
                ref_val = resolve_dotpath(result.data, expected_val)
                expected_val = to_comparable(ref_val)
            except (KeyError, IndexError, TypeError):
                pass  # treat as literal
        if actual_str != expected_val:
            pytest.fail(
                f"Assertion failed: {path}\n"
                f"  expected: {expected_val!r}\n"
                f"  actual:   {actual_str!r}"
            )


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



# ---------------------------------------------------------------------------
# v2 runners
# ---------------------------------------------------------------------------


def _run_taytsh_pipeline(source):
    module = taytsh_parse(source)
    errors, checker = check_with_info(module)
    if errors:
        return PhaseResult(errors=[str(e) for e in errors]), None, None
    return None, module, checker


def run_returns_v2(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_returns(module, checker)
    return PhaseResult(data=serialize_annotations(module, "returns"))


def run_scope_v2(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    return PhaseResult(data=serialize_annotations(module, "scope"))


def run_liveness_v2(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    return PhaseResult(data=serialize_annotations(module, "liveness"))


def run_strings_v2(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    analyze_strings(module, checker)
    return PhaseResult(data=serialize_annotations(module, "strings"))


def run_hoisting_v2(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_hoisting(module, checker)
    return PhaseResult(data=serialize_annotations(module, "hoisting"))


def run_ownership_v2(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    analyze_ownership(module, checker)
    return PhaseResult(data=serialize_annotations(module, "ownership"))


def _collect_calls(obj, calls, checker):
    """Walk AST collecting TCall nodes keyed by callee name."""
    if isinstance(obj, TCall):
        name = None
        if isinstance(obj.func, TVar):
            n = obj.func.name
            # Skip builtins and struct constructors — only user-defined fns
            t = checker.types.get(n)
            if t is not None and isinstance(t, StructT):
                name = None  # struct construction
            elif n in checker.functions:
                name = n
            else:
                name = None  # builtin or unknown
        elif isinstance(obj.func, TFieldAccess):
            name = obj.func.field
        if name is not None:
            is_tail = obj.annotations.get("callgraph.is_tail_call", False)
            calls.setdefault(name, {})["is_tail_call"] = is_tail
    if isinstance(obj, list):
        for item in obj:
            _collect_calls(item, calls, checker)
        return
    for attr in ("body", "value", "expr", "func", "target", "targets", "cond",
                 "then_body", "else_body", "then_expr", "else_expr", "left", "right",
                 "operand", "obj", "index", "low", "high", "args", "elements",
                 "entries", "iterable", "cases", "default", "catches", "finally_body",
                 "pattern"):
        child = getattr(obj, attr, None)
        if child is not None:
            if isinstance(child, list):
                for item in child:
                    _collect_calls(item, calls, checker)
            elif isinstance(child, tuple):
                for item in child:
                    _collect_calls(item, calls, checker)
            elif hasattr(child, "__dict__"):
                _collect_calls(child, calls, checker)


def _serialize_callgraph(module, checker):
    result = {}
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            d = {k[10:]: v for k, v in decl.annotations.items() if k.startswith("callgraph.")}
            calls = {}
            _collect_calls(decl.body, calls, checker)
            if calls:
                d["calls"] = calls
            result[decl.name] = d
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                d = {k[10:]: v for k, v in method.annotations.items() if k.startswith("callgraph.")}
                calls = {}
                _collect_calls(method.body, calls, checker)
                if calls:
                    d["calls"] = calls
                result[f"{decl.name}.{method.name}"] = d
    return result


def run_callgraph_v2(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_callgraph(module, checker)
    return PhaseResult(data=_serialize_callgraph(module, checker))


def transpile_code_v2_python(source: str) -> tuple[str | None, str | None]:
    """Transpile Taytsh source using python backend v2. Returns (output, error)."""
    try:
        signal.alarm(PARSE_TIMEOUT)
        module = taytsh_parse(source)
    except Exception as e:
        return (None, str(e))
    finally:
        signal.alarm(0)
    errors, checker = check_with_info(module)
    if errors:
        return (None, str(errors[0]))
    try:
        analyze_returns(module, checker)
        analyze_scope(module, checker)
        analyze_liveness(module, checker)
        return (emit_python_v2(module), None)
    except Exception as e:
        return (None, str(e))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transpiled_output_v2_python(codegen_v2_input: str) -> str:
    output, err = transpile_code_v2_python(codegen_v2_input)
    if err is not None:
        pytest.fail(f"Transpile error: {err}")
    if output is None:
        pytest.fail("No output from transpiler")
    return output


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    for section in TESTS.values():
        for name, cfg in section.items():
            test_dir = TESTS_DIR / cfg["dir"]
            run = cfg["run"]
            if run == "phase":
                fixture = f"{name}_input"
                if fixture in metafunc.fixturenames:
                    specs = discover_specs(test_dir)
                    params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in specs]
                    metafunc.parametrize(f"{fixture},{name}_expected", params)
            elif (
                run == "codegen_v2_python"
                and "codegen_v2_input" in metafunc.fixturenames
            ):
                tests = discover_codegen_v2_python_tests(test_dir)
                params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in tests]
                metafunc.parametrize("codegen_v2_input,codegen_v2_expected", params)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_returns_v2(returns_v2_input, returns_v2_expected):
    check_expected(returns_v2_expected, run_returns_v2(returns_v2_input), "returns_v2")


def test_scope_v2(scope_v2_input, scope_v2_expected):
    check_expected(scope_v2_expected, run_scope_v2(scope_v2_input), "scope_v2")


def test_liveness_v2(liveness_v2_input, liveness_v2_expected):
    check_expected(
        liveness_v2_expected, run_liveness_v2(liveness_v2_input), "liveness_v2"
    )


def test_strings_v2(strings_v2_input, strings_v2_expected):
    check_expected(
        strings_v2_expected, run_strings_v2(strings_v2_input), "strings_v2"
    )


def test_hoisting_v2(hoisting_v2_input, hoisting_v2_expected):
    check_expected(
        hoisting_v2_expected, run_hoisting_v2(hoisting_v2_input), "hoisting_v2"
    )


def test_ownership_v2(ownership_v2_input, ownership_v2_expected):
    check_expected(
        ownership_v2_expected, run_ownership_v2(ownership_v2_input), "ownership_v2"
    )


def test_callgraph_v2(callgraph_v2_input, callgraph_v2_expected):
    check_expected(
        callgraph_v2_expected, run_callgraph_v2(callgraph_v2_input), "callgraph_v2"
    )


def test_codegen_v2_python(
    codegen_v2_input: str,
    codegen_v2_expected: str,
    transpiled_output_v2_python: str,
):
    if not contains_normalized(transpiled_output_v2_python, codegen_v2_expected):
        pytest.fail(
            "Expected not found in output:\n"
            f"--- expected ---\n{codegen_v2_expected}\n"
            f"--- got ---\n{transpiled_output_v2_python}"
        )
