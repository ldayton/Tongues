"""Test runner for Tongues v2 test phases."""

import signal
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.backend_v2.python import emit_python as emit_python_v2
from src.middleend_v2.liveness import analyze_liveness
from src.middleend_v2.returns import analyze_returns
from src.middleend_v2.scope import analyze_scope
from src.taytsh import parse as taytsh_parse
from src.taytsh.ast import (
    TAssignStmt,
    TBinaryOp,
    TCall,
    TExprStmt,
    TFieldAccess,
    TFnDecl,
    TFnLit,
    TForStmt,
    TIfStmt,
    TIndex,
    TLetStmt,
    TListLit,
    TMapLit,
    TMatchStmt,
    TOpAssignStmt,
    TPatternType,
    TRange,
    TReturnStmt,
    TSetLit,
    TSlice,
    TStructDecl,
    TThrowStmt,
    TTernary,
    TTryStmt,
    TTupleAssignStmt,
    TTupleLit,
    TUnaryOp,
    TVar,
    TWhileStmt,
)
from src.taytsh.check import check_with_info

PARSE_TIMEOUT = 5
TESTS_DIR = Path(__file__).parent

# fmt: off
TESTS = {
    "middleend": {
        "scope_v2":    {"dir": "13_v2_scope",    "run": "phase"},
        "returns_v2":  {"dir": "14_v2_returns",  "run": "phase"},
        "liveness_v2": {"dir": "15_v2_liveness", "run": "phase"},
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
# v2 serializers
# ---------------------------------------------------------------------------


def _strip_prefix(annotations, prefix):
    return {k[len(prefix) :]: v for k, v in annotations.items() if k.startswith(prefix)}


def _serialize_returns_stmt(stmt):
    d = {"type": type(stmt).__name__}
    for k, v in getattr(stmt, "annotations", {}).items():
        if k.startswith("returns."):
            d[k[8:]] = v
    if isinstance(stmt, TMatchStmt):
        d["cases"] = [_strip_prefix(c.annotations, "returns.") for c in stmt.cases]
        if stmt.default is not None:
            d["default"] = _strip_prefix(stmt.default.annotations, "returns.")
    elif isinstance(stmt, TTryStmt):
        d["catches"] = [_strip_prefix(c.annotations, "returns.") for c in stmt.catches]
    return d


def _serialize_fn_returns(fn):
    d = _strip_prefix(fn.annotations, "returns.")
    d["body"] = [_serialize_returns_stmt(s) for s in fn.body]
    return d


def _serialize_returns(module):
    result = {}
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            result[decl.name] = _serialize_fn_returns(decl)
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                result[f"{decl.name}.{method.name}"] = _serialize_fn_returns(method)
    return result


def _serialize_scope_stmt(stmt):
    d = {"type": type(stmt).__name__}
    if isinstance(stmt, TForStmt):
        binder = {}
        for k, v in stmt.annotations.items():
            if k.startswith("scope.binder."):
                rest = k[len("scope.binder.") :]
                bname, attr = rest.split(".", 1)
                if bname not in binder:
                    binder[bname] = {}
                binder[bname][attr] = v
        if binder:
            d["binder"] = binder
    elif isinstance(stmt, TMatchStmt):
        cases = []
        for case in stmt.cases:
            cd = {}
            if isinstance(case.pattern, TPatternType):
                a = _strip_prefix(case.pattern.annotations, "scope.")
                if a:
                    cd["pattern"] = a
            cases.append(cd)
        d["cases"] = cases
        if stmt.default is not None:
            a = _strip_prefix(stmt.default.annotations, "scope.")
            if a:
                d["default"] = a
    elif isinstance(stmt, TTryStmt):
        d["catches"] = [_strip_prefix(c.annotations, "scope.") for c in stmt.catches]
    return d


def _collect_vars_expr(expr, result):
    if isinstance(expr, TVar):
        a = _strip_prefix(expr.annotations, "scope.")
        if a:
            if expr.name in result:
                result[expr.name].update(a)
            else:
                result[expr.name] = dict(a)
    elif isinstance(expr, TBinaryOp):
        _collect_vars_expr(expr.left, result)
        _collect_vars_expr(expr.right, result)
    elif isinstance(expr, TUnaryOp):
        _collect_vars_expr(expr.operand, result)
    elif isinstance(expr, TCall):
        _collect_vars_expr(expr.func, result)
        for a in expr.args:
            _collect_vars_expr(a.value, result)
    elif isinstance(expr, TFieldAccess):
        _collect_vars_expr(expr.obj, result)
    elif isinstance(expr, TIndex):
        _collect_vars_expr(expr.obj, result)
        _collect_vars_expr(expr.index, result)
    elif isinstance(expr, TTernary):
        _collect_vars_expr(expr.cond, result)
        _collect_vars_expr(expr.then_expr, result)
        _collect_vars_expr(expr.else_expr, result)
    elif isinstance(expr, TSlice):
        _collect_vars_expr(expr.obj, result)
        _collect_vars_expr(expr.low, result)
        _collect_vars_expr(expr.high, result)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _collect_vars_expr(e, result)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _collect_vars_expr(k, result)
            _collect_vars_expr(v, result)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _collect_vars_expr(e, result)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _collect_vars_expr(e, result)
    elif isinstance(expr, TFnLit):
        if isinstance(expr.body, list):
            _collect_vars_stmts(expr.body, result)
        else:
            _collect_vars_expr(expr.body, result)


def _collect_vars_stmts(stmts, result):
    for stmt in stmts:
        _collect_vars_stmt(stmt, result)


def _collect_vars_stmt(stmt, result):
    if isinstance(stmt, TExprStmt):
        _collect_vars_expr(stmt.expr, result)
    elif isinstance(stmt, TReturnStmt) and stmt.value is not None:
        _collect_vars_expr(stmt.value, result)
    elif isinstance(stmt, TThrowStmt):
        _collect_vars_expr(stmt.expr, result)
    elif isinstance(stmt, TLetStmt) and stmt.value is not None:
        _collect_vars_expr(stmt.value, result)
    elif isinstance(stmt, TAssignStmt):
        _collect_vars_expr(stmt.target, result)
        _collect_vars_expr(stmt.value, result)
    elif isinstance(stmt, TOpAssignStmt):
        _collect_vars_expr(stmt.target, result)
        _collect_vars_expr(stmt.value, result)
    elif isinstance(stmt, TTupleAssignStmt):
        for t in stmt.targets:
            _collect_vars_expr(t, result)
        _collect_vars_expr(stmt.value, result)
    elif isinstance(stmt, TIfStmt):
        _collect_vars_expr(stmt.cond, result)
        _collect_vars_stmts(stmt.then_body, result)
        if stmt.else_body is not None:
            _collect_vars_stmts(stmt.else_body, result)
    elif isinstance(stmt, TWhileStmt):
        _collect_vars_expr(stmt.cond, result)
        _collect_vars_stmts(stmt.body, result)
    elif isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                _collect_vars_expr(a, result)
        else:
            _collect_vars_expr(stmt.iterable, result)
        _collect_vars_stmts(stmt.body, result)
    elif isinstance(stmt, TMatchStmt):
        _collect_vars_expr(stmt.expr, result)
        for case in stmt.cases:
            _collect_vars_stmts(case.body, result)
        if stmt.default is not None:
            _collect_vars_stmts(stmt.default.body, result)
    elif isinstance(stmt, TTryStmt):
        _collect_vars_stmts(stmt.body, result)
        for catch in stmt.catches:
            _collect_vars_stmts(catch.body, result)
        if stmt.finally_body is not None:
            _collect_vars_stmts(stmt.finally_body, result)


def _serialize_fn_scope(fn):
    d = {}
    params = {}
    for p in fn.params:
        a = _strip_prefix(p.annotations, "scope.")
        if a:
            params[p.name] = a
    if params:
        d["params"] = params
    lets = {}
    for stmt in fn.body:
        if isinstance(stmt, TLetStmt):
            a = _strip_prefix(stmt.annotations, "scope.")
            if a:
                lets[stmt.name] = a
    if lets:
        d["lets"] = lets
    d["body"] = [_serialize_scope_stmt(s) for s in fn.body]
    vars_dict = {}
    _collect_vars_stmts(fn.body, vars_dict)
    if vars_dict:
        d["vars"] = vars_dict
    return d


def _serialize_scope(module):
    result = {}
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            result[decl.name] = _serialize_fn_scope(decl)
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                result[f"{decl.name}.{method.name}"] = _serialize_fn_scope(method)
    return result


def _collect_all_lets(stmts, lets):
    for stmt in stmts:
        if isinstance(stmt, TLetStmt):
            a = _strip_prefix(stmt.annotations, "liveness.")
            if a:
                lets[stmt.name] = a
        if isinstance(stmt, TIfStmt):
            _collect_all_lets(stmt.then_body, lets)
            if stmt.else_body is not None:
                _collect_all_lets(stmt.else_body, lets)
        elif isinstance(stmt, TWhileStmt):
            _collect_all_lets(stmt.body, lets)
        elif isinstance(stmt, TForStmt):
            _collect_all_lets(stmt.body, lets)
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                _collect_all_lets(case.body, lets)
            if stmt.default is not None:
                _collect_all_lets(stmt.default.body, lets)
        elif isinstance(stmt, TTryStmt):
            _collect_all_lets(stmt.body, lets)
            for catch in stmt.catches:
                _collect_all_lets(catch.body, lets)
            if stmt.finally_body is not None:
                _collect_all_lets(stmt.finally_body, lets)


def _serialize_liveness_stmt(stmt):
    d = {"type": type(stmt).__name__}
    for k, v in getattr(stmt, "annotations", {}).items():
        if k.startswith("liveness."):
            d[k[9:]] = v
    if isinstance(stmt, TMatchStmt):
        cases = []
        for case in stmt.cases:
            cd = {}
            if isinstance(case.pattern, TPatternType):
                a = _strip_prefix(case.pattern.annotations, "liveness.")
                if a:
                    cd.update(a)
            cases.append(cd)
        d["cases"] = cases
        if stmt.default is not None:
            dd = _strip_prefix(stmt.default.annotations, "liveness.")
            d["default"] = dd
    elif isinstance(stmt, TTryStmt):
        d["catches"] = [_strip_prefix(c.annotations, "liveness.") for c in stmt.catches]
    return d


def _serialize_fn_liveness(fn):
    d = {}
    lets = {}
    _collect_all_lets(fn.body, lets)
    if lets:
        d["lets"] = lets
    d["body"] = [_serialize_liveness_stmt(s) for s in fn.body]
    return d


def _serialize_liveness(module):
    result = {}
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            result[decl.name] = _serialize_fn_liveness(decl)
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                result[f"{decl.name}.{method.name}"] = _serialize_fn_liveness(method)
    return result


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
    return PhaseResult(data=_serialize_returns(module))


def run_scope_v2(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    return PhaseResult(data=_serialize_scope(module))


def run_liveness_v2(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    return PhaseResult(data=_serialize_liveness(module))


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
