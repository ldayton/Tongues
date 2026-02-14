"""Test runner for Tongues v2 test phases."""

import signal
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.test_runner import (
    check_cli_assertions,
    discover_cli_tests,
    run_cli,
)

from src.frontend import Frontend, compile as frontend_compile
from src.frontend.hierarchy import build_hierarchy
from src.frontend.names import resolve_names
from src.frontend.parse import parse
from src.frontend.subset import verify as verify_subset
from src.serialize import fields_to_dict, hierarchy_to_dict, signatures_to_dict

from src.backend_v2.perl import emit_perl as emit_perl_v2
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
    "cli": {
        "cli_v2":       {"dir": "02_v2_cli",       "run": "cli"},
    },
    "frontend": {
        "parse_v2":     {"dir": "03_v2_parse",     "run": "phase"},
        "subset_v2":    {"dir": "04_v2_subset",    "run": "phase"},
        "names_v2":     {"dir": "05_v2_names",     "run": "phase"},
        "sigs_v2":      {"dir": "06_v2_signatures", "run": "phase"},
        "fields_v2":    {"dir": "07_v2_fields",    "run": "phase"},
        "hierarchy_v2": {"dir": "08_v2_hierarchy", "run": "phase"},
        "inference_v2": {"dir": "09_v2_inference", "run": "phase"},
        "lowering_v2":  {"dir": "10_v2_lowering",  "run": "lowering"},
    },
    "taytsh": {
        "type_checking_v2": {"dir": "11_v2_type_checking", "run": "phase"},
    },
    "middleend": {
        "scope_v2":     {"dir": "14_v2_scope",     "run": "phase"},
        "returns_v2":   {"dir": "15_v2_returns",   "run": "phase"},
        "liveness_v2":  {"dir": "16_v2_liveness",  "run": "phase"},
        "strings_v2":   {"dir": "17_v2_strings",   "run": "phase"},
        "hoisting_v2":  {"dir": "18_v2_hoisting",  "run": "phase"},
        "ownership_v2": {"dir": "19_v2_ownership", "run": "phase"},
        "callgraph_v2": {"dir": "20_v2_callgraph", "run": "phase"},
    },
    "backend": {
        "codegen_v2_python": {"dir": "21_v2_codegen", "run": "codegen_v2_python"},
        "codegen_v2_perl": {"dir": "21_v2_codegen", "run": "codegen_v2_perl"},
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


def discover_codegen_v2_tests(test_dir: Path, lang: str) -> list[tuple[str, str, str]]:
    """Find v2 codegen tests for a language, returns (test_id, input, expected)."""
    results = []
    for test_file in sorted(test_dir.glob("*.tests")):
        tests = parse_codegen_file(test_file)
        # Skip files that are for other languages (e.g. perl.tests vs python.tests).
        if not any(lang in expected_by_lang for _, _, expected_by_lang in tests):
            continue
        for name, input_code, expected_by_lang in tests:
            expected = expected_by_lang.get(lang)
            if expected is None:
                pytest.fail(
                    f"{test_file.name}:{name} missing '--- {lang}' expected block"
                )
            test_id = f"{test_file.stem}/{name}[{lang}-v2]"
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


def run_parse_v2(source: str) -> PhaseResult:
    """Run the Python frontend parser, return ok/error result."""
    try:
        signal.alarm(PARSE_TIMEOUT)
        parse(source)
        return PhaseResult()
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    finally:
        signal.alarm(0)


def run_subset_v2(source: str) -> PhaseResult:
    """Run subset verification on Python source."""
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    result = verify_subset(ast_dict)
    return PhaseResult(
        errors=[e.message for e in result.errors()],
        warnings=[w.message for w in result.warnings()],
    )


def run_names_v2(source: str) -> PhaseResult:
    """Run name resolution on Python source."""
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
    """Parse -> names -> Frontend -> sigs. Raises on error."""
    ast_dict = parse(source)
    name_result = resolve_names(ast_dict)
    errors = name_result.errors()
    if errors:
        raise RuntimeError(errors[0].message)
    fe = Frontend()
    fe.init_from_names(source, name_result)
    fe.collect_sigs(ast_dict)
    return fe, ast_dict


def run_sigs_v2(source: str) -> PhaseResult:
    """Run signature collection on Python source."""
    try:
        fe, _ = _run_frontend_through_sigs(source)
        return PhaseResult(data=signatures_to_dict(fe.symbols))
    except Exception as e:
        return PhaseResult(errors=[str(e)])


def run_fields_v2(source: str) -> PhaseResult:
    """Run field collection on Python source."""
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


def run_hierarchy_v2(source: str) -> PhaseResult:
    """Run hierarchy analysis on Python source."""
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


def run_inference_v2(source: str) -> PhaseResult:
    """Run the full Python frontend pipeline (phases 2-9), checking inference errors."""
    try:
        frontend_compile(source)
        return PhaseResult()
    except Exception as e:
        return PhaseResult(errors=[str(e)])


def run_type_checking_v2(source: str) -> PhaseResult:
    """Run the Taytsh type checker on Taytsh source."""
    try:
        module = taytsh_parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    errors, checker = check_with_info(module)
    if errors:
        return PhaseResult(errors=[str(e) for e in errors])
    return PhaseResult()


def lower_to_taytsh(source: str) -> tuple[str | None, str | None]:
    """Lower Python source to Taytsh text. Returns (output, error)."""
    try:
        # TODO: replace with v2 lowering pipeline when implemented
        pytest.skip("v2 lowering not yet implemented")
    except Exception as e:
        return (None, str(e))


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
    for attr in (
        "body",
        "value",
        "expr",
        "func",
        "target",
        "targets",
        "cond",
        "then_body",
        "else_body",
        "then_expr",
        "else_expr",
        "left",
        "right",
        "operand",
        "obj",
        "index",
        "low",
        "high",
        "args",
        "elements",
        "entries",
        "iterable",
        "cases",
        "default",
        "catches",
        "finally_body",
        "pattern",
    ):
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
            d = {
                k[10:]: v
                for k, v in decl.annotations.items()
                if k.startswith("callgraph.")
            }
            calls = {}
            _collect_calls(decl.body, calls, checker)
            if calls:
                d["calls"] = calls
            result[decl.name] = d
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                d = {
                    k[10:]: v
                    for k, v in method.annotations.items()
                    if k.startswith("callgraph.")
                }
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


def _transpile_v2_with_emitter(source: str, emitter) -> tuple[str | None, str | None]:
    """Transpile Taytsh source for backend v2. Returns (output, error)."""
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
        return (emitter(module), None)
    except Exception as e:
        return (None, str(e))


def transpile_code_v2_python(source: str) -> tuple[str | None, str | None]:
    """Transpile Taytsh source using python backend v2. Returns (output, error)."""
    return _transpile_v2_with_emitter(source, emit_python_v2)


def transpile_code_v2_perl(source: str) -> tuple[str | None, str | None]:
    """Transpile Taytsh source using perl backend v2. Returns (output, error)."""
    return _transpile_v2_with_emitter(source, emit_perl_v2)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transpiled_output_v2_python(codegen_v2_python_input: str) -> str:
    output, err = transpile_code_v2_python(codegen_v2_python_input)
    if err is not None:
        pytest.fail(f"Transpile error: {err}")
    if output is None:
        pytest.fail("No output from transpiler")
    return output


@pytest.fixture
def transpiled_output_v2_perl(codegen_v2_perl_input: str) -> str:
    output, err = transpile_code_v2_perl(codegen_v2_perl_input)
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
            if run == "cli" and "cli_v2_spec" in metafunc.fixturenames:
                tests = discover_cli_tests(test_dir)
                params = [pytest.param(spec, id=tid) for tid, spec in tests]
                metafunc.parametrize("cli_v2_spec", params)
            elif run == "phase":
                fixture = f"{name}_input"
                if fixture in metafunc.fixturenames:
                    specs = discover_specs(test_dir)
                    params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in specs]
                    metafunc.parametrize(f"{fixture},{name}_expected", params)
            elif run == "lowering":
                fixture = f"{name}_input"
                if fixture in metafunc.fixturenames:
                    specs = discover_specs(test_dir)
                    params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in specs]
                    metafunc.parametrize(f"{fixture},{name}_expected", params)
            elif (
                run == "codegen_v2_python"
                and "codegen_v2_python_input" in metafunc.fixturenames
            ):
                tests = discover_codegen_v2_tests(test_dir, "python")
                params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in tests]
                metafunc.parametrize(
                    "codegen_v2_python_input,codegen_v2_python_expected", params
                )
            elif (
                run == "codegen_v2_perl"
                and "codegen_v2_perl_input" in metafunc.fixturenames
            ):
                tests = discover_codegen_v2_tests(test_dir, "perl")
                params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in tests]
                metafunc.parametrize(
                    "codegen_v2_perl_input,codegen_v2_perl_expected", params
                )


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_cli_v2(cli_v2_spec: dict) -> None:
    result = run_cli(cli_v2_spec)
    check_cli_assertions(result, cli_v2_spec["assertions"])


def test_parse_v2(parse_v2_input, parse_v2_expected):
    check_expected(
        parse_v2_expected, run_parse_v2(parse_v2_input), "parse_v2", lenient_errors=True
    )


def test_subset_v2(subset_v2_input, subset_v2_expected):
    check_expected(subset_v2_expected, run_subset_v2(subset_v2_input), "subset_v2")


def test_names_v2(names_v2_input, names_v2_expected):
    check_expected(names_v2_expected, run_names_v2(names_v2_input), "names_v2")


def test_sigs_v2(sigs_v2_input, sigs_v2_expected):
    check_expected(sigs_v2_expected, run_sigs_v2(sigs_v2_input), "sigs_v2")


def test_fields_v2(fields_v2_input, fields_v2_expected):
    check_expected(fields_v2_expected, run_fields_v2(fields_v2_input), "fields_v2")


def test_hierarchy_v2(hierarchy_v2_input, hierarchy_v2_expected):
    check_expected(
        hierarchy_v2_expected, run_hierarchy_v2(hierarchy_v2_input), "hierarchy_v2"
    )


def test_inference_v2(inference_v2_input, inference_v2_expected):
    check_expected(
        inference_v2_expected,
        run_inference_v2(inference_v2_input),
        "inference_v2",
        lenient_errors=True,
    )


def test_type_checking_v2(type_checking_v2_input, type_checking_v2_expected):
    check_expected(
        type_checking_v2_expected,
        run_type_checking_v2(type_checking_v2_input),
        "type_checking_v2",
    )


def test_lowering_v2(lowering_v2_input, lowering_v2_expected):
    output, err = lower_to_taytsh(lowering_v2_input)
    if lowering_v2_expected.startswith("error:"):
        expected_msg = lowering_v2_expected[6:].strip()
        if err is None:
            pytest.fail(f"Expected error containing '{expected_msg}', got success")
        if expected_msg and expected_msg.lower() not in (err or "").lower():
            pytest.fail(f"Expected error containing '{expected_msg}', got: {err}")
        return
    if err is not None:
        pytest.fail(f"Lowering error: {err}")
    if output is None:
        pytest.fail("No output from lowering")
    if not contains_normalized(output, lowering_v2_expected):
        pytest.fail(
            "Expected not found in output:\n"
            f"--- expected ---\n{lowering_v2_expected}\n"
            f"--- got ---\n{output}"
        )


def test_returns_v2(returns_v2_input, returns_v2_expected):
    check_expected(returns_v2_expected, run_returns_v2(returns_v2_input), "returns_v2")


def test_scope_v2(scope_v2_input, scope_v2_expected):
    check_expected(scope_v2_expected, run_scope_v2(scope_v2_input), "scope_v2")


def test_liveness_v2(liveness_v2_input, liveness_v2_expected):
    check_expected(
        liveness_v2_expected, run_liveness_v2(liveness_v2_input), "liveness_v2"
    )


def test_strings_v2(strings_v2_input, strings_v2_expected):
    check_expected(strings_v2_expected, run_strings_v2(strings_v2_input), "strings_v2")


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
    codegen_v2_python_input: str,
    codegen_v2_python_expected: str,
    transpiled_output_v2_python: str,
):
    if not contains_normalized(transpiled_output_v2_python, codegen_v2_python_expected):
        pytest.fail(
            "Expected not found in output:\n"
            f"--- expected ---\n{codegen_v2_python_expected}\n"
            f"--- got ---\n{transpiled_output_v2_python}"
        )


def test_codegen_v2_perl(
    codegen_v2_perl_input: str,
    codegen_v2_perl_expected: str,
    transpiled_output_v2_perl: str,
):
    if not contains_normalized(transpiled_output_v2_perl, codegen_v2_perl_expected):
        pytest.fail(
            "Expected not found in output:\n"
            f"--- expected ---\n{codegen_v2_perl_expected}\n"
            f"--- got ---\n{transpiled_output_v2_perl}"
        )
