"""Test runner for Tongues test phases."""

import subprocess
from pathlib import Path

import pytest

from src.frontend.typecollect import collect_types
from src.frontend.hierarchy import build_hierarchy
from src.frontend.pycheck import (
    run_pycheck as _run_pycheck,
    compute_expr_coverage,
    _EXPR_NODE_TYPES,
)
from src.frontend.bind import run_bind
from src.frontend.parse import parse, stamp_uids

from tests.harness import (
    EMITTERS,
    RUNTIMES,
    TESTS_DIR,
    TRANSPILED_BINARY,
    _TRANSPILED_MODULE,
    _available_targets,
    _cli_needs_backend,
    _run_inprocess,
    _transpiled_runtime,
    check_cli_assertions,
    check_expected,
    contains_normalized,
    discover_app_tests,
    discover_cli_tests,
    discover_codegen_tests,
    discover_specs,
    discover_taytsh_apps,
    emit_from_python,
    lower_to_taytsh,
    parse_simple_tests,
    run_callgraph,
    run_cli,
    run_fields,
    run_hierarchy,
    run_hoisting,
    run_liveness,
    run_names,
    run_ownership,
    run_parse,
    run_pycheck,
    run_returns,
    run_scope,
    run_sigs,
    run_strings,
    run_subset,
    run_taytsh_check,
    run_taytsh_parse,
    run_type_checking,
    taytsh_parse,
    taytsh_run,
    transpile_app,
    transpile_code,
)

# fmt: off
TESTS = {
    "cli": {
        "cli":       {"dir": "02_cli",       "run": "cli"},
    },
    "frontend": {
        "parse":     {"dir": "03_parse",     "run": "phase"},
        "subset":    {"dir": "04_subset",    "run": "phase"},
        "names":     {"dir": "05_names",     "run": "phase"},
        "sigs":      {"dir": "06_signatures", "run": "phase"},
        "fields":    {"dir": "07_fields",    "run": "phase"},
        "hierarchy": {"dir": "08_hierarchy", "run": "phase"},
        "pycheck":   {"dir": "09_pycheck",   "run": "phase"},
        "lowering":  {"dir": "10_lowering",  "run": "lowering"},
    },
    "middleend": {
        "type_checking": {"dir": "12_tycheck", "run": "phase", "glob": "lowered_*.tests"},
        "scope":     {"dir": "14_scope",     "run": "phase"},
        "returns":   {"dir": "15_returns",   "run": "phase"},
        "liveness":  {"dir": "16_liveness",  "run": "phase"},
        "strings":   {"dir": "17_strings",   "run": "phase"},
        "hoisting":  {"dir": "18_hoisting",  "run": "phase"},
        "ownership": {"dir": "19_ownership", "run": "phase"},
        "callgraph": {"dir": "20_callgraph", "run": "phase"},
    },
    "backend": {
        "codegen":        {"dir": "21_codegen", "run": "codegen"},
        "emit":           {"dir": "25_emit",    "run": "emit"},
        "app":            {"dir": "22_app",     "run": "app"},
        "ordering":       {"dir": "24_ordering", "run": "ordering"},
    },
    "taytsh": {
        "taytsh_parse": {"dir": "11_typarse",  "run": "phase"},
        "taytsh_check": {"dir": "12_tycheck",  "run": "phase", "glob": "[!l]*.tests"},
        "taytsh_app":   {"dir": "23_ty_app",   "run": "taytsh_app"},
    },
}
# fmt: on


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transpiled_output(codegen_input: str, codegen_lang: str) -> str:
    output, err = transpile_code(codegen_input, codegen_lang)
    if err is not None:
        pytest.fail(f"Transpile error: {err}")
    if output is None:
        pytest.fail("No output from transpiler")
    return output


@pytest.fixture
def emit_output(emit_input: str, emit_lang: str) -> str:
    output, err = emit_from_python(emit_input, emit_lang)
    if err is not None:
        pytest.fail(f"Emit error: {err}")
    if output is None:
        pytest.fail("No output from emitter")
    return output


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
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
                    specs = discover_specs(test_dir, cfg.get("glob", "*.tests"))
                    params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in specs]
                    metafunc.parametrize(f"{fixture},{name}_expected", params)
            elif run == "lowering":
                fixture = f"{name}_input"
                if fixture in metafunc.fixturenames:
                    specs = discover_specs(test_dir, cfg.get("glob", "*.tests"))
                    params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in specs]
                    metafunc.parametrize(f"{fixture},{name}_expected", params)
            elif run == "codegen" and "codegen_input" in metafunc.fixturenames:
                dirs = {
                    d.name
                    for d in test_dir.iterdir()
                    if d.is_dir() and d.name != "base"
                }
                langs = sorted(dirs & set(EMITTERS))
                all_tests = []
                for lang in langs:
                    for tid, inp, exp in discover_codegen_tests(test_dir, lang):
                        all_tests.append(pytest.param(inp, exp, lang, id=tid))
                for lang in sorted(set(EMITTERS) - dirs):
                    base_dir = test_dir / "base"
                    for base_file in sorted(base_dir.glob("*.tests")):
                        for name, _ in parse_simple_tests(base_file):
                            tid = f"{base_file.stem}/{name}[{lang}]"
                            all_tests.append(
                                pytest.param(
                                    "",
                                    "",
                                    lang,
                                    id=tid,
                                )
                            )
                metafunc.parametrize(
                    "codegen_input,codegen_expected,codegen_lang", all_tests
                )
            elif run == "emit" and "emit_input" in metafunc.fixturenames:
                dirs = {
                    d.name
                    for d in test_dir.iterdir()
                    if d.is_dir() and d.name != "base"
                }
                langs = sorted(dirs & set(EMITTERS))
                all_tests = []
                for lang in langs:
                    for tid, inp, exp in discover_codegen_tests(test_dir, lang):
                        all_tests.append(pytest.param(inp, exp, lang, id=tid))
                metafunc.parametrize("emit_input,emit_expected,emit_lang", all_tests)
            elif run == "taytsh_app" and "taytsh_app" in metafunc.fixturenames:
                apps = discover_taytsh_apps(test_dir)
                params = [pytest.param(p, id=p.stem) for p in apps]
                metafunc.parametrize("taytsh_app", params)
            elif run == "app" and "app_source" in metafunc.fixturenames:
                target_opt = metafunc.config.getoption("--target", default=None)
                targets = target_opt if target_opt else _available_targets()
                tests = discover_app_tests(test_dir, targets)
                params = [
                    pytest.param(path, target, id=tid) for tid, path, target in tests
                ]
                metafunc.parametrize("app_source,app_target", params)
            elif run == "ordering" and "ordering_source" in metafunc.fixturenames:
                target_opt = metafunc.config.getoption("--target", default=None)
                targets = target_opt if target_opt else _available_targets()
                ty_files = sorted(test_dir.glob("*.ty"))
                params = []
                for ty in ty_files:
                    for target in targets:
                        tid = f"{ty.stem}[{target}]"
                        params.append(pytest.param(ty, target, id=tid))
                metafunc.parametrize("ordering_source,ordering_target", params)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_cli(cli_spec: dict) -> None:
    if _cli_needs_backend(cli_spec):
        target = cli_spec["args"][cli_spec["args"].index("--target") + 1]
        pytest.skip(f"backend not yet implemented for '{target}'")
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


def test_pycheck(pycheck_input, pycheck_expected):
    check_expected(
        pycheck_expected,
        run_pycheck(pycheck_input),
        "pycheck",
        lenient_errors=True,
    )


def test_type_checking(type_checking_input, type_checking_expected):
    check_expected(
        type_checking_expected,
        run_type_checking(type_checking_input),
        "type_checking",
    )


def test_lowering(lowering_input, lowering_expected):
    output, err = lower_to_taytsh(lowering_input)
    if lowering_expected.startswith("error:"):
        expected_msg = lowering_expected[6:].strip()
        if err is None:
            pytest.fail(f"Expected error containing '{expected_msg}', got success")
        if expected_msg and expected_msg.lower() not in (err or "").lower():
            pytest.fail(f"Expected error containing '{expected_msg}', got: {err}")
        return
    if err is not None:
        pytest.fail(f"Lowering error: {err}")
    if output is None:
        pytest.fail("No output from lowering")
    if not contains_normalized(output, lowering_expected):
        pytest.fail(
            "Expected not found in output:\n"
            f"--- expected ---\n{lowering_expected}\n"
            f"--- got ---\n{output}"
        )


def test_returns(returns_input, returns_expected):
    check_expected(returns_expected, run_returns(returns_input), "returns")


def test_scope(scope_input, scope_expected):
    check_expected(scope_expected, run_scope(scope_input), "scope")


def test_liveness(liveness_input, liveness_expected):
    check_expected(liveness_expected, run_liveness(liveness_input), "liveness")


def test_strings(strings_input, strings_expected):
    check_expected(strings_expected, run_strings(strings_input), "strings")


def test_hoisting(hoisting_input, hoisting_expected):
    check_expected(hoisting_expected, run_hoisting(hoisting_input), "hoisting")


def test_ownership(ownership_input, ownership_expected):
    check_expected(ownership_expected, run_ownership(ownership_input), "ownership")


def test_callgraph(callgraph_input, callgraph_expected):
    check_expected(callgraph_expected, run_callgraph(callgraph_input), "callgraph")


def test_codegen(
    codegen_input: str,
    codegen_expected: str,
    codegen_lang: str,
    transpiled_output: str,
):
    if not contains_normalized(transpiled_output, codegen_expected):
        pytest.fail(
            "Expected not found in output:\n"
            f"--- expected ---\n{codegen_expected}\n"
            f"--- got ---\n{transpiled_output}"
        )


def test_emit(
    emit_input: str,
    emit_expected: str,
    emit_lang: str,
    emit_output: str,
):
    if not contains_normalized(emit_output, emit_expected):
        pytest.fail(
            "Expected not found in output:\n"
            f"--- expected ---\n{emit_expected}\n"
            f"--- got ---\n{emit_output}"
        )


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
    if TRANSPILED_BINARY is not None:
        argv = [TRANSPILED_BINARY, "taytsh", str(taytsh_app)]
        if _TRANSPILED_MODULE is not None:
            result = _run_inprocess(argv)
        else:
            cmd = [*_transpiled_runtime(), *argv]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            output = (result.stdout + result.stderr).decode(errors="replace").strip()
            pytest.fail(f"Exit code {result.returncode}:\n{output}")
        return
    source = taytsh_app.read_text()
    module = taytsh_parse(source)
    result = taytsh_run(module)
    if result.exit_code != 0:
        output = (result.stdout + result.stderr).decode(errors="replace").strip()
        pytest.fail(f"Exit code {result.exit_code}:\n{output}")


def test_app(app_source: Path, app_target: str) -> None:
    source = app_source.read_text()
    output, err = transpile_app(source, app_target)
    if err is not None:
        pytest.fail(f"Transpile error ({app_target}): {err}")
    runtime = RUNTIMES[app_target]
    result = subprocess.run(
        runtime,
        input=output.encode(),
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        stdout = result.stdout.decode(errors="replace")
        pytest.fail(
            f"App test failed with exit {result.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )


_EXPR_COVERAGE_SOURCE = """\
from dataclasses import dataclass
@dataclass
class Pt:
    x: int
    y: int
def f(p: Pt, xs: list[int], d: dict[str, int], s: set[int]) -> str:
    a: int = 1
    b: int = p.x + p.y
    c: int = -a
    ok: bool = a > b
    both: bool = ok and ok
    e: int = a if ok else b
    t: tuple[int, int] = (a, b)
    ns: list[int] = [a, b]
    ds: dict[str, int] = {"k": a}
    ss: set[int] = {a, b}
    lc: list[int] = [x for x in xs]
    sc: set[int] = {x for x in xs}
    dc: dict[str, int] = {k: a for k in ds}
    msg: str = f"{a}"
    n: int = (w := a)
    ln: int = len(xs)
    el: int = xs[a]
    return msg
"""


def test_expr_coverage() -> None:
    ast_dict = parse(_EXPR_COVERAGE_SOURCE)
    bind_result = run_bind(ast_dict)
    assert bind_result.subset_ok()
    assert bind_result.names_ok()
    hier_result = build_hierarchy(bind_result.known_classes, bind_result.class_bases)
    assert not hier_result.errors()
    tc_result = collect_types(
        ast_dict,
        bind_result.known_classes,
        bind_result.node_classes,
        bind_result.type_aliases,
        bind_result.class_bases,
        hier_result.hierarchy_roots,
    )
    assert not tc_result.errors()
    stamp_uids(ast_dict)
    inf_result = _run_pycheck(
        ast_dict,
        tc_result,
        hier_result,
        bind_result.known_classes,
        bind_result.class_bases,
        bind_result.flow_graphs,
    )
    assert not inf_result.errors()
    totals, covered = compute_expr_coverage(ast_dict, inf_result)
    for node_type in sorted(_EXPR_NODE_TYPES):
        assert totals.get(node_type, 0) > 0, (
            f"{node_type}: no instances found in test source"
        )
        assert covered.get(node_type, 0) > 0, (
            f"{node_type}: zero coverage ({totals[node_type]} instances)"
        )


def test_ordering(ordering_source: Path, ordering_target: str) -> None:
    source = ordering_source.read_text()
    output, err = transpile_code(source, ordering_target)
    if err is not None:
        pytest.fail(f"Transpile error ({ordering_target}): {err}")
    runtime = RUNTIMES[ordering_target]
    result = subprocess.run(
        runtime,
        input=output.encode(),
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        stdout = result.stdout.decode(errors="replace")
        pytest.fail(
            f"Ordering test failed with exit {result.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
