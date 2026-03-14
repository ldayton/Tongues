"""Frontend test phase: cli, parse, subset, names, sigs, fields, hierarchy, pycheck, lowering."""

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
    TESTS_DIR,
    _cli_needs_backend,
    check_cli_assertions,
    check_expected,
    contains_normalized,
    discover_cli_tests,
    discover_specs,
    emit_from_python,
    lower_to_taytsh,
    run_cli,
    run_fields,
    run_hierarchy,
    run_names,
    run_parse,
    run_pycheck,
    run_sigs,
    run_subset,
)

# fmt: off
TESTS = {
    "cli":       {"dir": "frontend/cli",        "run": "cli"},
    "parse":     {"dir": "frontend/parse",      "run": "phase"},
    "subset":    {"dir": "frontend/subset",     "run": "phase"},
    "names":     {"dir": "frontend/names",      "run": "phase"},
    "sigs":      {"dir": "frontend/signatures", "run": "phase"},
    "fields":    {"dir": "frontend/fields",     "run": "phase"},
    "hierarchy": {"dir": "frontend/hierarchy",  "run": "phase"},
    "pycheck":   {"dir": "frontend/pycheck",    "run": "phase"},
    "lowering":  {"dir": "frontend/lowering",   "run": "lowering"},
}
# fmt: on


def pytest_generate_tests(metafunc):
    for name, cfg in TESTS.items():
        test_dir = TESTS_DIR / cfg["dir"]
        run = cfg["run"]
        if run == "cli" and "cli_spec" in metafunc.fixturenames:
            tests = discover_cli_tests(test_dir)
            params = [pytest.param(spec, id=tid) for tid, spec in tests]
            metafunc.parametrize("cli_spec", params)
        elif run in ("phase", "lowering"):
            fixture = f"{name}_input"
            if fixture in metafunc.fixturenames:
                specs = discover_specs(test_dir, cfg.get("glob", "*.tests"))
                params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in specs]
                metafunc.parametrize(f"{fixture},{name}_expected", params)


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


@pytest.mark.parametrize(
    "source",
    [
        # Parameter named 'len' but builtin len() never called in body
        pytest.param(
            "def f(len: int) -> int:\n    return len\n",
            id="len-not-called-as-builtin",
        ),
        # Parameter named 'range' used as a plain variable
        pytest.param(
            "def f(range: int) -> int:\n    return range + 1\n",
            id="range-not-called-as-builtin",
        ),
        # Multiple params shadow builtins but none used as builtins
        pytest.param(
            "def f(type: str, len: int) -> str:\n    return type\n",
            id="multiple-shadows-none-used-as-builtin",
        ),
    ],
)
def test_param_shadow_no_warning_when_builtin_unused(source: str) -> None:
    """Parameter shadowing a builtin should not warn when the builtin is never called."""
    result = run_names(source)
    assert not result.errors, f"unexpected errors: {result.errors}"
    assert not result.warnings, f"unexpected warnings: {result.warnings}"


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


# ---------------------------------------------------------------------------
# Regression tests for pycheck gaps exposed by fallback engine removal
# ---------------------------------------------------------------------------


def test_empty_set_in_dictcomp() -> None:
    """Dict comp with set() value should infer map[int, set[int]], not map[int, string]."""
    source = (
        "def f(pairs: list[tuple[int, str]]) -> None:\n"
        "    deps: dict[int, set[int]] = {idx: set() for idx, _ in pairs}\n"
        "    _ = deps\n"
    )
    output, err = lower_to_taytsh(source)
    assert err is None, f"lowering error: {err}"
    assert output is not None
    assert "map[int, string]" not in output, (
        "dict comp value typed as string instead of set[int]"
    )


def test_tuple_full_slice() -> None:
    """t[:] on a fixed tuple should return the tuple type, not the element type."""
    source = (
        "def f(t: tuple[int, int, int]) -> tuple[int, int, int]:\n    return t[:]\n"
    )
    out = run_pycheck(source)
    assert not out.errors, f"unexpected errors: {out.errors}"


def test_isinstance_loop_var_reassignment() -> None:
    """Loop variable reassigned after isinstance narrowing should lower correctly (#217)."""
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class A:\n"
        "    x: int\n"
        "@dataclass\n"
        "class B:\n"
        "    y: int\n"
        "def convert(b: B) -> A:\n"
        "    return A(b.y)\n"
        "def f(items: list[A | B]) -> int:\n"
        "    total: int = 0\n"
        "    for item in items:\n"
        "        if isinstance(item, B):\n"
        "            item = convert(item)\n"
        "        total += item.x\n"
        "    return total\n"
        "def main() -> None:\n"
        "    print(f([A(1), B(2)]))\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    output, err = emit_from_python(source, "python")
    assert err is None, f"emit error: {err}"
    assert output is not None


def test_isinstance_loop_var_reassignment_with_accumulator() -> None:
    """Loop var reassignment + accumulator list used after loop should work (#217)."""
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class VList:\n"
        "    elements: list[int]\n"
        "@dataclass\n"
        "class VBytes:\n"
        "    data: bytes\n"
        "def coerce(a: VBytes) -> VList:\n"
        "    return VList([])\n"
        "def f(args: list[VList | VBytes]) -> int:\n"
        "    lists: list[VList] = []\n"
        "    for a in args:\n"
        "        if isinstance(a, VBytes):\n"
        "            a = coerce(a)\n"
        "        lists.append(a)\n"
        "    total: int = 0\n"
        "    for l in lists:\n"
        "        total += len(l.elements)\n"
        "    return total\n"
        "def main() -> None:\n"
        "    pass\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    output, err = emit_from_python(source, "python")
    assert err is None, f"emit error: {err}"
    assert output is not None


def test_emit_stamp_uids() -> None:
    """emit_from_python must stamp UIDs so pycheck types are available to lowering."""
    source = (
        "def main() -> None:\n"
        "    s: str = 'hello'\n"
        "    _ = s.strip()\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    output, err = emit_from_python(source, "python")
    assert err is None, f"emit error: {err}"
    assert output is not None
