"""Middleend test phase: typarse, tycheck, scope, returns, liveness, strings, hoisting, ownership, callgraph."""

import pytest

from tests.harness import (
    TESTS_DIR,
    check_expected,
    discover_specs,
    run_callgraph,
    run_hoisting,
    run_liveness,
    run_ownership,
    run_returns,
    run_scope,
    run_strings,
    run_taytsh_check,
    run_taytsh_parse,
    run_type_checking,
)

# fmt: off
TESTS = {
    "taytsh_parse":  {"dir": "11_typarse",  "run": "phase"},
    "taytsh_check":  {"dir": "12_tycheck",  "run": "phase", "glob": "[!l]*.tests"},
    "type_checking": {"dir": "12_tycheck",  "run": "phase", "glob": "lowered_*.tests"},
    "scope":         {"dir": "14_scope",    "run": "phase"},
    "returns":       {"dir": "15_returns",  "run": "phase"},
    "liveness":      {"dir": "16_liveness", "run": "phase"},
    "strings":       {"dir": "17_strings",  "run": "phase"},
    "hoisting":      {"dir": "18_hoisting", "run": "phase"},
    "ownership":     {"dir": "19_ownership","run": "phase"},
    "callgraph":     {"dir": "20_callgraph","run": "phase"},
}
# fmt: on


def pytest_generate_tests(metafunc):
    for name, cfg in TESTS.items():
        test_dir = TESTS_DIR / cfg["dir"]
        fixture = f"{name}_input"
        if fixture in metafunc.fixturenames:
            specs = discover_specs(test_dir, cfg.get("glob", "*.tests"))
            params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in specs]
            metafunc.parametrize(f"{fixture},{name}_expected", params)


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


def test_type_checking(type_checking_input, type_checking_expected):
    check_expected(
        type_checking_expected,
        run_type_checking(type_checking_input),
        "type_checking",
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
