"""Middleend test phase: scope, returns, liveness, strings, hoisting, ownership, callgraph."""

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
)

# fmt: off
TESTS = {
    "scope":     {"dir": "middleend/scope",    "run": "phase"},
    "returns":   {"dir": "middleend/returns",  "run": "phase"},
    "liveness":  {"dir": "middleend/liveness", "run": "phase"},
    "strings":   {"dir": "middleend/strings",  "run": "phase"},
    "hoisting":  {"dir": "middleend/hoisting", "run": "phase"},
    "ownership": {"dir": "middleend/ownership","run": "phase"},
    "callgraph": {"dir": "middleend/callgraph","run": "phase"},
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
