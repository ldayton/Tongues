"""Taytsh test phase: typarse, tycheck."""

import pytest

from tests.harness import (
    TESTS_DIR,
    check_expected,
    discover_specs,
    run_tycheck,
    run_typarse,
)

# fmt: off
TESTS = {
    "typarse": {"dir": "11_typarse", "run": "phase"},
    "tycheck": {"dir": "12_tycheck", "run": "phase"},
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


def test_typarse(typarse_input, typarse_expected):
    check_expected(
        typarse_expected,
        run_typarse(typarse_input),
        "typarse",
        lenient_errors=True,
    )


def test_tycheck(tycheck_input, tycheck_expected):
    check_expected(
        tycheck_expected,
        run_tycheck(tycheck_input),
        "tycheck",
        lenient_errors=True,
    )
