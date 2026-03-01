"""Lang-taytsh: run app/ordering/ty_app tests through treewalker and VM."""

from pathlib import Path

import pytest

from src.taytsh import parse as taytsh_parse
from src.taytsh.treewalker import run as taytsh_run
from src.taytsh.vm import vm_run
from tests.harness import TESTS_DIR, lower_to_taytsh

RUNNERS = ["treewalker", "vm"]

# fmt: off
TESTS = {
    "app":      {"dir": "backend/app",      "run": "app"},
    "ordering": {"dir": "backend/ordering", "run": "ordering"},
    "ty_app":   {"dir": "taytsh/app",       "run": "ty_app"},
}

# Lowered app tests that hit known runtime gaps.
# treewalker: fn-literal call dispatch not yet supported
# vm: IndexError in _b64decode_impl stack (#TBD), collection lowering
_APP_XFAIL_TREEWALKER = {
    "apptest_bytes", "apptest_dicts", "apptest_floats",
    "apptest_lists", "apptest_none", "apptest_sets",
    "apptest_truthiness", "apptest_tuples",
}
_APP_XFAIL_VM = {"apptest_base64", "apptest_dicts", "apptest_lists", "apptest_tuples"}
# fmt: on


def _run_module(module, runner: str) -> tuple[int, str]:
    """Run a parsed module and return (exit_code, combined_output)."""
    if runner == "treewalker":
        result = taytsh_run(module)
        output = (result.stdout + result.stderr).decode(errors="replace").strip()
        return result.exit_code, output
    result = vm_run(module)
    output = (result.stdout + result.stderr).strip()
    return result.exit_code, output


def pytest_generate_tests(metafunc):
    for _name, cfg in TESTS.items():
        test_dir = TESTS_DIR / cfg["dir"]
        run = cfg["run"]
        if run == "app" and "app_source" in metafunc.fixturenames:
            apps = sorted(test_dir.glob("apptest_*.py"))
            params = []
            for path in apps:
                for runner in RUNNERS:
                    xfails = (
                        _APP_XFAIL_TREEWALKER
                        if runner == "treewalker"
                        else _APP_XFAIL_VM
                    )
                    marks = (
                        [pytest.mark.xfail(strict=True)] if path.stem in xfails else []
                    )
                    params.append(
                        pytest.param(
                            path, runner, id=f"{path.stem}[{runner}]", marks=marks
                        )
                    )
            metafunc.parametrize("app_source,runner", params)
        elif run == "ordering" and "ordering_source" in metafunc.fixturenames:
            ty_files = sorted(test_dir.glob("*.ty"))
            params = []
            for path in ty_files:
                for runner in RUNNERS:
                    params.append(
                        pytest.param(path, runner, id=f"{path.stem}[{runner}]")
                    )
            metafunc.parametrize("ordering_source,runner", params)
        elif run == "ty_app" and "ty_app" in metafunc.fixturenames:
            ty_files = sorted(test_dir.glob("*.ty"))
            params = []
            for path in ty_files:
                for runner in RUNNERS:
                    params.append(
                        pytest.param(path, runner, id=f"{path.stem}[{runner}]")
                    )
            metafunc.parametrize("ty_app,runner", params)


def test_app(app_source: Path, runner: str) -> None:
    source = app_source.read_text()
    ty_text, err = lower_to_taytsh(source)
    if err is not None:
        pytest.fail(f"Lowering error: {err}")
    module = taytsh_parse(ty_text)
    exit_code, output = _run_module(module, runner)
    if exit_code != 0:
        pytest.fail(f"Exit code {exit_code}:\n{output}")


def test_ordering(ordering_source: Path, runner: str) -> None:
    source = ordering_source.read_text()
    module = taytsh_parse(source)
    exit_code, output = _run_module(module, runner)
    if exit_code != 0:
        pytest.fail(f"Exit code {exit_code}:\n{output}")


def test_ty_app(ty_app: Path, runner: str) -> None:
    source = ty_app.read_text()
    module = taytsh_parse(source)
    exit_code, output = _run_module(module, runner)
    if exit_code != 0:
        pytest.fail(f"Exit code {exit_code}:\n{output}")
