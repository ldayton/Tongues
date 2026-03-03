"""Lang-taytsh: round-trip app/ordering/ty_app through emit_taytsh then run."""

from pathlib import Path

import pytest

from src.backend.taytsh import emit_taytsh
from src.taytsh import parse as taytsh_parse
from src.taytsh.treewalker import run as taytsh_run
from src.taytsh.vm import vm_run
from tests.harness import TESTS_DIR, lower_to_taytsh, _transpile_with_emitter

RUNNERS = ["treewalker", "vm"]

# fmt: off
TESTS = {
    "app":      {"dir": "backend/app",      "run": "app"},
    "ordering": {"dir": "backend/ordering", "run": "ordering"},
    "ty_app":   {"dir": "taytsh/app",       "run": "ty_app"},
}

_APP_XFAIL_TREEWALKER: set[str] = set()
_APP_XFAIL_VM: set[str] = set()
# fmt: on


def _round_trip(ty_text: str) -> str:
    """Emit taytsh source through the backend, returning the emitted text."""
    output, err = _transpile_with_emitter(ty_text, emit_taytsh)
    if err is not None:
        pytest.fail(f"Taytsh emit error: {err}")
    return output


def _run_taytsh(ty_text: str, runner: str) -> tuple[int, str]:
    """Parse emitted taytsh and run through treewalker or VM."""
    module = taytsh_parse(ty_text)
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
    emitted = _round_trip(ty_text)
    exit_code, output = _run_taytsh(emitted, runner)
    if exit_code != 0:
        pytest.fail(f"Exit code {exit_code}:\n{output}")


def test_ordering(ordering_source: Path, runner: str) -> None:
    source = ordering_source.read_text()
    emitted = _round_trip(source)
    exit_code, output = _run_taytsh(emitted, runner)
    if exit_code != 0:
        pytest.fail(f"Exit code {exit_code}:\n{output}")


def test_ty_app(ty_app: Path, runner: str) -> None:
    source = ty_app.read_text()
    emitted = _round_trip(source)
    exit_code, output = _run_taytsh(emitted, runner)
    if exit_code != 0:
        pytest.fail(f"Exit code {exit_code}:\n{output}")
