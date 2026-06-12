"""Lang-taytsh: round-trip app/ordering/ty_app through emit_taytsh then run."""

from pathlib import Path

import pytest

from src.backend.taytsh import emit_taytsh
from src.taytsh import parse as taytsh_parse
from src.taytsh.vm import vm_run
from tests.harness import (
    TESTS_DIR,
    load_known_failures,
    lower_to_taytsh,
    _transpile_with_emitter,
)

# fmt: off
TESTS = {
    "app":      {"dir": "backend/app",      "run": "app"},
    "ordering": {"dir": "backend/ordering", "run": "ordering"},
    "ty_app":   {"dir": "taytsh/app",       "run": "ty_app"},
}

_APP_XFAIL_VM: set[str] = set()
# fmt: on


def _round_trip(ty_text: str) -> str:
    """Emit taytsh source through the backend, returning the emitted text."""
    output, err = _transpile_with_emitter(ty_text, emit_taytsh, "taytsh")
    if err is not None:
        pytest.fail(f"Taytsh emit error: {err}")
    return output


def _run_taytsh(ty_text: str) -> tuple[int, str]:
    """Parse emitted taytsh and run through the VM."""
    module = taytsh_parse(ty_text)
    result = vm_run(module)
    output = (result.stdout + result.stderr).strip()
    return result.exit_code, output


def pytest_generate_tests(metafunc):
    for _name, cfg in TESTS.items():
        test_dir = TESTS_DIR / cfg["dir"]
        run = cfg["run"]
        if run == "app" and "app_source" in metafunc.fixturenames:
            apps = sorted(test_dir.glob("apptest_*.py"))
            known = load_known_failures(test_dir)
            params = []
            for path in apps:
                entry = known.get((path.stem, "taytsh"))
                if path.stem in _APP_XFAIL_VM:
                    marks = [pytest.mark.xfail(strict=True)]
                elif entry is not None:
                    marks = [pytest.mark.xfail(strict=entry[1], reason=entry[0])]
                else:
                    marks = []
                params.append(pytest.param(path, id=path.stem, marks=marks))
            metafunc.parametrize("app_source", params)
        elif run == "ordering" and "ordering_source" in metafunc.fixturenames:
            ty_files = sorted(test_dir.glob("*.ty"))
            params = [pytest.param(path, id=path.stem) for path in ty_files]
            metafunc.parametrize("ordering_source", params)
        elif run == "ty_app" and "ty_app" in metafunc.fixturenames:
            ty_files = sorted(test_dir.glob("*.ty"))
            params = [pytest.param(path, id=path.stem) for path in ty_files]
            metafunc.parametrize("ty_app", params)


def test_app(app_source: Path) -> None:
    source = app_source.read_text()
    ty_text, err = lower_to_taytsh(source)
    if err is not None:
        pytest.fail(f"Lowering error: {err}")
    emitted = _round_trip(ty_text)
    exit_code, output = _run_taytsh(emitted)
    if exit_code != 0:
        pytest.fail(f"Exit code {exit_code}:\n{output}")


def test_ordering(ordering_source: Path) -> None:
    source = ordering_source.read_text()
    emitted = _round_trip(source)
    exit_code, output = _run_taytsh(emitted)
    if exit_code != 0:
        pytest.fail(f"Exit code {exit_code}:\n{output}")


def test_ty_app(ty_app: Path) -> None:
    source = ty_app.read_text()
    emitted = _round_trip(source)
    exit_code, output = _run_taytsh(emitted)
    if exit_code != 0:
        pytest.fail(f"Exit code {exit_code}:\n{output}")
