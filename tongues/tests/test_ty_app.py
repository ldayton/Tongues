"""Taytsh app tests: .ty programs on the treewalker."""

import subprocess
from pathlib import Path

import pytest

from tests.harness import (
    TESTS_DIR,
    TRANSPILED_BINARY,
    _TRANSPILED_MODULE,
    _run_inprocess,
    _transpiled_runtime,
    discover_taytsh_apps,
    taytsh_parse,
    taytsh_run,
)

TESTS = {
    "taytsh_app": {"dir": "23_ty_app", "run": "taytsh_app"},
}


def pytest_generate_tests(metafunc):
    for name, cfg in TESTS.items():
        test_dir = TESTS_DIR / cfg["dir"]
        if cfg["run"] == "taytsh_app" and "taytsh_app" in metafunc.fixturenames:
            apps = discover_taytsh_apps(test_dir)
            params = [pytest.param(p, id=p.stem) for p in apps]
            metafunc.parametrize("taytsh_app", params)


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
