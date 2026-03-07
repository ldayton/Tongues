"""Target test phase: app execution + ordering execution."""

from pathlib import Path

import pytest

from tests.harness import (
    TESTS_DIR,
    _available_targets,
    discover_app_tests,
    run_target_code,
    transpile_app,
    transpile_code,
)

# fmt: off
TESTS = {
    "app":      {"dir": "backend/app",      "run": "app"},
    "ordering": {"dir": "backend/ordering", "run": "ordering"},
}
# fmt: on


def pytest_generate_tests(metafunc):
    for name, cfg in TESTS.items():
        test_dir = TESTS_DIR / cfg["dir"]
        run = cfg["run"]
        if run == "app" and "app_source" in metafunc.fixturenames:
            target_opt = metafunc.config.getoption("--target", default=None)
            targets = target_opt if target_opt else _available_targets()
            tests = discover_app_tests(test_dir, targets)
            params = [pytest.param(path, target, id=tid) for tid, path, target in tests]
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


@pytest.mark.timeout(30)
def test_app(app_source: Path, app_target: str) -> None:
    source = app_source.read_text()
    output, err = transpile_app(source, app_target)
    if err is not None:
        pytest.fail(f"Transpile error ({app_target}): {err}")
    result = run_target_code(output, app_target, timeout=30)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        stdout = result.stdout.decode(errors="replace")
        pytest.fail(
            f"App test failed with exit {result.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )


def test_ordering(ordering_source: Path, ordering_target: str) -> None:
    source = ordering_source.read_text()
    output, err = transpile_code(source, ordering_target)
    if err is not None:
        pytest.fail(f"Transpile error ({ordering_target}): {err}")
    result = run_target_code(output, ordering_target, timeout=30)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        stdout = result.stdout.decode(errors="replace")
        pytest.fail(
            f"Ordering test failed with exit {result.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
