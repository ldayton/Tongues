"""Target test phase: app execution + ordering execution."""

import re
import subprocess
from pathlib import Path

import pytest

from tests.harness import (
    RUNTIMES,
    TESTS_DIR,
    discover_app_tests,
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
            targets = target_opt if target_opt else sorted(RUNTIMES)
            tests = discover_app_tests(test_dir, targets)
            params = [pytest.param(path, target, id=tid) for tid, path, target in tests]
            metafunc.parametrize("app_source,app_target", params)
        elif run == "ordering" and "ordering_source" in metafunc.fixturenames:
            target_opt = metafunc.config.getoption("--target", default=None)
            targets = target_opt if target_opt else sorted(RUNTIMES)
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
    runtime = RUNTIMES[app_target]
    result = subprocess.run(
        runtime,
        input=output.encode(),
        capture_output=True,
        timeout=30,
    )
    stdout = result.stdout.decode(errors="replace")
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        pytest.fail(
            f"App test failed with exit {result.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    match = re.search(r"(\d+) passed, (\d+) failed", stdout)
    if match is None:
        pytest.fail(f"No test summary in output ({app_target}):\n{stdout}")
    passed, failed = int(match.group(1)), int(match.group(2))
    if passed == 0:
        pytest.fail(f"No tests ran ({app_target}): 0 passed\n{stdout}")
    if failed > 0:
        pytest.fail(f"{failed} test(s) failed ({app_target}):\n{stdout}")


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
