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

# App tests with known failures — xfail until the underlying issues are fixed.
# Each entry is "stem[target]" matching the test id.
KNOWN_FAILURES: set[str] = {
    "apptest_base64[javascript]", "apptest_base64[ruby]",
    "apptest_bits[perl]",
    "apptest_bitset[javascript]",
    "apptest_bools[javascript]", "apptest_bools[perl]", "apptest_bools[python]", "apptest_bools[ruby]",
    "apptest_bytearray[python]",
    "apptest_bytes[javascript]", "apptest_bytes[perl]", "apptest_bytes[python]", "apptest_bytes[ruby]",
    "apptest_dicts[javascript]", "apptest_dicts[perl]", "apptest_dicts[python]", "apptest_dicts[ruby]",
    "apptest_enums[javascript]", "apptest_enums[perl]", "apptest_enums[python]", "apptest_enums[ruby]",
    "apptest_floats[javascript]", "apptest_floats[perl]", "apptest_floats[ruby]",
    "apptest_ints[perl]",
    "apptest_json[javascript]", "apptest_json[perl]", "apptest_json[python]", "apptest_json[ruby]",
    "apptest_lists[javascript]", "apptest_lists[perl]", "apptest_lists[python]", "apptest_lists[ruby]",
    "apptest_none[javascript]",
    "apptest_sets[javascript]", "apptest_sets[perl]", "apptest_sets[python]", "apptest_sets[ruby]",
    "apptest_sha256[javascript]", "apptest_sha256[perl]", "apptest_sha256[ruby]",
    "apptest_softfloat[javascript]", "apptest_softfloat[perl]",
    "apptest_string_unicode_param[python]",
    "apptest_strings[javascript]", "apptest_strings[perl]", "apptest_strings[ruby]",
    "apptest_sub_interface_field[javascript]", "apptest_sub_interface_field[perl]",
    "apptest_sub_interface_field[python]", "apptest_sub_interface_field[ruby]",
    "apptest_tuples[javascript]", "apptest_tuples[perl]", "apptest_tuples[python]", "apptest_tuples[ruby]",
    "apptest_utf8[javascript]", "apptest_utf8[ruby]",
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
            params = []
            for tid, path, target in tests:
                marks = []
                if tid in KNOWN_FAILURES:
                    marks.append(
                        pytest.mark.xfail(reason="known failure", strict=False)
                    )
                params.append(pytest.param(path, target, id=tid, marks=marks))
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
