"""Linker test phase: multi-file project merge via --project."""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.harness import (
    TESTS_DIR,
    TONGUES_DIR,
    check_cli_assertions,
    discover_linker_tests,
    run_linker_test,
)

TESTS = {
    "linker": {"dir": "frontend/linker", "run": "linker"},
}

FIXTURES = TESTS_DIR / "frontend" / "merge" / "fixtures"


def pytest_generate_tests(metafunc):
    for name, cfg in TESTS.items():
        test_dir = TESTS_DIR / cfg["dir"]
        if cfg["run"] == "linker" and "linker_spec" in metafunc.fixturenames:
            tests = discover_linker_tests(test_dir)
            params = [pytest.param(spec, id=tid) for tid, spec in tests]
            metafunc.parametrize("linker_spec", params)


def test_linker(linker_spec: dict) -> None:
    result = run_linker_test(linker_spec)
    err = check_cli_assertions(result, linker_spec["assertions"])
    if err is not None:
        pytest.fail(err)


class TestGatherProjectFiles:
    """Tests for directory-based project gathering (bin/tongues <dir>)."""

    def _run_bin(self, fixture_dir, extra_args=None):
        args = [sys.executable, str(TONGUES_DIR / "bin" / "tongues")]
        if extra_args:
            args.extend(extra_args)
        args.append(str(fixture_dir))
        return subprocess.run(args, capture_output=True, cwd=TONGUES_DIR)

    def test_skips_hidden_files(self):
        result = self._run_bin(FIXTURES / "gather_test", ["--stop-at", "parse"])
        output = result.stdout.decode()
        assert ".hidden_file.py" not in output

    def test_skips_pycache(self):
        result = self._run_bin(FIXTURES / "gather_test", ["--stop-at", "parse"])
        output = result.stdout.decode()
        assert "cached" not in output

    def test_skips_tongues_skip(self):
        result = self._run_bin(FIXTURES / "gather_test", ["--stop-at", "parse"])
        output = result.stdout.decode()
        assert "skipped" not in output

    def test_includes_visible(self):
        result = self._run_bin(FIXTURES / "gather_test", ["--stop-at", "parse"])
        assert result.returncode == 0, result.stderr.decode()
        output = result.stdout.decode()
        assert "visible" in output

    def test_empty_dir(self):
        result = self._run_bin(FIXTURES / "empty", ["--stop-at", "subset"])
        assert result.returncode != 0
        assert b"no .py files" in result.stderr

    def test_skip_fixture_only_has_b(self):
        result = self._run_bin(FIXTURES / "skip", ["--stop-at", "parse"])
        assert result.returncode == 0, result.stderr.decode()
        output = result.stdout.decode()
        assert "b.py" in output
        assert "requests" not in output
