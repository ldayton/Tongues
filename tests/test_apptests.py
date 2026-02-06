"""Pytest-based apptest runner for Tongues transpiler."""

import subprocess
from pathlib import Path

import pytest


def test_apptest(
    apptest: Path, target, executable: list[str], run_env: dict[str, str] | None
):
    """Run a transpiled apptest and verify it executes successfully."""
    try:
        result = subprocess.run(
            executable, capture_output=True, text=True, timeout=10, env=run_env
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Test timed out after 10 seconds")
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        pytest.fail(f"Test failed with exit code {result.returncode}:\n{output}")
