"""Pytest-based apptest runner for Tongues transpiler."""

import subprocess
from pathlib import Path

import pytest


def test_apptest(apptest: Path, target, executable: list[str]):
    """Run a transpiled apptest and verify it executes successfully."""
    try:
        result = subprocess.run(executable, capture_output=True, text=True, timeout=3)
    except subprocess.TimeoutExpired:
        pytest.fail("Test timed out after 3 seconds")
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        pytest.fail(f"Test failed with exit code {result.returncode}:\n{output}")
