"""Pytest configuration for Tongues test suite."""

import os
import sys
from pathlib import Path

# Add tongues directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add src directory for lib.* imports (used by tests.shared.test_harness)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_addoption(parser):
    parser.addoption(
        "--target",
        action="append",
        help="Target language(s) for app tests (repeatable)",
    )
    parser.addoption(
        "--transpiled",
        help="Path to transpiled binary (e.g. .out/tongues.py)",
    )


def pytest_configure(config):
    path = config.getoption("transpiled")
    if path is not None:
        resolved = str(Path(path).resolve())
        if not Path(resolved).is_file():
            raise FileNotFoundError("--transpiled binary not found: " + resolved)
        os.environ["TONGUES_TRANSPILED_BINARY"] = resolved
