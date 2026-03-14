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
    parser.addoption(
        "--timeout-override",
        type=int,
        default=None,
        help="Override all @pytest.mark.timeout markers with this value (seconds)",
    )


def pytest_collection_modifyitems(config, items):
    override = config.getoption("timeout_override")
    if override is None:
        return
    import pytest

    for item in items:
        if item.get_closest_marker("timeout"):
            item.own_markers = [m for m in item.own_markers if m.name != "timeout"]
            item.add_marker(pytest.mark.timeout(override))


def pytest_configure(config):
    path = config.getoption("transpiled")
    if path is not None:
        resolved = str(Path(path).resolve())
        if not Path(resolved).is_file():
            raise FileNotFoundError("--transpiled binary not found: " + resolved)
        os.environ["TONGUES_TRANSPILED_BINARY"] = resolved
