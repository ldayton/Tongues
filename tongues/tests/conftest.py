"""Pytest configuration for Tongues test suite."""

import sys
from pathlib import Path

# Add tongues directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_addoption(parser):
    parser.addoption(
        "--target",
        action="append",
        default=None,
        help="Target language(s) for app tests (repeatable)",
    )
    parser.addoption(
        "--transpiled",
        default=None,
        help="Path to transpiled binary (e.g. .out/tongues.py)",
    )


def pytest_configure(config):
    transpiled = config.getoption("--transpiled", default=None)
    if transpiled is not None:
        import tests.test_runner as runner

        runner.TRANSPILED_BINARY = transpiled
