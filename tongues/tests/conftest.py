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
