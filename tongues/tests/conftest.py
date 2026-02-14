"""Pytest configuration for Tongues test suite."""

import sys
from pathlib import Path

# Add tongues directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))
