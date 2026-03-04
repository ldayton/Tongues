"""Cross-lib import test module — imports from lib.levenshtein."""

from lib.levenshtein import levenshtein


def edit_distance(a: str, b: str) -> int:
    """Compute edit distance via cross-lib import."""
    return levenshtein(a, b)
