"""Cross-validate builtin dispatch sites against BUILTIN_NAMES."""

import re
from pathlib import Path

from src.taytsh.check import BUILTIN_NAMES
from src.taytsh.compiler import BUILTIN_TABLE

SRC = Path(__file__).resolve().parent.parent / "src" / "taytsh"


def _extract_dispatched_names(source: str, func_name: str) -> set[str]:
    """Extract builtin names from a dispatch function's if/elif chain."""
    lines = source.splitlines()
    # Find the 'def func_name(' line
    start = None
    for i, line in enumerate(lines):
        if re.search(rf"\bdef {func_name}\b", line):
            start = i
            break
    assert start is not None, f"could not find {func_name}"
    # Determine the indentation of the def line
    def_indent = len(lines[start]) - len(lines[start].lstrip())
    # Scan forward for the body; end at next def/class at same or lesser indent
    body_lines: list[str] = []
    for i in range(start + 1, len(lines)):
        line = lines[i]
        stripped = line.lstrip()
        if not stripped:
            continue
        cur_indent = len(line) - len(stripped)
        if cur_indent <= def_indent and (
            stripped.startswith("def ") or stripped.startswith("class ")
        ):
            break
        body_lines.append(line)
    body = "\n".join(body_lines)
    names: set[str] = set()
    names.update(re.findall(r'name == "(\w+)"', body))
    for m in re.finditer(r"name in \(([^)]+)\)", body):
        names.update(re.findall(r'"(\w+)"', m.group(1)))
    return names


def test_check_builtin_call_matches_builtin_names() -> None:
    source = (SRC / "check.py").read_text()
    dispatched = _extract_dispatched_names(source, "check_builtin_call")
    assert dispatched == BUILTIN_NAMES, (
        f"check_builtin_call vs BUILTIN_NAMES mismatch:\n"
        f"  missing handlers: {BUILTIN_NAMES - dispatched}\n"
        f"  extra handlers:   {dispatched - BUILTIN_NAMES}"
    )


def test_builtin_table_matches_builtin_names() -> None:
    assert set(BUILTIN_TABLE) == BUILTIN_NAMES, (
        f"BUILTIN_TABLE vs BUILTIN_NAMES mismatch:\n"
        f"  missing from table: {BUILTIN_NAMES - set(BUILTIN_TABLE)}\n"
        f"  extra in table:     {set(BUILTIN_TABLE) - BUILTIN_NAMES}"
    )
