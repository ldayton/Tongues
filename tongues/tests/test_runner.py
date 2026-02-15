"""Test runner for Tongues test phases."""

import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.frontend.fields import collect_fields
from src.frontend.hierarchy import build_hierarchy
from src.frontend.inference import run_inference as _run_inference
from src.frontend.names import resolve_names
from src.frontend.parse import parse
from src.frontend.signatures import collect_signatures
from src.frontend.subset import verify as verify_subset

TONGUES_DIR = Path(__file__).parent.parent


def parse_cli_test_file(path: Path) -> list[tuple[str, dict]]:
    """Parse a CLI .tests file into (name, spec) tuples."""
    lines = path.read_text().split("\n")
    result: list[tuple[str, dict]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("=== "):
            test_name = line[4:].strip()
            i += 1
            input_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                input_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            expected_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                expected_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            spec = _parse_cli_spec(input_lines, expected_lines)
            result.append((test_name, spec))
        else:
            i += 1
    return result


def _parse_cli_spec(input_lines: list[str], expected_lines: list[str]) -> dict:
    """Parse input + expected lines into a CLI test spec dict."""
    spec: dict = {"args": [], "stdin": None, "stdin_bytes": None, "assertions": []}
    body_start = 0
    if input_lines and input_lines[0].startswith("args:"):
        args_str = input_lines[0][5:].strip()
        spec["args"] = args_str.split() if args_str else []
        body_start = 1
    remaining = input_lines[body_start:]
    if remaining and remaining[0].startswith("stdin-bytes:"):
        hex_str = remaining[0][len("stdin-bytes:") :].strip()
        spec["stdin_bytes"] = bytes.fromhex(hex_str)
    else:
        spec["stdin"] = "\n".join(remaining)
    for line in expected_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("exit:"):
            spec["assertions"].append(("exit", int(line[5:].strip())))
        elif line.startswith("exit-not:"):
            spec["assertions"].append(("exit-not", int(line[9:].strip())))
        elif line.startswith("stderr:"):
            spec["assertions"].append(("stderr", line[7:].strip()))
        elif line.startswith("stderr-contains:"):
            spec["assertions"].append(("stderr-contains", line[16:].strip()))
        elif line.startswith("stderr-empty:"):
            spec["assertions"].append(("stderr-empty", None))
        elif line.startswith("stdout-contains:"):
            spec["assertions"].append(("stdout-contains", line[16:].strip()))
        elif line.startswith("stdout-empty:"):
            spec["assertions"].append(("stdout-empty", None))
    return spec


def discover_cli_tests(test_dir: Path) -> list[tuple[str, dict]]:
    """Find all CLI tests across .tests files."""
    results = []
    for test_file in sorted(test_dir.glob("*.tests")):
        for name, spec in parse_cli_test_file(test_file):
            results.append((f"{test_file.stem}/{name}", spec))
    return results


def run_cli(spec: dict) -> subprocess.CompletedProcess[bytes]:
    """Run tongues CLI from a test spec."""
    cmd = [sys.executable, "-m", "src.tongues", *spec["args"]]
    if spec["stdin_bytes"] is not None:
        stdin_data = spec["stdin_bytes"]
    elif spec["stdin"] is not None:
        stdin_data = spec["stdin"].encode()
    else:
        stdin_data = b""
    return subprocess.run(cmd, input=stdin_data, capture_output=True, cwd=TONGUES_DIR)


def check_cli_assertions(
    result: subprocess.CompletedProcess[bytes], assertions: list[tuple]
) -> None:
    """Check all assertions against a CLI result."""
    for kind, value in assertions:
        if kind == "exit":
            assert result.returncode == value, (
                f"expected exit {value}, got {result.returncode}"
                f"\nstderr: {result.stderr.decode(errors='replace')}"
            )
        elif kind == "exit-not":
            assert result.returncode != value, (
                f"expected exit != {value}, got {result.returncode}"
            )
        elif kind == "stderr":
            actual = result.stderr.decode(errors="replace").rstrip("\n")
            assert actual == value, f"expected stderr {value!r}, got {actual!r}"
        elif kind == "stderr-contains":
            actual = result.stderr.decode(errors="replace")
            assert value in actual, (
                f"expected stderr to contain {value!r}, got {actual!r}"
            )
        elif kind == "stderr-empty":
            assert result.stderr == b"", f"expected empty stderr, got {result.stderr!r}"
        elif kind == "stdout-contains":
            actual = result.stdout.decode(errors="replace")
            assert value in actual, (
                f"expected stdout to contain {value!r}, got {actual!r}"
            )
        elif kind == "stdout-empty":
            assert result.stdout == b"", (
                f"expected empty stdout, got {result.stdout[:200]!r}"
            )


from src.backend.perl import emit_perl as emit_perl
from src.backend.python import emit_python as emit_python
from src.backend.ruby import emit_ruby as emit_ruby
from src.middleend.callgraph import analyze_callgraph
from src.middleend.hoisting import analyze_hoisting
from src.middleend.liveness import analyze_liveness
from src.middleend.ownership import analyze_ownership
from src.middleend.returns import analyze_returns
from src.middleend.scope import analyze_scope
from src.middleend.strings import analyze_strings
from src.taytsh import check as taytsh_check_fn, parse as taytsh_parse
from src.taytsh.runtime import run as taytsh_run
from src.taytsh.ast import (
    TCall,
    TFieldAccess,
    TFnDecl,
    TStructDecl,
    TVar,
    serialize_annotations,
)
from src.taytsh.check import StructT, check_with_info

PARSE_TIMEOUT = 5
TESTS_DIR = Path(__file__).parent

# fmt: off
TESTS = {
    "cli": {
        "cli":       {"dir": "02_cli",       "run": "cli"},
    },
    "frontend": {
        "parse":     {"dir": "03_parse",     "run": "phase"},
        "subset":    {"dir": "04_subset",    "run": "phase"},
        "names":     {"dir": "05_names",     "run": "phase"},
        "sigs":      {"dir": "06_signatures", "run": "phase"},
        "fields":    {"dir": "07_fields",    "run": "phase"},
        "hierarchy": {"dir": "08_hierarchy", "run": "phase"},
        "inference": {"dir": "09_inference", "run": "phase"},
        "lowering":  {"dir": "10_lowering",  "run": "lowering"},
    },
    "middleend": {
        "type_checking": {"dir": "13_type_checking", "run": "phase"},
        "scope":     {"dir": "14_scope",     "run": "phase"},
        "returns":   {"dir": "15_returns",   "run": "phase"},
        "liveness":  {"dir": "16_liveness",  "run": "phase"},
        "strings":   {"dir": "17_strings",   "run": "phase"},
        "hoisting":  {"dir": "18_hoisting",  "run": "phase"},
        "ownership": {"dir": "19_ownership", "run": "phase"},
        "callgraph": {"dir": "20_callgraph", "run": "phase"},
    },
    "backend": {
        "codegen":        {"dir": "21_codegen", "run": "codegen"},
        "app":            {"dir": "22_app",     "run": "app"},
    },
    "taytsh": {
        "taytsh_parse": {"dir": "11_taytsh_parse", "run": "phase"},
        "taytsh_check": {"dir": "12_taytsh_check", "run": "phase"},
        "taytsh_app":   {"dir": "23_taytsh_app",   "run": "taytsh_app"},
    },
}
# fmt: on

EMITTERS = {
    "python": emit_python,
    "perl": emit_perl,
    "ruby": emit_ruby,
}

RUNTIMES = {
    "python": [sys.executable],
    "perl": ["perl"],
    "ruby": ["ruby"],
}


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def _timeout_handler(signum, frame):
    raise TimeoutError("parse() timed out")


signal.signal(signal.SIGALRM, _timeout_handler)


# ---------------------------------------------------------------------------
# Discovery + parsing
# ---------------------------------------------------------------------------


def parse_spec_file(path: Path) -> list[tuple[str, str, str]]:
    """Parse a .tests file into (name, input, expected) tuples."""
    lines = path.read_text().split("\n")
    result: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("=== "):
            test_name = line[4:].strip()
            i += 1
            input_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                input_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            expected_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                expected_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            test_input = "\n".join(input_lines)
            expected = "\n".join(expected_lines).strip()
            result.append((test_name, test_input, expected))
        else:
            i += 1
    return result


def discover_specs(test_dir: Path) -> list[tuple[str, str, str]]:
    """Glob *.tests in test_dir, return (test_id, input, expected) tuples."""
    results = []
    for test_file in sorted(test_dir.glob("*.tests")):
        for name, input_code, expected in parse_spec_file(test_file):
            results.append((f"{test_file.stem}/{name}", input_code, expected))
    return results


def parse_simple_tests(path: Path) -> list[tuple[str, str]]:
    """Parse a file of '=== name' + content blocks into (name, content) pairs."""
    lines = path.read_text().split("\n")
    result: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("=== "):
            name = lines[i][4:].strip()
            i += 1
            content_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("=== "):
                content_lines.append(lines[i])
                i += 1
            content = "\n".join(content_lines).strip()
            result.append((name, content))
        else:
            i += 1
    return result


def discover_codegen_tests(
    test_dir: Path, lang: str
) -> list[tuple[str, str, str]]:
    """Join base/*.tests with {lang}/*.tests by name. Fails on mismatch."""
    base_dir = test_dir / "base"
    lang_dir = test_dir / lang
    results = []
    for base_file in sorted(base_dir.glob("*.tests")):
        lang_file = lang_dir / base_file.name
        base_tests = parse_simple_tests(base_file)
        if not base_tests:
            continue
        if not lang_file.exists():
            pytest.fail(f"{lang}/{base_file.name} missing")
        lang_tests = parse_simple_tests(lang_file)
        base_names = [n for n, _ in base_tests]
        lang_names = [n for n, _ in lang_tests]
        if base_names != lang_names:
            base_set, lang_set = set(base_names), set(lang_names)
            missing = base_set - lang_set
            extra = lang_set - base_set
            pytest.fail(
                f"{base_file.name}: base/lang name mismatch for {lang}\n"
                f"  missing: {missing}\n  extra: {extra}"
            )
        lang_by_name = dict(lang_tests)
        for name, source in base_tests:
            test_id = f"{base_file.stem}/{name}[{lang}]"
            results.append((test_id, source, lang_by_name[name]))
    return results


# ---------------------------------------------------------------------------
# Result + assertion checker
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict | None = None


def resolve_dotpath(obj: object, path: str) -> object:
    """Resolve a dot-separated path against a nested dict/list structure."""
    parts = path.split(".")
    current = obj
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "length":
            return len(current)
        if isinstance(current, list):
            current = current[int(part)]
            i += 1
        elif isinstance(current, dict):
            if part in current:
                current = current[part]
                i += 1
            else:
                found = False
                for j in range(i + 1, len(parts)):
                    composite = ".".join(parts[i : j + 1])
                    if composite in current:
                        current = current[composite]
                        i = j + 1
                        found = True
                        break
                if not found:
                    raise KeyError(part)
        else:
            raise KeyError(
                f"cannot traverse {type(current).__name__} with key {part!r}"
            )
    return current


def to_comparable(value: object) -> str:
    """Convert a value to its string form for comparison."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def check_expected(
    expected: str, result: PhaseResult, phase: str, *, lenient_errors: bool = False
) -> None:
    if expected == "ok":
        if result.errors:
            pytest.fail(f"Expected ok, got error: {result.errors[0]}")
        return
    if expected.startswith("error:"):
        expected_msg = expected[6:].strip()
        if not result.errors:
            pytest.fail(f"Expected error containing '{expected_msg}', got ok")
        if not lenient_errors and expected_msg:
            found = any(expected_msg.lower() in e.lower() for e in result.errors)
            if not found:
                pytest.fail(
                    f"Expected error containing '{expected_msg}', got: {result.errors}"
                )
        return
    if expected.startswith("warning:"):
        expected_msg = expected[8:].strip()
        if not result.warnings:
            pytest.fail(f"Expected warning containing '{expected_msg}', got none")
        found = any(expected_msg.lower() in w.lower() for w in result.warnings)
        if not found:
            pytest.fail(
                f"Expected warning containing '{expected_msg}', got: {result.warnings}"
            )
        return
    if result.errors:
        pytest.fail(f"{phase} failed: {result.errors[0]}")
    assert result.data is not None, f"No data returned from {phase}"
    for line in expected.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            pytest.fail(f"Bad assertion (no '='): {line}")
        path, expected_val = line.split("=", 1)
        path = path.strip()
        expected_val = expected_val.strip()
        try:
            actual = resolve_dotpath(result.data, path)
        except (KeyError, IndexError, TypeError) as e:
            pytest.fail(f"Path '{path}' not found in result: {e}")
        actual_str = to_comparable(actual)
        # Cross-reference: if RHS looks like a dotpath, resolve it too
        if "." in expected_val and " " not in expected_val:
            try:
                ref_val = resolve_dotpath(result.data, expected_val)
                expected_val = to_comparable(ref_val)
            except (KeyError, IndexError, TypeError):
                pass  # treat as literal
        if actual_str != expected_val:
            pytest.fail(
                f"Assertion failed: {path}\n"
                f"  expected: {expected_val!r}\n"
                f"  actual:   {actual_str!r}"
            )


def contains_normalized(haystack: str, needle: str) -> bool:
    """Check if needle appears in haystack, normalizing line-by-line whitespace.

    Each needle line is matched as a substring within the corresponding haystack
    line (after stripping), and all needle lines must appear as consecutive
    haystack lines.
    """
    needle_lines = [line.strip() for line in needle.strip().split("\n") if line.strip()]
    haystack_lines = [line.strip() for line in haystack.split("\n") if line.strip()]
    if not needle_lines:
        return True
    for i in range(len(haystack_lines)):
        if needle_lines[0] in haystack_lines[i]:
            match = True
            for j in range(1, len(needle_lines)):
                if (
                    i + j >= len(haystack_lines)
                    or needle_lines[j] not in haystack_lines[i + j]
                ):
                    match = False
                    break
            if match:
                return True
    return False


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def run_parse(source: str) -> PhaseResult:
    """Run the Python frontend parser, return ok/error result."""
    try:
        signal.alarm(PARSE_TIMEOUT)
        parse(source)
        return PhaseResult()
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    finally:
        signal.alarm(0)


def run_subset(source: str) -> PhaseResult:
    """Run subset verification on Python source."""
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    result = verify_subset(ast_dict)
    return PhaseResult(
        errors=[e.message for e in result.errors()],
        warnings=[w.message for w in result.warnings()],
    )


def run_names(source: str) -> PhaseResult:
    """Run name resolution on Python source."""
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    result = resolve_names(ast_dict)
    return PhaseResult(
        errors=[e.message for e in result.errors()],
        warnings=[w.message for w in result.warnings],
    )


def run_sigs(source: str) -> PhaseResult:
    """Run signature collection on Python source."""
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    result = verify_subset(ast_dict)
    sub_errors = result.errors()
    if sub_errors:
        return PhaseResult(errors=[e.message for e in sub_errors])
    name_result = resolve_names(ast_dict)
    name_errors = name_result.errors()
    if name_errors:
        return PhaseResult(errors=[e.message for e in name_errors])
    # Build known_classes and node_classes from name table
    known_classes: set[str] = set()
    node_classes: set[str] = set()
    table = name_result.table
    for name, info in table.module_names.items():
        if info.kind == "class":
            known_classes.add(name)
            if len(info.bases) > 0:
                # Check if any base is "Node" or ends with "Node"
                for base in info.bases:
                    if base == "Node" or base.endswith("Node"):
                        node_classes.add(name)
    sig_result = collect_signatures(ast_dict, known_classes, node_classes)
    sig_errors = sig_result.errors()
    if sig_errors:
        return PhaseResult(errors=[str(e) for e in sig_errors])
    return PhaseResult(data=sig_result.to_dict())


def run_fields(source: str) -> PhaseResult:
    """Run field collection on Python source."""
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    result = verify_subset(ast_dict)
    sub_errors = result.errors()
    if sub_errors:
        return PhaseResult(errors=[e.message for e in sub_errors])
    name_result = resolve_names(ast_dict)
    name_errors = name_result.errors()
    if name_errors:
        return PhaseResult(errors=[e.message for e in name_errors])
    # Build known_classes, node_classes, hierarchy_roots from name table
    known_classes: set[str] = set()
    node_classes: set[str] = set()
    base_counts: dict[str, int] = {}
    parent_of: dict[str, str] = {}
    table = name_result.table
    for name, info in table.module_names.items():
        if info.kind == "class":
            known_classes.add(name)
            for base in info.bases:
                if base == "Node" or base.endswith("Node"):
                    node_classes.add(name)
                if base in known_classes or base[0:1].isupper():
                    if base not in base_counts:
                        base_counts[base] = 0
                    base_counts[base] += 1
                    parent_of[name] = base
    # Hierarchy roots: classes that are bases of others but have no parent themselves
    hierarchy_roots: set[str] = set()
    for base_name in base_counts:
        if base_name not in parent_of:
            hierarchy_roots.add(base_name)
    sig_result = collect_signatures(ast_dict, known_classes, node_classes)
    sig_errors = sig_result.errors()
    if sig_errors:
        return PhaseResult(errors=[str(e) for e in sig_errors])
    field_result = collect_fields(
        ast_dict, known_classes, node_classes, hierarchy_roots, sig_result
    )
    field_errors = field_result.errors()
    if field_errors:
        return PhaseResult(errors=[str(e) for e in field_errors])
    return PhaseResult(data=field_result.to_dict())


def run_hierarchy(source: str) -> PhaseResult:
    """Run hierarchy analysis on Python source."""
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    result = verify_subset(ast_dict)
    sub_errors = result.errors()
    if sub_errors:
        return PhaseResult(errors=[e.message for e in sub_errors])
    name_result = resolve_names(ast_dict)
    name_errors = name_result.errors()
    if name_errors:
        return PhaseResult(errors=[e.message for e in name_errors])
    # Build known_classes and class_bases from name table
    known_classes: set[str] = set()
    class_bases: dict[str, list[str]] = {}
    table = name_result.table
    for name, info in table.module_names.items():
        if info.kind == "class":
            known_classes.add(name)
            class_bases[name] = list(info.bases)
    hier_result = build_hierarchy(known_classes, class_bases)
    hier_errors = hier_result.errors()
    if hier_errors:
        return PhaseResult(errors=[str(e) for e in hier_errors])
    return PhaseResult(data=hier_result.to_dict())


def run_inference(source: str) -> PhaseResult:
    """Run the full Python frontend pipeline (phases 2-9), checking inference errors."""
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    result = verify_subset(ast_dict)
    sub_errors = result.errors()
    if sub_errors:
        return PhaseResult(errors=[e.message for e in sub_errors])
    name_result = resolve_names(ast_dict)
    name_errors = name_result.errors()
    if name_errors:
        return PhaseResult(errors=[e.message for e in name_errors])
    known_classes: set[str] = set()
    node_classes: set[str] = set()
    class_bases: dict[str, list[str]] = {}
    base_counts: dict[str, int] = {}
    parent_of: dict[str, str] = {}
    table = name_result.table
    for name, info in table.module_names.items():
        if info.kind == "class":
            known_classes.add(name)
            class_bases[name] = list(info.bases)
            for base in info.bases:
                if base == "Node" or base.endswith("Node"):
                    node_classes.add(name)
                if base in known_classes or base[0:1].isupper():
                    if base not in base_counts:
                        base_counts[base] = 0
                    base_counts[base] += 1
                    parent_of[name] = base
    hierarchy_roots: set[str] = set()
    for base_name in base_counts:
        if base_name not in parent_of:
            hierarchy_roots.add(base_name)
    sig_result = collect_signatures(ast_dict, known_classes, node_classes)
    sig_errors = sig_result.errors()
    if sig_errors:
        return PhaseResult(errors=[str(e) for e in sig_errors])
    field_result = collect_fields(
        ast_dict, known_classes, node_classes, hierarchy_roots, sig_result
    )
    field_errors = field_result.errors()
    if field_errors:
        return PhaseResult(errors=[str(e) for e in field_errors])
    from src.frontend.hierarchy import build_hierarchy

    hier_result = build_hierarchy(known_classes, class_bases)
    hier_errors = hier_result.errors()
    if hier_errors:
        return PhaseResult(errors=[str(e) for e in hier_errors])
    inf_result = _run_inference(
        ast_dict, sig_result, field_result, hier_result, known_classes, class_bases
    )
    inf_errors = inf_result.errors()
    if inf_errors:
        return PhaseResult(errors=[str(e) for e in inf_errors])
    return PhaseResult()


def run_type_checking(source: str) -> PhaseResult:
    """Run the Taytsh type checker on Taytsh source."""
    try:
        module = taytsh_parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    errors, checker = check_with_info(module)
    if errors:
        return PhaseResult(errors=[str(e) for e in errors])
    return PhaseResult()


def lower_to_taytsh(source: str) -> tuple[str | None, str | None]:
    """Lower Python source to Taytsh text. Returns (output, error)."""
    try:
        ast_dict = parse(source)
        result = verify_subset(ast_dict)
        sub_errors = result.errors()
        if sub_errors:
            return (None, sub_errors[0].message)
        name_result = resolve_names(ast_dict)
        name_errors = name_result.errors()
        if name_errors:
            return (None, name_errors[0].message)
        known_classes: set[str] = set()
        node_classes: set[str] = set()
        class_bases: dict[str, list[str]] = {}
        base_counts: dict[str, int] = {}
        parent_of: dict[str, str] = {}
        table = name_result.table
        for name, info in table.module_names.items():
            if info.kind == "class":
                known_classes.add(name)
                class_bases[name] = list(info.bases)
                for base in info.bases:
                    if base == "Node" or base.endswith("Node"):
                        node_classes.add(name)
                    if base in known_classes or base[0:1].isupper():
                        if base not in base_counts:
                            base_counts[base] = 0
                        base_counts[base] += 1
                        parent_of[name] = base
        hierarchy_roots: set[str] = set()
        for base_name in base_counts:
            if base_name not in parent_of:
                hierarchy_roots.add(base_name)
        sig_result = collect_signatures(ast_dict, known_classes, node_classes)
        sig_errors = sig_result.errors()
        if sig_errors:
            return (None, str(sig_errors[0]))
        field_result = collect_fields(
            ast_dict, known_classes, node_classes, hierarchy_roots, sig_result
        )
        field_errors = field_result.errors()
        if field_errors:
            return (None, str(field_errors[0]))
        from src.frontend.hierarchy import build_hierarchy

        hier_result = build_hierarchy(known_classes, class_bases)
        hier_errors = hier_result.errors()
        if hier_errors:
            return (None, str(hier_errors[0]))
        from src.frontend.lowering import lower
        from src.taytsh.emit import to_source

        module, lower_errors = lower(
            ast_dict,
            sig_result,
            field_result,
            hier_result,
            known_classes,
            class_bases,
            source,
        )
        if lower_errors:
            return (None, str(lower_errors[0]))
        if module is None:
            return (None, "lowering produced no module")
        output = to_source(module)
        return (output, None)
    except Exception as e:
        return (None, str(e))


def _run_taytsh_pipeline(source):
    module = taytsh_parse(source)
    errors, checker = check_with_info(module)
    if errors:
        return PhaseResult(errors=[str(e) for e in errors]), None, None
    return None, module, checker


def run_returns(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_returns(module, checker)
    return PhaseResult(data=serialize_annotations(module, "returns"))


def run_scope(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    return PhaseResult(data=serialize_annotations(module, "scope"))


def run_liveness(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    return PhaseResult(data=serialize_annotations(module, "liveness"))


def run_strings(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    analyze_strings(module, checker)
    return PhaseResult(data=serialize_annotations(module, "strings"))


def run_hoisting(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_hoisting(module, checker)
    return PhaseResult(data=serialize_annotations(module, "hoisting"))


def run_ownership(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    analyze_ownership(module, checker)
    return PhaseResult(data=serialize_annotations(module, "ownership"))


def _collect_calls(obj, calls, checker):
    """Walk AST collecting TCall nodes keyed by callee name."""
    if isinstance(obj, TCall):
        name = None
        if isinstance(obj.func, TVar):
            n = obj.func.name
            # Skip builtins and struct constructors — only user-defined fns
            t = checker.types.get(n)
            if t is not None and isinstance(t, StructT):
                name = None  # struct construction
            elif n in checker.functions:
                name = n
            else:
                name = None  # builtin or unknown
        elif isinstance(obj.func, TFieldAccess):
            name = obj.func.field
        if name is not None:
            is_tail = obj.annotations.get("callgraph.is_tail_call", False)
            calls.setdefault(name, {})["is_tail_call"] = is_tail
    if isinstance(obj, list):
        for item in obj:
            _collect_calls(item, calls, checker)
        return
    for attr in (
        "body",
        "value",
        "expr",
        "func",
        "target",
        "targets",
        "cond",
        "then_body",
        "else_body",
        "then_expr",
        "else_expr",
        "left",
        "right",
        "operand",
        "obj",
        "index",
        "low",
        "high",
        "args",
        "elements",
        "entries",
        "iterable",
        "cases",
        "default",
        "catches",
        "finally_body",
        "pattern",
    ):
        child = getattr(obj, attr, None)
        if child is not None:
            if isinstance(child, list):
                for item in child:
                    _collect_calls(item, calls, checker)
            elif isinstance(child, tuple):
                for item in child:
                    _collect_calls(item, calls, checker)
            elif hasattr(child, "__dict__"):
                _collect_calls(child, calls, checker)


def _serialize_callgraph(module, checker):
    result = {}
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            d = {
                k[10:]: v
                for k, v in decl.annotations.items()
                if k.startswith("callgraph.")
            }
            calls = {}
            _collect_calls(decl.body, calls, checker)
            if calls:
                d["calls"] = calls
            result[decl.name] = d
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                d = {
                    k[10:]: v
                    for k, v in method.annotations.items()
                    if k.startswith("callgraph.")
                }
                calls = {}
                _collect_calls(method.body, calls, checker)
                if calls:
                    d["calls"] = calls
                result[f"{decl.name}.{method.name}"] = d
    return result


def run_callgraph(source: str) -> PhaseResult:
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_callgraph(module, checker)
    return PhaseResult(data=_serialize_callgraph(module, checker))


def run_taytsh_parse(source: str) -> PhaseResult:
    try:
        signal.alarm(PARSE_TIMEOUT)
        module = taytsh_parse(source)
        return PhaseResult(
            data={
                "strict_math": module.strict_math,
                "strict_tostring": module.strict_tostring,
            }
        )
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    finally:
        signal.alarm(0)


def run_taytsh_check(source: str) -> PhaseResult:
    try:
        signal.alarm(PARSE_TIMEOUT)
        errors = taytsh_check_fn(source)
        if errors:
            return PhaseResult(errors=[str(e) for e in errors])
        return PhaseResult()
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    finally:
        signal.alarm(0)


def discover_taytsh_apps(test_dir: Path) -> list[Path]:
    return sorted(test_dir.glob("*.ty"))


def _transpile_with_emitter(source: str, emitter) -> tuple[str | None, str | None]:
    """Transpile Taytsh source. Returns (output, error)."""
    try:
        signal.alarm(PARSE_TIMEOUT)
        module = taytsh_parse(source)
    except Exception as e:
        return (None, str(e))
    finally:
        signal.alarm(0)
    errors, checker = check_with_info(module)
    if errors:
        return (None, str(errors[0]))
    try:
        analyze_returns(module, checker)
        analyze_scope(module, checker)
        analyze_liveness(module, checker)
        return (emitter(module), None)
    except Exception as e:
        return (None, str(e))


def transpile_code(source: str, lang: str) -> tuple[str | None, str | None]:
    """Transpile Taytsh source to the given language. Returns (output, error)."""
    emitter = EMITTERS.get(lang)
    if emitter is None:
        return (None, f"no emitter for '{lang}'")
    return _transpile_with_emitter(source, emitter)


def transpile_app(source: str, target: str) -> tuple[str | None, str | None]:
    """Transpile Python apptest source to target language. Returns (output, error)."""
    taytsh_text, err = lower_to_taytsh(source)
    if err is not None:
        return (None, err)
    emitter = EMITTERS.get(target)
    if emitter is None:
        return (None, f"no emitter for target '{target}'")
    return _transpile_with_emitter(taytsh_text, emitter)


def discover_app_tests(
    test_dir: Path, targets: list[str]
) -> list[tuple[str, Path, str]]:
    """Find all app tests. Returns (test_id, source_path, target)."""
    results = []
    for test_file in sorted(test_dir.glob("apptest_*.py")):
        for target in targets:
            test_id = f"{test_file.stem}[{target}]"
            results.append((test_id, test_file, target))
    return results


def _available_targets() -> list[str]:
    """Return targets whose runtimes are available."""
    available = []
    for target in sorted(RUNTIMES):
        cmd = RUNTIMES[target]
        if target == "python" or shutil.which(cmd[0]):
            available.append(target)
    return available


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transpiled_output(codegen_input: str, codegen_lang: str) -> str:
    output, err = transpile_code(codegen_input, codegen_lang)
    if err is not None:
        pytest.fail(f"Transpile error: {err}")
    if output is None:
        pytest.fail("No output from transpiler")
    return output


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    for section in TESTS.values():
        for name, cfg in section.items():
            test_dir = TESTS_DIR / cfg["dir"]
            run = cfg["run"]
            if run == "cli" and "cli_spec" in metafunc.fixturenames:
                tests = discover_cli_tests(test_dir)
                params = [pytest.param(spec, id=tid) for tid, spec in tests]
                metafunc.parametrize("cli_spec", params)
            elif run == "phase":
                fixture = f"{name}_input"
                if fixture in metafunc.fixturenames:
                    specs = discover_specs(test_dir)
                    params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in specs]
                    metafunc.parametrize(f"{fixture},{name}_expected", params)
            elif run == "lowering":
                fixture = f"{name}_input"
                if fixture in metafunc.fixturenames:
                    specs = discover_specs(test_dir)
                    params = [pytest.param(inp, exp, id=tid) for tid, inp, exp in specs]
                    metafunc.parametrize(f"{fixture},{name}_expected", params)
            elif run == "codegen" and "codegen_input" in metafunc.fixturenames:
                dirs = {
                    d.name
                    for d in test_dir.iterdir()
                    if d.is_dir() and d.name != "base"
                }
                langs = sorted(dirs & set(EMITTERS))
                all_tests = []
                for lang in langs:
                    for tid, inp, exp in discover_codegen_tests(test_dir, lang):
                        all_tests.append(pytest.param(inp, exp, lang, id=tid))
                for lang in sorted(set(EMITTERS) - dirs):
                    base_dir = test_dir / "base"
                    for base_file in sorted(base_dir.glob("*.tests")):
                        for name, _ in parse_simple_tests(base_file):
                            tid = f"{base_file.stem}/{name}[{lang}]"
                            all_tests.append(pytest.param(
                                "", "", lang, id=tid,
                            ))
                metafunc.parametrize("codegen_input,codegen_expected,codegen_lang", all_tests)
            elif run == "taytsh_app" and "taytsh_app" in metafunc.fixturenames:
                apps = discover_taytsh_apps(test_dir)
                params = [pytest.param(p, id=p.stem) for p in apps]
                metafunc.parametrize("taytsh_app", params)
            elif run == "app" and "app_source" in metafunc.fixturenames:
                target_opt = metafunc.config.getoption("--target", default=None)
                targets = target_opt if target_opt else _available_targets()
                tests = discover_app_tests(test_dir, targets)
                params = [
                    pytest.param(path, target, id=tid) for tid, path, target in tests
                ]
                metafunc.parametrize("app_source,app_target", params)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def _cli_needs_backend(spec: dict) -> bool:
    """True if the test will reach the backend (no --stop-at, expects exit 0)."""
    args = spec["args"]
    if "--stop-at" in args:
        return False
    expects_success = any(k == "exit" and v == 0 for k, v in spec["assertions"])
    if not expects_success:
        return False
    if "--target" not in args:
        return False
    target = args[args.index("--target") + 1]
    return target not in EMITTERS


def test_cli(cli_spec: dict) -> None:
    if _cli_needs_backend(cli_spec):
        target = cli_spec["args"][cli_spec["args"].index("--target") + 1]
        pytest.skip(f"backend not yet implemented for '{target}'")
    result = run_cli(cli_spec)
    check_cli_assertions(result, cli_spec["assertions"])


def test_parse(parse_input, parse_expected):
    check_expected(parse_expected, run_parse(parse_input), "parse", lenient_errors=True)


def test_subset(subset_input, subset_expected):
    check_expected(subset_expected, run_subset(subset_input), "subset")


def test_names(names_input, names_expected):
    check_expected(names_expected, run_names(names_input), "names")


def test_sigs(sigs_input, sigs_expected):
    check_expected(sigs_expected, run_sigs(sigs_input), "sigs")


def test_fields(fields_input, fields_expected):
    check_expected(fields_expected, run_fields(fields_input), "fields")


def test_hierarchy(hierarchy_input, hierarchy_expected):
    check_expected(hierarchy_expected, run_hierarchy(hierarchy_input), "hierarchy")


def test_inference(inference_input, inference_expected):
    check_expected(
        inference_expected,
        run_inference(inference_input),
        "inference",
        lenient_errors=True,
    )


def test_type_checking(type_checking_input, type_checking_expected):
    check_expected(
        type_checking_expected,
        run_type_checking(type_checking_input),
        "type_checking",
    )


def test_lowering(lowering_input, lowering_expected):
    output, err = lower_to_taytsh(lowering_input)
    if lowering_expected.startswith("error:"):
        expected_msg = lowering_expected[6:].strip()
        if err is None:
            pytest.fail(f"Expected error containing '{expected_msg}', got success")
        if expected_msg and expected_msg.lower() not in (err or "").lower():
            pytest.fail(f"Expected error containing '{expected_msg}', got: {err}")
        return
    if err is not None:
        pytest.fail(f"Lowering error: {err}")
    if output is None:
        pytest.fail("No output from lowering")
    if not contains_normalized(output, lowering_expected):
        pytest.fail(
            "Expected not found in output:\n"
            f"--- expected ---\n{lowering_expected}\n"
            f"--- got ---\n{output}"
        )


def test_returns(returns_input, returns_expected):
    check_expected(returns_expected, run_returns(returns_input), "returns")


def test_scope(scope_input, scope_expected):
    check_expected(scope_expected, run_scope(scope_input), "scope")


def test_liveness(liveness_input, liveness_expected):
    check_expected(liveness_expected, run_liveness(liveness_input), "liveness")


def test_strings(strings_input, strings_expected):
    check_expected(strings_expected, run_strings(strings_input), "strings")


def test_hoisting(hoisting_input, hoisting_expected):
    check_expected(hoisting_expected, run_hoisting(hoisting_input), "hoisting")


def test_ownership(ownership_input, ownership_expected):
    check_expected(ownership_expected, run_ownership(ownership_input), "ownership")


def test_callgraph(callgraph_input, callgraph_expected):
    check_expected(callgraph_expected, run_callgraph(callgraph_input), "callgraph")


def test_codegen(
    codegen_input: str,
    codegen_expected: str,
    codegen_lang: str,
    transpiled_output: str,
):
    if not contains_normalized(transpiled_output, codegen_expected):
        pytest.fail(
            "Expected not found in output:\n"
            f"--- expected ---\n{codegen_expected}\n"
            f"--- got ---\n{transpiled_output}"
        )


def test_taytsh_parse(taytsh_parse_input, taytsh_parse_expected):
    check_expected(
        taytsh_parse_expected,
        run_taytsh_parse(taytsh_parse_input),
        "taytsh_parse",
        lenient_errors=True,
    )


def test_taytsh_check(taytsh_check_input, taytsh_check_expected):
    check_expected(
        taytsh_check_expected,
        run_taytsh_check(taytsh_check_input),
        "taytsh_check",
        lenient_errors=True,
    )


def test_taytsh_app(taytsh_app: Path):
    source = taytsh_app.read_text()
    module = taytsh_parse(source)
    result = taytsh_run(module)
    if result.exit_code != 0:
        output = (result.stdout + result.stderr).decode(errors="replace").strip()
        pytest.fail(f"Exit code {result.exit_code}:\n{output}")


def test_app(app_source: Path, app_target: str) -> None:
    source = app_source.read_text()
    output, err = transpile_app(source, app_target)
    if err is not None:
        pytest.fail(f"Transpile error ({app_target}): {err}")
    runtime = RUNTIMES[app_target]
    result = subprocess.run(
        runtime,
        input=output.encode(),
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        stdout = result.stdout.decode(errors="replace")
        pytest.fail(
            f"App test failed with exit {result.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
