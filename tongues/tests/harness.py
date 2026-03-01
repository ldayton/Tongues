"""Shared test infrastructure for Tongues test phases."""

import importlib.util
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.frontend.typecollect import collect_signatures, collect_types
from src.frontend.hierarchy import build_hierarchy
from src.frontend.pycheck import (
    run_pycheck as _run_pycheck,
    compute_expr_coverage,
    _EXPR_NODE_TYPES,
)
from src.frontend.bind import run_bind, resolve_names, verify as verify_subset
from src.frontend.parse import parse, stamp_uids
from src.frontend.types import JDict, JList, JStr, JInt, JFloat, JBool, JNull

TONGUES_DIR = Path(__file__).parent.parent
LIB_DIR = TONGUES_DIR / "src" / "lib"

TRANSPILED_BINARY: str | None = os.environ.get("TONGUES_TRANSPILED_BINARY")

_TRANSPILED_MODULE = None
if TRANSPILED_BINARY is not None and TRANSPILED_BINARY.endswith(".py"):
    _spec = importlib.util.spec_from_file_location(
        "tongues_transpiled", TRANSPILED_BINARY
    )
    _TRANSPILED_MODULE = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _TRANSPILED_MODULE
    _spec.loader.exec_module(_TRANSPILED_MODULE)

EXT_TO_LANG = {".py": "python", ".rb": "ruby", ".pl": "perl"}

_LIB_IMPORT_RE = re.compile(r"^from lib\.(\w+) import", re.MULTILINE)


def _find_lib_imports(source: str) -> list[str]:
    """Extract unique lib module names from 'from lib.X import' statements."""
    return list(dict.fromkeys(_LIB_IMPORT_RE.findall(source)))


def _read_lib_sources(names: list[str]) -> list[tuple[str, str]]:
    """Read lib modules. Returns [(import_path, source)] e.g. [('lib/base64.py', '...')]."""
    result: list[tuple[str, str]] = []
    for name in names:
        file_path = LIB_DIR / f"{name}.py"
        import_path = f"lib/{name}.py"
        result.append((import_path, file_path.read_text()))
    return result


def _build_project_input(
    app_path: str, app_source: str, lib_sources: list[tuple[str, str]]
) -> bytes:
    """Build NUL-delimited project input."""
    parts = [app_path, app_source]
    for import_path, source in lib_sources:
        parts.append(import_path)
        parts.append(source)
    return "\0".join(parts).encode()


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


def parse_linker_test_file(path: Path) -> list[tuple[str, dict]]:
    """Parse a linker .tests file with file: directives into (name, spec) tuples.

    Format:
        === test name
        file: a.py
        <source for a.py>
        file: b.py
        <source for b.py>
        args: --project --stop-at subset
        ---
        exit: 0
        ---
    """
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
            spec = _parse_linker_spec(input_lines, expected_lines)
            result.append((test_name, spec))
        else:
            i += 1
    return result


def _parse_linker_spec(input_lines: list[str], expected_lines: list[str]) -> dict:
    """Parse input with file: directives + expected into a linker test spec."""
    spec: dict = {"files": [], "args": [], "assertions": []}
    current_path: str | None = None
    current_lines: list[str] = []
    for line in input_lines:
        if line.startswith("file: "):
            if current_path is not None:
                spec["files"].append((current_path, "\n".join(current_lines)))
            current_path = line[6:].strip()
            current_lines = []
        elif line.startswith("args: "):
            if current_path is not None:
                spec["files"].append((current_path, "\n".join(current_lines)))
                current_path = None
                current_lines = []
            spec["args"] = line[6:].strip().split()
        else:
            current_lines.append(line)
    if current_path is not None:
        spec["files"].append((current_path, "\n".join(current_lines)))
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


def discover_linker_tests(test_dir: Path) -> list[tuple[str, dict]]:
    """Find all linker tests across .tests files."""
    results = []
    for test_file in sorted(test_dir.glob("*.tests")):
        for name, spec in parse_linker_test_file(test_file):
            results.append((f"{test_file.stem}/{name}", spec))
    return results


def run_linker_test(spec: dict) -> subprocess.CompletedProcess[bytes]:
    """Run a linker test: encode files as NUL-delimited --project stdin."""
    parts = []
    for path, source in spec["files"]:
        parts.append(path)
        parts.append(source)
    stdin_data = "\0".join(parts).encode() if parts else b""
    if TRANSPILED_BINARY is not None:
        if _TRANSPILED_MODULE is not None:
            argv = [TRANSPILED_BINARY, *spec["args"]]
            return _run_inprocess(argv, stdin_data=stdin_data)
        cmd = [*_transpiled_cmd(), *spec["args"]]
    else:
        cmd = [sys.executable, "-m", "src.tongues", *spec["args"]]
    return subprocess.run(cmd, input=stdin_data, capture_output=True, cwd=TONGUES_DIR)


def run_cli(spec: dict) -> subprocess.CompletedProcess[bytes]:
    """Run tongues CLI from a test spec."""
    if spec["stdin_bytes"] is not None:
        stdin_data = spec["stdin_bytes"]
    elif spec["stdin"] is not None:
        stdin_data = spec["stdin"].encode()
    else:
        stdin_data = b""
    if TRANSPILED_BINARY is not None:
        if _TRANSPILED_MODULE is not None:
            argv = [TRANSPILED_BINARY, *spec["args"]]
            return _run_inprocess(argv, stdin_data=stdin_data)
        cmd = [*_transpiled_cmd(), *spec["args"]]
    else:
        cmd = [sys.executable, "-m", "src.tongues", *spec["args"]]
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
from src.middleend.callgraph_serial import serialize_callgraph
from src.middleend.hoisting import analyze_hoisting
from src.middleend.liveness import analyze_liveness
from src.middleend.ownership import analyze_ownership
from src.middleend.returns import analyze_returns
from src.middleend.scope import analyze_scope
from src.middleend.strings import analyze_strings
from src.taytsh import check as taytsh_check_fn, parse as taytsh_parse
from src.taytsh.treewalker import run as taytsh_run
from src.taytsh.ast import (
    serialize_annotations,
)
from src.taytsh.check import check_with_info

PARSE_TIMEOUT = 5
TESTS_DIR = Path(__file__).parent

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


def discover_specs(
    test_dir: Path, pattern: str = "*.tests"
) -> list[tuple[str, str, str]]:
    """Glob *.tests in test_dir, return (test_id, input, expected) tuples."""
    results = []
    for test_file in sorted(test_dir.glob(pattern)):
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


def discover_codegen_tests(test_dir: Path, lang: str) -> list[tuple[str, str, str]]:
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
    reveals: list[tuple[int, str]] = field(default_factory=list)


def _unwrap_jvalue(obj: object) -> object:
    """Unwrap JsonValue types to plain Python equivalents."""
    if isinstance(obj, JDict):
        return obj.entries
    if isinstance(obj, JList):
        return obj.items
    if isinstance(obj, JStr):
        return obj.value
    if isinstance(obj, JInt):
        return obj.value
    if isinstance(obj, JFloat):
        return obj.value
    if isinstance(obj, JBool):
        return obj.value
    if isinstance(obj, JNull):
        return None
    return obj


def resolve_dotpath(obj: object, path: str) -> object:
    """Resolve a dot-separated path against a nested dict/list structure."""
    parts = path.split(".")
    current = _unwrap_jvalue(obj)
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "length":
            return len(current)
        if isinstance(current, list):
            current = _unwrap_jvalue(current[int(part)])
            i += 1
        elif isinstance(current, dict):
            if part in current:
                current = _unwrap_jvalue(current[part])
                i += 1
            else:
                found = False
                for j in range(i + 1, len(parts)):
                    composite = ".".join(parts[i : j + 1])
                    if composite in current:
                        current = _unwrap_jvalue(current[composite])
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


def _check_reveals(
    assertions: list[tuple[int, str]], actuals: list[tuple[int, str]]
) -> None:
    for lineno, expected_type in assertions:
        found = False
        for actual_line, actual_type in actuals:
            if actual_line == lineno:
                if actual_type != expected_type:
                    pytest.fail(
                        f"reveal_type at line {lineno}: expected '{expected_type}', got '{actual_type}'"
                    )
                found = True
                break
        if not found:
            pytest.fail(f"No reveal_type found at line {lineno}")


def check_expected(
    expected: str, result: PhaseResult, phase: str, *, lenient_errors: bool = False
) -> None:
    reveal_assertions: list[tuple[int, str]] = []
    verdict_lines: list[str] = []
    for line in expected.split("\n"):
        stripped = line.strip()
        if stripped.startswith("reveal:"):
            rest = stripped[7:]
            eq_pos = rest.index("=")
            lineno = int(rest[:eq_pos].strip())
            expected_type = rest[eq_pos + 1 :].strip()
            reveal_assertions.append((lineno, expected_type))
        else:
            verdict_lines.append(line)
    expected = "\n".join(verdict_lines).strip()
    if not expected:
        expected = "ok"
    if expected == "ok":
        if result.errors:
            pytest.fail(f"Expected ok, got error: {result.errors[0]}")
        _check_reveals(reveal_assertions, result.reveals)
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
# Transpiled binary dispatch
# ---------------------------------------------------------------------------


def _transpiled_runtime() -> list[str]:
    """Get runtime command for the transpiled binary based on its extension."""
    assert TRANSPILED_BINARY is not None
    ext = Path(TRANSPILED_BINARY).suffix
    lang = EXT_TO_LANG.get(ext)
    if lang is None:
        pytest.fail(f"Unknown transpiled binary extension: {ext}")
    return RUNTIMES[lang]


def _transpiled_cmd(*extra: str) -> list[str]:
    """Build a command to invoke the transpiled binary."""
    assert TRANSPILED_BINARY is not None
    binary = str((TONGUES_DIR / TRANSPILED_BINARY).resolve())
    return [*_transpiled_runtime(), binary, *extra]


def _run_inprocess(
    argv: list[str], *, stdin_data: bytes = b""
) -> subprocess.CompletedProcess:
    """Run the transpiled Python module in-process with the given argv."""
    old_argv = sys.argv
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdin = sys.stdin
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    stdin_buf = io.BytesIO(stdin_data)
    stdin_wrapper = io.TextIOWrapper(stdin_buf)
    returncode = 0
    try:
        sys.argv = argv
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf
        sys.stdin = stdin_wrapper
        _TRANSPILED_MODULE.main()
    except SystemExit as e:
        returncode = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        stderr_buf.write(str(e) + "\n")
        returncode = 1
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin = old_stdin
    return subprocess.CompletedProcess(
        args=argv,
        returncode=returncode,
        stdout=stdout_buf.getvalue().encode(),
        stderr=stderr_buf.getvalue().encode(),
    )


def _run_transpiled(
    source: str,
    args: list[str],
    *,
    is_taytsh: bool = False,
    expect_json: bool = True,
) -> PhaseResult:
    """Run a phase via the transpiled binary subprocess."""
    suffix = ".ty" if is_taytsh else ".py"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        tmp.write(source)
        tmp.flush()
        cmd_args = list(args)
        if is_taytsh:
            argv = [TRANSPILED_BINARY, "taytsh", *cmd_args, tmp.name]
        else:
            argv = [TRANSPILED_BINARY, *cmd_args, tmp.name]
        if _TRANSPILED_MODULE is not None:
            result = _run_inprocess(argv)
        else:
            cmd = [*_transpiled_runtime(), *argv]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
        Path(tmp.name).unlink(missing_ok=True)
    stderr_text = result.stderr.decode(errors="replace").strip()
    if result.returncode != 0:
        errors = [line for line in stderr_text.split("\n") if line.strip()]
        return PhaseResult(errors=errors)
    if not expect_json:
        warnings = (
            [line for line in stderr_text.split("\n") if line.strip()]
            if stderr_text
            else []
        )
        return PhaseResult(warnings=warnings)
    warnings = (
        [line for line in stderr_text.split("\n") if line.strip()]
        if stderr_text
        else []
    )
    stdout_text = result.stdout.decode(errors="replace").strip()
    if not stdout_text:
        return PhaseResult(warnings=warnings)
    try:
        data = json.loads(stdout_text)
    except json.JSONDecodeError:
        return PhaseResult(errors=[f"Invalid JSON output: {stdout_text[:200]}"])
    return PhaseResult(data=data, warnings=warnings)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def run_parse(source: str) -> PhaseResult:
    """Run the Python frontend parser, return ok/error result."""
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "parse"])
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
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "subset"], expect_json=False)
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
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "names"])
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
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "signatures"])
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    bind_result = run_bind(ast_dict)
    if not bind_result.subset_ok():
        return PhaseResult(errors=[e.message for e in bind_result.subset_violations])
    if not bind_result.names_ok():
        return PhaseResult(errors=[e.message for e in bind_result.name_violations])
    sig_result = collect_signatures(
        ast_dict,
        bind_result.known_classes,
        bind_result.node_classes,
        bind_result.type_aliases,
        bind_result.class_bases,
    )
    sig_errors = sig_result.errors()
    if sig_errors:
        return PhaseResult(errors=[str(e) for e in sig_errors])
    return PhaseResult(data=sig_result.to_dict())


def run_fields(source: str) -> PhaseResult:
    """Run field collection on Python source."""
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "fields"])
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    bind_result = run_bind(ast_dict)
    if not bind_result.subset_ok():
        return PhaseResult(errors=[e.message for e in bind_result.subset_violations])
    if not bind_result.names_ok():
        return PhaseResult(errors=[e.message for e in bind_result.name_violations])
    hier_result = build_hierarchy(bind_result.known_classes, bind_result.class_bases)
    hier_errors = hier_result.errors()
    if hier_errors:
        return PhaseResult(errors=[str(e) for e in hier_errors])
    tc_result = collect_types(
        ast_dict,
        bind_result.known_classes,
        bind_result.node_classes,
        bind_result.type_aliases,
        bind_result.class_bases,
        set(hier_result.hierarchy_roots),
    )
    tc_errors = tc_result.errors()
    if tc_errors:
        return PhaseResult(errors=[str(e) for e in tc_errors])
    return PhaseResult(data=tc_result.fields_to_dict())


def run_hierarchy(source: str) -> PhaseResult:
    """Run hierarchy analysis on Python source."""
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "hierarchy"])
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    bind_result = run_bind(ast_dict)
    if not bind_result.subset_ok():
        return PhaseResult(errors=[e.message for e in bind_result.subset_violations])
    if not bind_result.names_ok():
        return PhaseResult(errors=[e.message for e in bind_result.name_violations])
    hier_result = build_hierarchy(bind_result.known_classes, bind_result.class_bases)
    hier_errors = hier_result.errors()
    if hier_errors:
        return PhaseResult(errors=[str(e) for e in hier_errors])
    return PhaseResult(data=hier_result.to_dict())


def run_pycheck(source: str) -> PhaseResult:
    """Run the full Python frontend pipeline (phases 2-9), checking type errors."""
    if TRANSPILED_BINARY is not None:
        result = _run_transpiled(source, ["--stop-at", "pycheck"])
        if result.errors:
            return result
        if result.data and "reveals" in result.data:
            reveals = []
            for rev in result.data["reveals"]:
                reveals.append((rev["line"], rev["type"]))
            return PhaseResult(reveals=reveals)
        return result
    try:
        ast_dict = parse(source)
    except Exception as e:
        return PhaseResult(errors=[str(e)])
    bind_result = run_bind(ast_dict)
    if not bind_result.subset_ok():
        return PhaseResult(errors=[e.message for e in bind_result.subset_violations])
    if not bind_result.names_ok():
        return PhaseResult(errors=[e.message for e in bind_result.name_violations])
    hier_result = build_hierarchy(bind_result.known_classes, bind_result.class_bases)
    hier_errors = hier_result.errors()
    if hier_errors:
        return PhaseResult(errors=[str(e) for e in hier_errors])
    tc_result = collect_types(
        ast_dict,
        bind_result.known_classes,
        bind_result.node_classes,
        bind_result.type_aliases,
        bind_result.class_bases,
        hier_result.hierarchy_roots,
    )
    tc_errors = tc_result.errors()
    if tc_errors:
        return PhaseResult(errors=[str(e) for e in tc_errors])
    inf_result = _run_pycheck(
        ast_dict,
        tc_result,
        hier_result,
        bind_result.known_classes,
        bind_result.class_bases,
        bind_result.flow_graphs,
    )
    inf_errors = inf_result.errors()
    if inf_errors:
        return PhaseResult(errors=[str(e) for e in inf_errors])
    return PhaseResult(reveals=inf_result.reveals())


def lower_to_taytsh(source: str) -> tuple[str | None, str | None]:
    """Lower Python source to Taytsh text. Returns (output, error)."""
    lib_names = _find_lib_imports(source)
    if TRANSPILED_BINARY is not None:
        if lib_names:
            lib_sources = _read_lib_sources(lib_names)
            stdin_data = _build_project_input("apptest.py", source, lib_sources)
            argv = [TRANSPILED_BINARY, "--project", "--stop-at", "lowering-text"]
            if _TRANSPILED_MODULE is not None:
                result = _run_inprocess(argv, stdin_data=stdin_data)
            else:
                cmd = [*_transpiled_runtime(), *argv]
                result = subprocess.run(
                    cmd, input=stdin_data, capture_output=True, timeout=30
                )
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp:
                tmp.write(source)
                tmp.flush()
                argv = [TRANSPILED_BINARY, "--stop-at", "lowering-text", tmp.name]
                if _TRANSPILED_MODULE is not None:
                    result = _run_inprocess(argv)
                else:
                    cmd = [*_transpiled_runtime(), *argv]
                    result = subprocess.run(cmd, capture_output=True, timeout=30)
                Path(tmp.name).unlink(missing_ok=True)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            return (None, stderr.split("\n")[0] if stderr else "lowering failed")
        return (result.stdout.decode(errors="replace"), None)
    try:
        from src.frontend.lowering import lower
        from src.taytsh.emit import to_source
        from src.taytsh.ast import TModule

        def _lower_single(src: str):
            """Lower a single source to TModule. Returns (module, error)."""
            ast_dict = parse(src)
            stamp_uids(ast_dict)
            bind_result = run_bind(ast_dict)
            if not bind_result.subset_ok():
                return (None, bind_result.subset_violations[0].message)
            if not bind_result.names_ok():
                return (None, bind_result.name_violations[0].message)
            hier_result = build_hierarchy(
                bind_result.known_classes, bind_result.class_bases
            )
            hier_errors = hier_result.errors()
            if hier_errors:
                return (None, str(hier_errors[0]))
            tc_result = collect_types(
                ast_dict,
                bind_result.known_classes,
                bind_result.node_classes,
                bind_result.type_aliases,
                bind_result.class_bases,
                hier_result.hierarchy_roots,
            )
            tc_errors = tc_result.errors()
            if tc_errors:
                return (None, str(tc_errors[0]))
            inf_result = _run_pycheck(
                ast_dict,
                tc_result,
                hier_result,
                bind_result.known_classes,
                bind_result.class_bases,
                bind_result.flow_graphs,
            )
            inf_errors = inf_result.errors()
            if inf_errors:
                return (None, str(inf_errors[0]))
            module, lower_errors = lower(
                ast_dict,
                tc_result,
                hier_result,
                bind_result.known_classes,
                bind_result.class_bases,
                inf_result,
            )
            if lower_errors:
                return (None, str(lower_errors[0]))
            if module is None:
                return (None, "lowering produced no module")
            return (module, None)

        # Lower lib modules first, then app module
        all_decls: list = []
        if lib_names:
            lib_sources = _read_lib_sources(lib_names)
            for _path, lib_src in lib_sources:
                lib_module, err = _lower_single(lib_src)
                if err is not None:
                    return (None, err)
                all_decls.extend(lib_module.decls)

        app_module, err = _lower_single(source)
        if err is not None:
            return (None, err)
        all_decls.extend(app_module.decls)

        merged = TModule(decls=all_decls)
        output = to_source(merged)
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
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "returns"], is_taytsh=True)
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_returns(module, checker)
    return PhaseResult(data=serialize_annotations(module, "returns"))


def run_scope(source: str) -> PhaseResult:
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "scope"], is_taytsh=True)
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    return PhaseResult(data=serialize_annotations(module, "scope"))


def run_liveness(source: str) -> PhaseResult:
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "liveness"], is_taytsh=True)
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    return PhaseResult(data=serialize_annotations(module, "liveness"))


def run_strings(source: str) -> PhaseResult:
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "strings"], is_taytsh=True)
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    analyze_strings(module, checker)
    return PhaseResult(data=serialize_annotations(module, "strings"))


def run_hoisting(source: str) -> PhaseResult:
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "hoisting"], is_taytsh=True)
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_hoisting(module, checker)
    return PhaseResult(data=serialize_annotations(module, "hoisting"))


def run_ownership(source: str) -> PhaseResult:
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "ownership"], is_taytsh=True)
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    analyze_ownership(module, checker)
    return PhaseResult(data=serialize_annotations(module, "ownership"))


def run_callgraph(source: str) -> PhaseResult:
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "callgraph"], is_taytsh=True)
    err, module, checker = _run_taytsh_pipeline(source)
    if err:
        return err
    analyze_callgraph(module, checker)
    return PhaseResult(data=serialize_callgraph(module, checker))


def run_typarse(source: str) -> PhaseResult:
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(source, ["--stop-at", "parse"], is_taytsh=True)
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


def run_tycheck(source: str) -> PhaseResult:
    if TRANSPILED_BINARY is not None:
        return _run_transpiled(
            source, ["--stop-at", "check"], is_taytsh=True, expect_json=False
        )
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


def discover_ty_apps(test_dir: Path) -> list[Path]:
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
    if TRANSPILED_BINARY is not None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ty", delete=False) as tmp:
            tmp.write(source)
            tmp.flush()
            argv = [TRANSPILED_BINARY, "taytsh", "--emit", lang, tmp.name]
            if _TRANSPILED_MODULE is not None:
                result = _run_inprocess(argv)
            else:
                cmd = [*_transpiled_runtime(), *argv]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
            Path(tmp.name).unlink(missing_ok=True)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            return (None, stderr.split("\n")[0] if stderr else "transpile failed")
        return (result.stdout.decode(errors="replace"), None)
    emitter = EMITTERS.get(lang)
    if emitter is None:
        return (None, f"no emitter for '{lang}'")
    return _transpile_with_emitter(source, emitter)


def emit_from_python(source: str, lang: str) -> tuple[str | None, str | None]:
    """Emit backend output from Python source, bypassing the Taytsh parser.

    Pipeline: parse -> subset -> names -> sigs -> fields -> hierarchy -> lower()
    -> check_with_info() -> middleend -> emitter. Returns (output, error).
    """
    if TRANSPILED_BINARY is not None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(source)
            tmp.flush()
            argv = [TRANSPILED_BINARY, "--target", lang, tmp.name]
            if _TRANSPILED_MODULE is not None:
                result = _run_inprocess(argv)
            else:
                cmd = [*_transpiled_runtime(), *argv]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
            Path(tmp.name).unlink(missing_ok=True)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            return (None, stderr.split("\n")[0] if stderr else "emit failed")
        return (result.stdout.decode(errors="replace"), None)
    emitter = EMITTERS.get(lang)
    if emitter is None:
        return (None, f"no emitter for '{lang}'")
    try:
        ast_dict = parse(source)
        bind_result = run_bind(ast_dict)
        if not bind_result.subset_ok():
            return (None, bind_result.subset_violations[0].message)
        if not bind_result.names_ok():
            return (None, bind_result.name_violations[0].message)
        hier_result = build_hierarchy(
            bind_result.known_classes, bind_result.class_bases
        )
        hier_errors = hier_result.errors()
        if hier_errors:
            return (None, str(hier_errors[0]))
        tc_result = collect_types(
            ast_dict,
            bind_result.known_classes,
            bind_result.node_classes,
            bind_result.type_aliases,
            bind_result.class_bases,
            hier_result.hierarchy_roots,
        )
        tc_errors = tc_result.errors()
        if tc_errors:
            return (None, str(tc_errors[0]))
        inf_result = _run_pycheck(
            ast_dict,
            tc_result,
            hier_result,
            bind_result.known_classes,
            bind_result.class_bases,
            bind_result.flow_graphs,
        )
        inf_errors = inf_result.errors()
        if inf_errors:
            return (None, str(inf_errors[0]))
        from src.frontend.lowering import lower

        module, lower_errors = lower(
            ast_dict,
            tc_result,
            hier_result,
            bind_result.known_classes,
            bind_result.class_bases,
            inf_result,
        )
        if lower_errors:
            return (None, str(lower_errors[0]))
        if module is None:
            return (None, "lowering produced no module")
        errors, checker = check_with_info(module)
        if errors:
            return (None, str(errors[0]))
        analyze_returns(module, checker)
        analyze_scope(module, checker)
        analyze_liveness(module, checker)
        return (emitter(module), None)
    except Exception as e:
        return (None, str(e))


def transpile_app(source: str, target: str) -> tuple[str | None, str | None]:
    """Transpile Python apptest source to target language. Returns (output, error)."""
    lib_names = _find_lib_imports(source)
    if TRANSPILED_BINARY is not None:
        if lib_names:
            lib_sources = _read_lib_sources(lib_names)
            stdin_data = _build_project_input("apptest.py", source, lib_sources)
            argv = [TRANSPILED_BINARY, "--project", "--target", target]
            if _TRANSPILED_MODULE is not None:
                result = _run_inprocess(argv, stdin_data=stdin_data)
            else:
                cmd = [*_transpiled_runtime(), *argv]
                result = subprocess.run(
                    cmd, input=stdin_data, capture_output=True, timeout=30
                )
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp:
                tmp.write(source)
                tmp.flush()
                argv = [TRANSPILED_BINARY, "--target", target, tmp.name]
                if _TRANSPILED_MODULE is not None:
                    result = _run_inprocess(argv)
                else:
                    cmd = [*_transpiled_runtime(), *argv]
                    result = subprocess.run(cmd, capture_output=True, timeout=30)
                Path(tmp.name).unlink(missing_ok=True)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            return (None, stderr.split("\n")[0] if stderr else "transpile failed")
        return (result.stdout.decode(errors="replace"), None)
    emitter = EMITTERS.get(target)
    if emitter is None:
        return (None, f"no emitter for target '{target}'")
    # For lib imports, work directly with TModule to preserve default params
    if lib_names:
        try:
            from src.frontend.lowering import lower
            from src.taytsh.ast import TModule

            def _lower_single_to_module(src: str):
                """Lower a single source to TModule."""
                ast_dict = parse(src)
                stamp_uids(ast_dict)
                bind_result = run_bind(ast_dict)
                if not bind_result.subset_ok():
                    return (None, bind_result.subset_violations[0].message)
                if not bind_result.names_ok():
                    return (None, bind_result.name_violations[0].message)
                hier_result = build_hierarchy(
                    bind_result.known_classes, bind_result.class_bases
                )
                hier_errors = hier_result.errors()
                if hier_errors:
                    return (None, str(hier_errors[0]))
                tc_result = collect_types(
                    ast_dict,
                    bind_result.known_classes,
                    bind_result.node_classes,
                    bind_result.type_aliases,
                    bind_result.class_bases,
                    hier_result.hierarchy_roots,
                )
                tc_errors = tc_result.errors()
                if tc_errors:
                    return (None, str(tc_errors[0]))
                inf_result = _run_pycheck(
                    ast_dict,
                    tc_result,
                    hier_result,
                    bind_result.known_classes,
                    bind_result.class_bases,
                    bind_result.flow_graphs,
                )
                inf_errors = inf_result.errors()
                if inf_errors:
                    return (None, str(inf_errors[0]))
                module, lower_errors = lower(
                    ast_dict,
                    tc_result,
                    hier_result,
                    bind_result.known_classes,
                    bind_result.class_bases,
                    inf_result,
                )
                if lower_errors:
                    return (None, str(lower_errors[0]))
                if module is None:
                    return (None, "lowering produced no module")
                return (module, None)

            # Lower lib modules first, then app module
            all_decls: list = []
            lib_sources = _read_lib_sources(lib_names)
            for _path, lib_src in lib_sources:
                lib_module, err = _lower_single_to_module(lib_src)
                if err is not None:
                    return (None, err)
                all_decls.extend(lib_module.decls)
            app_module, err = _lower_single_to_module(source)
            if err is not None:
                return (None, err)
            all_decls.extend(app_module.decls)
            merged = TModule(decls=all_decls)

            # Run Taytsh checker and analyzers directly on merged module
            errors, checker = check_with_info(merged)
            if errors:
                return (None, str(errors[0]))
            analyze_returns(merged, checker)
            analyze_scope(merged, checker)
            analyze_liveness(merged, checker)
            return (emitter(merged), None)
        except Exception as e:
            return (None, str(e))
    taytsh_text, err = lower_to_taytsh(source)
    if err is not None:
        return (None, err)
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
