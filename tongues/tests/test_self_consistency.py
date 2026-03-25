"""Per-language self-consistency: fixed-point bootstrap and reference output."""

import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.timeout(600)

TONGUES_DIR = Path(__file__).parent.parent
SRC_DIR = TONGUES_DIR / "src"
OUT_DIR = TONGUES_DIR / ".out"
TESTS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Determine which binary we're testing
# ---------------------------------------------------------------------------


def _binary_path() -> str | None:
    return os.environ.get("TONGUES_TRANSPILED_BINARY")


def _binary_ext() -> str | None:
    p = _binary_path()
    if p is None:
        return None
    return Path(p).suffix


# ---------------------------------------------------------------------------
# Project stdin helpers
# ---------------------------------------------------------------------------


def _gather_project_files(project_root: Path) -> list[tuple[str, str]]:
    """Collect .py files as (relpath, source) pairs, matching bin/tongues logic."""
    results: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d != "__pycache__"
        )
        for fname in sorted(filenames):
            if not fname.endswith(".py") or fname.startswith("."):
                continue
            full_path = os.path.join(dirpath, fname)
            with open(full_path) as f:
                source = f.read()
            for line in source.split("\n", 5)[:5]:
                if "tongues: skip" in line:
                    break
            else:
                relpath = os.path.relpath(full_path, project_root)
                results.append((relpath, source))
    results.sort(key=lambda x: x[0])
    return results


def _build_project_stdin(files: list[tuple[str, str]]) -> bytes:
    """Build NUL-delimited stdin from project files."""
    parts: list[str] = []
    for path, source in files:
        parts.append(path)
        parts.append(source)
    return "\0".join(parts).encode()


def _project_stdin() -> bytes:
    return _build_project_stdin(_gather_project_files(SRC_DIR))


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------


def _run_original(args: list[str], stdin_data: bytes = b"") -> str:
    """Run the original transpiler with given args, return stdout."""
    result = subprocess.run(
        [sys.executable, str(TONGUES_DIR / "bin" / "tongues"), *args],
        input=stdin_data,
        capture_output=True,
        cwd=TONGUES_DIR,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout.decode()


def _load_python_binary(source: str, name: str):
    """Write transpiled Python to .out/ and load as a module."""
    path = OUT_DIR / f"{name}.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"tongues_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_python_binary(mod, args: list[str], stdin_data: bytes = b"") -> str:
    """Run a transpiled Python module's main() in-process, return stdout."""
    old_argv, old_stdout, old_stderr, old_stdin = (
        sys.argv,
        sys.stdout,
        sys.stderr,
        sys.stdin,
    )
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    stdin_wrapper = io.TextIOWrapper(io.BytesIO(stdin_data))
    try:
        sys.argv = args
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf
        sys.stdin = stdin_wrapper
        mod.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            raise RuntimeError(
                f"main() exited with code {e.code}: {stderr_buf.getvalue()}"
            )
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin = old_stdin
    return stdout_buf.getvalue()


def _run_ruby_binary(rb_path: str, args: list[str], stdin_data: bytes = b"") -> str:
    """Run a transpiled Ruby binary via subprocess, return stdout."""
    result = subprocess.run(
        ["ruby", "-W0", "-e", f"load '{rb_path}'; main", "--", *args],
        input=stdin_data,
        capture_output=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout.decode()


def _run_perl_binary(pl_path: str, args: list[str], stdin_data: bytes = b"") -> str:
    """Run a transpiled Perl binary via subprocess, return stdout."""
    result = subprocess.run(
        ["perl", "-e", f"do '{pl_path}'; die $@ if $@; main()", "--", *args],
        input=stdin_data,
        capture_output=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout.decode()


def _run_js_binary(js_path: str, args: list[str], stdin_data: bytes = b"") -> str:
    """Run a transpiled JS binary via node helper, return stdout."""
    helper = str(TESTS_DIR / "run-js-main.js")
    result = subprocess.run(
        ["node", helper, js_path, *args],
        input=stdin_data,
        capture_output=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout.decode()


def _dispatch(args: list[str], stdin_data: bytes = b"") -> str:
    """Run the transpiled binary indicated by TRANSPILED_BINARY env var."""
    path = _binary_path()
    assert path is not None, "TRANSPILED_BINARY not set"
    ext = _binary_ext()
    if ext == ".py":
        spec = importlib.util.spec_from_file_location("tongues_sc", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return _run_python_binary(mod, ["sc.py", *args], stdin_data)
    elif ext == ".rb":
        return _run_ruby_binary(path, args, stdin_data)
    elif ext == ".pl":
        return _run_perl_binary(path, args, stdin_data)
    elif ext == ".js":
        return _run_js_binary(path, args, stdin_data)
    elif ext == ".ty":
        # Taytsh uses the original Python runtime to execute
        result = subprocess.run(
            [
                sys.executable,
                str(TONGUES_DIR / "bin" / "tongues"),
                "taytsh",
                *args,
                path,
            ],
            input=stdin_data,
            capture_output=True,
            cwd=TONGUES_DIR,
            timeout=600,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        return result.stdout.decode()
    else:
        raise ValueError(f"Unknown binary extension: {ext}")


def _transpile_source(target: str) -> str:
    """Use the original transpiler to transpile its own source to the given target."""
    result = subprocess.run(
        [
            sys.executable,
            str(TONGUES_DIR / "bin" / "tongues"),
            "--strict-tostring",
            "--target",
            target,
            "src",
        ],
        capture_output=True,
        cwd=TONGUES_DIR,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout.decode()


# ---------------------------------------------------------------------------
# Determine language from extension for skip logic
# ---------------------------------------------------------------------------

_EXT_TO_LANG = {
    ".py": "python",
    ".rb": "ruby",
    ".pl": "perl",
    ".js": "javascript",
    ".ty": "taytsh",
}
_EXT_TO_TARGET = {".py": "python", ".rb": "ruby", ".pl": "perl", ".js": "javascript"}


def _current_lang() -> str | None:
    ext = _binary_ext()
    return _EXT_TO_LANG.get(ext) if ext else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fixed_point():
    """The transpiled binary self-transpiles to its own language; output matches the on-disk stage 2 file."""
    path = _binary_path()
    if path is None:
        pytest.skip("no --transpiled binary")
    ext = _binary_ext()
    target = _EXT_TO_TARGET.get(ext)
    if target is None and ext == ".ty":
        # Taytsh fixed-point: self-transpile to taytsh, compare with on-disk .ty
        stdin_data = _project_stdin()
        stage2_on_disk = Path(path).read_text()
        stage3 = _dispatch(["--project", "--stop-at", "lowering-text"], stdin_data)
        assert stage2_on_disk == stage3, (
            "Taytsh fixed-point failed: on-disk .ty != re-emitted"
        )
        return
    if target is None:
        pytest.skip(f"unknown extension {ext}")
    stdin_data = _project_stdin()
    stage2_on_disk = Path(path).read_text()
    stage3 = _dispatch(["--project", "--target", target], stdin_data)
    assert stage2_on_disk == stage3, (
        f"Fixed-point failed for {target}: on-disk stage2 != self-transpiled stage3"
    )


def test_phase_agreement():
    """Transpiled Python binary produces same intermediate output as the original."""
    if _current_lang() != "python":
        pytest.skip("python-only test")
    path = _binary_path()
    assert path is not None
    spec = importlib.util.spec_from_file_location("tongues_pa", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    stdin_data = _project_stdin()
    for phase in ["parse", "names", "pycheck", "lowering", "lowering-text"]:
        original = _run_original(["--project", "--stop-at", phase], stdin_data)
        transpiled = _run_python_binary(
            mod, ["pa.py", "--project", "--stop-at", phase], stdin_data
        )
        assert original == transpiled, f"--stop-at {phase}: original != transpiled"


@pytest.mark.parametrize("target", ["ruby", "perl"])
def test_cross_target_agreement(target: str):
    """Transpiled Python binary produces same backend output as the original."""
    if _current_lang() != "python":
        pytest.skip("python-only test")
    path = _binary_path()
    assert path is not None
    spec = importlib.util.spec_from_file_location(f"tongues_ct_{target}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    stdin_data = _project_stdin()
    original = _run_original(["--project", "--target", target], stdin_data)
    transpiled = _run_python_binary(
        mod, ["ct.py", "--project", "--target", target], stdin_data
    )
    assert original == transpiled, f"--target {target}: original != transpiled"


def test_taytsh_emit_round_trip():
    """Lowering to Taytsh text, parsing, and re-emitting produces identical output."""
    if _current_lang() not in ("python", None):
        pytest.skip("python-only test (uses src.taytsh imports)")
    from src.taytsh.emit import to_source
    from src.taytsh.parse import Parser
    from src.taytsh.tokens import tokenize

    stdin_data = _project_stdin()
    ty_text = _run_original(["--project", "--stop-at", "lowering-text"], stdin_data)
    tokens = tokenize(ty_text)
    module = Parser(tokens).parse_program()
    ty_text_2 = to_source(module)
    assert ty_text == ty_text_2, "Taytsh emit round-trip is not idempotent"


def test_taytsh_unicode_escape_round_trip():
    """Unicode escape sequences in taytsh text survive tokenize-parse-emit-retokenize."""
    if _current_lang() not in ("python", None):
        pytest.skip("python-only test (uses src.taytsh imports)")
    from src.taytsh.emit import to_source
    from src.taytsh.parse import Parser
    from src.taytsh.tokens import tokenize

    source = (
        'fn Main() -> void {\n'
        '    let bmp: string = "caf\\u00e9"\n'
        '    let astral: string = "\\U0001f600"\n'
        '    let mixed: string = "\\u03b1 \\U0001f601 \\u03b2"\n'
        '    let r: rune = \'\\u00e9\'\n'
        '    WritelnOut(bmp)\n'
        '}\n'
    )
    tokens1 = tokenize(source)
    module1 = Parser(tokens1).parse_program()
    emitted = to_source(module1)
    tokens2 = tokenize(emitted)
    module2 = Parser(tokens2).parse_program()
    emitted2 = to_source(module2)
    assert emitted == emitted2, (
        f"Unicode escape round-trip not idempotent:\n"
        f"--- first emit ---\n{emitted}\n"
        f"--- second emit ---\n{emitted2}"
    )


def test_reference_emit():
    """Transpiled binary emits Python output; save to .out/ for cross-language comparison."""
    path = _binary_path()
    if path is None:
        pytest.skip("no --transpiled binary")
    ext = _binary_ext()
    if ext == ".ty":
        pytest.skip("Taytsh cannot target Python directly")
    lang = _current_lang()
    assert lang is not None
    stdin_data = _project_stdin()
    output = _dispatch(["--project", "--target", "python"], stdin_data)
    assert len(output) > 0, "reference emit produced empty output"
    ref_path = OUT_DIR / f"reference-python-{lang}.txt"
    ref_path.parent.mkdir(exist_ok=True)
    ref_path.write_text(output)
