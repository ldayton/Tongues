"""Fixed-point bootstrap verification: the transpiled transpiler reproduces itself."""

import importlib.util
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TONGUES_DIR = Path(__file__).parent.parent
SRC_DIR = TONGUES_DIR / "src"
OUT_DIR = TONGUES_DIR / ".out"


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


def _transpile_source(target: str) -> str:
    """Use the original transpiler to transpile its own source to the given target."""
    result = subprocess.run(
        [
            sys.executable,
            str(TONGUES_DIR / "bin" / "tongues"),
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


def test_fixed_point():
    """Stage 1 produces stage 2; stage 2 self-transpiles to identical stage 3."""
    stage2 = _transpile_source("python")
    mod = _load_python_binary(stage2, "fp_stage2")

    files = _gather_project_files(SRC_DIR)
    stdin_data = _build_project_stdin(files)
    stage3 = _run_python_binary(
        mod, ["stage2.py", "--project", "--target", "python"], stdin_data
    )

    assert stage2 == stage3, "stage2 != stage3: transpiler did not reach a fixed point"


@pytest.mark.xfail(reason="Ruby binary hits bytes iteration bug on full self-compile")
def test_cross_language_equivalence():
    """Python and Ruby transpiled binaries produce identical output."""
    if shutil.which("ruby") is None:
        pytest.skip("ruby not available")

    # Transpile the source to both Python and Ruby binaries
    py_source = _transpile_source("python")
    rb_source = _transpile_source("ruby")
    py_mod = _load_python_binary(py_source, "xl_python")
    rb_path = OUT_DIR / "xl_ruby.rb"
    rb_path.parent.mkdir(exist_ok=True)
    rb_path.write_text(rb_source)

    # Have each binary compile tongues source to Python
    files = _gather_project_files(SRC_DIR)
    stdin_data = _build_project_stdin(files)
    from_python = _run_python_binary(
        py_mod, ["xl_python.py", "--project", "--target", "python"], stdin_data
    )
    from_ruby = _run_ruby_binary(
        str(rb_path), ["--project", "--target", "python"], stdin_data
    )

    assert from_python == from_ruby, (
        "Python and Ruby transpiled binaries produced different output"
    )
