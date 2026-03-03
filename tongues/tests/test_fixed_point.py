"""Fixed-point bootstrap verification: the transpiled transpiler reproduces itself."""

import importlib.util
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.timeout(60)

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


def _project_stdin() -> bytes:
    """Gather project files and build stdin data (cached-ish via call site)."""
    return _build_project_stdin(_gather_project_files(SRC_DIR))


def test_fixed_point():
    """Stage 1 produces stage 2; stage 2 self-transpiles to identical stage 3."""
    stage2 = _transpile_source("python")
    mod = _load_python_binary(stage2, "fp_stage2")

    stdin_data = _project_stdin()
    stage3 = _run_python_binary(
        mod, ["stage2.py", "--project", "--target", "python"], stdin_data
    )

    assert stage2 == stage3, "stage2 != stage3: transpiler did not reach a fixed point"


def test_taytsh_emit_round_trip():
    """Lowering to Taytsh text, parsing, and re-emitting produces identical output."""
    from src.taytsh.emit import to_source
    from src.taytsh.parse import Parser
    from src.taytsh.tokens import tokenize

    stdin_data = _project_stdin()
    ty_text = _run_original(["--project", "--stop-at", "lowering-text"], stdin_data)

    tokens = tokenize(ty_text)
    module = Parser(tokens).parse_program()
    ty_text_2 = to_source(module)

    assert ty_text == ty_text_2, "Taytsh emit round-trip is not idempotent"


@pytest.mark.parametrize("target", ["ruby", "perl"])
def test_cross_target_agreement(target: str):
    """Transpiled Python binary produces same backend output as the original."""
    py_source = _transpile_source("python")
    py_mod = _load_python_binary(py_source, f"ct_{target}")

    stdin_data = _project_stdin()
    original = _run_original(["--project", "--target", target], stdin_data)
    transpiled = _run_python_binary(
        py_mod, ["ct.py", "--project", "--target", target], stdin_data
    )

    assert original == transpiled, f"--target {target}: original != transpiled"


@pytest.mark.parametrize(
    "phase",
    [
        "parse",
        "names",
        "pycheck",
        "lowering",
        "lowering-text",
    ],
)
def test_phase_output_agreement(phase: str):
    """Transpiled Python binary produces same intermediate output as the original."""
    py_source = _transpile_source("python")
    py_mod = _load_python_binary(py_source, f"po_{phase}")

    stdin_data = _project_stdin()
    original = _run_original(["--project", "--stop-at", phase], stdin_data)
    transpiled = _run_python_binary(
        py_mod, ["po.py", "--project", "--stop-at", phase], stdin_data
    )

    assert original == transpiled, f"--stop-at {phase}: original != transpiled"


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


@pytest.mark.timeout(600)
def test_cross_language_equivalence_perl():
    """Python and Perl transpiled binaries produce identical output."""
    if shutil.which("perl") is None:
        pytest.skip("perl not available")

    py_source = _transpile_source("python")
    pl_source = _transpile_source("perl")
    py_mod = _load_python_binary(py_source, "xl_py_perl")
    pl_path = OUT_DIR / "xl_perl.pl"
    pl_path.parent.mkdir(exist_ok=True)
    pl_path.write_text(pl_source)

    files = _gather_project_files(SRC_DIR)
    stdin_data = _build_project_stdin(files)
    from_python = _run_python_binary(
        py_mod, ["xl_py_perl.py", "--project", "--target", "python"], stdin_data
    )
    from_perl = _run_perl_binary(
        str(pl_path), ["--project", "--target", "python"], stdin_data
    )

    assert from_python == from_perl, (
        "Python and Perl transpiled binaries produced different output"
    )
