"""Fixed-point bootstrap verification: the transpiled transpiler reproduces itself."""

import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

TONGUES_DIR = Path(__file__).parent.parent
SRC_DIR = TONGUES_DIR / "src"


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


def _run_module_inprocess(mod, argv: list[str], stdin_data: bytes = b"") -> str:
    """Run a transpiled module's main() in-process, return stdout."""
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
        sys.argv = argv
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


def test_fixed_point():
    """Stage 1 produces stage 2; stage 2 self-transpiles to identical stage 3."""
    # Stage 1: original transpiler transpiles itself
    result = subprocess.run(
        [
            sys.executable,
            str(TONGUES_DIR / "bin" / "tongues"),
            "--target",
            "python",
            "src",
        ],
        capture_output=True,
        cwd=TONGUES_DIR,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    stage2 = result.stdout.decode()

    # Stage 2: load the transpiled binary and have it transpile the same source
    # Write to a temp file so importlib can resolve the module properly
    # (dataclasses needs sys.modules[cls.__module__] to exist)
    stage2_path = TONGUES_DIR / ".out" / "stage2.py"
    stage2_path.parent.mkdir(exist_ok=True)
    stage2_path.write_text(stage2)
    spec = importlib.util.spec_from_file_location("tongues_stage2", stage2_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    files = _gather_project_files(SRC_DIR)
    stdin_data = _build_project_stdin(files)
    stage3 = _run_module_inprocess(
        mod, ["stage2.py", "--project", "--target", "python"], stdin_data
    )

    assert stage2 == stage3, "stage2 != stage3: transpiler did not reach a fixed point"
