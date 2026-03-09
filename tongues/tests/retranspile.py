#!/usr/bin/env python3
"""Run a transpiled binary on a source directory, targeting its own language.

Usage: retranspile.py <binary> <src_dir> <target> <output_file>

Loads the transpiled binary and feeds it the project source via --project mode.
For Python, runs in-process; for Ruby/Perl/JS, runs as a subprocess.
"""

import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent


def gather_project_files(root):
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d != "__pycache__"
        )
        for fname in sorted(filenames):
            if not fname.endswith(".py") or fname.startswith("."):
                continue
            path = os.path.join(dirpath, fname)
            with open(path) as f:
                source = f.read()
            for line in source.split("\n", 5)[:5]:
                if "tongues: skip" in line:
                    break
            else:
                results.append((os.path.relpath(path, root), source))
    results.sort()
    return results


def build_project_stdin(files):
    parts = []
    for relpath, source in files:
        parts.append(relpath)
        parts.append(source)
    return "\0".join(parts)


def run_python(binary_path, stdin_data, target):
    spec = importlib.util.spec_from_file_location("tongues_retranspile", binary_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tongues_retranspile"] = mod
    spec.loader.exec_module(mod)
    old_argv, old_stdout, old_stderr, old_stdin = (
        sys.argv,
        sys.stdout,
        sys.stderr,
        sys.stdin,
    )
    out = io.StringIO()
    err = io.StringIO()
    try:
        sys.argv = ["retranspile.py", "--project", "--target", target]
        sys.stdout = out
        sys.stderr = err
        sys.stdin = io.TextIOWrapper(io.BytesIO(stdin_data.encode("utf-8")))
        mod.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            sys.argv, sys.stdout, sys.stderr, sys.stdin = (
                old_argv,
                old_stdout,
                old_stderr,
                old_stdin,
            )
            print(
                f"Retranspile failed (exit {e.code}): {err.getvalue()}", file=sys.stderr
            )
            sys.exit(1)
    finally:
        sys.argv, sys.stdout, sys.stderr, sys.stdin = (
            old_argv,
            old_stdout,
            old_stderr,
            old_stdin,
        )
    return out.getvalue()


def run_subprocess(cmd, stdin_data):
    result = subprocess.run(
        cmd,
        input=stdin_data.encode("utf-8"),
        capture_output=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(
            f"Retranspile failed (exit {result.returncode}): {result.stderr.decode(errors='replace')}",
            file=sys.stderr,
        )
        sys.exit(1)
    return result.stdout.decode()


def run_ruby(binary_path, stdin_data, target):
    return run_subprocess(
        [
            "ruby",
            "-W0",
            "-e",
            f"load '{binary_path}'; main",
            "--",
            "--project",
            "--target",
            target,
        ],
        stdin_data,
    )


def run_perl(binary_path, stdin_data, target):
    return run_subprocess(
        [
            "perl",
            "-e",
            f"do '{binary_path}'; die $@ if $@; main()",
            "--",
            "--project",
            "--target",
            target,
        ],
        stdin_data,
    )


def run_javascript(binary_path, stdin_data, target):
    helper = str(TESTS_DIR / "run-js-main.js")
    return run_subprocess(
        ["node", helper, binary_path, "--project", "--target", target],
        stdin_data,
    )


def run_java(binary_path, stdin_data, target):
    import shutil
    import tempfile

    # Compile the Java file to a temp directory
    tmpdir = tempfile.mkdtemp()
    try:
        shutil.copy(binary_path, os.path.join(tmpdir, "Main.java"))
        compile_result = subprocess.run(
            ["javac", "-encoding", "UTF-8", "Main.java"],
            cwd=tmpdir,
            capture_output=True,
        )
        if compile_result.returncode != 0:
            print(
                f"Java compilation failed: {compile_result.stderr.decode(errors='replace')}",
                file=sys.stderr,
            )
            sys.exit(1)
        return run_subprocess(
            ["java", "-cp", tmpdir, "Main", "--project", "--target", target],
            stdin_data,
        )
    finally:
        shutil.rmtree(tmpdir)


RUNNERS = {
    ".py": run_python,
    ".rb": run_ruby,
    ".pl": run_perl,
    ".js": run_javascript,
    ".java": run_java,
}


if __name__ == "__main__":
    binary_path = os.path.abspath(sys.argv[1])
    src_dir = sys.argv[2]
    target = sys.argv[3]
    output_file = sys.argv[4]

    ext = os.path.splitext(binary_path)[1]
    runner = RUNNERS.get(ext)
    if runner is None:
        print(f"Unknown binary extension: {ext}", file=sys.stderr)
        sys.exit(1)

    files = gather_project_files(src_dir)
    stdin_data = build_project_stdin(files)
    output = runner(binary_path, stdin_data, target)

    with open(output_file, "w") as f:
        f.write(output)
