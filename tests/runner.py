"""Apptest runner: transpile each apptest_*.py through Tongues, then execute it."""

import glob
import os
import subprocess
import sys

TESTS_DIR = os.path.join(os.path.dirname(__file__), "pypy")
OUT_DIR = os.path.join(TESTS_DIR, ".out")
TONGUES_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "tongues")


def find_apptests():
    return sorted(glob.glob(os.path.join(TESTS_DIR, "apptest_*.py")))


def transpile(source_path, out_path):
    """Transpile a Python file through Tongues (Python → Python). Returns error or None."""
    with open(source_path) as f:
        source = f.read()
    result = subprocess.run(
        [sys.executable, "-m", "src.tongues", "--target", "python"],
        input=source,
        capture_output=True,
        text=True,
        cwd=TONGUES_DIR,
    )
    if result.returncode != 0:
        return result.stderr.strip()
    with open(out_path, "w") as f:
        f.write(result.stdout)
    subprocess.run(
        ["uvx", "ruff", "format", "--quiet", out_path],
        capture_output=True,
    )
    return None


def execute(out_path):
    """Execute transpiled Python code. Returns (success, output)."""
    try:
        result = subprocess.run(
            [sys.executable, out_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        return False, "timed out"


def main():
    apptests = find_apptests()
    if not apptests:
        print("No apptest_*.py files found in " + TESTS_DIR)
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)

    passed = 0
    failed = 0
    errors = []

    for path in apptests:
        name = os.path.basename(path)
        out_path = os.path.join(OUT_DIR, name)

        err = transpile(path, out_path)
        if err is not None:
            failed += 1
            errors.append((name, "transpile: " + err))
            print("  FAIL " + name + " (transpile)")
            continue

        ok, output = execute(out_path)
        if ok:
            passed += 1
            print("  PASS " + name)
        else:
            failed += 1
            errors.append((name, "execute: " + output))
            print("  FAIL " + name + " (execute)")

    print()
    if errors:
        print("FAILURES:")
        for name, msg in errors:
            print("  " + name)
            for line in msg.split("\n"):
                print("    " + line)
        print()

    print(str(passed) + " passed, " + str(failed) + " failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
