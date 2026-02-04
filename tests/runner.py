"""Apptest runner: transpile each apptest_*.py through Tongues, then execute it."""

import glob
import os
import subprocess
import sys

TESTS_DIR = os.path.join(os.path.dirname(__file__), "pypy")
OUT_DIR = os.path.join(TESTS_DIR, ".out")
TONGUES_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "tongues")

TARGETS = {
    "python": {"ext": ".py", "run": lambda p: [sys.executable, p]},
    "javascript": {"ext": ".js", "run": lambda p: ["node", p]},
    "typescript": {"ext": ".ts", "run": lambda p: ["npx", "tsx", p]},
    "ruby": {"ext": ".rb", "run": lambda p: ["ruby", p]},
}


def find_apptests():
    return sorted(glob.glob(os.path.join(TESTS_DIR, "apptest_*.py")))


def transpile(source_path, target, out_path):
    """Transpile a Python file through Tongues. Returns error or None."""
    with open(source_path) as f:
        source = f.read()
    result = subprocess.run(
        [sys.executable, "-m", "src.tongues", "--target", target],
        input=source,
        capture_output=True,
        text=True,
        cwd=TONGUES_DIR,
    )
    if result.returncode != 0:
        return result.stderr.strip()
    with open(out_path, "w") as f:
        f.write(result.stdout)
    if target == "python":
        subprocess.run(
            ["uvx", "ruff", "format", "--quiet", out_path],
            capture_output=True,
        )
    elif target == "javascript":
        subprocess.run(
            ["npx", "@biomejs/biome", "format", "--write", out_path],
            capture_output=True,
        )
    return None


def execute(out_path, run_cmd):
    """Execute transpiled code. Returns (success, output)."""
    try:
        result = subprocess.run(
            run_cmd(out_path),
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

    passed = 0
    failed = 0
    errors = []

    for target, cfg in TARGETS.items():
        target_dir = os.path.join(OUT_DIR, target)
        os.makedirs(target_dir, exist_ok=True)
        print(target + ":")
        for path in apptests:
            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(target_dir, stem + cfg["ext"])
            label = target + "/" + stem

            err = transpile(path, target, out_path)
            if err is not None:
                failed += 1
                errors.append((label, "transpile: " + err))
                print("  FAIL " + stem + " (transpile)")
                continue

            ok, output = execute(out_path, cfg["run"])
            if ok:
                passed += 1
                print("  PASS " + stem)
            else:
                failed += 1
                errors.append((label, "execute: " + output))
                print("  FAIL " + stem + " (execute)")
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
