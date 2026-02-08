"""CLI tests for tongues entry point (01-cli-spec).

Test cases live in 01_cli/*.tests files. Format:

    === test name
    args: --target go --stop-at parse
    source code here
    (stdin for the transpiler)
    ---
    exit: 0
    stderr: error: some message
    stdout-contains: "keyword"
    stdout-empty: true
    stderr-empty: true
    exit-not: 2
    ---

Special directives in the input section:
    args:           CLI arguments (first line, required)
    stdin-bytes:    hex-encoded raw bytes instead of text (e.g. "ff fe")

Assertion directives in the expected section:
    exit:             exact exit code
    exit-not:         exit code must NOT equal this
    stderr:           exact stderr content (trailing newline added)
    stderr-contains:  stderr must contain substring
    stderr-empty:     stderr must be empty
    stdout-contains:  stdout must contain substring
    stdout-empty:     stdout must be empty
"""

import subprocess
import sys
from pathlib import Path

import pytest

CLI_DIR = Path(__file__).parent / "01_cli"
TONGUES_DIR = Path(__file__).parent.parent


def parse_cli_test_file(path: Path) -> list[tuple[str, dict]]:
    """Parse a .tests file into (name, spec) tuples.

    Each spec dict has keys: args, stdin, stdin_bytes, assertions.
    """
    lines = path.read_text().split("\n")
    result: list[tuple[str, dict]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("=== "):
            test_name = line[4:].strip()
            i += 1
            # Read input section (args line + stdin)
            input_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                input_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            # Read expected section
            expected_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("---"):
                expected_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i] == "---":
                i += 1
            spec = _parse_spec(input_lines, expected_lines)
            result.append((test_name, spec))
        else:
            i += 1
    return result


def _parse_spec(input_lines: list[str], expected_lines: list[str]) -> dict:
    """Parse input + expected lines into a test spec dict."""
    spec: dict = {
        "args": [],
        "stdin": None,
        "stdin_bytes": None,
        "assertions": [],
    }
    # First line must be args:
    body_start = 0
    if input_lines and input_lines[0].startswith("args:"):
        args_str = input_lines[0][5:].strip()
        spec["args"] = args_str.split() if args_str else []
        body_start = 1

    # Check for stdin-bytes directive
    remaining = input_lines[body_start:]
    if remaining and remaining[0].startswith("stdin-bytes:"):
        hex_str = remaining[0][len("stdin-bytes:") :].strip()
        spec["stdin_bytes"] = bytes.fromhex(hex_str)
    else:
        spec["stdin"] = "\n".join(remaining)

    # Parse assertions
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


def discover_cli_tests() -> list[tuple[str, dict]]:
    """Find all CLI tests across .tests files."""
    results = []
    for test_file in sorted(CLI_DIR.glob("*.tests")):
        tests = parse_cli_test_file(test_file)
        for name, spec in tests:
            test_id = f"{test_file.stem}/{name}"
            results.append((test_id, spec))
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
    return subprocess.run(
        cmd,
        input=stdin_data,
        capture_output=True,
        cwd=TONGUES_DIR,
    )


def check_assertions(
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


def pytest_generate_tests(metafunc):
    """Parametrize test_cli over all .tests files."""
    if "cli_spec" in metafunc.fixturenames:
        tests = discover_cli_tests()
        params = [pytest.param(spec, id=test_id) for test_id, spec in tests]
        metafunc.parametrize("cli_spec", params)


def test_cli(cli_spec: dict) -> None:
    """Run a single CLI test case from .tests file."""
    result = run_cli(cli_spec)
    check_assertions(result, cli_spec["assertions"])
