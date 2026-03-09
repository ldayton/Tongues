#!/usr/bin/env python3
"""Native Python test harness for transpiled Tongues binaries.

Loads the transpiled file once, then runs all .tests cases in-process.
Shared parsing/assertion logic is transpiled from tests/shared/test_harness.py.

Supports parallel execution with -n <num> or -n auto (like pytest-xdist).
"""

import glob
import importlib.util
import io
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

TONGUES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(TONGUES_DIR, "tests")
LIB_DIR = os.path.join(TONGUES_DIR, "src", "lib")

# Phase -> test config
# Runners: cli, linker, phase, lowering, codegen, emit, app, ordering, ty_app
TESTS = [
    (
        "cli",
        [
            ("cli", {"dir": "frontend/cli", "run": "cli"}),
        ],
    ),
    (
        "linker",
        [
            ("linker", {"dir": "frontend/linker", "run": "linker"}),
        ],
    ),
    (
        "frontend",
        [
            (
                "parse",
                {
                    "dir": "frontend/parse",
                    "run": "phase",
                    "taytsh": False,
                    "args": ["--stop-at", "parse"],
                    "json": True,
                },
            ),
            (
                "subset",
                {
                    "dir": "frontend/subset",
                    "run": "phase",
                    "taytsh": False,
                    "args": ["--stop-at", "subset"],
                    "json": False,
                },
            ),
            (
                "names",
                {
                    "dir": "frontend/names",
                    "run": "phase",
                    "taytsh": False,
                    "args": ["--stop-at", "names"],
                    "json": True,
                },
            ),
            (
                "sigs",
                {
                    "dir": "frontend/signatures",
                    "run": "phase",
                    "taytsh": False,
                    "args": ["--stop-at", "signatures"],
                    "json": True,
                },
            ),
            (
                "fields",
                {
                    "dir": "frontend/fields",
                    "run": "phase",
                    "taytsh": False,
                    "args": ["--stop-at", "fields"],
                    "json": True,
                },
            ),
            (
                "hierarchy",
                {
                    "dir": "frontend/hierarchy",
                    "run": "phase",
                    "taytsh": False,
                    "args": ["--stop-at", "hierarchy"],
                    "json": True,
                },
            ),
            (
                "pycheck",
                {
                    "dir": "frontend/pycheck",
                    "run": "phase",
                    "taytsh": False,
                    "args": ["--stop-at", "pycheck"],
                    "json": True,
                },
            ),
            ("lowering", {"dir": "frontend/lowering", "run": "lowering"}),
        ],
    ),
    (
        "middleend",
        [
            (
                "scope",
                {
                    "dir": "middleend/scope",
                    "run": "phase",
                    "taytsh": True,
                    "args": ["--stop-at", "scope"],
                    "json": True,
                },
            ),
            (
                "returns",
                {
                    "dir": "middleend/returns",
                    "run": "phase",
                    "taytsh": True,
                    "args": ["--stop-at", "returns"],
                    "json": True,
                },
            ),
            (
                "liveness",
                {
                    "dir": "middleend/liveness",
                    "run": "phase",
                    "taytsh": True,
                    "args": ["--stop-at", "liveness"],
                    "json": True,
                },
            ),
            (
                "strings",
                {
                    "dir": "middleend/strings",
                    "run": "phase",
                    "taytsh": True,
                    "args": ["--stop-at", "strings"],
                    "json": True,
                },
            ),
            (
                "hoisting",
                {
                    "dir": "middleend/hoisting",
                    "run": "phase",
                    "taytsh": True,
                    "args": ["--stop-at", "hoisting"],
                    "json": True,
                },
            ),
            (
                "ownership",
                {
                    "dir": "middleend/ownership",
                    "run": "phase",
                    "taytsh": True,
                    "args": ["--stop-at", "ownership"],
                    "json": True,
                },
            ),
            (
                "callgraph",
                {
                    "dir": "middleend/callgraph",
                    "run": "phase",
                    "taytsh": True,
                    "args": ["--stop-at", "callgraph"],
                    "json": True,
                },
            ),
        ],
    ),
    (
        "backend",
        [
            ("codegen", {"dir": "backend/codegen", "run": "codegen"}),
            ("emit", {"dir": "backend/emit", "run": "emit"}),
            ("app", {"dir": "backend/app", "run": "app"}),
            ("ordering", {"dir": "backend/ordering", "run": "ordering"}),
        ],
    ),
    (
        "taytsh",
        [
            (
                "typarse",
                {
                    "dir": "taytsh/typarse",
                    "run": "phase",
                    "taytsh": True,
                    "args": ["--stop-at", "parse"],
                    "json": True,
                },
            ),
            (
                "tycheck",
                {
                    "dir": "taytsh/tycheck",
                    "run": "phase",
                    "taytsh": True,
                    "args": ["--stop-at", "check"],
                    "json": True,
                },
            ),
            ("ty_app", {"dir": "taytsh/app", "run": "ty_app"}),
        ],
    ),
]

EMITTER_LANGS = ["python"]
RUNTIMES = {
    "python": ["python3"],
}

# Reference to the transpiled module's main function, set after loading.
_main_func = None
_use_vm = False
_vm_compiled = None

# Worker process state (used by parallel execution)
_worker_mod = None
_worker_main_func = None
_worker_harness_mod = None
_worker_use_vm = False
_worker_vm_compiled = None


# ---------------------------------------------------------------------------
# VM mode: parse + compile .ty once, invoke per test
# ---------------------------------------------------------------------------


_vm_mod = None  # reference to the transpiled module for VM API access


def _load_vm_module(ty_path):
    global _vm_compiled
    with open(ty_path) as f:
        source = f.read()
    module = _vm_mod.taytsh_taytsh_parse(source)
    _vm_compiled = _vm_mod.vm_prepare(module)
    print("VM module compiled")


def _run_vm_inprocess(argv, stdin_data="", stdin_bytes=None):
    raw = stdin_bytes if stdin_bytes is not None else stdin_data.encode("utf-8")
    instance = _vm_mod.VM(module=_vm_compiled)
    instance.builtins.vm = instance
    try:
        result = instance.invoke(raw, ["tongues"] + list(argv))
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit": 1}
    stdout = (
        result.stdout
        if isinstance(result.stdout, str)
        else result.stdout.decode("utf-8")
    )
    stderr = (
        result.stderr
        if isinstance(result.stderr, str)
        else result.stderr.decode("utf-8")
    )
    return {"stdout": stdout, "stderr": stderr, "exit": result.exit_code}


# ---------------------------------------------------------------------------
# In-process execution
# ---------------------------------------------------------------------------


class _StdinWrapper(io.StringIO):
    """StringIO with a .buffer attribute for code that reads binary stdin."""

    def __init__(self, text, raw_bytes=None):
        super().__init__(text)
        self.buffer = io.BytesIO(
            raw_bytes if raw_bytes is not None else text.encode("utf-8")
        )


def run_inprocess(argv, stdin_data="", stdin_bytes=None):
    if _use_vm:
        return _run_vm_inprocess(argv, stdin_data=stdin_data, stdin_bytes=stdin_bytes)
    old_argv = sys.argv[:]
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdin = sys.stdin
    out = io.StringIO()
    err = io.StringIO()
    code = 0
    try:
        sys.argv = ["tongues"] + list(argv)
        sys.stdout = out
        sys.stderr = err
        sys.stdin = _StdinWrapper(stdin_data, raw_bytes=stdin_bytes)
        _main_func()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        err.write(str(e) + "\n")
        code = 1
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin = old_stdin
    return {"stdout": out.getvalue(), "stderr": err.getvalue(), "exit": code}


def _get_json_parse():
    """Get json_parse function from harness module (works in main and worker processes)."""
    if _worker_harness_mod is not None:
        return _worker_harness_mod.json_parse
    # Fall back to global (main process)
    return json_parse


def run_transpiled_phase(source, cli_args, is_taytsh=False, expect_json=True):
    suffix = ".ty" if is_taytsh else ".py"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(source)
        if is_taytsh:
            argv = ["taytsh"] + list(cli_args) + [tmp_path]
        else:
            argv = list(cli_args) + [tmp_path]
        result = run_inprocess(argv)
    finally:
        os.unlink(tmp_path)
    stderr_text = result["stderr"].strip()
    if result["exit"] != 0:
        errors = [line for line in stderr_text.split("\n") if line]
        return {"errors": errors, "warnings": [], "data": None, "reveals": []}
    warnings = [line for line in stderr_text.split("\n") if line] if stderr_text else []
    if not expect_json:
        return {"errors": [], "warnings": warnings, "data": None, "reveals": []}
    stdout_text = result["stdout"].strip()
    if not stdout_text:
        return {"errors": [], "warnings": warnings, "data": None, "reveals": []}
    try:
        data = _get_json_parse()(stdout_text)
    except Exception:
        return {
            "errors": [f"Invalid JSON output: {stdout_text[:200]}"],
            "warnings": [],
            "data": None,
            "reveals": [],
        }
    return {"errors": [], "warnings": warnings, "data": data, "reveals": []}


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------


def run_cli_tests(test_dir):
    results = []
    for f in sorted(glob.glob(os.path.join(test_dir, "*.tests"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        with open(f) as fh:
            content = fh.read()
        for name, spec in parse_cli_test_file(content):
            test_id = f"{stem}/{name}"
            if cli_needs_backend(spec.args, spec.assertions, EMITTER_LANGS):
                results.append(("skip", test_id, None))
                continue
            if spec.stdin_hex:
                raw = bytes.fromhex(spec.stdin_hex)
                result = run_inprocess(
                    spec.args, stdin_data=raw.decode("latin-1"), stdin_bytes=raw
                )
            else:
                result = run_inprocess(spec.args, stdin_data=spec.stdin)
            err = check_cli_assertions(
                result["exit"], result["stdout"], result["stderr"], spec.assertions
            )
            if err:
                results.append(("fail", test_id, err))
            else:
                results.append(("pass", test_id, None))
    return results


def run_linker_tests(test_dir):
    results = []
    for f in sorted(glob.glob(os.path.join(test_dir, "*.tests"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        with open(f) as fh:
            content = fh.read()
        for name, spec in parse_linker_test_file(content):
            test_id = f"{stem}/{name}"
            parts = []
            for lf in spec.files:
                parts.append(lf.path)
                parts.append(lf.source)
            stdin_data = "\0".join(parts)
            args = spec.args
            if "--target" in args:
                idx = args.index("--target")
                target = args[idx + 1]
                if target not in EMITTER_LANGS:
                    results.append(("skip", test_id, None))
                    continue
            result = run_inprocess(args, stdin_data=stdin_data)
            err = check_cli_assertions(
                result["exit"], result["stdout"], result["stderr"], spec.assertions
            )
            if err:
                results.append(("fail", test_id, err))
            else:
                results.append(("pass", test_id, None))
    return results


def run_phase_tests(test_dir, phase_name, cfg):
    results = []
    for f in sorted(glob.glob(os.path.join(test_dir, "*.tests"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        with open(f) as fh:
            content = fh.read()
        for entry in parse_spec_file(content):
            test_id = f"{stem}/{entry.name}"
            lenient = phase_name in ("parse", "pycheck", "typarse", "tycheck")
            phase_result = run_transpiled_phase(
                entry.input_,
                cfg["args"],
                is_taytsh=cfg["taytsh"],
                expect_json=cfg["json"],
            )
            reveals = phase_result["reveals"]
            if (
                phase_name in ("pycheck", "tycheck")
                and not phase_result["errors"]
                and phase_result["data"]
            ):
                if isinstance(phase_result["data"], JsonObject):
                    try:
                        reveals_arr = json_get_items(
                            json_get_field(phase_result["data"], "reveals")
                        )
                        reveals = [
                            (
                                int(json_get_number(json_get_field(r, "line"))),
                                json_get_string(json_get_field(r, "type")),
                            )
                            for r in reveals_arr
                        ]
                    except Exception:
                        pass
            err = check_expected(
                entry.expected,
                phase_result["errors"],
                phase_result["warnings"],
                phase_result["data"],
                reveals,
                phase_name,
                lenient,
            )
            if err:
                results.append(("fail", test_id, err))
            else:
                results.append(("pass", test_id, None))
    return results


def run_lowering_tests(test_dir):
    results = []
    for f in sorted(glob.glob(os.path.join(test_dir, "*.tests"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        with open(f) as fh:
            content = fh.read()
        for entry in parse_spec_file(content):
            test_id = f"{stem}/{entry.name}"
            fd, tmp_path = tempfile.mkstemp(suffix=".py")
            try:
                with os.fdopen(fd, "w") as tf:
                    tf.write(entry.input_)
                result = run_inprocess(["--stop-at", "lowering-text", tmp_path])
            finally:
                os.unlink(tmp_path)
            if entry.expected.startswith("error:"):
                expected_msg = entry.expected[6:].strip()
                if result["exit"] == 0:
                    results.append(
                        (
                            "fail",
                            test_id,
                            f"Expected error containing '{expected_msg}', got success",
                        )
                    )
                    continue
                stderr_line = (result["stderr"].strip().split("\n") or [""])[0]
                if expected_msg and expected_msg.lower() not in stderr_line.lower():
                    results.append(
                        (
                            "fail",
                            test_id,
                            f"Expected error containing '{expected_msg}', got: {stderr_line}",
                        )
                    )
                    continue
                results.append(("pass", test_id, None))
                continue
            if result["exit"] != 0:
                err_msg = (result["stderr"].strip().split("\n") or ["lowering failed"])[
                    0
                ]
                results.append(("fail", test_id, f"Lowering error: {err_msg}"))
                continue
            output = result["stdout"]
            if not contains_normalized(output, entry.expected):
                results.append(
                    (
                        "fail",
                        test_id,
                        f"Expected not found in output:\n--- expected ---\n{entry.expected}\n--- got ---\n{output}",
                    )
                )
                continue
            results.append(("pass", test_id, None))
    return results


def run_codegen_tests(test_dir):
    results = []
    base_dir = os.path.join(test_dir, "base")
    if not os.path.isdir(base_dir):
        return results
    lang_dirs = sorted(
        [
            d
            for d in os.listdir(test_dir)
            if d != "base"
            and os.path.isdir(os.path.join(test_dir, d))
            and d in EMITTER_LANGS
        ]
    )
    for lang in lang_dirs:
        lang_dir = os.path.join(test_dir, lang)
        for base_file in sorted(glob.glob(os.path.join(base_dir, "*.tests"))):
            basename = os.path.basename(base_file)
            stem = os.path.splitext(basename)[0]
            lang_file = os.path.join(lang_dir, basename)
            with open(base_file) as fh:
                base_tests = parse_simple_tests(fh.read())
            if not base_tests:
                continue
            if not os.path.exists(lang_file):
                for entry in base_tests:
                    results.append(
                        (
                            "fail",
                            f"{stem}/{entry.name}[{lang}]",
                            f"{lang}/{basename} missing",
                        )
                    )
                continue
            with open(lang_file) as fh:
                lang_tests = parse_simple_tests(fh.read())
            base_names = [e.name for e in base_tests]
            lang_names = [e.name for e in lang_tests]
            if base_names != lang_names:
                for entry in base_tests:
                    results.append(
                        (
                            "fail",
                            f"{stem}/{entry.name}[{lang}]",
                            "base/lang name mismatch",
                        )
                    )
                continue
            lang_by_name = {e.name: e.content for e in lang_tests}
            for entry in base_tests:
                test_id = f"{stem}/{entry.name}[{lang}]"
                expected = lang_by_name[entry.name]
                fd, tmp_path = tempfile.mkstemp(suffix=".ty")
                try:
                    with os.fdopen(fd, "w") as tf:
                        tf.write(entry.content)
                    result = run_inprocess(["taytsh", "--emit", lang, tmp_path])
                finally:
                    os.unlink(tmp_path)
                if result["exit"] != 0:
                    stderr = (
                        result["stderr"].strip().split("\n") or ["transpile failed"]
                    )[0]
                    results.append(("fail", test_id, f"Transpile error: {stderr}"))
                    continue
                output = result["stdout"]
                if not contains_normalized(output, expected):
                    results.append(
                        (
                            "fail",
                            test_id,
                            f"Expected not found in output:\n--- expected ---\n{expected}\n--- got ---\n{output}",
                        )
                    )
                    continue
                results.append(("pass", test_id, None))
    return results


def run_emit_tests(test_dir):
    results = []
    base_dir = os.path.join(test_dir, "base")
    if not os.path.isdir(base_dir):
        return results
    lang_dirs = sorted(
        [
            d
            for d in os.listdir(test_dir)
            if d != "base"
            and os.path.isdir(os.path.join(test_dir, d))
            and d in EMITTER_LANGS
        ]
    )
    for lang in lang_dirs:
        lang_dir = os.path.join(test_dir, lang)
        for base_file in sorted(glob.glob(os.path.join(base_dir, "*.tests"))):
            basename = os.path.basename(base_file)
            stem = os.path.splitext(basename)[0]
            lang_file = os.path.join(lang_dir, basename)
            with open(base_file) as fh:
                base_tests = parse_simple_tests(fh.read())
            if not base_tests:
                continue
            if not os.path.exists(lang_file):
                continue
            with open(lang_file) as fh:
                lang_tests = parse_simple_tests(fh.read())
            lang_by_name = {e.name: e.content for e in lang_tests}
            for entry in base_tests:
                if entry.name not in lang_by_name:
                    continue
                test_id = f"{stem}/{entry.name}[{lang}]"
                expected = lang_by_name[entry.name]
                fd, tmp_path = tempfile.mkstemp(suffix=".py")
                try:
                    with os.fdopen(fd, "w") as tf:
                        tf.write(entry.content)
                    result = run_inprocess(["--target", lang, tmp_path])
                finally:
                    os.unlink(tmp_path)
                if result["exit"] != 0:
                    stderr = (result["stderr"].strip().split("\n") or ["emit failed"])[
                        0
                    ]
                    results.append(("fail", test_id, f"Emit error: {stderr}"))
                    continue
                output = result["stdout"]
                if not contains_normalized(output, expected):
                    results.append(
                        (
                            "fail",
                            test_id,
                            f"Expected not found in output:\n--- expected ---\n{expected}\n--- got ---\n{output}",
                        )
                    )
                    continue
                results.append(("pass", test_id, None))
    return results


def _find_lib_imports(source):
    """Find lib imports from source text (simple, no AST)."""
    names = []
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith("from lib.") and "import" in stripped:
            # from lib.foo import ...
            mod = stripped.split("from lib.")[1].split(" ")[0].split(".")[0]
            if mod and mod not in names:
                names.append(mod)
        elif stripped.startswith("import lib."):
            mod = stripped.split("import lib.")[1].split(" ")[0].split(".")[0]
            if mod and mod not in names:
                names.append(mod)
    return names


def run_app_tests(test_dir):
    results = []
    available = [lang for lang, cmd in RUNTIMES.items() if _runtime_available(cmd)]
    for test_file in sorted(glob.glob(os.path.join(test_dir, "apptest_*.py"))):
        stem = os.path.splitext(os.path.basename(test_file))[0]
        with open(test_file) as fh:
            source = fh.read()
        lib_names = find_lib_imports(source)
        # Transitively resolve cross-lib imports
        seen = list(lib_names)
        queue = list(lib_names)
        while queue:
            name = queue.pop(0)
            lib_path = os.path.join(LIB_DIR, f"{name}.py")
            if not os.path.exists(lib_path):
                continue
            with open(lib_path) as fh:
                lib_source = fh.read()
            for dep in find_lib_imports(lib_source):
                if dep not in seen:
                    seen.append(dep)
                    queue.append(dep)
        lib_names = seen
        for target in available:
            test_id = f"{stem}[{target}]"
            if not lib_names:
                fd, tmp_path = tempfile.mkstemp(suffix=".py")
                try:
                    with os.fdopen(fd, "w") as tf:
                        tf.write(source)
                    result = run_inprocess(["--target", target, tmp_path])
                finally:
                    os.unlink(tmp_path)
            else:
                parts = []
                for name in lib_names:
                    lib_path = os.path.join(LIB_DIR, f"{name}.py")
                    with open(lib_path) as fh:
                        parts.append((f"lib/{name}.py", fh.read()))
                stdin_data = build_project_input("apptest.py", source, parts)
                result = run_inprocess(
                    ["--project", "--target", target], stdin_data=stdin_data
                )
            if result["exit"] != 0:
                stderr = (result["stderr"].strip().split("\n") or ["transpile failed"])[
                    0
                ]
                results.append(
                    ("fail", test_id, f"Transpile error ({target}): {stderr}")
                )
                continue
            transpiled_code = result["stdout"]
            runtime = RUNTIMES[target]
            try:
                proc = subprocess.run(
                    runtime,
                    input=transpiled_code,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if proc.returncode != 0:
                    output = proc.stdout + proc.stderr
                    results.append(
                        (
                            "fail",
                            test_id,
                            f"App test failed with exit {proc.returncode}\n{output}",
                        )
                    )
                    continue
            except subprocess.TimeoutExpired:
                results.append(("fail", test_id, "App test timed out"))
                continue
            results.append(("pass", test_id, None))
    return results


def run_ty_app_tests(test_dir):
    results = []
    for test_file in sorted(glob.glob(os.path.join(test_dir, "*.ty"))):
        stem = os.path.splitext(os.path.basename(test_file))[0]
        test_id = stem
        result = run_inprocess(["taytsh", test_file])
        if result["exit"] != 0:
            output = (result["stdout"] + result["stderr"]).strip()
            results.append(("fail", test_id, f"Exit code {result['exit']}:\n{output}"))
            continue
        results.append(("pass", test_id, None))
    return results


def run_ordering_tests(test_dir):
    results = []
    available = [lang for lang, cmd in RUNTIMES.items() if _runtime_available(cmd)]
    for test_file in sorted(glob.glob(os.path.join(test_dir, "*.ty"))):
        stem = os.path.splitext(os.path.basename(test_file))[0]
        for target in available:
            test_id = f"{stem}[{target}]"
            result = run_inprocess(["taytsh", "--emit", target, test_file])
            if result["exit"] != 0:
                stderr = (result["stderr"].strip().split("\n") or ["transpile failed"])[
                    0
                ]
                results.append(
                    ("fail", test_id, f"Transpile error ({target}): {stderr}")
                )
                continue
            transpiled_code = result["stdout"]
            runtime = RUNTIMES[target]
            try:
                proc = subprocess.run(
                    runtime,
                    input=transpiled_code,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if proc.returncode != 0:
                    output = proc.stdout + proc.stderr
                    results.append(
                        (
                            "fail",
                            test_id,
                            f"Ordering test failed with exit {proc.returncode}\n{output}",
                        )
                    )
                    continue
            except subprocess.TimeoutExpired:
                results.append(("fail", test_id, "Ordering test timed out"))
                continue
            results.append(("pass", test_id, None))
    return results


def _runtime_available(cmd):
    try:
        subprocess.run(["which", cmd[0]], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Parallel execution support
# ---------------------------------------------------------------------------


def _get_cpu_count():
    """Get available CPU count, preferring sched_getaffinity for container awareness."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _parse_num_workers(value):
    """Parse -n argument: 'auto' or integer."""
    if value == "auto":
        return _get_cpu_count()
    try:
        n = int(value)
        if n < 1:
            raise ValueError("Worker count must be positive")
        return n
    except ValueError:
        print(f"Invalid -n value: {value}", file=sys.stderr)
        sys.exit(1)


def _worker_init(transpiled_path, harness_path, via_vm_path):
    """Initialize worker process: load transpiled module and harness."""
    global _worker_mod, _worker_main_func, _worker_harness_mod
    global \
        _worker_use_vm, \
        _worker_vm_compiled, \
        _vm_mod, \
        _vm_compiled, \
        _main_func, \
        _use_vm
    # Load transpiled module
    spec = importlib.util.spec_from_file_location("tongues_transpiled", transpiled_path)
    _worker_mod = importlib.util.module_from_spec(spec)
    sys.modules["tongues_transpiled"] = _worker_mod
    spec.loader.exec_module(_worker_mod)
    _worker_main_func = getattr(_worker_mod, "main", None)
    # Load harness module
    harness_spec = importlib.util.spec_from_file_location(
        "test_harness_transpiled", harness_path
    )
    _worker_harness_mod = importlib.util.module_from_spec(harness_spec)
    sys.modules["test_harness_transpiled"] = _worker_harness_mod
    harness_spec.loader.exec_module(_worker_harness_mod)
    # Import harness functions into globals
    for name in dir(_worker_harness_mod):
        if not name.startswith("_"):
            globals()[name] = getattr(_worker_harness_mod, name)
    # Set up module-level references for run_inprocess
    _vm_mod = _worker_mod
    _main_func = _worker_main_func
    # Load VM module if requested
    if via_vm_path:
        with open(via_vm_path) as f:
            source = f.read()
        module = _worker_mod.taytsh_taytsh_parse(source)
        _worker_vm_compiled = _worker_mod.vm_prepare(module)
        _vm_compiled = _worker_vm_compiled
        _worker_use_vm = True
        _use_vm = True


def _run_single_test(args):
    """Run a single test case in a worker process. Returns (phase, test_id, status, error)."""
    global _main_func, _use_vm, _vm_mod, _vm_compiled
    # Restore worker state to module globals
    _main_func = _worker_main_func
    _use_vm = _worker_use_vm
    _vm_mod = _worker_mod
    _vm_compiled = _worker_vm_compiled
    # Get harness functions from the loaded module
    h = _worker_harness_mod
    phase_name, test_id, test_type, test_data = args
    try:
        if test_type == "cli":
            spec = test_data
            if spec.stdin_hex:
                raw = bytes.fromhex(spec.stdin_hex)
                result = run_inprocess(
                    spec.args, stdin_data=raw.decode("latin-1"), stdin_bytes=raw
                )
            else:
                result = run_inprocess(spec.args, stdin_data=spec.stdin)
            err = h.check_cli_assertions(
                result["exit"], result["stdout"], result["stderr"], spec.assertions
            )
            if err:
                return (phase_name, test_id, "fail", err)
            return (phase_name, test_id, "pass", None)
        elif test_type == "phase":
            entry, cfg = test_data
            phase_result = run_transpiled_phase(
                entry.input_,
                cfg["args"],
                is_taytsh=cfg["taytsh"],
                expect_json=cfg["json"],
            )
            reveals = phase_result["reveals"]
            if (
                phase_name in ("pycheck", "tycheck")
                and not phase_result["errors"]
                and phase_result["data"]
            ):
                if isinstance(phase_result["data"], h.JsonObject):
                    try:
                        reveals_arr = h.json_get_items(
                            h.json_get_field(phase_result["data"], "reveals")
                        )
                        reveals = [
                            (
                                int(h.json_get_number(h.json_get_field(r, "line"))),
                                h.json_get_string(h.json_get_field(r, "type")),
                            )
                            for r in reveals_arr
                        ]
                    except Exception:
                        pass
            lenient = phase_name in ("parse", "pycheck", "typarse", "tycheck")
            err = h.check_expected(
                entry.expected,
                phase_result["errors"],
                phase_result["warnings"],
                phase_result["data"],
                reveals,
                phase_name,
                lenient,
            )
            if err:
                return (phase_name, test_id, "fail", err)
            return (phase_name, test_id, "pass", None)
        elif test_type == "ty_app":
            test_file = test_data
            result = run_inprocess(["taytsh", test_file])
            if result["exit"] != 0:
                output = (result["stdout"] + result["stderr"]).strip()
                return (
                    phase_name,
                    test_id,
                    "fail",
                    f"Exit code {result['exit']}:\n{output}",
                )
            return (phase_name, test_id, "pass", None)
        elif test_type == "skip":
            return (phase_name, test_id, "skip", None)
        else:
            return (phase_name, test_id, "fail", f"Unknown test type: {test_type}")
    except Exception as e:
        import traceback

        return (
            phase_name,
            test_id,
            "fail",
            f"Exception: {e}\n{traceback.format_exc()}",
        )


def _collect_tests(tests_config):
    """Collect all tests without running them. Returns list of (phase, test_id, type, data)."""
    collected = []
    for section_name, phases in tests_config:
        for phase_name, cfg in phases:
            test_dir = os.path.join(TESTS_DIR, cfg["dir"])
            if not os.path.isdir(test_dir):
                continue
            runner_name = cfg["run"]
            if runner_name == "cli":
                for f in sorted(glob.glob(os.path.join(test_dir, "*.tests"))):
                    stem = os.path.splitext(os.path.basename(f))[0]
                    with open(f) as fh:
                        content = fh.read()
                    for name, spec in parse_cli_test_file(content):
                        test_id = f"{stem}/{name}"
                        if cli_needs_backend(spec.args, spec.assertions, EMITTER_LANGS):
                            collected.append((phase_name, test_id, "skip", None))
                        else:
                            collected.append((phase_name, test_id, "cli", spec))
            elif runner_name == "phase":
                for f in sorted(glob.glob(os.path.join(test_dir, "*.tests"))):
                    stem = os.path.splitext(os.path.basename(f))[0]
                    with open(f) as fh:
                        content = fh.read()
                    for entry in parse_spec_file(content):
                        test_id = f"{stem}/{entry.name}"
                        collected.append((phase_name, test_id, "phase", (entry, cfg)))
            elif runner_name == "ty_app":
                for test_file in sorted(glob.glob(os.path.join(test_dir, "*.ty"))):
                    stem = os.path.splitext(os.path.basename(test_file))[0]
                    collected.append((phase_name, stem, "ty_app", test_file))
            # Note: linker, lowering, codegen, emit, app, ordering tests are complex
            # and not parallelized - they run sequentially in the main process
    return collected


def _run_tests_parallel(
    collected, num_workers, transpiled_path, harness_path, via_vm_path
):
    """Run collected tests in parallel using process pool."""
    results = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(transpiled_path, harness_path, via_vm_path),
    ) as executor:
        futures = {executor.submit(_run_single_test, t): t for t in collected}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                test = futures[future]
                results.append((test[0], test[1], "fail", f"Worker error: {e}"))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python test-transpiled.py <transpiled.py> [options]",
            file=sys.stderr,
        )
        print("Options:", file=sys.stderr)
        print("  --via-vm <tongues.ty>  Run tests through the VM", file=sys.stderr)
        print("  --target <name>        Set target name for reporting", file=sys.stderr)
        print(
            "  -n <num|auto>          Number of parallel workers (default: 1)",
            file=sys.stderr,
        )
        sys.exit(1)

    via_vm_path = None
    target_name = None
    num_workers = 1
    filtered_argv = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--via-vm":
            if i + 1 >= len(sys.argv):
                print("--via-vm requires a path to a .ty file", file=sys.stderr)
                sys.exit(1)
            raw = sys.argv[i + 1]
            via_vm_path = raw if os.path.isabs(raw) else os.path.join(TONGUES_DIR, raw)
            i += 2
        elif sys.argv[i] == "--target":
            if i + 1 >= len(sys.argv):
                print("--target requires a name", file=sys.stderr)
                sys.exit(1)
            target_name = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "-n":
            if i + 1 >= len(sys.argv):
                print("-n requires a number or 'auto'", file=sys.stderr)
                sys.exit(1)
            num_workers = _parse_num_workers(sys.argv[i + 1])
            i += 2
        else:
            filtered_argv.append(sys.argv[i])
            i += 1

    transpiled_path = (
        os.path.join(TONGUES_DIR, filtered_argv[0])
        if not os.path.isabs(filtered_argv[0])
        else filtered_argv[0]
    )
    if not os.path.exists(transpiled_path):
        print(f"Transpiled file not found: {transpiled_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading transpiled binary: {transpiled_path}")
    t0 = time.monotonic()
    spec = importlib.util.spec_from_file_location("tongues_transpiled", transpiled_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tongues_transpiled"] = mod
    try:
        spec.loader.exec_module(mod)
    except SyntaxError as e:
        print("Failed to load transpiled binary: syntax error", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(1)
    t1 = time.monotonic()
    print(f"Loaded in {t1 - t0:.1f}s")

    _main_func = getattr(mod, "main", None)
    if _main_func is None:
        print("Transpiled binary has no main() function", file=sys.stderr)
        sys.exit(1)

    harness_path = os.path.join(TONGUES_DIR, ".out", "test_harness.py")
    if not os.path.exists(harness_path):
        print(f"Transpiled harness not found: {harness_path}", file=sys.stderr)
        sys.exit(1)
    harness_spec = importlib.util.spec_from_file_location(
        "test_harness_transpiled", harness_path
    )
    harness_mod = importlib.util.module_from_spec(harness_spec)
    sys.modules["test_harness_transpiled"] = harness_mod
    harness_spec.loader.exec_module(harness_mod)

    # Import harness functions into this module's global scope
    for name in dir(harness_mod):
        if not name.startswith("_"):
            globals()[name] = getattr(harness_mod, name)

    _vm_mod = mod

    if via_vm_path is not None:
        if not os.path.exists(via_vm_path):
            print(f"VM module not found: {via_vm_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading VM module: {via_vm_path}")
        vm_t0 = time.monotonic()
        _load_vm_module(via_vm_path)
        print(f"VM compiled in {time.monotonic() - vm_t0:.1f}s")
        _use_vm = True

    print()
    if num_workers > 1:
        print(f"Running with {num_workers} workers")

    total_pass = 0
    total_fail = 0
    total_skip = 0
    failures = []

    # Runners that must run sequentially (complex setup or external processes)
    SEQUENTIAL_RUNNERS = {
        "linker": run_linker_tests,
        "lowering": run_lowering_tests,
        "codegen": run_codegen_tests,
        "emit": run_emit_tests,
        "app": run_app_tests,
        "ordering": run_ordering_tests,
    }

    # Parallelizable test types
    PARALLEL_RUNNERS = {"cli", "phase", "ty_app"}

    if num_workers > 1:
        # Collect parallelizable tests
        parallel_tests = _collect_tests(TESTS)
        if parallel_tests:
            print(f"Collected {len(parallel_tests)} parallelizable tests")
            t_start = time.monotonic()
            parallel_results = _run_tests_parallel(
                parallel_tests, num_workers, transpiled_path, harness_path, via_vm_path
            )
            t_elapsed = time.monotonic() - t_start
            print(f"Parallel execution completed in {t_elapsed:.1f}s")
            # Group results by phase for reporting
            phase_results_map = {}
            for phase, tid, status, err in parallel_results:
                if phase not in phase_results_map:
                    phase_results_map[phase] = []
                phase_results_map[phase].append((status, tid, err))
            # Report parallel results
            for phase in dict.fromkeys(t[0] for t in parallel_tests):
                if phase in phase_results_map:
                    phase_results = phase_results_map[phase]
                    print(f"::group::{phase}")
                    pass_count = sum(1 for s, _, _ in phase_results if s == "pass")
                    fail_count = sum(1 for s, _, _ in phase_results if s == "fail")
                    skip_count = sum(1 for s, _, _ in phase_results if s == "skip")
                    total_pass += pass_count
                    total_fail += fail_count
                    total_skip += skip_count
                    status = "FAIL" if fail_count > 0 else "ok"
                    counts = f"{pass_count} passed"
                    if fail_count > 0:
                        counts += f", {fail_count} failed"
                    if skip_count > 0:
                        counts += f", {skip_count} skipped"
                    print(f"{phase}: {status} ({counts})")
                    for s, tid, err in phase_results:
                        if s == "fail":
                            failures.append((phase, tid, err))
                            print(f"  FAIL {tid}")
                    print("::endgroup::")
        # Run sequential tests
        for section_name, phases in TESTS:
            for phase_name, cfg in phases:
                runner_name = cfg["run"]
                if runner_name not in SEQUENTIAL_RUNNERS:
                    continue
                test_dir = os.path.join(TESTS_DIR, cfg["dir"])
                if not os.path.isdir(test_dir):
                    continue
                print(f"::group::{phase_name}")
                phase_results = SEQUENTIAL_RUNNERS[runner_name](test_dir)
                pass_count = sum(1 for s, _, _ in phase_results if s == "pass")
                fail_count = sum(1 for s, _, _ in phase_results if s == "fail")
                skip_count = sum(1 for s, _, _ in phase_results if s == "skip")
                total_pass += pass_count
                total_fail += fail_count
                total_skip += skip_count
                status = "FAIL" if fail_count > 0 else "ok"
                counts = f"{pass_count} passed"
                if fail_count > 0:
                    counts += f", {fail_count} failed"
                if skip_count > 0:
                    counts += f", {skip_count} skipped"
                print(f"{phase_name}: {status} ({counts})")
                for s, tid, err in phase_results:
                    if s == "fail":
                        failures.append((phase_name, tid, err))
                        print(f"  FAIL {tid}")
                print("::endgroup::")
    else:
        # Sequential execution (original behavior)
        RUNNERS = {
            "cli": run_cli_tests,
            "linker": run_linker_tests,
            "lowering": run_lowering_tests,
            "codegen": run_codegen_tests,
            "emit": run_emit_tests,
            "app": run_app_tests,
            "ty_app": run_ty_app_tests,
            "ordering": run_ordering_tests,
        }
        for section_name, phases in TESTS:
            for phase_name, cfg in phases:
                test_dir = os.path.join(TESTS_DIR, cfg["dir"])
                if not os.path.isdir(test_dir):
                    continue
                print(f"::group::{phase_name}")
                runner_name = cfg["run"]
                if runner_name == "phase":
                    phase_results = run_phase_tests(test_dir, phase_name, cfg)
                elif runner_name in RUNNERS:
                    phase_results = RUNNERS[runner_name](test_dir)
                else:
                    phase_results = []
                pass_count = sum(1 for s, _, _ in phase_results if s == "pass")
                fail_count = sum(1 for s, _, _ in phase_results if s == "fail")
                skip_count = sum(1 for s, _, _ in phase_results if s == "skip")
                total_pass += pass_count
                total_fail += fail_count
                total_skip += skip_count
                status = "FAIL" if fail_count > 0 else "ok"
                counts = f"{pass_count} passed"
                if fail_count > 0:
                    counts += f", {fail_count} failed"
                if skip_count > 0:
                    counts += f", {skip_count} skipped"
                print(f"{phase_name}: {status} ({counts})")
                for s, tid, err in phase_results:
                    if s == "fail":
                        failures.append((phase_name, tid, err))
                        print(f"  FAIL {tid}")
                print("::endgroup::")

    print()
    if failures:
        print("=" * 60)
        print(f"FAILURES [{target_name}]" if target_name else "FAILURES")
        print("=" * 60)
        for phase, tid, err in failures:
            print()
            print(f"{phase} :: {tid}")
            print(err)
            # GitHub Actions error annotation
            title = f"{phase} :: {tid}"
            print(
                f"::error title={title}::{err.splitlines()[0] if err else 'Test failed'}"
            )
        print()

    print("=" * 60)
    total = total_pass + total_fail + total_skip
    prefix = f"[{target_name}] " if target_name else ""
    summary_line = f"{prefix}{total} tests: {total_pass} passed, {total_fail} failed, {total_skip} skipped"
    print(summary_line)
    print("=" * 60)

    # GitHub Actions notice annotation
    if total_fail == 0:
        print(f"::notice::{summary_line}")

    # GitHub Actions job summary
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            status_emoji = "✅" if total_fail == 0 else "❌"
            f.write(f"## {status_emoji} {target_name or 'Test Results'}\n\n")
            f.write(f"| Passed | Failed | Skipped | Total |\n")
            f.write(f"|--------|--------|---------|-------|\n")
            f.write(f"| {total_pass} | {total_fail} | {total_skip} | {total} |\n\n")
            if failures:
                f.write("### Failures\n\n")
                for phase, tid, err in failures:
                    f.write(
                        f"<details><summary><code>{phase} :: {tid}</code></summary>\n\n"
                    )
                    f.write(f"```\n{err}\n```\n\n</details>\n\n")

    sys.exit(1 if total_fail > 0 else 0)
