"""Pytest configuration for Tongues test suite."""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tongues"))

from src.frontend import Frontend
from src.frontend.parse import parse
from src.frontend.subset import verify as verify_subset
from src.frontend.names import resolve_names
from src.middleend import analyze
from src.backend.go import GoBackend
from src.backend.java import JavaBackend
from src.backend.javascript import JsBackend
from src.backend.lua import LuaBackend
from src.backend.perl import PerlBackend
from src.backend.python import PythonBackend
from src.backend.ruby import RubyBackend
from src.backend.typescript import TsBackend
from src.backend.csharp import CSharpBackend
from src.backend.php import PhpBackend

# Skip specific (apptest, language) combinations that are known to fail.
# Format: apptest_stem -> set of languages to skip
_SKIP_LANGS: dict[str, set[str]] = {
    "apptest_bools": {
        "dart",
        "go",
        "lua",
        "perl",
        "php",
        "ruby",
        "swift",
    },
    "apptest_bytes": {
        "c",
        "dart",
        "go",
        "java",
        "lua",
        "perl",
        "php",
        "ruby",
        "rust",
        "swift",
        "typescript",
        "zig",
    },
    "apptest_chars": {
        "c",
        "dart",
        "go",
        "java",
        "javascript",
        "lua",
        "perl",
        "php",
        "ruby",
        "swift",
        "typescript",
        "zig",
    },
    "apptest_dicts": {
        "c",
        "dart",
        "go",
        "java",
        "javascript",
        "lua",
        "perl",
        "php",
        "ruby",
        "rust",
        "swift",
        "typescript",
        "zig",
    },
    "apptest_floats": {
        "c",
        "dart",
        "go",
        "java",
        "javascript",
        "lua",
        "perl",
        "php",
        "ruby",
        "rust",
        "swift",
        "typescript",
        "zig",
    },
    "apptest_ints": {
        "c",
        "dart",
        "go",
        "java",
        "javascript",
        "lua",
        "perl",
        "php",
        "rust",
        "swift",
        "typescript",
        "zig",
    },
    "apptest_lists": {
        "c",
        "dart",
        "go",
        "java",
        "javascript",
        "lua",
        "perl",
        "php",
        "ruby",
        "rust",
        "swift",
        "typescript",
        "zig",
    },
    "apptest_none": {
        "c",
        "dart",
        "go",
        "java",
        "lua",
        "perl",
        "php",
        "ruby",
        "rust",
        "swift",
        "typescript",
        "zig",
    },
    "apptest_sets": {
        "c",
        "dart",
        "go",
        "java",
        "javascript",
        "lua",
        "perl",
        "php",
        "ruby",
        "rust",
        "swift",
        "typescript",
        "zig",
    },
    "apptest_strings": {
        "c",
        "dart",
        "go",
        "java",
        "javascript",
        "lua",
        "perl",
        "php",
        "ruby",
        "rust",
        "swift",
        "typescript",
        "zig",
    },
    "apptest_truthiness": {
        "c",
        "dart",
        "go",
        "java",
        "javascript",
        "lua",
        "perl",
        "php",
        "ruby",
        "rust",
        "swift",
        "typescript",
        "zig",
    },
    "apptest_tuples": {
        "c",
        "dart",
        "go",
        "java",
        "javascript",
        "lua",
        "perl",
        "php",
        "ruby",
        "rust",
        "swift",
        "typescript",
        "zig",
    },
}

TESTS_DIR = Path(__file__).parent
APP_DIR = TESTS_DIR / "app"
CODEGEN_DIR = TESTS_DIR / "codegen"
OUT_DIR = APP_DIR / ".out"
TONGUES_DIR = TESTS_DIR.parent / "tongues"

BACKENDS = {
    "csharp": CSharpBackend,
    "go": GoBackend,
    "java": JavaBackend,
    "javascript": JsBackend,
    "lua": LuaBackend,
    "perl": PerlBackend,
    "php": PhpBackend,
    "python": PythonBackend,
    "ruby": RubyBackend,
    "typescript": TsBackend,
}


class TranspileError(Exception):
    """Raised when transpilation fails."""


class CompileError(Exception):
    """Raised when compilation fails."""


@dataclass
class Target:
    """Configuration for a language target."""

    name: str
    ext: str
    run_cmd: list[str] | None = None  # None means needs compilation first
    compile_cmd: list[str] | None = None
    format_cmd: list[str] | None = None

    def get_run_command(self, path: Path) -> list[str]:
        """Return the command to run the output file."""
        if self.run_cmd:
            return [arg.format(path=path) for arg in self.run_cmd]
        # For compiled languages, return the executable path
        return [str(path.with_suffix(""))]

    def get_compile_command(self, path: Path) -> list[str] | None:
        """Return the command to compile the output file."""
        if not self.compile_cmd:
            return None
        return [
            arg.format(path=path, out=path.with_suffix("")) for arg in self.compile_cmd
        ]

    def get_format_command(self, path: Path) -> list[str] | None:
        """Return the command to format the output file."""
        if not self.format_cmd:
            return None
        return [arg.format(path=path) for arg in self.format_cmd]


TARGETS: dict[str, Target] = {
    "python": Target(
        name="python",
        ext=".py",
        run_cmd=["uv", "run", "--no-project", "--python", "~=3.12", "python", "{path}"],
        format_cmd=["uvx", "ruff", "format", "--quiet", "{path}"],
    ),
    "javascript": Target(
        name="javascript",
        ext=".js",
        run_cmd=["node", "{path}"],
        format_cmd=["npx", "@biomejs/biome", "format", "--write", "{path}"],
    ),
    "typescript": Target(
        name="typescript",
        ext=".ts",
        run_cmd=["npx", "tsx", "{path}"],
        format_cmd=["npx", "@biomejs/biome", "format", "--write", "{path}"],
    ),
    "ruby": Target(
        name="ruby",
        ext=".rb",
        run_cmd=["ruby", "{path}"],
        format_cmd=[
            "rubocop",
            "-A",
            "--fail-level",
            "fatal",
            "-o",
            "/dev/null",
            "{path}",
        ],
    ),
    "java": Target(
        name="java",
        ext=".java",
        run_cmd=["java", "{path}"],
        format_cmd=[
            "java",
            "-jar",
            "/usr/local/lib/google-java-format.jar",
            "-i",
            "{path}",
        ],
    ),
    "dart": Target(
        name="dart",
        ext=".dart",
        run_cmd=["dart", "run", "{path}"],
        format_cmd=["dart", "format", "{path}"],
    ),
    "go": Target(
        name="go",
        ext=".go",
        run_cmd=["go", "run", "{path}"],
        format_cmd=["gofmt", "-w", "{path}"],
    ),
    "lua": Target(
        name="lua",
        ext=".lua",
        run_cmd=["lua", "{path}"],
        format_cmd=["stylua", "{path}"],
    ),
    "php": Target(
        name="php",
        ext=".php",
        run_cmd=["php", "{path}"],
        format_cmd=["php-cs-fixer", "fix", "--quiet", "{path}"],
    ),
    "c": Target(
        name="c",
        ext=".c",
        compile_cmd=["gcc", "-std=c11", "-o", "{out}", "{path}", "-lm"],
        format_cmd=["clang-format", "-i", "{path}"],
    ),
    "perl": Target(
        name="perl",
        ext=".pl",
        run_cmd=["perl", "{path}"],
        format_cmd=["perltidy", "-b", "-bext=/", "{path}"],
    ),
    "rust": Target(
        name="rust",
        ext=".rs",
        compile_cmd=["rustc", "-o", "{out}", "{path}"],
    ),
    "swift": Target(
        name="swift",
        ext=".swift",
        run_cmd=["swift", "{path}"],
    ),
    "zig": Target(
        name="zig",
        ext=".zig",
        compile_cmd=[
            "zig",
            "build-exe",
            "{path}",
            "-fno-emit-bin",
            "-fno-emit-asm",
            "--cache-dir",
            "/tmp/zig-cache",
            "-femit-bin={out}",
        ],
    ),
}

PYTHON_TARGET_VERSION = (3, 12)


def _get_clean_env() -> dict[str, str]:
    """Get environment without venv contamination for running target Python."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("VIRTUAL_ENV")}
    venv_bin = os.environ.get("VIRTUAL_ENV", "")
    if venv_bin:
        env["PATH"] = os.pathsep.join(
            p
            for p in os.environ.get("PATH", "").split(os.pathsep)
            if not p.startswith(venv_bin)
        )
    return env


def _verify_python_target_version() -> None:
    """Verify Python target uses the expected version."""
    target = TARGETS["python"]
    cmd = target.run_cmd[:6] + ["-c", "import sys; print(sys.version_info[:2])"]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=_get_clean_env(), cwd="/tmp"
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to check Python target version: {result.stderr}")
    actual = eval(result.stdout.strip())
    if actual[0] != PYTHON_TARGET_VERSION[0] or actual[1] != PYTHON_TARGET_VERSION[1]:
        raise RuntimeError(
            f"Python target version mismatch: expected {PYTHON_TARGET_VERSION}, got {actual}"
        )


def discover_apptests() -> list[Path]:
    """Find all apptest_*.py files."""
    return sorted(APP_DIR.glob("apptest_*.py"))


def parse_codegen_file(path: Path) -> list[tuple[str, str, dict[str, str]]]:
    """Parse .tests file into (name, input, {lang: expected}) tuples."""
    lines = path.read_text().split("\n")
    result: list[tuple[str, str, dict[str, str]]] = []
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
            expected_by_lang: dict[str, str] = {}
            while i < len(lines) and lines[i].startswith("--- "):
                lang = lines[i][4:].strip()
                i += 1
                expected_lines: list[str] = []
                while i < len(lines) and not lines[i].startswith("---"):
                    expected_lines.append(lines[i])
                    i += 1
                expected_by_lang[lang] = "\n".join(expected_lines)
            if i < len(lines) and lines[i] == "---":
                i += 1
            test_input = "\n".join(input_lines)
            result.append((test_name, test_input, expected_by_lang))
        else:
            i += 1
    return result


def discover_codegen_tests() -> list[tuple[str, str, str, str, bool]]:
    """Find all codegen tests, returns (test_id, input, lang, expected, has_explicit_expected)."""
    results = []
    for test_file in sorted(CODEGEN_DIR.glob("*.tests")):
        tests = parse_codegen_file(test_file)
        for name, input_code, expected_by_lang in tests:
            langs = list(expected_by_lang.keys())
            if "python" not in langs:
                langs.append("python")
            for lang in langs:
                has_explicit = lang in expected_by_lang
                expected = expected_by_lang.get(lang, input_code)
                test_id = f"{test_file.stem}/{name}[{lang}]"
                results.append((test_id, input_code, lang, expected, has_explicit))
    return results


def pytest_configure(config):
    """Verify Python target version at session start."""
    _verify_python_target_version()


def pytest_addoption(parser):
    """Add --target option for filtering targets."""
    parser.addoption(
        "--target",
        action="append",
        default=[],
        help="Run only specified targets (can be used multiple times)",
    )


def pytest_generate_tests(metafunc):
    """Parametrize tests over test combinations."""
    if "apptest" in metafunc.fixturenames and "target" in metafunc.fixturenames:
        apptests = discover_apptests()
        target_filter = metafunc.config.getoption("target")
        targets = (
            [TARGETS[t] for t in target_filter if t in TARGETS]
            if target_filter
            else list(TARGETS.values())
        )
        params = []
        for apptest in apptests:
            for target in targets:
                test_id = f"{target.name}/{apptest.stem}"
                if target.name in _SKIP_LANGS.get(apptest.stem, set()):
                    params.append(
                        pytest.param(
                            apptest,
                            target,
                            id=test_id,
                            marks=pytest.mark.skip(reason="known failure"),
                        )
                    )
                else:
                    params.append(pytest.param(apptest, target, id=test_id))
        metafunc.parametrize("apptest,target", params)

    if "codegen_input" in metafunc.fixturenames:
        target_filter = metafunc.config.getoption("target")
        tests = discover_codegen_tests()
        if target_filter:
            tests = [
                (tid, inp, lang, exp, has_exp)
                for tid, inp, lang, exp, has_exp in tests
                if lang in target_filter
            ]
        params = [
            pytest.param(input_code, lang, expected, has_explicit, id=test_id)
            for test_id, input_code, lang, expected, has_explicit in tests
        ]
        metafunc.parametrize(
            "codegen_input,codegen_lang,codegen_expected,codegen_has_explicit", params
        )


@pytest.fixture
def output_path(apptest: Path, target: Target, tmp_path: Path) -> Path:
    """Compute output path for transpiled file."""
    target_dir = OUT_DIR / target.name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{apptest.stem}{target.ext}"


@pytest.fixture
def transpiled(apptest: Path, target: Target, output_path: Path) -> Path:
    """Transpile apptest to target language."""
    source = apptest.read_text()
    result = subprocess.run(
        [sys.executable, "-m", "src.tongues", "--target", target.name],
        input=source,
        capture_output=True,
        text=True,
        cwd=TONGUES_DIR,
    )
    if result.returncode != 0:
        raise TranspileError(result.stderr.strip())
    output_path.write_text(result.stdout)
    return output_path


@pytest.fixture
def formatted(transpiled: Path, target: Target) -> Path:
    """Apply language formatter (optional, no-fail)."""
    fmt_cmd = target.get_format_command(transpiled)
    if fmt_cmd:
        try:
            subprocess.run(fmt_cmd, capture_output=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # Formatting is optional
    return transpiled


@pytest.fixture
def compiled(formatted: Path, target: Target) -> Path:
    """Compile for compiled languages (C/Rust/Zig)."""
    compile_cmd = target.get_compile_command(formatted)
    if compile_cmd:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise CompileError(result.stderr.strip())
    return formatted


@pytest.fixture
def executable(compiled: Path, target: Target) -> list[str]:
    """Return the command to execute the test."""
    return target.get_run_command(compiled)


@pytest.fixture
def run_env(target: Target) -> dict[str, str] | None:
    """Return clean environment for Python target, None for others."""
    if target.name == "python":
        return _get_clean_env()
    return None


def transpile_code(source: str, target: str) -> tuple[str | None, str | None]:
    """Transpile source to target language. Returns (output, error)."""
    ast_dict = parse(source)
    result = verify_subset(ast_dict)
    errors = result.errors()
    if errors:
        return (None, errors[0].message)
    name_result = resolve_names(ast_dict)
    errors = name_result.errors()
    if errors:
        return (None, errors[0].message)
    fe = Frontend()
    module = fe.transpile(source, ast_dict, name_result=name_result)
    analyze(module)
    backend_cls = BACKENDS.get(target)
    if backend_cls is None:
        return (None, f"unknown target: {target}")
    be = backend_cls()
    code = be.emit(module)
    return (code, None)


@pytest.fixture
def transpiled_output(codegen_input: str, codegen_lang: str) -> str:
    """Transpile codegen input to target language."""
    output, err = transpile_code(codegen_input, codegen_lang)
    if err is not None:
        pytest.fail(f"Transpile error: {err}")
    if output is None:
        pytest.fail("No output from transpiler")
    return output
