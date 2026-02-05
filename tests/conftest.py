"""Pytest configuration for Tongues apptest suite."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent / "app"
OUT_DIR = TESTS_DIR / ".out"
TONGUES_DIR = Path(__file__).parent.parent / "tongues"


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
        run_cmd=[sys.executable, "{path}"],
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


def discover_apptests() -> list[Path]:
    """Find all apptest_*.py files."""
    return sorted(TESTS_DIR.glob("apptest_*.py"))


def pytest_addoption(parser):
    """Add --target option for filtering targets."""
    parser.addoption(
        "--target",
        action="append",
        default=[],
        help="Run only specified targets (can be used multiple times)",
    )


def pytest_generate_tests(metafunc):
    """Parametrize tests over (apptest, target) combinations."""
    if "apptest" in metafunc.fixturenames and "target" in metafunc.fixturenames:
        apptests = discover_apptests()
        target_filter = metafunc.config.getoption("target")
        targets = (
            [TARGETS[t] for t in target_filter if t in TARGETS]
            if target_filter
            else list(TARGETS.values())
        )
        params = [
            pytest.param(apptest, target, id=f"{target.name}/{apptest.stem}")
            for apptest in apptests
            for target in targets
        ]
        metafunc.parametrize("apptest,target", params)


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
