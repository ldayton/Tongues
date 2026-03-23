"""Codegen test phase: codegen snapshots + emit snapshots."""

import pytest

from tests.harness import (
    EMITTERS,
    TESTS_DIR,
    contains_normalized,
    discover_codegen_tests,
    emit_from_python,
    transpile_code,
)

# fmt: off
TESTS = {
    "codegen": {"dir": "backend/codegen", "run": "codegen"},
    "emit":    {"dir": "backend/emit",    "run": "emit"},
}
# fmt: on


@pytest.fixture
def transpiled_output(codegen_input: str, codegen_lang: str) -> str:
    output, err = transpile_code(codegen_input, codegen_lang)
    if err is not None:
        pytest.fail(f"Transpile error: {err}")
    if output is None:
        pytest.fail("No output from transpiler")
    return output


@pytest.fixture
def emit_output(emit_input: str, emit_lang: str) -> str:
    output, err = emit_from_python(emit_input, emit_lang)
    if err is not None:
        pytest.fail(f"Emit error: {err}")
    if output is None:
        pytest.fail("No output from emitter")
    return output


def _collect_snapshot_tests(metafunc, test_dir, label, param_names):
    """Collect snapshot tests, requiring every EMITTER to have a test directory."""
    dirs = {d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "base"}
    missing = set(EMITTERS) - dirs
    if missing:
        pytest.fail(
            f"Missing {label} test dirs for: {', '.join(sorted(missing))}. "
            f"Expected dirs in {test_dir} for all EMITTERS."
        )
    langs = sorted(dirs & set(EMITTERS))
    counts = {}
    all_tests = []
    for lang in langs:
        lang_tests = list(discover_codegen_tests(test_dir, lang))
        counts[lang] = len(lang_tests)
        for tid, inp, exp in lang_tests:
            all_tests.append(pytest.param(inp, exp, lang, id=tid))
    expected_count = counts[langs[0]]
    unequal = {l: c for l, c in counts.items() if c != expected_count}
    if unequal:
        pytest.fail(
            f"Mismatched {label} test counts: "
            + ", ".join(f"{l}={c}" for l, c in sorted(counts.items()))
        )
    metafunc.parametrize(param_names, all_tests)


def pytest_generate_tests(metafunc):
    for name, cfg in TESTS.items():
        test_dir = TESTS_DIR / cfg["dir"]
        run = cfg["run"]
        if run == "codegen" and "codegen_input" in metafunc.fixturenames:
            _collect_snapshot_tests(
                metafunc,
                test_dir,
                "codegen",
                "codegen_input,codegen_expected,codegen_lang",
            )
        elif run == "emit" and "emit_input" in metafunc.fixturenames:
            _collect_snapshot_tests(
                metafunc,
                test_dir,
                "emit",
                "emit_input,emit_expected,emit_lang",
            )


def test_codegen(
    codegen_input: str,
    codegen_expected: str,
    codegen_lang: str,
    transpiled_output: str,
):
    if not contains_normalized(transpiled_output, codegen_expected):
        pytest.fail(
            "Expected not found in output:\n"
            f"--- expected ---\n{codegen_expected}\n"
            f"--- got ---\n{transpiled_output}"
        )


def test_emit(
    emit_input: str,
    emit_expected: str,
    emit_lang: str,
    emit_output: str,
):
    if not contains_normalized(emit_output, emit_expected):
        pytest.fail(
            "Expected not found in output:\n"
            f"--- expected ---\n{emit_expected}\n"
            f"--- got ---\n{emit_output}"
        )
