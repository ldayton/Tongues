"""Codegen test phase: codegen snapshots + emit snapshots."""

import pytest

from tests.harness import (
    EMITTERS,
    TESTS_DIR,
    contains_normalized,
    discover_codegen_tests,
    emit_from_python,
    parse_simple_tests,
    transpile_code,
)

# fmt: off
TESTS = {
    "codegen": {"dir": "21_codegen", "run": "codegen"},
    "emit":    {"dir": "25_emit",    "run": "emit"},
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


def pytest_generate_tests(metafunc):
    for name, cfg in TESTS.items():
        test_dir = TESTS_DIR / cfg["dir"]
        run = cfg["run"]
        if run == "codegen" and "codegen_input" in metafunc.fixturenames:
            dirs = {
                d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "base"
            }
            langs = sorted(dirs & set(EMITTERS))
            all_tests = []
            for lang in langs:
                for tid, inp, exp in discover_codegen_tests(test_dir, lang):
                    all_tests.append(pytest.param(inp, exp, lang, id=tid))
            for lang in sorted(set(EMITTERS) - dirs):
                base_dir = test_dir / "base"
                for base_file in sorted(base_dir.glob("*.tests")):
                    for name, _ in parse_simple_tests(base_file):
                        tid = f"{base_file.stem}/{name}[{lang}]"
                        all_tests.append(
                            pytest.param(
                                "",
                                "",
                                lang,
                                id=tid,
                            )
                        )
            metafunc.parametrize(
                "codegen_input,codegen_expected,codegen_lang", all_tests
            )
        elif run == "emit" and "emit_input" in metafunc.fixturenames:
            dirs = {
                d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "base"
            }
            langs = sorted(dirs & set(EMITTERS))
            all_tests = []
            for lang in langs:
                for tid, inp, exp in discover_codegen_tests(test_dir, lang):
                    all_tests.append(pytest.param(inp, exp, lang, id=tid))
            metafunc.parametrize("emit_input,emit_expected,emit_lang", all_tests)


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
