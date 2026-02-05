"""Pytest-based codegen tests for Tongues transpiler."""

import pytest


def test_codegen(
    codegen_input: str,
    codegen_lang: str,
    codegen_expected: str,
    codegen_has_explicit: bool,
    transpiled_output: str,
):
    """Verify transpiler output matches expected code."""
    if not contains_normalized(transpiled_output, codegen_expected):
        pytest.fail(
            f"Expected not found in output:\n--- expected ---\n{codegen_expected}\n--- got ---\n{transpiled_output}"
        )

    # For Python with explicit expected, verify semantic equivalence (expressions only)
    if codegen_lang == "python" and codegen_has_explicit:
        # Skip semantic check for statements (def, class, etc.)
        input_stripped = codegen_input.strip()
        if not any(input_stripped.startswith(kw) for kw in ("def ", "class ", "@", "if ", "for ", "while ")):
            try:
                input_result = eval(input_stripped)
                output_result = eval(transpiled_output.strip())
                if input_result != output_result:
                    pytest.fail(
                        f"Semantic mismatch:\n  input  {input_stripped!r} = {input_result!r}\n  output {transpiled_output.strip()!r} = {output_result!r}"
                    )
            except SyntaxError:
                pass  # Not a simple expression, skip semantic check


def contains_normalized(haystack: str, needle: str) -> bool:
    """Check if needle appears in haystack, normalizing line-by-line whitespace."""
    needle_lines = [line.strip() for line in needle.strip().split("\n") if line.strip()]
    haystack_lines = [line.strip() for line in haystack.split("\n") if line.strip()]
    if not needle_lines:
        return True
    for i in range(len(haystack_lines)):
        if haystack_lines[i] == needle_lines[0]:
            match = True
            for j in range(1, len(needle_lines)):
                if i + j >= len(haystack_lines) or haystack_lines[i + j] != needle_lines[j]:
                    match = False
                    break
            if match:
                return True
    return False
