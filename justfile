set shell := ["bash", "-o", "pipefail", "-cu"]

# Verify all transpiler source is subset-compliant
subset:
    #!/usr/bin/env bash
    set -euo pipefail
    cd tongues
    failed=0
    for f in $(fd -e py . src) tests/run_tests.py; do
        if ! uv run python -m src.tongues --verify < "$f" 2>/dev/null; then
            echo "FAIL: $f"
            uv run python -m src.tongues --verify < "$f" 2>&1 | head -5
            failed=1
        fi
    done
    exit $failed

# Run phase tests (subset/names verification)
test-phases:
    python3 tongues/tests/run_testsw.py tongues/tests/phases/

# Run codegen tests
test-codegen:
    python3 tongues/tests/run_codegen_testsw.py tongues/tests/codegen/

# Run Python apptests
test-apptests:
    uv run --directory tongues pytest ../tests/test_apptests.py -k "python" -v

# Run all transpiler tests
test: test-phases test-codegen test-apptests

# Lint (--fix to apply changes)
lint *ARGS:
    uvx ruff check {{ if ARGS == "--fix" { "--fix" } else { "" } }} tongues/

# Format (--fix to apply changes)
fmt *ARGS:
    uvx ruff format {{ if ARGS == "--fix" { "" } else { "--check" } }} .

check: fmt lint subset test
