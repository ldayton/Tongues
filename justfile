set shell := ["bash", "-o", "pipefail", "-cu"]

# Verify all transpiler source is subset-compliant
subset:
    #!/usr/bin/env bash
    set -euo pipefail
    cd tongues
    failed=0
    for f in $(find src -name '*.py') tests/run_tests.py; do
        if ! uv run python -m src.tongues --verify < "$f" 2>/dev/null; then
            echo "FAIL: $f"
            uv run python -m src.tongues --verify < "$f" 2>&1 | head -5
            failed=1
        fi
    done
    exit $failed

# Run codegen tests locally
test-codegen-local:
    uv run --directory tongues pytest ../tests/test_codegen.py -v

# Run apptests locally for a specific language (or all if not specified)
test-apptests-local lang="":
    uv run --directory tongues pytest ../tests/test_apptests.py {{ if lang != "" { "--target " + lang } else { "" } }} -v

# Lint (--fix to apply changes)
lint *ARGS:
    uvx ruff check {{ if ARGS == "--fix" { "--fix" } else { "" } }} tongues/

# Format (--fix to apply changes)
fmt *ARGS:
    uvx ruff format {{ if ARGS == "--fix" { "" } else { "--check" } }} .

check: fmt lint subset test-codegen
    #!/usr/bin/env bash
    failed=0
    for lang in c csharp dart go java javascript lua perl php python ruby rust swift typescript zig; do
        just test-apptests "$lang" || failed=1
    done
    exit $failed

# Build Docker image for a language
docker-build lang:
    docker build -t tongues-{{lang}} docker/{{lang}}

# Run codegen tests in Docker (uses python image)
test-codegen:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        bash -c "rm -rf tongues/.venv && uv run --directory tongues pytest ../tests/test_codegen.py -v"

# Run apptests in Docker for a language (image must have python+uv installed)
test-apptests lang:
    docker build -t tongues-{{lang}} docker/{{lang}}
    docker run --rm -v "$(pwd):/workspace" tongues-{{lang}} \
        bash -c "rm -rf tongues/.venv && uv run --directory tongues pytest ../tests/test_apptests.py --target {{lang}} -v"

# Run all tests in Docker
test: test-codegen \
    (test-apptests "c") \
    (test-apptests "csharp") \
    (test-apptests "dart") \
    (test-apptests "go") \
    (test-apptests "java") \
    (test-apptests "javascript") \
    (test-apptests "lua") \
    (test-apptests "perl") \
    (test-apptests "php") \
    (test-apptests "python") \
    (test-apptests "ruby") \
    (test-apptests "rust") \
    (test-apptests "swift") \
    (test-apptests "typescript") \
    (test-apptests "zig")
