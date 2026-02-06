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

# Run codegen tests
test-codegen:
    uv run --directory tongues pytest ../tests/test_codegen.py -v

# Run apptests for a specific language (or all if not specified)
test-apptests lang="":
    uv run --directory tongues pytest ../tests/test_apptests.py {{ if lang != "" { "--target " + lang } else { "" } }} -v

# Run all transpiler tests
test: test-codegen test-apptests

# Lint (--fix to apply changes)
lint *ARGS:
    uvx ruff check {{ if ARGS == "--fix" { "--fix" } else { "" } }} tongues/

# Format (--fix to apply changes)
fmt *ARGS:
    uvx ruff format {{ if ARGS == "--fix" { "" } else { "--check" } }} .

check: fmt lint subset test

check-docker: fmt lint subset test-docker

# Build Docker image for a language
docker-build lang:
    docker build -t tongues-{{lang}} docker/{{lang}}

# Run codegen tests in Docker (uses python image)
test-codegen-docker:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest ../tests/test_codegen.py -v

# Run apptests in Docker for a language (image must have python+uv installed)
test-apptests-docker lang="python":
    docker build -t tongues-{{lang}} docker/{{lang}}
    docker run --rm -v "$(pwd):/workspace" tongues-{{lang}} \
        uv run --directory tongues pytest ../tests/test_apptests.py --target {{lang}} -v

# Run all Docker tests (all languages with Docker images)
test-docker: test-codegen-docker \
    (test-apptests-docker "c") \
    (test-apptests-docker "csharp") \
    (test-apptests-docker "dart") \
    (test-apptests-docker "go") \
    (test-apptests-docker "java") \
    (test-apptests-docker "javascript") \
    (test-apptests-docker "lua") \
    (test-apptests-docker "perl") \
    (test-apptests-docker "php") \
    (test-apptests-docker "python") \
    (test-apptests-docker "ruby") \
    (test-apptests-docker "rust") \
    (test-apptests-docker "swift") \
    (test-apptests-docker "typescript") \
    (test-apptests-docker "zig")
