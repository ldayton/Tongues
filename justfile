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
    uv run --directory tongues pytest ../tests/test_15_codegen.py -v

# Run apptests locally for a specific language (or all if not specified)
test-apptests-local lang="":
    uv run --directory tongues pytest ../tests/test_15_app.py {{ if lang != "" { "--target " + lang } else { "" } }} -v

# Lint (--fix to apply changes)
lint *ARGS:
    uvx ruff check {{ if ARGS == "--fix" { "--fix" } else { "" } }} tongues/

# Format (--fix to apply changes)
fmt *ARGS:
    uvx ruff format {{ if ARGS == "--fix" { "" } else { "--check" } }} .

check:
    #!/usr/bin/env bash
    declare -A results
    failed=0
    just fmt && results[fmt]=✅ || { results[fmt]=❌; failed=1; }
    just lint && results[lint]=✅ || { results[lint]=❌; failed=1; }
    just subset && results[subset]=✅ || { results[subset]=❌; failed=1; }
    just test-codegen && results[codegen]=✅ || { results[codegen]=❌; failed=1; }
    for lang in c csharp dart go java javascript lua perl php python ruby rust swift typescript zig; do
        just test-apptests "$lang" && results[$lang]=✅ || { results[$lang]=❌; failed=1; }
    done
    echo ""
    echo "══════════════════════════════════════"
    echo "           CHECK SUMMARY"
    echo "══════════════════════════════════════"
    printf "%-12s %s\n" "TARGET" "STATUS"
    printf "%-12s %s\n" "──────" "──────"
    for t in fmt lint subset codegen; do
        printf "%-12s %s\n" "$t" "${results[$t]}"
    done
    echo "──────────── ──────"
    for lang in c csharp dart go java javascript lua perl php python ruby rust swift typescript zig; do
        printf "%-12s %s\n" "$lang" "${results[$lang]}"
    done
    echo "══════════════════════════════════════"
    if [ $failed -eq 0 ]; then echo "✅ ALL PASSED"; else echo "❌ SOME FAILED"; fi
    echo "══════════════════════════════════════"
    exit $failed

# Build Docker image for a language
docker-build lang:
    docker build -t tongues-{{lang}} docker/{{lang}}

# Run codegen tests in Docker (uses python image)
test-codegen:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        bash -c "rm -rf tongues/.venv && uv run --directory tongues pytest ../tests/test_15_codegen.py -v"

# Run apptests in Docker for a language (image must have python+uv installed)
test-apptests lang:
    docker build -t tongues-{{lang}} docker/{{lang}}
    docker run --rm -v "$(pwd):/workspace" tongues-{{lang}} \
        bash -c "rm -rf tongues/.venv && uv run --directory tongues pytest ../tests/test_15_app.py --target {{lang}} -v"

# Check if formatters are installed
formatters:
    #!/usr/bin/env bash
    failed=0
    printf "%-12s %-30s %s\n" "LANG" "FORMATTER" "STATUS"
    printf "%-12s %-30s %s\n" "----" "---------" "------"
    check() {
        lang=$1; name=$2; cmd=$3
        if eval "$cmd" >/dev/null 2>&1; then
            status="✅"
        else
            status="❌"
            failed=1
        fi
        printf "%-12s %-30s %s\n" "$lang" "$name" "$status"
    }
    check "c"          "clang-format"              "command -v clang-format"
    check "csharp"     "csharpier"                 "command -v dotnet-csharpier || dotnet tool list -g | grep -q csharpier"
    check "dart"       "dart format"               "command -v dart"
    check "go"         "gofmt"                     "command -v gofmt"
    check "java"       "google-java-format"        "test -f /opt/java-tools/google-java-format.jar"
    check "javascript" "biome (via npx)"           "command -v npx"
    check "lua"        "stylua"                    "command -v stylua"
    check "perl"       "perltidy"                  "command -v perltidy"
    check "php"        "php-cs-fixer"              "command -v php-cs-fixer"
    check "python"     "ruff (via uvx)"            "command -v uvx"
    check "ruby"       "rubocop"                   "command -v rubocop"
    check "rust"       "rustfmt"                   "command -v rustfmt"
    check "swift"      "swiftformat"               "command -v swiftformat"
    check "typescript" "biome (via npx)"           "command -v npx"
    check "zig"        "zig fmt"                   "command -v zig"
    exit $failed

# Check local runtime versions against Dockerfile expectations
versions:
    #!/usr/bin/env bash
    failed=0
    printf "%-12s %-20s %-20s %s\n" "LANG" "EXPECTED" "LOCAL" "STATUS"
    printf "%-12s %-20s %-20s %s\n" "----" "--------" "-----" "------"
    check() {
        lang=$1; expected=$2; cmd=$3
        local_ver=$(eval "$cmd" 2>/dev/null || echo "not found")
        if echo "$local_ver" | grep -q "$expected"; then
            status="✅"
        else
            status="❌"
            failed=1
        fi
        printf "%-12s %-20s %-20s %s\n" "$lang" "$expected" "$local_ver" "$status"
    }
    check "c"          "13."     "gcc --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1"
    check "csharp"     "8."      "dotnet --version | cut -d. -f1-2"
    check "dart"       "3.2"     "dart --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1"
    check "go"         "1.21"    "go version | grep -oE 'go[0-9]+\.[0-9]+' | sed 's/go//'"
    check "java"       "21"      "java --version 2>&1 | head -1 | grep -oE '[0-9]+' | head -1"
    check "javascript" "21"      "node --version | grep -oE '[0-9]+' | head -1"
    check "lua"        "5.4"     "lua -v 2>&1 | grep -oE '[0-9]+\.[0-9]+'"
    check "perl"       "5.38"    "perl -v | grep -oE 'v[0-9]+\.[0-9]+' | sed 's/v//'"
    check "php"        "8.3"     "php --version | head -1 | grep -oE '[0-9]+\.[0-9]+'"
    check "python"     "3.12"    "python --version | grep -oE '[0-9]+\.[0-9]+'"
    check "python3"    "3.12"    "python3 --version | grep -oE '[0-9]+\.[0-9]+'"
    check "ruby"       "3."      "ruby --version | grep -oE '[0-9]+\.[0-9]+'"
    check "rust"       "1.75"    "rustc --version | grep -oE '[0-9]+\.[0-9]+'"
    check "swift"      "6."      "xcrun swift --version 2>&1 | grep -oE 'Swift version [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+'"
    check "typescript" "5.3"     "tsc --version | grep -oE '[0-9]+\.[0-9]+'"
    check "zig"        "0.14"    "zig version | grep -oE '[0-9]+\.[0-9]+'"
    exit $failed

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

# Run all tests locally (requires matching runtime versions)
test-local:
    #!/usr/bin/env bash
    declare -A results
    failed=0
    just versions && results[versions]=✅ || { results[versions]=❌; failed=1; }
    just test-codegen-local && results[codegen]=✅ || { results[codegen]=❌; failed=1; }
    for lang in c csharp dart go java javascript lua perl php python ruby rust swift typescript zig; do
        just test-apptests-local "$lang" && results[$lang]=✅ || { results[$lang]=❌; failed=1; }
    done
    echo ""
    echo "══════════════════════════════════════"
    echo "         TEST-LOCAL SUMMARY"
    echo "══════════════════════════════════════"
    printf "%-12s %s\n" "TARGET" "STATUS"
    printf "%-12s %s\n" "──────" "──────"
    for t in versions codegen; do
        printf "%-12s %s\n" "$t" "${results[$t]}"
    done
    echo "──────────── ──────"
    for lang in c csharp dart go java javascript lua perl php python ruby rust swift typescript zig; do
        printf "%-12s %s\n" "$lang" "${results[$lang]}"
    done
    echo "══════════════════════════════════════"
    if [ $failed -eq 0 ]; then echo "✅ ALL PASSED"; else echo "❌ SOME FAILED"; fi
    echo "══════════════════════════════════════"
    exit $failed

# Run full check locally (requires matching runtime versions)
check-local:
    #!/usr/bin/env bash
    declare -A results
    failed=0
    just versions && results[versions]=✅ || { results[versions]=❌; failed=1; }
    just fmt && results[fmt]=✅ || { results[fmt]=❌; failed=1; }
    just lint && results[lint]=✅ || { results[lint]=❌; failed=1; }
    just subset && results[subset]=✅ || { results[subset]=❌; failed=1; }
    just test-codegen-local && results[codegen]=✅ || { results[codegen]=❌; failed=1; }
    for lang in c csharp dart go java javascript lua perl php python ruby rust swift typescript zig; do
        just test-apptests-local "$lang" && results[$lang]=✅ || { results[$lang]=❌; failed=1; }
    done
    echo ""
    echo "══════════════════════════════════════"
    echo "        CHECK-LOCAL SUMMARY"
    echo "══════════════════════════════════════"
    printf "%-12s %s\n" "TARGET" "STATUS"
    printf "%-12s %s\n" "──────" "──────"
    for t in versions fmt lint subset codegen; do
        printf "%-12s %s\n" "$t" "${results[$t]}"
    done
    echo "──────────── ──────"
    for lang in c csharp dart go java javascript lua perl php python ruby rust swift typescript zig; do
        printf "%-12s %s\n" "$lang" "${results[$lang]}"
    done
    echo "══════════════════════════════════════"
    if [ $failed -eq 0 ]; then echo "✅ ALL PASSED"; else echo "❌ SOME FAILED"; fi
    echo "══════════════════════════════════════"
    exit $failed
