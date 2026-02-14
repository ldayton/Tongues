set shell := ["bash", "-o", "pipefail", "-cu"]

# Verify all transpiler source is subset-compliant
subset: subset-tongues subset-taytsh

# Verify tongues source is subset-compliant
subset-tongues:
    #!/usr/bin/env bash
    set -euo pipefail
    failed=0
    for f in $(find tongues/src -name '*.py'); do
        [ ! -s "$f" ] && continue
        if ! uv run --directory tongues python -m src.tongues --stop-at subset < "$f" 2>/dev/null; then
            echo "FAIL: $f"
            uv run --directory tongues python -m src.tongues --stop-at subset < "$f" 2>&1 | head -5
            failed=1
        fi
    done
    exit $failed

# Verify taytsh source is subset-compliant
subset-taytsh:
    #!/usr/bin/env bash
    set -euo pipefail
    failed=0
    for f in $(find taytsh/src -name '*.py'); do
        [ ! -s "$f" ] && continue
        if ! uv run --directory tongues python -m src.tongues --stop-at subset < "$f" 2>/dev/null; then
            echo "FAIL: $f"
            uv run --directory tongues python -m src.tongues --stop-at subset < "$f" 2>&1 | head -5
            failed=1
        fi
    done
    exit $failed

# Run CLI tests locally
test-cli-local:
    uv run --directory tongues pytest tests/test_runner.py -k test_cli -v

# Run parse tests locally
test-parse-local:
    uv run --directory tongues pytest tests/test_runner.py -k test_parse -v

# Run subset tests locally
test-subset-local:
    uv run --directory tongues pytest tests/test_runner.py -k test_subset -v

# Run names tests locally
test-names-local:
    uv run --directory tongues pytest tests/test_runner.py -k test_names -v

# Run signatures tests locally
test-signatures-local:
    uv run --directory tongues pytest tests/test_runner.py -k test_sigs -v

# Run fields tests locally
test-fields-local:
    uv run --directory tongues pytest tests/test_runner.py -k test_fields -v

# Run hierarchy tests locally
test-hierarchy-local:
    uv run --directory tongues pytest tests/test_runner.py -k test_hierarchy -v

# Run inference tests locally
test-inference-local:
    uv run --directory tongues pytest tests/test_runner.py -k test_inference -v

# Run lowering tests locally
test-lowering-local:
    uv run --directory tongues pytest tests/test_runner.py -k test_lowering -v

# Run middleend tests locally
test-middleend-local:
    uv run --directory tongues pytest tests/test_runner.py -k "test_type_checking or test_scope or test_returns or test_liveness or test_strings or test_hoisting or test_ownership or test_callgraph" -v

# Run codegen and app tests locally
test-codegen-local:
    uv run --directory tongues pytest tests/test_runner.py -k "test_codegen or test_app" -v

# Run taytsh tests locally
test-taytsh-local:
    uv run --directory taytsh pytest tests/test_runner.py -v

# Lint all (--fix to apply changes)
lint *ARGS: (lint-tongues ARGS) (lint-taytsh ARGS)

# Lint tongues (--fix to apply changes)
lint-tongues *ARGS:
    uv run --directory tongues ruff check {{ if ARGS == "--fix" { "--fix" } else { "" } }} src/

# Lint taytsh (--fix to apply changes)
lint-taytsh *ARGS:
    uv run --directory taytsh ruff check {{ if ARGS == "--fix" { "--fix" } else { "" } }} src/

# Format all (--fix to apply changes)
fmt *ARGS: (fmt-tongues ARGS) (fmt-taytsh ARGS)

# Format tongues (--fix to apply changes)
fmt-tongues *ARGS:
    uv run --directory tongues ruff format {{ if ARGS == "--fix" { "" } else { "--check" } }} .

# Format taytsh (--fix to apply changes)
fmt-taytsh *ARGS:
    uv run --directory taytsh ruff format {{ if ARGS == "--fix" { "" } else { "--check" } }} .

check:
    #!/usr/bin/env bash
    declare -A results
    failed=0
    just fmt-tongues && results[fmt-tongues]=✅ || { results[fmt-tongues]=❌; failed=1; }
    just fmt-taytsh && results[fmt-taytsh]=✅ || { results[fmt-taytsh]=❌; failed=1; }
    just lint-tongues && results[lint-tongues]=✅ || { results[lint-tongues]=❌; failed=1; }
    just lint-taytsh && results[lint-taytsh]=✅ || { results[lint-taytsh]=❌; failed=1; }
    just subset-tongues && results[subset-tongues]=✅ || { results[subset-tongues]=❌; failed=1; }
    just subset-taytsh && results[subset-taytsh]=✅ || { results[subset-taytsh]=❌; failed=1; }
    just test-cli && results[cli]=✅ || { results[cli]=❌; failed=1; }
    just test-parse && results[parse]=✅ || { results[parse]=❌; failed=1; }
    just test-subset && results[subset-tests]=✅ || { results[subset-tests]=❌; failed=1; }
    just test-names && results[names]=✅ || { results[names]=❌; failed=1; }
    just test-signatures && results[signatures]=✅ || { results[signatures]=❌; failed=1; }
    just test-fields && results[fields]=✅ || { results[fields]=❌; failed=1; }
    just test-hierarchy && results[hierarchy]=✅ || { results[hierarchy]=❌; failed=1; }
    just test-inference && results[inference]=✅ || { results[inference]=❌; failed=1; }
    just test-lowering && results[lowering]=✅ || { results[lowering]=❌; failed=1; }
    just test-middleend && results[middleend]=✅ || { results[middleend]=❌; failed=1; }
    just test-codegen && results[codegen]=✅ || { results[codegen]=❌; failed=1; }
    just test-taytsh && results[taytsh]=✅ || { results[taytsh]=❌; failed=1; }
    echo ""
    echo "══════════════════════════════════════"
    echo "           CHECK SUMMARY"
    echo "══════════════════════════════════════"
    printf "%-14s %s\n" "TARGET" "STATUS"
    printf "%-14s %s\n" "──────" "──────"
    for t in fmt-tongues fmt-taytsh lint-tongues lint-taytsh subset-tongues subset-taytsh cli parse subset-tests names signatures fields hierarchy inference lowering middleend codegen taytsh; do
        printf "%-14s %s\n" "$t" "${results[$t]}"
    done
    echo "══════════════════════════════════════"
    if [ $failed -eq 0 ]; then echo "✅ ALL PASSED"; else echo "❌ SOME FAILED"; fi
    echo "══════════════════════════════════════"
    exit $failed

# Build Docker image for a language
docker-build lang:
    docker build -t tongues-{{lang}} docker/{{lang}}

# Run CLI tests in Docker
test-cli:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k test_cli -v

# Run parse tests in Docker
test-parse:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k test_parse -v

# Run subset tests in Docker
test-subset:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k test_subset -v

# Run names tests in Docker
test-names:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k test_names -v

# Run signatures tests in Docker
test-signatures:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k test_sigs -v

# Run fields tests in Docker
test-fields:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k test_fields -v

# Run hierarchy tests in Docker
test-hierarchy:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k test_hierarchy -v

# Run inference tests in Docker
test-inference:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k test_inference -v

# Run lowering tests in Docker
test-lowering:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k test_lowering -v

# Run middleend tests in Docker
test-middleend:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k "test_type_checking or test_scope or test_returns or test_liveness or test_strings or test_hoisting or test_ownership or test_callgraph" -v

# Run codegen and app tests in Docker
test-codegen:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k "test_codegen or test_app" -v

# Run taytsh tests in Docker
test-taytsh:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory taytsh pytest tests/test_runner.py -v

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
test: test-codegen test-taytsh

# Run all tests locally (requires matching runtime versions)
test-local:
    #!/usr/bin/env bash
    declare -A results
    failed=0
    just versions && results[versions]=✅ || { results[versions]=❌; failed=1; }
    just test-cli-local && results[cli]=✅ || { results[cli]=❌; failed=1; }
    just test-parse-local && results[parse]=✅ || { results[parse]=❌; failed=1; }
    just test-subset-local && results[subset]=✅ || { results[subset]=❌; failed=1; }
    just test-names-local && results[names]=✅ || { results[names]=❌; failed=1; }
    just test-signatures-local && results[signatures]=✅ || { results[signatures]=❌; failed=1; }
    just test-fields-local && results[fields]=✅ || { results[fields]=❌; failed=1; }
    just test-hierarchy-local && results[hierarchy]=✅ || { results[hierarchy]=❌; failed=1; }
    just test-inference-local && results[inference]=✅ || { results[inference]=❌; failed=1; }
    just test-lowering-local && results[lowering]=✅ || { results[lowering]=❌; failed=1; }
    just test-middleend-local && results[middleend]=✅ || { results[middleend]=❌; failed=1; }
    just test-codegen-local && results[codegen]=✅ || { results[codegen]=❌; failed=1; }
    just test-taytsh-local && results[taytsh]=✅ || { results[taytsh]=❌; failed=1; }
    echo ""
    echo "══════════════════════════════════════"
    echo "         TEST-LOCAL SUMMARY"
    echo "══════════════════════════════════════"
    printf "%-14s %s\n" "TARGET" "STATUS"
    printf "%-14s %s\n" "──────" "──────"
    for t in versions cli parse subset names signatures fields hierarchy inference lowering middleend codegen taytsh; do
        printf "%-14s %s\n" "$t" "${results[$t]}"
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
    just fmt-tongues && results[fmt-tongues]=✅ || { results[fmt-tongues]=❌; failed=1; }
    just fmt-taytsh && results[fmt-taytsh]=✅ || { results[fmt-taytsh]=❌; failed=1; }
    just lint-tongues && results[lint-tongues]=✅ || { results[lint-tongues]=❌; failed=1; }
    just lint-taytsh && results[lint-taytsh]=✅ || { results[lint-taytsh]=❌; failed=1; }
    just subset-tongues && results[subset-tongues]=✅ || { results[subset-tongues]=❌; failed=1; }
    just subset-taytsh && results[subset-taytsh]=✅ || { results[subset-taytsh]=❌; failed=1; }
    just test-cli-local && results[cli]=✅ || { results[cli]=❌; failed=1; }
    just test-parse-local && results[parse]=✅ || { results[parse]=❌; failed=1; }
    just test-subset-local && results[subset-tests]=✅ || { results[subset-tests]=❌; failed=1; }
    just test-names-local && results[names]=✅ || { results[names]=❌; failed=1; }
    just test-signatures-local && results[signatures]=✅ || { results[signatures]=❌; failed=1; }
    just test-fields-local && results[fields]=✅ || { results[fields]=❌; failed=1; }
    just test-hierarchy-local && results[hierarchy]=✅ || { results[hierarchy]=❌; failed=1; }
    just test-inference-local && results[inference]=✅ || { results[inference]=❌; failed=1; }
    just test-lowering-local && results[lowering]=✅ || { results[lowering]=❌; failed=1; }
    just test-middleend-local && results[middleend]=✅ || { results[middleend]=❌; failed=1; }
    just test-codegen-local && results[codegen]=✅ || { results[codegen]=❌; failed=1; }
    just test-taytsh-local && results[taytsh]=✅ || { results[taytsh]=❌; failed=1; }
    echo ""
    echo "══════════════════════════════════════"
    echo "        CHECK-LOCAL SUMMARY"
    echo "══════════════════════════════════════"
    printf "%-14s %s\n" "TARGET" "STATUS"
    printf "%-14s %s\n" "──────" "──────"
    for t in versions fmt-tongues fmt-taytsh lint-tongues lint-taytsh subset-tongues subset-taytsh cli parse subset-tests names signatures fields hierarchy inference lowering middleend codegen taytsh; do
        printf "%-14s %s\n" "$t" "${results[$t]}"
    done
    echo "══════════════════════════════════════"
    if [ $failed -eq 0 ]; then echo "✅ ALL PASSED"; else echo "❌ SOME FAILED"; fi
    echo "══════════════════════════════════════"
    exit $failed

# Install VS Code syntax highlighting extension for Taytsh
vscode:
    #!/usr/bin/env bash
    cd editors/vscode
    rm -f taytsh-syntax-*.vsix
    npx @vscode/vsce package --allow-missing-repository
    shopt -s nullglob
    vsix=(taytsh-syntax-*.vsix)
    if [ ${#vsix[@]} -ne 1 ]; then
        echo "expected exactly one VSIX, found ${#vsix[@]}"
        ls -la
        exit 1
    fi
    code --install-extension "${vsix[0]}"
