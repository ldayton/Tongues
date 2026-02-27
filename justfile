set shell := ["bash", "-o", "pipefail", "-cu"]

# Quick pre-push check: lint, fmt, subset, test-local
prep:
    #!/usr/bin/env bash
    log=/tmp/tongues-prep-$(date +%s).log
    rc=0
    { just _lint-fmt-subset && just test-local; } 2>&1 | tee "$log" || rc=$?
    echo "$log"
    exit $rc

# Run lint, fmt, subset in parallel
_lint-fmt-subset:
    #!/usr/bin/env bash
    set -uo pipefail
    pids=() results=()
    just lint & pids+=($!)
    just fmt & pids+=($!)
    just subset & pids+=($!)
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    exit $failed

# Verify all transpiler source is subset-compliant
subset:
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

# Run all frontend tests locally (cli, parse, subset, names, sigs, fields, hierarchy, inference, lowering)
test-frontend-local:
    uv run --directory tongues pytest tests/test_runner.py -k "test_cli or test_parse or test_subset or test_names or test_sigs or test_fields or test_hierarchy or test_inference or test_lowering" -v -n auto

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

# Run all middleend tests locally (type checking, scope, returns, liveness, strings, hoisting, ownership, callgraph, taytsh, tycheck-gen)
test-middleend-local:
    uv run --directory tongues pytest tests/test_runner.py -k "test_type_checking or test_scope or test_returns or test_liveness or (test_strings and not apptest) or test_hoisting or test_ownership or test_callgraph or test_taytsh or test_tycheck_gen" tests/test_taytsh_vm.py tests/test_tycheck_gen.py -v -n auto

# Run backend tests (codegen + apptests) locally
test-backend-local:
    uv run --directory tongues pytest tests/test_runner.py -k "test_codegen or test_app" -v -n auto

# Run declaration ordering tests locally
test-ordering-local:
    uv run --directory tongues pytest tests/test_runner.py -k test_ordering -v

# Run taytsh tests locally
test-taytsh-local:
    uv run --directory tongues pytest tests/test_runner.py -k "test_taytsh" tests/test_taytsh_vm.py -v

# Run generative type-checker tests locally
test-tycheck-gen-local:
    uv run --directory tongues pytest tests/test_tycheck_gen.py -v

# Lint (--fix to apply changes)
lint *ARGS:
    uv run --directory tongues ruff check {{ if ARGS == "--fix" { "--fix" } else { "" } }} src/

# Format (--fix to apply changes)
fmt *ARGS:
    uv run --directory tongues ruff format {{ if ARGS == "--fix" { "" } else { "--check" } }} .

check:
    #!/usr/bin/env bash
    declare -A results
    failed=0
    just fmt && results[fmt]=✅ || { results[fmt]=❌; failed=1; }
    just lint && results[lint]=✅ || { results[lint]=❌; failed=1; }
    just subset && results[subset]=✅ || { results[subset]=❌; failed=1; }
    just test-frontend && results[frontend]=✅ || { results[frontend]=❌; failed=1; }
    just test-middleend && results[middleend]=✅ || { results[middleend]=❌; failed=1; }
    just test-backend && results[backend]=✅ || { results[backend]=❌; failed=1; }
    echo ""
    echo "══════════════════════════════════════"
    echo "           CHECK SUMMARY"
    echo "══════════════════════════════════════"
    printf "%-14s %s\n" "TARGET" "STATUS"
    printf "%-14s %s\n" "──────" "──────"
    for t in fmt lint subset frontend middleend backend; do
        printf "%-14s %s\n" "$t" "${results[$t]}"
    done
    echo "══════════════════════════════════════"
    if [ $failed -eq 0 ]; then echo "✅ ALL PASSED"; else echo "❌ SOME FAILED"; fi
    echo "══════════════════════════════════════"
    exit $failed

# Self-transpile: emit to .out/tongues.{ext}
self-transpile target="python":
    #!/usr/bin/env bash
    set -euo pipefail
    declare -A ext=([python]=py [ruby]=rb [perl]=pl)
    mkdir -p tongues/.out
    cd tongues && uv run bin/tongues --target {{target}} -o ".out/tongues.${ext[{{target}}]}" src
    if [ "{{target}}" = "python" ]; then
        uv run python3 -c "import ast; ast.parse(open('.out/tongues.py').read())"
    fi

# Run test suite against a transpiled binary locally
test-transpiled-local target="python":
    #!/usr/bin/env bash
    set -euo pipefail
    declare -A ext=([python]=py [ruby]=rb [perl]=pl)
    case "{{target}}" in
        python)
            uv run --directory tongues pytest tests/test_runner.py \
                --transpiled ".out/tongues.${ext[{{target}}]}" -v
            ;;
        ruby)
            cd tongues && ruby bin/test-transpiled.rb ".out/tongues.rb"
            ;;
        perl)
            cd tongues && perl bin/test-transpiled.pl ".out/tongues.pl"
            ;;
        *)
            echo "No native test harness for {{target}}, falling back to pytest"
            uv run --directory tongues pytest tests/test_runner.py \
                --transpiled ".out/tongues.${ext[{{target}}]}" -v
            ;;
    esac

# Run test suite against a transpiled binary in Docker
test-transpiled target="python":
    #!/usr/bin/env bash
    set -euo pipefail
    declare -A ext=([python]=py [ruby]=rb [perl]=pl)
    docker build -t tongues-{{target}} docker/{{target}}
    docker run --rm -v "$(pwd):/workspace" tongues-{{target}} \
        uv run --directory tongues pytest tests/test_runner.py \
        --transpiled ".out/tongues.${ext[{{target}}]}" -v

# Build Docker image for a language
docker-build lang:
    docker build -t tongues-{{lang}} docker/{{lang}}

# Run all frontend tests in Docker
test-frontend:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k "test_cli or test_parse or test_subset or test_names or test_sigs or test_fields or test_hierarchy or test_inference or test_lowering" -v

# Run all middleend tests in Docker
test-middleend:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k "test_type_checking or test_scope or test_returns or test_liveness or (test_strings and not apptest) or test_hoisting or test_ownership or test_callgraph or test_taytsh or test_tycheck_gen" tests/test_taytsh_vm.py tests/test_tycheck_gen.py -v

# Run backend tests (codegen + apptests) in Docker
test-backend:
    docker build -t tongues-python docker/python
    docker run --rm -v "$(pwd):/workspace" tongues-python \
        uv run --directory tongues pytest tests/test_runner.py -k "test_codegen or test_app" -v

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
test: test-frontend test-middleend test-backend self-transpile test-transpiled (self-transpile "ruby") (test-transpiled "ruby") (self-transpile "perl") (test-transpiled "perl")

# Run all tests locally (requires matching runtime versions)
test-local:
    #!/usr/bin/env bash
    declare -A results
    failed=0
    just versions && results[versions]=✅ || { results[versions]=❌; failed=1; }
    uv run --directory tongues pytest tests/test_runner.py tests/test_taytsh_vm.py tests/test_tycheck_gen.py -v -n auto \
        && results[tests]=✅ || { results[tests]=❌; failed=1; }
    # Self-transpile + test all three targets in parallel
    _st() {
        local lang=$1
        just self-transpile "$lang" && just test-transpiled-local "$lang"
    }
    _st python & pid_py=$!
    _st ruby & pid_rb=$!
    _st perl & pid_pl=$!
    wait $pid_py && results[transpiled-py]=✅ || { results[transpiled-py]=❌; failed=1; }
    wait $pid_rb && results[transpiled-rb]=✅ || { results[transpiled-rb]=❌; failed=1; }
    wait $pid_pl && results[transpiled-pl]=✅ || { results[transpiled-pl]=❌; failed=1; }
    echo ""
    echo "══════════════════════════════════════"
    echo "         TEST-LOCAL SUMMARY"
    echo "══════════════════════════════════════"
    printf "%-14s %s\n" "TARGET" "STATUS"
    printf "%-14s %s\n" "──────" "──────"
    for t in versions tests transpiled-py transpiled-rb transpiled-pl; do
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
