set shell := ["bash", "-o", "pipefail", "-cu"]

# Quick pre-push check: style + test (--fix to auto-fix style)
prep *ARGS:
    #!/usr/bin/env bash
    log=/tmp/tongues-prep-$(date +%s).log
    rc=0
    { just style {{ARGS}} && just test; } 2>&1 | tee "$log" || rc=$?
    echo "$log"
    exit $rc

# Run lint, fmt, subset in parallel (--fix to auto-fix)
style *ARGS:
    #!/usr/bin/env bash
    set -uo pipefail
    pids=()
    just lint {{ARGS}} & pids+=($!)
    just fmt {{ARGS}} & pids+=($!)
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

# Run all frontend tests (cli, parse, subset, names, sigs, fields, hierarchy, pycheck, lowering, linker)
test-frontend:
    uv run --directory tongues pytest tests/test_frontend.py tests/test_frontend_linker.py -v -n auto

# Run CLI tests
test-cli:
    uv run --directory tongues pytest tests/test_frontend.py -k test_cli -v

# Run parse tests
test-parse:
    uv run --directory tongues pytest tests/test_frontend.py -k test_parse -v

# Run subset tests
test-subset:
    uv run --directory tongues pytest tests/test_frontend.py -k test_subset -v

# Run names tests
test-names:
    uv run --directory tongues pytest tests/test_frontend.py -k test_names -v

# Run signatures tests
test-signatures:
    uv run --directory tongues pytest tests/test_frontend.py -k test_sigs -v

# Run fields tests
test-fields:
    uv run --directory tongues pytest tests/test_frontend.py -k test_fields -v

# Run hierarchy tests
test-hierarchy:
    uv run --directory tongues pytest tests/test_frontend.py -k test_hierarchy -v

# Run pycheck tests
test-pycheck:
    uv run --directory tongues pytest tests/test_frontend.py -k test_pycheck -v

# Run lowering tests
test-lowering:
    uv run --directory tongues pytest tests/test_frontend.py -k test_lowering -v

# Run linker tests
test-linker:
    uv run --directory tongues pytest tests/test_frontend_linker.py -v

# Run all middleend tests (scope, returns, liveness, strings, hoisting, ownership, callgraph)
test-middleend:
    uv run --directory tongues pytest tests/test_middleend.py -v -n auto

# Run all taytsh tests (typarse, tycheck, app, vm, gen-check)
test-taytsh:
    uv run --directory tongues pytest tests/test_taytsh.py tests/test_taytsh_app.py tests/test_taytsh_vm.py tests/test_taytsh_gen_check.py -v

# Run generative type-checker tests
test-tycheck-gen:
    uv run --directory tongues pytest tests/test_taytsh_gen_check.py -v

# Run backend tests (codegen + apptests)
test-backend:
    uv run --directory tongues pytest tests/test_backend_codegen.py tests/test_backend_target.py -v -n auto

# Run declaration ordering tests
test-ordering:
    uv run --directory tongues pytest tests/test_backend_target.py -k test_ordering -v

# Run softfloat library tests
test-lib:
    uv run --directory tongues pytest tests/test_lib_softfloat.py -v

# Lint (--fix to apply changes)
lint *ARGS:
    uv run --directory tongues ruff check {{ if ARGS == "--fix" { "--fix" } else { "" } }} src/

# Format (--fix to apply changes)
fmt *ARGS:
    uv run --directory tongues ruff format {{ if ARGS == "--fix" { "" } else { "--check" } }} .
    npx prettier {{ if ARGS == "--fix" { "--write" } else { "--check" } }} spec/*.md

# Self-transpile: emit to .out/tongues.{ext}
_self-transpile target="python":
    #!/usr/bin/env bash
    set -euo pipefail
    declare -A ext=([python]=py [ruby]=rb [perl]=pl [taytsh]=ty)
    mkdir -p tongues/.out
    cd tongues && uv run bin/tongues --target {{target}} -o ".out/tongues.${ext[{{target}}]}" src
    if [ "{{target}}" = "python" ]; then
        uv run python3 -c "import ast; ast.parse(open('.out/tongues.py').read())"
    fi

# Self-transpile and test against transpiled Python binary
lang-python *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just _self-transpile python
    uv run --directory tongues pytest tests/test_frontend.py tests/test_middleend.py \
        tests/test_backend_codegen.py tests/test_backend_target.py tests/test_taytsh_app.py \
        tests/test_frontend_linker.py \
        --transpiled ".out/tongues.py" -v {{ ARGS }}

# Self-transpile and test against transpiled Ruby binary
lang-ruby:
    #!/usr/bin/env bash
    set -euo pipefail
    just _self-transpile ruby
    cd tongues
    printf 'tests/shared/test_harness.py\0%s\0lib/json.py\0%s' \
        "$(<tests/shared/test_harness.py)" "$(<src/lib/json.py)" \
        | uv run bin/tongues --project --target ruby -o .out/test_harness.rb
    ruby tests/test-transpiled.rb ".out/tongues.rb"

# Self-transpile and test against transpiled Perl binary
lang-perl:
    #!/usr/bin/env bash
    set -euo pipefail
    just _self-transpile perl
    cd tongues
    printf 'tests/shared/test_harness.py\0%s\0lib/json.py\0%s' \
        "$(<tests/shared/test_harness.py)" "$(<src/lib/json.py)" \
        | uv run bin/tongues --project --target perl -o .out/test_harness.pl
    perl tests/test-transpiled.pl ".out/tongues.pl"

# Self-transpile to Taytsh and test through treewalker
lang-taytsh-treewalker *ARGS:
    just _self-transpile taytsh
    just _lang-taytsh-treewalker-frontend {{ ARGS }}
    just _lang-taytsh-treewalker-middleend {{ ARGS }}
    just _lang-taytsh-treewalker-backend {{ ARGS }}
    just _lang-taytsh-treewalker-apptest {{ ARGS }}

_lang-taytsh-treewalker-frontend *ARGS:
    uv run --directory tongues pytest tests/test_frontend.py tests/test_frontend_linker.py \
        --transpiled ".out/tongues.ty" --taytsh-runner treewalker -v {{ ARGS }}

_lang-taytsh-treewalker-middleend *ARGS:
    uv run --directory tongues pytest tests/test_middleend.py \
        --transpiled ".out/tongues.ty" --taytsh-runner treewalker -v {{ ARGS }}

_lang-taytsh-treewalker-backend *ARGS:
    uv run --directory tongues pytest tests/test_backend_codegen.py \
        --transpiled ".out/tongues.ty" --taytsh-runner treewalker -v {{ ARGS }}

_lang-taytsh-treewalker-apptest *ARGS:
    uv run --directory tongues pytest tests/test_backend_target.py tests/test_taytsh_app.py \
        --transpiled ".out/tongues.ty" --taytsh-runner treewalker -v {{ ARGS }}

# Self-transpile to Taytsh and test through VM
lang-taytsh-vm *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    just _self-transpile taytsh
    uv run --directory tongues pytest tests/test_frontend.py tests/test_middleend.py \
        tests/test_backend_codegen.py tests/test_backend_target.py tests/test_taytsh_app.py \
        tests/test_frontend_linker.py \
        --transpiled ".out/tongues.ty" --taytsh-runner vm -v {{ ARGS }}

# Run a just target inside Docker
docker target lang="python":
    docker build -t tongues-{{lang}} docker/{{lang}}
    docker run --rm -v "$(pwd):/workspace" tongues-{{lang}} just {{target}}

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
    printf "%-12s %-10s %-10s %-6s  %-20s %s\n" "LANG" "EXPECTED" "LOCAL" "STATUS" "MAC" "LINUX"
    printf "%-12s %-10s %-10s %-6s  %-20s %s\n" "----" "--------" "-----" "------" "---" "-----"
    check() {
        lang=$1; expected=$2; mac=$3; linux=$4; cmd=$5
        local_ver=$(eval "$cmd" 2>/dev/null || echo "not found")
        if echo "$local_ver" | grep -q "$expected"; then
            status="✅"
        else
            status="❌"
            failed=1
        fi
        printf "%-12s %-10s %-10s %-6s  %-20s %s\n" "$lang" "$expected" "$local_ver" "$status" "$mac" "$linux"
    }
    check "c"          "13."   "brew gcc@13"        "brew gcc@13"        "gcc --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1"
    check "csharp"     "8."    "brew dotnet@8"      "brew dotnet@8"      "dotnet --version | cut -d. -f1-2"
    check "dart"       "3."    "brew dart-sdk"      "brew dart-sdk"      "dart --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1"
    check "go"         "1.21"  "brew go@1.21"       "brew go@1.21"       "go version | grep -oE 'go[0-9]+\.[0-9]+' | sed 's/go//'"
    check "java"       "21"    "brew openjdk@21"    "brew openjdk@21"    "java --version 2>&1 | head -1 | grep -oE '[0-9]+' | head -1"
    check "javascript" "20"    "brew node@20"       "brew node@20"       "node --version | grep -oE '[0-9]+' | head -1"
    check "lua"        "5.4"   "brew lua"           "brew lua"           "lua -v 2>&1 | grep -oE '[0-9]+\.[0-9]+'"
    check "perl"       "5."    "brew perl"          "brew perl"          "perl -v | grep -oE 'v[0-9]+\.[0-9]+' | sed 's/v//'"
    check "php"        "8.3"   "brew php@8.3"       "brew php@8.3"       "php --version | head -1 | grep -oE '[0-9]+\.[0-9]+'"
    check "python"     "3.12"  "brew python@3.12"   "brew python@3.12"   "python --version | grep -oE '[0-9]+\.[0-9]+'"
    check "python3"    "3.12"  "brew python@3.12"   "brew python@3.12"   "python3 --version | grep -oE '[0-9]+\.[0-9]+'"
    check "ruby"       "3."    "brew ruby@3.3"      "brew ruby@3.3"      "ruby --version | grep -oE '[0-9]+\.[0-9]+'"
    check "rust"       "1.75"  "rustup"             "rustup"             "rustc --version | grep -oE '[0-9]+\.[0-9]+'"
    check "swift"      "6."    "xcode"              "brew swift"         "swift --version 2>&1 | grep -oE 'Swift version [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+'"
    check "typescript" "5.3"   ".direnv npm"        ".direnv npm"        "tsc --version | grep -oE '[0-9]+\.[0-9]+'"
    check "zig"        "0.14"  "brew zig@0.14"      "brew zig@0.14"      "zig version | grep -oE '[0-9]+\.[0-9]+'"
    exit $failed

# Run all tests (requires matching runtime versions)
test:
    #!/usr/bin/env bash
    declare -A results
    failed=0
    just versions && results[versions]=✅ || { results[versions]=❌; failed=1; }
    uv run --directory tongues pytest tests/test_frontend.py tests/test_frontend_linker.py \
        tests/test_middleend.py tests/test_taytsh.py tests/test_taytsh_app.py \
        tests/test_taytsh_vm.py tests/test_taytsh_gen_check.py \
        tests/test_backend_codegen.py tests/test_backend_target.py \
        tests/test_lib_softfloat.py -v -n auto \
        && results[tests]=✅ || { results[tests]=❌; failed=1; }
    # Self-transpile + test all three targets in parallel
    _st() {
        local lang=$1
        just "lang-$lang"
    }
    _st python & pid_py=$!
    _st ruby & pid_rb=$!
    _st perl & pid_pl=$!
    _st taytsh-treewalker & pid_ty_tw=$!
    _st taytsh-vm & pid_ty_vm=$!
    wait $pid_py && results[lang-python]=✅ || { results[lang-python]=❌; failed=1; }
    wait $pid_rb && results[lang-ruby]=✅ || { results[lang-ruby]=❌; failed=1; }
    wait $pid_pl && results[lang-perl]=✅ || { results[lang-perl]=❌; failed=1; }
    wait $pid_ty_tw && results[lang-taytsh-tw]=✅ || { results[lang-taytsh-tw]=❌; failed=1; }
    wait $pid_ty_vm && results[lang-taytsh-vm]=✅ || { results[lang-taytsh-vm]=❌; failed=1; }
    echo ""
    echo "══════════════════════════════════════"
    echo "           TEST SUMMARY"
    echo "══════════════════════════════════════"
    printf "%-16s %s\n" "TARGET" "STATUS"
    printf "%-16s %s\n" "──────" "──────"
    for t in versions tests lang-python lang-ruby lang-perl lang-taytsh-tw lang-taytsh-vm; do
        printf "%-16s %s\n" "$t" "${results[$t]}"
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
