set shell := ["bash", "-o", "pipefail", "-cu"]

export VIRTUAL_ENV := ""

# Run lint and format checks (--fix to auto-fix)
style *ARGS:
    #!/usr/bin/env bash
    set -uo pipefail
    start=$SECONDS
    pids=()
    just -f {{justfile()}} lint {{ARGS}} & pids+=($!)
    just -f {{justfile()}} fmt {{ARGS}} & pids+=($!)
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    elapsed=$((SECONDS - start))
    if [ $failed -eq 0 ]; then
        printf '\033[32m[style] %ds\033[0m\n' "$elapsed"
    else
        printf '\033[31m[style] %ds (FAILED)\033[0m\n' "$elapsed"
        exit 1
    fi

# Lint (--fix to apply changes)
lint *ARGS:
    #!/usr/bin/env bash
    set -uo pipefail
    start=$SECONDS
    uv run --directory tongues ruff check {{ if ARGS == "--fix" { "--fix" } else { "" } }} src/; rc=$?
    elapsed=$((SECONDS - start))
    if [ $rc -eq 0 ]; then
        printf '\033[32m[lint] %ds\033[0m\n' "$elapsed"
    else
        printf '\033[31m[lint] %ds (FAILED)\033[0m\n' "$elapsed"
    fi
    exit $rc

# Format (--fix to apply changes)
fmt *ARGS:
    #!/usr/bin/env bash
    set -uo pipefail
    start=$SECONDS
    failed=0
    uv run --directory tongues ruff format {{ if ARGS == "--fix" { "" } else { "--check" } }} . || failed=1
    npx prettier {{ if ARGS == "--fix" { "--write" } else { "--check" } }} spec/*.md || failed=1
    elapsed=$((SECONDS - start))
    if [ $failed -eq 0 ]; then
        printf '\033[32m[fmt] %ds\033[0m\n' "$elapsed"
    else
        printf '\033[31m[fmt] %ds (FAILED)\033[0m\n' "$elapsed"
    fi
    exit $failed

# Transpile Tongues to target language
_transpile-tongues target:
    #!/usr/bin/env bash
    set -euo pipefail
    start=$SECONDS
    declare -A ext=([python]=py [ruby]=rb [perl]=pl [javascript]=js [java]=java [taytsh]=ty)
    mkdir -p tongues/.out
    cd tongues && uv run bin/tongues --target {{target}} -o ".out/tongues.${ext[{{target}}]}" src
    printf '\033[32m[transpile-tongues-{{target}}] %ds\033[0m\n' "$((SECONDS - start))"

# Transpile the shared test harness to target language
_transpile-harness target:
    #!/usr/bin/env bash
    set -euo pipefail
    start=$SECONDS
    declare -A ext=([python]=py [ruby]=rb [perl]=pl [javascript]=js)
    cd tongues
    printf 'tests/shared/test_harness.py\0%s\0lib/json.py\0%s' \
        "$(<tests/shared/test_harness.py)" "$(<src/lib/json.py)" \
        | uv run bin/tongues --project --target {{target}} -o ".out/test_harness.${ext[{{target}}]}"
    printf '\033[32m[transpile-harness:{{target}}] %ds\033[0m\n' "$((SECONDS - start))"

# Transpile Tongues and run test suite in target language
lang target:
    #!/usr/bin/env bash
    set -uo pipefail
    declare -A ext=([python]=py [ruby]=rb [perl]=pl [javascript]=js)
    declare -A runner=([python]=python3 [ruby]=ruby [perl]=perl [javascript]=node)
    just -f {{justfile()}} _transpile-tongues {{target}}
    just -f {{justfile()}} _transpile-harness {{target}}
    pids=()
    failed=0
    if [ "{{target}}" = "python" ]; then
        just -f {{justfile()}} _pyright & pids+=($!)
        just -f {{justfile()}} idempotence & pids+=($!)
    else
        just -f {{justfile()}} cross-equivalence {{target}} & pids+=($!)
    fi
    just -f {{justfile()}} _treewalker {{target}} & pids+=($!)
    just -f {{justfile()}} _vm {{target}} & pids+=($!)
    just -f {{justfile()}} _vm-test-tongues {{target}} & pids+=($!)
    start=$SECONDS
    cd tongues
    ${runner[{{target}}]} tests/test-transpiled.${ext[{{target}}]} .out/tongues.${ext[{{target}}]} --target test-tongues-{{target}}; rc=$?
    elapsed=$((SECONDS - start))
    if [ $rc -eq 0 ]; then
        printf '\033[32m[test-tongues-{{target}}] %ds\033[0m\n' "$elapsed"
    else
        printf '\033[31m[test-tongues-{{target}}] %ds (FAILED)\033[0m\n' "$elapsed"
        failed=1
    fi
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    exit $failed

# Self-transpile to Java, compile, and run tests
lang-java:
    #!/usr/bin/env bash
    set -uo pipefail
    just -f {{justfile()}} _transpile-tongues java
    just -f {{justfile()}} _compile-java
    pids=()
    failed=0
    just -f {{justfile()}} cross-equivalence java & pids+=($!)
    just -f {{justfile()}} _treewalker-java & pids+=($!)
    just -f {{justfile()}} _vm-java & pids+=($!)
    start=$SECONDS
    cd tongues
    java -cp .out/java-classes TestTranspiled .out/java-classes --target test-tongues-java; rc=$?
    elapsed=$((SECONDS - start))
    if [ $rc -eq 0 ]; then
        printf '\033[32m[test-tongues-java] %ds\033[0m\n' "$elapsed"
    else
        printf '\033[31m[test-tongues-java] %ds (FAILED)\033[0m\n' "$elapsed"
        failed=1
    fi
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    exit $failed

# Compile transpiled Java to .class files
_compile-java:
    #!/usr/bin/env bash
    set -euo pipefail
    start=$SECONDS
    echo "Compiling self-transpiled Java..."
    mkdir -p tongues/.out/java-classes
    cp tongues/.out/tongues.java tongues/.out/java-classes/Main.java
    if ! javac -encoding UTF-8 tongues/.out/java-classes/Main.java -d tongues/.out/java-classes 2>&1; then
        printf '\033[31m[compile:java] %ds (FAILED)\033[0m\n' "$((SECONDS - start))"
        exit 1
    fi
    # Also compile the test harness
    if ! javac -encoding UTF-8 tongues/tests/TestTranspiled.java -d tongues/.out/java-classes 2>&1; then
        printf '\033[31m[compile:java] harness failed\033[0m\n'
        exit 1
    fi
    printf '\033[32m[compile:java] %ds\033[0m\n' "$((SECONDS - start))"

# Run taytsh app tests through the treewalker with Java binary
_treewalker-java:
    #!/usr/bin/env bash
    set -uo pipefail
    start=$SECONDS
    passed=0; failed=0
    cd tongues
    for f in tests/taytsh/app/*.ty; do
        name=$(basename "$f" .ty)
        if java -cp .out/java-classes Main taytsh "$f" >/dev/null 2>&1; then
            passed=$((passed + 1))
        else
            echo "  FAIL $name"
            failed=$((failed + 1))
        fi
    done
    elapsed=$((SECONDS - start))
    if [ $failed -eq 0 ]; then
        printf '\033[32m[tw-taytsh-apptests-java] %ds (%d passed)\033[0m\n' "$elapsed" "$passed"
    else
        printf '\033[31m[tw-taytsh-apptests-java] %ds (%d passed, %d failed)\033[0m\n' "$elapsed" "$passed" "$failed"
        exit 1
    fi

# Run taytsh app tests through the VM with Java binary
_vm-java:
    #!/usr/bin/env bash
    set -uo pipefail
    start=$SECONDS
    passed=0; failed=0
    cd tongues
    for f in tests/taytsh/app/*.ty; do
        name=$(basename "$f" .ty)
        if java -cp .out/java-classes Main taytsh --vm "$f" >/dev/null 2>&1; then
            passed=$((passed + 1))
        else
            echo "  FAIL $name"
            failed=$((failed + 1))
        fi
    done
    elapsed=$((SECONDS - start))
    if [ $failed -eq 0 ]; then
        printf '\033[32m[vm-taytsh-apptests-java] %ds (%d passed)\033[0m\n' "$elapsed" "$passed"
    else
        printf '\033[31m[vm-taytsh-apptests-java] %ds (%d passed, %d failed)\033[0m\n' "$elapsed" "$passed" "$failed"
        exit 1
    fi

# Run test suite with Java binary
_test-tongues-java:
    #!/usr/bin/env bash
    set -uo pipefail
    start=$SECONDS
    cd tongues
    java -cp .out/java-classes TestTranspiled .out/java-classes --target test-tongues-java; rc=$?
    elapsed=$((SECONDS - start))
    if [ $rc -eq 0 ]; then
        printf '\033[32m[test-tongues-java] %ds\033[0m\n' "$elapsed"
    else
        printf '\033[31m[test-tongues-java] %ds (FAILED)\033[0m\n' "$elapsed"
    fi
    exit $rc

# Run test suite through VM with Java binary (not yet implemented)
_vm-test-tongues-java:
    #!/usr/bin/env bash
    # VM tests for Java not yet implemented - would require --via-vm support in TestTranspiled.java
    printf '\033[31m[vm-test-tongues-java] not implemented\033[0m\n'
    exit 1

# Type-check transpiled Python output
_pyright:
    #!/usr/bin/env bash
    set -euo pipefail
    start=$SECONDS
    uvx pyright tongues/.out/tongues.py; rc=$?
    elapsed=$((SECONDS - start))
    if [ $rc -eq 0 ]; then
        printf '\033[32m[pyright] %ds\033[0m\n' "$elapsed"
    else
        printf '\033[31m[pyright] %ds (FAILED)\033[0m\n' "$elapsed"
    fi
    exit $rc

# Idempotence: Python binary compiles src/ to Python, verify identical output
idempotence:
    #!/usr/bin/env bash
    set -euo pipefail
    start=$SECONDS
    cd tongues
    uv run python3 tests/retranspile.py ".out/tongues.py" src python ".out/tongues-2.py"; rc=$?
    if [ $rc -ne 0 ]; then
        printf '\033[31m[idempotence] %ds (FAILED)\033[0m\n' "$((SECONDS - start))"
        exit 1
    fi
    if diff -q ".out/tongues.py" ".out/tongues-2.py" >/dev/null 2>&1; then
        rm ".out/tongues-2.py"
        printf '\033[32m[idempotence] %ds\033[0m\n' "$((SECONDS - start))"
    else
        printf '\033[31m[idempotence] %ds (FAILED)\033[0m\n' "$((SECONDS - start))"
        diff --unified=3 ".out/tongues.py" ".out/tongues-2.py" | head -30
        rm ".out/tongues-2.py"
        exit 1
    fi

# Cross-equivalence: non-Python binary compiles src/ to its own language, verify identical output
cross-equivalence target:
    #!/usr/bin/env bash
    set -euo pipefail
    declare -A ext=([ruby]=rb [perl]=pl [javascript]=js [java]=java)
    e=${ext[{{target}}]}
    start=$SECONDS
    cd tongues
    uv run python3 tests/retranspile.py ".out/tongues.$e" src {{target}} ".out/tongues-2.$e"; rc=$?
    if [ $rc -ne 0 ]; then
        printf '\033[31m[cross-equivalence-{{target}}] %ds (FAILED)\033[0m\n' "$((SECONDS - start))"
        exit 1
    fi
    if diff -q ".out/tongues.$e" ".out/tongues-2.$e" >/dev/null 2>&1; then
        rm ".out/tongues-2.$e"
        printf '\033[32m[cross-equivalence-{{target}}] %ds\033[0m\n' "$((SECONDS - start))"
    else
        printf '\033[31m[cross-equivalence-{{target}}] %ds (FAILED)\033[0m\n' "$((SECONDS - start))"
        diff --unified=3 ".out/tongues.$e" ".out/tongues-2.$e" | head -30
        rm ".out/tongues-2.$e"
        exit 1
    fi

# Run full test suite through the VM in target language
_vm-test-tongues target:
    #!/usr/bin/env bash
    set -uo pipefail
    declare -A ext=([python]=py [ruby]=rb [perl]=pl [javascript]=js)
    declare -A runner=([python]=python3 [ruby]=ruby [perl]=perl [javascript]=node)
    just -f {{justfile()}} _transpile-tongues taytsh
    start=$SECONDS
    cd tongues
    ${runner[{{target}}]} tests/test-transpiled.${ext[{{target}}]} \
        .out/tongues.${ext[{{target}}]} --via-vm .out/tongues.ty --target vm-test-tongues-{{target}}; rc=$?
    elapsed=$((SECONDS - start))
    if [ $rc -eq 0 ]; then
        printf '\033[32m[vm-test-tongues-{{target}}] %ds\033[0m\n' "$elapsed"
    else
        printf '\033[31m[vm-test-tongues-{{target}}] %ds (FAILED)\033[0m\n' "$elapsed"
    fi
    exit $rc

# Run taytsh app tests through the treewalker in target language
_treewalker target:
    #!/usr/bin/env bash
    set -uo pipefail
    declare -A ext=([python]=py [ruby]=rb [perl]=pl [javascript]=js)
    declare -A runner=([python]=python3 [ruby]=ruby [perl]=perl [javascript]=node)
    start=$SECONDS
    passed=0; failed=0
    cd tongues
    for f in tests/taytsh/app/*.ty; do
        name=$(basename "$f" .ty)
        if ${runner[{{target}}]} ".out/tongues.${ext[{{target}}]}" taytsh "$f" >/dev/null 2>&1; then
            passed=$((passed + 1))
        else
            echo "  FAIL $name"
            failed=$((failed + 1))
        fi
    done
    elapsed=$((SECONDS - start))
    if [ $failed -eq 0 ]; then
        printf '\033[32m[tw-taytsh-apptests-{{target}}] %ds (%d passed)\033[0m\n' "$elapsed" "$passed"
    else
        printf '\033[31m[tw-taytsh-apptests-{{target}}] %ds (%d passed, %d failed)\033[0m\n' "$elapsed" "$passed" "$failed"
        exit 1
    fi

# Run taytsh app tests through the VM in target language
_vm target:
    #!/usr/bin/env bash
    set -uo pipefail
    declare -A ext=([python]=py [ruby]=rb [perl]=pl [javascript]=js)
    declare -A runner=([python]=python3 [ruby]=ruby [perl]=perl [javascript]=node)
    start=$SECONDS
    passed=0; failed=0
    cd tongues
    for f in tests/taytsh/app/*.ty; do
        name=$(basename "$f" .ty)
        if ${runner[{{target}}]} ".out/tongues.${ext[{{target}}]}" taytsh --vm "$f" >/dev/null 2>&1; then
            passed=$((passed + 1))
        else
            echo "  FAIL $name"
            failed=$((failed + 1))
        fi
    done
    elapsed=$((SECONDS - start))
    if [ $failed -eq 0 ]; then
        printf '\033[32m[vm-taytsh-apptests-{{target}}] %ds (%d passed)\033[0m\n' "$elapsed" "$passed"
    else
        printf '\033[31m[vm-taytsh-apptests-{{target}}] %ds (%d passed, %d failed)\033[0m\n' "$elapsed" "$passed" "$failed"
        exit 1
    fi

# Generative / property-based tests (require pytest + hypothesis)
pytest-gen:
    #!/usr/bin/env bash
    set -uo pipefail
    pids=()
    failed=0
    just -f {{justfile()}} _pytest-gen-tycheck & pids+=($!)
    just -f {{justfile()}} _pytest-gen-softfloat & pids+=($!)
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    exit $failed

_pytest-gen-tycheck:
    #!/usr/bin/env bash
    set -uo pipefail
    start=$SECONDS
    uv run --directory tongues pytest tests/test_taytsh_gen_check.py -v; rc=$?
    elapsed=$((SECONDS - start))
    if [ $rc -eq 0 ]; then
        printf '\033[32m[pytest-gen-tycheck] %ds\033[0m\n' "$elapsed"
    else
        printf '\033[31m[pytest-gen-tycheck] %ds (FAILED)\033[0m\n' "$elapsed"
    fi
    exit $rc

_pytest-gen-softfloat:
    #!/usr/bin/env bash
    set -uo pipefail
    start=$SECONDS
    uv run --directory tongues pytest tests/test_gen_softfloat.py -v; rc=$?
    elapsed=$((SECONDS - start))
    if [ $rc -eq 0 ]; then
        printf '\033[32m[pytest-gen-softfloat] %ds\033[0m\n' "$elapsed"
    else
        printf '\033[31m[pytest-gen-softfloat] %ds (FAILED)\033[0m\n' "$elapsed"
    fi
    exit $rc

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

# Run a just target inside Docker
docker target lang:
    docker build -t tongues-{{lang}} docker/{{lang}}
    docker run --rm -v "$(pwd):/workspace" tongues-{{lang}} just -f justfile.v2 {{target}}
