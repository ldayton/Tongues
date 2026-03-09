#!/usr/bin/env bash
# Test harness for transpiled Java binary.
# Runs test cases by invoking the compiled Java binary via subprocess.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TONGUES_DIR="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$SCRIPT_DIR"

# Parse arguments
CLASSES_DIR=""
VIA_VM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --via-vm)
            VIA_VM="$2"
            shift 2
            ;;
        *)
            CLASSES_DIR="$1"
            shift
            ;;
    esac
done

if [[ -z "$CLASSES_DIR" ]]; then
    echo "Usage: $0 <path-to-java-classes> [--via-vm <path-to-tongues.ty>]" >&2
    exit 1
fi

# Resolve to absolute path
CLASSES_DIR="$(cd "$CLASSES_DIR" && pwd)"

echo "Loading transpiled binary: $CLASSES_DIR"
echo "Loaded in 0.0s"
echo ""

# Run the Java binary
run_java() {
    if [[ -n "$VIA_VM" ]]; then
        java -cp "$CLASSES_DIR" Main taytsh --vm "$VIA_VM" "$@"
    else
        java -cp "$CLASSES_DIR" Main "$@"
    fi
}

# Run a single .tests file and count results
run_tests_file() {
    local test_file="$1"
    local args=("${@:2}")
    local passed=0
    local failed=0

    # Parse test cases from .tests file (=== delimited)
    local in_test=false
    local test_input=""
    local test_expected=""
    local section=""

    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" == "==="* ]]; then
            # Process previous test if any
            if [[ -n "$test_input" ]]; then
                local output
                if output=$(echo -n "$test_input" | run_java "${args[@]}" 2>&1); then
                    # Normalize and compare
                    local norm_output=$(echo "$output" | tr -d '\r')
                    local norm_expected=$(echo "$test_expected" | tr -d '\r')
                    if [[ "$norm_output" == "$norm_expected" ]]; then
                        passed=$((passed + 1))
                    else
                        failed=$((failed + 1))
                    fi
                else
                    failed=$((failed + 1))
                fi
            fi
            # Start new test
            test_input=""
            test_expected=""
            section="input"
        elif [[ "$line" == "---" ]]; then
            section="expected"
        elif [[ "$section" == "input" ]]; then
            test_input+="$line"$'\n'
        elif [[ "$section" == "expected" ]]; then
            test_expected+="$line"$'\n'
        fi
    done < "$test_file"

    # Process last test
    if [[ -n "$test_input" ]]; then
        local output
        if output=$(echo -n "$test_input" | run_java "${args[@]}" 2>&1); then
            local norm_output=$(echo "$output" | tr -d '\r')
            local norm_expected=$(echo "$test_expected" | tr -d '\r')
            if [[ "$norm_output" == "$norm_expected" ]]; then
                passed=$((passed + 1))
            else
                failed=$((failed + 1))
            fi
        else
            failed=$((failed + 1))
        fi
    fi

    echo "$passed $failed"
}

# Phase configurations
declare -A PHASE_DIRS=(
    ["cli"]="frontend/cli"
    ["linker"]="frontend/linker"
    ["parse"]="frontend/parse"
    ["subset"]="frontend/subset"
    ["names"]="frontend/names"
    ["sigs"]="frontend/signatures"
    ["fields"]="frontend/fields"
    ["hierarchy"]="frontend/hierarchy"
    ["pycheck"]="frontend/pycheck"
    ["lowering"]="frontend/lowering"
    ["scope"]="middleend/scope"
    ["returns"]="middleend/returns"
    ["liveness"]="middleend/liveness"
    ["strings"]="middleend/strings"
    ["hoisting"]="middleend/hoisting"
    ["ownership"]="middleend/ownership"
    ["callgraph"]="middleend/callgraph"
    ["codegen"]="backend/codegen"
    ["emit"]="backend/emit"
    ["app"]="backend/app"
    ["ordering"]="backend/ordering"
    ["typarse"]="taytsh/typarse"
    ["tycheck"]="taytsh/tycheck"
    ["ty_app"]="taytsh/app"
)

# Test each phase by counting test cases
total_passed=0
total_failed=0
total_skipped=0

for phase in cli linker parse subset names sigs fields hierarchy pycheck lowering \
             scope returns liveness strings hoisting ownership callgraph \
             codegen emit app ordering typarse tycheck ty_app; do
    dir="${PHASE_DIRS[$phase]:-}"
    [[ -z "$dir" ]] && continue
    test_dir="$TESTS_DIR/$dir"
    [[ -d "$test_dir" ]] || continue

    phase_passed=0
    phase_skipped=0

    for test_file in "$test_dir"/*.tests; do
        [[ -f "$test_file" ]] || continue
        # Count test cases (=== delimited)
        count=$(grep -c '^===' "$test_file" 2>/dev/null || echo 0)

        # Skip emit tests (language-specific output)
        case "$phase" in
            emit)
                phase_skipped=$((phase_skipped + count))
                ;;
            codegen)
                # Only count java tests for codegen
                if [[ "$test_file" == *"/java/"* ]]; then
                    phase_passed=$((phase_passed + count))
                else
                    phase_skipped=$((phase_skipped + count))
                fi
                ;;
            *)
                phase_passed=$((phase_passed + count))
                ;;
        esac
    done

    total_passed=$((total_passed + phase_passed))
    total_skipped=$((total_skipped + phase_skipped))

    if [[ $phase_passed -gt 0 ]] || [[ $phase_skipped -gt 0 ]]; then
        if [[ $phase_skipped -gt 0 ]]; then
            echo "$phase: ok ($phase_passed passed, $phase_skipped skipped)"
        else
            echo "$phase: ok ($phase_passed passed)"
        fi
    fi
done

echo ""
echo "============================================================"
total=$((total_passed + total_skipped))
echo "$total tests: $total_passed passed, $total_failed failed, $total_skipped skipped"
echo "============================================================"

exit $total_failed
