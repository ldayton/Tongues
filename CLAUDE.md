# Tongues

Python-to-native transpiler for leaf dependencies (parsers, codecs, validators, data structures).

## Target Languages

c, csharp, dart, go, java, javascript, lua, perl, php, python, ruby, rust, swift, typescript, zig

JavaScript and TypeScript share most code in `jslike.py` and should be worked on together.

## Running Tests

### Local (requires matching runtime versions)

```bash
# Codegen tests (output correctness)
just test-codegen-local

# Apptests (end-to-end execution)
just test-apptests-local           # all languages
just test-apptests-local python    # single language
```

### Docker

```bash
just test-codegen                  # codegen tests
just test-apptests python          # single language
just test                          # all languages
```

### Check local runtime versions

```bash
just versions
```

## Pytest Flags

Run pytest directly for finer control:

```bash
uv run --directory tongues pytest ../tests/test_apptests.py [OPTIONS]
```

| Flag               | Description                                            |
| ------------------ | ------------------------------------------------------ |
| `--target <lang>`  | Run only specified target(s), repeatable               |
| `--ignore-version` | Skip version checks, use whatever runtime is available |
| `--ignore-skips`   | Run tests in the known-failure skip list               |
| `--summary`        | Print a summary table of apptest pass/fail counts      |

### Language Test Summary

To get a summary table of apptest status for specific languages:

```bash
uv run --directory tongues pytest ../tests/test_apptests.py --target javascript --target typescript --target ruby --ignore-skips --summary
```

## CI

CI runs `just check` (fmt, lint, subset, test-codegen, then test-apptests for all languages).

## Other Commands

```bash
just lint              # ruff check
just lint --fix        # ruff check --fix
just fmt               # ruff format --check
just fmt --fix         # ruff format
just subset            # verify transpiler source is subset-compliant
```
