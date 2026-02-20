# Tongues

Python-to-native transpiler for leaf dependencies (parsers, codecs, validators, data structures).

## Target Languages

c, csharp, dart, go, java, javascript, lua, perl, php, python, ruby, rust, swift, typescript, zig

JavaScript and TypeScript share most code in `jslike.py` and should be worked on together.

## Running Tests

### Local (requires matching runtime versions)

```bash
just test-backend-local            # codegen + apptests
just test-local                    # all stages
```

### Docker

```bash
just test-backend                  # codegen + apptests
just test                          # backend + taytsh
```

### Individual test stages

Each stage has `-local` and Docker variants:

```bash
just test-cli-local                # CLI argument handling
just test-parse-local              # parser
just test-subset-local             # subset compliance
just test-names-local              # name resolution
just test-signatures-local         # signature extraction
just test-fields-local             # field analysis
just test-hierarchy-local          # type hierarchy
just test-inference-local          # type inference
just test-lowering-local           # Python → Taytsh lowering
just test-middleend-local          # type checking, scope, returns, liveness, strings, hoisting, ownership, callgraph
just test-backend-local            # codegen + apptests
just test-taytsh-local             # Taytsh parser, checker, apptests
```

### Check local runtime versions

```bash
just versions
```

## Pytest Flags

```bash
uv run --directory tongues pytest tests/test_runner.py [OPTIONS]
```

| Flag              | Description                              |
| ----------------- | ---------------------------------------- |
| `--target <lang>` | Run only specified target(s), repeatable |
| `-k <pattern>`    | Filter tests by name pattern             |

## Other Commands

```bash
just lint              # ruff check
just lint --fix        # ruff check --fix
just fmt               # ruff format --check
just fmt --fix         # ruff format
just subset            # verify transpiler source is subset-compliant
just check-local       # full check locally (fmt, lint, subset, all tests)
just self-transpile    # emit self-transpiled Python to .out/
```
