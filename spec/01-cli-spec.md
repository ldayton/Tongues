# Phase 1: CLI

**Module:** `tongues.py`

Program entry point. Parses command-line arguments, reads source input, invokes the compilation pipeline, writes output. Written in the Tongues subset—uses only allowed I/O primitives (stdin/stdout/stderr, no file I/O).

| Responsibility      | Implementation                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------- |
| Argument parsing    | Manual `sys.argv` processing (no argparse)                                                                    |
| Source input        | `sys.stdin.read()`                                                                                            |
| Target selection    | `--target` flag (15 languages, default: go)                                                                   |
| Phase control       | `--stop-at` flag to halt pipeline at any phase                                                                |
| Output              | `print()` to stdout                                                                                           |
| Error reporting     | `print(..., file=sys.stderr)` with exit code                                                                  |
| Pipeline invocation | `parse()` → `verify_subset()` → `resolve_names()` → `Frontend().transpile()` → `analyze()` → `backend.emit()` |

## Usage

```
tongues [OPTIONS] < input.py

Options:
  --target TARGET   Output language: c, csharp, dart, go, java, javascript,
                    lua, perl, php, python, ruby, rust, swift, typescript, zig
  --stop-at PHASE   Stop after phase (see below)
  --help            Show this help message
```

File redirection is handled by the shell; the transpiler itself only uses stdin/stdout.

## Phase Control

The `--stop-at` flag halts the pipeline after the specified phase and outputs the intermediate representation:

| Phase        | Output Format                                  |
| ------------ | ---------------------------------------------- |
| `parse`      | JSON: `{"type": "Module", "body": [...]}`      |
| `subset`     | Nothing (validation only)                      |
| `names`      | JSON: `{"names": {...}, "scopes": [...]}`      |
| `signatures` | JSON: `{"functions": {...}, "methods": {...}}` |
| `fields`     | JSON: `{"classes": {...}}`                     |
| `hierarchy`  | JSON: `{"ancestors": {...}, "root": ...}`      |
| `inference`  | JSON: typed AST                                |
| `lowering`   | JSON: IR module                                |
| `analyze`    | JSON: annotated IR                             |
| (omitted)    | Target language source code                    |

`--stop-at` halts after the named phase completes successfully. If any phase fails, the pipeline stops at the failure point regardless of `--stop-at`:

| Scenario                           | Behavior                      |
| ---------------------------------- | ----------------------------- |
| `--stop-at inference`, parse fails | Exit 1, parse error to stderr |
| `--stop-at subset`, subset passes  | Exit 0, no output             |
| `--stop-at parse`, parse succeeds  | Exit 0, AST JSON to stdout    |

This enables testing each phase independently and introspecting intermediate representations.

## Skip Directive

Files containing `tongues: skip` in the first 5 lines are skipped during `--stop-at subset`.

## Exit Codes

| Code | Meaning                                                               |
| ---- | --------------------------------------------------------------------- |
| 0    | Success                                                               |
| 1    | Compilation error (parse failure, subset violation, type error, etc.) |
| 2    | Usage error (unknown flag, unknown target, unknown phase)             |

## Input Handling

| Input               | Behavior                                |
| ------------------- | --------------------------------------- |
| Zero bytes          | error: `no input provided`, exit 2      |
| Whitespace only     | Passed to parser (may fail at parse)    |
| Invalid UTF-8       | error: `invalid utf-8 in input`, exit 1 |
| No trailing newline | Accepted                                |

## Errors

| Condition      | Diagnostic                      | Exit |
| -------------- | ------------------------------- | ---- |
| Unknown target | error: `unknown target 'foo'`   | 2    |
| Unknown phase  | error: `unknown phase 'foo'`    | 2    |
| Unknown flag   | error: `unknown flag 'foo'`     | 2    |
| Empty input    | error: `no input provided`      | 2    |
| Invalid UTF-8  | error: `invalid utf-8 in input` | 1    |

## Postconditions

Source read from stdin; target language selected; pipeline invoked; output written to stdout or error reported to stderr with non-zero exit.
