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

| Phase        | Output                                |
| ------------ | ------------------------------------- |
| `parse`      | Dict-based AST as JSON                |
| `subset`     | Exit 0 if valid, exit 1 with errors   |
| `names`      | NameTable as JSON                     |
| `signatures` | SigTable as JSON                      |
| `fields`     | FieldTable as JSON                    |
| `hierarchy`  | SubtypeRel as JSON                    |
| `inference`  | TypedAST as JSON                      |
| `lowering`   | IR Module as JSON                     |
| `analyze`    | Annotated IR as JSON                  |
| (omitted)    | Full transpile to `--target` language |

This enables testing each phase independently and introspecting intermediate representations.

## Skip Directive

Files containing `tongues: skip` in the first 5 lines are skipped during `--stop-at subset`.

## Postconditions

Source read from stdin; target language selected; pipeline invoked; output written to stdout or error reported to stderr with non-zero exit.
