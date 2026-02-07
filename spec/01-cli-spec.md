# Phase 1: CLI

**Module:** `cli.py`

Program entry point. Parses command-line arguments, reads source input, invokes the compilation pipeline, writes output. Written in the Tongues subset—uses only allowed I/O primitives (stdin/stdout/stderr, no file I/O).

| Responsibility      | Implementation                                                       |
| ------------------- | -------------------------------------------------------------------- |
| Argument parsing    | Manual `sys.argv` processing (no argparse)                           |
| Source input        | `sys.stdin.read()`                                                   |
| Target selection    | `--target` flag: `go`, `rust`, `c`, `java`, `py`                     |
| Output              | `print()` to stdout                                                  |
| Error reporting     | `print(..., file=sys.stderr)` with exit code                         |
| Pipeline invocation | Call `frontend.compile()` → `middleend.analyze()` → `backend.emit()` |

## Usage

```
tongues [OPTIONS] < input.py > output.go

Options:
  --target TARGET   Output language: go, rust, c, java, py (default: go)
  --verify          Check subset compliance only, no codegen
  --help            Show this help message
```

File redirection is handled by the shell; the transpiler itself only uses stdin/stdout.

## Postconditions

Source read from stdin; target language selected; pipeline invoked; output written to stdout or error reported to stderr with non-zero exit.
