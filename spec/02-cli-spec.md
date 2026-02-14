# Phase 1: CLI

**Module:** `tongues.py`

Program entry point. Parses command-line arguments, reads source input, invokes the compilation pipeline, writes output. Written in the Tongues subset — uses only allowed I/O primitives (stdin/stdout/stderr, no file I/O).

## Usage

```
tongues [OPTIONS] < input.py

Options:
  --target TARGET     Output language (see Target Selection)
  --stop-at PHASE     Stop after phase (see Phase Control)
  --strict            Enable strict math and strict tostring
  --strict-math       Enable strict math mode
  --strict-tostring   Enable strict tostring mode
  --help              Show this help message
```

File redirection is handled by the shell; the transpiler itself only uses stdin/stdout.

## Target Selection

The `--target` flag selects the output language. Default: `go`.

| Target     | Flag value   |
| ---------- | ------------ |
| C          | `c`          |
| C#         | `csharp`     |
| Dart       | `dart`       |
| Go         | `go`         |
| Java       | `java`       |
| JavaScript | `javascript` |
| Lua        | `lua`        |
| Perl       | `perl`       |
| PHP        | `php`        |
| Python     | `python`     |
| Ruby       | `ruby`       |
| Rust       | `rust`       |
| Swift      | `swift`      |
| TypeScript | `typescript` |
| Zig        | `zig`        |

## Phase Control

The `--stop-at` flag halts the pipeline after the named phase and outputs the intermediate representation:

| Phase        | Output Format             |
| ------------ | ------------------------- |
| `parse`      | JSON: dict-based AST      |
| `subset`     | Nothing (validation only) |
| `names`      | JSON: name table          |
| `signatures` | JSON: signature table     |
| `fields`     | JSON: field table         |
| `hierarchy`  | JSON: subtype relations   |
| `inference`  | JSON: typed AST           |
| `lowering`   | JSON: Taytsh IR Module    |
| `analyze`    | JSON: annotated Taytsh IR |
| (omitted)    | Target language source    |

`--stop-at` halts after the named phase completes successfully. If any phase fails, the pipeline stops at the failure point regardless of `--stop-at`.

## Strict Mode

Three flags control cross-target consistency. Strict flags are stored on the Taytsh Module node. See `12-taytsh-ir-spec.md` for full semantics.

| Flag                | Effect                                                      |
| ------------------- | ----------------------------------------------------------- |
| `--strict-math`     | Bit-identical arithmetic across targets (all 15 targets)    |
| `--strict-tostring` | Canonical `ToString` format across targets (all 15 targets) |
| `--strict`          | Enables both (all 15 targets)                               |

Strict mode can also be set within source files via pragmas:

```
@@["strict_math", "strict_tostring"]
```

Pragmas must appear before any declarations and are equivalent to the corresponding flags.

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
