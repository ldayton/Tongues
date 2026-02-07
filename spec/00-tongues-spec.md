# Tongues Specification

Tongues is a transpiler for a well-behaved subset of Python. Programs in this subset are valid Python—they execute normally with CPython, work with standard IDEs, debuggers, test frameworks, and tooling. The subset trades dynamic flexibility for compile-time guarantees: all types are known, all calls resolve statically, all ownership is inferrable. This enables generation of idiomatic output in multiple target languages from a single source —- all without shipping a runtime dependency.

Sequential pipeline with clean phase boundaries. Each phase completes before the next starts.

## Design Principles

- **Sequential**: Each phase completes fully before the next starts
- **Monotonic**: Phase N reads outputs of phases 1..N-1; invariants accumulate
- **Fail fast**: Reject bad input at the earliest possible phase
- **Self-hosting**: Tongues can transpile itself; the compiler is written in the subset it accepts

## Supported Output Languages

Tongues supports these target languages:

| Language   | Min Version  |
| ---------- | ------------ |
| C          | GCC 13       |
| C#         | .NET 8       |
| Dart       | Dart 3.2     |
| Go         | Go 1.21      |
| Java       | Temurin 21   |
| Javascript | Node.js 21   |
| Lua        | Lua 5.4      |
| Perl       | Perl 5.38    |
| PHP        | PHP 8.3      |
| Python     | CPython 3.12 |
| Ruby       | Ruby 3.2     |
| Rust       | Rust 1.75    |
| Swift      | Swift 6.0    |
| Typescript | tsc 5.3      |
| Zig        | Zig 0.14     |

## Pipeline Overview

| Phase | Stage     | Module          | Description                                         |
| :---: | --------- | --------------- | --------------------------------------------------- |
|   1   | cli       | `cli.py`        | Parse arguments, read input, invoke pipeline        |
|   2   | frontend  | `parse.py`      | Tokenize and parse source; produce dict-based AST   |
|   3   | frontend  | `subset.py`     | Reject unsupported Python features early            |
|   4   | frontend  | `names.py`      | Scope analysis and name binding                     |
|   5   | frontend  | `signatures.py` | Type syntax parsing and kind checking               |
|   6   | frontend  | `fields.py`     | Dataflow over `__init__`; infer field types         |
|   7   | frontend  | `hierarchy.py`  | Class hierarchy; subtyping relations                |
|   8   | frontend  | `inference.py`  | Bidirectional type inference (↑synth / ↓check)      |
|   9   | frontend  | `lowering.py`   | Type-directed elaboration to IR                     |
|  10   | middleend | `scope.py`      | Variable declarations, reassignments, modifications |
|  11   | middleend | `returns.py`    | Return pattern analysis                             |
|  12   | middleend | `liveness.py`   | Unused values, catch vars, bindings                 |
|  13   | middleend | `hoisting.py`   | Variables needing hoisting for Go emission          |
|  14   | middleend | `ownership.py`  | Ownership inference and escape analysis             |
|  15   | backend   | `<lang>.py`     | Emit target language source from annotated IR       |

## Frontend Summary

| Module          | Knows types? | Knows IR? | Output                                |
| --------------- | :----------: | :-------: | ------------------------------------- |
| `parse.py`      |      no      |    no     | dict-based AST                        |
| `subset.py`     |      no      |    no     | (rejects bad input or passes through) |
| `names.py`      |      no      |    no     | NameTable { name → kind }             |
| `signatures.py` | yes (parse)  |    no     | SigTable { func → (params, ret) }     |
| `fields.py`     | yes (infer)  |    no     | FieldTable { class → [(name, type)] } |
| `hierarchy.py`  |  yes (sub)   |    no     | SubtypeRel { class → ancestors }      |
| `inference.py`  | yes (bidir)  |    no     | TypedAST (↑synth / ↓check / narrow)   |
| `lowering.py`   | yes (narrow) |    yes    | IR Module                             |

## Middleend Summary

Read-only analysis passes that annotate IR nodes in place. No transformations—just computing properties needed for code generation.

| Module         | Depends on     | Annotations added                                                                              |
| -------------- | -------------- | ---------------------------------------------------------------------------------------------- |
| `scope.py`     | —              | `is_reassigned`, `is_modified`, `is_unused`, `is_declaration`, `is_interface`, `narrowed_type` |
| `returns.py`   | —              | `needs_named_returns`                                                                          |
| `liveness.py`  | scope, returns | `initial_value_unused`, `catch_var_unused`, `binding_unused`                                   |
| `hoisting.py`  | scope, returns | `hoisted_vars`, `rune_vars`                                                                    |
| `ownership.py` | scope          | `ownership`, `region`, `escapes`                                                               |
