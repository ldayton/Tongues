# Tongues Specification v2

Tongues is a transpiler for a well-behaved subset of Python. Programs in this subset are valid Python — they execute normally with CPython, work with standard IDEs, debuggers, test frameworks, and tooling. The subset trades dynamic flexibility for compile-time guarantees: all types are known, all calls resolve statically, all ownership is inferrable. This enables generation of idiomatic output in multiple target languages from a single source, all without shipping a runtime dependency.

The frontend parses Python, checks the subset, resolves names and types, then lowers to Taytsh — a statically-typed, target-neutral intermediate language. The middleend annotates the IR with analysis results. Backends walk the annotated IR and emit target language source.

## Design Principles

- **Self-hosting**: the compiler is written in the subset it accepts
- **Append-only**: the lowerer produces IR; middleend passes add annotations but never modify the tree; backends only read
- **Closed-world**: single-file compilation; all types, implementations, and call targets are visible

## Supported Output Languages

| Language   | Min Version  |
| ---------- | ------------ |
| C          | GCC 13       |
| C#         | .NET 8       |
| Dart       | Dart 3.2     |
| Go         | Go 1.21      |
| Java       | Temurin 21   |
| JavaScript | Node.js 21   |
| Lua        | Lua 5.4      |
| Perl       | Perl 5.38    |
| PHP        | PHP 8.3      |
| Python     | CPython 3.12 |
| Ruby       | Ruby 3.2     |
| Rust       | Rust 1.75    |
| Swift      | Swift 6.0    |
| TypeScript | tsc 5.3      |
| Zig        | Zig 0.15     |

## Pipeline Overview

| Phase | Stage     | Module                   | Description                                    |
| :---: | --------- | ------------------------ | ---------------------------------------------- |
|   1   | cli       | `tongues.py`             | Parse arguments, read input, invoke pipeline   |
|   2   | frontend  | `frontend/parse.py`      | Tokenize and parse source; produce dict AST    |
|   3   | frontend  | `frontend/subset.py`     | Reject unsupported Python features             |
|   4   | frontend  | `frontend/names.py`      | Scope analysis and name binding                |
|   5   | frontend  | `frontend/signatures.py` | Type syntax parsing and kind checking          |
|   6   | frontend  | `frontend/fields.py`     | Dataflow over `__init__`; infer field types    |
|   7   | frontend  | `frontend/hierarchy.py`  | Class hierarchy; subtyping relations           |
|   8   | frontend  | `frontend/inference.py`  | Bidirectional type inference                   |
|   9   | frontend  | `frontend/lowering.py`   | Type-directed lowering to Taytsh IR            |
|  10   | middleend | `middleend/scope.py`     | Binding reassignment, constness, narrowing     |
|  11   | middleend | `middleend/returns.py`   | Return pattern analysis                        |
|  12   | middleend | `middleend/liveness.py`  | Unused values, catch vars, bindings            |
|  13   | middleend | `middleend/strings.py`   | String content classification and usage        |
|  14   | middleend | `middleend/hoisting.py`  | Variable pre-declaration, continue workarounds |
|  15   | middleend | `middleend/ownership.py` | Ownership inference and escape analysis        |
|  16   | middleend | `middleend/callgraph.py` | Throw propagation, recursion, tail calls       |
|  17   | backend   | `backend/<lang>.py`      | Emit target language source from annotated IR  |

## Type System

The frontend parses Python type annotations and maps them to Taytsh types. From phase 5 onward, all type representations use the Taytsh type system:

| Taytsh Type    | Meaning                       |
| -------------- | ----------------------------- |
| `int`          | signed integer, ≥32 bits      |
| `float`        | IEEE 754 binary64             |
| `bool`         | boolean                       |
| `byte`         | unsigned 8-bit integer        |
| `bytes`        | byte sequence                 |
| `string`       | rune sequence                 |
| `rune`         | single Unicode character      |
| `list[T]`      | ordered mutable sequence      |
| `map[K, V]`    | mutable key-value mapping     |
| `set[T]`       | mutable unique collection     |
| `(T, U, ...)`  | fixed heterogeneous tuple     |
| `T?`           | `T` or `nil`                  |
| `A \| B`       | union type                    |
| `fn[T..., R]`  | function type                 |
| `nil`          | null value / type             |
| struct name    | user-defined struct           |
| interface name | sealed interface              |
| enum name      | enumeration                   |
| `void`         | return-type marker (no value) |

See `12-taytsh-ir-spec.md` for the full type system, built-in functions, and grammar.

## Frontend Summary

| Module                   | Knows types? | Knows IR? | Output                             |
| ------------------------ | :----------: | :-------: | ---------------------------------- |
| `frontend/parse.py`      |      no      |    no     | dict-based AST                     |
| `frontend/subset.py`     |      no      |    no     | (rejects bad input or passes)      |
| `frontend/names.py`      |      no      |    no     | NameTable { name → kind }          |
| `frontend/signatures.py` | yes (parse)  |    yes    | SigTable { func → FuncInfo }       |
| `frontend/fields.py`     | yes (infer)  |    yes    | FieldTable { class → ClassFields } |
| `frontend/hierarchy.py`  |  yes (sub)   |    no     | SubtypeRel { root, node types }    |
| `frontend/inference.py`  | yes (bidir)  |    no     | TypedAST (annotated dict-AST)      |
| `frontend/lowering.py`   | yes (narrow) |    yes    | Taytsh IR Module                   |

## Middleend Summary

Read-only analysis passes that annotate Taytsh IR nodes. No transformations — just computing properties needed for code generation. Annotations are namespaced per pass (`scope.*`, `returns.*`, etc.) and write-once.

| Module                   | Depends on     | Annotations produced                                                                                                                                                          |
| ------------------------ | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `middleend/scope.py`     | —              | `scope.is_reassigned`, `scope.is_const`, `scope.is_modified`, `scope.is_unused`, `scope.narrowed_type`, `scope.is_interface`, `scope.is_function_ref`, `scope.case_interface` |
| `middleend/returns.py`   | —              | `returns.always_returns`, `returns.needs_named_returns`, `returns.may_return_nil`, `returns.body_has_return`                                                                  |
| `middleend/liveness.py`  | scope          | `liveness.initial_value_unused`, `liveness.catch_var_unused`, `liveness.match_var_unused`, `liveness.tuple_unused_indices`                                                    |
| `middleend/strings.py`   | scope,liveness | `strings.content`, `strings.indexed`, `strings.iterated`, `strings.len_called`, `strings.builder`                                                                             |
| `middleend/hoisting.py`  | scope,strings? | `hoisting.hoisted_vars`, `hoisting.func_hoisted_vars`, `hoisting.has_continue`, `hoisting.has_break`, `hoisting.rune_vars`                                                    |
| `middleend/ownership.py` | scope,liveness | `ownership.kind`, `ownership.escapes`, `ownership.region`                                                                                                                     |
| `middleend/callgraph.py` | —              | `callgraph.throws`, `callgraph.is_recursive`, `callgraph.recursive_group`, `callgraph.is_tail_call`                                                                           |

See `11-MIDDLEEND-SPEC-v2.md` for pass ordering, dependency graph, and target-conditional execution.
