# Tongues Frontend v2

The frontend accepts Python source and produces a Taytsh Module. It operates in nine sequential phases: accept input via the CLI, parse the source into a dict-based AST, enforce subset restrictions, resolve names, collect signatures, infer field types, build the class hierarchy, run bidirectional type inference, and lower the typed AST to Taytsh IR. Each phase completes before the next starts. Phase outputs accumulate — later phases read the outputs of all prior phases.

## Pipeline

```
source.py → parse → subset → names → signatures → fields → hierarchy → inference → lowering → Taytsh Module
```

All phases are pure functions of their inputs. No phase modifies a prior phase's output. The pipeline is deterministic — same input produces same output.

## Phase Summaries

### 1. CLI (`tongues.py`)

Program entry point. Parses command-line arguments, reads source from stdin, selects the target language, invokes the pipeline, writes output to stdout. Written in the Tongues subset.

### 2. Parse (`frontend/parse.py`)

Hand-written recursive descent parser. Tokenizes Python source and produces a dict-based AST following the structure of Python's `ast` module. Parses the full Python grammar — subset restrictions are enforced in phase 3.

### 3. Subset (`frontend/subset.py`)

Syntactic gate. Walks the AST and rejects Python constructs that have no Taytsh equivalent: nested functions, closures, dynamic attribute access, bare collection types, `*args`/`**kwargs`, and others. What passes this phase is guaranteed to be translatable.

### 4. Names (`frontend/names.py`)

Scope analysis. Builds a symbol table mapping every name to its declaration. Enforces the flat scoping model that Taytsh requires: module scope and function-local scope, no nesting, no `global`/`nonlocal`.

### 5. Signatures (`frontend/signatures.py`)

Type syntax parsing. Collects function and method signatures, translates Python type annotations into Taytsh types, validates type constructor arity. After this phase, every function has a typed parameter list and return type in the Taytsh type system.

### 6. Fields (`frontend/fields.py`)

Field type inference. Analyzes class bodies and `__init__` methods to determine struct field types, default values, constructor parameter order, and discriminator fields for tagged unions.

### 7. Hierarchy (`frontend/hierarchy.py`)

Inheritance analysis. Builds ancestor chains, classifies hierarchy roots as Taytsh interfaces and their subclasses as implementing structs, computes least upper bound for union types.

### 8. Inference (`frontend/inference.py`)

Bidirectional type inference with flow-sensitive narrowing. Computes Taytsh types for all expressions, infers local variable types, enforces type safety constraints, resolves optional/nil semantics, and validates iterator consumption.

### 9. Lowering (`frontend/lowering.py`)

Translation to Taytsh IR. Walks the typed AST and emits a Taytsh Module: structs, interfaces, enums, functions, and constants. Resolves Python-specific semantics (floor division, value-returning `and`/`or`, negative indexing, truthiness) into Taytsh forms. Attaches provenance annotations for backends that want to reconstruct idiomatic source patterns.

## Phase Artifacts

Each phase produces an artifact consumed by subsequent phases. No phase modifies a prior artifact.

| Phase      | Produces       | Consumed by                                                       |
| ---------- | -------------- | ----------------------------------------------------------------- |
| parse      | dict-based AST | subset, names, signatures, fields, hierarchy, inference, lowering |
| subset     | (validation)   | —                                                                 |
| names      | NameTable      | signatures, fields, inference                                     |
| signatures | SigTable       | fields, inference, lowering                                       |
| fields     | FieldTable     | inference, lowering                                               |
| hierarchy  | SubtypeRel     | inference, lowering                                               |
| inference  | TypedAST       | lowering                                                          |
| lowering   | Taytsh Module  | middleend, backends                                               |

## Concept Map

Python source constructs map to Taytsh IR constructs during lowering:

| Python                     | Taytsh                           |
| -------------------------- | -------------------------------- |
| `class Foo:`               | `struct Foo { ... }`             |
| `class Foo(Base):`         | `struct Foo : Base { ... }`      |
| hierarchy root class       | `interface Name {}`              |
| `def f(x: int) -> str:`    | `fn F(x: int) -> string { ... }` |
| `def m(self, x: int):`     | method inside struct             |
| `int`                      | `int`                            |
| `float`                    | `float`                          |
| `bool`                     | `bool`                           |
| `str`                      | `string`                         |
| `bytes`                    | `bytes`                          |
| `list[T]`                  | `list[T]`                        |
| `dict[K, V]`               | `map[K, V]`                      |
| `set[T]`                   | `set[T]`                         |
| `tuple[A, B]`              | `(A, B)`                         |
| `T \| None`, `Optional[T]` | `T?`                             |
| `A \| B`                   | `A \| B`                         |
| `Callable[[A], R]`         | `fn[A, R]`                       |
| `None`                     | `nil`                            |
| `if`/`elif`/`else`         | `if`/`else if`/`else`            |
| `for x in xs:`             | `for x in xs { ... }`            |
| `for i in range(n):`       | `for i in range(n) { ... }`      |
| `while cond:`              | `while cond { ... }`             |
| `match`/`case`             | `match`/`case`                   |
| `try`/`except`/`finally`   | `try`/`catch`/`finally`          |
| `raise E(msg)`             | `throw E(msg)`                   |
| `assert cond`              | `Assert(cond)`                   |
| `print(x)`                 | `WritelnOut(ToString(x))`        |
| `len(xs)`                  | `Len(xs)`                        |
| `xs.append(v)`             | `Append(xs, v)`                  |
| `d.get(k)`                 | `Get(d, k)`                      |
| `f"{x}: {y}"`              | `Format("{}: {}", x, y)`         |
| `isinstance(x, T)`         | `match` case                     |
| `x is None`                | `x == nil`                       |
| `x[-1]`                    | `x[Len(x) - 1]`                  |
| `xs[:n]`                   | `xs[0:n]`                        |
| `a < b < c`                | `a < b && b < c`                 |
| `[x*2 for x in xs]`        | for loop + `Append`              |

## Error Model

Every frontend error includes:

| Field   | Content                          |
| ------- | -------------------------------- |
| phase   | which phase detected the error   |
| message | human-readable diagnostic        |
| line    | 1-indexed source line number     |
| column  | 0-indexed column within the line |

Errors are written to stderr in the format `error: {message} at line {line}`. The pipeline halts at the first error — no attempt to recover or report multiple errors.

Exit code 1 for compilation errors (parse, subset, type, lowering). Exit code 2 for usage errors (unknown flag, unknown target).
