# Frontend

The frontend accepts Python source and produces a Taytsh Module. It operates in sequential phases, each completing before the next starts. Phase outputs accumulate — later phases read the outputs of all prior phases. All phases are pure functions of their inputs. No phase modifies a prior phase's output.

```
source → parse → [merge] → subset → names → signatures → hierarchy → fields → pycheck → lowering → Taytsh Module
```

The merge phase runs only when the input is a directory (project mode). In single-file mode it is skipped.

## Phases

### Parse

Hand-written recursive descent parser. Tokenizes Python source and produces a dict-based AST following the structure of Python's `ast` module. Parses the full Python grammar — subset restrictions are enforced later.

### Merge (project mode only)

Gathers all `.py` files from the input directory, resolves cross-file imports, detects name collisions, and merges into a single AST.

### Subset

Early, cheap syntactic filter. Rejects constructs that are known to be untranslatable — closures, dynamic dispatch, reflection, missing type annotations, `Any` — before any type information exists. Cannot alone determine whether code is translatable; that requires the type-level checks in pycheck. Its role is to fail fast on things that definitely won't work.

### Names

Scope analysis. Builds a symbol table mapping every name to its declaration. Identifies class names, base classes, type aliases, and builds control flow graphs for each function body. Enforces the flat scoping model: module scope and function-local scope, no nesting.

### Signatures

Reads type annotations from function and method declarations and converts them to `TypeNode`s. Produces a `FuncInfo` (parameter types, return type) for every function and method. Purely mechanical — parses annotation syntax into the type representation, with no inference. Its output is what makes pycheck possible: without declared signatures, there is nothing to check call sites or return statements against.

### Hierarchy

Builds the class inheritance tree from the base class declarations that names collected. Identifies hierarchy roots (classes with children that become interfaces in the IR), node types (classes in a hierarchy), and exception types. Validates that base classes exist and that there are no cycles. Its output tells downstream phases how types relate: pycheck uses it for assignability checks and narrowing, lowering uses it to decide whether a class becomes a struct or an interface.

### Fields

Reads class bodies to extract the typed fields of each struct — their names, types, ordering, defaults, and constructor parameters. Depends on hierarchy (needs to know hierarchy roots to enforce discriminator field requirements) and signatures (needs method return types to validate field initializers). Its output completes the `TypeCollectResult` that pycheck and lowering both consume for struct field access.

### Pycheck

Bidirectional type checker. Synthesis (bottom-up) infers the type of an expression from its structure; checking (top-down) propagates an expected type into an expression from its context. This is why annotations are required at boundaries — parameters, returns, empty collections — but not at every expression: synthesis handles most cases, and checking fills in the rest. Enforces the semantic constraints that the subset's syntactic restrictions exist to enable — optionals must be narrowed before use, unions must be discriminated before member access, iterators must be eagerly consumed. Its output is a total map from every expression to its concrete type.

### Lowering

Pure translation pass from Python AST to typed Taytsh IR. Consumes pycheck's type map to make type-directed code generation decisions (operator dispatch, coercion insertion, truthiness expansion, method routing, etc.). Never infers types — only looks them up. Where Python's type semantics differ from the IR's, a thin adjustment layer translates between them (e.g. `string` → `rune` for character iteration, `bytes[i]` → `int`).

## Phase Artifacts

Each phase produces an artifact consumed by subsequent phases. No phase modifies a prior artifact.

| Phase      | Produces        | Consumed by                                                            |
| ---------- | --------------- | ---------------------------------------------------------------------- |
| parse      | dict-based AST  | merge, subset, names, signatures, fields, hierarchy, pycheck, lowering |
| merge      | merged AST      | subset onward (project mode only)                                      |
| subset     | (validation)    | —                                                                      |
| names      | NameTable, CFGs | signatures, hierarchy, fields, pycheck                                 |
| signatures | FuncInfo table  | fields, pycheck, lowering                                              |
| hierarchy  | HierarchyResult | fields, pycheck, lowering                                              |
| fields     | ClassInfo table | pycheck, lowering                                                      |
| pycheck    | expr_types map  | lowering                                                               |
| lowering   | Taytsh Module   | middleend, backends                                                    |

## Error Model

Every frontend error includes a phase tag, human-readable message, 1-indexed source line, and 0-indexed column. Errors are written to stderr. In project mode, errors from phases after merge prepend the source file path. Exit code 1 for compilation errors, exit code 2 for usage errors.
