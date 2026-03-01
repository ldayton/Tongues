# Middleend

The middleend accepts a Taytsh Module from the frontend and annotates it with analysis results that backends need for idiomatic code generation. It does not transform the IR — the tree structure is unchanged on exit. Every pass reads the IR (and possibly annotations from prior passes) and writes new annotations under a namespaced key prefix. The annotation map is monotonically increasing over an information ordering — each pass can only add information, never retract it. This guarantees that independent passes commute and that analysis results are stable regardless of pass ordering.

Passes are intra-procedural unless noted otherwise. Each pass processes every function and method independently.

```
Taytsh Module → type checking → scope → returns → liveness → strings → hoisting → ownership → callgraph → annotated Module
```

Some passes are conditional — they run only when the target set includes languages that need their output. The four unconditional passes (type checking, scope, returns, liveness) always run.

## Passes

### Type Checking

Validates the Taytsh IR against the type system defined in the Taytsh spec. Enforces type safety, exhaustive pattern matching, no-closure invariants, no-shadowing rules, and nil safety. This is a standalone validator for Taytsh itself — independent of the frontend's pycheck, which validates Python source. A Taytsh program produced by hand or by a tool other than the Tongues frontend must still pass this checker.

Produces no annotations. Rejects invalid programs with diagnostics.

### Scope

Intra-procedural binding analysis. Walks each function body and classifies every variable binding: whether it is reassigned, whether it is effectively constant, whether parameters are mutated or unused. Tracks type narrowing at use sites — after `!= nil` checks, inside `match` arms, after guard returns — recording the narrowed type so backends can emit precise types. Detects interface usage in type-switch cases (needed for Go's type-switch emission).

| Annotation              | Type   | On         | Meaning                                                 |
| ----------------------- | ------ | ---------- | ------------------------------------------------------- |
| `scope.is_reassigned`   | bool   | let        | binding is assigned more than once                      |
| `scope.is_const`        | bool   | let        | binding is never reassigned                             |
| `scope.is_modified`     | bool   | param      | parameter is mutated (field/index write, mutating call) |
| `scope.is_unused`       | bool   | param      | parameter is never read                                 |
| `scope.narrowed_type`   | string | ident use  | type after narrowing at this use site                   |
| `scope.is_interface`    | bool   | ident use  | value is consumed as an interface type                  |
| `scope.is_function_ref` | bool   | ident use  | value is used as a function reference                   |
| `scope.case_interface`  | string | match case | which interface this case binding is consumed as        |

### Returns

Intra-procedural control flow analysis. Identifies blocks that always terminate (via `return`, `throw`, or `Exit()`), functions that may return nil, and try/catch structures that need named returns for Go. Composite rules: an `if`/`else` always-returns iff both branches do; a `match` always-returns iff all cases do; loops conservatively never always-return.

| Annotation                    | Type | On    | Meaning                                     |
| ----------------------------- | ---- | ----- | ------------------------------------------- |
| `returns.always_returns`      | bool | block | every path through this block terminates    |
| `returns.needs_named_returns` | bool | fn    | Go needs named return variables             |
| `returns.may_return_nil`      | bool | fn    | at least one path returns a nil-typed value |
| `returns.body_has_return`     | bool | try   | try body contains a return statement        |

### Liveness

Backward must-analysis. Walks each function body from exits to entries, identifying bindings whose initial value is overwritten on all paths before any read — letting backends skip the initializer. Also identifies unused catch bindings, unused match bindings, and unused tuple assignment targets. Conservative: loops are assumed to execute zero times, so an assignment inside a loop does not kill the initial value.

| Annotation                      | Type   | On           | Meaning                                         |
| ------------------------------- | ------ | ------------ | ----------------------------------------------- |
| `liveness.initial_value_unused` | bool   | let          | initial value is dead (overwritten before read) |
| `liveness.catch_var_unused`     | bool   | catch        | catch binding is never read                     |
| `liveness.match_var_unused`     | bool   | case         | match binding is never read                     |
| `liveness.tuple_unused_indices` | string | tuple assign | comma-separated indices of unused targets       |

### Strings (conditional)

Intra-procedural string classification. Runs when the target set includes languages whose native string encoding is not rune-indexed (Go, Rust, C, Zig, Java, C#, JavaScript, TypeScript, Dart, Swift, Lua, PHP). Classifies every string binding's content (ASCII, BMP, or unknown) and tracks how it is used (indexed, iterated, length-checked). Detects the string builder pattern — a loop accumulating via `s = Concat(s, expr)` — so backends can emit `StringBuilder`/`strings.Builder`/etc.

Content classification forms a three-element lattice: `ascii ⊑ bmp ⊑ unknown`. Propagation through operations is the lattice join: concatenation of two ASCII strings is `ascii`; concatenation of ASCII and BMP is `bmp`. Every string operation is monotone — no operation can narrow the content class, only widen or preserve it.

| Annotation           | Type   | On       | Meaning                                    |
| -------------------- | ------ | -------- | ------------------------------------------ |
| `strings.content`    | string | let      | `"ascii"`, `"bmp"`, or `"unknown"`         |
| `strings.indexed`    | bool   | let      | string is used in index or slice position  |
| `strings.iterated`   | bool   | let      | string is iterated with `for`              |
| `strings.len_called` | bool   | let      | `Len()` is called on this string           |
| `strings.builder`    | string | for loop | comma-separated accumulator names, or `""` |

Depends on: `scope.*`, `liveness.*`.

### Hoisting (conditional)

Intra-procedural analysis solving three target-specific code generation problems. Detects `continue` inside loop bodies (Lua needs a transformation). Detects `break` inside match cases (C# switch-in-loop workaround). Collects variables that hold runes (Go needs explicit `rune` type declarations for string indexing).

| Annotation              | Type   | On             | Meaning                                              |
| ----------------------- | ------ | -------------- | ---------------------------------------------------- |
| `hoisting.hoisted_vars` | string | control struct | variables declared inside but used after, with types |
| `hoisting.has_continue` | bool   | loop           | body contains `continue`                             |
| `hoisting.has_break`    | bool   | match          | case body contains `break`                           |
| `hoisting.rune_vars`    | string | fn             | comma-separated variable names, or `""`              |

Can use `strings.indexed` when available; otherwise performs its own string-indexing detection.

### Ownership (conditional)

Intra-procedural ownership analysis. Runs when the target set includes languages with manual memory management (C, Rust, Zig, Swift). Classifies every binding as owned (this scope frees), borrowed (owner elsewhere), or shared (ref-counted). Detects value escapes — assignments to struct fields, collection mutations, returns, throws — that force a value into shared ownership.

| Annotation          | Type   | On   | Meaning                                       |
| ------------------- | ------ | ---- | --------------------------------------------- |
| `ownership.kind`    | string | let  | `"owned"`, `"borrowed"`, or `"shared"`        |
| `ownership.region`  | string | let  | opaque scope identifier for cleanup placement |
| `ownership.escapes` | bool   | expr | value escapes its declaration scope           |

Depends on: `scope.*`, `liveness.*`.

### Call Graph (inter-procedural)

The only inter-procedural pass. Builds a call graph across all functions and methods in the module, then computes the least fixed point of a monotone function on the powerset lattice of exception types: `throw` and calls are joins (add types to the set), `try`/`catch` is a meet (removes types). Processing SCCs in reverse topological order guarantees convergence. Also detects direct and mutual recursion via SCC analysis and identifies tail calls.

| Annotation                  | Type   | On        | Meaning                                          |
| --------------------------- | ------ | --------- | ------------------------------------------------ |
| `callgraph.throws`          | string | fn        | semicolon-separated exception type names, sorted |
| `callgraph.is_recursive`    | bool   | fn        | function is directly or mutually recursive       |
| `callgraph.recursive_group` | string | fn        | opaque SCC identifier for mutual recursion       |
| `callgraph.is_tail_call`    | bool   | call site | call is in tail position                         |

Throw sources include explicit `throw`, built-in operations that can fail (indexing, division, parsing), and strict-math traps. `try`/`catch` filters throws: a typed catch removes its type, a catch-all removes everything. Tail position is defined inductively; try bodies and loops are never tail.

## Annotation Conventions

All annotations are keyed by pass name prefix (`scope.`, `returns.`, `liveness.`, `strings.`, `hoisting.`, `ownership.`, `callgraph.`). Values are `bool`, `int`, `string`, or `(int, int)`. Backends read annotations but never write them.

## Pass Dependencies

```
type checking  (independent)
scope          (independent)
returns        (independent)
liveness       (reads scope.is_unused)
strings        (reads scope.*, liveness.*)
hoisting       (reads strings.indexed if available)
ownership      (reads scope.*, liveness.*)
callgraph      (independent)
```

Passes without dependencies can run in any order relative to each other. In practice the pipeline runs sequentially in the order listed.
