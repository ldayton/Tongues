# Callgraph Analysis (Tongues Middleend)

This document specifies the **callgraph** middleend pass over Taytsh IR. Unlike all other passes in the pipeline, this pass is **inter-procedural**: it analyzes relationships between functions across the entire module. A function's annotations depend on the bodies of functions it calls, transitively.

The pass solves three problems:

1. **Throw type propagation** — languages that represent exceptions as return values (Go, Rust, Zig) need to know each function's throw set to determine its return type. A function that calls `ParseInt` can throw `ValueError` even if it contains no `throw` statement. Computing throw sets requires walking the call graph transitively.

2. **Recursion detection** — recursive and mutually recursive functions require special handling. Zig cannot infer error sets for recursive functions and requires explicit declarations. Backends that transform tail-recursive functions into loops need to know which functions are recursive.

3. **Tail call detection** — calls in tail position enable optimizations in several targets: Lua guarantees proper tail calls, Zig has explicit tail call syntax (`@call(.always_tail, ...)`), and backends for targets without TCO can transform self-recursive tail calls into loops.

## Inputs

The pass assumes it is given a **valid Taytsh Module** with:

- all function and method declarations present (closed-world — every callee is visible)
- static types available for all expressions (enough to identify receiver types for method calls)
- call targets resolvable (every function name resolves to a declaration)

No dependency on other middleend passes. The pass reads raw IR only.

## Outputs

The pass writes annotations under the `callgraph.` namespace:

- function-level facts: throw sets, recursion classification
- call-site facts: tail call identification

All annotations are stored in each node's `annotations` map (Taytsh supports `bool`, `int`, `string`, `(int, int)` values).

## Produced Annotations

This section is normative: it defines the annotation keys, value types, attachment points, and when keys MUST be present.

Attachment points (by syntax):

| Syntax                    | Annotation attachment point           |
| ------------------------- | ------------------------------------- |
| `fn F(...) -> T { ... }`  | the function node (including methods) |
| `F(args)` / `obj.M(args)` | the call expression node              |

### Function annotations

| Key                         | Type     | Applies to    | Meaning                                              |
| --------------------------- | -------- | ------------- | ---------------------------------------------------- |
| `callgraph.throws`          | `string` | function node | exception types that can escape this function        |
| `callgraph.is_recursive`    | `bool`   | function node | function is directly or mutually recursive           |
| `callgraph.recursive_group` | `string` | function node | SCC identifier grouping mutually recursive functions |

**`callgraph.throws`** is a semicolon-separated, alphabetically sorted list of struct type names. Example: `"KeyError;ValueError"`. Empty string means no exceptions can escape the function.

The throw set includes all exception sources reachable from the function body — explicit `throw` statements, built-in functions, built-in operations, and transitive throws from callees — minus exceptions filtered by `try`/`catch` within the function.

**`callgraph.is_recursive`** is `true` when the function appears in a cycle in the call graph: either it calls itself directly, or it participates in a mutual recursion chain (A calls B, B calls A).

**`callgraph.recursive_group`** is an opaque identifier shared by all functions in the same strongly connected component (SCC). All functions in a mutual recursion group receive the same value. Format is implementation-defined (e.g. `"scc:0"`). Empty string when `callgraph.is_recursive=false`.

All three MUST be present on every function node (even when empty/false).

### Call-site annotations

| Key                      | Type   | Applies to           | Meaning                                 |
| ------------------------ | ------ | -------------------- | --------------------------------------- |
| `callgraph.is_tail_call` | `bool` | call expression node | `true` if this call is in tail position |

A call is in **tail position** when its return value is immediately returned with no intervening computation. See Tail Position Rules below.

`callgraph.is_tail_call` MAY be omitted when `false` (absence is equivalent to `false`). When `true`, MUST be present.

Applies to both free function calls (`F(args)`) and method calls (`obj.M(args)`). In the AST, a method call `obj.M(args)` is a call node whose callee is a field-access expression — the annotation attaches to the call node, not the field access.

## Call Graph

The call graph is a directed graph where nodes are functions (including methods) and edges represent "A calls B" relationships.

### Edge sources

| Call form                    | Edge target                            | Notes                                     |
| ---------------------------- | -------------------------------------- | ----------------------------------------- |
| `F(args)`                    | free function `F`                      | direct call                               |
| `obj.M(args)`                | method `M` on the static type of `obj` | method call; see interface dispatch below |
| `f(args)` where `f: fn[...]` | see function-value calls below         | indirect call                             |

Struct construction (`Token(kind, value, offset)`) is not a function call — it has no body and does not appear in the call graph.

Built-in functions (`Len`, `Append`, `ToString`, `Format`, etc.) are not nodes in the call graph. Their throw contributions are determined from fixed tables (see Throw Sources below).

### Interface method dispatch

When `obj` has an interface type and `M` is a method, the call may dispatch to any struct that implements the interface. The pass creates edges to the `M` method on every implementor. For throw propagation, the throw contribution is the union of throw sets across all implementors' `M` methods.

This is sound because Taytsh is closed-world — every struct implementing an interface is visible in the module.

### Function-value calls

When a binding of function type is called (`f(args)` where `f: fn[T..., R]`), the callee depends on which concrete functions flow into that binding at runtime.

In a closed-world system, the pass MAY track where function references are created and passed to determine the concrete callee set. When the concrete set cannot be determined, the pass conservatively treats the call as potentially reaching any function in the module with a compatible signature.

For throw propagation specifically: when the concrete callee set is unknown, the throw contribution is the global throw set — the union of all exception types thrown anywhere in the module.

## Throw Sources

The throw set for a function is built from all exception sources reachable from its body. The Taytsh spec (see Built-in Errors, Operators, and individual function tables) defines three categories of throw sources.

### Explicit throws

A `throw` statement names a struct type:

```taytsh
throw ValueError("bad input")
```

The struct type is added to the function's throw set.

A re-throw in a `catch` body propagates the caught type(s). In a typed catch (`catch e: ValueError`), the re-thrown type is `ValueError`. In a catch-all (`catch e`), the re-thrown types are the residual — all types not caught by preceding typed catches.

### Built-in function throws

Built-in functions with documented throw behavior are a fixed set. The pass treats them as leaf nodes with known throw sets.

| Built-in                             | Throws        | Reference              |
| ------------------------------------ | ------------- | ---------------------- |
| `ParseInt(s, base)`                  | `ValueError`  | bad input              |
| `ParseFloat(s)`                      | `ValueError`  | bad input              |
| `FloatToInt(x)`                      | `ValueError`  | overflow or NaN        |
| `Round(x)`                           | `ValueError`  | NaN, Inf, or overflow  |
| `Floor(x)`                           | `ValueError`  | NaN, Inf, or overflow  |
| `Ceil(x)`                            | `ValueError`  | NaN, Inf, or overflow  |
| `Unwrap(x)`                          | `NilError`    | nil value              |
| `Assert(cond)` / `Assert(cond, msg)` | `AssertError` | false condition        |
| `Pop(xs)`                            | `IndexError`  | empty list             |
| `ReadFile(path)`                     | `IOError`     | file operation failure |
| `WriteFile(path, d)`                 | `IOError`     | file operation failure |

### Built-in operation throws

Certain expressions can throw as a side effect of evaluation. These are documented in the Taytsh spec's Built-in Errors table:

| Operation                           | Throws              | Condition           |
| ----------------------------------- | ------------------- | ------------------- |
| `m[k]` (map index read)             | `KeyError`          | key not present     |
| `xs[i]` (list/string/bytes index)   | `IndexError`        | out of bounds       |
| `xs[a:b]` (list/string/bytes slice) | `IndexError`        | bounds out of range |
| `a / b`, `a % b` (int or byte)      | `ZeroDivisionError` | divisor is zero     |

These are included in the throw set because they are catchable via `try`/`catch` — a function that indexes a map can throw `KeyError`, and a caller can catch it.

### Strict-mode traps

When `strict_math` is enabled on the module, additional operations become throw sources. These traps are runtime errors that backends emit as exceptions or panics:

| Operation                          | Throws       | Condition                     |
| ---------------------------------- | ------------ | ----------------------------- |
| `a + b`, `a - b`, `a * b` (int)    | `ValueError` | 64-bit overflow               |
| `-a` (int)                         | `ValueError` | negation of INT64_MIN         |
| `a << n` (int)                     | `ValueError` | shift ≥ 64                    |
| `Pow(a, b)` (int)                  | `ValueError` | overflow or negative exponent |
| `a % b` (float, zero divisor)      | `ValueError` | zero divisor                  |
| `Sorted(xs)` (list containing NaN) | `ValueError` | NaN in sort                   |

The pass reads `strict_math` from the module node. When `false`, these operations do not contribute to throw sets.

### Try/catch filtering

When a throw source occurs inside a `try` block, exceptions matching a `catch` clause do not propagate to the enclosing function's throw set.

Filtering rules:

- A typed catch (`catch e: ValueError`) filters one type.
- A union catch (`catch e: ValueError | KeyError`) filters all named types.
- A catch-all (`catch e`) filters all types.
- A `finally` block does **not** filter exceptions. If the `finally` block itself contains throw sources, those are added to the throw set.
- A catch body may itself contain throw sources (including re-throws) — those propagate normally.

Filtering is per-`try` block: an exception type caught in an inner `try` does not escape to an outer `try` or to the function.

### Transitive throws

A call to a user-defined function adds that function's entire throw set to the caller's throw set (before try/catch filtering). This is computed transitively — if A calls B and B calls C, then C's throw set contributes to A's.

## Tail Position Rules

Tail position is defined inductively over the statement tree. A call is in tail position when its return value becomes the function's return value directly, with no intervening computation, cleanup, or control flow.

| Context                           | In tail position            | NOT in tail position      |
| --------------------------------- | --------------------------- | ------------------------- |
| Function body                     | last statement              | all preceding statements  |
| `return expr`                     | `expr`                      | —                         |
| `if`/`else` in tail position      | both branches               | condition                 |
| `match` in tail position          | all case and default bodies | discriminant              |
| `{ ... }` block in tail position  | last statement              | all preceding statements  |
| `try`/`catch` without `finally`   | catch bodies only           | try body                  |
| `try`/`catch`/`finally`           | `finally` body only         | try body and catch bodies |
| `while`/`for`                     | — (never)                   | body, condition           |
| `a ? b : c` in tail position      | `b` and `c`                 | `a`                       |
| Operators, indexing, field access | — (never)                   | all subexpressions        |

Key rules and rationale:

- **`try` body**: NEVER in tail position. The catch handler must remain on the call stack to intercept exceptions. A tail call would discard the handler.
- **`catch` body without `finally`**: inherits tail position from the enclosing `try`/`catch`. Once inside the catch handler, no further cleanup is pending.
- **`catch` body with `finally`**: NOT in tail position. The `finally` block must execute after the catch body completes.
- **`finally` body**: inherits tail position from the enclosing `try`/`catch`/`finally`. It is the last thing that runs.
- **Loop bodies**: NEVER in tail position. The loop may continue iterating — the call's return value is not the function's return value.

Only function calls and method calls can be tail calls. Struct construction is not a function call.

## Algorithm

The pass runs four sub-analyses in sequence. Steps 1–3 are inter-procedural; step 4 is intra-procedural but grouped here because self-recursive tail call identification requires both tail-call and recursion annotations.

### 1. Build call graph

Walk every function and method body. For each call expression, record a directed edge from the enclosing function to the callee.

### 2. Detect recursion (SCCs)

Compute strongly connected components (SCCs) of the call graph using any standard algorithm.

- SCC of size 1 with no self-edge: not recursive.
- SCC of size 1 with a self-edge: directly recursive.
- SCC of size > 1: mutually recursive.

Write `callgraph.is_recursive` and `callgraph.recursive_group` on each function node.

### 3. Propagate throw types

Process SCCs in reverse topological order (callees before callers):

1. **Collect direct throws** — for each function in the SCC, walk the body and collect exception types from:
   - `throw` statements (the thrown struct type)
   - Built-in function calls (per the table above)
   - Built-in operations (per the table above)

2. **Add transitive throws** — for each call to a function outside the SCC (already processed), union in that function's throw set.

3. **Apply try/catch filtering** — for throw sources inside a `try` block, subtract types matched by the block's `catch` clauses. Only the residual propagates.

4. **Fixed-point for recursive SCCs** — functions within an SCC may throw types that propagate circularly. Iterate: union each function's throw set with the throw sets of its SCC-internal callees. Repeat until no set changes. Termination is guaranteed because throw sets grow monotonically and the universe of exception types is finite (all struct types used in `throw` statements and built-in operations across the module).

5. **Write annotations** — set `callgraph.throws` on each function node.

### 4. Detect tail calls

For each function, walk the body threading a `tail: bool` flag according to the rules in Tail Position Rules. When a function call or method call is reached with `tail=true`, set `callgraph.is_tail_call=true` on the call expression node.

## Examples

```taytsh
fn ParseOrDefault(s: string) -> int {
    try {
        return ParseInt(s, 10)
    } catch e: ValueError {
        return 0
    }
}
-- function: callgraph.throws=""                (ValueError is caught)
-- function: callgraph.is_recursive=false
-- ParseInt call: callgraph.is_tail_call=false  (inside try body)
```

```taytsh
fn ParseBoth(a: string, b: string) -> (int, int) {
    let x: int = ParseInt(a, 10)
    let y: int = ParseInt(b, 10)
    return (x, y)
}
-- function: callgraph.throws="ValueError"  (ParseInt throws, not caught)
-- function: callgraph.is_recursive=false
```

```taytsh
fn Lookup(m: map[string, int], key: string) -> int {
    return m[key]
}
-- function: callgraph.throws="KeyError"  (map index can throw)
-- function: callgraph.is_recursive=false
```

```taytsh
fn Eval(node: Node) -> int {
    match node {
        case lit: Literal {
            return lit.value
        }
        case bin: BinOp {
            return Eval(bin.left) + Eval(bin.right)
        }
    }
}
-- function: callgraph.throws=""
-- function: callgraph.is_recursive=true, callgraph.recursive_group="scc:0"
-- Eval calls: callgraph.is_tail_call=false  (result is added, not returned)
```

```taytsh
fn Last(xs: list[int]) -> int {
    if Len(xs) == 1 {
        return xs[0]
    }
    return Last(xs[1:Len(xs)])
}
-- function: callgraph.throws="IndexError"  (index and slice operations)
-- function: callgraph.is_recursive=true
-- Last call: callgraph.is_tail_call=true   (return Last(...) in tail position)
```

```taytsh
fn IsEven(n: int) -> bool {
    if n == 0 { return true }
    return IsOdd(n - 1)
}

fn IsOdd(n: int) -> bool {
    if n == 0 { return false }
    return IsEven(n - 1)
}
-- IsEven: callgraph.is_recursive=true,  callgraph.recursive_group="scc:0"
-- IsOdd:  callgraph.is_recursive=true,  callgraph.recursive_group="scc:0"
-- IsOdd call in IsEven:  callgraph.is_tail_call=true
-- IsEven call in IsOdd:  callgraph.is_tail_call=true
-- both: callgraph.throws=""
```

```taytsh
fn SafeProcess(input: string) -> string {
    try {
        return Transform(input)
    } catch e: ValueError {
        return Fallback(input)
    } finally {
        Cleanup()
    }
}
-- Transform call:  callgraph.is_tail_call=false  (inside try body)
-- Fallback call:   callgraph.is_tail_call=false  (finally must run after catch)
-- Cleanup call:    callgraph.is_tail_call=false   (returns void, not function's value)
```

```taytsh
fn Propagate(s: string) -> int {
    try {
        return ParseInt(s, 10)
    } catch e: ValueError {
        throw ValueError(Concat("bad: ", e.message))
    }
}
-- function: callgraph.throws="ValueError"  (catch body re-throws)
```

```taytsh
fn ReadConfig(path: string) -> string {
    let content: string | bytes = ReadFile(path)
    match content {
        case s: string {
            return s
        }
        case b: bytes {
            return Decode(b)
        }
    }
}
-- function: callgraph.throws="IOError"  (ReadFile can throw, not caught)
```

## Postconditions

- Every function node has `callgraph.throws: string`, `callgraph.is_recursive: bool`, and `callgraph.recursive_group: string`.
- Call expression nodes in tail position have `callgraph.is_tail_call=true` (absent means `false`).
- Throw sets are closed: if function A calls function B, every type in B's throw set that is not filtered by a `try`/`catch` enclosing the call site appears in A's throw set.
- All functions in the same SCC share the same `callgraph.recursive_group` value.
- Non-recursive functions have `callgraph.recursive_group=""`.
