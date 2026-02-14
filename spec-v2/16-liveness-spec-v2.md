# Liveness Analysis (Tongues Middleend)

This document specifies the **liveness** middleend pass over Taytsh IR. The pass is **intra-procedural**: it analyzes each function body independently and writes results into Taytsh node `annotations`.

The pass identifies dead stores and unused bindings so backends can emit cleaner code — omitting initializers, replacing binding names with `_`, and suppressing unused-variable warnings.

## Inputs

The pass assumes it is given a **valid Taytsh Module** with:

- names resolved (each identifier use resolves to a specific binding)
- binding sites identifiable (`let`, parameters, `for` binders, `match`/`catch` binders)

The pass MAY read `scope.is_unused` from the scope pass to skip already-known unused parameters. It can run independently with degraded precision if scope annotations are absent.

## Outputs

The pass writes annotations under the `liveness.` namespace:

- dead-store detection for `let` declarations
- unused-binding detection for `catch` and `match` binders
- per-index unused detection for tuple assignments

All annotations are stored in each node's `annotations` map (Taytsh supports `bool`, `int`, `string`, `(int, int)` values).

## Produced Annotations

This section is normative: it defines the annotation keys, value types, attachment points, and when keys MUST be present.

Attachment points (by syntax):

| Syntax                                    | Annotation attachment point   |
| ----------------------------------------- | ----------------------------- |
| `let x: T = expr` / `let x: T`            | the `let` declaration node    |
| `catch e: E { ... }` / `catch e { ... }`  | the catch binding node        |
| `case v: T { ... }` / `default v { ... }` | the case/default binding node |
| `a, b = expr`                             | the tuple assignment node     |

### Let declarations

| Key                             | Type   | Applies to             | Meaning                                                           |
| ------------------------------- | ------ | ---------------------- | ----------------------------------------------------------------- |
| `liveness.initial_value_unused` | `bool` | `let` declaration node | `true` if the initial value is overwritten before it is ever read |

A `let` declaration's initial value is **unused** when every control-flow path from the declaration reaches an assignment to the same binding before any read of it. This covers both explicit initializers (`let x: int = 0`) and implicit zero values (`let x: int`).

When `true`, backends can suppress the initializer:
- Emit `let x: int` instead of `let x: int = 0`.
- Go emits `var x int` instead of `x := 0`.
- Rust emits `let x: i32;` (uninitialized binding).

`liveness.initial_value_unused` MUST be present on every `let` declaration node (even when `false`).

### Catch bindings

| Key                         | Type   | Applies to         | Meaning                                                            |
| --------------------------- | ------ | ------------------ | ------------------------------------------------------------------ |
| `liveness.catch_var_unused` | `bool` | catch binding node | `true` if the catch variable is never referenced in the catch body |

When `true`, backends can emit anonymous catch forms:
- Java: `catch (Exception _)`
- Go: `_ = err`
- Python: `except ValueError:` (no `as e`)

`liveness.catch_var_unused` MUST be present on every catch binding node (even when `false`).

### Match bindings

| Key                         | Type   | Applies to                | Meaning                                                    |
| --------------------------- | ------ | ------------------------- | ---------------------------------------------------------- |
| `liveness.match_var_unused` | `bool` | case/default binding node | `true` if the binding is never referenced in the case body |

When `true`, backends can emit patterns without bindings:
- Rust: `Foo(_)` instead of `Foo(v)`
- Go: `case Foo:` without variable assignment

This applies to both `case v: T { ... }` and `default v { ... }` bindings.

`liveness.match_var_unused` MUST be present on every case/default binding node that introduces a binding (cases without bindings, like `case nil` and enum cases, are not annotated).

### Tuple assignments

| Key                             | Type     | Applies to            | Meaning                                                                                 |
| ------------------------------- | -------- | --------------------- | --------------------------------------------------------------------------------------- |
| `liveness.tuple_unused_indices` | `string` | tuple assignment node | comma-separated 0-based indices of targets that are never read before being overwritten |

For a tuple assignment `a, b, c = F()`, if `a` is never read before being overwritten (or never read at all), the annotation is `"0"`. If both `a` and `c` are unused, it is `"0,2"`.

When present, backends can emit `_` for unused targets:
- Go: `_, b, _ := F()`
- Python: `_, b, _ = F()`
- Rust: `let (_, b, _) = F();`

`liveness.tuple_unused_indices` MUST be present on every tuple assignment node. When no indices are unused, the value is `""` (empty string).

## Algorithm (per function)

For each function (including methods):

1. **Collect bindings and their declaration sites.**

2. **Scan catch and match bindings** — for each catch binding and case/default binding, check whether the binding is referenced anywhere in the corresponding body. Set `liveness.catch_var_unused` and `liveness.match_var_unused` accordingly. This is a simple containment check — no dataflow needed.

3. **Analyze initial-value liveness** — for each `let` declaration, determine whether the initial value is read before being overwritten:
   - Walk forward from the declaration through the statement list.
   - If the binding is read (appears in an expression) before any assignment to it, the initial value is live → `false`.
   - If the binding is assigned before any read, the initial value is dead → `true`.
   - At control-flow branches (`if`/`else`, `match`, `try`/`catch`), the initial value is unused only if it is overwritten on **every** path before being read on **any** path.
   - Conservatively assume loops may execute zero times — an assignment inside a loop body does not guarantee the initial value is dead.

4. **Analyze tuple assignment targets** — for each tuple assignment, apply the same forward analysis per-target: is the target read before being overwritten? Collect unused indices.

## Examples

```taytsh
fn Example() -> void {
    let x: int = 0
    let y: int = 0
    x = ParseInt(ReadAll(), 10)
    WritelnOut(ToString(x + y))
}
-- let x: liveness.initial_value_unused=true  (overwritten before read)
-- let y: liveness.initial_value_unused=false (read in the WritelnOut call)
```

```taytsh
fn Safe(s: string) -> int {
    try {
        return ParseInt(s, 10)
    } catch e: ValueError {
        return 0
    }
}
-- catch binding e: liveness.catch_var_unused=true (e never referenced)
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
-- case lit: liveness.match_var_unused=false (lit.value is accessed)
-- case bin: liveness.match_var_unused=false (bin.left, bin.right accessed)
```

```taytsh
fn Quotient(a: int, b: int) -> int {
    let q: int
    let r: int
    q, r = DivMod(a, b)
    return q
}
-- tuple assignment: liveness.tuple_unused_indices="1" (r never read)
```

## Postconditions

- Every `let` declaration node has `liveness.initial_value_unused: bool`.
- Every catch binding node has `liveness.catch_var_unused: bool`.
- Every case/default binding node (that introduces a binding) has `liveness.match_var_unused: bool`.
- Every tuple assignment node has `liveness.tuple_unused_indices: string`.
