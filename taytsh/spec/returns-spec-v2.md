# Returns Analysis (Tongues Middleend)

This document specifies the **returns** middleend pass over Taytsh IR. The pass is **intra-procedural**: it analyzes each function body independently and writes results into Taytsh node `annotations`.

The pass performs pure control-flow analysis — it walks the statement tree, tracks which paths terminate, and records the results. It does not read annotations from other passes.

## Inputs

The pass assumes it is given a **valid Taytsh Module** with:

- static types available for expressions (at least enough to determine whether a return expression's type includes `nil`)
- try/catch structure intact (the pass inspects try bodies and catch bodies separately)

## Outputs

The pass writes annotations under the `returns.` namespace:

- block-level facts: whether a block always terminates
- function-level facts: Go named-return need, nil-returning possibility
- try-level facts: whether the try body contains returns

All annotations are stored in each node's `annotations` map (Taytsh supports `bool`, `int`, `string`, `(int, int)` values).

## Produced Annotations

This section is normative: it defines the annotation keys, value types, attachment points, and when keys MUST be present.

Attachment points (by syntax):

| Syntax | Annotation attachment point |
| ------ | --------------------------- |
| `fn F(...) -> T { ... }` | the function node |
| `{ stmt; stmt; ... }` | the block node |
| `try { ... } catch ...` | the try node |

### Block annotations

| Key | Type | Applies to | Meaning |
| --- | ---- | ---------- | ------- |
| `returns.always_returns` | `bool` | block node | `true` if every control-flow path through the block terminates |

A block **always returns** if every path through it ends in a terminator. The terminators are:

1. `return` (with or without a value)
2. `throw`
3. `Exit()`

A block always returns if its statement list contains a terminator, or if it ends with a composite statement that always returns:

- `if`/`else`: always returns iff both branches always return. An `if` without `else` never always-returns (the implicit fall-through path is non-terminating).
- `match`: always returns iff every case body (and `default` if present) always returns. Since `match` is exhaustive, all paths are covered.
- `try`/`catch`: always returns iff the try body always returns AND every catch body always returns. A `finally` block does not affect termination — it executes unconditionally but does not itself terminate the function.
- `while` / `for`: conservatively never always-returns. The loop may execute zero iterations, so the pass does not assume the body runs.

Statements after a terminator are unreachable. The pass does not need to flag them — it simply stops scanning the statement list once a terminator is found.

`returns.always_returns` MUST be present on every block node (even when `false`).

### Function annotations

| Key | Type | Applies to | Meaning |
| --- | ---- | ---------- | ------- |
| `returns.needs_named_returns` | `bool` | function node | `true` if the function requires Go-style named returns |
| `returns.may_return_nil` | `bool` | function node | `true` if the function may return `nil` at runtime |

**`returns.needs_named_returns`**: set `true` when a function contains a try/catch where any catch body (or the try body itself) contains a `return` statement. Go transforms try/catch into `defer`/`recover`, and deferred functions can only set return values through named returns.

**`returns.may_return_nil`**: set `true` if any of:
- The function body contains `return nil`.
- The function body contains a `return` whose expression has a static type that includes `nil` (i.e. `T?` or a union containing `nil`), and the expression is not narrowed to exclude `nil` at that point.

This annotation is independent of whether the declared return type includes `nil`. A function declared `-> int` that never returns `nil` gets `false`; a function declared `-> int?` that always narrows before returning also gets `false`.

Both MUST be present on every function node (even when `false`).

### Try annotations

| Key | Type | Applies to | Meaning |
| --- | ---- | ---------- | ------- |
| `returns.body_has_return` | `bool` | try node | `true` if the try body contains any `return` statement |

Languages that transform try/catch into non-exception mechanisms need to know whether the try body contains returns:

- Lua (`pcall`): a return inside pcall must be propagated via flag variables.
- Perl (`eval`): similar propagation needed.
- Zig (error unions): return inside error-handling blocks requires propagation.

The check is recursive — a `return` nested inside `if`, `while`, `for`, `match`, or another `try` within the try body counts.

`returns.body_has_return` MUST be present on every try node (even when `false`).

## Algorithm (per function)

For each function (including methods):

1. **Walk the body block**, computing `returns.always_returns` bottom-up:
   - For each statement list, scan until a terminator or the end.
   - For composite statements (`if`/`else`, `match`, `try`/`catch`), recurse into sub-blocks and combine results per the rules above.
   - Set `returns.always_returns` on each block node.

2. **Detect named-return need**:
   - For each `try` node in the function, check whether any catch body or the try body contains a `return` (at any nesting depth).
   - If so, set `returns.needs_named_returns=true` on the function node.

3. **Detect try-body returns**:
   - For each `try` node, recursively check the try body for any `return` statement.
   - Set `returns.body_has_return` accordingly.

4. **Detect nil returns**:
   - For each `return` statement, check whether the expression is `nil` or has a static type including `nil`.
   - If any such return exists, set `returns.may_return_nil=true` on the function node.

Steps 1–4 can be combined into a single recursive walk.

## Examples

```taytsh
fn Find(xs: list[int], target: int) -> int? {
    for x in xs {
        if x == target {
            return x
        }
    }
    return nil
}
-- function node: returns.needs_named_returns=false, returns.may_return_nil=true
-- the for body block: returns.always_returns=false (while/for never always-return)
-- the if body block: returns.always_returns=true (contains return)
-- the function body block: returns.always_returns=true (ends with return nil)
```

```taytsh
fn ParseOrDefault(s: string) -> int {
    try {
        return ParseInt(s, 10)
    } catch e: ValueError {
        return 0
    }
}
-- function node: returns.needs_named_returns=true (catch body has return)
-- function node: returns.may_return_nil=false
-- try node: returns.body_has_return=true
-- try body block: returns.always_returns=true
-- catch body block: returns.always_returns=true
```

```taytsh
fn Describe(v: int | string | nil) -> string {
    match v {
        case n: int {
            return Concat("int: ", ToString(n))
        }
        case s: string {
            return Concat("string: ", s)
        }
        case nil {
            return "nil"
        }
    }
}
-- function node: returns.needs_named_returns=false, returns.may_return_nil=false
-- function body block: returns.always_returns=true (match is exhaustive, all arms return)
```

## Postconditions

- Every block node has `returns.always_returns: bool`.
- Every function node has `returns.needs_named_returns: bool` and `returns.may_return_nil: bool`.
- Every try node has `returns.body_has_return: bool`.
- No annotations from other passes are read or written.
