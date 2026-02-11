# Hoisting Analysis (Tongues Middleend)

This document specifies the **hoisting** middleend pass over Taytsh IR. The pass is **intra-procedural**: it analyzes each function body independently and writes results into Taytsh node `annotations`.

The pass solves three problems for backends with strict declaration-before-use requirements:

1. **Variable hoisting** — languages like Go require variables to be declared in the scope where they are used. When a `let` declaration appears inside a control structure (`if`, `try`, `while`, `for`, `match`) but the variable is referenced after the structure exits, the backend must emit the declaration before the structure. This pass identifies those variables and annotates the enclosing structure.

2. **Continue transformation** — Lua lacks native `continue` (before 5.2) and uses `goto`-based workarounds. This pass marks loops containing `continue` so backends can emit the transformation.

3. **Rune variable collection** — Go indexes strings by bytes, not runes. When a Taytsh string variable is character-indexed, the Go backend must convert it to `[]rune` at scope entry. This pass identifies which string variables need rune conversion.

This pass is **conditional** — it only runs when the target set includes Go, Lua, or another language requiring pre-declaration.

## Inputs

The pass assumes it is given a **valid Taytsh Module** with:

- static types available for all bindings
- scope annotations available: `scope.is_reassigned` (to know which variables are assigned inside control structures)

## Outputs

The pass writes annotations under the `hoisting.` namespace:

- control-structure annotations: variables to hoist
- loop annotations: continue presence
- function annotations: string variables needing rune conversion

All annotations are stored in each node's `annotations` map (Taytsh supports `bool`, `int`, `string`, `(int, int)` values).

## Produced Annotations

This section is normative: it defines the annotation keys, value types, attachment points, and when keys MUST be present.

Attachment points (by syntax):

| Syntax | Annotation attachment point |
| ------ | --------------------------- |
| `if ... { } else { }` | the `if` node |
| `try { } catch ... { }` | the `try` node |
| `while ... { }` | the `while` node |
| `for ... in ... { }` | the `for` node |
| `match ... { case ... }` | the `match` node |
| `fn F(...) -> T { ... }` | the function node |

### Control-structure annotations

| Key | Type | Applies to | Meaning |
| --- | ---- | ---------- | ------- |
| `hoisting.hoisted_vars` | `string` | `if`, `try`, `while`, `for`, `match` nodes | variables to declare before this structure |

A variable needs hoisting when:
1. Its `let` declaration appears inside the control structure's body (or a branch of it).
2. The variable is referenced after the control structure — in a subsequent statement in the enclosing block.

The annotation value is a semicolon-separated list of `name:type` pairs, where `type` is a Taytsh type string. Example: `"x:int;y:string"`. When no variables need hoisting, the value is `""` (empty string).

The type of a hoisted variable is its declared type from the `let` site. When the variable is assigned in multiple branches with different narrowed types, the declared type (the common supertype) is used.

`hoisting.hoisted_vars` MUST be present on every `if`, `try`, `while`, `for`, and `match` node (even when `""`).

### Loop annotations

| Key | Type | Applies to | Meaning |
| --- | ---- | ---------- | ------- |
| `hoisting.has_continue` | `bool` | `while`, `for` nodes | `true` if the loop body contains a `continue` statement |

The check is recursive — a `continue` nested inside an `if` or `match` within the loop body counts. A `continue` inside a nested loop does **not** count — it belongs to the inner loop.

When `true`, backends lacking native `continue` emit workarounds:
- Lua: `goto continue_label` with a label at the loop end, or a `repeat until true` wrapping pattern.

`hoisting.has_continue` MUST be present on every `while` and `for` node (even when `false`).

### Function annotations

| Key | Type | Applies to | Meaning |
| --- | ---- | ---------- | ------- |
| `hoisting.rune_vars` | `string` | function node | string variables that are character-indexed in the function body |

The annotation value is a comma-separated list of variable names. Example: `"s,name"`. When no string variables are indexed, the value is `""` (empty string).

A string variable needs rune conversion when it appears as the target of an index expression (`s[i]`) or a slice expression (`s[a:b]`) in the function body. The Go backend emits `sRunes := []rune(s)` at function entry for each listed variable and uses `sRunes[i]` at index sites.

`hoisting.rune_vars` MUST be present on every function node (even when `""`).

## Algorithm (per function)

For each function (including methods):

1. **Collect rune variables** — walk the function body and find every string-typed variable that appears as the base of an index or slice expression. Record the variable name. Write `hoisting.rune_vars` on the function node.

2. **Detect continue in loops** — for each `while` and `for` node, recursively check the body for `continue` statements, stopping at nested loops (they own their own `continue`). Write `hoisting.has_continue`.

3. **Detect hoisted variables** — for each control structure (`if`, `try`, `while`, `for`, `match`):
   - Collect all `let` declarations inside the structure's body (all branches).
   - For each such declaration, check whether the variable name is referenced in any statement **after** the control structure in the enclosing block.
   - If so, the variable needs hoisting. Record its name and declared type.
   - Write `hoisting.hoisted_vars` on the control structure node.

   Nesting: when control structures are nested, a variable declared in an inner structure that is used after the **outer** structure gets hoisted to the outer structure's annotation. Each variable appears in at most one `hoisting.hoisted_vars` — the outermost structure that requires the hoist.

## Examples

```taytsh
fn Classify(x: int) -> string {
    let label: string
    if x > 0 {
        label = "positive"
    } else {
        label = "negative"
    }
    return label
}
-- if node: hoisting.hoisted_vars="" (label is declared BEFORE the if, no hoist needed)
```

```taytsh
fn Parse(s: string) -> int {
    let result: int
    try {
        let n: int = ParseInt(s, 10)
        result = n * 2
    } catch e: ValueError {
        result = 0
    }
    return result
}
-- try node: hoisting.hoisted_vars="" (result declared before try; n is not used after try)
```

```taytsh
fn FindFirst(xs: list[string], prefix: string) -> string {
    let found: string = ""
    for x in xs {
        if StartsWith(x, prefix) {
            found = x
            break
        }
    }
    return found
}
-- for node: hoisting.has_continue=false, hoisting.hoisted_vars=""
```

```taytsh
fn SumFiltered(xs: list[int]) -> int {
    let total: int = 0
    for x in xs {
        if x < 0 {
            continue
        }
        total += x
    }
    return total
}
-- for node: hoisting.has_continue=true, hoisting.hoisted_vars=""
```

```taytsh
fn CharAt(s: string, i: int) -> rune {
    return s[i]
}
-- function node: hoisting.rune_vars="s"
```

## Postconditions

- Every `if`, `try`, `while`, `for`, and `match` node has `hoisting.hoisted_vars: string`.
- Every `while` and `for` node has `hoisting.has_continue: bool`.
- Every function node has `hoisting.rune_vars: string`.
- All hoisted variable types are concrete Taytsh types (no unresolved or top types).
