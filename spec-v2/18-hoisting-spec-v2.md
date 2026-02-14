# Hoisting Analysis (Tongues Middleend)

This document specifies the **hoisting** middleend pass over Taytsh IR. The pass is **intra-procedural**: it analyzes each function body independently and writes results into Taytsh node `annotations`.

The pass solves three problems for backends with strict declaration or scoping requirements:

1. **Continue transformation** — Lua lacks native `continue` (before 5.2) and uses `goto`-based workarounds. This pass marks loops containing `continue` so backends can emit the transformation.

2. **Break-in-switch detection** — languages that emit `match`/type-switch as native `switch`/`case` (C#) have a conflict: their `break` exits the switch, not an enclosing loop. When a Taytsh `break` (which targets the enclosing loop) appears inside a match/type-switch case body, backends must emit flag-variable workarounds. This pass marks affected match/type-switch nodes.

3. **Rune variable collection** — Go indexes strings by bytes, not runes. When a Taytsh string variable is character-indexed, the Go backend must convert it to `[]rune` at scope entry. This pass identifies which string variables need rune conversion.

This pass is **conditional** — it only runs when the target set includes Go, Lua, C#, or another language requiring these workarounds.

## Inputs

The pass assumes it is given a **valid Taytsh Module** with:

- when the strings pass is active: `strings.indexed` available (to derive `hoisting.rune_vars` without redundant string-indexing detection)

## Outputs

The pass writes annotations under the `hoisting.` namespace:

- loop annotations: continue presence
- match/type-switch annotations: break presence
- function annotations: string variables needing rune conversion

All annotations are stored in each node's `annotations` map (Taytsh supports `bool`, `int`, `string`, `(int, int)` values).

## Produced Annotations

This section is normative: it defines the annotation keys, value types, attachment points, and when keys MUST be present.

Attachment points (by syntax):

| Syntax                        | Annotation attachment point |
| ----------------------------- | --------------------------- |
| `while ... { }`               | the `while` node            |
| `for ... in ... { }`          | the `for` node              |
| `match ... { case ... }`      | the `match` node            |
| `match ... { case v: T ... }` | the type-switch node        |
| `fn F(...) -> T { ... }`      | the function node           |

### Loop annotations

| Key                     | Type   | Applies to           | Meaning                                                 |
| ----------------------- | ------ | -------------------- | ------------------------------------------------------- |
| `hoisting.has_continue` | `bool` | `while`, `for` nodes | `true` if the loop body contains a `continue` statement |

The check is recursive — a `continue` nested inside an `if` or `match` within the loop body counts. A `continue` inside a nested loop does **not** count — it belongs to the inner loop.

When `true`, backends lacking native `continue` emit workarounds:
- Lua: `goto continue_label` with a label at the loop end, or a `repeat until true` wrapping pattern.

`hoisting.has_continue` MUST be present on every `while` and `for` node (even when `false`).

### Match/type-switch annotations

| Key                  | Type   | Applies to                 | Meaning                                                                          |
| -------------------- | ------ | -------------------------- | -------------------------------------------------------------------------------- |
| `hoisting.has_break` | `bool` | `match`, type-switch nodes | `true` if any case body contains a `break` statement targeting an enclosing loop |

Languages that emit `match`/type-switch as native `switch`/`case` statements (C#, and potentially others) face a conflict: their `break` keyword exits the switch, not an enclosing loop. When a Taytsh `break` (which always targets the nearest enclosing `while`/`for` loop) appears inside a match or type-switch case body, the backend must emit a flag-variable workaround to propagate the break past the switch.

The check is recursive — a `break` nested inside an `if` or `try` within the case body counts. A `break` inside a nested `while`/`for` loop does **not** count — it targets that inner loop. A `break` inside a nested match/type-switch **does** count — match/type-switch nodes do not intercept `break`.

`hoisting.has_break` MUST be present on every `match` and type-switch node (even when `false`).

### Function annotations

| Key                  | Type     | Applies to    | Meaning                                                          |
| -------------------- | -------- | ------------- | ---------------------------------------------------------------- |
| `hoisting.rune_vars` | `string` | function node | string variables that are character-indexed in the function body |

A comma-separated list of variable names. Example: `"s,name"`. When no string variables are indexed, the value is `""` (empty string).

A string variable needs rune conversion when it appears as the target of an index expression (`s[i]`) or a slice expression (`s[a:b]`) in the function body. The Go backend emits `sRunes := []rune(s)` at function entry for each listed variable and uses `sRunes[i]` at index sites.

`hoisting.rune_vars` MUST be present on every function node (even when `""`).

## Algorithm (per function)

For each function (including methods):

1. **Collect rune variables** — when the strings pass is active, derive from `strings.indexed`: a variable needs rune conversion iff `strings.indexed=true` and `strings.content!="ascii"`. Otherwise, walk the function body and find every string-typed variable that appears as the base of an index or slice expression. Record the variable name. Write `hoisting.rune_vars` on the function node.

2. **Detect continue in loops** — for each `while` and `for` node, recursively check the body for `continue` statements, stopping at nested loops (they own their own `continue`). Write `hoisting.has_continue`.

3. **Detect break in match/type-switch** — for each `match` and type-switch node, recursively check case bodies for `break` statements, stopping at nested loops (they own their own `break`). Do NOT stop at nested match/type-switch nodes — they do not intercept `break`. Write `hoisting.has_break`.

## Examples

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
-- for node: hoisting.has_continue=true
```

```taytsh
fn CharAt(s: string, i: int) -> rune {
    return s[i]
}
-- function node: hoisting.rune_vars="s"
```

```taytsh
fn ProcessItems(items: list[Foo | int]) -> void {
    for item in items {
        match item {
            case f: Foo {
                if ShouldSkip(f) {
                    break
                }
                Process(f)
            }
            case n: int {
                WritelnOut(ToString(n))
            }
        }
    }
}
-- match node: hoisting.has_break=true (break inside case targets the for loop)
-- for node: hoisting.has_continue=false
```

## Postconditions

- Every `while` and `for` node has `hoisting.has_continue: bool`.
- Every `match` and type-switch node has `hoisting.has_break: bool`.
- Every function node has `hoisting.rune_vars: string`.
