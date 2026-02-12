# Hoisting Analysis (Tongues Middleend)

This document specifies the **hoisting** middleend pass over Taytsh IR. The pass is **intra-procedural**: it analyzes each function body independently and writes results into Taytsh node `annotations`.

The pass solves five problems for backends with strict declaration or scoping requirements:

1. **Variable hoisting** — languages like Go require variables to be declared in the scope where they are used. When a `let` declaration appears inside a control structure (`if`, `try`, `while`, `for`, `match`, type switch) but the variable is referenced after the structure exits, the backend must emit the declaration before the structure. This pass identifies those variables and annotates the enclosing structure.

2. **Function-scope pre-declaration** — languages like Perl have strict block scoping where `my` declarations inside control-flow blocks are invisible to sibling blocks and subsequent statements. Backends that need all hoisted variables at function entry rather than before individual control structures use the aggregated function-level annotation.

3. **Continue transformation** — Lua lacks native `continue` (before 5.2) and uses `goto`-based workarounds. This pass marks loops containing `continue` so backends can emit the transformation.

4. **Break-in-switch detection** — languages that emit `match`/type-switch as native `switch`/`case` (C#) have a conflict: their `break` exits the switch, not an enclosing loop. When a Taytsh `break` (which targets the enclosing loop) appears inside a match/type-switch case body, backends must emit flag-variable workarounds. This pass marks affected match/type-switch nodes.

5. **Rune variable collection** — Go indexes strings by bytes, not runes. When a Taytsh string variable is character-indexed, the Go backend must convert it to `[]rune` at scope entry. This pass identifies which string variables need rune conversion.

This pass is **conditional** — it only runs when the target set includes Go, Lua, Perl, C#, or another language requiring pre-declaration or switch workarounds.

## Inputs

The pass assumes it is given a **valid Taytsh Module** with:

- static types available for all bindings and expressions (sufficient to resolve hoisted variable types in all cases, including tuple assignments inferred from function return types or call return types)
- scope annotations available: `scope.is_reassigned` (to know which variables are assigned inside control structures)
- when the strings pass is active: `strings.indexed` available (to derive `hoisting.rune_vars` without redundant string-indexing detection)

## Outputs

The pass writes annotations under the `hoisting.` namespace:

- control-structure annotations: variables to hoist
- loop annotations: continue presence
- match/type-switch annotations: break presence
- function annotations: string variables needing rune conversion, aggregated hoisted variables

All annotations are stored in each node's `annotations` map (Taytsh supports `bool`, `int`, `string`, `(int, int)` values).

## Produced Annotations

This section is normative: it defines the annotation keys, value types, attachment points, and when keys MUST be present.

Attachment points (by syntax):

| Syntax                        | Annotation attachment point |
| ----------------------------- | --------------------------- |
| `if ... { } else { }`         | the `if` node               |
| `try { } catch ... { }`       | the `try` node              |
| `while ... { }`               | the `while` node            |
| `for ... in ... { }`          | the `for` node              |
| `match ... { case ... }`      | the `match` node            |
| `match ... { case v: T ... }` | the type-switch node        |
| `fn F(...) -> T { ... }`      | the function node           |

### Control-structure annotations

| Key                     | Type     | Applies to                                              | Meaning                                    |
| ----------------------- | -------- | ------------------------------------------------------- | ------------------------------------------ |
| `hoisting.hoisted_vars` | `string` | `if`, `try`, `while`, `for`, `match`, type-switch nodes | variables to declare before this structure |

A variable needs hoisting when:
1. Its `let` declaration appears inside the control structure's body (or a branch of it).
2. The variable is referenced after the control structure — in a subsequent statement in the enclosing block.

The annotation value is a semicolon-separated list of `name:type` pairs, where `type` is a Taytsh type string. Example: `"x:int;y:string"`. When no variables need hoisting, the value is `""` (empty string).

The type of a hoisted variable is its declared type from the `let` site. When the variable is assigned in multiple branches with different narrowed types, the declared type (the common supertype) is used. When the type must be inferred (e.g. individual targets of a tuple assignment), the pass resolves it from context — function return type, call return type, or expression type at the assignment site. The type MUST always be resolved; a missing or unresolved type is a pass error.

`hoisting.hoisted_vars` MUST be present on every `if`, `try`, `while`, `for`, `match`, and type-switch node (even when `""`).

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

| Key                          | Type     | Applies to    | Meaning                                                          |
| ---------------------------- | -------- | ------------- | ---------------------------------------------------------------- |
| `hoisting.rune_vars`         | `string` | function node | string variables that are character-indexed in the function body |
| `hoisting.func_hoisted_vars` | `string` | function node | all hoisted variables aggregated across the function             |

**`hoisting.rune_vars`**: a comma-separated list of variable names. Example: `"s,name"`. When no string variables are indexed, the value is `""` (empty string).

A string variable needs rune conversion when it appears as the target of an index expression (`s[i]`) or a slice expression (`s[a:b]`) in the function body. The Go backend emits `sRunes := []rune(s)` at function entry for each listed variable and uses `sRunes[i]` at index sites.

`hoisting.rune_vars` MUST be present on every function node (even when `""`).

**`hoisting.func_hoisted_vars`**: the union of all `hoisting.hoisted_vars` entries across every control structure in the function, collected into a single semicolon-separated `name:type` list. Same format as `hoisting.hoisted_vars`. When no variables are hoisted anywhere in the function, the value is `""`.

Languages with strict block scoping (Perl) need all hoisted variables pre-declared at function scope — a `my` declaration inside a control-flow block is invisible to sibling blocks and subsequent statements. These backends read `hoisting.func_hoisted_vars` to emit all pre-declarations at function entry, rather than iterating over individual control-structure annotations.

`hoisting.func_hoisted_vars` MUST be present on every function node (even when `""`).

## Algorithm (per function)

For each function (including methods):

1. **Collect rune variables** — when the strings pass is active, derive from `strings.indexed`: a variable needs rune conversion iff `strings.indexed=true` and `strings.content!="ascii"`. Otherwise, walk the function body and find every string-typed variable that appears as the base of an index or slice expression. Record the variable name. Write `hoisting.rune_vars` on the function node.

2. **Detect continue in loops** — for each `while` and `for` node, recursively check the body for `continue` statements, stopping at nested loops (they own their own `continue`). Write `hoisting.has_continue`.

3. **Detect break in match/type-switch** — for each `match` and type-switch node, recursively check case bodies for `break` statements, stopping at nested loops (they own their own `break`). Do NOT stop at nested match/type-switch nodes — they do not intercept `break`. Write `hoisting.has_break`.

4. **Detect hoisted variables** — for each control structure (`if`, `try`, `while`, `for`, `match`, type switch):
   - Collect all `let` declarations inside the structure's body (all branches).
   - For each such declaration, check whether the variable name is referenced in any statement **after** the control structure in the enclosing block.
   - If so, the variable needs hoisting. Record its name and declared type. The type MUST be resolved — when the declaring `let` has a known type, use it; when the type must be inferred (e.g. from a tuple assignment), resolve from context (function return type, call return type, expression type).
   - Write `hoisting.hoisted_vars` on the control structure node.

   Nesting: when control structures are nested, a variable declared in an inner structure that is used after the **outer** structure gets hoisted to the outer structure's annotation. Each variable appears in at most one `hoisting.hoisted_vars` — the outermost structure that requires the hoist.

5. **Aggregate function-scope hoisted variables** — collect the union of all `hoisting.hoisted_vars` entries written in step 4 across the entire function. Write `hoisting.func_hoisted_vars` on the function node.

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

```taytsh
fn TryParse(s: string) -> int {
    let result: int
    try {
        let n: int = ParseInt(s, 10)
        result = n
    } catch e: ValueError {
        result = 0
    }
    return result
}
-- function node: hoisting.func_hoisted_vars="" (result declared before try; n not used after)
```

```taytsh
fn Classify(x: int) -> string {
    if x > 0 {
        let label: string = "positive"
        let code: int = 1
        return Concat(label, ToString(code))
    } else {
        let label: string = "negative"
        return label
    }
    -- no variables used after the if, so no hoisting needed
}
-- if node: hoisting.hoisted_vars=""
-- function node: hoisting.func_hoisted_vars=""
```

## Postconditions

- Every `if`, `try`, `while`, `for`, `match`, and type-switch node has `hoisting.hoisted_vars: string`.
- Every `while` and `for` node has `hoisting.has_continue: bool`.
- Every `match` and type-switch node has `hoisting.has_break: bool`.
- Every function node has `hoisting.rune_vars: string`.
- Every function node has `hoisting.func_hoisted_vars: string`.
- All hoisted variable types (in both `hoisting.hoisted_vars` and `hoisting.func_hoisted_vars`) are concrete Taytsh types — never empty, never unresolved. When the declaring `let` lacks an explicit type, the pass infers it from context.
