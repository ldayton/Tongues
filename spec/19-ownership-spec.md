# Ownership Analysis (Tongues Middleend)

This document specifies the **ownership** middleend pass over Taytsh IR. The pass is **intra-procedural**: it analyzes each function body independently and writes results into Taytsh node `annotations`.

The pass determines memory ownership for every binding and detects when values escape their defining scope. This information lets non-GC backends (C, Rust, Zig, Swift) emit correct memory management without re-analyzing the IR.

This pass is **conditional** — it only runs when the target set includes C, Rust, Zig, or Swift. For GC targets the pass is a no-op.

## Inputs

The pass assumes it is given a **valid Taytsh Module** with:

- static types available for all bindings and expressions
- scope annotations available: `scope.is_modified` (mutation tracking), `scope.is_const` (immutability guarantees)
- liveness annotations available: `liveness.initial_value_unused` (to skip dead stores)

## Outputs

The pass writes annotations under the `ownership.` namespace:

- binding-level facts: ownership kind and lifetime region
- expression-level facts: whether a value escapes its scope

All annotations are stored in each node's `annotations` map (Taytsh supports `bool`, `int`, `string`, `(int, int)` values).

## Background

Taytsh defines reference semantics for mutable types (`list`, `map`, `set`, structs) — assignment and parameter passing create aliases. The ownership pass determines how backends should represent this: which scope is responsible for freeing a value, whether a value is moved or borrowed, and when defensive copies are needed.

Immutable types (`int`, `float`, `bool`, `byte`, `rune`, `string`, `bytes`, tuples, enums) have no observable mutation, so the value/reference distinction is unobservable. The ownership pass still annotates bindings of these types (a string assigned to a struct field escapes, for instance), but backends have more freedom in representation.

## Ownership Patterns

The pass classifies each binding based on how its value originates and flows:

| Pattern                               | `ownership.kind` | Rationale                          |
| ------------------------------------- | ---------------- | ---------------------------------- |
| Constructor call (`Foo(...)`)         | `owned`          | caller creates the value           |
| Collection literal (`[...]`, `{...}`) | `owned`          | caller creates the value           |
| Function return value                 | `owned`          | ownership transfers to caller      |
| Parameter (default)                   | `borrowed`       | caller retains ownership           |
| Field access (`x.field`)              | `borrowed`       | owner is the containing struct     |
| Index expression (`xs[i]`)            | `borrowed`       | owner is the containing collection |
| Ambiguous / multiply-assigned         | `shared`         | runtime reference counting needed  |

A binding starts with an initial classification from its origin. Subsequent assignments may change the classification — a binding that receives values from multiple sources with different ownership is conservatively marked `shared`.

## Produced Annotations

This section is normative: it defines the annotation keys, value types, attachment points, and when keys MUST be present.

Attachment points (by syntax):

| Syntax                                           | Annotation attachment point |
| ------------------------------------------------ | --------------------------- |
| `fn F(p: T, ...) { ... }`                        | each parameter binding `p`  |
| `let x: T = expr`                                | the `let` binding `x`       |
| `for x in xs { ... }` / `for i, x in xs { ... }` | loop binder(s) `i`, `x`     |
| `case v: T { ... }` / `default v { ... }`        | case binding `v`            |
| `catch e: E { ... }` / `catch e { ... }`         | catch binding `e`           |
| any expression                                   | the expression node         |

### Binding annotations

| Key                | Type     | Applies to               | Meaning                                |
| ------------------ | -------- | ------------------------ | -------------------------------------- |
| `ownership.kind`   | `string` | binding declaration node | `"owned"`, `"borrowed"`, or `"shared"` |
| `ownership.region` | `string` | binding declaration node | lifetime region identifier             |

**`ownership.kind`** classifies how the binding relates to the value's lifetime:

- `"owned"` — this binding owns the value. The scope containing this binding is responsible for cleanup. C: `free()` at scope exit. Rust: value is moved in, drop at scope exit. Zig: allocator free at scope exit.
- `"borrowed"` — this binding references a value owned elsewhere. The binding must not outlive the owner. C: no `free()`. Rust: `&T` or `&mut T`. Zig: pointer, no dealloc.
- `"shared"` — ownership cannot be determined statically. Backends emit reference-counted wrappers. C: `rc_retain`/`rc_release`. Rust: `Rc<T>` or `Arc<T>`. Swift: ARC (default behavior).

**`ownership.region`** is an opaque identifier grouping bindings into the same lifetime scope. Bindings in the same region share a lifetime — they are created and destroyed together. The identifier format is implementation-defined (e.g. `"fn:Foo"`, `"block:3"`, `"loop:1"`).

C uses the region to determine which scope calls `free()`. Rust uses it to inform lifetime annotations on references. Two bindings with the same region identifier can safely alias each other.

Both `ownership.kind` and `ownership.region` MUST be present on every binding declaration node.

### Expression annotations

| Key                 | Type   | Applies to      | Meaning                                       |
| ------------------- | ------ | --------------- | --------------------------------------------- |
| `ownership.escapes` | `bool` | expression node | `true` if the value escapes the current scope |

A value **escapes** its scope when it is stored somewhere that outlives the scope in which it was created:

1. **Stored in a struct field**: `obj.field = expr` — the value outlives the current scope if `obj` does.
2. **Appended to a collection**: `Append(xs, expr)`, `Add(s, expr)`, map index assignment — the value outlives the current scope if the collection does.
3. **Returned from the function**: `return expr` — the value escapes to the caller's scope.
4. **Thrown**: `throw expr` — the value escapes to a catch site potentially in another function.

When `ownership.escapes=true`, backends emit defensive copies for borrowed values:
- C: `strdup()` for strings, `memcpy` + deep copy for structs.
- Rust: `.clone()`.
- Zig: allocator copy.

`ownership.escapes` is only set on expression nodes where the value escapes. Absence is equivalent to `false` — most expressions do not escape.

## Algorithm (per function)

For each function (including methods):

1. **Skip dead stores** — read `liveness.initial_value_unused`. Do not analyze ownership of initializers that are never read.

2. **Classify bindings** — for each binding declaration, determine `ownership.kind` from the origin of its value:
   - Constructor calls, collection literals, and function return values → `"owned"`.
   - Parameters → `"borrowed"` (caller retains ownership).
   - Field access and index expressions → `"borrowed"` (owner is the containing object).
   - If the binding is assigned from multiple sources with differing ownership, or if the analysis cannot determine a single owner → `"shared"`.

3. **Assign regions** — group bindings by the scope in which they are created. Each function body, loop body, and block introduces a potential region. Bindings whose values originate in the same scope share a region identifier.

4. **Detect escapes** — walk assignment targets, function returns, `throw` expressions, and mutating built-in calls. For each value expression in an escaping position:
   - If the value's binding is `"borrowed"`, mark the expression `ownership.escapes=true` — the backend must copy.
   - If the value's binding is `"owned"`, the ownership transfers — no copy needed, but the original binding should not be used after the transfer (Rust move semantics). Mark `ownership.escapes=true` to signal the transfer.

5. **Propagate through control flow** — at `if`/`else`, `match`, and `try`/`catch` boundaries, merge ownership: if a binding receives an owned value on one path and a borrowed value on another, conservatively mark it `"shared"`.

6. **Read scope annotations** — use `scope.is_modified` to identify parameters that are mutated. A mutated borrowed parameter may need `&mut` (Rust) or pointer passing (C). Use `scope.is_const` to identify bindings that are never reassigned — these can use simpler ownership (no move tracking needed).

## Examples

```taytsh
fn MakeToken(kind: TokenKind, value: string) -> Token {
    let t: Token = Token(kind, value, 0)
    return t
}
-- parameter kind:  ownership.kind="borrowed", ownership.region="fn:MakeToken"
-- parameter value: ownership.kind="borrowed", ownership.region="fn:MakeToken"
-- let t:           ownership.kind="owned",    ownership.region="fn:MakeToken"
-- t in return:     ownership.escapes=true (returned)
```

```taytsh
fn Append3(xs: list[int], a: int, b: int, c: int) -> void {
    Append(xs, a)
    Append(xs, b)
    Append(xs, c)
}
-- parameter xs: ownership.kind="borrowed", scope.is_modified=true
-- a, b, c in Append calls: ownership.escapes=true (stored in collection)
```

```taytsh
fn First(xs: list[string]) -> string {
    let s: string = xs[0]
    return s
}
-- let s: ownership.kind="borrowed" (value from index expression)
-- s in return: ownership.escapes=true (C backend emits strdup)
```

```taytsh
fn Build(flag: bool) -> list[int] {
    let result: list[int]
    if flag {
        result = [1, 2, 3]
    } else {
        result = [4, 5, 6]
    }
    return result
}
-- let result: ownership.kind="owned" (both branches assign owned values)
```

## Postconditions

- Every binding declaration node has `ownership.kind: string` (`"owned"`, `"borrowed"`, or `"shared"`).
- Every binding declaration node has `ownership.region: string`.
- Expression nodes in escaping positions have `ownership.escapes: bool` (absent means `false`).
- No annotation from this pass contradicts scope or liveness annotations.
