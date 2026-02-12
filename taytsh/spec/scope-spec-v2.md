# Scope Analysis (Tongues Middleend)

This document specifies the **scope** middleend pass over Taytsh IR. The pass is **intra-procedural**: it analyzes each function body independently and writes results into Taytsh node `annotations`.

Taytsh already has:

- explicit local declarations via `let`
- lexical block scoping (no shadowing — each name bound at most once per function)
- concrete static types (no `object` top type)
- explicit function types (`fn[T..., R]`)
- no module-level variables/constants (top-level declarations are only `fn`, `struct`, `interface`, `enum`)

As a result, this pass does **not** attempt to infer declarations from “first assignment”, nor does it track module constants.

## Inputs

The pass assumes it is given a **valid Taytsh Module** with:

- names resolved (each identifier use resolves to a specific binding or symbol)
- static types available for expressions (at least enough to recognize interface-typed uses and to stringify a type)

## Outputs

The pass writes annotations under the `scope.` namespace:

- binding-level facts: reassignment / constness
- parameter facts: modified / unused
- use-site facts: narrowed type, interface-typed, function reference
- type-switch case facts: interface usage within case body

All annotations are stored in each node’s `annotations` map (Taytsh supports `bool`, `int`, `string`, `(int, int)` values).

## Binding Sites

The pass treats the following constructs as introducing **bindings** (lexically scoped names):

- function parameters (including `self` for methods)
- `let name: T = expr`
- `for` loop binders (`for x in xs { ... }`, `for i, x in xs { ... }`)
- `match` case binders (`case v: T { ... }`, `default v { ... }`)
- `catch` binders (`catch e: SomeError { ... }`, `catch e { ... }`)

Taytsh forbids shadowing — each name is bound at most once per function. The pass can resolve names by spelling without tracking binding identity.

## Produced Annotations

This section is normative: it defines the annotation keys, value types, attachment points, and when keys MUST be present.

Notation:

- **binding declaration node**: the AST node that introduces a binding (parameter, `let` binder, loop binder, etc.)
- **identifier use node**: an identifier expression that resolves to a binding or symbol
- For declaration nodes, required boolean keys MUST be present even when `false`.
- For identifier use nodes, boolean keys MAY be omitted when `false` (absence is equivalent to `false`).

Attachment points (by syntax):

| Syntax                                           | Annotation attachment point       |
| ------------------------------------------------ | --------------------------------- |
| `fn F(p: T, ...) { ... }`                        | each parameter binding `p`        |
| `let x: T = expr`                                | the `let` binding `x`             |
| `for x in xs { ... }` / `for i, x in xs { ... }` | the loop binder(s) `i`, `x`       |
| `case v: T { ... }` / `default v { ... }`        | the case binding `v` (if present) |
| `catch e: E { ... }` / `catch e { ... }`         | the catch binding `e`             |
| identifier expression `x`                        | the identifier use node           |

### Binding declarations

These annotations are attached to the node that introduces the binding (parameter node, `let` binder node, loop binder node, etc.).

| Key                   | Type   | Applies to               | Meaning                                                  |
| --------------------- | ------ | ------------------------ | -------------------------------------------------------- |
| `scope.is_reassigned` | `bool` | binding declaration node | `true` if the binding is assigned after its introduction |
| `scope.is_const`      | `bool` | binding declaration node | `true` if the binding is never reassigned                |

Notes:

- The initializing value in a `let` does **not** count as a reassignment.
- The implicit per-iteration update of a `for` binder does **not** count as reassignment; only explicit assignments in the loop body do.
- The implicit assignment performed by `match`/`catch` binders does **not** count as reassignment; only explicit assignments in the case/catch body do.
- For parameters, “reassigned” means the parameter binding is assigned in the body (e.g. `p = ...`).
- Reassignment is defined on the binding itself: `x = expr` (including tuple assignment targets) counts; `x.field = expr` and `x[i] = expr` do **not** count as reassigning `x`.
- Backends may use `scope.is_const` to emit `const`/`final`/immutable bindings when the target supports it.
- `scope.is_const` MUST be logically equivalent to `!scope.is_reassigned` for the same binding.
- `scope.is_const` is about **rebinding**, not deep immutability: a binding can be `scope.is_const=true` even if its value is a mutable list/map/set/struct that is mutated.

Example:

```taytsh
fn Example() -> void {
    let x: int = 1
    let y: int = 2
    x = 3
    WritelnOut(ToString(y))
}

-- on the `let x` binder: scope.is_reassigned=true,  scope.is_const=false
-- on the `let y` binder: scope.is_reassigned=false, scope.is_const=true
```

### Parameters

These annotations are attached to parameter declaration nodes.

| Key                 | Type   | Applies to                 | Meaning                                                      |
| ------------------- | ------ | -------------------------- | ------------------------------------------------------------ |
| `scope.is_modified` | `bool` | parameter declaration node | `true` if the parameter is reassigned or mutated in the body |
| `scope.is_unused`   | `bool` | parameter declaration node | `true` if the parameter is never referenced                  |

The pass considers a parameter **mutated** if any of the following occur with the parameter binding as the *base* value:

1. It appears on the left-hand side of an assignment, directly or via a derived lvalue:
   - `p = expr`
   - `p.field = expr`
   - `p[i] = expr`
   - tuple assignment containing any of the above targets
2. It is passed as the receiver to a `void`-returning method call:
   - `p.Mutate(...)` where `Mutate`’s return type is `void`
3. It is passed in the first argument position to a known mutating built-in:
   - lists: `Append`, `Insert`, `Pop`, `RemoveAt`
   - maps: index assignment (`m[k] = v`), `Delete`
   - sets: `Add`, `Remove`

This definition is intentionally conservative and syntactic: it tracks explicit mutation forms in Taytsh.

Additional rules:

- `scope.is_modified` MUST be `true` if the parameter binding is reassigned (i.e. the binding’s `scope.is_reassigned=true`).
- `scope.is_unused` is based on resolved uses of the parameter binding (shadowing does not count as use).
- This pass does not require alias analysis: mutation through an alias MAY be missed unless the implementation explicitly tracks it.

Examples:

```taytsh
fn P1(xs: list[int]) -> void {
    Append(xs, 1)         -- xs: scope.is_modified=true
}

fn P2(x: int) -> int {
    return 0              -- x: scope.is_unused=true, scope.is_modified=false
}

fn P3(xs: list[int]) -> void {
    let a: list[int] = xs
    Append(a, 1)          -- xs: MAY remain scope.is_modified=false without alias tracking
}
```

### Type-switch case bindings

This annotation is attached to type-switch case and default binding nodes.

| Key                    | Type     | Applies to                            | Meaning                                                            |
| ---------------------- | -------- | ------------------------------------- | ------------------------------------------------------------------ |
| `scope.case_interface` | `string` | type-switch case/default binding node | interface the binding is used as in the case body, or `""` if none |

**`scope.case_interface`** resolves which interface a type-switch case binding is consumed as within its case body. In a type switch `match v { case x: T { BODY } }`, the binding `x` is narrowed to concrete type `T`. If `BODY` passes `x` to a function or calls a method on `x` that belongs to an interface `I` rather than to `T` directly, the annotation records `"I"`.

Go emits type switches as `switch v := expr.(type)` and needs to know whether a case body uses the binding through an interface — `v := expr.(InterfaceName)` vs `v := expr.(ConcreteType)`. Without this annotation, the Go backend must perform its own recursive AST walk per case body.

The analysis walks the case body once and checks whether the binding appears as receiver or argument in any call whose resolved target is an interface method. If multiple interfaces match, the first one found is recorded (in practice, Taytsh's type system ensures at most one interface is relevant per case).

When the binding is not used through any interface method, the value is `""`.

`scope.case_interface` MUST be present on every type-switch case and default binding node (even when `""`).

Example:

```taytsh
interface Printable {
    fn Display() -> string
}

struct Foo {
    fn Display() -> string { return "foo" }
    fn FooOnly() -> void { }
}

fn Process(v: Foo | int) -> void {
    match v {
        case f: Foo {
            let s: string = f.Display()
            WritelnOut(s)
        }
        -- case binding f: scope.case_interface="Printable"
        --   (Display is a method of interface Printable)
        case n: int {
            WritelnOut(ToString(n))
        }
        -- case binding n: scope.case_interface=""
    }
}
```

### Identifier uses (variable references)

These annotations are attached to identifier expression nodes (variable references).

| Key                     | Type     | Applies to          | Meaning                                                         |
| ----------------------- | -------- | ------------------- | --------------------------------------------------------------- |
| `scope.narrowed_type`   | `string` | identifier use node | a more precise type for this use site (Taytsh type syntax)      |
| `scope.is_interface`    | `bool`   | identifier use node | `true` if the static type at this use site is an interface type |
| `scope.is_function_ref` | `bool`   | identifier use node | `true` if the identifier resolves to a top-level `fn` symbol    |

`scope.narrowed_type` is present only when the use-site type is strictly narrower than the binding’s declared type.

`scope.narrowed_type` value format:

- MUST be a Taytsh type string (e.g. `int`, `Node`, `int?`, `int | string`, `(int, string)`, `list[int]`, `fn[int, bool]`)
- SHOULD use canonical union normalization (flatten, dedupe, unordered) and `T?` sugar when applicable

`scope.is_interface` semantics:

- Set `scope.is_interface=true` iff the identifier use node’s static type is an interface type (not merely a union containing an interface member).
- This is a use-site property: the same binding can have `scope.is_interface=true` at some uses and `false` at others if narrowing changes the static type.

`scope.is_function_ref` semantics:

- Set `scope.is_function_ref=true` iff the identifier use node resolves to a top-level `fn` symbol.
- The key applies both when the function is used as a value (`let f = AddOne`) and when it appears as the callee expression (`AddOne(1)`), since both are identifier uses.

Examples:

```taytsh
fn Narrow(x: int?) -> int {
    if x == nil { return 0 }
    return x                -- identifier `x` here: scope.narrowed_type="int"
}

fn Ref(f: fn[int, int], x: int) -> int {
    return f(x)              -- identifier `f` is not a function ref (it’s a parameter)
}

fn AddOne(x: int) -> int { return x + 1 }

fn DirectRef() -> fn[int, int] {
    return AddOne            -- identifier `AddOne` here: scope.is_function_ref=true
}
```

## Narrowing Rules

Taytsh supports flow-sensitive narrowing for optionals/unions:

- after `x != nil`, remove `nil` from `x`’s type for the dominated region
- inside a `match` arm `case v: T { ... }`, treat the case binding `v` as type `T`
- in a `default v { ... }` arm, treat `v` as the residual type (uncovered union members / uncovered interface variants)

This pass records the final use-site type it observes for a variable reference as `scope.narrowed_type`. The source of truth for flow types may be:

- a prior type-checking pass that annotates each expression with its static type, or
- a minimal flow analysis performed by this pass (at least for `!= nil` checks and `match` arms).

Regardless of implementation, the observable contract is the same: a backend can read `scope.narrowed_type` at a use site to avoid emitting redundant casts/tests.

## Algorithm (per function)

For each function (including methods):

1. **Collect bindings**
   - Create a binding record for each parameter and each binder introduced in the body.
2. **Track references**
   - For each identifier use that resolves to a binding, record a “use”.
   - For each identifier use that resolves to a top-level function symbol, set `scope.is_function_ref=true` on that node.
3. **Track writes**
   - For each assignment, determine which binding(s) it writes:
     - `name = expr` writes the `name` binding
     - `a.b = expr` writes (mutates) the base expression `a` (for `scope.is_modified` when `a` is a parameter binding)
     - `a[i] = expr` writes (mutates) the base expression `a` (for `scope.is_modified` when `a` is a parameter binding)
     - tuple assignment writes each target independently
   - Mark `scope.is_reassigned=true` on the written binding’s declaration node (excluding its `let` initializer).
4. **Track parameter modification/usage**
   - `scope.is_unused=true` iff no resolved identifier use refers to that parameter binding.
   - `scope.is_modified=true` iff the parameter binding is reassigned or matches any mutation rule above.
5. **Record use-site types**
   - For each identifier use, compare its use-site static type against the binding’s declared type.
   - If narrower, write `scope.narrowed_type` as a Taytsh type string on the identifier node.
   - If the use-site static type is an interface, write `scope.is_interface=true` on the identifier node.
6. **Resolve type-switch case interfaces**
   - For each type-switch case/default binding, walk the case body.
   - Check whether the binding appears as receiver or argument in a call to a method defined on an interface type.
   - If so, record the interface name as `scope.case_interface`. Otherwise, record `""`.
7. **Derive constness**
   - For each binding declaration, set:
     - `scope.is_const = !scope.is_reassigned`

## Postconditions

- Every binding declaration node has `scope.is_reassigned: bool` and `scope.is_const: bool` (both present even when `false`).
- Every parameter declaration node has `scope.is_modified: bool` and `scope.is_unused: bool` (both present even when `false`).
- Identifier use nodes MAY have `scope.is_function_ref: bool` (when absent, treated as `false`).
- Identifier use nodes MAY have `scope.narrowed_type: string` (absent means “no narrower type recorded”).
- Identifier use nodes MAY have `scope.is_interface: bool` (when absent, treated as `false`).
- Every type-switch case and default binding node has `scope.case_interface: string` (present even when `""`).
