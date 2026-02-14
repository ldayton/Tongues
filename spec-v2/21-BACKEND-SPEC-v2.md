# Tongues Backend Specification v2

A Tongues backend walks the annotated Taytsh IR and emits source code for one target language. It reads the IR tree and its annotations (written by the lowerer and middleend passes). It never modifies the IR.

## Backend Structure

Each backend is a tree walker. For each IR node type, the backend has an emitter that decides how to render that node. The emitter may:

1. Emit the node's desugared form directly (always correct).
2. Check provenance annotations and emit an idiomatic alternative.
3. Read annotations to refine the emission (const, unused, etc.).
4. Recognize structural patterns in the IR and emit target idioms.

These are not phases — they happen together during the single tree walk. The emitter for a `for` node might check provenance, read scope annotations, and pattern-match the body, all in one method.

## Annotation Consumption

### scope.*

Used by all backends.

`scope.is_const` — emit immutable bindings where the target supports them:

| Target     | const form           | Notes                                            |
| ---------- | -------------------- | ------------------------------------------------ |
| C          | `const`              | only for primitives and pointers-to-const        |
| C#         | `readonly` / `const` | `const` for compile-time, `readonly` for runtime |
| Dart       | `final`              |                                                  |
| Go         | `const`              | only for untyped constants; else no effect       |
| Java       | `final`              |                                                  |
| JavaScript | `const`              |                                                  |
| Lua        | `local`              | no const; annotation unused                      |
| Perl       | `my`                 | no const; annotation unused                      |
| PHP        | n/a                  | no block-scoped const                            |
| Python     | n/a                  | no const; could emit `Final[]` for type checkers |
| Ruby       | n/a                  | no local const                                   |
| Rust       | `let` (default)      | `let mut` when `scope.is_const=false`            |
| Swift      | `let`                | `var` when `scope.is_const=false`                |
| TypeScript | `const`              |                                                  |
| Zig        | `const`              | `var` when `scope.is_const=false`                |

`scope.is_unused` — suppress unused parameter warnings:

| Target | Mechanism                                         |
| ------ | ------------------------------------------------- |
| Go     | assign to `_` if unused                           |
| Rust   | prefix with `_`                                   |
| Zig    | assign to `_`                                     |
| C      | `(void)param;`                                    |
| Others | no action needed (no unused-param compile errors) |

`scope.is_modified` — informs pass-by-value targets whether a parameter's value is mutated. Backends for value-semantics languages (Go, Rust, Swift, C, Zig) use this to decide pointer/reference passing. If `scope.is_modified=true` and the parameter is a struct, the backend may need to pass by pointer.

`scope.narrowed_type` — backends that require explicit casts or type assertions can skip them when narrowing has already been proven. Go emits the type assertion result directly; Rust skips `unwrap()`; Java skips the cast.

`scope.is_interface` — backends that need explicit downcasting (Go, Java, C#) use this to know when a variable holds an interface-typed value that will need a type assertion at use sites.

`scope.is_function_ref` — backends where function references need special syntax (Go is direct, Python is direct, Java needs method references, C needs function pointers) use this to emit the right form.

`scope.case_interface` — Go uses this in type-switch emission to choose the right type assertion. When a case binding is used through interface methods, Go emits `v := expr.(InterfaceName)` instead of `v := expr.(ConcreteType)`.

### returns.*

Used by a subset of backends.

`returns.needs_named_returns` — Go only. When a try/catch body contains returns, Go must use named return values so that deferred recover can set them.

`returns.may_return_nil` — Dart uses this to emit `dynamic` or nullable return types. Go uses this for pointer returns. Java/C# may add `@Nullable`.

`returns.body_has_return` — Lua (pcall wrapper needs to propagate return values via flags), Perl (eval block needs similar), Zig (errdefer/error union return propagation).

`returns.always_returns` — all backends can use this to suppress "missing return" warnings or omit unreachable default returns.

### liveness.*

Used by all backends for cleaner output.

`liveness.initial_value_unused` — suppress the initializer. Emit `let x: int` instead of `let x: int = 0` when the zero value is immediately overwritten. Go emits `var x int` instead of `x := 0`.

`liveness.catch_var_unused` — emit anonymous catch. Java: `catch (Exception _)`. Go: `_ = err`. Python: `except ValueError:` (no `as e`).

`liveness.match_var_unused` — emit pattern without binding. Rust: `Foo(_)`. Go: `case Foo:` without assignment.

`liveness.tuple_unused_indices` — emit `_` for unused targets. Go: `_, b := f()`. Python: `_, b = f()`. Rust: `let (_, b) = f();`.

### strings.*

Used by all backends except Python, Ruby, and Perl (native-rune targets where string operations are already correct).

`strings.content` — select string operation implementation based on character content:

| Content     | UTF-8 targets (Go, Rust, C, Zig, Lua, PHP) | UTF-16 targets (Java, C#, JS, TS, Dart, Swift)          |
| ----------- | ------------------------------------------ | ------------------------------------------------------- |
| `"ascii"`   | native byte operations (`s[i]`, `len(s)`)  | native code-unit operations (`.charAt()`, `.length`)    |
| `"bmp"`     | still need multi-byte decode               | native code-unit operations safely (no surrogate pairs) |
| `"unknown"` | full rune-safe operations                  | full codepoint-safe operations                          |

Target-specific ASCII-mode examples:

| Target     | ASCII `s[i]`                | ASCII `Len(s)` |
| ---------- | --------------------------- | -------------- |
| Go         | `s[i]` (byte access)        | `len(s)`       |
| Rust       | `s.as_bytes()[i] as char`   | `s.len()`      |
| C          | `s[i]`                      | `strlen(s)`    |
| Java       | `s.charAt(i)`               | `s.length()`   |
| JavaScript | `s[i]` or `s.charCodeAt(i)` | `s.length`     |
| PHP        | `$s[$i]`                    | `strlen($s)`   |

`strings.indexed` — when `false`, skip rune-indexing machinery entirely:

| Target     | Effect when `strings.indexed=false`               |
| ---------- | ------------------------------------------------- |
| Go         | no `[]rune` conversion needed                     |
| Rust       | no `.chars().nth()` needed; plain `&str` suffices |
| C          | no UTF-8 index-to-byte-offset mapping             |
| Java       | no `codePointAt()` / `offsetByCodePoints()`       |
| JavaScript | no surrogate-aware indexing                       |
| PHP        | no `mb_substr()`                                  |

`strings.iterated` — when `true` and `strings.indexed=false`, use sequential decoding instead of random-access conversion:

| Target | Sequential iteration form                     |
| ------ | --------------------------------------------- |
| Go     | `for _, r := range s` (UTF-8 decode per rune) |
| Rust   | `for c in s.chars()`                          |
| C      | sequential UTF-8 walk                         |
| Java   | `s.codePoints().forEach(...)`                 |

When `strings.indexed=true`, the rune conversion needed for indexing also handles iteration.

`strings.len_called` — when `false`, skip rune-counting calls:

| Target | Effect when `strings.len_called=false` |
| ------ | -------------------------------------- |
| Go     | no `utf8.RuneCountInString(s)`         |
| Rust   | no `.chars().count()`                  |
| PHP    | no `mb_strlen()`                       |
| C      | no UTF-8 rune-counting loop            |

When `strings.content="ascii"`, rune count equals byte count regardless of `strings.len_called` — the backend uses the native length operation.

`strings.builder` — when non-empty, emit efficient string building instead of quadratic concatenation:

| Target     | Builder mechanism                       |
| ---------- | --------------------------------------- |
| Go         | `strings.Builder` with `.WriteString()` |
| Java / C#  | `StringBuilder` with `.append()`        |
| Rust       | `String` with `.push_str()`             |
| C          | growable buffer with doubling           |
| Zig        | `ArrayList(u8)` or equivalent           |
| JavaScript | array `.push()` + `.join("")`           |
| Dart       | `StringBuffer` with `.write()`          |
| PHP        | array `$parts[]` + `implode()`          |
| Lua        | table `insert` + `table.concat()`       |

The backend transforms the loop: replace the `let ACC = ""` + loop-with-Concat pattern into builder initialization, append calls, and final `.toString()`/`.String()` extraction.

### hoisting.*

Used by Go, Lua, Perl, and C#.

`hoisting.hoisted_vars` — Go emits `var` declarations before the control structure. Lua emits `local` declarations before the block.

`hoisting.func_hoisted_vars` — Perl reads this to emit `my` declarations at function scope. Perl's block-scoped `my` declarations inside control-flow blocks are invisible to sibling blocks, so all hoisted variables must be pre-declared at function entry.

`hoisting.has_continue` — Lua emits `goto continue_label` with a label at the loop end, since Lua lacks native `continue` (before 5.2) or uses `repeat until true` wrapping.

`hoisting.has_break` — C# uses this for match/type-switch nodes emitted as native `switch`/`case`. When `true`, C# emits a flag variable to propagate `break` past the switch to the enclosing loop, since C#'s `break` exits the switch rather than the loop.

`hoisting.rune_vars` — Go emits `xRunes := []rune(x)` at function entry for string variables that are indexed, then uses `xRunes[i]` at index sites. When the strings pass is active, `hoisting.rune_vars` is derived from `strings.indexed` — only bindings with `strings.indexed=true` and `strings.content!="ascii"` need rune conversion.

### ownership.*

Used by C, Rust, Zig, Swift.

`ownership.kind` — determines how values are passed and stored. C: owned values are freed by the current scope; borrowed are not. Rust: owned uses move semantics; borrowed uses `&`/`&mut`. Zig: similar to C with explicit allocator patterns. Swift: informs ARC behavior.

`ownership.escapes` — C: emit `strdup()` for escaping strings, `memcpy` for escaping structs. Rust: forces a `.clone()`. Zig: allocator copy.

`ownership.region` — C: determines which scope calls `free()`. Rust: informs lifetime annotations.

### callgraph.*

Used by Go, Rust, Zig, Lua.

`callgraph.throws` — backends that represent exceptions as return values use the throw set to determine the function's error return type:

| Target | Mechanism                                                                        |
| ------ | -------------------------------------------------------------------------------- |
| Go     | error return type; empty throw set means no error return                         |
| Rust   | `Result<T, E>` where `E` is an enum of the throw set types; empty means bare `T` |
| Zig    | error union `!T` with error set derived from throw types; empty means bare `T`   |
| Java   | `throws` clause on method signature (optional but improves generated code)       |
| Others | no action needed (native exceptions propagate implicitly)                        |

`callgraph.is_recursive` — backends that need special handling for recursive functions:

| Target | Mechanism                                                                 |
| ------ | ------------------------------------------------------------------------- |
| Zig    | recursive functions cannot infer error sets; must declare them explicitly |
| Rust   | recursive functions returning `impl Trait` need `Box<dyn>` indirection    |
| Others | no special handling needed                                                |

`callgraph.recursive_group` — identifies mutually recursive function groups. Zig and Rust use this to co-declare error sets or types for the group.

`callgraph.is_tail_call` — backends that support tail call optimization:

| Target | Mechanism                                                                |
| ------ | ------------------------------------------------------------------------ |
| Lua    | proper tail calls are guaranteed; backend emits `return f()` form        |
| Zig    | `@call(.always_tail, f, args)` for self-tail-calls                       |
| Go     | self-recursive tail calls can be transformed to loops                    |
| Rust   | self-recursive tail calls can be transformed to loops (no TCO guarantee) |
| Others | no action needed (tail position has no special syntax)                   |

## Provenance Consumption

Backends read provenance annotations from IR nodes and decide whether to emit the idiomatic form or the desugared form. The desugared form is always correct. A backend that doesn't recognize a provenance tag simply ignores it.

### Single-node provenance

These are stamped on one IR node and require no context beyond that node.

**in_operator / not_in_operator** — `Contains(xs, v)` with provenance.

| Target     | Idiomatic form       | Notes                             |
| ---------- | -------------------- | --------------------------------- |
| Python     | `v in xs`            |                                   |
| Ruby       | `xs.include?(v)`     | already the Contains emission     |
| JavaScript | `xs.includes(v)`     | already the Contains emission     |
| Lua        | table lookup pattern | no direct `in`; ignore provenance |
| Perl       | `grep { $_ eq $v }`  | or `exists` for hashes            |

For most backends, `Contains` already emits idiomatically regardless of provenance. The provenance is mainly useful for Python's `v in xs` (operand order reversal) and `v not in xs` (single keyword instead of `not v in xs`).

**open_start / open_end** — slice with `0` or `Len(x)` bound.

| Target | Idiomatic form | Notes                             |
| ------ | -------------- | --------------------------------- |
| Python | `xs[:n]`       | drop the 0                        |
| Ruby   | `xs[0...n]`    | Ruby slicing already handles this |
| Go     | `xs[:n]`       | drop the 0                        |
| Rust   | `&xs[..n]`     | drop the 0                        |

The backend checks provenance on the slice node and omits the redundant bound. Without provenance, the backend would emit `xs[0:n]` which is correct but not idiomatic.

**negative_index** — `x[Len(x) - n]` with provenance.

| Target | Idiomatic form | Notes |
| ------ | -------------- | ----- |
| Python | `x[-n]`        |       |
| Ruby   | `x[-n]`        |       |
| Perl   | `$x[-n]`       |       |

The backend pattern-matches `Len(x) - n` (guaranteed by the frontend) and emits the negative index directly. Other backends emit the arithmetic form.

**string_multiply / list_multiply** — `Repeat(s, n)` with provenance.

| Target | Idiomatic form | Notes               |
| ------ | -------------- | ------------------- |
| Python | `s * n`        |                     |
| Ruby   | `s * n`        |                     |
| Perl   | `$s x $n`      | Perl's `x` operator |

Other backends emit their Repeat implementation (loop, library call, etc.).

**truthiness** — `Len(xs) > 0` or `s != ""` with provenance.

| Target     | Idiomatic form   | Notes                                       |
| ---------- | ---------------- | ------------------------------------------- |
| Python     | `if xs:`         | truthy check                                |
| Ruby       | `if xs`          | but empty collections are truthy in Ruby!   |
| JavaScript | `if (xs.length)` | or `if (s)` for strings                     |
| Perl       | `if (@xs)`       | or `if ($s)`                                |
| Lua        | n/a              | tables are always truthy; ignore provenance |

**Caution**: Ruby and Lua have different truthiness rules than Python. Ruby treats empty arrays/hashes as truthy. Lua treats tables as truthy. The backend MUST only use the provenance form when the target's truthiness semantics match the desugared form's semantics for the specific type. When in doubt, emit the desugared form.

**enumerate** — `for i, v in xs` where the index was from `enumerate()`.

| Target | Idiomatic form                        | Notes |
| ------ | ------------------------------------- | ----- |
| Python | `for i, v in enumerate(xs)`           |       |
| Rust   | `for (i, v) in xs.iter().enumerate()` |       |
| Swift  | `for (i, v) in xs.enumerated()`       |       |

Most other backends use a manual counter or their native indexed iteration.

### Multi-statement provenance

These are stamped on the `for` node but the idiomatic form collapses multiple statements (the preceding accumulator `let` + the loop) into one expression.

**list_comprehension / dict_comprehension / set_comprehension**

The desugared pattern (guaranteed by the frontend):
```
let ACC: COLL_TYPE              -- accumulator declaration
for VAR in ITERABLE {
    (if GUARD {)?               -- optional filter
        MUTATE(ACC, EXPR)       -- Append / map insert / Add
    (})?
}
```

The backend recognizes this by:
1. The `for` node has the comprehension provenance tag.
2. The loop body contains exactly one mutation call (possibly inside one `if`).
3. The mutation target is the accumulator declared immediately before the loop.

To emit the comprehension, the backend:
1. Extracts EXPR from the mutation call (second arg of Append, value in map insert, arg of Add).
2. Extracts VAR and ITERABLE from the for node.
3. Extracts GUARD from the if condition, if present.
4. Suppresses the preceding `let` statement.
5. Emits the comprehension as an assignment: `ACC = [EXPR for VAR in ITERABLE (if GUARD)?]`.

Statement suppression requires the backend to look ahead in statement lists. When processing a `let` node, the backend peeks at the next statement; if it's a `for` with comprehension provenance whose body mutates the just-declared variable, the backend suppresses the `let` and defers to the `for` emitter.

| Target | Idiomatic form             |
| ------ | -------------------------- |
| Python | `[expr for x in xs]`       |
| Dart   | `[for (var x in xs) expr]` |

All other backends emit the loop form directly.

**chained_comparison** — `a < b && b < c` with provenance on the `&&` node.

| Target | Idiomatic form |
| ------ | -------------- |
| Python | `a < b < c`    |

The backend pattern-matches the `&&` node: left is `a OP1 b`, right is `b OP2 c`, and `b` is the same expression on both sides. Emits the chained form. Only Python benefits; all other backends emit the `&&` form.

## Target-Specific Idioms

These are patterns the backend recognizes from IR structure alone — no provenance needed. They are target idioms that have no Python source form.

### Negated conditions

```taytsh
if !cond { BODY }
```

| Target | Idiomatic form               |
| ------ | ---------------------------- |
| Ruby   | `unless cond`                |
| Perl   | `unless (cond)`              |
| Others | `if (!cond)` / `if not cond` |

The backend checks if the condition is a `!` expression with no `else` branch.

### Optional unwrapping

```taytsh
if x != nil {
    -- use x (scope.narrowed_type present)
}
```

| Target | Idiomatic form                     |
| ------ | ---------------------------------- |
| Rust   | `if let Some(x) = x { ... }`       |
| Swift  | `if let x = x { ... }`             |
| Go     | direct nil check (no special form) |
| Kotlin | `x?.let { ... }` (if ever added)   |

The backend sees a nil check where the true branch has `scope.narrowed_type` set, and emits the target's optional-binding syntax.

### Optional unwrapping with early return

```taytsh
if x == nil { return ... }
-- use x (scope.narrowed_type present after the if)
```

| Target | Idiomatic form                                    |
| ------ | ------------------------------------------------- |
| Swift  | `guard let x = x else { return ... }`             |
| Go     | `if x == nil { return ... }`                      |
| Rust   | `let Some(x) = x else { return ... };` (let-else) |

The backend sees a nil-eq check whose body is a single return, followed by uses of the variable with a narrowed type.

### Format as template literals

```taytsh
Format("hello, {}", name)
```

| Target     | Idiomatic form                  |
| ---------- | ------------------------------- |
| JavaScript | `` `hello, ${name}` ``          |
| TypeScript | `` `hello, ${name}` ``          |
| Kotlin     | `"hello, $name"`                |
| C#         | `$"hello, {name}"`              |
| Python     | `f"hello, {name}"`              |
| Others     | format function / concatenation |

The backend's `Format` emitter can always choose to emit template/interpolation syntax when the target supports it. This is a per-call decision — no provenance or annotation needed.

### Error handling patterns

Taytsh try/catch maps to fundamentally different mechanisms per target:

| Target | Mechanism                                |
| ------ | ---------------------------------------- |
| C      | setjmp/longjmp or error return codes     |
| Go     | error returns + defer/recover for panics |
| Rust   | Result<T, E> + ? operator                |
| Zig    | error unions + try/catch keywords        |
| Lua    | pcall/xpcall                             |
| Perl   | eval { } / die + $@                      |
| Others | native try/catch/except                  |

Go, Rust, and Zig transform exception semantics into return-value semantics. This is the most complex backend transformation and relies heavily on `callgraph.throws` (to determine the error return type), `returns.needs_named_returns`, `returns.body_has_return`, and `scope.is_modified` for parameters that cross try/catch boundaries.

### Loop forms

Taytsh `for i in range(n)` maps to target-native loop syntax:

| Target     | Emission                              |
| ---------- | ------------------------------------- |
| C          | `for (int i = 0; i < n; i++)`         |
| Go         | `for i := 0; i < n; i++`              |
| Java       | `for (int i = 0; i < n; i++)`         |
| Python     | `for i in range(n)`                   |
| Rust       | `for i in 0..n`                       |
| Ruby       | `(0...n).each do \|i\|`  or `n.times` |
| Swift      | `for i in 0..<n`                      |
| Zig        | `for (0..n) \|i\|` (or while loop)    |
| JavaScript | `for (let i = 0; i < n; i++)`         |
| Others     | C-style for or while equivalent       |

This is purely a backend decision based on the `range` node parameters. No middleend involvement.

### Collection literals

Taytsh map/set literals need target-specific construction:

| Target     | Map literal            | Set literal            |
| ---------- | ---------------------- | ---------------------- |
| Python     | `{"a": 1}`             | `{1, 2, 3}`            |
| JavaScript | `new Map([["a", 1]])`  | `new Set([1, 2, 3])`   |
| Go         | `map[K]V{"a": 1}`      | custom set type        |
| Java       | `Map.of("a", 1)`       | `Set.of(1, 2, 3)`      |
| Rust       | `HashMap::from([...])` | `HashSet::from([...])` |
| C          | custom hash table init | custom set init        |

## Provenance Consumption Summary

Which backends act on each provenance form:

| Provenance         | Backends that use it                 |
| ------------------ | ------------------------------------ |
| chained_comparison | Python                               |
| list_comprehension | Python, Dart                         |
| dict_comprehension | Python, Dart                         |
| set_comprehension  | Python                               |
| in_operator        | Python (operand reversal)            |
| not_in_operator    | Python                               |
| truthiness         | Python, JavaScript, Perl (with care) |
| enumerate          | Python, Rust, Swift                  |
| string_multiply    | Python, Ruby, Perl                   |
| list_multiply      | Python, Ruby                         |
| negative_index     | Python, Ruby, Perl                   |
| open_start         | Python, Go, Rust                     |
| open_end           | Python, Go, Rust                     |

Most provenance forms benefit 1-3 backends. Python benefits from all of them (unsurprising — the source language is Python). Several provenance forms (in_operator for non-Python, string_multiply, list_multiply) are consumed by backends whose `Contains`/`Repeat` emission is already idiomatic, making the provenance tag redundant for them in practice.

## Backend Complexity Profile

Not all backends are equal in complexity. Rough ranking by implementation difficulty:

| Tier        | Backends              | Why                                                          |
| ----------- | --------------------- | ------------------------------------------------------------ |
| Low         | Python, Ruby, Perl    | Dynamic typing, close to source semantics.                   |
|             |                       | Native rune strings — skip strings/hoisting/ownership/       |
|             |                       | callgraph passes.                                            |
| ----------- | --------------------- | ------------------------------------------------------------ |
| Medium      | JavaScript,           | GC, native exceptions, some type ceremony.                   |
|             | TypeScript, PHP,      | Consume strings pass for encoding-aware operations.          |
|             | Dart, Java, C#        | Hoisting needed for C# (break-in-switch).                    |
| ----------- | --------------------- | ------------------------------------------------------------ |
| Medium-High | Lua                   | Native exceptions via pcall but needs hoisting (continue     |
|             |                       | workaround, variable pre-declaration) and callgraph          |
|             |                       | (tail calls). Returns pass for pcall return propagation.     |
| ----------- | --------------------- | ------------------------------------------------------------ |
| High        | Go                    | Error returns from callgraph.throws, variable hoisting,      |
|             |                       | rune conversion, no exceptions, no pattern matching —        |
|             |                       | consumes the most passes of any backend.                     |
| ----------- | --------------------- | ------------------------------------------------------------ |
| High        | Rust, Swift           | Ownership/lifetimes/ARC from ownership pass. Rust needs      |
|             |                       | error-return transformation from callgraph.throws.           |
|             |                       | Swift has native exceptions but needs ARC reasoning.         |
| ----------- | --------------------- | ------------------------------------------------------------ |
| High        | C, Zig                | Manual memory from ownership pass, no exceptions,            |
|             |                       | no GC, no standard collections. C uses setjmp/longjmp;       |
|             |                       | Zig uses error unions from callgraph.throws.                 |
| ----------- | --------------------- | ------------------------------------------------------------ |

### Possible Future Targets

| Tier        | Backends              | Why                                                           |
| ----------- | --------------------- | ------------------------------------------------------------- |
| Medium      | C++                   | Smart pointers map directly to ownership.kind (unique_ptr     |
|             |                       | for owned, shared_ptr for shared, const T& for borrowed).     |
|             |                       | Native exceptions, STL covers all collections. Byte-indexed   |
|             |                       | strings need strings pass. No pattern matching — use          |
|             |                       | std::variant + std::visit for interfaces and unions.          |
| ----------- | --------------------- | ------------------------------------------------------------  |
| Medium      | Scala                 | JVM — GC, native exceptions. Excellent pattern matching       |
|             |                       | (case classes map to Taytsh interfaces, match is exhaustive). |
|             |                       | val/var from scope.is_const. Option[T] for T?. UTF-16         |
|             |                       | strings need strings pass for BMP/unknown distinction.        |
| ----------- | --------------------- | ------------------------------------------------------------  |
| Medium-High | OCaml                 | GC, native exceptions (raise/try...with), excellent pattern   |
|             |                       | matching via variants. Mutability inversion: every reassigned |
|             |                       | binding needs ref/!/:= — scope.is_const is critical.          |
|             |                       | Byte-indexed strings need strings pass. Mutable record        |
|             |                       | fields and Hashtbl/Array for reference semantics.             |
| ----------- | --------------------- | ------------------------------------------------------------  |
| Absurd      | Bash                  | No types, no structs, no floats without forking bc/awk.       |
|             |                       | ID-based object system with global variables for fields.      |
|             |                       | Function returns via global __retval (subshells break         |
|             |                       | reference semantics). Error propagation via global error      |
|             |                       | state + return codes. Needs callgraph (error returns),        |
|             |                       | hoisting (function-scoped locals), strings (byte-indexed).    |
| ----------- | --------------------- | ------------------------------------------------------------  |
