# Tongues Backend Specification

A Tongues backend walks the annotated Taytsh IR and emits source code for one target language. It reads the IR tree and its annotations (written by the lowerer and middleend passes). It never modifies the IR.

## Backend Structure

Each backend is a tree walker. For each IR node type, the backend has an emitter that decides how to render that node. The emitter may:

1. Emit the node's desugared form directly (always correct).
2. Check provenance annotations and emit an idiomatic alternative.
3. Read annotations to refine the emission (const, unused, etc.).
4. Recognize structural patterns in the IR and emit target idioms.

These are not phases — they happen together during the single tree walk. The emitter for a `for` node might check provenance, read scope annotations, and pattern-match the body, all in one method.

## Declaration Ordering

The Taytsh IR has forward-reference semantics: all top-level declarations are visible throughout the module regardless of position (see `12-taytsh-ir-spec.md`, Module Structure). Target languages do not necessarily share this property. Backends emitting languages where class/struct declarations are evaluated top-to-bottom must emit declarations in dependency order.

### Dependency graph

Nodes are top-level declarations: structs, interfaces, enums, and top-level lets.

Edges: declaration A depends on declaration B when:

- A implements B (struct `: Interface`)
- A has a field whose type mentions B (directly or as a type parameter)
- A is a top-level `let` whose initializer references B

Functions are excluded — function bodies are not evaluated at definition time in any target, and function signatures referencing not-yet-defined types are legal in every target Tongues supports.

### Algorithm

Topologically sort declarations so that dependencies are emitted before dependents. When no dependency exists between two declarations, preserve the original IR order for determinism.

Cycles between struct field types are possible (`struct A { b: B? }`, `struct B { a: A? }`). These references are always through optional, collection, or function types — a struct cannot directly contain itself. All target languages handle this via reference/pointer semantics. When a cycle exists, the backend breaks it arbitrarily; the circular references work regardless of order because the types are indirect.

### Target classification

| Ordering     | Targets                                         | Reason                                                               |
| ------------ | ----------------------------------------------- | -------------------------------------------------------------------- |
| Not needed   | Go, Rust, Swift, Zig, C#, Dart, Java            | All declarations within a file/package/module are mutually visible   |
| Required     | Python, JavaScript, TypeScript, Ruby, Perl, Lua | Class/struct declarations are evaluated top-to-bottom                |
| Forward decl | C                                               | Requires forward declarations (`struct Foo;`) before first reference |

### Target-specific notes

**Python** — Class bases and `@dataclass` field annotations are evaluated at class definition time. Dependency ordering alone is sufficient; `from __future__ import annotations` is not needed because the topological sort guarantees all referenced types exist before use.

**JavaScript / TypeScript** — `class` declarations are not hoisted. A class extending another must appear after the base class definition.

**Ruby** — A class inheriting from another must appear after the parent. Mixins and reopened classes are not used by Tongues, so simple dependency order suffices.

**Perl** — Package definitions are evaluated top-to-bottom. Parent packages must be defined before `use parent`.

**Lua** — All definitions (`local X = ...`) are sequential. Metatables and class-emulating tables must be defined before use.

**C** — Emit forward declarations for all struct types at the top of the file, then emit full definitions in dependency order. This two-pass approach avoids issues with mutually recursive struct pointers.

### Interaction with project compilation

The project merge phase (`04a-project-and-imports.md`) sorts files by import dependencies, not by type dependencies. Two structs in different files may have a type dependency without an import dependency (because all names become visible after merge). The backend's declaration ordering pass is therefore necessary even when the project merge has already run.

## Idiomatic Output

Every backend emits unconditionally idiomatic output by default. `for i in range(n)` becomes `for i in 0..n` in Rust, `for (int i = 0; i < n; i++)` in Java, `map[K]V{"a": 1}` becomes `HashMap::from(...)` in Rust — regardless of annotations. This is the backend's native translation of IR constructs into the target's natural forms.

Provenance annotations are an upgrade path. The lowerer desugars Python idioms into simpler IR — `x[-1]` becomes `x[Len(x) - 1]`, `[x*2 for x in xs]` becomes a loop with `Append`. Without provenance, the backend emits the desugared form literally in target-native syntax, which is correct and already idiomatic. With provenance, backends that have a corresponding higher-level form can raise the output: Python reconstructs `x[-1]`, Go emits `xs[:n]` instead of `xs[0:n]`, Dart emits collection-for instead of a loop.

Without provenance, emit the IR literally in target-native syntax. With provenance, reconstruct the higher-level idiom when the target supports it.

## Annotation Consumption

### scope.\*

Used by all backends.

`scope.is_const` — emit immutable bindings where the target supports them (`const`, `final`, `let` vs `var`, etc.). Targets without const locals (Lua, Perl, PHP, Python, Ruby) ignore this annotation.

`scope.is_unused` — suppress unused parameter warnings in targets that enforce them (Go, Rust, Zig, C). Others ignore.

`scope.is_modified` — informs pass-by-value targets whether a parameter's value is mutated. Backends for value-semantics languages (Go, Rust, Swift, C, Zig) use this to decide pointer/reference passing. If `scope.is_modified=true` and the parameter is a struct, the backend may need to pass by pointer.

`scope.narrowed_type` — backends that require explicit casts or type assertions can skip them when narrowing has already been proven. Go emits the type assertion result directly; Rust skips `unwrap()`; Java skips the cast.

`scope.is_interface` — backends that need explicit downcasting (Go, Java, C#) use this to know when a variable holds an interface-typed value that will need a type assertion at use sites.

`scope.is_function_ref` — backends where function references need special syntax (Go is direct, Python is direct, Java needs method references, C needs function pointers) use this to emit the right form.

`scope.case_interface` — Go uses this in type-switch emission to choose the right type assertion. When a case binding is used through interface methods, Go emits `v := expr.(InterfaceName)` instead of `v := expr.(ConcreteType)`.

### returns.\*

Used by a subset of backends.

`returns.needs_named_returns` — Go only. When a try/catch body contains returns, Go must use named return values so that deferred recover can set them.

`returns.may_return_nil` — Dart uses this to emit `dynamic` or nullable return types. Go uses this for pointer returns. Java/C# may add `@Nullable`.

`returns.body_has_return` — Lua (pcall wrapper needs to propagate return values via flags), Perl (eval block needs similar), Zig (errdefer/error union return propagation).

`returns.always_returns` — all backends can use this to suppress "missing return" warnings or omit unreachable default returns.

### liveness.\*

Used by all backends for cleaner output.

`liveness.initial_value_unused` — suppress the initializer. Emit `let x: int` instead of `let x: int = 0` when the zero value is immediately overwritten. Go emits `var x int` instead of `x := 0`.

`liveness.catch_var_unused` — emit anonymous catch. Java: `catch (Exception _)`. Go: `_ = err`. Python: `except ValueError:` (no `as e`).

`liveness.match_var_unused` — emit pattern without binding. Rust: `Foo(_)`. Go: `case Foo:` without assignment.

`liveness.tuple_unused_indices` — emit `_` for unused targets. Go: `_, b := f()`. Python: `_, b = f()`. Rust: `let (_, b) = f();`.

### strings.\*

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

### hoisting.\*

Used by Go, Lua, Perl, and C#.

`hoisting.hoisted_vars` — Go emits `var` declarations before the control structure. Lua emits `local` declarations before the block.

`hoisting.func_hoisted_vars` — Perl reads this to emit `my` declarations at function scope. Perl's block-scoped `my` declarations inside control-flow blocks are invisible to sibling blocks, so all hoisted variables must be pre-declared at function entry.

`hoisting.has_continue` — Lua emits `goto continue_label` with a label at the loop end, since Lua lacks native `continue` (before 5.2) or uses `repeat until true` wrapping.

`hoisting.has_break` — C# uses this for match/type-switch nodes emitted as native `switch`/`case`. When `true`, C# emits a flag variable to propagate `break` past the switch to the enclosing loop, since C#'s `break` exits the switch rather than the loop.

`hoisting.rune_vars` — Go emits `xRunes := []rune(x)` at function entry for string variables that are indexed, then uses `xRunes[i]` at index sites. When the strings pass is active, `hoisting.rune_vars` is derived from `strings.indexed` — only bindings with `strings.indexed=true` and `strings.content!="ascii"` need rune conversion.

### ownership.\*

Used by C, Rust, Zig, Swift.

`ownership.kind` — determines how values are passed and stored. C: owned values are freed by the current scope; borrowed are not. Rust: owned uses move semantics; borrowed uses `&`/`&mut`. Zig: similar to C with explicit allocator patterns. Swift: informs ARC behavior.

`ownership.escapes` — C: emit `strdup()` for escaping strings, `memcpy` for escaping structs. Rust: forces a `.clone()`. Zig: allocator copy.

`ownership.region` — C: determines which scope calls `free()`. Rust: informs lifetime annotations.

### callgraph.\*

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

**in_operator / not_in_operator** — `Contains(xs, v)` with provenance. Mainly useful for Python (`v in xs`, operand order reversal). Most other backends already emit `Contains` idiomatically regardless of provenance.

**open_start / open_end** — slice with `0` or `Len(x)` bound. Backends that support open-ended slices (Python `xs[:n]`, Go `xs[:n]`, Rust `&xs[..n]`) omit the redundant bound. Others emit the arithmetic form.

**negative_index** — `x[Len(x) - n]` with provenance. Targets with native negative indexing (Python, Ruby, Perl) emit `x[-n]` directly. Others emit the arithmetic form.

**string_multiply / list_multiply** — `Repeat(s, n)` with provenance. Targets with native repetition operators (Python `s * n`, Ruby `s * n`, Perl `$s x $n`) use them. Others emit their Repeat implementation.

**truthiness** — `Len(xs) > 0` or `s != ""` with provenance.

| Target     | Idiomatic form   | Notes                                       |
| ---------- | ---------------- | ------------------------------------------- |
| Python     | `if xs:`         | truthy check                                |
| Ruby       | `if xs`          | but empty collections are truthy in Ruby!   |
| JavaScript | `if (xs.length)` | or `if (s)` for strings                     |
| Perl       | `if (@xs)`       | or `if ($s)`                                |
| Lua        | n/a              | tables are always truthy; ignore provenance |

**Caution**: Ruby and Lua have different truthiness rules than Python. Ruby treats empty arrays/hashes as truthy. Lua treats tables as truthy. The backend MUST only use the provenance form when the target's truthiness semantics match the desugared form's semantics for the specific type. When in doubt, emit the desugared form.

**enumerate** — `for i, v in xs` where the index was from `enumerate()`. Targets with native enumerate (Python, Rust `.iter().enumerate()`, Swift `.enumerated()`) reconstruct the idiomatic form. Others use a manual counter.

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
|             |                       | Function returns via global \_\_retval (subshells break       |
|             |                       | reference semantics). Error propagation via global error      |
|             |                       | state + return codes. Needs callgraph (error returns),        |
|             |                       | hoisting (function-scoped locals), strings (byte-indexed).    |
| ----------- | --------------------- | ------------------------------------------------------------  |
