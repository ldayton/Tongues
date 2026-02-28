# Taytsh Type Checking

**Module:** `taytsh/src/check.py`

Standalone type checker for Taytsh IR. Validates a parsed Taytsh module against the type system defined in the IR spec, enforcing type safety, scoping rules, exhaustive matching, and no-closure invariants. Operates on the parsed AST produced by `taytsh/src/parse.py` — no Python source or frontend tables are involved. This is independent from the frontend's pycheck phase (spec 09), which validates the Python-to-IR translation; this phase validates the IR itself.

## Inputs

- **Module**: parsed Taytsh AST from `taytsh/src/parse.py` — a list of top-level declarations (functions, structs, interfaces, enums)

## Type Representation

Types are resolved from parse-time AST type nodes into checked type objects during the first pass. All subsequent checking uses resolved types.

### Primitives

| Kind     | Singleton  | Description        |
| -------- | ---------- | ------------------ |
| `int`    | `INT_T`    | signed integer     |
| `float`  | `FLOAT_T`  | IEEE 754 binary64  |
| `bool`   | `BOOL_T`   | boolean            |
| `byte`   | `BYTE_T`   | unsigned 8-bit     |
| `bytes`  | `BYTES_T`  | byte sequence      |
| `string` | `STRING_T` | rune sequence      |
| `rune`   | `RUNE_T`   | Unicode character  |
| `nil`    | `NIL_T`    | nil value/type     |
| `void`   | `VOID_T`   | return-type marker |

### Compound Types

| Type Class   | Fields                                | Taytsh Syntax   |
| ------------ | ------------------------------------- | --------------- |
| `ListT`      | `element`                             | `list[T]`       |
| `MapT`       | `key`, `value`                        | `map[K, V]`     |
| `SetT`       | `element`                             | `set[T]`        |
| `TupleT`     | `elements`                            | `(A, B, C)`     |
| `FnT`        | `params`, `ret`                       | `fn[A, B, R]`   |
| `StructT`    | `name`, `fields`, `methods`, `parent` | `StructName`    |
| `InterfaceT` | `name`, `variants`                    | `InterfaceName` |
| `EnumT`      | `name`, `variants`                    | `EnumName`      |
| `UnionT`     | `members`                             | `A \| B`        |

### `void` Restrictions

`void` is valid only as a function return type (after `->`) or as the last element of an `fn[...]` type. It cannot appear as a variable type, parameter type, collection element type, tuple element type, union member, or optional base.

## Type Resolution

Parse-time type AST nodes are resolved to checked types during declaration collection:

| AST Type Node  | Resolved To                                         |
| -------------- | --------------------------------------------------- |
| Primitive name | corresponding singleton                             |
| `list[T]`      | `ListT(resolve(T))`                                 |
| `map[K, V]`    | `MapT(resolve(K), resolve(V))` — K must be hashable |
| `set[T]`       | `SetT(resolve(T))` — T must be hashable             |
| `(A, B, ...)`  | `TupleT([resolve(A), resolve(B), ...])`             |
| `fn[A, B, R]`  | `FnT([resolve(A), resolve(B)], resolve(R))`         |
| `T?`           | `UnionT([resolve(T), NIL_T])` (normalized)          |
| `A \| B`       | `UnionT(normalize([resolve(A), resolve(B)]))`       |
| `IDENT`        | struct, interface, or enum lookup                   |

### Union Normalization

1. **Flatten**: nested unions are expanded (`A | (B | C)` → `A | B | C`)
2. **Deduplicate**: identical members are collapsed (`A | A` → `A`)
3. **Reduce**: a single remaining member becomes that type (not a 1-element union)

### Double Optional

`T??` is rejected in source-level type annotations — optionals do not nest. Computed types (e.g. `Get` on `map[K, V?]` returning `V?`) normalize via union flattening and deduplication, which collapses the redundant `nil` without error.

### Tuple Arity

Tuple types require at least two elements — `(T)` is not a valid type.

### Hashability

Map keys and set elements must be hashable types:

| Hashable                          | Not Hashable                    |
| --------------------------------- | ------------------------------- |
| `int`, `float`, `bool`            | `list[T]`, `map[K,V]`, `set[T]` |
| `byte`, `bytes`, `string`, `rune` | structs, interfaces             |
| enums                             | function types                  |
| tuples of hashable types          | unions                          |

## Two-Pass Checking

### Pass 1: Declaration Collection

Three sequential loops register all declarations before checking bodies:

**Loop 1 — Register names**: every struct, interface, and enum name is registered as a placeholder in the type namespace. Duplicate names are errors.

**Loop 2 — Resolve structure**: for each declaration:
- **Structs**: resolve field types and method signatures; if `: Name` is present, validate that `Name` refers to an interface (not a struct, enum, or undefined name — error: `parent is not an interface`) and register as a variant of that interface
- **Interfaces**: validate the body is empty; record as sealed interface
- **Enums**: record variant names; duplicate variants are errors

**Loop 3 — Register functions**: resolve all top-level function signatures (parameter types, return types). Record parameter names for named argument support.

### Pass 2: Body Checking

For each function declaration, a new scope is entered, parameters are declared, and the body is checked statement by statement. For each struct, methods are checked with `this` bound to the enclosing struct type.

## Scoping

Lexical block scoping with no shadowing. The checker maintains a scope stack.

### Scope Operations

| Operation     | Effect                                                  |
| ------------- | ------------------------------------------------------- |
| `enter_scope` | push new scope frame                                    |
| `exit_scope`  | pop current scope frame                                 |
| `declare`     | bind name in current scope                              |
| `lookup`      | search scopes innermost-out, then functions, then types |

### Binding Rules

| Condition                      | Result                               |
| ------------------------------ | ------------------------------------ |
| Name is reserved               | error: `reserved name`               |
| Name exists in current scope   | error: `shadows outer binding`       |
| Name exists in any outer scope | error: `shadows outer binding`       |
| Name is `_`                    | silently discarded, never registered |

The `_` discard identifier may appear multiple times in bindings (e.g., `for _, _ in m`) and is never accessible as a variable.

### Scope Boundaries

| Construct       | Creates Scope | Bindings Scoped To |
| --------------- | ------------- | ------------------ |
| Function body   | yes           | function           |
| `if` / `else`   | yes           | branch block       |
| `while`         | yes           | loop body          |
| `for`           | yes           | loop body          |
| `match` case    | yes           | case body          |
| `catch`         | yes           | catch body         |
| `try`/`finally` | yes           | block body         |

Loop variables, match bindings, and catch bindings are scoped to their block — they are not accessible after the construct ends.

### Reserved Names

Built-in function names are reserved and cannot be used for any binding — local variables, parameters, loop variables, match bindings, catch bindings, struct names, enum names, interface names, or function names.

Reserved names include: `Abs`, `Min`, `Max`, `Sum`, `Pow`, `Round`, `Floor`, `Ceil`, `Sqrt`, `DivMod`, `Len`, `Concat`, `Append`, `Insert`, `Pop`, `RemoveAt`, `IndexOf`, `Contains`, `Reversed`, `Sorted`, `RangeList`, `Map`, `Get`, `Delete`, `Keys`, `Values`, `Items`, `Merge`, `Set`, `Remove`, `Union`, `Intersection`, `Difference`, `ToString`, `Format`, `Encode`, `Decode`, `Bytes`, `BytesFrom`, `Split`, `SplitN`, `SplitWhitespace`, `Join`, `Find`, `RFind`, `Count`, `Replace`, `Repeat`, `Reverse`, `StartsWith`, `EndsWith`, `Upper`, `Lower`, `Trim`, `TrimStart`, `TrimEnd`, `RuneFromInt`, `RuneToInt`, `ParseInt`, `ParseFloat`, `FormatInt`, `IntToFloat`, `FloatToInt`, `ByteToInt`, `IntToByte`, `IsDigit`, `IsAlpha`, `IsAlnum`, `IsSpace`, `IsUpper`, `IsLower`, `IsType`, `WriteOut`, `WriteErr`, `WritelnOut`, `WritelnErr`, `ReadLine`, `ReadAll`, `ReadBytes`, `ReadBytesN`, `ReadFile`, `ReadFileBytes`, `WriteFile`, `Args`, `GetEnv`, `Exit`, `Assert`, `Unwrap`, `IsNaN`, `IsInf`, `WrappingAdd`, `WrappingSub`, `WrappingMul`.

Exception: `Add` is a built-in but not reserved — it can be used as a binding name.

Exception: `ToString` is reserved for bindings, but structs may declare a `fn ToString(this) -> string` method to override the default representation.

### Top-Level vs Local

Top-level names (functions, types) and local names occupy disjoint namespaces. A local variable may share a name with a top-level function or type — the local binding takes precedence within its scope.

## Assignability

A type `source` is assignable to `target` when:

| Rule                | Condition                                                    |
| ------------------- | ------------------------------------------------------------ |
| Identity            | types are structurally equal                                 |
| Struct → Interface  | source is a struct whose `parent` matches the interface name |
| `nil` → Optional    | source is `nil` and target contains `nil`                    |
| Type → Union member | source is assignable to any member of target union           |
| Union → Union       | every member of source is assignable to target               |
| Union → Single type | all source members are assignable to target                  |

Collections, tuples, and function types use structural equality — no covariance or contravariance. `list[Dog]` is not assignable to `list[Animal]`.

No implicit numeric coercion: `bool` is not assignable to `int`, `int` is not assignable to `float`. All conversions require explicit function calls (`IntToFloat`, `FloatToInt`, etc.).

## Zero Values

Types with zero values can omit the initializer in `let` declarations:

| Type            | Zero Value  | Has Zero Value                        |
| --------------- | ----------- | ------------------------------------- |
| `int`           | `0`         | yes                                   |
| `float`         | `0.0`       | yes                                   |
| `bool`          | `false`     | yes                                   |
| `byte`          | `0x00`      | yes                                   |
| `bytes`         | `b""`       | yes                                   |
| `string`        | `""`        | yes                                   |
| `rune`          | `'\0'`      | yes                                   |
| `list[T]`       | `[]`        | yes                                   |
| `map[K, V]`     | `Map()`     | yes                                   |
| `set[T]`        | `Set()`     | yes                                   |
| `(A, B, ...)`   | per-element | only if all elements have zero values |
| `T?`            | `nil`       | yes                                   |
| `A \| B \| nil` | `nil`       | yes (if contains nil)                 |
| struct          | —           | no                                    |
| interface       | —           | no                                    |
| enum            | —           | no                                    |
| `fn[...]`       | —           | no                                    |
| `A \| B`        | —           | no (without nil)                      |

For types without a zero value, omitting the initializer is permitted but the variable is **uninitialized** until assigned. Reading an uninitialized variable is a compile error: `variable used before assignment`.

### Definite Assignment

Variables declared without an initializer for types without zero values are **uninitialized** at their declaration point. The checker tracks uninitialized variables and enforces:

1. **Read check**: Reading an uninitialized variable is an error: `variable used before assignment`.
2. **Assignment**: Assigning to an uninitialized variable marks it as initialized.
3. **Branching**: After `if`/`else`, a variable is initialized only if it was initialized in **all** branches. After `if` without `else`, the variable remains uninitialized unless the then-body always exits.
4. **Loops**: Variables initialized only inside a loop body are still considered uninitialized after the loop (the body might not execute).
5. **Match**: Like if/else — initialized only if initialized in all cases (when exhaustive).

## Statement Checking

### `let`

Validates the initializer (if present) is assignable to the declared type. If no initializer is provided, the variable is recorded as **uninitialized** for types without zero values. For types with zero values, the checker implicitly initializes to the zero value. For types without zero values, the variable remains uninitialized — any read before assignment is an error.

### Assignment

Validates the target is a valid lvalue (variable, field access, or index expression). Validates the source is assignable to the target's type.

| Invalid Target | Diagnostic                       |
| -------------- | -------------------------------- |
| Literal        | `invalid assignment target`      |
| Binary expr    | `invalid assignment target`      |
| Call result    | `invalid assignment target`      |
| `this`         | `cannot assign to this`          |
| Tuple element  | `cannot assign to tuple element` |
| String index   | `cannot assign to string index`  |
| Bytes index    | `cannot assign to bytes index`   |
| Slice expr     | `cannot assign to slice`         |

### Compound Assignment

Compound `X=` is valid iff the binary operator `X` accepts the target's type per the binary operator table and the result is assignable back to the target.

### Tuple Assignment

Validates the right-hand side is a tuple type, the arity matches the number of targets, and each element is assignable to the corresponding target.

### Expression Statements

Bare expression statements (without assignment) must be function calls or method calls. Non-call expressions as statements are rejected: `expression has no effect`.

### Return

In a non-void function, validates the return expression's type is assignable to the function's declared return type. In a void function, rejects return with a value. Bare `return` in a non-void function is `missing return value`.

### Return-Path Completeness

Every non-void function must return or throw on all control-flow paths. A block is *complete* if it ends with a `return`, `throw`, or a construct whose branches are all complete (e.g. an `if`/`else` where both branches are complete, a `match` that is exhaustive and every case is complete, or a `try`/`catch` where the try body and all catch bodies are complete). A `while` with a literal `true` condition and no reachable `break` is complete — control never exits normally. "Literal `true`" means the token `true`, not a constant expression that evaluates to true. "Reachable `break`" means syntactically reachable — a `break` inside `if false { break }` is still reachable because the checker does not evaluate conditions. The checker performs no constant folding or constant propagation. A non-void function whose body is not complete produces: `not all paths return`.

### If

Validates the condition is `bool`. Performs nil narrowing and IsType narrowing (see Nil Narrowing and IsType Narrowing). Narrowing creates a new scope with the variable rebound to the narrowed type.

### While

Validates the condition is `bool`. Checks body in loop context (enabling `break`/`continue`).

### For

**Range form**: all arguments must be `int`; the loop variable is declared as `int`.

**Collection form**: the iterable must be a collection type. Binds loop variables according to the iterable:

| Iterable    | One variable | Two variables      |
| ----------- | ------------ | ------------------ |
| `list[T]`   | `v: T`       | `i: int, v: T`     |
| `string`    | `ch: rune`   | `i: int, ch: rune` |
| `bytes`     | `b: byte`    | `i: int, b: byte`  |
| `map[K, V]` | `k: K`       | `k: K, v: V`       |
| `set[T]`    | `v: T`       | error: not allowed |

Union and optional types are not iterable — the iterable expression must be a concrete collection type. `cannot iterate union`.

Loop variables are read-only — assigning to a loop variable is rejected: `cannot assign to loop variable`.

### Unreachable Code

Statements after `return`, `throw`, `break`, or `continue` within the same block are rejected: `unreachable code`.

### Break / Continue

Valid only inside a loop. Outside a loop: `break outside of loop` / `continue outside of loop`.

### Throw

Type-checks the expression. A type is throwable if it is a struct type, an interface type, or a union whose members are all throwable. Primitives, collections, `nil`, and function types are not throwable: `cannot throw T`.

### Try / Catch

Checks the try body. For each catch clause:
- If typed, validates that every type is a struct — for union catches (`catch e: A | B`), each member must be a struct
- A typed catch naming a type already covered by a preceding catch is rejected: `duplicate catch`
- An untyped catch is a catch-all; subsequent catches are rejected: `unreachable catch`
- A catch-all binding's type is the residual: the union of all struct types in the program not covered by preceding typed catches
- Catch bindings are scoped to the catch block

A `finally` block must not contain `return`, `throw`, `break`, or `continue`. These are rejected: `control flow in finally`.

Checks the finally body if present.

## Expression Checking

### Literals

| Literal   | Type     | Notes                                             |
| --------- | -------- | ------------------------------------------------- |
| `42`      | `int`    |                                                   |
| `3.14`    | `float`  |                                                   |
| `true`    | `bool`   |                                                   |
| `"hello"` | `string` |                                                   |
| `'λ'`     | `rune`   |                                                   |
| `b"\x89"` | `bytes`  |                                                   |
| `0xff`    | `byte`   | if expected type is `int`, produces `int` instead |
| `nil`     | `nil`    |                                                   |

**Integer range**: integer literals outside the range [-2^63, 2^63) are rejected: `integer literal too large`.

**Byte literal context**: a `0x` hex literal produces `byte` by default. It produces `int` instead only when the expected type is `int` — i.e. when it appears as the initializer of a `let` with declared type `int`, as an argument to a parameter typed `int`, the right-hand side of an assignment to a target typed `int`, or a return expression in a function returning `int`. In all other contexts (bare expressions, operator operands), the type is `byte`.

### Variables

Lookup through scopes (innermost-out), then functions, then types. Undefined name: `undefined name 'x'`.

### `this`

`this` is valid only inside a method body. Using `this` outside a method is rejected: `this outside method`.

### Binary Operators

| Category   | Operators               | Operand Types                                              | Result    |
| ---------- | ----------------------- | ---------------------------------------------------------- | --------- |
| Logical    | `&&`, `\|\|`            | `bool`, `bool`                                             | `bool`    |
| Equality   | `==`, `!=`              | identical types; rejected if type transitively contains fn | `bool`    |
| Ordering   | `<`, `<=`, `>`, `>=`    | same type: `int`, `float`, `byte`, `rune`, `string`        | `bool`    |
| Arithmetic | `+`, `-`, `*`, `/`, `%` | same type: `int`, `float`, `byte`                          | same type |
| Bitwise    | `&`, `\|`, `^`          | same type: `int`, `byte`                                   | same type |
| Shift      | `<<`, `>>`              | left: `int`/`byte`, right: `int`                           | left type |
| Shift      | `>>>`                   | left: `int`/`byte`, right: `int`; strict math only         | left type |

For `==` and `!=`, `nil` widens to the other operand's type. `x == nil` is valid when `x` is `T?` or a union containing `nil`.

Equality on function types is rejected: `equality not defined for fn`.

Unions and optionals are rejected for ordering, arithmetic, bitwise, and shift operations: `not defined for union`.

`>>>` (logical right shift) is available only when the module has the `strict_math` annotation. Using `>>>` without strict math is rejected: `strict math only`.

### Unary Operators

| Operator | Operand Types          | Result    |
| -------- | ---------------------- | --------- |
| `-`      | `int`, `float`, `byte` | same type |
| `!`      | `bool`                 | `bool`    |
| `~`      | `int`, `byte`          | same type |

### Ternary

Condition must be `bool`. If both branches have the same type, the result is that type. Otherwise, the result is the normalized union of both branch types (e.g., branches of `int` and `nil` produce `int?`).

### Field Access

- **Enum variant**: `EnumName.Variant` — validated against the enum's variant list
- **Struct field**: `expr.field` — validated against the struct's field map
- **Struct method**: `expr.method` — validated against the struct's method map (used in call context)
- **Tuple access**: `expr.N` — bounds-checked against tuple length
- **Interface field access**: rejected — interfaces have no fields or methods; use `match` to narrow to a concrete struct first: `cannot access field on interface`
- **Union field access**: requires all union members to have the field with the same type; `nil` members block field access

### Indexing

| Collection | Index Type | Result Type |
| ---------- | ---------- | ----------- |
| `list[T]`  | `int`      | `T`         |
| `string`   | `int`      | `rune`      |
| `bytes`    | `int`      | `byte`      |
| `map[K,V]` | `K`        | `V`         |

Union types cannot be indexed: `cannot index union`.

### Slicing

Both bounds must be `int`. Slicing preserves the collection type: `list[T][a:b]` → `list[T]`, `string[a:b]` → `string`, `bytes[a:b]` → `bytes`.

### Function Calls

**User functions**: argument count checked against parameter count; each argument type checked for assignability to the corresponding parameter type.

**Named arguments**: all arguments must be named or all positional — mixing is rejected (`cannot mix positional and named`). Named arguments are resolved by name and may appear in any order. Unknown parameter names are rejected (`no parameter 'c'`). Named arguments are valid only for user-defined functions and struct constructors. Built-in functions and function-value calls do not support named arguments.

**Struct constructors**: positional or named field arguments. Positional arguments follow field declaration order. Named arguments use `field: value` syntax. All fields must be provided.

**Method calls**: validated on the receiver struct. `this` is implicitly bound.

**Function value calls**: variables typed as `fn[A, R]` are callable; argument count and types are checked.

**Non-callable**: calling a non-function type is rejected (`cannot call int`).

### Collection Literals

**Context type**: a context type is available when a collection literal appears as: the initializer of a `let` with a declared type, an argument to a parameter with a known type, the right-hand side of an assignment to a typed target, or a return expression in a function with a declared return type.

**List literals**: with a context type `list[T]`, each element is checked for assignability to `T`. Without a context type, all elements must have exactly the same type (strict equality, not assignability) and that type becomes the element type. Empty lists require a context type; empty lists without one are rejected: `cannot infer type`.

**Map literals**: with a context type `map[K, V]`, each key is checked for assignability to `K` and each value to `V`. Without a context type, all keys must have the same type and all values must have the same type (strict equality). Empty maps require a context type. Duplicate keys are rejected: `duplicate key`.

**Set literals**: with a context type `set[T]`, each element is checked for assignability to `T`. Without a context type, all elements must have exactly the same type (strict equality, not assignability). Empty sets require a context type.

**Tuple literals**: each element is independently typed. Context types (from `let` declarations or parameters) propagate to elements — each element is checked for assignability to the corresponding context element type.

### Function Literals

Function literals use arrow syntax with explicit parameter types and return type. Block body (`{ }`) or expression body (`=>`).

**No closures**: function literal bodies may only reference their own parameters, top-level functions, registered types, and built-in names. Referencing a variable from an enclosing scope is rejected: `cannot capture 'name'`.

The closure check scans the literal body for variable references, tracking which names are locally bound (parameters, let bindings, for loop variables, match/catch bindings). Any reference not resolvable to a local binding, top-level declaration, or built-in is a capture error.

**Bound methods**: `instance.Method` used as a value is rejected because binding `this` is implicitly capturing state: `cannot capture 'this'`.

**Return-path completeness**: non-void function literals with block bodies are subject to the same return-path rule as named functions — all control-flow paths must return or throw.

## Nil Narrowing

The checker performs nil narrowing in `if` statements. When the condition is `x != nil` or `x == nil`:

| Condition  | Then-branch type    | Else-branch type    |
| ---------- | ------------------- | ------------------- |
| `x != nil` | non-nil part of `x` | (unchanged)         |
| `x == nil` | (unchanged)         | non-nil part of `x` |

"Non-nil part" means:
- `T?` narrows to `T`
- `A | B | nil` narrows to `A | B`
- `A | nil` narrows to `A`

Narrowing creates a new scope in the appropriate branch with the variable rebound to the narrowed type.

## IsType Narrowing

`IsType(expr, "TypeName")` is an expression-level type test that returns `bool`. When it appears in a condition, the checker narrows the tested variable to the named struct type in scopes where the check is known to hold. This complements `match` — it handles contexts where a statement-level pattern match cannot apply.

### Narrowable Types

`IsType` narrows when the tested expression's current type is:
- An **interface** whose variants include the named type
- A **union** containing an interface whose variants include the named type (e.g. `Node?` where `Node` has variant `Lit`)

If the type name does not resolve to a known variant, no narrowing occurs.

### Narrowing Contexts

| Context | Example | Narrowed Scope |
| --- | --- | --- |
| `if` then-branch | `if IsType(n, "Lit") { n.value }` | then-block |
| `if` else-branch (negated) | `if !IsType(n, "Lit") { } else { n.value }` | else-block |
| Guard pattern | `if !IsType(n, "Lit") { return }` then `n.value` | statements after the `if` |
| `&&` right operand | `IsType(n, "Lit") && n.value > 0` | right side of `&&` |
| Ternary then-branch | `IsType(n, "Lit") ? n.value : 0` | then-expression |

Guard narrowing applies when the `if` body always exits (via `return` or `throw`) and the condition is a negated `IsType`. After the `if`, the variable is narrowed to the tested type.

### Field Access Paths

`IsType` supports dotted paths as well as simple variables:

```
if IsType(c.func, "Var") {
    return c.func.name
}
```

The narrowed path `c.func` is tracked and subsequent field accesses through it resolve against the narrowed type.

### Relationship to match

`match` and `IsType` are complementary:
- `match` is a statement that exhaustively dispatches on all variants — use it for full type dispatch
- `IsType` is an expression that tests a single variant — use it in boolean combinations, ternary expressions, guard patterns, and field-access subjects where `match` cannot apply

## Match Checking

`match` dispatches on the runtime type of a value. The checker validates exhaustiveness, case validity, and binding types.

### Matchable Types

Only certain types can be the scrutinee of a `match`:

| Type       | Matchable | Cases Match Against          |
| ---------- | --------- | ---------------------------- |
| Interface  | yes       | implementing struct types    |
| Enum       | yes       | `EnumName.Variant` values    |
| Union      | yes       | member types                 |
| Optional   | yes       | inner type + `nil`           |
| Primitive  | no        | `cannot match on int`        |
| Collection | no        | `cannot match on list`       |
| Struct     | no        | `cannot match on StructName` |

### Case Validation

| Case Form           | Validates                                  |
| ------------------- | ------------------------------------------ |
| `case v: Type`      | `Type` is a member of the scrutinee's type |
| `case Enum.Variant` | variant exists in the enum                 |
| `case nil`          | scrutinee contains `nil`                   |

Invalid cases produce errors:
- `not a member` — case type not in union
- `not a variant` — struct not implementing the matched interface
- `not a variant of EnumName` — enum variant from wrong enum
- `duplicate case` — type or variant already covered
- `default must be last` — default not in final position

### Exhaustiveness

All possible types or values must be covered by cases, or `default` must be present. Non-exhaustive matches produce: `non-exhaustive match`.

For interfaces, all registered implementing structs must be covered. Covering all individual variants of an interface in a union satisfies the interface member.

For enums, all variants must be covered.

For unions, each member type must be covered. If a union contains an interface, either the interface itself or all its variants must be covered.

For optionals, both the value type and `nil` must be covered.

### Default Binding

`default` with a binding name (`default o { ... }`) binds the value to the residual type — the union of types not covered by preceding cases. For enums, the residual is the enum type itself.

## Built-in Functions

The checker validates every built-in function call for argument count, argument types, and return type. Built-in functions are recognized by name and are not shadowable by user functions (except when a local variable shares the name, which takes precedence in scope).

### Numeric

| Function            | Signature                                  |
| ------------------- | ------------------------------------------ |
| `Abs(x)`            | `int \| float -> same`                     |
| `Min(a, b)`         | `int \| float \| byte, same -> same`       |
| `Max(a, b)`         | `int \| float \| byte, same -> same`       |
| `Sum(xs)`           | `list[int] -> int`, `list[float] -> float` |
| `Pow(a, b)`         | `int, int -> int`, `float, float -> float` |
| `Round(x)`          | `float -> int`                             |
| `Floor(x)`          | `float -> int`                             |
| `Ceil(x)`           | `float -> int`                             |
| `DivMod(a, b)`      | `int, int -> (int, int)`                   |
| `Sqrt(x)`           | `float -> float`                           |
| `WrappingAdd(a, b)` | `int, int -> int`; strict math only        |
| `WrappingSub(a, b)` | `int, int -> int`; strict math only        |
| `WrappingMul(a, b)` | `int, int -> int`; strict math only        |

`WrappingAdd`, `WrappingSub`, and `WrappingMul` are available only when the module has the `strict_math` annotation. Using them without strict math is rejected: `strict math only`.

### Bytes

| Function        | Signature             |
| --------------- | --------------------- |
| `Bytes(n)`      | `int -> bytes`        |
| `BytesFrom(xs)` | `list[byte] -> bytes` |
| `Encode(s)`     | `string -> bytes`     |
| `Decode(b)`     | `bytes -> string`     |

### Strings

| Function                                                              | Signature                                                                             |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `Concat(a, b)`                                                        | `string, string -> string` or `bytes, bytes -> bytes`                                 |
| `Upper(s)` / `Lower(s)`                                               | `string -> string`                                                                    |
| `Trim(s, chars)`                                                      | `string, string -> string`                                                            |
| `TrimStart` / `TrimEnd`                                               | `string, string -> string`                                                            |
| `Split(s, sep)`                                                       | `string, string -> list[string]`                                                      |
| `SplitN(s, sep, max)`                                                 | `string, string, int -> list[string]`                                                 |
| `SplitWhitespace(s)`                                                  | `string -> list[string]`                                                              |
| `Join(sep, parts)`                                                    | `string, list[string] -> string`                                                      |
| `Find(s, sub)` / `RFind`                                              | `string, string -> int`                                                               |
| `Count(s, sub)`                                                       | `string, string -> int`                                                               |
| `Contains(s, sub)`                                                    | overloaded (see Collections)                                                          |
| `Replace(s, old, new)`                                                | `string, string, string -> string`                                                    |
| `Repeat(s, n)`                                                        | `string, int -> string` or `list[T], int -> list[T]`                                  |
| `Reverse(s)`                                                          | `string -> string`                                                                    |
| `StartsWith` / `EndsWith`                                             | `string, string -> bool`                                                              |
| `IsDigit` / `IsAlpha` / `IsAlnum` / `IsSpace` / `IsUpper` / `IsLower` | `string \| rune -> bool`                                                              |
| `ParseInt(s, base)`                                                   | `string, int -> int`                                                                  |
| `ParseFloat(s)`                                                       | `string -> float`                                                                     |
| `FormatInt(n, base)`                                                  | `int, int -> string`                                                                  |
| `RuneFromInt(n)`                                                      | `int -> rune`                                                                         |
| `RuneToInt(c)`                                                        | `rune -> int`                                                                         |
| `Format(tmpl, args...)`                                               | `string, string... -> string` (variadic; validates `{}` count matches argument count) |

### Collections

| Function                      | Signature                                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `Len(x)`                      | `string \| bytes \| list \| map \| set -> int`                                                   |
| `Append(xs, v)`               | `list[T], T -> void`                                                                             |
| `Insert(xs, i, v)`            | `list[T], int, T -> void`                                                                        |
| `Pop(xs)`                     | `list[T] -> T`                                                                                   |
| `RemoveAt(xs, i)`             | `list[T], int -> void`                                                                           |
| `IndexOf(xs, v)`              | `list[T], T -> int`                                                                              |
| `Contains(c, v)`              | `list[T], T -> bool` or `set[T], T -> bool` or `map[K,V], K -> bool` or `string, string -> bool` |
| `Reversed(xs)`                | `list[T] -> list[T]`                                                                             |
| `Sorted(xs)`                  | `list[T] -> list[T]` (T must be ordered)                                                         |
| `Map()`                       | `-> map[K,V]` (requires context type)                                                            |
| `Set()`                       | `-> set[T]` (requires context type)                                                              |
| `Get(m, k)` / `Get(m, k, d)`  | `map[K,V], K -> V?` or `map[K,V], K, V -> V`                                                     |
| `Delete(m, k)`                | `map[K,V], K -> void`                                                                            |
| `Keys(m)`                     | `map[K,V] -> list[K]`                                                                            |
| `Values(m)`                   | `map[K,V] -> list[V]`                                                                            |
| `Items(m)`                    | `map[K,V] -> list[(K,V)]`                                                                        |
| `Merge(m1, m2)`               | `map[K,V], map[K,V] -> map[K,V]`                                                                 |
| `Concat(a, b)`                | `list[T], list[T] -> list[T]`                                                                    |
| `RangeList(start, end, step)` | `int, int, int -> list[int]`                                                                     |
| `Add(s, v)`                   | `set[T], T -> void`                                                                              |
| `Remove(s, v)`                | `set[T], T -> void`                                                                              |
| `Union(a, b)`                 | `set[T], set[T] -> set[T]`                                                                       |
| `Intersection(a, b)`          | `set[T], set[T] -> set[T]`                                                                       |
| `Difference(a, b)`            | `set[T], set[T] -> set[T]`                                                                       |

### Conversions

| Function        | Signature      |
| --------------- | -------------- |
| `IntToFloat(n)` | `int -> float` |
| `FloatToInt(x)` | `float -> int` |
| `ByteToInt(b)`  | `byte -> int`  |
| `IntToByte(n)`  | `int -> byte`  |
| `ToString(x)`   | `T -> string`  |

### I/O

| Function             | Signature                         |
| -------------------- | --------------------------------- |
| `WriteOut(d)`        | `string \| bytes -> void`         |
| `WriteErr(d)`        | `string \| bytes -> void`         |
| `WritelnOut(d)`      | `string \| bytes -> void`         |
| `WritelnErr(d)`      | `string \| bytes -> void`         |
| `ReadLine()`         | `-> string?`                      |
| `ReadAll()`          | `-> string`                       |
| `ReadBytes()`        | `-> bytes`                        |
| `ReadBytesN(n)`      | `int -> bytes`                    |
| `ReadFile(path)`      | `string -> string`                |
| `ReadFileBytes(path)` | `string -> bytes`                 |
| `WriteFile(path, d)` | `string, string \| bytes -> void` |
| `Args()`             | `-> list[string]`                 |
| `GetEnv(name)`       | `string -> string?`               |
| `Exit(code)`         | `int -> void`                     |

### Other

| Function            | Signature              |
| ------------------- | ---------------------- |
| `Assert(cond)`      | `bool -> void`         |
| `Assert(cond, msg)` | `bool, string -> void` |
| `Unwrap(x)`         | `T? -> T`              |
| `IsNaN(x)`          | `float -> bool`        |
| `IsInf(x)`          | `float -> bool`        |
| `IsType(x, name)`   | `T, string -> bool`   |

## Built-in Structs

Seven error structs are pre-registered in the type namespace. They are available without declaration and can be used in `throw`, `catch`, and constructor expressions:

```
struct KeyError { message: string }
struct IndexError { message: string }
struct ZeroDivisionError { message: string }
struct AssertError { message: string }
struct NilError { message: string }
struct ValueError { message: string }
struct IOError { message: string }
```

## Module Validation

After checking all declarations and bodies, the checker validates the module entrypoint:

| Condition               | Diagnostic                     |
| ----------------------- | ------------------------------ |
| No `Main` function      | `missing Main`                 |
| `Main` has parameters   | `Main must take no parameters` |
| `Main` returns non-void | `Main must return void`        |

## Errors

All errors carry a line and column position from the AST node that caused the error.

| Condition                   | Diagnostic                             |
| --------------------------- | -------------------------------------- |
| Type mismatch in `let`      | `cannot assign string to int`          |
| Type mismatch in assignment | `cannot assign string to int`          |
| Type mismatch in return     | `cannot return string`                 |
| Missing return value        | `missing return value`                 |
| Return value from void      | `cannot return value from void`        |
| Type mismatch in argument   | `cannot pass string as int`            |
| Arity mismatch              | `expected 2 arguments, got 3`          |
| Undefined name              | `undefined name 'x'`                   |
| Reserved name binding       | `reserved name`                        |
| Shadow outer binding        | `shadows outer binding`                |
| Duplicate top-level name    | `duplicate name`                       |
| Duplicate type name         | `duplicate type name`                  |
| Duplicate function name     | `duplicate function name`              |
| Invalid assignment target   | `invalid assignment target`            |
| Assign to this              | `cannot assign to this`                |
| Assign to tuple element     | `cannot assign to tuple element`       |
| Initializer required        | `initializer required`                 |
| `void` as value type        | `void is not a value type`             |
| Double optional             | `double optional`                      |
| Single-element tuple        | `tuple requires at least two elements` |
| Non-exhaustive match        | `non-exhaustive match`                 |
| Duplicate match case        | `duplicate case`                       |
| Invalid match case type     | `not a member`                         |
| Wrong enum in match         | `not a variant of EnumName`            |
| Non-matchable scrutinee     | `cannot match on int`                  |
| Default not last            | `default must be last`                 |
| Closure capture             | `cannot capture 'name'`                |
| Bound method as value       | `cannot capture 'this'`                |
| Call non-function           | `cannot call int`                      |
| Mixed positional/named args | `cannot mix positional and named`      |
| Unknown parameter name      | `no parameter 'c'`                     |
| Break outside loop          | `break outside of loop`                |
| Continue outside loop       | `continue outside of loop`             |
| Missing Main                | `missing Main`                         |
| Main with parameters        | `Main must take no parameters`         |
| Main non-void return        | `Main must return void`                |
| Ordering on union           | `not defined for union`                |
| Arithmetic on union         | `not defined for union`                |
| Index on union              | `cannot index union`                   |
| Iterate union/optional      | `cannot iterate union`                 |
| Field access on interface   | `cannot access field on interface`     |
| Equality on fn type         | `equality not defined for fn`          |
| Not all paths return        | `not all paths return`                 |
| Unreachable code            | `unreachable code`                     |
| Duplicate typed catch       | `duplicate catch`                      |
| Catch after catch-all       | `unreachable catch`                    |
| Parent not an interface     | `parent is not an interface`           |
| Integer literal overflow    | `integer literal too large`            |
| Strict-math-only feature    | `strict math only`                     |
| Bare non-call expression    | `expression has no effect`             |
| Assign to loop variable     | `cannot assign to loop variable`       |
| Assign to slice             | `cannot assign to slice`               |
| `this` outside method       | `this outside method`                  |
| Non-hashable map key/set el | `not hashable`                         |
| `Sorted` non-ordered type   | `not ordered`                          |
| Unknown semantic annotation | `unknown annotation`                   |
| Empty collection no context | `cannot infer type`                    |
| Map literal duplicate key   | `duplicate key`                        |
| Throw non-throwable type    | `cannot throw T`                       |
| Control flow in finally     | `control flow in finally`              |

## Public API

| Function                  | Returns                       | Description                                      |
| ------------------------- | ----------------------------- | ------------------------------------------------ |
| `check(module)`           | `list[CheckError]`            | validate module, return errors                   |
| `check_with_info(module)` | `(list[CheckError], Checker)` | validate module, return errors and checker state |

`check_with_info` exposes the `Checker` instance after validation, giving downstream passes access to the resolved type namespace, function signatures, and parameter name maps.

## Postconditions

- All declarations registered with resolved types
- All function and method bodies type-checked
- All expressions have valid types
- Every non-void function returns or throws on all control-flow paths
- No unreachable code after `return`, `throw`, `break`, or `continue`
- Scoping enforced — no shadowing, no access after scope exit
- Reserved names protected
- Match exhaustiveness verified
- No closures — function literals reference only local bindings and top-level names
- Module has exactly one `Main` function with correct signature
- `void` restricted to return-type position
- Union types normalized (flattened, deduplicated)
- All built-in function calls validated for arity and argument types
- Strict-math-only features (`>>>`, `WrappingAdd`, `WrappingSub`, `WrappingMul`) rejected without `strict_math` annotation
- Integer literals within [-2^63, 2^63)
- Struct parents validated as interfaces
- Expression statements are function or method calls only
- Loop variables are read-only
- Non-void function literals return or throw on all paths
- Catch-all residual computed from all struct types in the program
- `finally` blocks contain no `return`, `throw`, `break`, or `continue`
