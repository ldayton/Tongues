# Intermediate Representation

IR for Python → {Go, TS, Rust, C} transpilation.

```
source.py → [Python AST] → Frontend → [IR] → Backend → target code
```

**Frontend** (one implementation): Type inference, symbol resolution, scope analysis, ownership analysis.

**Backends** (per target): Pure syntax emission, ~500-800 lines each.

**TypeScript strategy**: Backend emits `.ts` files; `tsc` produces `.js` + `.d.ts`. No separate JS backend needed.

See [subpy.md](subpy.md) for the supported Python subset.

## Types

### Primitives

```
Primitive { kind: "string" | "int" | "bool" | "float" | "byte" | "rune" | "void" }
```

| IR       | Go        | TS        | Rust     | C               |
| -------- | --------- | --------- | -------- | --------------- |
| `int`    | `int`     | `number`  | `i64`    | `int64_t`       |
| `float`  | `float64` | `number`  | `f64`    | `double`        |
| `bool`   | `bool`    | `boolean` | `bool`   | `bool`          |
| `string` | `string`  | `string`  | `String` | `Str` (ptr+len) |
| `byte`   | `byte`    | `number`  | `u8`     | `uint8_t`       |
| `rune`   | `rune`    | `number`  | `char`   | `int32_t`       |
| `void`   | —         | `void`    | `()`     | `void`          |

### Collections

```
Slice { element: Type }
```

| IR         | Go    | TS         | Rust     | C                                      |
| ---------- | ----- | ---------- | -------- | -------------------------------------- |
| `Slice(T)` | `[]T` | `Array<T>` | `Vec<T>` | `struct { T *data; size_t len, cap; }` |

```
Array { element: Type, size: int }
```

| IR            | Go     | TS         | Rust     | C      |
| ------------- | ------ | ---------- | -------- | ------ |
| `Array(T, N)` | `[N]T` | `Array<T>` | `[T; N]` | `T[N]` |

```
Map { key: Type, value: Type }
```

| IR         | Go        | TS          | Rust           | C              |
| ---------- | --------- | ----------- | -------------- | -------------- |
| `Map(K,V)` | `map[K]V` | `Map<K, V>` | `HashMap<K,V>` | generated hash |

```
Set { element: Type }
```

| IR       | Go           | TS       | Rust          | C              |
| -------- | ------------ | -------- | ------------- | -------------- |
| `Set(T)` | `map[T]bool` | `Set<T>` | `HashSet<T>`  | generated hash |

```
Tuple { elements: [Type] }
```

| IR             | Go           | TS           | Rust         | C                    |
| -------------- | ------------ | ------------ | ------------ | -------------------- |
| `Tuple(T1,T2)` | `(T1, T2)`   | `[T1, T2]`   | `(T1, T2)`   | struct or out-params |

Used for multi-value returns like `tuple[Node | None, str]`. Go emits multiple return values; other backends use native tuples.

### Pointers and Optionals

```
Pointer { target: Type, owned: bool }
```

| IR                        | Go   | TS  | Rust     | C              |
| ------------------------- | ---- | --- | -------- | -------------- |
| `Pointer(T, owned=true)`  | `*T` | ref | `Box<T>` | `T*` (owns)    |
| `Pointer(T, owned=false)` | `*T` | ref | `&'a T`  | `T*` (borrows) |

```
Optional { inner: Type }
```

| IR            | Go         | TS          | Rust        | C           |
| ------------- | ---------- | ----------- | ----------- | ----------- |
| `Optional(T)` | `*T` / nil | `T \| null` | `Option<T>` | `T*` + NULL |

### Type References

```
StructRef { name: string }          // Reference to defined struct
Interface { name: string }          // Dynamic dispatch (Go interface, Rust dyn Trait)
Union { variants: [Type] }          // Sum type (Go interface, Rust enum, C tagged union)
```

### Functions

```
FuncType { params: [Type], ret: Type, captures: bool }
```

| IR                              | Go          | TS         | Rust           | C               |
| ------------------------------- | ----------- | ---------- | -------------- | --------------- |
| `FuncType(..., captures=false)` | `func(...)` | `function` | `fn(...)`      | `T (*)(…)`      |
| `FuncType(..., captures=true)`  | `func(...)` | `function` | `impl Fn(...)` | struct + fn ptr |

### Strings

```
StringSlice                         // Borrowed immutable view
```

| IR            | Go       | TS       | Rust   | C             |
| ------------- | -------- | -------- | ------ | ------------- |
| `StringSlice` | `string` | `string` | `&str` | `const char*` |

## Source Locations

All nodes carry source position:

```
Loc { line: int, col: int, end_line: int, end_col: int }
```

Line is 1-indexed, column is 0-indexed. Enables error messages, source maps, IDE integration.

## Declarations

### Module

```
Module {
    name: string
    doc: string?                // module docstring
    structs: [Struct]
    interfaces: [InterfaceDef]
    functions: [Function]
    constants: [Constant]
}
```

### Struct

```
Struct {
    name: string
    doc: string?                // class docstring
    fields: [Field]
    methods: [Function]
    implements: [string]        // interface names
}

Field { name: string, typ: Type, default: Expr? }
```

### Interface

```
InterfaceDef {
    name: string
    methods: [MethodSig]
}

MethodSig { name: string, params: [Param], ret: Type }
```

### Function

```
Function {
    name: string
    params: [Param]
    ret: Type
    body: [Stmt]
    doc: string?                // function/method docstring
    receiver: Receiver?         // present for methods
    fallible: bool              // can raise ParseError
}

Receiver {
    name: string                // "self", "p", etc.
    typ: StructRef
    mutable: bool               // Rust: &mut self
    pointer: bool               // Go: *T receiver
}

Param {
    name: string
    typ: Type
    default: Expr?
    mutable: bool               // Rust: mut
}
```

**Default parameters:** Go lacks default arguments. Emission strategy by pattern:

| Pattern                  | Go Emission                 | Example                                             |
| ------------------------ | --------------------------- | --------------------------------------------------- |
| `bool = False`           | Omit; zero value is `false` | `func F(debug bool)`                                |
| `T \| None = None`       | Omit; zero value is `nil`   | `func F(opts *Options)`                             |
| Multiple optional        | Options struct              | `func F(opts FOptions)` with `FOptions{Field: val}` |
| Single trailing optional | Variadic                    | `func F(required int, opt ...string)`               |

### Constant

```
Constant { name: string, typ: Type, value: Expr }
```

## Statements

All statements carry `loc: Loc`.

### Variables

```
VarDecl { name: string, typ: Type, value: Expr?, mutable: bool }
```

| IR              | Go           | TS      | Rust      | C            |
| --------------- | ------------ | ------- | --------- | ------------ |
| `mutable=true`  | `var` / `:=` | `let`   | `let mut` | no qualifier |
| `mutable=false` | `const`      | `const` | `let`     | `const`      |

### Assignment

```
Assign { target: LValue, value: Expr }
OpAssign { target: LValue, op: string, value: Expr }    // +=, -=, *=, etc.
```

### Control Flow

```
If {
    cond: Expr
    then_body: [Stmt]
    else_body: [Stmt]
    init: VarDecl?              // Go: if x := ...; cond { }
}

TypeSwitch {
    expr: Expr
    binding: string             // variable name in each case
    cases: [TypeCase]
    default: [Stmt]
}

TypeCase { typ: Type, body: [Stmt] }

Match {
    expr: Expr
    cases: [MatchCase]
    default: [Stmt]
}

MatchCase { patterns: [Expr], body: [Stmt] }
```

`TypeSwitch` translates `isinstance` chains:

| IR           | Go                | TS                          | Rust                   | C                 |
| ------------ | ----------------- | --------------------------- | ---------------------- | ----------------- |
| `TypeSwitch` | `switch x.(type)` | `if (x instanceof T)` chain | `match x { T => ... }` | `switch (x->tag)` |

### Loops

```
ForRange {
    index: string?              // None if unused
    value: string?
    iterable: Expr
    body: [Stmt]
}

ForClassic {
    init: Stmt?
    cond: Expr?
    post: Stmt?
    body: [Stmt]
}

While { cond: Expr, body: [Stmt] }

Break { label: string? }
Continue { label: string? }
```

### Error Handling

**Two failure modes:**
- **Soft failure**: Returns `None` — "try alternative parsing"
- **Hard failure**: Raises exception — "committed and failed"

Exceptions are NOT used for control flow. Most propagate to top; few backtracking points catch them.

**Error type:**
```
ParseError { message: string, pos: int, line: int? }
MatchedPairError extends ParseError   // unclosed construct at EOF
```

**Fallible functions** (marked with `fallible: bool` on Function):
```
Raise { error_type: string, message: Expr, pos: Expr }
```

| IR      | Go           | TS      | Rust                     | C                 |
| ------- | ------------ | ------- | ------------------------ | ----------------- |
| `Raise` | `panic(...)` | `throw` | `return Err(ParseError)` | `longjmp` or goto |

**Calling fallible functions:**

Bare calls propagate errors automatically:
| Context | Go           | TS           | Rust      | C            |
| ------- | ------------ | ------------ | --------- | ------------ |
| Default | panic floats | throw floats | `call()?` | check + goto |

**Backtracking** (rare):
```
TryCatch {
    body: [Stmt]
    catch_var: string?          // None if error ignored
    catch_body: [Stmt]
    reraise: bool               // catch cleans up then re-raises
}
```

| IR         | Go              | TS          | Rust                      | C                |
| ---------- | --------------- | ----------- | ------------------------- | ---------------- |
| `TryCatch` | `defer/recover` | `try/catch` | `match call() { Ok/Err }` | `setjmp/longjmp` |

**Soft failure** (return None to signal "try alternative"):
```
SoftFail { }                    // return None / (None, "")
```

| IR         | Go           | TS            | Rust          | C             |
| ---------- | ------------ | ------------- | ------------- | ------------- |
| `SoftFail` | `return nil` | `return null` | `return None` | `return NULL` |

### Returns and Blocks

```
Return { value: Expr? }
ExprStmt { expr: Expr }                 // expression used as statement
Block { body: [Stmt] }                  // scoped statement group
```

## Expressions

All expressions carry `typ: Type` and `loc: Loc`.

### Literals

```
IntLit { value: int }
FloatLit { value: float }
StringLit { value: string }
BoolLit { value: bool }
NilLit { }                              // nil/null/None
```

### Access

```
Var { name: string }

FieldAccess {
    obj: Expr
    field: string
    through_pointer: bool       // Go auto-deref
}

Index {
    obj: Expr
    index: Expr
    bounds_check: bool          // C can skip
    returns_optional: bool      // Go map returns (v, ok)
}

SliceExpr { obj: Expr, low: Expr?, high: Expr? }
```

### Calls

```
Call { func: string, args: [Expr] }

MethodCall {
    obj: Expr
    method: string
    args: [Expr]
    receiver_type: Type         // resolved by frontend
}

StaticCall { on_type: Type, method: string, args: [Expr] }
```

### Operators

```
BinaryOp { op: string, left: Expr, right: Expr }
UnaryOp { op: string, operand: Expr }
Ternary { cond: Expr, then_expr: Expr, else_expr: Expr }
```

Go lacks ternary; backend emits IIFE: `func() T { if cond { return a } return b }()`

### Type Operations

```
Cast { expr: Expr, to_type: Type }
TypeAssert { expr: Expr, asserted: Type, safe: bool }
IsType { expr: Expr, tested_type: Type }
IsNil { expr: Expr, negated: bool }
Len { expr: Expr }
```

`IsNil` is explicit (not `BinaryOp("==", x, NilLit)`):

| IR      | Go         | TS           | Rust          | C           |
| ------- | ---------- | ------------ | ------------- | ----------- |
| `IsNil` | `x == nil` | `x === null` | `x.is_none()` | `x == NULL` |

### Allocation

```
MakeSlice { element_type: Type, length: Expr?, capacity: Expr? }
MakeMap { key_type: Type, value_type: Type }
SliceLit { element_type: Type, elements: [Expr] }
MapLit { key_type: Type, value_type: Type, entries: [(Expr, Expr)] }
SetLit { element_type: Type, elements: [Expr] }
StructLit { struct_name: string, fields: {string: Expr} }
```

### Strings

```
StringConcat { parts: [Expr] }
StringFormat { template: string, args: [Expr] }
```

| IR             | Go            | TS               | Rust      | C          |
| -------------- | ------------- | ---------------- | --------- | ---------- |
| `StringConcat` | `+`           | `+`              | `format!` | `snprintf` |
| `StringFormat` | `fmt.Sprintf` | template literal | `format!` | `snprintf` |

## LValues

Assignment targets:

```
VarLV { name: string }
FieldLV { obj: Expr, field: string }
IndexLV { obj: Expr, index: Expr }
DerefLV { ptr: Expr }
```

## Union Types

All unions are **closed** (fixed variants) and discriminated via `.kind` string field.

```
Union { name: string, variants: [StructRef] }
```

| IR      | Go                      | TS             | Rust   | C            |
| ------- | ----------------------- | -------------- | ------ | ------------ |
| `Union` | interface + type switch | class + `kind` | `enum` | tagged union |

