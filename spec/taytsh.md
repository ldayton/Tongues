# Taytsh Specification

Taytsh is a statically-typed nominally-typed monomorphic intermediate language that serves as the target-neutral IR for the Tongues transpiler. It has reference semantics for mutable types, lexical block scoping, sealed interfaces, exhaustive pattern matching, no closures, no imports, and no user-defined generics. Programs are single-file, closed-world, and free of implicit coercion.

## Introduction

Tongues transpiles a well-behaved subset of Python into 15 target languages. Python's semantics — floor division, value-returning `and`/`or`, type-dependent truthiness, negative indexing, open-ended slices, chained comparisons — must be resolved before code generation, but the targets themselves disagree on fundamentals: string encoding, integer width, value vs reference passing, loop forms, and pattern matching support. The IR must absorb Python's quirks on one side and remain neutral across diverse backends on the other.

Taytsh is that IR. The lowerer resolves Python semantics into conventional forms (truncating division, boolean `&&`/`||`, explicit nil checks), but defers representation choices. Taytsh has one loop-over-collection form (`for...in`), not separate range and iterator loops; strings are rune-indexed, not pre-lowered into target-specific character operations. The middleend analyzes how values are actually used — whether a `for...in` over a range should become a classic for loop, whether a string needs rune-level access or can stay in the target's native encoding — and records those decisions as annotations. Backends read the annotated IR and translate to syntax.

This spec defines Taytsh using a textual syntax to facilitate exposition. The grammar is parseable by recursive descent with at most two tokens of lookahead.

## Module Structure

A program is a single UTF-8 encoded source file. All declarations — functions, structs, interfaces, and enums — live in one flat namespace. There are no imports, no modules, no packages, no forward declarations. Every name is visible throughout the file regardless of declaration order.

This closed-world property means the compiler can see every type that exists, every function that can be called, and every interface implementation. Exhaustiveness checks in `match` are total — there are no unknown variants.

### Entrypoint

```
fn Main() -> void {
    let input: string = ReadAll()
    WritelnOut(input)
}
```

`Main` is the program entrypoint. It takes no parameters and returns `void`. Command-line arguments and environment variables are accessed via `Args()` and `GetEnv()`.

A valid program must contain exactly one `Main` function. Normal return from `Main` implies exit code 0.

## Type System

Every value in Taytsh is an `obj`. The terms "object" and "value" are synonymous — there is no distinction between primitive and reference types at this level of abstraction.

```
obj
├── int
├── float
├── bool
├── byte
├── bytes
├── string
├── rune
├── list[T]
├── map[K, V]
├── set[T]
├── (T, U, ...)       -- tuples
├── fn[T..., R]       -- function values
├── A | B | ...       -- union types
├── nil               -- value in expressions, type in type position
├── structs
├── interfaces
└── enums
```

`void` is a return-type marker meaning "no value." It may appear after `->` in function declarations and as the last element of `fn[...]` types. It is not a value type — it cannot be used for variables, parameters, or collection elements.

`T?` is sugar for `T` or `nil`. `nil` is a value in expression position and a type in type position.

No implicit coercion exists anywhere in the language. All type conversions are explicit function calls.

Generic type parameters are built-in only (`list[T]`, `map[K, V]`, `set[T]`, `fn[T..., R]`). User-defined structs, interfaces, and functions are monomorphic.

### ToString

`ToString` is a built-in free function defined on all types — the only universal operation. The output format is target-native by default; `--strict-tostring` specifies a canonical format — see Strict ToString below.

| Function      | Signature       | Description           |
| ------------- | --------------- | --------------------- |
| `ToString(x)` | `obj -> string` | string representation |

### Throw and Catch

Any `obj` can be thrown. `catch` blocks follow the same type-matching semantics as `match` cases — each `catch` names a type and binds a variable of that type. Unlike `match`, `catch` does not require exhaustiveness; unmatched exceptions propagate to the caller.

```
throw 42
throw "something went wrong"
throw ValueError("unexpected token")
```

```
try {
    RiskyOperation()
} catch e: ValueError {
    WritelnErr(e.message)
} catch e: obj {
    WritelnErr(Concat("unexpected: ", ToString(e)))
}
```

Catching `obj` is the catch-all. Catching `nil` handles a thrown `nil`.

### Built-in Errors

Built-in operations throw the following error types. All are catchable via `try`/`catch`.

```
struct KeyError { message: string }
struct IndexError { message: string }
struct ZeroDivisionError { message: string }
struct AssertError { message: string }
struct NilError { message: string }
struct ValueError { message: string }
struct IOError { message: string }
```

| Error               | Thrown by                                                       |
| ------------------- | --------------------------------------------------------------- |
| `KeyError`          | map indexing with missing key                                   |
| `IndexError`        | out-of-bounds index or slice, `Pop` on empty list               |
| `ZeroDivisionError` | int or byte `/` or `%` with zero divisor                        |
| `AssertError`       | `Assert` failure                                                |
| `NilError`          | `Unwrap` on nil                                                 |
| `ValueError`        | `ParseInt`/`ParseFloat` bad input, `FloatToInt` overflow or NaN |
| `IOError`           | `ReadFile`/`WriteFile` failure                                  |

## Numeric Types

```
let x: int = 42
let ratio: float = 3.14
let done: bool = true
```

| Type    | Literal         | Description                      |
| ------- | --------------- | -------------------------------- |
| `int`   | `42`            | signed integer, at least 53 bits |
| `float` | `3.14`          | IEEE 754 binary64                |
| `bool`  | `true`, `false` | boolean                          |

Target representations:

| Target     | `int`     | `float`   | Notes                              |
| ---------- | --------- | --------- | ---------------------------------- |
| C          | `int64_t` | `double`  |                                    |
| C#         | `long`    | `double`  |                                    |
| Dart       | `int`     | `double`  |                                    |
| Go         | `int`     | `float64` | `int` is 64-bit on modern targets  |
| Java       | `long`    | `double`  |                                    |
| JavaScript | `number`  | `number`  | IEEE 754 double; 53-bit int range  |
| Lua        | `integer` | `number`  |                                    |
| Perl       | scalar    | scalar    |                                    |
| PHP        | `int`     | `float`   | 64-bit on modern targets           |
| Python     | `int`     | `float`   | int is arbitrary precision         |
| Ruby       | `Integer` | `Float`   | Integer is arbitrary precision     |
| Rust       | `i64`     | `f64`     |                                    |
| Swift      | `Int`     | `Double`  | `Int` is 64-bit on modern targets  |
| TypeScript | `number`  | `number`  | IEEE 754 double; 53-bit int range  |
| Zig        | `i64`     | `f64`     | `f64` is IEEE-compliant by default |

### Functions

| Function       | Signature                | Description                                              |
| -------------- | ------------------------ | -------------------------------------------------------- |
| `Abs(x)`       | `T -> T`                 | absolute value                                           |
| `Min(a, b)`    | `T, T -> T`              | smaller of two values                                    |
| `Max(a, b)`    | `T, T -> T`              | larger of two values                                     |
| `Sum(xs)`      | `list[T] -> T`           | sum of elements                                          |
| `Pow(a, b)`    | `T, T -> T`              | exponentiation                                           |
| `Round(x)`     | `float -> int`           | round to nearest integer                                 |
| `Floor(x)`     | `float -> int`           | largest integer ≤ x                                      |
| `Ceil(x)`      | `float -> int`           | smallest integer ≥ x                                     |
| `Sqrt(x)`      | `float -> float`         | square root; IEEE 754 correctly rounded                  |
| `DivMod(a, b)` | `int, int -> (int, int)` | quotient and remainder (truncating, same as `/` and `%`) |

`T` in `Min`, `Max` is `int`, `float`, or `byte`. `T` in `Abs`, `Sum`, `Pow` is `int` or `float`. `Sqrt`, `Floor`, and `Ceil` are `float` only. No implicit coercion between numeric types — `Min(int, float)` is a type error. `bool` and `int` are distinct types with no implicit coercion in either direction.

## Bytes

```
let tag: byte = 0xff
let buf: bytes = ReadBytes()
let header: bytes = b"\x89PNG"
let first: byte = buf[0]
let rest: bytes = buf[1:10]
```

| Type    | Literal      | Description             |
| ------- | ------------ | ----------------------- |
| `byte`  | `0xff`       | unsigned 8-bit integer  |
| `bytes` | `b"\x89PNG"` | indexable byte sequence |

`bytes` is an ordered sequence of `byte` values. Indexing (`buf[i]`) yields a `byte`; throws `IndexError` if out of bounds. Slicing (`buf[a:b]`) yields a `bytes`; throws `IndexError` if either bound is out of range.

`byte` is a numeric type. Integer literals in the range 0–255 are accepted where a `byte` is expected. `0xff`-style hex literals always produce `byte` — for a hex int value, use `ByteToInt(0xff)`. Arithmetic and bitwise operators work on `byte` pairs (see Operators). For byte↔int conversion, see Conversions.

### Functions

| Function       | Signature               | Description   |
| -------------- | ----------------------- | ------------- |
| `Len(b)`       | `bytes -> int`          | byte count    |
| `Concat(a, b)` | `bytes, bytes -> bytes` | concatenation |
| `Encode(s)`    | `string -> bytes`       | UTF-8 encode  |
| `Decode(b)`    | `bytes -> string`       | UTF-8 decode  |

## Operators

```
let neg: int = -x
let valid: bool = !done
let low: int = mask & 0xff
let sum: int = a + b
let avg: int = total / count
let odd: bool = n % 2 != 0
let big: int = Pow(2, 10)
let inRange: bool = 0 <= x && x <= 255
let either: bool = a || b
let shifted: int = flags << 2
let abs: int = x > 0 ? x : -x
```

| Operator | Operands     | Result | Prec | Assoc | Description                                                                        |
| -------- | ------------ | ------ | ---- | ----- | ---------------------------------------------------------------------------------- |
| `?:`     | `bool, T, T` | `T`    | 1    | right | ternary conditional                                                                |
| `\|\|`   | `bool, bool` | `bool` | 2    | left  | logical or, short-circuit                                                          |
| `&&`     | `bool, bool` | `bool` | 3    | left  | logical and, short-circuit                                                         |
| `==`     | `T, T`       | `bool` | 4    | none  | equality; deep structural for structs/collections, IEEE 754 for float (NaN != NaN) |
| `!=`     | `T, T`       | `bool` | 4    | none  | inequality                                                                         |
| `<`      | `T, T`       | `bool` | 4    | none  | less than (int, float, byte, rune, string)                                         |
| `<=`     | `T, T`       | `bool` | 4    | none  | less or equal (int, float, byte, rune, string)                                     |
| `>`      | `T, T`       | `bool` | 4    | none  | greater than (int, float, byte, rune, string)                                      |
| `>=`     | `T, T`       | `bool` | 4    | none  | greater or equal (int, float, byte, rune, string)                                  |
| `\|`     | `T, T`       | `T`    | 5    | left  | bitwise or (int, byte)                                                             |
| `^`      | `T, T`       | `T`    | 6    | left  | bitwise xor (int, byte)                                                            |
| `&`      | `T, T`       | `T`    | 7    | left  | bitwise and (int, byte)                                                            |
| `<<`     | `T, int`     | `T`    | 8    | left  | left shift (int, byte); right operand must be non-negative                         |
| `>>`     | `T, int`     | `T`    | 8    | left  | arithmetic right shift (int, byte); right operand non-negative                     |
| `+`      | `T, T`       | `T`    | 9    | left  | addition (int, float, byte)                                                        |
| `-`      | `T, T`       | `T`    | 9    | left  | subtraction (int, float, byte)                                                     |
| `*`      | `T, T`       | `T`    | 10   | left  | multiplication (int, float, byte)                                                  |
| `/`      | `T, T`       | `T`    | 10   | left  | division (int, float, byte); int truncates toward zero                             |
| `%`      | `T, T`       | `T`    | 10   | left  | remainder (int, float, byte); int sign follows dividend                            |
| `-`      | `T`          | `T`    | 11   | right | negation (int, float, byte)                                                        |
| `!`      | `bool`       | `bool` | 11   | right | logical not                                                                        |
| `~`      | `T`          | `T`    | 11   | right | bitwise complement (int, byte)                                                     |

All operators require operands to be the same type — no implicit coercion. `int + float` is a type error; lowering must insert explicit casts. For int `/`, result truncates toward zero (`-7 / 2 == -3`); for int `%`, sign follows dividend (`-7 % 2 == -1`). For float `/`, result is IEEE 754 division. For float `%`, result is IEEE 754 remainder (`fmod`); sign follows dividend; behavior with zero divisor is unspecified. Byte arithmetic wraps mod 256. String comparisons are lexicographic.

Comparisons are binary — `a < b < c` is not valid. Python's chained comparisons are desugared by the lowerer into `&&`-connected binary comparisons. A middleend raising pass can reconstruct chains for targets that support them (Python).

## Strings

Target languages disagree on what a string is. Some are byte-oriented (Go, Rust), some are UTF-16 (Java, JavaScript), and indexing `s[i]` means different things in each. Taytsh defines `string` as a sequence of runes, so indexing and length have consistent character-level semantics across all targets.

```
let name: string = "hello"
let ch: rune = 'λ'
let first: rune = name[0]
let n: int = Len("café")        -- 4, not 5
```

| Type     | Literal   | Description              |
| -------- | --------- | ------------------------ |
| `string` | `"hello"` | sequence of runes        |
| `rune`   | `'λ'`     | single Unicode character |

`string[i]` yields a `rune`; throws `IndexError` if out of bounds. `string[lo:hi]` yields a `string`; throws `IndexError` if either bound is out of range. `Len(s)` returns the rune count.

### Functions

| Function              | Signature                  | Description                    |
| --------------------- | -------------------------- | ------------------------------ |
| `Len(s)`              | `string -> int`            | rune count                     |
| `Concat(a, b)`        | `string, string -> string` | concatenation                  |
| `RuneFromInt(n)`      | `int -> rune`              | code point to rune             |
| `RuneToInt(c)`        | `rune -> int`              | rune to code point             |
| `ParseInt(s, base)`   | `string, int -> int`       | parse integer in given base    |
| `ParseFloat(s)`       | `string -> float`          | parse float                    |
| `FormatInt(n, base)`  | `int, int -> string`       | format integer in given base   |
| `Upper(s)`            | `string -> string`         | uppercase                      |
| `Lower(s)`            | `string -> string`         | lowercase                      |
| `Trim(s, chars)`      | `string, string -> string` | trim characters from both ends |
| `TrimStart(s, chars)` | `string, string -> string` | trim characters from start     |
| `TrimEnd(s, chars)`   | `string, string -> string` | trim characters from end       |

| `Split(s, sep)`        | `string, string -> list[string]`      | split by separator; empty sep is error |
| `SplitN(s, sep, max)`  | `string, string, int -> list[string]` | split by separator; at most max pieces; max ≤ 0 throws ValueError |
| `SplitWhitespace(s)`   | `string -> list[string]`              | split on whitespace runs, strip ends   |
| `Join(sep, parts)`     | `string, list[string] -> string`      | join with separator                    |
| `Find(s, sub)`         | `string, string -> int`               | index of substring, -1 if missing; empty sub returns 0 |
| `RFind(s, sub)`        | `string, string -> int`               | last occurrence, -1 if missing         |
| `Count(s, sub)`        | `string, string -> int`               | count non-overlapping occurrences      |
| `Contains(s, sub)`     | `string, string -> bool`              | substring test                         |
| `Replace(s, old, new)` | `string, string, string -> string`    | replace all occurrences                |
| `Repeat(s, n)`         | `string, int -> string`               | repeat n times; n ≤ 0 yields empty     |
| `StartsWith(s, pre)`   | `string, string -> bool`              | prefix test                            |
| `EndsWith(s, suf)`     | `string, string -> bool`              | suffix test                            |
| `IsDigit(x)`           | `string \| rune -> bool`              | all characters are digits; false for empty string              |
| `IsAlpha(x)`           | `string \| rune -> bool`              | all characters are letters; false for empty string             |
| `IsAlnum(x)`           | `string \| rune -> bool`              | all characters are letters or digits; false for empty string   |
| `IsSpace(x)`           | `string \| rune -> bool`              | all characters are whitespace; false for empty string          |
| `IsUpper(x)`           | `string \| rune -> bool`              | all characters are uppercase; false for empty string           |
| `IsLower(x)`           | `string \| rune -> bool`              | all characters are lowercase; false for empty string           |

### Format Strings

```
let msg: string = Format("hello, {}", name)
let line: string = Format("{}: {}", ToString(lineno), text)
```

`Format(template, args...)` interpolates arguments into a template string. `{}` placeholders are filled left to right. All arguments must be `string` — callers insert explicit conversions (`ToString`, etc.) before passing. The number of `{}` must match the number of arguments. `Format` is a built-in variadic function — user-defined functions cannot declare variadic parameters.

| Function                    | Signature                     | Description                     |
| --------------------------- | ----------------------------- | ------------------------------- |
| `Format(template, args...)` | `string, string... -> string` | positional string interpolation |

## Functions

```
fn Gcd(a: int, b: int) -> int {
    while b != 0 {
        let t: int = b
        b = a % b
        a = t
    }
    return a
}

fn Greet(name: string) -> void {
    WritelnOut(Concat("hello, ", name))
}
```

A function has a name, typed parameters, a return type, and a body. `void` is a return-type marker meaning no return value — it is not a type. The body is a block of statements enclosed in `{}`.

```
fn Factorial(n: int) -> int {
    if n <= 1 {
        return 1
    }
    return n * Factorial(n - 1)
}
```

### Function Types

Functions are values. The type of a function is written `fn[ParamTypes..., ReturnType]` — the last element is the return type, everything before it is a parameter type.

```
let predicate: fn[int, bool] = IsEven
let transform: fn[string, string] = Upper
let callback: fn[void] = DoNothing
```

A named function can be used as a value by name. Function values are called with the same `()` syntax as named functions.

```
fn Apply(xs: list[int], f: fn[int, int]) -> list[int] {
    let result: list[int]
    for x in xs {
        Append(result, f(x))
    }
    return result
}

let doubled: list[int] = Apply(numbers, Double)
```

### Function Literals

Anonymous functions use arrow syntax. Block body with `{ }` or expression body with `=>`.

```
let double: fn[int, int] = (x: int) -> int { return x * 2 }
let negate: fn[int, int] = (x: int) -> int => -x
let greet: fn[string, void] = (name: string) -> void {
    WritelnOut(Concat("hello, ", name))
}
```

Function literals cannot close over surrounding state — the body can only reference its own parameters. Capturing variables from an enclosing scope is a compile error.

```
let offset: int = 10
let shift: fn[int, int] = (x: int) -> int => x + offset  -- error: cannot capture 'offset'
```

Bound methods are not values for the same reason — binding `self` is capturing state.

```
let s: Span = Span(0, 10)
let f: fn[int] = s.Len  -- error: cannot capture 'self'
```

### Higher-Order Functions

```
fn Filter(xs: list[int], f: fn[int, bool]) -> list[int] {
    let result: list[int]
    for x in xs {
        if f(x) {
            Append(result, x)
        }
    }
    return result
}

let evens: list[int] = Filter(numbers, IsEven)
let small: list[int] = Filter(numbers, (x: int) -> bool => x < 10)
```

## Variables

```
let x: int = 42
let name: string = "hello"
let done: bool = false
let items: list[int]
```

`let` declares a variable with an explicit type. The initializer is optional for types with zero values — omitting it gives the zero value. Structs, interfaces, enums, function types, and union types not containing `nil` have no zero value and require an explicit initializer.

### Zero values

| Type          | Zero value                                                                |
| ------------- | ------------------------------------------------------------------------- |
| `int`         | `0`                                                                       |
| `float`       | `0.0`                                                                     |
| `bool`        | `false`                                                                   |
| `byte`        | `0x00`                                                                    |
| `bytes`       | `b""`                                                                     |
| `string`      | `""`                                                                      |
| `rune`        | `'\0'`                                                                    |
| `list[T]`     | `[]`                                                                      |
| `map[K, V]`   | `Map()`                                                                   |
| `set[T]`      | `Set()`                                                                   |
| `(T, U, ...)` | tuple of element zero values; no zero value if any element type lacks one |
| `T?`          | `nil`                                                                     |
| `A \| B`      | `nil` if union contains `nil`; otherwise no zero value                    |
| `obj`         | `nil`                                                                     |

Structs, interfaces, enums, function types, and union types not containing `nil` have no zero value.

`let` bindings are mutable. Backends may emit `const`/`final` when they detect a binding is never reassigned.

## Scoping

Taytsh uses lexical block scoping. A variable is visible from its `let` declaration to the end of the enclosing `{}` block. A variable does not exist before its declaration — there is no hoisting.

```
fn Example() -> void {
    let x: int = 1          -- x visible from here to end of function
    if x > 0 {
        let y: int = 2      -- y visible only inside this block
        WritelnOut(ToString(y))
    }
    -- y is not accessible here
}
```

### Loop and binding scopes

`for` loop variables are scoped to the loop body. `match`/`case` bindings are scoped to the case body. `catch` bindings are scoped to the catch body.

```
for i, v in items {
    -- i and v are scoped here
}
-- i and v are not accessible here
```

### No shadowing

A name may be bound at most once within a function. Declaring a variable with the same name as an outer binding — including parameters, loop variables, and case/catch bindings — is a compile error. `_` is the exception — it may appear multiple times in bindings (e.g. `for _, _ in m`) and is never accessible as a variable.

```
fn Bad(x: int) -> void {
    let x: int = 10              -- error: x already bound (parameter)
}

fn AlsoBad() -> void {
    let x: int = 1
    if true {
        let x: int = 2          -- error: x already bound (outer block)
    }
}
```

Every name in a function body resolves to exactly one binding. Backends never need to rename variables to avoid scope collisions.

### Reserved names

Built-in function names (`Len`, `Append`, `ToString`, `WriteOut`, `WritelnErr`, etc.) are reserved. No binding — top-level declaration, local variable, parameter, or loop variable — may use a reserved name. The one exception is `ToString`: structs may declare a `fn ToString(self) -> string` method to override the default string representation (see Strict ToString). This keeps name resolution trivial — a call to `Len(xs)` always means the built-in, with no overload resolution or import precedence to consider.

### Top-level declarations

Functions, structs, interfaces, and enums live in a single flat namespace and are mutually visible regardless of declaration order (see Module Structure). Top-level names and local names occupy disjoint namespaces — a local variable may share a name with a top-level declaration, and the local binding takes precedence within its scope.

## Assignment

```
x = 10
name = "world"
token.kind = TokenKind.Eof
items[0] = 99
```

Assignment writes to a variable, field, or index. The target must already be declared.

### Compound assignment

```
x += 1
total -= cost
mask &= 0xff
count *= 2
```

Compound assignment (`+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`) applies the operator and assigns the result.

### Tuple assignment

```
let q: int
let r: int
q, r = DivMod(17, 5)
```

Tuple assignment destructures a tuple or multi-return value into multiple targets. The right-hand side must be a tuple expression in parentheses or a function call returning a tuple.

## Return

```
fn Add(a: int, b: int) -> int {
    return a + b
}

fn Log(msg: string) -> void {
    WritelnOut(msg)
    return
}
```

`return` exits the current function. A `void` function uses bare `return` or omits it.

## If

```
if x > 0 {
    WritelnOut("positive")
}

if x > 0 {
    WritelnOut("positive")
} else {
    WritelnOut("non-positive")
}

if x > 0 {
    WritelnOut("positive")
} else if x == 0 {
    WritelnOut("zero")
} else {
    WritelnOut("negative")
}
```

Braces are required. There is no parenthesized condition — the expression goes directly after `if`. `else if` chains are supported.

## While

```
while n > 0 {
    n = n / 10
    digits += 1
}
```

Executes the body while the condition is true. The condition must be `bool`.

## For

```
for value in items {
    WritelnOut(ToString(value))
}

for i, ch in name {
    WritelnOut(Concat(ToString(i), Concat(": ", ToString(RuneToInt(ch)))))
}

for _, v in pairs {
    total += v
}
```

Iterates over a collection. The loop variable(s) before `in` bind the index and/or value for each element. Use `_` to discard a variable. A single variable before `in` binds the value; two variables bind index and value.

### Iterable types

| Type        | One variable | Two variables      |
| ----------- | ------------ | ------------------ |
| `list[T]`   | `v: T`       | `i: int, v: T`     |
| `string`    | `ch: rune`   | `i: int, ch: rune` |
| `bytes`     | `b: byte`    | `i: int, b: byte`  |
| `map[K, V]` | `k: K`       | `k: K, v: V`       |
| `set[T]`    | `v: T`       | not allowed        |

Map and set iteration order is unspecified. Sets do not support the two-variable form. Mutating a collection while iterating it is undefined behavior.

### Range

`range` is loop syntax, not a function — it can only appear as the target of a `for` loop.

```
for i in range(10) {
    WritelnOut(ToString(i))
}

for i in range(2, 10) {
    WritelnOut(ToString(i))
}

for i in range(10, 0, -1) {
    WritelnOut(ToString(i))
}
```

`range(end)` iterates from `0` to `end - 1`. `range(start, end)` iterates from `start` to `end - 1`. `range(start, end, step)` iterates from `start` toward `end` (exclusive) by `step`. `step` must be nonzero; negative `step` counts downward. All arguments must be `int`. The loop variable binds the current value. The two-variable form is not supported — the value is the index.

## Break and Continue

```
while true {
    let line: string? = ReadLine()
    if line == nil {
        break
    }
    if StartsWith(line, "#") {
        continue
    }
    Process(line)
}
```

`break` exits the innermost loop. `continue` skips to the next iteration.

## Try / Catch / Throw

Any `obj` can be thrown — see the Type System section for details. `catch` blocks use the same type-matching semantics as `match` cases. Unlike `match`, `catch` does not require exhaustiveness; unmatched exceptions propagate to the caller.

```
try {
    let n: int = ParseInt(input, 10)
    WritelnOut(ToString(n))
} catch e: ValueError {
    WritelnErr(Concat("bad input: ", e.message))
}
```

`try` executes the body. If an exception is thrown, control transfers to the first `catch` whose type matches. Each `catch` binds the exception to a named variable of that type.

```
try {
    RiskyOperation()
} catch e: KeyError {
    WritelnErr(e.message)
} catch e: obj {
    WritelnErr(Concat("unexpected: ", ToString(e)))
} finally {
    Cleanup()
}
```

`finally` is optional and executes unconditionally — on normal exit, after a catch, or on return from within the try body.

```
throw ValueError("unexpected token")
throw "something went wrong"
```

`throw` accepts any expression. `throw` with an existing variable re-throws it.

`catch` can name a union of types to handle multiple exception types in one clause. The binding's type is the union — access to shared fields like `.message` requires all member types to have that field with the same type.

```
try {
    Process(input)
} catch e: ValueError | KeyError {
    WritelnErr(e.message)
}
```

### Assert

`Assert(cond)` and `Assert(cond, msg)` evaluate the condition and throw `AssertError` if false. The optional second argument provides the error message. Backends emit language-appropriate mechanisms (panic, exception, process exit).

```
Assert(n > 0)
Assert(Len(items) == expected, "wrong count")
```

## Tuples

```
let pair: (int, string) = (1, "hello")
let x: int = pair.0
let y: string = pair.1

let q: int
let r: int
q, r = DivMod(17, 5)
```

Tuples are fixed-size, heterogeneous, immutable. Elements are accessed by position with `.0`, `.1`, etc. Tuple assignment destructures into multiple targets.

Tuple types have two or more elements — there is no single-element tuple.

## Lists

```
let xs: list[int] = [1, 2, 3]
let empty: list[string] = []
let first: int = xs[0]
let mid: list[int] = xs[1:3]
let n: int = Len(xs)
```

`list[T]` is an ordered, mutable, variable-length sequence of elements of type `T`. Indexing (`xs[i]`) yields a `T`; throws `IndexError` if out of bounds. Slicing (`xs[a:b]`) yields a new `list[T]`; throws `IndexError` if either bound is out of range.

### Functions

| Function           | Signature                 | Description                                        |
| ------------------ | ------------------------- | -------------------------------------------------- |
| `Len(xs)`          | `list[T] -> int`          | element count                                      |
| `Append(xs, v)`    | `list[T], T -> void`      | append element to end                              |
| `Insert(xs, i, v)` | `list[T], int, T -> void` | insert element at index                            |
| `Pop(xs)`          | `list[T] -> T`            | remove and return last; throws IndexError if empty |
| `RemoveAt(xs, i)`  | `list[T], int -> void`    | remove element at index                            |
| `IndexOf(xs, v)`   | `list[T], T -> int`       | index of first occurrence, -1 if missing           |
| `Contains(xs, v)`  | `list[T], T -> bool`      | membership test                                    |
| `Repeat(xs, n)`    | `list[T], int -> list[T]` | repeat list n times; n ≤ 0 yields empty list       |
| `Reversed(xs)`     | `list[T] -> list[T]`      | new list in reverse order                          |
| `Sorted(xs)`       | `list[T] -> list[T]`      | new list in ascending order                        |

`Sorted` requires `T` to be an ordered type (`int`, `float`, `byte`, `rune`, `string`).

## Maps

```
let ages: map[string, int] = {"alice": 30, "bob": 25}
let empty: map[string, int] = Map()
let age: int = ages["alice"]
ages["charlie"] = 35
```

`map[K, V]` is an unordered mutable mapping from keys of type `K` to values of type `V`. `K` must be a hashable type (primitives, strings, runes, enums, tuples of hashable types).

Indexing (`m[k]`) yields a `V`; throws `KeyError` if `k` is not present. Assigning to an index (`m[k] = v`) inserts or updates.

### Functions

| Function             | Signature                           | Description                                |
| -------------------- | ----------------------------------- | ------------------------------------------ |
| `Len(m)`             | `map[K, V] -> int`                  | number of entries                          |
| `Map()`              | `-> map[K, V]`                      | empty map (type from context)              |
| `Contains(m, k)`     | `map[K, V], K -> bool`              | key membership test                        |
| `Get(m, k)`          | `map[K, V], K -> V?`                | value for key, or nil if missing           |
| `Get(m, k, default)` | `map[K, V], K, V -> V`              | value for key, or default if missing       |
| `Delete(m, k)`       | `map[K, V], K -> void`              | remove entry by key                        |
| `Keys(m)`            | `map[K, V] -> list[K]`              | list of keys; order unspecified            |
| `Values(m)`          | `map[K, V] -> list[V]`              | list of values; order unspecified          |
| `Items(m)`           | `map[K, V] -> list[(K, V)]`         | list of key-value pairs; order unspecified |
| `Merge(m1, m2)`      | `map[K, V], map[K, V] -> map[K, V]` | new map; m2 wins on key conflict           |

`Get` is a built-in with two arities. `Assert` (see Try/Catch) likewise accepts one or two arguments. User-defined functions cannot declare optional parameters.

Iteration with `for k, v in m` yields key-value pairs.

## Sets

```
let seen: set[int] = {1, 2, 3}
let empty: set[string] = Set()
```

`set[T]` is an unordered mutable collection of unique values of type `T`. `T` must be a hashable type. Non-empty sets can use literal syntax `{v1, v2, ...}`. Empty sets use `Set()` — `{}` is not valid since it would be ambiguous.

### Functions

| Function         | Signature           | Description                   |
| ---------------- | ------------------- | ----------------------------- |
| `Len(s)`         | `set[T] -> int`     | number of elements            |
| `Set()`          | `-> set[T]`         | empty set (type from context) |
| `Add(s, v)`      | `set[T], T -> void` | add element                   |
| `Remove(s, v)`   | `set[T], T -> void` | remove element                |
| `Contains(s, v)` | `set[T], T -> bool` | membership test               |

## Collection Equality

`==` and `!=` work on lists, maps, and sets with deep structural comparison. Lists compare element-wise in order. Maps compare by key-value pairs regardless of insertion order. Sets compare by membership.



## Indexing and Slicing

Indexing and slicing apply to `string`, `bytes`, and `list[T]`. The rules below are universal across all three.

### Non-negative indices

Taytsh indices are non-negative. Negative indexing is a Python feature resolved during lowering:

- Literal negative indices: `x[-1]` lowers to `x[Len(x) - 1]`
- Dynamic indices that may be negative: the lowerer inserts `i < 0 ? Len(x) + i : i`

At the Taytsh level, a negative index that reaches runtime is out of bounds and throws `IndexError`.

### Slice bounds

Both bounds of a slice are always present. Open-ended slices are a Python feature resolved during lowering:

- `xs[:3]` lowers to `xs[0:3]`
- `xs[2:]` lowers to `xs[2:Len(xs)]`
- `xs[:-1]` lowers to `xs[0:Len(xs) - 1]`

The grammar enforces this — `[Expr : Expr]` requires both expressions. Backends that want to emit idiomatic open-ended slices can detect `0` or `Len(x)` bounds, or use the `open_start` / `open_end` provenance annotations.

If `lo > hi`, the slice throws `IndexError`. Slicing always produces a copy. `xs[0:Len(xs)]` is a full copy of `xs`.

## Value and Reference Semantics

Mutable types — `list[T]`, `map[K, V]`, `set[T]`, and structs — have reference semantics. Assignment and parameter passing create aliases. Mutation through one alias is visible through all others.

```
let a: list[int] = [1, 2, 3]
let b: list[int] = a
Append(b, 4)
-- Len(a) == 4. a and b are aliases of the same list.
```

```
fn Bump(t: Token) -> void {
    t.offset += 1       -- caller sees the change
}
```

Immutable types — primitives (`int`, `float`, `bool`, `byte`, `rune`), `string`, `bytes`, tuples, and enums — have no observable mutation, so the value/reference distinction is unobservable. Backends represent them however they like.

Slicing is the one explicit carve-out: `xs[0:3]` returns a new value that does not alias the source (see Indexing and Slicing).

### Representation deferral

Reference semantics defines what the program means, not how backends represent it. The middleend analyzes whether a mutable value is actually aliased — whether more than one live variable refers to the same underlying data. The common case (a list is created, mutated, and returned but never shared) does not require reference types at the backend level. Backends for value-semantics languages (Go, Rust, Swift, C, Zig) can emit value types for non-aliased values and reference types only where aliasing occurs.

## Optional

```
let x: int? = nil
let y: int? = 42
```

`T?` is sugar for `T` or `nil`. Any type can be made optional: `int?`, `string?`, `list[int]?`, `Token?`. `T??` is not valid — optionals do not nest. See also Union Types below.

### Nil narrowing

```
fn Lookup(m: map[string, int], key: string) -> int {
    let v: int? = Get(m, key)
    if v != nil {
        return v + 1
    }
    return 0
}
```

After a `!= nil` check, the variable's type narrows from `T?` to `T` within the branch. This is the primary way to work with optional values.

### Unwrap

| Function    | Signature | Description                           |
| ----------- | --------- | ------------------------------------- |
| `Unwrap(x)` | `T? -> T` | extract value; throws NilError if nil |

`Unwrap` extracts the value when the program can guarantee non-nil but control flow doesn't prove it.

## Union Types

```
let v: int | string = 42
let w: string | bytes = "hello"
fn Process(v: int | string | nil) -> void { ... }
```

`A | B` is a union type — a value that is one of the listed member types. `|` separates members in type position (after `:` or `->`), where it is unambiguous with the bitwise OR operator (which lives in expression position).

### Syntax and parsing

`|` binds tighter than `,` because `|` is consumed within the `Type` parse before returning to the enclosing list. So `fn[int | string, bool]` is `fn[(int | string), bool]`, and `map[string, int | float]` is `map[string, (int | float)]`.

### Interaction with `T?`

`?` applies to the entire union: `int | string?` parses as `(int | string)?` = `int | string | nil`. To include nil in a union without `?`, write `| nil` explicitly. `int | nil` is equivalent to `int?`. Both forms are valid.

`T??` remains invalid. `void` cannot appear as a union member (not a value type).

### Interaction with interfaces

Interfaces and unions are distinct concepts. `Node` (an interface) and `Literal | BinOp` (a union of its implementors) are different types, even if extensionally equivalent. Mixing is allowed: `Node | int` means "any Node variant or an int."

### Interaction with `obj`

Any union containing `obj` reduces to `obj`.

### Normalization rules

1. **Flatten:** `A | (B | C)` → `A | B | C`
2. **Deduplicate:** `A | A` → `A`
3. **Absorb obj:** `A | obj` → `obj`
4. **Single remaining:** if reduced to 1 type, it's that type (not a 1-element union)
5. **Unordered:** `int | string` = `string | int`

### Zero values

Union types containing `nil` have zero value `nil`. Others have no zero value and require an explicit initializer (like structs and enums).

### Type narrowing

- After `match`, the variable narrows to the matched type
- After `!= nil`, nil is removed from the union
- If removing nil leaves one type, the variable narrows to that type (consistent with existing `T?` narrowing)

### Equality and operators

- `==` / `!=`: defined on unions. Equal iff same variant type and equal values.
- Ordering (`<`, `<=`, etc.): not defined on unions. Narrow first.
- Arithmetic: not defined on unions. Narrow first.
- `ToString`: works (dispatches to held type).

## Conversions

```
let f: float = IntToFloat(42)
let n: int = FloatToInt(3.14)
let b: byte = IntToByte(65)
let code: int = ByteToInt(b)
```

| Function        | Signature      | Description            |
| --------------- | -------------- | ---------------------- |
| `IntToFloat(n)` | `int -> float` | exact if representable |
| `FloatToInt(x)` | `float -> int` | truncate toward zero   |
| `ByteToInt(b)`  | `byte -> int`  | widen to int           |
| `IntToByte(n)`  | `int -> byte`  | low 8 bits             |

No implicit coercion between any types. All conversions are explicit function calls. `ToString` is universal (see Type System). `ParseInt` is in the Strings section. `RuneToInt` and `RuneFromInt` handle rune↔int. `ByteToInt` and `IntToByte` handle byte↔int.

## Structs

```
struct Token {
    kind: TokenKind
    value: string
    offset: int
}

-- positional construction
let t: Token = Token(TokenKind.Ident, "foo", 0)

-- or named construction
let t2: Token = Token(kind: TokenKind.Ident, value: "foo", offset: 0)
let k: TokenKind = t.kind
t.offset = 10
```

A struct has a name and one or more typed fields. Construction is positional or named. Positional arguments follow the field declaration order. Named arguments use `field: value` syntax and may appear in any order. Mixing positional and named arguments in a single construction is not allowed. Fields are accessed with `.` and are mutable.

### Methods

```
struct Span {
    start: int
    end: int

    fn Len(self) -> int {
        return self.end - self.start
    }
}

let s: Span = Span(0, 10)
let n: int = s.Len()
```

Methods are functions declared inside a struct with `self` as the first parameter. `self` is the receiver instance; its type is the enclosing struct. Methods are called with `.` syntax.

## Interfaces

```
interface Node {}

struct Literal : Node {
    value: int
}

struct BinOp : Node {
    op: string
    left: Node
    right: Node
}
```

An interface declares a type that can be one of several struct variants. Structs declare which interface they implement with `: InterfaceName`. A struct may implement at most one interface.

Every interface is automatically sealed — the compiler sees all implementations (see Module Structure) and can verify exhaustive matching.

Interface bodies are empty. They define no fields or methods; they exist solely as the root of a closed type hierarchy. Common operations are free functions that match internally.

## Enums

```
enum TokenKind {
    Ident
    Number
    String
    LParen
    RParen
    Plus
    Minus
    Eof
}

let k: TokenKind = TokenKind.Ident
```

An enum defines a set of named constants with no associated data. Enum values are accessed as `EnumName.Variant` and compared with `==` and `!=`.

## Match

`match` dispatches on the runtime type of a value. Each `case` names a type and binds a variable of that type, or names an enum value. A match must be exhaustive — every possible type or value must be covered. `default` satisfies exhaustiveness for any remaining cases.

### Interface matching

```
match node {
    case lit: Literal {
        WritelnOut(ToString(lit.value))
    }
    case bin: BinOp {
        Eval(bin.left)
        WriteOut(bin.op)
        Eval(bin.right)
    }
}
```

Each case names a variant struct and binds a typed variable. Exhaustive — every struct implementing the interface must have a case (or `default`).

### Enum matching

```
match kind {
    case TokenKind.Ident {
        ParseIdent()
    }
    case TokenKind.Number {
        ParseNumber()
    }
    default {
        ParseOther()
    }
}
```

Each case names a qualified enum value. Exhaustive — every variant must have a case (or `default`).

### Optional matching

```
match result {
    case v: int {
        WritelnOut(ToString(v))
    }
    case nil {
        WritelnOut("absent")
    }
}
```

`case nil` matches the nil case. Since `nil` is a type in type position, no binding is needed — the value is always `nil`. Exhaustive — both the value type and `nil` must be covered.

### Obj matching

```
match value {
    case n: int {
        WritelnOut(ToString(n))
    }
    case s: string {
        WritelnOut(s)
    }
    default {
        WritelnOut("something else")
    }
}
```

When matching on `obj`, cases name concrete types. `default` is required — the set of all types cannot be enumerated.

### Union matching

```
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
```

Each union member must be covered by a case, or `default` must be present. `case nil` handles nil members (same as optional matching).

When a union member is an interface, a case naming the interface matches all its variants. Individual variant cases also work — the exhaustiveness checker recognizes that covering all variants covers the interface.

```
fn Process(v: Node | int) -> void {
    match v {
        case n: int {
            WritelnOut(ToString(n))
        }
        case node: Node {
            EvalNode(node)
        }
    }
}
```

### Default

`default` matches any value not covered by preceding cases. `default x: obj` binds the value for use in the body.

```
match value {
    case n: int {
        WritelnOut(ToString(n))
    }
    default o: obj {
        WritelnErr(Concat("unexpected: ", ToString(o)))
    }
}
```

## I/O

I/O covers stdin, stdout, stderr, and file read/write.

```
WritelnOut(ToString(42))
WritelnErr("error: bad input")
let line: string? = ReadLine()
let input: string = ReadAll()
let data: bytes = ReadBytes()
let chunk: bytes = ReadBytesN(1024)
WriteOut(Encode("binary output"))
let content: string | bytes = ReadFile("input.txt")
WriteFile("output.txt", result)
let args: list[string] = Args()
let home: string? = GetEnv("HOME")
Exit(1)
```

### Output

Four output functions write to stdout or stderr. Each accepts `string` or `bytes`. The `Writeln` variants append a newline after writing.

| Function        | Signature                 | Description      |
| --------------- | ------------------------- | ---------------- |
| `WriteOut(d)`   | `string \| bytes -> void` | write to stdout  |
| `WriteErr(d)`   | `string \| bytes -> void` | write to stderr  |
| `WritelnOut(d)` | `string \| bytes -> void` | stdout + newline |
| `WritelnErr(d)` | `string \| bytes -> void` | stderr + newline |

To write other types, convert first: `WritelnOut(ToString(n))`.

### Input

| Function        | Signature      | Description                                            |
| --------------- | -------------- | ------------------------------------------------------ |
| `ReadLine()`    | `-> string?`   | read one line from stdin, strip \n or \r\n; nil on EOF |
| `ReadAll()`     | `-> string`    | read all of stdin as string                            |
| `ReadBytes()`   | `-> bytes`     | read all of stdin as bytes                             |
| `ReadBytesN(n)` | `int -> bytes` | read up to n bytes from stdin                          |

### Files

| Function             | Signature                         | Description       |
| -------------------- | --------------------------------- | ----------------- |
| `ReadFile(path)`     | `string -> string \| bytes`       | read entire file  |
| `WriteFile(path, d)` | `string, string \| bytes -> void` | write entire file |

`ReadFile` returns `string` if the file contains valid UTF-8, `bytes` otherwise. `WriteFile` accepts `string` or `bytes` — string arguments are written as UTF-8 — and writes the contents to the file, creating it if it doesn't exist and overwriting if it does. Both throw `IOError` on failure (missing file, permission denied, etc.).

### System

| Function       | Signature           | Description                        |
| -------------- | ------------------- | ---------------------------------- |
| `Args()`       | `-> list[string]`   | command-line arguments             |
| `GetEnv(name)` | `string -> string?` | environment variable; nil if unset |
| `Exit(code)`   | `int -> void`       | terminate with exit code           |

## Math Semantics

By default, Taytsh targets correct-enough portable behavior across all backends. Where targets disagree on edge-case behavior, Taytsh leaves it unspecified rather than imposing costly emulation. Two strict flags trade performance for cross-target consistency: `--strict-math` (arithmetic) and `--strict-tostring` (string representation). `--strict` enables both.

### Integers

Integers are signed, at least 53 bits. Most targets provide 64-bit integers; JavaScript and TypeScript provide 53-bit integer range (IEEE 754 double). Overflow beyond the target's integer range is unspecified — programs that depend on specific overflow or wrapping behavior are out of scope.

### Division and Remainder

Int and byte division by zero throws `ZeroDivisionError`. `/` truncates toward zero; `%` follows the dividend's sign. These are specified in the Operators section.

Float division by zero follows IEEE 754: `1.0 / 0.0` is `+Inf`, `-1.0 / 0.0` is `-Inf`, `0.0 / 0.0` is `NaN`. Float remainder by zero (`%` with zero divisor) is unspecified — targets disagree (some return NaN, some throw).

### Bitwise Operations

`~x` is two's complement: `~x == -(x + 1)`. This identity holds for any integer width ≥ the value's bit length.

Shift amounts must be non-negative. Behavior when the shift amount equals or exceeds the integer's bit width is unspecified.

### Floating Point

Floats are IEEE 754 binary64 (double precision) on all targets.

| Rule            | Description                                                                                    |
| --------------- | ---------------------------------------------------------------------------------------------- |
| NaN != NaN      | NaN is not equal to itself                                                                     |
| -0.0 == 0.0     | negative zero equals positive zero                                                             |
| NaN propagation | `Min` and `Max` propagate NaN; targets disagree (some swallow NaN), backends may need wrappers |
| NaN ordering    | `Sorted` on a list containing NaN is unspecified; strict mode traps                            |

| Function   | Signature       | Description                                |
| ---------- | --------------- | ------------------------------------------ |
| `IsNaN(x)` | `float -> bool` | true if x is NaN                           |
| `IsInf(x)` | `float -> bool` | true if x is positive or negative infinity |

### Rounding

`Round(x)` uses half-away-from-zero: `Round(0.5) == 1`, `Round(-0.5) == -1`. `Floor` and `Ceil` follow standard mathematical definitions. All three return `int` and throw `ValueError` if the argument is NaN, positive infinity, or negative infinity, or if the result is outside the representable integer range.

### Conversions

`FloatToInt(x)` truncates toward zero. Throws `ValueError` if the value is NaN or outside the representable integer range.

`IntToFloat(n)` converts exactly if representable; may lose precision for large values. This is not a trap — the result is the nearest representable double.

### Exponentiation

`Pow(0, 0) == 1`. Integer `Pow` with negative exponent is unspecified. Overflow follows integer overflow rules (unspecified).

### Parsing

`ParseInt(s, base)` and `ParseFloat(s)` throw `ValueError` on invalid input. Valid bases for `ParseInt` are 2–36.

### Strict Math

The `--strict-math` flag enables bit-identical arithmetic across targets. All 12 strict-mode targets produce exactly the same integer and float results for the same inputs.

Excluded targets:

| Target     | Reason                                                    |
| ---------- | --------------------------------------------------------- |
| JavaScript | 53-bit integer range; cannot represent 64-bit integers    |
| TypeScript | same as JavaScript                                        |
| C#         | .NET JIT performs FMA contraction; no in-code opt-out     |
| Swift      | LLVM backend performs FMA contraction; no in-code opt-out |

When strict math is enabled:

| Property                          | Default mode        | Strict mode         |
| --------------------------------- | ------------------- | ------------------- |
| Integer width                     | at least 53 bits    | exactly 64 bits     |
| Integer overflow                  | unspecified         | traps               |
| Shift ≥ 64                        | unspecified         | traps               |
| Negation of `INT64_MIN`           | unspecified         | traps               |
| `Pow(int, int)` overflow          | unspecified         | traps               |
| `Pow(int, int)` negative exponent | unspecified         | traps               |
| `Min`/`Max` with NaN              | backends may differ | NaN propagates      |
| `Round`                           | half-away-from-zero | half-away-from-zero |
| Float `%` with zero divisor       | unspecified         | traps               |
| `Sorted` with NaN                 | unspecified         | traps               |
| Available targets                 | all 16              | 12                  |

Strict mode integers are signed two's complement, exactly 64 bits. Overflow on any integer operation — addition, subtraction, multiplication, negation, left shift, exponentiation — is a runtime error. Targets with arbitrary-precision integers (Python, Ruby) emit range checks to enforce 64-bit bounds. Targets with native overflow detection use it: Rust (`checked_add`), Java (`Math.addExact`), C (`__builtin_add_overflow`), Zig (`@addWithOverflow`). Remaining targets (Go, Dart, Lua, PHP, Perl) emit manual comparison checks.

Strict mode floats are IEEE 754 binary64, same as default mode. Basic operations (`+`, `-`, `*`, `/`) are already bit-identical across all targets by IEEE 754 mandate. Strict mode additionally specifies `Min`, `Max`, and float `%` behavior — backends emit inline wrappers where the native function disagrees. FMA contraction is controlled per-target: C (`#pragma STDC FP_CONTRACT OFF`), Zig (`f64` is IEEE-compliant by default), Rust (no contraction by default). Interpreted and VM targets (Python, Ruby, Go, Java, Dart, Lua, PHP, Perl) do not perform FMA contraction.

`Pow(float, float)` with non-integer exponent is not available in strict mode — underlying `exp`/`log` implementations are not mandated by IEEE 754 and differ across targets. Integer exponents use binary exponentiation (emitted inline, exact within 64-bit range).

The strict math flag is recorded in the IR module metadata. Middleend passes and backends read it to select checked or unchecked emission.

### Strict ToString

The `--strict-tostring` flag specifies a canonical `ToString` format for every type, so string output is identical across all 16 targets.

| Property                   | Default mode  | Strict mode                                             |
| -------------------------- | ------------- | ------------------------------------------------------- |
| `int`                      | target-native | decimal, e.g. `"42"`, `"-7"`                            |
| `float`                    | target-native | shortest round-trip                                     |
| `float` special values     | target-native | `"NaN"`, `"Inf"`, `"-Inf"`                              |
| `float` negative zero      | target-native | `"-0.0"`                                                |
| `bool`                     | target-native | `"true"` / `"false"`                                    |
| `byte`                     | target-native | decimal, e.g. `"255"`                                   |
| `rune`                     | target-native | the character itself                                    |
| `nil`                      | target-native | `"nil"`                                                 |
| `string`                   | identity      | identity                                                |
| `bytes`                    | target-native | all hex, e.g. `"b\"\\x89\\x50\\x4e\\x47\""`             |
| `list[T]`                  | target-native | `[1, 2, 3]`                                             |
| `map[K, V]`                | target-native | `{"a": 1, "b": 2}` sorted keys                          |
| `set[T]`                   | target-native | `{1, 2, 3}` sorted                                      |
| `(T, U, ...)`              | target-native | `(1, "hello")`                                          |
| `struct`                   | target-native | `Token{kind: TokenKind.Ident, value: "foo", offset: 0}` |
| `enum`                     | target-native | `TokenKind.Ident`                                       |
| `fn[...]`                  | target-native | `fn[int, int]`                                          |
| struct `ToString` override | n/a           | `ToString(self)` method                                 |
| Available targets          | all 16        | all 16                                                  |

"Shortest round-trip" means the fewest decimal digits such that parsing the string back produces the exact same float — algorithms like Ryū compute this. Backends that lack a native shortest-round-trip conversion emit a custom formatter.

In composite contexts (collections, tuples, struct fields), `string` values are quoted with `"` and `rune` values are quoted with `'`. All other types are unquoted. This avoids ambiguity when parsing ToString output visually.

Map and set `ToString` output sorts elements for deterministic output regardless of runtime iteration order.

Structs may declare a `fn ToString(self) -> string` method to override the default format. `ToString` is normally a reserved name; this is the one permitted exception. The override applies in both default and strict modes. Enums always use the `EnumName.Variant` format.

The strict tostring flag is recorded in the IR module metadata.

### Strict

`--strict` enables `--strict-math` and `--strict-tostring`. The available target set is the intersection — 12 targets (excluding JavaScript, TypeScript, C#, and Swift, per `--strict-math` restrictions).

### Pragmas

Strict flags can be set within the source file using pragma comments on the first lines, before any declarations:

```
-- pragma strict-math
-- pragma strict-tostring
```

Pragmas are equivalent to the corresponding command-line flags. Both `-- pragma strict-math` and `-- pragma strict-tostring` together are equivalent to `--strict`. Pragmas must appear before any non-comment, non-blank content. A pragma comment is exactly `-- pragma strict-math` or `-- pragma strict-tostring` — no other forms are recognized.

## Source Metadata

Every Taytsh node has a `metadata` field of type `map[string, obj]`. The lowerer populates it during IR construction. Metadata is advisory — backends and raising passes may use it but are never required to.

The `pos` key is present on every node. All other keys are optional and node-type-specific.

### Position

Every node carries `pos`: a `(int, int)` of `(line, col)`, 1-indexed, for error reporting and source maps.

### Literal Notation

Literal nodes carry keys that preserve the original Python source form, enabling backends to emit idiomatic notation.

| Key          | Type     | Node            | Values                             | Purpose                    |
| ------------ | -------- | --------------- | ---------------------------------- | -------------------------- |
| `base`       | `string` | Integer literal | `"dec"`, `"hex"`, `"oct"`, `"bin"` | preserve source notation   |
| `large`      | `bool`   | Integer literal |                                    | value exceeds 64-bit range |
| `separators` | `bool`   | Integer literal |                                    | source used `_` separators |
| `scientific` | `bool`   | Float literal   |                                    | preserve `1e10` form       |
| `separators` | `bool`   | Float literal   |                                    | source used `_` separators |

A backend that doesn't support hex literals can emit decimal instead. The `large` flag lets backends that lack arbitrary-precision integers emit a compile error or use a big-integer library.

### Lowering Provenance

Some Python patterns are desugared during lowering into simpler IR forms. The original pattern is recorded under the `provenance` key (type `string`) so middleend raising passes can reconstruct idiomatic forms for targets that support them.

| `provenance` value   | Taytsh form                    | Python source               |
| -------------------- | ------------------------------ | --------------------------- |
| `chained_comparison` | `a < b && b < c`               | `a < b < c`                 |
| `list_comprehension` | for loop + `Append`            | `[x*2 for x in items]`      |
| `dict_comprehension` | for loop + map insert          | `{k: v for k, v in items}`  |
| `set_comprehension`  | for loop + `Add`               | `{x for x in items}`        |
| `in_operator`        | `Contains(xs, v)`              | `v in xs`                   |
| `not_in_operator`    | `!Contains(xs, v)`             | `v not in xs`               |
| `truthiness`         | `Len(xs) > 0`, `s != ""`, etc. | `if items:`, `if s:`        |
| `enumerate`          | `for i, v in xs`               | `for i, v in enumerate(xs)` |
| `string_multiply`    | `Repeat(s, n)`                 | `s * n`                     |
| `list_multiply`      | `Repeat(xs, n)`                | `xs * n`                    |
| `negative_index`     | `x[Len(x) - 1]`                | `x[-1]`                     |
| `open_start`         | `xs[0:n]`                      | `xs[:n]`                    |
| `open_end`           | `xs[n:Len(xs)]`                | `xs[n:]`                    |

Provenance is advisory — the lowered form is always correct as-is.

## Middleend Annotations

Every Taytsh node has an `annotations` field of type `map[string, obj]`. Middleend passes read the IR and write analysis results into this map. Keys are namespaced by pass (e.g. `"scope.is_reassigned"`, `"ownership.region"`).

Annotations are write-once — a pass sets a key, and no later pass overwrites it. Backends read annotations but never write them.

The specific annotations produced by each pass are defined in the middleend specs, not this document.

## Grammar

Notation is EBNF: `|` alternation, `( )?` optional, `( )*` zero or more, `( )+` one or more. Terminals in `'quotes'`; `\` has no escape meaning within quotes or character classes, so `'\'` is a literal backslash and `[^"\]` excludes `"` and `\`. The grammar targets recursive descent with at most two tokens of lookahead. The few spots requiring the second token are noted inline.

### Tokens

```
INT        = [0-9]+
BYTE       = '0x' hex hex
FLOAT      = [0-9]+ '.' [0-9]+ ( [eE] [+-]? [0-9]+ )?
           | [0-9]+ [eE] [+-]? [0-9]+
STRING     = '"' ( escape | [^"\] )* '"'
RUNE       = "'" ( escape | [^'\] ) "'"
BYTES      = 'b"' ( escape | [^"\] )* '"'
IDENT      = [a-zA-Z_] [a-zA-Z0-9_]*
escape     = '\' ( 'n' | 'r' | 't' | '\' | '"' | "'" | '0'
           | 'x' hex hex )
hex        = [0-9a-fA-F]
```

Keywords are reserved and cannot appear as `IDENT`:

```
bool      break     byte      bytes     case      catch
continue  default   else      enum      false     finally
float     fn        for       if        in        int
interface let       list      map       match     nil
obj       range     return    rune      self      set
string    struct    throw     true      try       void
while
```

Line comments start with `--` and run to end of line. No block comments.

Whitespace (spaces, tabs, newlines) separates tokens but is otherwise insignificant. There are no semicolons — statements are self-delimiting because every statement form begins with a keyword or an expression, and the parser knows when each form ends by its structure.

### Top Level

```
Program       = Decl*
Decl          = FnDecl | StructDecl | InterfaceDecl | EnumDecl

FnDecl        = 'fn' IDENT '(' ParamList ')' '->' Type Block
ParamList     = ( 'self' ( ',' Param )* | Param ( ',' Param )* )?
Param         = IDENT ':' Type
Block         = '{' Stmt* '}'

StructDecl    = 'struct' IDENT ( ':' IDENT )? '{' StructBody '}'
StructBody    = FnDecl* FieldDecl ( FieldDecl | FnDecl )*
FieldDecl     = IDENT ':' Type

InterfaceDecl = 'interface' IDENT '{' '}'

EnumDecl      = 'enum' IDENT '{' IDENT+ '}'
```

### Statements

Every statement form starts with a distinct keyword except assignment and expression statements, which both start with an expression. Those are merged into `ExprStmt` and disambiguated by what follows.

```
Stmt       = LetStmt
           | IfStmt
           | WhileStmt
           | ForStmt
           | MatchStmt
           | TryStmt
           | ReturnStmt
           | 'break'
           | 'continue'
           | 'throw' Expr
           | ExprStmt

LetStmt    = 'let' IDENT ':' Type ( '=' Expr )?
ReturnStmt = 'return' Expr?
IfStmt     = 'if' Expr Block ( 'else' ( IfStmt | Block ) )?
WhileStmt  = 'while' Expr Block
ForStmt    = 'for' Binding 'in' ( Expr | Range ) Block
Binding    = IDENT ( ',' IDENT )?
Range      = 'range' '(' Expr ( ',' Expr ( ',' Expr )? )? ')'
MatchStmt  = 'match' Expr '{' Case+ Default? '}'
           | 'match' Expr '{' Default '}'
Case       = 'case' Pattern Block
Pattern    = IDENT ':' TypeName
           | IDENT '.' IDENT
           | 'nil'
Default    = 'default' ( IDENT ':' 'obj' )? Block
TryStmt    = 'try' Block ( Catch+ ( 'finally' Block )? | 'finally' Block )
Catch      = 'catch' IDENT ':' TypeName ( '|' TypeName )* Block
TypeName   = IDENT | 'obj' | 'nil' | 'int' | 'float' | 'bool'
           | 'byte' | 'bytes' | 'string' | 'rune'
           | 'list' '[' Type ']'
           | 'map' '[' Type ',' Type ']'
           | 'set' '[' Type ']'
           | '(' Type ',' Type ( ',' Type )* ')'
           | 'fn' '[' Type ( ',' Type )* ']'
```

`ForStmt` needs two tokens of lookahead: after the first `IDENT`, peek for `,` (two bindings) versus `in` (one binding). `_` is a regular identifier; discard semantics are not a grammar concern.

`Pattern` in a `Case`: if the token is `nil`, no lookahead needed. Otherwise, after the first `IDENT`, peek for `:` (type binding) versus `.` (enum value). `Default` is unambiguous — it starts with `default`, then peek for `IDENT` (binding) versus `{` (no binding).

```
ExprStmt   = Expr ( AssignTail )?
AssignTail = AssignOp Expr
           | ( ',' Expr )+ '=' Expr
AssignOp   = '=' | '+=' | '-=' | '*=' | '/=' | '%='
           | '&=' | '|=' | '^=' | '<<=' | '>>='
```

The left-hand `Expr` in an assignment must be a valid target (identifier, field access, or index expression). This is a semantic check, not a grammatical one. The `( ',' Expr )+` form handles tuple assignment.

### Expressions

One nonterminal per precedence level. Left-associative levels use iteration (`*`), right-associative levels use right-recursion.

```
Expr       = Ternary
Ternary    = Or ( '?' Expr ':' Ternary )?
Or         = And ( '||' And )*
And        = Compare ( '&&' Compare )*
Compare    = BitOr ( CompOp BitOr )?
CompOp     = '==' | '!=' | '<' | '<=' | '>' | '>='
BitOr      = BitXor ( '|' BitXor )*
BitXor     = BitAnd ( '^' BitAnd )*
BitAnd     = Shift ( '&' Shift )*
Shift      = Sum ( ( '<<' | '>>' ) Sum )*
Sum        = Product ( ( '+' | '-' ) Product )*
Product    = Unary ( ( '*' | '/' | '%' ) Unary )*
Unary      = ( '-' | '!' | '~' ) Unary
           | Postfix
Postfix    = Primary ( Suffix )*
Suffix     = '.' IDENT | '.' INT
           | '[' Expr ( ':' Expr )? ']'
           | '(' ArgList ')'
ArgList    = ( Arg ( ',' Arg )* )?
Arg        = IDENT ':' Expr
           | Expr
Primary    = INT | FLOAT | BYTE | STRING | RUNE | BYTES
           | 'true' | 'false' | 'nil'
           | IDENT
           | '(' Expr ( ',' Expr )+ ')'
           | '(' Expr ')'
           | '[' ( Expr ( ',' Expr )* )? ']'
           | '{' Expr ':' Expr ( ',' Expr ':' Expr )* '}'
           | '{' Expr ( ',' Expr )* '}'
           | FnLiteral
FnLiteral  = '(' ParamList ')' '->' Type Block
           | '(' ParamList ')' '->' Type '=>' Expr
```

`Ternary` is right-associative: the "then" branch is a full `Expr`, the "else" branch right-recurses into `Ternary`.

`Compare` allows at most one comparison operator — `a < b < c` is a parse error. Chained comparisons are desugared during lowering into `&&`-connected binary comparisons.

`Suffix` with `'['` uses one token of lookahead inside the brackets: after the first `Expr`, peek for `':'` to distinguish indexing from slicing. `Arg` in an `ArgList` needs two tokens of lookahead: if the next tokens are `IDENT ':'`, it's a named argument; otherwise parse as a positional expression. Mixing named and positional arguments is a semantic error, not a grammar concern.

`'('` in `Primary` has three interpretations: function literal, tuple, or parenthesized expression. If the contents match `ParamList ')' '->'`, it's a function literal. Otherwise, parse the first `Expr`, then peek for `','` (tuple) versus `)` (parenthesized expression). `'['` as a `Primary` is a list literal; as a `Suffix` it's indexing/slicing — the parser knows which by context (a `Suffix` only follows a `Primary`). `'{'` is a map or set literal: parse the first `Expr`, then peek for `':'` (map) versus `','` or `}` (set). Empty maps use `Map()`, empty sets use `Set()`.

### Types

```
Type       = UnionType ( '?' )?
UnionType  = BaseType ( '|' BaseType )*
BaseType   = 'obj'
           | 'int' | 'float' | 'bool'
           | 'byte' | 'bytes'
           | 'string' | 'rune'
           | 'nil'
           | 'void'
           | 'list' '[' Type ']'
           | 'map' '[' Type ',' Type ']'
           | 'set' '[' Type ']'
           | '(' Type ',' Type ( ',' Type )* ')'
           | 'fn' '[' Type ( ',' Type )* ']'
           | IDENT
```

Every alternative in `BaseType` starts with a distinct token, so no lookahead is needed. `|` in `UnionType` is unambiguous with bitwise OR because types only appear after `:` or `->` (type position), while `|` as an operator appears between expressions (expression position). After each `BaseType`, peek for `|` to continue the union or anything else to stop — zero additional lookahead beyond the standard single-token peek. `?` for optional applies to the entire union and is a single-token peek after the union; types only appear in declaration contexts (after `:` or `->`) where `?` is unambiguous with the ternary operator.

`|` in `Catch` uses the same `TypeName` production — after each `TypeName`, peek for `|` to continue or `{` to stop.

Tuple types require at least two elements — `(T)` is not a type. `IDENT` covers user-defined names: structs, interfaces, enums.
