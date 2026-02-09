# IR Specification

> **Note:** This document is an exploration of a future IR iteration, using language-like syntax to clarify thinking. It is not consistent with the rest of the spec.

## Introduction

The current IR is a large set of specialized nodes that carry Python-flavored semantics: `PythonAnd`, `PythonOr`, `PythonMod`, `FloorDiv`, `SentinelToOptional`, `CharAt`, `StringNonEmpty`, and dozens more. Lowering eagerly commits to representations — `range(a, b)` becomes `ForClassic`, string indexing becomes `CharAt`/`CharLen`/`Substring` — and every backend must interpret these nodes individually. The middleend is limited to read-only annotation passes.

This IR replaces that with a small, target-neutral language. Python's quirks (floor division, value-returning `and`/`or`, type-dependent truthiness) are resolved during lowering into conventional semantics (truncating division, boolean `&&`/`||`), but representation choices are deliberately deferred. The IR has one loop-over-collection form (`for...in`), not two; strings are rune-indexed at the IR level, not pre-lowered into character operations. The middleend analyzes how values are actually used — whether a `for...in` over a range should become a classic for loop, whether a string needs rune-level access or can stay in the target's native representation — and makes those decisions as shared passes rather than per-backend logic. Backends receive already-decided representations and translate to syntax.

This spec provides a textual syntax for the IR to facilitate exposition. The grammar is parseable by recursive descent with at most two tokens of lookahead.

## Numeric Types

```
let x: int = 42
let ratio: float = 3.14
let done: bool = true
```

| Type    | Literal         | Description                      |
| ------- | --------------- | -------------------------------- |
| `int`   | `42`            | signed integer, at least 64 bits |
| `float` | `3.14`          | floating point, at least 64 bits |
| `bool`  | `true`, `false` | boolean                          |

Target representations:

| Target     | `int`     | `float`   | Notes                             |
| ---------- | --------- | --------- | --------------------------------- |
| C          | `int64_t` | `double`  |                                   |
| C#         | `long`    | `double`  |                                   |
| Dart       | `int`     | `double`  |                                   |
| Go         | `int`     | `float64` | `int` is 64-bit on modern targets |
| Java       | `long`    | `double`  |                                   |
| JavaScript | `number`  | `number`  | IEEE 754 double; 53-bit int range |
| Lua        | `integer` | `number`  |                                   |
| Perl       | scalar    | scalar    |                                   |
| PHP        | `int`     | `float`   | 64-bit on modern targets          |
| Python     | `int`     | `float`   | int is arbitrary precision        |
| Ruby       | `Integer` | `Float`   | Integer is arbitrary precision    |
| Rust       | `i64`     | `f64`     |                                   |
| Swift      | `Int`     | `Double`  | `Int` is 64-bit on modern targets |
| TypeScript | `number`  | `number`  | IEEE 754 double; 53-bit int range |
| Zig        | `i64`     | `f64`     |                                   |

### Functions

| Function       | Signature                | Description              |
| -------------- | ------------------------ | ------------------------ |
| `Abs(x)`       | `T -> T`                 | absolute value           |
| `Min(a, b)`    | `T, T -> T`              | smaller of two values    |
| `Max(a, b)`    | `T, T -> T`              | larger of two values     |
| `Sum(xs)`      | `list[T] -> T`           | sum of elements          |
| `Pow(a, b)`    | `int, int -> int`        | exponentiation           |
| `Round(x)`     | `float -> int`           | round to nearest integer |
| `DivMod(a, b)` | `int, int -> (int, int)` | quotient and remainder   |

`T` in `Abs`, `Min`, `Max`, `Sum` is `int` or `float`. No implicit coercion between numeric types — `Min(int, float)` is a type error. `bool` and `int` are distinct types with no implicit coercion in either direction.

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

`bytes` is an ordered sequence of `byte` values. Indexing (`buf[i]`) yields a `byte`. Slicing (`buf[a:b]`) yields a `bytes`.

### Functions

| Function | Signature      | Description |
| -------- | -------------- | ----------- |
| `Len(b)` | `bytes -> int` | byte count  |

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

| Operator | Operands     | Result | Prec | Assoc | Description                                                         |
| -------- | ------------ | ------ | ---- | ----- | ------------------------------------------------------------------- |
| `?:`     | `bool, T, T` | `T`    | 1    | right | ternary conditional                                                 |
| `\|\|`   | `bool, bool` | `bool` | 2    | left  | logical or, short-circuit                                           |
| `&&`     | `bool, bool` | `bool` | 3    | left  | logical and, short-circuit                                          |
| `==`     | `T, T`       | `bool` | 4    | none  | equality; structural for structs, IEEE 754 for float (NaN != NaN)   |
| `!=`     | `T, T`       | `bool` | 4    | none  | inequality                                                          |
| `<`      | `int, int`   | `bool` | 4    | none  | less than                                                           |
| `<=`     | `int, int`   | `bool` | 4    | none  | less or equal                                                       |
| `>`      | `int, int`   | `bool` | 4    | none  | greater than                                                        |
| `>=`     | `int, int`   | `bool` | 4    | none  | greater or equal                                                    |
| `\|`     | `int, int`   | `int`  | 5    | left  | bitwise or                                                          |
| `^`      | `int, int`   | `int`  | 6    | left  | bitwise xor                                                         |
| `&`      | `int, int`   | `int`  | 7    | left  | bitwise and                                                         |
| `<<`     | `int, int`   | `int`  | 8    | left  | left shift; right operand must be non-negative                      |
| `>>`     | `int, int`   | `int`  | 8    | left  | arithmetic right shift (sign-extending); right operand non-negative |
| `+`      | `int, int`   | `int`  | 9    | left  | addition                                                            |
| `-`      | `int, int`   | `int`  | 9    | left  | subtraction                                                         |
| `*`      | `int, int`   | `int`  | 10   | left  | multiplication                                                      |
| `/`      | `int, int`   | `int`  | 10   | left  | truncating division (toward zero); `-7 / 2 == -3`                   |
| `%`      | `int, int`   | `int`  | 10   | left  | truncating remainder; sign follows dividend; `-7 % 2 == -1`         |
| `-`      | `int`        | `int`  | 11   | right | negation (prefix)                                                   |
| `!`      | `bool`       | `bool` | 11   | right | logical not                                                         |
| `~`      | `int`        | `int`  | 11   | right | bitwise complement (two's complement)                               |

Arithmetic operators (`+`, `-`, `*`, `/`, `%`, unary `-`) also work on `float` operands, returning `float`. `Pow` also accepts float operands. For float `/`, result is IEEE 754 division (not truncating). Comparison operators (`<`, `<=`, `>`, `>=`) also work on `float` and `string` (lexicographic). All arithmetic and comparison operators require both operands to be the same type — no implicit coercion. `int + float` is a type error; lowering must insert explicit casts.

Comparisons are binary — `a < b < c` is not valid. Python's chained comparisons are desugared by the lowerer into `&&`-connected binary comparisons. A middleend raising pass can reconstruct chains for targets that support them (Python).

## Strings

Target languages disagree on what a string is. Some are byte-oriented (Go, Rust), some are UTF-16 (Java, JavaScript), and indexing `s[i]` means different things in each. The IR defines `string` as a sequence of runes, so indexing and length have consistent character-level semantics across all targets.

```
let name: string = "hello"
let ch: rune = 'λ'
let first: rune = name[0]
let n: int = Len("café")        -- 4, not 5
```

| Type     | Literal   | Description              |
| -------- | --------- | ------------------------ |
| `string` | `"hello"` | sequence of UTF-8 runes  |
| `rune`   | `'λ'`     | single Unicode character |

`string[i]` yields a `rune`. `Len(s)` returns the rune count.

### Functions

| Function               | Signature                             | Description                            |
| ---------------------- | ------------------------------------- | -------------------------------------- |
| `Len(s)`               | `string -> int`                       | rune count                             |
| `CharAt(s, i)`         | `string, int -> rune`                 | rune at position                       |
| `Substring(s, lo, hi)` | `string, int, int -> string`          | substring by rune position             |
| `Concat(a, b)`         | `string, string -> string`            | concatenation                          |
| `RuneFromInt(n)`       | `int -> rune`                         | code point to rune                     |
| `RuneToInt(c)`         | `rune -> int`                         | rune to code point                     |
| `IntToStr(n)`          | `int -> string`                       | integer to string                      |
| `ParseInt(s, base)`    | `string, int -> int`                  | parse integer in given base            |
| `Upper(s)`             | `string -> string`                    | uppercase                              |
| `Lower(s)`             | `string -> string`                    | lowercase                              |
| `Trim(s, chars)`       | `string, string -> string`            | trim characters from both ends         |
| `TrimStart(s, chars)`  | `string, string -> string`            | trim characters from start             |
| `TrimEnd(s, chars)`    | `string, string -> string`            | trim characters from end               |
| `RuneToStr(c)`         | `rune -> string`                      | single-rune string                     |
| `Split(s, sep)`        | `string, string -> list[string]`      | split by separator                     |
| `SplitN(s, sep, max)`  | `string, string, int -> list[string]` | split by separator; at most max pieces |
| `SplitWhitespace(s)`   | `string -> list[string]`              | split on whitespace runs, strip ends   |
| `Join(sep, parts)`     | `string, list[string] -> string`      | join with separator                    |
| `Find(s, sub)`         | `string, string -> int`               | index of substring, -1 if missing      |
| `Replace(s, old, new)` | `string, string, string -> string`    | replace all occurrences                |
| `StartsWith(s, pre)`   | `string, string -> bool`              | prefix test                            |
| `EndsWith(s, suf)`     | `string, string -> bool`              | suffix test                            |
| `IsDigit(s)`           | `string -> bool`                      | all characters are digits              |
| `IsAlpha(s)`           | `string -> bool`                      | all characters are letters             |
| `IsAlnum(s)`           | `string -> bool`                      | all characters are letters or digits   |
| `IsSpace(s)`           | `string -> bool`                      | all characters are whitespace          |
| `IsUpper(s)`           | `string -> bool`                      | all characters are uppercase           |
| `IsLower(s)`           | `string -> bool`                      | all characters are lowercase           |

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
    Print(Concat("hello, ", name))
}
```

A function has a name, typed parameters, a return type, and a body. `void` means no return value. The body is a block of statements enclosed in `{}`.

```
fn Factorial(n: int) -> int {
    if n <= 1 {
        return 1
    }
    return n * Factorial(n - 1)
}
```

## Variables

```
let x: int = 42
let name: string = "hello"
let done: bool = false
let items: list[int]
```

`let` declares a variable with an explicit type. The initializer is optional — omitting it gives the zero value for the type.

`let` bindings are mutable. Backends may emit `const`/`final` when they detect a binding is never reassigned.

## Assignment

```
x = 10
name = "world"
token.kind = "eof"
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

Tuple assignment destructures a tuple or multi-return value into multiple targets.

## Return

```
fn Add(a: int, b: int) -> int {
    return a + b
}

fn Log(msg: string) -> void {
    Print(msg)
    return
}
```

`return` exits the current function. A `void` function uses bare `return` or omits it.

## If

```
if x > 0 {
    Print("positive")
}

if x > 0 {
    Print("positive")
} else {
    Print("non-positive")
}

if x > 0 {
    Print("positive")
} else if x == 0 {
    Print("zero")
} else {
    Print("negative")
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

## For Range

```
for value in items {
    Print(IntToStr(value))
}

for i, ch in name {
    Print(Concat(IntToStr(i), Concat(": ", Concat(IntToStr(RuneToInt(ch)), "\n"))))
}

for _, v in pairs {
    total += v
}

for i in items {
    Print(IntToStr(i))
}
```

Iterates over a collection. The loop variable(s) before `in` bind the index and/or value for each element. Use `_` to discard a variable. A single variable before `in` binds the value; two variables bind index and value.

## Break and Continue

```
while true {
    let line: string = ReadLine()
    if line == "" {
        break
    }
    if StartsWith(line, "#") {
        continue
    }
    Process(line)
}
```

`break` exits the innermost loop. `continue` skips to the next iteration.

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

`list[T]` is an ordered, mutable, variable-length sequence of elements of type `T`. Indexing (`xs[i]`) yields a `T`. Slicing (`xs[a:b]`) yields a new `list[T]`.

### Functions

| Function           | Signature                 | Description                              |
| ------------------ | ------------------------- | ---------------------------------------- |
| `Len(xs)`          | `list[T] -> int`          | element count                            |
| `Append(xs, v)`    | `list[T], T -> void`      | append element to end                    |
| `Insert(xs, i, v)` | `list[T], int, T -> void` | insert element at index                  |
| `Pop(xs)`          | `list[T] -> T`            | remove and return last element           |
| `RemoveAt(xs, i)`  | `list[T], int -> void`    | remove element at index                  |
| `IndexOf(xs, v)`   | `list[T], T -> int`       | index of first occurrence, -1 if missing |
| `Contains(xs, v)`  | `list[T], T -> bool`      | membership test                          |
| `Repeat(xs, n)`    | `list[T], int -> list[T]` | repeat list n times                      |
| `Reversed(xs)`     | `list[T] -> list[T]`      | new list in reverse order                |
| `Sorted(xs)`       | `list[T] -> list[T]`      | new list in ascending order              |

`Sorted` requires `T` to be an ordered type (`int`, `float`, `string`).

## Maps

```
let ages: map[string, int] = {"alice": 30, "bob": 25}
let empty: map[string, int] = Map()
let age: int = ages["alice"]
ages["charlie"] = 35
```

`map[K, V]` is an unordered mutable mapping from keys of type `K` to values of type `V`. `K` must be a hashable type (primitives, strings, runes, tuples of hashable types).

Indexing (`m[k]`) yields a `V`. Assigning to an index (`m[k] = v`) inserts or updates.

### Functions

| Function             | Signature                   | Description                          |
| -------------------- | --------------------------- | ------------------------------------ |
| `Len(m)`             | `map[K, V] -> int`          | number of entries                    |
| `Map()`              | `-> map[K, V]`              | empty map (type from context)        |
| `Contains(m, k)`     | `map[K, V], K -> bool`      | key membership test                  |
| `Get(m, k, default)` | `map[K, V], K, V -> V`      | value for key, or default if missing |
| `Delete(m, k)`       | `map[K, V], K -> void`      | remove entry by key                  |
| `Keys(m)`            | `map[K, V] -> list[K]`      | list of keys                         |
| `Values(m)`          | `map[K, V] -> list[V]`      | list of values                       |
| `Items(m)`           | `map[K, V] -> list[(K, V)]` | list of key-value pairs              |

Iteration with `for k, v in m` yields key-value pairs.

## Sets

```
let seen: set[int] = Set(1, 2, 3)
let empty: set[string] = Set()
```

`set[T]` is an unordered mutable collection of unique values of type `T`. `T` must be a hashable type.

### Functions

| Function         | Signature           | Description               |
| ---------------- | ------------------- | ------------------------- |
| `Len(s)`         | `set[T] -> int`     | number of elements        |
| `Set(...)`       | `T... -> set[T]`    | construct set from values |
| `Add(s, v)`      | `set[T], T -> void` | add element               |
| `Remove(s, v)`   | `set[T], T -> void` | remove element            |
| `Contains(s, v)` | `set[T], T -> bool` | membership test           |

## Collection Equality

`==` and `!=` work on lists, maps, and sets with deep structural comparison. Lists compare element-wise in order. Maps compare by key-value pairs regardless of insertion order. Sets compare by membership.

`<`, `<=`, `>`, `>=` work on lists only, comparing lexicographically.

## Optional

```
let x: int? = nil
let y: int? = 42
```

`T?` is the optional type — a value of type `T` or `nil`. `nil` is a literal representing the absence of a value. Any type can be made optional: `int?`, `string?`, `list[int]?`, `Token?`. `T??` is not valid — optionals do not nest.

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

| Function    | Signature | Description                |
| ----------- | --------- | -------------------------- |
| `Unwrap(x)` | `T? -> T` | extract value; trap if nil |

`Unwrap` extracts the value when the program can guarantee non-nil but control flow doesn't prove it.

## Conversions

```
let f: float = IntToFloat(42)
let n: int = FloatToInt(3.14)
```

| Function        | Signature      | Description            |
| --------------- | -------------- | ---------------------- |
| `IntToFloat(n)` | `int -> float` | exact if representable |
| `FloatToInt(x)` | `float -> int` | truncate toward zero   |

No implicit coercion between any types. All conversions are explicit function calls. `IntToStr` and `ParseInt` are in the Strings section. `RuneToInt` and `RuneFromInt` handle rune↔int.

## Structs

```
struct Token {
    kind: TokenKind
    value: string
    offset: int
}

let t: Token = Token(TokenKind.Ident, "foo", 0)
let k: TokenKind = t.kind
t.offset = 10
```

A struct has a name and typed fields. Construction is positional — arguments follow the field declaration order. Fields are accessed with `.` and are mutable.

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

There are no imports — the entire program is one module — so every interface is automatically sealed. The IR can see all implementations and verify exhaustive matching.

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

```
match node {
    case lit: Literal {
        Print(IntToStr(lit.value))
    }
    case bin: BinOp {
        Eval(bin.left)
        Print(bin.op)
        Eval(bin.right)
    }
}

match kind {
    case TokenKind.Ident {
        ParseIdent()
    }
    case TokenKind.Number {
        ParseNumber()
    }
    case TokenKind.LParen {
        ParseGroup()
    }
}
```

`match` dispatches on the runtime type of an interface value or the value of an enum. For interfaces, each `case` names a variant type and binds a variable of that type. For enums, each `case` names a qualified enum value.

A match must be exhaustive — every variant or enum value must have a case.

## TODO

| Section      | Topic                 | Notes                                                                                                       |
| ------------ | --------------------- | ----------------------------------------------------------------------------------------------------------- |
| Declarations | Module structure      | constants, entrypoint                                                                                       |
| Strings      | Format strings        | interpolation                                                                                               |
| Strings      | Repetition            | `s * n` → `Repeat(s, n)`                                                                                    |
| Strings      | RFind                 | index of last substring occurrence                                                                          |
| Strings      | Count                 | count substring occurrences                                                                                 |
| Strings      | Encoding / decoding   | `bytes.decode()`, `str.encode()`                                                                            |
| Collections  | Merge                 | map merge (`Merge(m1, m2)`)                                                                                 |
| Collections  | Contains for strings  | `Contains(s, sub)` for substring test — add to string functions                                             |
| Math         | Numeric semantics     | division by zero, overflow, bitwise width, shift range, rounding, float edge cases, conversions, ParseInt   |
| Control flow | Try/catch/raise       |                                                                                                             |
| Control flow | Match/case            | match on non-interface/enum types? default case?                                                            |
| Control flow | Assert                |                                                                                                             |
| Control flow | Yield / generators    |                                                                                                             |
| Functions    | References            | callable values, bound methods                                                                              |
| Metadata     | Source metadata       | literal form (hex/octal/binary), large int flag, source positions — fully specified here                    |
| Metadata     | Middleend annotations | format for attaching analysis results to IR nodes; contents defined by middleend specs, not this doc        |
| I/O          | I/O                   | Print, ReadLine, ReadAll, ReadBytes, ReadBytesN, ReadBytesLine, WriteBytes, WriteStderr, Args, GetEnv, Exit |
| Appendix     | Grammar reference     | complete grammar for the IR textual syntax                                                                  |

## TOC

1. Introduction ✓
2. Primitives ✓
   - Numeric types (int, float, bool) + functions ✓
   - Bytes (byte, bytes, indexing, slicing, Len) ✓
3. Strings ✓
   - string, rune + functions ✓
   - Format strings
   - Repetition (`Repeat`)
   - RFind, Count
   - Encoding / decoding (bytes↔string)
4. Operators ✓
5. Collections ✓
   - Tuples ✓
   - Lists ✓ (merge pending)
   - Maps ✓ (merge pending)
   - Sets ✓
   - Membership (`Contains`) ✓
   - Equality and comparison ✓
6. Type System ✓
   - Optional and nil ✓
   - Conversions ✓
7. Declarations ✓
   - Structs ✓
   - Interfaces ✓
   - Enums ✓
   - Functions ✓ (references, defaults pending)
   - Module structure (constants, entrypoint pending)
8. Statements
   - Variables ✓ (zero values pending)
   - Assignment ✓
   - Return ✓
   - If ✓
   - While ✓
   - For ✓ (iterability, range, map iteration pending)
   - Break / continue ✓
   - Try / catch / raise
   - Match / case ✓
   - Assert
   - Yield / generators
9. Math Semantics
    - Division by zero (integer: trap; float: IEEE 754 ±Inf/NaN)
    - Integer overflow (wrap, trap, or bigint — mode TBD)
    - Bitwise complement and width (`~` depends on integer representation)
    - Shift out of range (behavior when shift amount ≥ bit width)
    - Round tie-breaking (half-to-even vs half-away-from-zero)
    - Float edge cases (NaN propagation in comparisons, negative zero)
    - Float-to-int conversion on overflow/NaN
    - Exponentiation corner cases (`Pow(0, 0)`, large exponents)
    - ParseInt failure mode (trap vs sentinel)
10. Built-in Functions
    - I/O (Print, ReadLine, ReadAll, ReadBytes, ReadBytesN, ReadBytesLine, WriteBytes, WriteStderr, Args, GetEnv, Exit)
11. Metadata
    - Source metadata: literal form, large int flag, source positions (fully specified here)
    - Middleend annotations: format for attaching analysis results; contents defined by middleend specs
12. Appendix: Grammar

## Grammar

Notation is EBNF: `|` alternation, `( )?` optional, `( )*` zero or more, `( )+` one or more. Terminals in `'quotes'`. The grammar targets recursive descent with at most two tokens of lookahead. The few spots requiring the second token are noted inline.

### Tokens

```
INT        = [0-9]+ | '0x' [0-9a-fA-F]+ | '0o' [0-7]+ | '0b' [01]+
FLOAT      = [0-9]+ '.' [0-9]+
STRING     = '"' ( escape | [^"\] )* '"'
RUNE       = "'" ( escape | [^'\] ) "'"
BYTES      = 'b"' ( escape | [^"\] )* '"'
IDENT      = [a-zA-Z_] [a-zA-Z0-9_]*
escape     = '\' ( 'n' | 'r' | 't' | '\' | '"' | "'" | '0'
           | 'x' hex hex | 'u' hex hex hex hex )
hex        = [0-9a-fA-F]
```

Keywords are reserved and cannot appear as `IDENT`:

```
bool    break     byte    bytes     case      continue
else    enum      false   float    fn        for
if      in        int     interface let       list
map     match     nil     return   rune      self
set     string    struct  true     void      while
```

Line comments start with `--` and run to end of line. No block comments.

Whitespace (spaces, tabs, newlines) separates tokens but is otherwise insignificant. There are no semicolons — statements are self-delimiting because every statement form begins with a keyword or an expression, and the parser knows when each form ends by its structure.

### Top Level

```
Program       = Decl*
Decl          = FnDecl | StructDecl | InterfaceDecl | EnumDecl

FnDecl        = 'fn' IDENT '(' ParamList ')' '->' Type Block
ParamList     = ( Param ( ',' Param )* )?
Param         = IDENT ':' Type
Block         = '{' Stmt* '}'

StructDecl    = 'struct' IDENT ( ':' IDENT )? '{' StructBody '}'
StructBody    = ( FieldDecl | FnDecl )*
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
           | ReturnStmt
           | 'break'
           | 'continue'
           | ExprStmt

LetStmt    = 'let' IDENT ':' Type ( '=' Expr )?
ReturnStmt = 'return' Expr?
IfStmt     = 'if' Expr Block ( 'else' ( IfStmt | Block ) )?
WhileStmt  = 'while' Expr Block
ForStmt    = 'for' Binding 'in' Expr Block
Binding    = IDENT ( ',' IDENT )?
MatchStmt  = 'match' Expr '{' Case+ '}'
Case       = 'case' Pattern Block
Pattern    = IDENT ':' IDENT
           | IDENT '.' IDENT
```

`ForStmt` needs two tokens of lookahead: after the first `IDENT`, peek for `,` (two bindings) versus `in` (one binding). `_` is a regular identifier; discard semantics are not a grammar concern.

`Pattern` in a `Case` also needs two tokens: after the first `IDENT`, peek for `:` (interface variant binding) versus `.` (enum value).

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
ArgList    = ( Expr ( ',' Expr )* )?
Primary    = INT | FLOAT | STRING | RUNE | BYTES
           | 'true' | 'false' | 'nil'
           | IDENT
           | '(' Expr ( ',' Expr )+ ')'
           | '(' Expr ')'
           | '[' ( Expr ( ',' Expr )* )? ']'
           | '{' Expr ':' Expr ( ',' Expr ':' Expr )* '}'
```

`Ternary` is right-associative: the "then" branch is a full `Expr`, the "else" branch right-recurses into `Ternary`.

`Compare` allows at most one comparison operator — `a < b < c` is a parse error. Chained comparisons are desugared during lowering into `&&`-connected binary comparisons.

`Suffix` with `'['` uses one token of lookahead inside the brackets: after the first `Expr`, peek for `':'` to distinguish indexing from slicing.

`'('` is disambiguated by what follows: parse the first `Expr`, then peek for `','` (tuple) versus `)` (parenthesized expression). `'['` as a `Primary` is a list literal; as a `Suffix` it's indexing/slicing — the parser knows which by context (a `Suffix` only follows a `Primary`). `'{'` is always a map literal (no empty map literal; use `Map()`). Sets have no literal syntax; use `Set(...)`.

### Types

```
Type       = BaseType ( '?' )?
BaseType   = 'int' | 'float' | 'bool'
           | 'byte' | 'bytes'
           | 'string' | 'rune'
           | 'void'
           | 'list' '[' Type ']'
           | 'map' '[' Type ',' Type ']'
           | 'set' '[' Type ']'
           | '(' Type ',' Type ( ',' Type )* ')'
           | IDENT
```

Every alternative in `BaseType` starts with a distinct token, so no lookahead is needed. `?` for optional is a single-token peek after the base type; types only appear in declaration contexts (after `:` or `->`) where `?` is unambiguous with the ternary operator.

Tuple types require at least two elements — `(T)` is not a type. `IDENT` covers user-defined names: structs, interfaces, enums.
