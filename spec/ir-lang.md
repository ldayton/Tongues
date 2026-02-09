# IR Specification

> **Note:** This document is an exploration of a future IR iteration, using language-like syntax to clarify thinking. It is not consistent with the rest of the spec.

## Introduction

The current IR is a large set of specialized nodes that carry Python-flavored semantics: `PythonAnd`, `PythonOr`, `PythonMod`, `FloorDiv`, `SentinelToOptional`, `CharAt`, `StringNonEmpty`, and dozens more. Lowering eagerly commits to representations — `range(a, b)` becomes `ForClassic`, string indexing becomes `CharAt`/`CharLen`/`Substring` — and every backend must interpret these nodes individually. The middleend is limited to read-only annotation passes.

This IR replaces that with a small, target-neutral language. Python's quirks (floor division, value-returning `and`/`or`, type-dependent truthiness) are resolved during lowering into conventional semantics (truncating division, boolean `&&`/`||`), but representation choices are deliberately deferred. The IR has one loop-over-collection form (`for...in`), not two; strings are rune-indexed at the IR level, not pre-lowered into character operations. The middleend analyzes how values are actually used — whether a `for...in` over a range should become a classic for loop, whether a string needs rune-level access or can stay in the target's native representation — and makes those decisions as shared passes rather than per-backend logic. Backends receive already-decided representations and translate to syntax.

This spec provides a textual syntax for the IR to facilitate exposition. The "language" is LL(1) parseable.

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
let big: int = 2 ** 10
let inRange: bool = 0 <= x <= 255
let either: bool = a || b
let shifted: int = flags << 2
let abs: int = x > 0 ? x : -x
```

| Operator | Operands     | Result | Prec | Assoc | Description                                                         |
| -------- | ------------ | ------ | ---- | ----- | ------------------------------------------------------------------- |
| `?:`     | `bool, T, T` | `T`    | 1    | right | ternary conditional                                                 |
| `\|\|`   | `bool, bool` | `bool` | 2    | left  | logical or, short-circuit                                           |
| `&&`     | `bool, bool` | `bool` | 3    | left  | logical and, short-circuit                                          |
| `==`     | `T, T`       | `bool` | 4    | chain | equality; structural for structs, IEEE 754 for float (NaN != NaN)   |
| `!=`     | `T, T`       | `bool` | 4    | chain | inequality                                                          |
| `<`      | `int, int`   | `bool` | 4    | chain | less than                                                           |
| `<=`     | `int, int`   | `bool` | 4    | chain | less or equal                                                       |
| `>`      | `int, int`   | `bool` | 4    | chain | greater than                                                        |
| `>=`     | `int, int`   | `bool` | 4    | chain | greater or equal                                                    |
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
| `**`     | `int, int`   | `int`  | 11   | right | exponentiation; right operand must be non-negative                  |
| `-`      | `int`        | `int`  | 12   | right | negation (prefix)                                                   |
| `!`      | `bool`       | `bool` | 12   | right | logical not                                                         |
| `~`      | `int`        | `int`  | 12   | right | bitwise complement (two's complement)                               |

Arithmetic operators (`+`, `-`, `*`, `/`, `%`, `**`, unary `-`) also work on `float` operands, returning `float`. For float `/`, result is IEEE 754 division (not truncating). Comparison operators (`<`, `<=`, `>`, `>=`) also work on `float` and `string` (lexicographic). All arithmetic and comparison operators require both operands to be the same type — no implicit coercion. `int + float` is a type error; lowering must insert explicit casts.

Chained comparisons (`a < b < c`) evaluate each operand once. They desugar to `a < b && b < c` but `b` is only computed once.

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

| Function               | Signature                             | Description                              |
| ---------------------- | ------------------------------------- | ---------------------------------------- |
| `Len(s)`               | `string -> int`                       | rune count                               |
| `CharAt(s, i)`         | `string, int -> rune`                 | rune at position                         |
| `Substring(s, lo, hi)` | `string, int, int -> string`          | substring by rune position               |
| `Concat(a, b)`         | `string, string -> string`            | concatenation                            |
| `Chr(n)`               | `int -> rune`                         | code point to rune                       |
| `Ord(c)`               | `rune -> int`                         | rune to code point                       |
| `IntToStr(n)`          | `int -> string`                       | integer to string                        |
| `ParseInt(s, base)`    | `string, int -> int`                  | parse integer in given base              |
| `Upper(s)`             | `string -> string`                    | uppercase                                |
| `Lower(s)`             | `string -> string`                    | lowercase                                |
| `Strip(s, chars)`      | `string, string -> string`            | trim characters from both ends           |
| `LStrip(s, chars)`     | `string, string -> string`            | trim characters from left                |
| `RStrip(s, chars)`     | `string, string -> string`            | trim characters from right               |
| `RuneToStr(c)`         | `rune -> string`                      | single-rune string                       |
| `Split(s, sep, max)`   | `string, string, int -> list[string]` | split by separator; max=-1 for unlimited |
| `SplitWhitespace(s)`   | `string -> list[string]`              | split on whitespace runs, strip ends     |
| `Join(sep, parts)`     | `string, list[string] -> string`      | join with separator                      |
| `Find(s, sub)`         | `string, string -> int`               | index of substring, -1 if missing        |
| `Replace(s, old, new)` | `string, string, string -> string`    | replace all occurrences                  |
| `StartsWith(s, pre)`   | `string, string -> bool`              | prefix test                              |
| `EndsWith(s, suf)`     | `string, string -> bool`              | suffix test                              |
| `IsDigit(s)`           | `string -> bool`                      | all characters are digits                |
| `IsAlpha(s)`           | `string -> bool`                      | all characters are letters               |
| `IsAlnum(s)`           | `string -> bool`                      | all characters are letters or digits     |
| `IsSpace(s)`           | `string -> bool`                      | all characters are whitespace            |
| `IsUpper(s)`           | `string -> bool`                      | all characters are uppercase             |
| `IsLower(s)`           | `string -> bool`                      | all characters are lowercase             |

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
    Print(Concat(IntToStr(i), Concat(": ", Concat(IntToStr(Ord(ch)), "\n"))))
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

## TODO

| Section      | Topic                 | Notes                                                                                                       |
| ------------ | --------------------- | ----------------------------------------------------------------------------------------------------------- |
| Type system  | Grammar               | optional `T?`, tuples, callable, union                                                                      |
| Type system  | Optional and nil      | `T?`, nil checks, unwrap                                                                                    |
| Type system  | Conversions and casts | int↔float, int↔string, ord/chr already covered                                                              |
| Declarations | Structs               | fields, methods, constructors                                                                               |
| Declarations | Interfaces            | hierarchy root, type switch, casts                                                                          |
| Declarations | Enums                 |                                                                                                             |
| Declarations | Module structure      | imports, constants, entrypoint                                                                              |
| Strings      | Format strings        | interpolation                                                                                               |
| Strings      | Repetition            | `s * n` → `Repeat(s, n)`                                                                                    |
| Strings      | RFind                 | index of last substring occurrence                                                                          |
| Strings      | Count                 | count substring occurrences                                                                                 |
| Strings      | Encoding / decoding   | `bytes.decode()`, `str.encode()`                                                                            |
| Collections  | Tuples                | literals, indexing, unpacking                                                                               |
| Collections  | Lists                 | literals, append, pop, insert, index, slice, comprehensions                                                 |
| Collections  | Maps                  | literals, get, set, delete, keys/values/items, comprehensions, merge (`d1 \| d2`)                           |
| Collections  | Sets                  | literals, add, remove, contains, comprehensions, frozenset                                                  |
| Collections  | Membership testing    | `in` / `not in` for lists, maps, sets, strings, tuples                                                      |
| Collections  | Equality / comparison | deep `==` on list/map/set; lexicographic `<` on list                                                        |
| Collections  | Repetition            | `xs * n`                                                                                                    |
| Collections  | Sorted / reversed     | builtins producing new or reversed collections                                                              |
| Control flow | Try/catch/raise       |                                                                                                             |
| Control flow | Match/case            |                                                                                                             |
| Control flow | Assert                |                                                                                                             |
| Control flow | Yield / generators    |                                                                                                             |
| Functions    | References            | callable values, bound methods                                                                              |
| Metadata     | Source metadata       | literal form (hex/octal/binary), large int flag, source positions — fully specified here                     |
| Metadata     | Middleend annotations | format for attaching analysis results to IR nodes; contents defined by middleend specs, not this doc         |
| I/O          | I/O                   | Print, ReadLine, ReadAll, ReadBytes, ReadBytesN, ReadBytesLine, WriteBytes, WriteStderr, Args, GetEnv, Exit |
| Appendix     | Grammar reference     | complete LL(1) grammar for the IR textual syntax                                                            |

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
5. Collections
   - Tuples (literals, indexing, unpacking)
   - Lists (literals, append, pop, insert, index, slice, comprehensions, repetition)
   - Maps (literals, get, set, delete, keys/values/items, comprehensions, merge)
   - Sets (literals, add, remove, contains, comprehensions, frozenset)
   - Membership (`in` / `not in`)
   - Equality and comparison
   - Sorted / reversed
6. Type System
   - Type grammar (`T?`, union, callable, tuple types)
   - Optional and nil (`T?`, nil checks, unwrap)
   - Conversions and casts (int↔float, int↔string)
7. Declarations
   - Structs (fields, methods, constructors)
   - Interfaces (hierarchy root, type switch, casts)
   - Enums
   - Functions ✓ (references, defaults pending)
   - Module structure (imports, constants, entrypoint)
8. Statements
   - Variables ✓ (zero values pending)
   - Assignment ✓
   - Return ✓
   - If ✓
   - While ✓
   - For ✓ (iterability, range, map iteration pending)
   - Break / continue ✓
   - Try / catch / raise
   - Match / case
   - Assert
   - Yield / generators
9. Built-in Functions
    - I/O (Print, ReadLine, ReadAll, ReadBytes, ReadBytesN, ReadBytesLine, WriteBytes, WriteStderr, Args, GetEnv, Exit)
10. Metadata
    - Source metadata: literal form, large int flag, source positions (fully specified here)
    - Middleend annotations: format for attaching analysis results; contents defined by middleend specs
11. Appendix: LL(1) Grammar
