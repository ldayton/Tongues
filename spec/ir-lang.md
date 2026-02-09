# IR Specification

> **Note:** This document is an exploration of a future IR iteration, using language-like syntax to clarify thinking. It is not consistent with the rest of the spec.

This spec provides a textual syntax for the Tongues IR, to facilitate exposition. The "language" is LL(1) parseable.

Grammar notation: `|` alternatives, `*` repetition, `?` optional, `'x'` terminal.

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
| `Abs(x)`       | `int -> int`             | absolute value           |
| `Min(a, b)`    | `int, int -> int`        | smaller of two values    |
| `Max(a, b)`    | `int, int -> int`        | larger of two values     |
| `Pow(a, b)`    | `int, int -> int`        | exponentiation           |
| `Round(x)`     | `float -> int`           | round to nearest integer |
| `DivMod(a, b)` | `int, int -> (int, int)` | quotient and remainder   |

## Bytes

```
let tag: byte = 0xff
let buf: bytes = read_bytes()
let header: bytes = b"\x89PNG"
let first: byte = buf[0]
```

`byte` is an unsigned 8-bit integer. `bytes` is a byte sequence for binary I/O. The `b"..."` literal creates a `bytes` value.

### Functions

| Function         | Signature      | Description                   |
| ---------------- | -------------- | ----------------------------- |
| `Len(b)`         | `bytes -> int` | byte count                    |
| `ReadBytes()`    | `-> bytes`     | read all bytes from stdin     |
| `ReadBytesN(n)`  | `int -> bytes` | read up to n bytes from stdin |
| `WriteBytes(b)`  | `bytes -> int` | write bytes to stdout         |
| `WriteStderr(b)` | `bytes -> int` | write bytes to stderr         |

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

Arithmetic operators (`+`, `-`, `*`, `/`, `%`, `**`, unary `-`) also work on `float` operands, returning `float`. For float `/`, result is IEEE 754 division (not truncating). Comparison operators (`<`, `<=`, `>`, `>=`) also work on `float` and `string` (lexicographic).

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

| Function               | Signature                          | Description                          |
| ---------------------- | ---------------------------------- | ------------------------------------ |
| `Len(s)`               | `string -> int`                    | rune count                           |
| `CharAt(s, i)`         | `string, int -> rune`              | rune at position                     |
| `Substring(s, lo, hi)` | `string, int, int -> string`       | substring by rune position           |
| `Concat(a, b)`         | `string, string -> string`         | concatenation                        |
| `Chr(n)`               | `int -> rune`                      | code point to rune                   |
| `Ord(c)`               | `rune -> int`                      | rune to code point                   |
| `IntToStr(n)`          | `int -> string`                    | integer to string                    |
| `ParseInt(s, base)`    | `string, int -> int`               | parse integer in given base          |
| `Upper(s)`             | `string -> string`                 | uppercase                            |
| `Lower(s)`             | `string -> string`                 | lowercase                            |
| `Strip(s, chars)`      | `string, string -> string`         | trim characters from both ends       |
| `LStrip(s, chars)`     | `string, string -> string`         | trim characters from left            |
| `RStrip(s, chars)`     | `string, string -> string`         | trim characters from right           |
| `Split(s, sep)`        | `string, string -> list[string]`   | split by separator                   |
| `Join(sep, parts)`     | `string, list[string] -> string`   | join with separator                  |
| `Find(s, sub)`         | `string, string -> int`            | index of substring, -1 if missing    |
| `Replace(s, old, new)` | `string, string, string -> string` | replace all occurrences              |
| `StartsWith(s, pre)`   | `string, string -> bool`           | prefix test                          |
| `EndsWith(s, suf)`     | `string, string -> bool`           | suffix test                          |
| `IsDigit(s)`           | `string -> bool`                   | all characters are digits            |
| `IsAlpha(s)`           | `string -> bool`                   | all characters are letters           |
| `IsAlnum(s)`           | `string -> bool`                   | all characters are letters or digits |
| `IsSpace(s)`           | `string -> bool`                   | all characters are whitespace        |
| `IsUpper(s)`           | `string -> bool`                   | all characters are uppercase         |
| `IsLower(s)`           | `string -> bool`                   | all characters are lowercase         |

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
