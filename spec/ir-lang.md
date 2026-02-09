# IR Specification

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
| `Split(s, sep)`        | `string, string -> []string`       | split by separator                   |
| `Join(sep, parts)`     | `string, []string -> string`       | join with separator                  |
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

## Basic Expressions

```
x                               -- variable
token.kind                      -- field access
node.left.value                 -- chained field access
items[0]                        -- index
table["key"]                    -- map lookup
items[1:5]                      -- slice
items[:n]                       -- slice from start
items[2:]                       -- slice to end
-x                              -- negation
!done                           -- logical not
~mask                           -- bitwise complement
a + b                           -- arithmetic
n % 2                           -- modulo
2 ** 10                         -- exponentiation
x == y                          -- equality
a < b < c                       -- chained comparison
a && b                          -- logical and
x || y                          -- logical or
flags & 0xff                    -- bitwise and
n << 2                          -- shift
x > 0 ? x : -x                 -- ternary
n as float                      -- cast
node as! Token                  -- checked downcast
node is Token                   -- type test
x is nil                        -- nil check
x is !nil                       -- negated nil check
```

A bare identifier is a variable reference. Dot chains access struct fields left to right.

`**` is right-associative. All other binary operators are left-associative. Chained comparisons (`a < b < c`) evaluate each operand once.

### Precedence

| Level | Operators         | Assoc |
| ----- | ----------------- | ----- |
| 1     | `?:`              | right |
| 2     | `\|\|`            | left  |
| 3     | `&&`              | left  |
| 4     | `== != < <= > >=` | chain |
| 5     | `\|`              | left  |
| 6     | `^`               | left  |
| 7     | `&`               | left  |
| 8     | `<< >>`           | left  |
| 9     | `+ -`             | left  |
| 10    | `* / %`           | left  |
| 11    | `**`              | right |
| 12    | `- ! ~` (prefix)  | right |
| 13    | `. [] () as is`   | left  |

### Indexing vs slicing

After `[`, the parser reads an expression then checks for `:`. No colon means index, colon means slice. All slice bounds are optional.

```
items[i]                        -- Index
items[lo:hi]                    -- SliceExpr
items[:hi]                      -- SliceExpr, low omitted
items[lo:]                      -- SliceExpr, high omitted
items[::step]                   -- SliceExpr, step only
```

### Casts and type operations

`as`, `as!`, and `is` bind at postfix level (tighter than any binary operator).

```
n as float                      -- Cast: convert int to float
node as! Token                  -- TypeAssert: checked downcast, panics on mismatch
node is Token                   -- IsType: runtime type test, returns bool
x is nil                        -- IsNil: nil check
x is !nil                       -- IsNil(negated): not-nil check
```
