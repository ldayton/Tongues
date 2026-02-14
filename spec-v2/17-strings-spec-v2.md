# Strings Analysis (Tongues Middleend)

This document specifies the **strings** middleend pass over Taytsh IR. The pass is **intra-procedural**: it analyzes each function body independently and writes results into Taytsh node `annotations`.

The pass classifies string-typed bindings by content (ASCII, BMP, or unknown) and by usage pattern (indexed, iterated, length-counted), and detects string-builder accumulation in loops. This lets backends skip expensive rune-decoding machinery when the data or access pattern doesn't require it.

This pass is **conditional** — it only runs when the target set includes a language whose native string encoding disagrees with Taytsh's rune-sequence model. For native-rune targets (Python, Ruby, Perl) the pass is a no-op.

## Inputs

The pass assumes it is given a **valid Taytsh Module** with:

- static types available for all bindings and expressions
- scope annotations available: `scope.is_reassigned` (to determine single- vs multi-assignment bindings), `scope.is_const`
- liveness annotations available: `liveness.initial_value_unused` (to skip dead stores in content analysis)

## Outputs

The pass writes annotations under the `strings.` namespace:

- binding-level facts: content classification, usage patterns
- loop-level facts: builder pattern detection

All annotations are stored in each node's `annotations` map (Taytsh supports `bool`, `int`, `string`, `(int, int)` values).

## Background

Taytsh defines `string` as a sequence of runes — `s[i]` yields the i-th character, `Len(s)` counts characters. Target languages represent strings in three encoding families:

| Encoding          | Targets                                       | Native `s[i]`                 | Native `Len(s)`                     |
| ----------------- | --------------------------------------------- | ----------------------------- | ----------------------------------- |
| Native rune       | Python, Ruby, Perl                            | character (correct)           | character count (correct)           |
| UTF-8 bytes       | Go, Rust, C, Zig, Lua, PHP                    | byte (wrong)                  | byte count (wrong)                  |
| UTF-16 code units | Java, C#, JavaScript, TypeScript, Dart, Swift | code unit (wrong for non-BMP) | code unit count (wrong for non-BMP) |

Swift uses UTF-8 storage internally (since Swift 5) and lacks native integer subscripting. It is grouped with UTF-16 targets because BMP optimizations (no surrogate pairs) apply when the backend uses UTF-16 views or `Array<Character>` conversion.

Without analysis, backends defensively emit rune-safe operations everywhere — Go converts to `[]rune`, Rust uses `.chars().nth(i)`, Java uses `codePointAt()` with `offsetByCodePoints()`, C handrolls UTF-8 decoding.

Two orthogonal facts eliminate most of this:

1. **Content** — when a string is provably ASCII (U+0000–U+007F), all three encoding families agree: byte index = code unit index = character index, and all length measures agree. When provably BMP (U+0000–U+FFFF), UTF-16 code unit indexing is correct (no surrogate pairs).

2. **Usage** — many strings are never indexed or sliced. They're compared, concatenated, passed to `Find`/`Contains`/`Split`, or printed. Only `s[i]`, `s[a:b]`, `Len(s)`, and `for ch in s` require encoding-aware operations. A string that's only concatenated and compared needs no rune machinery in any backend.

## Content Classification

The pass classifies each string-typed binding into one of three levels forming a lattice:

```
ascii ⊂ bmp ⊂ unknown
```

- **`"ascii"`** — every character is in U+0000–U+007F. All encoding families agree on indexing, length, and iteration.
- **`"bmp"`** — every character is in U+0000–U+FFFF. No surrogate pairs; UTF-16 code unit indexing is correct. UTF-8 targets still need multi-byte decoding.
- **`"unknown"`** — the string may contain any Unicode character.

### ASCII sources

These expressions produce provably-ASCII strings:

| Source                                                                                  | Rationale                                          |
| --------------------------------------------------------------------------------------- | -------------------------------------------------- |
| String literal with all characters in U+0000–U+007F                                     | Content is statically known                        |
| `ToString(x)` where x is `int`, `float`, `bool`, or `byte`                              | Digits, `-`, `.`, `e`, `true`, `false`             |
| `FormatInt(n, base)`                                                                    | Digits and ASCII letters                           |
| `Concat(a, b)` where both are ASCII                                                     | Concatenation preserves content class              |
| `Format(template, args...)` where template and all args are ASCII                       | Interpolation preserves content class              |
| `Lower(s)` / `Upper(s)` where s is ASCII                                                | ASCII case mapping stays in ASCII                  |
| `Trim(s, chars)` / `TrimStart` / `TrimEnd` where s is ASCII                             | Substring of ASCII is ASCII                        |
| `Replace(s, old, new)` where all three are ASCII                                        | Replacement within ASCII stays ASCII               |
| `Repeat(s, n)` where s is ASCII                                                         | Repetition preserves content class                 |
| `Join(sep, parts)` where sep is ASCII and parts elements are traceable to ASCII sources | Join preserves ASCII when all components are ASCII |
| `s[a:b]` (slice) where `s` is ASCII                                                     | Substring of ASCII is ASCII                        |
| `Reverse(s)` where `s` is ASCII                                                         | Reversal preserves content class                   |

### BMP sources

Everything classified as ASCII is also BMP. Additionally:

| Source                                                     | Rationale                                                       |
| ---------------------------------------------------------- | --------------------------------------------------------------- |
| String literal with all characters in U+0000–U+FFFF        | Content is statically known                                     |
| Operations on BMP inputs (same propagation rules as ASCII) | BMP is closed under concatenation, case mapping, trimming, etc. |

### Unknown sources

Any source not matching the above:

| Source                                           | Rationale                                  |
| ------------------------------------------------ | ------------------------------------------ |
| `ReadAll()`, `ReadLine()`, `ReadFile()`          | Arbitrary external input                   |
| `Decode(b)`                                      | Arbitrary byte content                     |
| Parameters                                       | Caller could pass any string               |
| User-defined function return values              | Intra-procedural; callee body not analyzed |
| Collection element access (`xs[i]`, `Get(m, k)`) | Element content not tracked                |
| Field access (`obj.field`)                       | Field content not tracked                  |

### Join at control-flow merges

When a binding has multiple sources (if/else branches, reassignment), the classification is the join (least precise) across all sources:

|             | ascii   | bmp     | unknown |
| ----------- | ------- | ------- | ------- |
| **ascii**   | ascii   | bmp     | unknown |
| **bmp**     | bmp     | bmp     | unknown |
| **unknown** | unknown | unknown | unknown |

Dead stores (`liveness.initial_value_unused=true`) do not contribute — the overwritten initializer is never observed.

## Produced Annotations

This section is normative: it defines the annotation keys, value types, attachment points, and when keys MUST be present.

Attachment points (by syntax):

| Syntax                                           | Annotation attachment point               |
| ------------------------------------------------ | ----------------------------------------- |
| `fn F(p: string, ...) { ... }`                   | parameter binding `p` (when string-typed) |
| `let x: string = expr`                           | the `let` binding `x` (when string-typed) |
| `for x in xs { ... }` / `for i, x in xs { ... }` | loop binder(s) that are string-typed      |
| `case v: T { ... }` / `default v { ... }`        | case binding `v` (when string-typed)      |
| `for ... in ... { }` / `while ... { }`           | loop node                                 |

"String-typed" means the binding's declared type is `string`, `string?`, or any union containing `string`.

### Binding annotations

| Key                  | Type     | Applies to                       | Meaning                                                                 |
| -------------------- | -------- | -------------------------------- | ----------------------------------------------------------------------- |
| `strings.content`    | `string` | string-typed binding declaration | `"ascii"`, `"bmp"`, or `"unknown"`                                      |
| `strings.indexed`    | `bool`   | string-typed binding declaration | `true` if the binding appears as base of `s[i]` or `s[a:b]`             |
| `strings.iterated`   | `bool`   | string-typed binding declaration | `true` if the binding is iterated via `for ch in s` or `for i, ch in s` |
| `strings.len_called` | `bool`   | string-typed binding declaration | `true` if `Len(s)` is called with this binding as the argument          |

**`strings.content`** classifies what characters can appear in the string, determined by tracing value origins through the Content Classification rules above.

Backends combine content with usage to choose emission strategy:

- `"ascii"` + any usage → native byte/code-unit operations are correct. No rune machinery needed in any backend.
- `"bmp"` + indexed → UTF-16 backends can use native code-unit indexing safely. UTF-8 backends still need multi-byte decoding.
- `"unknown"` + indexed → all non-native-rune backends need rune conversion or rune-aware indexing.

**`strings.indexed`** identifies bindings that appear in random-access positions. When `false`, backends can skip rune-indexing entirely — Go needs no `[]rune` conversion, Rust needs no `.chars().nth()`, Java needs no `codePointAt()`.

**`strings.iterated`** identifies bindings iterated character-by-character. When `strings.iterated=true` and `strings.indexed=false`, backends can use sequential decoding — Go: `for _, r := range s`, Rust: `for c in s.chars()`, C: sequential UTF-8 walk — which is cheaper than random-access rune conversion.

**`strings.len_called`** identifies bindings whose character count is needed. When `false`, backends skip rune-counting calls — Go: no `utf8.RuneCountInString`, Rust: no `.chars().count()`, PHP: no `mb_strlen`.

All four MUST be present on every string-typed binding declaration.

### Loop annotations

| Key               | Type     | Applies to           | Meaning                                             |
| ----------------- | -------- | -------------------- | --------------------------------------------------- |
| `strings.builder` | `string` | `for`, `while` nodes | comma-separated accumulator variable names, or `""` |

**`strings.builder`** identifies loops that build strings via repeated concatenation. When non-empty, backends emit efficient builder mechanisms instead of quadratic string concatenation:

| Target     | Mechanism                     |
| ---------- | ----------------------------- |
| Go         | `strings.Builder`             |
| Java / C#  | `StringBuilder`               |
| Rust       | `String` with `push_str`      |
| C          | growable buffer with doubling |
| Zig        | `ArrayList(u8)` or equivalent |
| JavaScript | array `push` + `join`         |

`strings.builder` MUST be present on every `for` and `while` node (even when `""`).

## Builder Pattern

A loop has the builder pattern for a variable `ACC` when all of the following hold:

1. `ACC` is a string-typed binding declared before the loop.
2. The loop body contains at least one assignment of the form `ACC = Concat(ACC, EXPR)`.
3. Every assignment to `ACC` inside the loop body has the form `ACC = Concat(ACC, EXPR)` — the accumulator is always the first argument.
4. `ACC` is not read inside the loop body except as the first argument to `Concat` in those accumulating assignments.

Assignments inside control structures within the loop body (if, match, try/catch) are included — the pattern holds as long as all paths that assign `ACC` use the accumulating form.

Multiple accumulators in the same loop are independently detected and listed comma-separated.

Multiple appends per iteration are valid:

```taytsh
for x in items {
    s = Concat(s, prefix)
    s = Concat(s, ToString(x))
}
-- strings.builder="s" on the for node
```

Conditional appends are valid:

```taytsh
for x in items {
    if x > 0 {
        s = Concat(s, ToString(x))
    }
}
-- strings.builder="s" on the for node
```

Non-Concat assignments break the pattern:

```taytsh
for x in items {
    s = Concat(s, ToString(x))
    s = Upper(s)
}
-- strings.builder="" (Upper breaks the accumulation pattern)
```

## Algorithm (per function)

For each function (including methods):

1. **Identify string-typed bindings** — collect every binding whose declared type is or contains `string`.

2. **Classify content** — for each string-typed binding:
   - If `scope.is_reassigned=false`, analyze the single source expression against the content classification rules.
   - If `scope.is_reassigned=true`, analyze every assignment source and join the results.
   - Skip initializers flagged by `liveness.initial_value_unused=true`.
   - For compound expressions (`Concat`, `Format`, etc.), recursively determine operand content and apply the propagation rules.
   - For self-referential assignments (`s = Concat(s, expr)` in a loop), compute content as a fixed-point: initialize the binding's content from its first non-dead assignment, then re-evaluate all assignments treating the binding's current content as its own operand. Repeat until stable. The lattice is three-valued and monotonically widening, so this terminates in at most two iterations.

3. **Scan usage** — walk the function body once:
   - Index expression `s[i]` or slice `s[a:b]`: set `strings.indexed=true` on the base binding.
   - `for` loop with the binding as iterable: set `strings.iterated=true`.
   - `Len(s)` call: set `strings.len_called=true` on the argument binding.

4. **Detect builder patterns** — for each `for` and `while` node:
   - Identify candidate accumulators: string-typed bindings declared before the loop and assigned inside it.
   - For each candidate, verify the builder pattern conditions.
   - Write `strings.builder` with the comma-separated names of valid accumulators, or `""`.

## Examples

```taytsh
fn FormatList(xs: list[int]) -> string {
    let result: string = "["
    for i, x in xs {
        if i > 0 {
            result = Concat(result, ", ")
        }
        result = Concat(result, ToString(x))
    }
    result = Concat(result, "]")
    return result
}
-- let result:  strings.content="ascii", strings.indexed=false,
--              strings.iterated=false, strings.len_called=false
-- for node:    strings.builder="result"
```

```taytsh
fn CharAt(s: string, i: int) -> rune {
    return s[i]
}
-- parameter s: strings.content="unknown", strings.indexed=true,
--              strings.iterated=false, strings.len_called=false
```

```taytsh
fn CountAlpha(s: string) -> int {
    let n: int = 0
    for ch in s {
        if IsAlpha(ch) {
            n += 1
        }
    }
    return n
}
-- parameter s: strings.content="unknown", strings.indexed=false,
--              strings.iterated=true, strings.len_called=false
-- for node:    strings.builder=""
```

```taytsh
fn HexHeader(n: int) -> string {
    let label: string = FormatInt(n, 16)
    let padded: string = Concat("0x", label)
    let width: int = Len(padded)
    return padded
}
-- let label:  strings.content="ascii", strings.indexed=false,
--             strings.iterated=false, strings.len_called=false
-- let padded: strings.content="ascii", strings.indexed=false,
--             strings.iterated=false, strings.len_called=true
```

```taytsh
fn Truncate(s: string, max: int) -> string {
    if Len(s) <= max {
        return s
    }
    return s[0:max]
}
-- parameter s: strings.content="unknown", strings.indexed=true,
--              strings.iterated=false, strings.len_called=true
```

```taytsh
fn Label(code: int, name: string) -> string {
    let label: string
    if code > 0 {
        label = FormatInt(code, 10)
    } else {
        label = name
    }
    return label
}
-- parameter name: strings.content="unknown"
-- let label:      strings.content="unknown"
--     (FormatInt → ascii, name → unknown; join(ascii, unknown) = unknown)
```

```taytsh
fn Greet() -> string {
    let greeting: string = "こんにちは"
    let name: string = "Alice"
    return Concat(greeting, Concat(", ", name))
}
-- let greeting: strings.content="bmp",   strings.indexed=false,
--               strings.iterated=false, strings.len_called=false
-- let name:     strings.content="ascii", strings.indexed=false,
--               strings.iterated=false, strings.len_called=false
```

```taytsh
fn ReadLines() -> string {
    let result: string = ""
    let line: string? = ReadLine()
    while line != nil {
        result = Concat(result, Concat(line, "\n"))
        line = ReadLine()
    }
    return result
}
-- let result:  strings.content="unknown", strings.indexed=false,
--              strings.iterated=false, strings.len_called=false
-- while node:  strings.builder="result"
```

## Relationship to Other Passes

The strings pass subsumes the string-indexing detection currently performed by the hoisting pass (`hoisting.rune_vars`). When the strings pass is active, `hoisting.rune_vars` can be derived from `strings.indexed` — a binding with `strings.indexed=true` and `strings.content!="ascii"` needs rune conversion.

The strings pass refines the hoisting decision: `hoisting.rune_vars` is function-scoped (convert at function entry), while `strings.indexed` on specific bindings allows backends to place rune conversion closer to the point of use.

## Postconditions

- Every string-typed binding declaration has `strings.content: string` (`"ascii"`, `"bmp"`, or `"unknown"`).
- Every string-typed binding declaration has `strings.indexed: bool`, `strings.iterated: bool`, and `strings.len_called: bool`.
- Every `for` and `while` node has `strings.builder: string`.
- Content classifications are sound: `"ascii"` guarantees all runtime values through the binding consist only of U+0000–U+007F characters. `"bmp"` guarantees U+0000–U+FFFF.
- Usage annotations are complete: if a binding appears in an indexing, slicing, iteration, or `Len` position anywhere in the function, the corresponding flag is `true`.
