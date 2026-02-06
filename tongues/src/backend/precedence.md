# Backend Operator Precedence

## Integration Status

| Backend | Table/Function | Integrated | Issues |
|---------|----------------|------------|--------|
| **c.py** | `_C_PREC` + `_c_prec()` | No | — |
| **go.py** | `_GO_PREC` + `_go_prec()` | No | — |
| **dart.py** | `_DART_PREC` + `_dart_prec()` | No | — |
| **csharp.py** | `_CS_PREC` + `_cs_prec()` | No | Duplicate: `_OP_PRECEDENCE` already exists and is integrated |
| **java.py** | `_JAVA_PREC` + `_java_prec()` | No | Duplicate: `_OP_PRECEDENCE` already exists and is integrated |
| **rust.py** | `_RUST_PREC` + `_rust_prec()` | Yes | — |
| **zig.py** | `_ZIG_PREC` + `_zig_prec()` | Yes | — |
| **swift.py** | `_SWIFT_PREC` + `_swift_prec()` | Yes | Missing non-associative comparison handling |
| **python.py** | `_PRECEDENCE` | Yes | — |
| **ruby.py** | `_PRECEDENCE` | Yes | — |
| **lua.py** | `_PRECEDENCE` | Yes | — |
| **perl.py** | `_PRECEDENCE` | Yes | — |
| **php.py** | `_PRECEDENCE` | Yes | — |
| **jslike.py** | `_op_precedence()` | Yes | — |

## Implementation Patterns

| Backend | Implementation | Signature | Non-associative handling |
|---------|---------------|-----------|-------------------------|
| **rust** | `_wrap_prec()` method | `(expr, parent_op, is_right)` | Yes (comparisons) |
| **zig** | `_wrap_prec()` method | `(expr, parent_op, is_right)` | Yes (comparisons + bitwise/cmp mixing) |
| **swift** | `_wrap_prec()` method | `(expr, parent_op, is_right)` | No |
| **python** | `_needs_parens()` func | `(child_op, parent_op, is_left)` | Yes (chained comparisons) |
| **ruby** | `_needs_parens()` func | `(child_op, parent_op, is_left)` | Yes (comparisons on right) |
| **lua** | `_needs_parens()` func | `(child_op, parent_op, is_left)` | Yes (comparisons on right) |
| **perl** | `_needs_parens()` func | `(child_op, parent_op, is_left)` | Yes (includes string cmp ops) |
| **php** | `_needs_parens()` func | `(child_op, parent_op, is_left)` | Yes (comparisons on right) |
| **jslike** | `_expr_with_precedence()` | `(expr, parent_op, is_right)` | Special `**` handling |
| **csharp** | `_needs_parens()` func | `(child: BinaryOp, parent_op)` | No `is_left` param |
| **java** | `_needs_parens()` func | `(child: BinaryOp, parent_op)` | No `is_left` param |

## Precedence Families

### Rust-style (bitwise tighter than comparisons)

Used by: **rust**, **swift**, **python**

```
|| < && < == < | < ^ < & < << < + < *
```

Bitwise ops at distinct levels, all tighter than comparisons.

### Go-style (bitwise grouped with arithmetic)

Used by: **go**, **zig**

```
|| < && < == < (+ | ^) < (* & <<)
```

Bitwise OR/XOR with addition, bitwise AND with multiplication.

### C-style (bitwise looser than comparisons)

Used by: **c**, **dart**, **csharp**, **java**, **jslike** (JS/TS), **php**

```
|| < && < | < ^ < & < == < < < << < + < *
```

Classic C gotcha: `a & b == c` parses as `a & (b == c)`.

### Scripting (no bitwise)

Used by: **ruby**, **perl**

```
|| < && < == < + < *
```

No bitwise operators in precedence table.

## Special Operators by Language

| Language | Special Ops | Notes |
|----------|-------------|-------|
| **zig** | `and`, `or` | Keywords instead of `&&`, `||` |
| **lua** | `~=`, `~`, `..`, `^` | `~=` is !=, `~` is xor, `..` is concat, `^` is exp |
| **perl** | `eq ne lt gt le ge`, `.` | String comparison ops, `.` is concat |
| **php** | `or`, `and`, `.`, `===`, `!==` | 4 logical levels: `or` < `||` < `and` < `&&` |
| **jslike** | `===`, `!==`, `**` | Strict equality, exponentiation |
| **dart** | `~/` | Integer division |
| **ruby** | `**` | Exponentiation |
| **python** | `**` | Exponentiation |

## TODO

1. **Integrate c.py**: Add `_wrap_prec()` or `_needs_parens()` using `_C_PREC`
2. **Integrate go.py**: Add `_wrap_prec()` or `_needs_parens()` using `_GO_PREC`
3. **Integrate dart.py**: Add `_wrap_prec()` or `_needs_parens()` using `_DART_PREC`
4. **Fix csharp.py**: Remove duplicate `_CS_PREC`, add `is_left` param to `_needs_parens()`
5. **Fix java.py**: Remove duplicate `_JAVA_PREC`, add `is_left` param to `_needs_parens()`
6. **Fix swift.py**: Add non-associative comparison handling like Rust/Zig
