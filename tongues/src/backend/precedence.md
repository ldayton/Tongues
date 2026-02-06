# Backend Operator Precedence

## Integration Status

All backends use consistent naming:
- **Table**: `_PRECEDENCE`
- **Lookup**: `_prec(op)`
- **Decision**: `_needs_parens(child_op, parent_op, is_left)`
- **Method**: `_maybe_paren(expr, parent_op, is_left)`

| Backend       | Integrated | Notes                        |
| ------------- | ---------- | ---------------------------- |
| **c.py**      | Yes        | —                            |
| **go.py**     | Yes        | —                            |
| **dart.py**   | Yes        | —                            |
| **csharp.py** | Yes        | Inline `_needs_parens` calls |
| **java.py**   | Yes        | Inline `_needs_parens` calls |
| **rust.py**   | Yes        | —                            |
| **zig.py**    | Yes        | —                            |
| **swift.py**  | Yes        | —                            |
| **python.py** | Yes        | —                            |
| **ruby.py**   | Yes        | —                            |
| **lua.py**    | Yes        | —                            |
| **perl.py**   | Yes        | —                            |
| **php.py**    | Yes        | —                            |
| **jslike.py** | Yes        | —                            |

## Standard Pattern

```python
_PRECEDENCE: dict[str, int] = {
    "||": 1,
    "&&": 2,
    # ... language-specific order
}

def _prec(op: str) -> int:
    return _PRECEDENCE.get(op, DEFAULT)

def _needs_parens(child_op: str, parent_op: str, is_left: bool) -> bool:
    child_prec = _prec(child_op)
    parent_prec = _prec(parent_op)
    if child_prec < parent_prec:
        return True
    if child_prec == parent_prec and not is_left:
        return child_op in COMPARISON_OPS  # non-associative
    return False

# Method on emitter class
def _maybe_paren(self, expr: Expr, parent_op: str, is_left: bool) -> str:
    match expr:
        case BinaryOp(op=child_op):
            if _needs_parens(child_op, parent_op, is_left):
                return f"({self._expr(expr)})"
        case Ternary():
            return f"({self._expr(expr)})"
    return self._expr(expr)
```

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

| Language   | Special Ops                    | Notes                                            |
| ---------- | ------------------------------ | ------------------------------------------------ |
| **zig**    | `and`, `or`                    | Keywords instead of `&&`, `\|\|`                 |
| **lua**    | `~=`, `~`, `..`, `^`           | `~=` is !=, `~` is xor, `..` is concat, `^` exp  |
| **perl**   | `eq ne lt gt le ge`, `.`       | String comparison ops, `.` is concat             |
| **php**    | `or`, `and`, `.`, `===`, `!==` | 4 logical levels: `or` < `\|\|` < `and` < `&&`   |
| **jslike** | `===`, `!==`, `**`             | Strict equality, exponentiation                  |
| **dart**   | `~/`                           | Integer division                                 |
| **ruby**   | `**`                           | Exponentiation                                   |
| **python** | `**`                           | Exponentiation                                   |
