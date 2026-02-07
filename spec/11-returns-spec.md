# Phase 11: Returns

**Module:** `middleend/returns.py`

Analyze return patterns: which statements contain returns, which always return, which functions need named returns for Go emission, which may return null.

## Analysis Functions

| Function          | Purpose                                  |
| ----------------- | ---------------------------------------- |
| `contains_return` | Does statement list contain any Return?  |
| `always_returns`  | Does statement list return on all paths? |

## Annotations

### Function Annotations

| Annotation            | Set When                                              | Used By          |
| --------------------- | ----------------------------------------------------- | ---------------- |
| `needs_named_returns` | TryCatch contains catch-body returns                  | Go               |
| `may_return_null`     | Function body contains `return None` or null-yielding | Dart, Go, others |

### TryCatch Annotations

| Annotation        | Set When                     | Used By        |
| ----------------- | ---------------------------- | -------------- |
| `body_has_return` | Try body contains any Return | Lua, Perl, Zig |

Languages using `pcall`/`eval` blocks (Lua, Perl) or lacking exceptions (Zig) need to know if the try body contains returns to emit workarounds (flag variables, labeled loops).

## Nullable Return Detection

A function `may_return_null` if any of:
- Explicit `return None`
- Return of variable known to be None (via assignment tracking)
- Return of expression typed `T | None` without narrowing

```python
def find(xs: list[int], target: int) -> int | None:
    for x in xs:
        if x == target:
            return x
    return None  # sets may_return_null = True
```

Backends like Dart use this to emit `dynamic` return type when declared type doesn't include null but function returns null anyway.

## Postconditions

- `Function.needs_named_returns` set for Go try-catch patterns
- `Function.may_return_null` set when function may return null
- `TryCatch.body_has_return` set when try body contains returns
