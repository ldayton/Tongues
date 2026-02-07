# Phase 11: Returns

**Module:** `middleend/returns.py`

Analyze return patterns: which statements contain returns, which always return, which functions need named returns for Go emission.

| Function          | Purpose                                  |
| ----------------- | ---------------------------------------- |
| `contains_return` | Does statement list contain any Return?  |
| `always_returns`  | Does statement list return on all paths? |

## Postconditions

`Function.needs_named_returns` set for functions with TryCatch containing catch-body returns.
