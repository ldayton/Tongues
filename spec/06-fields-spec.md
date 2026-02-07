# Phase 6: Fields

**Module:** `frontend/fields.py`

Analyze `__init__` bodies to infer field types. Since phase 3 guarantees annotations or obvious types, analysis is simple pattern matching:

| Pattern                  | Inference                                |
| ------------------------ | ---------------------------------------- |
| `self.x: T = ...`        | Field `x` has type `T`                   |
| `self.x = param`         | Field `x` has type of `param` (SigTable) |
| `self.x = literal`       | Field `x` has type of literal            |
| `self.x = Constructor()` | Field `x` has type `Constructor`         |

No full dataflow needed. Walk `__init__` assignments, resolve RHS types via SigTable.

## Postconditions

- FieldTable maps every class to `[(field_name, type)]`; all fields typed; init order captured
- Fields assigned `None` or conditionally assigned wrapped in `Optional[T]}`
- No manual type override tables needed; types inferred from `__init__` patterns

## Prior Art

- [Java definite assignment](https://docs.oracle.com/javase/specs/jls/se9/html/jls-16.html)
- [TypeScript strictPropertyInitialization](https://www.typescriptlang.org/docs/handbook/2/classes.html)
