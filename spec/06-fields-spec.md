# Phase 6: Fields

**Module:** `frontend/fields.py`

Collect field types from class definitions. Fields are declared in class bodies (dataclass style) or assigned in `__init__`.

## Field Patterns

| Pattern                  | Inference                                     |
| ------------------------ | --------------------------------------------- |
| `x: int` (class body)    | Field `x` has type `int`                      |
| `self.x: T = ...`        | Field `x` has type `T`                        |
| `self.x = param`         | Field `x` has type of `param` (from SigTable) |
| `self.x = literal`       | Field `x` has type of literal                 |
| `self.x = Constructor()` | Field `x` has type `Constructor`              |
| `self.kind = "literal"`  | Constant field for discriminated unions       |

## Init Parameter Mapping

For `self.x = param` assignments, track the mapping from parameter to field. This enables backends to generate idiomatic constructors.

```python
class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x  # param_to_field: {"x": "x", "y": "y"}
        self.y = y
```

## Errors

| Condition                   | Diagnostic                              |
| --------------------------- | --------------------------------------- |
| Field type not inferable    | error: `cannot infer type of field 'x'` |
| Duplicate field declaration | error: `field 'x' already declared`     |

## Output

FieldTable mapping each class to:
- `fields`: dict of field name → (name, type)
- `init_params`: parameter order for constructor
- `param_to_field`: mapping from init param to field name
- `const_fields`: constant string fields (for discriminated unions)

## Postconditions

All class fields collected with types; init parameter order captured; constant fields identified.

## Prior Art

- [Java definite assignment](https://docs.oracle.com/javase/specs/jls/se9/html/jls-16.html)
- [TypeScript strictPropertyInitialization](https://www.typescriptlang.org/docs/handbook/2/classes.html)
