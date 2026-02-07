# Phase 6: Fields

**Module:** `frontend/fields.py`

Collect field types from class definitions. Fields are declared in class bodies (dataclass style) or assigned in `__init__`.

## Inputs

- **AST**: dict-based AST from Phase 2
- **SigTable**: from Phase 5 (for parameter types in `self.x = param`)

## Field Patterns

| Pattern                  | Inference                                     |
| ------------------------ | --------------------------------------------- |
| `x: int` (class body)    | Field `x` has type `int`                      |
| `self.x: T = ...`        | Field `x` has type `T`                        |
| `self.x = param`         | Field `x` has type of `param` (from SigTable) |
| `self.x = literal`       | Field `x` has type of literal                 |
| `self.x = Constructor()` | Field `x` has type `Constructor`              |
| `self.kind = "literal"`  | Constant field for discriminated unions       |

## Field Sources

| Source                | Priority | Notes                     |
| --------------------- | -------- | ------------------------- |
| Class body annotation | 1        | `x: int` or `x: int = 0`  |
| `__init__` assignment | 2        | Only if not in class body |

If both exist, class body annotation wins. Type mismatch between body annotation and init assignment is an error.

## Class Body Defaults

| Pattern                                     | Stored                                |
| ------------------------------------------- | ------------------------------------- |
| `x: int`                                    | `has_default=False`                   |
| `x: int = 0`                                | `has_default=True, default=IntLit(0)` |
| `x: list[int] = field(default_factory=...)` | Not allowed (subset)                  |

## Inheritance

FieldTable stores only the class's own fields. Inherited fields accessed via Phase 7 hierarchy.

| Class                       | `fields` contains  |
| --------------------------- | ------------------ |
| `class A:` with `x: int`    | `{x: int}`         |
| `class B(A):` with `y: str` | `{y: str}` (not x) |

## Init Parameter Mapping

For `self.x = param` assignments, track the mapping from parameter to field. This enables backends to generate idiomatic constructors.

```python
class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x  # param_to_field: {"x": "x", "y": "y"}
        self.y = y
```

## Const Fields

A field is const if assigned a string literal directly and not reassigned elsewhere.

| Pattern             | const_fields      |
| ------------------- | ----------------- |
| `self.kind = "add"` | `{"kind": "add"}` |
| `self.tag = "foo"`  | `{"tag": "foo"}`  |
| `self.kind = param` | not const         |

Used for discriminated union narrowing. Field name is typically `kind` but any name works.

## Conditional and Computed Fields

| Pattern               | Behavior                      |
| --------------------- | ----------------------------- |
| `if cond: self.x = 1` | error: conditional assignment |
| `self.x = a + b`      | Type inferred from expression |
| `self.x = func()`     | Type is return type of func   |

## Field Ordering

Fields ordered by declaration: class body fields first (top to bottom), then init-only fields (assignment order). This order used for constructor parameter generation.

## Dataclass Handling

`@dataclass` classes use class body annotations as the canonical field source. No `__init__` analysis needed—dataclass generates it. `@dataclass(kw_only=True)` marks all fields as keyword-only in init_params.

## Errors

| Condition                   | Diagnostic                                          |
| --------------------------- | --------------------------------------------------- |
| Field type not inferable    | error: `cannot infer type of field 'x'`             |
| Duplicate field declaration | error: `field 'x' already declared`                 |
| Type mismatch body vs init  | error: `field 'x' declared as int but assigned str` |
| Conditional assignment      | error: `conditional field assignment not allowed`   |
| Field assigned outside init | error: `field 'x' must be assigned in __init__`     |

## Output

FieldTable mapping each class to:
- `fields`: dict of field name → (name, type, has_default, default)
- `init_params`: parameter order for constructor
- `param_to_field`: mapping from init param to field name
- `const_fields`: constant string fields (for discriminated unions)
- `is_dataclass`: whether class has `@dataclass` decorator

## Postconditions

All class fields collected with types; init parameter order captured; constant fields identified.

## Prior Art

- [Java definite assignment](https://docs.oracle.com/javase/specs/jls/se9/html/jls-16.html)
- [TypeScript strictPropertyInitialization](https://www.typescriptlang.org/docs/handbook/2/classes.html)
