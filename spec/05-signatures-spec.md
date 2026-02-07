# Phase 5: Signatures

**Module:** `frontend/signatures.py`

Collect function and method signatures. Parse type annotations into internal representations and validate arity.

## Signature Collection

| Construct               | Collected                            |
| ----------------------- | ------------------------------------ |
| `def f(x: int) -> str:` | Function with params and return type |
| `def m(self, x: int):`  | Method (self excluded from params)   |
| `def f(x: int = 0):`    | Parameter with default value         |
| `def f() -> None:`      | Void return type                     |

## Type Arity

Type constructors must have correct number of arguments:

| Constructor                | Args | Valid                  | Invalid                  |
| -------------------------- | ---- | ---------------------- | ------------------------ |
| `list`, `set`, `frozenset` | 1    | `list[int]`            | `list`, `list[int, str]` |
| `dict`                     | 2    | `dict[str, int]`       | `dict[str]`, `dict`      |
| `tuple`                    | 1+   | `tuple[int, str]`      | `tuple`                  |
| `Callable`                 | 2    | `Callable[[int], str]` | `Callable[int]`          |
| `Optional`                 | 1    | `Optional[int]`        | `Optional`               |

## Errors

| Condition           | Diagnostic                                           |
| ------------------- | ---------------------------------------------------- |
| Wrong arity         | error: `list requires 1 type argument, got 0`        |
| Unknown type        | error: `unknown type 'Foo'`                          |
| Missing return type | error: `function 'f' missing return type annotation` |
| Missing param type  | error: `parameter 'x' missing type annotation`       |

## Output

SigTable mapping each function/method to:
- `name`: function name
- `params`: list of (name, type, has_default, default_value)
- `return_type`: return type
- `is_method`: whether it's a method
- `receiver_type`: class name if method

## Postconditions

All function/method signatures collected; all type annotations parsed and validated; arity errors reported.

## Prior Art

- [Kind (type theory)](https://en.wikipedia.org/wiki/Kind_(type_theory))
- [PEP 484](https://peps.python.org/pep-0484/)
