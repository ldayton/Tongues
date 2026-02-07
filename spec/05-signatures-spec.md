# Phase 5: Signatures

**Module:** `frontend/signatures.py`

Collect function and method signatures. Parse type annotations into IR types and validate arity.

## Inputs

- **AST**: dict-based AST from Phase 2
- **NameTable**: from Phase 4 (for resolving type names)

## Signature Collection

| Construct               | Collected                            |
| ----------------------- | ------------------------------------ |
| `def f(x: int) -> str:` | Function with params and return type |
| `def m(self, x: int):`  | Method (self excluded from params)   |
| `def f(x: int = 0):`    | Parameter with default value         |
| `def f() -> None:`      | Void return type                     |

## Parameter Modifiers

| Signature               | Params Representation                                      |
| ----------------------- | ---------------------------------------------------------- |
| `def f(a, b):`          | `[(a, T, pos_or_kw), (b, T, pos_or_kw)]`                   |
| `def f(a, /, b):`       | `[(a, T, positional), (b, T, pos_or_kw)]`                  |
| `def f(a, *, b):`       | `[(a, T, pos_or_kw), (b, T, keyword)]`                     |
| `def f(a, /, b, *, c):` | `[(a, T, positional), (b, T, pos_or_kw), (c, T, keyword)]` |

## Default Values

| Pattern          | Stored                                    |
| ---------------- | ----------------------------------------- |
| `x: int`         | `has_default=False`                       |
| `x: int = 0`     | `has_default=True, default=IntLit(0)`     |
| `x: str = ""`    | `has_default=True, default=StringLit("")` |
| `x: bool = True` | `has_default=True, default=BoolLit(True)` |
| `x: T = None`    | `has_default=True, default=NoneLit`       |

Only literal defaults allowed (enforced by subset phase).

## Self Parameter

For methods, `self` is not stored in `params`. Its type is `StructRef(receiver_type)` and available via the `receiver_type` field.

## Forward References

Type names resolved against the module's class definitions. Forward references allowed—a parameter may reference a class defined later in the file. Resolution uses NameTable from Phase 4.

## Type Representation

Python annotations are parsed directly into IR types (see Type System in 00-tongues-spec):

| Python Annotation     | IR Type                                                   |
| --------------------- | --------------------------------------------------------- |
| `int`                 | `INT`                                                     |
| `str`                 | `STRING`                                                  |
| `bool`                | `BOOL`                                                    |
| `float`               | `FLOAT`                                                   |
| `bytes`               | `BYTES`                                                   |
| `list[T]`             | `Slice(T)`                                                |
| `dict[K, V]`          | `Map(K, V)`                                               |
| `set[T]`              | `Set(T)`                                                  |
| `tuple[A, B, C]`      | `Tuple(A, B, C)`                                          |
| `tuple[T, ...]`       | `Tuple(T, variadic=True)`                                 |
| `A \| B`              | `Optional(A)` if B is None, else union via `InterfaceRef` |
| `Callable[[A, B], R]` | `FuncType(params=(A, B), ret=R)`                          |
| `ClassName`           | `StructRef("ClassName")`                                  |

## Type Normalization

| Input         | Normalized Form                           |
| ------------- | ----------------------------------------- |
| `Optional[T]` | `Optional(T)`                             |
| `T \| None`   | `Optional(T)`                             |
| `A \| B`      | `InterfaceRef` if common base, else error |
| `Union[A, A]` | `A` (deduplicated)                        |

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
- `params`: list of (name, type, modifier, has_default, default_value)
- `return_type`: return type
- `is_method`: whether it's a method
- `receiver_type`: class name if method

## Postconditions

All function/method signatures collected; all type annotations parsed and validated; arity errors reported.

## Prior Art

- [Kind (type theory)](https://en.wikipedia.org/wiki/Kind_(type_theory))
- [PEP 484](https://peps.python.org/pep-0484/)
