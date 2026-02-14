# Phase 5: Signatures

**Module:** `frontend/signatures.py`

Collect function and method signatures. Parse Python type annotations into Taytsh types, classify parameter kinds, record default values, and detect mutated parameters.

## Inputs

- **AST**: dict-based AST from phase 2
- **NameTable**: from phase 4 (for resolving type names)

## Signature Collection

Top-level functions and class methods are collected in a single pass over the module body.

| Construct               | Collected as                         |
| ----------------------- | ------------------------------------ |
| `def f(x: int) -> str:` | Free function in `symbols.functions` |
| `class C:` methods      | Methods in struct's `methods` dict   |

For classes, every `FunctionDef` in the class body is collected as a method with `is_method=True` and `receiver_type` set to the class name. Node subclass methods (excluding `__init__`, `__repr__`, `to_sexp`, `kind`, `ToSexp`, `GetKind`) are registered in a `method_to_structs` mapping for dispatch.

## Parameter Kinds

Parameters are classified by position relative to `/` and `*` markers:

| Signature               | Params                                            |
| ----------------------- | ------------------------------------------------- |
| `def f(a, b):`          | `[(a, pos_or_kw), (b, pos_or_kw)]`                |
| `def f(a, /, b):`       | `[(a, positional), (b, pos_or_kw)]`               |
| `def f(a, *, b):`       | `[(a, pos_or_kw), (b, keyword)]`                  |
| `def f(a, /, b, *, c):` | `[(a, positional), (b, pos_or_kw), (c, keyword)]` |

The three parameter groups come from the AST:

| AST field     | Position        | Modifier     |
| ------------- | --------------- | ------------ |
| `posonlyargs` | Before `/`      | `positional` |
| `args`        | Between `/`-`*` | `pos_or_kw`  |
| `kwonlyargs`  | After `*`       | `keyword`    |

## Self Parameter

For methods, `self` is filtered out of the parameter list. It does not appear in `params`. The receiver type is stored separately:

```
FuncInfo.receiver_type = "ClassName"
FuncInfo.is_method = True
```

## Default Values

Defaults are lowered to IR expressions at collection time.

| Pattern          | Stored                                          |
| ---------------- | ----------------------------------------------- |
| `x: int`         | `has_default=False, default_value=None`         |
| `x: int = 0`     | `has_default=True, default_value=IntLit(0)`     |
| `x: str = ""`    | `has_default=True, default_value=StringLit("")` |
| `x: bool = True` | `has_default=True, default_value=BoolLit(True)` |
| `x: T = None`    | `has_default=True, default_value=NilLit`        |

Only literal defaults are allowed (enforced by the subset phase). For positional and regular parameters, defaults are matched from the tail — Python's `defaults` list covers the rightmost N parameters of `posonlyargs + args` combined. Keyword-only parameters have a parallel `kw_defaults` list with `None` entries for parameters without defaults.

## Type Translation

Python annotations are parsed to strings, then translated to Taytsh types:

| Python Annotation     | Taytsh Type                  |
| --------------------- | ---------------------------- |
| `int`                 | `int`                        |
| `str`                 | `string`                     |
| `bool`                | `bool`                       |
| `float`               | `float`                      |
| `bytes`               | `bytes`                      |
| `list[T]`             | `list[T]`                    |
| `dict[K, V]`          | `map[K, V]`                  |
| `set[T]`              | `set[T]`                     |
| `tuple[A, B, C]`      | `(A, B, C)`                  |
| `tuple[T, ...]`       | variadic tuple               |
| `T \| None`           | `T?`                         |
| `Optional[T]`         | `T?`                         |
| `A \| B`              | interface ref if common base |
| `Callable[[A, B], R]` | `fn[A, B, R]`                |
| `ClassName`           | struct ref                   |

When a type annotation cannot be parsed, the parameter falls back to `interface("any")`.

## Mutated Parameter Detection

Before building the parameter list, the function body is scanned for mutations to parameters. A parameter is considered mutated if:

| Pattern            | Mutation type    |
| ------------------ | ---------------- |
| `param.append(x)`  | Method call      |
| `param.extend(xs)` | Method call      |
| `param.clear()`    | Method call      |
| `param.pop()`      | Method call      |
| `param[i] = x`     | Subscript assign |

Mutated `list[T]` parameters are wrapped in `Pointer(list[T])` to signal pass-by-reference semantics to backends.

## Type Normalization

| Input         | Normalized Form    |
| ------------- | ------------------ |
| `Optional[T]` | `T?`               |
| `T \| None`   | `T?`               |
| `A \| B`      | interface ref      |
| `Union[A, A]` | `A` (deduplicated) |

## Type Arity

Type constructors must have the correct number of arguments:

| Constructor                | Args | Valid                  | Invalid                  |
| -------------------------- | ---- | ---------------------- | ------------------------ |
| `list`, `set`, `frozenset` | 1    | `list[int]`            | `list`, `list[int, str]` |
| `dict`                     | 2    | `dict[str, int]`       | `dict[str]`, `dict`      |
| `tuple`                    | 1+   | `tuple[int, str]`      | `tuple`                  |
| `Callable`                 | 2    | `Callable[[int], str]` | `Callable[int]`          |
| `Optional`                 | 1    | `Optional[int]`        | `Optional`               |

## Errors

| Condition           | Diagnostic                                            |
| ------------------- | ----------------------------------------------------- |
| Missing return type | error: `function 'f' missing return type annotation`  |
| Missing param type  | error: `parameter 'x' missing type annotation in f()` |
| Wrong arity         | error: `list requires 1 type argument, got 0`         |
| Unknown type        | error: `unknown type 'Foo'`                           |

## Output

For each function: `FuncInfo` containing:

| Field            | Content                            |
| ---------------- | ---------------------------------- |
| `name`           | function name                      |
| `params`         | list of `ParamInfo`                |
| `return_type`    | Taytsh return type                 |
| `return_py_type` | original Python return type string |
| `is_method`      | whether it's a method              |
| `receiver_type`  | class name if method, else `""`    |

For each parameter: `ParamInfo` containing:

| Field           | Content                                        |
| --------------- | ---------------------------------------------- |
| `name`          | parameter name                                 |
| `typ`           | Taytsh type (with Pointer wrapping if mutated) |
| `py_type`       | original Python type string                    |
| `has_default`   | whether a default value exists                 |
| `default_value` | IR expression for the default, or `None`       |
| `modifier`      | `positional`, `pos_or_kw`, or `keyword`        |

## Postconditions

All function and method signatures collected. All type annotations parsed to Taytsh types. Parameter kinds classified. Default values lowered to IR. Mutated list parameters wrapped in Pointer. Arity errors reported.
