# Phase 6: Fields

**Module:** `frontend/fields.py`

Collect field types from class definitions. Fields are declared in class bodies (dataclass-style annotations) or assigned in `__init__`. Also determines constructor parameters, constant discriminator fields, and auto-generated kind values.

## Inputs

- **AST**: dict-based AST from phase 2
- **SigTable**: from phase 5 (for parameter types in `self.x = param`)
- **Base class info**: from NameTable (phase 4), for detecting hierarchy root and suppressing kind generation on root class

## Field Discovery

Fields come from two sources, collected in order:

### Class Body Annotations

Class-level `x: T` and `x: T = default` statements. Collected first and take priority.

| Pattern                      | Result                                               |
| ---------------------------- | ---------------------------------------------------- |
| `x: int`                     | field `x`, type `int`, no default                    |
| `x: int = 0`                 | field `x`, type `int`, default `IntLit(0)`           |
| `x: str = "foo"`             | field `x`, type `string`, default `StringLit("foo")` |
| `x: bool = True`             | field `x`, type `bool`, default `BoolLit(True)`      |
| `field(default_factory=...)` | error: not allowed                                   |

Duplicate class-body field names are errors.

### `__init__` Assignments

After class body fields, `__init__` is walked for `self.attr` assignments (both `self.x = ...` and `self.x: T = ...`). The walk traverses nested control flow but skips nested functions.

| Pattern              | Inference                                                   |
| -------------------- | ----------------------------------------------------------- |
| `self.x = param`     | type from parameter annotation; records param→field mapping |
| `self.x = "literal"` | constant string field (discriminator)                       |
| `self.x: T = ...`    | type from annotation `T`                                    |
| `self.x = expr`      | type inferred from expression via callbacks                 |

If a field was already declared in the class body, the `__init__` assignment checks for type consistency but does not create a new entry.

## Field Sources and Priority

| Source                | Priority | Notes                             |
| --------------------- | -------- | --------------------------------- |
| Class body annotation | 1        | `x: int` or `x: int = 0`          |
| `__init__` assignment | 2        | Only if not already in class body |

Type mismatches between body annotation and init assignment are errors:

```
error: field 'x' declared as int but assigned str
```

## Default Values

Only literal constants are stored as defaults:

| AST Value    | IR Expression |
| ------------ | ------------- |
| `bool` const | `BoolLit`     |
| `int` const  | `IntLit`      |
| `str` const  | `StringLit`   |
| other        | `None`        |

## Inherited Fields

`StructInfo.fields` stores only the class's own fields. Inherited fields are accessed via the hierarchy (phase 7).

| Class                       | `fields` contains    |
| --------------------------- | -------------------- |
| `class A:` with `x: int`    | `{x: int}`           |
| `class B(A):` with `y: str` | `{y: str}` (not `x`) |

## Constructor Parameters

`init_params` records parameter order for constructor generation:

- **With `__init__`**: parameters are collected from the function signature (excluding `self`)
- **Dataclass without `__init__`**: init_params is populated from class body field names in declaration order

`needs_constructor` is set to `True` when:
- The class has `init_params`
- The class has computed init assignments (not simple `self.x = param`)
- The class is an exception

## Discriminator Fields

### Explicit Const Fields

A field is const if assigned a string literal directly in `__init__`:

| Pattern             | `const_fields`    |
| ------------------- | ----------------- |
| `self.kind = "add"` | `{"kind": "add"}` |
| `self.tag = "foo"`  | `{"tag": "foo"}`  |
| `self.kind = param` | not const         |

### Auto-Generated Kind

For classes with an `__init__` that are not the hierarchy root, if no `kind` const field exists and no init parameter maps to `kind`, a kind value is auto-generated from the class name using PascalCase→kebab-case conversion:

| Class Name | Generated `const_fields["kind"]` |
| ---------- | -------------------------------- |
| `Add`      | `"add"`                          |
| `BinaryOp` | `"binary-op"`                    |
| `ForRange` | `"for-range"`                    |
| `IfStmt`   | `"if-stmt"`                      |

The hierarchy root does not get a kind value.

### Param-to-Field Mapping

For `self.x = param` assignments, the mapping from parameter name to field name is recorded:

```python
class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x  # param_to_field: {"x": "x", "y": "y"}
        self.y = y
```

## Conditional and Outside-Init Restrictions

| Condition                  | Behavior                                        |
| -------------------------- | ----------------------------------------------- |
| `if cond: self.x = 1`      | error: conditional field assignment not allowed |
| `for ...: self.x = 1`      | error: conditional field assignment not allowed |
| `self.x = ...` in non-init | error: field 'x' must be assigned in `__init__` |

Field assignments inside `if`, `for`, or `while` blocks within `__init__` are rejected. Methods other than `__init__` may not introduce new fields.

## Dataclass Handling

`@dataclass` classes use class body annotations as the canonical field source. `@dataclass(kw_only=True)` sets `kw_only=True` on the struct.

| Decorator                  | `is_dataclass` | `kw_only` |
| -------------------------- | -------------- | --------- |
| `@dataclass`               | `True`         | `False`   |
| `@dataclass(kw_only=True)` | `True`         | `True`    |
| `@dataclass(eq=True)`      | `True`         | `False`   |
| No decorator               | `False`        | `False`   |

## Node Subclass Field Tracking

For classes marked `is_node` (hierarchy subclasses), fields are registered in a `field_to_structs` mapping that tracks which structs contain each field name. Used by later phases for field access resolution.

## Type Unwrapping

Field types are unwrapped: `Pointer(StructRef(X))` → `StructRef(X)`. Fields store the inner type, not the pointer wrapper.

## Errors

| Condition                   | Diagnostic                                           |
| --------------------------- | ---------------------------------------------------- |
| Duplicate field declaration | error: `field 'x' already declared`                  |
| Type mismatch body vs init  | error: `field 'x' declared as int but assigned str`  |
| Conditional assignment      | error: `conditional field assignment not allowed: x` |
| Field outside init          | error: `field 'x' must be assigned in __init__`      |
| default_factory             | error: `field(default_factory=...) not allowed`      |

## Output

`StructInfo` updated with:

| Field               | Content                                          |
| ------------------- | ------------------------------------------------ |
| `fields`            | `{name → FieldInfo}` with type, default          |
| `init_params`       | parameter names in constructor order             |
| `param_to_field`    | `{param_name → field_name}`                      |
| `const_fields`      | `{field_name → string_value}` for discriminators |
| `is_dataclass`      | whether class has `@dataclass` decorator         |
| `kw_only`           | whether `@dataclass(kw_only=True)`               |
| `needs_constructor` | whether backends should emit a constructor       |

Each `FieldInfo` carries:

| Field         | Content                              |
| ------------- | ------------------------------------ |
| `name`        | field name                           |
| `typ`         | Taytsh type                          |
| `py_name`     | original Python field name           |
| `has_default` | whether a default value exists       |
| `default`     | IR expression for default, or `None` |

## Postconditions

All class fields collected with types. Constructor parameter order captured. Constant discriminator fields identified with auto-generated kind values for hierarchy subclasses. Conditional and outside-init field assignments rejected. Dataclass metadata recorded.
