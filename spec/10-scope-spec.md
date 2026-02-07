# Phase 10: Scope

**Module:** `middleend/scope.py`

Analyze variable scope: declarations, reassignments, parameter modifications. Walks each function body tracking which variables are declared vs assigned, and whether parameters are modified.

## Variable Annotations

| Annotation              | Meaning                                    |
| ----------------------- | ------------------------------------------ |
| `VarDecl.is_reassigned` | Variable assigned after declaration        |
| `VarDecl.is_const`      | Never reassigned after declaration         |
| `Assign.is_declaration` | First assignment to a new variable         |
| `Var.narrowed_type`     | Precise type at use site after type guards |
| `Var.is_interface`      | Expression statically typed as interface   |
| `Var.is_object_typed`   | Declared with `object` type (top type)     |
| `Var.is_function_ref`   | References a module-level function         |
| `Var.is_constant`       | References a module-level constant         |

## Parameter Annotations

| Annotation          | Meaning                                     |
| ------------------- | ------------------------------------------- |
| `Param.is_modified` | Parameter assigned/mutated in function body |
| `Param.is_unused`   | Parameter never referenced                  |
| `Param.is_callable` | Parameter has `FuncType` (callable)         |

## Annotation Details

### is_object_typed

Tracks variables declared with `object` type. Needed for C#'s string comparison:

```python
def compare(x: object, y: str) -> bool:
    return x == y  # x.is_object_typed=True → cast to string for comparison
```

### is_function_ref / is_constant

Tracks references to module-level functions and constants. Needed for Perl namespacing:

```python
MY_CONST = 42
def my_func(): ...

x = MY_CONST      # x.is_constant=True → emit main::MY_CONST()
f = my_func       # f.is_function_ref=True → emit \&my_func
```

### is_callable

Tracks parameters with callable types. Needed for Perl indirect call syntax:

```python
def apply(f: Callable[[int], int], x: int) -> int:
    return f(x)   # f.is_callable=True → emit $f->($x)
```

## Postconditions

- Every VarDecl, Param, and Assign annotated; reassignment counts accurate
- Variables annotated with `is_const` (never reassigned after declaration); enables `const`/`let` in TS, `final` in Java
- Variables annotated with `is_interface: bool` when statically typed as interface; enables direct `== nil` vs reflection-based nil check in Go
- Variables annotated with `is_object_typed` when declared as `object`; enables correct cast emission in C#
- Variables annotated with `is_function_ref` / `is_constant` for module-level references; enables correct namespace qualification in Perl
- Parameters annotated with `is_callable` when typed as `FuncType`; enables indirect call syntax in Perl
- Variables annotated with precise narrowed type at each use site (not just declaration type); eliminates redundant casts when type is statically known
