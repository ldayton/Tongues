# Phase 8: Inference

**Modules:** `frontend/inference.py`, `frontend/type_inference.py`, `frontend/validate.py`

Bidirectional type inference with flow-sensitive narrowing and validation. Computes Taytsh types for all expressions, infers local variable types from assignments, enforces type safety constraints, and validates iterator/generator consumption. This phase has two sub-phases: type computation (8a) annotates the AST with inferred types, and type validation (8b) walks the typed AST with a flow-sensitive environment to reject type errors.

## Inputs

- **AST**: dict-based AST from phase 2
- **NameTable**: from phase 4
- **SigTable**: from phase 5 (function/method signatures with Taytsh types)
- **FieldTable**: from phase 6 (field types)
- **SubtypeRel**: from phase 7 (hierarchy root, node types)

## Taytsh Type Language

### Primitives

| Python  | Taytsh   |
| ------- | -------- |
| `int`   | `int`    |
| `float` | `float`  |
| `str`   | `string` |
| `bytes` | `bytes`  |
| `bool`  | `bool`   |

### Compound Types

| Python             | Taytsh                    |
| ------------------ | ------------------------- |
| `list[T]`          | `list[T]`                 |
| `dict[K, V]`       | `map[K, V]`               |
| `set[T]`           | `set[T]`                  |
| `tuple[A, B, C]`   | `(A, B, C)`               |
| `tuple[T, ...]`    | variadic tuple            |
| `T \| None`        | `T?`                      |
| `A \| B`           | `A \| B` or interface ref |
| `Callable[[A], R]` | `fn[A, R]`                |
| `object`           | `interface("any")`        |

### Subtyping

| Relation       | Holds when              |
| -------------- | ----------------------- |
| `A <: A \| B`  | Always (union widening) |
| `C <: P`       | `C` inherits from `P`   |
| `bool <: int`  | Always (numeric)        |
| `int <: float` | Always (numeric)        |
| `T <: T`       | Always (reflexivity)    |

These are Python-side subtyping rules used during inference. Taytsh IR has no implicit numeric coercion — the lowering phase inserts explicit casts (`IntToFloat`, etc.) where needed.

Collections are invariant: `list[Dog]` is not a subtype of `list[Animal]`.

## Synthesis and Checking

### Synthesis

Expressions produce a type without external guidance:

| Expression     | Synthesized Type                       |
| -------------- | -------------------------------------- |
| `42`           | `int`                                  |
| `"hello"`      | `string`                               |
| `True`         | `bool`                                 |
| `3.14`         | `float`                                |
| `None`         | `interface("any")`                     |
| `x` (variable) | lookup in flow-sensitive env           |
| `f(args...)`   | return type of `f`                     |
| `[a, b, c]`    | `list[T]` where T = first element type |
| `{k: v}`       | `map[K, V]` from first entry           |
| `(a, b, c)`    | `(T, U, V)` per-element                |

### Checking

Expressions are verified against an expected type:

| Context              | Checked Expression       |
| -------------------- | ------------------------ |
| `def f(x: T)` call   | argument against param T |
| `return e` in `-> T` | `e` against return type  |
| `x: T = e`           | `e` against declared T   |
| `xs.append(e)`       | `e` against element type |
| `d[k] = v`           | key and value types      |

### Assignability

A type `actual` is assignable to `expected` when:
- They are the same type
- Either is `interface("any")`
- `actual` is `bool` and `expected` is `int` or `float`
- `actual` is `int` and `expected` is `float`
- `expected` is `T?` and `actual` is assignable to `T`
- `actual` is a struct in the node hierarchy and `expected` is the hierarchy root interface
- Both are collection types with assignable element types (invariant: same direction)
- Both are function types with matching parameter counts and compatible param/return types

## Local Variable Inference

Local variables without annotations get types from their first assignment:

| Pattern         | Inferred Type         |
| --------------- | --------------------- |
| `x = 42`        | `int`                 |
| `x = [1, 2, 3]` | `list[int]`           |
| `x = func()`    | return type of `func` |

Empty collections without annotations are errors — the element type cannot be inferred:

```
error: empty list needs type annotation
error: empty dict needs type annotation
```

## Type Narrowing

Flow-sensitive narrowing refines variable types within guarded branches. The validator maintains a `TypeEnv` that tracks both IR types and original Python source types (for union/optional awareness after IR erasure).

### Narrowing Guards

| Guard                     | Then-branch narrows to  | Else-branch narrows to  |
| ------------------------- | ----------------------- | ----------------------- |
| `isinstance(x, T)`        | `T`                     | remaining union members |
| `not isinstance(x, T)`    | remaining union members | `T`                     |
| `x is None`               | (unchanged)             | non-None part           |
| `x is not None`           | non-None part           | (unchanged)             |
| `if x:` (optional)        | non-None part           | (unchanged)             |
| `x.kind == "foo"`         | (deferred to lowering)  | (deferred to lowering)  |
| `assert isinstance(x, T)` | `T` (after assert)      | —                       |
| `assert x is not None`    | non-None part (after)   | —                       |

### Early Exit Narrowing

When a then-branch always returns, the else type propagates to subsequent code:

```python
if x is None:
    return 0
# x: non-None here
```

### Compound Guards

- `a and b`: narrowing propagates left-to-right through the chain
- `isinstance(x, A) or isinstance(x, B)`: extracts union `[A, B]` for the then-branch
- `not isinstance(x, T)`: inverts the narrowing (then gets complement, else gets T)

### Attribute Path Guards

`x.attr is not None` and `x.attr is None` are tracked as guarded attribute paths, allowing subsequent access to `x.attr` without optional errors.

### Walrus Narrowing

`if (val := func()):` narrows `val` to the non-None part of the return type when the function returns an optional.

### Source Type Tracking

The validator maintains `source_types` alongside IR types. This is necessary because some Python union types are represented as interface refs in IR, erasing the original union members. Source types preserve the original union structure for accurate narrowing and error reporting.

## Truthiness

Only types with unambiguous truthiness are allowed in boolean contexts (`if`, `while`, ternary conditions).

### Allowed

| Type        | `if x:` means   |
| ----------- | --------------- |
| `bool`      | boolean value   |
| `T?`        | `x is not None` |
| `list[T]`   | `len(x) > 0`    |
| `map[K, V]` | `len(x) > 0`    |
| `set[T]`    | `len(x) > 0`    |
| `string`    | `len(x) > 0`    |

### Rejected

| Type        | Diagnostic                                             |
| ----------- | ------------------------------------------------------ |
| `int`       | `truthiness of int not allowed (zero is valid data)`   |
| `float`     | `truthiness of float not allowed (zero is valid data)` |
| `string?`   | `ambiguous truthiness for optional str`                |
| `list[T]?`  | `ambiguous truthiness for optional list`               |
| `map[K,V]?` | `ambiguous truthiness for optional dict`               |
| `set[T]?`   | `ambiguous truthiness for optional set`                |

## Optionals and Nil

### Unguarded Access

Accessing fields or methods on an optional type without narrowing is an error:

```
error: cannot access 'attr' on optional type (may be None)
error: cannot call method on optional type (may be None)
```

## Interfaces and Polymorphism

Variables typed as the hierarchy root use `InterfaceRef`. Struct pointers in the hierarchy are assignable to the interface.

### Unguarded Union/Object Access

Operations on un-narrowed union or object types are errors:

| Operation                   | Diagnostic                                            |
| --------------------------- | ----------------------------------------------------- |
| `obj.attr` (object type)    | `cannot access attribute on object without narrowing` |
| `obj.method()` (union type) | `cannot call method on union type without narrowing`  |
| `obj + 1` (object type)     | `cannot use object in arithmetic without narrowing`   |
| `obj[i]` (object type)      | `cannot subscript object without narrowing`           |
| `union + 1` (optional)      | `cannot use T \| None in arithmetic (may be None)`    |

Exception: `.kind` access is always allowed on union types for discrimination.

### Union Field Access

For multi-variant unions (non-optional), field access is allowed only if all union members share the field:

```
error: attribute 'x' not available on all union members
```

### Interface Method Availability

Calling a method on an interface-typed variable checks that the method exists on the hierarchy root struct.

## Unions

When a source type is a union (`A | B`), the else-branch of an `isinstance` check narrows to the remaining members:

```python
def f(x: A | B | C) -> None:
    if isinstance(x, A):
        ...  # x: A
    # x: B | C here
```

## Tuples and Unpacking

### Tuple Index Bounds

Static tuple indexing is bounds-checked:

```
error: tuple index 2 out of bounds for tuple of length 2
```

### Tuple Unpacking

Count must match:

```
error: cannot unpack tuple of 3 elements into 2 targets
```

Optional tuples cannot be unpacked without guarding:

```
error: cannot unpack optional tuple without guard
```

## Collection Typing

### Element Type Enforcement

Collection mutations are type-checked against the element type:

| Operation         | Check                                        |
| ----------------- | -------------------------------------------- |
| `xs.append(v)`    | `v` assignable to element type               |
| `xs.extend(ys)`   | `ys` element type assignable to `xs` element |
| `xs.insert(i, v)` | `v` assignable to element type               |
| `s.add(v)`        | `v` assignable to set element type           |
| `d[k] = v`        | `k` assignable to key type, `v` to value     |

### Literal Homogeneity

List and dict literals must have consistent types:

```
error: mixed types in list literal: int and str
error: mixed key types in dict literal
error: mixed value types in dict literal
```

Exception: union-typed list contexts (e.g., return type is `list[int | str]`) allow mixed literals.

### Arithmetic Type Checking

String concatenation with non-string operands is rejected:

```
error: cannot add str and int
```

List concatenation requires matching element types:

```
error: cannot concatenate list[int] and list[str]
```

## Iterator Consumption

### Single-Use Iterators

`enumerate`, `zip`, and `reversed` return single-use iterators that must be consumed immediately.

| Allowed                     | Consumer        |
| --------------------------- | --------------- |
| `for i, x in enumerate(xs)` | for-loop header |
| `list(zip(xs, ys))`         | eager consumer  |
| `sorted(enumerate(xs))`     | eager consumer  |

Eager consumers: `list`, `tuple`, `set`, `dict`, `frozenset`, `sum`, `min`, `max`, `any`, `all`, `sorted`, plus `str.join`.

### Iterator Escape

| Escaped form          | Diagnostic                                         |
| --------------------- | -------------------------------------------------- |
| `e = enumerate(xs)`   | `cannot assign enumerate() to variable`            |
| `return zip(xs, ys)`  | `cannot return zip()`                              |
| `foo(enumerate(xs))`  | `cannot pass enumerate() to non-consumer function` |
| `g = (x for x in xs)` | `cannot assign generator expression to variable`   |
| `return (x for ...)`  | `cannot return generator expression`               |
| `foo(x for x in xs)`  | `cannot pass generator expression to non-consumer` |

Exception: `return list(enumerate(xs))` is allowed (eager consumer wrapping).

## Callable Typing

### Argument Type Checking

Function calls check argument types against parameter types:

```
error: argument 1 has type str, expected int
error: expected 2 arguments, got 3
```

### Callable Variable Invocation

Variables typed as `fn[A, R]` are checked for argument count and types at call sites.

### Function Reference Resolution

When a function name is passed where a callable is expected, its signature is resolved to a `FnT` for compatibility checking. Built-in functions (`len`, `str`, `int`, `bool`) have known signatures.

### Unbound Method Detection

Calling `ClassName.method(args)` without a receiver is rejected:

```
error: cannot call method without self: ClassName.method
```

## Errors

| Condition                   | Diagnostic                                            |
| --------------------------- | ----------------------------------------------------- |
| Type mismatch in assignment | `cannot assign str to int`                            |
| Type mismatch in return     | `cannot return str as int`                            |
| Type mismatch in call       | `argument 1 has type str, expected int`               |
| Arity mismatch              | `expected 2 arguments, got 3`                         |
| Unguarded optional access   | `cannot access 'x' on optional type (may be None)`    |
| Unguarded union method      | `cannot call method on union type without narrowing`  |
| Unguarded object operation  | `cannot access attribute on object without narrowing` |
| Ambiguous truthiness        | `truthiness of int not allowed (zero is valid data)`  |
| Iterator escape             | `cannot assign enumerate() to variable`               |
| Generator escape            | `cannot assign generator expression to variable`      |
| Tuple unpack mismatch       | `cannot unpack tuple of 3 elements into 2 targets`    |
| Tuple index out of bounds   | `tuple index 2 out of bounds for tuple of length 2`   |
| Empty collection untyped    | `empty list needs type annotation`                    |
| Mixed literal types         | `mixed types in list literal: int and str`            |
| Bad string arithmetic       | `cannot add str and int`                              |
| Invalid kind value          | `kind value 'foo' does not match any known type`      |
| Unbound method call         | `cannot call method without self: Foo.bar`            |
| Builtin type error          | `len() requires a sized type, got int`                |

## Output

The TypedAST is the input dict-AST with type annotations added:

| Node type       | Added field      | Contents                        |
| --------------- | ---------------- | ------------------------------- |
| All expressions | `_expr_type`     | Taytsh type                     |
| Variable refs   | `_resolved`      | declaration kind from NameTable |
| Narrowed uses   | `_narrowed_type` | more precise type after guards  |

Structure otherwise identical to phase 2 output. Subsequent phases traverse the same AST and read type information from annotations.

## Postconditions

- All expressions annotated with synthesized or checked types
- All local variables have inferred types
- Type narrowing applied per control flow (isinstance, is None, is not None, truthiness, assert, walrus)
- Optional types validated — unguarded access rejected
- Union types validated — unguarded member access rejected unless shared
- Iterator/generator consumption verified — escapes rejected
- Tuple unpacking count and optionality verified
- Collection element types enforced on mutation
- Literal homogeneity enforced
- Truthiness restricted to unambiguous types
- Callable argument types and arity checked
- Kind check values validated against known structs
