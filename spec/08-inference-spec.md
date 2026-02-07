# Tongues Type System

This document describes the type system enforced during phases 5–8. It requires knowing the types of expressions and cannot be checked from AST structure alone.

See [subset-spec.md](../03_subset/_subset-spec.md) for syntactic restrictions checkable on the AST.

---

## 1. Type Language

### Primitive Types

| Type    | Values                       |
| ------- | ---------------------------- |
| `int`   | Arbitrary-precision integers |
| `float` | IEEE 754 double-precision    |
| `str`   | Unicode strings              |
| `bytes` | Byte sequences               |
| `bool`  | `True`, `False`              |

### Compound Types

| Form             | Meaning                            |
| ---------------- | ---------------------------------- |
| `list[T]`        | Homogeneous sequence of `T`        |
| `dict[K, V]`     | Map from `K` to `V`                |
| `set[T]`         | Unordered collection of unique `T` |
| `tuple[A, B, C]` | Fixed-length heterogeneous product |

### Optional and Union Types

| Form        | Meaning                    |
| ----------- | -------------------------- |
| `T \| None` | Nullable `T`               |
| `A \| B`    | Value is either `A` or `B` |

Unions are unordered: `A | B` equals `B | A`.

### Callable Types

| Form                  | Meaning                       |
| --------------------- | ----------------------------- |
| `Callable[[A, B], R]` | Function from `(A, B)` to `R` |

Bound methods have the receiver already applied:

```python
class Parser:
    def parse(self, s: str) -> Node: ...

p.parse  # has type Callable[[str], Node], not Callable[[Parser, str], Node]
```

### Subtyping

Subtyping is structural for unions, nominal for classes:

| Relation      | Holds when              |
| ------------- | ----------------------- |
| `A <: A \| B` | Always (union widening) |
| `C <: P`      | `C` inherits from `P`   |
| `T <: T`      | Always (reflexivity)    |

Collections are invariant: `list[Dog]` is not a subtype of `list[Animal]`.

---

## 2. Typing Judgments

The type system uses bidirectional typing: expressions either **synthesize** a type or are **checked** against an expected type.

### Synthesis (Γ ⊢ e ⇒ T)

The expression `e` produces type `T` without external guidance.

| Expression     | Synthesized Type                  |
| -------------- | --------------------------------- |
| `42`           | `int`                             |
| `"hello"`      | `str`                             |
| `True`         | `bool`                            |
| `x` (variable) | lookup `x` in Γ                   |
| `obj.field`    | field type from class definition  |
| `f(args...)`   | return type of `f`                |
| `xs[i]`        | element type of `xs`              |
| `[a, b, c]`    | `list[T]` where `T` = common type |

### Checking (Γ ⊢ e ⇐ T)

The expression `e` is verified against expected type `T`.

| Context              | Checked Expression       |
| -------------------- | ------------------------ |
| `def f(x: T)`        | argument at call site    |
| `return e` in `-> T` | `e`                      |
| `x: T = e`           | `e`                      |
| `xs.append(e)`       | `e` against element type |

Checking succeeds if the synthesized type is a subtype of the expected type.

### Subsumption

When a synthesized type `S` is checked against `T`, the check passes if `S <: T`:

```python
def takes_animal(a: Animal) -> None: ...
dog: Dog = Dog()
takes_animal(dog)  # Dog <: Animal, passes
```

---

## 3. Type Narrowing

Type narrowing refines a variable's type based on control flow. After a guard, the variable has a more precise type in the guarded branch.

### Narrowing Guards

| Guard              | Narrows `x` to         | In Branch |
| ------------------ | ---------------------- | --------- |
| `isinstance(x, T)` | `T`                    | then      |
| `x is None`        | `None`                 | then      |
| `x is not None`    | non-None part of union | then      |
| `if x:` (optional) | non-None part          | then      |
| `x.kind == "foo"`  | variant with that kind | then      |

### Narrowing Propagates Past Guards

Early exits refine types in subsequent code:

```python
def process(x: int | None) -> int:
    if x is None:
        return 0
    # x: int here (None case returned)
    return x + 1
```

### Kind-Based Discrimination

Tagged unions use a `kind` field to discriminate:

```python
@dataclass
class Add:
    kind: str = "add"
    left: int
    right: int

@dataclass
class Sub:
    kind: str = "sub"
    left: int
    right: int

def eval(node: Add | Sub) -> int:
    if node.kind == "add":
        # node: Add here
        return node.left + node.right
    # node: Sub here
    return node.left - node.right
```

---

## 4. Truthiness and Coercion

Python's `if x:` has type-dependent semantics. Only types with unambiguous truthiness are allowed:

| Type        | `if x:` means   | Rationale   |
| ----------- | --------------- | ----------- |
| `bool`      | boolean value   | Direct test |
| `T \| None` | `x is not None` | Null check  |
| `list[T]`   | `len(x) > 0`    | Non-empty   |
| `str`       | `len(x) > 0`    | Non-empty   |

### Rejected: Ambiguous Truthiness

Types where truthiness conflates distinct conditions are rejected:

```python
x: list[int] | None = ...
if x:  # ERROR: does this mean non-None or non-empty?
    ...
```

Explicit checks required:

```python
if x is not None:      # tests None
    if x:              # tests empty (x: list[int] here)
        ...
```

---

## 5. Tuple Types and Unpacking

Tuples are fixed-length products with per-position types.

### Tuple Construction

```python
t: tuple[int, str, bool] = (1, "a", True)
```

Each position is checked independently.

### Tuple Unpacking

Unpacking requires the right-hand side to have known tuple type:

| Allowed                 | Type of RHS               |
| ----------------------- | ------------------------- |
| `a, b = func()`         | `func` returns tuple      |
| `a, b = (1, "x")`       | literal tuple             |
| `if t := f(): a, b = t` | guarded (proves non-None) |

Unpacking from an unguarded variable is rejected because tuple type may not be statically known:

```python
t = func()      # t: tuple[int, str] | None perhaps
a, b = t        # ERROR: may be None
```

---

## 6. Collection Typing

### Element Type Constraints

Collection methods are typed by element type:

```python
xs: list[int] = []
xs.append(1)      # OK: int matches element type
xs.append("x")    # ERROR: str is not int
```

### Dict Key and Value Types

```python
d: dict[str, int] = {}
d["key"] = 42     # OK
d[42] = "value"   # ERROR: key must be str, value must be int
```

---

## 7. Iterator Consumption

Certain functions return single-use iterators that must be consumed immediately.

### Single-Use Iterators

| Function    | Returns  |
| ----------- | -------- |
| `enumerate` | iterator |
| `zip`       | iterator |
| `reversed`  | iterator |

### Consumption Requirement

Iterators must appear in contexts that consume them eagerly:

| Allowed                     | Consumes iterator |
| --------------------------- | ----------------- |
| `for i, x in enumerate(xs)` | for-loop header   |
| `list(zip(xs, ys))`         | `list()` call     |
| `[x for x in reversed(xs)]` | comprehension     |

### Rejected: Iterator Escape

Iterators cannot be stored, returned, or passed to unknown consumers:

```python
e = enumerate(xs)       # ERROR: iterator assigned
return zip(xs, ys)      # ERROR: iterator returned
foo(reversed(xs))       # ERROR: foo may not consume eagerly
```

### Generator Expressions

Generator expressions are iterators. They may only appear as arguments to known eager consumers:

| Consumer      | Consumes eagerly |
| ------------- | ---------------- |
| `list()`      | yes              |
| `tuple()`     | yes              |
| `set()`       | yes              |
| `dict()`      | yes              |
| `frozenset()` | yes              |
| `sum()`       | yes              |
| `min()`       | yes              |
| `max()`       | yes              |
| `any()`       | yes              |
| `all()`       | yes              |
| `sorted()`    | yes              |
| `str.join()`  | yes              |

---

## 8. Callable Typing

### Function Types

All parameters and return types must be annotated:

```python
def add(a: int, b: int) -> int:
    return a + b
# add: Callable[[int, int], int]
```

### Bound Methods

When a method is accessed on an instance, the receiver is already bound:

```python
class Counter:
    def increment(self, n: int) -> int: ...

c = Counter()
f = c.increment  # f: Callable[[int], int]
```

### Higher-Order Functions

Functions accepting callbacks check argument types:

```python
def apply(f: Callable[[int], str], x: int) -> str:
    return f(x)

apply(str, 42)        # OK: str: Callable[[int], str]
apply(len, 42)        # ERROR: len expects sequence, not int
```
