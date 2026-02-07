# Phase 8: Inference

**Module:** `frontend/inference.py` (entry point), `frontend/type_inference.py` (helpers)

Bidirectional type inference with flow-sensitive narrowing. Computes types for all expressions, infers local variable types from assignments, and enforces type safety constraints that require knowing expression types.

See [03-subset-spec.md](03-subset-spec.md) for syntactic restrictions checkable on AST structure alone.

## Inputs

- **AST**: dict-based AST from Phase 2
- **NameTable**: from Phase 4 (for name resolution)
- **SigTable**: from Phase 5 (for function/method signatures)
- **FieldTable**: from Phase 6 (for field types)
- **SubtypeRel**: from Phase 7 (for subtype checks, hierarchy_root)

## 1. Type Language

### Primitive Types

| Type    | Values                       | IR Representation |
| ------- | ---------------------------- | ----------------- |
| `int`   | Arbitrary-precision integers | `INT`             |
| `float` | IEEE 754 double-precision    | `FLOAT`           |
| `str`   | Unicode strings              | `STRING`          |
| `bytes` | Byte sequences               | `BYTES`           |
| `bool`  | `True`, `False`              | `BOOL`            |

### Compound Types

| Form             | Meaning                            | IR Representation            |
| ---------------- | ---------------------------------- | ---------------------------- |
| `list[T]`        | Homogeneous sequence of `T`        | `Slice(T)`                   |
| `dict[K, V]`     | Map from `K` to `V`                | `Map(K, V)`                  |
| `set[T]`         | Unordered collection of unique `T` | `Set(T)`                     |
| `tuple[A, B, C]` | Fixed-length heterogeneous product | `Tuple((A, B, C))`           |
| `tuple[T, ...]`  | Variable-length homogeneous tuple  | `Tuple((T,), variadic=True)` |

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

p.parse  # Callable[[str], Node], not Callable[[Parser, str], Node]
```

### Subtyping

Subtyping is structural for unions, nominal for classes:

| Relation      | Holds when              |
| ------------- | ----------------------- |
| `A <: A \| B` | Always (union widening) |
| `C <: P`      | `C` inherits from `P`   |
| `T <: T`      | Always (reflexivity)    |

Collections are invariant: `list[Dog]` is not a subtype of `list[Animal]`. This enables efficient code generation—backends don't need runtime type checks or covariant array wrappers.

---

## 2. Typing Judgments

The type system uses bidirectional typing: expressions either **synthesize** a type or are **checked** against an expected type.

### Synthesis (Γ ⊢ e ⇒ T)

The expression `e` produces type `T` without external guidance.

| Expression      | Synthesized Type                 |
| --------------- | -------------------------------- |
| `42`            | `int`                            |
| `"hello"`       | `str`                            |
| `True`          | `bool`                           |
| `x` (variable)  | lookup `x` in Γ                  |
| `obj.field`     | field type from class definition |
| `f(args...)`    | return type of `f`               |
| `xs[i]`         | element type of `xs` ²           |
| `[a, b, c]`     | `list[T]` where T = common type  |
| `x + y` (arith) | `int` or `float` ¹               |
| `x == y`        | `bool`                           |
| `x and y`       | `bool`                           |

¹ Bool operands in arithmetic coerce to int: `True + 1` yields `int`. Lowering emits `Cast(BOOL, INT)`.

² For strings, `s[i]` yields `str` (a single Unicode code point), not a byte. Lowering emits `CharAt`.

### Large Integer Detection

Integer literals exceeding JavaScript's safe integer range (2⁵³-1) are annotated with `is_large_int: True`. This enables backends to emit BigInt literals or equivalent representations:

```python
x = 9007199254740992  # is_large_int=True, exceeds 2^53-1
y = 1000              # is_large_int=False
```

Expressions involving large integers propagate the annotation when the result may exceed safe range.

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

## 3. Local Variable Inference

Local variables without annotations get types from their first assignment. Types propagate through assignment chains.

### Inference from Assignment

| Pattern         | Inferred Type         |
| --------------- | --------------------- |
| `x = 42`        | `int`                 |
| `x = "hello"`   | `str`                 |
| `x = True`      | `bool`                |
| `x = [1, 2, 3]` | `list[int]`           |
| `x = {}`        | `dict[str, any]`      |
| `x = func()`    | return type of `func` |
| `x = obj.field` | field type            |
| `x = other_var` | type of `other_var`   |
| `x = param`     | type of parameter     |

### Annotated Assignments

Annotated assignments use the declared type:

```python
x: int = 0           # x: int
xs: list[str] = []   # xs: list[str], not list[any]
```

### Branch Unification

When a variable is assigned in both branches of an if/else, its type after the merge point is the common type:

```python
if cond:
    x = "hello"
else:
    x = "world"
# x: str here (both branches assign str)
```

If branches assign incompatible types, the variable's type is the least upper bound or an error is reported.

### Coercion Result Types

When coercions change the result type, expressions are annotated with `output_type` to track the actual emitted type distinct from the source type:

| Expression      | Source Type | Output Type | Reason                            |
| --------------- | ----------- | ----------- | --------------------------------- |
| `min(True, 1)`  | `bool`      | `int`       | Mixed bool/int yields int         |
| `max(False, 0)` | `bool`      | `int`       | Mixed bool/int yields int         |
| `True & False`  | `bool`      | `bool`      | Bool-only bitwise preserves bool  |
| `True & 1`      | `bool`      | `int`       | Mixed bool/int bitwise yields int |
| `-True`         | `bool`      | `int`       | Unary minus on bool yields int    |

Backends use `output_type` to determine whether the result needs bool-preserving emission (e.g., `!= 0` to convert back to bool).

---

## 4. Type Narrowing

Type narrowing refines a variable's type based on control flow. After a guard, the variable has a more precise type in the guarded branch.

### Narrowing Guards

| Guard                     | Narrows `x` to         | In Branch |
| ------------------------- | ---------------------- | --------- |
| `isinstance(x, T)`        | `T`                    | then      |
| `not isinstance(x, T)`    | remaining types        | then      |
| `x is None`               | `None`                 | then      |
| `x is not None`           | non-None part of union | then      |
| `if x:` (optional)        | non-None part          | then      |
| `x.kind == "foo"`         | variant with that kind | then      |
| `assert isinstance(x, T)` | `T`                    | after     |
| `assert x is not None`    | non-None part          | after     |

### Early Exit Narrowing

Early exits refine types in subsequent code:

```python
def process(x: int | None) -> int:
    if x is None:
        return 0
    # x: int here (None case returned)
    return x + 1
```

```python
def process(a: Animal) -> str:
    if not isinstance(a, Dog):
        return "not a dog"
    # a: Dog here
    return a.bark()
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

### Kind Alias Narrowing

Kind checks work through intermediate variables:

```python
k = node.kind
if k == "add":
    # node: Add here (k traced back to node.kind)
    ...
```

### Attribute Path Narrowing

Nested field access narrows based on kind:

```python
if stmt.body.kind == "block":
    # stmt.body: Block here
    ...
```

### Narrowing Scope

Narrowing applies only within the guarded scope. After branches merge, narrowing is lost:

```python
def f(x: int | None, flag: bool) -> int:
    if flag:
        if x is not None:
            y: int = x  # OK: x is int here
    return x + 1  # ERROR: x may be None (narrowing lost at merge)
```

---

## 5. Truthiness

Python's `if x:` has type-dependent semantics. Only types with unambiguous truthiness are allowed.

### Allowed

| Type         | `if x:` means   | Rationale   |
| ------------ | --------------- | ----------- |
| `bool`       | boolean value   | Direct test |
| `T \| None`  | `x is not None` | Null check  |
| `list[T]`    | `len(x) > 0`    | Non-empty   |
| `dict[K, V]` | `len(x) > 0`    | Non-empty   |
| `set[T]`     | `len(x) > 0`    | Non-empty   |
| `str`        | `len(x) > 0`    | Non-empty   |

### Rejected

| Type              | Rationale                          |
| ----------------- | ---------------------------------- |
| `int`             | Zero is valid data, not a sentinel |
| `float`           | Zero is valid data, not a sentinel |
| `list[T] \| None` | Ambiguous: non-None or non-empty?  |
| `dict[K,V]\|None` | Ambiguous: non-None or non-empty?  |
| `set[T] \| None`  | Ambiguous: non-None or non-empty?  |
| `str \| None`     | Ambiguous: non-None or non-empty?  |

For ambiguous cases, use explicit checks:

```python
if x is not None:      # tests None
    if x:              # tests empty (x: list[int] here)
        ...
```

---

## 6. Optional Type Representation

Optional types use sentinel values for efficient code generation. This avoids pointer indirection in target languages.

| Python Type    | IR Type         | Sentinel | Rationale                      |
| -------------- | --------------- | -------- | ------------------------------ |
| `str \| None`  | `STRING`        | `""`     | Empty string represents None   |
| `int \| None`  | `Optional(INT)` | varies   | -1 or pointer depending on use |
| `Node \| None` | `InterfaceRef`  | nil      | Interfaces are nilable         |

### Sentinel Detection

When a variable is annotated `T | None` and initialized to `None`, the compiler tracks this for sentinel-based code generation:

```python
result: str | None = None   # uses empty string sentinel
if found:
    result = value
return result               # returns "" if not found
```

### Interface Field Widening

When `None` flows into a field typed as `InterfaceRef` (without explicit `| None`), inference widens the field type to `Optional(InterfaceRef)`:

```python
@dataclass
class Parser:
    current: Token      # declared non-optional

p = Parser(current=None)  # None flows in → widen to Optional(Token)
```

This accommodates Python code that passes `None` without explicit optional annotation. Backends emit nullable types for widened fields.

### Any-Typed Assignment Specialization

When assigning a concrete literal to an `any`-typed variable, inference specializes to the concrete type to avoid boxing overhead:

```python
def process(x: object) -> None:
    y = 42          # y: int (not object), avoids boxing
    z = "hello"     # z: str (not object)
```

The specialization applies only to literals; variable-to-variable assignments preserve the declared type.

---

## 7. Class Hierarchy and Polymorphism

When classes form an inheritance hierarchy, the compiler detects a **hierarchy root** and generates interface-based polymorphism.

### Hierarchy Detection

| Classification   | Criteria                        | IR Type                    |
| ---------------- | ------------------------------- | -------------------------- |
| Hierarchy root   | No base, used as base by others | `InterfaceRef(name)`       |
| Node subclass    | Inherits from hierarchy root    | `Pointer(StructRef(name))` |
| Standalone class | No inheritance relationship     | `Pointer(StructRef(name))` |

### Polymorphic Variables

Variables typed as the hierarchy root or a union of subclasses use the interface type:

```python
class Node: ...
class Expr(Node): ...
class Stmt(Node): ...

def process(n: Node) -> None:      # n: InterfaceRef("Node")
    ...

def eval(e: Expr | Stmt) -> int:   # e: InterfaceRef("Node")
    ...
```

### Concrete vs Interface Context

Parameters and return types use concrete struct pointers; local variables unified across branches use the interface:

```python
def make_expr() -> Expr:           # returns Pointer(StructRef("Expr"))
    return Expr()

e: Node = make_expr()              # e: InterfaceRef("Node")
```

### Pointer Nullability

Struct references distinguish nullable from non-nullable:

| Python Type   | IR Type                      | Nullable |
| ------------- | ---------------------------- | -------- |
| `Foo`         | `Pointer(StructRef("Foo"))`  | no       |
| `Foo \| None` | `Optional(StructRef("Foo"))` | yes      |
| `Node` (root) | `InterfaceRef("Node")`       | yes ³    |

³ Interface references are implicitly nullable (can hold nil for any variant).

Fields and parameters annotated `Foo` (without `| None`) are non-nullable. Assignment of `None` to a non-nullable field is a type error. Backends emit `T` vs `T?` / `T | null` based on this distinction.

### Typed Nil vs Nil Interface

Some languages (notably Go) distinguish a nil interface from an interface holding a typed nil pointer. Inference tracks `may_be_typed_nil` for expressions that could hold a concrete type's nil:

```python
def get_node() -> Node | None:
    if condition:
        return None        # nil interface
    expr: Expr | None = None
    return expr            # typed nil (Expr's nil wrapped in Node interface)
```

Backends needing this distinction (Go) emit reflection-based nil checks for `may_be_typed_nil` expressions.

---

## 8. Tuple Types and Unpacking

Tuples are fixed-length products with per-position types.

### Fixed-Length Tuples

```python
t: tuple[int, str, bool] = (1, "a", True)
```

Each position is checked independently.

### Variable-Length Tuples

```python
t: tuple[int, ...] = (1, 2, 3, 4)
```

All elements must have the same type. Indexing returns that type.

### Tuple Unpacking

Unpacking requires the right-hand side to have known tuple type:

| Allowed                   | Type of RHS               |
| ------------------------- | ------------------------- |
| `a, b = func()`           | `func` returns tuple      |
| `a, b = (1, "x")`         | literal tuple             |
| `if t := f(): a, b = t`   | guarded (proves non-None) |
| `for a, b in items:`      | items is `list[tuple]`    |
| `for i, x in enumerate()` | enumerate yields tuples   |
| `for x, y in zip(a, b)`   | zip yields tuples         |
| `for k, v in d.items()`   | items yields tuples       |

### Rejected

```python
t: tuple[int, str] | None = func()
a, b = t        # ERROR: may be None

def get_pair() -> tuple[int, str]: ...
a, b, c = get_pair()  # ERROR: wrong count
```

---

## 9. Collection Typing

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

### Dict Key Equivalence

Python treats `True == 1` and `False == 0` as equivalent dict keys:

```python
d: dict[int, str] = {1: "one"}
d[True]   # returns "one" (True == 1)
d[False]  # KeyError (no key 0)
```

For `dict[int, V]`, bool keys coerce to int. Lowering emits `CoerceMapKey(key, target_key_type)` when key type differs from declared map key type.

### Invariance

Collections are invariant. This code is rejected:

```python
dogs: list[Dog] = [Dog()]
animals: list[Animal] = dogs  # ERROR: list[Dog] is not list[Animal]
```

This prevents unsound mutations:
```python
animals.append(Cat())  # would corrupt dogs list
```

### Comparison Semantics

Python treats `True == 1` and `False == 0` as equal. Some target languages use strict equality where `true !== 1`. Comparisons between bool and int are annotated with `use_loose_equality: True`:

```python
x: bool = True
if x == 1:      # use_loose_equality=True
    ...
if x == True:   # use_loose_equality=False (same types)
    ...
```

Backends emit loose equality operators (`==` in PHP) or explicit coercion based on this annotation.

---

## 10. Iterator Consumption

Certain functions return single-use iterators that must be consumed immediately.

### Single-Use Iterators

| Function    | Returns  |
| ----------- | -------- |
| `enumerate` | iterator |
| `zip`       | iterator |

Note: `range` is reusable and can be assigned or returned.

### Consumption Requirement

Iterators must appear in contexts that consume them eagerly:

| Allowed                     | Consumer        |
| --------------------------- | --------------- |
| `for i, x in enumerate(xs)` | for-loop header |
| `list(zip(xs, ys))`         | `list()` call   |
| `dict(enumerate(xs))`       | `dict()` call   |
| `[x for x in zip(a, b)]`    | comprehension   |

### Rejected: Iterator Escape

```python
e = enumerate(xs)       # ERROR: iterator assigned
return zip(xs, ys)      # ERROR: iterator returned
foo(enumerate(xs))      # ERROR: foo may not consume eagerly
```

### Generator Expressions

Generator expressions are single-use. They may only appear as arguments to known eager consumers:

| Consumer      | Allowed |
| ------------- | ------- |
| `list()`      | yes     |
| `tuple()`     | yes     |
| `set()`       | yes     |
| `dict()`      | yes     |
| `frozenset()` | yes     |
| `sum()`       | yes     |
| `min()`       | yes     |
| `max()`       | yes     |
| `any()`       | yes     |
| `all()`       | yes     |
| `sorted()`    | yes     |
| `str.join()`  | yes     |

Rejected:

```python
g = (x * 2 for x in xs)     # ERROR: assigned
return (x for x in xs)       # ERROR: returned
foo(x for x in xs)           # ERROR: unknown consumer
```

---

## 11. Callable Typing

### Function Types

All parameters and return types must be annotated:

```python
def add(a: int, b: int) -> int:
    return a + b
# add: Callable[[int, int], int]
```

### Self and Method Receivers

Inside a method, `self` has type `Pointer(StructRef(class_name))`:

```python
class Counter:
    value: int
    def increment(self, n: int) -> int:
        # self: *Counter
        self.value += n
        return self.value
```

### Bound Methods

When a method is accessed on an instance, the receiver is already bound:

```python
c = Counter()
f = c.increment  # f: Callable[[int], int]
```

### Higher-Order Functions

Functions accepting callbacks check argument and return types:

```python
def apply(f: Callable[[int], str], x: int) -> str:
    return f(x)

apply(str, 42)        # OK: str: Callable[[int], str]
apply(len, 42)        # ERROR: len expects sequence, not int
```

---

## 12. Object Type

The `object` type is the top type—any value can be assigned to it. To use a value typed as `object`, narrow with `isinstance`:

```python
def show(x: object) -> str:
    if isinstance(x, int):
        return str(x * 2)      # x: int here
    if isinstance(x, str):
        return x.upper()       # x: str here
    return repr(x)
```

Without narrowing, only operations valid for all types are allowed (essentially none).

---

## Errors

| Condition                   | Diagnostic                                                 |
| --------------------------- | ---------------------------------------------------------- |
| Type mismatch in assignment | error: `cannot assign str to int`                          |
| Type mismatch in return     | error: `expected int, got str`                             |
| Type mismatch in call       | error: `argument 1: expected int, got str`                 |
| Unnarrowed optional access  | error: `value may be None`                                 |
| Unnarrowed union access     | error: `value may be str` (when int expected)              |
| Ambiguous truthiness        | error: `ambiguous truthiness for list[int] \| None`        |
| Iterator escape             | error: `enumerate result must be consumed immediately`     |
| Generator escape            | error: `generator expression must be consumed immediately` |
| Tuple unpack count mismatch | error: `cannot unpack tuple[int, str] into 3 values`       |
| Tuple index out of bounds   | error: `tuple index 2 out of range for tuple[int, str]`    |
| Unknown type                | error: `unknown type 'Foo'`                                |

---

## Output: TypedAST

The TypedAST is the input dict-AST with type annotations added:

| Node type           | Added field      | Contents                               |
| ------------------- | ---------------- | -------------------------------------- |
| All expressions     | `_type`          | IR type (e.g., `INT`, `Slice(STRING)`) |
| Variable references | `_resolved`      | Declaration kind from NameTable        |
| Narrowed uses       | `_narrowed_type` | More precise type after guards         |

Structure otherwise identical to Phase 2 output. Subsequent phases can traverse the same AST and read type information from `_type` fields.

## Postconditions

- All expressions annotated with synthesized or checked types
- All local variables have inferred types
- Type narrowing applied per control flow
- Optional types resolved to sentinel or interface representations
- Iterator/generator consumption verified
- Type errors reported with line/column

---

## Prior Art

- [Bidirectional Typing](https://arxiv.org/abs/1908.05839) — Dunfield & Krishnaswami
- [Local Type Inference](https://www.cis.upenn.edu/~bcpierce/papers/lti-toplas.pdf) — Pierce & Turner
- [TypeScript Narrowing](https://www.typescriptlang.org/docs/handbook/2/narrowing.html)
- [Pyright Type Inference](https://github.com/microsoft/pyright/blob/main/docs/type-inference.md)
