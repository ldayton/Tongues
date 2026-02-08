# Phase 3: Subset

**Module:** `frontend/subset.py`

Syntactic restrictions for transpilation to statically-typed languages. These are checkable from AST structure alone.

See [08-inference-spec.md](08-inference-spec.md) for type-level invariants requiring inference.

## Errors

Subset violations are reported with the construct and reason:

| Condition          | Diagnostic                                    |
| ------------------ | --------------------------------------------- |
| Banned construct   | error: `{construct} not allowed: {reason}`    |
| Missing annotation | error: `missing type annotation for '{name}'` |
| Invalid decorator  | error: `decorator '{name}' not allowed`       |
| Banned builtin     | error: `builtin '{name}' not allowed`         |
| Invalid import     | error: `import of '{module}' not allowed`     |
| Mutable default    | error: `mutable default argument not allowed` |

---

## 1. Types & Annotations

### Supported Types

| Type       | Syntax                                 | Notes                                          |
| ---------- | -------------------------------------- | ---------------------------------------------- |
| Primitives | `int`, `float`, `str`, `bool`, `bytes` | `bytes` for binary I/O                         |
| Object     | `object`                               | Base type; narrow with `isinstance()` to use   |
| Optional   | `T \| None`, `Optional[T]`             | Nullable types                                 |
| Union      | `A \| B \| C`, `Union[A, B, C]`        | Discriminated by type or `.kind` field         |
| List       | `list[T]`, `List[T]`                   | Bare `list` banned                             |
| Dict       | `dict[K, V]`, `Dict[K, V]`             | Bare `dict` banned                             |
| Set        | `set[T]`, `Set[T]`                     | Bare `set` banned                              |
| Tuple      | `tuple[A, B, C]`, `Tuple[A, B, C]`     | Fixed-length, heterogeneous                    |
| Tuple      | `tuple[T, ...]`                        | Variable-length, homogeneous                   |
| Callable   | `Callable[[A, B], R]`                  | Function types; bound methods include receiver |

### Restrictions

| Restriction                         | Rationale                             |
| ----------------------------------- | ------------------------------------- |
| All annotations required            | Static typing for transpilation       |
| No bare `list`/`dict`/`set`/`tuple` | Element types must be known           |
| No `Any`                            | Use `object` + `isinstance()` instead |
| No `TypeVar`                        | No generics; monomorphic types only   |

```python
# Allowed
x: list[int] = []
d: dict[str, int] = {}
t: tuple[int, str] = (1, "a")
opt: int | None = None
result: int | str = parse(s)           # union
handler: Callable[[int], str] = func   # callable

# object requires isinstance() to use
def show(x: object) -> str:
    if isinstance(x, int):
        return str(x * 2)      # x is int here
    return repr(x)

# Not allowed
x: list = []        # bare collection
x: Any = foo()      # Any type (no checking)
```

---

## 2. Functions

| Allowed                           | Not Allowed          | Rationale                                      |
| --------------------------------- | -------------------- | ---------------------------------------------- |
| `def f(a: int, b: int) -> int`    | `def f(*args)`       | Static arity required for type checking        |
| `def f(a: int = 0) -> int`        | `def f(**kwargs)`    | Static parameter names required                |
| `def f(a: int, /, b: int) -> int` | `lambda x: x`        | All functions must be named for static binding |
| `def f(*, a: int) -> int`         | nested functions     | No closures; flat function namespace           |
| recursive functions               | `global`, `nonlocal` | Two-level scoping only (module + local)        |

- All parameters and return types must be annotated
- No mutable defaults (`[]`, `{}`) — shared-state bugs

---

## 3. Classes

| Allowed                           | Not Allowed                                  | Rationale                                   |
| --------------------------------- | -------------------------------------------- | ------------------------------------------- |
| `class Foo:`                      | `class Foo(A, B):` (multiple inheritance)    | Class hierarchy must be a tree, not DAG     |
| `class Foo(Base):`                | nested classes                               | Flat class namespace                        |
| `@dataclass`                      | `@staticmethod`, `@classmethod`, `@property` | Methods must have explicit receivers        |
| `@dataclass(unsafe_hash=True)`    | arbitrary decorators                         | No metaprogramming                          |
| `__init__`, `__new__`, `__repr__` | other dunder methods                         | No operator overloading or magic interfaces |

Exception multiple inheritance allowed: `class E(Base, Exception)` (marker only).

### Dataclass Restrictions

| Allowed                        | Not Allowed                   | Rationale                                   |
| ------------------------------ | ----------------------------- | ------------------------------------------- |
| `@dataclass`                   | `@dataclass(frozen=True)`     | Immutability can't be guaranteed in targets |
| `@dataclass(eq=True)`          | `@dataclass(order=True)`      | Comparison operators need explicit impl     |
| `@dataclass(unsafe_hash=True)` | `field(default_factory=list)` | No field() options                          |
| `@dataclass(kw_only=True)`     |                               |                                             |
| `x: int = 0`                   |                               |                                             |

---

## 4. Operators & Expressions

| Allowed              | Not Allowed                   | Rationale                                      |
| -------------------- | ----------------------------- | ---------------------------------------------- |
| `x == y`             | `x is y` (except `x is None`) | Identity vs equality; `is` only for None check |
| `x is None`          | `x is not y` (except None)    | Same as above                                  |
| `x in xs`            | `del x`                       | Reassign or let go out of scope                |
| `a < b < c` (chains) |                               |                                                |
| `x += 1`             |                               |                                                |

### Walrus Operator

| Allowed                          | Notes                        |
| -------------------------------- | ---------------------------- |
| `if (x := func()):`              | Scopes to enclosing function |
| `while (line := read()):`        | Useful for read loops        |
| `[y for x in xs if (y := f(x))]` | In comprehension conditions  |

The walrus operator `:=` is allowed and scopes to the enclosing function (not the comprehension).

### F-Strings

| Allowed  | Not Allowed  | Rationale                |
| -------- | ------------ | ------------------------ |
| `f"{x}"` | `f"{x!r}"`   | No conversion specifiers |
| `f"{x}"` | `f"{x:.2f}"` | No format specs          |

### Strings

| Restriction              | Rationale                       |
| ------------------------ | ------------------------------- |
| No surrogate code points | Not valid Unicode scalar values |

---

## 5. Statements & Control Flow

| Allowed                      | Not Allowed         | Rationale                              |
| ---------------------------- | ------------------- | -------------------------------------- |
| `if`/`elif`/`else`           | `with` statement    | Context managers need runtime protocol |
| `for`/`while`                | loop `else` clause  | Unusual semantics; use flag variable   |
| `try`/`except`/`finally`     | `try` `else` clause | Move else code after try block         |
| `match`/`case`               | bare `except:`      | Must specify exception type            |
| `raise`, `break`, `continue` | `async`/`await`     | Requires runtime scheduler             |

---

## 6. Iteration

### Allowed Everywhere
`range(n)`, `range(a, b)`, `range(a, b, step)` — reusable sequence with known length

### Eager Context Only
Must appear in for-loop header or eager consumer. Enforcement requires type checking (see [08-inference-spec.md](08-inference-spec.md#6-iterator-escape)).

| Function                                  | Rationale                   |
| ----------------------------------------- | --------------------------- |
| `enumerate(xs)`, `enumerate(xs, start=n)` | Returns single-use iterator |
| `zip(xs, ys)`                             | Returns single-use iterator |

```python
# Allowed — immediately consumed
for i, x in enumerate(xs): ...
list(zip(xs, ys))

# Not allowed — iterator escapes
e = enumerate(xs)      # assigned
return zip(xs, ys)     # returned
```

### Comprehensions

| Construct          | Example                | Rationale    |
| ------------------ | ---------------------- | ------------ |
| List comprehension | `[x*2 for x in xs]`    | Always eager |
| Set comprehension  | `{x*2 for x in xs}`    | Always eager |
| Dict comprehension | `{k: v for k,v in xs}` | Always eager |

### Generator Expressions
Allowed only as immediate argument to eager consumers. Enforcement requires type checking (see [08-inference-spec.md](08-inference-spec.md#5-generator-expression-consumers)).

| Consumer      | Example                      |
| ------------- | ---------------------------- |
| `tuple()`     | `tuple(x for x in iter)`     |
| `list()`      | `list(x for x in iter)`      |
| `set()`       | `set(x for x in iter)`       |
| `dict()`      | `dict((k,v) for ...)`        |
| `frozenset()` | `frozenset(x for x in iter)` |
| `any()`       | `any(p(x) for x in iter)`    |
| `all()`       | `all(p(x) for x in iter)`    |
| `sum()`       | `sum(x for x in iter)`       |
| `min()`       | `min(x for x in iter)`       |
| `max()`       | `max(x for x in iter)`       |
| `sorted()`    | `sorted(x for x in iter)`    |
| `str.join()`  | `",".join(s for s in iter)`  |

Not allowed: `g = (x for x in iter)`, `return (x for x in iter)`, `foo(x for x in iter)`.

### Generator Functions

| Allowed                              | Not Allowed                  | Rationale                    |
| ------------------------------------ | ---------------------------- | ---------------------------- |
| `for x in items: yield x`            | `while True: yield next()`   | Must be structural recursion |
| `yield from traverse(node.children)` | `yield from infinite_stream` | Bounded by tree depth        |

---

## 7. Builtins

### Allowed Functions

| Category    | Functions                                                    |
| ----------- | ------------------------------------------------------------ |
| Math        | `abs`, `min`, `max`, `sum`, `round`, `divmod`, `pow`         |
| Conversion  | `int`, `float`, `str`, `bool`, `bytes`, `chr`, `ord`         |
| Collections | `list`, `dict`, `set`, `tuple`, `frozenset`, `len`, `sorted` |
| Type check  | `isinstance`                                                 |
| Iteration   | `range`, `enumerate`, `zip`                                  |
| Formatting  | `repr`, `ascii`, `bin`, `hex`, `oct`                         |
| Boolean     | `all`, `any`                                                 |
| Other       | `slice`, `super`, `object`, `print`                          |

### Allowed Methods

| Type   | Methods                                                                                                                                                                                         |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `str`  | `join`, `split`, `strip`, `lstrip`, `rstrip`, `lower`, `upper`, `startswith`, `endswith`, `replace`, `find`, `rfind`, `count`, `isalnum`, `isalpha`, `isdigit`, `isspace`, `isupper`, `islower` |
| `list` | `append`, `extend`, `pop`, `insert`, `remove`, `copy`, `clear`, `index`, `count`, `reverse`, `sort`                                                                                             |
| `dict` | `get`, `keys`, `values`, `items`, `pop`, `setdefault`, `update`, `clear`, `copy`                                                                                                                |
| `set`  | `add`, `remove`, `discard`, `pop`, `clear`, `copy`, `union`, `intersection`, `difference`, `issubset`, `issuperset`                                                                             |

### Banned Functions

| Function                                   | Rationale                                       |
| ------------------------------------------ | ----------------------------------------------- |
| `type`                                     | Runtime type inspection; use `isinstance`       |
| `getattr`, `setattr`, `hasattr`, `delattr` | Dynamic attribute access; fields must be static |
| `vars`, `dir`, `globals`, `locals`         | Runtime introspection                           |
| `id`                                       | Object identity is implementation-dependent     |
| `callable`                                 | Runtime type inspection                         |
| `eval`, `exec`, `compile`, `__import__`    | Dynamic code execution                          |
| `iter`, `next`                             | Manual iteration; use for-loop                  |
| `map`, `filter`                            | Require callable argument; use comprehension    |
| `open`, `input`                            | I/O restricted to sys.stdin/stdout/stderr       |
| `issubclass`                               | Runtime type hierarchy inspection               |
| `hash`                                     | Use `@dataclass(unsafe_hash=True)` instead      |
| `format`                                   | Use f-strings instead                           |
| `memoryview`                               | Low-level memory access                         |
| `complex`                                  | Adds type complexity; rarely needed             |
| `aiter`, `anext`                           | Async iteration not supported                   |
| `reversed`                                 | Returns iterator; use slice `xs[::-1]` instead  |
| `breakpoint`                               | Debugger invocation                             |
| `help`                                     | Interactive help system                         |
| `exit`, `quit`                             | Use `sys.exit()` instead                        |

### Restricted Arguments

| Function    | Allowed                                                     | Not Allowed     | Rationale                        |
| ----------- | ----------------------------------------------------------- | --------------- | -------------------------------- |
| `min`/`max` | `min(a, b)`, `min(xs)`                                      | `key=`          | Key requires callable            |
|             |                                                             | `default=`      | Changes return type to Optional  |
| `sorted`    | `sorted(xs)`, `sorted(xs, reverse=True)`                    | `key=`          | Key requires callable            |
| `print`     | `print(x)`, `print(x, end="")`, `print(x, file=sys.stderr)` | multiple values | Simplifies output formatting     |
|             |                                                             | `sep=`          | Use string concatenation instead |

---

## 8. I/O

| Allowed                       | Rationale              |
| ----------------------------- | ---------------------- |
| `print(x)`                    | Standard output        |
| `print(x, end="")`            | Suppress newline       |
| `print(x, file=sys.stderr)`   | Error output           |
| `sys.stdin.readline()`        | Line input             |
| `sys.stdin.read()`            | Full input             |
| `sys.stdin.buffer.read()`     | Binary input           |
| `sys.stdin.buffer.read(n)`    | Binary input (n bytes) |
| `sys.stdin.buffer.readline()` | Binary line input      |
| `sys.stdout.buffer.write(b)`  | Binary output          |
| `sys.stderr.buffer.write(b)`  | Binary error output    |
| `sys.argv`                    | Command-line arguments |
| `os.getenv(name)`             | Environment variables  |
| `os.getenv(name, default)`    | With default           |

---

## 9. Imports

Every module in the program's import graph must be subset-compliant. The transpiler resolves all imports and verifies that every imported module is either a project file or an allowed stdlib module.

### Syntactic rules (subset checker)

| Allowed                | Not Allowed       | Rationale                      |
| ---------------------- | ----------------- | ------------------------------ |
| `from X import Y`      | `from X import *` | Star imports obscure bindings  |
| `from X import Y as Z` | `import X as Y`   | Module aliases obscure origins |
| `from . import module` |                   |                                |
| `import sys`           |                   | stdin/stdout/stderr/argv       |
| `import os`            |                   | `os.getenv()`                  |
| `import re`            |                   | Regular expressions            |

Bare `import` is restricted to `sys`, `os`, and `re`. These modules have built-in transpiler support and must be used via their module namespace (e.g., `sys.argv`, `os.getenv()`, `re.match()`), so `from sys/os/re import ...` is not allowed.

All other `from` imports are syntactically valid. Whether they resolve is a semantic question handled by import resolution.

### Semantic rules (import resolution)

Every import must resolve to one of:

| Source                | Examples                                                                                                                               |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Project file          | `from .module import X`, `from mypackage import Y`                                                                                     |
| Allowed stdlib module | `from typing import ...`, `from dataclasses import dataclass`, `from collections.abc import ...`, `from __future__ import annotations` |
| Allowed bare import   | `import sys`, `import os`, `import re`                                                                                                 |

Imports that do not resolve — other stdlib modules, external packages — are errors at this phase, not at subset checking.
