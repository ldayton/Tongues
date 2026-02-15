# Phase 3: Subset

**Module:** `frontend/subset.py`

Syntactic gate for transpilation. Walks the AST and rejects Python constructs that have no Taytsh equivalent. Everything that passes this phase is guaranteed to be translatable to Taytsh IR. Restrictions are checkable from AST structure alone — no type information needed.

See phase 8 (inference) for type-level invariants requiring inference.

## Type Annotations

### Supported Types

| Type       | Syntax                                 | Taytsh equivalent                         |
| ---------- | -------------------------------------- | ----------------------------------------- |
| Primitives | `int`, `float`, `str`, `bool`, `bytes` | `int`, `float`, `string`, `bool`, `bytes` |
| Object     | `object`                               | narrow with `isinstance` to use           |
| Optional   | `T \| None`, `Optional[T]`             | `T?`                                      |
| Union      | `A \| B \| C`, `Union[A, B, C]`        | `A \| B \| C`                             |
| List       | `list[T]`, `List[T]`                   | `list[T]`                                 |
| Dict       | `dict[K, V]`, `Dict[K, V]`             | `map[K, V]`                               |
| Set        | `set[T]`, `Set[T]`                     | `set[T]`                                  |
| Frozenset  | `frozenset[T]`                         | `set[T]`                                  |
| Tuple      | `tuple[A, B, C]`, `Tuple[A, B, C]`     | `(A, B, C)`                               |
| Tuple      | `tuple[T, ...]`                        | variable-length                           |
| Callable   | `Callable[[A, B], R]`                  | `fn[A, B, R]`                             |

### Restrictions

| Restriction                                     | Rationale                           |
| ----------------------------------------------- | ----------------------------------- |
| All annotations required                        | Taytsh requires static types        |
| No bare `list`/`dict`/`set`/`tuple`/`frozenset` | Element types must be known         |
| No `Any`                                        | Use `object` + `isinstance` instead |
| No `TypeVar`                                    | No generics; monomorphic types only |

## Functions

| Allowed                           | Not Allowed          | Rationale                               |
| --------------------------------- | -------------------- | --------------------------------------- |
| `def f(a: int, b: int) -> int`    | `def f(*args)`       | Taytsh requires static arity            |
| `def f(a: int = 0) -> int`        | `def f(**kwargs)`    | Taytsh requires static parameter names  |
|                                   | `f(*xs)`, `f(**d)`   | Static arity required at call sites     |
| `def f(a: int, /, b: int) -> int` | `lambda x: x`        | All functions must be named             |
| `def f(*, a: int) -> int`         | nested functions     | No closures; flat function namespace    |
| recursive functions               | `global`, `nonlocal` | Two-level scoping only (module + local) |

All parameters and return types must be annotated. No mutable defaults (`[]`, `{}`).

## Classes and Enumerations

### Classes

| Allowed                           | Not Allowed                                  | Rationale                     |
| --------------------------------- | -------------------------------------------- | ----------------------------- |
| `class Foo:`                      | `class Foo(A, B):` (multiple inheritance)    | Hierarchy must be a tree      |
| `class Foo(Base):`                | nested classes                               | Flat struct namespace         |
| `@dataclass`                      | `@staticmethod`, `@classmethod`, `@property` | Methods require explicit self |
| `@dataclass(unsafe_hash=True)`    | arbitrary decorators                         | No metaprogramming            |
| `__init__`, `__new__`, `__repr__` | other dunder methods                         | No operator overloading       |

Exception multiple inheritance allowed: `class E(Base, Exception)` (marker only).

### Dataclass Restrictions

| Allowed                        | Not Allowed                   |
| ------------------------------ | ----------------------------- |
| `@dataclass`                   | `@dataclass(frozen=True)`     |
| `@dataclass(eq=True)`          | `@dataclass(order=True)`      |
| `@dataclass(unsafe_hash=True)` | `field(default_factory=list)` |
| `@dataclass(kw_only=True)`     |                               |
| `x: int = 0`                   |                               |

### Enumerations

Classes whose fields are all class-level string constants and that serve as hierarchy roots are recognized as Taytsh enum candidates during lowering. No `enum.Enum` import is needed — enumeration semantics are inferred from structure.

## Operators and Expressions

| Allowed                    | Not Allowed                       | Rationale                            |
| -------------------------- | --------------------------------- | ------------------------------------ |
| `x == y`                   | `x is y` (neither side a literal) | `is` only when one side is a literal |
| `x is None`/`True`/`False` | `x is not y` (neither a literal)  | Same as above                        |
| `x in xs`, `x not in xs`   | `del x`                           | Reassign or let go out of scope      |
| `a < b < c` (chains)       |                                   |                                      |
| `x += 1`                   |                                   |                                      |

### Walrus Operator

| Allowed                          | Notes                        |
| -------------------------------- | ---------------------------- |
| `if (x := func()):`              | Scopes to enclosing function |
| `while (line := read()):`        | Useful for read loops        |
| `[y for x in xs if (y := f(x))]` | In comprehension conditions  |

### F-Strings

| Allowed  | Not Allowed  | Rationale                |
| -------- | ------------ | ------------------------ |
| `f"{x}"` | `f"{x!r}"`   | No conversion specifiers |
| `f"{x}"` | `f"{x:.2f}"` | No format specs          |

### Strings

| Restriction              | Rationale                       |
| ------------------------ | ------------------------------- |
| No surrogate code points | Not valid Unicode scalar values |

## Statements and Control Flow

| Allowed                  | Not Allowed         | Rationale                              |
| ------------------------ | ------------------- | -------------------------------------- |
| `if`/`elif`/`else`       | `with` statement    | Context managers need runtime protocol |
| `for`/`while`            | loop `else` clause  | Unusual semantics; use flag variable   |
| `try`/`except`/`finally` | `try` `else` clause | Move else code after try block         |
| `match`/`case`           | bare `except:`      | Must specify exception type            |
| `raise`, `raise from`    | `async`/`await`     | Requires runtime scheduler             |
| `break`, `continue`      |                     |                                        |
| `assert`, `pass`         |                     |                                        |

`assert expr` evaluates the expression and throws `AssertError` if falsy. The optional message form `assert expr, msg` is allowed.

## Iteration and Comprehensions

### Range

`range(n)`, `range(a, b)`, `range(a, b, step)` — maps to Taytsh `for i in range(...)`.

### Iterators

| Function                                  | Restriction                                      |
| ----------------------------------------- | ------------------------------------------------ |
| `enumerate(xs)`, `enumerate(xs, start=n)` | Must appear in for-loop header or eager consumer |
| `zip(xs, ys)`                             | Must appear in for-loop header or eager consumer |

Iterators must not escape: no assignment, no return, no passing to non-consumer functions. Enforcement requires type information (see phase 8).

### Comprehensions

| Construct          | Example                |
| ------------------ | ---------------------- |
| List comprehension | `[x*2 for x in xs]`    |
| Set comprehension  | `{x*2 for x in xs}`    |
| Dict comprehension | `{k: v for k,v in xs}` |

All comprehensions are eager. Lowered to for loops with provenance annotations.

### Generator Expressions

Allowed only as immediate argument to eager consumers:

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

Not allowed: `g = (x for x in iter)`, `return (x for x in iter)`.

### Generator Functions

| Allowed                              | Not Allowed                  |
| ------------------------------------ | ---------------------------- |
| `for x in items: yield x`            | `while True: yield next()`   |
| `yield from traverse(node.children)` | `yield from infinite_stream` |

## Built-in Functions and Methods

### Allowed Functions

| Category    | Functions                                                    |
| ----------- | ------------------------------------------------------------ |
| Math        | `abs`, `min`, `max`, `sum`, `round`, `divmod`, `pow`         |
| Conversion  | `int`, `float`, `str`, `bool`, `bytes`, `chr`, `ord`         |
| Collections | `list`, `dict`, `set`, `tuple`, `frozenset`, `len`, `sorted` |
| Type check  | `isinstance`                                                 |
| Iteration   | `range`, `enumerate`, `zip`, `reversed`                      |
| Formatting  | `repr`, `ascii`, `bin`, `hex`, `oct`                         |
| Boolean     | `all`, `any`                                                 |
| Other       | `slice`, `super`, `object`, `print`                          |

### Allowed Methods

Enforcement requires type information; verified during type inference (phase 8).

| Type   | Methods                                                                                                                                                                                         |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `str`  | `join`, `split`, `strip`, `lstrip`, `rstrip`, `lower`, `upper`, `startswith`, `endswith`, `replace`, `find`, `rfind`, `count`, `isalnum`, `isalpha`, `isdigit`, `isspace`, `isupper`, `islower` |
| `list` | `append`, `extend`, `pop`, `insert`, `remove`, `copy`, `clear`, `index`, `count`, `reverse`, `sort`                                                                                             |
| `dict` | `get`, `keys`, `values`, `items`, `pop`, `setdefault`, `update`, `clear`, `copy`                                                                                                                |
| `set`  | `add`, `remove`, `discard`, `pop`, `clear`, `copy`, `union`, `intersection`, `difference`, `issubset`, `issuperset`                                                                             |

### Banned Functions

| Function                                   | Rationale                           |
| ------------------------------------------ | ----------------------------------- |
| `type`                                     | Runtime type inspection             |
| `getattr`, `setattr`, `hasattr`, `delattr` | Dynamic attribute access            |
| `vars`, `dir`, `globals`, `locals`         | Runtime introspection               |
| `id`                                       | Object identity is target-dependent |
| `callable`                                 | Runtime type inspection             |
| `eval`, `exec`, `compile`, `__import__`    | Dynamic code execution              |
| `iter`, `next`                             | Manual iteration; use for-loop      |
| `map`, `filter`                            | Use comprehension                   |
| `open`, `input`                            | I/O restricted to sys.stdin/stdout  |
| `issubclass`                               | Runtime hierarchy inspection        |
| `hash`                                     | Use `@dataclass(unsafe_hash=True)`  |
| `format`                                   | Use f-strings                       |
| `memoryview`, `complex`                    | Unnecessary type complexity         |
| `aiter`, `anext`, `breakpoint`, `help`     | Runtime/interactive features        |
| `exit`, `quit`                             | Use `sys.exit()` instead            |

### Restricted Arguments

| Function    | Allowed                                                     | Not Allowed             |
| ----------- | ----------------------------------------------------------- | ----------------------- |
| `min`/`max` | `min(a, b)`, `min(xs)`                                      | `key=`, `default=`      |
| `sorted`    | `sorted(xs)`, `sorted(xs, reverse=True)`                    | `key=`                  |
| `print`     | `print(x)`, `print(x, end="")`, `print(x, file=sys.stderr)` | multiple values, `sep=` |

## I/O

| Allowed                      | Taytsh equivalent           |
| ---------------------------- | --------------------------- |
| `print(x)`                   | `WritelnOut(ToString(x))`   |
| `print(x, end="")`           | `WriteOut(ToString(x))`     |
| `print(x, file=sys.stderr)`  | `WritelnErr(ToString(x))`   |
| `sys.stdin.readline()`       | `ReadLine()`                |
| `sys.stdin.read()`           | `ReadAll()`                 |
| `sys.stdin.buffer.read()`    | `ReadBytes()`               |
| `sys.stdin.buffer.read(n)`   | `ReadBytesN(n)`             |
| `sys.stdout.buffer.write(b)` | `WriteOut(b)`               |
| `sys.stderr.buffer.write(b)` | `WriteErr(b)`               |
| `sys.argv`                   | `Args()`                    |
| `sys.exit(code)`             | `Exit(code)`                |
| `os.getenv(name)`            | `GetEnv(name)`              |
| `os.getenv(name, default)`   | `GetEnv(name)` with default |

## Imports

Every module in the program's import graph must be subset-compliant.

### Syntactic Rules

| Allowed                | Not Allowed       | Rationale                      |
| ---------------------- | ----------------- | ------------------------------ |
| `from X import Y`      | `from X import *` | Star imports obscure bindings  |
| `from X import Y as Z` | `import X as Y`   | Module aliases obscure origins |
| `from . import module` |                   |                                |
| `import sys`           |                   | stdin/stdout/stderr/argv       |
| `import os`            |                   | `os.getenv()`                  |

Bare `import` is restricted to `sys` and `os`.

### Semantic Rules

Every import must resolve to one of:

| Source                | Examples                                                                                                                               |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Project file          | `from .module import X`, `from mypackage import Y`                                                                                     |
| Allowed stdlib module | `from typing import ...`, `from dataclasses import dataclass`, `from collections.abc import ...`, `from __future__ import annotations` |
| Allowed bare import   | `import sys`, `import os`                                                                                                              |

Unresolvable imports — other stdlib modules, external packages — are errors.

## Errors

| Condition          | Diagnostic                                    |
| ------------------ | --------------------------------------------- |
| Banned construct   | error: `{construct} not allowed: {reason}`    |
| Missing annotation | error: `missing type annotation for '{name}'` |
| Invalid decorator  | error: `decorator '{name}' not allowed`       |
| Banned builtin     | error: `builtin '{name}' not allowed`         |
| Invalid import     | error: `import of '{module}' not allowed`     |
| Mutable default    | error: `mutable default argument not allowed` |
