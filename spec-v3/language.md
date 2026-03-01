# Tongues Source Language

Tongues source files are valid Python. They execute unmodified with identical semantics on CPython or PyPy. Tongues is not a Python-like language — it is a subset of Python, restricted to the fragment whose types are fully concrete and whose semantics can be determined statically.

## Principles

Every expression has exactly one concrete type. There is no top type, no `Any`, no partial generics like `list[any]`. Types are either declared in annotations or inferred from context, but never left ambiguous.

All dispatch is static. There is no dynamic attribute access, no reflection, no metaprogramming. Every method call and field access can be resolved at compile time.

All control flow is structured and all resources have lexical lifetimes. Iterators are consumed at the point of creation. Files are opened and closed within a single `with` block.

## Types

### Primitives

`int`, `float`, `str`, `bool`, `bytes`. No implicit conversions in source — `bool` is assignable to `int`, and `int` to `float`, but coercions are inserted explicitly during compilation.

### Collections

`list[T]`, `dict[K, V]`, `set[T]`, `tuple[A, B, ...]`. Always parameterized — bare `list`, `dict`, `set`, `tuple` are errors. Collections are invariant: `list[Dog]` is not assignable to `list[Animal]`.

Dict and set iteration order is not preserved across transpilation. Python guarantees insertion-order iteration for dicts, but the target languages do not. Code that depends on dict or set ordering may produce different results after transpilation.

### Optionals and Unions

`T | None` (or `Optional[T]`) for nullable types. `A | B | C` for unions.

### Callables

`Callable[[A, B], R]` for function types. Functions are first-class values that do not capture mutable state.

### Type Aliases

`TypeAlias` declarations expand at parse time via name substitution. They introduce no new types — the alias is replaced by its definition everywhere it appears.

### Structs and Interfaces

Classes map to structs. A class with subclasses becomes an interface (the base) with implementing structs (the subclasses). Single inheritance only. `@dataclass` is the only decorator, with limited arguments (`eq`, `order`, `unsafe_hash`, `kw_only`).

## Annotations

All function parameters (except `self`) and return types must be annotated. `Any` is banned — there are no escape hatches. Empty collection literals (`[]`, `{}`, `set()`) require a type annotation on the variable.

## Generics

Functions and classes can be parameterized by type variables. At every use site, type parameters are resolved to concrete types — the "every expression has one concrete type" invariant holds per-instantiation. There is no type erasure; generic code is monomorphized.

```python
T = TypeVar("T")

def first(xs: list[T]) -> T:
    return xs[0]

class Stack(Generic[T]):
    items: list[T]
    def push(self, item: T) -> None:
        self.items.append(item)
    def pop(self) -> T:
        return self.items.pop()
```

Type parameters may be bounded (`T = TypeVar("T", bound=Comparable)`) to constrain what types can be substituted. Unbounded type parameters accept any concrete type. Higher-kinded types and variance annotations are not supported.

## Functions and Scope

Two scope levels: module and function-local. No `global`/`nonlocal`. All parameters are explicit — no `*args`, `**kwargs`, or unpacking at call sites. Mutable defaults (`[]`, `{}`) are banned. No generator functions — `yield` and `yield from` are not allowed.

Local functions and lambda expressions are allowed. They must not capture mutable state from an enclosing scope — they are scoped helpers, not closures. Both desugar to module-level named functions during compilation.

Protocol methods are allowed: `__init__`, `__new__`, `__repr__`, `__str__`, `__eq__`, `__hash__`, `__lt__`. These have direct translations in every target language. `@dataclass` can generate `__eq__`, `__hash__`, and `__lt__` via `eq=True`, `unsafe_hash=True`, and `order=True`. Operator overloading (`__add__`, `__getitem__`, `__contains__`, `__len__`, `__bool__`, etc.) is not allowed.

## Type Safety

### Narrowing

Variables with optional, union, or interface types cannot be used until narrowed to a concrete type. Narrowing happens through:

- **isinstance**: `if isinstance(x, T)` narrows `x` to `T` in the then-branch and to the remaining union members in the else-branch. `isinstance(x, (A, B))` is equivalent to `isinstance(x, A) or isinstance(x, B)`.
- **match/case**: Pattern matching narrows each case arm to the matched type. isinstance chains can be written as match statements.
- **None checks**: `if x is not None` unwraps optionals. `if x is None: return` unwraps for subsequent code via early exit.
- **Truthiness**: `if x:` narrows optionals to their non-None type. Only types with unambiguous truthiness are allowed in boolean context — `int` and `float` are rejected because zero is valid data. Optional collections (`list[T]?`) are rejected because it's ambiguous whether the check is for None or emptiness.
- **Assert**: `assert isinstance(x, T)` and `assert x is not None` narrow for subsequent code.
- **Attribute paths**: `if x.attr is not None` tracks the narrowed attribute for subsequent access.

Early exit (`return`, `raise`, `break`, `continue`) in a then-branch narrows the else type into subsequent code.

### Collection Homogeneity

List and dict literals must have consistent element types. Collection mutations (`append`, `add`, `insert`, `d[k] = v`) are type-checked against the declared element type.

### Iterator Consumption

`enumerate`, `zip`, `reversed`, and generator expressions must be consumed immediately — in a for-loop header or as an argument to an eager consumer (`list`, `tuple`, `set`, `sorted`, `any`, `all`, `sum`, `min`, `max`, `str.join`). They cannot be assigned to variables, returned, or passed to other functions.

### Tuple Typing

Fixed-length tuples have per-element types. Static index access is bounds-checked. Unpacking must match the tuple length. Optional tuples require a guard before unpacking.

## Restricted Syntax

### Banned Constructs

`async`/`await`, `yield`/`yield from`, `global`, `nonlocal`, nested classes, multiple inheritance (except for exception marker bases), `for`/`while`/`try` else clauses, bare `except:`.

### Restricted Builtins

A whitelist of builtins is available. Banned categories: runtime introspection (`type`, `getattr`, `vars`, `dir`, `id`, `callable`), dynamic execution (`eval`, `exec`), manual iteration (`iter`, `next`, `map`, `filter`). `print` takes one positional argument.

`sorted`, `min`, and `max` accept an optional `key=` argument, which must be a function reference or lambda.

### Deletion

`del` is allowed on dict keys (`del d[k]`) and list indices (`del xs[i]`). Variable deletion is not allowed.

### Identity Comparison

`is` and `is not` are restricted to singleton comparisons: `x is None`, `x is True`, `x is False`.

### F-strings

Conversion specifiers `!r` and `!s` are allowed (desugared to `repr()` and `str()` calls). Format specs (`:,.2f`) are not.

### Exception Handling

`except` clauses accept multiple exception types: `except (ValueError, KeyError) as e:`.

### Star Unpacking

Star unpacking is allowed in list literals: `[*a, x, *b]`. It desugars to concatenation. Star unpacking in assignments (`a, *rest = xs`) is not allowed — it requires runtime length computation that cannot be resolved statically.

### Slicing

Slice-with-step syntax is allowed: `xs[::2]`, `xs[::-1]`. Step slices desugar to loops or reversal.

### Imports

`import` is limited to `sys` and `os`. `from X import Y` is allowed for `typing`, `dataclasses`, `collections.abc`, `__future__`, and project-local modules. No star imports.

## I/O

I/O is restricted to specific idioms that map to target-language equivalents:

- **stdout/stderr**: `print(x)`, `print(x, end="")`, `print(x, file=sys.stderr)`, `sys.stdout.buffer.write(b)`
- **stdin**: `sys.stdin.readline()`, `sys.stdin.read()`, `sys.stdin.buffer.read()`
- **files**: `with open(path, mode) as f:` with a single read or write operation
- **environment**: `os.getenv(name)`, `os.getenv(name, default)`
- **arguments**: `sys.argv`
- **exit**: `sys.exit(code)`
