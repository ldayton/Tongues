# Supported Python

Tongues transpiles a subset of Python. **Your code is still Python**—you can always run `python myfile.py` and get correct behavior. This document describes what's supported and what to avoid.

The core constraint: **everything knowable at runtime must be knowable at compile time.**

## Quick Reference

```python
# YES: Explicit, statically analyzable
def parse(source: str, pos: int) -> tuple[Node | None, str]:
    if node is None:
        return None, ""
    return node, text

# NO: Dynamic, requires runtime introspection
def parse(source, pos):  # missing types
    return getattr(obj, method_name)()  # dynamic dispatch
```

## Banned Constructs

### Metaprogramming

| Banned                    | Reason                     | Alternative                 |
| ------------------------- | -------------------------- | --------------------------- |
| `getattr(obj, name)`      | Runtime attribute lookup   | Direct field access         |
| `setattr(obj, name, val)` | Runtime attribute mutation | Direct assignment           |
| `hasattr(obj, name)`      | Runtime introspection      | `isinstance` or field check |
| `type(obj)`               | Runtime type inspection    | `isinstance`                |
| `eval()`, `exec()`        | Code generation            | Not supported               |
| `__class__`, `__dict__`   | Reflection                 | Not supported               |

### Generators and Lazy Evaluation

| Banned                | Reason               | Alternative                  |
| --------------------- | -------------------- | ---------------------------- |
| `yield`, `yield from` | Lazy evaluation      | Return list                  |
| `(x for x in items)`  | Generator expression | List comprehension           |
| `map(f, items)`       | Returns iterator     | `[f(x) for x in items]`      |
| `filter(f, items)`    | Returns iterator     | `[x for x in items if f(x)]` |
| `zip(a, b)`           | Returns iterator     | Indexed loop                 |
| `enumerate(items)`    | Returns iterator     | `range(len(items))`          |
| `reversed(items)`     | Returns iterator     | `items[::-1]`                |
| `iter()`, `next()`    | Iterator protocol    | Explicit indexing            |

### Async

| Banned       | Reason          | Alternative      |
| ------------ | --------------- | ---------------- |
| `async def`  | Coroutines      | Synchronous code |
| `await`      | Suspension      | Synchronous code |
| `async for`  | Async iteration | Synchronous loop |
| `async with` | Async context   | `try`/`finally`  |

### Closures and Nested Scope

| Banned           | Reason             | Alternative           |
| ---------------- | ------------------ | --------------------- |
| `lambda x: expr` | Anonymous function | Named function        |
| `nonlocal var`   | Closure mutation   | Pass as parameter     |
| `global var`     | Global mutation    | Pass as parameter     |
| Nested `def`     | Closure capture    | Module-level function |

### OOP Patterns

| Banned               | Reason               | Alternative                      |
| -------------------- | -------------------- | -------------------------------- |
| `@staticmethod`      | No `self` binding    | Module-level function            |
| `@classmethod`       | Class as parameter   | Module-level function            |
| `@property`          | Descriptor protocol  | Explicit getter method           |
| `@decorator`         | Runtime modification | Direct call                      |
| Multiple inheritance | Diamond problem      | Single inheritance + composition |
| Nested classes       | Scope complexity     | Module-level classes             |

### Dunder Methods

| Allowed                           | Banned                                                                   |
| --------------------------------- | ------------------------------------------------------------------------ |
| `__init__`, `__new__`, `__repr__` | All others: `__str__`, `__eq__`, `__hash__`, `__len__`, `__iter__`, etc. |

Dunder methods require runtime protocol dispatch. Use explicit methods instead:
- `__str__` → `def to_string(self) -> str`
- `__eq__` → `def equals(self, other: T) -> bool`
- `__len__` → `def length(self) -> int`

### Control Flow

| Banned                       | Reason                      | Alternative         |
| ---------------------------- | --------------------------- | ------------------- |
| `with` statement             | Context manager protocol    | `try`/`finally`     |
| `for`/`else`, `while`/`else` | Unusual semantics           | Flag variable       |
| Bare `except:`               | Catches everything          | `except Exception:` |
| `match`/`case`               | Pattern matching complexity | `if`/`elif` chain   |

### Python Idioms

| Banned              | Reason                     | Alternative            |
| ------------------- | -------------------------- | ---------------------- |
| `a < b < c`         | Chained comparison         | `a < b and b < c`      |
| `x or []`           | Falsy default              | `if x is None: x = []` |
| `x ** 2`            | Power operator             | `x * x`                |
| `x is y` (non-None) | Identity vs equality       | `x == y`               |
| `:=` walrus         | Complicates type inference | Separate assignment    |

### I/O and System

| Banned                     | Reason                | Alternative               |
| -------------------------- | --------------------- | ------------------------- |
| `print(a, b, c)`           | Variadic              | `print(f"{a} {b} {c}")`   |
| `print(..., sep=x)`        | Complexity            | Manual join               |
| `print(..., flush=True)`   | Complexity            | Not supported             |
| `open()`                   | File I/O              | Receive data as parameter |
| `input()`                  | Python-specific       | `sys.stdin.readline()`    |
| `sys.exit()`               | Control flow          | Return from `main()`      |
| `sys.stdout.write()`, etc. | Method I/O            | `print()`                 |
| `for line in sys.stdin`    | Iterator              | `while` + `readline()`    |
| `os.environ[name]`         | May raise KeyError    | `os.getenv(name)`         |
| `os.*`, `sys.*` (other)    | Platform APIs         | Not supported             |
| `import`                   | External dependencies | Self-contained code       |

Allowed imports: `__future__`, `typing`, `collections.abc`, `sys` (for `argv`, `stdin`, `stderr`), `os` (for `getenv`)

Files and network are permanently out of scope—they require resource management, error handling, and platform abstractions that don't map cleanly across languages.

### Assignment

| Banned              | Reason               | Alternative               |
| ------------------- | -------------------- | ------------------------- |
| `del name`          | Unbinding variables  | Reassign to sentinel      |
| `del obj.attr`      | Attribute deletion   | Set to None               |
| `a, b = variable`   | Unpack from variable | Unpack from call directly |
| `def f(x=[])`       | Mutable default      | `def f(x=None)`           |
| `*args`, `**kwargs` | Dynamic arguments    | Explicit parameters       |

`del items[i]` is allowed (collection mutation).

## Required Annotations

All type information must be explicit.

### Functions

```python
# Required: return type and all parameter types
def parse(source: str, pos: int) -> tuple[Node | None, str]:
    ...

# Banned: missing types
def parse(source, pos):
    ...

# Banned: bare collections
def get_items() -> list:  # needs list[T]
    ...
```

### Variables

```python
# Required for non-obvious types
items: list[str] = []
node: Node | None = None

# OK: type obvious from literal
count = 0
name = "default"
flag = True

# OK: type obvious from annotated function call
result = parse(source, 0)  # parse() return type is known
```

### Class Fields

```python
class Parser:
    # Annotate fields at class level or in __init__
    pos: int
    source: str
    errors: list[str]

    def __init__(self, source: str) -> None:
        self.pos = 0
        self.source = source
        self.errors = []
```

## Allowed Constructs

### Types

| Type       | Example                                |
| ---------- | -------------------------------------- |
| Primitives | `int`, `float`, `str`, `bool`, `bytes` |
| Optional   | `T \| None`                            |
| Union      | `A \| B \| C`                          |
| List       | `list[T]`                              |
| Dict       | `dict[K, V]`                           |
| Set        | `set[T]`                               |
| Tuple      | `tuple[A, B, C]`                       |
| Callable   | `Callable[[A, B], R]`                  |

### Control Flow

```python
# Conditionals
if condition:
    ...
elif other:
    ...
else:
    ...

# Loops
for item in items:
    ...

for i in range(n):
    ...

for i in range(start, stop):
    ...

for i in range(start, stop, step):
    ...

while condition:
    ...

# Loop control
break
continue

# Exception handling
try:
    ...
except ExceptionType as e:
    ...
finally:
    ...

raise ErrorType(message)
```

### Expressions

```python
# Arithmetic
a + b, a - b, a * b, a / b, a // b, a % b

# Comparison
a == b, a != b, a < b, a <= b, a > b, a >= b

# Boolean
a and b, a or b, not a

# Bitwise
a & b, a | b, a ^ b, ~a, a << n, a >> n

# Identity (None only)
x is None, x is not None

# Membership
x in collection, x not in collection

# Ternary
value if condition else other

# Indexing and slicing
items[i], items[-1], items[a:b], items[::step]

# Comprehensions
[expr for x in items]
[expr for x in items if cond]
{expr for x in items}
{k: v for x in items}
```

### Built-in Functions

| Category      | Allowed                                                      |
| ------------- | ------------------------------------------------------------ |
| Math          | `abs`, `min`, `max`, `sum`, `round`, `divmod`, `pow`         |
| Conversion    | `int`, `float`, `str`, `bool`, `bytes`, `chr`, `ord`         |
| Collections   | `list`, `dict`, `set`, `tuple`, `frozenset`, `len`, `sorted` |
| Type checking | `isinstance`                                                 |
| Iteration     | `range`                                                      |
| Formatting    | `repr`, `ascii`, `bin`, `hex`, `oct`                         |
| Boolean       | `all`, `any`                                                 |
| Other         | `slice`, `super`, `object`                                   |

### I/O Primitives

Tongues supports minimal I/O for building executable programs. These are the only I/O operations available—files and network are out of scope.

#### Entry Point and Arguments

```python
import sys

def main() -> int:
    """Program entry point. Returns exit code (0 for success)."""
    args = sys.argv[1:]  # standard Python idiom
    ...
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

Programs must define `main()` returning `int`. The `if __name__ == "__main__"` guard is recognized and transpiled to appropriate entry point code for each target language.

`sys.argv` behaves like Python: `argv[0]` is the program name, `argv[1:]` are user arguments. On Java, `argv[0]` is empty string (Java doesn't provide program name).

#### Output

```python
print(x)                       # stdout with newline
print(x, end='')               # stdout without newline
print(x, file=sys.stderr)      # stderr with newline
print(x, file=sys.stderr, end='')  # stderr without newline
```

Only single-value `print()` is supported. For multiple values, use f-strings: `print(f"{a} {b}")`.

`sep=` and `flush=` are not supported.

#### Input (Text)

```python
sys.stdin.readline()  # line with trailing \n, or '' on EOF
sys.stdin.read()      # entire contents as str
```

Standard Python semantics: `readline()` returns empty string on EOF, not None. The trailing newline is included (except possibly on the last line).

#### Input/Output (Binary)

```python
sys.stdin.buffer.read()       # all bytes
sys.stdin.buffer.read(n)      # up to n bytes (may return fewer on EOF)
sys.stdin.buffer.readline()   # line as bytes

sys.stdout.buffer.write(b)    # write bytes, returns count written
sys.stderr.buffer.write(b)    # write bytes to stderr
```

Standard Python semantics: `read(n)` may return fewer than n bytes at EOF—check `len(result)`.

#### Environment

```python
os.getenv(name)           # None if unset
os.getenv(name, default)  # default if unset
```

#### Target Language Mapping

| Python                   | Go                           | Rust                          | C                            | TypeScript                   |
| ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ---------------------------- |
| `sys.argv`               | `os.Args`                    | `std::env::args().collect()`  | `argv[0..argc]`              | `process.argv.slice(1)`      |
| `print(x)`               | `fmt.Println(x)`             | `println!("{}", x)`           | `printf("%s\n", x)`          | `console.log(x)`             |
| `print(x, file=stderr)`  | `fmt.Fprintln(os.Stderr, x)` | `eprintln!("{}", x)`          | `fprintf(stderr, "%s\n", x)` | `console.error(x)`           |
| `print(x, end='')`       | `fmt.Print(x)`               | `print!("{}", x)`             | `printf("%s", x)`            | `process.stdout.write(x)`    |
| `sys.stdin.readline()`   | `bufio.Scanner`              | `stdin().read_line()`         | `fgets()`                    | `readline` sync              |
| `sys.stdin.read()`       | `io.ReadAll(os.Stdin)`       | `io::read_to_string(stdin())` | `read()` loop                | `fs.readFileSync(0, 'utf8')` |
| `stdin.buffer.read()`    | `io.ReadAll(os.Stdin)`       | `stdin().read_to_end()`       | `fread()` loop               | `fs.readFileSync(0)`         |
| `stdin.buffer.read(n)`   | `io.ReadAtLeast`             | `stdin().take(n).read()`      | `fread(buf,1,n,stdin)`       | manual Buffer                |
| `stdout.buffer.write(b)` | `os.Stdout.Write(b)`         | `stdout().write_all(b)`       | `fwrite()`                   | `process.stdout.write(b)`    |
| `os.getenv(name)`        | `os.Getenv()`                | `std::env::var().ok()`        | `getenv()`                   | `process.env[name]`          |

### String Methods

```python
s.join(items)
s.split(sep), s.split()  # with/without separator
s.strip(), s.lstrip(), s.rstrip()
s.lower(), s.upper()
s.startswith(prefix), s.endswith(suffix)
s.replace(old, new)
s.find(sub), s.rfind(sub)
s.count(sub)
s.isalnum(), s.isalpha(), s.isdigit(), s.isspace()
s.isupper(), s.islower()
```

### List Methods

```python
items.append(x)
items.extend(other)
items.pop(), items.pop(i)
items.insert(i, x)
items.remove(x)
items.copy()
items.clear()
items.index(x)
items.count(x)
items.reverse()
items.sort()
```

### Dict Methods

```python
d.get(key), d.get(key, default)
d.keys(), d.values(), d.items()
d.pop(key), d.pop(key, default)
d.setdefault(key, default)
d.update(other)
d.clear()
d.copy()
```

### Set Methods

```python
s.add(x)
s.remove(x), s.discard(x)
s.pop()
s.clear()
s.copy()
s.union(other), s.intersection(other), s.difference(other)
s.issubset(other), s.issuperset(other)
```

## Style Recommendations

### Prefer Explicit Over Implicit

```python
# Good: explicit None check
if node is None:
    return default

# Avoid: truthiness for None check
if not node:  # also catches empty list, zero, etc.
    return default
```

### Prefer Simple Over Clever

```python
# Good: explicit loop
result = []
for item in items:
    if predicate(item):
        result.append(transform(item))

# Avoid: nested comprehension
result = [transform(item) for item in items if predicate(item)]  # OK but harder to debug
```

### Prefer Flat Over Nested

```python
# Good: early return
def process(node: Node | None) -> str:
    if node is None:
        return ""
    if node.kind != "word":
        return ""
    return node.value

# Avoid: nested conditionals
def process(node: Node | None) -> str:
    if node is not None:
        if node.kind == "word":
            return node.value
    return ""
```

## Verification

Use `tongues verify` to check compliance:

```bash
# Verify a file
tongues verify mycode.py

# Verify a directory
tongues verify src/

# Group errors by category
tongues verify src/ --by-category

# Verbose output with node statistics
tongues verify src/ -v
```

Violations are categorized:
- `async` — Async constructs
- `generator` — Generators and lazy evaluation
- `function` — Closures, decorators, banned dunders
- `class` — Multiple inheritance, class decorators
- `control` — `with`, loop `else`, bare `except`
- `builtin` — Banned built-in functions
- `reflection` — Runtime introspection
- `types` — Missing or invalid type annotations
- `import` — Disallowed imports
- `expression` — Banned operators or patterns
- `statement` — Banned statements
- `syntax` — Unsupported syntax
