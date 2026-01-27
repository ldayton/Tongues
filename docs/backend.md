# Backend Architecture

The backend (`src/backend/`) emits target language code from annotated IR. Each target language has its own emitter file.

```
Frontend → [IR] → Middleend → [Annotated IR] → Backend → target code
```

## Current State

| File | Lines | Status |
|------|-------|--------|
| `go.py` | 1,563 | Primary, fully working |
| `java.py` | 983 | Stub |
| `c.py` | 921 | Stub |
| `rust.py` | 906 | Stub |
| `csharp.py` | 891 | Stub |
| `zig.py` | 875 | Stub |
| `python.py` | 790 | Stub |
| `swift.py` | 756 | Stub |
| `typescript.py` | 755 | Stub |
| `ruby.py` | 675 | Stub |
| `util.py` | 70 | Shared utilities |

### Go Backend Structure

```
GoBackend class (~1,563 lines)
├── emit()                    # Entry point
├── _emit_header()            # Package, imports
├── _emit_string_helpers()    # Runtime helpers (embedded Go code)
├── _emit_struct()            # Struct definitions
├── _emit_function()          # Function/method definitions
├── _emit_stmt_*()            # Statement dispatch (18 methods)
├── _emit_expr_*()            # Expression dispatch (28 methods)
├── _emit_lvalue()            # LValue emission
├── _type_to_go()             # IR Type → Go type string
└── _to_camel()/_to_pascal()  # Name conversion
```

### Shared Utilities (util.py)

```python
# Name converters
to_snake(name)          # camelCase → snake_case
to_camel(name)          # snake_case → camelCase
to_pascal(name)         # snake_case → PascalCase
to_screaming_snake(name)  # → SCREAMING_SNAKE

# String escaping
escape_string(value)

# Base emitter class
class Emitter:
    def __init__(self, indent_str="    ")
    def line(self, text="")
    def output() -> str
```

### Current Issues

1. **Go backend doesn't use `util.py`** — reimplements name converters
2. **Inline method dispatch** — 50+ lines of if/elif for string methods
3. **Embedded runtime helpers** — 100+ lines of Go as Python string literal
4. **Ad-hoc state tracking** — multiple instance dicts for context

## Design Principles

### 1. IR-First Code Generation

From [Strumenta's transpiler guide](https://tomassetti.me/how-to-write-a-transpiler/):

> "The single biggest mistake in persons trying to implement a transpiler without experience is they try to generate directly the code of the target language from the AST of the original language."

**Correct pipeline (what we do):**
```
Source AST → IR → Target Code
```

The IR should contain all information needed for emission. Backends should not compensate for missing frontend work.

### 2. One File Per Target

Each target language gets its own emitter file. This is the standard approach used by:
- LLVM (separate backend directories per target)
- Kalai transpiler (separate passes per language)
- GraalVM Truffle (separate language implementations)

### 3. Visitor Pattern via Dispatch

```python
def _emit_stmt(self, stmt: Stmt) -> None:
    method = f"_emit_stmt_{type(stmt).__name__}"
    getattr(self, method, self._emit_stmt_default)(stmt)
```

Benefits:
- All code for one operation in one class
- AST/IR nodes know nothing about emission
- Fast dispatch via method lookup
- New targets = new classes, no IR changes

### 4. Iterate Then Generalize

From [Kalai's design](https://github.com/kalai-transpiler/kalai/blob/main/docs/Design.md):

> "When implementing new functionality, put it in a pass in the target language-specific pipeline phase, and then later refactor into the [shared] construct phase if/when commonalities across target languages are understood."

Don't prematurely extract shared code. When patterns genuinely emerge across Go/Rust/etc., then refactor.

## Target Structure

Minimal changes to improve consistency:

```
src/backend/
├── __init__.py       # Exports
├── base.py           # Emitter base class + shared dispatch
├── util.py           # Name converters, string escaping
├── go.py             # Go backend (extends Emitter)
├── rust.py           # Rust backend (extends Emitter)
└── ...
```

## Emitter Base Class

Move shared logic to a base class that all backends extend:

```python
# base.py
from abc import ABC, abstractmethod
from src.ir import Stmt, Expr, Module

class Emitter(ABC):
    """Base class for all target language emitters."""

    def __init__(self, indent_str: str = "    ") -> None:
        self.indent = 0
        self.lines: list[str] = []
        self._indent_str = indent_str

    def line(self, text: str = "") -> None:
        """Emit a line with current indentation."""
        if text:
            self.lines.append(self._indent_str * self.indent + text)
        else:
            self.lines.append("")

    def line_raw(self, text: str) -> None:
        """Emit a line without indentation (for inline continuations)."""
        if self.lines:
            self.lines[-1] += text
        else:
            self.lines.append(text)

    def output(self) -> str:
        """Return accumulated output as string."""
        return "\n".join(self.lines)

    def indent_block(self):
        """Context manager for indented blocks."""
        return _IndentContext(self)

    @abstractmethod
    def emit(self, module: Module) -> str:
        """Emit target code from IR module."""
        ...

    def emit_stmt(self, stmt: Stmt) -> None:
        """Dispatch to type-specific statement emitter."""
        method = f"emit_stmt_{type(stmt).__name__}"
        handler = getattr(self, method, None)
        if handler:
            handler(stmt)
        else:
            self.line(f"// TODO: {type(stmt).__name__}")

    def emit_expr(self, expr: Expr) -> str:
        """Dispatch to type-specific expression emitter."""
        method = f"emit_expr_{type(expr).__name__}"
        handler = getattr(self, method, None)
        if handler:
            return handler(expr)
        return f"/* TODO: {type(expr).__name__} */"


class _IndentContext:
    def __init__(self, emitter: Emitter):
        self.emitter = emitter

    def __enter__(self):
        self.emitter.indent += 1
        return self

    def __exit__(self, *args):
        self.emitter.indent -= 1
```

Usage:

```python
# go.py
from src.backend.base import Emitter
from src.backend.util import to_camel, to_pascal, escape_string

class GoBackend(Emitter):
    def __init__(self):
        super().__init__(indent_str="\t")  # Go uses tabs
        self._receiver_name = ""
        self._tuple_vars: dict[str, Tuple] = {}
        # ...

    def emit(self, module: Module) -> str:
        self.lines = []
        self._emit_header(module)
        for struct in module.structs:
            self._emit_struct(struct)
        for func in module.functions:
            self._emit_function(func)
        return self.output()

    def emit_stmt_If(self, stmt: If) -> None:
        cond = self.emit_expr(stmt.cond)
        self.line(f"if {cond} {{")
        with self.indent_block():
            for s in stmt.then_body:
                self.emit_stmt(s)
        if stmt.else_body:
            self.line("} else {")
            with self.indent_block():
                for s in stmt.else_body:
                    self.emit_stmt(s)
        self.line("}")
```

## Table-Driven Method Dispatch

Reduce verbose if/elif chains with lookup tables:

```python
# go.py

# String method mappings: Python method → Go code generator
STRING_METHODS: dict[str, Callable[[str, list[str]], str]] = {
    "join": lambda obj, args: f"strings.Join({args[0]}, {obj})",
    "startswith": lambda obj, args: f"strings.HasPrefix({obj}, {args[0]})",
    "endswith": lambda obj, args: f"strings.HasSuffix({obj}, {args[0]})",
    "replace": lambda obj, args: f"strings.ReplaceAll({obj}, {args[0]}, {args[1]})",
    "lower": lambda obj, args: f"strings.ToLower({obj})",
    "upper": lambda obj, args: f"strings.ToUpper({obj})",
    "strip": lambda obj, args: f"strings.TrimSpace({obj})",
    "lstrip": lambda obj, args: f'strings.TrimLeft({obj}, {args[0] if args else '" \\t\\n\\r"'})',
    "rstrip": lambda obj, args: f'strings.TrimRight({obj}, {args[0] if args else '" \\t\\n\\r"'})',
    "split": lambda obj, args: f"strings.Split({obj}, {args[0]})" if args else f"strings.Fields({obj})",
    "count": lambda obj, args: f"strings.Count({obj}, {args[0]})",
    "find": lambda obj, args: f"strings.Index({obj}, {args[0]})",
    "rfind": lambda obj, args: f"strings.LastIndex({obj}, {args[0]})",
}

# Character classification methods (need helper functions)
STRING_PREDICATES: dict[str, str] = {
    "isalnum": "_strIsAlnum",
    "isalpha": "_strIsAlpha",
    "isdigit": "_strIsDigit",
    "isspace": "_strIsSpace",
    "isupper": "_strIsUpper",
    "islower": "_strIsLower",
}

def emit_expr_MethodCall(self, expr: MethodCall) -> str:
    obj = self.emit_expr(expr.obj)
    args = [self.emit_expr(a) for a in expr.args]
    method = expr.method

    # String methods
    if method in STRING_METHODS:
        return STRING_METHODS[method](obj, args)
    if method in STRING_PREDICATES:
        helper = STRING_PREDICATES[method]
        return f"{helper}({obj})"

    # Slice methods
    if isinstance(expr.receiver_type, Slice):
        if method == "append" and args:
            return f"append({obj}, {args[0]})"
        if method == "extend" and args:
            return f"append({obj}, {args[0]}...)"
        if method == "pop" and not args:
            return f"{obj}[len({obj})-1]"
        if method == "copy":
            return f"append({obj}[:0:0], {obj}...)"

    # Default: PascalCase method call
    return f"{obj}.{to_pascal(method)}({', '.join(args)})"
```

## External Runtime Helpers

Instead of embedding Go code as Python strings, read from external files:

```
src/backend/
├── go.py
├── runtime/
│   └── go_helpers.go    # Actual Go file
└── ...
```

```python
# go.py
import importlib.resources

def _emit_runtime_helpers(self) -> None:
    """Emit helper functions from external Go file."""
    helpers = importlib.resources.read_text("src.backend.runtime", "go_helpers.go")
    # Skip package declaration, emit function bodies
    for line in helpers.split("\n"):
        if not line.startswith("package"):
            self.line_raw(line)
    self.line("")
```

Benefits:
- Syntax highlighting in IDE
- Go tooling can check the helpers
- Cleaner Python code

## Reference: How Other Compilers Organize Backends

### LLVM

```
llvm/lib/Target/
├── X86/
│   ├── X86.td              # TableGen target description
│   ├── X86InstrInfo.td     # Instruction definitions
│   ├── X86ISelLowering.cpp # Instruction selection
│   └── X86AsmPrinter.cpp   # Assembly emission
├── ARM/
│   └── ...
└── ...
```

Uses [TableGen](https://llvm.org/docs/TableGen/) for declarative target descriptions. Generates C++ code from `.td` files to reduce repetition.

### Kalai (Clojure → Rust/Java/C++)

```
kalai/
├── pass/
│   ├── kalai/         # Language-agnostic passes
│   └── target/
│       ├── rust.clj   # Rust-specific pass
│       ├── java.clj   # Java-specific pass
│       └── cpp.clj    # C++-specific pass
└── ...
```

**Nano-pass architecture**: Many small transformations rather than few large ones. Start target-specific, extract shared code only when patterns emerge.

Source: [Kalai Design](https://github.com/kalai-transpiler/kalai/blob/main/docs/Design.md)

### GraalVM Truffle

```
truffle/
├── api/               # Shared language implementation framework
└── ...

graaljs/               # JavaScript implementation
├── src/
│   └── com.oracle.truffle.js/
│       └── ...

truffleruby/           # Ruby implementation
├── src/
│   └── ...
```

Each language is a separate project that uses the shared Truffle framework. Languages share the compilation infrastructure but have independent implementations.

## Migration Strategy

Since the backend is already well-organized, changes are incremental:

1. **Extract `base.py`** from `util.py`
   - Move `Emitter` class
   - Add `emit_stmt` / `emit_expr` dispatch methods
   - Add `indent_block` context manager

2. **Update `go.py`** to use base class
   - Extend `Emitter`
   - Import name converters from `util.py`
   - Replace `self.output` list with inherited `self.lines`

3. **Extract method tables** (optional)
   - Create `STRING_METHODS`, `STRING_PREDICATES` dicts
   - Simplify `emit_expr_MethodCall`

4. **External runtime helpers** (optional)
   - Create `runtime/go_helpers.go`
   - Update `_emit_string_helpers` to read from file

Each step can be done independently and should pass tests before proceeding.

## When to Add New Backends

Follow the Kalai principle: implement target-specific first, then generalize.

1. **Copy `go.py`** as starting template
2. **Implement type emission** (`_type_to_X`)
3. **Implement statement emitters** (start with `VarDecl`, `Assign`, `If`, `Return`)
4. **Implement expression emitters** (start with literals, `Var`, `BinaryOp`, `Call`)
5. **Add runtime helpers** as needed
6. **Extract shared patterns** only when they genuinely span multiple backends

Don't create abstract "shared" code until you have at least 2-3 working backends that demonstrate the pattern.

## Backend Responsibilities

Backends (IR → target) handle only syntax:

1. **Name conversion** — snake_case → camelCase/PascalCase
2. **Syntax emission** — IR nodes → target syntax
3. **Error propagation** — Fallible calls: Go uses panic, Rust uses `?`, TS uses throw
4. **Idioms** — Target-specific patterns (defer/recover, try/catch)
5. **Formatting** — Indentation, line breaks

## Memory Strategy

### Rust Backend

Arena allocation with single lifetime `'arena` for all AST nodes:

```rust
struct Command<'arena> {
    words: Vec<'arena, &'arena Word<'arena>>,
}
```

Uses `bumpalo::Bump`. Sidesteps ownership inference.

### C Backend

Arena allocation with ptr+len strings:

```c
typedef struct { const char *data; size_t len; } Str;
typedef struct { char *base; char *ptr; size_t cap; } Arena;

void *arena_alloc(Arena *a, size_t size);
```

No per-node `free()`. Single `arena_free()` at end.

### Python Backend

Emit idiomatic Python, shedding restrictions. The source is written in restricted Python for transpilation; the Python backend produces clean Pythonic output.

**Easy transforms:**
```
lst[len(lst)-1]     →  lst[-1]
int(a / b)          →  a // b
a < b and b < c     →  a < b < c
if x is None: x=[]  →  x = x or []
TypeSwitch          →  match/case
```

**Pattern-based transforms:**
```
i = 0                       for i, item in enumerate(items):
for item in items:      →       process(item)
    process(item)
    i += 1

for i in range(len(a)):     for x, y in zip(a, b):
    x = a[i]            →       process(x, y)
    y = b[i]
    process(x, y)
```

**Not recoverable** (not in IR): `**kwargs`, decorators, generators, `async`/`await`.

## Ownership Model

AST is a **strict tree**: parent→child only, no cycles, no shared nodes, no back-references.

Nodes are immutable after construction. All nodes live until parse completion.

Arena allocation with single lifetime `'arena`. No reference counting needed.

**Ownership rule:** All child references are owned. No complex inference needed:

| Field Pattern   | Ownership        | Rust                          | C                   |
| --------------- | ---------------- | ----------------------------- | ------------------- |
| `field: Node`   | Owned            | `Box<Node>` or `&'arena Node` | `Node*` (arena)     |
| `field: [Node]` | Owned collection | `Vec<&'arena Node>`           | `NodeSlice` (arena) |
| `field: Node?`  | Owned optional   | `Option<&'arena Node>`        | `Node*` (nullable)  |

Back-references (if ever needed) would use indices, not pointers: `parent_idx: u32`.

## String Handling

Two string representations:

```
StringRef { start: u32, end: u32 }    // Byte range into source buffer
ArenaStr { ptr: *const u8, len: u32 } // Arena-allocated
```

| Field type                  | Representation | Example                                     |
| --------------------------- | -------------- | ------------------------------------------- |
| Parameter names, delimiters | `StringRef`    | `ParamExpansion.param`, `HereDoc.delimiter` |
| Constructed content         | `ArenaStr`     | `Word.value`, `AnsiCQuote.content`          |
| Operator literals           | `&'static str` | `Operator.op`                               |

Input buffer must outlive AST, or copy referenced ranges into arena at parse end.
