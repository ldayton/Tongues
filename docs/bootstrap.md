# Bootstrap Architecture

The transpiler will eventually be extracted to [Tongues](https://github.com/anthropics/Tongues), a standalone tool for multi-language transpilation. Tongues uses a two-layer architecture to solve the bootstrap problem.

## The Bootstrap Problem

A transpiler that converts Python to other languages faces a chicken-and-egg problem:

1. **Parsing requires Python internals** — `ast.parse()` is CPython-specific
2. **The transpiler should transpile itself** — Self-hosting proves correctness
3. **Some operations are inherently platform-specific** — I/O, CLI, file access

You can't transpile code that uses `ast.parse()`, `getattr()`, or `print()`. But a transpiler needs all of these.

## Solution: Two Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                        bootstrap/                                │
│  Full Python: ast.parse(), I/O, CLI, exec()                     │
│  NOT transpilable — runs only on CPython                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ dict-based AST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        src/tongues/                              │
│  Restricted Python: the Tongues subset                          │
│  TRANSPILABLE — runs on any target language                     │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### Bootstrap Layer (Python-only)

| Module | Purpose |
|--------|---------|
| `parser.py` | Wrap `ast.parse()`, convert to dict representation |
| `cli.py` | Argument parsing, file I/O, error reporting |
| `bootstrap.py` | Serialize AST to Python module for offline use |

The bootstrap layer uses whatever Python features are needed:
- `ast.parse()`, `ast.iter_fields()`
- `getattr()`, `hasattr()`
- f-strings, generators, comprehensions
- `print()`, `open()`, `Path.read_text()`

### Core Layer (Transpilable)

| Module | Purpose |
|--------|---------|
| `verify.py` | Validate code against the Tongues subset |
| `frontend.py` | Python AST → IR (future) |
| `middleend.py` | IR analysis passes (future) |
| `backend/*.py` | IR → target code (future) |

The core layer is written in the Tongues subset:
- No `ast` module — operates on `dict[str, Any]`
- No introspection — no `getattr`, `hasattr`, `type()`
- No I/O — receives data, returns results
- No generators — uses explicit loops
- No f-strings — uses concatenation

## Data Format: Dict-Based AST

The bootstrap layer converts Python's `ast.AST` nodes to plain dicts:

```python
# Python AST (not transpilable)
node = ast.parse("x + 1").body[0].value
isinstance(node, ast.BinOp)  # requires ast module
node.left                     # attribute access on AST class

# Dict-based AST (transpilable)
node = {"_type": "BinOp", "left": {...}, "right": {...}, "op": {...}}
node.get("_type") == "BinOp"  # plain dict operations
node.get("left")              # works in any language
```

This representation:
- Contains all information from the original AST
- Uses only primitive operations (dict access, string comparison)
- Can be serialized to any format (JSON, MessagePack, etc.)
- Works identically in Go, Rust, TypeScript, etc.

## Self-Hosting Path

### Phase 1: Python Bootstrap (Current)

```
Python source → [bootstrap: ast.parse()] → dict AST → [core: verify/transpile] → target code
```

The bootstrap layer runs on CPython. The core layer is written in Tongues-compliant Python.

### Phase 2: Transpiled Core

```
Python source → [bootstrap: ast.parse()] → dict AST → [Go/Rust core] → target code
```

Transpile the core layer to Go/Rust. The bootstrap layer still handles parsing.

### Phase 3: Native Parsers

```
Python source → [native parser] → dict AST → [native core] → target code
```

Write native Python parsers for each target. Now the entire toolchain runs natively.

### Phase 4: Full Self-Hosting

```
Tongues source → [Tongues-in-Tongues] → any target
```

The transpiler transpiles itself. Changes to the core propagate to all targets automatically.

## Implications for Parable

### Current State

Parable's transpiler (`transpiler/src/`) is a single-layer Python application:

```
parable.py → [frontend.py] → IR → [middleend.py] → [backend/go.py] → parable.go
```

It uses `ast.parse()` directly and cannot transpile itself.

### Migration Path

1. **Extract to Tongues subset** — Rewrite frontend/middleend/backend without Python-specific features
2. **Add bootstrap layer** — Move AST parsing to a separate non-transpilable module
3. **Verify compliance** — Use `tongues verify` to ensure core code is transpilable
4. **Transpile the transpiler** — Generate Go/Rust versions of the transpiler itself

### What Changes

| Component | Current | Future |
|-----------|---------|--------|
| AST parsing | `ast.parse()` in frontend | Bootstrap layer only |
| AST representation | `ast.AST` nodes | `dict[str, Any]` |
| Type checking | Python `isinstance()` | `node.get("_type") == "..."` |
| Iteration | `for x in items` | `while i < len(items)` |
| String formatting | f-strings | Concatenation |

### What Stays the Same

- IR definitions (`ir.py`)
- Middleend analysis passes
- Backend emission logic
- The restricted Python subset being transpiled

## Design Principles

### 1. Minimize Bootstrap Surface

The bootstrap layer should be as small as possible:
- Only truly untranspilable operations
- Thin wrappers, not business logic
- Easy to reimplement per-platform

### 2. Data Over Behavior

Pass data between layers, not objects:
- Dict-based AST, not class instances
- Plain return values, not exceptions for control flow
- Explicit parameters, not implicit context

### 3. Prove by Transpiling

If code is in the core layer, it must transpile:
- Run `tongues verify` in CI
- Any violation is a bug
- The verifier itself is verified

### 4. One Source of Truth

The Python implementation is canonical:
- Test in Python first
- Transpile to targets
- All targets pass the same tests
