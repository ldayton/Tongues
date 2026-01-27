# Middleend Architecture

The middleend (`src/middleend.py`) performs IR analysis passes. It annotates IR nodes with computed properties but never transforms the IR structure.

```
Frontend → [IR] → Middleend → [Annotated IR] → Backend
```

## Current State

Single 1,000-line file with clear section headers but repetitive traversal code.

### Analysis Passes

| Pass | Annotates | Purpose |
|------|-----------|---------|
| `_analyze_reassignments` | `VarDecl.is_reassigned`, `Assign.is_declaration`, `Param.is_modified` | Track variable mutations |
| `_analyze_initial_value_usage` | `VarDecl.initial_value_unused`, `TryCatch.has_returns` | Detect dead initial values |
| `_analyze_named_returns` | `Function.needs_named_returns` | Go named return requirement |
| `_analyze_hoisting_all` | `TryCatch.hoisted_vars`, `If.hoisted_vars` | Variables needing hoisting |
| `_analyze_unused_tuple_targets_all` | `TupleAssign.unused_indices` | Emit `_` for unused elements |

### Problem: Repetitive Traversal

The same pattern appears ~10 times:

```python
if isinstance(stmt, If):
    recurse(stmt.then_body)
    recurse(stmt.else_body)
elif isinstance(stmt, While):
    recurse(stmt.body)
elif isinstance(stmt, ForRange):
    recurse(stmt.body)
elif isinstance(stmt, ForClassic):
    recurse(stmt.body)
elif isinstance(stmt, Block):
    recurse(stmt.body)
elif isinstance(stmt, TryCatch):
    recurse(stmt.body)
    recurse(stmt.catch_body)
elif isinstance(stmt, (Match, TypeSwitch)):
    for case in stmt.cases:
        recurse(case.body)
    recurse(stmt.default)
```

Similarly, expression traversal uses brittle attribute lists:

```python
for attr in ('obj', 'left', 'right', 'operand', 'cond', ...):
    if hasattr(expr, attr):
        visit(getattr(expr, attr))
```

## Target Structure

```
src/middleend/
├── __init__.py          # exports analyze()
├── analyze.py           # Entry point, pass orchestration
├── walk.py              # Generic traversal infrastructure
├── reassignment.py      # Variable reassignment analysis
├── initial_value.py     # Initial value usage analysis
├── hoisting.py          # Variable hoisting analysis
└── unused.py            # Unused tuple target analysis
```

## Design: Generic Walker + Focused Passes

### walk.py — Traversal Infrastructure

```python
from typing import Iterator
from src.ir import (
    Stmt, Expr, If, While, ForRange, ForClassic, Block,
    TryCatch, Match, TypeSwitch, Var
)

def stmt_children(stmt: Stmt) -> list[list[Stmt]]:
    """Return child statement lists for recursion."""
    if isinstance(stmt, If):
        result = [stmt.then_body, stmt.else_body]
        if stmt.init:
            result.insert(0, [stmt.init])
        return result
    elif isinstance(stmt, While):
        return [stmt.body]
    elif isinstance(stmt, ForRange):
        return [stmt.body]
    elif isinstance(stmt, ForClassic):
        result = [stmt.body]
        if stmt.init:
            result.insert(0, [stmt.init])
        return result
    elif isinstance(stmt, Block):
        return [stmt.body]
    elif isinstance(stmt, TryCatch):
        return [stmt.body, stmt.catch_body]
    elif isinstance(stmt, (Match, TypeSwitch)):
        return [case.body for case in stmt.cases] + [stmt.default]
    return []


def iter_stmts(stmts: list[Stmt]) -> Iterator[Stmt]:
    """Yield all statements recursively (pre-order)."""
    for stmt in stmts:
        yield stmt
        for child_list in stmt_children(stmt):
            yield from iter_stmts(child_list)


def expr_children(expr: Expr) -> list[Expr]:
    """Return child expressions for recursion."""
    if expr is None:
        return []
    children: list[Expr] = []
    # Single child attributes
    for attr in ('obj', 'left', 'right', 'operand', 'cond', 'then_expr',
                 'else_expr', 'expr', 'index', 'low', 'high', 'ptr',
                 'value', 'message', 'pos', 'iterable', 'length', 'capacity'):
        child = getattr(expr, attr, None)
        if child is not None:
            children.append(child)
    # List attributes
    for attr in ('args', 'elements', 'parts'):
        child_list = getattr(expr, attr, None)
        if child_list:
            children.extend(child_list)
    # Dict/entries attributes
    if hasattr(expr, 'entries'):
        entries = expr.entries
        if isinstance(entries, dict):
            children.extend(entries.values())
        elif entries:
            for item in entries:
                if isinstance(item, tuple) and len(item) == 2:
                    children.append(item[1])
    if hasattr(expr, 'fields') and isinstance(expr.fields, dict):
        children.extend(expr.fields.values())
    return children


def iter_exprs(expr: Expr) -> Iterator[Expr]:
    """Yield expression and all children recursively."""
    if expr is None:
        return
    yield expr
    for child in expr_children(expr):
        yield from iter_exprs(child)


def collect_var_refs(stmts: list[Stmt]) -> set[str]:
    """Collect all variable names referenced in statements."""
    result: set[str] = set()
    for stmt in iter_stmts(stmts):
        for expr in _stmt_exprs(stmt):
            for e in iter_exprs(expr):
                if isinstance(e, Var):
                    result.add(e.name)
    return result


def _stmt_exprs(stmt: Stmt) -> list[Expr]:
    """Return expressions directly contained in a statement (not children)."""
    exprs: list[Expr] = []
    for attr in ('value', 'cond', 'expr', 'iterable', 'target'):
        e = getattr(stmt, attr, None)
        if e is not None:
            exprs.append(e)
    return exprs
```

### Visitor Base Class (optional, rustc-style)

```python
class StmtVisitor:
    """Base visitor with default traversal. Override visit_* for custom behavior."""

    def visit(self, stmts: list[Stmt]) -> None:
        for stmt in stmts:
            self.visit_stmt(stmt)

    def visit_stmt(self, stmt: Stmt) -> None:
        method = f'visit_{type(stmt).__name__}'
        getattr(self, method, self.visit_default)(stmt)

    def visit_default(self, stmt: Stmt) -> None:
        self.super_stmt(stmt)

    def super_stmt(self, stmt: Stmt) -> None:
        """Recurse into children. Call from visit_* to continue traversal."""
        for child_list in stmt_children(stmt):
            for child in child_list:
                self.visit_stmt(child)
```

Usage:

```python
class ReturnFinder(StmtVisitor):
    def __init__(self):
        self.found = False

    def visit_Return(self, stmt: Return) -> None:
        self.found = True
        # Don't call super_stmt — stop descent

def contains_return(stmts: list[Stmt]) -> bool:
    finder = ReturnFinder()
    finder.visit(stmts)
    return finder.found
```

### Example: Simplified Analysis Pass

Before (current):

```python
def _contains_return(stmts: list[Stmt]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, Return):
            return True
        if isinstance(stmt, If):
            if _contains_return(stmt.then_body) or _contains_return(stmt.else_body):
                return True
        elif isinstance(stmt, While):
            if _contains_return(stmt.body):
                return True
        elif isinstance(stmt, ForRange):
            if _contains_return(stmt.body):
                return True
        # ... 20 more lines
    return False
```

After (with walker):

```python
def contains_return(stmts: list[Stmt]) -> bool:
    return any(isinstance(s, Return) for s in iter_stmts(stmts))
```

## Reference: How Compilers Organize Middleends

### LLVM: Analysis vs Transform Passes

LLVM cleanly separates pass types:

| Type | Purpose | Modifies IR? |
|------|---------|--------------|
| Analysis | Compute info (dominator tree, alias analysis) | No |
| Transform | Modify code (DCE, inlining) | Yes |

Analysis results are **cached** and **invalidated** when transforms run. Transform passes declare what they preserve.

Source: [LLVM Pass Infrastructure](https://www.compilersutra.com/docs/llvm/llvm_pass_tracker/llvm_pass/)

### Rust Compiler: visit_X / super_X Pattern

Rust's MIR visitor uses macro-generated traits:

```rust
impl<'tcx> Visitor<'tcx> for MyAnalysis {
    fn visit_terminator(&mut self, term: &Terminator<'tcx>, loc: Location) {
        // Custom logic
        self.super_terminator(term, loc);  // Recurse
    }
}
```

- `visit_X()` — override for custom behavior
- `super_X()` — call to recurse into children

The macro generates all boilerplate; adding new IR nodes updates one place.

Source: [MIR Visitor Guide](https://rustc-dev-guide.rust-lang.org/mir/visitor.html)

### Go Compiler: Sequential Pass List

Go's SSA backend runs passes sequentially:

1. Machine-independent passes (DCE, nil check elimination)
2. Lowering pass (abstract → machine-specific ops)
3. Architecture-specific passes

Each pass transforms one function at a time. Simple and effective.

Source: [Go SSA README](https://go.dev/src/cmd/compile/internal/ssa/README)

### Visitor vs Fold

| Pattern | Creates New Structure? | Use Case |
|---------|------------------------|----------|
| **Visitor** | No | Analysis, annotation |
| **Fold** | Yes | Transformation (AST → IR) |

Our middleend is **analysis-only**, so visitor pattern is correct.

Source: [Rust Design Patterns](https://rust-unofficial.github.io/patterns/patterns/behavioural/visitor.html)

## Design Principles

### 1. Analysis-Only (No Transforms)

The middleend computes properties; it never rewrites IR. This matches LLVM's separation:

- **Frontend**: Produces IR
- **Middleend**: Annotates IR (read-only)
- **Backend**: Consumes annotated IR

### 2. Dynamic Annotations

Annotations are added as attributes, not defined in `ir.py`:

```python
stmt.is_declaration = True  # Added by middleend
stmt.hoisted_vars = [...]   # Added by middleend
```

This keeps IR definitions clean and allows passes to evolve independently.

### 3. Single Traversal Infrastructure

All passes share `walk.py`. Adding a new IR node type requires updating one file, not every analysis.

### 4. Passes Are Independent

Each pass can run in isolation. No pass depends on another's annotations (except where documented, e.g., `_analyze_named_returns` depends on `has_catch_returns`).

## Migration Strategy

1. **Create `middleend/` package** with `__init__.py`
2. **Extract `walk.py`** — `stmt_children`, `iter_stmts`, `expr_children`, `iter_exprs`, `collect_var_refs`
3. **Refactor existing code** to use `walk.py` helpers
4. **Extract passes** one at a time:
   - `reassignment.py`
   - `initial_value.py`
   - `hoisting.py`
   - `unused.py`
5. **Create `analyze.py`** — orchestrates passes, exports `analyze()`
6. **Update imports** in `cli.py`

Each step should pass tests before proceeding. The refactor to use `walk.py` can happen before splitting into files.
