# Frontend Architecture

The frontend (`src/frontend.py`) converts Python AST to language-agnostic IR. It handles all Python-specific semantics so backends only emit syntax.

```
source.py → [Python AST] → Frontend → [IR] → Middleend → Backend → target code
```

## Current State

Single 2,900-line file with interleaved concerns:

| Line Range | Concern | Lines |
|------------|---------|-------|
| 57-125 | Constants (`TYPE_MAP`, `KIND_TO_CLASS`) | ~70 |
| 127-148 | `TypeContext` + `Frontend.__init__` | ~20 |
| 149-346 | Collection passes | ~200 |
| 347-528 | IR building (`_build_*`) | ~180 |
| 536-797 | Type parsing (`_py_type_to_ir`, etc.) | ~260 |
| 798-1635 | Type inference | ~840 |
| 1636-2430 | Expression lowering | ~800 |
| 2432-2897 | Statement lowering | ~465 |

## Target Structure

```
src/frontend/
├── __init__.py       # exports Frontend
├── frontend.py       # Frontend class, transpile(), orchestration (~100 lines)
├── context.py        # FrontendState, TypeContext (shared mutable state)
├── collect.py        # Pass 1-5: class names, signatures, fields, constants
├── types.py          # _py_type_to_ir, _annotation_to_str, TYPE_MAP
├── inference.py      # _collect_var_types, _synthesize_type, _coerce
├── lower_expr.py     # Expression lowering
├── lower_stmt.py     # Statement lowering
└── constants.py      # KIND_TO_CLASS, domain constants
```

## Design: Module Functions + Shared Context

Use module-level functions with explicit context, not mixins or heavy OOP composition.

```python
# context.py
@dataclass
class FrontendState:
    symbols: SymbolTable
    node_types: set[str]
    current_func_info: FuncInfo | None = None
    current_class_name: str = ""
    type_ctx: TypeContext = field(default_factory=TypeContext)

@dataclass
class TypeContext:
    expected: Type | None = None
    var_types: dict[str, Type] = field(default_factory=dict)
    return_type: Type | None = None
    tuple_vars: dict[str, list[str]] = field(default_factory=dict)
    sentinel_ints: set[str] = field(default_factory=set)
    narrowed_vars: set[str] = field(default_factory=set)
```

```python
# lower_expr.py
from . import ir
from .context import FrontendState

def lower_expr(state: FrontendState, node: ast.expr) -> ir.Expr:
    handler = EXPR_DISPATCH.get(type(node))
    if handler:
        return handler(state, node)
    raise InternalError(f"unhandled expression: {type(node).__name__}")

def lower_expr_Call(state: FrontendState, node: ast.Call) -> ir.Expr:
    ...

def lower_expr_Name(state: FrontendState, node: ast.Name) -> ir.Expr:
    ...

EXPR_DISPATCH: dict[type, Callable] = {
    ast.Call: lower_expr_Call,
    ast.Name: lower_expr_Name,
    ast.Attribute: lower_expr_Attribute,
    ast.Subscript: lower_expr_Subscript,
    ast.BinOp: lower_expr_BinOp,
    ast.Compare: lower_expr_Compare,
    ast.BoolOp: lower_expr_BoolOp,
    ast.UnaryOp: lower_expr_UnaryOp,
    ast.IfExp: lower_expr_IfExp,
    ast.List: lower_expr_List,
    ast.Dict: lower_expr_Dict,
    ast.Set: lower_expr_Set,
    ast.Tuple: lower_expr_Tuple,
    ast.JoinedStr: lower_expr_JoinedStr,
    ast.Constant: lower_expr_Constant,
}
```

```python
# frontend.py
from . import collect, build, lower_expr, lower_stmt
from .context import FrontendState

class Frontend:
    def __init__(self):
        self.state = FrontendState(symbols=SymbolTable(), node_types=set())

    def transpile(self, source: str) -> Module:
        tree = ast.parse(source)
        collect.collect_all(self.state, tree)
        return build.build_module(self.state, tree)
```

## Rationale

### Why Not Mixins?

```python
# Mixins create implicit dependencies and diamond inheritance risk
class Frontend(CollectMixin, BuildMixin, TypesMixin, InferenceMixin, LowerExprMixin, LowerStmtMixin):
    ...
```

- Hard to trace where methods come from
- `self` becomes a god object
- Testing requires instantiating the full class

### Why Not Heavy Composition?

```python
# Composition adds indirection without benefit here
class Frontend:
    def __init__(self):
        self._collector = Collector(self)
        self._builder = Builder(self)
        self._inference = TypeInference(self)
        self._expr_lowerer = ExprLowerer(self)
        self._stmt_lowerer = StmtLowerer(self)
```

- Extra boilerplate for object creation
- Still tightly coupled (each component needs `self` reference)
- Python functions are the natural unit

### Why Module Functions?

- **Explicit dependencies**: State passed in, not accessed via `self`
- **Testable**: Each module can be tested with mock state
- **Fast dispatch**: Dict lookup beats `getattr(self, f"_lower_expr_{name}")`
- **Matches Python idiom**: How stdlib and most Python projects organize

## Reference: How Other Transpilers Organize

### Mypyc (Python → C)

```
mypyc/
├── ir/           # IR definitions only
├── irbuild/      # AST → IR (main.py is entry point)
├── transform/    # IR → IR passes (uninit, exceptions, refcount)
├── primitives/   # Primitive type operations
├── codegen/      # IR → C
└── analysis/     # Semantic analysis
```

Key insight: `irbuild/` is a package with multiple modules, not a single file. Top-level logic in `main.py`, with helpers split out.

Source: [mypyc developer docs](https://github.com/python/mypy/blob/master/mypyc/doc/dev-intro.md)

### Cython (Python/Cython → C)

```
Compiler/
├── Nodes.py              # Statement nodes
├── ExprNodes.py          # Expression nodes
├── Visitor.py            # Visitor pattern base
├── ParseTreeTransforms.py # Tree transform passes
└── TypeInference.py      # Type inference
```

Key insight: Statements and expressions in separate files. Visitor pattern with `visit_<NodeType>` naming.

Source: [Cython internals](https://docs.cython.org/en/latest/src/devguide/cython_internals.html)

### Common Patterns

| Pattern | Description |
|---------|-------------|
| **Nodes split by kind** | Statements separate from expressions |
| **Visitor dispatch** | Method/function naming `visit_<NodeType>` or `lower_<NodeType>` |
| **Metadata in maps** | `dict[NodeId, Metadata]` separate from AST for pass decoupling |
| **Lowering assumes correctness** | After type check, unexpected input is compiler bug (panic) |
| **Primitives separate** | Built-in operations in their own module |

Source: [Thunderseethe on lowering](https://thunderseethe.dev/posts/lowering-base-ir/)

## Module Responsibilities

### `constants.py`
Domain constants that don't change:
- `TYPE_MAP`: Python type names → IR primitives
- `KIND_TO_CLASS`: `.kind` string values → class names

### `context.py`
Shared mutable state:
- `FrontendState`: Symbols, node types, current function/class context
- `TypeContext`: Type inference state (expected type, var types, narrowing)

### `collect.py`
Five collection passes over the AST:
1. Collect class names and inheritance
2. Mark Node subclasses
3. Collect function/method signatures
4. Collect struct fields
5. Collect module constants

### `types.py`
Python type annotation → IR type conversion:
- `annotation_to_str()`: AST annotation → string
- `py_type_to_ir()`: Type string → IR Type
- `parse_callable_type()`: `Callable[[A, B], R]` → FuncType
- `split_union_types()`, `split_type_args()`: Type string parsing

### `inference.py`
Type inference and synthesis:
- `collect_var_types()`: Pre-scan function body for variable types
- `infer_iterable_type()`: Element type from for loops
- `synthesize_type()`: Type of lowered expression
- `coerce()`: Insert type conversions

### `lower_expr.py`
Expression lowering (AST expr → IR Expr):
- One function per AST expression type
- `EXPR_DISPATCH` dict for fast lookup
- Handles Python-specific patterns (string methods, built-ins, tuple indexing)

### `lower_stmt.py`
Statement lowering (AST stmt → IR Stmt):
- One function per AST statement type
- `STMT_DISPATCH` dict for fast lookup
- Handles isinstance chains → TypeSwitch conversion

### `frontend.py`
Orchestration only:
- `Frontend` class with `transpile()` method
- Coordinates passes: collect → build
- ~100 lines

## Migration Strategy

1. **Create `frontend/` package** with `__init__.py`
2. **Extract `constants.py`** (no dependencies, easy)
3. **Extract `context.py`** (TypeContext, FrontendState)
4. **Extract `types.py`** (type conversion, depends on constants)
5. **Extract `collect.py`** (collection passes, depends on types)
6. **Extract `inference.py`** (type inference, depends on types)
7. **Extract `lower_expr.py`** (expression lowering, depends on inference)
8. **Extract `lower_stmt.py`** (statement lowering, depends on lower_expr)
9. **Reduce `frontend.py`** to orchestration
10. **Update imports** in `cli.py` and tests

Each step should pass tests before proceeding.

## Frontend Responsibilities

The frontend (Python AST → IR) performs all analysis:

1. **Type inference** — Resolve types from annotations and usage
2. **Symbol resolution** — Build table of structs, methods, functions
3. **Scope analysis** — Variable lifetimes, hoisting requirements
4. **Ownership analysis** — Mark `Pointer.owned` for Rust/C (see Ownership Model)
5. **Method resolution** — Fill `MethodCall.receiver_type`
6. **Nil analysis** — `x is None` → `IsNil`
7. **Truthiness** — `if items:` → `if len(items) > 0`
8. **Type narrowing** — `isinstance` → `TypeSwitch` (see below)
9. **Fallibility analysis** — Mark functions containing `raise` as `fallible=true`

### Type Narrowing

Convert `isinstance` chains to `TypeSwitch`:

```python
# Python source
if isinstance(node, CommandSubstitution):
    process(node.command)
elif isinstance(node, ArithBinaryOp) or isinstance(node, ArithComma):
    process_binary(node)
```

```
// IR
TypeSwitch {
    expr: node
    binding: "node"
    cases: [
        TypeCase { typ: StructRef("CommandSubstitution"), body: [...] }
        TypeCase { typ: Union("", [StructRef("ArithBinaryOp"), StructRef("ArithComma")]), body: [...] }
    ]
}
```

**Collection element types:** When a field is annotated `list[Node]` but usage shows specific types, the frontend narrows:

| Annotation          | Actual Usage          | Resolved Type                         |
| ------------------- | --------------------- | ------------------------------------- |
| `list[Node]`        | Only `Word` assigned  | `Slice(StructRef("Word"))`            |
| `list[Node]`        | Mixed expansion types | `Slice(Union("Expansion", [...]))`    |
| `ArithNode \| None` | Type alias            | `Optional(Union("ArithNode", [...]))` |

Resolve Python `Union[...]` and `X | Y` type aliases to IR `Union` with explicit variants.

### Type Inference

Two-pass local inference. All function signatures are annotated; only local variables and expressions need inference.

**Pass 1 — Symbol Collection:**
- Struct definitions → field names and types
- Function signatures → parameter types, return types
- Module constants → names and types
- Union type aliases → variant lists

**Pass 2 — Expression Typing:**

Traverse each function body. Compute `Expr.typ` bottom-up:

| Expression | Type Rule |
|------------|-----------|
| `IntLit`, `FloatLit`, `StringLit`, `BoolLit` | Literal's intrinsic type |
| `NilLit` | Context-dependent (from annotation or assignment target) |
| `Var` | Lookup in scope: parameter type or previously inferred local |
| `FieldAccess` | Lookup field type on struct |
| `Index` on `Slice(T)` | `T` |
| `Index` on `Map(K,V)` | `V` |
| `Index` on `str` | `str` (single character) |
| `Call` | Function's return type from symbol table |
| `MethodCall` | Method's return type from struct definition |
| `BinaryOp` | Fixed rules: `int + int → int`, `str + str → str`, `x == y → bool` |
| `UnaryOp` | `!bool → bool`, `-int → int` |

**Local Variable Typing:**

On first assignment `x = expr`, record `x`'s type as `expr.typ`. Subsequent assignments must have compatible type (same type or subtype). Reassignment to incompatible type is a frontend error.

**Narrowing Scope:**

Maintain a `narrowed: dict[str, Type]` map per scope. When entering `if node.kind == "command":`, set `narrowed["node"] = StructRef("Command")`. Variable lookups check `narrowed` before declared type. On scope exit, restore previous narrowing state.

**Principle:** No unification, no constraint solving. Types flow forward from annotations and initializers. This suffices because all function boundaries are annotated. See [Pierce & Turner, Local Type Inference](https://www.cis.upenn.edu/~bcpierce/papers/lti-toplas.pdf).
