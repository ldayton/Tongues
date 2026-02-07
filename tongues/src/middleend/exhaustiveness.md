# Exhaustiveness Checking

## Problem Statement

Tongues lacks exhaustiveness checking for type-based dispatch. When switching on a
class hierarchy or Union type, there's no verification that all variants are handled.
This is a significant gap because:

1. **Self-hosting**: Tongues compiles itself. The backends use `match` statements
   on IR node types (`Expr`, `Stmt`, `Type`, `LValue`). Missing cases cause runtime
   `AttributeError` or silent bugs, not compile-time errors.

2. **Target language support**: Many targets (Rust, Swift, Java sealed classes) have
   native exhaustiveness. We should verify exhaustiveness in the frontend so backends
   can emit clean `match`/`switch` without synthetic defaults.

3. **Closed hierarchies**: The IR defines closed hierarchies. `Type` has exactly 15
   subtypes, `Stmt` has 22, `Expr` has 62, `LValue` has 5. These are known at compile
   time and should be checkable.

## Existing Infrastructure

### Phase 7: `hierarchy.py`

Already computes:
- `StructInfo.bases`: parent classes for each struct
- `SubtypeRel.node_types`: set of all Node subclass names
- `is_node_subclass()`: check if A is subtype of B
- Single inheritance guarantee (tree, not DAG)

### Phase 9: `lowering.py`

Converts `isinstance` chains to `TypeSwitch`:
```python
if isinstance(x, IntLit):
    ...
elif isinstance(x, FloatLit):
    ...
else:
    ...
```
Becomes:
```
TypeSwitch(expr=x, binding="x", cases=[TypeCase(IntLit), TypeCase(FloatLit)], default=[...])
```

### IR Definitions

```python
class TypeSwitch(Stmt):
    expr: Expr              # Expression being switched on
    binding: str            # Variable name bound in each case
    cases: list[TypeCase]   # Type-specific branches
    default: list[Stmt]     # Fallback (may be empty)

class TypeCase:
    typ: Type               # The type for this case
    body: list[Stmt]        # Statements to execute

class Union(Type):
    name: str
    variants: tuple[StructRef, ...]  # All possible types
```

## Required Changes

### 1. Build Subtype Index (Phase 7)

Add to `hierarchy.py`:

```python
def build_subtype_index(symbols: SymbolTable) -> dict[str, set[str]]:
    """Map each base class to all its concrete subtypes.

    For exhaustiveness, we need the inverse of the 'bases' relation.
    Given 'Expr', returns {'IntLit', 'FloatLit', 'StringLit', ...}.
    Only includes concrete (non-abstract) types.
    """
    index: dict[str, set[str]] = {}
    for name, info in symbols.structs.items():
        # Walk up inheritance chain, adding self to each ancestor's set
        current = name
        while current:
            parent_info = symbols.structs.get(current)
            if not parent_info:
                break
            for base in parent_info.bases:
                if base not in index:
                    index[base] = set()
                # Only add concrete types (those that can be instantiated)
                if not is_abstract(name, symbols):
                    index[base].add(name)
            current = parent_info.bases[0] if parent_info.bases else None
    return index

def is_abstract(name: str, symbols: SymbolTable) -> bool:
    """A class is abstract if it has subtypes but is never instantiated directly."""
    # Heuristic: if a class is only used as a base, it's abstract
    # More precise: check if class has abstract methods or is in bases but never constructed
    info = symbols.structs.get(name)
    if not info:
        return False
    # Check if any other class inherits from this
    for other_name, other_info in symbols.structs.items():
        if name in other_info.bases:
            return True  # Has children, likely abstract
    return False
```

Update `SubtypeRel`:
```python
@dataclass
class SubtypeRel:
    hierarchy_root: str | None = None
    node_types: set[str] = field(default_factory=set)
    exception_types: set[str] = field(default_factory=set)
    base_to_subtypes: dict[str, set[str]] = field(default_factory=dict)  # NEW

    def concrete_subtypes(self, base: str) -> set[str]:
        """Return all concrete types that are subtypes of base."""
        return self.base_to_subtypes.get(base, set())
```

### 2. Exhaustiveness Analysis (New Phase)

Create `middleend/exhaustiveness.py`:

```python
"""Phase 12.5: Exhaustiveness checking for type switches."""

from src.ir import (
    Module, Function, TypeSwitch, TypeCase, Match, MatchCase,
    StructRef, Union, Type, Stmt, If, Block, ForRange, ForClassic,
    While, TryCatch,
)

@dataclass
class ExhaustivenessResult:
    """Result of exhaustiveness analysis."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

def analyze_exhaustiveness(module: Module) -> ExhaustivenessResult:
    """Check all TypeSwitch and Match statements for exhaustiveness."""
    result = ExhaustivenessResult()
    subtypes = module.symbols.subtype_rel.base_to_subtypes

    for func in module.functions:
        check_function(func, subtypes, result)
    for struct in module.structs:
        for method in struct.methods:
            check_function(method, subtypes, result)

    return result

def check_function(func: Function, subtypes: dict[str, set[str]], result: ExhaustivenessResult):
    """Check all type switches in a function."""
    for stmt in walk_stmts(func.body):
        if isinstance(stmt, TypeSwitch):
            check_type_switch(stmt, subtypes, result)
        elif isinstance(stmt, Match):
            check_match(stmt, subtypes, result)

def check_type_switch(ts: TypeSwitch, subtypes: dict[str, set[str]], result: ExhaustivenessResult):
    """Verify TypeSwitch covers all subtypes or has a default."""
    expr_type = ts.expr.typ

    # Get the base type being switched on
    if isinstance(expr_type, StructRef):
        base_name = expr_type.name
    elif isinstance(expr_type, Union):
        # Union: check all variants are covered
        check_union_switch(ts, expr_type, result)
        return
    else:
        return  # Can't check exhaustiveness for other types

    # Get all concrete subtypes of the base
    required = subtypes.get(base_name, set())
    if not required:
        return  # No known subtypes, can't check

    # Collect covered types
    covered: set[str] = set()
    for case in ts.cases:
        if isinstance(case.typ, StructRef):
            covered.add(case.typ.name)
            # Also add subtypes of the covered type
            covered.update(subtypes.get(case.typ.name, set()))

    # Check for missing types
    missing = required - covered

    if missing and not ts.default:
        loc = f"{ts.loc.line}:{ts.loc.col}" if ts.loc else "?"
        result.errors.append(
            f"{loc}: Non-exhaustive TypeSwitch on '{base_name}': "
            f"missing {sorted(missing)}"
        )
    elif not missing and ts.default:
        loc = f"{ts.loc.line}:{ts.loc.col}" if ts.loc else "?"
        result.warnings.append(
            f"{loc}: TypeSwitch on '{base_name}' is exhaustive but has unreachable default"
        )

def check_union_switch(ts: TypeSwitch, union: Union, result: ExhaustivenessResult):
    """Check that all Union variants are covered."""
    required = {v.name for v in union.variants}
    covered = {case.typ.name for case in ts.cases if isinstance(case.typ, StructRef)}
    missing = required - covered

    if missing and not ts.default:
        loc = f"{ts.loc.line}:{ts.loc.col}" if ts.loc else "?"
        result.errors.append(
            f"{loc}: Non-exhaustive TypeSwitch on union '{union.name}': "
            f"missing {sorted(missing)}"
        )

def walk_stmts(stmts: list[Stmt]):
    """Yield all statements recursively."""
    for stmt in stmts:
        yield stmt
        if isinstance(stmt, If):
            yield from walk_stmts(stmt.then_body)
            yield from walk_stmts(stmt.else_body)
        elif isinstance(stmt, TypeSwitch):
            for case in stmt.cases:
                yield from walk_stmts(case.body)
            yield from walk_stmts(stmt.default)
        elif isinstance(stmt, Match):
            for case in stmt.cases:
                yield from walk_stmts(case.body)
        elif isinstance(stmt, (ForRange, ForClassic, While)):
            yield from walk_stmts(stmt.body)
        elif isinstance(stmt, Block):
            yield from walk_stmts(stmt.body)
        elif isinstance(stmt, TryCatch):
            yield from walk_stmts(stmt.try_body)
            for clause in stmt.catch_clauses:
                yield from walk_stmts(clause.body)
            yield from walk_stmts(stmt.finally_body)
```

### 3. Integration with Pipeline

Update `middleend/__init__.py`:
```python
from .exhaustiveness import analyze_exhaustiveness

def analyze(module: Module) -> None:
    analyze_scope(module)
    analyze_liveness(module)
    analyze_returns(module)
    analyze_hoisting(module)
    analyze_callbacks(module)
    analyze_ownership(module)

    # Exhaustiveness checking (errors are fatal)
    result = analyze_exhaustiveness(module)
    for error in result.errors:
        print(f"error: {error}", file=sys.stderr)
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if result.errors:
        sys.exit(1)
```

## Subset Checking Interactions

### Current State

`subset.py` (Phase 3) runs before type inference. It validates syntax constraints
but cannot check exhaustiveness because:
1. Types aren't resolved yet
2. Class hierarchy isn't built
3. `isinstance` chains aren't lowered to `TypeSwitch`

### Required Changes to Subset

The subset checker should enforce patterns that ENABLE exhaustiveness checking:

```python
# In subset.py, add to visit_Match():

def visit_Match(self, node: ASTNode) -> None:
    """Validate match statement patterns."""
    cases = node.get("cases", [])

    # Check for wildcard/default case
    has_wildcard = False
    for case in cases:
        pattern = case.get("pattern", {})
        if is_wildcard_pattern(pattern):
            has_wildcard = True
            # Wildcard must be last
            if case != cases[-1]:
                self.error(case, "match", "wildcard pattern must be last case")

    # If matching on a typed variable (isinstance pattern), require exhaustive or wildcard
    subject = node.get("subject", {})
    if is_name(subject) and not has_wildcard:
        # Can't check exhaustiveness here (no types), but warn about potential issue
        self.warning(node, "match",
            "match without wildcard case may be non-exhaustive; "
            "exhaustiveness will be verified after type inference")

def is_wildcard_pattern(pattern: ASTNode) -> bool:
    """Check if pattern is a wildcard (_) or captures all (case x:)."""
    ptype = pattern.get("_type", "")
    if ptype == "MatchAs" and pattern.get("pattern") is None:
        return True  # case _: or case x:
    return False
```

### Annotation for Intentional Non-Exhaustiveness

For cases where non-exhaustive matching is intentional (e.g., handling only some
variants and raising for others), add a pragma or annotation:

```python
# In source code:
match node:  # @non_exhaustive
    case IntLit():
        ...
    case FloatLit():
        ...
    # Intentionally not handling all Expr types

# Or explicit else with raise:
match node:
    case IntLit():
        ...
    case _:
        raise ValueError(f"Unexpected: {node}")  # Explicit handling
```

The subset checker should recognize the raise-in-default pattern as intentional.

## Self-Hosting Verification

Since Tongues compiles itself, the exhaustiveness checker will verify the backends:

1. **ruby.py `_expr` method**: Has ~80 cases for Expr types. Should cover all 62.
2. **ruby.py `_emit_stmt` method**: Has ~20 cases for Stmt types. Should cover all 22.
3. **All backends**: Similar patterns.

Expected findings when first enabled:
- Missing cases in some backends (bugs!)
- Unreachable default branches (dead code)
- Intentional partial handling that needs annotation

## Implementation Order

1. **Phase 1**: Add `base_to_subtypes` index to `hierarchy.py` and `SubtypeRel`
2. **Phase 2**: Create `exhaustiveness.py` with basic TypeSwitch checking
3. **Phase 3**: Integrate into middleend pipeline, initially as warnings only
4. **Phase 4**: Add subset.py validation for match patterns
5. **Phase 5**: Promote warnings to errors, fix all backends
6. **Phase 6**: Add Union exhaustiveness checking
7. **Phase 7**: Add pragma support for intentional non-exhaustiveness

## Testing

```python
# test_exhaustiveness.py

def test_exhaustive_switch():
    """TypeSwitch covering all subtypes should pass."""
    code = '''
class Base: pass
class A(Base): pass
class B(Base): pass

def foo(x: Base) -> int:
    if isinstance(x, A):
        return 1
    elif isinstance(x, B):
        return 2
    '''
    # Should pass - all subtypes covered

def test_non_exhaustive_error():
    """TypeSwitch missing subtypes without default should error."""
    code = '''
class Base: pass
class A(Base): pass
class B(Base): pass
class C(Base): pass

def foo(x: Base) -> int:
    if isinstance(x, A):
        return 1
    elif isinstance(x, B):
        return 2
    # Missing C!
    '''
    # Should error: "missing {'C'}"

def test_non_exhaustive_with_default():
    """TypeSwitch with default is always valid."""
    code = '''
class Base: pass
class A(Base): pass
class B(Base): pass

def foo(x: Base) -> int:
    if isinstance(x, A):
        return 1
    else:
        return 0  # Covers B and any future subtypes
    '''
    # Should pass - has default
```

## Future Extensions

1. **Match on literal values**: Check enum-like patterns (kind fields)
2. **Tuple matching**: Verify all combinations covered
3. **Guard analysis**: Account for guards in exhaustiveness
4. **IDE integration**: Provide "add missing cases" quick-fix
5. **Documentation generation**: Auto-generate "handles: X, Y, Z" docs
