# Fix Missing Narrowing in the Taytsh Checker

`just self-transpile` produces 675 errors. ~226 are narrowing-related. The spec
(`spec/13-type-checking.md`) defines what narrowing the checker must support. Some
errors are checker bugs (not implementing the spec correctly). Others require the
transpiler's lowering pass to emit Taytsh patterns the checker can already narrow.

## Fix 1: Assignment must check against declared type, not narrowed type

**Category:** Checker bug

**File:** `tongues/src/taytsh/check.py`, `check_assign_stmt` (line 1591)

**Bug:** After guard clause narrowing (e.g., `if r != nil { return }`), the checker
narrows `r` to `nil` in the continuing scope. When `r` is later reassigned
(`r = scan_...()`), `check_assign_stmt` calls `self.check_expr(stmt.target, None)`
at line 1605, which returns the **narrowed** type (`nil`). The assignment of
`string?` to `nil` then fails.

Narrowing creates a scope overlay for reads — it doesn't change the declaration. The
spec says assignment validates against "the target's type," which is the declared
type. The checker is incorrectly using the narrowed type.

**Fix:** When the target is a `TVar`, resolve the **declared** type (the original
binding, skipping narrowing overlays in the current scope) for the compatibility
check. After a successful assignment, call `self.narrow(name, val_type)` so
subsequent reads see the new type.

Concretely: add a `lookup_declared(name)` method that walks scopes but skips
entries added by `narrow()` (or walks from the declaring scope outward). Use it in
`check_assign_stmt` for the `is_assignable` check.

**Errors fixed:** ~8 — `tongues/src/middleend/scope.py` lines 773-876 (repeated
`r = scan_...; if r != nil { return r }` pattern), `tongues/src/taytsh/check.py`
lines 3146-3162.

---

## Fix 2: Definite assignment through if/elif/else with exit branches

**Category:** Checker bug

**File:** `tongues/src/taytsh/check.py`, `check_if_stmt` (lines 1746-1772)

**Spec reference:** Definite assignment rules (spec lines 203-211):
> After `if`/`else`, a variable is initialized only if it was initialized in
> **all** branches. After `if` without `else`, the variable remains uninitialized
> unless the then-body always exits.

**Bug:** The checker's `uninitialized` set merging doesn't handle nested if/elif
chains correctly. When code does:

```
if cond_a {
    x = val_a
} else {
    if cond_b {
        x = val_b
    } else {
        continue  // or return/break
    }
}
use(x)  // "variable used before assignment"
```

The transpiler flattens `if/elif/else` into nested `TIfStmt`. The outer if/else
both complete (the else's inner if covers all paths), but the checker's merging at
the outer level may not recognize that the nested else-body is complete.

**Fix:** Verify that `_block_is_complete` correctly handles the nested if pattern,
and that `check_if_stmt`'s uninitialized merging propagates inner results to the
outer level. The logic at lines 1746-1772 should already handle this if
`_block_is_complete` returns `True` for the else body — verify and fix if not.

Also review while-loop cases: `tongues/src/taytsh/emit.py` lines 290-310 and
655-660 have a `while current != nil` loop where a variable is assigned on every
non-`continue` path. The spec says "variables initialized only inside a loop body
are still considered uninitialized after the loop." These may need source-side
initialization (e.g., `let final_else: list[TStmt]? = nil`) rather than checker
changes.

**Errors fixed:** ~26 across `tongues/src/taytsh/emit.py`,
`tongues/src/frontend/parse.py`, `tongues/src/frontend/lowering.py`,
`tongues/src/frontend/subset.py`, `tongues/src/taytsh/check.py`.

---

## Fix 3: Multi-level field access path narrowing

**Category:** Checker bug (incomplete implementation of spec)

**File:** `tongues/src/taytsh/check.py`, `_istype_var_from_call` (line 620) and
`_lookup_field_type` (line 1092)

**Spec reference:** (spec lines 466-476):
> `IsType` supports dotted paths as well as simple variables:
> ```
> if IsType(c.func, "Var") {
>     return c.func.name
> }
> ```
> The narrowed path `c.func` is tracked and subsequent field accesses through it
> resolve against the narrowed type.

The spec says dotted paths are tracked. After `IsType(c.func, "Var")` narrows
`c.func` to `Var`, accessing `c.func.name` should resolve `.name` on `Var`. This
implies the checker must resolve field accesses on narrowed multi-component paths.

**Bug:** `_istype_var_from_call` only handles depth 1 (`TFieldAccess(obj=TVar)`).
A 2-level path like `cond.left.obj` (which is
`TFieldAccess(obj=TFieldAccess(obj=TVar("cond"), field="left"), field="obj")`) fails
because `first.obj` is a `TFieldAccess`, not a `TVar`. Similarly,
`_lookup_field_type` only handles 2-component paths (`"a.b"`), so looking up
`"cond.left.name"` (3 components) fails.

**Fix:**
1. Make `_istype_var_from_call` recursively build dotted paths from nested
   `TFieldAccess` nodes: walk up the chain collecting field names until hitting a
   `TVar` root, then join with `.`.
2. Extend `_lookup_field_type` to handle N-component paths: resolve each component
   by looking up the type at that level and resolving the next field.
3. Extend `_lookup_narrowed_path` to check prefixes (if `"cond.left"` is narrowed,
   `"cond.left.name"` should resolve `.name` on the narrowed type).

**Errors fixed:** ~10 — `tongues/src/taytsh/check.py` lines 547-583,
`tongues/src/backend/perl.py` lines 1600-1643.

---

## Fix 4: Lowering must emit narrowable patterns for isinstance

**Category:** Transpiler change (lowering pass)

**File:** `tongues/src/frontend/lowering.py`

The spec defines narrowing for `IsType(var, "T")` and `IsType(var.field, "T")` —
simple variables and dotted paths. The spec does **not** define narrowing for
indexed expressions (`IsType(body[0], "T")`), compound `||` conditions, or other
complex forms.

The Python source uses patterns the checker can't narrow:

1. `isinstance(body[0], TExprStmt)` → `IsType(body[0], "TExprStmt")` — indexed
   expression, not a variable or dotted path
2. `not isinstance(x, T) or x.op not in (...)` → `!IsType(x, "T") || ...` —
   compound `||` guard, spec only defines negated `IsType` guards
3. `isinstance(args[1].value, TStringLit)` → `IsType(args[1].value, "TStringLit")`
   — indexed + field access, not a simple dotted path

**Fix:** The lowering pass should restructure these into patterns the checker
handles. Specifically:

### 4a. Indexed isinstance → temp variable

When lowering `isinstance(container[i], T)` where the result is used in an `if`
condition and `container[i]` is accessed in the body, introduce a temp:

```python
# Before lowering:
if isinstance(body[0], TExprStmt):
    call = body[0].expr

# After lowering (Taytsh):
let _t0: TStmt = body[0]
if IsType(_t0, "TExprStmt") {
    let call: TExpr = _t0.expr
}
```

This is a lowering-phase transformation. The emitted Taytsh uses only spec-defined
narrowing (simple variable + IsType).

### 4b. Compound `||` guards → nested ifs

When lowering `if not isinstance(x, T) or other_cond: return`, restructure:

```python
# Before lowering:
if not isinstance(cond, TBinaryOp) or cond.op not in ("!=", "=="):
    return ...

# After lowering (Taytsh):
if !IsType(cond, "TBinaryOp") {
    return ...
}
if cond.op != "!=" && cond.op != "==" {
    return ...
}
```

Each guard is now a simple negated `IsType` guard (spec-defined) followed by a
field-level check on the narrowed variable.

### 4c. Indexed + field access → temp variable for the indexed part

```python
# Before lowering:
if isinstance(args[1].value, TStringLit):
    pat = args[1].value.value

# After lowering (Taytsh):
let _arg1_val: TExpr = args[1].value
if IsType(_arg1_val, "TStringLit") {
    let pat: string = _arg1_val.value
}
```

**Scope:** These transformations happen during the isinstance lowering in
`_lower_expr` (around line 2181) and the if-statement lowering. They only trigger
when the isinstance argument is not a simple variable or dotted path.

**Errors fixed:** ~55 across `tongues/src/backend/perl.py`,
`tongues/src/backend/python.py`, `tongues/src/backend/ruby.py`,
`tongues/src/middleend/returns.py`, `tongues/src/middleend/callgraph.py`,
`tongues/src/taytsh/check.py`, `tongues/src/taytsh/emit.py`.

---

## Fix 5: Ternary nil narrowing result type

**Category:** Checker bug (investigate)

**File:** `tongues/src/taytsh/check.py`, `check_ternary` (around line 2527)

**Spec reference:** (spec line 362):
> If both branches have the same type, the result is that type. Otherwise, the
> result is the normalized union of both branch types.

The spec also says narrowing applies in ternary branches (spec line 462).

**Bug:** The pattern `x if x is not None else default` should produce type `T` when
both branches yield `T`. If the checker narrows `x` to `T` in the then-branch (from
the `!= nil` condition) and the else-branch also produces `T`, the union of `T` and
`T` is `T`. This should already work.

**Fix:** Investigate whether the checker's ternary handling actually applies nil
narrowing to the then-expression. If it does, these errors may be caused by
something else (e.g., the else-expression producing `T | nil` because the variable
used there wasn't narrowed). Trace a specific case:
`tongues/src/taytsh/check.py` line 3133 —
`elem_expected if elem_expected is not None else first`.

**Errors fixed:** ~6 across `tongues/src/taytsh/check.py`,
`tongues/src/frontend/lowering.py`, `tongues/src/middleend/strings.py`.

---

## Priority Order

| Priority | Fix | Errors | Where |
|----------|-----|--------|-------|
| 1 | Fix 1: assignment vs narrowed type | ~8 | Checker |
| 2 | Fix 2: definite assignment | ~26 | Checker |
| 3 | Fix 3: multi-level dotted paths | ~10 | Checker |
| 4 | Fix 4: lowering narrowable patterns | ~55 | Transpiler lowering |
| 5 | Fix 5: ternary narrowing | ~6 | Checker (investigate) |

Fixes 1-3 and 5 are checker-only. Fix 4 is transpiler-only. All emit or validate
Taytsh as defined in the spec — no spec extensions.

## Testing

After each fix:
```bash
just self-transpile 2>&1 | grep -c '\[check\]'    # track total (currently 675)
uv run pytest tests/test_check_gen.py -v            # checker regression tests
```

## Key Code Locations

### Checker (`tongues/src/taytsh/check.py`)

| What | Line |
|------|------|
| Scope stack / `narrow()` / `lookup()` | 1024-1090 |
| `_lookup_narrowed_path` | 1104 |
| `_lookup_field_type` | 1092 |
| `_narrow_to_type` | 1113 |
| `_nil_check_var` | 533 |
| `_collect_nil_checks` | 587 |
| `_istype_var_from_call` | 620 |
| `_type_check_var` | 638 |
| `_collect_type_checks` | 655 |
| Guard narrowing in `check_stmts` | 1442-1502 |
| `check_if_stmt` nil/IsType narrowing | 1710-1744 |
| `check_if_stmt` uninitialized merging | 1746-1772 |
| `check_assign_stmt` | 1591-1615 |
| `check_ternary` | 2527-2570 |
| `&&`/`\|\|` short-circuit narrowing | 2254-2316 |
| `_block_is_complete` | 461 |
| `_body_always_exits` | 525 |

### Transpiler lowering (`tongues/src/frontend/lowering.py`)

| What | Line |
|------|------|
| isinstance → IsType lowering | 2181 |
| if-statement lowering | (search `_lower_if`) |
