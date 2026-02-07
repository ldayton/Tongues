# Phase 13: Hoisting

**Module:** `middleend/hoisting.py`

Compute variables needing hoisting for Go emission. Go requires variables to be declared before use, but Python allows first assignment in branches. This pass identifies variables that need to be hoisted to an outer scope.

Variables need hoisting when:
- First assigned inside a control structure (if/try/while/for/match)
- Used after that control structure exits

## Postconditions

- `If.hoisted_vars`, `TryCatch.hoisted_vars`, `While.hoisted_vars`, etc. contain `[(name, type)]` for variables needing hoisting
- All hoisted vars have concrete type (no `None`/`interface{}`/`any` fallback)
- Type derived from all assignment sites, not just first encountered
- String variables needing character indexing listed in `Function.rune_vars: list[str]`; Go backend emits `runes := []rune(s)` at scope entry, uses `runes[i]` thereafter

## Prior Art

- [Go variable scoping](https://go.dev/ref/spec#Declarations_and_scope)
