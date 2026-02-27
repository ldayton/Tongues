# Applying "Generating Well-Typed Terms That Are Not Useless" to Tongues

Paper: Frank, Quiring, Lampropoulos (POPL 2024)

## Current State

The existing generator (`test_check_gen.py`) implements a sophisticated top-down type-directed approach:

- `ExprGen.gen_expr(target_type, depth)` = the Palka et al. local approach
- `ScopeTracker` = the typing context Γ
- Weighted production rules with depth-limited fallbacks
- 27-boolean `FeatureVector` for coverage targeting
- 500-seed well-typed + 500-seed mutation (22 operators) + exhaustive match enumeration

This is exactly the "local" generator from the paper.

## The Problem

When generating fn literals, parameter types are chosen *before* the body:

```
pick param types → generate body → hope body uses params
```

When generating function calls, argument expressions are generated independently of how they're used inside the callee. The paper demonstrates this local approach yields ~30% parameter usage vs ~95% in real programs.

This matters less for checker correctness testing (unused args are still well-typed) but matters for:

- **Apptest quality**: used arguments exercise more codegen paths (register alloc, stack, calling conventions)
- **Codegen bug detection**: the paper found GHC strictness bugs 4x faster because used arguments trigger optimization paths
- **Self-transpile testing**: programs that look more like real code better test the pipeline
- **Mutation test sensitivity**: more data flow = more places where a type error is observable at runtime

## Concrete Enhancements

### A. Nonlocal fn-literal generation (highest impact)

The paper's `GenParam⊲` rule. Instead of picking parameter types up front:

1. Create fn literal with an *extensible* parameter list (initially empty)
2. Generate the body; when a typed hole needs filling, add a parameter of that type
3. Simultaneously add a corresponding argument at every call site using this fn value

Guarantees every fn-literal parameter is used in the body.

For `ExprGen`, this means:
- When generating a fn-typed expression, start with zero params
- While generating the body, allow a "GenParam" production that adds a fresh parameter of the needed type
- Track call sites that use this fn value and extend their argument lists in sync

Simplest implementation:

```
start with empty params → generate body → when body needs a type,
  50% chance: add a param of that type (and extend call sites)
  50% chance: use existing local approach
```

The `ScopeTracker` already supports the bookkeeping. Need a back-pointer from fn-literal bodies to their parameter lists and call sites.

### B. Nonlocal let-insertion (GenLet analog)

The paper's `GenLet` rule creates a let-binding *retroactively* when a variable of some type is needed.

Currently `StmtGen` generates `let` bindings with random types. Instead:
- When `gen_expr` needs a value of type `T` and no variable of that type is in scope, insert a `let` binding *above* the current statement
- Generate its initializer, reference it at the use site
- Guarantees the let-bound variable is used at least once

### C. Match generation guided by need (GenMatch analog)

Currently match generation picks a scrutinee type first, then generates case bodies.

The paper's `GenMatch` rule starts from a need: "I need a value of type `T`, and there's a variable `x: SomeInterface` in scope, so generate a match on `x` and fill the relevant case branch with a `T`-producing expression that uses the variant's fields."

Produces matches where case-bound variables (variant fields) are actually *used*.

### D. Argument-usage-aware weighting

After generating a program, measure what % of:
- fn-literal params
- function params
- let bindings
- match-bound variables

are actually referenced. Aim for ~95% usage rate. Feed back into weight tuning.

### E. Higher-order flow testing

The paper's arguments holes (`⊲α`) propagate type extensions through higher-order usage. When a function is passed as an argument, extending its parameter list also extends calls through the higher-order path.

The generator already has `higher_order` as a feature flag. The nonlocal approach would make higher-order generated code more interesting: a function passed to a higher-order combinator would actually *use* all its parameters at both definition and call sites.

## Implementation Priority

| Enhancement | Effort | Impact | Where it helps |
|------------|--------|--------|----------------|
| A. Nonlocal fn-literal | Medium | High | Codegen bugs, apptest quality |
| B. Nonlocal let-insertion | Low | Medium | Data flow coverage |
| C. Need-driven match gen | Medium | Medium | Match codegen, variant field usage |
| D. Usage metrics | Low | Low | Diagnostic, guides tuning |
| E. Higher-order propagation | High | Medium | Complex calling patterns |

## Key Paper Results for Reference

- Local (Palka et al.): ~30% parameter usage
- Nonlocal: 55% overall, 66% extensible lambda params, 100% let bindings
- Tunable from ~60% to ~99% for extensible lambda params
- 2x more effective, 4x faster at finding GHC-6.12 strictness analyzer bugs
- Real SML programs average 94.9% parameter usage (Table 1 in paper)
- Fully backwards-compatible: can mix local and nonlocal rules with weighted selection
- Metatheory mechanized in Coq; implementation in OCaml using mutable cells with parent pointers
