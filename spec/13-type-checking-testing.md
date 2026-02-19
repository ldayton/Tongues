# Type Checker Testing

**Module:** `taytsh/tests/check/`

Systematic testing of the Taytsh type checker (`taytsh/src/check.py`) through construction-based program generation and mutation-based oracles. The approach generates well-typed and ill-typed Taytsh programs that exercise combinations of type system features, then validates the checker's accept/reject decisions against known-correct expectations.

## Motivation

The type checker validates a large surface of interacting features: 9 primitive types, 4 collection types, unions, optionals, nil narrowing, match exhaustiveness over interfaces/enums/unions, sealed hierarchies, no-closure enforcement, scoping with no-shadowing, and 60+ overloaded built-in functions. Bugs cluster at feature intersections — a union containing an interface matched with a default binding, a nil-narrowed optional passed to an overloaded built-in, a function literal referencing a for-loop variable from an enclosing scope. Manual test cases cover known interactions but miss the combinatorial space.

## Approach

Construction-based generation with mutation oracles, drawing on Hephaestus (Chaliasos et al., PLDI 2022) for the mutation oracle model and Erwin (ICSE 2026) for bounded exhaustive enumeration of match patterns. No constraint solver is needed — Taytsh's type system is monomorphic, closed-world, and fully explicit, making construction-based generation tractable.

### Why Not CLP

Dewey et al. (ASE 2015) encode typing rules as Constraint Logic Programming predicates and use SWI-Prolog's backtracking search to generate well-typed programs. This is thorough but the encoding effort is proportional to the type system's complexity, and throughput is bounded by the solver. Taytsh's type system is simple enough that a direct Python generator maintaining type invariants by construction achieves higher throughput with less infrastructure.

### Why Not Differential Testing

There is only one Taytsh checker. Differential testing requires multiple independent implementations of the same specification. The mutation oracle provides an equivalent signal: a well-typed program that is rejected, or an ill-typed program that is accepted, is a bug regardless of any reference implementation.

## Architecture

```
Generator ──→ Well-typed program ──→ Checker ──→ accept? (expected: yes)
    │
    └──→ Mutator ──→ Ill-typed program ──→ Checker ──→ reject? (expected: yes)
```

Three components:

1. **Generator**: produces syntactically valid, well-typed Taytsh programs by construction
2. **Mutator**: applies type-breaking mutations to well-typed programs
3. **Harness**: runs the checker on generated programs and validates outcomes

### Generator

The generator builds Taytsh ASTs bottom-up, maintaining a type environment that guarantees well-typedness at every step. It does not produce text — it constructs the same AST nodes that `taytsh/src/parse.py` produces, and feeds them directly to `check()`.

#### Type Pool

Each generated program starts by assembling a **type pool** — the set of types available in the program. The pool is built by random selection from:

| Category    | Types                                                     |
| ----------- | --------------------------------------------------------- |
| Primitives  | `int`, `float`, `bool`, `byte`, `bytes`, `string`, `rune` |
| Collections | `list[T]`, `map[K, V]`, `set[T]` (T/K/V drawn from pool)  |
| Tuples      | `(T, U)`, `(T, U, V)` (elements drawn from pool)          |
| Functions   | `fn[T..., R]` (params and return drawn from pool)         |
| Optionals   | `T?` for any T in pool                                    |
| Unions      | `A                                                        | B`, `A | B | C` (members drawn from pool) |
| Structs     | 1–4 structs with 1–4 fields typed from pool               |
| Interfaces  | 0–2 interfaces, each with 2–4 implementing structs        |
| Enums       | 0–2 enums with 2–6 variants                               |

Constraints enforced during pool construction:
- Map keys and set elements are hashable
- No `T??` (double optional)
- No `void` outside return-type position
- Tuple arity ≥ 2
- Union members are distinct after normalization
- At least one struct exists (needed for throw/catch testing)

#### Declaration Generation

After the type pool, the generator produces declarations:

**Structs and interfaces** are emitted first (order-independent in Taytsh). Each struct gets fields from the pool and optionally 0–2 methods. Methods receive `self` plus 0–2 parameters.

**Enums** are emitted with their variants.

**Functions** are generated with 0–4 parameters and a return type from the pool. The body is generated statement-by-statement with access to a scope stack tracking available bindings and their types.

**Main** is always generated last, with signature `fn Main() -> void`.

#### Statement Generation

Each statement is chosen from a weighted distribution, with weights adjusted by context (e.g., `break`/`continue` only inside loops, `return` biased toward function end):

| Statement   | Weight | Preconditions                 |
| ----------- | ------ | ----------------------------- |
| `let`       | 30     | —                             |
| assignment  | 20     | mutable binding in scope      |
| `if`        | 15     | —                             |
| `while`     | 8      | depth < max                   |
| `for`       | 12     | collection or range available |
| `match`     | 10     | matchable type in scope       |
| `try/catch` | 5      | —                             |
| `throw`     | 3      | struct type available         |
| `return`    | 10     | —                             |
| expr stmt   | 15     | callable in scope             |
| `break`     | 3      | inside loop                   |
| `continue`  | 3      | inside loop                   |

`let` declarations pick a type from the pool and generate an expression of that type (or omit the initializer if the type has a zero value, with some probability). The binding is registered in the current scope, checking for shadowing violations.

#### Expression Generation

Expressions are generated top-down from a target type. The generator picks from valid productions for the requested type:

| Target Type   | Possible Productions                                            |
| ------------- | --------------------------------------------------------------- |
| `int`         | literal, variable, arithmetic, comparison result, built-in call |
| `bool`        | literal, variable, comparison, logical op, built-in call        |
| `string`      | literal, variable, Concat, built-in call                        |
| `list[T]`     | literal, variable, Sorted, Reversed, Concat, slice              |
| `T?`          | nil, expression of type T, Get call                             |
| `A \| B`      | expression of type A, expression of type B                      |
| struct        | constructor (positional or named), variable                     |
| interface     | expression of any implementing struct type                      |
| enum          | `EnumName.Variant`                                              |
| `fn[T..., R]` | function name, function literal                                 |

Recursion depth is bounded. At maximum depth, the generator falls back to literals and variables.

#### Feature Combinatorics

The generator is parameterized by a **feature vector** — a set of features that must appear in the generated program:

| Feature              | What It Forces                                           |
| -------------------- | -------------------------------------------------------- |
| `union_type`         | at least one union type in the pool                      |
| `optional_type`      | at least one optional type used                          |
| `nil_narrowing`      | an `if x != nil` with usage in the narrowed branch       |
| `match_interface`    | match on an interface with case per variant              |
| `match_enum`         | match on an enum                                         |
| `match_union`        | match on a union type                                    |
| `match_optional`     | match on an optional with nil case                       |
| `match_default`      | match with a default arm (with or without binding)       |
| `match_default_bind` | match with `default x` binding the residual              |
| `try_catch_typed`    | try/catch with a typed catch                             |
| `try_catch_all`      | try/catch with an untyped catch-all                      |
| `try_catch_union`    | catch with `A \| B` type                                 |
| `try_finally`        | try with finally block                                   |
| `fn_literal`         | at least one function literal                            |
| `fn_value`           | function used as a value (assigned to fn-typed var)      |
| `higher_order`       | function accepting fn-typed parameter                    |
| `for_collection`     | for loop over list/string/bytes/map/set                  |
| `for_range`          | for loop with range                                      |
| `for_two_vars`       | for loop with index and value                            |
| `tuple_destructure`  | tuple assignment `q, r = ...`                            |
| `struct_method`      | method call on a struct                                  |
| `compound_assign`    | `+=`, `-=`, etc.                                         |
| `nested_collection`  | `list[list[T]]`, `map[K, list[V]]`, etc.                 |
| `union_field_access` | field access on union where all members share a field    |
| `bytes_ops`          | byte/bytes indexing, slicing, built-in calls             |
| `rune_ops`           | rune literals, RuneToInt/RuneFromInt                     |
| `overloaded_builtin` | Contains/Concat/Repeat/Len on different collection types |

The test harness drives generation by selecting feature combinations. Pairwise coverage (every pair of features appears together in at least one test) is the baseline; higher-strength coverage (triples) targets known-complex interactions.

### Mutator

The mutator takes a well-typed AST and applies one type-breaking transformation. The mutated program should be rejected by the checker. If the checker accepts it, that's a soundness bug.

#### Mutation Operators

| Mutation               | What It Does                                              | Expected Error                    |
| ---------------------- | --------------------------------------------------------- | --------------------------------- |
| `swap_type`            | change a `let` declaration's type to an incompatible one  | `cannot assign X to Y`            |
| `wrong_arg_type`       | replace a function argument with one of wrong type        | `cannot pass X as Y`              |
| `wrong_arg_count`      | add or remove a function argument                         | `expected N arguments, got M`     |
| `wrong_return_type`    | change return expression to wrong type                    | `cannot return X`                 |
| `missing_match_case`   | remove a case from an exhaustive match                    | `non-exhaustive match`            |
| `duplicate_match_case` | duplicate an existing match case                          | `duplicate case`                  |
| `wrong_match_case`     | add a case for a type not in the scrutinee                | `not a member` / `not a variant`  |
| `capture_variable`     | reference an outer variable inside a function literal     | `cannot capture 'name'`           |
| `shadow_binding`       | declare a variable with the same name as an outer binding | `shadows outer binding`           |
| `use_reserved_name`    | use a reserved name as a binding                          | `reserved name`                   |
| `assign_to_self`       | `self = ...` inside a method                              | `cannot assign to self`           |
| `assign_to_tuple_el`   | `pair.0 = ...`                                            | `cannot assign to tuple element`  |
| `void_as_value`        | use void-returning call as a value                        | `void is not a value type`        |
| `break_outside_loop`   | place `break` outside any loop                            | `break outside of loop`           |
| `index_union`          | index into a union-typed variable                         | `cannot index union`              |
| `arith_on_union`       | arithmetic on a union-typed variable                      | `not defined for union`           |
| `order_on_union`       | ordering comparison on a union-typed variable             | `not defined for union`           |
| `missing_initializer`  | omit initializer for a type without zero value            | `initializer required`            |
| `double_optional`      | declare a variable as `T??`                               | `double optional`                 |
| `call_non_function`    | call a non-function value                                 | `cannot call X`                   |
| `mixed_args`           | mix positional and named arguments                        | `cannot mix positional and named` |
| `wrong_named_arg`      | use a nonexistent parameter name                          | `no parameter 'x'`                |

Each mutation records the expected diagnostic so the harness can validate not just rejection but the correct error message.

### Exhaustiveness Enumerator

Match exhaustiveness is tested separately with bounded exhaustive enumeration. For a given type hierarchy or union:

1. Enumerate all subsets of cases (2^n for n variants/members)
2. For each subset, determine whether it should be exhaustive
3. Optionally add `default` to non-exhaustive subsets
4. Run the checker and validate the accept/reject decision

This is tractable because variant counts are small (2–6 for enums, 2–4 for interfaces, 2–4 for union members). The total enumeration per type is at most 2^6 = 64 subsets.

#### Interaction with Interfaces in Unions

When a union contains an interface, exhaustiveness has a subtle interaction: covering all individual variants of the interface satisfies the interface member without an explicit interface case. The enumerator generates both styles:

```
-- Style 1: interface case covers all variants
match v {
    case n: int { ... }
    case node: Node { ... }    -- covers Literal and BinOp
}

-- Style 2: variant cases cover the interface
match v {
    case n: int { ... }
    case lit: Literal { ... }
    case bin: BinOp { ... }
}
```

Both should be accepted. Removing any single case from either style should be rejected (unless `default` is present).

### Harness

The test harness orchestrates generation, mutation, and validation:

```python
def test_well_typed(feature_vector, seed):
    """Generate a well-typed program and verify the checker accepts it."""
    ast = generate(feature_vector, seed)
    errors = check(ast)
    assert errors == [], f"well-typed program rejected: {errors}"

def test_ill_typed(feature_vector, mutation, seed):
    """Generate, mutate, and verify the checker rejects with correct diagnostic."""
    ast = generate(feature_vector, seed)
    mutated, expected_error = mutate(ast, mutation)
    errors = check(mutated)
    assert any(expected_error in e.message for e in errors), (
        f"ill-typed program not rejected with '{expected_error}'"
    )

def test_exhaustiveness(type_config):
    """Bounded exhaustive enumeration of match patterns."""
    for subset in powerset(type_config.cases):
        ast = build_match_program(type_config, subset)
        errors = check(ast)
        if is_exhaustive(subset, type_config):
            assert errors == []
        else:
            assert any("non-exhaustive" in e.message for e in errors)
```

#### Seeded Randomness

Every generated program is determined by a seed. Failing tests report the seed for reproduction. Seeds are drawn sequentially for systematic coverage, not from `random.random()`.

#### Shrinking

When a test fails, the harness attempts to minimize the failing program by:

1. Removing declarations not transitively referenced by the failing construct
2. Removing statements from function bodies
3. Simplifying expressions (replace subexpressions with literals)
4. Removing unused fields from structs
5. Reducing enum variant counts

Each reduction step re-checks that the failure still reproduces. The output is a minimal Taytsh program exhibiting the bug.

## Feature Interaction Matrix

The following feature pairs are high-priority targets based on where type checker bugs cluster in practice (per Chaliasos et al.'s empirical study of JVM compiler bugs):

| Feature A            | Feature B            | Interaction Risk                                     |
| -------------------- | -------------------- | ---------------------------------------------------- |
| `union_type`         | `match_default_bind` | residual type computation for default binding        |
| `union_type`         | `nil_narrowing`      | narrowing a multi-member union removes nil           |
| `optional_type`      | `match_interface`    | interface? matched with variant cases + nil case     |
| `match_interface`    | `match_default_bind` | residual is union of uncovered variant structs       |
| `fn_literal`         | `for_collection`     | closure check must reject capture of loop variable   |
| `fn_literal`         | `try_catch_typed`    | closure check must reject capture of catch binding   |
| `fn_literal`         | `nil_narrowing`      | closure check in narrowed branch                     |
| `union_field_access` | `nil_narrowing`      | field access on narrowed union still valid?          |
| `try_catch_union`    | `match_union`        | catch union type then match within catch body        |
| `overloaded_builtin` | `union_type`         | Contains/Len on union-typed variable (should reject) |
| `nested_collection`  | `for_two_vars`       | iterating nested collection, inner type bindings     |
| `struct_method`      | `fn_value`           | bound method as value (should reject: captures self) |
| `compound_assign`    | `bytes_ops`          | byte compound assignment wraps mod 256               |
| `tuple_destructure`  | `overloaded_builtin` | DivMod result destructured then passed to built-in   |
| `match_enum`         | `match_default_bind` | default binding type is the enum itself              |

## Built-in Function Coverage

The generator tracks which built-in function signatures have been exercised. Each built-in has one or more **signature variants** (e.g., `Contains` has 4: list, set, map, string). The goal is to cover every variant at least once across the test suite.

| Built-in    | Variants                                                  |
| ----------- | --------------------------------------------------------- |
| `Len`       | string, bytes, list, map, set                             |
| `Contains`  | list, set, map, string                                    |
| `Concat`    | string, bytes, list                                       |
| `Repeat`    | string, list                                              |
| `Get`       | map 2-arg (returns V?), map 3-arg (returns V)             |
| `Assert`    | 1-arg (bool), 2-arg (bool, string)                        |
| `IsDigit`   | string, rune (same for IsAlpha, IsAlnum, etc.)            |
| `WriteOut`  | string, bytes (same for WriteErr, WritelnOut, WritelnErr) |
| `ReadFile`  | returns string \| bytes                                   |
| `WriteFile` | accepts string or bytes as second arg                     |

## Metrics

| Metric                    | Target                                             |
| ------------------------- | -------------------------------------------------- |
| Feature pair coverage     | 100% of pairs in the feature vector                |
| Feature triple coverage   | ≥ 80% of triples involving high-risk features      |
| Built-in variant coverage | 100% of signature variants                         |
| Mutation kill rate        | 100% (every mutation operator detected by checker) |
| Exhaustiveness coverage   | all 2^n subsets for n ≤ 6 variants                 |
| Programs per run          | ≥ 10,000 well-typed + 10,000 mutated               |

## Implementation Plan

### Phase 1: Infrastructure

- AST builder utilities: functions that construct valid Taytsh AST nodes without going through the parser
- Type pool construction with constraint enforcement
- Scope tracker that mirrors the checker's scoping rules
- Seeded random source

### Phase 2: Core Generator

- Expression generation for all primitive types
- Statement generation (let, assignment, if, while, for, return)
- Function and struct declaration generation
- Main function generation
- Basic harness: generate → check → assert no errors

### Phase 3: Advanced Features

- Union and optional type generation
- Match statement generation with exhaustiveness
- Try/catch generation
- Function literal generation with no-closure enforcement
- Nil narrowing in if-branches

### Phase 4: Mutator

- All mutation operators from the table above
- Expected-diagnostic validation
- Shrinking for failing tests

### Phase 5: Exhaustiveness Enumerator

- Powerset enumeration for enum, interface, union, and optional match
- Interface-in-union variant coverage

### Phase 6: Coverage Analysis

- Feature pair/triple tracking
- Built-in variant tracking
- Reports showing uncovered combinations

## References

- Chaliasos, Sotiropoulos, Drosos, Mitropoulos, Spinellis, Krintz. "Finding Typing Compiler Bugs." PLDI 2022.
- Chaliasos, Sotiropoulos, Spinellis, Krintz. "Well-Typed Programs Can Go Wrong." OOPSLA 2021.
- Dewey, Roesch, Hardekopf. "Fuzzing the Rust Typechecker Using CLP (T)." ASE 2015.
- Ploeger, Wüstholz, Christakis. "Erwin: Bounded Exhaustive Random Program Generation for Compiler and Language Testing." ICSE 2026.
- Liu. "A Generic Algorithm for Checking Exhaustivity of Pattern Matching." Scala 2016.
- Fetscher, Claessen, Palka, Hughes, Findler. "Making Random Judgments." ESOP 2015.
