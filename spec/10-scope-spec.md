# Phase 10: Scope

**Module:** `middleend/scope.py`

Analyze variable scope: declarations, reassignments, parameter modifications. Walks each function body tracking which variables are declared vs assigned, and whether parameters are modified.

| Annotation              | Meaning                                     |
| ----------------------- | ------------------------------------------- |
| `VarDecl.is_reassigned` | Variable assigned after declaration         |
| `Param.is_modified`     | Parameter assigned/mutated in function body |
| `Param.is_unused`       | Parameter never referenced                  |
| `Assign.is_declaration` | First assignment to a new variable          |
| `Expr.is_interface`     | Expression statically typed as interface    |
| `Name.narrowed_type`    | Precise type at use site after type guards  |

## Postconditions

- Every VarDecl, Param, and Assign annotated; reassignment counts accurate
- Variables annotated with `is_const` (never reassigned after declaration); enables `const`/`let` in TS, `final` in Java
- Expressions annotated with `is_interface: bool` when statically typed as interface; enables direct `== nil` vs reflection-based nil check in Go
- Variables annotated with precise narrowed type at each use site (not just declaration type); eliminates redundant casts when type is statically known
