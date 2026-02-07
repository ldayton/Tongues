# Phase 14: Ownership

**Module:** `middleend/ownership.py`

Infer ownership and region annotations for memory-safe code generation. Since phase 3 guarantees no back-references, no borrowed field storage, and strict tree structures, ownership analysis reduces to simple patterns:

| Pattern                    | Ownership        | Region              |
| -------------------------- | ---------------- | ------------------- |
| Constructor call (`Foo()`) | owned            | caller's region     |
| Factory function return    | owned            | caller's region     |
| Parameter                  | borrowed         | caller's region     |
| Field access               | borrowed         | object's region     |
| Return value               | owned (transfer) | caller's region     |
| Collection element         | owned            | collection's region |
| Explicit `.copy()` call    | owned (new)      | caller's region     |

## Escape Analysis

Escape analysis detects when borrowed references outlive their region:

| Violation                    | Diagnostic                                                                   |
| ---------------------------- | ---------------------------------------------------------------------------- |
| Borrowed ref stored in field | Error: "cannot store borrowed `x` in field; use `.copy()` or take ownership" |
| Borrowed ref returned        | Error: "reference to `x` escapes function scope"                             |
| Borrowed ref in collection   | Error: "cannot add borrowed `x` to collection; transfer ownership or copy"   |

## Ambiguous Ownership

Lobster-style fallback: When inference cannot determine ownership statically, mark as `shared`. Backends emit:
- Go: no change (GC handles)
- Rust: `Rc<T>` or `Arc<T>`
- C: reference-counted wrapper

## Annotations

| Annotation          | Meaning                                           |
| ------------------- | ------------------------------------------------- |
| `VarDecl.ownership` | `owned`, `borrowed`, or `shared`                  |
| `Param.ownership`   | `owned` (takes ownership) or `borrowed` (default) |
| `Field.ownership`   | `owned` (default) or `weak` (back-reference)      |
| `Expr.escapes`      | Expression's value escapes current scope          |

## Postconditions

- Every VarDecl, Param, Field annotated with ownership
- No escaping borrowed references (or diagnostic emitted)
- Ambiguous cases marked `shared` for runtime fallback
- Backends can emit memory management without re-analysis

## Prior Art

- [Tofte-Talpin Region Inference](https://www.sciencedirect.com/science/article/pii/S0890540196926139)
- [Lobster Compile-Time RC](https://aardappel.github.io/lobster/memory_management.html)
- [Cyclone Regions](https://www.cs.umd.edu/projects/cyclone/papers/cyclone-regions.pdf)
