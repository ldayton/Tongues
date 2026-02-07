# Phase 7: Hierarchy

**Module:** `frontend/hierarchy.py`

Build the inheritance tree and compute subtyping relations. Inheritance implies subtyping in Tongues. Since phase 3 guarantees single inheritance:

- Hierarchy is a tree, not DAG
- No diamond problem
- LUB is finding common ancestor (walk up both chains)
- Transitive closure is just ancestor list per class

## Postconditions

SubtypeRel maps every class to ancestors; `is_subtype(A, B)` works for any A, B; no cycles.

## Prior Art

- [Inheritance Is Not Subtyping](https://www.cs.utexas.edu/~wcook/papers/InheritanceSubtyping90/CookPOPL90.pdf)
- [Variance](https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science))
