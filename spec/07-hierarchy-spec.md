# Phase 7: Hierarchy

**Module:** `frontend/hierarchy.py`

Build the inheritance tree and compute subtyping relations. Since phase 3 guarantees single inheritance, the hierarchy is a tree—no diamond problem, no ambiguity.

## Subtype Relation

Inheritance implies subtyping. For every class, compute its ancestor chain:

```python
class Node: ...
class Expr(Node): ...
class BinOp(Expr): ...

# Ancestors:
# BinOp → [Expr, Node]
# Expr  → [Node]
# Node  → []
```

## Operations

| Operation          | Definition                                |
| ------------------ | ----------------------------------------- |
| `is_subtype(A, B)` | B is in A's ancestor chain (or A == B)    |
| `ancestors(A)`     | Ordered list from A up to root            |
| `lub(A, B)`        | First common ancestor (least upper bound) |

LUB is needed for union types—backends need a common representation for `A | B`.

## Classification

| Classification     | Criteria                        | Purpose                               |
| ------------------ | ------------------------------- | ------------------------------------- |
| Hierarchy root     | No base, used as base by others | Interface type for polymorphism       |
| Node subclass      | Inherits from hierarchy root    | Emitted with interface implementation |
| Exception subclass | Inherits from `Exception`       | Error handling codegen                |

## Errors

| Condition                | Diagnostic                                 |
| ------------------------ | ------------------------------------------ |
| Inheritance cycle        | error: `cycle in inheritance: A -> B -> A` |
| Unknown base class       | error: `unknown base class 'Foo'`          |
| Multiple hierarchy roots | warning: `multiple hierarchy roots found`  |

## Output

SubtypeRel containing:
- `ancestors`: dict mapping each class to its ordered ancestor list
- `hierarchy_root`: the polymorphic root class (or None)
- `node_types`: set of all hierarchy root subclasses
- `exception_types`: set of all Exception subclasses

## Postconditions

All classes have ancestor chains; `is_subtype(A, B)` works for any pair; `lub(A, B)` returns common ancestor; no cycles.

## Prior Art

- [Inheritance Is Not Subtyping](https://www.cs.utexas.edu/~wcook/papers/InheritanceSubtyping90/CookPOPL90.pdf)
- [Least Upper Bound](https://en.wikipedia.org/wiki/Join_and_meet)
