# Phase 7: Hierarchy

**Module:** `frontend/hierarchy.py`

Build the class hierarchy and classify structs. Since phase 3 guarantees single inheritance, the hierarchy is a tree. This phase detects the hierarchy root, marks node subclasses and exception subclasses, and validates that no cycles exist.

## Inputs

- **SymbolTable**: structs with `bases` populated from phases 4-6

## Ancestor Chains

Each struct's `bases` list contains its direct base class names. The hierarchy is traversed transitively to determine subclass relationships:

```python
class Node: ...       # root — bases: []
class Expr(Node): ... # bases: ["Node"]
class BinOp(Expr): ...# bases: ["Expr"] → transitive: Expr, Node
```

## Interface and Struct Classification

### Hierarchy Root Detection

A class is the hierarchy root if it:
1. Has no base classes
2. Is used as a base class (directly or transitively) by at least one other class
3. Is the only such root

Exception classes and their subclasses are excluded from root detection — `Exception` is a builtin, not a user-defined hierarchy root.

If zero or multiple roots exist, `hierarchy_root` is `None` and no interface is generated.

### Classification

| Classification     | Criteria                                  | Effect                         |
| ------------------ | ----------------------------------------- | ------------------------------ |
| Hierarchy root     | No base, used as base by others           | Becomes Taytsh interface       |
| Node subclass      | Transitively inherits from hierarchy root | `is_node = True` on StructInfo |
| Exception subclass | Transitively inherits from `Exception`    | `is_exception = True`          |
| Standalone class   | No inheritance relationship to root       | Neither flag set               |

The hierarchy root itself is classified as a node subclass (`is_node = True`).

### Exception Subclass Detection

A class is an exception subclass if:
- It is `Exception` itself, or
- Any of its bases is transitively an exception subclass

This handles both direct (`class E(Exception)`) and indirect (`class E(Base, Exception)`, `class E2(E)`) exception inheritance.

## Subtype Operations

The `SubtypeRel` provides:

| Operation            | Definition                         |
| -------------------- | ---------------------------------- |
| `is_node(name)`      | `name` is in `node_types` set      |
| `is_exception(name)` | `name` is in `exception_types` set |

Least upper bound and ancestor chain queries are handled by walking `bases` on the StructInfo directly, not precomputed in SubtypeRel.

## Errors

| Condition         | Diagnostic                    |
| ----------------- | ----------------------------- |
| Inheritance cycle | error: `cycle in inheritance` |

Cycle detection walks each class's base chain and raises if a class appears twice.

## Output

`SubtypeRel` containing:

| Field             | Content                                               |
| ----------------- | ----------------------------------------------------- |
| `hierarchy_root`  | name of the polymorphic root class, or `None`         |
| `node_types`      | set of all hierarchy root subclasses (including root) |
| `exception_types` | set of all `Exception` subclasses                     |

Side effect: each `StructInfo` in the SymbolTable has `is_node` and `is_exception` flags updated.

## Postconditions

No inheritance cycles. Hierarchy root identified (if unique). Every struct classified as node subclass, exception subclass, both, or neither. `is_node` and `is_exception` flags set on all StructInfo entries.
