# Phase 12: Liveness

**Module:** `middleend/liveness.py`

Analyze liveness: unused initial values, unused catch variables, unused bindings. Determines whether the initial value of a VarDecl is ever read before being overwritten.

| Annotation                     | Meaning                               |
| ------------------------------ | ------------------------------------- |
| `VarDecl.initial_value_unused` | Initial value overwritten before read |
| `TryCatch.catch_var_unused`    | Catch variable never referenced       |
| `TypeSwitch.binding_unused`    | Binding variable never referenced     |
| `TupleAssign.unused_indices`   | Which tuple targets are never used    |

## Postconditions

All liveness annotations set; enables dead store elimination in codegen.
