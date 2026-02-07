# Phase 4: Names

**Module:** `frontend/names.py`

Build a symbol table mapping names to their declarations. Since phase 3 guarantees no nested functions and no `global`/`nonlocal`, scoping collapses to:

| Scope   | Contains                                  |
| ------- | ----------------------------------------- |
| Builtin | `len`, `range`, `str`, `int`, `Exception` |
| Module  | Classes, functions, constants             |
| Class   | Fields, methods                           |
| Local   | Parameters, local variables               |

No enclosing scope. Resolution is a simple two-level lookup: local → module (→ builtin).

## Postconditions

All names resolve; no shadowing ambiguity; kind (class/function/variable/parameter/field) is known for each name.

## Prior Art

- [Scope Graphs](https://link.springer.com/chapter/10.1007/978-3-662-46669-8_9)
- [Python LEGB](https://realpython.com/python-scope-legb-rule/)
