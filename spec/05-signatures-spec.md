# Phase 5: Signatures

**Module:** `frontend/signatures.py`

Parse type annotations into internal type representations. Verify types are well-formed via kind checking—kinds classify type constructors the way types classify values. No higher-kinded types, so kind checking reduces to arity validation:

| Constructor   | Arity | Kind               |
| ------------- | ----- | ------------------ |
| `List`, `Set` | 1     | `* -> *`           |
| `Dict`        | 2     | `* -> * -> *`      |
| `Optional`    | 1     | `* -> *`           |
| `Callable`    | 2     | `[*...] -> * -> *` |

## Postconditions

All type annotations parsed to IR types; all types well-formed (correct arity, valid references); SigTable maps every function to `(params, return_type)`.

## Prior Art

- [Kind (type theory)](https://en.wikipedia.org/wiki/Kind_(type_theory))
- [PEP 484](https://peps.python.org/pep-0484/)
