# Phase 4: Names

**Module:** `frontend/names.py`

Build a symbol table mapping names to their declarations. Since phase 3 guarantees no nested functions and no `global`/`nonlocal`, scoping is simple.

## Inputs

- **AST**: dict-based AST from Phase 2

## Scopes

| Scope   | Contains                                                                   |
| ------- | -------------------------------------------------------------------------- |
| Builtin | Allowed builtins from [03-subset-spec.md §7](03-subset-spec.md#7-builtins) |
| Module  | Classes, functions, imports, variables                                     |
| Class   | Fields, methods                                                            |
| Local   | Parameters, local variables, imports                                       |

Resolution order: local → module → builtin. No enclosing function scope (phase 3 bans nested functions).

## Name Binding

| Construct         | Scope                    | Notes                                        |
| ----------------- | ------------------------ | -------------------------------------------- |
| `def f():`        | module                   | Function definition                          |
| `class C:`        | module                   | Class definition                             |
| `x: int = 1`      | module or local          | Annotated assignment                         |
| `x = 1`           | module or local          | Unannotated assignment                       |
| `from X import Y` | module or local          | Import binding                               |
| `def f(x: int):`  | local                    | Parameter                                    |
| `for x in xs:`    | local                    | Loop variable                                |
| `except E as e:`  | local                    | Exception binding                            |
| `[x for x in xs]` | local (to comprehension) | Does not leak to enclosing scope             |
| `if (x := f()):`  | local (to function)      | Walrus operator scopes to enclosing function |
| `case Foo(x=y):`  | local                    | Match pattern binding                        |

## Errors

| Condition                    | Diagnostic                               |
| ---------------------------- | ---------------------------------------- |
| Name not found               | error: `name 'x' is not defined`         |
| Redefinition at module level | error: `'x' already defined at line N`   |
| Parameter shadows builtin    | warning: `parameter 'x' shadows builtin` |

Forward references are allowed—a function may call another defined later in the file.

## Postconditions

All names resolve to a declaration with known kind (class/function/variable/parameter/field/import).

## Prior Art

- [Scope Graphs](https://link.springer.com/chapter/10.1007/978-3-662-46669-8_9)
- [Python LEGB](https://realpython.com/python-scope-legb-rule/)
