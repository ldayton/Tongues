# Phase 4: Names

**Module:** `frontend/names.py`

Scope analysis and name binding. Builds a symbol table mapping every name reference to its declaration. Since phase 3 guarantees no nested functions and no `global`/`nonlocal`, scoping is flat: local → module → builtin.

## Inputs

- **AST**: dict-based AST from phase 2

## Scope Model

| Scope   | Contains                                                                          |
| ------- | --------------------------------------------------------------------------------- |
| Builtin | Allowed builtins from phase 3 plus exception types                                |
| Module  | Classes, functions, imports, constants, type aliases, annotated module variables  |
| Class   | Methods, annotated fields, `__init__`-assigned fields                             |
| Local   | Parameters, assignments, for-loop targets, except bindings, imports, match/walrus |

Resolution order: local → module → builtin. No enclosing function scope (phase 3 bans nested functions). Class scope is not searched during name resolution — it exists only for field/method lookup by later phases.

## Binding Rules

Three sequential passes collect names before resolving references.

### Pass 1: Module-Level Names

Walks top-level statements and registers:

| Construct                   | Kind         | Notes                                |
| --------------------------- | ------------ | ------------------------------------ |
| `class Foo:`                | `class`      | Base class names recorded in `bases` |
| `def f():`                  | `function`   |                                      |
| `X = 42` (ALL_CAPS)         | `constant`   | Detected by naming convention        |
| `MyType = dict[str, int]`   | `type_alias` | PascalCase + type-expression value   |
| `x: int = 0`                | `variable`   | `constant` if ALL_CAPS               |
| `import sys`                | `import`     |                                      |
| `from X import Y`           | `import`     | Binds `Y` (or `Z` if `as Z`)         |
| `if TYPE_CHECKING:` imports | `import`     | Treated as module-level              |

Duplicate module-level names are errors. ALL_CAPS detection requires at least one uppercase letter and only uppercase letters, digits, and underscores. Type alias detection requires a PascalCase name assigned to a `Name`, `Subscript`, or `BinOp` node.

Between pass 1 and pass 2, base class names are validated — every name in a class's `bases` list must resolve.

### Pass 2: Class Members

For each class, collects members in two sub-passes:

| Source                            | Kind       | Priority |
| --------------------------------- | ---------- | -------- |
| `def method(self):` in class body | `function` | —        |
| `x: int` in class body            | `field`    | 1        |
| `self.x = ...` in `__init__`      | `field`    | 2        |
| `self.x: T = ...` in `__init__`   | `field`    | 2        |

Class body annotations are collected first. Then `__init__` is walked (including nested control flow, but not nested functions) for `self.attr` assignments. If a field was already declared in the class body, the `__init__` assignment is skipped.

### Pass 3: Locals and References

For each function/method, collects local bindings then resolves all `Name` nodes with `ctx=Load`.

| Construct         | Kind        | Notes                                  |
| ----------------- | ----------- | -------------------------------------- |
| `def f(x: int):`  | `parameter` | `self`/`cls` suppresses shadow warning |
| `x = expr`        | `variable`  |                                        |
| `x: int = expr`   | `variable`  |                                        |
| `a, b = expr`     | `variable`  | Recursively unpacks tuples/lists       |
| `for x in xs:`    | `variable`  |                                        |
| `except E as e:`  | `variable`  |                                        |
| `from X import Y` | `import`    | Local scope                            |
| `if (x := expr):` | `variable`  | Scopes to enclosing function           |
| `case Foo(x=y):`  | `variable`  | All match pattern forms                |

Local names are registered on first occurrence — subsequent assignments to the same name do not create new entries.

### Comprehension Scoping

Comprehension variables (`for x in xs` inside `[...]`, `{...}`, etc.) are scoped to the comprehension — they do not leak to the enclosing function. References inside comprehensions resolve against comprehension-local variables first, then fall through to the standard local → module → builtin chain.

Nested comprehensions inherit outer comprehension variables.

### Match Pattern Names

All pattern forms that bind names are collected:

| Pattern          | Binding                                  |
| ---------------- | ---------------------------------------- |
| `case x:`        | `MatchAs` with name                      |
| `case Foo(a=b):` | `MatchClass` keyword and positional      |
| `case [a, *b]:`  | `MatchSequence` + `MatchStar`            |
| `case {k: v}:`   | `MatchMapping` patterns and rest         |
| `case a \| b:`   | `MatchOr` — names from first alternative |

## Forward References

Forward references are allowed. A function may call another defined later in the file. A type annotation may reference a class defined later. This works because pass 1 collects all module-level names before pass 3 resolves references.

## Errors

| Condition                 | Diagnostic                                          |
| ------------------------- | --------------------------------------------------- |
| Name not found            | error: `name 'x' is not defined`                    |
| Name not found in init    | error: `name 'x' is not defined; cannot infer type` |
| Redefinition at module    | error: `'x' already defined at line N`              |
| Unknown base class        | error: `name 'Base' is not defined`                 |
| Parameter shadows builtin | warning: `parameter 'x' shadows builtin`            |

## Output

`NameResult` containing:

- **table**: `NameTable` with three maps:
  - `module_names`: `{name → NameInfo}`
  - `class_names`: `{class_name → {name → NameInfo}}`
  - `local_names`: `{(class_name, func_name) → {name → NameInfo}}`
- **violations**: list of errors
- **warnings**: list of warnings

Each `NameInfo` carries:

| Field        | Content                                                                                              |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| `name`       | bound name                                                                                           |
| `kind`       | `class`, `function`, `variable`, `parameter`, `field`, `constant`, `type_alias`, `import`, `builtin` |
| `scope`      | `builtin`, `module`, `class`, `local`                                                                |
| `lineno`     | declaration line (1-indexed)                                                                         |
| `col`        | declaration column (0-indexed)                                                                       |
| `decl_class` | containing class name, or `""`                                                                       |
| `decl_func`  | containing function name, or `""`                                                                    |
| `bases`      | base class names (classes only)                                                                      |

## Postconditions

All names resolve to a declaration with known kind and scope. Module-level names are unique. Base class names validated. Comprehension variables do not leak. Walrus operator variables scope to the enclosing function.
