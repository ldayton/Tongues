# Phase 2: Parse

**Module:** `frontend/parse.py`

Tokenize source code and parse into dict-based AST. Hand-written recursive descent parser, self-contained in a single file with no external dependencies. Parses the full Python grammar—subset restrictions are enforced in Phase 3.

The tokenizer uses explicit `while i < len(...)` loops (no generators). The parser produces dict-based AST nodes following the structure of Python's `ast` module.

## AST Structure

Nodes are plain dicts with a `_type` key identifying the node type. Every node carries position metadata:

```python
{"_type": "Name", "id": "x", "ctx": {"_type": "Load"},
 "lineno": 1, "col_offset": 0, "end_lineno": 1, "end_col_offset": 0}
```

Node types, field names, and nesting follow `ast.parse()` output — `FunctionDef` has `name`, `args`, `body`, `decorator_list`, `returns`; `BinOp` has `left`, `op`, `right`; operators are nested dicts like `{"_type": "Add"}`; context fields (`Load`/`Store`/`Del`) appear on `Name`, `Attribute`, `Subscript`, `List`, `Tuple`, `Starred`. Refer to the [ast module docs](https://docs.python.org/3/library/ast.html) for the full node catalog.

### Divergences from `ast`

| Area               | `ast` module                                                                                     | Tongues                                 |
| ------------------ | ------------------------------------------------------------------------------------------------ | --------------------------------------- |
| Representation     | Object instances; type via `__class__.__name__`                                                  | Plain dicts; type via `_type` key       |
| End positions      | Precise `end_lineno`/`end_col_offset`                                                            | Present but set to start position       |
| `type_comment`     | On `Assign`, `For`, `With`, `FunctionDef`, `arg`, etc.                                           | Omitted (pre-annotation legacy feature) |
| `Constant.kind`    | `"u"` for unicode literals                                                                       | Omitted                                 |
| Omitted node types | `Expression`, `Interactive`, `FunctionType`, `TypeAlias`, `TypeVar`, `TypeVarTuple`, `ParamSpec` | Not needed for transpilation            |

## Errors

| Condition           | Diagnostic                                      |
| ------------------- | ----------------------------------------------- |
| Unexpected token    | error: `unexpected token '{tok}' at line {n}`   |
| Unterminated string | error: `unterminated string literal`            |
| Invalid syntax      | error: `invalid syntax at line {n}, column {c}` |
| Mismatched parens   | error: `unmatched '(' at line {n}`              |

## Postconditions

Source code parsed to dict-based AST; structure follows `ast.parse()` output (see divergences above); all tokens consumed; syntax errors reported with line/column.

## Prior Art

- [Python ast module](https://docs.python.org/3/library/ast.html)
- [parso](https://github.com/davidhalter/parso)
- [Dragon Book Ch. 3-4](https://en.wikipedia.org/wiki/Compilers:_Principles,_Techniques,_and_Tools)
