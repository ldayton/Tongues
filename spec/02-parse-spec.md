# Phase 2: Parse

**Module:** `frontend/parse.py`

Tokenize source code and parse into dict-based AST. Hand-written parser based on parso, self-contained in a single file. Parses the full Python grammar—subset restrictions are enforced in Phase 3.

The tokenizer uses explicit `while i < len(...)` loops (no generators). The parser is a recursive descent parser producing dict-based AST nodes matching the structure of Python's `ast` module.

## Postconditions

Source code parsed to dict-based AST; structure matches `ast.parse()` output; all tokens consumed; syntax errors reported with line/column.

## Prior Art

- [parso](https://github.com/davidhalter/parso)
- [Dragon Book Ch. 3-4](https://en.wikipedia.org/wiki/Compilers:_Principles,_Techniques,_and_Tools)
