# Phase 2: Parse

**Module:** `frontend/parse.py`

Tokenize source code and parse into dict-based AST. Enables self-hosting by removing CPython bootstrap dependency.

| Component      | Lines | Description                                      |
| -------------- | ----- | ------------------------------------------------ |
| `tokenize.py`  | ~350  | While-loop state machine; returns `list[Token]`  |
| `grammar.py`   | ~250  | Pre-compiled DFA tables as static data           |
| `parse.py`     | ~175  | LR shift-reduce parser; stack-based              |
| `ast_build.py` | ~250  | Grammar rules → dict nodes matching `ast` module |

The tokenizer uses explicit `while i < len(...)` loops (no generators). Grammar tables are pre-compiled under CPython once, then embedded as data. The parser is a simple stack machine consuming tokens and emitting dict-based AST nodes.

## Subset Simplifications

The restricted subset eliminates major parsing pain points:

| Constraint               | Simplification                                     |
| ------------------------ | -------------------------------------------------- |
| f-strings: `{expr}` only | No `!conversion`, no `:format_spec`                |
| No generators            | Tokenizer returns `list[Token]`, not lazy iterator |
| No nested functions      | No closure/scope tracking during parse             |
| Walrus operator          | `x := expr` allowed; scopes to enclosing function  |
| No async/await           | No context-dependent keyword handling              |
| Single grammar version   | No version switching; one static grammar           |

## Postconditions

Source code parsed to dict-based AST; structure matches `ast.parse()` output; all tokens consumed; syntax errors reported with line/column.

## Prior Art

- [Dragon Book Ch. 3-4](https://en.wikipedia.org/wiki/Compilers:_Principles,_Techniques,_and_Tools)
- [pgen2](https://github.com/python/cpython/tree/main/Parser/pgen)
- [parso](https://github.com/davidhalter/parso)
