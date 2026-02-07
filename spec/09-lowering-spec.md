# Phase 9: Lowering

**Module:** `frontend/lowering.py`

Transform typed AST into language-agnostic IR. Lowering reads types computed by Phase 8, tracks narrowing context as it traverses control flow, and emits IR nodes that map directly to target language constructs.

## Goals

- **Semantic completeness**: Every source operation has an IR representation
- **Language agnosticism**: IR nodes avoid Python-specific semantics
- **Direct mapping**: Each IR node maps cleanly to target language constructs
- **Type preservation**: All expressions carry resolved types from Phase 8

## Transformation Rules

### Statements

| Python                       | IR Output                                  |
| ---------------------------- | ------------------------------------------ |
| `if cond:`                   | `If(cond, then_body, else_body)`           |
| `if isinstance(x, T):` chain | `TypeSwitch(expr, binding, cases)`         |
| `match x:`                   | `Match(expr, cases, default)`              |
| `for x in items:`            | `ForRange(index, value, iterable, body)`   |
| `for i in range(n):`         | `ForClassic(init, cond, post, body)`       |
| `for i in range(a, b):`      | `ForClassic(init, cond, post, body)`       |
| `while cond:`                | `While(cond, body)`                        |
| `try: ... except E as e:`    | `TryCatch(body, catches)`                  |
| `raise E(msg, pos)`          | `Raise(error_type, message, pos)`          |
| `return expr`                | `Return(value)`                            |
| `x = expr`                   | `Assign(target, value)` or `VarDecl`       |
| `a, b = expr`                | `TupleAssign(targets, value)`              |
| `x += n`                     | `OpAssign(target, op, value)`              |
| `assert cond`                | `Assert(test, message)`                    |
| `print(x)`                   | `Print(value, newline=True, stderr=False)` |
| `print(x, end="")`           | `Print(value, newline=False)`              |
| `print(x, file=sys.stderr)`  | `Print(value, stderr=True)`                |
| `pass`                       | `NoOp()` or omit                           |
| `if __name__ == "__main__":` | `EntryPoint(function_name)`                |

### Expressions

| Python                 | IR Output                               |
| ---------------------- | --------------------------------------- |
| `42`, `3.14`, `"str"`  | `IntLit`, `FloatLit`, `StringLit`       |
| `True`, `False`        | `BoolLit(value)`                        |
| `None`                 | `NilLit(typ)`                           |
| `x`                    | `Var(name, typ)`                        |
| `obj.field`            | `FieldAccess(obj, field)`               |
| `obj[i]`               | `Index(obj, index)`                     |
| `obj[a:b]`             | `SliceExpr(obj, low, high, step)`       |
| `a + b`, `a - b`, etc. | `BinaryOp(op, left, right)`             |
| `a < b < c`            | `ChainedCompare(operands, ops)`         |
| `-x`, `not x`, `~x`    | `UnaryOp(op, operand)`                  |
| `x and y`, `x or y`    | `BinaryOp("&&", ...)`, `BinaryOp("      |  | ", ...)` |
| `a if cond else b`     | `Ternary(cond, then_expr, else_expr)`   |
| `f(args)`              | `Call(func, args)`                      |
| `obj.method(args)`     | `MethodCall(obj, method, args)`         |
| `obj.method` (no call) | `FuncRef(name, obj)` with receiver type |
| `isinstance(x, T)`     | `IsType(expr, tested_type)`             |
| `x is None`            | `IsNil(expr, negated=False)`            |
| `x is not None`        | `IsNil(expr, negated=True)`             |
| `[a, b, c]`            | `SliceLit(element_type, elements)`      |
| `{a, b, c}`            | `SetLit(element_type, elements)`        |
| `{k: v, ...}`          | `MapLit(key_type, value_type, entries)` |
| `(a, b, c)`            | `TupleLit(elements)`                    |
| `ClassName(args)`      | `StructLit(struct_name, fields)`        |
| `[x for x in xs]`      | `ListComp(element, generators)`         |
| `{x for x in xs}`      | `SetComp(element, generators)`          |
| `{k: v for ...}`       | `DictComp(key, value, generators)`      |

### Builtins

| Python                   | IR Output                |
| ------------------------ | ------------------------ |
| `len(x)`                 | `Len(expr)`              |
| `min(a, b)`              | `MinExpr(left, right)`   |
| `max(a, b)`              | `MaxExpr(left, right)`   |
| `int(s)`, `int(s, base)` | `ParseInt(string, base)` |
| `str(n)` (for int)       | `IntToStr(value)`        |
| `abs(x)`                 | `Call("abs", [x])`       |
| `chr(n)`                 | `Cast(expr, STRING)`     |
| `ord(c)`                 | `Cast(expr, INT)`        |

### String Operations

| Python                | IR Output                                |
| --------------------- | ---------------------------------------- |
| `s[i]`                | `CharAt(string, index)`                  |
| `len(s)` (char count) | `CharLen(string)`                        |
| `s[a:b]`              | `Substring(string, low, high)`           |
| `s1 + s2`             | `StringConcat(parts)`                    |
| `f"{x}..."`           | `StringFormat(template, args)`           |
| `s.strip(chars)`      | `TrimChars(string, chars, mode="both")`  |
| `s.lstrip(chars)`     | `TrimChars(string, chars, mode="left")`  |
| `s.rstrip(chars)`     | `TrimChars(string, chars, mode="right")` |
| `c.isalnum()`         | `CharClassify(kind="alnum", char)`       |
| `c.isdigit()`         | `CharClassify(kind="digit", char)`       |
| `c.isalpha()`         | `CharClassify(kind="alpha", char)`       |
| `c.isspace()`         | `CharClassify(kind="space", char)`       |
| `c.isupper()`         | `CharClassify(kind="upper", char)`       |
| `c.islower()`         | `CharClassify(kind="lower", char)`       |

### I/O

| Python                       | IR Output                        |
| ---------------------------- | -------------------------------- |
| `sys.stdin.readline()`       | `ReadLine()`                     |
| `sys.stdin.read()`           | `ReadAll()`                      |
| `sys.stdin.buffer.read()`    | `ReadBytes()`                    |
| `sys.stdin.buffer.read(n)`   | `ReadBytesN(count)`              |
| `sys.stdout.buffer.write(b)` | `WriteBytes(data, stderr=False)` |
| `sys.stderr.buffer.write(b)` | `WriteBytes(data, stderr=True)`  |
| `sys.argv`                   | `Args()`                         |
| `os.getenv(name)`            | `GetEnv(name, default=None)`     |
| `os.getenv(name, default)`   | `GetEnv(name, default)`          |

### Declarations

| Python                  | IR Output                           |
| ----------------------- | ----------------------------------- |
| `class Foo:`            | `Struct(name, fields, methods)`     |
| `class Foo(Base):`      | `Struct` with `implements=[Base]`   |
| `class E(Exception):`   | `Struct` with `is_exception=True`   |
| `def f(x: int) -> str:` | `Function(name, params, ret, body)` |
| `def m(self, x):`       | `Function` with `receiver`          |
| `X = 42` (module level) | `Constant(name, typ, value)`        |

### Special Patterns

| Python                            | IR Output                                   |
| --------------------------------- | ------------------------------------------- |
| `xs[-1]`                          | `LastElement(sequence)`                     |
| `&expr` (address-of)              | `AddrOf(operand)`                           |
| Sentinel int `== None`            | `BinaryOp("==", expr, IntLit(sentinel))`    |
| `x if sentinel else None`         | `SentinelToOptional(expr, sentinel)`        |
| `[]T` to `[]Interface` conversion | `SliceConvert(source, target_element_type)` |

---

## Narrowing Context

Lowering tracks type narrowing as it traverses control flow. Types from Phase 8 are primary; on-demand narrowing handles patterns requiring flow context.

| Pattern                    | Mechanism                                           |
| -------------------------- | --------------------------------------------------- |
| `isinstance(x, T)`         | Pre-computed in Phase 8, read from `_expr_type`     |
| `x.kind == "value"`        | `kind_to_class` mapping built from dataclass fields |
| `k = x.kind; if k == "v":` | `kind_source_vars` tracks alias to original         |
| `x.body.kind == "value"`   | `narrowed_attr_paths` for nested access             |
| `if x is not None:`        | Tracked per-scope, reset at merge points            |

### Context State

| Field                 | Purpose                                              |
| --------------------- | ---------------------------------------------------- |
| `var_types`           | Current type of each variable in scope               |
| `narrowed_vars`       | Variables with narrowed types in current branch      |
| `narrowed_attr_paths` | Attribute paths (`("x", "body")`) narrowed by kind   |
| `kind_to_class`       | Maps `"add"` → `"Add"` for discriminated unions      |
| `kind_source_vars`    | Maps alias `k` → original `x` for deferred narrowing |
| `union_types`         | Variables typed as union, tracking possible variants |
| `optional_strings`    | Variables using empty string as None sentinel        |
| `sentinel_ints`       | Variables using -1 (or other) as None sentinel       |

---

## Errors

| Condition                | Diagnostic                                           |
| ------------------------ | ---------------------------------------------------- |
| Unsupported AST node     | error: `lowering: unsupported node type '{type}'`    |
| Unknown struct reference | error: `lowering: unknown struct '{name}'`           |
| Missing expression type  | error: `lowering: no type for expression at {loc}`   |
| Invalid field access     | error: `lowering: '{struct}' has no field '{field}'` |

---

## Invariants

After lowering, these properties hold:

**Types**
- All `Expr` nodes have non-None `typ` field
- All `StructRef` names exist in `Module.structs`
- All `InterfaceRef` names exist in `Module.interfaces` (or are `"any"`)

**Structure**
- No Python AST nodes remain in output
- All `Var` references resolve to visible declarations
- `Module.hierarchy_root` set if class hierarchy detected
- Exception structs have `is_exception=True` and optional `embedded_type`

**Semantics**
- Truthy tests emit `Truthy(expr)`, not expanded comparisons
- Constructors emit `StructLit` with all fields, not field-by-field assignment
- Bound method references have `FuncType.receiver` set
- Character operations use semantic nodes (`CharAt`, `CharLen`, `Substring`)

**Annotations not yet set** (added by middleend):
- `is_reassigned`, `is_modified`, `is_unused` on variables/params
- `hoisted_vars` on control flow statements
- `needs_named_returns` on functions
- `ownership`, `region`, `escapes` annotations

---

## Non-Goals

Phase 9 explicitly does NOT:

- Determine variable reassignment or mutation (Phase 10: scope)
- Analyze return patterns (Phase 11: returns)
- Detect unused bindings (Phase 12: liveness)
- Compute hoisting requirements (Phase 13: hoisting)
- Infer ownership or escape analysis (Phase 14: ownership)
- Emit target language code (Phase 15: backend)

---

## Postconditions

IR Module complete with:
- `structs`: All class definitions with fields and methods
- `interfaces`: Hierarchy root interface if polymorphism detected
- `functions`: All function definitions with typed parameters and bodies
- `constants`: Module-level constant bindings
- `enums`: Enumeration definitions (if any)
- `exports`: Public API declarations
- `entrypoint`: Main function marker (if `if __name__ == "__main__"` present)

All IR nodes typed. No AST remnants. Ready for middleend annotation passes.

---

## Prior Art

- [Three-address code](https://en.wikipedia.org/wiki/Three-address_code)
- [Cornell CS 4120 IR notes](https://www.cs.cornell.edu/courses/cs4120/2023sp/notes/ir/)
- [TypeScript Flow Nodes](https://effectivetypescript.com/2024/03/24/flownodes/)
- [Pyright Lazy Evaluation](https://github.com/microsoft/pyright/blob/main/docs/mypy-comparison.md)
