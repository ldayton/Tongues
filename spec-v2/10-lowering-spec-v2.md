# Phase 9: Lowering

**Modules:** `frontend/lowering.py`, `frontend/builders.py`

Transform the typed Python AST into Taytsh IR. Lowering reads types computed by phase 8, tracks narrowing context as it traverses control flow, resolves Python-specific semantics (floor division, value-returning `and`/`or`, truthiness, negative indexing, chained comparisons) into conventional forms, and emits typed IR nodes that map directly to target language constructs. The output is a complete `Module` ready for middleend annotation passes.

## Inputs

- **TypedAST**: from phase 8 (dict-AST with `_expr_type` annotations)
- **SymbolTable**: structs with fields, methods, `init_params`, `const_fields`, `param_to_field`
- **SigTable**: from phase 5 (function/method signatures, parameter defaults)
- **SubtypeRel**: from phase 7 (`hierarchy_root`, `node_types`)
- **Source text**: original Python source (for literal format detection)

## Goals

- **Semantic completeness**: every source operation has an IR representation
- **Language agnosticism**: IR nodes avoid Python-specific semantics
- **Direct mapping**: each IR node maps cleanly to target language constructs
- **Type preservation**: all expressions carry resolved types from phase 8

## Declarations

### Module Assembly

`build_module` walks the top-level AST body and assembles an IR `Module`:

1. Extract module docstring (first `Expr` if string constant)
2. Build constants (module-level `ALL_CAPS` assignments and class-level constants)
3. Build hierarchy root interface (if polymorphism detected), with `GetKind`/`ToSexp` methods and `kind` field
4. Build structs from `ClassDef` nodes, including method bodies and optional constructor functions
5. Build top-level functions with bodies
6. Append constructor functions after regular functions
7. Detect entry point (`if __name__ == "__main__": main()`)

### Structs

| Python                  | IR Output                             |
| ----------------------- | ------------------------------------- |
| `class Foo:`            | `Struct(name, fields, methods)`       |
| `class Foo(Base):`      | `Struct` with `implements=[Base]`     |
| `class E(Exception):`   | `Struct` with `is_exception=True`     |
| `@dataclass class Foo:` | `Struct` with fields from annotations |

Struct fields are ordered per `init_params` from phase 6. Constant discriminator fields (`const_fields`) are added to each struct. Constructor functions (`NewXxx`) are generated for classes with `needs_constructor=True`.

### Functions

| Python                  | IR Output                           |
| ----------------------- | ----------------------------------- |
| `def f(x: int) -> str:` | `Function(name, params, ret, body)` |
| `def m(self, x):`       | `Function` with `receiver`          |

Parameters carry types from phase 5. Default values are lowered to IR expressions. Mutated list parameters retain their `Pointer(Slice)` wrapping.

### Constants

| Python                | IR Output                         |
| --------------------- | --------------------------------- |
| `X = 42` (ALL_CAPS)   | `Constant(name, typ, value)`      |
| `Foo.BAR = 1` (class) | `Constant("Foo_BAR", typ, value)` |

### Entry Point

| Python                       | IR Output                   |
| ---------------------------- | --------------------------- |
| `if __name__ == "__main__":` | `EntryPoint(function_name)` |

The function name is extracted from the guarded call.

## Statements

| Python                       | IR Output                                  |
| ---------------------------- | ------------------------------------------ |
| `if cond:`                   | `If(cond, then_body, else_body)`           |
| `if isinstance(x, T):` chain | `TypeSwitch(expr, binding, cases)`         |
| `match x:`                   | `Match(expr, cases, default)`              |
| `for x in items:`            | `ForRange(index, value, iterable, body)`   |
| `for i in range(n):`         | `ForClassic(init, cond, post, body)`       |
| `for i in range(a, b):`      | `ForClassic(init, cond, post, body)`       |
| `for i in range(a, b, -1):`  | `ForClassic` with negative step            |
| `while cond:`                | `While(cond, body)`                        |
| `try: ... except E as e:`    | `TryCatch(body, catches)`                  |
| `raise E(msg)`               | `Raise(error_type, message)`               |
| `return expr`                | `Return(value)`                            |
| `x = expr`                   | `Assign(target, value)` or `VarDecl`       |
| `a, b = expr`                | `TupleAssign(targets, value)`              |
| `x += n`                     | `OpAssign(target, op, value)`              |
| `assert cond`                | `Assert(test, message)`                    |
| `print(x)`                   | `Print(value, newline=True, stderr=False)` |
| `print(x, end="")`           | `Print(value, newline=False)`              |
| `print(x, file=sys.stderr)`  | `Print(value, stderr=True)`                |
| `pass`                       | omitted                                    |
| `if __name__ == "__main__":` | `EntryPoint(function_name)`                |

### If Statements

If statements are lowered with narrowing-aware dispatch:

**isinstance chain** → `TypeSwitch`: when the condition is `isinstance(x, A)` and elif branches are `isinstance(x, B)`, etc. on the same variable, the entire chain is collapsed into a single `TypeSwitch` with per-case bindings.

**isinstance in AND** → narrowing: `if isinstance(x, T) and x.field:` narrows `x` to `T` in the then-branch body.

**Kind-based narrowing** → type narrowing: `if kind == "value":` narrows based on the `kind_to_class` mapping built from `const_fields`.

**Regular if** → `If` with condition converted to bool.

### For Statements

**range() patterns** → `ForClassic(init, cond, post, body)`:
- `range(n)` → `i = 0; i < n; i++`
- `range(a, b)` → `i = a; i < b; i++`
- `range(a, b, step)` → `i = a; i < b; i += step` (with `>` for negative step)

**Collection iteration** → `ForRange(index, value, iterable, body)`:
- Element type inferred from collection type
- Union type information propagated from list to loop variable
- `Pointer(Slice)` targets are dereferenced for iteration

**Tuple unpacking in for** → synthetic field accesses:
- `for a, b in list_of_tuples:` unpacks via `tuple.F0`, `tuple.F1` field access

**enumerate** → `ForRange` with index and value bindings extracted from the tuple target.

### Assignment

Assignment lowering handles several patterns:

**Simple assignment** → `Assign(target, value)` with type coercion.

**Tuple-returning function** → `TupleAssign` with synthetic variable expansion: `x = func()` where func returns `(A, B)` creates synthetic variables `x0`, `x1` tracked in `tuple_vars`.

**Tuple unpacking** → `TupleAssign(targets, value)` from function calls, method calls, tuple literals, or subscript access.

**Multiple targets** → `a = b = value` emits multiple `Assign` statements in a `Block`.

**Annotated assignment** → `AnnAssign` lowers with explicit type context from the annotation. `str | None` lowers to `string?`, `int | None` lowers to `int?`.

### Try / Except

`TryCatch(body, catches)` with catch clauses carrying exception type and binding name. Bare `raise` inside a catch is lowered as re-raise.

### Assert

`Assert(test, message)` with condition converted to bool and optional message extracted.

## Expressions

### Literals

| Python           | IR Output                                      |
| ---------------- | ---------------------------------------------- |
| `42`             | `IntLit(value, format?)` — format: hex/oct/bin |
| `3.14`           | `FloatLit(value, format?)` — format: exp       |
| `True` / `False` | `BoolLit(value)`                               |
| `"hello"`        | `StringLit(value)`                             |
| `b"\x89PNG"`     | `SliceLit(BYTE, [byte values])`                |
| `None`           | `NilLit(typ)` — typed nil                      |

Literal format detection reads the original source text to preserve notation (hex `0xff`, octal `0o77`, binary `0b101`, scientific `1e10`).

### Names

| Python          | IR Output                                          |
| --------------- | -------------------------------------------------- |
| `x` (variable)  | `Var(name, typ)` from `var_types`                  |
| `x` (constant)  | `Var(name, typ)` from `symbols.constants`          |
| `x` (tuple var) | `TupleLit(x0, x1, ...)` expanded from `tuple_vars` |

### Attribute Access

| Python       | IR Output                                        |
| ------------ | ------------------------------------------------ |
| `obj.field`  | `FieldAccess(obj, field, typ)`                   |
| `Cls.CONST`  | `Var("Cls_CONST", typ)` — class constant         |
| `node.field` | `TypeAssert` + `FieldAccess` if narrowing needed |
| `node.kind`  | `FieldAccess` directly (always on interface)     |

Node interface field access: only `kind` is accessible without narrowing. Other fields on interface-typed variables require a `TypeAssert` to the specific struct type, determined by the narrowing context.

### Subscript

| Python         | IR Output                                             |
| -------------- | ----------------------------------------------------- |
| `xs[i]`        | `Index(obj, index)` with element type                 |
| `xs[-1]`       | `Index(obj, Len(obj) - 1)` — negative index resolved  |
| `xs[a:b]`      | `SliceExpr(obj, low, high)`                           |
| `xs[:n]`       | `SliceExpr(obj, 0, n)`                                |
| `xs[n:]`       | `SliceExpr(obj, n, Len(obj))`                         |
| `s[i]`         | `CharAt(string, index)` — string indexing yields rune |
| `tuple[0]`     | `FieldAccess(tuple, "F0")` — tuple index to field     |
| `tuple_var[0]` | `Var("tuple_var0")` — expanded synthetic variable     |

Negative indices with literal `-N` are converted to `Len(obj) - N`. Dynamic indices that may be negative use a conditional form.

### Function Calls

**Built-in functions** → specialized IR nodes (see Built-in Translation below).

**Struct constructors**:
- Structs with `needs_constructor` → `Call("NewXxx", args)` to the generated constructor
- Simple structs → `StructLit(name, fields)` with fields ordered per `init_params`
- Keyword arguments merged to positional order via `param_to_field`
- Default values filled from field defaults
- Constant fields (`const_fields`) added automatically

**Free function calls** → `Call(name, args)`:
- Return type from `symbols.functions`
- Keyword and default arguments resolved
- Pointer-to-slice parameters: `&` added via `add_address_of_for_ptr_params`
- Slice-from-pointer: `*` dereference via `deref_for_func_slice_params`
- Interface coercion: arguments wrapped in `TypeAssert` when function expects Node types

**Method calls** → `MethodCall(obj, method, args)`:
- Receiver type determines dispatch
- Same argument handling as free functions

### Comprehensions

| Python            | IR Output                          |
| ----------------- | ---------------------------------- |
| `[x for x in xs]` | `ListComp(element, generators)`    |
| `{x for x in xs}` | `SetComp(element, generators)`     |
| `{k: v for ...}`  | `DictComp(key, value, generators)` |
| `(x for x in xs)` | `ListComp` (eagerly consumed)      |

Generator expressions are lowered as `ListComp` since the subset requires eager consumption.

### Ternary

`a if cond else b` → `Ternary(cond, then_expr, else_expr)`. Supports attribute-path kind narrowing in the condition.

### F-Strings

`f"hello {name}"` → `StringFormat(template, args)`. Constant parts are preserved; expressions are lowered and placed as arguments.

## Operators

### Arithmetic

| Python          | IR Output                                                                                                         |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `a + b` (int)   | `BinaryOp("+", left, right)` — type `int`                                                                         |
| `a + b` (float) | `BinaryOp("+", left, right)` — type `float`                                                                       |
| `s1 + s2`       | `Concat(left, right)` — string concat                                                                             |
| `a // b`        | `FloorDiv(left, right)` — floors toward -∞                                                                        |
| `a / b` (int)   | `BinaryOp("/", IntToFloat(left), IntToFloat(right))` — true division yields `float`                               |
| `a % b`         | `PythonMod(left, right)` — Python mod semantics                                                                   |
| `a ** b`        | `Pow(left, right)`                                                                                                |
| `divmod(a, b)`  | `(FloorDiv(left, right), PythonMod(left, right))` — Python floor semantics, not Taytsh `DivMod` (which truncates) |

`FloorDiv` is distinct from `BinaryOp("/")` because Python's `//` floors toward negative infinity, while Taytsh's int `/` truncates toward zero.

`PythonMod` emits `((a % b) + b) % b` for targets where `%` has different sign behavior.

### Bitwise

| Python   | IR Output                                  |
| -------- | ------------------------------------------ |
| `a & b`  | `BinaryOp("&", left, right)` — type `int`  |
| `a \| b` | `BinaryOp("\|", left, right)` — type `int` |
| `a ^ b`  | `BinaryOp("^", left, right)` — type `int`  |
| `a << b` | `BinaryOp("<<", left, right)`              |
| `a >> b` | `ArithmeticShiftRight(left, right)`        |
| `~x`     | `UnaryOp("~", operand)` — type `int`       |

`ArithmeticShiftRight` is distinct because Python's `>>` is arithmetic (sign-preserving), while some targets have logical right shift.

### Comparison

| Python           | IR Output                                       |
| ---------------- | ----------------------------------------------- |
| `a == b`         | `BinaryOp("==", left, right)`                   |
| `a < b`          | `BinaryOp("<", left, right)`                    |
| `a < b < c`      | `ChainedCompare(operands, ops)`                 |
| `x is None`      | `IsNil(expr)`                                   |
| `x is not None`  | `IsNil(expr, negated=True)`                     |
| `x in (a, b, c)` | `a == x \|\| b == x \|\| c == x`                |
| `x in coll`      | `Contains(coll, x)` (see Collection Operations) |

### Boolean

| Python    | IR Output                                   |
| --------- | ------------------------------------------- |
| `a and b` | `BinaryOp("&&", left, right)` — both bool   |
| `a or b`  | `BinaryOp("\|\|", left, right)` — both bool |
| `not x`   | `UnaryOp("!", as_bool(x))`                  |

Python's `and`/`or` return values (not booleans), but since the subset enforces boolean context, they lower to `&&`/`||`. The `not` operator first converts the operand to bool if needed.

**isinstance narrowing in AND**: `isinstance(x, T) and expr` temporarily narrows `x` to `T` while lowering `expr`.

### Unary

| Python  | IR Output                                        |
| ------- | ------------------------------------------------ |
| `-x`    | `UnaryOp("-", operand)` — preserves operand type |
| `+x`    | operand passed through                           |
| `not x` | `UnaryOp("!", as_bool(operand))`                 |
| `~x`    | `UnaryOp("~", operand)` — type `int`             |

## Built-in Translation

| Python             | IR Output                                   |
| ------------------ | ------------------------------------------- |
| `len(x)`           | `Len(expr)` — dereferences `Pointer(Slice)` |
| `min(a, b)`        | `Min(left, right)`                          |
| `max(a, b)`        | `Max(left, right)`                          |
| `abs(x)`           | `Abs(x)`                                    |
| `int(s)`           | `ParseInt(string, 10)`                      |
| `int(s, base)`     | `ParseInt(string, base)`                    |
| `float(s)`         | `ParseFloat(s)`                             |
| `str(n)` (int)     | `ToString(value)`                           |
| `bool(x)`          | type-specific conversion                    |
| `chr(n)`           | `RuneFromInt(n)`                            |
| `ord(c)`           | `RuneToInt(c)`                              |
| `isinstance(x, T)` | `IsType(expr, tested_type)`                 |
| `sorted(xs)`       | `Sorted(xs)`, reversed if `reverse=True`    |
| `list(xs)`         | copy or covariant list conversion           |

`bool(x)` conversion is type-specific: `int != 0`, `str != ""`, `len(coll) != 0`, etc.

`list(xs)` with covariant conversion (e.g., `list[Derived]` to `list[Interface]`) emits a conversion loop.

## Truthiness Resolution

Truthy tests are specialized by type to avoid ambiguity:

| Type                    | `if x:` lowers to           |
| ----------------------- | --------------------------- |
| `bool`                  | condition used directly     |
| `T?` / `Pointer`        | `IsNil(expr, negated=True)` |
| `InterfaceRef`          | `IsNil(expr, negated=True)` |
| `string`                | `Truthy(expr)` (non-empty)  |
| `Slice` / `Map` / `Set` | `Truthy(expr)` (non-empty)  |
| `int` / `float`         | `Truthy(expr)` (non-zero)   |

The `not` operator on non-bool types first converts to bool via the same type-specific rules, then applies `!`.

## String Operations

| Python                            | IR Output                                   |
| --------------------------------- | ------------------------------------------- |
| `s[i]`                            | `s[i]` — yields `rune`                      |
| `len(s)`                          | `Len(s)` — rune count                       |
| `s[a:b]`                          | `s[a:b]` — substring                        |
| `s1 + s2`                         | `Concat(s1, s2)`                            |
| `s1 == s2`, `s1 < s2`             | `==`, `<` (string comparison)               |
| `f"{x}..."`                       | `Format(template, args)`                    |
| `s.find(sub)`                     | `Find(s, sub)`                              |
| `s.rfind(sub)`                    | `RFind(s, sub)`                             |
| `s.split(sep)`                    | `Split(s, sep)`                             |
| `s.split()`                       | `SplitWhitespace(s)`                        |
| `s.replace(old, new)`             | `Replace(s, old, new)`                      |
| `s.count(sub)`                    | `Count(s, sub)`                             |
| `s.startswith(x)`                 | `StartsWith(s, x)`                          |
| `s.startswith((a, b))`            | `StartsWith` per prefix, joined with `\|\|` |
| `s.endswith(x)`                   | `EndsWith(s, x)`                            |
| `s.endswith((a, b))`              | `EndsWith` per suffix, joined with `\|\|`   |
| `s * n`                           | `Repeat(s, n)`                              |
| `s.strip(chars)`                  | `Trim(s, chars)`                            |
| `s.lstrip(chars)`                 | `TrimStart(s, chars)`                       |
| `s.rstrip(chars)`                 | `TrimEnd(s, chars)`                         |
| `s.lower()`                       | `Lower(s)`                                  |
| `s.upper()`                       | `Upper(s)`                                  |
| `sep.join(parts)`                 | `Join(sep, parts)`                          |
| `c.isdigit()`, `.isalpha()`, etc. | `IsDigit(c)`, `IsAlpha(c)`, etc.            |
| `s.encode("utf-8")`               | `Encode(s)`                                 |
| `for i, c in enumerate(s)`        | `for i, ch in s` (string iteration)         |

## Bytes Operations

| Python              | IR Output                  |
| ------------------- | -------------------------- |
| `b.decode("utf-8")` | `Decode(b)` — UTF-8 decode |
| `s.encode("utf-8")` | `Encode(s)` — UTF-8 encode |

## Collection Operations

### List

| Python            | IR Output                            |
| ----------------- | ------------------------------------ |
| `[a, b, c]`       | list literal with element type       |
| `[]` (typed)      | empty list literal with element type |
| `xs.append(v)`    | `Append(xs, v)`                      |
| `xs.extend(ys)`   | loop + `Append` per element          |
| `xs.insert(i, v)` | `Insert(xs, i, v)`                   |
| `xs.pop()`        | `Pop(xs)`                            |
| `xs.remove(v)`    | `RemoveAt(xs, IndexOf(xs, v))`       |
| `xs.index(v)`     | `IndexOf(xs, v)`                     |
| `xs.count(v)`     | loop counting `== v` matches         |
| `xs.copy()`       | `xs[0:Len(xs)]` (full slice)         |
| `xs.clear()`      | reassign empty list                  |
| `xs.reverse()`    | reassign `Reversed(xs)`              |
| `xs.sort()`       | reassign `Sorted(xs)`                |
| `x in xs`         | `Contains(xs, x)`                    |
| `xs * n`          | `Repeat(xs, n)`                      |
| `list1 == list2`  | `==` (deep structural)               |
| `list1 < list2`   | `<` (element-wise comparison)        |

Element types are inferred from the expected type context (bidirectional inference) or from the first element.

### Map

| Python               | IR Output                            |
| -------------------- | ------------------------------------ |
| `{k: v, ...}`        | map literal with key and value types |
| `{}` (typed)         | `Map()` (empty map)                  |
| `k in d`             | `Contains(d, k)`                     |
| `d.get(k)`           | `Get(d, k)`                          |
| `d.get(k, default)`  | `Get(d, k, default)`                 |
| `d.pop(k)`           | `Get(d, k)` + `Delete(d, k)`         |
| `d.setdefault(k, v)` | `Contains` + conditional insert      |
| `d.update(other)`    | loop insert                          |
| `d.keys()`           | `Keys(d)`                            |
| `d.values()`         | `Values(d)`                          |
| `d.items()`          | `Items(d)`                           |
| `d.copy()`           | `Merge(d, Map())`                    |
| `d.clear()`          | reassign `Map()`                     |
| `d1 \| d2`           | `Merge(d1, d2)`                      |
| `d1 \|= d2`          | in-place `Merge`                     |
| `map1 == map2`       | `==` (deep structural)               |

### Set

| Python              | IR Output                                   |
| ------------------- | ------------------------------------------- |
| `{a, b, c}`         | set literal with element type               |
| `set()` (typed)     | `Set()` (empty set)                         |
| `s.add(v)`          | `Add(s, v)`                                 |
| `s.remove(v)`       | `Remove(s, v)`                              |
| `s.discard(v)`      | `Contains` + `Remove` (no error if missing) |
| `s.pop()`           | iterate + `Remove`                          |
| `s.copy()`          | iterate + `Add` to new set                  |
| `s.clear()`         | reassign `Set()`                            |
| `s.union(t)`        | iterate + `Add` to new set                  |
| `s.intersection(t)` | iterate + `Contains` check + `Add`          |
| `s.difference(t)`   | iterate + `!Contains` check + `Add`         |
| `s.issubset(t)`     | iterate + `Contains` check                  |
| `s.issuperset(t)`   | iterate `t` + `Contains` in `s`             |
| `x in s`            | `Contains(s, x)`                            |
| `set1 == set2`      | `==` (deep structural)                      |

### Tuple

| Python      | IR Output                                    |
| ----------- | -------------------------------------------- |
| `(a, b, c)` | `TupleLit(elements)` — element types tracked |
| `t[0]`      | `FieldAccess(t, "F0")`                       |

Tuple element access is lowered to named field access (`F0`, `F1`, etc.) since tuples are represented as anonymous structs.

### Contains (Polymorphic)

The `in` operator dispatches by collection type:

| Collection | IR Output          |
| ---------- | ------------------ |
| `list[T]`  | `Contains(xs, v)`  |
| `set[T]`   | `Contains(s, v)`   |
| `map[K,V]` | `Contains(m, k)`   |
| `string`   | `Contains(s, sub)` |

## I/O

| Python                       | IR Output                     |
| ---------------------------- | ----------------------------- |
| `print(x)`                   | `Print(value, newline=True)`  |
| `print(x, end="")`           | `Print(value, newline=False)` |
| `print(x, file=sys.stderr)`  | `Print(value, stderr=True)`   |
| `sys.stdin.readline()`       | `ReadLine()`                  |
| `sys.stdin.read()`           | `ReadAll()`                   |
| `sys.stdin.buffer.read()`    | `ReadBytes()`                 |
| `sys.stdin.buffer.read(n)`   | `ReadBytesN(count)`           |
| `sys.stdout.buffer.write(b)` | `WriteOut(data)`              |
| `sys.stderr.buffer.write(b)` | `WriteErr(data)`              |
| `sys.argv`                   | `Args()`                      |
| `os.getenv(name)`            | `GetEnv(name, default=None)`  |
| `os.getenv(name, default)`   | `GetEnv(name, default)`       |

## Annotations

### Literal Format

Literal format is detected from the original source text and preserved on IR literal nodes:

| Literal Format | Stored As                | Purpose                  |
| -------------- | ------------------------ | ------------------------ |
| `0xff`         | `IntLit(format="hex")`   | preserve hex notation    |
| `0o77`         | `IntLit(format="oct")`   | preserve octal notation  |
| `0b101`        | `IntLit(format="bin")`   | preserve binary notation |
| `1e10`         | `FloatLit(format="exp")` | preserve scientific form |

### Source Location

Every IR node carries a `Loc` with `line`, `col`, `end_line`, `end_col` extracted from the Python AST node. Used for error reporting and source maps.

## Narrowing Context

Lowering maintains a `TypeContext` that tracks type narrowing as it traverses control flow.

### Context State

| Field                 | Purpose                                              |
| --------------------- | ---------------------------------------------------- |
| `var_types`           | current type of each variable in scope               |
| `narrowed_vars`       | variables with narrowed types in current branch      |
| `narrowed_attr_paths` | attribute paths (`("x", "body")`) narrowed by kind   |
| `kind_source_vars`    | maps alias `k` → original `x` for deferred narrowing |
| `kind_to_class`       | maps `"add"` → `"Add"` from `const_fields`           |
| `union_types`         | variables typed as union, tracking possible variants |
| `list_element_unions` | list variable → structs that can be in the list      |
| `unified_to_node`     | variables unified to the Node interface type         |
| `tuple_vars`          | tuple variable → synthetic field names               |
| `return_type`         | expected return type of the current function         |
| `expected`            | bidirectional type context for collection literals   |

### Narrowing Patterns

| Pattern                    | Mechanism                                         |
| -------------------------- | ------------------------------------------------- |
| `isinstance(x, T)`         | pre-computed in phase 8, read from `_expr_type`   |
| `x.kind == "value"`        | `kind_to_class` mapping built from `const_fields` |
| `k = x.kind; if k == "v":` | `kind_source_vars` tracks alias to original       |
| `x.body.kind == "value"`   | `narrowed_attr_paths` for nested attribute access |
| `if x is not None:`        | tracked per scope, reset at merge points          |

Narrowing state is saved before entering branches and restored after, so that narrowing in one branch does not leak to sibling branches.

### Tuple Variable Expansion

When a function returns a tuple and the result is assigned to a single variable, synthetic variables are created:

```
x = func_returning_pair()
# Creates x0, x1 in var_types
# x tracked in tuple_vars as ["x0", "x1"]
# Later: x[0] → Var("x0"), x[1] → Var("x1")
```

## Errors

| Condition                | Diagnostic                                    |
| ------------------------ | --------------------------------------------- |
| Unsupported AST node     | `lowering: unsupported node type '{type}'`    |
| Unknown struct reference | `lowering: unknown struct '{name}'`           |
| Missing expression type  | `lowering: no type for expression at {loc}`   |
| Invalid field access     | `lowering: '{struct}' has no field '{field}'` |

## Invariants

After lowering, these properties hold:

**Types**
- All `Expr` nodes have non-None `typ` field
- All `StructRef` names exist in `Module.structs`
- All `InterfaceRef` names exist in `Module.interfaces`

**Structure**
- No Python AST nodes remain in output
- All `Var` references resolve to visible declarations
- `Module.hierarchy_root` set if class hierarchy detected
- Exception structs have `is_exception=True`

**Semantics**
- Truthy tests emit type-specific nodes, not generic comparisons
- Constructors emit `StructLit` with all fields, not field-by-field assignment
- Character operations use semantic nodes (`CharAt`, `CharLen`, `Substring`)
- Collection methods emit Taytsh built-in calls (`Append`, `Get`, `Contains`, etc.)
- String operations emit Taytsh built-in calls (`Find`, `Split`, `Replace`, etc.)
- Arithmetic uses `FloorDiv`, `PythonMod`, `ArithmeticShiftRight` for Python-specific semantics
- Empty collections carry explicit element types
- Negative indices resolved to `Len(obj) - N`
- Chained comparisons preserved as `ChainedCompare`
- F-strings lowered to `StringFormat` with argument list

**Not yet set** (added by middleend):
- `is_reassigned`, `is_modified`, `is_unused` on variables/params
- `hoisted_vars` on control flow statements
- `needs_named_returns` on functions
- Ownership and escape annotations

## Postconditions

IR Module complete with:

| Field            | Content                                                   |
| ---------------- | --------------------------------------------------------- |
| `structs`        | all class definitions with fields and methods             |
| `interfaces`     | hierarchy root interface if polymorphism detected         |
| `functions`      | all function definitions with typed parameters and bodies |
| `constants`      | module-level constant bindings                            |
| `enums`          | enumeration definitions                                   |
| `exports`        | public API declarations                                   |
| `entrypoint`     | main function marker                                      |
| `hierarchy_root` | name of the polymorphic root class, or None               |

All IR nodes typed. No Python AST remnants. Ready for middleend annotation passes.
