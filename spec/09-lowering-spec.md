# Phase 9: Lowering

**Module:** `frontend/lowering.py`

Transform typed AST into language-agnostic IR. Lowering reads types computed by Phase 8, tracks narrowing context as it traverses control flow, and emits IR nodes that map directly to target language constructs.

## Inputs

- **TypedAST**: from Phase 8 (dict-AST with `_type` annotations)
- **FieldTable**: from Phase 6 (for const_fields, field types, init_params)
- **SubtypeRel**: from Phase 7 (for hierarchy_root, ancestor chains)
- **SigTable**: from Phase 5 (for function signatures)

## Goals

- **Semantic completeness**: Every source operation has an IR representation
- **Language agnosticism**: IR nodes avoid Python-specific semantics
- **Direct mapping**: Each IR node maps cleanly to target language constructs
- **Type preservation**: All expressions carry resolved types from Phase 8

## Transformation Rules

### Statements

| Python                       | IR Output                                    |
| ---------------------------- | -------------------------------------------- |
| `if cond:`                   | `If(cond, then_body, else_body)`             |
| `if isinstance(x, T):` chain | `TypeSwitch(expr, binding, cases)` ³         |
| `match x:`                   | `Match(expr, cases, default)`                |
| `for x in items:`            | `ForRange(index, value, iterable, body)`     |
| `for i in range(n):`         | `ForClassic(init, cond, post, body)`         |
| `for i in range(a, b):`      | `ForClassic(init, cond, post, body)`         |
| `while cond:`                | `While(cond, body)`                          |
| `try: ... except E as e:`    | `TryCatch(body, catches, catch_unreachable)` |
| `raise E(msg, pos)`          | `Raise(error_type, message, pos)`            |
| `return expr`                | `Return(value)`                              |
| `x = expr`                   | `Assign(target, value)` or `VarDecl`         |
| `a, b = expr`                | `TupleAssign(targets, value)`                |
| `x += n`                     | `OpAssign(target, op, value)`                |
| `assert cond`                | `Assert(test, message)`                      |
| `print(x)`                   | `Print(value, newline=True, stderr=False)`   |
| `print(x, end="")`           | `Print(value, newline=False)`                |
| `print(x, file=sys.stderr)`  | `Print(value, stderr=True)`                  |
| `pass`                       | `NoOp()` or omit                             |
| `if __name__ == "__main__":` | `EntryPoint(function_name)`                  |

³ Each `TypeSwitch` case includes `narrowed_binding` name for targets requiring new bindings (e.g., `nodeExpr` for case `Expr`).

`TryCatch.catch_unreachable` is `True` for panic-based languages (Zig) where exceptions don't exist and catch blocks are emitted as dead code for structural compatibility.

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
| `a + b`, `a - b`, etc. | `BinaryOp(op, left, right)` ²           |
| `a < b < c`            | `ChainedCompare(operands, ops)`         |
| `-x`, `not x`, `~x`    | `UnaryOp(op, operand)`                  |
| `x and y`, `x or y`    | `BinaryOp("&&", ...)`, `BinaryOp("      |  | ", ...)` |
| `a if cond else b`     | `Ternary(cond, then_expr, else_expr)` ⁶ |
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
| `ClassName(args)`      | `StructLit(struct_name, fields)` ¹      |
| `[x for x in xs]`      | `ListComp(element, generators)`         |
| `{x for x in xs}`      | `SetComp(element, generators)`          |
| `{k: v for ...}`       | `DictComp(key, value, generators)`      |

¹ `StructLit.fields` ordered per `init_params` from Phase 6.

² Bool operands in arithmetic emit `Cast(BOOL, INT)` wrapper per Phase 8.

⁶ `Ternary` has optional flags: `needs_iife=True` for languages without ternary operator (Go), `needs_parens=True` for nested ternary disambiguation (PHP 8+), `then_may_be_falsy=True` when `a and b or c` idiom fails (Lua).

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

### Truthiness

Truthy tests are specialized by type for unambiguous backend emission:

| Context                   | IR Output                   |
| ------------------------- | --------------------------- |
| `if bool_expr:`           | condition used directly     |
| `if optional_expr:`       | `IsNil(expr, negated=True)` |
| `if string_expr:`         | `StringNonEmpty(expr)`      |
| `if list_expr:`           | `SliceNonEmpty(expr)`       |
| `if dict_expr:`           | `MapNonEmpty(expr)`         |
| `if set_expr:`            | `SetNonEmpty(expr)`         |
| `if int_expr:` (rejected) | error: ambiguous truthiness |

Generic `Truthy(expr)` is only emitted when type is unknown (e.g., `object`).

### Falsiness

The `not` operator on non-bool types emits `Falsy(expr)`, specialized by type:

| Context         | IR Output                    |
| --------------- | ---------------------------- |
| `not bool_expr` | `UnaryOp("!", expr)`         |
| `not optional`  | `IsNil(expr, negated=False)` |
| `not string`    | `StringEmpty(expr)`          |
| `not list`      | `SliceEmpty(expr)`           |
| `not dict`      | `MapEmpty(expr)`             |
| `not set`       | `SetEmpty(expr)`             |
| `not int`       | `BinaryOp("==", expr, 0)`    |

### IsNil vs IsEmpty

`IsNil` tests for null/None. `IsEmpty` tests for zero-length. For non-nullable collections, these are distinct:

| Expression          | Type (non-nullable) | IR Output        |
| ------------------- | ------------------- | ---------------- |
| `if xs:`            | `list[int]`         | `SliceNonEmpty`  |
| `if xs is not None` | `list[int]`         | `BoolLit(True)`  |
| `if xs:`            | `list[int] \| None` | error: ambiguous |

For nullable collections, explicit checks are required.

### String Operations

| Python                     | IR Output                                            |
| -------------------------- | ---------------------------------------------------- |
| `s[i]`                     | `CharAt(string, index)`                              |
| `len(s)` (char count)      | `CharLen(string)`                                    |
| `s[a:b]`                   | `Substring(string, low, high, clamp=True)`           |
| `s1 + s2`                  | `StringConcat(parts)`                                |
| `s1 == s2`, `s1 < s2`      | `StringCompare(left, right, op)`                     |
| `f"{x}..."`                | `StringFormat(template, args)`                       |
| `s.find(sub)`              | `StringFind(string, needle, not_found=-1)`           |
| `s.rfind(sub)`             | `StringRfind(string, needle, not_found=-1)`          |
| `s.split(sep)`             | `StringSplit(string, sep, maxsplit, whitespace)`     |
| `s.split()`                | `StringSplit(string, None, -1, whitespace=True)`     |
| `s.replace(old, new)`      | `StringReplace(string, pattern, repl, literal=True)` |
| `s.endswith(x)`            | `StringEndsWith(string, suffix)`                     |
| `s.endswith((a, b))`       | `StringEndsWithAny(string, suffixes)`                |
| `s.startswith(x)`          | `StringStartsWith(string, prefix)`                   |
| `s.startswith((a, b))`     | `StringStartsWithAny(string, prefixes)`              |
| `s * n`                    | `StringRepeat(string, count, clamp_negative=True)`   |
| `s.strip(chars)`           | `TrimChars(string, chars, mode="both")`              |
| `s.lstrip(chars)`          | `TrimChars(string, chars, mode="left")`              |
| `s.rstrip(chars)`          | `TrimChars(string, chars, mode="right")`             |
| `c.isalnum()`              | `CharClassify(kind="alnum", char)`                   |
| `c.isdigit()`              | `CharClassify(kind="digit", char)`                   |
| `c.isalpha()`              | `CharClassify(kind="alpha", char)`                   |
| `c.isspace()`              | `CharClassify(kind="space", char)`                   |
| `c.isupper()`              | `CharClassify(kind="upper", char)`                   |
| `c.islower()`              | `CharClassify(kind="lower", char)`                   |
| `for i, c in enumerate(s)` | `ForStringChars(string, index, value, body)`         |

The `literal` flag on `StringReplace` indicates the pattern should be treated as literal text, not a regex. Backends using regex-based replacement (Perl, Ruby) apply appropriate escaping.

### Bytes Operations

| Python               | IR Output                               |
| -------------------- | --------------------------------------- |
| `b.decode()`         | `BytesDecode(bytes, encoding="utf8")`   |
| `b.decode("latin1")` | `BytesDecode(bytes, encoding)`          |
| `s.encode()`         | `StringEncode(string, encoding="utf8")` |
| `str(b)`             | `BytesDecode(bytes, encoding="utf8")`   |
| `bytes(s, "utf8")`   | `StringEncode(string, encoding)`        |

### Collection Operations

| Python              | IR Output                                        |
| ------------------- | ------------------------------------------------ |
| `x in set_`         | `SetContains(set, element)`                      |
| `x in map_`         | `MapContains(map, key)`                          |
| `x in list_`        | `SliceContains(slice, element)`                  |
| `x in string`       | `StringContains(string, substring)`              |
| `x in tuple_set`    | `TupleSetContains(set, element)`                 |
| `tuple_map[k]`      | `TupleMapGet(map, key)`                          |
| `len(list_)`        | `Len(slice)`                                     |
| `len(map_)`         | `MapLen(map)`                                    |
| `len(set_)`         | `SetLen(set)`                                    |
| `list_.pop()`       | `ListPop(list, index=None)`                      |
| `list_.pop(i)`      | `ListPop(list, index)`                           |
| `list_.append(x)`   | `SliceAppend(slice, element)`                    |
| `list_.extend(xs)`  | `SliceExtend(slice, elements)`                   |
| `list_.insert(i,x)` | `ListInsert(list, index, val, clamp=True)`       |
| `list_ * n`         | `ArrayRepeat(array, count, clamp_negative=True)` |
| `map_.get(k)`       | `MapGet(map, key, default=None)`                 |
| `map_.get(k, d)`    | `MapGet(map, key, default)`                      |
| `map1 \| map2`      | `MapMerge(left, right)`                          |
| `map1 \|= map2`     | `MapMergeInPlace(target, value)`                 |
| `set_.add(x)`       | `SetAdd(set, element)`                           |
| `d.keys()`          | `DictKeys(map)`                                  |
| `d.values()`        | `DictValues(map)`                                |
| `d.items()`         | `DictItems(map)`                                 |
| `d.keys() & other`  | `SetOp(DictKeys(d), other, op="&")`              |
| `list1 == list2`    | `ArrayEquals(left, right)`                       |
| `map1 == map2`      | `MapEquals(left, right)`                         |
| `set1 == set2`      | `SetEquals(left, right)`                         |
| `list1 < list2`     | `ArrayCompare(left, right, op)`                  |

Dict views (`DictKeys`, `DictValues`, `DictItems`) support set operations when used with `&`, `|`, `-`, `^`.

`TupleSetContains` and `TupleMapGet` are emitted when sets/maps contain tuple elements/keys, since these require value equality rather than reference equality.

`MapLen` and `SetLen` are distinct from `Len` because some languages (Lua) require iteration to compute map/set length.

`ListInsert` with `clamp=True` indicates out-of-bounds indices are clamped to valid range (Python semantics) rather than raising errors.

### Arithmetic Operations

| Python        | IR Output                             |
| ------------- | ------------------------------------- |
| `a // b`      | `FloorDiv(left, right)`               |
| `a / b` (int) | `FloorDiv(left, right)`               |
| `a % b`       | `PythonMod(left, right)`              |
| `divmod(a,b)` | `DivMod(left, right)`                 |
| `a >> b`      | `ArithmeticShiftRight(left, right)` ⁴ |
| `a << b`      | `BinaryOp("<<", left, right)`         |

⁴ Python's `>>` is arithmetic (sign-preserving). Some languages have logical right shift. `ArithmeticShiftRight` emits `// (1 << n)` or equivalent for correct semantics.

`FloorDiv` is distinct from `BinaryOp("/")` because Python's `//` always floors toward negative infinity, while many languages truncate toward zero.

`PythonMod` emits `((a % b) + b) % b` for languages where `%` has different sign behavior. Backends may optimize when operands are known non-negative.

`DivMod` returns a tuple of `(quotient, remainder)`. Backends may inline as `TupleLit(a // b, a % b)` or use native divmod operations.

### Boolean Operations

| Python    | IR Output                          |
| --------- | ---------------------------------- |
| `a and b` | `PythonAnd(left, right)` ⁵         |
| `a or b`  | `PythonOr(left, right)` ⁵          |
| `not x`   | `UnaryOp("!", operand)` or `Falsy` |

⁵ Python's `and`/`or` return values, not booleans: `x and y` returns `x` if falsy, else `y`. `PythonAnd` and `PythonOr` preserve these semantics. When both operands are `bool` and result is used as `bool`, backends may emit native `&&`/`||`.

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

| Python                  | IR Output                             |
| ----------------------- | ------------------------------------- |
| `class Foo:`            | `Struct(name, fields, methods)`       |
| `class Foo(Base):`      | `Struct` with `implements=[Base]`     |
| `class E(Exception):`   | `Struct` with `is_exception=True`     |
| `def f(x: int) -> str:` | `Function(name, params, ret, body)`   |
| `def m(self, x):`       | `Function` with `receiver`            |
| `X = 42` (module level) | `Constant(name, typ, value)`          |
| `f` (function as value) | `FuncRef(name, needs_boxing=True)` ⁷  |
| `obj.method` (no call)  | `FuncRef(name, receiver, bound=True)` |

⁷ `needs_boxing` indicates the function reference is used as a first-class value (assigned, passed, returned) and may need wrapping (e.g., `(Action)` cast in C#).

### Entrypoint

| Python                       | IR Output                                  |
| ---------------------------- | ------------------------------------------ |
| `if __name__ == "__main__":` | `EntryPoint(function_name, internal_name)` |

The `internal_name` field (e.g., `_main`) is used when the target language requires a wrapper function. Swift and Zig emit `_{name}` internally to avoid conflicts with the language's `main` convention.

### Special Patterns

| Python                            | IR Output                                   |
| --------------------------------- | ------------------------------------------- |
| `xs[-1]`                          | `LastElement(sequence)`                     |
| `xs[len(xs) - n]`                 | `LastElement(sequence, offset=n)`           |
| `xs[i]` (Lua target)              | `Index(obj, index, base=1)`                 |
| `&expr` (address-of)              | `AddrOf(operand)`                           |
| Sentinel int `== None`            | `BinaryOp("==", expr, IntLit(sentinel))`    |
| `x if sentinel else None`         | `SentinelToOptional(expr, sentinel)`        |
| `[]T` to `[]Interface` conversion | `SliceConvert(source, target_element_type)` |
| Struct assigned to interface var  | `InterfaceCast(expr, target_interface)`     |
| `d[k1][k2]` (nested dict)         | `NestedMapIndex(outer, k1, k2)`             |
| `d[k].append(x)` (mutate value)   | `MutableMapAccess(map, key)` + method       |
| Empty `{}` with known type        | `MapLit(key_type, value_type, [])`          |
| Empty `[]` with known type        | `SliceLit(element_type, [])`                |
| Empty `set()` with known type     | `SetLit(element_type, [])`                  |
| `range(n)`                        | `Range(0, n, 1)`                            |
| `range(a, b, -1)`                 | `Range(a, b, -1, negative_step=True)`       |

The `base` field on `Index` is `0` by default; set to `1` for 1-indexed languages (Lua). Lowering adds the annotation; backends adjust accordingly.

`NestedMapIndex` handles chained dictionary access where the outer access needs `.get_mut()` (Rust) or similar mutable access patterns.

`MutableMapAccess` is emitted when a dict value is accessed and then mutated (e.g., `d[k].append(x)`). This enables backends to use mutable reference patterns.

Empty collection literals carry explicit element types to enable turbofish syntax (Rust) or explicit type parameters where inference fails.

---

## Narrowing Context

Lowering tracks type narrowing as it traverses control flow. Types from Phase 8 are primary; on-demand narrowing handles patterns requiring flow context.

| Pattern                    | Mechanism                                                 |
| -------------------------- | --------------------------------------------------------- |
| `isinstance(x, T)`         | Pre-computed in Phase 8, read from `_expr_type`           |
| `x.kind == "value"`        | `kind_to_class` mapping built from const_fields (Phase 6) |
| `k = x.kind; if k == "v":` | `kind_source_vars` tracks alias to original               |
| `x.body.kind == "value"`   | `narrowed_attr_paths` for nested access                   |
| `if x is not None:`        | Tracked per-scope, reset at merge points                  |

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
- Truthy tests emit type-specific nodes (`StringNonEmpty`, `SliceNonEmpty`, etc.), not expanded comparisons
- Falsy tests emit type-specific nodes (`StringEmpty`, `SliceEmpty`, etc.) or `Falsy` for unknown types
- Constructors emit `StructLit` with all fields, not field-by-field assignment
- Bound method references have `FuncType.receiver` set
- Character operations use semantic nodes (`CharAt`, `CharLen`, `Substring`)
- Struct-to-interface assignments emit `InterfaceCast`
- Collection methods emit type-specific IR (`SliceAppend`, `MapGet`, `SetContains`, etc.)
- Collection length uses `MapLen`/`SetLen` for maps/sets, `Len` for slices
- String operations emit semantic nodes (`StringFind`, `StringSplit`, `StringReplace`, etc.)
- Arithmetic uses `FloorDiv`, `PythonMod`, `ArithmeticShiftRight` for Python-specific semantics
- Boolean `and`/`or` emit `PythonAnd`/`PythonOr` when value-returning behavior is needed
- Empty collections carry explicit element types for type inference in target languages
- Tuple-element collections use `TupleSetContains`/`TupleMapGet` for value equality

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
- `entrypoint`: Main function marker with `internal_name` (if `if __name__ == "__main__"` present)
- `used_tuple_types`: Set of tuple signatures for backends needing type definitions (C, Java)
- `used_slice_types`: Set of slice element types for backends needing type definitions (C)

All IR nodes typed. No AST remnants. Ready for middleend annotation passes.

---

## Prior Art

- [Three-address code](https://en.wikipedia.org/wiki/Three-address_code)
- [Cornell CS 4120 IR notes](https://www.cs.cornell.edu/courses/cs4120/2023sp/notes/ir/)
- [TypeScript Flow Nodes](https://effectivetypescript.com/2024/03/24/flownodes/)
- [Pyright Lazy Evaluation](https://github.com/microsoft/pyright/blob/main/docs/mypy-comparison.md)
