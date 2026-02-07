# Phase 9: Lowering

**Module:** `frontend/lowering.py`

Translate TypedAST to IR. Lowering primarily reads types from Phase 8, but may invoke type inference for expressions requiring full narrowing context. Pattern-match on AST nodes and emit IR.

| AST Node                 | IR Output                              |
| ------------------------ | -------------------------------------- |
| `BinOp(+, a, b)` : `int` | `ir.BinOp(Add, lower(a), lower(b))`    |
| `Call(f, args)` : `T`    | `ir.Call(f, [lower(a) for a in args])` |
| `Attribute(obj, field)`  | `ir.FieldAccess(lower(obj), field)`    |

## Type Resolution Strategy

Following TypeScript's architecture, Phase 9 builds narrowing context (tracking `isinstance` checks, kind comparisons, alias assignments) as it traverses control flow. When a pre-computed type is unavailable, it requests the narrowed type on-demand.

| Pattern                  | Strategy                               |
| ------------------------ | -------------------------------------- |
| `isinstance(x, T)`       | Pre-computed in Phase 8                |
| `x.kind == "value"`      | On-demand with kind→struct mapping     |
| `kind = x.kind; if ...`  | On-demand with alias tracking          |
| `x.attr.kind == "value"` | On-demand with attribute path tracking |

## Postconditions

- IR Module complete; all IR nodes typed; no AST remnants in output
- Truthy checks (`if x`, `if s`) emit `Truthy(expr)`, not `BinaryOp(Len(x), ">", 0)`
- No marker variables (`_pass`, `_skip_docstring`); use `NoOp` or omit
- Bound method references emit `FuncRef(obj, method)` with `FuncType.receiver` set; backends emit correct function pointer signatures
- String operations emit semantic IR: `CharAt`, `CharLen`, `Substring` (not Python method names)
- Character classification emits semantic IR: `IsAlnum`, `IsDigit`, `IsAlpha`, `IsSpace`, `IsUpper`, `IsLower`; backends map to `unicode.IsLetter`/`Character.isDigit`/regex
- String trimming with char set emits `TrimChars(expr, chars, mode)` where mode is `left`/`right`/`both`; backends map to regex or stdlib
- `TryCatch` carries exception type for catch clause
- Constructors emit `StructLit` with all fields (not field-by-field assignment)
- `range()` iteration emits `ForClassic` (not `Call` to range helper)
- While loops with index iteration pattern (`while i < len(x): ... x[i] ... i += 1`) emit `ForRange` or `ForClassic`; enables Go `for i, c := range` and avoids manual index management
- Module exports emit `Export` nodes (not hardcoded in backend)
- Enum definitions emit `Enum` IR (not prefixed constants)
- Exception classes marked with `extends_error: bool` for target Error inheritance
- Numeric conversion emits `ParseInt(expr)` / `IntToStr(expr)` semantic IR (not helper function names); backends map to `int()`/`strconv.Atoi()`/`Integer.parseInt()`
- Sentinel-to-optional conversion emits `SentinelToOptional(expr, sentinel)` IR; backends map to `None if x == sentinel else x` / `if x == sentinel { return nil }`
- Index with `-1` emits `LastElement(expr)` IR; backends map to `[-1]`/`[len-1]`/`getLast()`
- Conditional expressions emit `Ternary(cond, then, else)` with `needs_statement: bool` flag; backends without ternary operator emit if/else variable assignment
- Pointer creation emits `AddrOf(expr)` semantic IR (not helper function)
- Interface definitions emit `InterfaceDef` with explicit field list (not inferred by backend); includes discriminant fields like `kind: string` for tagged unions
- Type switch emits `TypeSwitch` with `binding` field containing the narrowed variable name; no hardcoded naming conventions in backend
- Slice covariance emits `SliceConvert(source, target_element_type)` when element types differ but are compatible; backends handle covariant/invariant semantics per language
- Comprehensions emit `ListComp`, `SetComp`, `DictComp` with `element`, `target`, `iterable`, `condition` fields; backends emit idiomatic loops or builtins
- Slicing emits `SliceExpr(obj, low, high, step)` with optional fields; backends map to `[a:b]`/`substring()`/`subList()`
- Bitwise operations emit `BinaryOp(op, left, right)` where op is `&`/`|`/`^`/`~`/`<<`/`>>`
- Entry point `main() -> int` with `if __name__ == "__main__"` emits `EntryPoint` IR; backends emit `func main()` / `public static void main()` / `fn main()`
- `print(x)` emits `Print(expr, newline=True, stderr=False)`; `print(x, end='')` sets `newline=False`; `print(x, file=sys.stderr)` sets `stderr=True`
- `sys.stdin.readline()` emits `ReadLine()`; `sys.stdin.read()` emits `ReadAll()`
- `sys.stdin.buffer.read()` emits `ReadBytes()`; `sys.stdin.buffer.read(n)` emits `ReadBytesN(n)`
- `sys.stdout.buffer.write(b)` emits `WriteBytes(expr, stderr=False)`
- `sys.argv` emits `Args` IR; backends map to `os.Args`/`args`/`argv`
- `os.getenv(name)` emits `GetEnv(name, default)` with optional default

## Prior Art

- [Three-address code](https://en.wikipedia.org/wiki/Three-address_code)
- [Cornell CS 4120 IR notes](https://www.cs.cornell.edu/courses/cs4120/2023sp/notes/ir/)
- [TypeScript Flow Nodes](https://effectivetypescript.com/2024/03/24/flownodes/)
- [Pyright Lazy Evaluation](https://github.com/microsoft/pyright/blob/main/docs/mypy-comparison.md)
