# Phase 15: Backend

**Module:** `backend/<lang>.py`

Emit target language source from annotated IR. Each backend is a single module that walks the IR and produces output text.

| Target | Module    | Output                   |
| ------ | --------- | ------------------------ |
| Go     | `go.py`   | `.go` source files       |
| C      | `c.py`    | `.c` / `.h` source files |
| Rust   | `rust.py` | `.rs` source files       |

Walk the annotated IR and emit target language source. The backend reads all annotations from phases 1–14 but adds none—pure output generation.

## IR to Target Mapping

| IR Node      | Go Output                | C Output                  | Rust Output             |
| ------------ | ------------------------ | ------------------------- | ----------------------- |
| `Function`   | `func name(...) { ... }` | `type name(...) { ... }`  | `fn name(...) { ... }`  |
| `Struct`     | `type Name struct { }`   | `typedef struct { } Name` | `struct Name { }`       |
| `VarDecl`    | `var x T` or `x := ...`  | `T x = ...`               | `let mut x = ...`       |
| `MethodCall` | `obj.Method(args)`       | `Method(obj, args)`       | `obj.method(args)`      |
| `Print`      | `fmt.Println(x)`         | `printf("%s\n", x)`       | `println!("{}", x)`     |
| `ReadLine`   | `bufio.Scanner`          | `fgets()`                 | `stdin().read_line()`   |
| `ReadAll`    | `io.ReadAll(os.Stdin)`   | `read()` loop             | `read_to_string(stdin)` |
| `Args`       | `os.Args`                | `argv[0..argc]`           | `env::args().collect()` |
| `GetEnv`     | `os.Getenv()`            | `getenv()`                | `env::var().ok()`       |

## Middleend Annotation Consumption

The backend consumes middleend annotations:
- `is_reassigned` → Go: `var` vs `:=`; Rust: `let mut` vs `let`; TS: `let` vs `const`
- `hoisted_vars` → Go: emit declarations before control structure
- `needs_named_returns` → Go: use named return values
- `initial_value_unused` → Go: omit initializer, use zero value
- `is_interface` → Go: direct `== nil` vs `_isNilInterface()` reflection
- `narrowed_type` → Java/TS: omit redundant casts when type is known
- `rune_vars` → Go: emit `[]rune` conversion at scope entry
- `ownership=owned` → Rust: owned value; C: arena-allocated
- `ownership=borrowed` → Rust: `&T`; C: pointer to caller's memory
- `ownership=shared` → Rust: `Rc<T>`; C: refcounted wrapper
- `ownership=weak` → Rust: `Weak<T>`; C: non-owning pointer (back-refs)

## Postconditions

Valid target language source emitted; all IR nodes consumed; output compiles with target toolchain.
