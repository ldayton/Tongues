# Tongues

Write algorithms once in Python. Get native, idiomatic code in every language.

---

## The Problem

You build something foundational—a parser, a codec, a validator. It's good. Now npm needs it. PyPI needs it. Crates.io needs it. Go needs it.

Your options:
- **Manual ports** — They drift. Bugs fixed in one, not the others.
- **FFI bindings** — Deployment hell. Performance overhead. Build complexity.
- **WASM everywhere** — Runtime bloat. Ecosystem friction. Not actually native.

## The Solution

Write it once in restricted Python. Get native, human-readable code in 10+ languages. One test suite. One source of truth. Zero runtime dependencies.

```
source.py → frontend → IR → backend → target code (Go, Rust, etc)
```

Your Python code *runs*. Test with pytest, debug with pdb, iterate fast—then emit to every ecosystem.

## Backends

| Language   | Reference        | Homebrew         | Status      |
|------------|------------------|------------------|-------------|
| Bash       | GNU Bash   5.2   | `bash`           | Future      |
| C          | Clang       18   | `llvm@18`        | In progress |
| C#         | .NET       9.0   | `dotnet@9`       | In progress |
| Go         | Go        1.23   | `go@1.23`        | In progress |
| Java       | OpenJDK     21   | `openjdk@21`     | In progress |
| Lua        | Lua        5.4   | `lua`            | Future      |
| Perl       | Perl      5.40   | `perl`           | Future      |
| PHP        | PHP        8.3   | `php@8.3`        | Future      |
| Python     | CPython   3.12   | `python@3.12`    | In progress |
| Ruby       | CRuby      3.3   | `ruby@3.3`       | In progress |
| Rust       | rustc     1.82   | `rust`           | In progress |
| Swift      | Swift     5.10   | `swift`          | In progress |
| TypeScript | tsc        5.6   | `typescript`     | In progress |
| Zig        | Zig       0.13   | `zig`            | In progress |

We target language versions from ~3 years ago. New enough for modern idioms, old enough to be everywhere—LTS distros, corporate environments, CI images. No bleeding-edge features, no legacy baggage.

## For What

Leaf dependencies. The code everyone imports but that imports nothing:

- Parsers and lexers
- Codecs (Base64, MessagePack, Protobuf wire format)
- Validators (JSON Schema, semver, email)
- Text algorithms (diff, Levenshtein, glob matching)
- Data structures (tries, bloom filters, ropes)
- Hashing and checksums
- State machines and interpreters

If it needs syscalls, networking, or threads—we're not your tool. If it's pure transforms on in-memory data—we are.

## Output Quality

Not transpiled garbage. Idiomatic code that looks like a native wrote it:

- Proper language conventions (camelCase in JS, snake_case in Rust)
- Native types (Go slices, Rust Vecs, TS arrays)
- Clean control flow (no goto spaghetti)
- Readable enough to debug directly
- Python output is more idiomatic than the restricted Python input

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Python    │     │   Frontend  │     │  Middleend  │     │   Backend   │
│   Source    │────▶│  (Python →  │────▶│   (IR →     │────▶│  (IR →      │
│             │     │     IR)     │     │  annotated) │     │   target)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

- **Frontend**: Python AST to language-agnostic IR. All source semantics resolved here.
- **Middleend**: Scope analysis, flow tracking, annotations. Read-only—never transforms.
- **Backend**: Pure syntax emission. No source knowledge, no target heuristics.

## Proof

[Parable](https://github.com/ldayton/Parable)—an 11,000-line bash parser. Python, JavaScript, Go, all from one source. 20,000+ tests passing on all targets.

## Status

Extracting from Parable.

## License

MIT
