# Tongues

Write your library once in Python. Get native, idiomatic code in every language.

---

## For What

Leaf dependencies. The code everyone imports but that imports nothing:

- Parsers and lexers
- Codecs (Base64, MessagePack, Protobuf wire format)
- Validators (JSON Schema, semver, email)
- Text algorithms (diff, Levenshtein, glob matching)
- Data structures (tries, bloom filters, ropes)
- Hashing and checksums
- State machines and interpreters

If it needs syscalls, networking, or threads—-Tongues isn't your tool. If you're content with stdin, stdout, stderr, argv, env vars, and main(), then Tongues is a perfect fit.

## Status

The transpiler currently supports these target languages, although output isn't yet idiomatic:

| Language   | Min Version  | Status    |
| ---------- | ------------ | --------- |
| C          | GCC 13       | Prototype |
| C#         | .NET 8       | Prototype |
| Dart       | Dart 3.2     | Prototype |
| Go         | Go 1.21      | Prototype |
| Java       | Temurin 21   | Prototype |
| Javascript | Node.js 21   | Prototype |
| Lua        | Lua 5.4      | Prototype |
| Perl       | Perl 5.38    | Prototype |
| PHP        | PHP 8.3      | Prototype |
| Python     | CPython 3.12 | Prototype |
| Ruby       | Ruby 3.2     | Prototype |
| Typescript | tsc 5.3      | Prototype |
| Swift      | Swift 5.9    | WIP       |
| Rust       | Rust 1.75    | WIP       |
| Zig        | Zig 0.11     | WIP       |

We target language versions from ~3 years ago. New enough for modern idioms, old enough to be everywhere—LTS distros, corporate environments, CI images. No bleeding-edge features, no legacy baggage.

## License

MIT
