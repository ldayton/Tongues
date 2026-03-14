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

The transpiler currently supports these target languages:

| Language   | Min Version  | Status |
| ---------- | ------------ | ------ |
| Java       | Temurin 21   | Alpha  |
| Javascript | Node.js 21   | Alpha  |
| Perl       | Perl 5.38    | Alpha  |
| Python     | CPython 3.12 | Alpha  |
| Ruby       | Ruby 3.2     | Alpha  |
| C          | GCC 13       |        |
| C#         | .NET 8       |        |
| Dart       | Dart 3.2     |        |
| Go         | Go 1.21      |        |
| Lua        | Lua 5.4      |        |
| PHP        | PHP 8.3      |        |
| Rust       | Rust 1.75    |        |
| Swift      | Swift 6.0    |        |
| Typescript | tsc 5.3      |        |
| Zig        | Zig 0.14     |        |

We target language versions from ~3 years ago. New enough for modern idioms, old enough to be everywhere—LTS distros, corporate environments, CI images. No bleeding-edge features, no legacy baggage.

## License

MIT
