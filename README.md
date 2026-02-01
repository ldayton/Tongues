# Tongues

Write algorithms once in Python. Get native, idiomatic code in every language.

---

## Status

The transpiler currently supports these target languages, although output isn't yet idiomatic:

| Language   | Reference    | Released | Status      |
| ---------- | ------------ | -------- | ----------- |
| Go         | Go 1.21      | Aug 2023 | Prototype   |
| Java       | Temurin 21   | Sep 2023 | Prototype   |
| Javascript | Node.js 21   | Oct 2023 | Prototype   |
| Lua        | Lua 5.4      | May 2023 | Prototype   |
| Python     | CPython 3.12 | Oct 2023 | Prototype   |
| Ruby       | Ruby 3.3     | Dec 2023 | Prototype   |
| Typescript | tsc 5.3      | Nov 2023 | Prototype   |
| C#         | .NET 8       | Nov 2023 | Prototype   |
| Perl       | Perl 5.38    | Jul 2023 | Prototype   |
| PHP        | PHP 8.3      | Nov 2023 | Prototype   |
| C          | GCC 13       | Jul 2023 | In Progress |
| Dart       | Dart 3.2     | Nov 2023 | Future      |
| Rust       | Rust 1.75    | Dec 2023 | Future      |
| Swift      | Swift 5.9    | Sep 2023 | Future      |
| Zig        | Zig 0.11     | Aug 2023 | Future      |

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

If it needs syscalls, networking, or threads—-Tongues isn't your tool. If you're content with stdin, stdout, stderr, argv, env vars, and main(), then Tongues is a perfect fit.

## Proof

[Parable](https://github.com/ldayton/Parable)—an 11,000-line bash parser. Python, JavaScript, Go, all from one source. 20,000+ tests passing on all targets.

## License

MIT
