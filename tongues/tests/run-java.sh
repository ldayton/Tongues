#!/usr/bin/env bash
# Compile and run a Java program read from stdin.
# Used as the RUNTIMES entry for Java app/ordering tests.
set -euo pipefail
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
cat > "$tmpdir/Main.java"
javac -encoding UTF-8 "$tmpdir/Main.java" -d "$tmpdir" 2>&1
java -cp "$tmpdir" Main
