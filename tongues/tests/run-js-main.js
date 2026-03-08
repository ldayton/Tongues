#!/usr/bin/env node
"use strict";

// Helper to load a transpiled JS binary and call main().
// Usage: node run-js-main.js <path-to-transpiled.js> [args...]

const nodeFs = require("fs");
const nodePath = require("path");
const vm = require("vm");

const filePath = process.argv[2];
if (!filePath) {
    process.stderr.write("Usage: node run-js-main.js <file.js> [args...]\n");
    process.exit(1);
}

const resolved = nodePath.resolve(filePath);
let code = nodeFs.readFileSync(resolved, "utf-8");

// Convert block-scoped top-level declarations to var (same as test-transpiled.js)
code = code.replace(/^class\s+(\w+)/gm, "var $1 = class $1");
code = code.replace(/^const /gm, "var ");
code = code.replace(/^let /gm, "var ");

globalThis.require = require;
process.argv = ["node", resolved, ...process.argv.slice(3)];

vm.runInThisContext(code, { filename: resolved });
main();
