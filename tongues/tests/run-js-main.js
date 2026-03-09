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

// Intercept process.exit to wait for stdout to flush
const originalExit = process.exit;
process.exit = (code) => {
    // Force synchronous write of any pending stdout data
    if (process.stdout._writableState && process.stdout._writableState.buffered.length > 0) {
        const nodeFs = require("fs");
        for (const chunk of process.stdout._writableState.buffered) {
            nodeFs.writeSync(1, chunk.chunk);
        }
    }
    originalExit(code);
};

vm.runInThisContext(code, { filename: resolved });
main();
