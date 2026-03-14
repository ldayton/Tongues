#!/usr/bin/env node
"use strict";

const nodeFs = require("fs");
const nodePath = require("path");
const vm = require("vm");

// Load the transpiled tongues module into global scope.
// Same loading strategy as test-transpiled.js: convert top-level block-scoped
// declarations to var so they become globalThis properties.
globalThis.require = require;

const tonguesPath = nodePath.join(__dirname, "..", "lib", "tongues.js");
let code = nodeFs.readFileSync(tonguesPath, "utf-8");
code = code.replace(/^class\s+(\w+)/gm, "var $1 = class $1");
code = code.replace(/^const /gm, "var ");
code = code.replace(/^let /gm, "var ");
vm.runInThisContext(code, { filename: tonguesPath });

// ---- CLI wrapper (mirrors bin/tongues Python logic) ----

function shouldSkipFile(source) {
    const lines = source.split("\n", 5);
    for (const line of lines) {
        if (line.includes("tongues: skip")) return true;
    }
    return false;
}

function resolveImport(importingFile, module, level, projectRoot) {
    let relPath;
    if (level > 0) {
        let dirPath = nodePath.dirname(importingFile);
        let up = level - 1;
        while (up > 0) {
            dirPath = nodePath.dirname(dirPath);
            up -= 1;
        }
        relPath = module ? nodePath.join(dirPath, ...module.split(".")) : dirPath;
    } else {
        relPath = nodePath.join(projectRoot, ...module.split("."));
    }
    const initPath = nodePath.join(relPath, "__init__.py");
    try { if (nodeFs.statSync(initPath).isFile()) return initPath; } catch {}
    const modulePath = relPath + ".py";
    try { if (nodeFs.statSync(modulePath).isFile()) return modulePath; } catch {}
    return null;
}

function verifyProject(targetPath) {
    const result = new ProjectVerifyResult();
    let stat;
    try { stat = nodeFs.statSync(targetPath); } catch { return result; }
    if (stat.isFile()) {
        const source = nodeFs.readFileSync(targetPath, "utf-8");
        if (shouldSkipFile(source)) return result;
        result.file_results.set(targetPath, verify(frontend_parse_parse(source)));
        return result;
    }

    const projectRoot = targetPath;
    const pending = [];
    const visited = new Set();
    for (const entry of nodeFs.readdirSync(projectRoot)) {
        if (entry.endsWith(".py")) {
            const fullPath = nodePath.join(projectRoot, entry);
            try { if (nodeFs.statSync(fullPath).isFile()) pending.push(fullPath); } catch {}
        }
    }

    while (pending.length > 0) {
        const filePath = pending.pop();
        if (visited.has(filePath)) continue;
        visited.add(filePath);
        const source = nodeFs.readFileSync(filePath, "utf-8");
        if (shouldSkipFile(source)) continue;
        const astDict = frontend_parse_parse(source);
        result.file_results.set(filePath, verify(astDict));
        for (const imp of extract_imports(astDict)) {
            if (imp.level === 0) continue;
            const resolved = resolveImport(filePath, imp.module, imp.level, projectRoot);
            if (resolved !== null && !visited.has(resolved)) pending.push(resolved);
        }
    }
    return result;
}

function gatherProjectFiles(projectRoot) {
    const results = [];
    function walk(dir) {
        let entries;
        try { entries = nodeFs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
        const dirs = [];
        const files = [];
        for (const entry of entries) {
            if (entry.name.startsWith(".")) continue;
            if (entry.isDirectory() && entry.name !== "__pycache__") dirs.push(entry.name);
            else if (entry.isFile() && entry.name.endsWith(".py")) files.push(entry.name);
        }
        dirs.sort();
        files.sort();
        for (const fname of files) {
            const fullPath = nodePath.join(dir, fname);
            const relPath = nodePath.relative(projectRoot, fullPath);
            const source = nodeFs.readFileSync(fullPath, "utf-8");
            if (shouldSkipFile(source)) continue;
            results.push([relPath, source]);
        }
        for (const d of dirs) {
            walk(nodePath.join(dir, d));
        }
    }
    walk(projectRoot);
    results.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    return results;
}

function findInputArg(args) {
    const flagsWithValue = new Set(["--target", "--stop-at", "-o", "--output"]);
    let i = 0;
    while (i < args.length) {
        if (flagsWithValue.has(args[i])) i += 2;
        else if (args[i].startsWith("-")) i += 1;
        else return args[i];
    }
    return null;
}

// ---- Entry point ----

const args = process.argv.slice(2);

// Dispatch taytsh subcommand
if (args.length > 0 && args[0] === "taytsh") {
    process.exit(cli_main(args.slice(1)));
}

// Handle --verify flag
const verifyIdx = args.indexOf("--verify");
if (verifyIdx !== -1) {
    const verifyPath = args[verifyIdx + 1];
    if (verifyPath === undefined) {
        console.error("error: --verify requires a path argument");
        process.exit(2);
    }
    const result = verifyProject(verifyPath);
    const errors = result.errors();
    if (errors.length > 0) {
        for (const e of errors) {
            console.error(String(e));
        }
        process.exit(1);
    }
    process.exit(0);
}

// Check if positional arg is a directory (project mode)
const inputArg = findInputArg(args);
if (inputArg !== null) {
    let isDir = false;
    try { isDir = nodeFs.statSync(inputArg).isDirectory(); } catch {}
    if (isDir) {
        const [target, stop_at, strict_math, strict_tostring, , , output_file] = parse_args();
        const files = gatherProjectFiles(inputArg);
        if (files.length === 0) {
            console.error("error: no .py files found in directory");
            process.exit(1);
        }
        process.exit(main_project(files, target, stop_at, strict_math, strict_tostring, output_file));
    }
}

// Default: single-file mode
main();
