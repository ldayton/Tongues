#!/usr/bin/env node
"use strict";

// Native Node.js test harness for transpiled Tongues binaries.
// Loads the transpiled file once, then runs all .tests cases in-process.
// Shared parsing/assertion logic is transpiled from tests/shared/test_harness.py.

const nodeFs = require("fs");
const nodePath = require("path");
const vm = require("vm");
const os = require("os");
const { spawnSync } = require("child_process");

const TONGUES_DIR = nodePath.resolve(__dirname, "..");
const TESTS_DIR = nodePath.join(TONGUES_DIR, "tests");
const LIB_DIR = nodePath.join(TONGUES_DIR, "src", "lib");

// Phase -> test config: {dir, run, taytsh?, args?, json?}
// Runners: cli, linker, phase, lowering, codegen, emit, app, ordering, ty_app
const TESTS = [
    ["cli", [
        ["cli", { dir: "frontend/cli", run: "cli" }],
    ]],
    ["linker", [
        ["linker", { dir: "frontend/linker", run: "linker" }],
    ]],
    ["frontend", [
        ["parse",     { dir: "frontend/parse",      run: "phase", taytsh: false, args: ["--stop-at", "parse"],      json: true  }],
        ["subset",    { dir: "frontend/subset",      run: "phase", taytsh: false, args: ["--stop-at", "subset"],     json: false }],
        ["names",     { dir: "frontend/names",       run: "phase", taytsh: false, args: ["--stop-at", "names"],      json: true  }],
        ["sigs",      { dir: "frontend/signatures",  run: "phase", taytsh: false, args: ["--stop-at", "signatures"], json: true  }],
        ["fields",    { dir: "frontend/fields",      run: "phase", taytsh: false, args: ["--stop-at", "fields"],     json: true  }],
        ["hierarchy", { dir: "frontend/hierarchy",   run: "phase", taytsh: false, args: ["--stop-at", "hierarchy"],  json: true  }],
        ["pycheck",   { dir: "frontend/pycheck",     run: "phase", taytsh: false, args: ["--stop-at", "pycheck"],    json: true  }],
        ["lowering",  { dir: "frontend/lowering",    run: "lowering" }],
    ]],
    ["middleend", [
        ["scope",     { dir: "middleend/scope",     run: "phase", taytsh: true, args: ["--stop-at", "scope"],     json: true }],
        ["returns",   { dir: "middleend/returns",   run: "phase", taytsh: true, args: ["--stop-at", "returns"],   json: true }],
        ["liveness",  { dir: "middleend/liveness",  run: "phase", taytsh: true, args: ["--stop-at", "liveness"],  json: true }],
        ["strings",   { dir: "middleend/strings",   run: "phase", taytsh: true, args: ["--stop-at", "strings"],   json: true }],
        ["hoisting",  { dir: "middleend/hoisting",  run: "phase", taytsh: true, args: ["--stop-at", "hoisting"],  json: true }],
        ["ownership", { dir: "middleend/ownership", run: "phase", taytsh: true, args: ["--stop-at", "ownership"], json: true }],
        ["callgraph", { dir: "middleend/callgraph", run: "phase", taytsh: true, args: ["--stop-at", "callgraph"], json: true }],
    ]],
    ["backend", [
        ["codegen",  { dir: "backend/codegen",  run: "codegen" }],
        ["emit",     { dir: "backend/emit",     run: "emit" }],
        ["app",      { dir: "backend/app",      run: "app" }],
        ["ordering", { dir: "backend/ordering", run: "ordering" }],
    ]],
    ["taytsh", [
        ["typarse", { dir: "taytsh/typarse", run: "phase", taytsh: true, args: ["--stop-at", "parse"], json: true }],
        ["tycheck", { dir: "taytsh/tycheck", run: "phase", taytsh: true, args: ["--stop-at", "check"], json: true }],
        ["ty_app",  { dir: "taytsh/app",     run: "ty_app" }],
    ]],
];

const EMITTER_LANGS = ["javascript", "perl", "python", "ruby"];
const RUNTIMES = {
    javascript: ["node"],
    perl: ["perl"],
    python: ["python3"],
    ruby: ["ruby"],
};

// ---------------------------------------------------------------------------
// Loading transpiled files into global scope
// ---------------------------------------------------------------------------

// Make module-scoped builtins available globally for vm.runInThisContext
globalThis.require = require;

function loadGlobal(filePath) {
    let code = nodeFs.readFileSync(filePath, "utf-8");
    // Convert block-scoped top-level (column 0) declarations to var so they
    // become properties of globalThis when evaluated via vm.runInThisContext.
    // Only modify column-0 declarations to preserve scoping inside functions.
    code = code.replace(/^class\s+(\w+)/gm, "var $1 = class $1");
    code = code.replace(/^const /gm, "var ");
    code = code.replace(/^let /gm, "var ");
    vm.runInThisContext(code, { filename: filePath });
}

// ---------------------------------------------------------------------------
// In-process execution
// ---------------------------------------------------------------------------

function runInprocess(argv, stdinData) {
    if (stdinData === undefined) stdinData = "";
    const origArgv = process.argv;
    const origStdoutWrite = process.stdout.write;
    const origStderrWrite = process.stderr.write;
    const origExit = process.exit;
    const origReadFileSync = nodeFs.readFileSync;
    let outBuf = "";
    let errBuf = "";
    let exitCode = 0;
    process.argv = ["node", "tongues", ...argv];
    process.stdout.write = (chunk) => { outBuf += String(chunk); return true; };
    process.stderr.write = (chunk) => { errBuf += String(chunk); return true; };
    const exitSentinel = Symbol("exit");
    process.exit = (code) => { throw { [exitSentinel]: true, code: code || 0 }; };
    nodeFs.readFileSync = (p, enc) => {
        if (p === "/dev/stdin" || p === 0) {
            if (enc) return typeof stdinData === "string" ? stdinData : stdinData.toString(enc);
            return typeof stdinData === "string" ? Buffer.from(stdinData, "utf-8") : stdinData;
        }
        return origReadFileSync.call(nodeFs, p, enc);
    };
    try {
        main();
    } catch (e) {
        if (e && e[exitSentinel]) {
            exitCode = e.code;
        } else {
            const msg = e instanceof Error ? e.message : String(e);
            errBuf += msg + "\n";
            exitCode = 1;
        }
    } finally {
        process.argv = origArgv;
        process.stdout.write = origStdoutWrite;
        process.stderr.write = origStderrWrite;
        process.exit = origExit;
        nodeFs.readFileSync = origReadFileSync;
    }
    return { stdout: outBuf, stderr: errBuf, exit: exitCode };
}

function runTranspiledPhase(source, cliArgs, isTaytsh, expectJson) {
    if (expectJson === undefined) expectJson = true;
    const suffix = isTaytsh ? ".ty" : ".py";
    const tmpFile = nodePath.join(os.tmpdir(), `test_${Date.now()}_${Math.random().toString(36).slice(2)}${suffix}`);
    nodeFs.writeFileSync(tmpFile, source);
    let argv;
    if (isTaytsh) {
        argv = ["taytsh", ...cliArgs, tmpFile];
    } else {
        argv = [...cliArgs, tmpFile];
    }
    const result = runInprocess(argv);
    try { nodeFs.unlinkSync(tmpFile); } catch {}
    const stderrText = result.stderr.trim();
    if (result.exit !== 0) {
        const errors = stderrText.split("\n").filter(s => s !== "");
        return { errors, warnings: [], data: null, reveals: [] };
    }
    const warnings = stderrText === "" ? [] : stderrText.split("\n").filter(s => s !== "");
    if (!expectJson) {
        return { errors: [], warnings, data: null, reveals: [] };
    }
    const stdoutText = result.stdout.trim();
    if (stdoutText === "") {
        return { errors: [], warnings, data: null, reveals: [] };
    }
    let data;
    try {
        data = json_parse(stdoutText);
    } catch {
        return { errors: ["Invalid JSON output: " + stdoutText.slice(0, 200)], warnings: [], data: null, reveals: [] };
    }
    return { errors: [], warnings, data, reveals: [] };
}

// ---------------------------------------------------------------------------
// File utilities
// ---------------------------------------------------------------------------

function globTests(dir, pattern) {
    if (!pattern) pattern = "*.tests";
    if (!nodeFs.existsSync(dir)) return [];
    const suffix = pattern.replace("*", "");
    return nodeFs.readdirSync(dir)
        .filter(f => f.endsWith(suffix))
        .sort()
        .map(f => nodePath.join(dir, f));
}

function globFiles(dir, pattern) {
    if (!nodeFs.existsSync(dir)) return [];
    const suffix = pattern.replace("*", "");
    return nodeFs.readdirSync(dir)
        .filter(f => f.endsWith(suffix))
        .sort()
        .map(f => nodePath.join(dir, f));
}

function subdirs(dir) {
    if (!nodeFs.existsSync(dir)) return [];
    return nodeFs.readdirSync(dir)
        .filter(d => {
            try { return nodeFs.statSync(nodePath.join(dir, d)).isDirectory(); } catch { return false; }
        })
        .sort();
}

function basename(filePath, ext) {
    return nodePath.basename(filePath, ext);
}

// ---------------------------------------------------------------------------
// Test runners
// ---------------------------------------------------------------------------

function runCliTests(testDir) {
    const results = [];
    for (const f of globTests(testDir)) {
        const stem = basename(f, ".tests");
        const tests = parse_cli_test_file(nodeFs.readFileSync(f, "utf-8"));
        for (const [name, spec] of tests) {
            const testId = `${stem}/${name}`;
            if (cli_needs_backend(spec.args, spec.assertions, EMITTER_LANGS)) {
                results.push(["skip", testId, null]);
                continue;
            }
            let stdinData;
            if (spec.stdin_hex !== "") {
                stdinData = Buffer.from(spec.stdin_hex, "hex");
            } else {
                stdinData = spec.stdin;
            }
            const result = runInprocess(spec.args, stdinData);
            const err = check_cli_assertions(result.exit, result.stdout, result.stderr, spec.assertions);
            results.push([err === "" ? "pass" : "fail", testId, err === "" ? null : err]);
        }
    }
    return results;
}

function runLinkerTests(testDir) {
    const results = [];
    for (const f of globTests(testDir)) {
        const stem = basename(f, ".tests");
        const tests = parse_linker_test_file(nodeFs.readFileSync(f, "utf-8"));
        for (const [name, spec] of tests) {
            const testId = `${stem}/${name}`;
            const parts = [];
            for (const lf of spec.files) {
                parts.push(lf.path, lf.source);
            }
            const stdinData = parts.join("\0");
            const args = spec.args;
            const targetIdx = args.indexOf("--target");
            if (targetIdx !== -1) {
                const target = args[targetIdx + 1];
                if (!EMITTER_LANGS.includes(target)) {
                    results.push(["skip", testId, null]);
                    continue;
                }
            }
            const result = runInprocess(args, stdinData);
            const err = check_cli_assertions(result.exit, result.stdout, result.stderr, spec.assertions);
            results.push([err === "" ? "pass" : "fail", testId, err === "" ? null : err]);
        }
    }
    return results;
}

function runPhaseTests(testDir, phaseName, cfg) {
    const results = [];
    for (const f of globTests(testDir, cfg.glob)) {
        const stem = basename(f, ".tests");
        const tests = parse_spec_file(nodeFs.readFileSync(f, "utf-8"));
        for (const entry of tests) {
            const testId = `${stem}/${entry.name}`;
            const lenient = ["parse", "pycheck", "typarse", "tycheck"].includes(phaseName);
            const phaseResult = runTranspiledPhase(entry.input, cfg.args, cfg.taytsh, cfg.json);
            let reveals = phaseResult.reveals;
            if (["pycheck", "tycheck"].includes(phaseName) && phaseResult.errors.length === 0 && phaseResult.data) {
                if (phaseResult.data instanceof JsonObject) {
                    try {
                        const revealsArr = json_get_items(json_get_field(phaseResult.data, "reveals"));
                        reveals = revealsArr.map(r => [
                            Math.trunc(json_get_number(json_get_field(r, "line"))),
                            json_get_string(json_get_field(r, "type")),
                        ]);
                    } catch {}
                }
            }
            let err;
            try {
                err = check_expected(entry.expected, phaseResult.errors, phaseResult.warnings,
                    phaseResult.data, reveals, phaseName, lenient);
            } catch (exc) {
                err = "harness crash: " + (exc instanceof Error ? exc.message : String(exc));
            }
            results.push([err === "" ? "pass" : "fail", testId, err === "" ? null : err]);
        }
    }
    return results;
}

function runLoweringTests(testDir) {
    const results = [];
    for (const f of globTests(testDir)) {
        const stem = basename(f, ".tests");
        const tests = parse_spec_file(nodeFs.readFileSync(f, "utf-8"));
        for (const entry of tests) {
            const testId = `${stem}/${entry.name}`;
            const tmpFile = nodePath.join(os.tmpdir(), `test_${Date.now()}_${Math.random().toString(36).slice(2)}.py`);
            nodeFs.writeFileSync(tmpFile, entry.input);
            const result = runInprocess(["--stop-at", "lowering-text", tmpFile]);
            try { nodeFs.unlinkSync(tmpFile); } catch {}
            if (entry.expected.startsWith("error:")) {
                const expectedMsg = entry.expected.slice(6).trim();
                if (result.exit === 0) {
                    results.push(["fail", testId, `Expected error containing '${expectedMsg}', got success`]);
                    continue;
                }
                const stderr = result.stderr.trim();
                const firstLine = stderr.split("\n")[0] || "";
                if (expectedMsg !== "" && !firstLine.toLowerCase().includes(expectedMsg.toLowerCase())) {
                    results.push(["fail", testId, `Expected error containing '${expectedMsg}', got: ${firstLine}`]);
                    continue;
                }
                results.push(["pass", testId, null]);
                continue;
            }
            if (result.exit !== 0) {
                const errMsg = result.stderr.trim().split("\n")[0] || "lowering failed";
                results.push(["fail", testId, `Lowering error: ${errMsg}`]);
                continue;
            }
            if (!contains_normalized(result.stdout, entry.expected)) {
                results.push(["fail", testId, `Expected not found in output:\n--- expected ---\n${entry.expected}\n--- got ---\n${result.stdout}`]);
                continue;
            }
            results.push(["pass", testId, null]);
        }
    }
    return results;
}

function runCodegenTests(testDir) {
    const results = [];
    const baseDir = nodePath.join(testDir, "base");
    if (!nodeFs.existsSync(baseDir)) return results;
    const langDirs = subdirs(testDir)
        .filter(d => d !== "base" && EMITTER_LANGS.includes(d));
    for (const lang of langDirs) {
        const langDir = nodePath.join(testDir, lang);
        for (const baseFile of globTests(baseDir)) {
            const baseName = nodePath.basename(baseFile);
            const stem = basename(baseFile, ".tests");
            const langFile = nodePath.join(langDir, baseName);
            const baseTests = parse_simple_tests(nodeFs.readFileSync(baseFile, "utf-8"));
            if (baseTests.length === 0) continue;
            if (!nodeFs.existsSync(langFile)) {
                for (const entry of baseTests) {
                    results.push(["fail", `${stem}/${entry.name}[${lang}]`, `${lang}/${baseName} missing`]);
                }
                continue;
            }
            const langTests = parse_simple_tests(nodeFs.readFileSync(langFile, "utf-8"));
            const baseNames = baseTests.map(e => e.name);
            const langNames = langTests.map(e => e.name);
            if (baseNames.join("\0") !== langNames.join("\0")) {
                for (const entry of baseTests) {
                    results.push(["fail", `${stem}/${entry.name}[${lang}]`, "base/lang name mismatch"]);
                }
                continue;
            }
            const langByName = new Map(langTests.map(e => [e.name, e.content]));
            for (const entry of baseTests) {
                const testId = `${stem}/${entry.name}[${lang}]`;
                const expected = langByName.get(entry.name);
                const tmpFile = nodePath.join(os.tmpdir(), `test_${Date.now()}_${Math.random().toString(36).slice(2)}.ty`);
                nodeFs.writeFileSync(tmpFile, entry.content);
                const result = runInprocess(["taytsh", "--emit", lang, tmpFile]);
                try { nodeFs.unlinkSync(tmpFile); } catch {}
                if (result.exit !== 0) {
                    const stderr = result.stderr.trim().split("\n")[0] || "transpile failed";
                    results.push(["fail", testId, `Transpile error: ${stderr}`]);
                    continue;
                }
                if (!contains_normalized(result.stdout, expected)) {
                    results.push(["fail", testId, `Expected not found in output:\n--- expected ---\n${expected}\n--- got ---\n${result.stdout}`]);
                    continue;
                }
                results.push(["pass", testId, null]);
            }
        }
    }
    return results;
}

function runEmitTests(testDir) {
    const results = [];
    const baseDir = nodePath.join(testDir, "base");
    if (!nodeFs.existsSync(baseDir)) return results;
    const langDirs = subdirs(testDir)
        .filter(d => d !== "base" && EMITTER_LANGS.includes(d));
    for (const lang of langDirs) {
        const langDir = nodePath.join(testDir, lang);
        for (const baseFile of globTests(baseDir)) {
            const baseName = nodePath.basename(baseFile);
            const stem = basename(baseFile, ".tests");
            const langFile = nodePath.join(langDir, baseName);
            const baseTests = parse_simple_tests(nodeFs.readFileSync(baseFile, "utf-8"));
            if (baseTests.length === 0) continue;
            if (!nodeFs.existsSync(langFile)) continue;
            const langTests = parse_simple_tests(nodeFs.readFileSync(langFile, "utf-8"));
            const langByName = new Map(langTests.map(e => [e.name, e.content]));
            for (const entry of baseTests) {
                if (!langByName.has(entry.name)) continue;
                const testId = `${stem}/${entry.name}[${lang}]`;
                const expected = langByName.get(entry.name);
                const tmpFile = nodePath.join(os.tmpdir(), `test_${Date.now()}_${Math.random().toString(36).slice(2)}.py`);
                nodeFs.writeFileSync(tmpFile, entry.content);
                const result = runInprocess(["--target", lang, tmpFile]);
                try { nodeFs.unlinkSync(tmpFile); } catch {}
                if (result.exit !== 0) {
                    const stderr = result.stderr.trim().split("\n")[0] || "emit failed";
                    results.push(["fail", testId, `Emit error: ${stderr}`]);
                    continue;
                }
                if (!contains_normalized(result.stdout, expected)) {
                    results.push(["fail", testId, `Expected not found in output:\n--- expected ---\n${expected}\n--- got ---\n${result.stdout}`]);
                    continue;
                }
                results.push(["pass", testId, null]);
            }
        }
    }
    return results;
}

function runtimeAvailable(lang) {
    const cmd = RUNTIMES[lang];
    if (!cmd) return false;
    try {
        const r = spawnSync("which", [cmd[0]], { encoding: "utf-8", timeout: 5000 });
        return r.status === 0;
    } catch { return false; }
}

function runAppTests(testDir) {
    const results = [];
    const available = Object.keys(RUNTIMES).filter(runtimeAvailable).sort();
    for (const testFile of globFiles(testDir, "apptest_*.py").concat(globFiles(testDir, "apptest_*.py")).filter((v, i, a) => a.indexOf(v) === i)) {
        // Re-glob properly
    }
    const appFiles = nodeFs.existsSync(testDir)
        ? nodeFs.readdirSync(testDir).filter(f => f.startsWith("apptest_") && f.endsWith(".py")).sort().map(f => nodePath.join(testDir, f))
        : [];
    for (const testFile of appFiles) {
        const stem = basename(testFile, ".py");
        const source = nodeFs.readFileSync(testFile, "utf-8");
        let libNames = find_lib_imports(source);
        // Transitively resolve cross-lib imports
        const seen = new Set(libNames);
        const queue = [...libNames];
        while (queue.length > 0) {
            const name = queue.shift();
            const libPath = nodePath.join(LIB_DIR, `${name}.py`);
            if (!nodeFs.existsSync(libPath)) continue;
            const deps = find_lib_imports(nodeFs.readFileSync(libPath, "utf-8"));
            for (const dep of deps) {
                if (!seen.has(dep)) {
                    seen.add(dep);
                    libNames.push(dep);
                    queue.push(dep);
                }
            }
        }
        for (const target of available) {
            const testId = `${stem}[${target}]`;
            let result;
            if (libNames.length === 0) {
                const tmpFile = nodePath.join(os.tmpdir(), `test_${Date.now()}_${Math.random().toString(36).slice(2)}.py`);
                nodeFs.writeFileSync(tmpFile, source);
                result = runInprocess(["--target", target, tmpFile]);
                try { nodeFs.unlinkSync(tmpFile); } catch {}
            } else {
                const libSources = libNames.map(name => {
                    const libPath = nodePath.join(LIB_DIR, `${name}.py`);
                    return [`lib/${name}.py`, nodeFs.readFileSync(libPath, "utf-8")];
                });
                const stdinData = build_project_input("apptest.py", source, libSources);
                result = runInprocess(["--project", "--target", target], stdinData);
            }
            if (result.exit !== 0) {
                const stderr = result.stderr.trim().split("\n")[0] || "transpile failed";
                results.push(["fail", testId, `Transpile error (${target}): ${stderr}`]);
                continue;
            }
            const transpiledCode = result.stdout;
            const runtime = RUNTIMES[target];
            const run = spawnSync(runtime[0], runtime.slice(1), {
                input: transpiledCode,
                encoding: "utf-8",
                timeout: 10000,
            });
            const output = (run.stdout || "") + (run.stderr || "");
            if (run.status !== 0) {
                results.push(["fail", testId, `App test failed with exit ${run.status}\n${output}`]);
                continue;
            }
            results.push(["pass", testId, null]);
        }
    }
    return results;
}

function runTyAppTests(testDir) {
    const results = [];
    for (const testFile of globFiles(testDir, "*.ty")) {
        const stem = basename(testFile, ".ty");
        const testId = stem;
        const result = runInprocess(["taytsh", testFile]);
        if (result.exit !== 0) {
            const output = (result.stdout + result.stderr).trim();
            results.push(["fail", testId, `Exit code ${result.exit}:\n${output}`]);
            continue;
        }
        results.push(["pass", testId, null]);
    }
    return results;
}

function runOrderingTests(testDir) {
    const results = [];
    const available = Object.keys(RUNTIMES).filter(runtimeAvailable).sort();
    for (const testFile of globFiles(testDir, "*.ty")) {
        const stem = basename(testFile, ".ty");
        for (const target of available) {
            const testId = `${stem}[${target}]`;
            const result = runInprocess(["taytsh", "--emit", target, testFile]);
            if (result.exit !== 0) {
                const stderr = result.stderr.trim().split("\n")[0] || "transpile failed";
                results.push(["fail", testId, `Transpile error (${target}): ${stderr}`]);
                continue;
            }
            const transpiledCode = result.stdout;
            const runtime = RUNTIMES[target];
            const run = spawnSync(runtime[0], runtime.slice(1), {
                input: transpiledCode,
                encoding: "utf-8",
                timeout: 10000,
            });
            const output = (run.stdout || "") + (run.stderr || "");
            if (run.status !== 0) {
                results.push(["fail", testId, `Ordering test failed with exit ${run.status}\n${output}`]);
                continue;
            }
            results.push(["pass", testId, null]);
        }
    }
    return results;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

if (process.argv.length < 3) {
    process.stderr.write("Usage: node test-transpiled.js <transpiled.js>\n");
    process.exit(1);
}

const transpiledPath = nodePath.resolve(TONGUES_DIR, process.argv[2]);
if (!nodeFs.existsSync(transpiledPath)) {
    process.stderr.write(`Transpiled file not found: ${transpiledPath}\n`);
    process.exit(1);
}

console.log(`Loading transpiled binary: ${transpiledPath}`);
const t0 = Date.now();
try {
    loadGlobal(transpiledPath);
} catch (e) {
    process.stderr.write("Failed to load transpiled binary:\n");
    process.stderr.write(String(e).split("\n").slice(0, 5).join("\n") + "\n");
    process.exit(1);
}
const t1 = Date.now();
console.log(`Loaded in ${((t1 - t0) / 1000).toFixed(1)}s`);

const harnessPath = nodePath.join(TONGUES_DIR, ".out", "test_harness.js");
if (!nodeFs.existsSync(harnessPath)) {
    process.stderr.write(`Transpiled harness not found: ${harnessPath}\n`);
    process.exit(1);
}
try {
    loadGlobal(harnessPath);
} catch (e) {
    process.stderr.write(`Failed to load transpiled harness:\n${e}\n`);
    process.exit(1);
}
console.log();

let totalPass = 0;
let totalFail = 0;
let totalSkip = 0;
const failures = [];

for (const [sectionName, phases] of TESTS) {
    for (const [phaseName, cfg] of phases) {
        const testDir = nodePath.join(TESTS_DIR, cfg.dir);
        if (!nodeFs.existsSync(testDir)) continue;
        let phaseResults;
        switch (cfg.run) {
            case "cli":      phaseResults = runCliTests(testDir); break;
            case "linker":   phaseResults = runLinkerTests(testDir); break;
            case "phase":    phaseResults = runPhaseTests(testDir, phaseName, cfg); break;
            case "lowering": phaseResults = runLoweringTests(testDir); break;
            case "codegen":  phaseResults = runCodegenTests(testDir); break;
            case "emit":     phaseResults = runEmitTests(testDir); break;
            case "app":      phaseResults = runAppTests(testDir); break;
            case "ty_app":   phaseResults = runTyAppTests(testDir); break;
            case "ordering": phaseResults = runOrderingTests(testDir); break;
            default:         phaseResults = []; break;
        }
        let pass = 0, failCount = 0, skip = 0;
        for (const [s] of phaseResults) {
            if (s === "pass") pass++;
            else if (s === "fail") failCount++;
            else if (s === "skip") skip++;
        }
        totalPass += pass;
        totalFail += failCount;
        totalSkip += skip;
        const status = failCount > 0 ? "FAIL" : "ok";
        let counts = `${pass} passed`;
        if (failCount > 0) counts += `, ${failCount} failed`;
        if (skip > 0) counts += `, ${skip} skipped`;
        console.log(`${phaseName}: ${status} (${counts})`);
        for (const [s, tid, err] of phaseResults) {
            if (s === "fail") {
                failures.push([phaseName, tid, err]);
                console.log(`  FAIL ${tid}`);
            }
        }
    }
}

console.log();
if (failures.length > 0) {
    console.log("=".repeat(60));
    console.log("FAILURES");
    console.log("=".repeat(60));
    for (const [phase, tid, err] of failures) {
        console.log();
        console.log(`${phase} :: ${tid}`);
        console.log(err);
    }
    console.log();
}

console.log("=".repeat(60));
const total = totalPass + totalFail + totalSkip;
console.log(`${total} tests: ${totalPass} passed, ${totalFail} failed, ${totalSkip} skipped`);
console.log("=".repeat(60));

process.exit(totalFail > 0 ? 1 : 0);
