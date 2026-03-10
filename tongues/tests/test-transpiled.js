#!/usr/bin/env node
"use strict";

// Native Node.js test harness for transpiled Tongues binaries.
// Loads the transpiled file once, then runs all .tests cases in-process.
// Shared parsing/assertion logic is transpiled from tests/shared/test_harness.py.
// Supports parallel execution with -n <num> or -n auto (like pytest-xdist).

const nodeFs = require("fs");
const nodePath = require("path");
const vm = require("vm");
const os = require("os");
const { spawnSync, fork } = require("child_process");

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

const EMITTER_LANGS = ["javascript"];
const RUNTIMES = {
    javascript: ["node"],
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
// VM mode: parse + compile .ty once, invoke per test
// ---------------------------------------------------------------------------

let _vmCompiled = null;

function loadVmModule(tyPath) {
    const source = nodeFs.readFileSync(tyPath, "utf-8");
    const module = taytsh_taytsh_parse(source);
    _vmCompiled = vm_prepare(module);
    console.log("VM module compiled");
}

function runVmInprocess(argv, stdinData) {
    if (stdinData === undefined) stdinData = "";
    const stdinBuf = typeof stdinData === "string"
        ? Buffer.from(stdinData, "utf-8")
        : (Buffer.isBuffer(stdinData) ? stdinData : Buffer.from(stdinData));
    const instance = new VM(_vmCompiled);
    const result = instance.invoke(stdinBuf, ["tongues", ...argv]);
    return {
        stdout: typeof result.stdout === "string" ? result.stdout : result.stdout.toString("utf-8"),
        stderr: typeof result.stderr === "string" ? result.stderr : result.stderr.toString("utf-8"),
        exit: result.exit_code,
    };
}

// ---------------------------------------------------------------------------
// In-process execution
// ---------------------------------------------------------------------------

let runInprocess = function runInprocess(argv, stdinData) {
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
};

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
            let annotations = {};
            if (["pycheck", "tycheck"].includes(phaseName) && phaseResult.errors.length === 0 && phaseResult.data) {
                if (phaseResult.data instanceof JsonObject) {
                    try {
                        const revealsArr = json_get_items(json_get_field(phaseResult.data, "reveals"));
                        reveals = revealsArr.map(r => [
                            Math.trunc(json_get_number(json_get_field(r, "line"))),
                            json_get_string(json_get_field(r, "type")),
                        ]);
                    } catch {}
                    try {
                        const annsObj = json_get_field(phaseResult.data, "annotations");
                        if (annsObj instanceof JsonObject) {
                            for (const [lineStr, lineAnns] of annsObj.entries) {
                                const lineDict = {};
                                if (lineAnns instanceof JsonObject) {
                                    for (const [k, v] of lineAnns.entries) {
                                        if (v instanceof JsonString) lineDict[k] = v.value;
                                    }
                                }
                                annotations[parseInt(lineStr)] = lineDict;
                            }
                        }
                    } catch {}
                }
            }
            let err;
            try {
                err = check_expected(entry.expected, phaseResult.errors, phaseResult.warnings,
                    phaseResult.data, reveals, annotations, phaseName, lenient);
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
// Parallel execution support
// ---------------------------------------------------------------------------

// Helper to extract plain JSON-serializable data from spec/entry objects
function serializeCliSpec(spec) {
    return {
        args: Array.from(spec.args),
        stdin: spec.stdin,
        stdin_hex: spec.stdin_hex,
        assertions: Array.from(spec.assertions).map(a => ({
            type: a.type,
            value: a.value,
        })),
    };
}
function serializeLinkerSpec(spec) {
    return {
        args: Array.from(spec.args),
        files: Array.from(spec.files).map(f => ({ path: f.path, source: f.source })),
        assertions: Array.from(spec.assertions).map(a => ({
            type: a.type,
            value: a.value,
        })),
    };
}
function serializeEntry(entry) {
    return { name: entry.name, input: entry.input, expected: entry.expected };
}
function serializeSimpleEntry(entry) {
    return { name: entry.name, content: entry.content };
}

function collectTests() {
    const collected = [];
    for (const [sectionName, phases] of TESTS) {
        for (const [phaseName, cfg] of phases) {
            const testDir = nodePath.join(TESTS_DIR, cfg.dir);
            if (!nodeFs.existsSync(testDir)) continue;
            // Serialize cfg to plain object
            const plainCfg = { dir: cfg.dir, run: cfg.run, taytsh: cfg.taytsh, args: cfg.args ? Array.from(cfg.args) : null, json: cfg.json, glob: cfg.glob };
            switch (cfg.run) {
                case "cli":
                    for (const f of globTests(testDir)) {
                        const stem = basename(f, ".tests");
                        const content = nodeFs.readFileSync(f, "utf-8");
                        const tests = parse_cli_test_file(content);
                        for (const [name, spec] of tests) {
                            const testId = `${stem}/${name}`;
                            if (cli_needs_backend(spec.args, spec.assertions, EMITTER_LANGS)) {
                                collected.push([phaseName, testId, "skip", null]);
                            } else {
                                collected.push([phaseName, testId, "cli", serializeCliSpec(spec)]);
                            }
                        }
                    }
                    break;
                case "linker":
                    for (const f of globTests(testDir)) {
                        const stem = basename(f, ".tests");
                        const content = nodeFs.readFileSync(f, "utf-8");
                        const tests = parse_linker_test_file(content);
                        for (const [name, spec] of tests) {
                            const testId = `${stem}/${name}`;
                            const args = spec.args;
                            const targetIdx = args.indexOf("--target");
                            if (targetIdx !== -1 && !EMITTER_LANGS.includes(args[targetIdx + 1])) {
                                collected.push([phaseName, testId, "skip", null]);
                            } else {
                                collected.push([phaseName, testId, "linker", serializeLinkerSpec(spec)]);
                            }
                        }
                    }
                    break;
                case "phase":
                    for (const f of globTests(testDir, cfg.glob)) {
                        const stem = basename(f, ".tests");
                        const content = nodeFs.readFileSync(f, "utf-8");
                        const tests = parse_spec_file(content);
                        for (const entry of tests) {
                            const testId = `${stem}/${entry.name}`;
                            collected.push([phaseName, testId, "phase", { entry: serializeEntry(entry), cfg: plainCfg }]);
                        }
                    }
                    break;
                case "lowering":
                    for (const f of globTests(testDir)) {
                        const stem = basename(f, ".tests");
                        const content = nodeFs.readFileSync(f, "utf-8");
                        const tests = parse_spec_file(content);
                        for (const entry of tests) {
                            const testId = `${stem}/${entry.name}`;
                            collected.push([phaseName, testId, "lowering", serializeEntry(entry)]);
                        }
                    }
                    break;
                case "codegen": {
                    const baseDir = nodePath.join(testDir, "base");
                    if (!nodeFs.existsSync(baseDir)) break;
                    const langDirs = subdirs(testDir).filter(d => d !== "base" && EMITTER_LANGS.includes(d));
                    for (const lang of langDirs) {
                        const langDir = nodePath.join(testDir, lang);
                        for (const baseFile of globTests(baseDir)) {
                            const baseName = nodePath.basename(baseFile);
                            const stem = basename(baseFile, ".tests");
                            const langFile = nodePath.join(langDir, baseName);
                            const baseContent = nodeFs.readFileSync(baseFile, "utf-8");
                            const baseTests = parse_simple_tests(baseContent);
                            if (baseTests.length === 0) continue;
                            if (!nodeFs.existsSync(langFile)) {
                                for (const entry of baseTests) {
                                    collected.push([phaseName, `${stem}/${entry.name}[${lang}]`, "prefail", `${lang}/${baseName} missing`]);
                                }
                                continue;
                            }
                            const langContent = nodeFs.readFileSync(langFile, "utf-8");
                            const langTests = parse_simple_tests(langContent);
                            const baseNames = baseTests.map(e => e.name);
                            const langNames = langTests.map(e => e.name);
                            if (baseNames.join("\0") !== langNames.join("\0")) {
                                for (const entry of baseTests) {
                                    collected.push([phaseName, `${stem}/${entry.name}[${lang}]`, "prefail", "base/lang name mismatch"]);
                                }
                                continue;
                            }
                            const langByName = new Map(langTests.map(e => [e.name, e.content]));
                            for (const entry of baseTests) {
                                const testId = `${stem}/${entry.name}[${lang}]`;
                                // Already serializable - content, expected, lang are all strings
                                collected.push([phaseName, testId, "codegen", { content: String(entry.content), expected: String(langByName.get(entry.name)), lang }]);
                            }
                        }
                    }
                    break;
                }
                case "emit": {
                    const baseDir = nodePath.join(testDir, "base");
                    if (!nodeFs.existsSync(baseDir)) break;
                    const langDirs = subdirs(testDir).filter(d => d !== "base" && EMITTER_LANGS.includes(d));
                    for (const lang of langDirs) {
                        const langDir = nodePath.join(testDir, lang);
                        for (const baseFile of globTests(baseDir)) {
                            const baseName = nodePath.basename(baseFile);
                            const stem = basename(baseFile, ".tests");
                            const langFile = nodePath.join(langDir, baseName);
                            const baseContent = nodeFs.readFileSync(baseFile, "utf-8");
                            const baseTests = parse_simple_tests(baseContent);
                            if (baseTests.length === 0) continue;
                            if (!nodeFs.existsSync(langFile)) continue;
                            const langContent = nodeFs.readFileSync(langFile, "utf-8");
                            const langTests = parse_simple_tests(langContent);
                            const langByName = new Map(langTests.map(e => [e.name, e.content]));
                            for (const entry of baseTests) {
                                if (!langByName.has(entry.name)) continue;
                                const testId = `${stem}/${entry.name}[${lang}]`;
                                // Already serializable
                                collected.push([phaseName, testId, "emit", { content: String(entry.content), expected: String(langByName.get(entry.name)), lang }]);
                            }
                        }
                    }
                    break;
                }
                case "app": {
                    const available = Object.keys(RUNTIMES).filter(runtimeAvailable).sort();
                    const appFiles = nodeFs.existsSync(testDir)
                        ? nodeFs.readdirSync(testDir).filter(f => f.startsWith("apptest_") && f.endsWith(".py")).sort().map(f => nodePath.join(testDir, f))
                        : [];
                    for (const testFile of appFiles) {
                        const stem = basename(testFile, ".py");
                        const source = nodeFs.readFileSync(testFile, "utf-8");
                        let libNames = find_lib_imports(source);
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
                        // libParts is array of [string, string] - already serializable
                        const libParts = libNames.map(name => {
                            const libPath = nodePath.join(LIB_DIR, `${name}.py`);
                            return [`lib/${name}.py`, nodeFs.readFileSync(libPath, "utf-8")];
                        });
                        for (const target of available) {
                            const testId = `${stem}[${target}]`;
                            collected.push([phaseName, testId, "app", { source: String(source), libParts, target }]);
                        }
                    }
                    break;
                }
                case "ty_app":
                    for (const testFile of globFiles(testDir, "*.ty")) {
                        const stem = basename(testFile, ".ty");
                        collected.push([phaseName, stem, "ty_app", testFile]);
                    }
                    break;
                case "ordering": {
                    const available = Object.keys(RUNTIMES).filter(runtimeAvailable).sort();
                    for (const testFile of globFiles(testDir, "*.ty")) {
                        const stem = basename(testFile, ".ty");
                        for (const target of available) {
                            const testId = `${stem}[${target}]`;
                            collected.push([phaseName, testId, "ordering", { testFile, target }]);
                        }
                    }
                    break;
                }
            }
        }
    }
    return collected;
}

function runSingleTest(phaseName, testId, testType, testData) {
    switch (testType) {
        case "skip":
            return [phaseName, testId, "skip", null];
        case "prefail":
            return [phaseName, testId, "fail", testData];
        case "cli": {
            const spec = testData;
            let stdinData;
            if (spec.stdin_hex !== "") {
                stdinData = Buffer.from(spec.stdin_hex, "hex");
            } else {
                stdinData = spec.stdin;
            }
            const result = runInprocess(spec.args, stdinData);
            const err = check_cli_assertions(result.exit, result.stdout, result.stderr, spec.assertions);
            return [phaseName, testId, err === "" ? "pass" : "fail", err === "" ? null : err];
        }
        case "linker": {
            const spec = testData;
            const parts = [];
            for (const lf of spec.files) {
                parts.push(lf.path, lf.source);
            }
            const stdinData = parts.join("\0");
            const result = runInprocess(spec.args, stdinData);
            const err = check_cli_assertions(result.exit, result.stdout, result.stderr, spec.assertions);
            return [phaseName, testId, err === "" ? "pass" : "fail", err === "" ? null : err];
        }
        case "phase": {
            const { entry, cfg } = testData;
            const lenient = ["parse", "pycheck", "typarse", "tycheck"].includes(phaseName);
            const phaseResult = runTranspiledPhase(entry.input, cfg.args, cfg.taytsh, cfg.json);
            let reveals = phaseResult.reveals;
            let annotations = {};
            if (["pycheck", "tycheck"].includes(phaseName) && phaseResult.errors.length === 0 && phaseResult.data) {
                if (phaseResult.data instanceof JsonObject) {
                    try {
                        const revealsArr = json_get_items(json_get_field(phaseResult.data, "reveals"));
                        reveals = revealsArr.map(r => [
                            Math.trunc(json_get_number(json_get_field(r, "line"))),
                            json_get_string(json_get_field(r, "type")),
                        ]);
                    } catch {}
                    try {
                        const annsObj = json_get_field(phaseResult.data, "annotations");
                        if (annsObj instanceof JsonObject) {
                            for (const [lineStr, lineAnns] of annsObj.entries) {
                                const lineDict = {};
                                if (lineAnns instanceof JsonObject) {
                                    for (const [k, v] of lineAnns.entries) {
                                        if (v instanceof JsonString) lineDict[k] = v.value;
                                    }
                                }
                                annotations[parseInt(lineStr)] = lineDict;
                            }
                        }
                    } catch {}
                }
            }
            let err;
            try {
                err = check_expected(entry.expected, phaseResult.errors, phaseResult.warnings,
                    phaseResult.data, reveals, annotations, phaseName, lenient);
            } catch (exc) {
                err = "harness crash: " + (exc instanceof Error ? exc.message : String(exc));
            }
            return [phaseName, testId, err === "" ? "pass" : "fail", err === "" ? null : err];
        }
        case "lowering": {
            const entry = testData;
            const tmpFile = nodePath.join(os.tmpdir(), `test_${Date.now()}_${Math.random().toString(36).slice(2)}.py`);
            nodeFs.writeFileSync(tmpFile, entry.input);
            const result = runInprocess(["--stop-at", "lowering-text", tmpFile]);
            try { nodeFs.unlinkSync(tmpFile); } catch {}
            if (entry.expected.startsWith("error:")) {
                const expectedMsg = entry.expected.slice(6).trim();
                if (result.exit === 0) {
                    return [phaseName, testId, "fail", `Expected error containing '${expectedMsg}', got success`];
                }
                const stderr = result.stderr.trim();
                const firstLine = stderr.split("\n")[0] || "";
                if (expectedMsg !== "" && !firstLine.toLowerCase().includes(expectedMsg.toLowerCase())) {
                    return [phaseName, testId, "fail", `Expected error containing '${expectedMsg}', got: ${firstLine}`];
                }
                return [phaseName, testId, "pass", null];
            }
            if (result.exit !== 0) {
                const errMsg = result.stderr.trim().split("\n")[0] || "lowering failed";
                return [phaseName, testId, "fail", `Lowering error: ${errMsg}`];
            }
            if (!contains_normalized(result.stdout, entry.expected)) {
                return [phaseName, testId, "fail", `Expected not found in output:\n--- expected ---\n${entry.expected}\n--- got ---\n${result.stdout}`];
            }
            return [phaseName, testId, "pass", null];
        }
        case "codegen": {
            const { content, expected, lang } = testData;
            const tmpFile = nodePath.join(os.tmpdir(), `test_${Date.now()}_${Math.random().toString(36).slice(2)}.ty`);
            nodeFs.writeFileSync(tmpFile, content);
            const result = runInprocess(["taytsh", "--emit", lang, tmpFile]);
            try { nodeFs.unlinkSync(tmpFile); } catch {}
            if (result.exit !== 0) {
                const stderr = result.stderr.trim().split("\n")[0] || "transpile failed";
                return [phaseName, testId, "fail", `Transpile error: ${stderr}`];
            }
            if (!contains_normalized(result.stdout, expected)) {
                return [phaseName, testId, "fail", `Expected not found in output:\n--- expected ---\n${expected}\n--- got ---\n${result.stdout}`];
            }
            return [phaseName, testId, "pass", null];
        }
        case "emit": {
            const { content, expected, lang } = testData;
            const tmpFile = nodePath.join(os.tmpdir(), `test_${Date.now()}_${Math.random().toString(36).slice(2)}.py`);
            nodeFs.writeFileSync(tmpFile, content);
            const result = runInprocess(["--target", lang, tmpFile]);
            try { nodeFs.unlinkSync(tmpFile); } catch {}
            if (result.exit !== 0) {
                const stderr = result.stderr.trim().split("\n")[0] || "emit failed";
                return [phaseName, testId, "fail", `Emit error: ${stderr}`];
            }
            if (!contains_normalized(result.stdout, expected)) {
                return [phaseName, testId, "fail", `Expected not found in output:\n--- expected ---\n${expected}\n--- got ---\n${result.stdout}`];
            }
            return [phaseName, testId, "pass", null];
        }
        case "app": {
            const { source, libParts, target } = testData;
            let result;
            if (libParts.length === 0) {
                const tmpFile = nodePath.join(os.tmpdir(), `test_${Date.now()}_${Math.random().toString(36).slice(2)}.py`);
                nodeFs.writeFileSync(tmpFile, source);
                result = runInprocess(["--target", target, tmpFile]);
                try { nodeFs.unlinkSync(tmpFile); } catch {}
            } else {
                const stdinData = build_project_input("apptest.py", source, libParts);
                result = runInprocess(["--project", "--target", target], stdinData);
            }
            if (result.exit !== 0) {
                const stderr = result.stderr.trim().split("\n")[0] || "transpile failed";
                return [phaseName, testId, "fail", `Transpile error (${target}): ${stderr}`];
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
                return [phaseName, testId, "fail", `App test failed with exit ${run.status}\n${output}`];
            }
            return [phaseName, testId, "pass", null];
        }
        case "ty_app": {
            const testFile = testData;
            const result = runInprocess(["taytsh", testFile]);
            if (result.exit !== 0) {
                const output = (result.stdout + result.stderr).trim();
                return [phaseName, testId, "fail", `Exit code ${result.exit}:\n${output}`];
            }
            return [phaseName, testId, "pass", null];
        }
        case "ordering": {
            const { testFile, target } = testData;
            const result = runInprocess(["taytsh", "--emit", target, testFile]);
            if (result.exit !== 0) {
                const stderr = result.stderr.trim().split("\n")[0] || "transpile failed";
                return [phaseName, testId, "fail", `Transpile error (${target}): ${stderr}`];
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
                return [phaseName, testId, "fail", `Ordering test failed with exit ${run.status}\n${output}`];
            }
            return [phaseName, testId, "pass", null];
        }
        default:
            return [phaseName, testId, "fail", `Unknown test type: ${testType}`];
    }
}

// ---------------------------------------------------------------------------
// Worker mode (when spawned via fork)
// ---------------------------------------------------------------------------

if (process.env._TONGUES_WORKER === "1") {
    // Worker process - initialize and process tasks via IPC
    const transpiledPath = process.env._TONGUES_TRANSPILED;
    const harnessPath = process.env._TONGUES_HARNESS;
    const viaVmPath = process.env._TONGUES_VM || null;
    loadGlobal(transpiledPath);
    loadGlobal(harnessPath);
    if (viaVmPath) {
        loadVmModule(viaVmPath);
        runInprocess = runVmInprocess;
    }
    process.send({ type: "ready" });
    process.on("message", (msg) => {
        if (msg.type === "task") {
            const { id, phaseName, testId, testType, testData } = msg;
            try {
                const result = runSingleTest(phaseName, testId, testType, testData);
                process.send({ type: "result", id, result });
            } catch (err) {
                process.send({ type: "result", id, result: [phaseName, testId, "fail", `Worker error: ${err.message || err}`] });
            }
        } else if (msg.type === "exit") {
            process.exit(0);
        }
    });
} else {

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

if (process.argv.length < 3) {
    process.stderr.write("Usage: node test-transpiled.js <transpiled.js> [--via-vm <tongues.ty>] [--target <name>] [-n <num|auto>]\n");
    process.exit(1);
}

const viaVmIdx = process.argv.indexOf("--via-vm");
let viaVmPath = null;
if (viaVmIdx !== -1) {
    if (viaVmIdx + 1 >= process.argv.length) {
        process.stderr.write("--via-vm requires a path to a .ty file\n");
        process.exit(1);
    }
    viaVmPath = nodePath.resolve(TONGUES_DIR, process.argv[viaVmIdx + 1]);
}

const targetIdx = process.argv.indexOf("--target");
let targetName = null;
if (targetIdx !== -1) {
    if (targetIdx + 1 >= process.argv.length) {
        process.stderr.write("--target requires a name\n");
        process.exit(1);
    }
    targetName = process.argv[targetIdx + 1];
}

// Parse -n argument for parallel workers (default: auto = CPU count)
const nIdx = process.argv.indexOf("-n");
let numWorkers = os.cpus().length;
if (nIdx !== -1) {
    if (nIdx + 1 >= process.argv.length) {
        process.stderr.write("-n requires a number or 'auto'\n");
        process.exit(1);
    }
    const nVal = process.argv[nIdx + 1];
    numWorkers = nVal === "auto" ? os.cpus().length : parseInt(nVal, 10);
    if (isNaN(numWorkers) || numWorkers < 1) {
        process.stderr.write("-n requires a positive number or 'auto'\n");
        process.exit(1);
    }
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

if (viaVmPath) {
    if (!nodeFs.existsSync(viaVmPath)) {
        process.stderr.write(`VM module not found: ${viaVmPath}\n`);
        process.exit(1);
    }
    console.log(`Loading VM module: ${viaVmPath}`);
    const vmT0 = Date.now();
    loadVmModule(viaVmPath);
    console.log(`VM compiled in ${((Date.now() - vmT0) / 1000).toFixed(1)}s`);
    runInprocess = runVmInprocess;
}

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

// Collect all tests
const collected = collectTests();
console.log(`Collected ${collected.length} tests`);
console.log(`Running with ${numWorkers} workers`);
console.log();

const vmTag = viaVmPath ? "[vm] " : "";
let totalPass = 0;
let totalFail = 0;
let totalSkip = 0;
const failures = [];

async function runParallel() {
    const tStart = Date.now();
    // Fork worker processes
    const workers = [];
    const workerEnv = {
        _TONGUES_WORKER: "1",
        _TONGUES_TRANSPILED: transpiledPath,
        _TONGUES_HARNESS: harnessPath,
        _TONGUES_VM: viaVmPath || "",
    };
    for (let i = 0; i < numWorkers; i++) {
        const worker = fork(__filename, [], { env: { ...process.env, ...workerEnv }, silent: true });
        workers.push({ proc: worker, busy: false, ready: false, pending: null });
    }
    // Wait for all workers to be ready
    await Promise.all(workers.map(w => new Promise(resolve => {
        w.proc.once("message", (msg) => {
            if (msg.type === "ready") { w.ready = true; resolve(); }
        });
    })));
    // Task queue and results
    const taskQueue = collected.map((t, i) => ({ id: i, task: t }));
    const results = new Array(collected.length);
    let completed = 0;
    let nextTask = 0;
    // Dispatch tasks to workers
    function dispatch(worker) {
        if (nextTask >= taskQueue.length) return;
        const { id, task } = taskQueue[nextTask++];
        const [phaseName, testId, testType, testData] = task;
        worker.busy = true;
        worker.pending = { id, timeout: setTimeout(() => {
            // Timeout - kill and respawn worker
            worker.proc.kill();
            results[id] = [phaseName, testId, "fail", "Test timed out after 3s"];
            completed++;
            printResult(results[id]);
            // Respawn
            const newProc = fork(__filename, [], { env: { ...process.env, ...workerEnv }, silent: true });
            worker.proc = newProc;
            worker.busy = false;
            worker.ready = false;
            newProc.once("message", (msg) => {
                if (msg.type === "ready") { worker.ready = true; dispatch(worker); }
            });
            newProc.on("message", handleMessage.bind(null, worker));
        }, 3000) };
        worker.proc.send({ type: "task", id, phaseName, testId, testType, testData });
    }
    function handleMessage(worker, msg) {
        if (msg.type === "result") {
            clearTimeout(worker.pending.timeout);
            results[msg.id] = msg.result;
            completed++;
            printResult(msg.result);
            worker.busy = false;
            worker.pending = null;
            dispatch(worker);
        }
    }
    function printResult([phaseName, testId, status, err]) {
        if (status === "pass") {
            console.log(`PASS ${vmTag}${phaseName}::${testId}`);
            totalPass++;
        } else if (status === "skip") {
            console.log(`SKIP ${vmTag}${phaseName}::${testId}`);
            totalSkip++;
        } else {
            console.log(`FAIL ${vmTag}${phaseName}::${testId}`);
            if (err) {
                for (const line of String(err).split("\n")) {
                    console.log(`  ${line}`);
                }
            }
            totalFail++;
            failures.push([phaseName, testId, err]);
        }
    }
    // Set up message handlers and start dispatching
    for (const worker of workers) {
        worker.proc.on("message", handleMessage.bind(null, worker));
        dispatch(worker);
    }
    // Wait for all tasks to complete
    await new Promise(resolve => {
        const check = setInterval(() => {
            if (completed >= collected.length) {
                clearInterval(check);
                resolve();
            }
        }, 50);
    });
    // Terminate workers
    for (const worker of workers) {
        worker.proc.send({ type: "exit" });
    }
    const tElapsed = (Date.now() - tStart) / 1000;
    console.log(`Completed in ${tElapsed.toFixed(1)}s`);
}

function runSerial() {
    const tStart = Date.now();
    for (const [phaseName, testId, testType, testData] of collected) {
        const [, , status, err] = runSingleTest(phaseName, testId, testType, testData);
        if (status === "pass") {
            console.log(`PASS ${vmTag}${phaseName}::${testId}`);
            totalPass++;
        } else if (status === "skip") {
            console.log(`SKIP ${vmTag}${phaseName}::${testId}`);
            totalSkip++;
        } else {
            console.log(`FAIL ${vmTag}${phaseName}::${testId}`);
            if (err) {
                for (const line of err.split("\n")) {
                    console.log(`  ${line}`);
                }
            }
            totalFail++;
            failures.push([phaseName, testId, err]);
        }
    }
    const tElapsed = (Date.now() - tStart) / 1000;
    console.log(`Completed in ${tElapsed.toFixed(1)}s`);
}

// Run tests - wrap in async IIFE for parallel execution
(async () => {
    if (numWorkers > 1) {
        await runParallel();
    } else {
        runSerial();
    }

    console.log();
    if (failures.length > 0) {
        console.log("=".repeat(60));
        console.log(targetName ? `FAILURES [${targetName}]` : "FAILURES");
        console.log("=".repeat(60));
        for (const [phase, tid, err] of failures) {
            console.log(`  ${phase}::${tid}`);
        }
        console.log();
    }

    console.log("=".repeat(60));
    const total = totalPass + totalFail + totalSkip;
    const prefix = targetName ? `[${targetName}] ` : "";
    const summaryLine = `${prefix}${total} tests: ${totalPass} passed, ${totalFail} failed, ${totalSkip} skipped`;
    console.log(summaryLine);
    console.log("=".repeat(60));

    // GitHub Actions notice annotation
    if (totalFail === 0) {
        console.log(`::notice::${summaryLine}`);
    }

    // GitHub Actions job summary
    const summaryFile = process.env.GITHUB_STEP_SUMMARY;
    if (summaryFile) {
        const statusEmoji = totalFail === 0 ? "✅" : "❌";
        let md = `## ${statusEmoji} ${targetName || "Test Results"}\n\n`;
        md += `| Passed | Failed | Skipped | Total |\n`;
        md += `|--------|--------|---------|-------|\n`;
        md += `| ${totalPass} | ${totalFail} | ${totalSkip} | ${total} |\n\n`;
        if (failures.length > 0) {
            md += "### Failures\n\n";
            for (const [phase, tid, err] of failures) {
                md += `<details><summary><code>${phase} :: ${tid}</code></summary>\n\n`;
                md += `\`\`\`\n${err}\n\`\`\`\n\n</details>\n\n`;
            }
        }
        nodeFs.appendFileSync(summaryFile, md);
    }

    process.exit(totalFail > 0 ? 1 : 0);
})();

} // end if not worker mode
