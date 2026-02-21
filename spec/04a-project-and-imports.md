# Phase 3a: Project Compilation

**Related:** `04-subset-spec.md` (import syntax and allowed stdlib), `02-cli-spec.md` (input handling), `05-names-spec.md` (name binding)

When the input is a directory, the compiler gathers all Python files, resolves cross-file imports, detects name collisions, merges the files into a single AST, and feeds it through the existing single-file pipeline. Output is always a single file.

This phase runs between parse (phase 2) and subset (phase 3). It is skipped entirely when the input is one file — single-file compilation is unchanged.

## CLI

```
tongues ./myproject --target go -o out.go
```

When the positional argument is a directory, the compiler enters project mode. All other flags behave identically to single-file mode.

| Input         | Syntax                | Behavior                              |
| ------------- | --------------------- | ------------------------------------- |
| Directory     | `tongues ./src`       | Gather .py files, compile as project  |
| Single file   | `tongues foo.py`      | Existing single-file pipeline         |
| Stdin         | `tongues < foo.py`    | Existing single-file pipeline         |

Project mode is not available via stdin. A directory with zero `.py` files is an error.

## Gathering

The compiler recursively walks the input directory and collects all `.py` files. The walk skips:

- Entries starting with `.` (hidden files/directories)
- `__pycache__` directories
- Files containing `tongues: skip` in the first 5 lines (see `02-cli-spec.md`)

Files are sorted lexicographically for deterministic ordering. The set of collected paths is the **universe** — no file outside the universe may be referenced by a project import.

The **project root** is the input directory path (normalized, trailing separator removed).

## Import Classification

Every import in the universe falls into one of three categories:

| Category | Examples | Handling |
|----------|----------|----------|
| Stdlib | `import sys`, `import os`, `from typing import ...`, `from dataclasses import dataclass`, `from collections.abc import ...`, `from __future__ import annotations` | Kept as-is. Validated by subset (phase 3). |
| Project | `from .module import X`, `from .module import X as Y`, `from . import module`, `from pkg.sub import X` | Resolved to a file in the universe. Stripped during merge. |
| Invalid | `import requests`, `from numpy import array` | Error: unresolved import. |

Classification reuses the `IMPORT_ONLY_MODULES` and `ALLOWED_FROM_MODULES` sets from `subset.py`. An `ImportFrom` node that is not stdlib and has `level > 0` (relative) is always a project import. An `ImportFrom` with `level == 0` and module not in the allowed sets is a project import (absolute).

The syntactic rules from `04-subset-spec.md` apply to all imports regardless of category: no star imports, no `import X as Y`, etc. Phase 3a validates these before attempting resolution.

## Import Resolution

Resolution takes: importing file path, module string, relative level, project root, and the universe (set of paths).

### Relative imports (level > 0)

1. Let `dir` = directory of the importing file.
2. Go up `level - 1` directories from `dir`.
3. If module is non-empty: replace dots with `/`, append to the current directory. Try `{dir}/{module_path}.py`, then `{dir}/{module_path}/__init__.py`. First match in the universe wins.
4. If module is empty (`from . import name`): for each imported name, try `{dir}/{name}.py`, then `{dir}/{name}/__init__.py`. First match in the universe wins.

### Absolute project imports (level == 0)

Same as relative but anchored at project root: `{project_root}/{module_path}.py`, then `{project_root}/{module_path}/__init__.py`.

### Result

Resolution returns a path in the universe or failure. Failure is an unresolved import error.

## Dependency Order

Build a directed graph: file A depends on file B if A has a project import that resolves to B. Compute a topological sort of this graph.

Circular imports are allowed. When cycles exist, the topological sort breaks them arbitrarily (lexicographic order as tiebreaker). This is safe for the IR — the Taytsh type checker resolves all names regardless of position (see `12-taytsh-ir-spec.md`, Module Structure). However, the emitted target code may require declarations in dependency order. Backends handle this with a separate type-dependency sort (see `21-BACKEND-SPEC.md`, Declaration Ordering).

## Name Collection

For each file in the universe, collect all module-level names and their kinds:

| AST node type | Names collected |
|---------------|-----------------|
| `ClassDef` | class name |
| `FunctionDef` | function name |
| `Assign` | target names (Name nodes only) |
| `AnnAssign` | target name (Name node only) |

Names introduced by import statements are **not** collected — they are bindings to names defined elsewhere, not definitions. This matches the `05-names-spec.md` distinction: imports have kind `import`, while the actual definitions have kind `class`, `function`, `variable`, etc.

## Collision Detection

Two files in the universe may not define the same module-level name. For each name that appears in more than one file:

```
error: duplicate name 'Token' defined in src/parse.py:15 and src/check.py:42
```

All collisions are reported (not just the first). Any collision is a fatal error — the merge does not proceed.

Names prefixed with `_` (conventionally private) are subject to the same rule. The flat namespace has no file-private scope.

## Import Binding Map

For each file, build a map from locally-used name to its definition location:

| Import form | Local name | Resolves to |
|-------------|------------|-------------|
| `from .parse import Token` | `Token` | `Token` in `src/parse.py` |
| `from .parse import Token as Tok` | `Tok` | `Token` in `src/parse.py` |
| `from . import parse` | `parse` | module object — see below |

### Module-as-name imports

`from . import parse` imports the module as a name. References appear as `parse.Token`, `parse.tokenize()`. During merge, these are rewritten:

- `parse.Token` → `Token` (the definition's original name, now in flat namespace)
- `parse.tokenize()` → `tokenize()` (same)

The merge walker replaces `Attribute` nodes where the value is a module-name binding with plain `Name` nodes referencing the target name. If the attribute name is not a module-level name in the target file, it is an error.

### Aliased imports

`from .parse import Token as Tok` binds `Tok` locally to mean `Token`. During merge, all references to `Tok` in the importing file are rewritten to `Token`. This rewriting is purely local to the importing file's AST.

## AST Merge

After resolution and collision detection:

1. For each file in dependency order:
   a. Walk the AST body. Remove all project import nodes (`ImportFrom` nodes classified as project imports).
   b. For aliased imports: walk the file's AST and rename all references from the alias to the original name.
   c. For module-as-name imports: walk the file's AST and rewrite `module.attr` attribute access to plain `attr` name references.
   d. Tag each top-level AST node with `_source_file: str` (the file path relative to project root).
2. Concatenate all files' AST body lists in dependency order into a single body list.
3. Construct a merged AST dict with the combined body.

The merged AST is then passed to phase 3 (subset) and onward. Since project imports have been stripped, the subset phase only sees stdlib imports and validates them with existing logic.

After merge, the names phase (phase 4) sees a single AST where all definitions from all files are module-level. Imported names (e.g. `Token` imported from `parse.py` into `check.py`) are no longer import bindings — they are direct references to the class/function definition now present in the merged module scope. This matches the `05-names-spec.md` resolution model: local → module → builtin.

## Entry Point

The lowerer identifies the entry point by looking for `if __name__ == '__main__':` in the merged AST. In project mode, exactly one file should contain this guard. If zero files contain it, there is no entry point (valid for library compilation). If multiple files contain it, only the one from the file detected as the main module (last in dependency order, or containing the guard) is used.

## Error Reporting

Every error in phases 3+ may include a source file path if the AST node carries `_source_file`. The error format becomes:

```
src/parse.py:15:8: error: missing type annotation for 'x'
```

When `_source_file` is absent (single-file mode), the format is unchanged:

```
15:8: error: missing type annotation for 'x'
```

Pipeline phases do not need to be aware of project mode. The `_source_file` tag is carried through the AST as an opaque attribute. Phases that report errors using AST node positions check for `_source_file` and prepend it when present.

## Errors

| Condition | Diagnostic | Exit |
|-----------|------------|------|
| No .py files in directory | `error: no .py files found in directory` | 1 |
| Unresolved import | `{file}:{line}:{col}: unresolved import: {module}` | 1 |
| Name collision | `error: duplicate name '{name}' defined in {file1}:{line1} and {file2}:{line2}` | 1 |
| Module-as-name: unknown attr | `{file}:{line}:{col}: '{module}.{attr}' does not exist in {target_file}` | 1 |

## Phase Control Interaction

`--stop-at` works with project mode:

| Phase | Behavior in project mode |
|-------|--------------------------|
| `parse` | Output: JSON array of `{path, ast}` objects (one per file) |
| `subset` | Validates the merged AST |
| (other phases) | Operate on merged AST, same as single-file |
