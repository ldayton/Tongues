"""Subset-compliant entry point."""

from __future__ import annotations

import sys

from .frontend.parse import parse, ParseError
from .frontend.subset import (
    verify as verify_subset,
    IMPORT_ONLY_MODULES,
    ALLOWED_FROM_MODULES,
)
from .frontend.names import NameInfo, NameTable, resolve_names
from .frontend.signatures import annotation_to_str, collect_signatures
from .frontend.fields import collect_fields
from .frontend.hierarchy import build_hierarchy
from .frontend.inference import run_inference
from .frontend.lowering import lower
from .taytsh.ast import to_dict as module_to_dict
from .taytsh.check import Checker
from .middleend.returns import analyze_returns
from .middleend.scope import analyze_scope
from .middleend.liveness import analyze_liveness
from .backend.python import emit_python
from .backend.perl import emit_perl
from .backend.ruby import emit_ruby

TARGETS: list[str] = [
    "c",
    "csharp",
    "dart",
    "go",
    "java",
    "javascript",
    "lua",
    "perl",
    "php",
    "python",
    "ruby",
    "rust",
    "swift",
    "typescript",
    "zig",
]

PHASES: list[str] = [
    "parse",
    "subset",
    "names",
    "signatures",
    "fields",
    "hierarchy",
    "inference",
    "lowering",
    "analyze",
]

USAGE: str = """\
tongues [OPTIONS] [INPUT] [-o OUTPUT]

Options:
  --target TARGET     Output language: c, csharp, dart, go, java, javascript,
                      lua, perl, php, python, ruby, rust, swift, typescript, zig
  --stop-at PHASE     Stop after phase: parse, subset, names, signatures,
                      fields, hierarchy, inference, lowering, analyze
  --project           Read NUL-delimited multi-file input (path\\0source\\0...)
  --strict            Enable strict math and strict tostring
  --strict-math       Enable strict math mode
  --strict-tostring   Enable strict tostring mode
  -o, --output FILE   Write output to FILE instead of stdout
  --help              Show this help message
"""


def should_skip_file(source: str) -> bool:
    """Check if file has a tongues: skip directive in first 5 lines."""
    lines = source.split("\n", 5)
    i = 0
    while i < len(lines) and i < 5:
        if "tongues: skip" in lines[i]:
            return True
        i += 1
    return False


def read_source(input_file: str | None) -> tuple[str, int]:
    """Read source from file or stdin. Returns (source, exit_code) where exit_code 0 means OK."""
    if input_file is not None:
        try:
            with open(input_file, "rb") as f:
                raw = f.read()
        except OSError:
            print("error: cannot open '" + input_file + "'", file=sys.stderr)
            return ("", 1)
    else:
        raw = sys.stdin.buffer.read()
    if len(raw) > 0:
        try:
            source = raw.decode("utf-8")
        except ValueError:
            print("error: invalid utf-8 in input", file=sys.stderr)
            return ("", 1)
        return (source, 0)
    return ("", 0)


def write_output(output: str, output_file: str | None) -> int:
    """Write output to file or stdout. Returns 0 on success, 1 on error."""
    if output_file is not None:
        try:
            with open(output_file, "w") as f:
                f.write(output)
        except OSError:
            print("error: cannot write '" + output_file + "'", file=sys.stderr)
            return 1
        return 0
    print(output)
    return 0


# --- JSON serialization (subset-compliant, no json module) ---


def _json_escape(s: str) -> str:
    """Escape a string for JSON output."""
    result: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            result.append("\\\\")
        elif c == '"':
            result.append('\\"')
        elif c == "\n":
            result.append("\\n")
        elif c == "\r":
            result.append("\\r")
        elif c == "\t":
            result.append("\\t")
        else:
            result.append(c)
        i += 1
    return "".join(result)


def _to_json(obj: object, indent: int, level: int) -> str:
    """Recursively serialize an object to JSON string."""
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        if obj:
            return "true"
        return "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return str(obj)
    if isinstance(obj, str):
        return '"' + _json_escape(obj) + '"'
    if isinstance(obj, list):
        if len(obj) == 0:
            return "[]"
        parts: list[str] = []
        pad = " " * (indent * (level + 1))
        pad_close = " " * (indent * level)
        i = 0
        while i < len(obj):
            parts.append(pad + _to_json(obj[i], indent, level + 1))
            i += 1
        return "[\n" + ",\n".join(parts) + "\n" + pad_close + "]"
    if isinstance(obj, dict):
        if len(obj) == 0:
            return "{}"
        parts: list[str] = []
        pad = " " * (indent * (level + 1))
        pad_close = " " * (indent * level)
        keys = list(obj.keys())
        i = 0
        while i < len(keys):
            k = keys[i]
            v = obj[k]
            key_str = '"' + _json_escape(str(k)) + '"'
            val_str = _to_json(v, indent, level + 1)
            parts.append(pad + key_str + ": " + val_str)
            i += 1
        return "{\n" + ",\n".join(parts) + "\n" + pad_close + "}"
    return '"<unserializable>"'


def to_json(obj: object) -> str:
    """Serialize object to pretty-printed JSON."""
    return _to_json(obj, 2, 0)


# --- Name table serialization ---


def _name_info_to_dict(info: NameInfo) -> dict[str, object]:
    """Convert a NameInfo to a serializable dict."""
    d: dict[str, object] = {
        "kind": info.kind,
        "scope": info.scope,
        "lineno": info.lineno,
        "col": info.col,
    }
    if info.decl_class != "":
        d["decl_class"] = info.decl_class
    if info.decl_func != "":
        d["decl_func"] = info.decl_func
    if len(info.bases) > 0:
        d["bases"] = info.bases
    return d


def _name_table_to_dict(table: NameTable) -> dict[str, object]:
    """Convert a NameTable to spec-compliant format: {"names": {...}, "scopes": [...]}."""
    names: dict[str, object] = {}
    keys = list(table.module_names.keys())
    i = 0
    while i < len(keys):
        name = keys[i]
        names[name] = _name_info_to_dict(table.module_names[name])
        i += 1
    scopes: list[object] = []
    ckeys = list(table.class_names.keys())
    i = 0
    while i < len(ckeys):
        cname = ckeys[i]
        scope_names: dict[str, object] = {}
        mkeys = list(table.class_names[cname].keys())
        j = 0
        while j < len(mkeys):
            mname = mkeys[j]
            scope_names[mname] = _name_info_to_dict(table.class_names[cname][mname])
            j += 1
        scopes.append({"scope": cname, "names": scope_names})
        i += 1
    lkeys = list(table.local_names.keys())
    i = 0
    while i < len(lkeys):
        lkey = lkeys[i]
        if str(lkey[0]) != "":
            scope_key = str(lkey[0]) + ":" + str(lkey[1])
        else:
            scope_key = str(lkey[1])
        scope_names: dict[str, object] = {}
        skeys = list(table.local_names[lkey].keys())
        j = 0
        while j < len(skeys):
            sname = skeys[j]
            scope_names[sname] = _name_info_to_dict(table.local_names[lkey][sname])
            j += 1
        scopes.append({"scope": scope_key, "names": scope_names})
        i += 1
    result: dict[str, object] = {"names": names}
    if len(scopes) > 0:
        result["scopes"] = scopes
    return result


# --- Pragma extraction ---


def _extract_pragmas(
    source: str,
) -> tuple[str, bool, bool]:
    """Strip @@[...] pragma lines from the start of source.

    Returns (remaining_source, strict_math, strict_tostring).
    """
    strict_math = False
    strict_tostring = False
    lines = source.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "":
            i += 1
            continue
        if not stripped.startswith("@@["):
            break
        body = stripped[3:]
        if not body.endswith("]"):
            break
        body = body[:-1]
        entries = body.split(",")
        j = 0
        while j < len(entries):
            entry = entries[j].strip().strip('"')
            if entry == "strict_math":
                strict_math = True
            elif entry == "strict_tostring":
                strict_tostring = True
            j += 1
        i += 1
    remaining = "\n".join(lines[i:])
    return (remaining, strict_math, strict_tostring)


# --- Error reporting ---


def _print_errors(errors: list[object]) -> None:
    """Print a list of error objects to stderr."""
    i = 0
    while i < len(errors):
        print(str(errors[i]), file=sys.stderr)
        i += 1


# --- Pipeline ---


# --- Project merge (Phase 3a) ---


def _classify_import(node: dict[str, object]) -> str:
    """Classify an ImportFrom node as 'stdlib' or 'project'."""
    level = node.get("level", 0)
    if not isinstance(level, int):
        level = 0
    if level > 0:
        return "project"
    module = node.get("module", "")
    if module is None:
        module = ""
    if not isinstance(module, str):
        module = ""
    if module in ALLOWED_FROM_MODULES:
        return "stdlib"
    if module in IMPORT_ONLY_MODULES:
        return "stdlib"
    parts = module.split(".")
    if len(parts) > 0 and parts[0] in ALLOWED_FROM_MODULES:
        return "stdlib"
    if len(parts) > 0 and parts[0] in IMPORT_ONLY_MODULES:
        return "stdlib"
    return "project"


def _resolve_project_import(
    importing_file: str,
    module: str,
    level: int,
    names: list[dict[str, object]],
    universe: set[str],
) -> list[tuple[str, str]]:
    """Resolve project import to paths in universe. Returns [(path, "")] or [("", error_msg)]."""
    if level > 0:
        slash_idx = importing_file.rfind("/")
        if slash_idx >= 0:
            dir_path = importing_file[:slash_idx]
        else:
            dir_path = ""
        up = level - 1
        while up > 0:
            slash_idx = dir_path.rfind("/")
            if slash_idx >= 0:
                dir_path = dir_path[:slash_idx]
            else:
                dir_path = ""
            up -= 1
        if module != "":
            module_path = module.replace(".", "/")
            if dir_path != "":
                candidate = dir_path + "/" + module_path
            else:
                candidate = module_path
            py_path = candidate + ".py"
            init_path = candidate + "/__init__.py"
            if py_path in universe:
                return [(py_path, "")]
            if init_path in universe:
                return [(init_path, "")]
            return [("", importing_file + ": unresolved import: " + module)]
        else:
            results: list[tuple[str, str]] = []
            i = 0
            while i < len(names):
                name_node = names[i]
                name = ""
                if isinstance(name_node, dict):
                    n = name_node.get("name", "")
                    if isinstance(n, str):
                        name = n
                if name != "" and name != "*":
                    if dir_path != "":
                        candidate = dir_path + "/" + name
                    else:
                        candidate = name
                    py_path = candidate + ".py"
                    init_path = candidate + "/__init__.py"
                    if py_path in universe:
                        results.append((py_path, ""))
                    elif init_path in universe:
                        results.append((init_path, ""))
                    else:
                        results.append(
                            ("", importing_file + ": unresolved import: " + name)
                        )
                i += 1
            return results
    else:
        module_path = module.replace(".", "/")
        py_path = module_path + ".py"
        init_path = module_path + "/__init__.py"
        if py_path in universe:
            return [(py_path, "")]
        if init_path in universe:
            return [(init_path, "")]
        return [("", importing_file + ": unresolved import: " + module)]


def _dependency_order(files: list[str], deps: dict[str, list[str]]) -> list[str]:
    """Topological sort with lexicographic tiebreaker, cycle-tolerant."""
    in_degree: dict[str, int] = {}
    i = 0
    while i < len(files):
        in_degree[files[i]] = 0
        i += 1
    i = 0
    while i < len(files):
        f = files[i]
        dep_list = deps.get(f)
        if dep_list is not None:
            j = 0
            while j < len(dep_list):
                d = dep_list[j]
                if d in in_degree:
                    in_degree[d] = in_degree[d] + 1
                j += 1
        i += 1
    ready: list[str] = []
    i = 0
    while i < len(files):
        if in_degree[files[i]] == 0:
            ready.append(files[i])
        i += 1
    ready.sort()
    result: list[str] = []
    while len(ready) > 0:
        node = ready[0]
        ready = ready[1:]
        result.append(node)
        dep_list = deps.get(node)
        if dep_list is not None:
            j = 0
            while j < len(dep_list):
                d = dep_list[j]
                if d in in_degree:
                    in_degree[d] = in_degree[d] - 1
                    if in_degree[d] == 0:
                        k = 0
                        inserted = False
                        while k < len(ready):
                            if d < ready[k]:
                                ready.insert(k, d)
                                inserted = True
                                break
                            k += 1
                        if not inserted:
                            ready.append(d)
                j += 1
    if len(result) < len(files):
        remaining: list[str] = []
        i = 0
        while i < len(files):
            found = False
            j = 0
            while j < len(result):
                if files[i] == result[j]:
                    found = True
                    break
                j += 1
            if not found:
                remaining.append(files[i])
            i += 1
        remaining.sort()
        i = 0
        while i < len(remaining):
            result.append(remaining[i])
            i += 1
    return result


def _collect_module_names(
    ast_dict: dict[str, object],
) -> list[tuple[str, int, int, dict[str, object]]]:
    """Collect (name, lineno, col, stmt) for module-level definitions."""
    result: list[tuple[str, int, int, dict[str, object]]] = []
    body = ast_dict.get("body", [])
    if not isinstance(body, list):
        return result
    i = 0
    while i < len(body):
        stmt = body[i]
        if not isinstance(stmt, dict):
            i += 1
            continue
        node_type = stmt.get("_type", "")
        lineno = stmt.get("lineno", 0)
        if not isinstance(lineno, int):
            lineno = 0
        col = stmt.get("col_offset", 0)
        if not isinstance(col, int):
            col = 0
        if node_type == "ClassDef":
            name = stmt.get("name", "")
            if isinstance(name, str) and name != "":
                result.append((name, lineno, col, stmt))
        elif node_type == "FunctionDef":
            name = stmt.get("name", "")
            if isinstance(name, str) and name != "":
                result.append((name, lineno, col, stmt))
        elif node_type == "Assign":
            targets = stmt.get("targets", [])
            if isinstance(targets, list):
                j = 0
                while j < len(targets):
                    target = targets[j]
                    if isinstance(target, dict) and target.get("_type") == "Name":
                        tid = target.get("id", "")
                        if isinstance(tid, str) and tid != "":
                            result.append((tid, lineno, col, stmt))
                    j += 1
        elif node_type == "AnnAssign":
            target = stmt.get("target", {})
            if isinstance(target, dict) and target.get("_type") == "Name":
                tid = target.get("id", "")
                if isinstance(tid, str) and tid != "":
                    result.append((tid, lineno, col, stmt))
        i += 1
    return result


def _detect_collisions(
    file_names: dict[str, list[tuple[str, int, int]]],
) -> list[str]:
    """Return error messages for cross-file name collisions."""
    name_to_locs: dict[str, list[tuple[str, int]]] = {}
    fkeys = list(file_names.keys())
    i = 0
    while i < len(fkeys):
        f = fkeys[i]
        names = file_names[f]
        j = 0
        while j < len(names):
            name, lineno, col = names[j]
            if name not in name_to_locs:
                name_to_locs[name] = []
            name_to_locs[name].append((f, lineno))
            j += 1
        i += 1
    errors: list[str] = []
    nkeys = list(name_to_locs.keys())
    nkeys.sort()
    i = 0
    while i < len(nkeys):
        name = nkeys[i]
        locs = name_to_locs[name]
        if len(locs) > 1:
            locs.sort()
            msg = "error: duplicate name '" + name + "' defined in "
            j = 0
            while j < len(locs):
                if j > 0:
                    msg = msg + " and "
                msg = msg + locs[j][0] + ":" + str(locs[j][1])
                j += 1
            errors.append(msg)
        i += 1
    return errors


def _ast_equal(a: object, b: object) -> bool:
    """Deep structural comparison of AST nodes, ignoring position metadata."""
    ignore: set[str] = {
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
        "_source_file",
        "_remove",
    }
    work_a: list[object] = [a]
    work_b: list[object] = [b]
    wi = 0
    while wi < len(work_a):
        x = work_a[wi]
        y = work_b[wi]
        if isinstance(x, dict) and isinstance(y, dict):
            x_keys: list[str] = []
            xk = list(x.keys())
            ki = 0
            while ki < len(xk):
                if xk[ki] not in ignore:
                    x_keys.append(xk[ki])
                ki += 1
            y_keys: list[str] = []
            yk = list(y.keys())
            ki = 0
            while ki < len(yk):
                if yk[ki] not in ignore:
                    y_keys.append(yk[ki])
                ki += 1
            if len(x_keys) != len(y_keys):
                return False
            x_keys.sort()
            y_keys.sort()
            ki = 0
            while ki < len(x_keys):
                if x_keys[ki] != y_keys[ki]:
                    return False
                ki += 1
            ki = 0
            while ki < len(x_keys):
                work_a.append(x[x_keys[ki]])
                work_b.append(y[y_keys[ki]])
                ki += 1
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                return False
            li = 0
            while li < len(x):
                work_a.append(x[li])
                work_b.append(y[li])
                li += 1
        else:
            if x != y:
                return False
        wi += 1
    return True


def _collect_definition_refs(node: dict[str, object]) -> set[str]:
    """Collect all Name.id values referenced in a definition's body."""
    refs: set[str] = set()
    node_type = node.get("_type", "")
    seeds: list[object] = []
    if node_type == "FunctionDef":
        body = node.get("body", [])
        if isinstance(body, list):
            seeds.append(body)
        args = node.get("args")
        if isinstance(args, dict):
            seeds.append(args)
        returns = node.get("returns")
        if returns is not None:
            seeds.append(returns)
        deco = node.get("decorator_list", [])
        if isinstance(deco, list):
            seeds.append(deco)
    elif node_type == "ClassDef":
        body = node.get("body", [])
        if isinstance(body, list):
            seeds.append(body)
        bases = node.get("bases", [])
        if isinstance(bases, list):
            seeds.append(bases)
        deco = node.get("decorator_list", [])
        if isinstance(deco, list):
            seeds.append(deco)
    elif node_type == "Assign":
        value = node.get("value")
        if value is not None:
            seeds.append(value)
    elif node_type == "AnnAssign":
        value = node.get("value")
        if value is not None:
            seeds.append(value)
        ann = node.get("annotation")
        if ann is not None:
            seeds.append(ann)
    work: list[object] = list(seeds)
    wi = 0
    while wi < len(work):
        item = work[wi]
        if isinstance(item, dict):
            if item.get("_type") == "Name":
                nid = item.get("id")
                if isinstance(nid, str):
                    refs.add(nid)
            keys = list(item.keys())
            ki = 0
            while ki < len(keys):
                val = item[keys[ki]]
                if isinstance(val, dict) or isinstance(val, list):
                    work.append(val)
                ki += 1
        elif isinstance(item, list):
            li = 0
            while li < len(item):
                child = item[li]
                if isinstance(child, dict) or isinstance(child, list):
                    work.append(child)
                li += 1
        wi += 1
    return refs


def _compute_module_stems(paths: list[str]) -> dict[str, str]:
    """Compute unique module stems for each file path."""
    raw_stems: dict[str, str] = {}
    i = 0
    while i < len(paths):
        path = paths[i]
        slash_idx = path.rfind("/")
        if slash_idx >= 0:
            filename = path[slash_idx + 1 :]
            parent = path[:slash_idx]
        else:
            filename = path
            parent = ""
        if filename == "__init__.py":
            parent_slash = parent.rfind("/")
            if parent_slash >= 0:
                stem = parent[parent_slash + 1 :]
            else:
                stem = parent
            if stem == "":
                stem = "__init__"
        else:
            if filename.endswith(".py"):
                stem = filename[:-3]
            else:
                stem = filename
        raw_stems[path] = stem
        i += 1
    stem_to_paths: dict[str, list[str]] = {}
    i = 0
    while i < len(paths):
        path = paths[i]
        stem = raw_stems[path]
        if stem not in stem_to_paths:
            stem_to_paths[stem] = []
        stem_to_paths[stem].append(path)
        i += 1
    result: dict[str, str] = {}
    skeys = list(stem_to_paths.keys())
    si = 0
    while si < len(skeys):
        stem = skeys[si]
        colliding = stem_to_paths[stem]
        if len(colliding) == 1:
            result[colliding[0]] = stem
        else:
            ci = 0
            while ci < len(colliding):
                path = colliding[ci]
                slash_idx = path.rfind("/")
                if slash_idx >= 0:
                    parent = path[:slash_idx]
                    parent_slash = parent.rfind("/")
                    if parent_slash >= 0:
                        parent_name = parent[parent_slash + 1 :]
                    else:
                        parent_name = parent
                    result[path] = parent_name + "_" + stem
                else:
                    result[path] = stem
                ci += 1
        si += 1
    return result


def _is_all_caps(name: str) -> bool:
    """Check if name follows ALL_CAPS convention."""
    if len(name) == 0:
        return False
    has_letter = False
    i = 0
    while i < len(name):
        c = name[i]
        if c != "_" and not c.isupper() and not c.isdigit():
            return False
        if c.isupper():
            has_letter = True
        i += 1
    return has_letter


def _prefix_name(name: str, stem: str) -> str:
    """Compute prefixed name for collision resolution."""
    if len(name) > 0 and name[0] == "_":
        if _is_all_caps(name[1:]):
            return "_" + stem.upper() + "_" + name[1:]
        return "_" + stem + "_" + name[1:]
    if _is_all_caps(name):
        return stem.upper() + "_" + name
    return stem + "_" + name


def _plan_collision_resolution(
    file_names: dict[str, list[tuple[str, int, int, dict[str, object]]]],
    stems: dict[str, str],
) -> tuple[set[str], dict[str, dict[str, str]]]:
    """Plan collision resolution. Returns (dedup_names, file_renames)."""
    name_to_defs: dict[str, list[tuple[str, dict[str, object]]]] = {}
    fkeys = list(file_names.keys())
    fi = 0
    while fi < len(fkeys):
        f = fkeys[fi]
        names = file_names[f]
        ni = 0
        while ni < len(names):
            name = names[ni][0]
            ast_node = names[ni][3]
            if name not in name_to_defs:
                name_to_defs[name] = []
            name_to_defs[name].append((f, ast_node))
            ni += 1
        fi += 1
    dedup_candidates: set[str] = set()
    file_renames: dict[str, dict[str, str]] = {}
    nkeys = list(name_to_defs.keys())
    ni = 0
    while ni < len(nkeys):
        name = nkeys[ni]
        defs = name_to_defs[name]
        if len(defs) > 1:
            all_equal = True
            di = 1
            while di < len(defs):
                if not _ast_equal(defs[0][1], defs[di][1]):
                    all_equal = False
                    break
                di += 1
            if all_equal:
                dedup_candidates.add(name)
            else:
                di = 0
                while di < len(defs):
                    f = defs[di][0]
                    stem = stems[f]
                    prefixed = _prefix_name(name, stem)
                    if f not in file_renames:
                        file_renames[f] = {}
                    file_renames[f][name] = prefixed
                    di += 1
        ni += 1
    changed = True
    while changed:
        changed = False
        all_prefixed: set[str] = set()
        rkeys = list(file_renames.keys())
        ri = 0
        while ri < len(rkeys):
            rmap = file_renames[rkeys[ri]]
            mk = list(rmap.keys())
            mi = 0
            while mi < len(mk):
                all_prefixed.add(mk[mi])
                mi += 1
            ri += 1
        to_demote: list[str] = []
        dedup_list = list(dedup_candidates)
        di = 0
        while di < len(dedup_list):
            name = dedup_list[di]
            defs = name_to_defs[name]
            refs = _collect_definition_refs(defs[0][1])
            ref_list = list(refs)
            ri = 0
            unsafe = False
            while ri < len(ref_list):
                if ref_list[ri] in all_prefixed:
                    unsafe = True
                    break
                ri += 1
            if unsafe:
                to_demote.append(name)
            di += 1
        dmi = 0
        while dmi < len(to_demote):
            name = to_demote[dmi]
            dedup_candidates.discard(name)
            defs = name_to_defs[name]
            di = 0
            while di < len(defs):
                f = defs[di][0]
                stem = stems[f]
                prefixed = _prefix_name(name, stem)
                if f not in file_renames:
                    file_renames[f] = {}
                file_renames[f][name] = prefixed
                di += 1
            changed = True
            dmi += 1
    return (dedup_candidates, file_renames)


def _rewrite_names(node: dict[str, object], rename_map: dict[str, str]) -> None:
    """Recursively rename Name nodes and definition names per rename_map. In-place."""
    work: list[object] = [node]
    wi = 0
    while wi < len(work):
        item = work[wi]
        if isinstance(item, dict):
            ntype = item.get("_type")
            if ntype == "Name":
                nid = item.get("id")
                if isinstance(nid, str) and nid in rename_map:
                    item["id"] = rename_map[nid]
            elif ntype == "FunctionDef" or ntype == "ClassDef":
                def_name = item.get("name")
                if isinstance(def_name, str) and def_name in rename_map:
                    item["name"] = rename_map[def_name]
            keys = list(item.keys())
            ki = 0
            while ki < len(keys):
                val = item[keys[ki]]
                if isinstance(val, dict) or isinstance(val, list):
                    work.append(val)
                ki += 1
        elif isinstance(item, list):
            li = 0
            while li < len(item):
                child = item[li]
                if isinstance(child, dict) or isinstance(child, list):
                    work.append(child)
                li += 1
        wi += 1


def _rewrite_module_attrs(
    node: dict[str, object],
    module_bindings: dict[str, str],
    file_name_map: dict[str, dict[str, str]],
) -> list[str]:
    """Rewrite module.attr Attribute nodes to plain Name nodes. Returns errors."""
    errors: list[str] = []
    work: list[object] = [node]
    wi = 0
    while wi < len(work):
        item = work[wi]
        if isinstance(item, dict):
            keys = list(item.keys())
            ki = 0
            while ki < len(keys):
                val = item[keys[ki]]
                if isinstance(val, dict):
                    if val.get("_type") == "Attribute":
                        value_node = val.get("value")
                        if (
                            isinstance(value_node, dict)
                            and value_node.get("_type") == "Name"
                        ):
                            mod_name = value_node.get("id", "")
                            if (
                                isinstance(mod_name, str)
                                and mod_name in module_bindings
                            ):
                                target_file = module_bindings[mod_name]
                                attr = val.get("attr", "")
                                if not isinstance(attr, str):
                                    attr = ""
                                target_name_map = file_name_map.get(target_file)
                                if (
                                    target_name_map is not None
                                    and attr in target_name_map
                                ):
                                    final_name = target_name_map[attr]
                                    lineno = val.get("lineno", 0)
                                    col = val.get("col_offset", 0)
                                    end_lineno = val.get("end_lineno", 0)
                                    end_col = val.get("end_col_offset", 0)
                                    source_file = val.get("_source_file", "")
                                    val.clear()
                                    val["_type"] = "Name"
                                    val["id"] = final_name
                                    val["ctx"] = {"_type": "Load"}
                                    val["lineno"] = lineno
                                    val["col_offset"] = col
                                    val["end_lineno"] = end_lineno
                                    val["end_col_offset"] = end_col
                                    if source_file != "":
                                        val["_source_file"] = source_file
                                elif target_name_map is not None:
                                    lineno = val.get("lineno", 0)
                                    col = val.get("col_offset", 0)
                                    errors.append(
                                        str(lineno)
                                        + ":"
                                        + str(col)
                                        + ": '"
                                        + mod_name
                                        + "."
                                        + attr
                                        + "' does not exist in "
                                        + target_file
                                    )
                                else:
                                    work.append(val)
                            else:
                                work.append(val)
                        else:
                            work.append(val)
                    else:
                        work.append(val)
                elif isinstance(val, list):
                    work.append(val)
                ki += 1
        elif isinstance(item, list):
            li = 0
            while li < len(item):
                child = item[li]
                if isinstance(child, dict) or isinstance(child, list):
                    work.append(child)
                li += 1
        wi += 1
    return errors


def _tag_source_file(node: dict[str, object], source_file: str) -> None:
    """Tag all dict nodes in the subtree with _source_file."""
    work: list[object] = [node]
    wi = 0
    while wi < len(work):
        item = work[wi]
        if isinstance(item, dict):
            if "_type" in item:
                item["_source_file"] = source_file
            keys = list(item.keys())
            ki = 0
            while ki < len(keys):
                val = item[keys[ki]]
                if isinstance(val, dict) or isinstance(val, list):
                    work.append(val)
                ki += 1
        elif isinstance(item, list):
            li = 0
            while li < len(item):
                child = item[li]
                if isinstance(child, dict) or isinstance(child, list):
                    work.append(child)
                li += 1
        wi += 1


def _stdlib_import_seen(
    names_list: object,
    stdlib_seen: set[str],
) -> bool:
    """Check if all bound names in an import are already seen. Adds unseen to set.

    Returns True if every name was already in stdlib_seen (skip the import).
    When only some names are new, removes already-seen aliases from the list.
    """
    if not isinstance(names_list, list):
        return False
    new_indices: list[int] = []
    ni = 0
    while ni < len(names_list):
        alias = names_list[ni]
        if isinstance(alias, dict):
            name = alias.get("name", "")
            asname = alias.get("asname")
            bound = asname if isinstance(asname, str) and asname != "" else name
            if isinstance(bound, str) and bound != "":
                if bound not in stdlib_seen:
                    new_indices.append(ni)
                    stdlib_seen.add(bound)
        ni += 1
    if len(new_indices) == 0:
        return True
    if len(new_indices) < len(names_list):
        # Partial overlap: keep only new aliases
        kept: list[object] = []
        ki = 0
        while ki < len(new_indices):
            kept.append(names_list[new_indices[ki]])
            ki += 1
        names_list.clear()
        ki = 0
        while ki < len(kept):
            names_list.append(kept[ki])
            ki += 1
    return False


def merge_project(
    file_asts: list[tuple[str, dict[str, object]]],
) -> tuple[dict[str, object] | None, list[str]]:
    """Full project merge. Returns (merged_ast, errors)."""
    errors: list[str] = []
    # Build universe
    universe: set[str] = set()
    i = 0
    while i < len(file_asts):
        universe.add(file_asts[i][0])
        i += 1
    # Classify imports and resolve project imports
    deps: dict[str, list[str]] = {}
    file_import_info: dict[
        str, list[tuple[dict[str, object], str, list[tuple[str, str]]]]
    ] = {}
    i = 0
    while i < len(file_asts):
        path = file_asts[i][0]
        ast_dict = file_asts[i][1]
        body = ast_dict.get("body", [])
        if not isinstance(body, list):
            i += 1
            continue
        file_deps: list[str] = []
        import_entries: list[tuple[dict[str, object], str, list[tuple[str, str]]]] = []
        j = 0
        while j < len(body):
            stmt = body[j]
            if isinstance(stmt, dict) and stmt.get("_type") == "ImportFrom":
                classification = _classify_import(stmt)
                if classification == "project":
                    module = stmt.get("module", "")
                    if module is None:
                        module = ""
                    if not isinstance(module, str):
                        module = ""
                    level = stmt.get("level", 0)
                    if not isinstance(level, int):
                        level = 0
                    names_list = stmt.get("names", [])
                    if not isinstance(names_list, list):
                        names_list = []
                    resolved = _resolve_project_import(
                        path, module, level, names_list, universe
                    )
                    import_entries.append((stmt, module, resolved))
                    k = 0
                    while k < len(resolved):
                        rpath, rerr = resolved[k]
                        if rpath != "":
                            found = False
                            m = 0
                            while m < len(file_deps):
                                if file_deps[m] == rpath:
                                    found = True
                                    break
                                m += 1
                            if not found:
                                file_deps.append(rpath)
                        elif rerr != "":
                            errors.append(rerr)
                        k += 1
            j += 1
        deps[path] = file_deps
        file_import_info[path] = import_entries
        i += 1
    if len(errors) > 0:
        return (None, errors)
    # Collect module names per file (with AST nodes)
    all_file_names: dict[str, list[tuple[str, int, int, dict[str, object]]]] = {}
    i = 0
    while i < len(file_asts):
        path = file_asts[i][0]
        ast_dict = file_asts[i][1]
        all_file_names[path] = _collect_module_names(ast_dict)
        i += 1
    # Compute module stems and plan collision resolution
    file_list: list[str] = []
    i = 0
    while i < len(file_asts):
        file_list.append(file_asts[i][0])
        i += 1
    stems = _compute_module_stems(file_list)
    dedup_names, file_renames = _plan_collision_resolution(all_file_names, stems)
    # Build file_name_map for _rewrite_module_attrs
    file_name_map: dict[str, dict[str, str]] = {}
    fkeys = list(all_file_names.keys())
    i = 0
    while i < len(fkeys):
        f = fkeys[i]
        names = all_file_names[f]
        name_map: dict[str, str] = {}
        j = 0
        while j < len(names):
            original = names[j][0]
            f_renames = file_renames.get(f, {})
            if original in f_renames:
                name_map[original] = f_renames[original]
            else:
                name_map[original] = original
            j += 1
        file_name_map[f] = name_map
        i += 1
    # Dependency order
    ordered = _dependency_order(file_list, deps)
    # Merge
    merged_body: list[dict[str, object]] = []
    dedup_seen: set[str] = set()
    stdlib_seen: set[str] = set()
    oi = 0
    while oi < len(ordered):
        path = ordered[oi]
        ast_dict: dict[str, object] | None = None
        ai = 0
        while ai < len(file_asts):
            if file_asts[ai][0] == path:
                ast_dict = file_asts[ai][1]
                break
            ai += 1
        if ast_dict is None:
            oi += 1
            continue
        body = ast_dict.get("body", [])
        if not isinstance(body, list):
            oi += 1
            continue
        # Build import binding maps for this file
        rename_map: dict[str, str] = {}
        module_bindings: dict[str, str] = {}
        import_entries = file_import_info.get(path, [])
        ei = 0
        while ei < len(import_entries):
            stmt, module, resolved = import_entries[ei]
            names_list = stmt.get("names", [])
            if not isinstance(names_list, list):
                names_list = []
            level = stmt.get("level", 0)
            if not isinstance(level, int):
                level = 0
            if module == "" and level > 0:
                # from . import X — each name is a module
                ni = 0
                while ni < len(names_list):
                    alias = names_list[ni]
                    if isinstance(alias, dict):
                        name = alias.get("name", "")
                        asname = alias.get("asname")
                        if isinstance(name, str) and name != "" and name != "*":
                            bound = (
                                asname
                                if isinstance(asname, str) and asname != ""
                                else name
                            )
                            if ni < len(resolved):
                                rpath = resolved[ni][0]
                                if rpath != "":
                                    module_bindings[bound] = rpath
                    ni += 1
            else:
                source_file = ""
                if len(resolved) > 0:
                    source_file = resolved[0][0]
                source_renames = file_renames.get(source_file, {})
                ni = 0
                while ni < len(names_list):
                    alias = names_list[ni]
                    if isinstance(alias, dict):
                        name = alias.get("name", "")
                        asname = alias.get("asname")
                        if isinstance(name, str) and name != "" and name != "*":
                            bound = (
                                asname
                                if isinstance(asname, str) and asname != ""
                                else name
                            )
                            if name in source_renames:
                                rename_map[bound] = source_renames[name]
                            elif bound != name:
                                rename_map[bound] = name
                    ni += 1
            ei += 1
        # If __init__.py defines names matching module bindings,
        # bare references should resolve to the init definitions
        if len(module_bindings) > 0:
            slash_idx = path.rfind("/")
            if slash_idx >= 0:
                init_path = path[:slash_idx] + "/__init__.py"
                init_names = file_name_map.get(init_path)
                if init_names is not None:
                    new_mb: dict[str, str] = {}
                    mb_keys = list(module_bindings.keys())
                    mbi = 0
                    while mbi < len(mb_keys):
                        bound = mb_keys[mbi]
                        if bound in init_names:
                            rename_map[bound] = init_names[bound]
                        else:
                            new_mb[bound] = module_bindings[bound]
                        mbi += 1
                    module_bindings = new_mb
        # Add own collision renames (override import renames for local defs)
        own_renames = file_renames.get(path, {})
        okeys = list(own_renames.keys())
        oki = 0
        while oki < len(okeys):
            rename_map[okeys[oki]] = own_renames[okeys[oki]]
            oki += 1
        # Apply rewrites
        if len(rename_map) > 0:
            _rewrite_names(ast_dict, rename_map)
        if len(module_bindings) > 0:
            rewrite_errors = _rewrite_module_attrs(
                ast_dict, module_bindings, file_name_map
            )
            ri = 0
            while ri < len(rewrite_errors):
                errors.append(path + ":" + rewrite_errors[ri])
                ri += 1
        # Mark project import stmts for removal
        ei = 0
        while ei < len(import_entries):
            import_entries[ei][0]["_remove"] = True
            ei += 1
        # Mark dedup duplicate definitions (keep first occurrence)
        bi = 0
        while bi < len(body):
            bstmt = body[bi]
            if isinstance(bstmt, dict):
                stype = bstmt.get("_type", "")
                def_name = ""
                if stype == "ClassDef" or stype == "FunctionDef":
                    n = bstmt.get("name", "")
                    if isinstance(n, str):
                        def_name = n
                elif stype == "Assign":
                    targets = bstmt.get("targets", [])
                    if isinstance(targets, list) and len(targets) > 0:
                        t = targets[0]
                        if isinstance(t, dict) and t.get("_type") == "Name":
                            n = t.get("id", "")
                            if isinstance(n, str):
                                def_name = n
                elif stype == "AnnAssign":
                    target = bstmt.get("target", {})
                    if isinstance(target, dict) and target.get("_type") == "Name":
                        n = target.get("id", "")
                        if isinstance(n, str):
                            def_name = n
                if def_name != "" and def_name in dedup_names:
                    if def_name in dedup_seen:
                        bstmt["_remove"] = True
                    else:
                        dedup_seen.add(def_name)
            bi += 1
        # Filter body: remove project imports, dedup dups, dedup stdlib imports
        new_body: list[dict[str, object]] = []
        bi = 0
        while bi < len(body):
            bstmt = body[bi]
            if isinstance(bstmt, dict) and bstmt.get("_remove") is True:
                bi += 1
                continue
            # Deduplicate stdlib imports
            if isinstance(bstmt, dict):
                skip_stdlib = False
                btype = bstmt.get("_type", "")
                if btype == "ImportFrom" and _classify_import(bstmt) == "stdlib":
                    skip_stdlib = _stdlib_import_seen(
                        bstmt.get("names", []), stdlib_seen
                    )
                elif btype == "Import":
                    skip_stdlib = _stdlib_import_seen(
                        bstmt.get("names", []), stdlib_seen
                    )
                if skip_stdlib:
                    bi += 1
                    continue
            if isinstance(bstmt, dict):
                _tag_source_file(bstmt, path)
            new_body.append(bstmt)
            bi += 1
        mi = 0
        while mi < len(new_body):
            merged_body.append(new_body[mi])
            mi += 1
        oi += 1
    if len(errors) > 0:
        return (None, errors)
    return ({"_type": "Module", "body": merged_body}, [])


def _pipeline_post_parse(
    ast_dict: dict[str, object],
    source: str,
    target: str,
    stop_at: str | None,
    strict_math: bool,
    strict_tostring: bool,
) -> tuple[int, str]:
    """Run pipeline phases after parsing. Returns (exit_code, output)."""
    # Phase 3: Subset
    result = verify_subset(ast_dict)
    errors = result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return (1, "")
    if stop_at == "subset":
        return (0, "")
    # Phase 4: Names
    name_result = resolve_names(ast_dict)
    errors = name_result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return (1, "")
    if stop_at == "names":
        return (0, to_json(_name_table_to_dict(name_result.table)))
    # Phase 5: Signatures
    known_classes: set[str] = set()
    node_classes: set[str] = set()
    mkeys = list(name_result.table.module_names.keys())
    ki = 0
    while ki < len(mkeys):
        mname = mkeys[ki]
        info = name_result.table.module_names[mname]
        if info.kind == "class":
            known_classes.add(mname)
            bi = 0
            while bi < len(info.bases):
                base = info.bases[bi]
                if base == "Node" or base.endswith("Node"):
                    node_classes.add(mname)
                bi += 1
        ki += 1
    # Collect type alias definitions from AST
    type_aliases: dict[str, str] = {}
    ta_body = ast_dict.get("body", [])
    if isinstance(ta_body, list):
        tai = 0
        while tai < len(ta_body):
            ta_stmt = ta_body[tai]
            if isinstance(ta_stmt, dict) and ta_stmt.get("_type") == "Assign":
                ta_targets = ta_stmt.get("targets", [])
                if isinstance(ta_targets, list) and len(ta_targets) == 1:
                    ta_target = ta_targets[0]
                    if isinstance(ta_target, dict) and ta_target.get("_type") == "Name":
                        ta_name = ta_target.get("id", "")
                        if isinstance(ta_name, str):
                            ta_info = name_result.table.module_names.get(ta_name)
                            if ta_info is not None and ta_info.kind == "type_alias":
                                ta_value = ta_stmt.get("value", {})
                                ta_str = annotation_to_str(ta_value)
                                if ta_str != "":
                                    type_aliases[ta_name] = ta_str
            tai += 1
    sig_result = collect_signatures(ast_dict, known_classes, node_classes, type_aliases)
    errors = sig_result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return (1, "")
    if stop_at == "signatures":
        return (0, to_json(sig_result.to_dict()))
    # Phase 6: Fields
    hierarchy_roots: set[str] = set()
    base_counts: dict[str, int] = {}
    parent_of: dict[str, str] = {}
    ki = 0
    mkeys2 = list(name_result.table.module_names.keys())
    while ki < len(mkeys2):
        mname = mkeys2[ki]
        info = name_result.table.module_names[mname]
        if info.kind == "class":
            bi = 0
            while bi < len(info.bases):
                base = info.bases[bi]
                if base not in base_counts:
                    base_counts[base] = 0
                base_counts[base] = base_counts[base] + 1
                parent_of[mname] = base
                bi += 1
        ki += 1
    bkeys = list(base_counts.keys())
    ki = 0
    while ki < len(bkeys):
        bname = bkeys[ki]
        if bname not in parent_of:
            hierarchy_roots.add(bname)
        ki += 1
    field_result = collect_fields(
        ast_dict, known_classes, node_classes, hierarchy_roots, sig_result
    )
    errors = field_result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return (1, "")
    if stop_at == "fields":
        return (0, to_json(field_result.to_dict()))
    # Phase 7: Hierarchy
    class_bases: dict[str, list[str]] = {}
    ki = 0
    while ki < len(mkeys2):
        mname = mkeys2[ki]
        info = name_result.table.module_names[mname]
        if info.kind == "class":
            class_bases[mname] = list(info.bases)
        ki += 1
    hier_result = build_hierarchy(known_classes, class_bases)
    errors = hier_result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return (1, "")
    if stop_at == "hierarchy":
        return (0, to_json(hier_result.to_dict()))
    # Phase 8: Inference
    inf_result = run_inference(
        ast_dict, sig_result, field_result, hier_result, known_classes, class_bases
    )
    errors = inf_result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return (1, "")
    if stop_at == "inference":
        return (0, to_json(ast_dict))
    # Phase 9: Lowering
    module, lower_errors = lower(
        ast_dict,
        sig_result,
        field_result,
        hier_result,
        known_classes,
        class_bases,
        source,
    )
    if len(lower_errors) > 0:
        _print_errors(lower_errors)
        return (1, "")
    if module is None:
        print("error: lowering produced no module", file=sys.stderr)
        return (1, "")
    if strict_math:
        module.strict_math = True
    if strict_tostring:
        module.strict_tostring = True
    if stop_at == "lowering":
        return (0, to_json(module_to_dict(module)))
    # Phase 10: Type check
    checker = Checker()
    checker.collect_declarations(module)
    if len(checker.errors) > 0:
        _print_errors(checker.errors)
        return (1, "")
    checker.check_bodies(module)
    if len(checker.errors) > 0:
        _print_errors(checker.errors)
        return (1, "")
    # Phases 11-16: Middleend
    analyze_returns(module, checker)
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    if stop_at == "analyze":
        return (0, to_json(module_to_dict(module)))
    # Phase 17: Backend
    emitters: dict[str, object] = {
        "python": emit_python,
        "perl": emit_perl,
        "ruby": emit_ruby,
    }
    if target not in emitters:
        print(
            "error: backend not yet implemented for '" + target + "'", file=sys.stderr
        )
        return (1, "")
    emitter = emitters[target]
    return (0, emitter(module))


def run_pipeline(
    source: str,
    target: str,
    stop_at: str | None,
    strict_math: bool,
    strict_tostring: bool,
) -> tuple[int, str]:
    """Run the transpilation pipeline. Returns (exit_code, output)."""
    source, pragma_math, pragma_tostring = _extract_pragmas(source)
    if pragma_math:
        strict_math = True
    if pragma_tostring:
        strict_tostring = True
    if stop_at == "subset" and should_skip_file(source):
        return (0, "")
    # Phase 2: Parse
    try:
        ast_dict = parse(source)
    except ParseError as e:
        print(
            "error:" + str(e.lineno) + ":" + str(e.col) + ": " + e.msg, file=sys.stderr
        )
        return (1, "")
    if stop_at == "parse":
        return (0, to_json(ast_dict))
    return _pipeline_post_parse(
        ast_dict, source, target, stop_at, strict_math, strict_tostring
    )


def parse_args() -> tuple[str, str | None, bool, bool, bool, str | None, str | None]:
    """Parse command-line arguments. Returns (target, stop_at, strict_math, strict_tostring, project, input_file, output_file)."""
    args = sys.argv[1:]
    target: str | None = None
    stop_at: str | None = None
    strict_math = False
    strict_tostring = False
    project = False
    input_file: str | None = None
    output_file: str | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--help" or arg == "-h":
            print(USAGE, end="")
            sys.exit(0)
        elif arg == "--target":
            if i + 1 >= len(args):
                print("error: --target requires an argument", file=sys.stderr)
                sys.exit(2)
            target = args[i + 1]
            i += 2
        elif arg == "--stop-at":
            if i + 1 >= len(args):
                print("error: --stop-at requires an argument", file=sys.stderr)
                sys.exit(2)
            stop_at = args[i + 1]
            i += 2
        elif arg == "-o" or arg == "--output":
            if i + 1 >= len(args):
                print("error: " + arg + " requires an argument", file=sys.stderr)
                sys.exit(2)
            output_file = args[i + 1]
            i += 2
        elif arg == "--strict":
            strict_math = True
            strict_tostring = True
            i += 1
        elif arg == "--strict-math":
            strict_math = True
            i += 1
        elif arg == "--strict-tostring":
            strict_tostring = True
            i += 1
        elif arg == "--project":
            project = True
            i += 1
        elif arg.startswith("-"):
            print("error: unknown flag '" + arg + "'", file=sys.stderr)
            sys.exit(2)
        else:
            if input_file is not None:
                print("error: unexpected argument '" + arg + "'", file=sys.stderr)
                sys.exit(2)
            input_file = arg
            i += 1
    if stop_at is not None and stop_at not in PHASES:
        print("error: unknown phase '" + stop_at + "'", file=sys.stderr)
        sys.exit(2)
    if target is None:
        if stop_at is not None:
            target = "python"
        else:
            print("error: --target is required", file=sys.stderr)
            sys.exit(2)
    target_str: str = target if target is not None else ""
    if target_str not in TARGETS:
        print("error: unknown target '" + target_str + "'", file=sys.stderr)
        sys.exit(2)
    return (
        target_str,
        stop_at,
        strict_math,
        strict_tostring,
        project,
        input_file,
        output_file,
    )


def _parse_project_input(data: str) -> list[tuple[str, str]]:
    """Parse NUL-delimited path\\0source\\0 pairs."""
    parts = data.split("\0")
    result: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(parts):
        result.append((parts[i], parts[i + 1]))
        i += 2
    return result


def main() -> int:
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "taytsh":
        from .taytsh.cli import main as taytsh_main

        return taytsh_main(sys.argv[2:])
    target, stop_at, strict_math, strict_tostring, project, input_file, output_file = (
        parse_args()
    )
    if project:
        source, err = read_source(input_file)
        if err != 0:
            return err
        if len(source) == 0:
            print("error: no input provided", file=sys.stderr)
            return 2
        files = _parse_project_input(source)
        if len(files) == 0:
            print("error: no .py files found in directory", file=sys.stderr)
            return 1
        return main_project(
            files, target, stop_at, strict_math, strict_tostring, output_file
        )
    source, err = read_source(input_file)
    if err != 0:
        return err
    if len(source) == 0:
        print("error: no input provided", file=sys.stderr)
        return 2
    exit_code, output = run_pipeline(
        source, target, stop_at, strict_math, strict_tostring
    )
    if exit_code != 0:
        return exit_code
    if len(output) > 0:
        return write_output(output, output_file)
    return 0


def main_project(
    files: list[tuple[str, str]],
    target: str,
    stop_at: str | None,
    strict_math: bool,
    strict_tostring: bool,
    output_file: str | None,
) -> int:
    """Project-mode entry point. files is [(relpath, source)]."""
    # Parse all files
    file_asts: list[tuple[str, dict[str, object]]] = []
    i = 0
    while i < len(files):
        path = files[i][0]
        source = files[i][1]
        source, pragma_math, pragma_tostring = _extract_pragmas(source)
        if pragma_math:
            strict_math = True
        if pragma_tostring:
            strict_tostring = True
        try:
            ast_dict = parse(source)
        except ParseError as e:
            print(
                path + ":" + str(e.lineno) + ":" + str(e.col) + ": " + e.msg,
                file=sys.stderr,
            )
            return 1
        file_asts.append((path, ast_dict))
        i += 1
    if stop_at == "parse":
        items: list[dict[str, object]] = []
        j = 0
        while j < len(file_asts):
            items.append({"path": file_asts[j][0], "ast": file_asts[j][1]})
            j += 1
        output = to_json(items)
        return write_output(output, output_file)
    # Merge
    merged_ast, merge_errors = merge_project(file_asts)
    if len(merge_errors) > 0:
        _print_errors(merge_errors)
        return 1
    if merged_ast is None:
        print("error: merge produced no AST", file=sys.stderr)
        return 1
    # Collect all sources for lowering (concatenated)
    all_source_parts: list[str] = []
    k = 0
    while k < len(files):
        all_source_parts.append(files[k][1])
        k += 1
    combined_source = "\n".join(all_source_parts)
    exit_code, output = _pipeline_post_parse(
        merged_ast, combined_source, target, stop_at, strict_math, strict_tostring
    )
    if exit_code != 0:
        return exit_code
    if len(output) > 0:
        return write_output(output, output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
