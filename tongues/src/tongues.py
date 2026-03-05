"""Subset-compliant entry point."""

from __future__ import annotations

import sys

from .frontend.parse import parse, ParseError, stamp_uids
from .frontend.bind import (
    run_bind,
    NameInfo,
    NameTable,
    IMPORT_ONLY_MODULES,
    ALLOWED_FROM_MODULES,
)
from .frontend.typecollect import collect_signatures, collect_types
from .frontend.hierarchy import build_hierarchy
from .frontend.pycheck import run_pycheck
from .frontend.lowering import lower
from .frontend.types import (
    JsonValue,
    JStr,
    JInt,
    JFloat,
    JBool,
    JNull,
    JList,
    JDict,
    ASTNode,
    get_str,
    get_int,
    get_bool,
    get_node,
    get_nodes,
    has_key,
)
from .taytsh.ast import (
    TLetStmt,
    serialize_annotations,
    to_dict as module_to_dict,
)
from .taytsh.check import Checker, check_with_info
from .taytsh.emit import to_source
from .taytsh.parse import Parser as TaytshParser
from .taytsh.tokens import tokenize as taytsh_tokenize
from .middleend.callgraph import analyze_callgraph
from .middleend.callgraph_serial import serialize_callgraph
from .middleend.hoisting import analyze_hoisting
from .middleend.liveness import analyze_liveness
from .middleend.ownership import analyze_ownership
from .middleend.returns import analyze_returns
from .middleend.scope import analyze_scope
from .middleend.strings import analyze_strings
from .backend.python import emit_python
from .backend.perl import emit_perl
from .backend.ruby import emit_ruby
from .backend.taytsh import emit_taytsh

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
    "taytsh",
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
    "pycheck",
    "lowering",
    "lowering-text",
    "analyze",
]

USAGE: str = """\
tongues [OPTIONS] [INPUT] [-o OUTPUT]

Options:
  --target TARGET     Output language: c, csharp, dart, go, java, javascript,
                      lua, perl, php, python, ruby, rust, swift, typescript, zig
  --stop-at PHASE     Stop after phase: parse, subset, names, signatures,
                      fields, hierarchy, pycheck, lowering, lowering-text,
                      analyze
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
    for line in lines[:5]:
        if "tongues: skip" in line:
            return True
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
        source = ""
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
    print(output, end="")
    return 0


# --- JSON serialization (subset-compliant, no json module) ---


def _json_escape(s: str) -> str:
    """Escape a string for JSON output."""
    result: list[str] = []
    for c in s:
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
    return "".join(result)


def _to_json(obj: JsonValue, indent: int, level: int) -> str:
    """Recursively serialize a JsonValue to JSON string."""
    if isinstance(obj, JNull):
        return "null"
    if isinstance(obj, JBool):
        if obj.value:
            return "true"
        return "false"
    if isinstance(obj, JInt):
        return str(obj.value)
    if isinstance(obj, JFloat):
        return str(obj.value)
    if isinstance(obj, JStr):
        return '"' + _json_escape(obj.value) + '"'
    if isinstance(obj, JList):
        if len(obj.items) == 0:
            return "[]"
        parts: list[str] = []
        pad = " " * (indent * (level + 1))
        pad_close = " " * (indent * level)
        for item in obj.items:
            parts.append(pad + _to_json(item, indent, level + 1))
        return "[\n" + ",\n".join(parts) + "\n" + pad_close + "]"
    if isinstance(obj, JDict):
        if len(obj.entries) == 0:
            return "{}"
        parts: list[str] = []
        pad = " " * (indent * (level + 1))
        pad_close = " " * (indent * level)
        for k, v in obj.entries.items():
            key_str = '"' + _json_escape(str(k)) + '"'
            val_str = _to_json(v, indent, level + 1)
            parts.append(pad + key_str + ": " + val_str)
        return "{\n" + ",\n".join(parts) + "\n" + pad_close + "}"
    return '"<unserializable>"'


def to_json(obj: JsonValue) -> str:
    """Serialize JsonValue to pretty-printed JSON."""
    return _to_json(obj, 2, 0)


# --- Name table serialization ---


def _name_info_to_dict(info: NameInfo) -> JsonValue:
    """Convert a NameInfo to a JsonValue dict."""
    d: dict[str, JsonValue] = {
        "kind": JStr(info.kind),
        "scope": JStr(info.scope),
        "lineno": JInt(info.lineno),
        "col": JInt(info.col),
    }
    if info.decl_class != "":
        d["decl_class"] = JStr(info.decl_class)
    if info.decl_func != "":
        d["decl_func"] = JStr(info.decl_func)
    if len(info.bases) > 0:
        bases_jv: list[JsonValue] = []
        for base in info.bases:
            bases_jv.append(JStr(base))
        d["bases"] = JList(bases_jv)
    return JDict(d)


def _name_table_to_dict(table: NameTable) -> JsonValue:
    """Convert a NameTable to spec-compliant format: {"names": {...}, "scopes": [...]}."""
    names: dict[str, JsonValue] = {}
    for name, info in table.module_names.items():
        names[name] = _name_info_to_dict(info)
    scopes: list[JsonValue] = []
    for cname, cmap in table.class_names.items():
        scope_names: dict[str, JsonValue] = {}
        for mname, minfo in cmap.items():
            scope_names[mname] = _name_info_to_dict(minfo)
        scopes.append(JDict({"scope": JStr(cname), "names": JDict(scope_names)}))
    for lkey, lmap in table.local_names.items():
        if str(lkey[0]) != "":
            scope_key = str(lkey[0]) + ":" + str(lkey[1])
        else:
            scope_key = str(lkey[1])
        scope_names: dict[str, JsonValue] = {}
        for sname, sinfo in lmap.items():
            scope_names[sname] = _name_info_to_dict(sinfo)
        scopes.append(JDict({"scope": JStr(scope_key), "names": JDict(scope_names)}))
    result: dict[str, JsonValue] = {"names": JDict(names)}
    if len(scopes) > 0:
        result["scopes"] = JList(scopes)
    return JDict(result)


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
        pragma_body = stripped[3:]
        if not pragma_body.endswith("]"):
            break
        pragma_body = pragma_body[:-1]
        for entry_raw in pragma_body.split(","):
            entry = entry_raw.strip().strip('"')
            if entry == "strict_math":
                strict_math = True
            elif entry == "strict_tostring":
                strict_tostring = True
        i += 1
    remaining = "\n".join(lines[i:])
    return (remaining, strict_math, strict_tostring)


# --- Error reporting ---


def _print_errors(errors: list[str]) -> None:
    """Print error strings to stderr."""
    for error in errors:
        print(error, file=sys.stderr)


# --- Pipeline ---


# --- Project merge (Phase 3a) ---


def _classify_import(node: ASTNode) -> str:
    """Classify an ImportFrom node as 'stdlib' or 'project'."""
    level = get_int(node, "level")
    if level > 0:
        return "project"
    module = get_str(node, "module")
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
    names: list[ASTNode],
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
            for name_node in names:
                name = get_str(name_node, "name")
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
    for f in files:
        in_degree[f] = 0
    for f in files:
        dep_list = deps.get(f)
        if dep_list is not None:
            for dep in dep_list:
                if dep in in_degree:
                    in_degree[dep] = in_degree[dep] + 1
    ready: list[str] = []
    for f in files:
        if in_degree[f] == 0:
            ready.append(f)
    ready.sort()
    result: list[str] = []
    while len(ready) > 0:
        node = ready[0]
        ready = ready[1:]
        result.append(node)
        dep_list = deps.get(node)
        if dep_list is not None:
            for dep in dep_list:
                if dep in in_degree:
                    in_degree[dep] = in_degree[dep] - 1
                    if in_degree[dep] == 0:
                        k = 0
                        inserted = False
                        while k < len(ready):
                            if dep < ready[k]:
                                ready.insert(k, dep)
                                inserted = True
                                break
                            k += 1
                        if not inserted:
                            ready.append(dep)
    if len(result) < len(files):
        remaining: list[str] = []
        for f in files:
            if f not in result:
                remaining.append(f)
        remaining.sort()
        for f in remaining:
            result.append(f)
    return result


def _collect_module_names(
    ast_dict: ASTNode,
) -> list[tuple[str, int, int, ASTNode]]:
    """Collect (name, lineno, col, stmt) for module-level definitions."""
    result: list[tuple[str, int, int, ASTNode]] = []
    body = get_nodes(ast_dict, "body")
    for stmt in body:
        node_type = get_str(stmt, "_type")
        lineno = get_int(stmt, "lineno")
        col = get_int(stmt, "col_offset")
        if node_type == "ClassDef":
            name = get_str(stmt, "name")
            if name != "":
                result.append((name, lineno, col, stmt))
        elif node_type == "FunctionDef":
            name = get_str(stmt, "name")
            if name != "":
                result.append((name, lineno, col, stmt))
        elif node_type == "Assign":
            for assign_tgt in get_nodes(stmt, "targets"):
                if get_str(assign_tgt, "_type") == "Name":
                    tid = get_str(assign_tgt, "id")
                    if tid != "":
                        result.append((tid, lineno, col, stmt))
        elif node_type == "TypeAlias":
            name_node = get_node(stmt, "name")
            tid = get_str(name_node, "id")
            if tid != "":
                result.append((tid, lineno, col, stmt))
        elif node_type == "AnnAssign":
            tgt = get_node(stmt, "target")
            if get_str(tgt, "_type") == "Name":
                tid = get_str(tgt, "id")
                if tid != "":
                    result.append((tid, lineno, col, stmt))
    return result


def _ast_equal(a: ASTNode, b: ASTNode) -> bool:
    """Deep structural comparison of AST nodes, ignoring position metadata."""
    ignore: set[str] = {
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
        "_source_file",
        "_remove",
    }
    work_a: list[JsonValue] = [JDict(a)]
    work_b: list[JsonValue] = [JDict(b)]
    wi = 0
    while wi < len(work_a):
        xa = work_a[wi]
        ya = work_b[wi]
        matched = _ast_equal_step(xa, ya, ignore, work_a, work_b)
        if not matched:
            return False
        wi += 1
    return True


def _ast_equal_step(
    xa: JsonValue,
    ya: JsonValue,
    ignore: set[str],
    work_a: list[JsonValue],
    work_b: list[JsonValue],
) -> bool:
    """Compare one pair of values in _ast_equal. Returns False if mismatch."""
    if isinstance(xa, JDict) and isinstance(ya, JDict):
        return _ast_equal_dicts(xa.entries, ya.entries, ignore, work_a, work_b)
    if isinstance(xa, JList) and isinstance(ya, JList):
        xl = xa.items
        yl = ya.items
        if len(xl) != len(yl):
            return False
        li = 0
        while li < len(xl):
            work_a.append(xl[li])
            work_b.append(yl[li])
            li += 1
        return True
    if isinstance(xa, JStr) and isinstance(ya, JStr):
        return xa.value == ya.value
    if isinstance(xa, JInt) and isinstance(ya, JInt):
        return xa.value == ya.value
    if isinstance(xa, JBool) and isinstance(ya, JBool):
        return xa.value == ya.value
    if isinstance(xa, JFloat) and isinstance(ya, JFloat):
        return xa.value == ya.value
    if isinstance(xa, JNull) and isinstance(ya, JNull):
        return True
    return xa == ya


def _ast_equal_dicts(
    xd: dict[str, JsonValue],
    yd: dict[str, JsonValue],
    ignore: set[str],
    work_a: list[JsonValue],
    work_b: list[JsonValue],
) -> bool:
    """Compare two ASTNode dicts in _ast_equal. Returns False if mismatch."""
    x_keys: list[str] = []
    xk = list(xd.keys())
    ki = 0
    while ki < len(xk):
        if xk[ki] not in ignore:
            x_keys.append(xk[ki])
        ki += 1
    y_keys: list[str] = []
    yk = list(yd.keys())
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
        work_a.append(xd[x_keys[ki]])
        work_b.append(yd[y_keys[ki]])
        ki += 1
    return True


def _collect_definition_refs(node: ASTNode) -> set[str]:
    """Collect all Name.id values referenced in a definition's body."""
    refs: set[str] = set()
    node_type = get_str(node, "_type")
    seeds: list[JsonValue] = []
    if node_type == "FunctionDef":
        body_val = node.get("body")
        if isinstance(body_val, JList):
            seeds.append(body_val)
        args_val = node.get("args")
        if isinstance(args_val, JDict):
            seeds.append(args_val)
        returns_val = node.get("returns")
        if returns_val is not None:
            if not isinstance(returns_val, JNull):
                seeds.append(returns_val)
        deco_val = node.get("decorator_list")
        if isinstance(deco_val, JList):
            seeds.append(deco_val)
    elif node_type == "ClassDef":
        body_val = node.get("body")
        if isinstance(body_val, JList):
            seeds.append(body_val)
        bases_val = node.get("bases")
        if isinstance(bases_val, JList):
            seeds.append(bases_val)
        deco_val = node.get("decorator_list")
        if isinstance(deco_val, JList):
            seeds.append(deco_val)
    elif node_type == "Assign":
        value_val = node.get("value")
        if value_val is not None:
            if not isinstance(value_val, JNull):
                seeds.append(value_val)
    elif node_type == "AnnAssign":
        value_val = node.get("value")
        if value_val is not None:
            if not isinstance(value_val, JNull):
                seeds.append(value_val)
        ann_val = node.get("annotation")
        if ann_val is not None:
            if not isinstance(ann_val, JNull):
                seeds.append(ann_val)
    work: list[JsonValue] = list(seeds)
    wi = 0
    while wi < len(work):
        item = work[wi]
        if isinstance(item, JDict):
            node_entries = item.entries
            if get_str(node_entries, "_type") == "Name":
                nid = get_str(node_entries, "id")
                if nid != "":
                    refs.add(nid)
            keys = list(node_entries.keys())
            ki = 0
            while ki < len(keys):
                val = node_entries[keys[ki]]
                if isinstance(val, JDict) or isinstance(val, JList):
                    work.append(val)
                ki += 1
        elif isinstance(item, JList):
            li = 0
            while li < len(item.items):
                child = item.items[li]
                if isinstance(child, JDict) or isinstance(child, JList):
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
    file_names: dict[str, list[tuple[str, int, int, ASTNode]]],
    stems: dict[str, str],
) -> tuple[set[str], dict[str, dict[str, str]]]:
    """Plan collision resolution. Returns (dedup_names, file_renames)."""
    name_to_defs: dict[str, list[tuple[str, ASTNode]]] = {}
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


def _rewrite_names(node: ASTNode, rename_map: dict[str, str]) -> None:
    """Recursively rename Name nodes and definition names per rename_map. In-place."""
    work: list[JsonValue] = [JDict(node)]
    wi = 0
    while wi < len(work):
        item = work[wi]
        if isinstance(item, JDict):
            node_entries = item.entries
            ntype = get_str(node_entries, "_type")
            if ntype == "Name":
                nid = get_str(node_entries, "id")
                if nid != "" and nid in rename_map:
                    node_entries["id"] = JStr(rename_map[nid])
            elif ntype == "FunctionDef" or ntype == "ClassDef":
                def_name = get_str(node_entries, "name")
                if def_name != "" and def_name in rename_map:
                    node_entries["name"] = JStr(rename_map[def_name])
            keys = list(node_entries.keys())
            ki = 0
            while ki < len(keys):
                val = node_entries[keys[ki]]
                if isinstance(val, JDict) or isinstance(val, JList):
                    work.append(val)
                ki += 1
        elif isinstance(item, JList):
            li = 0
            while li < len(item.items):
                child = item.items[li]
                if isinstance(child, JDict) or isinstance(child, JList):
                    work.append(child)
                li += 1
        wi += 1


def _rewrite_module_attrs(
    node: ASTNode,
    module_bindings: dict[str, str],
    file_name_map: dict[str, dict[str, str]],
) -> list[str]:
    """Rewrite module.attr Attribute nodes to plain Name nodes. Returns errors."""
    errors: list[str] = []
    work: list[JsonValue] = [JDict(node)]
    wi = 0
    while wi < len(work):
        item = work[wi]
        if isinstance(item, JDict):
            node_entries = item.entries
            keys = list(node_entries.keys())
            ki = 0
            while ki < len(keys):
                val = node_entries[keys[ki]]
                if isinstance(val, JDict):
                    val_entries = val.entries
                    if get_str(val_entries, "_type") == "Attribute":
                        value_node_v = val_entries.get("value")
                        value_node: ASTNode | None = None
                        if isinstance(value_node_v, JDict):
                            value_node = value_node_v.entries
                        if (
                            value_node is not None
                            and get_str(value_node, "_type") == "Name"
                        ):
                            mod_name = get_str(value_node, "id")
                            if mod_name != "" and mod_name in module_bindings:
                                target_file = module_bindings[mod_name]
                                attr = get_str(val_entries, "attr")
                                target_name_map = file_name_map.get(target_file)
                                if (
                                    target_name_map is not None
                                    and attr in target_name_map
                                ):
                                    final_name = target_name_map[attr]
                                    lineno = val_entries.get("lineno")
                                    col = val_entries.get("col_offset")
                                    end_lineno = val_entries.get("end_lineno")
                                    end_col = val_entries.get("end_col_offset")
                                    source_file = val_entries.get("_source_file")
                                    val_entries.clear()
                                    val_entries["_type"] = JStr("Name")
                                    val_entries["id"] = JStr(final_name)
                                    val_entries["ctx"] = JDict({"_type": JStr("Load")})
                                    if lineno is not None:
                                        val_entries["lineno"] = lineno
                                    if col is not None:
                                        val_entries["col_offset"] = col
                                    if end_lineno is not None:
                                        val_entries["end_lineno"] = end_lineno
                                    if end_col is not None:
                                        val_entries["end_col_offset"] = end_col
                                    if source_file is not None:
                                        if not isinstance(source_file, JNull):
                                            val_entries["_source_file"] = source_file
                                elif target_name_map is not None:
                                    lineno_i = get_int(val_entries, "lineno")
                                    col_i = get_int(val_entries, "col_offset")
                                    errors.append(
                                        str(lineno_i)
                                        + ":"
                                        + str(col_i)
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
                elif isinstance(val, JList):
                    work.append(val)
                ki += 1
        elif isinstance(item, JList):
            li = 0
            while li < len(item.items):
                child = item.items[li]
                if isinstance(child, JDict) or isinstance(child, JList):
                    work.append(child)
                li += 1
        wi += 1
    return errors


def _tag_source_file(node: ASTNode, source_file: str) -> None:
    """Tag all dict nodes in the subtree with _source_file."""
    work: list[JsonValue] = [JDict(node)]
    wi = 0
    while wi < len(work):
        item = work[wi]
        if isinstance(item, JDict):
            node_entries = item.entries
            if has_key(node_entries, "_type"):
                node_entries["_source_file"] = JStr(source_file)
            keys = list(node_entries.keys())
            ki = 0
            while ki < len(keys):
                val = node_entries[keys[ki]]
                if isinstance(val, JDict) or isinstance(val, JList):
                    work.append(val)
                ki += 1
        elif isinstance(item, JList):
            li = 0
            while li < len(item.items):
                child = item.items[li]
                if isinstance(child, JDict) or isinstance(child, JList):
                    work.append(child)
                li += 1
        wi += 1


def _stdlib_import_seen(
    names_list: list[JsonValue],
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
        alias_raw = names_list[ni]
        alias: ASTNode = {}
        if isinstance(alias_raw, JDict):
            alias = alias_raw.entries
        elif isinstance(alias_raw, dict):
            alias = alias_raw
        name = get_str(alias, "name")
        v = alias.get("asname")
        asname = ""
        if isinstance(v, JStr):
            asname = v.value
        bound = asname if asname != "" else name
        if bound != "":
            if bound not in stdlib_seen:
                new_indices.append(ni)
                stdlib_seen.add(bound)
        ni += 1
    if len(new_indices) == 0:
        return True
    if len(new_indices) < len(names_list):
        kept: list[JsonValue] = []
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
    file_asts: list[tuple[str, ASTNode]],
) -> tuple[ASTNode | None, list[str], dict[str, dict[str, str]]]:
    """Full project merge. Returns (merged_ast, errors)."""
    errors: list[str] = []
    universe: set[str] = set()
    i = 0
    while i < len(file_asts):
        universe.add(file_asts[i][0])
        i += 1
    deps: dict[str, list[str]] = {}
    file_import_info: dict[str, list[tuple[ASTNode, str, list[tuple[str, str]]]]] = {}
    i = 0
    while i < len(file_asts):
        path = file_asts[i][0]
        ast_dict = file_asts[i][1]
        ast_body = get_nodes(ast_dict, "body")
        file_deps: list[str] = []
        import_entries: list[tuple[ASTNode, str, list[tuple[str, str]]]] = []
        j = 0
        while j < len(ast_body):
            stmt = ast_body[j]
            if get_str(stmt, "_type") == "ImportFrom":
                classification = _classify_import(stmt)
                if classification == "project":
                    module = get_str(stmt, "module")
                    level = get_int(stmt, "level")
                    names_list = get_nodes(stmt, "names")
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
        return (None, errors, {})
    all_file_names: dict[str, list[tuple[str, int, int, ASTNode]]] = {}
    i = 0
    while i < len(file_asts):
        path = file_asts[i][0]
        ast_dict = file_asts[i][1]
        all_file_names[path] = _collect_module_names(ast_dict)
        i += 1
    file_list: list[str] = []
    i = 0
    while i < len(file_asts):
        file_list.append(file_asts[i][0])
        i += 1
    stems = _compute_module_stems(file_list)
    dedup_names, file_renames = _plan_collision_resolution(all_file_names, stems)
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
    ordered = _dependency_order(file_list, deps)
    merged_body: list[ASTNode] = []
    dedup_seen: set[str] = set()
    stdlib_seen: set[str] = set()
    oi = 0
    while oi < len(ordered):
        path = ordered[oi]
        found_ast: ASTNode | None = None
        ai = 0
        while ai < len(file_asts):
            if file_asts[ai][0] == path:
                found_ast = file_asts[ai][1]
                break
            ai += 1
        if found_ast is None:
            oi += 1
            continue
        ast_body = get_nodes(found_ast, "body")
        if len(ast_body) == 0:
            oi += 1
            continue
        rename_map: dict[str, str] = {}
        module_bindings: dict[str, str] = {}
        import_entries = file_import_info.get(path, [])
        ei = 0
        while ei < len(import_entries):
            stmt, module, resolved = import_entries[ei]
            names_list = get_nodes(stmt, "names")
            level = get_int(stmt, "level")
            if module == "" and level > 0:
                ni = 0
                while ni < len(names_list):
                    alias = names_list[ni]
                    name = get_str(alias, "name")
                    v = alias.get("asname")
                    asname = ""
                    if isinstance(v, JStr):
                        asname = v.value
                    if name != "" and name != "*":
                        bound = asname if asname != "" else name
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
                    name = get_str(alias, "name")
                    v = alias.get("asname")
                    asname = ""
                    if isinstance(v, JStr):
                        asname = v.value
                    if name != "" and name != "*":
                        bound = asname if asname != "" else name
                        if name in source_renames:
                            rename_map[bound] = source_renames[name]
                        elif bound != name:
                            rename_map[bound] = name
                    ni += 1
            ei += 1
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
        own_renames = file_renames.get(path, {})
        okeys = list(own_renames.keys())
        oki = 0
        while oki < len(okeys):
            rename_map[okeys[oki]] = own_renames[okeys[oki]]
            oki += 1
        if len(rename_map) > 0:
            _rewrite_names(found_ast, rename_map)
        if len(module_bindings) > 0:
            rewrite_errors = _rewrite_module_attrs(
                found_ast, module_bindings, file_name_map
            )
            ri = 0
            while ri < len(rewrite_errors):
                errors.append(path + ":" + rewrite_errors[ri])
                ri += 1
        ei = 0
        while ei < len(import_entries):
            import_entries[ei][0]["_remove"] = JBool(True)
            ei += 1
        bi = 0
        while bi < len(ast_body):
            bstmt = ast_body[bi]
            stype = get_str(bstmt, "_type")
            def_name = ""
            if stype == "ClassDef" or stype == "FunctionDef":
                def_name = get_str(bstmt, "name")
            elif stype == "TypeAlias":
                ta_name_node = get_node(bstmt, "name")
                def_name = get_str(ta_name_node, "id")
            elif stype == "Assign":
                targets = get_nodes(bstmt, "targets")
                if len(targets) > 0:
                    t = targets[0]
                    if get_str(t, "_type") == "Name":
                        def_name = get_str(t, "id")
            elif stype == "AnnAssign":
                ann_target = get_node(bstmt, "target")
                if get_str(ann_target, "_type") == "Name":
                    def_name = get_str(ann_target, "id")
            if def_name != "" and def_name in dedup_names:
                if def_name in dedup_seen:
                    bstmt["_remove"] = JBool(True)
                else:
                    dedup_seen.add(def_name)
            bi += 1
        new_body: list[ASTNode] = []
        bi = 0
        while bi < len(ast_body):
            bstmt = ast_body[bi]
            if get_bool(bstmt, "_remove"):
                bi += 1
                continue
            skip_stdlib = False
            btype = get_str(bstmt, "_type")
            if btype == "ImportFrom" and _classify_import(bstmt) == "stdlib":
                names_val = bstmt.get("names")
                if isinstance(names_val, JList):
                    skip_stdlib = _stdlib_import_seen(names_val.items, stdlib_seen)
                elif isinstance(names_val, list):
                    skip_stdlib = _stdlib_import_seen(names_val, stdlib_seen)
            elif btype == "Import":
                names_val = bstmt.get("names")
                if isinstance(names_val, JList):
                    skip_stdlib = _stdlib_import_seen(names_val.items, stdlib_seen)
                elif isinstance(names_val, list):
                    skip_stdlib = _stdlib_import_seen(names_val, stdlib_seen)
            if skip_stdlib:
                bi += 1
                continue
            _tag_source_file(bstmt, path)
            new_body.append(bstmt)
            bi += 1
        mi = 0
        while mi < len(new_body):
            merged_body.append(new_body[mi])
            mi += 1
        oi += 1
    if len(errors) > 0:
        return (None, errors, {})
    wrapped_body = JList([])
    wbi = 0
    while wbi < len(merged_body):
        wrapped_body.items.append(JDict(merged_body[wbi]))
        wbi += 1
    return ({"_type": JStr("Module"), "body": wrapped_body}, [], file_renames)


def _pipeline_post_parse(
    ast_dict: ASTNode,
    source: str,
    target: str,
    stop_at: str | None,
    strict_math: bool,
    strict_tostring: bool,
    file_renames: dict[str, dict[str, str]] | None = None,
) -> tuple[int, str]:
    """Run pipeline phases after parsing. Returns (exit_code, output)."""
    bind_result = run_bind(ast_dict)
    if not bind_result.subset_ok() and stop_at != "names":
        err_strs: list[str] = []
        sei = 0
        while sei < len(bind_result.subset_violations):
            err_strs.append(str(bind_result.subset_violations[sei]))
            sei += 1
        _print_errors(err_strs)
        return (1, "")
    if stop_at == "subset":
        if len(bind_result.subset_warnings) > 0:
            warn_strs: list[str] = []
            swi = 0
            while swi < len(bind_result.subset_warnings):
                warn_strs.append(str(bind_result.subset_warnings[swi]))
                swi += 1
            _print_errors(warn_strs)
        return (0, "")
    if not bind_result.names_ok():
        err_strs: list[str] = []
        nei = 0
        while nei < len(bind_result.name_violations):
            err_strs.append(str(bind_result.name_violations[nei]))
            nei += 1
        _print_errors(err_strs)
        return (1, "")
    if stop_at == "names":
        if len(bind_result.name_warnings) > 0:
            warn_strs: list[str] = []
            nwi = 0
            while nwi < len(bind_result.name_warnings):
                warn_strs.append(str(bind_result.name_warnings[nwi]))
                nwi += 1
            _print_errors(warn_strs)
        return (0, to_json(_name_table_to_dict(bind_result.table)))
    known_classes = bind_result.known_classes
    # Add aliases to known_classes (bare → prefixed) from file_renames
    if file_renames is not None:
        _bare_to_prefixed: dict[str, str] = {}
        _frk = list(file_renames.keys())
        _fri = 0
        while _fri < len(_frk):
            _f = _frk[_fri]
            _renames = file_renames[_f]
            _rk = list(_renames.keys())
            _ri = 0
            while _ri < len(_rk):
                _bare = _rk[_ri]
                _prefixed = _renames[_bare]
                if _bare not in _bare_to_prefixed:
                    _bare_to_prefixed[_bare] = _prefixed
                elif _bare_to_prefixed[_bare] != _prefixed:
                    _bare_to_prefixed[_bare] = ""  # Ambiguous
                _ri += 1
            _fri += 1
        _ak = list(_bare_to_prefixed.keys())
        _ai = 0
        while _ai < len(_ak):
            _bare = _ak[_ai]
            _prefixed = _bare_to_prefixed[_bare]
            if (
                _prefixed != ""
                and _prefixed in known_classes
                and _bare not in known_classes
            ):
                known_classes[_bare] = _prefixed
            _ai += 1
    node_classes = bind_result.node_classes
    class_bases = bind_result.class_bases
    if stop_at == "signatures":
        sig_result = collect_signatures(
            ast_dict, known_classes, node_classes, bind_result.type_aliases, class_bases
        )
        sig_errors = sig_result.errors()
        if len(sig_errors) > 0:
            err_strs: list[str] = []
            sei = 0
            while sei < len(sig_errors):
                err_strs.append(str(sig_errors[sei]))
                sei += 1
            _print_errors(err_strs)
            return (1, "")
        return (0, to_json(sig_result.to_dict()))
    hier_result = build_hierarchy(
        known_classes, class_bases, bind_result.class_source_files
    )
    hier_errors = hier_result.errors()
    if len(hier_errors) > 0:
        err_strs: list[str] = []
        hei = 0
        while hei < len(hier_errors):
            err_strs.append(str(hier_errors[hei]))
            hei += 1
        _print_errors(err_strs)
        return (1, "")
    if stop_at == "hierarchy":
        return (0, to_json(hier_result.to_dict()))
    hierarchy_roots: set[str] = set()
    hri = 0
    while hri < len(hier_result.hierarchy_roots):
        hierarchy_roots.add(hier_result.hierarchy_roots[hri])
        hri += 1
    tc_result = collect_types(
        ast_dict,
        known_classes,
        node_classes,
        bind_result.type_aliases,
        class_bases,
        hierarchy_roots,
    )
    tc_errors = tc_result.errors()
    if len(tc_errors) > 0:
        err_strs: list[str] = []
        tei = 0
        while tei < len(tc_errors):
            err_strs.append(str(tc_errors[tei]))
            tei += 1
        _print_errors(err_strs)
        return (1, "")
    if stop_at == "fields":
        return (0, to_json(tc_result.fields_to_dict()))
    stamp_uids(ast_dict)
    inf_result = run_pycheck(
        ast_dict,
        tc_result,
        hier_result,
        known_classes,
        class_bases,
        bind_result.flow_graphs,
    )
    inf_errors = inf_result.errors()
    if len(inf_errors) > 0:
        err_strs: list[str] = []
        iei = 0
        while iei < len(inf_errors):
            err_strs.append(str(inf_errors[iei]))
            iei += 1
        _print_errors(err_strs)
        return (1, "")
    if stop_at == "pycheck":
        reveals_out = JList([])
        inf_reveals = inf_result.reveals()
        ri = 0
        while ri < len(inf_reveals):
            rev = inf_reveals[ri]
            reveals_out.items.append(
                JDict({"line": JInt(rev[0]), "type": JStr(rev[1])})
            )
            ri += 1
        d: dict[str, JsonValue] = {"ast": JDict(ast_dict), "reveals": reveals_out}
        return (0, to_json(JDict(d)))
    module, lower_errors = lower(
        ast_dict,
        tc_result,
        hier_result,
        known_classes,
        class_bases,
        inf_result,
    )
    if len(lower_errors) > 0:
        err_strs: list[str] = []
        lei = 0
        while lei < len(lower_errors):
            err_strs.append(str(lower_errors[lei]))
            lei += 1
        _print_errors(err_strs)
        return (1, "")
    if module is None:
        print("error: lowering produced no module", file=sys.stderr)
        return (1, "")
    if strict_math:
        module.strict_math = True
    if strict_tostring:
        module.strict_tostring = True
    if stop_at == "lowering-text":
        return (0, to_source(module))
    if stop_at == "lowering":
        return (0, to_json(module_to_dict(module)))
    checker = Checker()
    checker.collect_declarations(module)
    if len(checker.errors) > 0:
        err_strs: list[str] = []
        cei = 0
        while cei < len(checker.errors):
            err_strs.append(str(checker.errors[cei]))
            cei += 1
        _print_errors(err_strs)
        return (1, "")
    checker.enter_scope()
    for cdecl in module.decls:
        if isinstance(cdecl, TLetStmt):
            checker.check_let_stmt(cdecl)
    checker.check_bodies(module)
    if len(checker.errors) > 0:
        err_strs: list[str] = []
        cei = 0
        while cei < len(checker.errors):
            err_strs.append(str(checker.errors[cei]))
            cei += 1
        _print_errors(err_strs)
        return (1, "")
    analyze_returns(module, checker)
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    if stop_at == "analyze":
        return (0, to_json(module_to_dict(module)))
    if target == "python":
        return (0, emit_python(module))
    if target == "perl":
        return (0, emit_perl(module))
    if target == "ruby":
        return (0, emit_ruby(module))
    if target == "taytsh":
        return (0, emit_taytsh(module))
    print("error: backend not yet implemented for '" + target + "'", file=sys.stderr)
    return (1, "")


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
    ast_dict: dict[str, JsonValue] = {}
    try:
        ast_dict = parse(source)
    except ParseError as e:
        print(
            "error:" + str(e.lineno) + ":" + str(e.col) + ": [parse] " + e.msg,
            file=sys.stderr,
        )
        return (1, "")
    if stop_at == "parse":
        return (0, to_json(JDict(ast_dict)))
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
            print(str(USAGE), end="")
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


TAYTSH_PHASES: list[str] = [
    "parse",
    "check",
    "returns",
    "scope",
    "liveness",
    "strings",
    "hoisting",
    "ownership",
    "callgraph",
]

TAYTSH_EMIT_TARGETS: list[str] = ["python", "perl", "ruby", "taytsh"]


def taytsh_pipeline(argv: list[str]) -> int:
    """Handle taytsh --stop-at/--emit subcommand."""
    stop_at: str | None = None
    emit_target: str | None = None
    strict_math = False
    strict_tostring = False
    filepath: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--stop-at":
            if i + 1 >= len(argv):
                print("error: --stop-at requires an argument", file=sys.stderr)
                return 2
            stop_at = argv[i + 1]
            i += 2
        elif arg == "--emit":
            if i + 1 >= len(argv):
                print("error: --emit requires an argument", file=sys.stderr)
                return 2
            emit_target = argv[i + 1]
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
        elif arg.startswith("-"):
            print("error: unknown flag '" + arg + "'", file=sys.stderr)
            return 2
        else:
            filepath = arg
            i += 1
    if stop_at is not None and stop_at not in TAYTSH_PHASES:
        print("error: unknown taytsh phase '" + stop_at + "'", file=sys.stderr)
        return 2
    if emit_target is not None and emit_target not in TAYTSH_EMIT_TARGETS:
        print("error: unknown emit target '" + emit_target + "'", file=sys.stderr)
        return 2
    source, err = read_source(filepath)
    if err != 0:
        return err
    tokens = taytsh_tokenize(source)
    parser = TaytshParser(tokens)
    module = parser.parse_program()
    if strict_math:
        module.strict_math = True
    if strict_tostring:
        module.strict_tostring = True
    if stop_at == "parse":
        d: dict[str, JsonValue] = {
            "strict_math": JBool(module.strict_math),
            "strict_tostring": JBool(module.strict_tostring),
        }
        print(to_json(JDict(d)))
        return 0
    check_result = check_with_info(module)
    errors = check_result[0]
    checker = check_result[1]
    if len(errors) > 0:
        ei = 0
        while ei < len(errors):
            print(str(errors[ei]), file=sys.stderr)
            ei += 1
        return 1
    if stop_at == "check":
        reveals_out = JList([])
        ri = 0
        while ri < len(checker.reveals):
            rev = checker.reveals[ri]
            reveals_out.items.append(
                JDict({"line": JInt(rev[0]), "type": JStr(rev[1])})
            )
            ri += 1
        print(to_json(JDict({"reveals": reveals_out})))
        return 0
    if stop_at == "returns":
        analyze_returns(module, checker)
        print(to_json(JDict(serialize_annotations(module, "returns"))))
        return 0
    if stop_at == "scope":
        analyze_scope(module, checker)
        print(to_json(JDict(serialize_annotations(module, "scope"))))
        return 0
    if stop_at == "liveness":
        analyze_scope(module, checker)
        analyze_liveness(module, checker)
        print(to_json(JDict(serialize_annotations(module, "liveness"))))
        return 0
    if stop_at == "strings":
        analyze_scope(module, checker)
        analyze_liveness(module, checker)
        analyze_strings(module, checker)
        print(to_json(JDict(serialize_annotations(module, "strings"))))
        return 0
    if stop_at == "hoisting":
        analyze_hoisting(module, checker)
        print(to_json(JDict(serialize_annotations(module, "hoisting"))))
        return 0
    if stop_at == "ownership":
        analyze_scope(module, checker)
        analyze_liveness(module, checker)
        analyze_ownership(module, checker)
        print(to_json(JDict(serialize_annotations(module, "ownership"))))
        return 0
    if stop_at == "callgraph":
        analyze_callgraph(module, checker)
        print(to_json(JDict(serialize_callgraph(module, checker))))
        return 0
    if emit_target is not None:
        analyze_returns(module, checker)
        analyze_scope(module, checker)
        analyze_liveness(module, checker)
        result = ""
        if emit_target == "python":
            result = emit_python(module)
        elif emit_target == "perl":
            result = emit_perl(module)
        elif emit_target == "ruby":
            result = emit_ruby(module)
        elif emit_target == "taytsh":
            result = emit_taytsh(module)
        print(result)
        return 0
    print("error: --stop-at or --emit required", file=sys.stderr)
    return 2


def main() -> None:
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "taytsh":
        taytsh_args = sys.argv[2:]
        has_pipeline_flag = False
        ti = 0
        while ti < len(taytsh_args):
            if taytsh_args[ti] == "--stop-at" or taytsh_args[ti] == "--emit":
                has_pipeline_flag = True
                break
            ti += 1
        if has_pipeline_flag:
            sys.exit(taytsh_pipeline(taytsh_args))
        from .taytsh.cli import cli_main as taytsh_main

        return taytsh_main(taytsh_args)  # type: ignore[return-value]
    target, stop_at, strict_math, strict_tostring, project, input_file, output_file = (
        parse_args()
    )
    if project:
        source, err = read_source(input_file)
        if err != 0:
            sys.exit(err)
        if len(source) == 0:
            print("error: no input provided", file=sys.stderr)
            sys.exit(2)
        files = _parse_project_input(source)
        if len(files) == 0:
            print("error: no .py files found in directory", file=sys.stderr)
            sys.exit(1)
        sys.exit(
            main_project(
                files, target, stop_at, strict_math, strict_tostring, output_file
            )
        )
    source, err = read_source(input_file)
    if err != 0:
        sys.exit(err)
    if len(source) == 0:
        print("error: no input provided", file=sys.stderr)
        sys.exit(2)
    exit_code, output = run_pipeline(
        source, target, stop_at, strict_math, strict_tostring
    )
    if exit_code != 0:
        sys.exit(exit_code)
    if len(output) > 0:
        sys.exit(write_output(output, output_file))


def main_project(
    files: list[tuple[str, str]],
    target: str,
    stop_at: str | None,
    strict_math: bool,
    strict_tostring: bool,
    output_file: str | None,
) -> int:
    """Project-mode entry point. files is [(relpath, source)]."""
    file_asts: list[tuple[str, ASTNode]] = []
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
                path
                + ":error:"
                + str(e.lineno)
                + ":"
                + str(e.col)
                + ": [parse] "
                + e.msg,
                file=sys.stderr,
            )
            return 1
        file_asts.append((path, ast_dict))
        i += 1
    if stop_at == "parse":
        items: list[JsonValue] = []
        j = 0
        while j < len(file_asts):
            items.append(
                JDict({"path": JStr(file_asts[j][0]), "ast": JDict(file_asts[j][1])})
            )
            j += 1
        output = to_json(JList(items))
        return write_output(output, output_file)
    merged_ast, merge_errors, file_renames = merge_project(file_asts)
    if len(merge_errors) > 0:
        _print_errors(merge_errors)
        return 1
    if merged_ast is None:
        print("error: merge produced no AST", file=sys.stderr)
        return 1
    all_source_parts: list[str] = []
    k = 0
    while k < len(files):
        all_source_parts.append(files[k][1])
        k += 1
    combined_source = "\n".join(all_source_parts)
    exit_code, output = _pipeline_post_parse(
        merged_ast,
        combined_source,
        target,
        stop_at,
        strict_math,
        strict_tostring,
        file_renames,
    )
    if exit_code != 0:
        return exit_code
    if len(output) > 0:
        return write_output(output, output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
