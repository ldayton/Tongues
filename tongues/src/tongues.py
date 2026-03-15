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
    collect_expr_annotations,
    serialize_annotations,
    to_dict as module_to_dict,
)
from .taytsh.check import Checker, check_with_info
from .taytsh.emit import to_source
from .taytsh.parse import Parser as TaytshParser
from .taytsh.tokens import tokenize as taytsh_tokenize
from .middleend.callgraph import analyze_callgraph
from .middleend.error_types import patch_error_types
from .middleend.callgraph_serial import serialize_callgraph
from .middleend.hoisting import analyze_hoisting
from .middleend.int_width import analyze_int_width
from .middleend.liveness import analyze_liveness
from .middleend.ownership import analyze_ownership
from .middleend.returns import analyze_returns
from .middleend.scope import analyze_scope
from .middleend.strings import analyze_strings
from .backend.java import emit_java
from .backend.javascript import emit_javascript
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

VERSION: str = "0.2.2"

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
  --version           Show version number
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
    if raw:
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
        if not obj.items:
            return "[]"
        parts: list[str] = []
        pad = " " * (indent * (level + 1))
        pad_close = " " * (indent * level)
        for item in obj.items:
            parts.append(pad + _to_json(item, indent, level + 1))
        return "[\n" + ",\n".join(parts) + "\n" + pad_close + "]"
    if isinstance(obj, JDict):
        if not obj.entries:
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
    if info.bases:
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
    if scopes:
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
    consumed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            consumed += 1
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
        consumed += 1
    remaining = "\n".join(lines[consumed:])
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
    if parts and parts[0] in ALLOWED_FROM_MODULES:
        return "stdlib"
    if parts and parts[0] in IMPORT_ONLY_MODULES:
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
    while ready:
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
        for x, y in zip(xl, yl):
            work_a.append(x)
            work_b.append(y)
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
    x_keys: list[str] = [k for k in xd if k not in ignore]
    y_keys: list[str] = [k for k in yd if k not in ignore]
    if len(x_keys) != len(y_keys):
        return False
    x_keys.sort()
    y_keys.sort()
    for xk, yk in zip(x_keys, y_keys):
        if xk != yk:
            return False
    for xk, yk in zip(x_keys, y_keys):
        work_a.append(xd[xk])
        work_b.append(yd[yk])
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
            for val in node_entries.values():
                if isinstance(val, (JDict, JList)):
                    work.append(val)
        elif isinstance(item, JList):
            for child in item.items:
                if isinstance(child, (JDict, JList)):
                    work.append(child)
        wi += 1
    return refs


def _compute_module_stems(paths: list[str]) -> dict[str, str]:
    """Compute unique module stems for each file path."""
    raw_stems: dict[str, str] = {}
    for path in paths:
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
            if not stem:
                stem = "__init__"
        else:
            if filename.endswith(".py"):
                stem = filename[:-3]
            else:
                stem = filename
        raw_stems[path] = stem
    stem_to_paths: dict[str, list[str]] = {}
    for path in paths:
        stem = raw_stems[path]
        if stem not in stem_to_paths:
            stem_to_paths[stem] = []
        stem_to_paths[stem].append(path)
    result: dict[str, str] = {}
    for stp_stem, colliding in stem_to_paths.items():
        if len(colliding) == 1:
            result[colliding[0]] = stp_stem
        else:
            for path in colliding:
                slash_idx = path.rfind("/")
                if slash_idx >= 0:
                    parent = path[:slash_idx]
                    parent_slash = parent.rfind("/")
                    if parent_slash >= 0:
                        parent_name = parent[parent_slash + 1 :]
                    else:
                        parent_name = parent
                    result[path] = parent_name + "_" + stp_stem
                else:
                    result[path] = stp_stem
    return result


def _is_all_caps(name: str) -> bool:
    """Check if name follows ALL_CAPS convention."""
    if not name:
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
    if name and name[0] == "_":
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
    for f, names in file_names.items():
        for entry in names:
            ename = entry[0]
            ast_node = entry[3]
            if ename not in name_to_defs:
                name_to_defs[ename] = []
            name_to_defs[ename].append((f, ast_node))
    dedup_candidates: set[str] = set()
    file_renames: dict[str, dict[str, str]] = {}
    for name, defs in name_to_defs.items():
        if len(defs) > 1:
            all_equal = True
            for d in defs[1:]:
                if not _ast_equal(defs[0][1], d[1]):
                    all_equal = False
                    break
            if all_equal:
                dedup_candidates.add(name)
            else:
                for d in defs:
                    f = d[0]
                    stem = stems[f]
                    prefixed = _prefix_name(name, stem)
                    if f not in file_renames:
                        file_renames[f] = {}
                    file_renames[f][name] = prefixed
    changed = True
    while changed:
        changed = False
        all_prefixed: set[str] = set()
        for rmap in file_renames.values():
            for k in rmap:
                all_prefixed.add(k)
        to_demote: list[str] = []
        for cand in list(dedup_candidates):
            defs = name_to_defs[cand]
            refs = _collect_definition_refs(defs[0][1])
            if any(r in all_prefixed for r in refs):
                to_demote.append(cand)
        for demoted in to_demote:
            dedup_candidates.discard(demoted)
            defs = name_to_defs[demoted]
            for d in defs:
                f = d[0]
                stem = stems[f]
                prefixed = _prefix_name(demoted, stem)
                if f not in file_renames:
                    file_renames[f] = {}
                file_renames[f][demoted] = prefixed
            changed = True
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
            for val in node_entries.values():
                if isinstance(val, (JDict, JList)):
                    work.append(val)
        elif isinstance(item, JList):
            for child in item.items:
                if isinstance(child, (JDict, JList)):
                    work.append(child)
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
            for val in list(node_entries.values()):
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
        elif isinstance(item, JList):
            for child in item.items:
                if isinstance(child, (JDict, JList)):
                    work.append(child)
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
            for val in node_entries.values():
                if isinstance(val, (JDict, JList)):
                    work.append(val)
        elif isinstance(item, JList):
            for child in item.items:
                if isinstance(child, (JDict, JList)):
                    work.append(child)
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
    for ni, alias_raw in enumerate(names_list):
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
    if not new_indices:
        return True
    if len(new_indices) < len(names_list):
        kept: list[JsonValue] = []
        for idx in new_indices:
            kept.append(names_list[idx])
        names_list.clear()
        for kept_item in kept:
            names_list.append(kept_item)
    return False


def merge_project(
    file_asts: list[tuple[str, ASTNode]],
) -> tuple[ASTNode | None, list[str], dict[str, dict[str, str]]]:
    """Full project merge. Returns (merged_ast, errors)."""
    errors: list[str] = []
    universe: set[str] = set()
    for path, _ in file_asts:
        universe.add(path)
    deps: dict[str, list[str]] = {}
    file_import_info: dict[str, list[tuple[ASTNode, str, list[tuple[str, str]]]]] = {}
    for path, ast_dict in file_asts:
        ast_body = get_nodes(ast_dict, "body")
        file_deps: list[str] = []
        import_entries: list[tuple[ASTNode, str, list[tuple[str, str]]]] = []
        for stmt in ast_body:
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
                    for rpath, rerr in resolved:
                        if rpath != "":
                            if rpath not in file_deps:
                                file_deps.append(rpath)
                        elif rerr != "":
                            errors.append(rerr)
        deps[path] = file_deps
        file_import_info[path] = import_entries
    if errors:
        return (None, errors, {})
    all_file_names: dict[str, list[tuple[str, int, int, ASTNode]]] = {}
    for path, ast_dict in file_asts:
        all_file_names[path] = _collect_module_names(ast_dict)
    file_list: list[str] = [path for path, _ in file_asts]
    stems = _compute_module_stems(file_list)
    dedup_names, file_renames = _plan_collision_resolution(all_file_names, stems)
    file_name_map: dict[str, dict[str, str]] = {}
    for f, names in all_file_names.items():
        name_map: dict[str, str] = {}
        f_renames = file_renames.get(f, {})
        for entry in names:
            original = entry[0]
            if original in f_renames:
                name_map[original] = f_renames[original]
            else:
                name_map[original] = original
        file_name_map[f] = name_map
    ordered = _dependency_order(file_list, deps)
    merged_body: list[ASTNode] = []
    dedup_seen: set[str] = set()
    stdlib_seen: set[str] = set()
    ast_by_path: dict[str, ASTNode] = {p: a for p, a in file_asts}
    for path in ordered:
        found_ast = ast_by_path.get(path)
        if found_ast is None:
            continue
        ast_body = get_nodes(found_ast, "body")
        if not ast_body:
            continue
        rename_map: dict[str, str] = {}
        module_bindings: dict[str, str] = {}
        import_entries = file_import_info.get(path, [])
        for imp_stmt, imp_module, imp_resolved in import_entries:
            names_list = get_nodes(imp_stmt, "names")
            level = get_int(imp_stmt, "level")
            if not imp_module and level > 0:
                for ni, alias in enumerate(names_list):
                    name = get_str(alias, "name")
                    v = alias.get("asname")
                    asname = ""
                    if isinstance(v, JStr):
                        asname = v.value
                    if name != "" and name != "*":
                        bound = asname if asname != "" else name
                        if ni < len(imp_resolved):
                            rpath = imp_resolved[ni][0]
                            if rpath != "":
                                module_bindings[bound] = rpath
            else:
                source_file = ""
                if imp_resolved:
                    source_file = imp_resolved[0][0]
                source_renames = file_renames.get(source_file, {})
                for alias in names_list:
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
        if module_bindings:
            slash_idx = path.rfind("/")
            if slash_idx >= 0:
                init_path = path[:slash_idx] + "/__init__.py"
                init_names = file_name_map.get(init_path)
                if init_names is not None:
                    new_mb: dict[str, str] = {}
                    for mb_key, mb_val in module_bindings.items():
                        if mb_key in init_names:
                            rename_map[mb_key] = init_names[mb_key]
                        else:
                            new_mb[mb_key] = mb_val
                    module_bindings = new_mb
        own_renames = file_renames.get(path, {})
        for k in own_renames:
            rename_map[k] = own_renames[k]
        if rename_map:
            _rewrite_names(found_ast, rename_map)
        if module_bindings:
            rewrite_errors = _rewrite_module_attrs(
                found_ast, module_bindings, file_name_map
            )
            for err in rewrite_errors:
                errors.append(path + ":" + err)
        for imp_entry in import_entries:
            imp_entry[0]["_remove"] = JBool(True)
        for bstmt in ast_body:
            stype = get_str(bstmt, "_type")
            def_name = ""
            if stype == "ClassDef" or stype == "FunctionDef":
                def_name = get_str(bstmt, "name")
            elif stype == "TypeAlias":
                ta_name_node = get_node(bstmt, "name")
                def_name = get_str(ta_name_node, "id")
            elif stype == "Assign":
                targets = get_nodes(bstmt, "targets")
                if targets:
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
        new_body: list[ASTNode] = []
        for bstmt in ast_body:
            if get_bool(bstmt, "_remove"):
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
                continue
            _tag_source_file(bstmt, path)
            new_body.append(bstmt)
        merged_body.extend(new_body)
    if errors:
        return (None, errors, {})
    wrapped_body = JList([])
    for b in merged_body:
        wrapped_body.items.append(JDict(b))
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
        err_strs: list[str] = [str(v) for v in bind_result.subset_violations]
        _print_errors(err_strs)
        return (1, "")
    if stop_at == "subset":
        if bind_result.subset_warnings:
            warn_strs: list[str] = [str(w) for w in bind_result.subset_warnings]
            _print_errors(warn_strs)
        return (0, "")
    if not bind_result.names_ok():
        err_strs: list[str] = [str(v) for v in bind_result.name_violations]
        _print_errors(err_strs)
        return (1, "")
    if stop_at == "names":
        if bind_result.name_warnings:
            warn_strs: list[str] = [str(w) for w in bind_result.name_warnings]
            _print_errors(warn_strs)
        return (0, to_json(_name_table_to_dict(bind_result.table)))
    known_classes = bind_result.known_classes
    known_funcs = bind_result.known_funcs
    # Add aliases to known_classes (bare → prefixed) from file_renames
    if file_renames is not None:
        _bare_to_prefixed: dict[str, str] = {}
        for _renames in file_renames.values():
            for _bare, _prefixed in _renames.items():
                if _bare not in _bare_to_prefixed:
                    _bare_to_prefixed[_bare] = _prefixed
                elif _bare_to_prefixed[_bare] != _prefixed:
                    _bare_to_prefixed[_bare] = ""  # Ambiguous
        for _bare, _prefixed in _bare_to_prefixed.items():
            if (
                _prefixed != ""
                and _prefixed in known_classes
                and _bare not in known_classes
            ):
                known_classes[_bare] = _prefixed
    node_classes = bind_result.node_classes
    class_bases = bind_result.class_bases
    if stop_at == "signatures":
        sig_result = collect_signatures(
            ast_dict, known_classes, node_classes, bind_result.type_aliases, class_bases
        )
        sig_errors = sig_result.errors()
        if sig_errors:
            err_strs: list[str] = [str(e) for e in sig_errors]
            _print_errors(err_strs)
            return (1, "")
        return (0, to_json(sig_result.to_dict()))
    hier_result = build_hierarchy(
        known_classes, class_bases, bind_result.class_source_files
    )
    hier_errors = hier_result.errors()
    if hier_errors:
        err_strs: list[str] = [str(e) for e in hier_errors]
        _print_errors(err_strs)
        return (1, "")
    if stop_at == "hierarchy":
        return (0, to_json(hier_result.to_dict()))
    hierarchy_roots: set[str] = set(hier_result.hierarchy_roots)
    tc_result = collect_types(
        ast_dict,
        known_classes,
        node_classes,
        bind_result.type_aliases,
        class_bases,
        hierarchy_roots,
    )
    tc_errors = tc_result.errors()
    if tc_errors:
        err_strs: list[str] = [str(e) for e in tc_errors]
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
    if inf_errors:
        err_strs: list[str] = [str(e) for e in inf_errors]
        _print_errors(err_strs)
        return (1, "")
    if stop_at == "pycheck":
        inf_reveals = inf_result.reveals()
        reveals_out = JList([])
        for rev in inf_reveals:
            reveals_out.items.append(
                JDict({"line": JInt(rev[0]), "type": JStr(rev[1])})
            )
        d: dict[str, JsonValue] = {"ast": JDict(ast_dict), "reveals": reveals_out}
        return (0, to_json(JDict(d)))
    module, lower_errors = lower(
        ast_dict,
        tc_result,
        hier_result,
        known_classes,
        class_bases,
        inf_result,
        known_funcs,
    )
    if lower_errors:
        err_strs: list[str] = [str(e) for e in lower_errors]
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
    try:
        checker.collect_declarations(module)
    except Exception as exc:
        if checker.errors:
            err_strs: list[str] = [str(e) for e in checker.errors]
            _print_errors(err_strs)
        print(
            "error: check phase crashed (malformed lowered AST): " + str(exc),
            file=sys.stderr,
        )
        return (1, "")
    if checker.errors:
        err_strs: list[str] = [str(e) for e in checker.errors]
        _print_errors(err_strs)
        return (1, "")
    checker.enter_scope()
    for cdecl in module.decls:
        if isinstance(cdecl, TLetStmt):
            checker.check_let_stmt(cdecl)
    checker.check_bodies(module)
    if checker.errors:
        err_strs: list[str] = [str(e) for e in checker.errors]
        _print_errors(err_strs)
        return (1, "")
    patch_error_types(module)
    analyze_returns(module, checker)
    analyze_scope(module, checker)
    analyze_liveness(module, checker)
    analyze_int_width(module, checker)
    if stop_at == "analyze":
        return (0, to_json(module_to_dict(module)))
    if target == "python":
        return (0, emit_python(module))
    if target == "java":
        return (0, emit_java(module))
    if target == "javascript":
        return (0, emit_javascript(module))
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
        if arg == "--version":
            print(VERSION)
            sys.exit(0)
        elif arg == "--help" or arg == "-h":
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

TAYTSH_EMIT_TARGETS: list[str] = [
    "java",
    "javascript",
    "python",
    "perl",
    "ruby",
    "taytsh",
]


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
    if errors:
        for e in errors:
            print(str(e), file=sys.stderr)
        return 1
    if stop_at == "check":
        reveals_out = JList([])
        for rev in checker.reveals:
            reveals_out.items.append(
                JDict({"line": JInt(rev[0]), "type": JStr(rev[1])})
            )
        expr_anns = collect_expr_annotations(module)
        anns_out = JDict({})
        for line, ann_dict in expr_anns.items():
            line_dict = JDict({})
            for k, v in ann_dict.items():
                line_dict.entries[k] = JStr(v)
            anns_out.entries[str(line)] = line_dict
        print(to_json(JDict({"reveals": reveals_out, "annotations": anns_out})))
        return 0
    if stop_at == "returns":
        patch_error_types(module)
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
        patch_error_types(module)
        analyze_returns(module, checker)
        analyze_scope(module, checker)
        analyze_liveness(module, checker)
        analyze_int_width(module, checker)
        result = ""
        if emit_target == "java":
            result = emit_java(module)
        elif emit_target == "javascript":
            result = emit_javascript(module)
        elif emit_target == "python":
            result = emit_python(module)
        elif emit_target == "perl":
            result = emit_perl(module)
        elif emit_target == "ruby":
            analyze_strings(module, checker)
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
        has_pipeline_flag = any(a == "--stop-at" or a == "--emit" for a in taytsh_args)
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
        if not source:
            print("error: no input provided", file=sys.stderr)
            sys.exit(2)
        files = _parse_project_input(source)
        if not files:
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
    if not source:
        print("error: no input provided", file=sys.stderr)
        sys.exit(2)
    exit_code, output = run_pipeline(
        source, target, stop_at, strict_math, strict_tostring
    )
    if exit_code != 0:
        sys.exit(exit_code)
    if output:
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
    for path, raw_source in files:
        file_source, pragma_math, pragma_tostring = _extract_pragmas(raw_source)
        if pragma_math:
            strict_math = True
        if pragma_tostring:
            strict_tostring = True
        try:
            ast_dict = parse(file_source)
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
    if stop_at == "parse":
        items: list[JsonValue] = []
        for p, a in file_asts:
            items.append(JDict({"path": JStr(p), "ast": JDict(a)}))
        output = to_json(JList(items))
        return write_output(output, output_file)
    merged_ast, merge_errors, file_renames = merge_project(file_asts)
    if merge_errors:
        _print_errors(merge_errors)
        return 1
    if merged_ast is None:
        print("error: merge produced no AST", file=sys.stderr)
        return 1
    combined_source = "\n".join(src for _, src in files)
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
    if output:
        return write_output(output, output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
