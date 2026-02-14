"""Subset-compliant entry point - reads from stdin, writes to stdout."""

from __future__ import annotations

import sys

from .frontend.parse import parse, ParseError
from .frontend.subset import verify as verify_subset
from .frontend.names import NameInfo, NameTable, resolve_names
from .frontend.signatures import SignatureResult, collect_signatures
from .frontend.fields import FieldResult, collect_fields
from .frontend.hierarchy import HierarchyResult, build_hierarchy
from .frontend.inference import InferenceResult, run_inference

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
tongues [OPTIONS] < input.py

Options:
  --target TARGET     Output language: c, csharp, dart, go, java, javascript,
                      lua, perl, php, python, ruby, rust, swift, typescript, zig
  --stop-at PHASE     Stop after phase: parse, subset, names, signatures,
                      fields, hierarchy, inference, lowering, analyze
  --strict            Enable strict math and strict tostring
  --strict-math       Enable strict math mode
  --strict-tostring   Enable strict tostring mode
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


def read_source() -> tuple[str, int]:
    """Read source from stdin with validation. Returns (source, exit_code) where exit_code 0 means OK."""
    raw = sys.stdin.buffer.read()
    if len(raw) > 0:
        try:
            source = raw.decode("utf-8")
        except ValueError:
            print("error: invalid utf-8 in input", file=sys.stderr)
            return ("", 1)
        return (source, 0)
    return ("", 0)


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
        parts = []
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
        scope_names = {}
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


# --- Error reporting ---


def _print_errors(errors: list[object]) -> None:
    """Print a list of error objects to stderr."""
    i = 0
    while i < len(errors):
        print(str(errors[i]), file=sys.stderr)
        i += 1


# --- Pipeline ---


def run_pipeline(
    target: str, stop_at: str | None, strict_math: bool, strict_tostring: bool
) -> int:
    """Run the transpilation pipeline, optionally stopping at a phase."""
    source, err = read_source()
    if err != 0:
        return err
    if len(source) == 0:
        print("error: no input provided", file=sys.stderr)
        return 2
    if stop_at == "subset" and should_skip_file(source):
        return 0
    # Phase 2: Parse
    try:
        ast_dict = parse(source)
    except ParseError as e:
        print(
            "error:" + str(e.lineno) + ":" + str(e.col) + ": " + e.msg, file=sys.stderr
        )
        return 1
    if stop_at == "parse":
        print(to_json(ast_dict))
        return 0
    # Phase 3: Subset
    result = verify_subset(ast_dict)
    errors = result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return 1
    if stop_at == "subset":
        return 0
    # Phase 4: Names
    name_result = resolve_names(ast_dict)
    errors = name_result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return 1
    if stop_at == "names":
        print(to_json(_name_table_to_dict(name_result.table)))
        return 0
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
    sig_result = collect_signatures(ast_dict, known_classes, node_classes)
    errors = sig_result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return 1
    if stop_at == "signatures":
        print(to_json(sig_result.to_dict()))
        return 0
    # Phase 6: Fields
    # Determine hierarchy roots
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
    field_result = collect_fields(ast_dict, known_classes, node_classes, hierarchy_roots, sig_result)
    errors = field_result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return 1
    if stop_at == "fields":
        print(to_json(field_result.to_dict()))
        return 0
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
        return 1
    if stop_at == "hierarchy":
        print(to_json(hier_result.to_dict()))
        return 0
    # Phase 8: Inference
    inf_result = run_inference(ast_dict, sig_result, field_result, hier_result, known_classes, class_bases)
    errors = inf_result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return 1
    if stop_at == "inference":
        print(to_json(ast_dict))
        return 0
    # Phase 9: Lowering
    # TODO: wire frontend.lowering — produces TModule, set strict flags on it
    if stop_at == "lowering":
        print("error: phase not yet implemented", file=sys.stderr)
        return 1
    # Phases 10-16: Analyze
    # TODO: wire middleend
    if stop_at == "analyze":
        print("error: phase not yet implemented", file=sys.stderr)
        return 1
    # Phase 17: Backend
    # TODO: wire backend
    print("error: phase not yet implemented", file=sys.stderr)
    return 1


def parse_args() -> tuple[str, str | None, bool, bool]:
    """Parse command-line arguments. Returns (target, stop_at, strict_math, strict_tostring)."""
    args = sys.argv[1:]
    target = "go"
    stop_at: str | None = None
    strict_math = False
    strict_tostring = False
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
        else:
            print("error: unknown flag '" + arg + "'", file=sys.stderr)
            sys.exit(2)
    if stop_at is not None and stop_at not in PHASES:
        print("error: unknown phase '" + stop_at + "'", file=sys.stderr)
        sys.exit(2)
    if target not in TARGETS:
        print("error: unknown target '" + target + "'", file=sys.stderr)
        sys.exit(2)
    return (target, stop_at, strict_math, strict_tostring)


def main() -> int:
    """Main entry point."""
    target, stop_at, strict_math, strict_tostring = parse_args()
    return run_pipeline(target, stop_at, strict_math, strict_tostring)


if __name__ == "__main__":
    sys.exit(main())
