"""Subset-compliant entry point."""

from __future__ import annotations

import sys

from .frontend.parse import parse, ParseError
from .frontend.subset import verify as verify_subset
from .frontend.names import NameInfo, NameTable, resolve_names
from .frontend.signatures import collect_signatures
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
    source: str, target: str, stop_at: str | None, strict_math: bool, strict_tostring: bool
) -> tuple[int, str]:
    """Run the transpilation pipeline. Returns (exit_code, output)."""
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
    sig_result = collect_signatures(ast_dict, known_classes, node_classes)
    errors = sig_result.errors()
    if len(errors) > 0:
        _print_errors(errors)
        return (1, "")
    if stop_at == "signatures":
        return (0, to_json(sig_result.to_dict()))
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
        ast_dict, sig_result, field_result, hier_result, known_classes, class_bases, source
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
        print("error: backend not yet implemented for '" + target + "'", file=sys.stderr)
        return (1, "")
    emitter = emitters[target]
    return (0, emitter(module))


def parse_args() -> tuple[str, str | None, bool, bool, str | None, str | None]:
    """Parse command-line arguments. Returns (target, stop_at, strict_math, strict_tostring, input_file, output_file)."""
    args = sys.argv[1:]
    target = "go"
    stop_at: str | None = None
    strict_math = False
    strict_tostring = False
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
    if target not in TARGETS:
        print("error: unknown target '" + target + "'", file=sys.stderr)
        sys.exit(2)
    return (target, stop_at, strict_math, strict_tostring, input_file, output_file)


def main() -> int:
    """Main entry point."""
    target, stop_at, strict_math, strict_tostring, input_file, output_file = parse_args()
    source, err = read_source(input_file)
    if err != 0:
        return err
    if len(source) == 0:
        print("error: no input provided", file=sys.stderr)
        return 2
    exit_code, output = run_pipeline(source, target, stop_at, strict_math, strict_tostring)
    if exit_code != 0:
        return exit_code
    if len(output) > 0:
        return write_output(output, output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
