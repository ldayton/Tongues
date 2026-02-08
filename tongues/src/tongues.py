"""Subset-compliant entry point - reads from stdin, writes to stdout."""

from __future__ import annotations

import sys

from .frontend import Frontend
from .frontend.parse import parse, ParseError
from .frontend.subset import verify as verify_subset
from .frontend.names import NameInfo, NameTable, resolve_names
from .middleend import analyze
from .serialize import (
    serialize as ir_serialize,
    signatures_to_dict,
    fields_to_dict,
    hierarchy_to_dict,
    module_to_dict,
)
from .backend.c import CBackend
from .backend.go import GoBackend
from .backend.java import JavaBackend
from .backend.javascript import JsBackend
from .backend.lua import LuaBackend
from .backend.perl import PerlBackend
from .backend.python import PythonBackend
from .backend.ruby import RubyBackend
from .backend.typescript import TsBackend
from .backend.csharp import CSharpBackend
from .backend.dart import DartBackend
from .backend.php import PhpBackend
from .backend.rust import RustBackend
from .backend.swift import SwiftBackend
from .backend.zig import ZigBackend

BACKENDS: dict[
    str,
    type[CBackend]
    | type[GoBackend]
    | type[JavaBackend]
    | type[JsBackend]
    | type[LuaBackend]
    | type[PerlBackend]
    | type[PythonBackend]
    | type[RubyBackend]
    | type[TsBackend]
    | type[CSharpBackend]
    | type[DartBackend]
    | type[PhpBackend]
    | type[RustBackend]
    | type[SwiftBackend]
    | type[ZigBackend],
] = {
    "c": CBackend,
    "csharp": CSharpBackend,
    "dart": DartBackend,
    "go": GoBackend,
    "java": JavaBackend,
    "javascript": JsBackend,
    "lua": LuaBackend,
    "perl": PerlBackend,
    "php": PhpBackend,
    "python": PythonBackend,
    "ruby": RubyBackend,
    "rust": RustBackend,
    "swift": SwiftBackend,
    "typescript": TsBackend,
    "zig": ZigBackend,
}

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
  --target TARGET   Output language: c, csharp, dart, go, java, javascript,
                    lua, perl, php, python, ruby, rust, swift, typescript, zig
  --stop-at PHASE   Stop after phase: parse, subset, names, signatures,
                    fields, hierarchy, inference, lowering, analyze
  --help            Show this help message
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


def run_pipeline(target: str, stop_at: str | None) -> int:
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
        print("error:" + str(e.lineno) + ":" + str(e.col) + ": " + e.msg, file=sys.stderr)
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
    # Phases 5+: Frontend (phased execution)
    fe = Frontend()
    try:
        fe.init_from_names(source, name_result)
    except Exception as e:
        print("error: " + str(e), file=sys.stderr)
        return 1
    # Phase 5: Signatures
    try:
        fe.collect_sigs(ast_dict)
    except Exception as e:
        print("error: " + str(e), file=sys.stderr)
        return 1
    if stop_at == "signatures":
        print(to_json(signatures_to_dict(fe.symbols)))
        return 0
    # Phase 6: Fields
    try:
        fe.collect_flds(ast_dict)
    except Exception as e:
        print("error: " + str(e), file=sys.stderr)
        return 1
    if stop_at == "fields":
        print(to_json(fields_to_dict(fe.symbols)))
        return 0
    # Phase 7: Hierarchy (computed during init_from_names)
    if stop_at == "hierarchy":
        print(to_json(hierarchy_to_dict(fe.symbols, fe.get_hierarchy_root())))
        return 0
    # Phases 8-9: Inference + Lowering
    try:
        module = fe.build_ir(ast_dict)
    except Exception as e:
        print("error: " + str(e), file=sys.stderr)
        return 1
    if stop_at == "inference":
        print(to_json(ir_serialize(ast_dict)))
        return 0
    if stop_at == "lowering":
        print(to_json(module_to_dict(module)))
        return 0
    # Phases 10-14: Analyze
    try:
        analyze(module)
    except Exception as e:
        print("error: " + str(e), file=sys.stderr)
        return 1
    if stop_at == "analyze":
        print(to_json(module_to_dict(module)))
        return 0
    # Phase 15: Backend
    backend_cls = BACKENDS[target]
    be = backend_cls()
    try:
        code = be.emit(module)
    except Exception as e:
        print("error: " + str(e), file=sys.stderr)
        return 1
    print(code)
    return 0


def parse_args() -> tuple[str, str | None]:
    """Parse command-line arguments. Returns (target, stop_at)."""
    args = sys.argv[1:]
    target = "go"
    stop_at: str | None = None
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
        else:
            print("error: unknown flag '" + arg + "'", file=sys.stderr)
            sys.exit(2)
    if stop_at is not None and stop_at not in PHASES:
        print("error: unknown phase '" + stop_at + "'", file=sys.stderr)
        sys.exit(2)
    if target not in BACKENDS:
        print("error: unknown target '" + target + "'", file=sys.stderr)
        sys.exit(2)
    return (target, stop_at)


def main() -> int:
    """Main entry point."""
    target, stop_at = parse_args()
    return run_pipeline(target, stop_at)


if __name__ == "__main__":
    sys.exit(main())
