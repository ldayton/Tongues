"""Tier 3: Thread pos through _typenode_to_ttype, _default_value_for_type,
_emit_hoisted_placeholders, _backpatch_hoisted; fix _build_constants; remove _P0."""

import re
import sys

TIER3_BODY = [
    "_typenode_to_ttype",
    "_default_value_for_type",
    "_emit_hoisted_placeholders",
]


def find_functions(lines):
    """Find top-level function ranges."""
    funcs = []
    i = 0
    while i < len(lines):
        m = re.match(r"^def (\w+)\(", lines[i])
        if m:
            fname = m.group(1)
            start = i
            j = i
            depth = 0
            sig_done = False
            while j < len(lines):
                for ch in lines[j]:
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                if depth <= 0 and ":" in lines[j].split("#")[0]:
                    sig_done = True
                    j += 1
                    break
                j += 1
            if not sig_done:
                i += 1
                continue
            while j < len(lines):
                line = lines[j]
                if re.match(r"^(def |class |@)", line):
                    break
                if (
                    line.strip()
                    and not line[0].isspace()
                    and not line.startswith("#")
                    and not line.startswith(")")
                ):
                    break
                j += 1
            funcs.append((start, j, fname))
            i = j
        else:
            i += 1
    return funcs


def transform(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    lines = content.split("\n")

    # Step 1: Replace _P0 -> pos in tier3 function bodies (not def lines)
    funcs = find_functions(lines)
    for start, end, fname in funcs:
        if fname not in TIER3_BODY:
            continue
        for j in range(start, end):
            if "_P0" in lines[j] and not lines[j].lstrip().startswith("def "):
                lines[j] = lines[j].replace("_P0", "pos")

    # Step 1b: Handle _build_constants — add pos = _node_pos(...) and replace _P0
    for start, end, fname in funcs:
        if fname != "_build_constants":
            continue
        # We need to find the right insertion points for pos = _node_pos(node)
        # and pos = _node_pos(item).
        # Strategy: find lines with _P0 and replace them, and add pos assignments
        # at the right indentation.
        j = start
        while j < end:
            line = lines[j]
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            # For module-level assigns/annassigns: add pos = _node_pos(node) before
            # the ttype = _typenode_to_ttype(...) line
            if (
                "ttype = _typenode_to_ttype(" in stripped
                and "class_body" not in "".join(lines[max(start, j - 15) : j])
            ):
                # Check if we're in the class body section by looking for 'item' vs 'node' context
                pass
            if "_P0" in line:
                # Determine context: are we inside the class_body loop?
                # Look backwards for 'item = class_body[j]' or 'node = body[i]'
                in_class_body = False
                for k in range(j, max(start, j - 30), -1):
                    if "item = class_body" in lines[k]:
                        in_class_body = True
                        break
                    if "node = body[" in lines[k]:
                        break
                if in_class_body:
                    lines[j] = lines[j].replace("_P0", "_node_pos(item)")
                else:
                    lines[j] = lines[j].replace("_P0", "_node_pos(node)")
            j += 1

    content = "\n".join(lines)

    # Step 2: Update function signatures

    # _typenode_to_ttype
    content = content.replace(
        "def _typenode_to_ttype(t: TypeNode) -> TType:",
        "def _typenode_to_ttype(pos: Pos, t: TypeNode) -> TType:",
    )

    # _default_value_for_type
    content = content.replace(
        "def _default_value_for_type(td: TypeNode) -> TExpr:",
        "def _default_value_for_type(pos: Pos, td: TypeNode) -> TExpr:",
    )

    # _emit_hoisted_placeholders
    content = content.replace(
        "def _emit_hoisted_placeholders(\n    names: list[str], env: _Env, pre_stmts: list[TStmt]\n) -> None:",
        "def _emit_hoisted_placeholders(\n    pos: Pos, names: list[str], env: _Env, pre_stmts: list[TStmt]\n) -> None:",
    )

    # _backpatch_hoisted — add pos: Pos param
    content = content.replace(
        "def _backpatch_hoisted(name: str, typ: TypeNode, env: _Env) -> None:",
        "def _backpatch_hoisted(pos: Pos, name: str, typ: TypeNode, env: _Env) -> None:",
    )

    # Step 3: Update callers

    # _typenode_to_ttype — callers within its own body (recursive)
    # Already replaced _P0 -> pos in body, now recursive calls need pos arg
    content = re.sub(
        r"(?<!def )_typenode_to_ttype\(([a-z])", r"_typenode_to_ttype(pos, \1", content
    )
    # Also handle _typenode_to_ttype(t.xxx) patterns already converted
    # The above regex handles these since t starts with lowercase

    # Handle _typenode_to_ttype(PrimitiveType(...)) in _backpatch_hoisted
    # (doesn't start with lowercase) — but _backpatch_hoisted's call is _typenode_to_ttype(typ)
    # which starts with lowercase, so it's covered.

    # Handle _typenode_to_ttype in _build_constants — these call _typenode_to_ttype(val_type),
    # _typenode_to_ttype(type_dict), _typenode_to_ttype(c_type_dict) — all lowercase, covered.

    # But we also need to handle _typenode_to_ttype(VOID_TYPE) or uppercase starts — none exist.

    # _default_value_for_type callers
    content = re.sub(
        r"(?<!def )_default_value_for_type\(", r"_default_value_for_type(pos, ", content
    )

    # _emit_hoisted_placeholders callers
    content = re.sub(
        r"(?<!def )_emit_hoisted_placeholders\(",
        r"_emit_hoisted_placeholders(pos, ",
        content,
    )

    # _backpatch_hoisted callers
    content = re.sub(
        r"(?<!def )_backpatch_hoisted\(", r"_backpatch_hoisted(pos, ", content
    )

    # Step 4: Remove _P0 definition
    content = content.replace("_P0 = Pos(0, 0)\n", "")

    with open(filepath, "w") as f:
        f.write(content)

    # Verify no _P0 remain
    remaining = content.count("_P0")
    print(f"Tier 3 transformation complete. Remaining _P0: {remaining}")


if __name__ == "__main__":
    transform(sys.argv[1])
