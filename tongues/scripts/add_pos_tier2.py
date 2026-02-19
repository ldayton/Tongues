"""Tier 2: Add pos: Pos parameter to helper functions and replace _P0 -> pos."""

import re
import sys

TIER2 = [
    "_make_call",
    "_len_expr",
    "_make_named_call",
    "_make_method_call",
    "_bool_to_int",
    "_make_compare_expr",
    "_lower_boolop_chain",
    "_lower_print_call",
    "_lower_struct_constructor",
    "_lower_string_method",
    "_lower_startswith_endswith",
    "_lower_set_method",
    "_lower_isinstance_chain",
    "_lower_bytes_method",
    "_lower_list_method",
    "_lower_dict_method",
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

    # First, find function ranges for tier2 functions and replace _P0 in their bodies
    funcs = find_functions(lines)
    for start, end, fname in funcs:
        if fname not in TIER2:
            continue
        for j in range(start, end):
            if "_P0" in lines[j] and not lines[j].lstrip().startswith("def "):
                lines[j] = lines[j].replace("_P0", "pos")

    content = "\n".join(lines)

    # Step 1: Add pos: Pos parameter to function signatures
    # _make_call
    content = content.replace(
        "def _make_call(name: str, args: list[TExpr]) -> TCall:",
        "def _make_call(pos: Pos, name: str, args: list[TExpr]) -> TCall:",
    )
    # _len_expr
    content = content.replace(
        "def _len_expr(obj: TExpr, obj_type: TypeNode) -> TExpr:",
        "def _len_expr(pos: Pos, obj: TExpr, obj_type: TypeNode) -> TExpr:",
    )
    # _make_named_call
    content = content.replace(
        "def _make_named_call(\n    name: str, pos_args: list[TExpr], named: list[tuple[str, TExpr]]\n) -> TCall:",
        "def _make_named_call(\n    pos: Pos, name: str, pos_args: list[TExpr], named: list[tuple[str, TExpr]]\n) -> TCall:",
    )
    # _make_method_call
    content = content.replace(
        "def _make_method_call(obj: TExpr, method: str, args: list[TExpr]) -> TCall:",
        "def _make_method_call(pos: Pos, obj: TExpr, method: str, args: list[TExpr]) -> TCall:",
    )
    # _bool_to_int
    content = content.replace(
        "def _bool_to_int(expr: TExpr) -> TExpr:",
        "def _bool_to_int(pos: Pos, expr: TExpr) -> TExpr:",
    )
    # _make_compare_expr
    content = content.replace(
        "def _make_compare_expr(left: TExpr, op_node: ASTNode, right: TExpr) -> TExpr:",
        "def _make_compare_expr(pos: Pos, left: TExpr, op_node: ASTNode, right: TExpr) -> TExpr:",
    )
    # _lower_boolop_chain
    content = content.replace(
        "def _lower_boolop_chain(\n    values: list[ASTNode], op_type: str, idx: int, env: _Env, ctx: _LowerCtx\n) -> TExpr:",
        "def _lower_boolop_chain(\n    pos: Pos, values: list[ASTNode], op_type: str, idx: int, env: _Env, ctx: _LowerCtx\n) -> TExpr:",
    )
    # _lower_print_call
    content = content.replace(
        "def _lower_print_call(\n    args: list[ASTNode], keywords: list[ASTNode], env: _Env, ctx: _LowerCtx\n) -> TExpr:",
        "def _lower_print_call(\n    pos: Pos, args: list[ASTNode], keywords: list[ASTNode], env: _Env, ctx: _LowerCtx\n) -> TExpr:",
    )
    # _lower_struct_constructor
    content = content.replace(
        "def _lower_struct_constructor(\n    class_name: str,\n    args: list[ASTNode],\n    keywords: list[ASTNode],\n    env: _Env,\n    ctx: _LowerCtx,\n) -> TExpr:",
        "def _lower_struct_constructor(\n    pos: Pos,\n    class_name: str,\n    args: list[ASTNode],\n    keywords: list[ASTNode],\n    env: _Env,\n    ctx: _LowerCtx,\n) -> TExpr:",
    )
    # _lower_string_method
    content = content.replace(
        "def _lower_string_method(\n    obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx\n) -> TExpr:",
        "def _lower_string_method(\n    pos: Pos, obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx\n) -> TExpr:",
    )
    # _lower_startswith_endswith
    content = content.replace(
        "def _lower_startswith_endswith(\n    func_name: str, obj: TExpr, args: list[ASTNode], env: _Env, ctx: _LowerCtx\n) -> TExpr:",
        "def _lower_startswith_endswith(\n    pos: Pos, func_name: str, obj: TExpr, args: list[ASTNode], env: _Env, ctx: _LowerCtx\n) -> TExpr:",
    )
    # _lower_set_method
    content = content.replace(
        "def _lower_set_method(\n    obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx\n) -> TExpr:",
        "def _lower_set_method(\n    pos: Pos, obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx\n) -> TExpr:",
    )
    # _lower_isinstance_chain
    content = content.replace(
        "def _lower_isinstance_chain(\n    chain: list[tuple[str, str, list[ASTNode], list[ASTNode] | None]],\n    else_body_nodes: list[ASTNode] | None,\n    env: _Env,\n    ctx: _LowerCtx,\n) -> list[TStmt]:",
        "def _lower_isinstance_chain(\n    pos: Pos,\n    chain: list[tuple[str, str, list[ASTNode], list[ASTNode] | None]],\n    else_body_nodes: list[ASTNode] | None,\n    env: _Env,\n    ctx: _LowerCtx,\n) -> list[TStmt]:",
    )
    # _lower_bytes_method
    content = content.replace(
        "def _lower_bytes_method(\n    obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx\n) -> TExpr:",
        "def _lower_bytes_method(\n    pos: Pos, obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx\n) -> TExpr:",
    )
    # _lower_list_method - has extra obj_node param, already has pos from tier1
    content = content.replace(
        "def _lower_list_method(\n    obj: TExpr,\n    obj_node: ASTNode,\n    method: str,\n    args: list[ASTNode],\n    env: _Env,\n    ctx: _LowerCtx,\n) -> TExpr:",
        "def _lower_list_method(\n    pos: Pos,\n    obj: TExpr,\n    obj_node: ASTNode,\n    method: str,\n    args: list[ASTNode],\n    env: _Env,\n    ctx: _LowerCtx,\n) -> TExpr:",
    )
    # _lower_dict_method - same pattern
    content = content.replace(
        "def _lower_dict_method(\n    obj: TExpr,\n    obj_node: ASTNode,\n    method: str,\n    args: list[ASTNode],\n    env: _Env,\n    ctx: _LowerCtx,\n) -> TExpr:",
        "def _lower_dict_method(\n    pos: Pos,\n    obj: TExpr,\n    obj_node: ASTNode,\n    method: str,\n    args: list[ASTNode],\n    env: _Env,\n    ctx: _LowerCtx,\n) -> TExpr:",
    )

    # Step 2: Update callers — use negative lookbehind to avoid matching def lines
    # _make_call
    content = re.sub(r'(?<!def )_make_call\((")', r"_make_call(pos, \1", content)
    content = re.sub(
        r"(?<!def )_make_call\(builtin,", r"_make_call(pos, builtin,", content
    )

    # _len_expr
    content = re.sub(r"(?<!def )_len_expr\(obj,", r"_len_expr(pos, obj,", content)

    # _make_named_call — multiline and inline
    content = re.sub(
        r'(?<!def )_make_named_call\(\n(\s+)"', r'_make_named_call(\n\1pos, "', content
    )
    content = re.sub(
        r'(?<!def )_make_named_call\((")', r"_make_named_call(pos, \1", content
    )

    # _make_method_call
    content = re.sub(
        r"(?<!def )_make_method_call\(obj,", r"_make_method_call(pos, obj,", content
    )
    content = re.sub(
        r"(?<!def )_make_method_call\(result,",
        r"_make_method_call(pos, result,",
        content,
    )

    # _bool_to_int — match calls not on def lines
    content = re.sub(
        r"(?<!def )_bool_to_int\(([a-z])", r"_bool_to_int(pos, \1", content
    )

    # _make_compare_expr
    content = re.sub(
        r"(?<!def )_make_compare_expr\(", r"_make_compare_expr(pos, ", content
    )

    # _lower_boolop_chain
    content = re.sub(
        r"(?<!def )_lower_boolop_chain\(values,",
        r"_lower_boolop_chain(pos, values,",
        content,
    )
    # Also recursive calls within the function itself
    content = re.sub(
        r"(?<!def )_lower_boolop_chain\(pos, values,\s*op_type,\s*idx \+ 1",
        r"_lower_boolop_chain(pos, values, op_type, idx + 1",
        content,
    )

    # _lower_print_call
    content = re.sub(
        r"(?<!def )_lower_print_call\(args,", r"_lower_print_call(pos, args,", content
    )

    # _lower_struct_constructor
    content = re.sub(
        r"(?<!def )_lower_struct_constructor\(fname,",
        r"_lower_struct_constructor(pos, fname,",
        content,
    )

    # _lower_string_method
    content = re.sub(
        r"(?<!def )_lower_string_method\(obj,",
        r"_lower_string_method(pos, obj,",
        content,
    )

    # _lower_startswith_endswith
    content = re.sub(
        r'(?<!def )_lower_startswith_endswith\((")',
        r"_lower_startswith_endswith(pos, \1",
        content,
    )

    # _lower_set_method
    content = re.sub(
        r"(?<!def )_lower_set_method\(obj,", r"_lower_set_method(pos, obj,", content
    )

    # _lower_isinstance_chain
    content = re.sub(
        r"(?<!def )_lower_isinstance_chain\(chain,",
        r"_lower_isinstance_chain(pos, chain,",
        content,
    )

    # _lower_bytes_method
    content = re.sub(
        r"(?<!def )_lower_bytes_method\(obj,", r"_lower_bytes_method(pos, obj,", content
    )

    # _lower_list_method — called with (obj, obj_node, ...)
    content = re.sub(
        r"(?<!def )_lower_list_method\(obj, obj_node,",
        r"_lower_list_method(pos, obj, obj_node,",
        content,
    )

    # _lower_dict_method — called with (obj, obj_node, ...)
    content = re.sub(
        r"(?<!def )_lower_dict_method\(obj, obj_node,",
        r"_lower_dict_method(pos, obj, obj_node,",
        content,
    )

    with open(filepath, "w") as f:
        f.write(content)
    print("Tier 2 transformation complete")


if __name__ == "__main__":
    transform(sys.argv[1])
