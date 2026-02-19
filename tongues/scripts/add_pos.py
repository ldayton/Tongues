"""Transform lowering.py: add pos = _node_pos(node) and replace _P0 -> pos in _lower_* functions."""

import re
import sys

TIER1 = {
    "_lower_constant": "node",
    "_lower_name": "node",
    "_lower_attribute": "node",
    "_lower_binop": "node",
    "_lower_boolop": "node",
    "_lower_compare": "node",
    "_lower_unaryop": "node",
    "_lower_call": "node",
    "_lower_name_call": "node",
    "_lower_method_call": "node",
    "_lower_subscript": "node",
    "_lower_ifexp": "node",
    "_lower_list_literal": "node",
    "_lower_dict_literal": "node",
    "_lower_set_literal": "node",
    "_lower_tuple_literal": "node",
    "_lower_fstring": "node",
    "_lower_as_bool": "node",
    "_lower_ternary_cond": "node",
    "_lower_in_expr": "left_node",
    "_lower_list_from_tuple": "node",
    "_lower_listcomp": "node",
    "_lower_return": "node",
    "_lower_assign": "node",
    "_lower_ann_assign": "node",
    "_lower_aug_assign": "node",
    "_lower_if": "node",
    "_lower_while": "node",
    "_lower_for": "node",
    "_lower_for_range": "target_node",
    "_lower_for_enumerate": "target_node",
    "_lower_try": "node",
    "_lower_raise": "node",
    "_lower_assert": "node",
    "_lower_stmt": "node",
    "_lower_expr_stmt": "node",
    "_lower_expr": "node",
    "_lower_single_compare": "left_node",
    "_lower_degenerate_tuple_compare": "left_node",
    "_lower_set_compare": "left_node",
    "_lower_list_compare": "left_node",
    "_lower_tuple_compare": "left_node",
    "_lower_tuple_concat": "left_node",
    "_lower_dict_literal_typed": "node",
    "_expand_listcomp": "node",
    "_expand_setcomp": "node",
    "_expand_dictcomp": "node",
    "_expand_genexpr_to_set_add": "genexpr",
    "_lower_extend_arg": "arg_node",
    "_ensure_set_expr": "arg_node",
    "_lower_list_method": "obj_node",
    "_lower_dict_method": "obj_node",
    "_method_side_effects": "value_node",
    "_lower_tuple_assign": "target_node",
    "_lower_isinstance_chain": None,  # no node param, handled in tier 2
    "_build_function": "node",
    "_build_method": "node",
    "_build_struct": "node",
}


def find_functions(lines):
    """Find top-level function ranges: [(start, end, name), ...]"""
    funcs = []
    i = 0
    while i < len(lines):
        m = re.match(r"^def (\w+)\(", lines[i])
        if m:
            fname = m.group(1)
            start = i
            # Skip past the function signature (multi-line: find the closing colon)
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
            # Now j points to line after the signature
            # Find end: next top-level def/class or non-indented code
            while j < len(lines):
                line = lines[j]
                # Top-level def or class or decorator
                if re.match(r"^(def |class |@)", line):
                    break
                # Module-level assignment or other code at col 0
                # BUT skip empty lines and comments and section dividers
                if line.strip() and not line[0].isspace() and not line.startswith("#"):
                    break
                j += 1
            funcs.append((start, j, fname))
            i = j
        else:
            i += 1
    return funcs


def find_insert_point(lines, start, end):
    """Find line index after def signature + docstring."""
    i = start
    depth = 0
    while i < end:
        for ch in lines[i]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        if depth <= 0 and ":" in lines[i].split("#")[0]:
            i += 1
            break
        i += 1
    # Skip docstring
    if i < end:
        stripped = lines[i].strip()
        if stripped.startswith('"""'):
            if stripped.count('"""') >= 2 and len(stripped) > 3:
                i += 1
            else:
                i += 1
                while i < end and '"""' not in lines[i]:
                    i += 1
                if i < end:
                    i += 1
    return i


def get_indent(lines, start, end, insert_at):
    for j in range(insert_at, end):
        if lines[j].strip():
            m = re.match(r"^(\s+)", lines[j])
            if m:
                return m.group(1)
    return "    "


def transform(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    funcs = find_functions(lines)
    transformed = 0
    for start, end, fname in reversed(funcs):
        if fname not in TIER1:
            continue
        node_param = TIER1[fname]
        if node_param is None:
            continue
        has_p0 = any("_P0" in lines[j] for j in range(start, end))
        if not has_p0:
            continue
        insert_at = find_insert_point(lines, start, end)
        indent = get_indent(lines, start, end, insert_at)
        pos_line = indent + "pos = _node_pos(" + node_param + ")\n"
        lines.insert(insert_at, pos_line)
        new_end = end + 1
        for j in range(insert_at + 1, new_end):
            if "_P0" in lines[j]:
                lines[j] = lines[j].replace("_P0", "pos")
        transformed += 1

    with open(filepath, "w") as f:
        f.writelines(lines)
    print(f"Transformed {transformed} functions")


if __name__ == "__main__":
    transform(sys.argv[1])
