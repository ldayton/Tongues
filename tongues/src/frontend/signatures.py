"""Phase 5: Function and method signature collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ast_compat import ASTNode, dict_walk, is_type
from .collection import CollectionCallbacks
from ..ir import (
    FuncInfo,
    InterfaceRef,
    ParamInfo,
    Pointer,
    Slice,
    VOID,
)

if TYPE_CHECKING:
    from ..ir import SymbolTable


def _all_param_args(args: ASTNode) -> list[ASTNode]:
    """Collect all parameter arg nodes from posonlyargs + args + kwonlyargs."""
    result: list[ASTNode] = []
    for a in args.get("posonlyargs", []):
        result.append(a)
    for a in args.get("args", []):
        result.append(a)
    for a in args.get("kwonlyargs", []):
        result.append(a)
    return result


def detect_mutated_params(node: ASTNode) -> set[str]:
    """Detect which parameters are mutated in the function body."""
    mutated = set()
    args = node.get("args", {})
    all_args = _all_param_args(args)
    param_names = {a.get("arg") for a in all_args if a.get("arg") != "self"}
    for stmt in dict_walk(node):
        # param.append(...), param.extend(...), param.clear(), param.pop()
        if is_type(stmt, ["Expr"]) and is_type(stmt.get("value"), ["Call"]):
            call = stmt.get("value")
            if is_type(call.get("func"), ["Attribute"]):
                func = call.get("func")
                if func.get("attr") in ("append", "extend", "clear", "pop"):
                    if (
                        is_type(func.get("value"), ["Name"])
                        and func.get("value", {}).get("id") in param_names
                    ):
                        mutated.add(func.get("value", {}).get("id"))
        # param[i] = ...
        if is_type(stmt, ["Assign"]):
            for target in stmt.get("targets", []):
                if is_type(target, ["Subscript"]):
                    if (
                        is_type(target.get("value"), ["Name"])
                        and target.get("value", {}).get("id") in param_names
                    ):
                        mutated.add(target.get("value", {}).get("id"))
    return mutated


def _make_param(
    arg: ASTNode,
    modifier: str,
    has_default: bool,
    default_node: ASTNode | None,
    mutated_params: set[str],
    callbacks: CollectionCallbacks,
    func_name: str = "",
) -> ParamInfo:
    """Build a ParamInfo from an AST arg node."""
    annotation = arg.get("annotation")
    if annotation is None:
        lineno = arg.get("lineno", 0)
        raise TypeError(
            f"{lineno}:0: [types] parameter '{arg.get('arg')}' missing type annotation in {func_name}()"
        )
    py_type = callbacks.annotation_to_str(annotation)
    typ = callbacks.py_type_to_ir(py_type, True) if py_type else InterfaceRef("any")
    if arg.get("arg") in mutated_params and isinstance(typ, Slice):
        typ = Pointer(typ)
    default_value = None
    if has_default and default_node is not None:
        default_value = callbacks.lower_expr(default_node)
    return ParamInfo(
        name=arg.get("arg"),
        typ=typ,
        has_default=has_default,
        default_value=default_value,
        modifier=modifier,
    )


def extract_func_info(
    node: ASTNode,
    callbacks: CollectionCallbacks,
    is_method: bool = False,
) -> FuncInfo:
    """Extract function signature information."""
    mutated_params = detect_mutated_params(node)
    params: list[ParamInfo] = []
    args = node.get("args", {})
    # Positional-only params (before /)
    posonlyargs = args.get("posonlyargs", [])
    # Regular params (between / and *)
    regular_args = args.get("args", [])
    # Keyword-only params (after *)
    kwonlyargs = args.get("kwonlyargs", [])
    # Defaults apply to the tail of posonlyargs + regular_args
    defaults = args.get("defaults", [])
    # kw_defaults is parallel to kwonlyargs (None entries for no default)
    kw_defaults = args.get("kw_defaults", [])
    # Filter self from regular args (methods)
    non_self_posonly = [a for a in posonlyargs if a.get("arg") != "self"]
    non_self_regular = [a for a in regular_args if a.get("arg") != "self"]
    # defaults covers the tail of posonlyargs + regular args combined
    n_positional = len(non_self_posonly) + len(non_self_regular)
    n_defaults = len(defaults) if defaults else 0
    func_name = node.get("name", "")
    # Positional-only params
    for i, arg in enumerate(non_self_posonly):
        has_default = i >= n_positional - n_defaults
        default_node = None
        if has_default:
            default_idx = i - (n_positional - n_defaults)
            if defaults and default_idx < len(defaults):
                default_node = defaults[default_idx]
        params.append(
            _make_param(
                arg,
                "positional",
                has_default,
                default_node,
                mutated_params,
                callbacks,
                func_name,
            )
        )
    # Regular params
    for i, arg in enumerate(non_self_regular):
        global_i = len(non_self_posonly) + i
        has_default = global_i >= n_positional - n_defaults
        default_node = None
        if has_default:
            default_idx = global_i - (n_positional - n_defaults)
            if defaults and default_idx < len(defaults):
                default_node = defaults[default_idx]
        params.append(
            _make_param(
                arg,
                "pos_or_kw",
                has_default,
                default_node,
                mutated_params,
                callbacks,
                func_name,
            )
        )
    # Keyword-only params
    for i, arg in enumerate(kwonlyargs):
        has_default = False
        default_node = None
        if i < len(kw_defaults) and kw_defaults[i] is not None:
            has_default = True
            default_node = kw_defaults[i]
        params.append(
            _make_param(
                arg,
                "keyword",
                has_default,
                default_node,
                mutated_params,
                callbacks,
                func_name,
            )
        )
    return_type = VOID
    returns = node.get("returns")
    if returns:
        py_return = callbacks.annotation_to_str(returns)
        return_type = callbacks.py_return_type_to_ir(py_return)
    else:
        lineno = node.get("lineno", 0)
        raise TypeError(
            f"{lineno}:0: [types] function '{func_name}' missing return type annotation"
        )
    return FuncInfo(
        name=node.get("name"),
        params=params,
        return_type=return_type,
        is_method=is_method,
    )


def collect_class_methods(
    node: ASTNode,
    symbols: SymbolTable,
    callbacks: CollectionCallbacks,
) -> None:
    """Collect method signatures for a class."""
    info = symbols.structs[node.get("name")]
    for stmt in node.get("body", []):
        if is_type(stmt, ["FunctionDef"]):
            func_info = extract_func_info(stmt, callbacks, is_method=True)
            func_info.is_method = True
            func_info.receiver_type = node.get("name")
            info.methods[stmt.get("name")] = func_info
    # Build method-to-struct mapping for Node subclasses
    if info.is_node:
        excluded_methods = {
            "to_sexp",
            "kind",
            "__init__",
            "__repr__",
            "ToSexp",
            "GetKind",
        }
        for method_name in info.methods:
            if method_name not in excluded_methods:
                symbols.method_to_structs[method_name] = info.name


def collect_signatures(
    tree: ASTNode,
    symbols: SymbolTable,
    callbacks: CollectionCallbacks,
) -> None:
    """Pass 5: Collect function and method signatures."""
    for node in tree.get("body", []):
        if is_type(node, ["FunctionDef"]):
            symbols.functions[node.get("name")] = extract_func_info(node, callbacks)
        elif is_type(node, ["ClassDef"]):
            collect_class_methods(node, symbols, callbacks)
