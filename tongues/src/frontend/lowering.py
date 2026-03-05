"""Phase 10: Lowering — Python AST to Taytsh IR.

Transforms the typed Python dict-AST into Taytsh IR nodes (TModule from
taytsh/ast.py), using type information from phases 5-9 (signatures, fields,
hierarchy, pycheck).

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..taytsh.ast import (
    Ann,
    Pos,
    TArg,
    TAssignStmt,
    TBinaryOp,
    TBoolLit,
    TBreakStmt,
    TBytesLit,
    TCall,
    TCatch,
    TContinueStmt,
    TDecl,
    TDefault,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFieldDecl,
    TFnDecl,
    TFnLit,
    TFloatLit,
    TFuncType,
    TForStmt,
    TIdentType,
    TIfStmt,
    TIndex,
    TIntLit,
    TInterfaceDecl,
    TLetStmt,
    TListLit,
    TListType,
    TMapLit,
    TMapType,
    TMatchCase,
    TMatchStmt,
    TModule,
    TModuleItem,
    TNilLit,
    TOpAssignStmt,
    TOptionalType,
    TParam,
    TPatternNil,
    TPatternType,
    TPrimitive,
    TRange,
    TReturnStmt,
    TRuneLit,
    TSetLit,
    TSetType,
    TSlice,
    TStringLit,
    TStructDecl,
    TStmt,
    TThrowStmt,
    TTernary,
    TTupleAccess,
    TTupleAssignStmt,
    TTupleLit,
    TTupleType,
    TTryStmt,
    TType,
    TUnaryOp,
    TUnionType,
    TVar,
    TWhileStmt,
)
from .typecollect import (
    ClassInfo,
    ParamInfo,
    TypeCollectResult,
    annotation_to_str,
    py_type_to_type_dict,
)
from .hierarchy import HierarchyResult
from .pycheck import PycheckResult
from .types import (
    TypeNode,
    PrimitiveType,
    SliceType,
    MapType,
    SetType,
    TupleType,
    OptionalType,
    PointerType,
    StructRef,
    InterfaceRef,
    FuncType,
    UnionType,
    LiteralType,
    BoolLit,
    IntLit,
    FloatLit,
    StringLit,
    NilLit,
    INT_TYPE,
    BOOL_TYPE,
    STR_TYPE,  # noqa: F401 — used by _collection_element_type (added on main)
    VOID_TYPE,
    contains_any,
    JsonValue,
    JStr,
    JInt,
    JBool,
    JFloat,
    JDict,
    JList,
    JNull,
    ASTNode,
    get_str,
    get_int,
    get_bool,
    get_node,
    get_nodes,
)


TAYTSH_KEYWORDS: set[str] = {
    "bool",
    "break",
    "byte",
    "bytes",
    "case",
    "catch",
    "continue",
    "default",
    "else",
    "enum",
    "false",
    "finally",
    "float",
    "fn",
    "for",
    "if",
    "in",
    "int",
    "interface",
    "let",
    "list",
    "map",
    "match",
    "nil",
    "range",
    "return",
    "rune",
    "set",
    "string",
    "struct",
    "throw",
    "true",
    "this",
    "try",
    "void",
    "while",
}


def _safe_name(name: str) -> str:
    """Rename if name collides with a Taytsh keyword."""
    if name in TAYTSH_KEYWORDS:
        return name + "_"
    return name


def _name_ann(safe: str, original: str) -> Ann:
    """Annotation recording the original Python name, if renamed."""
    if safe == original:
        return {}
    return {"name.original." + safe: original}


# ---------------------------------------------------------------------------
# Type dict to TType conversion
# ---------------------------------------------------------------------------

_LOWER_ANCESTORS: dict[str, list[str]] = {}


def _resolve_struct_union(t: UnionType) -> TypeNode:
    """Resolve a union of related structs to their common ancestor interface."""
    names: list[str] = []
    for v in t.variants:
        if isinstance(v, PointerType) and isinstance(v.target, StructRef):
            names.append(v.target.name)
        elif isinstance(v, StructRef):
            names.append(v.name)
        else:
            return t
    if len(names) < 2:
        return t
    chain0 = _ancestor_chain_hier(names[0])
    common: set[str] = set(chain0)
    for name in names[1:]:
        chain_i = _ancestor_chain_hier(name)
        chain_set: set[str] = set(chain_i)
        new_common: set[str] = set()
        for ancestor in chain0:
            if ancestor in chain_set and ancestor in common:
                new_common.add(ancestor)
        common = new_common
    for ancestor in chain0:
        if ancestor in common:
            return InterfaceRef(ancestor)
    return t


def _ancestor_chain_hier(name: str) -> list[str]:
    """Get ancestor chain from the lowering ancestors dict."""
    chain: list[str] = [name]
    visited: set[str] = set()
    visited.add(name)
    cur = name
    while True:
        parents = _LOWER_ANCESTORS.get(cur)
        if parents is None or not parents:
            break
        parent = parents[0]
        if parent in visited:
            break
        chain.append(parent)
        visited.add(parent)
        cur = parent
    return chain


def _typenode_to_ttype(pos: Pos, t: TypeNode) -> TType:
    """Convert a TypeNode (from signatures/pycheck) to a Taytsh TType node."""
    if isinstance(t, PrimitiveType):
        return TPrimitive(pos, t.kind)
    if isinstance(t, SliceType):
        if isinstance(t.element, PrimitiveType) and t.element.kind == "byte":
            return TPrimitive(pos, "bytes")
        return TListType(pos, _typenode_to_ttype(pos, t.element))
    if isinstance(t, MapType):
        val_ttype: TType = _typenode_to_ttype(pos, t.value)
        if isinstance(val_ttype, TPrimitive) and val_ttype.kind == "void":
            val_ttype = TPrimitive(pos, "nil")
        return TMapType(pos, _typenode_to_ttype(pos, t.key), val_ttype)
    if isinstance(t, SetType):
        return TSetType(pos, _typenode_to_ttype(pos, t.element))
    if isinstance(t, TupleType):
        if t.variadic and t.elements:
            return TListType(pos, _typenode_to_ttype(pos, t.elements[0]))
        parts: list[TType] = []
        for elem in t.elements:
            parts.append(_typenode_to_ttype(pos, elem))
        if len(parts) >= 2:
            return TTupleType(pos, parts)
        if len(parts) == 1:
            return TListType(pos, parts[0])
        return TPrimitive(pos, "error")
    if isinstance(t, OptionalType):
        return TOptionalType(pos, _typenode_to_ttype(pos, t.inner))
    if isinstance(t, PointerType):
        if isinstance(t.target, StructRef):
            return TIdentType(pos, t.target.name)
        return _typenode_to_ttype(pos, t.target)
    if isinstance(t, StructRef):
        if t.name == "dict":
            return TMapType(pos, TPrimitive(pos, "string"), TPrimitive(pos, "error"))
        if t.name == "list":
            return TListType(pos, TPrimitive(pos, "error"))
        return TIdentType(pos, t.name)
    if isinstance(t, InterfaceRef):
        if t.name == "any":
            return TPrimitive(pos, "void")
        return TIdentType(pos, t.name)
    if isinstance(t, FuncType):
        fn_parts: list[TType] = []
        for param in t.params:
            fn_parts.append(_typenode_to_ttype(pos, param))
        fn_parts.append(_typenode_to_ttype(pos, t.ret))
        return TFuncType(pos, fn_parts)
    if isinstance(t, UnionType):
        resolved = _resolve_struct_union(t)
        if not isinstance(resolved, UnionType):
            return _typenode_to_ttype(pos, resolved)
        parts2: list[TType] = []
        for variant in t.variants:
            parts2.append(_typenode_to_ttype(pos, variant))
        if len(parts2) >= 2:
            return TUnionType(pos, parts2)
        return TPrimitive(pos, "error")
    if isinstance(t, LiteralType):
        return _typenode_to_ttype(pos, t.base)
    return TPrimitive(pos, "error")


def _emit_hoisted_placeholders(
    pos: Pos, names: list[str], env: _Env, pre_stmts: list[TStmt]
) -> None:
    """Emit placeholder TLetStmt for hoisted variables, to be back-patched later."""
    for hname in names:
        env.declared.add(hname)
        safe = _safe_name(hname)
        placeholder = TLetStmt(
            pos,
            safe,
            TPrimitive(pos, "int"),
            TIntLit(pos, 0, "0", {}),
            _name_ann(safe, hname),
        )
        env.hoisted_stmts[hname] = placeholder
        pre_stmts.append(placeholder)


def _type_has_zero_value(td: TypeNode) -> bool:
    """Return True if the type has a zero/default value."""
    kind = _type_dict_kind(td)
    if kind in (
        "int",
        "float",
        "string",
        "bool",
        "byte",
        "bytes",
        "rune",
        "Slice",
        "Map",
        "Set",
        "Optional",
    ):
        return True
    if isinstance(td, TupleType) and not td.variadic:
        return all(_type_has_zero_value(e) for e in td.elements)
    return False


def _backpatch_hoisted(pos: Pos, name: str, typ: TypeNode, env: _Env) -> None:
    """Back-patch a hoisted placeholder TLetStmt with the real type."""
    placeholder = env.hoisted_stmts.get(name)
    if placeholder is None:
        return
    ttype = _typenode_to_ttype(pos, typ)
    placeholder.typ = ttype
    if _type_has_zero_value(typ):
        placeholder.value = _default_value_for_type(pos, typ)
    else:
        placeholder.value = None
    env.var_types[name] = typ


def _unwrap_pointer(td: TypeNode) -> TypeNode:
    """Unwrap Pointer wrapper to get the actual type."""
    if isinstance(td, PointerType):
        return td.target
    return td


def _lookup_method_params(
    obj_type: TypeNode, method_name: str, ctx: _LowerCtx
) -> list[ParamInfo] | None:
    """Look up method parameter info from signatures, skipping 'self'."""
    class_name = ""
    if isinstance(obj_type, StructRef):
        class_name = obj_type.name
    elif isinstance(obj_type, InterfaceRef):
        class_name = obj_type.name
    if not class_name:
        return None
    class_methods = ctx.tc_result.methods.get(class_name)
    if class_methods is None:
        return None
    func_info = class_methods.get(method_name)
    if func_info is None:
        return None
    # Skip 'self' parameter
    result: list[ParamInfo] = []
    for param in func_info.params:
        if param.name != "self":
            result.append(param)
    return result


def _resolve_kwargs_to_positional(
    pos: Pos,
    args: list[ASTNode],
    keywords: list[ASTNode],
    params: list[ParamInfo] | None,
    env: "_Env",
    ctx: "_LowerCtx",
) -> list[TArg]:
    """Merge positional args and keyword args into positional TArg list.

    Uses the parameter list to place keyword args in the correct position.
    Falls back to appending kwargs after positional args if no param info.
    """
    if params is not None:
        # Build slots indexed by param position
        n = len(params)
        slots: list[TExpr | None] = []
        _si = 0
        while _si < n:
            slots.append(None)
            _si += 1
        # Place positional args
        idx = 0
        for a in args:
            if idx < n:
                slots[idx] = _lower_expr(a, env, ctx)
                idx += 1
        # Place keyword args by name lookup
        for kw in keywords:
            kw_name = get_str(kw, "arg")
            kw_val = get_node(kw, "value")
            if kw_name and kw_val:
                j = 0
                while j < n:
                    if params[j].name == kw_name:
                        slots[j] = _lower_expr(kw_val, env, ctx)
                        break
                    j += 1
        # Emit non-None slots as positional args
        result: list[TArg] = []
        i = 0
        while i < n:
            slot = slots[i]
            if slot is not None:
                result.append(TArg(pos, None, slot))
            i += 1
        return result
    # No param info — positional args then kwargs in order (best effort)
    result: list[TArg] = []
    for a in args:
        result.append(TArg(pos, None, _lower_expr(a, env, ctx)))
    for kw in keywords:
        kw_name = get_str(kw, "arg")
        kw_val = get_node(kw, "value")
        if kw_name and kw_val:
            result.append(TArg(pos, None, _lower_expr(kw_val, env, ctx)))
    return result


def _type_dict_kind(td: TypeNode) -> str:
    """Get the kind string from a TypeNode for dispatch."""
    if isinstance(td, PrimitiveType):
        return td.kind
    if isinstance(td, PointerType):
        return _type_dict_kind(td.target)
    if isinstance(td, TupleType) and td.variadic:
        return "Slice"
    if isinstance(td, SliceType):
        return "Slice"
    if isinstance(td, MapType):
        return "Map"
    if isinstance(td, SetType):
        return "Set"
    if isinstance(td, TupleType):
        return "Tuple"
    if isinstance(td, OptionalType):
        return "Optional"
    if isinstance(td, StructRef):
        return "StructRef"
    if isinstance(td, InterfaceRef):
        return "InterfaceRef"
    if isinstance(td, FuncType):
        return "FuncType"
    if isinstance(td, UnionType):
        return "Union"
    return "unknown"


def _is_type_dict(td: TypeNode | None, names: list[str]) -> bool:
    """Check if TypeNode matches any of the given kind/type names."""
    if td is None:
        return False
    k = _type_dict_kind(td)
    for name in names:
        if k == name:
            return True
    return False


def _is_optional_type(td: TypeNode) -> bool:
    return isinstance(td, OptionalType)


def _is_variadic_tuple(td: TypeNode) -> bool:
    return isinstance(td, TupleType) and td.variadic


def _is_single_elem_tuple(td: TypeNode) -> bool:
    if not isinstance(td, TupleType):
        return False
    if td.variadic:
        return False
    return len(td.elements) == 1


def _is_set_of_genexpr(node: ASTNode) -> bool:
    """Check if node is set(genexpr) or set(listcomp)."""
    if not _is_ast(node, "Call"):
        return False
    func = get_node(node, "func")
    if not _is_ast(func, "Name") or get_str(func, "id") != "set":
        return False
    args = get_nodes(node, "args")
    if len(args) != 1:
        return False
    return _is_ast(args[0], "GeneratorExp") or _is_ast(args[0], "ListComp")


def _expand_genexpr_to_set_add(
    var_name: str, genexpr: ASTNode, env: _Env, ctx: _LowerCtx
) -> list[TStmt]:
    """Expand set(genexpr) into for loop with Add statements."""
    pos = _node_pos(genexpr)
    elt = get_node(genexpr, "elt")
    generators = get_nodes(genexpr, "generators")
    if not generators:
        return []
    gen = generators[0]
    if not isinstance(gen, dict):
        return []
    target = get_node(gen, "target")
    iter_node = get_node(gen, "iter")
    orig_name = get_str(target, "id")
    target_name = _safe_name(orig_name)
    t_ann = _name_ann(target_name, orig_name)
    # Detect range() iterator → TRange
    if _is_ast(iter_node, "Call"):
        rfunc = get_node(iter_node, "func")
        if _is_ast(rfunc, "Name") and get_str(rfunc, "id") == "range":
            rargs = get_nodes(iter_node, "args")
            range_lowered: list[TExpr] = []
            for ra in rargs:
                range_lowered.append(_lower_expr(ra, env, ctx))
            iter_expr: TExpr = TRange(pos, range_lowered, {})
        else:
            iter_expr = _lower_expr(iter_node, env, ctx)
    else:
        iter_expr = _lower_expr(iter_node, env, ctx)
    comp_env = env.copy()
    comp_env.declared.add(orig_name)
    elt_expr = _lower_expr(elt, comp_env, ctx)
    result_var = TVar(pos, var_name, {})
    add_call = _make_call(pos, "Add", [result_var, elt_expr])
    body: list[TStmt] = [TExprStmt(pos, add_call, {})]
    ifs = get_nodes(gen, "ifs")
    if ifs and isinstance(ifs[0], dict):
        cond = _lower_as_bool(ifs[0], comp_env, ctx)
        body = [TIfStmt(pos, cond, body, None, {})]
    for_stmt = TForStmt(pos, [target_name], iter_expr, body, t_ann)
    return [for_stmt]


def _is_map_type(td: TypeNode) -> bool:
    return isinstance(td, MapType)


def _lower_dict_literal_typed(
    node: ASTNode, type_dict: TypeNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower a Dict literal with known target type, converting bool keys to int when needed."""
    pos = _node_pos(node)
    keys = get_nodes(node, "keys")
    values = get_nodes(node, "values")
    key_is_int = (
        isinstance(type_dict, MapType)
        and isinstance(type_dict.key, PrimitiveType)
        and type_dict.key.kind == "int"
    )
    entries: list[tuple[TExpr, TExpr]] = []
    i = 0
    while i < len(keys):
        k = keys[i]
        v: ASTNode | None = values[i] if i < len(values) else None
        if v is not None:
            if key_is_int and _is_ast(k, "Constant"):
                kval = k.get("value")
                if isinstance(kval, JBool):
                    if kval.value:
                        key_expr = TIntLit(pos, 1, "1", {})
                    else:
                        key_expr = TIntLit(pos, 0, "0", {})
                else:
                    key_expr = _lower_expr(k, env, ctx)
            else:
                key_expr = _lower_expr(k, env, ctx)
            entries.append((key_expr, _lower_expr(v, env, ctx)))
        i += 1
    if not entries:
        return _make_call(pos, "Map", [])
    return TMapLit(pos, entries, {})


def _is_bytes_slice(td: TypeNode) -> bool:
    if isinstance(td, SliceType):
        if isinstance(td.element, PrimitiveType) and td.element.kind == "byte":
            return True
    return False


def _default_value_for_type(pos: Pos, td: TypeNode) -> TExpr:
    """Return a zero/default value for a given TypeNode."""
    kind = _type_dict_kind(td)
    if kind == "float":
        return TFloatLit(pos, 0.0, "0.0", {})
    if kind == "string":
        return TStringLit(pos, "", {})
    if kind == "bool":
        return TBoolLit(pos, False, {})
    if kind == "bytes":
        return TBytesLit(pos, b"", {})
    if isinstance(td, TupleType) and len(td.elements) >= 2:
        parts: list[TExpr] = []
        for elem in td.elements:
            parts.append(_default_value_for_type(pos, elem))
        return TTupleLit(pos, parts, {})
    if (
        isinstance(td, SliceType)
        and isinstance(td.element, PrimitiveType)
        and td.element.kind == "byte"
    ):
        return TBytesLit(pos, b"", {})
    if kind == "Slice":
        return TListLit(pos, [], {})
    if kind == "Map":
        return _make_call(pos, "Map", [])
    if kind == "Set":
        return _make_call(pos, "Set", [])
    if kind == "Optional" or kind == "InterfaceRef" or kind == "StructRef":
        return TNilLit(pos, {})
    return TIntLit(pos, 0, "0", {})


def _types_comparable(left: TypeNode, right: TypeNode) -> bool:
    """Check if two types can be compared for equality."""
    lk = _type_dict_kind(left)
    rk = _type_dict_kind(right)
    if lk == "void" or rk == "void":
        return True
    if lk == "InterfaceRef" or rk == "InterfaceRef":
        return True
    if lk == rk:
        return True
    if (lk == "rune" and rk == "string") or (lk == "string" and rk == "rune"):
        return True
    numeric = {"bool", "int", "float", "byte"}
    if lk in numeric and rk in numeric:
        return True
    if _is_optional_type(left) or _is_optional_type(right):
        return True
    if lk in ("Slice", "Map", "Set", "Tuple") and rk in (
        "Slice",
        "Map",
        "Set",
        "Tuple",
    ):
        return lk == rk
    return False


def _is_struct_type(td: TypeNode) -> bool:
    if isinstance(td, PointerType):
        return isinstance(td.target, StructRef)
    if isinstance(td, StructRef):
        return True
    return False


def _struct_name(td: TypeNode) -> str:
    """Get struct name from TypeNode."""
    if isinstance(td, PointerType):
        if isinstance(td.target, StructRef):
            return td.target.name
    if isinstance(td, StructRef):
        return td.name
    return ""


def _is_interface_type(td: TypeNode) -> bool:
    return isinstance(td, InterfaceRef)


def _interface_name(td: TypeNode) -> str:
    if isinstance(td, InterfaceRef):
        return td.name
    return ""


def _is_non_zero_default(dv: TypeNode | None) -> bool:
    """True when a default value is not the zero value for its type."""
    if dv is None:
        return False
    if isinstance(dv, NilLit):
        return False
    if isinstance(dv, BoolLit):
        return dv.value
    if isinstance(dv, IntLit):
        return dv.value != 0
    if isinstance(dv, FloatLit):
        return dv.value != 0.0
    if isinstance(dv, StringLit):
        return dv.value != ""
    return False


# ---------------------------------------------------------------------------
# Lowering context
# ---------------------------------------------------------------------------


class LoweringError:
    """An error found during lowering."""

    def __init__(
        self, lineno: int, col: int, message: str, source_file: str = ""
    ) -> None:
        self.lineno: int = lineno
        self.col: int = col
        self.message: str = message
        self.source_file: str = source_file

    def __repr__(self) -> str:
        file_prefix = ""
        if self.source_file:
            file_prefix = self.source_file + ":"
        return (
            file_prefix
            + "error:"
            + str(self.lineno)
            + ":"
            + str(self.col)
            + ": [lowering] "
            + self.message
        )


class _LowerCtx:
    """Module-level context for lowering."""

    def __init__(
        self,
        tc_result: TypeCollectResult,
        hier_result: HierarchyResult,
        known_classes: dict[str, str],
        class_bases: dict[str, list[str]],
        pycheck_result: PycheckResult,
    ) -> None:
        self.tc_result: TypeCollectResult = tc_result
        self.hier_result: HierarchyResult = hier_result
        self.known_classes: dict[str, str] = known_classes
        self.class_bases: dict[str, list[str]] = class_bases
        self.errors: list[LoweringError] = []
        self.isinstance_temp_counter: int = 0
        self.comp_counter: int = 0
        self.pycheck_result: PycheckResult = pycheck_result
        self.class_nodes: dict[str, ASTNode] = {}
        self.func_nodes: dict[str, ASTNode] = {}


class _Env:
    """Scope-level environment for variable tracking."""

    def __init__(self) -> None:
        self.var_types: dict[str, TypeNode] = {}
        self.declared: set[str] = set()
        self.return_type: TypeNode = VOID_TYPE
        self.hoisted_stmts: dict[str, TLetStmt] = {}
        self.isinstance_subs: dict[str, str] = {}
        self.pre_stmts: list[TStmt] = []

    def copy(self) -> _Env:
        env = _Env()
        keys = list(self.var_types.keys())
        for key in keys:
            env.var_types[key] = self.var_types[key]
        dkeys = list(self.declared)
        for dkey in dkeys:
            env.declared.add(dkey)
        env.return_type = self.return_type
        env.hoisted_stmts = self.hoisted_stmts
        skeys = list(self.isinstance_subs.keys())
        for skey in skeys:
            env.isinstance_subs[skey] = self.isinstance_subs[skey]
        return env


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _node_pos(node: ASTNode) -> Pos:
    """Extract position from dict-AST node."""
    return Pos(
        get_int(node, "lineno"),
        get_int(node, "col_offset"),
        get_str(node, "_source_file"),
    )


def _is_ast(node: ASTNode, type_name: str) -> bool:
    """Check if node is a dict-AST of given type."""
    if isinstance(node, JDict):
        return get_str(node.entries, "_type") == type_name
    if isinstance(node, dict):
        return get_str(node, "_type") == type_name
    return False


# ---------------------------------------------------------------------------
# Type inference helpers (derive types from signatures and annotations)
# ---------------------------------------------------------------------------


def _func_return_type(ctx: _LowerCtx, name: str) -> TypeNode:
    """Get return type of a function from signatures."""
    info = ctx.tc_result.functions.get(name)
    if info is not None:
        return info.return_type
    return VOID_TYPE


def _is_nil_guard_test(test: ASTNode, body: ASTNode) -> bool:
    """Check if test is `x is not None` (or `x != None`) and body is `x`."""
    if not _is_ast(test, "Compare"):
        return False
    ops = get_nodes(test, "ops")
    comps = get_nodes(test, "comparators")
    if len(ops) != 1 or len(comps) != 1:
        return False
    op = ops[0]
    comp = comps[0]
    if not isinstance(op, dict):
        return False
    op_type = get_str(op, "_type")
    if op_type != "IsNot" and op_type != "NotEq":
        return False
    if not (_is_ast(comp, "Constant") and isinstance(comp.get("value"), JNull)):
        return False
    left = get_node(test, "left")
    if not _is_ast(body, "Name") or not _is_ast(left, "Name"):
        return False
    return get_str(body, "id") == get_str(left, "id")


def _infer_expr_type(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TypeNode:
    """Look up the type of an expression from pycheck results."""
    return _lookup_expr_type(node, env, ctx)


def _adjust_pycheck_type(node: ASTNode, pt: TypeNode, env: _Env) -> TypeNode:
    """Translate pycheck type conventions to lowering conventions."""
    # Literal[x] → base type (lowering doesn't distinguish literals)
    if isinstance(pt, LiteralType):
        pt = pt.base
    # pycheck says str; lowering says rune (loop variable over string)
    if isinstance(pt, PrimitiveType) and pt.kind == "string":
        if _is_ast(node, "Name"):
            vt = env.var_types.get(get_str(node, "id"))
            if isinstance(vt, PrimitiveType) and vt.kind == "rune":
                return vt
    # Normalize bytes: pycheck returns SliceType(byte), lowering uses PrimitiveType("bytes")
    if (
        isinstance(pt, SliceType)
        and isinstance(pt.element, PrimitiveType)
        and pt.element.kind == "byte"
    ):
        pt = PrimitiveType("bytes")
    if not isinstance(node, dict):
        return pt
    t = get_str(node, "_type")
    # Python: bytes[i] returns int, not byte (pycheck uses taytsh convention)
    if t == "Subscript" and isinstance(pt, PrimitiveType) and pt.kind == "byte":
        slc = get_node(node, "slice")
        if not _is_ast(slc, "Slice"):
            return INT_TYPE
    # pycheck returns SetType for dict.keys()/items(), lowering uses SliceType
    if isinstance(pt, SetType) and t == "Call":
        func = get_node(node, "func")
        if (
            isinstance(func, dict)
            and get_str(func, "_type") == "Attribute"
            and get_str(func, "attr") in ("keys", "items")
        ):
            return SliceType(pt.element)
    # Lowering promotes bool to int for bitwise ops (IR uses int arithmetic)
    if t == "BinOp":
        op = get_node(node, "op")
        op_t = get_str(op, "_type")
        if op_t in ("BitAnd", "BitOr", "BitXor", "LShift", "RShift"):
            if isinstance(pt, PrimitiveType) and pt.kind == "bool":
                return INT_TYPE
    return pt


def _is_any_type(t: TypeNode) -> bool:
    return isinstance(t, InterfaceRef) and t.name == "any"


def _is_synthetic(node: ASTNode) -> bool:
    """Check if node was created synthetically during lowering."""
    if not isinstance(node, dict):
        return False
    syn = node.get("_synthetic")
    return isinstance(syn, JBool) and syn.value


def _infer_synthetic_type(node: ASTNode, env: _Env) -> TypeNode:
    """Infer type for synthetic nodes created during lowering."""
    t = get_str(node, "_type")
    if t == "Name":
        name = get_str(node, "id")
        vt = env.var_types.get(name)
        if vt is not None:
            return vt
    elif t == "BoolOp":
        return BOOL_TYPE
    return VOID_TYPE


def _is_empty_collection_node(node: ASTNode) -> bool:
    """Check if node is an empty collection literal or no-arg constructor call."""
    t = get_str(node, "_type")
    if t in ("Dict", "Set", "List"):
        return True
    if t == "Call":
        func = get_node(node, "func")
        if func and get_str(func, "_type") == "Name":
            if get_str(func, "id") in ("set", "dict", "list", "frozenset"):
                args = get_nodes(node, "args")
                if not args:
                    return True
    return False


def _lookup_expr_type(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TypeNode:
    """Look up expression type from pycheck results."""
    if _is_synthetic(node):
        return _infer_synthetic_type(node, env)
    uid_jv = node.get("_uid") if isinstance(node, dict) else None
    if not isinstance(uid_jv, JInt):
        return _env_name_fallback(node, env)
    pt = ctx.pycheck_result.expr_types.get(uid_jv.value)
    if pt is None:
        return _env_name_fallback(node, env)
    if _is_any_type(pt):
        return _env_name_fallback(node, env)
    check_t = pt.ret if isinstance(pt, FuncType) else pt
    if contains_any(check_t):
        if isinstance(node, dict) and _is_empty_collection_node(node):
            return check_t
        return _env_name_fallback(node, env)
    return _adjust_pycheck_type(node, pt, env)


def _env_name_fallback(node: ASTNode, env: _Env) -> TypeNode:
    """For Name nodes without pycheck coverage, check env.var_types."""
    if isinstance(node, dict) and get_str(node, "_type") == "Name":
        vt = env.var_types.get(get_str(node, "id"))
        if vt is not None:
            return vt
    return VOID_TYPE


# ---------------------------------------------------------------------------
# Expression lowering
# ---------------------------------------------------------------------------


def _make_call(pos: Pos, name: str, args: list[TExpr]) -> TCall:
    """Helper to create a simple function call."""
    targs: list[TArg] = []
    for arg in args:
        targs.append(TArg(pos, None, arg))
    return TCall(pos, TVar(pos, name, {}), targs, {})


def _len_expr(pos: Pos, obj: TExpr, obj_type: TypeNode) -> TExpr:
    """Len(obj) for most types, literal int for fixed-size tuples."""
    if isinstance(obj_type, TupleType):
        n = len(obj_type.elements)
        return TIntLit(pos, n, str(n), {})
    return _make_call(pos, "Len", [obj])


def _make_named_call(
    pos: Pos, name: str, pos_args: list[TExpr], named: list[tuple[str, TExpr]]
) -> TCall:
    """Helper to create a function call with named arguments."""
    targs: list[TArg] = []
    for pa in pos_args:
        targs.append(TArg(pos, None, pa))
    for pair in named:
        targs.append(TArg(pos, pair[0], pair[1]))
    return TCall(pos, TVar(pos, name, {}), targs, {})


def _make_method_call(pos: Pos, obj: TExpr, method: str, args: list[TExpr]) -> TCall:
    """Helper to create a method call."""
    targs: list[TArg] = []
    for arg in args:
        targs.append(TArg(pos, None, arg))
    return TCall(pos, TFieldAccess(pos, obj, method, {}), targs, {})


def _lower_expr(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Python expression AST node to a Taytsh expression."""
    pos = _node_pos(node)
    t = get_str(node, "_type")
    if t == "Constant":
        return _lower_constant(node, env, ctx)
    if t == "Name":
        return _lower_name(node, env, ctx)
    if t == "Attribute":
        return _lower_attribute(node, env, ctx)
    if t == "BinOp":
        return _lower_binop(node, env, ctx)
    if t == "BoolOp":
        return _lower_boolop(node, env, ctx)
    if t == "Compare":
        return _lower_compare(node, env, ctx)
    if t == "UnaryOp":
        return _lower_unaryop(node, env, ctx)
    if t == "Call":
        return _lower_call(node, env, ctx)
    if t == "Subscript":
        return _lower_subscript(node, env, ctx)
    if t == "IfExp":
        return _lower_ifexp(node, env, ctx)
    if t == "List":
        return _lower_list_literal(node, env, ctx)
    if t == "Dict":
        return _lower_dict_literal(node, env, ctx)
    if t == "Set":
        return _lower_set_literal(node, env, ctx)
    if t == "Tuple":
        return _lower_tuple_literal(node, env, ctx)
    if t == "JoinedStr":
        return _lower_fstring(node, env, ctx)
    if t == "ListComp" or t == "GeneratorExp":
        return _lower_listcomp(node, env, ctx)
    if t == "SetComp":
        return _lower_setcomp(node, env, ctx)
    if t == "DictComp":
        return _lower_dictcomp(node, env, ctx)
    low_sf = get_str(node, "_source_file")
    ctx.errors.append(
        LoweringError(0, 0, "unsupported expression type '" + str(t) + "'", low_sf)
    )
    return TVar(pos, "__error__", {})


def _lower_constant(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Constant node."""
    pos = _node_pos(node)
    val = node.get("value")
    if isinstance(val, JBool):
        return TBoolLit(pos, val.value, {})
    if isinstance(val, JInt):
        return TIntLit(pos, val.value, str(val.value), {})
    if isinstance(val, JFloat):
        return TFloatLit(pos, val.value, repr(val.value), {})
    if isinstance(val, JStr):
        if get_bool(node, "_is_bytes"):
            byte_vals: list[int] = []
            for ch in val.value:
                byte_vals.append(ord(ch))
            return TBytesLit(pos, bytes(byte_vals), {})
        return TStringLit(pos, val.value, {})
    if isinstance(val, JNull) or val is None:
        return TNilLit(pos, {})
    return TNilLit(pos, {})


def _lower_name(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Name node."""
    pos = _node_pos(node)
    name = get_str(node, "id")
    if name == "True":
        return TBoolLit(pos, True, {})
    if name == "False":
        return TBoolLit(pos, False, {})
    if name == "None":
        return TNilLit(pos, {})
    if name == "_":
        return TNilLit(pos, {})
    if name == "self":
        return TVar(pos, "this", {})
    safe = _safe_name(name)
    return TVar(pos, safe, _name_ann(safe, name))


def _lower_attribute(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower an Attribute access node."""
    pos = _node_pos(node)
    attr = get_str(node, "attr")
    obj_node = get_node(node, "value")
    # Class constant access: ClassName.CONST → Var("ClassName_CONST")
    if _is_ast(obj_node, "Name"):
        obj_name = get_str(obj_node, "id")
        if obj_name in ctx.known_classes and attr.isupper():
            return TVar(pos, obj_name + "_" + attr, {})
        # sys.argv → Args()
        if obj_name == "sys" and attr == "argv":
            return _make_call(pos, "Args", [])
        if obj_name == "sys" and attr == "maxsize":
            return TIntLit(pos, 9223372036854775807, "9223372036854775807", {})
        # sys.stdin.readline() etc are handled in _lower_call
    # sys.stdin / sys.stdout / sys.stderr attribute chains
    if _is_ast(obj_node, "Attribute"):
        inner_obj = get_node(obj_node, "value")
        inner_attr = get_str(obj_node, "attr")
        if _is_ast(inner_obj, "Name") and get_str(inner_obj, "id") == "sys":
            if inner_attr == "stdin" and attr == "buffer":
                # Return a placeholder for sys.stdin.buffer
                return TVar(pos, "__stdin_buffer__", {})
            if inner_attr == "stdout" and attr == "buffer":
                return TVar(pos, "__stdout_buffer__", {})
            if inner_attr == "stderr" and attr == "buffer":
                return TVar(pos, "__stderr_buffer__", {})
    obj = _lower_expr(obj_node, env, ctx)
    return TFieldAccess(pos, obj, attr, {})


def _bool_to_int(pos: Pos, expr: TExpr) -> TExpr:
    """Convert bool expression to int: b ? 1 : 0."""
    return TTernary(
        pos,
        expr,
        TIntLit(pos, 1, "1", {}),
        TIntLit(pos, 0, "0", {}),
        {},
    )


def _coerce_arithmetic(
    pos: Pos,
    left: TExpr,
    right: TExpr,
    left_type: TypeNode | None,
    right_type: TypeNode | None,
) -> tuple[TExpr, TExpr]:
    """Insert bool→int and int→float coercions for arithmetic operands."""
    lt_bool = _is_type_dict(left_type, ["bool"])
    rt_bool = _is_type_dict(right_type, ["bool"])
    if lt_bool:
        left = _bool_to_int(pos, left)
    if rt_bool:
        right = _bool_to_int(pos, right)
    lt_float = _is_type_dict(left_type, ["float"])
    rt_float = _is_type_dict(right_type, ["float"])
    lt_int = _is_type_dict(left_type, ["int"]) or lt_bool
    rt_int = _is_type_dict(right_type, ["int"]) or rt_bool
    if lt_float and rt_int:
        right = _make_call(pos, "IntToFloat", [right])
    elif rt_float and lt_int:
        left = _make_call(pos, "IntToFloat", [left])
    return left, right


def _coerce_bitwise(
    pos: Pos,
    left: TExpr,
    right: TExpr,
    left_type: TypeNode | None,
    right_type: TypeNode | None,
) -> tuple[TExpr, TExpr]:
    """Insert bool→int coercions for bitwise operands."""
    if _is_type_dict(left_type, ["bool"]):
        left = _bool_to_int(pos, left)
    if _is_type_dict(right_type, ["bool"]):
        right = _bool_to_int(pos, right)
    return left, right


def _lower_binop(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a BinOp node."""
    pos = _node_pos(node)
    op_node = get_node(node, "op")
    op_type = get_str(op_node, "_type")
    left_node = get_node(node, "left")
    right_node = get_node(node, "right")
    left = _lower_expr(left_node, env, ctx)
    right = _lower_expr(right_node, env, ctx)
    left_type = _infer_expr_type(left_node, env, ctx)
    right_type = _infer_expr_type(right_node, env, ctx)
    if op_type == "Add":
        if _is_type_dict(left_type, ["string"]) or _is_type_dict(
            right_type, ["string"]
        ):
            if _is_type_dict(left_type, ["rune"]):
                left = _make_call(pos, "ToString", [left])
            if _is_type_dict(right_type, ["rune"]):
                right = _make_call(pos, "ToString", [right])
            return _make_call(pos, "Concat", [left, right])
        if _is_type_dict(left_type, ["bytes"]) or _is_type_dict(right_type, ["bytes"]):
            return _make_call(pos, "Concat", [left, right])
        if _is_type_dict(left_type, ["Slice"]) or _is_type_dict(right_type, ["Slice"]):
            return _make_call(pos, "Concat", [left, right])
        if _is_type_dict(left_type, ["Tuple"]) or _is_type_dict(right_type, ["Tuple"]):
            return _lower_tuple_concat(left_node, right_node, env, ctx)
        left, right = _coerce_arithmetic(pos, left, right, left_type, right_type)
        return TBinaryOp(pos, "+", left, right, {})
    if op_type == "Sub":
        if _is_type_dict(left_type, ["Set"]):
            return _make_call(pos, "Difference", [left, right])
        if _is_type_dict(left_type, ["Slice"]) and _is_type_dict(right_type, ["Slice"]):
            return _make_call(
                pos,
                "Difference",
                [
                    _make_call(pos, "SetFromList", [left]),
                    _make_call(pos, "SetFromList", [right]),
                ],
            )
        left, right = _coerce_arithmetic(pos, left, right, left_type, right_type)
        return TBinaryOp(pos, "-", left, right, {})
    if op_type == "Mult":
        if _is_type_dict(left_type, ["string", "bytes", "Slice", "Tuple"]):
            return _make_call(pos, "Repeat", [left, right])
        if _is_type_dict(right_type, ["string", "bytes", "Slice", "Tuple"]):
            return _make_call(pos, "Repeat", [right, left])
        left, right = _coerce_arithmetic(pos, left, right, left_type, right_type)
        return TBinaryOp(pos, "*", left, right, {})
    if op_type == "FloorDiv":
        left, right = _coerce_arithmetic(pos, left, right, left_type, right_type)
        return _make_call(pos, "FloorDiv", [left, right])
    if op_type == "Div":
        # True division: IntToFloat(a) / IntToFloat(b)
        if _is_type_dict(left_type, ["int", "bool"]):
            if _is_type_dict(left_type, ["bool"]):
                left = _bool_to_int(pos, left)
            left = _make_call(pos, "IntToFloat", [left])
        if _is_type_dict(right_type, ["int", "bool"]):
            if _is_type_dict(right_type, ["bool"]):
                right = _bool_to_int(pos, right)
            right = _make_call(pos, "IntToFloat", [right])
        return TBinaryOp(pos, "/", left, right, {})
    if op_type == "Mod":
        left, right = _coerce_arithmetic(pos, left, right, left_type, right_type)
        return _make_call(pos, "PythonMod", [left, right])
    if op_type == "Pow":
        left, right = _coerce_arithmetic(pos, left, right, left_type, right_type)
        return _make_call(pos, "Pow", [left, right])
    if op_type == "BitAnd":
        if _is_type_dict(left_type, ["Set"]):
            return _make_call(pos, "Intersection", [left, right])
        if _is_type_dict(left_type, ["Slice"]) and _is_type_dict(right_type, ["Slice"]):
            return _make_call(
                pos,
                "Intersection",
                [
                    _make_call(pos, "SetFromList", [left]),
                    _make_call(pos, "SetFromList", [right]),
                ],
            )
        left, right = _coerce_bitwise(pos, left, right, left_type, right_type)
        return TBinaryOp(pos, "&", left, right, {})
    if op_type == "BitOr":
        # Dict merge: a | b
        if _is_type_dict(left_type, ["Map"]):
            return _make_call(pos, "Merge", [left, right])
        # Set union: a | b
        if _is_type_dict(left_type, ["Set"]):
            return _make_call(pos, "Union", [left, right])
        if _is_type_dict(left_type, ["Slice"]) and _is_type_dict(right_type, ["Slice"]):
            return _make_call(
                pos,
                "Union",
                [
                    _make_call(pos, "SetFromList", [left]),
                    _make_call(pos, "SetFromList", [right]),
                ],
            )
        left, right = _coerce_bitwise(pos, left, right, left_type, right_type)
        return TBinaryOp(pos, "|", left, right, {})
    if op_type == "BitXor":
        if _is_type_dict(left_type, ["Set"]):
            u = _make_call(pos, "Union", [left, right])
            i = _make_call(pos, "Intersection", [left, right])
            return _make_call(pos, "Difference", [u, i])
        if _is_type_dict(left_type, ["Slice"]) and _is_type_dict(right_type, ["Slice"]):
            ls = _make_call(pos, "SetFromList", [left])
            rs = _make_call(pos, "SetFromList", [right])
            u = _make_call(pos, "Union", [ls, rs])
            i = _make_call(pos, "Intersection", [ls, rs])
            return _make_call(pos, "Difference", [u, i])
        left, right = _coerce_bitwise(pos, left, right, left_type, right_type)
        return TBinaryOp(pos, "^", left, right, {})
    if op_type == "LShift":
        left, right = _coerce_bitwise(pos, left, right, left_type, right_type)
        return TBinaryOp(pos, "<<", left, right, {})
    if op_type == "RShift":
        left, right = _coerce_bitwise(pos, left, right, left_type, right_type)
        return TBinaryOp(pos, ">>", left, right, {})
    return TBinaryOp(pos, "+", left, right, {})


def _lower_tuple_concat(
    left_node: ASTNode, right_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Inline tuple concatenation: (a, b) + (c, d) → (a, b, c, d)."""
    pos = _node_pos(left_node)
    left_is_literal = _is_ast(left_node, "Tuple")
    right_is_literal = _is_ast(right_node, "Tuple")
    if not left_is_literal or not right_is_literal:
        left = _lower_expr(left_node, env, ctx)
        right = _lower_expr(right_node, env, ctx)
        return _make_call(pos, "Concat", [left, right])
    left_elts = get_nodes(left_node, "elts")
    right_elts = get_nodes(right_node, "elts")
    all_elts: list[ASTNode] = []
    for e in left_elts:
        all_elts.append(e)
    for e in right_elts:
        all_elts.append(e)
    if not all_elts:
        return TNilLit(pos, {})
    elements: list[TExpr] = []
    for e in all_elts:
        elements.append(_lower_expr(e, env, ctx))
    if len(elements) == 1:
        return TListLit(pos, elements, {})
    return TTupleLit(pos, elements, {})


def _lower_boolop_chain(
    pos: Pos, values: list[ASTNode], op_type: str, idx: int, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Recursively lower non-bool and/or chain to nested ternaries."""
    v = values[idx]
    if not isinstance(v, dict):
        return TBoolLit(pos, True, {})
    if idx == len(values) - 1:
        return _lower_expr(v, env, ctx)
    left = _lower_expr(v, env, ctx)
    if isinstance(left, TBoolLit):
        if op_type == "And":
            if left.value:
                return _lower_boolop_chain(pos, values, op_type, idx + 1, env, ctx)
            return left
        else:
            if left.value:
                return left
            return _lower_boolop_chain(pos, values, op_type, idx + 1, env, ctx)
    cond = _lower_as_bool(v, env, ctx)
    rest = _lower_boolop_chain(pos, values, op_type, idx + 1, env, ctx)
    if op_type == "And":
        return TTernary(pos, cond, rest, left, {})
    return TTernary(pos, cond, left, rest, {})


def _lower_boolop(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a BoolOp node (and/or)."""
    pos = _node_pos(node)
    op_node = get_node(node, "op")
    op_type = get_str(op_node, "_type")
    values = get_nodes(node, "values")
    if len(values) < 2:
        if len(values) == 1 and isinstance(values[0], dict):
            return _lower_expr(values[0], env, ctx)
        return TBoolLit(pos, True, {})
    # Check if all operands are bool — if so, use && / ||
    all_bool = True
    for v in values:
        vt = _infer_expr_type(v, env, ctx)
        if not _is_type_dict(vt, ["bool"]):
            all_bool = False
    if all_bool:
        op_str = "&&" if op_type == "And" else "||"
        first = values[0]
        if not isinstance(first, dict):
            return TBoolLit(pos, True, {})
        result: TExpr = _lower_as_bool(first, env, ctx)
        for v in values[1:]:
            right = _lower_as_bool(v, env, ctx)
            result = TBinaryOp(pos, op_str, result, right, {})
        return result
    # Non-bool operands: use ternaries for Python short-circuit semantics
    # a and b → truthy(a) ? b : a
    # a or b  → truthy(a) ? a : b
    # Build right-to-left for chaining: a and b and c → truthy(a) ? (truthy(b) ? c : b) : a
    return _lower_boolop_chain(pos, values, op_type, 0, env, ctx)


def _lower_compare(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Compare node."""
    pos = _node_pos(node)
    left_node = get_node(node, "left")
    ops = get_nodes(node, "ops")
    comparators = get_nodes(node, "comparators")
    if not ops or not comparators:
        return TBoolLit(pos, True, {})
    # Single comparison
    if len(ops) == 1:
        op_node = ops[0]
        comp_node = comparators[0]
        if not isinstance(op_node, dict) or not isinstance(comp_node, dict):
            return TBoolLit(pos, True, {})
        return _lower_single_compare(left_node, op_node, comp_node, env, ctx)
    # Chained comparison: a < b < c → a < b && b < c
    left = _lower_expr(left_node, env, ctx)
    parts: list[TExpr] = []
    prev_expr = left
    for op_n, comp_n in zip(ops, comparators):
        if not isinstance(op_n, dict) or not isinstance(comp_n, dict):
            continue
        right = _lower_expr(comp_n, env, ctx)
        cmp = _make_compare_expr(pos, prev_expr, op_n, right)
        parts.append(cmp)
        prev_expr = right
    if not parts:
        return TBoolLit(pos, True, {})
    result = parts[0]
    for part in parts[1:]:
        result = TBinaryOp(pos, "&&", result, part, {})
    return result


def _lower_degenerate_tuple_compare(
    left_node: ASTNode,
    op_type: str,
    right_node: ASTNode,
    left_elts: list[ASTNode] | None,
    right_elts: list[ASTNode] | None,
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Handle comparisons involving 0 or 1 element tuples."""
    pos = _node_pos(left_node)
    left_len = len(left_elts) if left_elts is not None else -1
    right_len = len(right_elts) if right_elts is not None else -1
    # () vs () → equal
    if left_len == 0 and right_len == 0:
        if op_type in ("Eq", "LtE", "GtE"):
            return TBoolLit(pos, True, {})
        return TBoolLit(pos, False, {})
    # () vs non-empty → empty is less
    if left_len == 0:
        if op_type in ("Lt", "LtE"):
            return TBoolLit(pos, True, {})
        if op_type in ("Eq", "Gt", "GtE"):
            return TBoolLit(pos, False, {})
        if op_type == "NotEq":
            return TBoolLit(pos, True, {})
    if right_len == 0:
        if op_type in ("Gt", "GtE"):
            return TBoolLit(pos, True, {})
        if op_type in ("Eq", "Lt", "LtE"):
            return TBoolLit(pos, False, {})
        if op_type == "NotEq":
            return TBoolLit(pos, True, {})
    # (x,) vs (y,) → compare x and y directly
    if (
        left_elts is not None
        and right_elts is not None
        and left_len == 1
        and right_len == 1
    ):
        le = left_elts[0]
        re = right_elts[0]
        if isinstance(le, dict) and isinstance(re, dict):
            return _lower_single_compare(le, {"_type": JStr(op_type)}, re, env, ctx)
    # (x,) vs (a, b) or vice versa → compare first elements, then length
    if (
        left_elts is not None
        and right_elts is not None
        and left_len == 1
        and right_len >= 2
    ):
        le = left_elts[0]
        if isinstance(le, dict) and isinstance(right_elts[0], dict):
            first_cmp = _lower_single_compare(
                le, {"_type": JStr("Lt")}, right_elts[0], env, ctx
            )
            first_eq = _lower_single_compare(
                le, {"_type": JStr("Eq")}, right_elts[0], env, ctx
            )
            if op_type == "Lt":
                return TBinaryOp(pos, "||", first_cmp, first_eq, {})
            if op_type == "LtE":
                return TBoolLit(pos, True, {})
    # Fallback: lower normally (may cause type errors for unsupported cases)
    left = _lower_expr(left_node, env, ctx)
    right = _lower_expr(right_node, env, ctx)
    op_dict: ASTNode = {"_type": JStr(op_type)}
    return _make_compare_expr(pos, left, op_dict, right)


def _lower_set_compare(
    left_node: ASTNode, op_type: str, right_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Desugar set ordering operators to subset/superset checks."""
    pos = _node_pos(left_node)
    left = _lower_expr(left_node, env, ctx)
    right = _lower_expr(right_node, env, ctx)
    # a <= b (subset): Len(Difference(a, b)) == 0
    # a < b (proper subset): a != b && Len(Difference(a, b)) == 0
    # a >= b (superset): Len(Difference(b, a)) == 0
    # a > b (proper superset): a != b && Len(Difference(b, a)) == 0
    if op_type in ("LtE", "Lt"):
        diff = _make_call(pos, "Difference", [left, right])
    else:
        diff = _make_call(pos, "Difference", [right, left])
    is_sub = TBinaryOp(
        pos,
        "==",
        _make_call(pos, "Len", [diff]),
        TIntLit(pos, 0, "0", {}),
        {},
    )
    if op_type in ("LtE", "GtE"):
        return is_sub
    # proper subset/superset: also require not equal
    not_eq = TBinaryOp(pos, "!=", left, right, {})
    return TBinaryOp(pos, "&&", not_eq, is_sub, {})


def _lower_list_compare(
    left_node: ASTNode, op_type: str, right_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Desugar list ordering to ListCompare builtin."""
    pos = _node_pos(left_node)
    left = _lower_expr(left_node, env, ctx)
    right = _lower_expr(right_node, env, ctx)
    cmp = _make_call(pos, "ListCompare", [left, right])
    zero = TIntLit(pos, 0, "0", {})
    op_map: dict[str, str] = {"Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">="}
    return TBinaryOp(pos, op_map[op_type], cmp, zero, {})


def _lower_tuple_compare(
    left_node: ASTNode, op_type: str, right_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Desugar tuple ordering into element-by-element comparison."""
    pos = _node_pos(left_node)
    left_elts: list[ASTNode] = []
    right_elts: list[ASTNode] = []
    if _is_ast(left_node, "Tuple"):
        left_elts = [e for e in get_nodes(left_node, "elts") if isinstance(e, dict)]
    if _is_ast(right_node, "Tuple"):
        right_elts = [e for e in get_nodes(right_node, "elts") if isinstance(e, dict)]
    # If not literal tuples, fall back to equality comparison
    if not left_elts and not _is_ast(left_node, "Tuple"):
        left = _lower_expr(left_node, env, ctx)
        right = _lower_expr(right_node, env, ctx)
        return TBoolLit(pos, False, {})
    if not right_elts and not _is_ast(right_node, "Tuple"):
        left = _lower_expr(left_node, env, ctx)
        right = _lower_expr(right_node, env, ctx)
        return TBoolLit(pos, False, {})
    min_len = min(len(left_elts), len(right_elts))
    # Build from the innermost comparison outward
    # For <: a0 < b0 || (a0 == b0 && (a1 < b1 || (a1 == b1 && ... tail)))
    # tail: if same length, False for <, True for <=; if different length, shorter < longer
    if op_type in ("Lt", "LtE"):
        if len(left_elts) < len(right_elts):
            tail: TExpr = TBoolLit(pos, True, {})
        elif len(left_elts) > len(right_elts):
            tail = TBoolLit(pos, False, {})
        else:
            tail = TBoolLit(pos, op_type == "LtE", {})
        cmp_op = "Lt"
    else:
        # Gt, GtE: reverse — a > b is b < a
        if len(left_elts) > len(right_elts):
            tail = TBoolLit(pos, True, {})
        elif len(left_elts) < len(right_elts):
            tail = TBoolLit(pos, False, {})
        else:
            tail = TBoolLit(pos, op_type == "GtE", {})
        cmp_op = "Gt"
    result = tail
    i = min_len - 1
    while i >= 0:
        le = left_elts[i]
        re = right_elts[i]
        a = _lower_expr(le, env, ctx)
        b = _lower_expr(re, env, ctx)
        lt_type = _infer_expr_type(le, env, ctx)
        rt_type = _infer_expr_type(re, env, ctx)
        a_c, b_c = _coerce_compare(pos, a, b, lt_type, rt_type)
        op_str = "<" if cmp_op == "Lt" else ">"
        elem_lt = TBinaryOp(pos, op_str, a_c, b_c, {})
        a2 = _lower_expr(le, env, ctx)
        b2 = _lower_expr(re, env, ctx)
        a2_c, b2_c = _coerce_compare(pos, a2, b2, lt_type, rt_type)
        elem_eq = TBinaryOp(pos, "==", a2_c, b2_c, {})
        inner = TBinaryOp(pos, "&&", elem_eq, result, {})
        result = TBinaryOp(pos, "||", elem_lt, inner, {})
        i -= 1
    return result


def _lower_single_compare(
    left_node: ASTNode, op_node: ASTNode, comp_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower a single comparison operation."""
    pos = _node_pos(left_node)
    op_type = get_str(op_node, "_type")
    # is None → IsNil(x) (keep IsNil to avoid checker narrowing to nil in then-body)
    if op_type == "Is":
        if _is_ast(comp_node, "Constant") and isinstance(comp_node.get("value"), JNull):
            left = _lower_expr(left_node, env, ctx)
            return _make_call(pos, "IsNil", [left])
    # is not None → x != nil (TVar) or !IsNil(expr)
    if op_type == "IsNot":
        if _is_ast(comp_node, "Constant") and isinstance(comp_node.get("value"), JNull):
            left = _lower_expr(left_node, env, ctx)
            if isinstance(left, TVar):
                return TBinaryOp(pos, "!=", left, TNilLit(pos, {}), {})
            return TUnaryOp(pos, "!", _make_call(pos, "IsNil", [left]), {})
    # isinstance check in compare context
    # in operator
    if op_type == "In":
        return _lower_in_expr(left_node, comp_node, env, ctx)
    if op_type == "NotIn":
        inner = _lower_in_expr(left_node, comp_node, env, ctx)
        return TUnaryOp(pos, "!", inner, {})
    # Degenerate tuple comparisons: (), (x,) — can't represent in Taytsh
    if _is_ast(left_node, "Tuple") or _is_ast(comp_node, "Tuple"):
        left_elts: list[ASTNode] | None = (
            get_nodes(left_node, "elts") if _is_ast(left_node, "Tuple") else None
        )
        right_elts: list[ASTNode] | None = (
            get_nodes(comp_node, "elts") if _is_ast(comp_node, "Tuple") else None
        )
        left_len = len(left_elts) if left_elts is not None else 2
        right_len = len(right_elts) if right_elts is not None else 2
        if left_len < 2 or right_len < 2:
            return _lower_degenerate_tuple_compare(
                left_node, op_type, comp_node, left_elts, right_elts, env, ctx
            )
    # None == X or X == None (non-optional) → false/true
    left_is_none = _is_ast(left_node, "Constant") and isinstance(
        left_node.get("value"), JNull
    )
    right_is_none = _is_ast(comp_node, "Constant") and isinstance(
        comp_node.get("value"), JNull
    )
    if left_is_none != right_is_none:
        # One side is None, other is not
        other_type = _infer_expr_type(
            comp_node if left_is_none else left_node, env, ctx
        )
        if not _is_optional_type(other_type):
            if op_type in ("Eq",):
                return TBoolLit(pos, False, {})
            if op_type in ("NotEq",):
                return TBoolLit(pos, True, {})
    left_type = _infer_expr_type(left_node, env, ctx)
    right_type = _infer_expr_type(comp_node, env, ctx)
    # rune vs single-char string literal → promote string to rune
    if op_type in ("Eq", "NotEq"):
        promoted = _maybe_promote_rune_compare(
            pos, left_node, comp_node, left_type, right_type, op_node, env, ctx
        )
        if promoted is not None:
            return promoted
    # Cross-type equality for incompatible types → false/true
    if op_type in ("Eq", "NotEq") and not _types_comparable(left_type, right_type):
        return TBoolLit(pos, op_type == "NotEq", {})
    # Set ordering: <= (subset), < (proper subset), >= (superset), > (proper superset)
    if (
        op_type in ("LtE", "Lt", "GtE", "Gt")
        and _is_type_dict(left_type, ["Set"])
        and _is_type_dict(right_type, ["Set"])
    ):
        return _lower_set_compare(left_node, op_type, comp_node, env, ctx)
    # List ordering: lexicographic via ListCompare builtin
    if (
        op_type in ("LtE", "Lt", "GtE", "Gt")
        and _is_type_dict(left_type, ["Slice"])
        and _is_type_dict(right_type, ["Slice"])
    ):
        return _lower_list_compare(left_node, op_type, comp_node, env, ctx)
    # Tuple ordering: element-by-element desugaring
    if op_type in ("LtE", "Lt", "GtE", "Gt"):
        lt = _type_dict_kind(left_type)
        rt = _type_dict_kind(right_type)
        if lt == "Tuple" or rt == "Tuple":
            return _lower_tuple_compare(left_node, op_type, comp_node, env, ctx)
    left = _lower_expr(left_node, env, ctx)
    right = _lower_expr(comp_node, env, ctx)
    left, right = _coerce_compare(pos, left, right, left_type, right_type)
    return _make_compare_expr(pos, left, op_node, right)


def _maybe_promote_rune_compare(
    pos: Pos,
    left_node: ASTNode,
    right_node: ASTNode,
    left_type: TypeNode | None,
    right_type: TypeNode | None,
    op_node: ASTNode,
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr | None:
    """Promote single-char string literal to rune when compared with rune."""
    lt_rune = _is_type_dict(left_type, ["rune"])
    rt_rune = _is_type_dict(right_type, ["rune"])
    if lt_rune:
        rch = _single_char_str_value(right_node)
        if rch is not None:
            lhs = _lower_expr(left_node, env, ctx)
            return _make_compare_expr(pos, lhs, op_node, TRuneLit(pos, rch, {}))
    if rt_rune:
        lch = _single_char_str_value(left_node)
        if lch is not None:
            rhs = _lower_expr(right_node, env, ctx)
            return _make_compare_expr(pos, TRuneLit(pos, lch, {}), op_node, rhs)
    return None


def _single_char_str_value(node: ASTNode) -> str | None:
    """Return the character if node is a single-char string constant, else None."""
    if not _is_ast(node, "Constant"):
        return None
    val = node.get("value")
    if not isinstance(val, JStr):
        return None
    if len(val.value) != 1:
        return None
    return val.value


def _coerce_compare(
    pos: Pos,
    left: TExpr,
    right: TExpr,
    left_type: TypeNode | None,
    right_type: TypeNode | None,
) -> tuple[TExpr, TExpr]:
    """Insert coercions for comparison operands."""
    lt_bool = _is_type_dict(left_type, ["bool"])
    rt_bool = _is_type_dict(right_type, ["bool"])
    lt_int = _is_type_dict(left_type, ["int"])
    rt_int = _is_type_dict(right_type, ["int"])
    lt_float = _is_type_dict(left_type, ["float"])
    rt_float = _is_type_dict(right_type, ["float"])
    # bool vs int → convert bool to int
    if lt_bool and rt_int:
        left = _bool_to_int(pos, left)
    elif rt_bool and lt_int:
        right = _bool_to_int(pos, right)
    elif lt_bool and rt_bool:
        left = _bool_to_int(pos, left)
        right = _bool_to_int(pos, right)
    # float vs int → convert int to float
    elif lt_float and rt_int:
        right = _make_call(pos, "IntToFloat", [right])
    elif rt_float and lt_int:
        left = _make_call(pos, "IntToFloat", [left])
    # bool vs float → bool to int to float
    elif lt_bool and rt_float:
        left = _make_call(pos, "IntToFloat", [_bool_to_int(pos, left)])
    elif rt_bool and lt_float:
        right = _make_call(pos, "IntToFloat", [_bool_to_int(pos, right)])
    # Note: byte vs int coercion is handled by _lower_subscript wrapping
    # bytes indexing with ByteToInt, so we don't need to coerce here
    # rune vs string → convert rune to string
    lt_rune = _is_type_dict(left_type, ["rune"])
    rt_rune = _is_type_dict(right_type, ["rune"])
    lt_str = _is_type_dict(left_type, ["string"])
    rt_str = _is_type_dict(right_type, ["string"])
    if lt_rune and rt_str:
        left = _make_call(pos, "ToString", [left])
    elif rt_rune and lt_str:
        right = _make_call(pos, "ToString", [right])
    return left, right


def _make_compare_expr(pos: Pos, left: TExpr, op_node: ASTNode, right: TExpr) -> TExpr:
    """Create a comparison expression from lowered operands."""
    op_type = get_str(op_node, "_type")
    op_map: dict[str, str] = {
        "Eq": "==",
        "NotEq": "!=",
        "Lt": "<",
        "LtE": "<=",
        "Gt": ">",
        "GtE": ">=",
    }
    op_str = op_map.get(op_type, "==")
    return TBinaryOp(pos, op_str, left, right, {})


def _lower_in_expr(
    left_node: ASTNode, right_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower 'x in collection' expression."""
    pos = _node_pos(left_node)
    # x in (1, 2, 3) → x == 1 || x == 2 || x == 3
    if _is_ast(right_node, "Tuple"):
        elts = get_nodes(right_node, "elts")
        left = _lower_expr(left_node, env, ctx)
        if not elts:
            return TBoolLit(pos, False, {})
        left_type = _infer_expr_type(left_node, env, ctx)
        lt_rune = _is_type_dict(left_type, ["rune"])
        parts: list[TExpr] = []
        for e in elts:
            if lt_rune:
                rch = _single_char_str_value(e)
                if rch is not None:
                    parts.append(TBinaryOp(pos, "==", left, TRuneLit(pos, rch, {}), {}))
                    continue
            right = _lower_expr(e, env, ctx)
            parts.append(TBinaryOp(pos, "==", left, right, {}))
        result: TExpr = parts[0]
        for part in parts[1:]:
            result = TBinaryOp(pos, "||", result, part, {})
        return result
    # x in collection → Contains(collection, x)
    # Type mismatch on map keys → always false
    left_type = _infer_expr_type(left_node, env, ctx)
    right_type = _infer_expr_type(right_node, env, ctx)
    if _is_type_dict(right_type, ["Map"]):
        if isinstance(right_type, MapType):
            lt = left_type
            if isinstance(lt, OptionalType):
                lt = lt.inner
            lk = _type_dict_kind(lt)
            rk = _type_dict_kind(right_type.key)
            if lk and rk and lk != rk:
                return TBoolLit(pos, False, {})
    left = _lower_expr(left_node, env, ctx)
    right = _lower_expr(right_node, env, ctx)
    return _make_call(pos, "Contains", [right, left])


def _lower_unaryop(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a UnaryOp node."""
    pos = _node_pos(node)
    op_node = get_node(node, "op")
    op_type = get_str(op_node, "_type")
    operand_node = get_node(node, "operand")
    if op_type == "Not":
        operand_type = _infer_expr_type(operand_node, env, ctx)
        # not None → True
        if _is_ast(operand_node, "Constant") and isinstance(
            operand_node.get("value"), JNull
        ):
            return TBoolLit(pos, True, {})
        if _is_optional_type(operand_type):
            # not x (optional) → IsNil(x)
            operand = _lower_expr(operand_node, env, ctx)
            return _make_call(pos, "IsNil", [operand])
        if _is_type_dict(operand_type, ["bool"]):
            operand = _lower_expr(operand_node, env, ctx)
            return TUnaryOp(pos, "!", operand, {})
        # Non-bool not: lower as bool then negate
        return TUnaryOp(pos, "!", _lower_as_bool(operand_node, env, ctx), {})
    if op_type == "USub":
        operand_type = _infer_expr_type(operand_node, env, ctx)
        operand = _lower_expr(operand_node, env, ctx)
        if _is_type_dict(operand_type, ["bool"]):
            operand = _bool_to_int(pos, operand)
        return TUnaryOp(pos, "-", operand, {})
    if op_type == "UAdd":
        return _lower_expr(operand_node, env, ctx)
    if op_type == "Invert":
        operand_type = _infer_expr_type(operand_node, env, ctx)
        operand = _lower_expr(operand_node, env, ctx)
        if _is_type_dict(operand_type, ["bool"]):
            operand = _bool_to_int(pos, operand)
        return TUnaryOp(pos, "~", operand, {})
    return _lower_expr(operand_node, env, ctx)


def _lower_call(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Call node."""
    pos = _node_pos(node)
    func_node = get_node(node, "func")
    args = get_nodes(node, "args")
    keywords = get_nodes(node, "keywords")
    # Direct function call
    if _is_ast(func_node, "Name"):
        fname = get_str(func_node, "id")
        return _lower_name_call(fname, args, keywords, node, env, ctx)
    # Parameterized constructor: set[T](), dict[K,V](), list[T](), frozenset[T]()
    if _is_ast(func_node, "Subscript"):
        sub_value = get_node(func_node, "value")
        if _is_ast(sub_value, "Name"):
            sub_name = get_str(sub_value, "id")
            if sub_name in ("set", "dict", "list", "frozenset"):
                return _lower_name_call(sub_name, args, keywords, node, env, ctx)
    # Method call
    if _is_ast(func_node, "Attribute"):
        return _lower_method_call(func_node, args, keywords, node, env, ctx)
    # Fallback
    func = _lower_expr(func_node, env, ctx)
    lowered_args: list[TArg] = []
    for a in args:
        lowered_args.append(TArg(pos, None, _lower_expr(a, env, ctx)))
    return TCall(pos, func, lowered_args, {})


def _lower_arithmetic_call(
    fname: str,
    pos: Pos,
    args: list[ASTNode],
    keywords: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr | None:
    """Lower arithmetic builtins: sum, round, min, max, pow, abs, divmod."""
    if fname == "sum":
        if args and isinstance(args[0], dict):
            return _make_call(pos, "Sum", [_lower_expr(args[0], env, ctx)])
    if fname == "round":
        if args and isinstance(args[0], dict):
            round_args: list[TExpr] = [_lower_expr(args[0], env, ctx)]
            if len(args) > 1 and isinstance(args[1], dict):
                round_args.append(_lower_expr(args[1], env, ctx))
            return _make_call(pos, "Round", round_args)
    if fname == "min" or fname == "max":
        builtin = "Min" if fname == "min" else "Max"
        lowered: list[TExpr] = []
        for a in args:
            at = _infer_expr_type(a, env, ctx)
            la = _lower_expr(a, env, ctx)
            if _is_type_dict(at, ["bool"]):
                la = _bool_to_int(pos, la)
            lowered.append(la)
        key_node = _get_keyword_value(keywords, "key")
        if key_node is not None and len(lowered) == 1:
            arg_type = _infer_expr_type(args[0], env, ctx)
            elem_type = _collection_element_type(arg_type)
            key_fn = _lower_key_func(pos, key_node, elem_type, env, ctx)
            return _make_call(pos, builtin, [lowered[0], key_fn])
        if len(lowered) == 1:
            return _make_call(pos, builtin, lowered)
        if len(lowered) >= 3:
            result = _make_call(pos, builtin, [lowered[0], lowered[1]])
            for item in lowered[2:]:
                result = _make_call(pos, builtin, [result, item])
            return result
        return _make_call(pos, builtin, lowered)
    if fname == "pow":
        if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            pow_a = _lower_expr(args[0], env, ctx)
            pow_b = _lower_expr(args[1], env, ctx)
            if _is_type_dict(_infer_expr_type(args[0], env, ctx), ["bool"]):
                pow_a = _bool_to_int(pos, pow_a)
            if _is_type_dict(_infer_expr_type(args[1], env, ctx), ["bool"]):
                pow_b = _bool_to_int(pos, pow_b)
            return _make_call(pos, "Pow", [pow_a, pow_b])
    if fname == "abs":
        if args and isinstance(args[0], dict):
            abs_a = _lower_expr(args[0], env, ctx)
            if _is_type_dict(_infer_expr_type(args[0], env, ctx), ["bool"]):
                abs_a = _bool_to_int(pos, abs_a)
            return _make_call(pos, "Abs", [abs_a])
    if fname == "divmod":
        if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            div_a = _lower_expr(args[0], env, ctx)
            div_b = _lower_expr(args[1], env, ctx)
            return TTupleLit(
                pos,
                [
                    _make_call(pos, "FloorDiv", [div_a, div_b]),
                    _make_call(pos, "PythonMod", [div_a, div_b]),
                ],
                {},
            )
    return None


def _lower_conversion_call(
    fname: str, pos: Pos, args: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> TExpr | None:
    """Lower conversion builtins: int, float, str, bool, chr, ord, repr, bytes, hex."""
    if fname == "int":
        if not args:
            return TIntLit(pos, 0, "0", {})
        if len(args) >= 1 and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            arg = _lower_expr(args[0], env, ctx)
            if len(args) >= 2 and isinstance(args[1], dict):
                base = _lower_expr(args[1], env, ctx)
                return _make_call(pos, "ParseInt", [arg, base])
            if _is_type_dict(arg_type, ["int"]):
                return arg
            if _is_type_dict(arg_type, ["bool"]):
                return TTernary(
                    pos,
                    arg,
                    TIntLit(pos, 1, "1", {}),
                    TIntLit(pos, 0, "0", {}),
                    {},
                )
            if _is_type_dict(arg_type, ["float"]):
                return _make_call(pos, "FloatToInt", [arg])
            if _is_type_dict(arg_type, ["byte"]):
                return _make_call(pos, "ByteToInt", [arg])
            return _make_call(pos, "ParseInt", [arg, TIntLit(pos, 10, "10", {})])
    if fname == "float":
        if not args:
            return TFloatLit(pos, 0.0, "0.0", {})
        if args and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            arg = _lower_expr(args[0], env, ctx)
            if _is_type_dict(arg_type, ["float"]):
                return arg
            if _is_type_dict(arg_type, ["int"]):
                return _make_call(pos, "IntToFloat", [arg])
            return _make_call(pos, "ParseFloat", [arg])
    if fname == "str":
        if args and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            if _is_type_dict(arg_type, ["string"]):
                return _lower_expr(args[0], env, ctx)
            return _make_call(pos, "ToString", [_lower_expr(args[0], env, ctx)])
    if fname == "bool":
        if not args:
            return TBoolLit(pos, False, {})
        if args and isinstance(args[0], dict):
            if _is_ast(args[0], "Constant") and isinstance(args[0].get("value"), JNull):
                return TBoolLit(pos, False, {})
            if _is_ast(args[0], "Dict"):
                return TBoolLit(pos, len(get_nodes(args[0], "keys")) > 0, {})
            if _is_ast(args[0], "Set"):
                return TBoolLit(pos, len(get_nodes(args[0], "elts")) > 0, {})
            if _is_ast(args[0], "List"):
                return TBoolLit(pos, len(get_nodes(args[0], "elts")) > 0, {})
            arg_type = _infer_expr_type(args[0], env, ctx)
            arg = _lower_expr(args[0], env, ctx)
            if _is_optional_type(arg_type):
                return TUnaryOp(pos, "!", _make_call(pos, "IsNil", [arg]), {})
            if _is_type_dict(arg_type, ["int"]):
                return TBinaryOp(pos, "!=", arg, TIntLit(pos, 0, "0", {}), {})
            if _is_type_dict(arg_type, ["float"]):
                return TBinaryOp(pos, "!=", arg, TFloatLit(pos, 0.0, "0.0", {}), {})
            if _is_type_dict(arg_type, ["string"]):
                return TBinaryOp(pos, "!=", arg, TStringLit(pos, "", {}), {})
            if _is_type_dict(arg_type, ["bool"]):
                return arg
            if isinstance(arg_type, TupleType):
                if arg_type.elements:
                    return TBoolLit(pos, True, {})
                return TBoolLit(pos, False, {})
            if _is_type_dict(arg_type, ["bytes", "Slice", "Map", "Set"]):
                return TBinaryOp(
                    pos,
                    "!=",
                    _make_call(pos, "Len", [arg]),
                    TIntLit(pos, 0, "0", {}),
                    {},
                )
            return TBinaryOp(pos, "!=", arg, TIntLit(pos, 0, "0", {}), {})
    if fname == "chr":
        if args and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            arg = _lower_expr(args[0], env, ctx)
            if _is_type_dict(arg_type, ["byte"]):
                rune = _make_call(
                    pos, "RuneFromInt", [_make_call(pos, "ByteToInt", [arg])]
                )
            else:
                rune = _make_call(pos, "RuneFromInt", [arg])
            return _make_call(pos, "ToString", [rune])
    if fname == "ord":
        if args and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            arg = _lower_expr(args[0], env, ctx)
            if _is_type_dict(arg_type, ["string"]):
                # If arg is a string subscript (s[i]), the lowered result could be:
                # - TIndex directly (string indexing returns rune in taytsh)
                # - ToString(TIndex) if lowering wrapped it for Python semantics
                # For ord(), we need the rune, so use TIndex directly without [0].
                if isinstance(arg, TIndex):
                    return _make_call(pos, "RuneToInt", [arg])
                if isinstance(arg, TCall) and arg.func == "ToString":
                    inner = arg.args[0]
                    if isinstance(inner, TIndex):
                        return _make_call(pos, "RuneToInt", [inner])
                indexed = TIndex(pos, arg, TIntLit(pos, 0, "0", {}), {})
                return _make_call(pos, "RuneToInt", [indexed])
            return _make_call(pos, "RuneToInt", [arg])
    if fname == "repr":
        if args and isinstance(args[0], dict):
            return _make_call(pos, "ToRepr", [_lower_expr(args[0], env, ctx)])
    if fname == "bytes":
        if not args:
            return TBytesLit(pos, b"", {})
        if len(args) == 1 and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            if _is_type_dict(arg_type, ["int"]):
                return _make_call(pos, "Bytes", [_lower_expr(args[0], env, ctx)])
            if _is_type_dict(arg_type, ["Slice"]):
                return _make_call(pos, "BytesFrom", [_lower_expr(args[0], env, ctx)])
    if fname == "hex":
        if args and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            arg = _lower_expr(args[0], env, ctx)
            if _is_type_dict(arg_type, ["byte"]):
                int_arg = _make_call(pos, "ByteToInt", [arg])
            else:
                int_arg = arg
            return _make_call(
                pos,
                "FormatInt",
                [int_arg, TIntLit(pos, 16, "16", {})],
            )
    return None


def _string_to_char_list(pos: Pos, s: str) -> list[TExpr]:
    """Split a string constant into a list of single-character TStringLits."""
    elems: list[TExpr] = []
    ci = 0
    while ci < len(s):
        elems.append(TStringLit(pos, s[ci : ci + 1], {}))
        ci += 1
    return elems


def _lower_collection_call(
    fname: str,
    pos: Pos,
    args: list[ASTNode],
    keywords: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr | None:
    """Lower collection builtins: sorted, list, set, frozenset, tuple, dict."""
    if fname == "sorted":
        if args and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            arg = _lower_expr(args[0], env, ctx)
            if isinstance(arg_type, MapType):
                arg = _make_call(pos, "Keys", [arg])
            key_node = _get_keyword_value(keywords, "key")
            sorted_args: list[TExpr] = [arg]
            if key_node is not None:
                elem_type = _collection_element_type(arg_type)
                key_fn = _lower_key_func(pos, key_node, elem_type, env, ctx)
                sorted_args.append(key_fn)
            is_reversed = _has_keyword_true(keywords, "reverse")
            if is_reversed:
                return _make_call(
                    pos, "Reversed", [_make_call(pos, "Sorted", sorted_args)]
                )
            return _make_call(pos, "Sorted", sorted_args)
    if fname == "list":
        if args and isinstance(args[0], dict):
            if _is_ast(args[0], "Call"):
                rfunc = get_node(args[0], "func")
                if _is_ast(rfunc, "Name") and get_str(rfunc, "id") == "range":
                    rargs = get_nodes(args[0], "args")
                    if len(rargs) == 1 and isinstance(rargs[0], dict):
                        end = _lower_expr(rargs[0], env, ctx)
                        return _make_call(
                            pos,
                            "RangeList",
                            [
                                TIntLit(pos, 0, "0", {}),
                                end,
                                TIntLit(pos, 1, "1", {}),
                            ],
                        )
                    if (
                        len(rargs) == 2
                        and isinstance(rargs[0], dict)
                        and isinstance(rargs[1], dict)
                    ):
                        start = _lower_expr(rargs[0], env, ctx)
                        end = _lower_expr(rargs[1], env, ctx)
                        return _make_call(
                            pos,
                            "RangeList",
                            [start, end, TIntLit(pos, 1, "1", {})],
                        )
                    if (
                        len(rargs) >= 3
                        and isinstance(rargs[0], dict)
                        and isinstance(rargs[1], dict)
                        and isinstance(rargs[2], dict)
                    ):
                        start = _lower_expr(rargs[0], env, ctx)
                        end = _lower_expr(rargs[1], env, ctx)
                        step = _lower_expr(rargs[2], env, ctx)
                        return _make_call(pos, "RangeList", [start, end, step])
            arg_type = _infer_expr_type(args[0], env, ctx)
            if _is_type_dict(arg_type, ["string"]):
                if _is_ast(args[0], "Constant"):
                    s_jv = args[0].get("value")
                    if isinstance(s_jv, JStr):
                        return TListLit(pos, _string_to_char_list(pos, s_jv.value), {})
                return _make_call(pos, "Chars", [_lower_expr(args[0], env, ctx)])
            if _is_ast(args[0], "Call"):
                rfunc = get_node(args[0], "func")
                if _is_ast(rfunc, "Name") and get_str(rfunc, "id") == "zip":
                    return _lower_expr(args[0], env, ctx)
                if _is_ast(rfunc, "Attribute") and get_str(rfunc, "attr") in (
                    "keys",
                    "values",
                    "items",
                ):
                    return _lower_expr(args[0], env, ctx)
            if isinstance(arg_type, SetType):
                return _make_call(pos, "Sorted", [_lower_expr(args[0], env, ctx)])
            if _is_type_dict(arg_type, ["Map"]):
                keys = _make_call(pos, "Keys", [_lower_expr(args[0], env, ctx)])
                return _make_call_ann(
                    pos, "ListFrom", [keys], {"provenance": "dict_keys"}
                )
            return _make_call(pos, "ListFrom", [_lower_expr(args[0], env, ctx)])
    if fname == "set" or fname == "frozenset":
        if not args:
            return _make_call(pos, "Set", [])
        if len(args) == 1 and isinstance(args[0], dict):
            if _is_ast(args[0], "GeneratorExp") or _is_ast(args[0], "ListComp"):
                return _make_call(pos, "SetFromList", [_lower_expr(args[0], env, ctx)])
            if _is_ast(args[0], "Call"):
                rfunc = get_node(args[0], "func")
                if _is_ast(rfunc, "Name") and get_str(rfunc, "id") == "range":
                    range_list = _lower_extend_arg(args[0], env, ctx)
                    return _make_call(pos, "SetFromList", [range_list])
                if _is_ast(rfunc, "Attribute") and get_str(rfunc, "attr") in (
                    "keys",
                    "values",
                    "items",
                ):
                    return _make_call(
                        pos, "SetFromList", [_lower_expr(args[0], env, ctx)]
                    )
            arg_type = _infer_expr_type(args[0], env, ctx)
            if _is_type_dict(arg_type, ["string"]) and _is_ast(args[0], "Constant"):
                s_jv = args[0].get("value")
                if isinstance(s_jv, JStr):
                    return _make_call(
                        pos,
                        "SetFromList",
                        [TListLit(pos, _string_to_char_list(pos, s_jv.value), {})],
                    )
            if _is_type_dict(arg_type, ["Slice"]):
                return _make_call(pos, "SetFromList", [_lower_expr(args[0], env, ctx)])
            if _is_ast(args[0], "Tuple") or isinstance(arg_type, TupleType):
                lowered_arg = _lower_expr(args[0], env, ctx)
                if isinstance(lowered_arg, TTupleLit):
                    return _make_call(
                        pos,
                        "SetFromList",
                        [TListLit(pos, lowered_arg.elements, {})],
                    )
                return _make_call(pos, "SetFromList", [lowered_arg])
            if _is_type_dict(arg_type, ["Map"]):
                keys = _make_call(pos, "Keys", [_lower_expr(args[0], env, ctx)])
                return _make_call_ann(
                    pos, "SetFromList", [keys], {"provenance": "dict_keys"}
                )
            return _make_call(pos, "SetFromList", [_lower_expr(args[0], env, ctx)])
    if fname == "tuple":
        if not args:
            return TListLit(pos, [], {})
        if len(args) == 1 and isinstance(args[0], dict):
            if _is_ast(args[0], "Call"):
                rfunc = get_node(args[0], "func")
                if _is_ast(rfunc, "Name") and get_str(rfunc, "id") == "range":
                    return _lower_extend_arg(args[0], env, ctx)
            arg_type = _infer_expr_type(args[0], env, ctx)
            if _is_type_dict(arg_type, ["string"]) and _is_ast(args[0], "Constant"):
                s_jv = args[0].get("value")
                if isinstance(s_jv, JStr):
                    return TListLit(pos, _string_to_char_list(pos, s_jv.value), {})
            if _is_type_dict(arg_type, ["Set"]):
                return _make_call(pos, "Sorted", [_lower_expr(args[0], env, ctx)])
            arg = _lower_expr(args[0], env, ctx)
            return TSlice(
                pos,
                arg,
                TIntLit(pos, 0, "0", {}),
                _make_call(pos, "Len", [arg]),
                {},
            )
    if fname == "dict":
        if not args:
            return _make_call(pos, "Map", [])
        if len(args) == 1 and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            if isinstance(arg_type, MapType):
                items = _make_call(pos, "Items", [_lower_expr(args[0], env, ctx)])
                return _make_call(pos, "MapFromPairs", [items])
            return _make_call(pos, "MapFromPairs", [_lower_expr(args[0], env, ctx)])
    return None


def _lower_name_call(
    fname: str,
    args: list[ASTNode],
    keywords: list[ASTNode],
    node: ASTNode,
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Lower a direct function call by name."""
    pos = _node_pos(node)
    if fname == "len":
        if args and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            if isinstance(arg_type, TupleType):
                n = len(arg_type.elements)
                return TIntLit(pos, n, str(n), {})
            if _is_ast(args[0], "Tuple"):
                elts = get_nodes(args[0], "elts")
                n = len(elts)
                return TIntLit(pos, n, str(n), {})
            if _is_sys_argv(args[0]):
                return TBinaryOp(
                    pos,
                    "+",
                    _make_call(pos, "Len", [_make_call(pos, "Args", [])]),
                    TIntLit(pos, 1, "1", {}),
                    {},
                )
            return _make_call(pos, "Len", [_lower_expr(args[0], env, ctx)])
    arith = _lower_arithmetic_call(fname, pos, args, keywords, env, ctx)
    if arith is not None:
        return arith
    conv = _lower_conversion_call(fname, pos, args, env, ctx)
    if conv is not None:
        return conv
    coll = _lower_collection_call(fname, pos, args, keywords, env, ctx)
    if coll is not None:
        return coll
    if fname == "zip":
        if len(args) >= 2 and all(isinstance(a, dict) for a in args):
            zip_args = [_lower_expr(a, env, ctx) for a in args]
            return _make_call(pos, "Zip", zip_args)
    if fname == "isinstance":
        if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            tnames = _isinstance_types_from_args(args)
            if not tnames:
                return TBoolLit(pos, True, {})
            lowered_arg = _lower_expr(args[0], env, ctx)
            result_expr: TExpr = _make_call(
                pos, "IsType", [lowered_arg, TStringLit(pos, tnames[0], {})]
            )
            for tname in tnames[1:]:
                right: TExpr = _make_call(
                    pos,
                    "IsType",
                    [lowered_arg, TStringLit(pos, tname, {})],
                )
                ann: Ann = {"provenance": "isinstance_tuple"} if len(tnames) > 1 else {}
                result_expr = TBinaryOp(pos, "||", result_expr, right, ann)
            return result_expr
    if fname == "any" or fname == "all":
        if args and isinstance(args[0], dict):
            a0_type = get_str(args[0], "_type")
            if a0_type == "GeneratorExp" or a0_type == "ListComp":
                return _lower_any_all(fname, args[0], env, ctx)
        return TBoolLit(pos, True, {})
    if fname == "print":
        return _lower_print_call(pos, args, keywords, env, ctx)
    if fname in (
        "TypeError",
        "NotImplementedError",
        "RuntimeError",
        "KeyError",
        "IndexError",
    ):
        exc_args: list[TArg] = []
        if args and isinstance(args[0], dict):
            exc_args.append(TArg(pos, None, _lower_expr(args[0], env, ctx)))
        else:
            exc_args.append(TArg(pos, None, TStringLit(pos, "", {})))
        return TCall(pos, TVar(pos, fname, {}), exc_args, {})
    if fname in ctx.known_classes:
        return _lower_struct_constructor(pos, fname, args, keywords, env, ctx)
    lowered_args: list[TArg] = []
    if keywords:
        func_info = ctx.tc_result.functions.get(fname)
        params: list[ParamInfo] | None = None
        if func_info is not None:
            params = func_info.params
        lowered_args = _resolve_kwargs_to_positional(
            pos, args, keywords, params, env, ctx
        )
    else:
        for a in args:
            lowered_args.append(TArg(pos, None, _lower_expr(a, env, ctx)))
    safe = _safe_name(fname)
    return TCall(pos, TVar(pos, safe, _name_ann(safe, fname)), lowered_args, {})


def _has_keyword_true(keywords: list[ASTNode], name: str) -> bool:
    """Check if keywords contain name=True."""
    for kw in keywords:
        if get_str(kw, "arg") == name:
            val_node = get_node(kw, "value")
            v = val_node.get("value")
            if isinstance(v, JBool) and v.value:
                return True
    return False


def _get_keyword_value(keywords: list[ASTNode], name: str) -> ASTNode | None:
    """Get value node for a keyword argument."""
    for kw in keywords:
        if get_str(kw, "arg") == name:
            val_node = get_node(kw, "value")
            if val_node:
                return val_node
    return None


def _collection_element_type(t: TypeNode) -> TypeNode:
    """Get element type from a collection type."""
    if isinstance(t, SliceType):
        return t.element
    if isinstance(t, SetType):
        return t.element
    if isinstance(t, MapType):
        return t.key
    if isinstance(t, PrimitiveType) and t.kind == "string":
        return STR_TYPE
    return INT_TYPE


def _infer_key_lambda_body_type(
    body: ASTNode, elem_type: TypeNode, env: _Env, ctx: _LowerCtx
) -> TypeNode:
    """Infer return type of a key= lambda body (pycheck doesn't walk lambdas)."""
    t = get_str(body, "_type")
    if t == "UnaryOp":
        operand = get_node(body, "operand")
        return _infer_key_lambda_body_type(operand, elem_type, env, ctx)
    if t == "Subscript" and isinstance(elem_type, TupleType):
        slc = get_node(body, "slice")
        if _is_ast(slc, "Constant") and isinstance(slc.get("value"), JInt):
            idx = get_int(slc, "value")
            if 0 <= idx < len(elem_type.elements):
                return elem_type.elements[idx]
    if t == "Call":
        func = get_node(body, "func")
        if _is_ast(func, "Name"):
            fname = get_str(func, "id")
            if fname == "len" or fname == "ord" or fname == "int":
                return INT_TYPE
            if fname == "str":
                return STR_TYPE
            return _func_return_type(ctx, fname)
    if t == "Attribute":
        obj = get_node(body, "value")
        obj_type = _infer_key_lambda_body_type(obj, elem_type, env, ctx)
        attr = get_str(body, "attr")
        if isinstance(obj_type, PointerType):
            obj_type = obj_type.target
        if _is_struct_type(obj_type):
            cls_info = ctx.tc_result.classes.get(_struct_name(obj_type))
            if cls_info is not None:
                fi = cls_info.fields.get(attr)
                if fi is not None:
                    return fi.typ
    if t == "Name":
        vt = env.var_types.get(get_str(body, "id"))
        if vt is not None:
            return vt
    return elem_type


def _lower_key_func(
    pos: Pos,
    key_node: ASTNode,
    elem_type: TypeNode,
    env: _Env,
    ctx: _LowerCtx,
) -> TFnLit:
    """Lower a key= argument (lambda or named function) to a TFnLit."""
    if _is_ast(key_node, "Lambda"):
        args_node = get_node(key_node, "args")
        param_args = get_nodes(args_node, "args") if args_node else []
        param_name = "x"
        if param_args and isinstance(param_args[0], dict):
            param_name = get_str(param_args[0], "arg")
        param_ttype = _typenode_to_ttype(pos, elem_type)
        param = TParam(pos, param_name, param_ttype, {})
        body_node = get_node(key_node, "body")
        inner_env = env.copy()
        inner_env.var_types[param_name] = elem_type
        inner_env.declared.add(param_name)
        body_expr = _lower_expr(body_node, inner_env, ctx)
        ret_type = _infer_key_lambda_body_type(body_node, elem_type, inner_env, ctx)
        lambda_ret = _typenode_to_ttype(pos, ret_type)
        return TFnLit(
            pos,
            [param],
            lambda_ret,
            [TExprStmt(pos, body_expr, {})],
            {"fn_lit.arrow": "true"},
        )
    if _is_ast(key_node, "Name"):
        fname = get_str(key_node, "id")
        func_info = ctx.tc_result.functions.get(fname)
        if func_info is not None and func_info.params:
            p = func_info.params[0]
            param_ttype = _typenode_to_ttype(pos, p.typ)
            param_name = p.name
            param = TParam(pos, param_name, param_ttype, {})
            func_ret = _typenode_to_ttype(pos, func_info.return_type)
            inner_env = env.copy()
            inner_env.var_types[param_name] = p.typ
            inner_env.declared.add(param_name)
            body_call = _lower_user_func_body_as_expr(
                pos, fname, param_name, inner_env, ctx
            )
            if body_call is not None:
                return TFnLit(
                    pos,
                    [param],
                    func_ret,
                    [TExprStmt(pos, body_call, {})],
                    {"fn_lit.arrow": "true"},
                )
        if fname == "len":
            param_ttype = _typenode_to_ttype(pos, elem_type)
            param_name = _key_param_name(elem_type)
            param = TParam(pos, param_name, param_ttype, {})
            ret_ttype = TPrimitive(pos, "int")
            body_expr = _make_call(pos, "Len", [TVar(pos, param_name, {})])
            return TFnLit(
                pos,
                [param],
                ret_ttype,
                [TExprStmt(pos, body_expr, {})],
                {"fn_lit.arrow": "true"},
            )
    param_ttype = _typenode_to_ttype(pos, elem_type)
    param_name = _key_param_name(elem_type)
    param = TParam(pos, param_name, param_ttype, {})
    ret_ttype = TPrimitive(pos, "int")
    inner_env = env.copy()
    inner_env.var_types[param_name] = elem_type
    inner_env.declared.add(param_name)
    body_expr = _lower_expr(key_node, inner_env, ctx)
    return TFnLit(
        pos,
        [param],
        ret_ttype,
        [TExprStmt(pos, body_expr, {})],
        {"fn_lit.arrow": "true"},
    )


def _key_param_name(elem_type: TypeNode) -> str:
    """Choose a sensible parameter name for a key function based on element type."""
    if isinstance(elem_type, PrimitiveType):
        if elem_type.kind == "string":
            return "s"
        if elem_type.kind == "int":
            return "x"
        if elem_type.kind == "float":
            return "x"
    return "x"


def _lower_user_func_body_as_expr(
    pos: Pos, fname: str, param_name: str, env: _Env, ctx: _LowerCtx
) -> TExpr | None:
    """Inline a single-expression user function as a TExpr for key= inlining."""
    node = ctx.func_nodes.get(fname)
    if node is None:
        return None
    body = get_nodes(node, "body")
    if len(body) == 1 and _is_ast(body[0], "Return"):
        ret_val = get_node(body[0], "value")
        if ret_val:
            return _lower_expr(ret_val, env, ctx)
    return None


def _lower_print_call(
    pos: Pos, args: list[ASTNode], keywords: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower print() to WritelnOut/WriteOut/WritelnErr or Print()."""
    # Get the argument (print typically has one arg in subset)
    arg_expr: TExpr = TStringLit(pos, "", {})
    is_string = True
    if args and isinstance(args[0], dict):
        arg_type = _infer_expr_type(args[0], env, ctx)
        arg_expr = _lower_expr(args[0], env, ctx)
        is_string = _is_type_dict(arg_type, ["string"])
        if not is_string:
            # Check for explicit str() wrapping — treat as string
            if _is_ast(args[0], "Call"):
                call_func = get_node(args[0], "func")
                if _is_ast(call_func, "Name") and get_str(call_func, "id") == "str":
                    is_string = True
    # Check for end=""
    end_val = _get_keyword_value(keywords, "end")
    no_newline = False
    if end_val is not None:
        v = end_val.get("value")
        if isinstance(v, JStr) and not v.value:
            no_newline = True
    # Check for file=sys.stderr
    file_val = _get_keyword_value(keywords, "file")
    is_stderr = False
    if file_val is not None:
        if _is_ast(file_val, "Attribute"):
            attr = get_str(file_val, "attr")
            obj = get_node(file_val, "value")
            if (
                _is_ast(obj, "Name")
                and get_str(obj, "id") == "sys"
                and attr == "stderr"
            ):
                is_stderr = True
    # For string values: use WritelnOut/WriteOut/WritelnErr directly
    if is_string:
        if is_stderr:
            return _make_call(pos, "WritelnErr", [arg_expr])
        if no_newline:
            return _make_call(pos, "WriteOut", [arg_expr])
        return _make_call(pos, "WritelnOut", [arg_expr])
    # For non-string values: use Print() with named args
    if is_stderr:
        return _make_named_call(
            pos, "Print", [arg_expr], [("stderr", TBoolLit(pos, True, {}))]
        )
    if no_newline:
        return _make_named_call(
            pos, "Print", [arg_expr], [("newline", TBoolLit(pos, False, {}))]
        )
    return _make_named_call(
        pos, "Print", [arg_expr], [("newline", TBoolLit(pos, True, {}))]
    )


def _lower_struct_constructor(
    pos: Pos,
    class_name: str,
    args: list[ASTNode],
    keywords: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Lower a struct constructor call."""
    if ctx.hier_result.is_hierarchy_root(class_name):
        ctx.errors.append(
            LoweringError(
                pos.line,
                pos.col,
                "cannot construct interface '" + class_name + "'",
            )
        )
    lowered_args: list[TArg] = []
    for a in args:
        lowered_args.append(TArg(pos, None, _lower_expr(a, env, ctx)))
    for kw in keywords:
        kw_name = get_str(kw, "arg")
        kw_val = get_node(kw, "value")
        if kw_name and kw_val:
            lowered_args.append(TArg(pos, kw_name, _lower_expr(kw_val, env, ctx)))
    return TCall(pos, TVar(pos, class_name, {}), lowered_args, {})


def _try_lower_stdlib_method(
    pos: Pos,
    obj_node: ASTNode,
    method_name: str,
    args: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr | None:
    """Try to lower stdlib method calls (sys, os, dict.fromkeys). Returns None if not applicable."""
    if _is_ast(obj_node, "Name") and get_str(obj_node, "id") == "sys":
        if method_name == "exit":
            exit_args: list[TExpr] = []
            if args and isinstance(args[0], dict):
                exit_args.append(_lower_expr(args[0], env, ctx))
            else:
                exit_args.append(TIntLit(pos, 0, "0", {}))
            return _make_call(pos, "Exit", exit_args)
    if _is_ast(obj_node, "Attribute"):
        inner_obj = get_node(obj_node, "value")
        inner_attr = get_str(obj_node, "attr")
        if _is_ast(inner_obj, "Name") and get_str(inner_obj, "id") == "sys":
            if inner_attr == "stdin":
                if method_name == "readline":
                    return _make_call(pos, "ReadLine", [])
                if method_name == "read":
                    return _make_call(pos, "ReadAll", [])
    if _is_ast(obj_node, "Attribute"):
        inner_obj = get_node(obj_node, "value")
        inner_attr = get_str(obj_node, "attr")
        if _is_ast(inner_obj, "Attribute"):
            inner2_obj = get_node(inner_obj, "value")
            inner2_attr = get_str(inner_obj, "attr")
            if _is_ast(inner2_obj, "Name") and get_str(inner2_obj, "id") == "sys":
                if inner2_attr == "stdin" and inner_attr == "buffer":
                    if method_name == "read":
                        if args and isinstance(args[0], dict):
                            return _make_call(
                                pos, "ReadBytesN", [_lower_expr(args[0], env, ctx)]
                            )
                        return _make_call(pos, "ReadBytes", [])
                if inner2_attr == "stdout" and inner_attr == "buffer":
                    if method_name == "write":
                        if args and isinstance(args[0], dict):
                            return _make_call(
                                pos, "WriteOut", [_lower_expr(args[0], env, ctx)]
                            )
                if inner2_attr == "stderr" and inner_attr == "buffer":
                    if method_name == "write":
                        if args and isinstance(args[0], dict):
                            return _make_call(
                                pos, "WriteErr", [_lower_expr(args[0], env, ctx)]
                            )
    if _is_ast(obj_node, "Name") and get_str(obj_node, "id") == "os":
        if method_name == "getenv":
            lowered: list[TExpr] = []
            for a in args:
                lowered.append(_lower_expr(a, env, ctx))
            return _make_call(pos, "GetEnv", lowered)
    if _is_ast(obj_node, "Name") and get_str(obj_node, "id") == "dict":
        if method_name == "fromkeys":
            fk_args: list[TExpr] = []
            for a in args:
                fk_args.append(_lower_expr(a, env, ctx))
            if len(fk_args) == 1:
                fk_args.append(TNilLit(pos, {}))
            return _make_call(pos, "MapFromKeys", fk_args)
    return None


def _lower_method_call(
    func_node: ASTNode,
    args: list[ASTNode],
    keywords: list[ASTNode],
    node: ASTNode,
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Lower a method call."""
    pos = _node_pos(node)
    method_name = get_str(func_node, "attr")
    obj_node = get_node(func_node, "value")
    obj_type = _infer_expr_type(obj_node, env, ctx)
    stdlib = _try_lower_stdlib_method(pos, obj_node, method_name, args, env, ctx)
    if stdlib is not None:
        return stdlib
    obj = _lower_expr(obj_node, env, ctx)
    # Unwrap pointer for type dispatch
    actual_type = _unwrap_pointer(obj_type)
    # String methods
    if _is_type_dict(actual_type, ["string"]):
        return _lower_string_method(pos, obj, method_name, args, env, ctx)
    # Bytes methods
    if _is_type_dict(actual_type, ["bytes"]) or _is_bytes_slice(actual_type):
        return _lower_bytes_method(pos, obj, method_name, args, env, ctx)
    # List methods
    if _is_type_dict(actual_type, ["Slice"]):
        return _lower_list_method(
            pos, obj, obj_node, method_name, args, env, ctx, type_name="list"
        )
    # Tuple methods (count/index reuse list lowering)
    if _is_type_dict(actual_type, ["Tuple"]):
        return _lower_list_method(
            pos, obj, obj_node, method_name, args, env, ctx, type_name="tuple"
        )
    # Dict methods
    if _is_type_dict(actual_type, ["Map"]):
        return _lower_dict_method(pos, obj, obj_node, method_name, args, env, ctx)
    # Set methods
    if _is_type_dict(actual_type, ["Set"]):
        return _lower_set_method(pos, obj, method_name, args, env, ctx)
    # Struct method call
    lowered_args: list[TArg] = []
    if keywords:
        method_params = _lookup_method_params(actual_type, method_name, ctx)
        lowered_args = _resolve_kwargs_to_positional(
            pos, args, keywords, method_params, env, ctx
        )
    else:
        for a in args:
            lowered_args.append(TArg(pos, None, _lower_expr(a, env, ctx)))
    return TCall(pos, TFieldAccess(pos, obj, method_name, {}), lowered_args, {})


def _lower_string_method(
    pos: Pos, obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower string method calls."""
    lowered: list[TExpr] = []
    for a in args:
        lowered.append(_lower_expr(a, env, ctx))
    if method == "find":
        return _make_call(pos, "Find", [obj] + lowered)
    if method == "rfind":
        return _make_call(pos, "RFind", [obj] + lowered)
    if method == "split":
        if not lowered:
            return _make_call(pos, "SplitWhitespace", [obj])
        if len(lowered) == 2:
            plus_one = TBinaryOp(pos, "+", lowered[1], TIntLit(pos, 1, "1", {}), {})
            return _make_call(pos, "SplitN", [obj, lowered[0], plus_one])
        return _make_call(pos, "Split", [obj] + lowered)
    if method == "replace":
        if len(lowered) == 3:
            return _make_call(pos, "ReplaceCount", [obj] + lowered)
        return _make_call(pos, "Replace", [obj] + lowered)
    if method == "count":
        return _make_call(pos, "Count", [obj] + lowered)
    if method == "startswith":
        return _lower_startswith_endswith(pos, "StartsWith", obj, args, env, ctx)
    if method == "endswith":
        return _lower_startswith_endswith(pos, "EndsWith", obj, args, env, ctx)
    if method == "strip":
        if not lowered:
            lowered = [TStringLit(pos, " \t\n\r\x0b\x0c", {})]
        return _make_call(pos, "Trim", [obj] + lowered)
    if method == "lstrip":
        if not lowered:
            lowered = [TStringLit(pos, " \t\n\r\x0b\x0c", {})]
        return _make_call(pos, "TrimStart", [obj] + lowered)
    if method == "rstrip":
        if not lowered:
            lowered = [TStringLit(pos, " \t\n\r\x0b\x0c", {})]
        return _make_call(pos, "TrimEnd", [obj] + lowered)
    if method == "lower":
        return _make_call(pos, "Lower", [obj])
    if method == "upper":
        return _make_call(pos, "Upper", [obj])
    if method == "join":
        return _make_call(pos, "Join", [obj] + lowered)
    if method == "isdigit":
        return _make_call(pos, "IsDigit", [obj])
    if method == "isalpha":
        return _make_call(pos, "IsAlpha", [obj])
    if method == "isalnum":
        return _make_call(pos, "IsAlnum", [obj])
    if method == "isspace":
        return _make_call(pos, "IsSpace", [obj])
    if method == "isupper":
        return _make_call(pos, "IsUpper", [obj])
    if method == "islower":
        return _make_call(pos, "IsLower", [obj])
    if method == "encode":
        return _make_call(pos, "Encode", [obj])
    if method == "index":
        return _make_call(pos, "Find", [obj] + lowered)
    if method == "removeprefix":
        rp_arg = lowered[0] if lowered else TStringLit(pos, "", {})
        rp_cond = _make_call(pos, "StartsWith", [obj, rp_arg])
        rp_then = TSlice(
            pos,
            obj,
            _make_call(pos, "Len", [rp_arg]),
            _make_call(pos, "Len", [obj]),
            {},
        )
        return TTernary(pos, rp_cond, rp_then, obj, {"provenance": "removeprefix"})
    if method == "removesuffix":
        rs_arg = lowered[0] if lowered else TStringLit(pos, "", {})
        rs_cond = _make_call(pos, "EndsWith", [obj, rs_arg])
        rs_end = TBinaryOp(
            pos,
            "-",
            _make_call(pos, "Len", [obj]),
            _make_call(pos, "Len", [rs_arg]),
            {},
        )
        rs_then = TSlice(pos, obj, TIntLit(pos, 0, "0", {}), rs_end, {})
        return TTernary(pos, rs_cond, rs_then, obj, {"provenance": "removesuffix"})
    if method == "partition" or method == "rpartition":
        func = "Find" if method == "partition" else "RFind"
        sep = lowered[0] if lowered else TStringLit(pos, "", {})
        find = _make_call(pos, func, [obj, sep])
        cond = TBinaryOp(pos, ">=", find, TIntLit(pos, 0, "0", {}), {})
        idx = _make_call(pos, func, [obj, sep])
        sep_len = _make_call(pos, "Len", [sep])
        obj_len = _make_call(pos, "Len", [obj])
        before = TSlice(pos, obj, TIntLit(pos, 0, "0", {}), idx, {})
        after = TSlice(pos, obj, TBinaryOp(pos, "+", idx, sep_len, {}), obj_len, {})
        found = TTupleLit(pos, [before, sep, after], {})
        empty = TStringLit(pos, "", {})
        if method == "partition":
            not_found = TTupleLit(pos, [obj, empty, TStringLit(pos, "", {})], {})
        else:
            not_found = TTupleLit(pos, [TStringLit(pos, "", {}), empty, obj], {})
        return TTernary(pos, cond, found, not_found, {"provenance": method})
    if method == "zfill":
        return _make_method_call(pos, obj, method, lowered)
    ctx.errors.append(
        LoweringError(pos.line, pos.col, "unsupported method '" + method + "' on str")
    )
    return TNilLit(pos, {})


def _lower_startswith_endswith(
    pos: Pos, func_name: str, obj: TExpr, args: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower startswith/endswith, handling tuple argument."""
    if args and isinstance(args[0], dict):
        arg = args[0]
        if _is_ast(arg, "Tuple"):
            # Tuple argument: startswith(("a", "b")) → StartsWith(s, "a") || StartsWith(s, "b")
            elts = get_nodes(arg, "elts")
            parts: list[TExpr] = []
            for e in elts:
                lowered_e = _lower_expr(e, env, ctx)
                parts.append(_make_call(pos, func_name, [obj, lowered_e]))
            if not parts:
                return TBoolLit(pos, False, {})
            result: TExpr = parts[0]
            for part in parts[1:]:
                result = TBinaryOp(pos, "||", result, part, {})
            return result
        # Single argument
        lowered_arg = _lower_expr(arg, env, ctx)
        return _make_call(pos, func_name, [obj, lowered_arg])
    return _make_call(pos, func_name, [obj])


def _lower_list_method(
    pos: Pos,
    obj: TExpr,
    obj_node: ASTNode,
    method: str,
    args: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
    type_name: str,
) -> TExpr:
    """Lower list method calls."""
    pos = _node_pos(obj_node)
    lowered: list[TExpr] = []
    for a in args:
        lowered.append(_lower_expr(a, env, ctx))
    if method == "append":
        if lowered and args:
            obj_type = _infer_expr_type(obj_node, env, ctx)
            if isinstance(obj_type, SliceType) and _is_type_dict(
                obj_type.element, ["string"]
            ):
                arg_type = _infer_expr_type(args[0], env, ctx)
                if _is_type_dict(arg_type, ["rune"]):
                    lowered[0] = _make_call(pos, "ToString", [lowered[0]])
        return _make_call(pos, "Append", [obj] + lowered)
    if method == "insert":
        return _make_call(pos, "Insert", [obj] + lowered)
    if method == "pop":
        if not lowered:
            return _make_call(pos, "Pop", [obj])
        return TIndex(pos, obj, lowered[0], {})
    if method == "index":
        if len(lowered) >= 2:
            val = lowered[0]
            start = lowered[1]
            sliced = TSlice(pos, obj, start, _make_call(pos, "Len", [obj]), {})
            return TBinaryOp(
                pos, "+", _make_call(pos, "IndexOf", [sliced, val]), start, {}
            )
        return _make_call(pos, "IndexOf", [obj] + lowered)
    if method == "remove":
        if lowered:
            return _make_call(
                pos, "RemoveAt", [obj, _make_call(pos, "IndexOf", [obj, lowered[0]])]
            )
        return _make_call(pos, "RemoveAt", [obj])
    if method == "copy":
        return TSlice(
            pos,
            obj,
            TIntLit(pos, 0, "0", {}),
            _make_call(pos, "Len", [obj]),
            {},
        )
    if method == "count":
        return _make_call(pos, "Count", [obj] + lowered)
    if method == "clear":
        return TListLit(pos, [], {})
    if method == "reverse":
        return TNilLit(pos, {})
    if method == "sort":
        return TNilLit(pos, {})
    ctx.errors.append(
        LoweringError(
            pos.line, pos.col, "unsupported method '" + method + "' on " + type_name
        )
    )
    return TNilLit(pos, {})


def _lower_dict_method(
    pos: Pos,
    obj: TExpr,
    obj_node: ASTNode,
    method: str,
    args: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Lower dict method calls."""
    pos = _node_pos(obj_node)
    lowered: list[TExpr] = []
    for a in args:
        lowered.append(_lower_expr(a, env, ctx))
    if method == "get":
        return _make_call(pos, "Get", [obj] + lowered)
    if method == "keys":
        return _make_call(pos, "Keys", [obj])
    if method == "values":
        return _make_call(pos, "Values", [obj])
    if method == "items":
        return _make_call(pos, "Items", [obj])
    if method == "copy":
        return _make_call(pos, "Merge", [obj, _make_call(pos, "Map", [])])
    if method == "pop":
        return TIndex(pos, obj, lowered[0], {})
    if method == "setdefault":
        return _make_call(pos, "Get", [obj] + lowered)
    if method == "update":
        return _make_call(pos, "Merge", [obj] + lowered)
    if method == "popitem":
        return _make_call(pos, "PopItem", [obj])
    ctx.errors.append(
        LoweringError(pos.line, pos.col, "unsupported method '" + method + "' on dict")
    )
    return TNilLit(pos, {})


def _method_side_effects(value_node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Return side-effect statements for methods that need post-assignment work."""
    pos = _node_pos(value_node)
    if not isinstance(value_node, dict) or not _is_ast(value_node, "Call"):
        return []
    func = get_node(value_node, "func")
    if not _is_ast(func, "Attribute"):
        return []
    method = get_str(func, "attr")
    obj_node = get_node(func, "value")
    obj_type = _infer_expr_type(obj_node, env, ctx)
    actual = _unwrap_pointer(obj_type)
    vargs = get_nodes(value_node, "args")
    # dict.pop(k) → Delete(d, k)
    if _is_type_dict(actual, ["Map"]) and method == "pop":
        if vargs and isinstance(vargs[0], dict):
            obj = _lower_expr(obj_node, env, ctx)
            key = _lower_expr(vargs[0], env, ctx)
            return [TExprStmt(pos, _make_call(pos, "Delete", [obj, key]), {})]
    # dict.setdefault(k, v) → if !Contains(d, k) { d[k] = v }
    if _is_type_dict(actual, ["Map"]) and method == "setdefault":
        if (
            len(vargs) >= 2
            and isinstance(vargs[0], dict)
            and isinstance(vargs[1], dict)
        ):
            obj = _lower_expr(obj_node, env, ctx)
            key = _lower_expr(vargs[0], env, ctx)
            default = _lower_expr(vargs[1], env, ctx)
            cond = TUnaryOp(pos, "!", _make_call(pos, "Contains", [obj, key]), {})
            assign = TAssignStmt(pos, TIndex(pos, obj, key, {}), default, {})
            return [TIfStmt(pos, cond, [assign], None, {})]
    # list.pop(i) → RemoveAt(xs, i)
    if _is_type_dict(actual, ["Slice"]) and method == "pop":
        if vargs and isinstance(vargs[0], dict):
            obj = _lower_expr(obj_node, env, ctx)
            idx = _lower_expr(vargs[0], env, ctx)
            return [TExprStmt(pos, _make_call(pos, "RemoveAt", [obj, idx]), {})]
    # list.sort() → xs = Sorted(xs)
    if _is_type_dict(actual, ["Slice"]) and method == "sort":
        obj = _lower_expr(obj_node, env, ctx)
        return [TAssignStmt(pos, obj, _make_call(pos, "Sorted", [obj]), {})]
    # list.reverse() → xs = Reversed(xs)
    if _is_type_dict(actual, ["Slice"]) and method == "reverse":
        obj = _lower_expr(obj_node, env, ctx)
        return [TAssignStmt(pos, obj, _make_call(pos, "Reversed", [obj]), {})]
    return []


def _lower_set_method(
    pos: Pos, obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower set method calls."""
    lowered: list[TExpr] = []
    for a in args:
        lowered.append(_lower_expr(a, env, ctx))
    if method == "add":
        return _make_call(pos, "Add", [obj] + lowered)
    if method == "remove":
        return _make_call(pos, "Remove", [obj] + lowered)
    if method == "discard":
        return _make_call(pos, "Remove", [obj] + lowered)
    if method == "pop":
        return _make_call(pos, "Pop", [obj])
    if method == "copy":
        return _make_call(pos, "Union", [obj, _make_call(pos, "Set", [])])
    if method == "union":
        if len(lowered) >= 1:
            result: TExpr = obj
            for lv in lowered:
                result = _make_call(pos, "Union", [result, lv])
            return result
    if method == "intersection":
        if len(lowered) >= 1:
            result_i: TExpr = obj
            for lv in lowered:
                result_i = _make_call(pos, "Intersection", [result_i, lv])
            return result_i
    if method == "difference":
        if len(lowered) >= 1:
            result_d: TExpr = obj
            for lv in lowered:
                result_d = _make_call(pos, "Difference", [result_d, lv])
            return result_d
    if method == "symmetric_difference":
        if len(lowered) == 1:
            u = _make_call(pos, "Union", [obj, lowered[0]])
            i2 = _make_call(pos, "Intersection", [obj, lowered[0]])
            return _make_call(pos, "Difference", [u, i2])
    if method == "issubset":
        if len(lowered) == 1:
            diff = _make_call(pos, "Difference", [obj, lowered[0]])
            return TBinaryOp(
                pos,
                "==",
                _make_call(pos, "Len", [diff]),
                TIntLit(pos, 0, "0", {}),
                {},
            )
    if method == "issuperset":
        if len(lowered) == 1:
            diff = _make_call(pos, "Difference", [lowered[0], obj])
            return TBinaryOp(
                pos,
                "==",
                _make_call(pos, "Len", [diff]),
                TIntLit(pos, 0, "0", {}),
                {},
            )
    if method == "isdisjoint":
        if len(lowered) == 1:
            inter = _make_call(pos, "Intersection", [obj, lowered[0]])
            return TBinaryOp(
                pos,
                "==",
                _make_call(pos, "Len", [inter]),
                TIntLit(pos, 0, "0", {}),
                {},
            )
    ctx.errors.append(
        LoweringError(pos.line, pos.col, "unsupported method '" + method + "' on set")
    )
    return TNilLit(pos, {})


def _lower_bytes_method(
    pos: Pos, obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower bytes method calls."""
    if method == "decode":
        return _make_call(pos, "Decode", [obj])
    if method == "upper":
        return _make_call(pos, "Upper", [obj])
    if method == "lower":
        return _make_call(pos, "Lower", [obj])
    if method == "startswith":
        return _lower_startswith_endswith(pos, "StartsWith", obj, args, env, ctx)
    if method == "endswith":
        return _lower_startswith_endswith(pos, "EndsWith", obj, args, env, ctx)
    lowered: list[TExpr] = []
    for a in args:
        lowered.append(_lower_expr(a, env, ctx))
    if method == "find":
        return _make_call(pos, "Find", [obj] + lowered)
    if method == "rfind":
        return _make_call(pos, "RFind", [obj] + lowered)
    if method == "count":
        return _make_call(pos, "Count", [obj] + lowered)
    if method == "strip":
        if not lowered:
            lowered = [TStringLit(pos, " \t\n\r\x0b\x0c", {})]
        return _make_call(pos, "Trim", [obj] + lowered)
    if method == "lstrip":
        if not lowered:
            lowered = [TStringLit(pos, " \t\n\r\x0b\x0c", {})]
        return _make_call(pos, "TrimStart", [obj] + lowered)
    if method == "rstrip":
        if not lowered:
            lowered = [TStringLit(pos, " \t\n\r\x0b\x0c", {})]
        return _make_call(pos, "TrimEnd", [obj] + lowered)
    if method == "split":
        if not lowered:
            return _make_call(pos, "SplitWhitespace", [obj])
        if len(lowered) == 2:
            plus_one = TBinaryOp(pos, "+", lowered[1], TIntLit(pos, 1, "1", {}), {})
            return _make_call(pos, "SplitN", [obj, lowered[0], plus_one])
        return _make_call(pos, "Split", [obj] + lowered)
    if method == "join":
        return _make_call(pos, "Join", [obj] + lowered)
    if method == "replace":
        if len(lowered) == 3:
            return _make_call(pos, "ReplaceCount", [obj] + lowered)
        return _make_call(pos, "Replace", [obj] + lowered)
    if method == "hex":
        return _make_method_call(pos, obj, method, lowered)
    ctx.errors.append(
        LoweringError(pos.line, pos.col, "unsupported method '" + method + "' on bytes")
    )
    return TNilLit(pos, {})


def _is_sys_argv(node: ASTNode) -> bool:
    """Check if a node is sys.argv."""
    if not _is_ast(node, "Attribute"):
        return False
    obj = get_node(node, "value")
    return (
        _is_ast(obj, "Name")
        and get_str(obj, "id") == "sys"
        and get_str(node, "attr") == "argv"
    )


def _get_const_int(node: ASTNode) -> int | None:
    """Extract a constant integer value from a Constant node."""
    if not _is_ast(node, "Constant"):
        return None
    val = node.get("value")
    if isinstance(val, JInt):
        return val.value
    return None


def _is_neg_const(node: ASTNode) -> int | None:
    """Detect a negative constant: Constant(-3) or UnaryOp(USub, Constant(3)).
    Returns the positive magnitude or None."""
    if _is_ast(node, "Constant"):
        val = node.get("value")
        if isinstance(val, JInt) and val.value < 0:
            return -val.value
    if _is_ast(node, "UnaryOp"):
        op_node = get_node(node, "op")
        if get_str(op_node, "_type") == "USub":
            operand = get_node(node, "operand")
            if _is_ast(operand, "Constant"):
                val = operand.get("value")
                if isinstance(val, JInt):
                    return val.value
    return None


def _lower_slice_bound(
    pos: Pos,
    jv: JsonValue | None,
    default: TExpr,
    env: _Env,
    ctx: _LowerCtx,
    obj: TExpr | None = None,
    obj_type: TypeNode | None = None,
) -> TExpr:
    """Lower a single slice bound, returning default if absent.
    If obj/obj_type are provided, resolves negative constants to Len(obj) - N."""
    if isinstance(jv, JDict):
        if obj is not None and obj_type is not None:
            neg = _is_neg_const(jv.entries)
            if neg is not None:
                return TBinaryOp(
                    pos,
                    "-",
                    _len_expr(pos, obj, obj_type),
                    TIntLit(pos, neg, str(neg), {}),
                    {},
                )
        return _lower_expr(jv.entries, env, ctx)
    return default


def _lower_slice(
    pos: Pos,
    obj: TExpr,
    obj_type: TypeNode,
    slice_node: ASTNode,
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Lower a slice access xs[a:b] or xs[a:b:c] into TSlice or hoisted loop."""
    lower_jv = slice_node.get("lower")
    upper_jv = slice_node.get("upper")
    step_jv = slice_node.get("step")
    has_step = step_jv is not None and not isinstance(step_jv, JNull)
    is_bytes = _is_type_dict(obj_type, ["bytes"]) or _is_bytes_slice(obj_type)
    if has_step and not isinstance(obj_type, TupleType) and not is_bytes:
        assert step_jv is not None
        return _lower_step_slice(
            pos, obj, obj_type, lower_jv, upper_jv, step_jv, env, ctx
        )
    low = _lower_slice_bound(
        pos, lower_jv, TIntLit(pos, 0, "0", {}), env, ctx, obj, obj_type
    )
    high = _lower_slice_bound(
        pos, upper_jv, _len_expr(pos, obj, obj_type), env, ctx, obj, obj_type
    )
    # For dynamic lower bounds that could be negative, wrap in Max(0, lo)
    if isinstance(lower_jv, JDict) and _is_neg_const(lower_jv.entries) is None:
        ci = _get_const_int(lower_jv.entries)
        if ci is None:
            low = _make_call(pos, "Max", [TIntLit(pos, 0, "0", {}), low])
    return TSlice(pos, obj, low, high, {})


def _lower_step_slice(
    pos: Pos,
    obj: TExpr,
    obj_type: TypeNode,
    lower_jv: JsonValue | None,
    upper_jv: JsonValue | None,
    step_jv: JsonValue,
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Lower a step-slice xs[a:b:c] into Reversed/Reverse or a hoisted for-loop."""
    is_string = _is_type_dict(obj_type, ["string"])
    no_lower = lower_jv is None or isinstance(lower_jv, JNull)
    no_upper = upper_jv is None or isinstance(upper_jv, JNull)
    # xs[::-1] / s[::-1] → Reversed(xs) / Reverse(s)
    step_entries = step_jv.entries if isinstance(step_jv, JDict) else None
    if no_lower and no_upper and step_entries is not None:
        neg = _is_neg_const(step_entries)
        if neg == 1:
            ann: Ann = {"provenance": "reversed_slice"}
            if is_string:
                return _make_call_ann(pos, "Reverse", [obj], ann)
            return _make_call_ann(pos, "Reversed", [obj], ann)
    # Arbitrary step: hoist a for-loop that accumulates elements
    step_expr = (
        _lower_expr(step_entries, env, ctx)
        if step_entries is not None
        else TIntLit(pos, 1, "1", {})
    )
    # Determine if step is negative
    step_neg = step_entries is not None and _is_neg_const(step_entries) is not None
    if no_lower:
        if step_neg:
            start = TBinaryOp(
                pos, "-", _len_expr(pos, obj, obj_type), TIntLit(pos, 1, "1", {}), {}
            )
        else:
            start = TIntLit(pos, 0, "0", {})
    else:
        start = _lower_slice_bound(
            pos, lower_jv, TIntLit(pos, 0, "0", {}), env, ctx, obj, obj_type
        )
    if no_upper:
        if step_neg:
            end = TIntLit(pos, -1, "-1", {})
        else:
            end = _len_expr(pos, obj, obj_type)
    else:
        end = _lower_slice_bound(
            pos, upper_jv, _len_expr(pos, obj, obj_type), env, ctx, obj, obj_type
        )
    # Allocate accumulator variable
    cid = ctx.comp_counter
    ctx.comp_counter = cid + 1
    rname = "__comp_" + str(cid) + "__"
    result_var = TVar(pos, rname, {})
    idx_name = "__i"
    idx_var = TVar(pos, idx_name, {})
    # Element access: obj[__i] (for strings, wrap in ToString)
    idx_expr: TExpr = TIndex(pos, obj, idx_var, {})
    if is_string:
        idx_expr = _make_call(pos, "ToString", [idx_expr])
    # Build: let acc: list[T]/string = []/""
    if is_string:
        let_type: TType = TPrimitive(pos, "string")
        let_init: TExpr = TStringLit(pos, "", {})
        # Accumulate with Concat
        append_stmt: TStmt = TAssignStmt(
            pos, result_var, _make_call(pos, "Concat", [result_var, idx_expr]), {}
        )
        body: list[TStmt] = [append_stmt]
    else:
        elt_ttype = _elem_type_from_obj(pos, obj_type)
        let_type = TListType(pos, elt_ttype)
        let_init = TListLit(pos, [], {})
        append_call = _make_call(pos, "Append", [result_var, idx_expr])
        body = [TExprStmt(pos, append_call, {})]
    let_stmt = TLetStmt(pos, rname, let_type, let_init, {})
    range_expr = TRange(pos, [start, end, step_expr], {})
    for_ann: Ann = {"provenance": "step_slice"}
    for_stmt = TForStmt(pos, [idx_name], range_expr, body, for_ann)
    env.pre_stmts.append(let_stmt)
    env.pre_stmts.append(for_stmt)
    return result_var


def _make_call_ann(pos: Pos, name: str, args: list[TExpr], ann: Ann) -> TCall:
    """Create a function call with annotations."""
    targs: list[TArg] = []
    for arg in args:
        targs.append(TArg(pos, None, arg))
    return TCall(pos, TVar(pos, name, {}), targs, ann)


def _elem_type_from_obj(pos: Pos, obj_type: TypeNode) -> TType:
    """Extract element TType from a list/slice type node."""
    if isinstance(obj_type, SliceType):
        return _typenode_to_ttype(pos, obj_type.element)
    return TPrimitive(pos, "int")


def _lower_subscript(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Subscript node."""
    pos = _node_pos(node)
    obj_node = get_node(node, "value")
    slice_node = get_node(node, "slice")
    # sys.argv subscript/slice: offset indices by -1 since Args() excludes program name
    if _is_sys_argv(obj_node):
        args_call = _make_call(pos, "Args", [])
        if _is_ast(slice_node, "Slice"):
            lower_jv = slice_node.get("lower")
            upper_jv = slice_node.get("upper")
            has_upper = upper_jv is not None and not isinstance(upper_jv, JNull)
            low_val: int | None = None
            if isinstance(lower_jv, JDict):
                low_val = _get_const_int(lower_jv.entries)
            if low_val is not None and not has_upper:
                if low_val <= 1:
                    return args_call
                low = TIntLit(pos, low_val - 1, str(low_val - 1), {})
                high = _make_call(pos, "Len", [args_call])
                return TSlice(pos, _make_call(pos, "Args", []), low, high, {})
        else:
            argv_idx = _get_const_int(slice_node)
            if argv_idx is not None and argv_idx >= 1:
                idx = TIntLit(pos, argv_idx - 1, str(argv_idx - 1), {})
                return TIndex(pos, args_call, idx, {})
    # hex(x)[2:] → FormatInt(x, 16) (hex() includes "0x" prefix, FormatInt does not)
    if (
        _is_ast(slice_node, "Slice")
        and _is_ast(obj_node, "Call")
        and _is_ast(get_node(obj_node, "func"), "Name")
        and get_str(get_node(obj_node, "func"), "id") == "hex"
    ):
        lower_jv = slice_node.get("lower")
        upper_jv = slice_node.get("upper")
        if (
            isinstance(lower_jv, JDict)
            and _get_const_int(lower_jv.entries) == 2
            and (upper_jv is None or isinstance(upper_jv, JNull))
        ):
            return _lower_expr(obj_node, env, ctx)
    obj = _lower_expr(obj_node, env, ctx)
    obj_type = _infer_expr_type(obj_node, env, ctx)
    if _is_ast(slice_node, "Slice"):
        return _lower_slice(pos, obj, obj_type, slice_node, env, ctx)
    # Tuple index: t[0] → t.0 (only for multi-element tuples, not single-element)
    if _is_type_dict(obj_type, ["Tuple"]) and not _is_single_elem_tuple(obj_type):
        if _is_ast(slice_node, "Constant"):
            idx_jv = slice_node.get("value")
            if isinstance(idx_jv, JInt):
                idx_val = idx_jv.value
                if idx_val < 0 and isinstance(obj_type, TupleType):
                    idx_val = len(obj_type.elements) + idx_val
                return TTupleAccess(pos, obj, idx_val, {})
        if _is_ast(slice_node, "UnaryOp"):
            op_node = get_node(slice_node, "op")
            if get_str(op_node, "_type") == "USub":
                operand = get_node(slice_node, "operand")
                if _is_ast(operand, "Constant"):
                    op_val_jv = operand.get("value")
                    if isinstance(op_val_jv, JInt) and isinstance(obj_type, TupleType):
                        idx_val = len(obj_type.elements) - op_val_jv.value
                        return TTupleAccess(pos, obj, idx_val, {})
    # Negative index: xs[-1] → xs[Len(xs) - 1]
    is_string = _is_type_dict(obj_type, ["string"])
    is_bytes = _is_type_dict(obj_type, ["bytes"]) or _is_bytes_slice(obj_type)
    if _is_ast(slice_node, "Constant"):
        val_jv = slice_node.get("value")
        if isinstance(val_jv, JInt) and val_jv.value < 0:
            n = -val_jv.value
            idx_expr = TBinaryOp(
                pos,
                "-",
                _len_expr(pos, obj, obj_type),
                TIntLit(pos, n, str(n), {}),
                {},
            )
            result = TIndex(pos, obj, idx_expr, {})
            if is_string:
                return _make_call(pos, "ToString", [result])
            if is_bytes:
                return _make_call(pos, "ByteToInt", [result])
            return result
    if _is_ast(slice_node, "UnaryOp"):
        op_node = get_node(slice_node, "op")
        if get_str(op_node, "_type") == "USub":
            operand = get_node(slice_node, "operand")
            if _is_ast(operand, "Constant"):
                op_val_jv = operand.get("value")
                if isinstance(op_val_jv, JInt):
                    idx_expr = TBinaryOp(
                        pos,
                        "-",
                        _len_expr(pos, obj, obj_type),
                        TIntLit(pos, op_val_jv.value, str(op_val_jv.value), {}),
                        {},
                    )
                    result = TIndex(pos, obj, idx_expr, {})
                    if is_string:
                        return _make_call(pos, "ToString", [result])
                    if is_bytes:
                        return _make_call(pos, "ByteToInt", [result])
                    return result
    # Normal index
    idx = _lower_expr(slice_node, env, ctx)
    result = TIndex(pos, obj, idx, {})
    if is_string:
        return _make_call(pos, "ToString", [result])
    if is_bytes:
        return _make_call(pos, "ByteToInt", [result])
    return result


def _lower_ternary_cond(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a ternary condition, using == nil / != nil for nil checks so the
    type checker can narrow optional variables in then/else branches."""
    pos = _node_pos(node)
    if _is_ast(node, "Compare"):
        cmp_ops = get_nodes(node, "ops")
        comps = get_nodes(node, "comparators")
        if len(cmp_ops) == 1 and len(comps) == 1:
            op_type = get_str(cmp_ops[0], "_type")
            comp = comps[0]
            left_node = get_node(node, "left")
            if (
                op_type == "Is"
                and _is_ast(comp, "Constant")
                and isinstance(comp.get("value"), JNull)
            ):
                left = _lower_expr(left_node, env, ctx)
                return TBinaryOp(pos, "==", left, TNilLit(pos, {}), {})
            if (
                op_type == "IsNot"
                and _is_ast(comp, "Constant")
                and isinstance(comp.get("value"), JNull)
            ):
                left = _lower_expr(left_node, env, ctx)
                return TBinaryOp(pos, "!=", left, TNilLit(pos, {}), {})
    return _lower_as_bool(node, env, ctx)


def _lower_ifexp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower an IfExp (ternary) node."""
    pos = _node_pos(node)
    test = get_node(node, "test")
    body = get_node(node, "body")
    orelse = get_node(node, "orelse")
    cond = _lower_ternary_cond(test, env, ctx)
    then_expr = _lower_expr(body, env, ctx)
    else_expr = _lower_expr(orelse, env, ctx)
    return TTernary(pos, cond, then_expr, else_expr, {})


def _lower_list_literal(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a List literal, desugaring star unpacking to Concat chains."""
    pos = _node_pos(node)
    elts = get_nodes(node, "elts")
    has_starred = False
    for e in elts:
        if _is_ast(e, "Starred"):
            has_starred = True
            break
    if not has_starred:
        elements: list[TExpr] = []
        for e in elts:
            elements.append(_lower_expr(e, env, ctx))
        return TListLit(pos, elements, {})
    # Group into runs of plain elements and starred expressions
    parts: list[TExpr] = []
    plain: list[TExpr] = []
    for e in elts:
        if _is_ast(e, "Starred"):
            if plain:
                parts.append(TListLit(pos, plain, {}))
                plain = []
            parts.append(_lower_expr(get_node(e, "value"), env, ctx))
        else:
            plain.append(_lower_expr(e, env, ctx))
    if plain:
        parts.append(TListLit(pos, plain, {}))
    result = parts[0]
    ann: Ann = {"provenance": "star_unpack"}
    for part in parts[1:]:
        call = _make_call(pos, "Concat", [result, part])
        call.annotations = ann
        result = call
    return result


def _lower_dict_literal(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Dict literal."""
    pos = _node_pos(node)
    keys = get_nodes(node, "keys")
    values = get_nodes(node, "values")
    entries: list[tuple[TExpr, TExpr]] = []
    i = 0
    while i < len(keys):
        k = keys[i]
        v: ASTNode | None = values[i] if i < len(values) else None
        if v is not None:
            entries.append((_lower_expr(k, env, ctx), _lower_expr(v, env, ctx)))
        i += 1
    if not entries:
        return _make_call(pos, "Map", [])
    return TMapLit(pos, entries, {})


def _lower_set_literal(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Set literal."""
    pos = _node_pos(node)
    elts = get_nodes(node, "elts")
    elements: list[TExpr] = []
    for e in elts:
        elements.append(_lower_expr(e, env, ctx))
    return TSetLit(pos, elements, {})


def _lower_list_from_tuple(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Tuple AST node as a list literal (for variadic tuples)."""
    pos = _node_pos(node)
    elts = get_nodes(node, "elts")
    elements: list[TExpr] = []
    for e in elts:
        elements.append(_lower_expr(e, env, ctx))
    return TListLit(pos, elements, {})


def _lower_tuple_literal(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Tuple literal."""
    pos = _node_pos(node)
    elts = get_nodes(node, "elts")
    elements: list[TExpr] = []
    for e in elts:
        elements.append(_lower_expr(e, env, ctx))
    if not elements:
        return TListLit(pos, [], {})
    if len(elements) == 1:
        return TListLit(pos, elements, {})
    return TTupleLit(pos, elements, {})


def _lower_fstring(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a JoinedStr (f-string) node to Format(template, args)."""
    pos = _node_pos(node)
    values = get_nodes(node, "values")
    template_parts: list[str] = []
    fmt_args: list[TExpr] = []
    for v in values:
        vtype = get_str(v, "_type")
        if vtype == "Constant":
            val = v.get("value")
            if isinstance(val, JStr):
                template_parts.append(val.value)
        elif vtype == "FormattedValue":
            template_parts.append("{}")
            inner = get_node(v, "value")
            lowered: TExpr = _lower_expr(inner, env, ctx)
            conversion = get_int(v, "conversion")
            if conversion == 114:  # !r
                lowered = _make_call(pos, "ToRepr", [lowered])
            elif conversion == 115:  # !s
                lowered = _make_call(pos, "ToString", [lowered])
            fmt_args.append(lowered)
    template = "".join(template_parts)
    all_args: list[TExpr] = [TStringLit(pos, template, {})]
    for _fa in fmt_args:
        all_args.append(_fa)
    return _make_call(pos, "Format", all_args)


def _lower_any_all(fname: str, node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower any(genexpr)/all(genexpr) via pre_stmts hoisting."""
    pos = _node_pos(node)
    elt = get_node(node, "elt")
    generators = get_nodes(node, "generators")
    is_any = fname == "any"
    default_val = not is_any
    if not generators:
        return TBoolLit(pos, default_val, {})
    gen = generators[0]
    if not isinstance(gen, dict):
        return TBoolLit(pos, default_val, {})
    target = get_node(gen, "target")
    iter_node = get_node(gen, "iter")
    binding, b_ann = _extract_binding(target)
    iter_expr = _lower_extend_arg(iter_node, env, ctx)
    comp_env = env.copy()
    for b in binding:
        comp_env.declared.add(b)
    elt_expr = _lower_expr(elt, comp_env, ctx)
    cid = ctx.comp_counter
    ctx.comp_counter = cid + 1
    rname = "__comp_" + str(cid) + "__"
    result_var = TVar(pos, rname, {})
    bool_type: TType = TPrimitive(pos, "bool")
    let_stmt = TLetStmt(pos, rname, bool_type, TBoolLit(pos, default_val, {}), {})
    cond: TExpr = elt_expr
    if not is_any:
        cond = TUnaryOp(pos, "!", elt_expr, {})
    set_val = TBoolLit(pos, is_any, {})
    inner_body: list[TStmt] = [
        TAssignStmt(pos, result_var, set_val, {}),
        TBreakStmt(pos, {}),
    ]
    gen_ifs = get_nodes(gen, "ifs")
    if gen_ifs and isinstance(gen_ifs[0], dict):
        gen_cond = _lower_as_bool(gen_ifs[0], comp_env, ctx)
        check_body: list[TStmt] = [TIfStmt(pos, cond, inner_body, None, {})]
        body: list[TStmt] = [TIfStmt(pos, gen_cond, check_body, None, {})]
    else:
        body = [TIfStmt(pos, cond, inner_body, None, {})]
    for_ann: Ann = {}
    for_ann.update(b_ann)
    for_stmt = TForStmt(pos, binding, iter_expr, body, for_ann)
    env.pre_stmts.append(let_stmt)
    env.pre_stmts.append(for_stmt)
    return result_var


def _set_comp_var_types(
    binding: list[str],
    iter_node: ASTNode,
    comp_env: _Env,
    env: _Env,
    ctx: _LowerCtx,
) -> None:
    """Set loop variable types in comp_env from the iter expression type."""
    iter_type = _infer_expr_type(iter_node, env, ctx)
    elem_type: TypeNode = VOID_TYPE
    if isinstance(iter_type, SliceType):
        elem_type = iter_type.element
    elif isinstance(iter_type, SetType):
        elem_type = iter_type.element
    if _is_type_dict(elem_type, ["void"]):
        return
    if len(binding) == 1:
        comp_env.var_types[binding[0]] = elem_type
    elif isinstance(elem_type, TupleType) and len(elem_type.elements) == len(binding):
        bi = 0
        while bi < len(binding):
            comp_env.var_types[binding[bi]] = elem_type.elements[bi]
            bi += 1


def _lower_listcomp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a ListComp/GeneratorExp in expression context via pre_stmts hoisting."""
    pos = _node_pos(node)
    elt = get_node(node, "elt")
    generators = get_nodes(node, "generators")
    if not generators:
        return TListLit(pos, [], {})
    gen = generators[0]
    if not isinstance(gen, dict):
        return TListLit(pos, [], {})
    # Build innermost env with all generator bindings and types
    comp_env = env.copy()
    for g in generators:
        if isinstance(g, dict):
            gt = get_node(g, "target")
            g_iter = get_node(g, "iter")
            gb, _ = _extract_binding(gt)
            for b in gb:
                comp_env.declared.add(b)
            _set_comp_var_types(gb, g_iter, comp_env, env, ctx)
    elt_expr = _lower_expr(elt, comp_env, ctx)
    # Infer element type for the result list
    elt_type_node = _infer_expr_type(elt, comp_env, ctx)
    elt_ttype = _typenode_to_ttype(pos, elt_type_node)
    if isinstance(elt_ttype, TPrimitive) and elt_ttype.kind == "void":
        elt_ttype = TPrimitive(pos, "int")
    cid = ctx.comp_counter
    ctx.comp_counter = cid + 1
    rname = "__comp_" + str(cid) + "__"
    result_var = TVar(pos, rname, {})
    list_type: TType = TListType(pos, elt_ttype)
    let_stmt = TLetStmt(pos, rname, list_type, TListLit(pos, [], {}), {})
    append_call = _make_call(pos, "Append", [result_var, elt_expr])
    body: list[TStmt] = [TExprStmt(pos, append_call, {})]
    # Build nested for loops from innermost to outermost
    gi = len(generators) - 1
    while gi >= 0:
        g = generators[gi]
        if isinstance(g, dict):
            gt = get_node(g, "target")
            g_iter = get_node(g, "iter")
            gb, g_ann = _extract_binding(gt)
            g_iter_expr = _lower_extend_arg(g_iter, comp_env, ctx)
            g_ifs = get_nodes(g, "ifs")
            if g_ifs and isinstance(g_ifs[0], dict):
                cond = _lower_as_bool(g_ifs[0], comp_env, ctx)
                body = [TIfStmt(pos, cond, body, None, {})]
            f_ann: Ann = {}
            f_ann.update(g_ann)
            body = [TForStmt(pos, gb, g_iter_expr, body, f_ann)]
        gi -= 1
    env.pre_stmts.append(let_stmt)
    for s in body:
        env.pre_stmts.append(s)
    return result_var


def _lower_setcomp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a SetComp in expression context via pre_stmts hoisting."""
    pos = _node_pos(node)
    elt = get_node(node, "elt")
    generators = get_nodes(node, "generators")
    empty_set = _make_call(pos, "Set", [])
    if not generators:
        return empty_set
    gen = generators[0]
    if not isinstance(gen, dict):
        return empty_set
    target = get_node(gen, "target")
    iter_node = get_node(gen, "iter")
    binding, b_ann = _extract_binding(target)
    iter_expr = _lower_extend_arg(iter_node, env, ctx)
    comp_env = env.copy()
    for b in binding:
        comp_env.declared.add(b)
    elt_expr = _lower_expr(elt, comp_env, ctx)
    cid = ctx.comp_counter
    ctx.comp_counter = cid + 1
    rname = "__comp_" + str(cid) + "__"
    result_var = TVar(pos, rname, {})
    set_type: TType = TSetType(pos, TPrimitive(pos, "int"))
    let_stmt = TLetStmt(pos, rname, set_type, empty_set, {})
    add_call = _make_call(pos, "Add", [result_var, elt_expr])
    body: list[TStmt] = [TExprStmt(pos, add_call, {})]
    ifs = get_nodes(gen, "ifs")
    if ifs and isinstance(ifs[0], dict):
        cond = _lower_as_bool(ifs[0], comp_env, ctx)
        body = [TIfStmt(pos, cond, body, None, {})]
    for_ann: Ann = {}
    for_ann.update(b_ann)
    for_stmt = TForStmt(pos, binding, iter_expr, body, for_ann)
    env.pre_stmts.append(let_stmt)
    env.pre_stmts.append(for_stmt)
    return result_var


def _lower_dictcomp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a DictComp in expression context via pre_stmts hoisting."""
    pos = _node_pos(node)
    key_node = get_node(node, "key")
    value_node = get_node(node, "value")
    generators = get_nodes(node, "generators")
    if not generators:
        return _make_call(pos, "Map", [])
    gen = generators[0]
    if not isinstance(gen, dict):
        return _make_call(pos, "Map", [])
    target = get_node(gen, "target")
    iter_node = get_node(gen, "iter")
    binding, b_ann = _extract_binding(target)
    # Detect .items() early for type inference
    is_items = False
    if _is_ast(iter_node, "Call"):
        iter_func = get_node(iter_node, "func")
        if _is_ast(iter_func, "Attribute") and get_str(iter_func, "attr") == "items":
            is_items = True
    # Set loop variable types in comp_env for type inference
    comp_env = env.copy()
    for b in binding:
        comp_env.declared.add(b)
    if is_items:
        dict_node = get_node(get_node(iter_node, "func"), "value")
        dict_type = _infer_expr_type(dict_node, env, ctx)
        if isinstance(dict_type, MapType) and len(binding) == 2:
            comp_env.var_types[binding[0]] = dict_type.key
            comp_env.var_types[binding[1]] = dict_type.value
    else:
        _set_comp_var_types(binding, iter_node, comp_env, env, ctx)
    key_expr = _lower_expr(key_node, comp_env, ctx)
    val_expr = _lower_expr(value_node, comp_env, ctx)
    # Infer key/value types from comprehension expressions
    key_type_node = _infer_expr_type(key_node, comp_env, ctx)
    val_type_node = _infer_expr_type(value_node, comp_env, ctx)
    key_ttype = _typenode_to_ttype(pos, key_type_node)
    val_ttype = _typenode_to_ttype(pos, val_type_node)
    if isinstance(key_ttype, TPrimitive) and key_ttype.kind == "void":
        key_ttype = TPrimitive(pos, "string")
    if isinstance(val_ttype, TPrimitive) and val_ttype.kind == "void":
        val_ttype = TPrimitive(pos, "string")
    cid = ctx.comp_counter
    ctx.comp_counter = cid + 1
    rname = "__comp_" + str(cid) + "__"
    result_var = TVar(pos, rname, {})
    map_type: TType = TMapType(pos, key_ttype, val_ttype)
    let_stmt = TLetStmt(pos, rname, map_type, _make_call(pos, "Map", []), {})
    idx_target = TIndex(pos, result_var, key_expr, {})
    body: list[TStmt] = [TAssignStmt(pos, idx_target, val_expr, {})]
    ifs = get_nodes(gen, "ifs")
    if ifs and isinstance(ifs[0], dict):
        cond = _lower_as_bool(ifs[0], comp_env, ctx)
        body = [TIfStmt(pos, cond, body, None, {})]
    for_ann: Ann = {}
    for_ann.update(b_ann)
    if is_items:
        for_ann["for.items"] = "true"
        iter_expr = _lower_expr(
            get_node(get_node(iter_node, "func"), "value"), env, ctx
        )
    else:
        iter_expr = _lower_extend_arg(iter_node, env, ctx)
    for_stmt = TForStmt(pos, binding, iter_expr, body, for_ann)
    env.pre_stmts.append(let_stmt)
    env.pre_stmts.append(for_stmt)
    return result_var


def _expand_listcomp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Expand a ListComp into: let __result__ = []; for x in xs { Append(__result__, elt) }; return __result__."""
    pos = _node_pos(node)
    elt = get_node(node, "elt")
    generators = get_nodes(node, "generators")
    if not generators:
        return [TReturnStmt(pos, TListLit(pos, [], {}), {})]
    gen = generators[0]
    if not isinstance(gen, dict):
        return [TReturnStmt(pos, TListLit(pos, [], {}), {})]
    target = get_node(gen, "target")
    iter_node = get_node(gen, "iter")
    orig_name = get_str(target, "id")
    target_name = _safe_name(orig_name)
    t_ann = _name_ann(target_name, orig_name)
    iter_expr = _lower_expr(iter_node, env, ctx)
    # Add loop var to env
    comp_env = env.copy()
    comp_env.declared.add(orig_name)
    elt_expr = _lower_expr(elt, comp_env, ctx)
    result_var = TVar(pos, "__result__", {})
    # Build: let __result__: list[...] = []
    ret_type = env.return_type
    result_type: TType = _typenode_to_ttype(pos, ret_type)
    let_stmt = TLetStmt(pos, "__result__", result_type, TListLit(pos, [], {}), {})
    # Build: for target_name in iter { Append(__result__, elt) }
    append_call = _make_call(pos, "Append", [result_var, elt_expr])
    body: list[TStmt] = [TExprStmt(pos, append_call, {})]
    # Handle optional filter (ifs in generator)
    ifs = get_nodes(gen, "ifs")
    if ifs and isinstance(ifs[0], dict):
        cond = _lower_as_bool(ifs[0], comp_env, ctx)
        body = [TIfStmt(pos, cond, body, None, {})]
    for_stmt = TForStmt(pos, [target_name], iter_expr, body, t_ann)
    # Return __result__
    return_stmt = TReturnStmt(pos, result_var, {})
    return [let_stmt, for_stmt, return_stmt]


def _expand_setcomp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Expand a SetComp into: let __result__ = {}; for x in xs { Add(__result__, elt) }; return __result__."""
    pos = _node_pos(node)
    elt = get_node(node, "elt")
    generators = get_nodes(node, "generators")
    empty_set = _make_call(pos, "Set", [])
    if not generators:
        return [TReturnStmt(pos, empty_set, {})]
    gen = generators[0]
    if not isinstance(gen, dict):
        return [TReturnStmt(pos, empty_set, {})]
    target = get_node(gen, "target")
    iter_node = get_node(gen, "iter")
    orig_name = get_str(target, "id")
    target_name = _safe_name(orig_name)
    t_ann = _name_ann(target_name, orig_name)
    iter_expr = _lower_expr(iter_node, env, ctx)
    comp_env = env.copy()
    comp_env.declared.add(orig_name)
    elt_expr = _lower_expr(elt, comp_env, ctx)
    result_var = TVar(pos, "__result__", {})
    ret_type = env.return_type
    result_type: TType = _typenode_to_ttype(pos, ret_type)
    let_stmt = TLetStmt(pos, "__result__", result_type, empty_set, {})
    add_call = _make_call(pos, "Add", [result_var, elt_expr])
    body: list[TStmt] = [TExprStmt(pos, add_call, {})]
    ifs = get_nodes(gen, "ifs")
    if ifs and isinstance(ifs[0], dict):
        cond = _lower_as_bool(ifs[0], comp_env, ctx)
        body = [TIfStmt(pos, cond, body, None, {})]
    for_stmt = TForStmt(pos, [target_name], iter_expr, body, t_ann)
    return_stmt = TReturnStmt(pos, result_var, {})
    return [let_stmt, for_stmt, return_stmt]


def _expand_dictcomp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Expand a DictComp into: let __result__ = Map(); for k in xs { __result__[key] = val }; return __result__."""
    pos = _node_pos(node)
    key_node = get_node(node, "key")
    value_node = get_node(node, "value")
    generators = get_nodes(node, "generators")
    if not generators:
        return [TReturnStmt(pos, _make_call(pos, "Map", []), {})]
    gen = generators[0]
    if not isinstance(gen, dict):
        return [TReturnStmt(pos, _make_call(pos, "Map", []), {})]
    target = get_node(gen, "target")
    iter_node = get_node(gen, "iter")
    orig_name = get_str(target, "id")
    target_name = _safe_name(orig_name)
    t_ann = _name_ann(target_name, orig_name)
    iter_expr = _lower_expr(iter_node, env, ctx)
    comp_env = env.copy()
    comp_env.declared.add(orig_name)
    key_expr = _lower_expr(key_node, comp_env, ctx)
    val_expr = _lower_expr(value_node, comp_env, ctx)
    result_var = TVar(pos, "__result__", {})
    ret_type = env.return_type
    result_type: TType = _typenode_to_ttype(pos, ret_type)
    let_stmt = TLetStmt(pos, "__result__", result_type, _make_call(pos, "Map", []), {})
    idx_target = TIndex(pos, result_var, key_expr, {})
    body: list[TStmt] = [TAssignStmt(pos, idx_target, val_expr, {})]
    ifs = get_nodes(gen, "ifs")
    if ifs and isinstance(ifs[0], dict):
        cond = _lower_as_bool(ifs[0], comp_env, ctx)
        body = [TIfStmt(pos, cond, body, None, {})]
    for_stmt = TForStmt(pos, [target_name], iter_expr, body, t_ann)
    return_stmt = TReturnStmt(pos, result_var, {})
    return [let_stmt, for_stmt, return_stmt]


# ---------------------------------------------------------------------------
# Truthiness (as_bool)
# ---------------------------------------------------------------------------


def _lower_as_bool(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower an expression as a boolean condition."""
    pos = _node_pos(node)
    # Structural cases that always need truthiness lowering, regardless of type
    t = get_str(node, "_type")
    if t == "Dict":
        return TBoolLit(pos, len(get_nodes(node, "keys")) > 0, {})
    if t == "Set" or t == "List":
        return TBoolLit(pos, len(get_nodes(node, "elts")) > 0, {})
    if t == "Compare":
        return _lower_expr(node, env, ctx)
    if t == "BoolOp":
        op_node = get_node(node, "op")
        op_str = "&&" if get_str(op_node, "_type") == "And" else "||"
        values = get_nodes(node, "values")
        if not values:
            return TBoolLit(pos, True, {})
        result: TExpr = _lower_as_bool(values[0], env, ctx)
        for val in values[1:]:
            right = _lower_as_bool(val, env, ctx)
            result = TBinaryOp(pos, op_str, result, right, {})
        return result
    expr_type = _infer_expr_type(node, env, ctx)
    if _is_type_dict(expr_type, ["bool"]):
        return _lower_expr(node, env, ctx)
    if _is_optional_type(expr_type):
        expr = _lower_expr(node, env, ctx)
        if isinstance(expr, TVar):
            return TBinaryOp(pos, "!=", expr, TNilLit(pos, {}), {})
        return TUnaryOp(pos, "!", _make_call(pos, "IsNil", [expr]), {})
    if _is_interface_type(expr_type):
        expr = _lower_expr(node, env, ctx)
        if isinstance(expr, TVar):
            return TBinaryOp(pos, "!=", expr, TNilLit(pos, {}), {})
        return TUnaryOp(pos, "!", _make_call(pos, "IsNil", [expr]), {})
    # Inline truthiness for known types
    if _is_type_dict(expr_type, ["string"]):
        expr = _lower_expr(node, env, ctx)
        return TBinaryOp(pos, "!=", expr, TStringLit(pos, "", {}), {})
    if _is_type_dict(expr_type, ["int"]):
        expr = _lower_expr(node, env, ctx)
        return TBinaryOp(pos, "!=", expr, TIntLit(pos, 0, "0", {}), {})
    if _is_type_dict(expr_type, ["float"]):
        expr = _lower_expr(node, env, ctx)
        return TBinaryOp(pos, "!=", expr, TFloatLit(pos, 0.0, "0.0", {}), {})
    if isinstance(expr_type, TupleType):
        if expr_type.elements:
            return TBoolLit(pos, True, {})
        return TBoolLit(pos, False, {})
    if _is_type_dict(expr_type, ["bytes", "Slice", "Map", "Set"]):
        expr = _lower_expr(node, env, ctx)
        return TBinaryOp(
            pos,
            "!=",
            _make_call(pos, "Len", [expr]),
            TIntLit(pos, 0, "0", {}),
            {},
        )
    return _lower_expr(node, env, ctx)


# ---------------------------------------------------------------------------
# Statement lowering
# ---------------------------------------------------------------------------


def _lower_with_open(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower with-open file I/O: read → ReadFile, write → WriteFile."""
    pos = _node_pos(node)
    items = get_nodes(node, "items")
    item = items[0]
    ctx_expr = get_node(item, "context_expr")
    args = get_nodes(ctx_expr, "args")
    path_expr = _lower_expr(args[0], env, ctx)
    mode = get_str(args[1], "value")
    body = get_nodes(node, "body")
    stmt = body[0]
    if mode == "rb":
        # with open(path, "rb") as f: raw = f.read() → raw = ReadFileBytes(path)
        targets = get_nodes(stmt, "targets")
        target_node = targets[0]
        name = get_str(target_node, "id")
        safe = _safe_name(name)
        ann = _name_ann(safe, name)
        call = _make_call(pos, "ReadFileBytes", [path_expr])
        val_type: TypeNode = PrimitiveType("bytes")
        if name not in env.declared:
            env.declared.add(name)
            env.var_types[name] = val_type
            ttype = _typenode_to_ttype(pos, val_type)
            return [TLetStmt(pos, safe, ttype, call, ann)]
        if name in env.hoisted_stmts:
            _backpatch_hoisted(pos, name, val_type, env)
        target: TExpr = TVar(pos, safe, ann)
        return [TAssignStmt(pos, target, call, {})]
    # mode "w" or "wb": with open(path, "w") as f: f.write(data) → WriteFile(path, data)
    call_node = get_node(stmt, "value")
    data_args = get_nodes(call_node, "args")
    data_expr = _lower_expr(data_args[0], env, ctx)
    call = _make_call(pos, "WriteFile", [path_expr, data_expr])
    return [TExprStmt(pos, call, {})]


def _lower_stmts(stmts: list[ASTNode], env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower a list of statements."""
    result: list[TStmt] = []
    for stmt_node in stmts:
        lowered = _lower_stmt(stmt_node, env, ctx)
        if env.pre_stmts:
            pre = env.pre_stmts
            env.pre_stmts = []
            for p in pre:
                result.append(p)
        for lo in lowered:
            result.append(lo)
    return result


def _lower_stmt(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower a single statement, may produce multiple IR statements."""
    pos = _node_pos(node)
    t = get_str(node, "_type")
    if t == "Return":
        return _lower_return(node, env, ctx)
    if t == "Assign":
        return _lower_assign(node, env, ctx)
    if t == "AnnAssign":
        return _lower_ann_assign(node, env, ctx)
    if t == "AugAssign":
        return _lower_aug_assign(node, env, ctx)
    if t == "If":
        return _lower_if(node, env, ctx)
    if t == "While":
        return _lower_while(node, env, ctx)
    if t == "For":
        return _lower_for(node, env, ctx)
    if t == "Expr":
        return _lower_expr_stmt(node, env, ctx)
    if t == "Try":
        return _lower_try(node, env, ctx)
    if t == "Raise":
        return _lower_raise(node, env, ctx)
    if t == "Assert":
        return _lower_assert(node, env, ctx)
    if t == "Break":
        return [TBreakStmt(pos, {})]
    if t == "Continue":
        return [TContinueStmt(pos, {})]
    if t == "With":
        return _lower_with_open(node, env, ctx)
    if t == "Match":
        return _lower_match(node, env, ctx)
    if t == "Pass":
        return []
    if t == "Delete":
        return _lower_delete(node, env, ctx)
    if t == "Import" or t == "ImportFrom":
        return []
    return []


def _lower_delete(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower del d[key] → Delete(d, key); del xs[i] → RemoveAt(xs, i)."""
    pos = _node_pos(node)
    result: list[TStmt] = []
    for target in get_nodes(node, "targets"):
        if get_str(target, "_type") != "Subscript":
            continue
        obj_node = get_node(target, "value")
        key_node = get_node(target, "slice")
        obj_type = _unwrap_pointer(_infer_expr_type(obj_node, env, ctx))
        obj = _lower_expr(obj_node, env, ctx)
        key = _lower_expr(key_node, env, ctx)
        ann: Ann = {"provenance": "del_subscript"}
        if _is_type_dict(obj_type, ["Map"]):
            result.append(TExprStmt(pos, _make_call(pos, "Delete", [obj, key]), ann))
        elif _is_type_dict(obj_type, ["Slice"]):
            result.append(TExprStmt(pos, _make_call(pos, "RemoveAt", [obj, key]), ann))
    return result


def _lower_return(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    pos = _node_pos(node)
    val_jv = node.get("value")
    if val_jv is None or isinstance(val_jv, JNull):
        return [TReturnStmt(pos, None, {})]
    ret_type = env.return_type
    if isinstance(ret_type, PrimitiveType) and ret_type.kind == "void":
        return [TReturnStmt(pos, None, {})]
    if isinstance(val_jv, JDict):
        return_val = val_jv.entries
        if _is_ast(return_val, "ListComp") or _is_ast(return_val, "GeneratorExp"):
            return _expand_listcomp(return_val, env, ctx)
        if _is_ast(return_val, "SetComp"):
            return _expand_setcomp(return_val, env, ctx)
        if _is_ast(return_val, "DictComp"):
            return _expand_dictcomp(return_val, env, ctx)
        expr = _lower_expr(return_val, env, ctx)
        return [TReturnStmt(pos, expr, {})]
    return [TReturnStmt(pos, None, {})]


def _lower_assign(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower an assignment statement."""
    pos = _node_pos(node)
    targets = get_nodes(node, "targets")
    value_node = get_node(node, "value")
    if not targets:
        return []
    target_node = targets[0]
    if not isinstance(target_node, dict):
        return []
    # Tuple unpacking: a, b = expr
    if _is_ast(target_node, "Tuple"):
        return _lower_tuple_assign(target_node, value_node, env, ctx)
    # Simple assignment
    if _is_ast(target_node, "Name"):
        name = get_str(target_node, "id")
        if name == "_":
            # _ = expr → just evaluate as expr statement
            expr = _lower_expr(value_node, env, ctx)
            return [TExprStmt(pos, expr, {})]
        value = _lower_expr(value_node, env, ctx)
        val_type: TypeNode = _infer_expr_type(value_node, env, ctx)
        if _is_type_dict(val_type, ["void"]):
            val_type = PrimitiveType("error")
        safe = _safe_name(name)
        ann = _name_ann(safe, name)
        if name not in env.declared:
            env.declared.add(name)
            env.var_types[name] = val_type
            ttype = _typenode_to_ttype(pos, val_type)
            stmts: list[TStmt] = [TLetStmt(pos, safe, ttype, value, ann)]
            stmts.extend(_method_side_effects(value_node, env, ctx))
            return stmts
        # Re-assignment
        if name in env.hoisted_stmts:
            _backpatch_hoisted(pos, name, val_type, env)
        target: TExpr = TVar(pos, safe, ann)
        stmts: list[TStmt] = [TAssignStmt(pos, target, value, {})]
        stmts.extend(_method_side_effects(value_node, env, ctx))
        return stmts
    # Attribute assignment: obj.field = expr
    if _is_ast(target_node, "Attribute"):
        attr = get_str(target_node, "attr")
        obj_node = get_node(target_node, "value")
        obj = _lower_expr(obj_node, env, ctx)
        target_fa: TExpr = TFieldAccess(pos, obj, attr, {})
        value = _lower_expr(value_node, env, ctx)
        return [TAssignStmt(pos, target_fa, value, {})]
    # Subscript assignment: xs[i] = expr
    if _is_ast(target_node, "Subscript"):
        obj_node = get_node(target_node, "value")
        slice_node = get_node(target_node, "slice")
        obj = _lower_expr(obj_node, env, ctx)
        # Slice assignment: xs[a:b] = ys → ReplaceSlice(xs, a, b, ys)
        if _is_ast(slice_node, "Slice"):
            obj_type = _infer_expr_type(obj_node, env, ctx)
            lower_jv = slice_node.get("lower")
            upper_jv = slice_node.get("upper")
            low = _lower_slice_bound(
                pos, lower_jv, TIntLit(pos, 0, "0", {}), env, ctx, obj, obj_type
            )
            high = _lower_slice_bound(
                pos, upper_jv, _len_expr(pos, obj, obj_type), env, ctx, obj, obj_type
            )
            value = _lower_expr(value_node, env, ctx)
            call = _make_call(pos, "ReplaceSlice", [obj, low, high, value])
            return [TExprStmt(pos, call, {})]
        idx = _lower_expr(slice_node, env, ctx)
        target = TIndex(pos, obj, idx, {})
        value = _lower_expr(value_node, env, ctx)
        return [TAssignStmt(pos, target, value, {})]
    return []


def _lower_tuple_assign(
    target_node: ASTNode, value_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> list[TStmt]:
    """Lower tuple unpacking assignment: a, b = expr."""
    pos = _node_pos(target_node)
    elts = get_nodes(target_node, "elts")
    # Special case: a, b = divmod(x, y) → DivMod(x, y)
    if _is_ast(value_node, "Call"):
        vfunc = get_node(value_node, "func")
        if _is_ast(vfunc, "Name") and get_str(vfunc, "id") == "divmod":
            vargs = get_nodes(value_node, "args")
            lowered_args: list[TExpr] = []
            arg_types: list[TypeNode | None] = []
            for a in vargs:
                at = _infer_expr_type(a, env, ctx)
                la = _lower_expr(a, env, ctx)
                if _is_type_dict(at, ["bool"]):
                    la = _bool_to_int(pos, la)
                    at = INT_TYPE
                lowered_args.append(la)
                arg_types.append(at)
            use_float = False
            for at2 in arg_types:
                if _is_type_dict(at2, ["float"]):
                    use_float = True
            if use_float:
                fa: list[TExpr] = []
                for la2 in lowered_args:
                    fa.append(la2)
                a_expr = fa[0] if fa else lowered_args[0]
                b_expr = fa[1] if len(fa) > 1 else lowered_args[1]
                value = TTupleLit(
                    pos,
                    [
                        _make_call(pos, "FloorDiv", [a_expr, b_expr]),
                        _make_call(pos, "PythonMod", [a_expr, b_expr]),
                    ],
                    {},
                )
            else:
                value = _make_call(pos, "DivMod", lowered_args)
            result_kind = "float" if use_float else "int"
            stmts: list[TStmt] = []
            targets: list[TExpr] = []
            for e in elts:
                if _is_ast(e, "Name"):
                    name = get_str(e, "id")
                    safe = _safe_name(name)
                    ann = _name_ann(safe, name)
                    if name not in env.declared:
                        env.declared.add(name)
                        env.var_types[name] = PrimitiveType(result_kind)
                        prim = TPrimitive(pos, result_kind)
                        init: TExpr = TIntLit(pos, 0, "0", {})
                        if use_float:
                            init = TFloatLit(pos, 0.0, "0.0", {})
                        stmts.append(TLetStmt(pos, safe, prim, init, ann))
                    elif name in env.hoisted_stmts:
                        _backpatch_hoisted(pos, name, PrimitiveType(result_kind), env)
                    targets.append(TVar(pos, safe, ann))
            stmts.append(TTupleAssignStmt(pos, targets, value, {}))
            return stmts
    value = _lower_expr(value_node, env, ctx)
    val_type = _infer_expr_type(value_node, env, ctx)
    elem_types: list[TypeNode] = []
    if isinstance(val_type, TupleType):
        for elem in val_type.elements:
            elem_types.append(elem)
    stmts: list[TStmt] = []
    targets: list[TExpr] = []
    for i, e in enumerate(elts):
        if _is_ast(e, "Name"):
            name = get_str(e, "id")
            safe = _safe_name(name)
            ann = _name_ann(safe, name)
            if name not in env.declared:
                env.declared.add(name)
                et = elem_types[i] if i < len(elem_types) else INT_TYPE
                env.var_types[name] = et
                ttype = _typenode_to_ttype(pos, et)
                init = _default_value_for_type(pos, et)
                stmts.append(TLetStmt(pos, safe, ttype, init, ann))
            elif name in env.hoisted_stmts:
                et = elem_types[i] if i < len(elem_types) else INT_TYPE
                _backpatch_hoisted(pos, name, et, env)
            targets.append(TVar(pos, safe, ann))
    stmts.append(TTupleAssignStmt(pos, targets, value, {}))
    return stmts


def _lower_ann_assign(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower an annotated assignment: x: int = 10."""
    pos = _node_pos(node)
    target_node = get_node(node, "target")
    ann_jv = node.get("annotation")
    value_jv = node.get("value")
    if not _is_ast(target_node, "Name"):
        return []
    name = get_str(target_node, "id")
    already_declared = name in env.declared
    # Get type from annotation
    ann_str = ""
    if isinstance(ann_jv, JDict):
        ann_str = annotation_to_str(ann_jv.entries)
    type_dict: TypeNode = VOID_TYPE
    if ann_str:
        errors: list[str] = []
        type_dict = py_type_to_type_dict(ann_str, ctx.known_classes, [], 0, 0)
    ttype = _typenode_to_ttype(pos, type_dict)
    if isinstance(ttype, TPrimitive) and ttype.kind == "void":
        ttype = TPrimitive(pos, "nil")
    env.declared.add(name)
    env.var_types[name] = type_dict
    if name in env.hoisted_stmts:
        _backpatch_hoisted(pos, name, type_dict, env)
        env.hoisted_stmts.pop(name)
    safe = _safe_name(name)
    ann = _name_ann(safe, name)
    val: TExpr | None = None
    stmts: list[TStmt] = []
    if isinstance(value_jv, JDict):
        value_node = value_jv.entries
        if (
            isinstance(type_dict, OptionalType)
            and _is_ast(value_node, "Call")
            and _is_ast(get_node(value_node, "func"), "Name")
            and _is_type_dict(
                _func_return_type(ctx, get_str(get_node(value_node, "func"), "id")),
                ["void"],
            )
        ):
            stmts.append(TExprStmt(pos, _lower_expr(value_node, env, ctx), {}))
            val = TNilLit(pos, {})
        elif _is_variadic_tuple(type_dict) and _is_ast(value_node, "Tuple"):
            val = _lower_list_from_tuple(value_node, env, ctx)
        elif _is_single_elem_tuple(type_dict) and _is_ast(value_node, "Tuple"):
            # tuple[int] = (42,) → list[int] = [42]
            val = _lower_list_from_tuple(value_node, env, ctx)
        elif _is_set_of_genexpr(value_node):
            # set(genexpr) → expand inline
            genexpr = get_nodes(value_node, "args")[0]
            if already_declared:
                target = TVar(pos, safe, ann)
                stmts.append(TAssignStmt(pos, target, _make_call(pos, "Set", []), {}))
            else:
                stmts.append(
                    TLetStmt(pos, safe, ttype, _make_call(pos, "Set", []), ann)
                )
            stmts.extend(_expand_genexpr_to_set_add(safe, genexpr, env, ctx))
            return stmts
        elif _is_map_type(type_dict) and _is_ast(value_node, "Dict"):
            # dict literal — convert bool keys to int when target type is int
            val = _lower_dict_literal_typed(value_node, type_dict, env, ctx)
        else:
            val = _lower_expr(value_node, env, ctx)
        if val is not None:
            if already_declared:
                target = TVar(pos, safe, ann)
                stmts.append(TAssignStmt(pos, target, val, {}))
            else:
                stmts.append(TLetStmt(pos, safe, ttype, val, ann))
        stmts.extend(_method_side_effects(value_node, env, ctx))
        return stmts
    if val is not None and not already_declared:
        stmts.append(TLetStmt(pos, safe, ttype, val, ann))
    return stmts


def _lower_aug_assign(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower augmented assignment: x += 1."""
    pos = _node_pos(node)
    target_node = get_node(node, "target")
    op_node = get_node(node, "op")
    value_node = get_node(node, "value")
    op_type = get_str(op_node, "_type")
    # list/string/bytes += other → target = Concat(target, other)
    if op_type == "Add":
        target_type = _infer_expr_type(target_node, env, ctx)
        if _is_type_dict(target_type, ["string", "bytes"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            if _is_type_dict(target_type, ["string"]):
                vtype = _infer_expr_type(value_node, env, ctx)
                if _is_type_dict(vtype, ["rune"]):
                    value = _make_call(pos, "ToString", [value])
            return [
                TAssignStmt(pos, target, _make_call(pos, "Concat", [target, value]), {})
            ]
        if _is_type_dict(target_type, ["Slice"]):
            target = _lower_expr(target_node, env, ctx)
            other = _lower_extend_arg(value_node, env, ctx)
            return [
                TAssignStmt(pos, target, _make_call(pos, "Concat", [target, other]), {})
            ]
        if _is_type_dict(target_type, ["Tuple"]):
            target = _lower_expr(target_node, env, ctx)
            other = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(pos, target, _make_call(pos, "Concat", [target, other]), {})
            ]
    # dict |= other → dict = Merge(dict, other)
    # set |= other → set = Union(set, other)
    if op_type == "BitOr":
        target_type = _infer_expr_type(target_node, env, ctx)
        if _is_type_dict(target_type, ["Map"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(pos, target, _make_call(pos, "Merge", [target, value]), {})
            ]
        if _is_type_dict(target_type, ["Set"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(pos, target, _make_call(pos, "Union", [target, value]), {})
            ]
    # set &= other → set = Intersection(set, other)
    if op_type == "BitAnd":
        target_type = _infer_expr_type(target_node, env, ctx)
        if _is_type_dict(target_type, ["Set"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(
                    pos,
                    target,
                    _make_call(pos, "Intersection", [target, value]),
                    {},
                )
            ]
    # set -= other → set = Difference(set, other)
    if op_type == "Sub":
        target_type = _infer_expr_type(target_node, env, ctx)
        if _is_type_dict(target_type, ["Set"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(
                    pos,
                    target,
                    _make_call(pos, "Difference", [target, value]),
                    {},
                )
            ]
    # tuple *= n → target = Repeat(target, n)
    if op_type == "Mult":
        target_type = _infer_expr_type(target_node, env, ctx)
        if _is_type_dict(target_type, ["Tuple"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(pos, target, _make_call(pos, "Repeat", [target, value]), {})
            ]
    # set ^= other → set = Difference(Union(set, other), Intersection(set, other))
    if op_type == "BitXor":
        target_type = _infer_expr_type(target_node, env, ctx)
        if _is_type_dict(target_type, ["Set"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            u = _make_call(pos, "Union", [target, value])
            inter = _make_call(pos, "Intersection", [target, value])
            return [
                TAssignStmt(pos, target, _make_call(pos, "Difference", [u, inter]), {})
            ]
    target = _lower_expr(target_node, env, ctx)
    value = _lower_expr(value_node, env, ctx)
    op_map: dict[str, str] = {
        "Add": "+=",
        "Sub": "-=",
        "Mult": "*=",
        "Div": "/=",
        "FloorDiv": "//=",
        "Mod": "%=",
        "BitAnd": "&=",
        "BitOr": "|=",
        "BitXor": "^=",
        "LShift": "<<=",
        "RShift": ">>=",
    }
    op_str = op_map.get(op_type, "+=")
    return [TOpAssignStmt(pos, target, op_str, value, {})]


def _scan_hoist_names(nodes: list[ASTNode], env: _Env) -> list[str]:
    """Scan for variable names first-assigned inside branch bodies."""
    result: list[str] = []
    seen: set[str] = set()
    stack: list[ASTNode] = []
    i = len(nodes) - 1
    while i >= 0:
        stack.append(nodes[i])
        i -= 1
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        t = get_str(node, "_type")
        if t == "Assign":
            targets = get_nodes(node, "targets")
            if targets:
                tgt = targets[0]
                if _is_ast(tgt, "Name"):
                    name = get_str(tgt, "id")
                    if name not in env.declared and name not in seen and name != "_":
                        result.append(name)
                        seen.add(name)
                elif _is_ast(tgt, "Tuple"):
                    elts = get_nodes(tgt, "elts")
                    for e in elts:
                        if _is_ast(e, "Name"):
                            ename = get_str(e, "id")
                            if (
                                ename not in env.declared
                                and ename not in seen
                                and ename != "_"
                            ):
                                result.append(ename)
                                seen.add(ename)
        elif t == "AnnAssign":
            tgt = get_node(node, "target")
            if _is_ast(tgt, "Name"):
                name = get_str(tgt, "id")
                if name not in env.declared and name not in seen and name != "_":
                    result.append(name)
                    seen.add(name)
        elif t in ("If", "While", "For"):
            body = get_nodes(node, "body")
            orelse = get_nodes(node, "orelse")
            j = len(orelse) - 1
            while j >= 0:
                stack.append(orelse[j])
                j -= 1
            j = len(body) - 1
            while j >= 0:
                stack.append(body[j])
                j -= 1
        elif t == "Try":
            body = get_nodes(node, "body")
            handlers = get_nodes(node, "handlers")
            orelse = get_nodes(node, "orelse")
            finalbody = get_nodes(node, "finalbody")
            j = len(finalbody) - 1
            while j >= 0:
                stack.append(finalbody[j])
                j -= 1
            j = len(orelse) - 1
            while j >= 0:
                stack.append(orelse[j])
                j -= 1
            j = len(handlers) - 1
            while j >= 0:
                h = handlers[j]
                hbody = get_nodes(h, "body")
                k = len(hbody) - 1
                while k >= 0:
                    stack.append(hbody[k])
                    k -= 1
                j -= 1
            j = len(body) - 1
            while j >= 0:
                stack.append(body[j])
                j -= 1
    return result


def _is_guard_body(body: list[ASTNode]) -> bool:
    """Check if a body is a guard clause (single return/continue/break/raise)."""
    if len(body) != 1:
        return False
    if not isinstance(body[0], dict):
        return False
    t = get_str(body[0], "_type")
    return t in ("Return", "Continue", "Break", "Raise")


def _is_negated_isinstance(node: ASTNode) -> bool:
    """Check if node is 'not isinstance(x, T)'."""
    if not _is_ast(node, "UnaryOp"):
        return False
    op = get_node(node, "op")
    if get_str(op, "_type") != "Not":
        return False
    operand = get_node(node, "operand")
    return _is_isinstance_call(operand)


def _subscript_key(node: ASTNode) -> str | None:
    """Build a substitution key for a Subscript node, e.g. 'body:0'."""
    if not _is_ast(node, "Subscript"):
        return None
    val = get_node(node, "value")
    slc = get_node(node, "slice")
    if not _is_ast(val, "Name"):
        return None
    if not _is_ast(slc, "Constant"):
        return None
    idx_jv = slc.get("value")
    if not isinstance(idx_jv, JInt):
        return None
    return get_str(val, "id") + ":" + str(idx_jv.value)


def _find_isinstance_subscripts(
    test: ASTNode,
) -> list[ASTNode]:
    """Find isinstance calls with Subscript first arguments in a condition."""
    results: list[ASTNode] = []
    if _is_isinstance_call(test):
        args = get_nodes(test, "args")
        if len(args) >= 1 and isinstance(args[0], dict):
            if _is_ast(args[0], "Subscript"):
                results.append(args[0])
            elif _is_ast(args[0], "Attribute"):
                inner = get_node(args[0], "value")
                if _is_ast(inner, "Subscript"):
                    results.append(inner)
    elif _is_ast(test, "BoolOp"):
        values = get_nodes(test, "values")
        for value in values:
            results.extend(_find_isinstance_subscripts(value))
    elif _is_ast(test, "UnaryOp"):
        operand = get_node(test, "operand")
        results.extend(_find_isinstance_subscripts(operand))
    return results


def _fresh_isinstance_temp(ctx: _LowerCtx) -> str:
    ctx.isinstance_temp_counter += 1
    return "_istype_tmp_" + str(ctx.isinstance_temp_counter)


def _make_name_node(name: str, src_node: ASTNode) -> ASTNode:
    """Create a Python AST Name node with JStr-wrapped values."""
    return {
        "_type": JStr("Name"),
        "_synthetic": JBool(True),
        "id": JStr(name),
        "lineno": src_node.get("lineno", JInt(0)),
        "col_offset": src_node.get("col_offset", JInt(0)),
        "_source_file": src_node.get("_source_file", JStr("")),
    }


def _replace_subscript_in_ast(node: ASTNode, key: str, name: str) -> None:
    """Replace Subscript nodes matching key with Name nodes in a Python AST tree."""
    if not isinstance(node, dict):
        return
    keys = list(node.keys())
    for k in keys:
        v = node[k]
        if isinstance(v, dict):
            if _subscript_key(v) == key:
                node[k] = _make_name_node(name, v)
            else:
                _replace_subscript_in_ast(v, key, name)
        elif isinstance(v, JDict):
            inner = v.entries
            if _subscript_key(inner) == key:
                node[k] = JDict(_make_name_node(name, inner))
            else:
                _replace_subscript_in_ast(inner, key, name)
        elif isinstance(v, JList):
            ji = 0
            while ji < len(v.items):
                item = v.items[ji]
                if isinstance(item, JDict):
                    if _subscript_key(item.entries) == key:
                        v.items[ji] = JDict(_make_name_node(name, item.entries))
                    else:
                        _replace_subscript_in_ast(item.entries, key, name)
                ji += 1


def _try_split_guards(
    node: ASTNode,
    test: ASTNode,
    body: list[ASTNode],
    orelse: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> list[TStmt] | None:
    """Split compound isinstance guards into sequential/nested ifs. Returns None if no split applies."""
    if _is_ast(test, "BoolOp") and not orelse and _is_guard_body(body):
        op = get_node(test, "op")
        if get_str(op, "_type") == "Or":
            guard_values = get_nodes(test, "values")
            has_isinstance = False
            for gv in guard_values:
                if _is_negated_isinstance(gv):
                    has_isinstance = True
            if has_isinstance and len(guard_values) >= 2:
                result_stmts: list[TStmt] = []
                for gv in guard_values:
                    part_node: dict[str, JsonValue] = node.copy()
                    part_node["test"] = JDict(gv)
                    part_node["orelse"] = JList([])
                    part_stmts = _lower_if(part_node, env, ctx)
                    for part_stmt in part_stmts:
                        result_stmts.append(part_stmt)
                return result_stmts
    if _is_ast(test, "BoolOp") and get_str(get_node(test, "op"), "_type") == "And":
        values = get_nodes(test, "values")
        first_sub_idx = -1
        for vi, val in enumerate(values):
            if _find_isinstance_subscripts(val):
                first_sub_idx = vi
                break
        if first_sub_idx > 0:
            guard_jv: list[JsonValue] = []
            for gval in values[:first_sub_idx]:
                guard_jv.append(JDict(gval))
            if len(guard_jv) == 1:
                guard_test: ASTNode = values[0]
            else:
                guard_test = {
                    "_type": JStr("BoolOp"),
                    "_synthetic": JBool(True),
                    "op": JDict({"_type": JStr("And")}),
                    "values": JList(guard_jv),
                }
            rest_jv: list[JsonValue] = []
            for rval in values[first_sub_idx:]:
                rest_jv.append(JDict(rval))
            if len(rest_jv) == 1:
                inner_test: ASTNode = values[first_sub_idx]
            else:
                inner_test = {
                    "_type": JStr("BoolOp"),
                    "_synthetic": JBool(True),
                    "op": JDict({"_type": JStr("And")}),
                    "values": JList(rest_jv),
                }
            body_jv: list[JsonValue] = []
            for b in body:
                body_jv.append(JDict(b))
            orelse_jv: list[JsonValue] = []
            for o in orelse:
                orelse_jv.append(JDict(o))
            inner_if: dict[str, JsonValue] = {
                "_type": node.get("_type", JStr("If")),
                "_synthetic": JBool(True),
                "test": JDict(inner_test),
                "body": JList(body_jv),
                "orelse": JList(orelse_jv),
                "lineno": node.get("lineno", JInt(0)),
                "col_offset": node.get("col_offset", JInt(0)),
            }
            sf = node.get("_source_file")
            if sf is not None:
                inner_if["_source_file"] = sf
            outer_if: dict[str, JsonValue] = {
                "_type": node.get("_type", JStr("If")),
                "_synthetic": JBool(True),
                "test": JDict(guard_test),
                "body": JList([JDict(inner_if)]),
                "orelse": JList(orelse_jv),
                "lineno": node.get("lineno", JInt(0)),
                "col_offset": node.get("col_offset", JInt(0)),
            }
            if sf is not None:
                outer_if["_source_file"] = sf
            return _lower_if(outer_if, env, ctx)
    return None


def _hoist_isinstance_temps(
    pos: Pos,
    test: ASTNode,
    body: list[ASTNode],
    orelse: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> list[TStmt]:
    """Extract indexed isinstance subscripts into temp let-bindings."""
    indexed_subs = _find_isinstance_subscripts(test)
    pre_temps: list[TStmt] = []
    seen_keys: set[str] = set()
    for sub_node in indexed_subs:
        key = _subscript_key(sub_node)
        if key is not None and key not in seen_keys:
            seen_keys.add(key)
            temp_name = _fresh_isinstance_temp(ctx)
            sub_type = _infer_expr_type(sub_node, env, ctx)
            ttype = _typenode_to_ttype(pos, sub_type)
            lowered_expr = _lower_expr(sub_node, env, ctx)
            pre_temps.append(TLetStmt(pos, temp_name, ttype, lowered_expr, {}))
            env.var_types[temp_name] = sub_type
            env.declared.add(temp_name)
            _replace_subscript_in_ast(test, key, temp_name)
            for b in body:
                _replace_subscript_in_ast(b, key, temp_name)
            for o in orelse:
                _replace_subscript_in_ast(o, key, temp_name)
    return pre_temps


def _lower_if(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower an if statement, detecting isinstance chains for match."""
    pos = _node_pos(node)
    test = get_node(node, "test")
    body = get_nodes(node, "body")
    orelse = get_nodes(node, "orelse")
    isinstance_result = _extract_isinstance_chain(node)
    if isinstance_result is not None:
        return _lower_isinstance_chain(
            pos, isinstance_result.cases, isinstance_result.else_body, env, ctx
        )
    split_result = _try_split_guards(node, test, body, orelse, env, ctx)
    if split_result is not None:
        return split_result
    pre_temps = _hoist_isinstance_temps(pos, test, body, orelse, env, ctx)
    # Hoist variables first-assigned inside branches
    all_branch_nodes: list[ASTNode] = []
    for b in body:
        all_branch_nodes.append(b)
    for o in orelse:
        all_branch_nodes.append(o)
    hoist_names = _scan_hoist_names(all_branch_nodes, env)
    pre_stmts: list[TStmt] = []
    _emit_hoisted_placeholders(pos, hoist_names, env, pre_stmts)
    cond = _lower_as_bool(test, env, ctx)
    # Drain pre_stmts from condition lowering (e.g. any/all expansions)
    if env.pre_stmts:
        cond_pre = env.pre_stmts
        env.pre_stmts = []
        for stmt in cond_pre:
            pre_stmts.append(stmt)
    then_body = _lower_stmts(body, env, ctx)
    else_body: list[TStmt] | None = None
    if orelse:
        else_body = _lower_stmts(orelse, env, ctx)
    pre_stmts.append(TIfStmt(pos, cond, then_body, else_body, {}))
    # Negated isinstance guard: narrow env for subsequent code
    if _is_negated_isinstance(test) and _is_guard_body(body) and not orelse:
        operand = get_node(test, "operand")
        args = get_nodes(operand, "args")
        if len(args) >= 2 and isinstance(args[0], dict) and _is_ast(args[0], "Name"):
            var_name = get_str(args[0], "id")
            type_name = _isinstance_type(operand)
            if type_name:
                env.var_types[var_name] = PointerType(StructRef(type_name))
    return pre_temps + pre_stmts


@dataclass
class _UnwrappedIsinstance:
    isinstance_node: ASTNode
    rest: list[ASTNode]


@dataclass
class _IsinstanceCase:
    var_name: str
    type_name: str
    body: list[ASTNode]
    extra_conds: list[ASTNode] | None


@dataclass
class _IsinstanceChainResult:
    cases: list[_IsinstanceCase]
    else_body: list[ASTNode] | None


def _unwrap_isinstance_and(test: ASTNode) -> _UnwrappedIsinstance | None:
    """If test is isinstance(x,T) AND ..., return the isinstance node and rest. Else None."""
    if not _is_ast(test, "BoolOp"):
        return None
    op = get_node(test, "op")
    if get_str(op, "_type") != "And":
        return None
    values = get_nodes(test, "values")
    if len(values) < 2:
        return None
    if not _is_isinstance_call(values[0]):
        return None
    return _UnwrappedIsinstance(isinstance_node=values[0], rest=values[1:])


def _extract_isinstance_chain(node: ASTNode) -> _IsinstanceChainResult | None:
    """Extract isinstance chain from if/elif. Returns cases + else_body or None."""
    test = get_node(node, "test")
    extra_conds: list[ASTNode] | None = None
    if _is_isinstance_call(test):
        isinstance_node = test
    else:
        unwrapped = _unwrap_isinstance_and(test)
        if unwrapped is None:
            return None
        isinstance_node = unwrapped.isinstance_node
        extra_conds = unwrapped.rest
    var_name = _isinstance_var(isinstance_node)
    if not var_name:
        return None
    type_names = _isinstance_types(isinstance_node)
    if not type_names:
        return None
    body = get_nodes(node, "body")
    result: list[_IsinstanceCase] = []
    for tn in type_names:
        result.append(
            _IsinstanceCase(
                var_name=var_name,
                type_name=tn,
                body=body,
                extra_conds=extra_conds,
            )
        )
    orelse = get_nodes(node, "orelse")
    # Check if elif is also isinstance on same var
    if len(orelse) == 1 and _is_ast(orelse[0], "If"):
        next_node = orelse[0]
        next_test = get_node(next_node, "test")
        next_isinstance: ASTNode | None = None
        if _is_isinstance_call(next_test):
            next_isinstance = next_test
        else:
            unwrapped2 = _unwrap_isinstance_and(next_test)
            if unwrapped2 is not None:
                next_isinstance = unwrapped2.isinstance_node
        if next_isinstance is not None and _isinstance_var(next_isinstance) == var_name:
            rest = _extract_isinstance_chain(next_node)
            if rest is not None:
                for case in rest.cases:
                    result.append(case)
                return _IsinstanceChainResult(cases=result, else_body=rest.else_body)
    # Trailing else (non-isinstance orelse)
    else_body: list[ASTNode] | None = None
    if orelse:
        else_body = orelse
    return _IsinstanceChainResult(cases=result, else_body=else_body)


def _is_isinstance_call(node: ASTNode) -> bool:
    """Check if node is isinstance(x, T)."""
    if not _is_ast(node, "Call"):
        return False
    func = get_node(node, "func")
    if not _is_ast(func, "Name"):
        return False
    return get_str(func, "id") == "isinstance"


def _isinstance_var(node: ASTNode) -> str:
    """Get variable name from isinstance(x, T)."""
    args = get_nodes(node, "args")
    if len(args) >= 1 and isinstance(args[0], dict):
        return get_str(args[0], "id")
    return ""


def _isinstance_type(node: ASTNode) -> str:
    """Get type name from isinstance(x, T)."""
    args = get_nodes(node, "args")
    if len(args) >= 2 and isinstance(args[1], dict):
        return get_str(args[1], "id")
    return ""


def _isinstance_types_from_args(args: list[ASTNode]) -> list[str]:
    """Get type name(s) from isinstance's second arg (raw args list)."""
    if len(args) < 2 or not isinstance(args[1], dict):
        return []
    second = args[1]
    if _is_ast(second, "Tuple"):
        elts = get_nodes(second, "elts")
        result: list[str] = []
        i = 0
        for e in elts:
            name = get_str(e, "id")
            if name:
                result.append(name)
        return result
    name = get_str(second, "id")
    if name:
        return [name]
    return []


def _isinstance_types(node: ASTNode) -> list[str]:
    """Get type name(s) from isinstance(x, T) or isinstance(x, (T1, T2))."""
    args = get_nodes(node, "args")
    return _isinstance_types_from_args(args)


def _lower_isinstance_chain(
    pos: Pos,
    chain: list[_IsinstanceCase],
    else_body_nodes: list[ASTNode] | None,
    env: _Env,
    ctx: _LowerCtx,
) -> list[TStmt]:
    """Lower isinstance chain to a match statement."""
    if not chain:
        return []
    has_extra = False
    # Hoist variables first-assigned inside branches
    all_body_nodes: list[ASTNode] = []
    for c in chain:
        if c.extra_conds is not None:
            has_extra = True
        for b in c.body:
            all_body_nodes.append(b)
    if else_body_nodes is not None:
        for ebn in else_body_nodes:
            all_body_nodes.append(ebn)
    hoist_names = _scan_hoist_names(all_body_nodes, env)
    pre_stmts: list[TStmt] = []
    _emit_hoisted_placeholders(pos, hoist_names, env, pre_stmts)
    var_name = chain[0].var_name
    sv = _safe_name(var_name)
    expr = TVar(pos, sv, _name_ann(sv, var_name))
    cases: list[TMatchCase] = []
    for c in chain:
        type_name = c.type_name
        body_stmts = c.body
        extra_conds = c.extra_conds
        binding_name = type_name[0].lower() + type_name[1:] if type_name else type_name
        if binding_name in env.declared:
            suffix = 2
            while binding_name + str(suffix) in env.declared:
                suffix += 1
            binding_name = binding_name + str(suffix)
        env.declared.add(binding_name)
        case_env = env.copy()
        if extra_conds is not None:
            # Lower extra conditions as && chain
            cond: TExpr = _lower_as_bool(extra_conds[0], case_env, ctx)
            for ec in extra_conds[1:]:
                right = _lower_as_bool(ec, case_env, ctx)
                cond = TBinaryOp(pos, "&&", cond, right, {})
            cond_pre: list[TStmt] = []
            if case_env.pre_stmts:
                cond_pre = case_env.pre_stmts
                case_env.pre_stmts = []
            then_body = _lower_stmts(body_stmts, case_env, ctx)
            nested_else: list[TStmt] | None = None
            if else_body_nodes is not None and else_body_nodes:
                nested_else = _lower_stmts(else_body_nodes, case_env, ctx)
            case_body: list[TStmt] = []
            for stmt in cond_pre:
                case_body.append(stmt)
            case_body.append(TIfStmt(pos, cond, then_body, nested_else, {}))
        else:
            case_body = _lower_stmts(body_stmts, case_env, ctx)
        pattern = TPatternType(pos, binding_name, TIdentType(pos, type_name), {})
        cases.append(TMatchCase(pos, pattern, case_body, {}))
    default: TDefault | None = None
    if else_body_nodes is not None and else_body_nodes:
        else_stmts = _lower_stmts(else_body_nodes, env, ctx)
        default = TDefault(pos, None, else_stmts, {})
    elif has_extra:
        default = TDefault(pos, None, [], {})
    else:
        default = TDefault(pos, None, [], {})
    pre_stmts.append(TMatchStmt(pos, expr, cases, default, {}))
    return pre_stmts


def _match_binding_name(type_name: str, env: _Env) -> str:
    """Generate a unique binding name for a match case pattern."""
    base = type_name[0].lower() + type_name[1:] if type_name else type_name
    if base not in env.declared and base not in TAYTSH_KEYWORDS:
        return base
    suffix = 0
    while base + str(suffix) in env.declared:
        suffix += 1
    return base + str(suffix)


def _lower_match(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower a match/case statement to TMatchStmt."""
    pos = _node_pos(node)
    subject = get_node(node, "subject")
    cases_nodes = get_nodes(node, "cases")
    # Hoist variables first-assigned in any case body
    all_body_nodes: list[ASTNode] = []
    for cn in cases_nodes:
        for b in get_nodes(cn, "body"):
            all_body_nodes.append(b)
    hoist_names = _scan_hoist_names(all_body_nodes, env)
    pre_stmts: list[TStmt] = []
    _emit_hoisted_placeholders(pos, hoist_names, env, pre_stmts)
    expr = _lower_expr(subject, env, ctx)
    subj_name = ""
    if _is_ast(subject, "Name"):
        subj_name = get_str(subject, "id")
    nil_cases: list[TMatchCase] = []
    type_cases: list[TMatchCase] = []
    default: TDefault | None = None
    for cn in cases_nodes:
        pattern = get_node(cn, "pattern")
        case_body_nodes = get_nodes(cn, "body")
        pt = get_str(pattern, "_type")
        if pt == "MatchAs" and isinstance(pattern.get("name"), JNull):
            # Wildcard _ → default
            default_body = _lower_stmts(case_body_nodes, env, ctx)
            default = TDefault(pos, None, default_body, {})
        elif pt == "MatchClass":
            cls = get_node(pattern, "cls")
            type_name = get_str(cls, "id")
            binding_name = _match_binding_name(type_name, env)
            env.declared.add(binding_name)
            case_env = env.copy()
            case_body = _lower_stmts(case_body_nodes, case_env, ctx)
            tp = TPatternType(pos, binding_name, TIdentType(pos, type_name), {})
            type_cases.append(TMatchCase(pos, tp, case_body, {}))
        elif pt == "MatchSingleton" or pt == "MatchValue":
            v = pattern.get("value")
            if isinstance(v, JNull):
                # case None → case nil
                case_body = _lower_stmts(case_body_nodes, env, ctx)
                nil_cases.append(TMatchCase(pos, TPatternNil(pos), case_body, {}))
        elif pt == "MatchOr":
            subs = get_nodes(pattern, "patterns")
            case_env = env.copy()
            case_body = _lower_stmts(case_body_nodes, case_env, ctx)
            # Emit one case arm per MatchClass alternative
            for sub in subs:
                if get_str(sub, "_type") == "MatchClass":
                    sub_cls = get_node(sub, "cls")
                    type_name = get_str(sub_cls, "id")
                    binding_name = _match_binding_name(type_name, env)
                    env.declared.add(binding_name)
                    tp = TPatternType(pos, binding_name, TIdentType(pos, type_name), {})
                    type_cases.append(TMatchCase(pos, tp, case_body, {}))
                elif get_str(sub, "_type") == "MatchSingleton":
                    v = sub.get("value")
                    if isinstance(v, JNull):
                        nil_cases.append(
                            TMatchCase(pos, TPatternNil(pos), case_body, {})
                        )
    if default is None:
        default = TDefault(pos, None, [], {})
    cases: list[TMatchCase] = nil_cases + type_cases
    pre_stmts.append(TMatchStmt(pos, expr, cases, default, {}))
    return pre_stmts


def _lower_while(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    pos = _node_pos(node)
    test = get_node(node, "test")
    body = get_nodes(node, "body")
    hoist_names = _scan_hoist_names(body, env)
    pre_stmts: list[TStmt] = []
    _emit_hoisted_placeholders(pos, hoist_names, env, pre_stmts)
    cond = _lower_as_bool(test, env, ctx)
    if env.pre_stmts:
        cond_pre = env.pre_stmts
        env.pre_stmts = []
        for stmt in cond_pre:
            pre_stmts.append(stmt)
    stmts = _lower_stmts(body, env, ctx)
    pre_stmts.append(TWhileStmt(pos, cond, stmts, {}))
    return pre_stmts


def _lower_for(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower a for statement."""
    pos = _node_pos(node)
    target_node = get_node(node, "target")
    iter_node = get_node(node, "iter")
    body = get_nodes(node, "body")
    hoist_names = _scan_hoist_names(body, env)
    pre_stmts: list[TStmt] = []
    _emit_hoisted_placeholders(pos, hoist_names, env, pre_stmts)
    # reversed(range(...)) → reversed TRange, reversed(xs) → Reversed(xs)
    if _is_ast(iter_node, "Call"):
        func = get_node(iter_node, "func")
        if _is_ast(func, "Name") and get_str(func, "id") == "reversed":
            result = _lower_for_reversed(target_node, iter_node, body, env, ctx)
            for r in result:
                pre_stmts.append(r)
            return pre_stmts
    # range() → TRange
    if _is_ast(iter_node, "Call"):
        func = get_node(iter_node, "func")
        if _is_ast(func, "Name") and get_str(func, "id") == "range":
            result = _lower_for_range(target_node, iter_node, body, env, ctx)
            for r in result:
                pre_stmts.append(r)
            return pre_stmts
    # enumerate() → indexed for
    if _is_ast(iter_node, "Call"):
        func = get_node(iter_node, "func")
        if _is_ast(func, "Name") and get_str(func, "id") == "enumerate":
            result = _lower_for_enumerate(target_node, iter_node, body, env, ctx)
            for r in result:
                pre_stmts.append(r)
            return pre_stmts
    # dict.items() → for k, v in d
    if _is_ast(iter_node, "Call"):
        func = get_node(iter_node, "func")
        if _is_ast(func, "Attribute") and get_str(func, "attr") == "items":
            obj_node = get_node(func, "value")
            iter_expr = _lower_expr(obj_node, env, ctx)
            binding, b_ann = _extract_binding(target_node)
            obj_type = _infer_expr_type(obj_node, env, ctx)
            if isinstance(obj_type, MapType) and len(binding) >= 2:
                env.var_types[binding[0]] = obj_type.key
                env.var_types[binding[1]] = obj_type.value
            body_stmts = _lower_stmts(body, env, ctx)
            for_ann: Ann = {}
            for_ann.update(b_ann)
            for_ann["for.items"] = "true"
            pre_stmts.append(TForStmt(pos, binding, iter_expr, body_stmts, for_ann))
            return pre_stmts
    # Tuple iteration: for x in t → for x in [t.0, t.1, ...]
    iter_type = _infer_expr_type(iter_node, env, ctx)
    if isinstance(iter_type, TupleType) and not iter_type.variadic:
        elems = iter_type.elements
        if elems:
            iter_lowered = _lower_expr(iter_node, env, ctx)
            items: list[TExpr] = []
            for j in range(len(elems)):
                items.append(TTupleAccess(pos, iter_lowered, j, {}))
            binding, b_ann = _extract_binding(target_node)
            body_stmts = _lower_stmts(body, env, ctx)
            list_expr = TListLit(pos, items, {})
            pre_stmts.append(TForStmt(pos, binding, list_expr, body_stmts, b_ann))
            return pre_stmts
    # Regular iteration: for x in xs
    binding, b_ann = _extract_binding(target_node)
    iter_expr = _lower_expr(iter_node, env, ctx)
    if len(binding) >= 2 and isinstance(iter_type, SliceType):
        elem = iter_type.element
        if isinstance(elem, TupleType) and len(elem.elements) == len(binding):
            b_ann["iter_kind"] = "tuple_unpack"
            for bname, btype in zip(binding, elem.elements):
                env.var_types[bname] = btype
    elif len(binding) == 1:
        elem_type: TypeNode = VOID_TYPE
        if _is_type_dict(iter_type, ["string"]):
            elem_type = PrimitiveType("rune")
        elif _is_type_dict(iter_type, ["bytes"]):
            elem_type = PrimitiveType("byte")
        elif isinstance(iter_type, SliceType):
            elem_type = iter_type.element
        elif isinstance(iter_type, SetType):
            elem_type = iter_type.element
        elif isinstance(iter_type, MapType):
            elem_type = iter_type.key
        if elem_type != VOID_TYPE:
            env.var_types[binding[0]] = elem_type
    body_stmts = _lower_stmts(body, env, ctx)
    pre_stmts.append(TForStmt(pos, binding, iter_expr, body_stmts, b_ann))
    return pre_stmts


def _extract_binding(target_node: ASTNode) -> tuple[list[str], Ann]:
    """Extract binding names from a for target, renaming Taytsh keywords."""
    if _is_ast(target_node, "Name"):
        orig = get_str(target_node, "id")
        safe = _safe_name(orig)
        return ([safe], _name_ann(safe, orig))
    if _is_ast(target_node, "Tuple"):
        elts = get_nodes(target_node, "elts")
        names: list[str] = []
        ann: Ann = {}
        for e in elts:
            if _is_ast(e, "Name"):
                orig = get_str(e, "id")
                safe = _safe_name(orig)
                names.append(safe)
                if safe != orig:
                    ann["name.original." + safe] = orig
        return (names, ann)
    return (["_"], {})


def _lower_for_range(
    target_node: ASTNode,
    iter_node: ASTNode,
    body: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> list[TStmt]:
    """Lower for i in range(...)."""
    pos = _node_pos(target_node)
    args = get_nodes(iter_node, "args")
    binding, b_ann = _extract_binding(target_node)
    if len(binding) == 1:
        env.var_types[binding[0]] = INT_TYPE
    range_args: list[TExpr] = []
    for a in args:
        range_args.append(_lower_expr(a, env, ctx))
    body_stmts = _lower_stmts(body, env, ctx)
    return [TForStmt(pos, binding, TRange(pos, range_args, {}), body_stmts, b_ann)]


def _lower_for_reversed(
    target_node: ASTNode,
    iter_node: ASTNode,
    body: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> list[TStmt]:
    """Lower for x in reversed(...)."""
    pos = _node_pos(target_node)
    rev_args = get_nodes(iter_node, "args")
    if not rev_args or not isinstance(rev_args[0], dict):
        return []
    inner = rev_args[0]
    binding, b_ann = _extract_binding(target_node)
    # reversed(range(...)) → range with reversed bounds
    if _is_ast(inner, "Call"):
        inner_func = get_node(inner, "func")
        if _is_ast(inner_func, "Name") and get_str(inner_func, "id") == "range":
            range_args = get_nodes(inner, "args")
            if len(binding) == 1:
                env.var_types[binding[0]] = INT_TYPE
            lowered: list[TExpr] = []
            for a in range_args:
                lowered.append(_lower_expr(a, env, ctx))
            if len(lowered) == 1:
                # reversed(range(n)) → range(n-1, -1, -1)
                n = lowered[0]
                rev_range = TRange(
                    pos,
                    [
                        TBinaryOp(pos, "-", n, TIntLit(pos, 1, "1", {}), {}),
                        TIntLit(pos, -1, "-1", {}),
                        TIntLit(pos, -1, "-1", {}),
                    ],
                    {},
                )
            elif len(lowered) == 2:
                # reversed(range(a, b)) → range(b-1, a-1, -1)
                rev_range = TRange(
                    pos,
                    [
                        TBinaryOp(pos, "-", lowered[1], TIntLit(pos, 1, "1", {}), {}),
                        TBinaryOp(pos, "-", lowered[0], TIntLit(pos, 1, "1", {}), {}),
                        TIntLit(pos, -1, "-1", {}),
                    ],
                    {},
                )
            else:
                # reversed(range(a, b, s)) — not supported, fall through
                rev_range = TRange(pos, lowered, {})
            b_ann["provenance"] = "reversed_range"
            body_stmts = _lower_stmts(body, env, ctx)
            return [TForStmt(pos, binding, rev_range, body_stmts, b_ann)]
    # reversed(xs) → for x in Reversed(xs)
    inner_expr = _lower_expr(inner, env, ctx)
    inner_type = _infer_expr_type(inner, env, ctx)
    if len(binding) == 1 and isinstance(inner_type, SliceType):
        env.var_types[binding[0]] = inner_type.element
    iter_expr = _make_call(pos, "Reversed", [inner_expr])
    body_stmts = _lower_stmts(body, env, ctx)
    return [TForStmt(pos, binding, iter_expr, body_stmts, b_ann)]


def _lower_for_enumerate(
    target_node: ASTNode,
    iter_node: ASTNode,
    body: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> list[TStmt]:
    """Lower for i, x in enumerate(xs)."""
    pos = _node_pos(target_node)
    args = get_nodes(iter_node, "args")
    if not args:
        return []
    inner = args[0]
    if not isinstance(inner, dict):
        return []
    binding, b_ann = _extract_binding(target_node)
    b_ann["for.enumerate"] = "true"
    inner_type = _infer_expr_type(inner, env, ctx)
    # Enumerate over fixed-size tuple → enumerate over list of accesses
    if isinstance(inner_type, TupleType) and not inner_type.variadic:
        elems = inner_type.elements
        if elems:
            iter_lowered = _lower_expr(inner, env, ctx)
            items_list: list[TExpr] = []
            for j in range(len(elems)):
                items_list.append(TTupleAccess(pos, iter_lowered, j, {}))
            body_stmts = _lower_stmts(body, env, ctx)
            return [
                TForStmt(
                    pos,
                    binding,
                    TListLit(pos, items_list, {}),
                    body_stmts,
                    b_ann,
                )
            ]
    iter_expr = _lower_expr(inner, env, ctx)
    # For enumerate over strings, change last binding to "ch"
    if _is_type_dict(inner_type, ["string"]) and len(binding) == 2:
        binding = [binding[0], "ch"]
    # Register loop variable types
    if len(binding) >= 1:
        env.var_types[binding[0]] = INT_TYPE
    if len(binding) >= 2:
        elem_type: TypeNode = VOID_TYPE
        if _is_type_dict(inner_type, ["string"]):
            elem_type = PrimitiveType("rune")
        elif _is_type_dict(inner_type, ["bytes"]):
            elem_type = PrimitiveType("byte")
        elif isinstance(inner_type, SliceType):
            elem_type = inner_type.element
        elif isinstance(inner_type, MapType):
            elem_type = inner_type.key
        if elem_type != VOID_TYPE:
            env.var_types[binding[1]] = elem_type
    body_stmts = _lower_stmts(body, env, ctx)
    return [TForStmt(pos, binding, iter_expr, body_stmts, b_ann)]


def _lower_extend_arg(arg_node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower an argument to list.extend() or list +=, handling range/string."""
    pos = _node_pos(arg_node)
    # extend(range(...)) → RangeList(...)
    if _is_ast(arg_node, "Call"):
        rfunc = get_node(arg_node, "func")
        if _is_ast(rfunc, "Name") and get_str(rfunc, "id") == "range":
            rargs = get_nodes(arg_node, "args")
            if len(rargs) == 1 and isinstance(rargs[0], dict):
                end = _lower_expr(rargs[0], env, ctx)
                return _make_call(
                    pos,
                    "RangeList",
                    [
                        TIntLit(pos, 0, "0", {}),
                        end,
                        TIntLit(pos, 1, "1", {}),
                    ],
                )
            if (
                len(rargs) == 2
                and isinstance(rargs[0], dict)
                and isinstance(rargs[1], dict)
            ):
                start = _lower_expr(rargs[0], env, ctx)
                end = _lower_expr(rargs[1], env, ctx)
                return _make_call(
                    pos,
                    "RangeList",
                    [start, end, TIntLit(pos, 1, "1", {})],
                )
            if (
                len(rargs) >= 3
                and isinstance(rargs[0], dict)
                and isinstance(rargs[1], dict)
                and isinstance(rargs[2], dict)
            ):
                start = _lower_expr(rargs[0], env, ctx)
                end = _lower_expr(rargs[1], env, ctx)
                step = _lower_expr(rargs[2], env, ctx)
                return _make_call(pos, "RangeList", [start, end, step])
    # extend("string") → ["s", "t", "r", ...]
    arg_type = _infer_expr_type(arg_node, env, ctx)
    if _is_type_dict(arg_type, ["string"]) and _is_ast(arg_node, "Constant"):
        s_jv = arg_node.get("value")
        if isinstance(s_jv, JStr):
            ext_elems: list[TExpr] = []
            ci = 0
            while ci < len(s_jv.value):
                ext_elems.append(TStringLit(pos, s_jv.value[ci : ci + 1], {}))
                ci += 1
            return TListLit(pos, ext_elems, {})
    return _lower_expr(arg_node, env, ctx)


def _ensure_set_expr(arg_node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Ensure an expression produces a set, wrapping in SetFromList if needed."""
    pos = _node_pos(arg_node)
    arg_type = _infer_expr_type(arg_node, env, ctx)
    if _is_type_dict(arg_type, ["Set"]):
        return _lower_expr(arg_node, env, ctx)
    # String → SetFromList(Chars(s))
    if _is_type_dict(arg_type, ["string"]):
        if _is_ast(arg_node, "Constant"):
            s_jv = arg_node.get("value")
            if isinstance(s_jv, JStr):
                ens_elems: list[TExpr] = []
                ci = 0
                while ci < len(s_jv.value):
                    ens_elems.append(TStringLit(pos, s_jv.value[ci : ci + 1], {}))
                    ci += 1
                return _make_call(pos, "SetFromList", [TListLit(pos, ens_elems, {})])
        return _make_call(
            pos,
            "SetFromList",
            [_make_call(pos, "Chars", [_lower_expr(arg_node, env, ctx)])],
        )
    # Map → SetFromList(Keys(map))
    if _is_type_dict(arg_type, ["Map"]):
        return _make_call(
            pos,
            "SetFromList",
            [_make_call(pos, "Keys", [_lower_expr(arg_node, env, ctx)])],
        )
    # List or other → SetFromList(list)
    return _make_call(pos, "SetFromList", [_lower_expr(arg_node, env, ctx)])


def _lower_expr_stmt(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower an expression statement."""
    pos = _node_pos(node)
    value = get_node(node, "value")
    if _is_ast(value, "Call"):
        func = get_node(value, "func")
        if _is_ast(func, "Name") and get_str(func, "id") == "assert_never":
            return [
                TThrowStmt(
                    pos,
                    TCall(
                        pos,
                        TVar(pos, "RuntimeError", {}),
                        [TArg(pos, None, TStringLit(pos, "unreachable", {}))],
                        {},
                    ),
                    {},
                )
            ]
    # Check for method calls that produce assignment side effects
    if _is_ast(value, "Call"):
        func = get_node(value, "func")
        if _is_ast(func, "Attribute"):
            method = get_str(func, "attr")
            obj_node = get_node(func, "value")
            obj_type = _infer_expr_type(obj_node, env, ctx)
            # list.clear() → xs = []
            if _is_type_dict(obj_type, ["Slice"]) and method == "clear":
                obj = _lower_expr(obj_node, env, ctx)
                return [TAssignStmt(pos, obj, TListLit(pos, [], {}), {})]
            # list.reverse() → xs = Reversed(xs)
            if _is_type_dict(obj_type, ["Slice"]) and method == "reverse":
                obj = _lower_expr(obj_node, env, ctx)
                return [TAssignStmt(pos, obj, _make_call(pos, "Reversed", [obj]), {})]
            # list.sort() → xs = Sorted(xs)
            if _is_type_dict(obj_type, ["Slice"]) and method == "sort":
                obj = _lower_expr(obj_node, env, ctx)
                return [TAssignStmt(pos, obj, _make_call(pos, "Sorted", [obj]), {})]
            # list.extend(other) → xs = Concat(xs, other)
            if _is_type_dict(obj_type, ["Slice"]) and method == "extend":
                vargs = get_nodes(value, "args")
                if vargs and isinstance(vargs[0], dict):
                    obj = _lower_expr(obj_node, env, ctx)
                    other_expr = _lower_extend_arg(vargs[0], env, ctx)
                    return [
                        TAssignStmt(
                            pos,
                            obj,
                            _make_call(pos, "Concat", [obj, other_expr]),
                            {},
                        )
                    ]
            # dict.pop(k) / dict.pop(k, default) → Delete(d, k)
            if _is_type_dict(obj_type, ["Map"]) and method == "pop":
                vargs = get_nodes(value, "args")
                if vargs and isinstance(vargs[0], dict):
                    obj = _lower_expr(obj_node, env, ctx)
                    key = _lower_expr(vargs[0], env, ctx)
                    return [TExprStmt(pos, _make_call(pos, "Delete", [obj, key]), {})]
            # list.pop() → Pop(xs); list.pop(i) → RemoveAt(xs, i)
            if _is_type_dict(obj_type, ["Slice"]) and method == "pop":
                vargs = get_nodes(value, "args")
                obj = _lower_expr(obj_node, env, ctx)
                if vargs and isinstance(vargs[0], dict):
                    idx = _lower_expr(vargs[0], env, ctx)
                    return [
                        TExprStmt(
                            pos,
                            _make_call(pos, "RemoveAt", [obj, idx]),
                            {},
                        )
                    ]
                return [TExprStmt(pos, _make_call(pos, "Pop", [obj]), {})]
            # dict.clear() → d = Map()
            if _is_type_dict(obj_type, ["Map"]) and method == "clear":
                obj = _lower_expr(obj_node, env, ctx)
                return [TAssignStmt(pos, obj, _make_call(pos, "Map", []), {})]
            # dict.update(other) → d = Merge(d, other)
            if _is_type_dict(obj_type, ["Map"]) and method == "update":
                vargs = get_nodes(value, "args")
                if vargs and isinstance(vargs[0], dict):
                    obj = _lower_expr(obj_node, env, ctx)
                    other = _lower_expr(vargs[0], env, ctx)
                    return [
                        TAssignStmt(
                            pos, obj, _make_call(pos, "Merge", [obj, other]), {}
                        )
                    ]
            # set.update(other, ...) → s = Union(s, SetFromList(other))
            if _is_type_dict(obj_type, ["Set"]) and method == "update":
                vargs = get_nodes(value, "args")
                if vargs:
                    obj = _lower_expr(obj_node, env, ctx)
                    result: TExpr = obj
                    for va in vargs:
                        other = _ensure_set_expr(va, env, ctx)
                        result = _make_call(pos, "Union", [result, other])
                    return [TAssignStmt(pos, obj, result, {})]
            # set.clear() → s = Set()
            if _is_type_dict(obj_type, ["Set"]) and method == "clear":
                obj = _lower_expr(obj_node, env, ctx)
                return [TAssignStmt(pos, obj, _make_call(pos, "Set", []), {})]
    expr = _lower_expr(value, env, ctx)
    return [TExprStmt(pos, expr, {})]


def _collect_exc_type(pos: Pos, node: ASTNode, out: list[TType]) -> None:
    """Append a single exception type from a Name node, skipping generic bases."""
    tname = get_str(node, "id")
    if tname == "AssertionError":
        tname = "AssertError"
    if tname != "Exception" and tname != "BaseException":
        out.append(TIdentType(pos, tname))


def _lower_try(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower a try/except statement."""
    pos = _node_pos(node)
    body = get_nodes(node, "body")
    handlers = get_nodes(node, "handlers")
    finalbody = get_nodes(node, "finalbody")
    body_stmts = _lower_stmts(body, env, ctx)
    catches: list[TCatch] = []
    for h in handlers:
        catch_name = get_str(h, "name")
        if not catch_name:
            catch_name = "e"
        exc_type_jv = h.get("type")
        exc_types: list[TType] = []
        if isinstance(exc_type_jv, JDict):
            exc_type_node = exc_type_jv.entries
            if _is_ast(exc_type_node, "Name"):
                _collect_exc_type(pos, exc_type_node, exc_types)
            elif _is_ast(exc_type_node, "Tuple"):
                for elt in get_nodes(exc_type_node, "elts"):
                    _collect_exc_type(pos, elt, exc_types)
        catch_body = _lower_stmts(get_nodes(h, "body"), env, ctx)
        sc = _safe_name(catch_name)
        catches.append(
            TCatch(pos, sc, exc_types, catch_body, _name_ann(sc, catch_name))
        )
    finally_body: list[TStmt] | None = None
    if finalbody:
        finally_body = _lower_stmts(finalbody, env, ctx)
    return [TTryStmt(pos, body_stmts, catches, finally_body, {})]


def _lower_raise(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower a raise statement."""
    pos = _node_pos(node)
    exc_jv = node.get("exc")
    if isinstance(exc_jv, JDict):
        expr = _lower_expr(exc_jv.entries, env, ctx)
        return [TThrowStmt(pos, expr, {})]
    return [TThrowStmt(pos, TVar(pos, "e", {}), {})]


def _lower_assert(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower an assert statement."""
    pos = _node_pos(node)
    test = get_node(node, "test")
    msg_jv = node.get("msg")
    cond = _lower_as_bool(test, env, ctx)
    args: list[TExpr] = [cond]
    if isinstance(msg_jv, JDict):
        args.append(_lower_expr(msg_jv.entries, env, ctx))
    call = _make_call(pos, "Assert", args)
    return [TExprStmt(pos, call, {})]


# ---------------------------------------------------------------------------
# Declaration building
# ---------------------------------------------------------------------------


def _build_function(
    node: ASTNode,
    env: _Env,
    ctx: _LowerCtx,
    is_entry_point: bool,
) -> TFnDecl:
    """Build a TFnDecl from a FunctionDef node."""
    pos = _node_pos(node)
    name = get_str(node, "name")
    if is_entry_point:
        name = "Main"
    # Get params and return type from signatures
    func_info = ctx.tc_result.functions.get(get_str(node, "name"))
    params: list[TParam] = []
    func_env = env.copy()
    func_env.hoisted_stmts = {}
    if func_info is not None:
        for p in func_info.params:
            if _is_non_zero_default(p.default_value):
                sf = get_str(node, "_source_file")
                ctx.errors.append(
                    LoweringError(
                        get_int(node, "lineno"),
                        get_int(node, "col_offset"),
                        name
                        + "() param '"
                        + p.name
                        + "' has a non-zero default parameter value",
                        sf,
                    )
                )
            if contains_any(p.typ):
                sf = get_str(node, "_source_file")
                ctx.errors.append(
                    LoweringError(
                        get_int(node, "lineno"),
                        get_int(node, "col_offset"),
                        name + "() param '" + p.name + "' has unresolved 'any' type",
                        sf,
                    )
                )
            ttype = _typenode_to_ttype(pos, p.typ)
            sp = _safe_name(p.name)
            params.append(TParam(pos, sp, ttype, _name_ann(sp, p.name), p.has_default))
            func_env.var_types[p.name] = p.typ
            func_env.declared.add(p.name)
        func_env.return_type = func_info.return_type
    if is_entry_point:
        func_env.return_type = VOID_TYPE
    ret_type: TType = TPrimitive(pos, "void")
    if is_entry_point:
        pass
    elif func_info is not None:
        if contains_any(func_info.return_type):
            sf = get_str(node, "_source_file")
            ctx.errors.append(
                LoweringError(
                    get_int(node, "lineno"),
                    get_int(node, "col_offset"),
                    name + "() return has unresolved 'any' type",
                    sf,
                )
            )
        ret_type = _typenode_to_ttype(pos, func_info.return_type)
    body_nodes = get_nodes(node, "body")
    body = _lower_stmts(body_nodes, func_env, ctx)
    return TFnDecl(pos, name, params, ret_type, body, {})


def _build_method(
    node: ASTNode,
    class_name: str,
    env: _Env,
    ctx: _LowerCtx,
    source_class: str | None = None,
) -> TFnDecl:
    """Build a TFnDecl from a method definition."""
    pos = _node_pos(node)
    name = get_str(node, "name")
    # Get method signature — from source class if inherited
    sig_class = source_class if source_class is not None else class_name
    class_methods = ctx.tc_result.methods.get(sig_class, {})
    func_info = class_methods.get(name)
    params: list[TParam] = []
    func_env = env.copy()
    func_env.hoisted_stmts = {}
    # Add this param
    self_type: TypeNode = PointerType(StructRef(class_name))
    func_env.var_types["this"] = self_type
    func_env.declared.add("this")
    params.append(TParam(pos, "this", None, {}))
    if func_info is not None:
        for p in func_info.params:
            if p.name != "self":
                if _is_non_zero_default(p.default_value):
                    sf = get_str(node, "_source_file")
                    ctx.errors.append(
                        LoweringError(
                            get_int(node, "lineno"),
                            get_int(node, "col_offset"),
                            class_name
                            + "."
                            + name
                            + "() param '"
                            + p.name
                            + "' has a non-zero default parameter value",
                            sf,
                        )
                    )
                if contains_any(p.typ):
                    sf = get_str(node, "_source_file")
                    ctx.errors.append(
                        LoweringError(
                            get_int(node, "lineno"),
                            get_int(node, "col_offset"),
                            class_name
                            + "."
                            + name
                            + "() param '"
                            + p.name
                            + "' has unresolved 'any' type",
                            sf,
                        )
                    )
                ttype = _typenode_to_ttype(pos, p.typ)
                sp = _safe_name(p.name)
                params.append(
                    TParam(pos, sp, ttype, _name_ann(sp, p.name), p.has_default)
                )
                func_env.var_types[p.name] = p.typ
                func_env.declared.add(p.name)
        func_env.return_type = func_info.return_type
    ret_type: TType = TPrimitive(pos, "void")
    if func_info is not None:
        if contains_any(func_info.return_type):
            sf = get_str(node, "_source_file")
            ctx.errors.append(
                LoweringError(
                    get_int(node, "lineno"),
                    get_int(node, "col_offset"),
                    class_name + "." + name + "() return has unresolved 'any' type",
                    sf,
                )
            )
        ret_type = _typenode_to_ttype(pos, func_info.return_type)
    body_nodes = get_nodes(node, "body")
    body = _lower_stmts(body_nodes, func_env, ctx)
    return TFnDecl(pos, name, params, ret_type, body, {})


def _collect_ancestor_fields(
    pos: Pos, name: str, ctx: _LowerCtx
) -> list[tuple[str, TType, bool]]:
    """Walk ancestors root-to-child, collecting fields from non-root ancestors."""
    chain: list[str] = []
    cur = name
    while True:
        ancs = ctx.hier_result.ancestors.get(cur)
        if ancs is None or not ancs:
            break
        chain.append(ancs[0])
        cur = ancs[0]
    # Reverse so we go root→child
    chain.reverse()
    result: list[tuple[str, TType, bool]] = []
    seen: set[str] = set()
    for anc in chain:
        anc_info = ctx.tc_result.classes.get(anc)
        if anc_info is not None:
            akeys: list[str] = []
            if anc_info.init_params:
                for p in anc_info.init_params:
                    field_name = anc_info.param_to_field.get(p, p)
                    if field_name not in seen:
                        akeys.append(field_name)
            for k in anc_info.field_order:
                if k not in seen and k not in akeys:
                    akeys.append(k)
            for fname in akeys:
                finfo = anc_info.fields.get(fname)
                if finfo is not None:
                    result.append(
                        (fname, _typenode_to_ttype(pos, finfo.typ), finfo.has_default)
                    )
                    seen.add(fname)
    return result


def _collect_ancestor_methods(
    name: str, own_names: set[str], ctx: _LowerCtx
) -> list[TFnDecl]:
    """Collect inherited methods from ancestor classes not overridden by own."""
    chain: list[str] = []
    cur = name
    while True:
        ancs = ctx.hier_result.ancestors.get(cur)
        if ancs is None or not ancs:
            break
        chain.append(ancs[0])
        cur = ancs[0]
    chain.reverse()
    result: list[TFnDecl] = []
    seen: set[str] = set(own_names)
    for anc in chain:
        anc_node = ctx.class_nodes.get(anc)
        if anc_node is not None:
            anc_body = get_nodes(anc_node, "body")
            env = _Env()
            for item in anc_body:
                if _is_ast(item, "FunctionDef"):
                    mname = get_str(item, "name")
                    if mname != "__init__" and mname not in seen:
                        result.append(
                            _build_method(item, name, env, ctx, source_class=anc)
                        )
                        seen.add(mname)
    return result


def _collect_field_keys(cls_info: ClassInfo, inherited: set[str]) -> list[str]:
    """Collect field keys: init_params first (mapped via param_to_field), then remaining field_order."""
    seen: set[str] = set(inherited)
    fkeys: list[str] = []
    if cls_info.init_params:
        for p in cls_info.init_params:
            field_name = cls_info.param_to_field.get(p, p)
            if field_name not in seen:
                fkeys.append(field_name)
                seen.add(field_name)
    for k in cls_info.field_order:
        if k not in seen:
            fkeys.append(k)
            seen.add(k)
    return fkeys


def _build_struct(
    node: ASTNode,
    ctx: _LowerCtx,
) -> TDecl | None:
    """Build a TStructDecl or TInterfaceDecl from a ClassDef node."""
    pos = _node_pos(node)
    name = get_str(node, "name")
    # Check if this is a hierarchy root → interface
    if ctx.hier_result.is_hierarchy_root(name):
        ann: Ann = {}
        bases_list = ctx.hier_result.ancestors.get(name)
        if bases_list is not None and bases_list:
            parent_root = ctx.hier_result.root_of(bases_list[0])
            if parent_root is not None:
                ann = {"_parent_interface": parent_root}
        iface_fields: list[TFieldDecl] = []
        cls_info = ctx.tc_result.classes.get(name)
        if cls_info is not None:
            fkeys = _collect_field_keys(cls_info, set())
            for fname in fkeys:
                finfo = cls_info.fields.get(fname)
                if finfo is not None:
                    if contains_any(finfo.typ):
                        sf = get_str(node, "_source_file")
                        ctx.errors.append(
                            LoweringError(
                                get_int(node, "lineno"),
                                get_int(node, "col_offset"),
                                name + "." + fname + " has unresolved 'any' type",
                                sf,
                            )
                        )
                    ftype = _typenode_to_ttype(pos, finfo.typ)
                    iface_fields.append(
                        TFieldDecl(pos, fname, ftype, finfo.has_default)
                    )
        return TInterfaceDecl(pos, name, ann, iface_fields)
    # Get bases
    bases = get_nodes(node, "bases")
    parent: str | None = None
    is_exception = False
    for b in bases:
        if _is_ast(b, "Name"):
            base_name = get_str(b, "id")
            if base_name == "Exception":
                is_exception = True
            elif base_name in ctx.known_classes:
                parent = base_name
    # Also check hierarchy result
    if ctx.hier_result.is_exception(name):
        is_exception = True
    if ctx.hier_result.is_node(name):
        parent = ctx.hier_result.root_of(name)
    # Build fields
    fields: list[TFieldDecl] = []
    cls_info = ctx.tc_result.classes.get(name)
    if cls_info is not None:
        if is_exception and not cls_info.fields:
            fields.append(TFieldDecl(pos, "message", TPrimitive(pos, "string")))
        else:
            # Collect inherited fields from ancestors
            ancestor_fields = _collect_ancestor_fields(pos, name, ctx)
            inherited_field_names: set[str] = set()
            for af_name, af_type, af_has_default in ancestor_fields:
                fields.append(TFieldDecl(pos, af_name, af_type, af_has_default))
                inherited_field_names.add(af_name)
            fkeys = _collect_field_keys(cls_info, inherited_field_names)
            for fname in fkeys:
                finfo = cls_info.fields.get(fname)
                if finfo is not None:
                    if contains_any(finfo.typ):
                        sf = get_str(node, "_source_file")
                        ctx.errors.append(
                            LoweringError(
                                get_int(node, "lineno"),
                                get_int(node, "col_offset"),
                                name + "." + fname + " has unresolved 'any' type",
                                sf,
                            )
                        )
                    ftype = _typenode_to_ttype(pos, finfo.typ)
                    fields.append(TFieldDecl(pos, fname, ftype, finfo.has_default))
    # Build methods — own + inherited from ancestors
    methods: list[TFnDecl] = []
    own_method_names: set[str] = set()
    body = get_nodes(node, "body")
    env = _Env()
    for item in body:
        if _is_ast(item, "FunctionDef"):
            mname = get_str(item, "name")
            if mname != "__init__":
                own_method_names.add(mname)
                methods.append(_build_method(item, name, env, ctx))
    ancestor_methods = _collect_ancestor_methods(name, own_method_names, ctx)
    for am in ancestor_methods:
        methods.append(am)
    ann: Ann = {}
    if is_exception:
        ann["_is_exception"] = "true"
    return TStructDecl(pos, name, parent, fields, methods, ann)


def _build_class_constants(class_node: ASTNode, ctx: _LowerCtx) -> list[TModuleItem]:
    """Extract class-level ALL_CAPS constants from a class body."""
    result: list[TModuleItem] = []
    class_name = get_str(class_node, "name")
    class_body = get_nodes(class_node, "body")
    for item in class_body:
        if _is_ast(item, "Assign"):
            targets = get_nodes(item, "targets")
            if targets:
                t = targets[0]
                if _is_ast(t, "Name"):
                    fname = get_str(t, "id")
                    if fname == fname.upper() and len(fname) > 1:
                        pos = _node_pos(item)
                        value_node = get_node(item, "value")
                        val_type: TypeNode = _infer_expr_type(value_node, _Env(), ctx)
                        if _is_type_dict(val_type, ["void"]):
                            val_type = PrimitiveType("error")
                        ttype = _typenode_to_ttype(pos, val_type)
                        value = _lower_expr(value_node, _Env(), ctx)
                        const_name = class_name + "_" + fname
                        result.append(TLetStmt(pos, const_name, ttype, value, {}))
        elif _is_ast(item, "AnnAssign"):
            target = get_node(item, "target")
            if _is_ast(target, "Name"):
                fname = get_str(target, "id")
                if fname == fname.upper() and len(fname) > 1:
                    pos = _node_pos(item)
                    ann_jv = item.get("annotation")
                    ann_str = ""
                    if isinstance(ann_jv, JDict):
                        ann_str = annotation_to_str(ann_jv.entries)
                    c_type_dict: TypeNode = VOID_TYPE
                    if ann_str:
                        c_type_dict = py_type_to_type_dict(
                            ann_str, ctx.known_classes, [], 0, 0
                        )
                    if _is_type_dict(c_type_dict, ["void"]):
                        value_node = get_node(item, "value")
                        c_type_dict = _infer_expr_type(value_node, _Env(), ctx)
                    if _is_type_dict(c_type_dict, ["void"]):
                        c_type_dict = PrimitiveType("error")
                    ttype = _typenode_to_ttype(pos, c_type_dict)
                    value_node = get_node(item, "value")
                    value = _lower_expr(value_node, _Env(), ctx)
                    const_name = class_name + "_" + fname
                    result.append(TLetStmt(pos, const_name, ttype, value, {}))
    return result


def _collect_constant_names(body: list[ASTNode], ctx: _LowerCtx) -> set[str]:
    """Collect names of module-level ALL_CAPS constants."""
    names: set[str] = set()
    for node in body:
        if _is_ast(node, "Assign"):
            targets = get_nodes(node, "targets")
            if targets:
                t = targets[0]
                if _is_ast(t, "Name"):
                    name = get_str(t, "id")
                    if name == name.upper() and name != "_" and len(name) > 1:
                        names.add(name)
        elif _is_ast(node, "AnnAssign"):
            target = get_node(node, "target")
            if _is_ast(target, "Name"):
                name = get_str(target, "id")
                if name == name.upper() and name != "_" and len(name) > 1:
                    names.add(name)
    return names


def _build_module_constant(
    node: ASTNode, constant_names: set[str], ctx: _LowerCtx
) -> TModuleItem | None:
    """Build a TLetStmt for a module-level constant, or None if not a constant."""
    if _is_ast(node, "Assign"):
        targets = get_nodes(node, "targets")
        if targets:
            t = targets[0]
            if _is_ast(t, "Name"):
                name = get_str(t, "id")
                if name in constant_names:
                    pos = _node_pos(node)
                    value_node = get_node(node, "value")
                    val_type: TypeNode = _infer_expr_type(value_node, _Env(), ctx)
                    if _is_type_dict(val_type, ["void"]):
                        val_type = PrimitiveType("error")
                    ttype = _typenode_to_ttype(pos, val_type)
                    value = _lower_expr(value_node, _Env(), ctx)
                    return TLetStmt(pos, name, ttype, value, {})
    elif _is_ast(node, "AnnAssign"):
        target = get_node(node, "target")
        if _is_ast(target, "Name"):
            name = get_str(target, "id")
            if name in constant_names:
                pos = _node_pos(node)
                ann_jv = node.get("annotation")
                ann_str = ""
                if isinstance(ann_jv, JDict):
                    ann_str = annotation_to_str(ann_jv.entries)
                type_dict: TypeNode = VOID_TYPE
                if ann_str:
                    type_dict = py_type_to_type_dict(
                        ann_str, ctx.known_classes, [], 0, 0
                    )
                if _is_type_dict(type_dict, ["void"]):
                    value_node = get_node(node, "value")
                    type_dict = _infer_expr_type(value_node, _Env(), ctx)
                if _is_type_dict(type_dict, ["void"]):
                    type_dict = PrimitiveType("error")
                ttype = _typenode_to_ttype(pos, type_dict)
                value_node = get_node(node, "value")
                value = _lower_expr(value_node, _Env(), ctx)
                return TLetStmt(pos, name, ttype, value, {})
    return None


def _detect_entry_point(body: list[ASTNode]) -> str | None:
    """Detect if __name__ == '__main__': main() pattern."""
    for node in body:
        if _is_ast(node, "If"):
            test = get_node(node, "test")
            if _is_name_main_check(test):
                if_body = get_nodes(node, "body")
                if if_body:
                    first = if_body[0]
                    if _is_ast(first, "Expr"):
                        val = get_node(first, "value")
                        if _is_ast(val, "Call"):
                            func = get_node(val, "func")
                            if _is_ast(func, "Name"):
                                return get_str(func, "id")
                return "main"
    return None


def _is_name_main_check(node: ASTNode) -> bool:
    """Check if node is __name__ == '__main__'."""
    if not _is_ast(node, "Compare"):
        return False
    left = get_node(node, "left")
    if not _is_ast(left, "Name") or get_str(left, "id") != "__name__":
        return False
    comparators = get_nodes(node, "comparators")
    if not comparators:
        return False
    comp = comparators[0]
    if _is_ast(comp, "Constant"):
        if get_str(comp, "value") == "__main__":
            return True
    return False


# ---------------------------------------------------------------------------
# Module assembly
# ---------------------------------------------------------------------------


def _build_module(tree: ASTNode, ctx: _LowerCtx) -> TModule:
    """Build a TModule from the top-level AST."""
    body = get_nodes(tree, "body")
    decls: list[TModuleItem] = []
    entry_point_func = _detect_entry_point(body)
    # Index class and function AST nodes
    for node in body:
        if _is_ast(node, "ClassDef"):
            ctx.class_nodes[get_str(node, "name")] = node
        elif _is_ast(node, "FunctionDef"):
            ctx.func_nodes[get_str(node, "name")] = node
    # Build structs/interfaces first (needed for type resolution)
    # Also extract class-level constants
    for node in body:
        if _is_ast(node, "ClassDef"):
            decl = _build_struct(node, ctx)
            if decl is not None:
                decls.append(decl)
            # Extract class-level constants
            class_constants = _build_class_constants(node, ctx)
            for cc in class_constants:
                decls.append(cc)
    # Build constants and functions in source order to preserve dependencies
    # (e.g., a constant that calls a function needs the function defined first)
    constant_names = _collect_constant_names(body, ctx)
    env = _Env()
    for node in body:
        if _is_ast(node, "FunctionDef"):
            fname = get_str(node, "name")
            is_entry = entry_point_func is not None and fname == entry_point_func
            decls.append(_build_function(node, env, ctx, is_entry))
        elif _is_ast(node, "Assign") or _is_ast(node, "AnnAssign"):
            const_decl = _build_module_constant(node, constant_names, ctx)
            if const_decl is not None:
                decls.append(const_decl)
    return TModule(decls)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lower(
    tree: ASTNode,
    tc_result: TypeCollectResult,
    hier_result: HierarchyResult,
    known_classes: dict[str, str],
    class_bases: dict[str, list[str]],
    pycheck_result: PycheckResult,
) -> tuple[TModule | None, list[LoweringError]]:
    """Lower the Python AST to Taytsh IR.

    Returns (module, errors). If errors is non-empty, module may be None.
    """
    while _LOWER_ANCESTORS:
        _LOWER_ANCESTORS.pop(list(_LOWER_ANCESTORS.keys())[0])
    akeys = list(hier_result.ancestors.keys())
    for ak in akeys:
        _LOWER_ANCESTORS[ak] = hier_result.ancestors[ak]
    ctx = _LowerCtx(tc_result, hier_result, known_classes, class_bases, pycheck_result)
    module = _build_module(tree, ctx)
    while _LOWER_ANCESTORS:
        _LOWER_ANCESTORS.pop(list(_LOWER_ANCESTORS.keys())[0])
    return (module, ctx.errors)
