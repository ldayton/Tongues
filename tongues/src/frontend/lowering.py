"""Phase 10: Lowering — Python AST to Taytsh IR.

Transforms the typed Python dict-AST into Taytsh IR nodes (TModule from
taytsh/ast.py), using type information from phases 5-9 (signatures, fields,
hierarchy, inference).

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from __future__ import annotations


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
    TPatternType,
    TPrimitive,
    TRange,
    TReturnStmt,
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
from .signatures import (
    ParamInfo,
    SignatureResult,
    annotation_to_str,
    py_type_to_type_dict,
)
from .fields import FieldResult
from .hierarchy import HierarchyResult
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
    INT_TYPE,
    FLOAT_TYPE,
    BOOL_TYPE,
    STR_TYPE,
    VOID_TYPE,
    contains_any,
    JsonValue,
    JStr,
    JInt,
    JBool,
    JFloat,
    JDict,
    JNull,
    ASTNode,
    get_str,
    get_int,
    get_bool,
    get_node,
    get_nodes,
    has_key,
)

_P0 = Pos(0, 0)
_EMPTY_ANN: Ann = {}

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
        return _EMPTY_ANN
    return {"name.original." + safe: original}


# ---------------------------------------------------------------------------
# Type dict to TType conversion
# ---------------------------------------------------------------------------


def _typenode_to_ttype(t: TypeNode) -> TType:
    """Convert a TypeNode (from signatures/inference) to a Taytsh TType node."""
    if isinstance(t, PrimitiveType):
        return TPrimitive(_P0, t.kind)
    if isinstance(t, SliceType):
        if isinstance(t.element, PrimitiveType) and t.element.kind == "byte":
            return TPrimitive(_P0, "bytes")
        return TListType(_P0, _typenode_to_ttype(t.element))
    if isinstance(t, MapType):
        val_ttype = _typenode_to_ttype(t.value)
        if isinstance(val_ttype, TPrimitive) and val_ttype.kind == "void":
            val_ttype = TPrimitive(_P0, "nil")
        return TMapType(_P0, _typenode_to_ttype(t.key), val_ttype)
    if isinstance(t, SetType):
        return TSetType(_P0, _typenode_to_ttype(t.element))
    if isinstance(t, TupleType):
        if t.variadic and len(t.elements) > 0:
            return TListType(_P0, _typenode_to_ttype(t.elements[0]))
        parts: list[TType] = []
        i = 0
        while i < len(t.elements):
            parts.append(_typenode_to_ttype(t.elements[i]))
            i += 1
        if len(parts) >= 2:
            return TTupleType(_P0, parts)
        if len(parts) == 1:
            return TListType(_P0, parts[0])
        return TPrimitive(_P0, "error")
    if isinstance(t, OptionalType):
        return TOptionalType(_P0, _typenode_to_ttype(t.inner))
    if isinstance(t, PointerType):
        if isinstance(t.target, StructRef):
            return TIdentType(_P0, t.target.name)
        return _typenode_to_ttype(t.target)
    if isinstance(t, StructRef):
        if t.name == "dict":
            return TMapType(_P0, TPrimitive(_P0, "string"), TPrimitive(_P0, "error"))
        if t.name == "list":
            return TListType(_P0, TPrimitive(_P0, "error"))
        return TIdentType(_P0, t.name)
    if isinstance(t, InterfaceRef):
        if t.name == "any":
            return TPrimitive(_P0, "void")
        return TIdentType(_P0, t.name)
    if isinstance(t, FuncType):
        fn_parts: list[TType] = []
        i = 0
        while i < len(t.params):
            fn_parts.append(_typenode_to_ttype(t.params[i]))
            i += 1
        fn_parts.append(_typenode_to_ttype(t.ret))
        return TFuncType(_P0, fn_parts)
    if isinstance(t, UnionType):
        parts2: list[TType] = []
        i = 0
        while i < len(t.variants):
            parts2.append(_typenode_to_ttype(t.variants[i]))
            i += 1
        if len(parts2) >= 2:
            return TUnionType(_P0, parts2)
        return TPrimitive(_P0, "error")
    return TPrimitive(_P0, "error")


def _emit_hoisted_lets(
    hoisted: list[tuple[str, TypeNode]], env: _Env, pre_stmts: list[TStmt]
) -> None:
    """Emit TLetStmt declarations for hoisted variables."""
    h = 0
    while h < len(hoisted):
        hname, htype = hoisted[h]
        env.declared.add(hname)
        env.var_types[hname] = htype
        safe = _safe_name(hname)
        ttype = _typenode_to_ttype(htype)
        pre_stmts.append(TLetStmt(_P0, safe, ttype, None, _name_ann(safe, hname)))
        h += 1


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
    if class_name == "":
        return None
    class_methods = ctx.sig_result.methods.get(class_name)
    if class_methods is None:
        return None
    func_info = class_methods.get(method_name)
    if func_info is None:
        return None
    # Skip 'self' parameter
    result: list[ParamInfo] = []
    i = 0
    while i < len(func_info.params):
        if func_info.params[i].name != "self":
            result.append(func_info.params[i])
        i += 1
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
    i = 0
    while i < len(names):
        if k == names[i]:
            return True
        i += 1
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
    elt = get_node(genexpr, "elt")
    generators = get_nodes(genexpr, "generators")
    if len(generators) == 0:
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
            ri = 0
            while ri < len(rargs):
                ra = rargs[ri]
                if isinstance(ra, dict):
                    range_lowered.append(_lower_expr(ra, env, ctx))
                ri += 1
            iter_expr: TExpr = TRange(_P0, range_lowered, {})
        else:
            iter_expr = _lower_expr(iter_node, env, ctx)
    else:
        iter_expr = _lower_expr(iter_node, env, ctx)
    comp_env = env.copy()
    comp_env.declared.add(orig_name)
    elt_expr = _lower_expr(elt, comp_env, ctx)
    result_var = TVar(_P0, var_name, _EMPTY_ANN)
    add_call = _make_call("Add", [result_var, elt_expr])
    body: list[TStmt] = [TExprStmt(_P0, add_call, _EMPTY_ANN)]
    ifs = get_nodes(gen, "ifs")
    if len(ifs) > 0 and isinstance(ifs[0], dict):
        cond = _lower_as_bool(ifs[0], comp_env, ctx)
        body = [TIfStmt(_P0, cond, body, None, _EMPTY_ANN)]
    for_stmt = TForStmt(_P0, [target_name], iter_expr, body, t_ann)
    return [for_stmt]


def _is_map_type(td: TypeNode) -> bool:
    return isinstance(td, MapType)


def _lower_dict_literal_typed(
    node: ASTNode, type_dict: TypeNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower a Dict literal with known target type, converting bool keys to int when needed."""
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
        v = values[i] if i < len(values) else None
        if isinstance(k, dict) and isinstance(v, dict):
            if key_is_int and _is_ast(k, "Constant"):
                kval = k.get("value")
                if isinstance(kval, JBool) and kval.value is True:
                    key_expr = TIntLit(_P0, 1, "1", _EMPTY_ANN)
                elif isinstance(kval, JBool) and kval.value is False:
                    key_expr = TIntLit(_P0, 0, "0", _EMPTY_ANN)
                else:
                    key_expr = _lower_expr(k, env, ctx)
            else:
                key_expr = _lower_expr(k, env, ctx)
            entries.append((key_expr, _lower_expr(v, env, ctx)))
        i += 1
    if len(entries) == 0:
        return _make_call("Map", [])
    return TMapLit(_P0, entries, _EMPTY_ANN)


def _is_bytes_slice(td: TypeNode) -> bool:
    if isinstance(td, SliceType):
        if isinstance(td.element, PrimitiveType) and td.element.kind == "byte":
            return True
    return False


def _default_value_for_type(td: TypeNode) -> TExpr:
    """Return a zero/default value for a given TypeNode."""
    kind = _type_dict_kind(td)
    if kind == "float":
        return TFloatLit(_P0, 0.0, "0.0", _EMPTY_ANN)
    if kind == "string":
        return TStringLit(_P0, "", _EMPTY_ANN)
    if kind == "bool":
        return TBoolLit(_P0, False, _EMPTY_ANN)
    if isinstance(td, TupleType) and len(td.elements) >= 2:
        parts: list[TExpr] = []
        i = 0
        while i < len(td.elements):
            parts.append(_default_value_for_type(td.elements[i]))
            i += 1
        return TTupleLit(_P0, parts, _EMPTY_ANN)
    if kind == "Slice":
        return TListLit(_P0, [], _EMPTY_ANN)
    if kind == "Map":
        return _make_call("Map", [])
    if kind == "Set":
        return _make_call("Set", [])
    if kind == "Optional" or kind == "InterfaceRef" or kind == "StructRef":
        return TNilLit(_P0, _EMPTY_ANN)
    return TIntLit(_P0, 0, "0", _EMPTY_ANN)


def _types_comparable(left: TypeNode, right: TypeNode) -> bool:
    """Check if two types can be compared for equality."""
    lk = _type_dict_kind(left)
    rk = _type_dict_kind(right)
    if lk == rk:
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
        if self.source_file != "":
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
        sig_result: SignatureResult,
        field_result: FieldResult,
        hier_result: HierarchyResult,
        known_classes: set[str],
        class_bases: dict[str, list[str]],
        source: str,
    ) -> None:
        self.sig_result: SignatureResult = sig_result
        self.field_result: FieldResult = field_result
        self.hier_result: HierarchyResult = hier_result
        self.known_classes: set[str] = known_classes
        self.class_bases: dict[str, list[str]] = class_bases
        self.source: str = source
        self.source_lines: list[str] = source.split("\n")
        self.errors: list[LoweringError] = []
        self.current_class: str = ""


class _Env:
    """Scope-level environment for variable tracking."""

    def __init__(self) -> None:
        self.var_types: dict[str, TypeNode] = {}
        self.declared: set[str] = set()
        self.return_type: TypeNode = VOID_TYPE

    def copy(self) -> _Env:
        env = _Env()
        keys = list(self.var_types.keys())
        i = 0
        while i < len(keys):
            env.var_types[keys[i]] = self.var_types[keys[i]]
            i += 1
        dkeys = list(self.declared)
        i = 0
        while i < len(dkeys):
            env.declared.add(dkeys[i])
            i += 1
        env.return_type = self.return_type
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


def _is_ast(node: JsonValue, type_name: str) -> bool:
    """Check if node is a dict-AST of given type."""
    if isinstance(node, JDict):
        return get_str(node.entries, "_type") == type_name
    if isinstance(node, dict):
        return get_str(node, "_type") == type_name
    return False


# ---------------------------------------------------------------------------
# Source text helpers (for literal format detection)
# ---------------------------------------------------------------------------


def _get_source_text(ctx: _LowerCtx, node: ASTNode) -> str:
    """Get source text for a node."""
    lineno = get_int(node, "lineno")
    col = get_int(node, "col_offset")
    end_col = get_int(node, "end_col_offset")
    if lineno == 0 or col == 0 and not has_key(node, "col_offset"):
        return ""
    line_idx = lineno - 1
    if line_idx < 0 or line_idx >= len(ctx.source_lines):
        return ""
    line = ctx.source_lines[line_idx]
    if has_key(node, "end_col_offset"):
        return line[col:end_col]
    return line[col:]


# ---------------------------------------------------------------------------
# Type inference helpers (derive types from signatures and annotations)
# ---------------------------------------------------------------------------


def _get_param_type(param: ParamInfo) -> TypeNode:
    return param.typ


def _func_return_type(ctx: _LowerCtx, name: str) -> TypeNode:
    """Get return type of a function from signatures."""
    info = ctx.sig_result.functions.get(name)
    if info is not None:
        return info.return_type
    return VOID_TYPE


def _method_return_type(ctx: _LowerCtx, class_name: str, method_name: str) -> TypeNode:
    """Get return type of a method from signatures."""
    class_methods = ctx.sig_result.methods.get(class_name)
    if class_methods is not None:
        info = class_methods.get(method_name)
        if info is not None:
            return info.return_type
    return VOID_TYPE


def _infer_expr_type(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TypeNode:
    """Infer the type of an expression node from context."""
    t = get_str(node, "_type")
    if t == "Constant":
        val = node.get("value")
        if isinstance(val, JBool):
            return BOOL_TYPE
        if isinstance(val, JInt):
            return INT_TYPE
        if isinstance(val, JFloat):
            return FLOAT_TYPE
        if isinstance(val, JStr):
            if get_bool(node, "_is_bytes"):
                return PrimitiveType("bytes")
            return STR_TYPE
        if isinstance(val, JNull):
            return VOID_TYPE
        return VOID_TYPE
    if t == "Name":
        name = get_str(node, "id")
        vt = env.var_types.get(name)
        if vt is not None:
            return vt
        fi = ctx.sig_result.functions.get(name)
        if fi is not None:
            param_types: list[TypeNode] = []
            j = 0
            while j < len(fi.params):
                param_types.append(fi.params[j].typ)
                j += 1
            return FuncType(param_types, fi.return_type)
        return VOID_TYPE
    if t == "Attribute":
        attr = get_str(node, "attr")
        obj_node = get_node(node, "value")
        obj_type = _infer_expr_type(obj_node, env, ctx)
        if _is_struct_type(obj_type):
            sname = _struct_name(obj_type)
            cls_info = ctx.field_result.classes.get(sname)
            if cls_info is not None:
                field_info = cls_info.fields.get(attr)
                if field_info is not None:
                    return field_info.typ
        return VOID_TYPE
    if t == "Call":
        func = get_node(node, "func")
        if _is_ast(func, "Name"):
            fname = get_str(func, "id")
            if fname == "len":
                return INT_TYPE
            if fname == "min" or fname == "max" or fname == "abs":
                call_args = get_nodes(node, "args")
                if len(call_args) > 0 and isinstance(call_args[0], dict):
                    at = _infer_expr_type(call_args[0], env, ctx)
                    if _is_type_dict(at, ["bool"]):
                        return INT_TYPE
                    return at
                return INT_TYPE
            if fname == "int":
                return INT_TYPE
            if fname == "float":
                return FLOAT_TYPE
            if fname == "str":
                return STR_TYPE
            if fname == "bool":
                return BOOL_TYPE
            if fname == "chr":
                return STR_TYPE
            if fname == "ord":
                return INT_TYPE
            if fname == "isinstance":
                return BOOL_TYPE
            if fname == "bytes":
                return PrimitiveType("bytes")
            if fname == "sorted":
                args = get_nodes(node, "args")
                if len(args) > 0 and isinstance(args[0], dict):
                    at = _infer_expr_type(args[0], env, ctx)
                    return at
                return SliceType(INT_TYPE)
            if fname == "list":
                args = get_nodes(node, "args")
                if len(args) > 0 and isinstance(args[0], dict):
                    at = _infer_expr_type(args[0], env, ctx)
                    if isinstance(at, SetType):
                        return SliceType(at.element)
                    return at
                return SliceType(INT_TYPE)
            if fname == "divmod":
                return TupleType([INT_TYPE, INT_TYPE], False)
            if fname == "set" or fname == "frozenset":
                args = get_nodes(node, "args")
                if len(args) > 0 and isinstance(args[0], dict):
                    at = _infer_expr_type(args[0], env, ctx)
                    if isinstance(at, SliceType):
                        return SetType(at.element)
                    if isinstance(at, SetType):
                        return at
                    if isinstance(at, TupleType):
                        return SetType(
                            at.elements[0] if len(at.elements) > 0 else INT_TYPE
                        )
                    if _is_type_dict(at, ["string"]):
                        return SetType(PrimitiveType("string"))
                return SetType(INT_TYPE)
            if fname in ctx.known_classes:
                return PointerType(StructRef(fname))
            rt = _func_return_type(ctx, fname)
            return rt
        if _is_ast(func, "Attribute"):
            method_name = get_str(func, "attr")
            obj_n = get_node(func, "value")
            obj_t = _infer_expr_type(obj_n, env, ctx)
            if _is_struct_type(obj_t):
                sname = _struct_name(obj_t)
                return _method_return_type(ctx, sname, method_name)
            if _is_type_dict(obj_t, ["string"]):
                if (
                    method_name == "find"
                    or method_name == "rfind"
                    or method_name == "index"
                    or method_name == "count"
                ):
                    return INT_TYPE
                if method_name == "split":
                    return SliceType(STR_TYPE)
                if method_name == "startswith" or method_name == "endswith":
                    return BOOL_TYPE
                if (
                    method_name == "isdigit"
                    or method_name == "isalpha"
                    or method_name == "isalnum"
                    or method_name == "isspace"
                    or method_name == "isupper"
                    or method_name == "islower"
                ):
                    return BOOL_TYPE
                if method_name == "encode":
                    return PrimitiveType("bytes")
                return STR_TYPE
            if _is_type_dict(obj_t, ["Slice"]):
                if method_name == "pop":
                    if isinstance(obj_t, SliceType):
                        return obj_t.element
                    return INT_TYPE
                if method_name == "index":
                    return INT_TYPE
                if method_name == "copy":
                    return obj_t
                return VOID_TYPE
            if _is_type_dict(obj_t, ["Map"]):
                if method_name == "get":
                    if isinstance(obj_t, MapType):
                        return OptionalType(obj_t.value)
                    return VOID_TYPE
                if method_name == "keys":
                    if isinstance(obj_t, MapType):
                        return SliceType(obj_t.key)
                    return SliceType(STR_TYPE)
                if method_name == "values":
                    if isinstance(obj_t, MapType):
                        return SliceType(obj_t.value)
                    return SliceType(INT_TYPE)
                if method_name == "items":
                    if isinstance(obj_t, MapType):
                        return SliceType(TupleType([obj_t.key, obj_t.value], False))
                    return SliceType(VOID_TYPE)
                if method_name == "pop":
                    if isinstance(obj_t, MapType):
                        return OptionalType(obj_t.value)
                    return VOID_TYPE
                if method_name == "copy":
                    return obj_t
                return VOID_TYPE
            if _is_type_dict(obj_t, ["Slice"]):
                if (
                    isinstance(obj_t, SliceType)
                    and isinstance(obj_t.element, PrimitiveType)
                    and obj_t.element.kind == "byte"
                ):
                    if method_name == "decode":
                        return STR_TYPE
        return VOID_TYPE
    if t == "BinOp":
        op = get_node(node, "op")
        op_type = get_str(op, "_type")
        left = get_node(node, "left")
        right_node = get_node(node, "right")
        lt = _infer_expr_type(left, env, ctx)
        rt = _infer_expr_type(right_node, env, ctx)
        if op_type == "Add":
            if _is_type_dict(lt, ["string"]) or _is_type_dict(rt, ["string"]):
                return STR_TYPE
            if _is_type_dict(lt, ["float"]):
                return FLOAT_TYPE
            return INT_TYPE
        if (
            op_type == "Sub"
            or op_type == "Mult"
            or op_type == "FloorDiv"
            or op_type == "Mod"
            or op_type == "Pow"
        ):
            if _is_type_dict(lt, ["float"]) or _is_type_dict(rt, ["float"]):
                return FLOAT_TYPE
            if op_type == "Mult" and _is_type_dict(lt, ["string"]):
                return STR_TYPE
            if op_type == "Mult" and _is_type_dict(lt, ["Slice"]):
                return lt
            return INT_TYPE
        if op_type == "Div":
            return FLOAT_TYPE
        if (
            op_type == "BitAnd"
            or op_type == "BitOr"
            or op_type == "BitXor"
            or op_type == "LShift"
            or op_type == "RShift"
        ):
            if _is_type_dict(lt, ["Map"]):
                return lt
            return INT_TYPE
        return INT_TYPE
    if t == "BoolOp":
        vals = get_nodes(node, "values")
        all_bools = True
        for bv in vals:
            if isinstance(bv, dict):
                bvt = _infer_expr_type(bv, env, ctx)
                if not _is_type_dict(bvt, ["bool"]):
                    all_bools = False
        if all_bools:
            return BOOL_TYPE
        op = get_node(node, "op")
        op_type = get_str(op, "_type")
        if op_type == "Or":
            last = vals[len(vals) - 1]
            if isinstance(last, dict):
                return _infer_expr_type(last, env, ctx)
        if len(vals) > 0 and isinstance(vals[0], dict):
            return _infer_expr_type(vals[0], env, ctx)
        return BOOL_TYPE
    if t == "Compare":
        return BOOL_TYPE
    if t == "UnaryOp":
        op = get_node(node, "op")
        op_type = get_str(op, "_type")
        if op_type == "Not":
            return BOOL_TYPE
        operand = get_node(node, "operand")
        result = _infer_expr_type(operand, env, ctx)
        if (op_type == "USub" or op_type == "Invert") and _is_type_dict(
            result, ["bool"]
        ):
            return INT_TYPE
        return result
    if t == "IfExp":
        body = get_node(node, "body")
        return _infer_expr_type(body, env, ctx)
    if t == "Subscript":
        obj = get_node(node, "value")
        obj_t = _infer_expr_type(obj, env, ctx)
        if _is_type_dict(obj_t, ["Slice"]):
            slc = get_node(node, "slice")
            if _is_ast(slc, "Slice"):
                return obj_t
            if isinstance(obj_t, SliceType):
                return obj_t.element
        if _is_type_dict(obj_t, ["Map"]):
            if isinstance(obj_t, MapType):
                return obj_t.value
        if _is_type_dict(obj_t, ["Tuple"]):
            slc = get_node(node, "slice")
            if _is_ast(slc, "Constant") and isinstance(obj_t, TupleType):
                idx_val = get_int(slc, "value")
                if has_key(slc, "value") and 0 <= idx_val < len(obj_t.elements):
                    return obj_t.elements[idx_val]
        if _is_type_dict(obj_t, ["string"]):
            slc = get_node(node, "slice")
            if _is_ast(slc, "Slice"):
                return STR_TYPE
            return STR_TYPE
        if _is_type_dict(obj_t, ["bytes"]):
            slc = get_node(node, "slice")
            if _is_ast(slc, "Slice"):
                return PrimitiveType("bytes")
            return INT_TYPE
        return VOID_TYPE
    if t == "List":
        elts = get_nodes(node, "elts")
        elem_type: TypeNode = VOID_TYPE
        if len(elts) > 0 and isinstance(elts[0], dict):
            elem_type = _infer_expr_type(elts[0], env, ctx)
        return SliceType(elem_type)
    if t == "Dict":
        ks = get_nodes(node, "keys")
        vs = get_nodes(node, "values")
        key_type: TypeNode = VOID_TYPE
        val_type: TypeNode = VOID_TYPE
        if len(ks) > 0 and isinstance(ks[0], dict):
            key_type = _infer_expr_type(ks[0], env, ctx)
        if len(vs) > 0 and isinstance(vs[0], dict):
            val_type = _infer_expr_type(vs[0], env, ctx)
        return MapType(key_type, val_type)
    if t == "Set":
        elts = get_nodes(node, "elts")
        elem_type2: TypeNode = VOID_TYPE
        if len(elts) > 0 and isinstance(elts[0], dict):
            elem_type2 = _infer_expr_type(elts[0], env, ctx)
        return SetType(elem_type2)
    if t == "Tuple":
        elts = get_nodes(node, "elts")
        parts: list[TypeNode] = []
        i = 0
        while i < len(elts):
            e = elts[i]
            if isinstance(e, dict):
                parts.append(_infer_expr_type(e, env, ctx))
            i += 1
        return TupleType(parts, False)
    if t == "JoinedStr":
        return STR_TYPE
    return VOID_TYPE


# ---------------------------------------------------------------------------
# Expression lowering
# ---------------------------------------------------------------------------


def _make_call(name: str, args: list[TExpr]) -> TCall:
    """Helper to create a simple function call."""
    targs: list[TArg] = []
    i = 0
    while i < len(args):
        targs.append(TArg(_P0, None, args[i]))
        i += 1
    return TCall(_P0, TVar(_P0, name, _EMPTY_ANN), targs, _EMPTY_ANN)


def _len_expr(obj: TExpr, obj_type: TypeNode) -> TExpr:
    """Len(obj) for most types, literal int for fixed-size tuples."""
    if isinstance(obj_type, TupleType):
        n = len(obj_type.elements)
        return TIntLit(_P0, n, str(n), _EMPTY_ANN)
    return _make_call("Len", [obj])


def _make_named_call(
    name: str, pos_args: list[TExpr], named: list[tuple[str, TExpr]]
) -> TCall:
    """Helper to create a function call with named arguments."""
    targs: list[TArg] = []
    i = 0
    while i < len(pos_args):
        targs.append(TArg(_P0, None, pos_args[i]))
        i += 1
    i = 0
    while i < len(named):
        targs.append(TArg(_P0, named[i][0], named[i][1]))
        i += 1
    return TCall(_P0, TVar(_P0, name, _EMPTY_ANN), targs, _EMPTY_ANN)


def _make_method_call(obj: TExpr, method: str, args: list[TExpr]) -> TCall:
    """Helper to create a method call."""
    targs: list[TArg] = []
    i = 0
    while i < len(args):
        targs.append(TArg(_P0, None, args[i]))
        i += 1
    return TCall(_P0, TFieldAccess(_P0, obj, method, _EMPTY_ANN), targs, _EMPTY_ANN)


def _lower_expr(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Python expression AST node to a Taytsh expression."""
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
        return _make_call("Set", [])
    if t == "DictComp":
        return _make_call("Map", [])
    low_sf = get_str(node, "_source_file")
    ctx.errors.append(
        LoweringError(0, 0, "unsupported expression type '" + str(t) + "'", low_sf)
    )
    return TVar(_P0, "__error__", _EMPTY_ANN)


def _lower_constant(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Constant node."""
    val = node.get("value")
    if isinstance(val, JBool):
        return TBoolLit(_P0, val.value, _EMPTY_ANN)
    if isinstance(val, JInt):
        return TIntLit(_P0, val.value, str(val.value), _EMPTY_ANN)
    if isinstance(val, JFloat):
        return TFloatLit(_P0, val.value, repr(val.value), _EMPTY_ANN)
    if isinstance(val, JStr):
        if get_bool(node, "_is_bytes"):
            return TBytesLit(_P0, val.value.encode("latin-1"), _EMPTY_ANN)
        return TStringLit(_P0, val.value, _EMPTY_ANN)
    if isinstance(val, JNull) or val is None:
        return TNilLit(_P0, _EMPTY_ANN)
    return TNilLit(_P0, _EMPTY_ANN)


def _lower_name(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Name node."""
    name = get_str(node, "id")
    if name == "True":
        return TBoolLit(_P0, True, _EMPTY_ANN)
    if name == "False":
        return TBoolLit(_P0, False, _EMPTY_ANN)
    if name == "None":
        return TNilLit(_P0, _EMPTY_ANN)
    if name == "_":
        return TNilLit(_P0, _EMPTY_ANN)
    safe = _safe_name(name)
    return TVar(_P0, safe, _name_ann(safe, name))


def _lower_attribute(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower an Attribute access node."""
    attr = get_str(node, "attr")
    obj_node = get_node(node, "value")
    # Class constant access: ClassName.CONST → Var("ClassName_CONST")
    if _is_ast(obj_node, "Name"):
        obj_name = get_str(obj_node, "id")
        if obj_name in ctx.known_classes and attr.isupper():
            return TVar(_P0, obj_name + "_" + attr, _EMPTY_ANN)
        # sys.argv → Args()
        if obj_name == "sys" and attr == "argv":
            return _make_call("Args", [])
        if obj_name == "sys" and attr == "maxsize":
            return TIntLit(_P0, 9223372036854775807, "9223372036854775807", _EMPTY_ANN)
        # sys.stdin.readline() etc are handled in _lower_call
    # sys.stdin / sys.stdout / sys.stderr attribute chains
    if _is_ast(obj_node, "Attribute"):
        inner_obj = get_node(obj_node, "value")
        inner_attr = get_str(obj_node, "attr")
        if _is_ast(inner_obj, "Name") and get_str(inner_obj, "id") == "sys":
            if inner_attr == "stdin" and attr == "buffer":
                # Return a placeholder for sys.stdin.buffer
                return TVar(_P0, "__stdin_buffer__", _EMPTY_ANN)
            if inner_attr == "stdout" and attr == "buffer":
                return TVar(_P0, "__stdout_buffer__", _EMPTY_ANN)
            if inner_attr == "stderr" and attr == "buffer":
                return TVar(_P0, "__stderr_buffer__", _EMPTY_ANN)
    obj = _lower_expr(obj_node, env, ctx)
    return TFieldAccess(_P0, obj, attr, _EMPTY_ANN)


def _bool_to_int(expr: TExpr) -> TExpr:
    """Convert bool expression to int: b ? 1 : 0."""
    return TTernary(
        _P0,
        expr,
        TIntLit(_P0, 1, "1", _EMPTY_ANN),
        TIntLit(_P0, 0, "0", _EMPTY_ANN),
        _EMPTY_ANN,
    )


def _coerce_arithmetic(
    left: TExpr,
    right: TExpr,
    left_type: TypeNode | None,
    right_type: TypeNode | None,
) -> tuple[TExpr, TExpr]:
    """Insert bool→int and int→float coercions for arithmetic operands."""
    lt_bool = _is_type_dict(left_type, ["bool"])
    rt_bool = _is_type_dict(right_type, ["bool"])
    if lt_bool:
        left = _bool_to_int(left)
    if rt_bool:
        right = _bool_to_int(right)
    lt_float = _is_type_dict(left_type, ["float"])
    rt_float = _is_type_dict(right_type, ["float"])
    lt_int = _is_type_dict(left_type, ["int"]) or lt_bool
    rt_int = _is_type_dict(right_type, ["int"]) or rt_bool
    if lt_float and rt_int:
        right = _make_call("IntToFloat", [right])
    elif rt_float and lt_int:
        left = _make_call("IntToFloat", [left])
    return left, right


def _lower_binop(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a BinOp node."""
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
            return _make_call("Concat", [left, right])
        if _is_type_dict(left_type, ["bytes"]) or _is_type_dict(right_type, ["bytes"]):
            return _make_call("Concat", [left, right])
        if _is_type_dict(left_type, ["Slice"]) or _is_type_dict(right_type, ["Slice"]):
            return _make_call("Concat", [left, right])
        if _is_type_dict(left_type, ["Tuple"]) or _is_type_dict(right_type, ["Tuple"]):
            return _lower_tuple_concat(left_node, right_node, env, ctx)
        left, right = _coerce_arithmetic(left, right, left_type, right_type)
        return TBinaryOp(_P0, "+", left, right, _EMPTY_ANN)
    if op_type == "Sub":
        if _is_type_dict(left_type, ["Set"]):
            return _make_call("Difference", [left, right])
        if _is_type_dict(left_type, ["Slice"]) and _is_type_dict(right_type, ["Slice"]):
            return _make_call(
                "Difference",
                [_make_call("SetFromList", [left]), _make_call("SetFromList", [right])],
            )
        left, right = _coerce_arithmetic(left, right, left_type, right_type)
        return TBinaryOp(_P0, "-", left, right, _EMPTY_ANN)
    if op_type == "Mult":
        if _is_type_dict(left_type, ["string", "bytes", "Slice", "Tuple"]):
            return _make_call("Repeat", [left, right])
        if _is_type_dict(right_type, ["string", "bytes", "Slice", "Tuple"]):
            return _make_call("Repeat", [right, left])
        left, right = _coerce_arithmetic(left, right, left_type, right_type)
        return TBinaryOp(_P0, "*", left, right, _EMPTY_ANN)
    if op_type == "FloorDiv":
        left, right = _coerce_arithmetic(left, right, left_type, right_type)
        return _make_call("FloorDiv", [left, right])
    if op_type == "Div":
        # True division: IntToFloat(a) / IntToFloat(b)
        if _is_type_dict(left_type, ["int", "bool"]):
            if _is_type_dict(left_type, ["bool"]):
                left = _bool_to_int(left)
            left = _make_call("IntToFloat", [left])
        if _is_type_dict(right_type, ["int", "bool"]):
            if _is_type_dict(right_type, ["bool"]):
                right = _bool_to_int(right)
            right = _make_call("IntToFloat", [right])
        return TBinaryOp(_P0, "/", left, right, _EMPTY_ANN)
    if op_type == "Mod":
        left, right = _coerce_arithmetic(left, right, left_type, right_type)
        return _make_call("PythonMod", [left, right])
    if op_type == "Pow":
        left, right = _coerce_arithmetic(left, right, left_type, right_type)
        return _make_call("Pow", [left, right])
    if op_type == "BitAnd":
        if _is_type_dict(left_type, ["Set"]):
            return _make_call("Intersection", [left, right])
        if _is_type_dict(left_type, ["Slice"]) and _is_type_dict(right_type, ["Slice"]):
            return _make_call(
                "Intersection",
                [_make_call("SetFromList", [left]), _make_call("SetFromList", [right])],
            )
        if _is_type_dict(left_type, ["bool"]):
            left = _bool_to_int(left)
        if _is_type_dict(right_type, ["bool"]):
            right = _bool_to_int(right)
        return TBinaryOp(_P0, "&", left, right, _EMPTY_ANN)
    if op_type == "BitOr":
        # Dict merge: a | b
        if _is_type_dict(left_type, ["Map"]):
            return _make_call("Merge", [left, right])
        # Set union: a | b
        if _is_type_dict(left_type, ["Set"]):
            return _make_call("Union", [left, right])
        if _is_type_dict(left_type, ["Slice"]) and _is_type_dict(right_type, ["Slice"]):
            return _make_call(
                "Union",
                [_make_call("SetFromList", [left]), _make_call("SetFromList", [right])],
            )
        if _is_type_dict(left_type, ["bool"]):
            left = _bool_to_int(left)
        if _is_type_dict(right_type, ["bool"]):
            right = _bool_to_int(right)
        return TBinaryOp(_P0, "|", left, right, _EMPTY_ANN)
    if op_type == "BitXor":
        if _is_type_dict(left_type, ["Set"]):
            u = _make_call("Union", [left, right])
            i = _make_call("Intersection", [left, right])
            return _make_call("Difference", [u, i])
        if _is_type_dict(left_type, ["Slice"]) and _is_type_dict(right_type, ["Slice"]):
            ls = _make_call("SetFromList", [left])
            rs = _make_call("SetFromList", [right])
            u = _make_call("Union", [ls, rs])
            i = _make_call("Intersection", [ls, rs])
            return _make_call("Difference", [u, i])
        if _is_type_dict(left_type, ["bool"]):
            left = _bool_to_int(left)
        if _is_type_dict(right_type, ["bool"]):
            right = _bool_to_int(right)
        return TBinaryOp(_P0, "^", left, right, _EMPTY_ANN)
    if op_type == "LShift":
        if _is_type_dict(left_type, ["bool"]):
            left = _bool_to_int(left)
        if _is_type_dict(right_type, ["bool"]):
            right = _bool_to_int(right)
        return TBinaryOp(_P0, "<<", left, right, _EMPTY_ANN)
    if op_type == "RShift":
        if _is_type_dict(left_type, ["bool"]):
            left = _bool_to_int(left)
        if _is_type_dict(right_type, ["bool"]):
            right = _bool_to_int(right)
        return TBinaryOp(_P0, ">>", left, right, _EMPTY_ANN)
    return TBinaryOp(_P0, "+", left, right, _EMPTY_ANN)


def _lower_tuple_concat(
    left_node: ASTNode, right_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Inline tuple concatenation: (a, b) + (c, d) → (a, b, c, d)."""
    left_is_literal = _is_ast(left_node, "Tuple")
    right_is_literal = _is_ast(right_node, "Tuple")
    if not left_is_literal or not right_is_literal:
        left = _lower_expr(left_node, env, ctx)
        right = _lower_expr(right_node, env, ctx)
        return _make_call("Concat", [left, right])
    left_elts = [e for e in get_nodes(left_node, "elts") if isinstance(e, dict)]
    right_elts = [e for e in get_nodes(right_node, "elts") if isinstance(e, dict)]
    all_elts = left_elts + right_elts
    if len(all_elts) == 0:
        return TNilLit(_P0, _EMPTY_ANN)
    elements: list[TExpr] = []
    for e in all_elts:
        elements.append(_lower_expr(e, env, ctx))
    if len(elements) == 1:
        return TListLit(_P0, elements, _EMPTY_ANN)
    return TTupleLit(_P0, elements, _EMPTY_ANN)


def _lower_boolop_chain(
    values: list[ASTNode], op_type: str, idx: int, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Recursively lower non-bool and/or chain to nested ternaries."""
    v = values[idx]
    if not isinstance(v, dict):
        return TBoolLit(_P0, True, _EMPTY_ANN)
    if idx == len(values) - 1:
        return _lower_expr(v, env, ctx)
    left = _lower_expr(v, env, ctx)
    if isinstance(left, TBoolLit):
        if op_type == "And":
            if left.value:
                return _lower_boolop_chain(values, op_type, idx + 1, env, ctx)
            return left
        else:
            if left.value:
                return left
            return _lower_boolop_chain(values, op_type, idx + 1, env, ctx)
    cond = _lower_as_bool(v, env, ctx)
    rest = _lower_boolop_chain(values, op_type, idx + 1, env, ctx)
    if op_type == "And":
        return TTernary(_P0, cond, rest, left, _EMPTY_ANN)
    return TTernary(_P0, cond, left, rest, _EMPTY_ANN)


def _lower_boolop(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a BoolOp node (and/or)."""
    op_node = get_node(node, "op")
    op_type = get_str(op_node, "_type")
    values = get_nodes(node, "values")
    if len(values) < 2:
        if len(values) == 1 and isinstance(values[0], dict):
            return _lower_expr(values[0], env, ctx)
        return TBoolLit(_P0, True, _EMPTY_ANN)
    # Check if all operands are bool — if so, use && / ||
    all_bool = True
    for v in values:
        if isinstance(v, dict):
            vt = _infer_expr_type(v, env, ctx)
            if not _is_type_dict(vt, ["bool"]):
                all_bool = False
    if all_bool:
        op_str = "&&" if op_type == "And" else "||"
        first = values[0]
        if not isinstance(first, dict):
            return TBoolLit(_P0, True, _EMPTY_ANN)
        result = _lower_as_bool(first, env, ctx)
        i = 1
        while i < len(values):
            v = values[i]
            if isinstance(v, dict):
                right = _lower_as_bool(v, env, ctx)
                result = TBinaryOp(_P0, op_str, result, right, _EMPTY_ANN)
            i += 1
        return result
    # Non-bool operands: use ternaries for Python short-circuit semantics
    # a and b → truthy(a) ? b : a
    # a or b  → truthy(a) ? a : b
    # Build right-to-left for chaining: a and b and c → truthy(a) ? (truthy(b) ? c : b) : a
    return _lower_boolop_chain(values, op_type, 0, env, ctx)


def _lower_compare(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Compare node."""
    left_node = get_node(node, "left")
    ops = get_nodes(node, "ops")
    comparators = get_nodes(node, "comparators")
    if len(ops) == 0 or len(comparators) == 0:
        return TBoolLit(_P0, True, _EMPTY_ANN)
    # Single comparison
    if len(ops) == 1:
        op_node = ops[0]
        comp_node = comparators[0]
        if not isinstance(op_node, dict) or not isinstance(comp_node, dict):
            return TBoolLit(_P0, True, _EMPTY_ANN)
        return _lower_single_compare(left_node, op_node, comp_node, env, ctx)
    # Chained comparison: a < b < c → a < b && b < c
    left = _lower_expr(left_node, env, ctx)
    parts: list[TExpr] = []
    prev_node = left_node
    prev_expr = left
    i = 0
    while i < len(ops):
        op_n = ops[i]
        comp_n = comparators[i]
        if not isinstance(op_n, dict) or not isinstance(comp_n, dict):
            i += 1
            continue
        right = _lower_expr(comp_n, env, ctx)
        cmp = _make_compare_expr(prev_expr, op_n, right)
        parts.append(cmp)
        prev_node = comp_n
        prev_expr = right
        i += 1
    if len(parts) == 0:
        return TBoolLit(_P0, True, _EMPTY_ANN)
    result = parts[0]
    i = 1
    while i < len(parts):
        result = TBinaryOp(_P0, "&&", result, parts[i], _EMPTY_ANN)
        i += 1
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
    left_len = len(left_elts) if left_elts is not None else -1
    right_len = len(right_elts) if right_elts is not None else -1
    # () vs () → equal
    if left_len == 0 and right_len == 0:
        if op_type in ("Eq", "LtE", "GtE"):
            return TBoolLit(_P0, True, _EMPTY_ANN)
        return TBoolLit(_P0, False, _EMPTY_ANN)
    # () vs non-empty → empty is less
    if left_len == 0:
        if op_type in ("Lt", "LtE"):
            return TBoolLit(_P0, True, _EMPTY_ANN)
        if op_type in ("Eq", "Gt", "GtE"):
            return TBoolLit(_P0, False, _EMPTY_ANN)
        if op_type == "NotEq":
            return TBoolLit(_P0, True, _EMPTY_ANN)
    if right_len == 0:
        if op_type in ("Gt", "GtE"):
            return TBoolLit(_P0, True, _EMPTY_ANN)
        if op_type in ("Eq", "Lt", "LtE"):
            return TBoolLit(_P0, False, _EMPTY_ANN)
        if op_type == "NotEq":
            return TBoolLit(_P0, True, _EMPTY_ANN)
    # (x,) vs (y,) → compare x and y directly
    if left_len == 1 and right_len == 1:
        le = left_elts[0]
        re = right_elts[0]
        if isinstance(le, dict) and isinstance(re, dict):
            return _lower_single_compare(le, {"_type": JStr(op_type)}, re, env, ctx)
    # (x,) vs (a, b) or vice versa → compare first elements, then length
    if left_len == 1 and right_len >= 2:
        le = left_elts[0]
        if isinstance(le, dict) and isinstance(right_elts[0], dict):
            first_cmp = _lower_single_compare(
                le, {"_type": JStr("Lt")}, right_elts[0], env, ctx
            )
            first_eq = _lower_single_compare(
                le, {"_type": JStr("Eq")}, right_elts[0], env, ctx
            )
            if op_type == "Lt":
                return TBinaryOp(_P0, "||", first_cmp, first_eq, _EMPTY_ANN)
            if op_type == "LtE":
                return TBoolLit(_P0, True, _EMPTY_ANN)
    # Fallback: lower normally (may cause type errors for unsupported cases)
    left = _lower_expr(left_node, env, ctx)
    right = _lower_expr(right_node, env, ctx)
    op_dict: ASTNode = {"_type": JStr(op_type)}
    return _make_compare_expr(left, op_dict, right)


def _lower_set_compare(
    left_node: ASTNode, op_type: str, right_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Desugar set ordering operators to subset/superset checks."""
    left = _lower_expr(left_node, env, ctx)
    right = _lower_expr(right_node, env, ctx)
    # a <= b (subset): Len(Difference(a, b)) == 0
    # a < b (proper subset): a != b && Len(Difference(a, b)) == 0
    # a >= b (superset): Len(Difference(b, a)) == 0
    # a > b (proper superset): a != b && Len(Difference(b, a)) == 0
    if op_type in ("LtE", "Lt"):
        diff = _make_call("Difference", [left, right])
    else:
        diff = _make_call("Difference", [right, left])
    is_sub = TBinaryOp(
        _P0,
        "==",
        _make_call("Len", [diff]),
        TIntLit(_P0, 0, "0", _EMPTY_ANN),
        _EMPTY_ANN,
    )
    if op_type in ("LtE", "GtE"):
        return is_sub
    # proper subset/superset: also require not equal
    not_eq = TBinaryOp(_P0, "!=", left, right, _EMPTY_ANN)
    return TBinaryOp(_P0, "&&", not_eq, is_sub, _EMPTY_ANN)


def _lower_list_compare(
    left_node: ASTNode, op_type: str, right_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Desugar list ordering to ListCompare builtin."""
    left = _lower_expr(left_node, env, ctx)
    right = _lower_expr(right_node, env, ctx)
    cmp = _make_call("ListCompare", [left, right])
    zero = TIntLit(_P0, 0, "0", _EMPTY_ANN)
    op_map: dict[str, str] = {"Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">="}
    return TBinaryOp(_P0, op_map[op_type], cmp, zero, _EMPTY_ANN)


def _lower_tuple_compare(
    left_node: ASTNode, op_type: str, right_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Desugar tuple ordering into element-by-element comparison."""
    left_elts: list[ASTNode] = []
    right_elts: list[ASTNode] = []
    if _is_ast(left_node, "Tuple"):
        left_elts = [e for e in get_nodes(left_node, "elts") if isinstance(e, dict)]
    if _is_ast(right_node, "Tuple"):
        right_elts = [e for e in get_nodes(right_node, "elts") if isinstance(e, dict)]
    # If not literal tuples, fall back to equality comparison
    if len(left_elts) == 0 and not _is_ast(left_node, "Tuple"):
        left = _lower_expr(left_node, env, ctx)
        right = _lower_expr(right_node, env, ctx)
        return TBoolLit(_P0, False, _EMPTY_ANN)
    if len(right_elts) == 0 and not _is_ast(right_node, "Tuple"):
        left = _lower_expr(left_node, env, ctx)
        right = _lower_expr(right_node, env, ctx)
        return TBoolLit(_P0, False, _EMPTY_ANN)
    min_len = min(len(left_elts), len(right_elts))
    # Build from the innermost comparison outward
    # For <: a0 < b0 || (a0 == b0 && (a1 < b1 || (a1 == b1 && ... tail)))
    # tail: if same length, False for <, True for <=; if different length, shorter < longer
    if op_type in ("Lt", "LtE"):
        if len(left_elts) < len(right_elts):
            tail: TExpr = TBoolLit(_P0, True, _EMPTY_ANN)
        elif len(left_elts) > len(right_elts):
            tail = TBoolLit(_P0, False, _EMPTY_ANN)
        else:
            tail = TBoolLit(_P0, op_type == "LtE", _EMPTY_ANN)
        cmp_op = "Lt"
    else:
        # Gt, GtE: reverse — a > b is b < a
        if len(left_elts) > len(right_elts):
            tail = TBoolLit(_P0, True, _EMPTY_ANN)
        elif len(left_elts) < len(right_elts):
            tail = TBoolLit(_P0, False, _EMPTY_ANN)
        else:
            tail = TBoolLit(_P0, op_type == "GtE", _EMPTY_ANN)
        cmp_op = "Gt"
    result = tail
    i = min_len - 1
    while i >= 0:
        le = left_elts[i]
        re = right_elts[i]
        if isinstance(le, dict) and isinstance(re, dict):
            a = _lower_expr(le, env, ctx)
            b = _lower_expr(re, env, ctx)
            lt_type = _infer_expr_type(le, env, ctx)
            rt_type = _infer_expr_type(re, env, ctx)
            a_c, b_c = _coerce_compare(a, b, lt_type, rt_type)
            op_str = "<" if cmp_op == "Lt" else ">"
            elem_lt = TBinaryOp(_P0, op_str, a_c, b_c, _EMPTY_ANN)
            # Need fresh lowered expressions for equality check
            a2 = _lower_expr(le, env, ctx)
            b2 = _lower_expr(re, env, ctx)
            a2_c, b2_c = _coerce_compare(a2, b2, lt_type, rt_type)
            elem_eq = TBinaryOp(_P0, "==", a2_c, b2_c, _EMPTY_ANN)
            inner = TBinaryOp(_P0, "&&", elem_eq, result, _EMPTY_ANN)
            result = TBinaryOp(_P0, "||", elem_lt, inner, _EMPTY_ANN)
        i -= 1
    return result


def _lower_single_compare(
    left_node: ASTNode, op_node: ASTNode, comp_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower a single comparison operation."""
    op_type = get_str(op_node, "_type")
    # is None → IsNil(x) (keep IsNil to avoid checker narrowing to nil in then-body)
    if op_type == "Is":
        if _is_ast(comp_node, "Constant") and isinstance(comp_node.get("value"), JNull):
            left = _lower_expr(left_node, env, ctx)
            return _make_call("IsNil", [left])
    # is not None → x != nil (TVar) or !IsNil(expr)
    if op_type == "IsNot":
        if _is_ast(comp_node, "Constant") and isinstance(comp_node.get("value"), JNull):
            left = _lower_expr(left_node, env, ctx)
            if isinstance(left, TVar):
                return TBinaryOp(_P0, "!=", left, TNilLit(_P0, _EMPTY_ANN), _EMPTY_ANN)
            return TUnaryOp(_P0, "!", _make_call("IsNil", [left]), _EMPTY_ANN)
    # isinstance check in compare context
    # in operator
    if op_type == "In":
        return _lower_in_expr(left_node, comp_node, env, ctx)
    if op_type == "NotIn":
        inner = _lower_in_expr(left_node, comp_node, env, ctx)
        return TUnaryOp(_P0, "!", inner, _EMPTY_ANN)
    # Degenerate tuple comparisons: (), (x,) — can't represent in Taytsh
    if _is_ast(left_node, "Tuple") or _is_ast(comp_node, "Tuple"):
        left_elts = (
            get_nodes(left_node, "elts") if _is_ast(left_node, "Tuple") else None
        )
        right_elts = (
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
        if other_type is not None and not _is_optional_type(other_type):
            if op_type in ("Eq",):
                return TBoolLit(_P0, False, _EMPTY_ANN)
            if op_type in ("NotEq",):
                return TBoolLit(_P0, True, _EMPTY_ANN)
    left_type = _infer_expr_type(left_node, env, ctx)
    right_type = _infer_expr_type(comp_node, env, ctx)
    # Cross-type equality for incompatible types → false/true
    if op_type in ("Eq", "NotEq") and not _types_comparable(left_type, right_type):
        return TBoolLit(_P0, op_type == "NotEq", _EMPTY_ANN)
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
        lt = _type_dict_kind(left_type) if left_type is not None else ""
        rt = _type_dict_kind(right_type) if right_type is not None else ""
        if lt == "Tuple" or rt == "Tuple":
            return _lower_tuple_compare(left_node, op_type, comp_node, env, ctx)
    left = _lower_expr(left_node, env, ctx)
    right = _lower_expr(comp_node, env, ctx)
    left, right = _coerce_compare(left, right, left_type, right_type)
    return _make_compare_expr(left, op_node, right)


def _coerce_compare(
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
        left = _bool_to_int(left)
    elif rt_bool and lt_int:
        right = _bool_to_int(right)
    elif lt_bool and rt_bool:
        left = _bool_to_int(left)
        right = _bool_to_int(right)
    # float vs int → convert int to float
    elif lt_float and rt_int:
        right = _make_call("IntToFloat", [right])
    elif rt_float and lt_int:
        left = _make_call("IntToFloat", [left])
    # bool vs float → bool to int to float
    elif lt_bool and rt_float:
        left = _make_call("IntToFloat", [_bool_to_int(left)])
    elif rt_bool and lt_float:
        right = _make_call("IntToFloat", [_bool_to_int(right)])
    else:
        # byte vs int → convert byte to int
        lt_byte = _is_type_dict(left_type, ["byte"])
        rt_byte = _is_type_dict(right_type, ["byte"])
        if lt_byte and rt_int:
            left = _make_call("ByteToInt", [left])
        elif rt_byte and lt_int:
            right = _make_call("ByteToInt", [right])
    return left, right


def _make_compare_expr(left: TExpr, op_node: ASTNode, right: TExpr) -> TExpr:
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
    return TBinaryOp(_P0, op_str, left, right, _EMPTY_ANN)


def _lower_in_expr(
    left_node: ASTNode, right_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower 'x in collection' expression."""
    # x in (1, 2, 3) → x == 1 || x == 2 || x == 3
    if _is_ast(right_node, "Tuple"):
        elts = get_nodes(right_node, "elts")
        left = _lower_expr(left_node, env, ctx)
        if len(elts) == 0:
            return TBoolLit(_P0, False, _EMPTY_ANN)
        parts: list[TExpr] = []
        i = 0
        while i < len(elts):
            e = elts[i]
            if isinstance(e, dict):
                right = _lower_expr(e, env, ctx)
                parts.append(TBinaryOp(_P0, "==", left, right, _EMPTY_ANN))
            i += 1
        result = parts[0]
        i = 1
        while i < len(parts):
            result = TBinaryOp(_P0, "||", result, parts[i], _EMPTY_ANN)
            i += 1
        return result
    # x in collection → Contains(collection, x)
    # Type mismatch on map keys → always false
    left_type = _infer_expr_type(left_node, env, ctx)
    right_type = _infer_expr_type(right_node, env, ctx)
    if _is_type_dict(right_type, ["Map"]):
        if isinstance(right_type, MapType):
            lk = _type_dict_kind(left_type)
            rk = _type_dict_kind(right_type.key)
            if lk != "" and rk != "" and lk != rk:
                return TBoolLit(_P0, False, _EMPTY_ANN)
    left = _lower_expr(left_node, env, ctx)
    right = _lower_expr(right_node, env, ctx)
    return _make_call("Contains", [right, left])


def _lower_unaryop(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a UnaryOp node."""
    op_node = get_node(node, "op")
    op_type = get_str(op_node, "_type")
    operand_node = get_node(node, "operand")
    if op_type == "Not":
        operand_type = _infer_expr_type(operand_node, env, ctx)
        # not None → True
        if _is_ast(operand_node, "Constant") and isinstance(
            operand_node.get("value"), JNull
        ):
            return TBoolLit(_P0, True, _EMPTY_ANN)
        if _is_optional_type(operand_type):
            # not x (optional) → IsNil(x)
            operand = _lower_expr(operand_node, env, ctx)
            return _make_call("IsNil", [operand])
        if _is_type_dict(operand_type, ["bool"]):
            operand = _lower_expr(operand_node, env, ctx)
            return TUnaryOp(_P0, "!", operand, _EMPTY_ANN)
        # Non-bool not: lower as bool then negate
        return TUnaryOp(_P0, "!", _lower_as_bool(operand_node, env, ctx), _EMPTY_ANN)
    if op_type == "USub":
        operand_type = _infer_expr_type(operand_node, env, ctx)
        operand = _lower_expr(operand_node, env, ctx)
        if _is_type_dict(operand_type, ["bool"]):
            operand = _bool_to_int(operand)
        return TUnaryOp(_P0, "-", operand, _EMPTY_ANN)
    if op_type == "UAdd":
        return _lower_expr(operand_node, env, ctx)
    if op_type == "Invert":
        operand_type = _infer_expr_type(operand_node, env, ctx)
        operand = _lower_expr(operand_node, env, ctx)
        if _is_type_dict(operand_type, ["bool"]):
            operand = _bool_to_int(operand)
        return TUnaryOp(_P0, "~", operand, _EMPTY_ANN)
    return _lower_expr(operand_node, env, ctx)


def _lower_call(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Call node."""
    func_node = get_node(node, "func")
    args = get_nodes(node, "args")
    keywords = get_nodes(node, "keywords")
    # Direct function call
    if _is_ast(func_node, "Name"):
        fname = get_str(func_node, "id")
        return _lower_name_call(fname, args, keywords, node, env, ctx)
    # Method call
    if _is_ast(func_node, "Attribute"):
        return _lower_method_call(func_node, args, keywords, node, env, ctx)
    # Fallback
    func = _lower_expr(func_node, env, ctx)
    lowered_args: list[TArg] = []
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, dict):
            lowered_args.append(TArg(_P0, None, _lower_expr(a, env, ctx)))
        i += 1
    return TCall(_P0, func, lowered_args, _EMPTY_ANN)


def _lower_name_call(
    fname: str,
    args: list[ASTNode],
    keywords: list[ASTNode],
    node: ASTNode,
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Lower a direct function call by name."""
    # Builtins
    if fname == "len":
        if len(args) > 0 and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            if isinstance(arg_type, TupleType):
                n = len(arg_type.elements)
                return TIntLit(_P0, n, str(n), _EMPTY_ANN)
            if _is_ast(args[0], "Tuple"):
                elts = get_nodes(args[0], "elts")
                n = len(elts)
                return TIntLit(_P0, n, str(n), _EMPTY_ANN)
            return _make_call("Len", [_lower_expr(args[0], env, ctx)])
    if fname == "min" or fname == "max":
        builtin = "Min" if fname == "min" else "Max"
        lowered: list[TExpr] = []
        i = 0
        while i < len(args):
            a = args[i]
            if isinstance(a, dict):
                at = _infer_expr_type(a, env, ctx)
                la = _lower_expr(a, env, ctx)
                if _is_type_dict(at, ["bool"]):
                    la = _bool_to_int(la)
                lowered.append(la)
            i += 1
        if len(lowered) == 1:
            # min(list) / max(list) — single iterable form
            # Lower to a loop: let __r = xs[0]; for __i in range(1, Len(xs)) { __r = Min(__r, xs[__i]) }; return __r
            # For expression context, we'll use a call to a synthetic helper.
            # But since we can't create functions inline, reduce to chained indexing
            # Actually, pass through as 1-arg call; checker will handle it.
            return _make_call(builtin, lowered)
        if len(lowered) >= 3:
            # Chain: min(a, b, c) → Min(Min(a, b), c)
            result = _make_call(builtin, [lowered[0], lowered[1]])
            j = 2
            while j < len(lowered):
                result = _make_call(builtin, [result, lowered[j]])
                j += 1
            return result
        return _make_call(builtin, lowered)
    if fname == "pow":
        if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            a = _lower_expr(args[0], env, ctx)
            b = _lower_expr(args[1], env, ctx)
            if _is_type_dict(_infer_expr_type(args[0], env, ctx), ["bool"]):
                a = _bool_to_int(a)
            if _is_type_dict(_infer_expr_type(args[1], env, ctx), ["bool"]):
                b = _bool_to_int(b)
            return _make_call("Pow", [a, b])
    if fname == "abs":
        if len(args) > 0 and isinstance(args[0], dict):
            a = _lower_expr(args[0], env, ctx)
            if _is_type_dict(_infer_expr_type(args[0], env, ctx), ["bool"]):
                a = _bool_to_int(a)
            return _make_call("Abs", [a])
    if fname == "int":
        if len(args) == 0:
            return TIntLit(_P0, 0, "0", _EMPTY_ANN)
        if len(args) >= 1 and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            arg = _lower_expr(args[0], env, ctx)
            if len(args) >= 2 and isinstance(args[1], dict):
                base = _lower_expr(args[1], env, ctx)
                return _make_call("ParseInt", [arg, base])
            if _is_type_dict(arg_type, ["int"]):
                return arg
            if _is_type_dict(arg_type, ["bool"]):
                return TTernary(
                    _P0,
                    arg,
                    TIntLit(_P0, 1, "1", _EMPTY_ANN),
                    TIntLit(_P0, 0, "0", _EMPTY_ANN),
                    _EMPTY_ANN,
                )
            if _is_type_dict(arg_type, ["float"]):
                return _make_call("FloatToInt", [arg])
            if _is_type_dict(arg_type, ["byte"]):
                return _make_call("ByteToInt", [arg])
            return _make_call("ParseInt", [arg, TIntLit(_P0, 10, "10", _EMPTY_ANN)])
    if fname == "float":
        if len(args) == 0:
            return TFloatLit(_P0, 0.0, "0.0", _EMPTY_ANN)
        if len(args) > 0 and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            arg = _lower_expr(args[0], env, ctx)
            if _is_type_dict(arg_type, ["float"]):
                return arg
            if _is_type_dict(arg_type, ["int"]):
                return _make_call("IntToFloat", [arg])
            return _make_call("ParseFloat", [arg])
    if fname == "str":
        if len(args) > 0 and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            if _is_type_dict(arg_type, ["string"]):
                return _lower_expr(args[0], env, ctx)
            return _make_call("ToString", [_lower_expr(args[0], env, ctx)])
    if fname == "bool":
        if len(args) == 0:
            return TBoolLit(_P0, False, _EMPTY_ANN)
        if len(args) > 0 and isinstance(args[0], dict):
            # bool(None) → false
            if _is_ast(args[0], "Constant") and isinstance(args[0].get("value"), JNull):
                return TBoolLit(_P0, False, _EMPTY_ANN)
            arg_type = _infer_expr_type(args[0], env, ctx)
            arg = _lower_expr(args[0], env, ctx)
            # bool(optional) → !IsNil(x)
            if _is_optional_type(arg_type):
                return TUnaryOp(_P0, "!", _make_call("IsNil", [arg]), _EMPTY_ANN)
            if _is_type_dict(arg_type, ["int"]):
                return TBinaryOp(
                    _P0, "!=", arg, TIntLit(_P0, 0, "0", _EMPTY_ANN), _EMPTY_ANN
                )
            if _is_type_dict(arg_type, ["float"]):
                return TBinaryOp(
                    _P0, "!=", arg, TFloatLit(_P0, 0.0, "0.0", _EMPTY_ANN), _EMPTY_ANN
                )
            if _is_type_dict(arg_type, ["string"]):
                return TBinaryOp(
                    _P0, "!=", arg, TStringLit(_P0, "", _EMPTY_ANN), _EMPTY_ANN
                )
            if _is_type_dict(arg_type, ["bool"]):
                return arg
            if isinstance(arg_type, TupleType):
                if len(arg_type.elements) > 0:
                    return TBoolLit(_P0, True, _EMPTY_ANN)
                return TBoolLit(_P0, False, _EMPTY_ANN)
            if _is_type_dict(arg_type, ["bytes", "Slice", "Map", "Set"]):
                return TBinaryOp(
                    _P0,
                    "!=",
                    _make_call("Len", [arg]),
                    TIntLit(_P0, 0, "0", _EMPTY_ANN),
                    _EMPTY_ANN,
                )
            return TBinaryOp(
                _P0, "!=", arg, TIntLit(_P0, 0, "0", _EMPTY_ANN), _EMPTY_ANN
            )
    if fname == "chr":
        if len(args) > 0 and isinstance(args[0], dict):
            rune = _make_call("RuneFromInt", [_lower_expr(args[0], env, ctx)])
            return _make_call("ToString", [rune])
    if fname == "ord":
        if len(args) > 0 and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            arg = _lower_expr(args[0], env, ctx)
            if _is_type_dict(arg_type, ["string"]):
                indexed = TIndex(_P0, arg, TIntLit(_P0, 0, "0", _EMPTY_ANN), _EMPTY_ANN)
                return _make_call("RuneToInt", [indexed])
            return _make_call("RuneToInt", [arg])
    if fname == "zip":
        if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            a = _lower_expr(args[0], env, ctx)
            b = _lower_expr(args[1], env, ctx)
            return _make_call("Zip", [a, b])
    if fname == "isinstance":
        if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            return TBoolLit(_P0, True, _EMPTY_ANN)
    if fname == "any" or fname == "all":
        return TBoolLit(_P0, True, _EMPTY_ANN)
    if fname == "repr":
        if len(args) > 0 and isinstance(args[0], dict):
            return _make_call("ToString", [_lower_expr(args[0], env, ctx)])
    if fname == "sorted":
        if len(args) > 0 and isinstance(args[0], dict):
            arg = _lower_expr(args[0], env, ctx)
            # Check for reverse=True
            is_reversed = _has_keyword_true(keywords, "reverse")
            if is_reversed:
                return _make_call("Reversed", [_make_call("Sorted", [arg])])
            return _make_call("Sorted", [arg])
    if fname == "list":
        if len(args) > 0 and isinstance(args[0], dict):
            # list(range(...)) → RangeList(start, end, step)
            if _is_ast(args[0], "Call"):
                rfunc = get_node(args[0], "func")
                if _is_ast(rfunc, "Name") and get_str(rfunc, "id") == "range":
                    rargs = get_nodes(args[0], "args")
                    if len(rargs) == 1 and isinstance(rargs[0], dict):
                        end = _lower_expr(rargs[0], env, ctx)
                        return _make_call(
                            "RangeList",
                            [
                                TIntLit(_P0, 0, "0", _EMPTY_ANN),
                                end,
                                TIntLit(_P0, 1, "1", _EMPTY_ANN),
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
                            "RangeList",
                            [start, end, TIntLit(_P0, 1, "1", _EMPTY_ANN)],
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
                        return _make_call("RangeList", [start, end, step])
            # list("string") → Chars("string")
            arg_type = _infer_expr_type(args[0], env, ctx)
            if _is_type_dict(arg_type, ["string"]):
                if _is_ast(args[0], "Constant"):
                    s_jv = args[0].get("value")
                    if isinstance(s_jv, JStr):
                        return TListLit(
                            _P0,
                            [TStringLit(_P0, c, _EMPTY_ANN) for c in s_jv.value],
                            _EMPTY_ANN,
                        )
                return _make_call("Chars", [_lower_expr(args[0], env, ctx)])
            # list(zip(...)) → Zip(...)
            if _is_ast(args[0], "Call"):
                rfunc = get_node(args[0], "func")
                if _is_ast(rfunc, "Name") and get_str(rfunc, "id") == "zip":
                    return _lower_expr(args[0], env, ctx)
            # list(set) → Sorted(set)
            if isinstance(arg_type, SetType):
                return _make_call("Sorted", [_lower_expr(args[0], env, ctx)])
            arg = _lower_expr(args[0], env, ctx)
            # list(xs) → xs[0:Len(xs)]
            return TSlice(
                _P0,
                arg,
                TIntLit(_P0, 0, "0", _EMPTY_ANN),
                _make_call("Len", [arg]),
                _EMPTY_ANN,
            )
    if fname == "bytes":
        if len(args) == 0:
            return TBytesLit(_P0, b"", _EMPTY_ANN)
        if len(args) == 1 and isinstance(args[0], dict):
            arg_type = _infer_expr_type(args[0], env, ctx)
            if _is_type_dict(arg_type, ["int"]):
                return _make_call("Bytes", [_lower_expr(args[0], env, ctx)])
            if _is_type_dict(arg_type, ["Slice"]):
                return _make_call("BytesFrom", [_lower_expr(args[0], env, ctx)])
    if fname == "set" or fname == "frozenset":
        if len(args) == 0:
            return _make_call("Set", [])
        if len(args) == 1 and isinstance(args[0], dict):
            # set(genexpr/listcomp) → SetFromList(listcomp)
            if _is_ast(args[0], "GeneratorExp") or _is_ast(args[0], "ListComp"):
                return _make_call("SetFromList", [_lower_expr(args[0], env, ctx)])
            # set(range(...)) → SetFromList(RangeList(...))
            if _is_ast(args[0], "Call"):
                rfunc = get_node(args[0], "func")
                if _is_ast(rfunc, "Name") and get_str(rfunc, "id") == "range":
                    range_list = _lower_extend_arg(args[0], env, ctx)
                    return _make_call("SetFromList", [range_list])
            # set("hello") → SetFromList(["h", "e", "l", "l", "o"])
            arg_type = _infer_expr_type(args[0], env, ctx)
            if _is_type_dict(arg_type, ["string"]) and _is_ast(args[0], "Constant"):
                s_jv = args[0].get("value")
                if isinstance(s_jv, JStr):
                    chars: list[TExpr] = [
                        TStringLit(_P0, c, _EMPTY_ANN) for c in s_jv.value
                    ]
                    return _make_call("SetFromList", [TListLit(_P0, chars, _EMPTY_ANN)])
            # set(list_expr) → SetFromList(list_expr)
            if _is_type_dict(arg_type, ["Slice"]):
                return _make_call("SetFromList", [_lower_expr(args[0], env, ctx)])
            # set(any_iterable) → SetFromList(expr)
            return _make_call("SetFromList", [_lower_expr(args[0], env, ctx)])
    if fname == "tuple":
        if len(args) == 0:
            return TListLit(_P0, [], _EMPTY_ANN)
        if len(args) == 1 and isinstance(args[0], dict):
            # tuple(range(...)) → RangeList(...)
            if _is_ast(args[0], "Call"):
                rfunc = get_node(args[0], "func")
                if _is_ast(rfunc, "Name") and get_str(rfunc, "id") == "range":
                    return _lower_extend_arg(args[0], env, ctx)
            # tuple("string") → ["c", "h", ...]
            arg_type = _infer_expr_type(args[0], env, ctx)
            if _is_type_dict(arg_type, ["string"]) and _is_ast(args[0], "Constant"):
                s_jv = args[0].get("value")
                if isinstance(s_jv, JStr):
                    return TListLit(
                        _P0,
                        [TStringLit(_P0, c, _EMPTY_ANN) for c in s_jv.value],
                        _EMPTY_ANN,
                    )
            # tuple(set) → Sorted(set)
            if _is_type_dict(arg_type, ["Set"]):
                return _make_call("Sorted", [_lower_expr(args[0], env, ctx)])
            # tuple(iterable) → copy as list
            arg = _lower_expr(args[0], env, ctx)
            return TSlice(
                _P0,
                arg,
                TIntLit(_P0, 0, "0", _EMPTY_ANN),
                _make_call("Len", [arg]),
                _EMPTY_ANN,
            )
    if fname == "dict":
        if len(args) == 0:
            return _make_call("Map", [])
        if len(args) == 1 and isinstance(args[0], dict):
            return _make_call("MapFromPairs", [_lower_expr(args[0], env, ctx)])
    if fname == "hex":
        if len(args) > 0 and isinstance(args[0], dict):
            return _make_call(
                "FormatInt",
                [_lower_expr(args[0], env, ctx), TIntLit(_P0, 16, "16", _EMPTY_ANN)],
            )
    if fname == "divmod":
        if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            a = _lower_expr(args[0], env, ctx)
            b = _lower_expr(args[1], env, ctx)
            return TTupleLit(
                _P0,
                [_make_call("FloorDiv", [a, b]), _make_call("PythonMod", [a, b])],
                _EMPTY_ANN,
            )
    if fname == "print":
        return _lower_print_call(args, keywords, env, ctx)
    # Python builtin exceptions → struct constructors with message field
    if fname in (
        "TypeError",
        "NotImplementedError",
        "RuntimeError",
        "KeyError",
        "IndexError",
    ):
        exc_args: list[TArg] = []
        if len(args) > 0 and isinstance(args[0], dict):
            exc_args.append(TArg(_P0, None, _lower_expr(args[0], env, ctx)))
        else:
            exc_args.append(TArg(_P0, None, TStringLit(_P0, "", _EMPTY_ANN)))
        return TCall(_P0, TVar(_P0, fname, _EMPTY_ANN), exc_args, _EMPTY_ANN)
    # Struct constructor
    if fname in ctx.known_classes:
        return _lower_struct_constructor(fname, args, keywords, env, ctx)
    # Regular function call
    lowered_args: list[TArg] = []
    if len(keywords) > 0:
        # Convert all args to named when keywords are present
        func_info = ctx.sig_result.functions.get(fname)
        i = 0
        while i < len(args):
            a = args[i]
            if isinstance(a, dict):
                pname: str | None = None
                if func_info is not None and i < len(func_info.params):
                    pname = _safe_name(func_info.params[i].name)
                lowered_args.append(TArg(_P0, pname, _lower_expr(a, env, ctx)))
            i += 1
        i = 0
        while i < len(keywords):
            kw = keywords[i]
            kw_name = get_str(kw, "arg")
            kw_val = get_node(kw, "value")
            if kw_name != "" and len(kw_val) > 0:
                lowered_args.append(
                    TArg(_P0, _safe_name(kw_name), _lower_expr(kw_val, env, ctx))
                )
            i += 1
    else:
        i = 0
        while i < len(args):
            a = args[i]
            if isinstance(a, dict):
                lowered_args.append(TArg(_P0, None, _lower_expr(a, env, ctx)))
            i += 1
    safe = _safe_name(fname)
    return TCall(_P0, TVar(_P0, safe, _name_ann(safe, fname)), lowered_args, _EMPTY_ANN)


def _has_keyword_true(keywords: list[ASTNode], name: str) -> bool:
    """Check if keywords contain name=True."""
    i = 0
    while i < len(keywords):
        kw = keywords[i]
        if get_str(kw, "arg") == name:
            val_node = get_node(kw, "value")
            v = val_node.get("value")
            if isinstance(v, JBool) and v.value is True:
                return True
        i += 1
    return False


def _get_keyword_value(keywords: list[ASTNode], name: str) -> ASTNode | None:
    """Get value node for a keyword argument."""
    i = 0
    while i < len(keywords):
        kw = keywords[i]
        if get_str(kw, "arg") == name:
            val_node = get_node(kw, "value")
            if len(val_node) > 0:
                return val_node
        i += 1
    return None


def _lower_print_call(
    args: list[ASTNode], keywords: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower print() to WritelnOut/WriteOut/WritelnErr or Print()."""
    # Get the argument (print typically has one arg in subset)
    arg_expr: TExpr = TStringLit(_P0, "", _EMPTY_ANN)
    is_string = True
    if len(args) > 0 and isinstance(args[0], dict):
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
        if isinstance(v, JStr) and v.value == "":
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
            return _make_call("WritelnErr", [arg_expr])
        if no_newline:
            return _make_call("WriteOut", [arg_expr])
        return _make_call("WritelnOut", [arg_expr])
    # For non-string values: use Print() with named args
    if is_stderr:
        return _make_named_call(
            "Print", [arg_expr], [("stderr", TBoolLit(_P0, True, _EMPTY_ANN))]
        )
    if no_newline:
        return _make_named_call(
            "Print", [arg_expr], [("newline", TBoolLit(_P0, False, _EMPTY_ANN))]
        )
    return _make_named_call(
        "Print", [arg_expr], [("newline", TBoolLit(_P0, True, _EMPTY_ANN))]
    )


def _lower_struct_constructor(
    class_name: str,
    args: list[ASTNode],
    keywords: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Lower a struct constructor call."""
    lowered_args: list[TArg] = []
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, dict):
            lowered_args.append(TArg(_P0, None, _lower_expr(a, env, ctx)))
        i += 1
    # Also lower keyword args
    i = 0
    while i < len(keywords):
        kw = keywords[i]
        kw_name = get_str(kw, "arg")
        kw_val = get_node(kw, "value")
        if kw_name != "" and len(kw_val) > 0:
            lowered_args.append(TArg(_P0, kw_name, _lower_expr(kw_val, env, ctx)))
        i += 1
    return TCall(_P0, TVar(_P0, class_name, _EMPTY_ANN), lowered_args, _EMPTY_ANN)


def _lower_method_call(
    func_node: ASTNode,
    args: list[ASTNode],
    keywords: list[ASTNode],
    node: ASTNode,
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Lower a method call."""
    method_name = get_str(func_node, "attr")
    obj_node = get_node(func_node, "value")
    obj_type = _infer_expr_type(obj_node, env, ctx)
    # sys.exit(n) → Exit(n)
    if _is_ast(obj_node, "Name") and get_str(obj_node, "id") == "sys":
        if method_name == "exit":
            exit_args: list[TExpr] = []
            if len(args) > 0 and isinstance(args[0], dict):
                exit_args.append(_lower_expr(args[0], env, ctx))
            else:
                exit_args.append(TIntLit(_P0, 0, "0", _EMPTY_ANN))
            return _make_call("Exit", exit_args)
    # sys.stdin methods
    if _is_ast(obj_node, "Attribute"):
        inner_obj = get_node(obj_node, "value")
        inner_attr = get_str(obj_node, "attr")
        if _is_ast(inner_obj, "Name") and get_str(inner_obj, "id") == "sys":
            if inner_attr == "stdin":
                if method_name == "readline":
                    return _make_call("ReadLine", [])
                if method_name == "read":
                    return _make_call("ReadAll", [])
    # sys.stdin.buffer.read / sys.stdout.buffer.write / sys.stderr.buffer.write
    if _is_ast(obj_node, "Attribute"):
        inner_obj = get_node(obj_node, "value")
        inner_attr = get_str(obj_node, "attr")
        if _is_ast(inner_obj, "Attribute"):
            inner2_obj = get_node(inner_obj, "value")
            inner2_attr = get_str(inner_obj, "attr")
            if _is_ast(inner2_obj, "Name") and get_str(inner2_obj, "id") == "sys":
                if inner2_attr == "stdin" and inner_attr == "buffer":
                    if method_name == "read":
                        if len(args) > 0 and isinstance(args[0], dict):
                            return _make_call(
                                "ReadBytesN", [_lower_expr(args[0], env, ctx)]
                            )
                        return _make_call("ReadBytes", [])
                if inner2_attr == "stdout" and inner_attr == "buffer":
                    if method_name == "write":
                        if len(args) > 0 and isinstance(args[0], dict):
                            return _make_call(
                                "WriteOut", [_lower_expr(args[0], env, ctx)]
                            )
                if inner2_attr == "stderr" and inner_attr == "buffer":
                    if method_name == "write":
                        if len(args) > 0 and isinstance(args[0], dict):
                            return _make_call(
                                "WriteErr", [_lower_expr(args[0], env, ctx)]
                            )
    # os.getenv
    if _is_ast(obj_node, "Name") and get_str(obj_node, "id") == "os":
        if method_name == "getenv":
            lowered: list[TExpr] = []
            i = 0
            while i < len(args):
                a = args[i]
                if isinstance(a, dict):
                    lowered.append(_lower_expr(a, env, ctx))
                i += 1
            return _make_call("GetEnv", lowered)
    # dict.fromkeys(keys, value?)
    if _is_ast(obj_node, "Name") and get_str(obj_node, "id") == "dict":
        if method_name == "fromkeys":
            fk_args: list[TExpr] = []
            fi = 0
            while fi < len(args):
                a = args[fi]
                if isinstance(a, dict):
                    fk_args.append(_lower_expr(a, env, ctx))
                fi += 1
            if len(fk_args) == 1:
                fk_args.append(TNilLit(_P0, _EMPTY_ANN))
            return _make_call("MapFromKeys", fk_args)
    obj = _lower_expr(obj_node, env, ctx)
    # Unwrap pointer for type dispatch
    actual_type = _unwrap_pointer(obj_type)
    # String methods
    if _is_type_dict(actual_type, ["string"]):
        return _lower_string_method(obj, method_name, args, env, ctx)
    # Bytes methods
    if _is_type_dict(actual_type, ["bytes"]) or _is_bytes_slice(actual_type):
        return _lower_bytes_method(obj, method_name, args, env, ctx)
    # List methods
    if _is_type_dict(actual_type, ["Slice"]):
        return _lower_list_method(obj, obj_node, method_name, args, env, ctx)
    # Dict methods
    if _is_type_dict(actual_type, ["Map"]):
        return _lower_dict_method(obj, obj_node, method_name, args, env, ctx)
    # Set methods
    if _is_type_dict(actual_type, ["Set"]):
        return _lower_set_method(obj, method_name, args, env, ctx)
    # Struct method call
    lowered_args: list[TArg] = []
    if len(keywords) > 0:
        # Convert all args to named when keywords are present
        method_params = _lookup_method_params(actual_type, method_name, ctx)
        i = 0
        while i < len(args):
            a = args[i]
            if isinstance(a, dict):
                pname: str | None = None
                if method_params is not None and i < len(method_params):
                    pname = _safe_name(method_params[i].name)
                lowered_args.append(TArg(_P0, pname, _lower_expr(a, env, ctx)))
            i += 1
        i = 0
        while i < len(keywords):
            kw = keywords[i]
            kw_name = get_str(kw, "arg")
            kw_val = get_node(kw, "value")
            if kw_name != "" and len(kw_val) > 0:
                lowered_args.append(
                    TArg(_P0, _safe_name(kw_name), _lower_expr(kw_val, env, ctx))
                )
            i += 1
    else:
        i = 0
        while i < len(args):
            a = args[i]
            if isinstance(a, dict):
                lowered_args.append(TArg(_P0, None, _lower_expr(a, env, ctx)))
            i += 1
    return TCall(
        _P0, TFieldAccess(_P0, obj, method_name, _EMPTY_ANN), lowered_args, _EMPTY_ANN
    )


def _lower_string_method(
    obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower string method calls."""
    lowered: list[TExpr] = []
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, dict):
            lowered.append(_lower_expr(a, env, ctx))
        i += 1
    if method == "find":
        return _make_call("Find", [obj] + lowered)
    if method == "rfind":
        return _make_call("RFind", [obj] + lowered)
    if method == "split":
        if len(lowered) == 0:
            return _make_call("SplitWhitespace", [obj])
        if len(lowered) == 2:
            plus_one = TBinaryOp(
                _P0, "+", lowered[1], TIntLit(_P0, 1, "1", _EMPTY_ANN), _EMPTY_ANN
            )
            return _make_call("SplitN", [obj, lowered[0], plus_one])
        return _make_call("Split", [obj] + lowered)
    if method == "replace":
        return _make_call("Replace", [obj] + lowered)
    if method == "count":
        return _make_call("Count", [obj] + lowered)
    if method == "startswith":
        return _lower_startswith_endswith("StartsWith", obj, args, env, ctx)
    if method == "endswith":
        return _lower_startswith_endswith("EndsWith", obj, args, env, ctx)
    if method == "strip":
        if len(lowered) == 0:
            lowered = [TStringLit(_P0, " \t\n\r\x0b\x0c", _EMPTY_ANN)]
        return _make_call("Trim", [obj] + lowered)
    if method == "lstrip":
        if len(lowered) == 0:
            lowered = [TStringLit(_P0, " \t\n\r\x0b\x0c", _EMPTY_ANN)]
        return _make_call("TrimStart", [obj] + lowered)
    if method == "rstrip":
        if len(lowered) == 0:
            lowered = [TStringLit(_P0, " \t\n\r\x0b\x0c", _EMPTY_ANN)]
        return _make_call("TrimEnd", [obj] + lowered)
    if method == "lower":
        return _make_call("Lower", [obj])
    if method == "upper":
        return _make_call("Upper", [obj])
    if method == "join":
        return _make_call("Join", [obj] + lowered)
    if method == "isdigit":
        return _make_call("IsDigit", [obj])
    if method == "isalpha":
        return _make_call("IsAlpha", [obj])
    if method == "isalnum":
        return _make_call("IsAlnum", [obj])
    if method == "isspace":
        return _make_call("IsSpace", [obj])
    if method == "isupper":
        return _make_call("IsUpper", [obj])
    if method == "islower":
        return _make_call("IsLower", [obj])
    if method == "encode":
        return _make_call("Encode", [obj])
    if method == "index":
        return _make_call("IndexOf", [obj] + lowered)
    return _make_method_call(obj, method, lowered)


def _lower_startswith_endswith(
    func_name: str, obj: TExpr, args: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower startswith/endswith, handling tuple argument."""
    if len(args) > 0 and isinstance(args[0], dict):
        arg = args[0]
        if _is_ast(arg, "Tuple"):
            # Tuple argument: startswith(("a", "b")) → StartsWith(s, "a") || StartsWith(s, "b")
            elts = get_nodes(arg, "elts")
            parts: list[TExpr] = []
            i = 0
            while i < len(elts):
                e = elts[i]
                if isinstance(e, dict):
                    lowered_e = _lower_expr(e, env, ctx)
                    parts.append(_make_call(func_name, [obj, lowered_e]))
                i += 1
            if len(parts) == 0:
                return TBoolLit(_P0, False, _EMPTY_ANN)
            result = parts[0]
            i = 1
            while i < len(parts):
                result = TBinaryOp(_P0, "||", result, parts[i], _EMPTY_ANN)
                i += 1
            return result
        # Single argument
        lowered_arg = _lower_expr(arg, env, ctx)
        return _make_call(func_name, [obj, lowered_arg])
    return _make_call(func_name, [obj])


def _lower_list_method(
    obj: TExpr,
    obj_node: ASTNode,
    method: str,
    args: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Lower list method calls."""
    lowered: list[TExpr] = []
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, dict):
            lowered.append(_lower_expr(a, env, ctx))
        i += 1
    if method == "append":
        return _make_call("Append", [obj] + lowered)
    if method == "insert":
        return _make_call("Insert", [obj] + lowered)
    if method == "pop":
        if len(lowered) == 0:
            return _make_call("Pop", [obj])
        return TIndex(_P0, obj, lowered[0], _EMPTY_ANN)
    if method == "index":
        if len(lowered) >= 2:
            val = lowered[0]
            start = lowered[1]
            sliced = TSlice(_P0, obj, start, _make_call("Len", [obj]), _EMPTY_ANN)
            return TBinaryOp(
                _P0, "+", _make_call("IndexOf", [sliced, val]), start, _EMPTY_ANN
            )
        return _make_call("IndexOf", [obj] + lowered)
    if method == "remove":
        if len(lowered) > 0:
            return _make_call(
                "RemoveAt", [obj, _make_call("IndexOf", [obj, lowered[0]])]
            )
        return _make_call("RemoveAt", [obj])
    if method == "copy":
        return TSlice(
            _P0,
            obj,
            TIntLit(_P0, 0, "0", _EMPTY_ANN),
            _make_call("Len", [obj]),
            _EMPTY_ANN,
        )
    if method == "clear":
        return TListLit(_P0, [], _EMPTY_ANN)
    if method == "reverse":
        return TNilLit(_P0, _EMPTY_ANN)
    if method == "sort":
        return TNilLit(_P0, _EMPTY_ANN)
    return _make_method_call(obj, method, lowered)


def _lower_dict_method(
    obj: TExpr,
    obj_node: ASTNode,
    method: str,
    args: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> TExpr:
    """Lower dict method calls."""
    lowered: list[TExpr] = []
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, dict):
            lowered.append(_lower_expr(a, env, ctx))
        i += 1
    if method == "get":
        return _make_call("Get", [obj] + lowered)
    if method == "keys":
        return _make_call("Keys", [obj])
    if method == "values":
        return _make_call("Values", [obj])
    if method == "items":
        return _make_call("Items", [obj])
    if method == "copy":
        return _make_call("Merge", [obj, _make_call("Map", [])])
    if method == "pop":
        return TIndex(_P0, obj, lowered[0], _EMPTY_ANN)
    if method == "setdefault":
        return _make_call("Get", [obj] + lowered)
    if method == "update":
        return _make_call("Merge", [obj] + lowered)
    if method == "popitem":
        return _make_call("PopItem", [obj])
    return _make_method_call(obj, method, lowered)


def _method_side_effects(value_node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Return side-effect statements for methods that need post-assignment work."""
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
        if len(vargs) > 0 and isinstance(vargs[0], dict):
            obj = _lower_expr(obj_node, env, ctx)
            key = _lower_expr(vargs[0], env, ctx)
            return [TExprStmt(_P0, _make_call("Delete", [obj, key]), _EMPTY_ANN)]
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
            cond = TUnaryOp(_P0, "!", _make_call("Contains", [obj, key]), _EMPTY_ANN)
            assign = TAssignStmt(
                _P0, TIndex(_P0, obj, key, _EMPTY_ANN), default, _EMPTY_ANN
            )
            return [TIfStmt(_P0, cond, [assign], None, _EMPTY_ANN)]
    # list.pop(i) → RemoveAt(xs, i)
    if _is_type_dict(actual, ["Slice"]) and method == "pop":
        if len(vargs) > 0 and isinstance(vargs[0], dict):
            obj = _lower_expr(obj_node, env, ctx)
            idx = _lower_expr(vargs[0], env, ctx)
            return [TExprStmt(_P0, _make_call("RemoveAt", [obj, idx]), _EMPTY_ANN)]
    # list.sort() → xs = Sorted(xs)
    if _is_type_dict(actual, ["Slice"]) and method == "sort":
        obj = _lower_expr(obj_node, env, ctx)
        return [TAssignStmt(_P0, obj, _make_call("Sorted", [obj]), _EMPTY_ANN)]
    # list.reverse() → xs = Reversed(xs)
    if _is_type_dict(actual, ["Slice"]) and method == "reverse":
        obj = _lower_expr(obj_node, env, ctx)
        return [TAssignStmt(_P0, obj, _make_call("Reversed", [obj]), _EMPTY_ANN)]
    return []


def _lower_set_method(
    obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower set method calls."""
    lowered: list[TExpr] = []
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, dict):
            lowered.append(_lower_expr(a, env, ctx))
        i += 1
    if method == "add":
        return _make_call("Add", [obj] + lowered)
    if method == "remove":
        return _make_call("Remove", [obj] + lowered)
    if method == "discard":
        return _make_call("Remove", [obj] + lowered)
    if method == "pop":
        return _make_call("Pop", [obj])
    if method == "copy":
        return _make_call("Union", [obj, _make_call("Set", [])])
    if method == "union":
        if len(lowered) >= 1:
            result = obj
            li = 0
            while li < len(lowered):
                result = _make_call("Union", [result, lowered[li]])
                li += 1
            return result
    if method == "intersection":
        if len(lowered) >= 1:
            result = obj
            li = 0
            while li < len(lowered):
                result = _make_call("Intersection", [result, lowered[li]])
                li += 1
            return result
    if method == "difference":
        if len(lowered) >= 1:
            result = obj
            li = 0
            while li < len(lowered):
                result = _make_call("Difference", [result, lowered[li]])
                li += 1
            return result
    if method == "symmetric_difference":
        if len(lowered) == 1:
            u = _make_call("Union", [obj, lowered[0]])
            i2 = _make_call("Intersection", [obj, lowered[0]])
            return _make_call("Difference", [u, i2])
    if method == "issubset":
        if len(lowered) == 1:
            diff = _make_call("Difference", [obj, lowered[0]])
            return TBinaryOp(
                _P0,
                "==",
                _make_call("Len", [diff]),
                TIntLit(_P0, 0, "0", _EMPTY_ANN),
                _EMPTY_ANN,
            )
    if method == "issuperset":
        if len(lowered) == 1:
            diff = _make_call("Difference", [lowered[0], obj])
            return TBinaryOp(
                _P0,
                "==",
                _make_call("Len", [diff]),
                TIntLit(_P0, 0, "0", _EMPTY_ANN),
                _EMPTY_ANN,
            )
    if method == "isdisjoint":
        if len(lowered) == 1:
            inter = _make_call("Intersection", [obj, lowered[0]])
            return TBinaryOp(
                _P0,
                "==",
                _make_call("Len", [inter]),
                TIntLit(_P0, 0, "0", _EMPTY_ANN),
                _EMPTY_ANN,
            )
    return _make_method_call(obj, method, lowered)


def _lower_bytes_method(
    obj: TExpr, method: str, args: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> TExpr:
    """Lower bytes method calls."""
    if method == "decode":
        return _make_call("Decode", [obj])
    if method == "startswith":
        return _lower_startswith_endswith("StartsWith", obj, args, env, ctx)
    if method == "endswith":
        return _lower_startswith_endswith("EndsWith", obj, args, env, ctx)
    lowered: list[TExpr] = []
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, dict):
            lowered.append(_lower_expr(a, env, ctx))
        i += 1
    return _make_method_call(obj, method, lowered)


def _lower_subscript(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Subscript node."""
    obj_node = get_node(node, "value")
    slice_node = get_node(node, "slice")
    obj = _lower_expr(obj_node, env, ctx)
    obj_type = _infer_expr_type(obj_node, env, ctx)
    # Slice access: xs[a:b]
    if _is_ast(slice_node, "Slice"):
        lower_jv = slice_node.get("lower")
        upper_jv = slice_node.get("upper")
        low: TExpr
        high: TExpr
        if lower_jv is None or isinstance(lower_jv, JNull):
            low = TIntLit(_P0, 0, "0", _EMPTY_ANN)
        elif isinstance(lower_jv, JDict):
            lower_val = lower_jv.entries
            if (
                isinstance(lower_val.get("value"), JNull)
                and get_str(lower_val, "_type") != "Constant"
            ):
                low = _lower_expr(lower_val, env, ctx)
            else:
                low = _lower_expr(lower_val, env, ctx)
        else:
            low = TIntLit(_P0, 0, "0", _EMPTY_ANN)
        if upper_jv is None or isinstance(upper_jv, JNull):
            high = _len_expr(obj, obj_type)
        elif isinstance(upper_jv, JDict):
            upper_val = upper_jv.entries
            if (
                isinstance(upper_val.get("value"), JNull)
                and get_str(upper_val, "_type") != "Constant"
            ):
                high = _lower_expr(upper_val, env, ctx)
            else:
                high = _lower_expr(upper_val, env, ctx)
        else:
            high = _len_expr(obj, obj_type)
        return TSlice(_P0, obj, low, high, _EMPTY_ANN)
    # Tuple index: t[0] → t.0 (only for multi-element tuples, not single-element)
    if _is_type_dict(obj_type, ["Tuple"]) and not _is_single_elem_tuple(obj_type):
        if _is_ast(slice_node, "Constant"):
            idx_jv = slice_node.get("value")
            if isinstance(idx_jv, JInt):
                return TTupleAccess(_P0, obj, idx_jv.value, _EMPTY_ANN)
    # Negative index: xs[-1] → xs[Len(xs) - 1]
    is_string = _is_type_dict(obj_type, ["string"])
    if _is_ast(slice_node, "Constant"):
        val_jv = slice_node.get("value")
        if isinstance(val_jv, JInt) and val_jv.value < 0:
            n = -val_jv.value
            idx_expr = TBinaryOp(
                _P0,
                "-",
                _len_expr(obj, obj_type),
                TIntLit(_P0, n, str(n), _EMPTY_ANN),
                _EMPTY_ANN,
            )
            result = TIndex(_P0, obj, idx_expr, _EMPTY_ANN)
            if is_string:
                return _make_call("ToString", [result])
            return result
    if _is_ast(slice_node, "UnaryOp"):
        op_node = get_node(slice_node, "op")
        if get_str(op_node, "_type") == "USub":
            operand = get_node(slice_node, "operand")
            if _is_ast(operand, "Constant"):
                op_val_jv = operand.get("value")
                if isinstance(op_val_jv, JInt):
                    idx_expr = TBinaryOp(
                        _P0,
                        "-",
                        _len_expr(obj, obj_type),
                        TIntLit(_P0, op_val_jv.value, str(op_val_jv.value), _EMPTY_ANN),
                        _EMPTY_ANN,
                    )
                    result = TIndex(_P0, obj, idx_expr, _EMPTY_ANN)
                    if is_string:
                        return _make_call("ToString", [result])
                    return result
    # Normal index
    idx = _lower_expr(slice_node, env, ctx)
    result = TIndex(_P0, obj, idx, _EMPTY_ANN)
    if is_string:
        return _make_call("ToString", [result])
    return result


def _lower_ternary_cond(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a ternary condition, using == nil / != nil for nil checks so the
    type checker can narrow optional variables in then/else branches."""
    if _is_ast(node, "Compare"):
        ops = get_nodes(node, "ops")
        comps = get_nodes(node, "comparators")
        if len(ops) == 1 and len(comps) == 1 and isinstance(ops[0], dict):
            op_type = get_str(ops[0], "_type")
            comp = comps[0]
            left_node = get_node(node, "left")
            if op_type == "Is" and _is_ast(comp, "Constant") and isinstance(
                comp.get("value"), JNull
            ):
                left = _lower_expr(left_node, env, ctx)
                return TBinaryOp(_P0, "==", left, TNilLit(_P0, _EMPTY_ANN), _EMPTY_ANN)
            if op_type == "IsNot" and _is_ast(comp, "Constant") and isinstance(
                comp.get("value"), JNull
            ):
                left = _lower_expr(left_node, env, ctx)
                return TBinaryOp(_P0, "!=", left, TNilLit(_P0, _EMPTY_ANN), _EMPTY_ANN)
    return _lower_as_bool(node, env, ctx)


def _lower_ifexp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower an IfExp (ternary) node."""
    test = get_node(node, "test")
    body = get_node(node, "body")
    orelse = get_node(node, "orelse")
    cond = _lower_ternary_cond(test, env, ctx)
    then_expr = _lower_expr(body, env, ctx)
    else_expr = _lower_expr(orelse, env, ctx)
    return TTernary(_P0, cond, then_expr, else_expr, _EMPTY_ANN)


def _lower_list_literal(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a List literal."""
    elts = get_nodes(node, "elts")
    elements: list[TExpr] = []
    i = 0
    while i < len(elts):
        e = elts[i]
        if isinstance(e, dict):
            elements.append(_lower_expr(e, env, ctx))
        i += 1
    return TListLit(_P0, elements, _EMPTY_ANN)


def _lower_dict_literal(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Dict literal."""
    keys = get_nodes(node, "keys")
    values = get_nodes(node, "values")
    entries: list[tuple[TExpr, TExpr]] = []
    i = 0
    while i < len(keys):
        k = keys[i]
        v = values[i] if i < len(values) else None
        if isinstance(k, dict) and isinstance(v, dict):
            entries.append((_lower_expr(k, env, ctx), _lower_expr(v, env, ctx)))
        i += 1
    if len(entries) == 0:
        return _make_call("Map", [])
    return TMapLit(_P0, entries, _EMPTY_ANN)


def _lower_set_literal(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Set literal."""
    elts = get_nodes(node, "elts")
    elements: list[TExpr] = []
    i = 0
    while i < len(elts):
        e = elts[i]
        if isinstance(e, dict):
            elements.append(_lower_expr(e, env, ctx))
        i += 1
    return TSetLit(_P0, elements, _EMPTY_ANN)


def _lower_list_from_tuple(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Tuple AST node as a list literal (for variadic tuples)."""
    elts = get_nodes(node, "elts")
    elements: list[TExpr] = []
    i = 0
    while i < len(elts):
        e = elts[i]
        if isinstance(e, dict):
            elements.append(_lower_expr(e, env, ctx))
        i += 1
    return TListLit(_P0, elements, _EMPTY_ANN)


def _lower_tuple_literal(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a Tuple literal."""
    elts = get_nodes(node, "elts")
    elements: list[TExpr] = []
    i = 0
    while i < len(elts):
        e = elts[i]
        if isinstance(e, dict):
            elements.append(_lower_expr(e, env, ctx))
        i += 1
    if len(elements) == 0:
        return TListLit(_P0, [], _EMPTY_ANN)
    if len(elements) == 1:
        return TListLit(_P0, elements, _EMPTY_ANN)
    return TTupleLit(_P0, elements, _EMPTY_ANN)


def _lower_fstring(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a JoinedStr (f-string) node to Format(template, args)."""
    values = get_nodes(node, "values")
    template_parts: list[str] = []
    fmt_args: list[TExpr] = []
    i = 0
    while i < len(values):
        v = values[i]
        vtype = get_str(v, "_type")
        if vtype == "Constant":
            val = v.get("value")
            if isinstance(val, JStr):
                template_parts.append(val.value)
        elif vtype == "FormattedValue":
            template_parts.append("{}")
            inner = get_node(v, "value")
            fmt_args.append(_lower_expr(inner, env, ctx))
        i += 1
    template = "".join(template_parts)
    all_args: list[TExpr] = [TStringLit(_P0, template, _EMPTY_ANN)] + fmt_args
    return _make_call("Format", all_args)


def _lower_listcomp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower a ListComp — fallback for expression contexts. Returns empty list."""
    return TListLit(_P0, [], _EMPTY_ANN)


def _expand_listcomp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Expand a ListComp into: let __result__ = []; for x in xs { Append(__result__, elt) }; return __result__."""
    elt = get_node(node, "elt")
    generators = get_nodes(node, "generators")
    if len(generators) == 0:
        return [TReturnStmt(_P0, TListLit(_P0, [], _EMPTY_ANN), _EMPTY_ANN)]
    gen = generators[0]
    if not isinstance(gen, dict):
        return [TReturnStmt(_P0, TListLit(_P0, [], _EMPTY_ANN), _EMPTY_ANN)]
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
    result_var = TVar(_P0, "__result__", _EMPTY_ANN)
    # Build: let __result__: list[...] = []
    ret_type = env.return_type
    result_type: TType = _typenode_to_ttype(ret_type)
    let_stmt = TLetStmt(
        _P0, "__result__", result_type, TListLit(_P0, [], _EMPTY_ANN), _EMPTY_ANN
    )
    # Build: for target_name in iter { Append(__result__, elt) }
    append_call = _make_call("Append", [result_var, elt_expr])
    body: list[TStmt] = [TExprStmt(_P0, append_call, _EMPTY_ANN)]
    # Handle optional filter (ifs in generator)
    ifs = get_nodes(gen, "ifs")
    if len(ifs) > 0 and isinstance(ifs[0], dict):
        cond = _lower_as_bool(ifs[0], comp_env, ctx)
        body = [TIfStmt(_P0, cond, body, None, _EMPTY_ANN)]
    for_stmt = TForStmt(_P0, [target_name], iter_expr, body, t_ann)
    # Return __result__
    return_stmt = TReturnStmt(_P0, result_var, _EMPTY_ANN)
    return [let_stmt, for_stmt, return_stmt]


def _expand_setcomp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Expand a SetComp into: let __result__ = {}; for x in xs { Add(__result__, elt) }; return __result__."""
    elt = get_node(node, "elt")
    generators = get_nodes(node, "generators")
    empty_set = _make_call("Set", [])
    if len(generators) == 0:
        return [TReturnStmt(_P0, empty_set, _EMPTY_ANN)]
    gen = generators[0]
    if not isinstance(gen, dict):
        return [TReturnStmt(_P0, empty_set, _EMPTY_ANN)]
    target = get_node(gen, "target")
    iter_node = get_node(gen, "iter")
    orig_name = get_str(target, "id")
    target_name = _safe_name(orig_name)
    t_ann = _name_ann(target_name, orig_name)
    iter_expr = _lower_expr(iter_node, env, ctx)
    comp_env = env.copy()
    comp_env.declared.add(orig_name)
    elt_expr = _lower_expr(elt, comp_env, ctx)
    result_var = TVar(_P0, "__result__", _EMPTY_ANN)
    ret_type = env.return_type
    result_type: TType = _typenode_to_ttype(ret_type)
    let_stmt = TLetStmt(_P0, "__result__", result_type, empty_set, _EMPTY_ANN)
    add_call = _make_call("Add", [result_var, elt_expr])
    body: list[TStmt] = [TExprStmt(_P0, add_call, _EMPTY_ANN)]
    ifs = get_nodes(gen, "ifs")
    if len(ifs) > 0 and isinstance(ifs[0], dict):
        cond = _lower_as_bool(ifs[0], comp_env, ctx)
        body = [TIfStmt(_P0, cond, body, None, _EMPTY_ANN)]
    for_stmt = TForStmt(_P0, [target_name], iter_expr, body, t_ann)
    return_stmt = TReturnStmt(_P0, result_var, _EMPTY_ANN)
    return [let_stmt, for_stmt, return_stmt]


def _expand_dictcomp(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Expand a DictComp into: let __result__ = Map(); for k in xs { __result__[key] = val }; return __result__."""
    key_node = get_node(node, "key")
    value_node = get_node(node, "value")
    generators = get_nodes(node, "generators")
    if len(generators) == 0:
        return [TReturnStmt(_P0, _make_call("Map", []), _EMPTY_ANN)]
    gen = generators[0]
    if not isinstance(gen, dict):
        return [TReturnStmt(_P0, _make_call("Map", []), _EMPTY_ANN)]
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
    result_var = TVar(_P0, "__result__", _EMPTY_ANN)
    ret_type = env.return_type
    result_type: TType = _typenode_to_ttype(ret_type)
    let_stmt = TLetStmt(
        _P0, "__result__", result_type, _make_call("Map", []), _EMPTY_ANN
    )
    idx_target = TIndex(_P0, result_var, key_expr, _EMPTY_ANN)
    body: list[TStmt] = [TAssignStmt(_P0, idx_target, val_expr, _EMPTY_ANN)]
    ifs = get_nodes(gen, "ifs")
    if len(ifs) > 0 and isinstance(ifs[0], dict):
        cond = _lower_as_bool(ifs[0], comp_env, ctx)
        body = [TIfStmt(_P0, cond, body, None, _EMPTY_ANN)]
    for_stmt = TForStmt(_P0, [target_name], iter_expr, body, t_ann)
    return_stmt = TReturnStmt(_P0, result_var, _EMPTY_ANN)
    return [let_stmt, for_stmt, return_stmt]


# ---------------------------------------------------------------------------
# Truthiness (as_bool)
# ---------------------------------------------------------------------------


def _lower_as_bool(node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower an expression as a boolean condition."""
    expr_type = _infer_expr_type(node, env, ctx)
    if _is_type_dict(expr_type, ["bool"]):
        return _lower_expr(node, env, ctx)
    if _is_optional_type(expr_type):
        expr = _lower_expr(node, env, ctx)
        if isinstance(expr, TVar):
            return TBinaryOp(_P0, "!=", expr, TNilLit(_P0, _EMPTY_ANN), _EMPTY_ANN)
        return _make_named_call(
            "IsNil", [expr], [("negated", TBoolLit(_P0, True, _EMPTY_ANN))]
        )
    if _is_interface_type(expr_type):
        expr = _lower_expr(node, env, ctx)
        if isinstance(expr, TVar):
            return TBinaryOp(_P0, "!=", expr, TNilLit(_P0, _EMPTY_ANN), _EMPTY_ANN)
        return _make_named_call(
            "IsNil", [expr], [("negated", TBoolLit(_P0, True, _EMPTY_ANN))]
        )
    # Inline truthiness for known types
    if _is_type_dict(expr_type, ["string"]):
        expr = _lower_expr(node, env, ctx)
        return TBinaryOp(_P0, "!=", expr, TStringLit(_P0, "", _EMPTY_ANN), _EMPTY_ANN)
    if _is_type_dict(expr_type, ["int"]):
        expr = _lower_expr(node, env, ctx)
        return TBinaryOp(_P0, "!=", expr, TIntLit(_P0, 0, "0", _EMPTY_ANN), _EMPTY_ANN)
    if _is_type_dict(expr_type, ["float"]):
        expr = _lower_expr(node, env, ctx)
        return TBinaryOp(
            _P0, "!=", expr, TFloatLit(_P0, 0.0, "0.0", _EMPTY_ANN), _EMPTY_ANN
        )
    if isinstance(expr_type, TupleType):
        if len(expr_type.elements) > 0:
            return TBoolLit(_P0, True, _EMPTY_ANN)
        return TBoolLit(_P0, False, _EMPTY_ANN)
    if _is_type_dict(expr_type, ["bytes", "Slice", "Map", "Set"]):
        expr = _lower_expr(node, env, ctx)
        return TBinaryOp(
            _P0,
            "!=",
            _make_call("Len", [expr]),
            TIntLit(_P0, 0, "0", _EMPTY_ANN),
            _EMPTY_ANN,
        )
    # Comparison/BoolOp already return bool
    t = get_str(node, "_type")
    if t == "Compare" or t == "BoolOp":
        return _lower_expr(node, env, ctx)
    return _lower_expr(node, env, ctx)


# ---------------------------------------------------------------------------
# Statement lowering
# ---------------------------------------------------------------------------


def _lower_stmts(stmts: list[ASTNode], env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower a list of statements."""
    result: list[TStmt] = []
    i = 0
    while i < len(stmts):
        s = stmts[i]
        if isinstance(s, dict):
            lowered = _lower_stmt(s, env, ctx)
            j = 0
            while j < len(lowered):
                result.append(lowered[j])
                j += 1
        i += 1
    return result


def _lower_stmt(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower a single statement, may produce multiple IR statements."""
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
        return [TBreakStmt(_P0, _EMPTY_ANN)]
    if t == "Continue":
        return [TContinueStmt(_P0, _EMPTY_ANN)]
    if t == "Pass":
        return []
    if t == "Import" or t == "ImportFrom":
        return []
    return []


def _lower_return(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    val_jv = node.get("value")
    if val_jv is None or isinstance(val_jv, JNull):
        return [TReturnStmt(_P0, None, _EMPTY_ANN)]
    ret_type = env.return_type
    if isinstance(ret_type, PrimitiveType) and ret_type.kind == "void":
        return [TReturnStmt(_P0, None, _EMPTY_ANN)]
    if isinstance(val_jv, JDict):
        val = val_jv.entries
        # Expand list comprehension into for loop + return
        if _is_ast(val, "ListComp") or _is_ast(val, "GeneratorExp"):
            return _expand_listcomp(val, env, ctx)
        if _is_ast(val, "SetComp"):
            return _expand_setcomp(val, env, ctx)
        if _is_ast(val, "DictComp"):
            return _expand_dictcomp(val, env, ctx)
        expr = _lower_expr(val, env, ctx)
        return [TReturnStmt(_P0, expr, _EMPTY_ANN)]
    return [TReturnStmt(_P0, None, _EMPTY_ANN)]


def _lower_assign(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower an assignment statement."""
    targets = get_nodes(node, "targets")
    value_node = get_node(node, "value")
    if len(targets) == 0:
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
            return [TExprStmt(_P0, expr, _EMPTY_ANN)]
        value = _lower_expr(value_node, env, ctx)
        val_type = _infer_expr_type(value_node, env, ctx)
        if _is_type_dict(val_type, ["void"]):
            val_type = PrimitiveType("error")
        safe = _safe_name(name)
        ann = _name_ann(safe, name)
        if name not in env.declared:
            env.declared.add(name)
            env.var_types[name] = val_type
            ttype = _typenode_to_ttype(val_type)
            stmts: list[TStmt] = [TLetStmt(_P0, safe, ttype, value, ann)]
            stmts.extend(_method_side_effects(value_node, env, ctx))
            return stmts
        # Re-assignment
        target = TVar(_P0, safe, ann)
        stmts: list[TStmt] = [TAssignStmt(_P0, target, value, _EMPTY_ANN)]
        stmts.extend(_method_side_effects(value_node, env, ctx))
        return stmts
    # Attribute assignment: obj.field = expr
    if _is_ast(target_node, "Attribute"):
        attr = get_str(target_node, "attr")
        obj_node = get_node(target_node, "value")
        obj = _lower_expr(obj_node, env, ctx)
        target = TFieldAccess(_P0, obj, attr, _EMPTY_ANN)
        value = _lower_expr(value_node, env, ctx)
        return [TAssignStmt(_P0, target, value, _EMPTY_ANN)]
    # Subscript assignment: xs[i] = expr
    if _is_ast(target_node, "Subscript"):
        obj_node = get_node(target_node, "value")
        slice_node = get_node(target_node, "slice")
        obj = _lower_expr(obj_node, env, ctx)
        # Slice assignment: xs[a:b] = ys → ReplaceSlice(xs, a, b, ys)
        if _is_ast(slice_node, "Slice"):
            lower_jv = slice_node.get("lower")
            upper_jv = slice_node.get("upper")
            low: TExpr
            high: TExpr
            if isinstance(lower_jv, JDict):
                low = _lower_expr(lower_jv.entries, env, ctx)
            else:
                low = TIntLit(_P0, 0, "0", _EMPTY_ANN)
            if isinstance(upper_jv, JDict):
                high = _lower_expr(upper_jv.entries, env, ctx)
            else:
                high = _make_call("Len", [obj])
            value = _lower_expr(value_node, env, ctx)
            call = _make_call("ReplaceSlice", [obj, low, high, value])
            return [TExprStmt(_P0, call, _EMPTY_ANN)]
        idx = _lower_expr(slice_node, env, ctx)
        target = TIndex(_P0, obj, idx, _EMPTY_ANN)
        value = _lower_expr(value_node, env, ctx)
        return [TAssignStmt(_P0, target, value, _EMPTY_ANN)]
    return []


def _lower_tuple_assign(
    target_node: ASTNode, value_node: ASTNode, env: _Env, ctx: _LowerCtx
) -> list[TStmt]:
    """Lower tuple unpacking assignment: a, b = expr."""
    elts = get_nodes(target_node, "elts")
    # Special case: a, b = divmod(x, y) → DivMod(x, y)
    if _is_ast(value_node, "Call"):
        vfunc = get_node(value_node, "func")
        if _is_ast(vfunc, "Name") and get_str(vfunc, "id") == "divmod":
            vargs = get_nodes(value_node, "args")
            lowered_args: list[TExpr] = []
            arg_types: list[TypeNode | None] = []
            ai = 0
            while ai < len(vargs):
                a = vargs[ai]
                if isinstance(a, dict):
                    at = _infer_expr_type(a, env, ctx)
                    la = _lower_expr(a, env, ctx)
                    if _is_type_dict(at, ["bool"]):
                        la = _bool_to_int(la)
                        at = INT_TYPE
                    lowered_args.append(la)
                    arg_types.append(at)
                ai += 1
            use_float = False
            for at in arg_types:
                if _is_type_dict(at, ["float"]):
                    use_float = True
            if use_float:
                fa: list[TExpr] = []
                for la2 in lowered_args:
                    fa.append(la2)
                a_expr = fa[0] if len(fa) > 0 else lowered_args[0]
                b_expr = fa[1] if len(fa) > 1 else lowered_args[1]
                value = TTupleLit(
                    _P0,
                    [
                        _make_call("FloorDiv", [a_expr, b_expr]),
                        _make_call("PythonMod", [a_expr, b_expr]),
                    ],
                    _EMPTY_ANN,
                )
            else:
                value = _make_call("DivMod", lowered_args)
            result_kind = "float" if use_float else "int"
            stmts: list[TStmt] = []
            targets: list[TExpr] = []
            i = 0
            while i < len(elts):
                e = elts[i]
                if isinstance(e, dict) and _is_ast(e, "Name"):
                    name = get_str(e, "id")
                    safe = _safe_name(name)
                    ann = _name_ann(safe, name)
                    if name not in env.declared:
                        env.declared.add(name)
                        env.var_types[name] = PrimitiveType(result_kind)
                        prim = TPrimitive(_P0, result_kind)
                        init: TExpr = TIntLit(_P0, 0, "0", _EMPTY_ANN)
                        if use_float:
                            init = TFloatLit(_P0, 0.0, "0.0", _EMPTY_ANN)
                        stmts.append(TLetStmt(_P0, safe, prim, init, ann))
                    targets.append(TVar(_P0, safe, ann))
                i += 1
            stmts.append(TTupleAssignStmt(_P0, targets, value, _EMPTY_ANN))
            return stmts
    value = _lower_expr(value_node, env, ctx)
    val_type = _infer_expr_type(value_node, env, ctx)
    elem_types: list[TypeNode] = []
    if isinstance(val_type, TupleType):
        i2 = 0
        while i2 < len(val_type.elements):
            elem_types.append(val_type.elements[i2])
            i2 += 1
    stmts: list[TStmt] = []
    targets: list[TExpr] = []
    i = 0
    while i < len(elts):
        e = elts[i]
        if isinstance(e, dict) and _is_ast(e, "Name"):
            name = get_str(e, "id")
            safe = _safe_name(name)
            ann = _name_ann(safe, name)
            if name not in env.declared:
                env.declared.add(name)
                et = elem_types[i] if i < len(elem_types) else INT_TYPE
                env.var_types[name] = et
                ttype = _typenode_to_ttype(et)
                init = _default_value_for_type(et)
                stmts.append(TLetStmt(_P0, safe, ttype, init, ann))
            targets.append(TVar(_P0, safe, ann))
        i += 1
    stmts.append(TTupleAssignStmt(_P0, targets, value, _EMPTY_ANN))
    return stmts


def _lower_ann_assign(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower an annotated assignment: x: int = 10."""
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
    if ann_str != "":
        errors: list[str] = []
        type_dict = py_type_to_type_dict(ann_str, ctx.known_classes, [], 0, 0)
    ttype = _typenode_to_ttype(type_dict)
    if isinstance(ttype, TPrimitive) and ttype.kind == "void":
        ttype = TPrimitive(_P0, "nil")
    env.declared.add(name)
    env.var_types[name] = type_dict
    safe = _safe_name(name)
    ann = _name_ann(safe, name)
    val: TExpr | None = None
    stmts: list[TStmt] = []
    if isinstance(value_jv, JDict):
        value_node = value_jv.entries
        # void-returning function assigned to optional → call + nil
        val_type = _infer_expr_type(value_node, env, ctx)
        if _is_type_dict(val_type, ["void"]) and isinstance(type_dict, OptionalType):
            val = _lower_expr(value_node, env, ctx)
            stmts.append(TExprStmt(_P0, val, _EMPTY_ANN))
            val = TNilLit(_P0, _EMPTY_ANN)
        elif _is_variadic_tuple(type_dict) and _is_ast(value_node, "Tuple"):
            val = _lower_list_from_tuple(value_node, env, ctx)
        elif _is_single_elem_tuple(type_dict) and _is_ast(value_node, "Tuple"):
            # tuple[int] = (42,) → list[int] = [42]
            val = _lower_list_from_tuple(value_node, env, ctx)
        elif _is_set_of_genexpr(value_node):
            # set(genexpr) → expand inline
            genexpr = get_nodes(value_node, "args")[0]
            if already_declared:
                target = TVar(_P0, safe, ann)
                stmts.append(
                    TAssignStmt(_P0, target, _make_call("Set", []), _EMPTY_ANN)
                )
            else:
                stmts.append(TLetStmt(_P0, safe, ttype, _make_call("Set", []), ann))
            stmts.extend(_expand_genexpr_to_set_add(safe, genexpr, env, ctx))
            return stmts
        elif _is_map_type(type_dict) and _is_ast(value_node, "Dict"):
            # dict literal — convert bool keys to int when target type is int
            val = _lower_dict_literal_typed(value_node, type_dict, env, ctx)
        else:
            val = _lower_expr(value_node, env, ctx)
        if already_declared:
            target = TVar(_P0, safe, ann)
            stmts.append(TAssignStmt(_P0, target, val, _EMPTY_ANN))
        else:
            stmts.append(TLetStmt(_P0, safe, ttype, val, ann))
        stmts.extend(_method_side_effects(value_node, env, ctx))
        return stmts
    if not already_declared:
        stmts.append(TLetStmt(_P0, safe, ttype, val, ann))
    return stmts


def _lower_aug_assign(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower augmented assignment: x += 1."""
    target_node = get_node(node, "target")
    op_node = get_node(node, "op")
    value_node = get_node(node, "value")
    op_type = get_str(op_node, "_type")
    # list/string/bytes += other → target = Concat(target, other)
    if op_type == "Add":
        target_type = _infer_expr_type(target_node, env, ctx)
        if target_type is not None and _is_type_dict(target_type, ["string", "bytes"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(
                    _P0, target, _make_call("Concat", [target, value]), _EMPTY_ANN
                )
            ]
        if target_type is not None and _is_type_dict(target_type, ["Slice"]):
            target = _lower_expr(target_node, env, ctx)
            other = _lower_extend_arg(value_node, env, ctx)
            return [
                TAssignStmt(
                    _P0, target, _make_call("Concat", [target, other]), _EMPTY_ANN
                )
            ]
        if target_type is not None and _is_type_dict(target_type, ["Tuple"]):
            target = _lower_expr(target_node, env, ctx)
            other = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(
                    _P0, target, _make_call("Concat", [target, other]), _EMPTY_ANN
                )
            ]
    # dict |= other → dict = Merge(dict, other)
    # set |= other → set = Union(set, other)
    if op_type == "BitOr":
        target_type = _infer_expr_type(target_node, env, ctx)
        if target_type is not None and _is_type_dict(target_type, ["Map"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(
                    _P0, target, _make_call("Merge", [target, value]), _EMPTY_ANN
                )
            ]
        if target_type is not None and _is_type_dict(target_type, ["Set"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(
                    _P0, target, _make_call("Union", [target, value]), _EMPTY_ANN
                )
            ]
    # set &= other → set = Intersection(set, other)
    if op_type == "BitAnd":
        target_type = _infer_expr_type(target_node, env, ctx)
        if target_type is not None and _is_type_dict(target_type, ["Set"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(
                    _P0, target, _make_call("Intersection", [target, value]), _EMPTY_ANN
                )
            ]
    # set -= other → set = Difference(set, other)
    if op_type == "Sub":
        target_type = _infer_expr_type(target_node, env, ctx)
        if target_type is not None and _is_type_dict(target_type, ["Set"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(
                    _P0, target, _make_call("Difference", [target, value]), _EMPTY_ANN
                )
            ]
    # tuple *= n → target = Repeat(target, n)
    if op_type == "Mult":
        target_type = _infer_expr_type(target_node, env, ctx)
        if target_type is not None and _is_type_dict(target_type, ["Tuple"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            return [
                TAssignStmt(
                    _P0, target, _make_call("Repeat", [target, value]), _EMPTY_ANN
                )
            ]
    # set ^= other → set = Difference(Union(set, other), Intersection(set, other))
    if op_type == "BitXor":
        target_type = _infer_expr_type(target_node, env, ctx)
        if target_type is not None and _is_type_dict(target_type, ["Set"]):
            target = _lower_expr(target_node, env, ctx)
            value = _lower_expr(value_node, env, ctx)
            u = _make_call("Union", [target, value])
            inter = _make_call("Intersection", [target, value])
            return [
                TAssignStmt(
                    _P0, target, _make_call("Difference", [u, inter]), _EMPTY_ANN
                )
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
    return [TOpAssignStmt(_P0, target, op_str, value, _EMPTY_ANN)]


def _scan_assign_targets(
    nodes: list[ASTNode], env: _Env, ctx: _LowerCtx
) -> list[tuple[str, TypeNode]]:
    """Scan AST nodes for first-assigned names not yet in env.declared."""
    result: list[tuple[str, TypeNode]] = []
    seen: set[str] = set()
    stack: list[ASTNode] = []
    i = len(nodes) - 1
    while i >= 0:
        stack.append(nodes[i])
        i -= 1
    while len(stack) > 0:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        t = get_str(node, "_type")
        if t == "Assign":
            targets = get_nodes(node, "targets")
            if len(targets) > 0:
                tgt = targets[0]
                if isinstance(tgt, dict) and _is_ast(tgt, "Name"):
                    name = get_str(tgt, "id")
                    if name not in env.declared and name not in seen and name != "_":
                        value_node = get_node(node, "value")
                        val_type = _infer_expr_type(value_node, env, ctx)
                        if _is_type_dict(val_type, ["void"]):
                            val_type = PrimitiveType("error")
                        result.append((name, val_type))
                        seen.add(name)
                elif isinstance(tgt, dict) and _is_ast(tgt, "Tuple"):
                    elts = get_nodes(tgt, "elts")
                    ei = 0
                    while ei < len(elts):
                        e = elts[ei]
                        if isinstance(e, dict) and _is_ast(e, "Name"):
                            ename = get_str(e, "id")
                            if (
                                ename not in env.declared
                                and ename not in seen
                                and ename != "_"
                            ):
                                result.append((ename, PrimitiveType("error")))
                                seen.add(ename)
                        ei += 1
        elif t == "AnnAssign":
            tgt = get_node(node, "target")
            if isinstance(tgt, dict) and _is_ast(tgt, "Name"):
                name = get_str(tgt, "id")
                if name not in env.declared and name not in seen and name != "_":
                    ann_jv = node.get("annotation")
                    ann_str = ""
                    if isinstance(ann_jv, JDict):
                        ann_str = annotation_to_str(ann_jv.entries)
                    td: TypeNode = VOID_TYPE
                    if ann_str != "":
                        td = py_type_to_type_dict(ann_str, ctx.known_classes, [], 0, 0)
                    if _is_type_dict(td, ["void"]):
                        value_node = get_node(node, "value")
                        td = _infer_expr_type(value_node, env, ctx)
                    if _is_type_dict(td, ["void"]):
                        td = PrimitiveType("error")
                    result.append((name, td))
                    seen.add(name)
        elif t == "If":
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
        elif t == "While":
            body = get_nodes(node, "body")
            j = len(body) - 1
            while j >= 0:
                stack.append(body[j])
                j -= 1
        elif t == "For":
            body = get_nodes(node, "body")
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
                if isinstance(h, dict):
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


def _lower_if(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower an if statement, detecting isinstance chains for match."""
    test = get_node(node, "test")
    body = get_nodes(node, "body")
    orelse = get_nodes(node, "orelse")
    # Check for isinstance chain → match statement
    isinstance_result = _extract_isinstance_chain(node)
    if isinstance_result is not None:
        chain, else_body_nodes = isinstance_result
        return _lower_isinstance_chain(chain, else_body_nodes, env, ctx)
    # Hoist variables first-assigned inside branches
    all_branch_nodes: list[ASTNode] = []
    bi = 0
    while bi < len(body):
        all_branch_nodes.append(body[bi])
        bi += 1
    bi = 0
    while bi < len(orelse):
        all_branch_nodes.append(orelse[bi])
        bi += 1
    hoisted = _scan_assign_targets(all_branch_nodes, env, ctx)
    pre_stmts: list[TStmt] = []
    _emit_hoisted_lets(hoisted, env, pre_stmts)
    cond = _lower_as_bool(test, env, ctx)
    then_body = _lower_stmts(body, env, ctx)
    else_body: list[TStmt] | None = None
    if len(orelse) > 0:
        else_body = _lower_stmts(orelse, env, ctx)
    pre_stmts.append(TIfStmt(_P0, cond, then_body, else_body, _EMPTY_ANN))
    return pre_stmts


def _extract_isinstance_chain(
    node: ASTNode,
) -> tuple[list[tuple[str, str, list[ASTNode]]], list[ASTNode] | None] | None:
    """Extract isinstance chain from if/elif. Returns (cases, else_body) or None."""
    test = get_node(node, "test")
    if not _is_isinstance_call(test):
        return None
    var_name = _isinstance_var(test)
    if var_name == "":
        return None
    type_name = _isinstance_type(test)
    body = get_nodes(node, "body")
    result: list[tuple[str, str, list[ASTNode]]] = [(var_name, type_name, body)]
    orelse = get_nodes(node, "orelse")
    # Check if elif is also isinstance on same var
    if len(orelse) == 1 and isinstance(orelse[0], dict) and _is_ast(orelse[0], "If"):
        next_node = orelse[0]
        next_test = get_node(next_node, "test")
        if _is_isinstance_call(next_test) and _isinstance_var(next_test) == var_name:
            rest = _extract_isinstance_chain(next_node)
            if rest is not None:
                rest_cases, rest_else = rest
                i = 0
                while i < len(rest_cases):
                    result.append(rest_cases[i])
                    i += 1
                return (result, rest_else)
    # Trailing else (non-isinstance orelse)
    else_body: list[ASTNode] | None = None
    if len(orelse) > 0:
        else_body = orelse
    return (result, else_body)


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


def _lower_isinstance_chain(
    chain: list[tuple[str, str, list[ASTNode]]],
    else_body_nodes: list[ASTNode] | None,
    env: _Env,
    ctx: _LowerCtx,
) -> list[TStmt]:
    """Lower isinstance chain to a match statement."""
    if len(chain) == 0:
        return []
    # Hoist variables first-assigned inside branches
    all_body_nodes: list[ASTNode] = []
    i = 0
    while i < len(chain):
        _, _, body_stmts = chain[i]
        j = 0
        while j < len(body_stmts):
            all_body_nodes.append(body_stmts[j])
            j += 1
        i += 1
    if else_body_nodes is not None:
        j = 0
        while j < len(else_body_nodes):
            all_body_nodes.append(else_body_nodes[j])
            j += 1
    hoisted = _scan_assign_targets(all_body_nodes, env, ctx)
    pre_stmts: list[TStmt] = []
    _emit_hoisted_lets(hoisted, env, pre_stmts)
    var_name = chain[0][0]
    sv = _safe_name(var_name)
    expr = TVar(_P0, sv, _name_ann(sv, var_name))
    cases: list[TMatchCase] = []
    i = 0
    while i < len(chain):
        _, type_name, body_stmts = chain[i]
        binding_name = type_name[0].lower() + type_name[1:] if type_name else type_name
        if binding_name in env.declared:
            suffix = 2
            while binding_name + str(suffix) in env.declared:
                suffix += 1
            binding_name = binding_name + str(suffix)
        env.declared.add(binding_name)
        # Create narrowed env for the case body
        case_env = env.copy()
        case_env.var_types[var_name] = PointerType(StructRef(type_name))
        case_body = _lower_stmts(body_stmts, case_env, ctx)
        pattern = TPatternType(
            _P0, binding_name, TIdentType(_P0, type_name), _EMPTY_ANN
        )
        cases.append(TMatchCase(_P0, pattern, case_body, _EMPTY_ANN))
        i += 1
    default: TDefault | None = None
    if else_body_nodes is not None and len(else_body_nodes) > 0:
        else_stmts = _lower_stmts(else_body_nodes, env, ctx)
        default = TDefault(_P0, None, else_stmts, _EMPTY_ANN)
    else:
        default = TDefault(_P0, None, [], _EMPTY_ANN)
    pre_stmts.append(TMatchStmt(_P0, expr, cases, default, _EMPTY_ANN))
    return pre_stmts


def _lower_while(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    test = get_node(node, "test")
    body = get_nodes(node, "body")
    hoisted = _scan_assign_targets(body, env, ctx)
    pre_stmts: list[TStmt] = []
    _emit_hoisted_lets(hoisted, env, pre_stmts)
    cond = _lower_as_bool(test, env, ctx)
    stmts = _lower_stmts(body, env, ctx)
    pre_stmts.append(TWhileStmt(_P0, cond, stmts, _EMPTY_ANN))
    return pre_stmts


def _lower_for(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower a for statement."""
    target_node = get_node(node, "target")
    iter_node = get_node(node, "iter")
    body = get_nodes(node, "body")
    hoisted = _scan_assign_targets(body, env, ctx)
    pre_stmts: list[TStmt] = []
    _emit_hoisted_lets(hoisted, env, pre_stmts)
    # range() → TRange
    if _is_ast(iter_node, "Call"):
        func = get_node(iter_node, "func")
        if _is_ast(func, "Name") and get_str(func, "id") == "range":
            result = _lower_for_range(target_node, iter_node, body, env, ctx)
            ri = 0
            while ri < len(result):
                pre_stmts.append(result[ri])
                ri += 1
            return pre_stmts
    # enumerate() → indexed for
    if _is_ast(iter_node, "Call"):
        func = get_node(iter_node, "func")
        if _is_ast(func, "Name") and get_str(func, "id") == "enumerate":
            result = _lower_for_enumerate(target_node, iter_node, body, env, ctx)
            ri = 0
            while ri < len(result):
                pre_stmts.append(result[ri])
                ri += 1
            return pre_stmts
    # dict.items() → for k, v in d
    if _is_ast(iter_node, "Call"):
        func = get_node(iter_node, "func")
        if _is_ast(func, "Attribute") and get_str(func, "attr") == "items":
            obj_node = get_node(func, "value")
            iter_expr = _lower_expr(obj_node, env, ctx)
            binding, b_ann = _extract_binding(target_node)
            body_stmts = _lower_stmts(body, env, ctx)
            pre_stmts.append(TForStmt(_P0, binding, iter_expr, body_stmts, b_ann))
            return pre_stmts
    # Tuple iteration: for x in t → for x in [t.0, t.1, ...]
    iter_type = _infer_expr_type(iter_node, env, ctx)
    if isinstance(iter_type, TupleType) and not iter_type.variadic:
        elems = iter_type.elements
        if len(elems) > 0:
            iter_lowered = _lower_expr(iter_node, env, ctx)
            items: list[TExpr] = []
            j = 0
            while j < len(elems):
                items.append(TTupleAccess(_P0, iter_lowered, j, _EMPTY_ANN))
                j += 1
            binding, b_ann = _extract_binding(target_node)
            body_stmts = _lower_stmts(body, env, ctx)
            list_expr = TListLit(_P0, items, _EMPTY_ANN)
            pre_stmts.append(TForStmt(_P0, binding, list_expr, body_stmts, b_ann))
            return pre_stmts
    # Regular iteration: for x in xs
    binding, b_ann = _extract_binding(target_node)
    iter_expr = _lower_expr(iter_node, env, ctx)
    body_stmts = _lower_stmts(body, env, ctx)
    pre_stmts.append(TForStmt(_P0, binding, iter_expr, body_stmts, b_ann))
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
        i = 0
        while i < len(elts):
            e = elts[i]
            if isinstance(e, dict) and _is_ast(e, "Name"):
                orig = get_str(e, "id")
                safe = _safe_name(orig)
                names.append(safe)
                if safe != orig:
                    ann["name.original." + safe] = orig
            i += 1
        return (names, ann)
    return (["_"], _EMPTY_ANN)


def _lower_for_range(
    target_node: ASTNode,
    iter_node: ASTNode,
    body: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> list[TStmt]:
    """Lower for i in range(...)."""
    args = get_nodes(iter_node, "args")
    binding, b_ann = _extract_binding(target_node)
    range_args: list[TExpr] = []
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, dict):
            range_args.append(_lower_expr(a, env, ctx))
        i += 1
    body_stmts = _lower_stmts(body, env, ctx)
    return [TForStmt(_P0, binding, TRange(_P0, range_args, {}), body_stmts, b_ann)]


def _lower_for_enumerate(
    target_node: ASTNode,
    iter_node: ASTNode,
    body: list[ASTNode],
    env: _Env,
    ctx: _LowerCtx,
) -> list[TStmt]:
    """Lower for i, x in enumerate(xs)."""
    args = get_nodes(iter_node, "args")
    if len(args) == 0:
        return []
    inner = args[0]
    if not isinstance(inner, dict):
        return []
    binding, b_ann = _extract_binding(target_node)
    inner_type = _infer_expr_type(inner, env, ctx)
    # Enumerate over fixed-size tuple → enumerate over list of accesses
    if isinstance(inner_type, TupleType) and not inner_type.variadic:
        elems = inner_type.elements
        if len(elems) > 0:
            iter_lowered = _lower_expr(inner, env, ctx)
            items_list: list[TExpr] = []
            j = 0
            while j < len(elems):
                items_list.append(TTupleAccess(_P0, iter_lowered, j, _EMPTY_ANN))
                j += 1
            body_stmts = _lower_stmts(body, env, ctx)
            return [
                TForStmt(
                    _P0,
                    binding,
                    TListLit(_P0, items_list, _EMPTY_ANN),
                    body_stmts,
                    b_ann,
                )
            ]
    iter_expr = _lower_expr(inner, env, ctx)
    # For enumerate over strings, change last binding to "ch"
    if _is_type_dict(inner_type, ["string"]) and len(binding) == 2:
        binding = [binding[0], "ch"]
    body_stmts = _lower_stmts(body, env, ctx)
    return [TForStmt(_P0, binding, iter_expr, body_stmts, b_ann)]


def _lower_extend_arg(arg_node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Lower an argument to list.extend() or list +=, handling range/string."""
    # extend(range(...)) → RangeList(...)
    if _is_ast(arg_node, "Call"):
        rfunc = get_node(arg_node, "func")
        if _is_ast(rfunc, "Name") and get_str(rfunc, "id") == "range":
            rargs = get_nodes(arg_node, "args")
            if len(rargs) == 1 and isinstance(rargs[0], dict):
                end = _lower_expr(rargs[0], env, ctx)
                return _make_call(
                    "RangeList",
                    [
                        TIntLit(_P0, 0, "0", _EMPTY_ANN),
                        end,
                        TIntLit(_P0, 1, "1", _EMPTY_ANN),
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
                    "RangeList",
                    [start, end, TIntLit(_P0, 1, "1", _EMPTY_ANN)],
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
                return _make_call("RangeList", [start, end, step])
    # extend("string") → ["s", "t", "r", ...]
    arg_type = _infer_expr_type(arg_node, env, ctx)
    if _is_type_dict(arg_type, ["string"]) and _is_ast(arg_node, "Constant"):
        s_jv = arg_node.get("value")
        if isinstance(s_jv, JStr):
            chars: list[TExpr] = [TStringLit(_P0, c, _EMPTY_ANN) for c in s_jv.value]
            return TListLit(_P0, chars, _EMPTY_ANN)
    return _lower_expr(arg_node, env, ctx)


def _ensure_set_expr(arg_node: ASTNode, env: _Env, ctx: _LowerCtx) -> TExpr:
    """Ensure an expression produces a set, wrapping in SetFromList if needed."""
    arg_type = _infer_expr_type(arg_node, env, ctx)
    if _is_type_dict(arg_type, ["Set"]):
        return _lower_expr(arg_node, env, ctx)
    # String → SetFromList(Chars(s))
    if _is_type_dict(arg_type, ["string"]):
        if _is_ast(arg_node, "Constant"):
            s_jv = arg_node.get("value")
            if isinstance(s_jv, JStr):
                chars: list[TExpr] = [
                    TStringLit(_P0, c, _EMPTY_ANN) for c in s_jv.value
                ]
                return _make_call("SetFromList", [TListLit(_P0, chars, _EMPTY_ANN)])
        return _make_call(
            "SetFromList", [_make_call("Chars", [_lower_expr(arg_node, env, ctx)])]
        )
    # Map → SetFromList(Keys(map))
    if _is_type_dict(arg_type, ["Map"]):
        return _make_call(
            "SetFromList", [_make_call("Keys", [_lower_expr(arg_node, env, ctx)])]
        )
    # List or other → SetFromList(list)
    return _make_call("SetFromList", [_lower_expr(arg_node, env, ctx)])


def _lower_expr_stmt(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower an expression statement."""
    value = get_node(node, "value")
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
                return [
                    TAssignStmt(_P0, obj, TListLit(_P0, [], _EMPTY_ANN), _EMPTY_ANN)
                ]
            # list.reverse() → xs = Reversed(xs)
            if _is_type_dict(obj_type, ["Slice"]) and method == "reverse":
                obj = _lower_expr(obj_node, env, ctx)
                return [
                    TAssignStmt(_P0, obj, _make_call("Reversed", [obj]), _EMPTY_ANN)
                ]
            # list.sort() → xs = Sorted(xs)
            if _is_type_dict(obj_type, ["Slice"]) and method == "sort":
                obj = _lower_expr(obj_node, env, ctx)
                return [TAssignStmt(_P0, obj, _make_call("Sorted", [obj]), _EMPTY_ANN)]
            # list.extend(other) → xs = Concat(xs, other)
            if _is_type_dict(obj_type, ["Slice"]) and method == "extend":
                vargs = get_nodes(value, "args")
                if len(vargs) > 0 and isinstance(vargs[0], dict):
                    obj = _lower_expr(obj_node, env, ctx)
                    other_expr = _lower_extend_arg(vargs[0], env, ctx)
                    return [
                        TAssignStmt(
                            _P0,
                            obj,
                            _make_call("Concat", [obj, other_expr]),
                            _EMPTY_ANN,
                        )
                    ]
            # dict.clear() → d = Map()
            if _is_type_dict(obj_type, ["Map"]) and method == "clear":
                obj = _lower_expr(obj_node, env, ctx)
                return [TAssignStmt(_P0, obj, _make_call("Map", []), _EMPTY_ANN)]
            # dict.update(other) → d = Merge(d, other)
            if _is_type_dict(obj_type, ["Map"]) and method == "update":
                vargs = get_nodes(value, "args")
                if len(vargs) > 0 and isinstance(vargs[0], dict):
                    obj = _lower_expr(obj_node, env, ctx)
                    other = _lower_expr(vargs[0], env, ctx)
                    return [
                        TAssignStmt(
                            _P0, obj, _make_call("Merge", [obj, other]), _EMPTY_ANN
                        )
                    ]
            # set.update(other, ...) → s = Union(s, SetFromList(other))
            if _is_type_dict(obj_type, ["Set"]) and method == "update":
                vargs = get_nodes(value, "args")
                if len(vargs) > 0:
                    obj = _lower_expr(obj_node, env, ctx)
                    result: TExpr = obj
                    vi = 0
                    while vi < len(vargs):
                        va = vargs[vi]
                        if isinstance(va, dict):
                            other = _ensure_set_expr(va, env, ctx)
                            result = _make_call("Union", [result, other])
                        vi += 1
                    return [TAssignStmt(_P0, obj, result, _EMPTY_ANN)]
            # set.clear() → s = Set()
            if _is_type_dict(obj_type, ["Set"]) and method == "clear":
                obj = _lower_expr(obj_node, env, ctx)
                return [TAssignStmt(_P0, obj, _make_call("Set", []), _EMPTY_ANN)]
    expr = _lower_expr(value, env, ctx)
    return [TExprStmt(_P0, expr, _EMPTY_ANN)]


def _lower_try(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower a try/except statement."""
    body = get_nodes(node, "body")
    handlers = get_nodes(node, "handlers")
    finalbody = get_nodes(node, "finalbody")
    body_stmts = _lower_stmts(body, env, ctx)
    catches: list[TCatch] = []
    i = 0
    while i < len(handlers):
        h = handlers[i]
        if isinstance(h, dict):
            catch_name = get_str(h, "name")
            if catch_name == "":
                catch_name = "e"
            exc_type_jv = h.get("type")
            exc_types: list[TType] = []
            if isinstance(exc_type_jv, JDict):
                exc_type_node = exc_type_jv.entries
                if _is_ast(exc_type_node, "Name"):
                    tname = get_str(exc_type_node, "id")
                    if tname == "AssertionError":
                        tname = "AssertError"
                    if tname != "Exception" and tname != "BaseException":
                        exc_types.append(TIdentType(_P0, tname))
            catch_body = _lower_stmts(get_nodes(h, "body"), env, ctx)
            sc = _safe_name(catch_name)
            catches.append(
                TCatch(_P0, sc, exc_types, catch_body, _name_ann(sc, catch_name))
            )
        i += 1
    finally_body: list[TStmt] | None = None
    if len(finalbody) > 0:
        finally_body = _lower_stmts(finalbody, env, ctx)
    return [TTryStmt(_P0, body_stmts, catches, finally_body, _EMPTY_ANN)]


def _lower_raise(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower a raise statement."""
    exc_jv = node.get("exc")
    if isinstance(exc_jv, JDict):
        expr = _lower_expr(exc_jv.entries, env, ctx)
        return [TThrowStmt(_P0, expr, _EMPTY_ANN)]
    return [TThrowStmt(_P0, TVar(_P0, "e", _EMPTY_ANN), _EMPTY_ANN)]


def _lower_assert(node: ASTNode, env: _Env, ctx: _LowerCtx) -> list[TStmt]:
    """Lower an assert statement."""
    test = get_node(node, "test")
    msg_jv = node.get("msg")
    cond = _lower_as_bool(test, env, ctx)
    args: list[TExpr] = [cond]
    if isinstance(msg_jv, JDict):
        args.append(_lower_expr(msg_jv.entries, env, ctx))
    call = _make_call("Assert", args)
    return [TExprStmt(_P0, call, _EMPTY_ANN)]


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
    name = get_str(node, "name")
    if is_entry_point:
        name = "Main"
    # Get params and return type from signatures
    func_info = ctx.sig_result.functions.get(get_str(node, "name"))
    params: list[TParam] = []
    func_env = env.copy()
    if func_info is not None:
        i = 0
        while i < len(func_info.params):
            p = func_info.params[i]
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
            ttype = _typenode_to_ttype(p.typ)
            sp = _safe_name(p.name)
            params.append(TParam(_P0, sp, ttype, _name_ann(sp, p.name), p.has_default))
            func_env.var_types[p.name] = p.typ
            func_env.declared.add(p.name)
            i += 1
        func_env.return_type = func_info.return_type
    if is_entry_point:
        func_env.return_type = VOID_TYPE
    ret_type: TType = TPrimitive(_P0, "void")
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
        ret_type = _typenode_to_ttype(func_info.return_type)
    body_nodes = get_nodes(node, "body")
    body = _lower_stmts(body_nodes, func_env, ctx)
    return TFnDecl(_P0, name, params, ret_type, body, _EMPTY_ANN)


def _build_method(
    node: ASTNode,
    class_name: str,
    env: _Env,
    ctx: _LowerCtx,
) -> TFnDecl:
    """Build a TFnDecl from a method definition."""
    name = get_str(node, "name")
    # Get method signature
    class_methods = ctx.sig_result.methods.get(class_name, {})
    func_info = class_methods.get(name)
    params: list[TParam] = []
    func_env = env.copy()
    # Add self param
    self_type: TypeNode = PointerType(StructRef(class_name))
    func_env.var_types["self"] = self_type
    func_env.declared.add("self")
    params.append(TParam(_P0, "self", None, _EMPTY_ANN))
    if func_info is not None:
        i = 0
        while i < len(func_info.params):
            p = func_info.params[i]
            if p.name != "self":
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
                ttype = _typenode_to_ttype(p.typ)
                sp = _safe_name(p.name)
                params.append(
                    TParam(_P0, sp, ttype, _name_ann(sp, p.name), p.has_default)
                )
                func_env.var_types[p.name] = p.typ
                func_env.declared.add(p.name)
            i += 1
        func_env.return_type = func_info.return_type
    ret_type: TType = TPrimitive(_P0, "void")
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
        ret_type = _typenode_to_ttype(func_info.return_type)
    body_nodes = get_nodes(node, "body")
    body = _lower_stmts(body_nodes, func_env, ctx)
    return TFnDecl(_P0, name, params, ret_type, body, _EMPTY_ANN)


def _collect_ancestor_fields(
    name: str, ctx: _LowerCtx
) -> list[tuple[str, TType, bool]]:
    """Walk ancestors root-to-child, collecting fields from non-root ancestors."""
    chain: list[str] = []
    cur = name
    while True:
        ancs = ctx.hier_result.ancestors.get(cur)
        if ancs is None or len(ancs) == 0:
            break
        chain.append(ancs[0])
        cur = ancs[0]
    # Reverse so we go root→child
    chain.reverse()
    result: list[tuple[str, TType, bool]] = []
    seen: set[str] = set()
    i = 0
    while i < len(chain):
        anc = chain[i]
        anc_info = ctx.field_result.classes.get(anc)
        if anc_info is not None:
            akeys: list[str] = []
            if anc_info.init_params:
                j = 0
                while j < len(anc_info.init_params):
                    p = anc_info.init_params[j]
                    field_name = anc_info.param_to_field.get(p, p)
                    if field_name not in seen:
                        akeys.append(field_name)
                    j += 1
            for k in anc_info.fields:
                if k not in seen and k not in akeys:
                    akeys.append(k)
            j = 0
            while j < len(akeys):
                fname = akeys[j]
                finfo = anc_info.fields.get(fname)
                if finfo is not None:
                    result.append(
                        (fname, _typenode_to_ttype(finfo.typ), finfo.has_default)
                    )
                    seen.add(fname)
                j += 1
        i += 1
    return result


def _build_struct(
    node: ASTNode,
    ctx: _LowerCtx,
) -> TDecl | None:
    """Build a TStructDecl or TInterfaceDecl from a ClassDef node."""
    name = get_str(node, "name")
    # Check if this is a hierarchy root → interface
    if ctx.hier_result.is_hierarchy_root(name):
        ann: Ann = {}
        bases_list = ctx.hier_result.ancestors.get(name)
        if bases_list is not None and len(bases_list) > 0:
            parent_root = ctx.hier_result.root_of(bases_list[0])
            if parent_root is not None:
                ann = {"_parent_interface": parent_root}
        return TInterfaceDecl(_P0, name, ann)
    # Get bases
    bases = get_nodes(node, "bases")
    parent: str | None = None
    is_exception = False
    i = 0
    while i < len(bases):
        b = bases[i]
        if isinstance(b, dict) and _is_ast(b, "Name"):
            base_name = get_str(b, "id")
            if base_name == "Exception":
                is_exception = True
            elif base_name in ctx.known_classes:
                parent = base_name
        i += 1
    # Also check hierarchy result
    if ctx.hier_result.is_exception(name):
        is_exception = True
    if ctx.hier_result.is_node(name):
        parent = ctx.hier_result.root_of(name)
    # Build fields
    fields: list[TFieldDecl] = []
    cls_info = ctx.field_result.classes.get(name)
    if cls_info is not None:
        if is_exception and len(cls_info.fields) == 0:
            fields.append(TFieldDecl(_P0, "message", TPrimitive(_P0, "string")))
        else:
            # Collect inherited fields from ancestors
            ancestor_fields = _collect_ancestor_fields(name, ctx)
            inherited_field_names: set[str] = set()
            af_i = 0
            while af_i < len(ancestor_fields):
                af_name, af_type, af_has_default = ancestor_fields[af_i]
                fields.append(TFieldDecl(_P0, af_name, af_type, af_has_default))
                inherited_field_names.add(af_name)
                af_i += 1
            # Build own field keys: init_params first, then remaining fields
            seen: set[str] = set(inherited_field_names)
            fkeys: list[str] = []
            if cls_info.init_params:
                j = 0
                while j < len(cls_info.init_params):
                    p = cls_info.init_params[j]
                    field_name = cls_info.param_to_field.get(p, p)
                    if field_name not in seen:
                        fkeys.append(field_name)
                        seen.add(field_name)
                    j += 1
            for k in cls_info.fields:
                if k not in seen:
                    fkeys.append(k)
                    seen.add(k)
            j = 0
            while j < len(fkeys):
                fname = fkeys[j]
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
                    ftype = _typenode_to_ttype(finfo.typ)
                    fields.append(TFieldDecl(_P0, fname, ftype, finfo.has_default))
                j += 1
    # Build methods
    methods: list[TFnDecl] = []
    body = get_nodes(node, "body")
    env = _Env()
    j = 0
    while j < len(body):
        item = body[j]
        if isinstance(item, dict) and _is_ast(item, "FunctionDef"):
            mname = get_str(item, "name")
            if mname != "__init__":
                methods.append(_build_method(item, name, env, ctx))
        j += 1
    return TStructDecl(_P0, name, parent, fields, methods, _EMPTY_ANN)


def _build_constants(body: list[ASTNode], ctx: _LowerCtx) -> list[TModuleItem]:
    """Extract module-level and class-level constants."""
    result: list[TModuleItem] = []
    i = 0
    while i < len(body):
        node = body[i]
        if not isinstance(node, dict):
            i += 1
            continue
        # Module-level ALL_CAPS assignments
        if _is_ast(node, "Assign"):
            targets = get_nodes(node, "targets")
            if len(targets) > 0:
                t = targets[0]
                if isinstance(t, dict) and _is_ast(t, "Name"):
                    name = get_str(t, "id")
                    if name == name.upper() and name != "_" and len(name) > 1:
                        value_node = get_node(node, "value")
                        val_type = _infer_expr_type(value_node, _Env(), ctx)
                        if _is_type_dict(val_type, ["void"]):
                            val_type = PrimitiveType("error")
                        ttype = _typenode_to_ttype(val_type)
                        value = _lower_expr(value_node, _Env(), ctx)
                        result.append(TLetStmt(_P0, name, ttype, value, _EMPTY_ANN))
        # Module-level ALL_CAPS annotated assignments
        if _is_ast(node, "AnnAssign"):
            target = get_node(node, "target")
            if isinstance(target, dict) and _is_ast(target, "Name"):
                name = get_str(target, "id")
                if name == name.upper() and name != "_" and len(name) > 1:
                    ann_jv = node.get("annotation")
                    ann_str = ""
                    if isinstance(ann_jv, JDict):
                        ann_str = annotation_to_str(ann_jv.entries)
                    type_dict: TypeNode = VOID_TYPE
                    if ann_str != "":
                        type_dict = py_type_to_type_dict(
                            ann_str, ctx.known_classes, [], 0, 0
                        )
                    if _is_type_dict(type_dict, ["void"]):
                        value_node = get_node(node, "value")
                        type_dict = _infer_expr_type(value_node, _Env(), ctx)
                    if _is_type_dict(type_dict, ["void"]):
                        type_dict = PrimitiveType("error")
                    ttype = _typenode_to_ttype(type_dict)
                    value_node = get_node(node, "value")
                    value = _lower_expr(value_node, _Env(), ctx)
                    result.append(TLetStmt(_P0, name, ttype, value, _EMPTY_ANN))
        # Class-level constants
        if _is_ast(node, "ClassDef"):
            class_name = get_str(node, "name")
            class_body = get_nodes(node, "body")
            j = 0
            while j < len(class_body):
                item = class_body[j]
                if isinstance(item, dict) and _is_ast(item, "Assign"):
                    targets = get_nodes(item, "targets")
                    if len(targets) > 0:
                        t = targets[0]
                        if isinstance(t, dict) and _is_ast(t, "Name"):
                            fname = get_str(t, "id")
                            if fname == fname.upper() and len(fname) > 1:
                                value_node = get_node(item, "value")
                                val_type = _infer_expr_type(value_node, _Env(), ctx)
                                if _is_type_dict(val_type, ["void"]):
                                    val_type = PrimitiveType("error")
                                ttype = _typenode_to_ttype(val_type)
                                value = _lower_expr(value_node, _Env(), ctx)
                                const_name = class_name + "_" + fname
                                result.append(
                                    TLetStmt(_P0, const_name, ttype, value, _EMPTY_ANN)
                                )
                if isinstance(item, dict) and _is_ast(item, "AnnAssign"):
                    target = get_node(item, "target")
                    if isinstance(target, dict) and _is_ast(target, "Name"):
                        fname = get_str(target, "id")
                        if fname == fname.upper() and len(fname) > 1:
                            ann_jv = item.get("annotation")
                            ann_str = ""
                            if isinstance(ann_jv, JDict):
                                ann_str = annotation_to_str(ann_jv.entries)
                            c_type_dict: TypeNode = VOID_TYPE
                            if ann_str != "":
                                c_type_dict = py_type_to_type_dict(
                                    ann_str, ctx.known_classes, [], 0, 0
                                )
                            if _is_type_dict(c_type_dict, ["void"]):
                                value_node = get_node(item, "value")
                                c_type_dict = _infer_expr_type(value_node, _Env(), ctx)
                            if _is_type_dict(c_type_dict, ["void"]):
                                c_type_dict = PrimitiveType("error")
                            ttype = _typenode_to_ttype(c_type_dict)
                            value_node = get_node(item, "value")
                            value = _lower_expr(value_node, _Env(), ctx)
                            const_name = class_name + "_" + fname
                            result.append(
                                TLetStmt(_P0, const_name, ttype, value, _EMPTY_ANN)
                            )
                j += 1
        i += 1
    return result


def _detect_entry_point(body: list[ASTNode]) -> str | None:
    """Detect if __name__ == '__main__': main() pattern."""
    i = 0
    while i < len(body):
        node = body[i]
        if isinstance(node, dict) and _is_ast(node, "If"):
            test = get_node(node, "test")
            if _is_name_main_check(test):
                if_body = get_nodes(node, "body")
                if len(if_body) > 0:
                    first = if_body[0]
                    if isinstance(first, dict) and _is_ast(first, "Expr"):
                        val = get_node(first, "value")
                        if _is_ast(val, "Call"):
                            func = get_node(val, "func")
                            if _is_ast(func, "Name"):
                                return get_str(func, "id")
                return "main"
        i += 1
    return None


def _is_name_main_check(node: ASTNode) -> bool:
    """Check if node is __name__ == '__main__'."""
    if not _is_ast(node, "Compare"):
        return False
    left = get_node(node, "left")
    if not _is_ast(left, "Name") or get_str(left, "id") != "__name__":
        return False
    comparators = get_nodes(node, "comparators")
    if len(comparators) < 1:
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
    # Build constants first
    constants = _build_constants(body, ctx)
    i = 0
    while i < len(constants):
        decls.append(constants[i])
        i += 1
    # Build structs/interfaces
    i = 0
    while i < len(body):
        node = body[i]
        if isinstance(node, dict) and _is_ast(node, "ClassDef"):
            decl = _build_struct(node, ctx)
            if decl is not None:
                decls.append(decl)
        i += 1
    # Build functions
    env = _Env()
    i = 0
    while i < len(body):
        node = body[i]
        if isinstance(node, dict) and _is_ast(node, "FunctionDef"):
            fname = get_str(node, "name")
            is_entry = entry_point_func is not None and fname == entry_point_func
            decls.append(_build_function(node, env, ctx, is_entry))
        i += 1
    return TModule(decls)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lower(
    tree: ASTNode,
    sig_result: SignatureResult,
    field_result: FieldResult,
    hier_result: HierarchyResult,
    known_classes: set[str],
    class_bases: dict[str, list[str]],
    source: str,
) -> tuple[TModule | None, list[LoweringError]]:
    """Lower the Python AST to Taytsh IR.

    Returns (module, errors). If errors is non-empty, module may be None.
    """
    ctx = _LowerCtx(
        sig_result, field_result, hier_result, known_classes, class_bases, source
    )
    module = _build_module(tree, ctx)
    return (module, ctx.errors)
