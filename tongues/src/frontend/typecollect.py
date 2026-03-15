"""Type metadata collection: signatures and fields."""

from __future__ import annotations

from .types import (
    TypeNode,
    PrimitiveType,
    SliceType,
    ByteArrayType,
    MapType,
    SetType,
    TupleType,
    OptionalType,
    UnionType,
    TypeGuardType,
    PointerType,
    StructRef,
    InterfaceRef,
    FuncType,
    NilLit,
    BoolLit,
    IntLit,
    FloatLit,
    StringLit,
    ListLit,
    MapLit,
    SetLit,
    TupleLit,
    typenode_to_dict,
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
    get_node,
    get_nodes,
    get_jlist,
    has_key,
)


# ---------------------------------------------------------------------------
# Shared AST helpers
# ---------------------------------------------------------------------------


def _is_type(node: ASTNode, type_names: list[str]) -> bool:
    """Check if node is one of the given AST types."""
    t = get_str(node, "_type")
    return t in type_names


def _dict_walk(node: ASTNode) -> list[ASTNode]:
    """Walk dict-based AST, returns list of all nodes."""
    result: list[ASTNode] = [node]
    for key in node:
        if not key.startswith("_"):
            value = node[key]
            if isinstance(value, JDict) and has_key(value.entries, "_type"):
                result = result + _dict_walk(value.entries)
            elif isinstance(value, JList):
                for item in value.items:
                    if isinstance(item, JDict) and has_key(item.entries, "_type"):
                        result = result + _dict_walk(item.entries)
    return result


# ---------------------------------------------------------------------------
# Signature data classes
# ---------------------------------------------------------------------------


class ParamInfo:
    """A single function/method parameter."""

    def __init__(
        self,
        name: str,
        typ: TypeNode,
        py_type: str,
        has_default: bool,
        default_value: TypeNode | None,
        modifier: str,
    ) -> None:
        self.name: str = name
        self.typ: TypeNode = typ
        self.py_type: str = py_type
        self.has_default: bool = has_default
        self.default_value: TypeNode | None = default_value
        self.modifier: str = modifier

    def to_dict(self) -> JsonValue:
        """Serialize to a JsonValue dict for test assertions."""
        dv: JsonValue = JNull()
        if self.default_value is not None:
            dv = typenode_to_dict(self.default_value)
        return JDict(
            {
                "name": JStr(self.name),
                "typ": typenode_to_dict(self.typ),
                "py_type": JStr(self.py_type),
                "has_default": JBool(self.has_default),
                "default_value": dv,
                "modifier": JStr(self.modifier),
            }
        )


class FuncInfo:
    """Signature information for a function or method."""

    def __init__(
        self,
        name: str,
        params: list[ParamInfo],
        return_type: TypeNode,
        return_py_type: str,
        is_method: bool,
        receiver_type: str,
    ) -> None:
        self.name: str = name
        self.params: list[ParamInfo] = params
        self.return_type: TypeNode = return_type
        self.return_py_type: str = return_py_type
        self.is_method: bool = is_method
        self.receiver_type: str = receiver_type

    def to_dict(self) -> JsonValue:
        """Serialize to a JsonValue dict for test assertions."""
        param_dicts: list[JsonValue] = []
        for param in self.params:
            param_dicts.append(param.to_dict())
        return JDict(
            {
                "params": JList(param_dicts),
                "return_type": typenode_to_dict(self.return_type),
                "return_py_type": JStr(self.return_py_type),
                "is_method": JBool(self.is_method),
                "receiver_type": JStr(self.receiver_type),
            }
        )


class TypeCollectError:
    """An error found during signature collection."""

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
            + ": [types] "
            + self.message
        )


class SignatureResult:
    """Result of signature collection."""

    def __init__(self) -> None:
        self.functions: dict[str, FuncInfo] = {}
        self.methods: dict[str, dict[str, FuncInfo]] = {}
        self.method_to_structs: dict[str, str] = {}
        self._errors: list[TypeCollectError] = []

    def add_error(self, lineno: int, col: int, message: str) -> None:
        self._errors.append(TypeCollectError(lineno, col, message))

    def errors(self) -> list[TypeCollectError]:
        return self._errors

    def to_dict(self) -> JsonValue:
        """Serialize to nested JsonValue dicts for test assertions."""
        funcs: dict[str, JsonValue] = {}
        for name in self.functions:
            funcs[name] = self.functions[name].to_dict()
        meths: dict[str, JsonValue] = {}
        for cname in self.methods:
            class_methods: dict[str, JsonValue] = {}
            for mname in self.methods[cname]:
                class_methods[mname] = self.methods[cname][mname].to_dict()
            meths[cname] = JDict(class_methods)
        result: dict[str, JsonValue] = {}
        if funcs:
            result["functions"] = JDict(funcs)
        if meths:
            result["methods"] = JDict(meths)
        return JDict(result)


# ---------------------------------------------------------------------------
# Annotation to string
# ---------------------------------------------------------------------------


def annotation_to_str(node: ASTNode | None) -> str:
    """Convert a type annotation AST node to its string representation."""
    if node is None:
        return ""
    node_t = get_str(node, "_type")
    if node_t == "Name":
        return get_str(node, "id")
    if node_t == "Constant":
        v = node.get("value")
        if v is None or isinstance(v, JNull):
            return "None"
        if isinstance(v, JStr):
            if v.value == "Ellipsis":
                return "..."
            return v.value
        if isinstance(v, JInt):
            return str(v.value)
        if isinstance(v, JFloat):
            return str(v.value)
        if isinstance(v, JBool):
            return str(v.value)
        return ""
    if node_t == "List":
        elts = get_nodes(node, "elts")
        parts: list[str] = []
        for elt in elts:
            parts.append(annotation_to_str(elt))
        return "[" + ", ".join(parts) + "]"
    if node_t == "Subscript":
        value_node = get_node(node, "value")
        base = annotation_to_str(value_node)
        slc = get_node(node, "slice")
        if get_str(slc, "_type") == "Tuple":
            elts = get_nodes(slc, "elts")
            parts: list[str] = []
            for elt in elts:
                parts.append(annotation_to_str(elt))
            return base + "[" + ", ".join(parts) + "]"
        return base + "[" + annotation_to_str(slc) + "]"
    if node_t == "BinOp":
        op = get_node(node, "op")
        if get_str(op, "_type") == "BitOr":
            left_node = get_node(node, "left")
            right_node = get_node(node, "right")
            left = annotation_to_str(left_node)
            right = annotation_to_str(right_node)
            return left + " | " + right
    if node_t == "Attribute":
        return get_str(node, "attr")
    return ""


# ---------------------------------------------------------------------------
# Type string parsing
# ---------------------------------------------------------------------------


def _find_bracket_end(s: str, start: int) -> int:
    """Find the matching ] for the [ at position start."""
    depth = 1
    i = start + 1
    while i < len(s):
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(s)


def _split_type_args(s: str) -> list[str]:
    """Split a comma-separated type argument string, respecting brackets."""
    result: list[str] = []
    depth = 0
    current: list[str] = []
    for c in s:
        if c == "[":
            depth += 1
            current.append(c)
        elif c == "]":
            depth -= 1
            current.append(c)
        elif c == "," and depth == 0:
            result.append("".join(current).strip())
            current: list[str] = []
        else:
            current.append(c)
    tail = "".join(current).strip()
    if tail:
        result.append(tail)
    return result


def _split_union_members(s: str) -> list[str]:
    """Split a union type string on ' | ', respecting brackets."""
    result: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "[":
            depth += 1
            current.append(c)
        elif c == "]":
            depth -= 1
            current.append(c)
        elif (
            c == " "
            and depth == 0
            and i + 2 < len(s)
            and s[i + 1] == "|"
            and s[i + 2] == " "
        ):
            result.append("".join(current).strip())
            current: list[str] = []
            i += 3
            continue
        else:
            current.append(c)
        i += 1
    tail = "".join(current).strip()
    if tail:
        result.append(tail)
    return result


# Type alias expansions, populated by collect_signatures()
_TYPE_ALIASES: dict[str, str] = {}
_EXPANDING_ALIASES: dict[str, bool] = {}

# Class base mappings, populated by collect_signatures()
_CLASS_BASES: dict[str, list[str]] = {}


# Primitive type mapping: Python name -> kind string
_PRIM_MAP: dict[str, str] = {
    "int": "int",
    "str": "string",
    "bool": "bool",
    "float": "float",
    "byte": "byte",
    "None": "void",
}

_BUILTIN_EXCEPTIONS: set[str] = {
    "Exception",
    "AssertionError",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "RuntimeError",
    "StopIteration",
    "ArithmeticError",
    "OverflowError",
    "ZeroDivisionError",
    "Base64Error",
    "JsonError",
}


def py_type_to_type_dict(
    py_type: str,
    known_classes: dict[str, str],
    errors: list[TypeCollectError],
    lineno: int,
    col: int,
) -> TypeNode:
    """Convert a Python type string to a TypeNode."""
    s = py_type.strip()
    if not s:
        return InterfaceRef("any")
    # Expand type aliases (with recursion guard)
    if s in _TYPE_ALIASES:
        if s in _EXPANDING_ALIASES:
            errors.append(
                TypeCollectError(lineno, col, "recursive type alias '" + s + "'")
            )
            return InterfaceRef("any")
        _EXPANDING_ALIASES[s] = True
        result = py_type_to_type_dict(
            _TYPE_ALIASES[s], known_classes, errors, lineno, col
        )
        _EXPANDING_ALIASES.pop(s)
        return result
    # Check for union (A | B) — only if the split produces multiple top-level members
    if " | " in s:
        members = _split_union_members(s)
        if len(members) > 1:
            return _resolve_union(members, known_classes, errors, lineno, col)
    # Check for subscript types: name[args]
    bracket = s.find("[")
    if bracket != -1:
        base = s[:bracket].strip()
        end = _find_bracket_end(s, bracket)
        inner = s[bracket + 1 : end].strip()
        return _resolve_subscript(base, inner, known_classes, errors, lineno, col)
    # Primitives
    if s in _PRIM_MAP:
        return PrimitiveType(_PRIM_MAP[s])
    # object -> any interface
    if s == "object":
        return InterfaceRef("any")
    # bytes -> Slice(byte)
    if s == "bytes":
        return SliceType(PrimitiveType("byte"))
    if s == "bytearray":
        return ByteArrayType(PrimitiveType("int"))
    # Bare collection types (no subscript) -> any element
    if s == "list":
        return SliceType(InterfaceRef("any"))
    if s == "dict":
        return MapType(InterfaceRef("any"), InterfaceRef("any"))
    if s == "set" or s == "frozenset":
        return SetType(InterfaceRef("any"))
    if s == "tuple":
        return TupleType([InterfaceRef("any")], True)
    # Known class -> Pointer(StructRef)
    canonical = known_classes.get(s)
    if canonical is not None:
        return PointerType(StructRef(canonical))
    # Builtin and lib exception classes
    if s in _BUILTIN_EXCEPTIONS:
        return PointerType(StructRef(s))
    errors.append(TypeCollectError(lineno, col, "unknown type '" + s + "'"))
    return InterfaceRef("any")


def _resolve_subscript(
    base: str,
    inner: str,
    known_classes: dict[str, str],
    errors: list[TypeCollectError],
    lineno: int,
    col: int,
) -> TypeNode:
    """Resolve a subscripted type like list[int], dict[str, int], etc."""
    args = _split_type_args(inner)
    if base == "list":
        if len(args) != 1:
            errors.append(
                TypeCollectError(
                    lineno,
                    col,
                    "list requires 1 type argument, got " + str(len(args)),
                )
            )
            return InterfaceRef("any")
        elem = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        return SliceType(elem)
    if base == "dict":
        if len(args) != 2:
            errors.append(
                TypeCollectError(
                    lineno,
                    col,
                    "dict requires 2 type arguments, got " + str(len(args)),
                )
            )
            return InterfaceRef("any")
        key = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        val = py_type_to_type_dict(args[1], known_classes, errors, lineno, col)
        return MapType(key, val)
    if base == "set" or base == "frozenset":
        if len(args) != 1:
            errors.append(
                TypeCollectError(
                    lineno,
                    col,
                    base + " requires 1 type argument, got " + str(len(args)),
                )
            )
            return InterfaceRef("any")
        elem = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        return SetType(elem)
    if base == "tuple":
        if not args:
            errors.append(
                TypeCollectError(lineno, col, "tuple requires at least 1 type argument")
            )
            return InterfaceRef("any")
        # Check for variadic tuple: tuple[T, ...]
        if len(args) == 2 and args[1] == "...":
            elem = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
            return TupleType([elem], True)
        elems: list[TypeNode] = []
        for arg in args:
            elems.append(py_type_to_type_dict(arg, known_classes, errors, lineno, col))
        return TupleType(elems, False)
    if base == "Optional":
        if len(args) != 1:
            errors.append(
                TypeCollectError(
                    lineno,
                    col,
                    "Optional requires 1 type argument, got " + str(len(args)),
                )
            )
            return InterfaceRef("any")
        inner_t = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        return OptionalType(inner_t)
    if base == "TypeGuard":
        if len(args) != 1:
            errors.append(
                TypeCollectError(
                    lineno,
                    col,
                    "TypeGuard requires exactly 1 type argument",
                )
            )
            return InterfaceRef("any")
        inner_t = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        return TypeGuardType(inner_t)
    if base == "Union":
        return _resolve_union(args, known_classes, errors, lineno, col)
    if base == "Callable":
        if len(args) != 2:
            errors.append(
                TypeCollectError(
                    lineno,
                    col,
                    "Callable requires 2 type arguments, got " + str(len(args)),
                )
            )
            return InterfaceRef("any")
        # First arg is a list of param types like [int, str]
        param_str = args[0].strip()
        param_types: list[TypeNode] = []
        if param_str.startswith("[") and param_str.endswith("]"):
            param_inner = param_str[1:-1].strip()
            if param_inner:
                param_parts = _split_type_args(param_inner)
                for part in param_parts:
                    param_types.append(
                        py_type_to_type_dict(part, known_classes, errors, lineno, col)
                    )
        ret = py_type_to_type_dict(args[1], known_classes, errors, lineno, col)
        return FuncType(param_types, ret)
    # Sequence/Iterable[T] -> Slice(T)
    if base == "Sequence" or base == "Iterable":
        if len(args) != 1:
            errors.append(
                TypeCollectError(
                    lineno,
                    col,
                    base + " requires 1 type argument, got " + str(len(args)),
                )
            )
            return InterfaceRef("any")
        elem = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        return SliceType(elem)
    # Mapping[K, V] -> Map(K, V)
    if base == "Mapping":
        if len(args) != 2:
            errors.append(
                TypeCollectError(
                    lineno,
                    col,
                    "Mapping requires 2 type arguments, got " + str(len(args)),
                )
            )
            return InterfaceRef("any")
        key = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        val = py_type_to_type_dict(args[1], known_classes, errors, lineno, col)
        return MapType(key, val)
    # Unknown subscript base
    errors.append(TypeCollectError(lineno, col, "unknown type '" + base + "'"))
    return InterfaceRef("any")


def _resolve_union(
    members: list[str],
    known_classes: dict[str, str],
    errors: list[TypeCollectError],
    lineno: int,
    col: int,
) -> TypeNode:
    """Resolve a union type from its member strings."""
    # Deduplicate
    unique: list[str] = []
    seen: set[str] = set()
    for member in members:
        m = member.strip()
        if m not in seen:
            unique.append(m)
            seen.add(m)
    # Single type after dedup
    if len(unique) == 1:
        return py_type_to_type_dict(unique[0], known_classes, errors, lineno, col)
    # Check for T | None -> Optional
    has_none = False
    non_none: list[str] = []
    for u in unique:
        if u == "None":
            has_none = True
        else:
            non_none.append(u)
    if has_none and len(non_none) == 1:
        inner = py_type_to_type_dict(non_none[0], known_classes, errors, lineno, col)
        return OptionalType(inner)
    if has_none and len(non_none) > 1:
        # Optional of a union — wrap the union in Optional
        inner = _resolve_non_none_union(non_none, known_classes, errors, lineno, col)
        return OptionalType(inner)
    # Non-None union -> InterfaceRef
    return _resolve_non_none_union(unique, known_classes, errors, lineno, col)


def _resolve_non_none_union(
    members: list[str],
    known_classes: dict[str, str],
    errors: list[TypeCollectError],
    lineno: int,
    col: int,
) -> TypeNode:
    """Resolve a union with no None members to a UnionType."""
    variants: list[TypeNode] = []
    for member in members:
        variants.append(
            py_type_to_type_dict(member, known_classes, errors, lineno, col)
        )
    if len(variants) == 1:
        return variants[0]
    return UnionType(variants)


# ---------------------------------------------------------------------------
# Default value lowering
# ---------------------------------------------------------------------------


def _lower_default(node: ASTNode) -> TypeNode | None:
    """Lower a default value AST node to a literal TypeNode."""
    if not isinstance(node, dict):
        return None
    t = get_str(node, "_type")
    if t == "Constant":
        v = node.get("value")
        if v is None or isinstance(v, JNull):
            return NilLit()
        if isinstance(v, JBool):
            return BoolLit(v.value)
        if isinstance(v, JInt):
            return IntLit(v.value)
        if isinstance(v, JFloat):
            return FloatLit(v.value)
        if isinstance(v, JStr):
            return StringLit(v.value)
    if t == "UnaryOp":
        op = get_node(node, "op")
        if get_str(op, "_type") == "USub":
            operand = get_node(node, "operand")
            if get_str(operand, "_type") == "Constant":
                v = operand.get("value")
                if isinstance(v, JInt):
                    return IntLit(-v.value)
                if isinstance(v, JFloat):
                    return FloatLit(-v.value)
    if t == "List":
        return ListLit([])
    if t == "Dict":
        return MapLit([])
    if t == "Set":
        return SetLit([])
    if t == "Tuple":
        return TupleLit([])
    return None


# ---------------------------------------------------------------------------
# Mutated parameter detection
# ---------------------------------------------------------------------------


def detect_mutated_params(node: ASTNode) -> set[str]:
    """Detect which parameters are mutated in the function body."""
    mutated: set[str] = set()
    args_node = get_node(node, "args")
    if not args_node:
        return mutated
    param_names: set[str] = set()
    posonlyargs = get_nodes(args_node, "posonlyargs")
    regular_args = get_nodes(args_node, "args")
    kwonlyargs = get_nodes(args_node, "kwonlyargs")
    for a in posonlyargs:
        name = get_str(a, "arg")
        if name and name != "self":
            param_names.add(name)
    for a in regular_args:
        name = get_str(a, "arg")
        if name and name != "self":
            param_names.add(name)
    for a in kwonlyargs:
        name = get_str(a, "arg")
        if name and name != "self":
            param_names.add(name)
    all_nodes = _dict_walk(node)
    for stmt in all_nodes:
        if _is_type(stmt, ["Expr"]):
            val = get_node(stmt, "value")
            if _is_type(val, ["Call"]):
                func = get_node(val, "func")
                if _is_type(func, ["Attribute"]):
                    attr = get_str(func, "attr")
                    if attr in ("append", "extend", "clear", "pop"):
                        obj = get_node(func, "value")
                        if _is_type(obj, ["Name"]):
                            obj_id = get_str(obj, "id")
                            if obj_id and obj_id in param_names:
                                mutated.add(obj_id)
        if _is_type(stmt, ["Assign"]):
            targets = get_nodes(stmt, "targets")
            for target in targets:
                if _is_type(target, ["Subscript"]):
                    obj = get_node(target, "value")
                    if _is_type(obj, ["Name"]):
                        obj_id = get_str(obj, "id")
                        if obj_id and obj_id in param_names:
                            mutated.add(obj_id)
    return mutated


# ---------------------------------------------------------------------------
# TypeNode helpers
# ---------------------------------------------------------------------------


def _is_slice_type(typ: TypeNode) -> bool:
    """Check if a TypeNode is a Slice (list) type."""
    return isinstance(typ, SliceType)


def _wrap_pointer(typ: TypeNode) -> TypeNode:
    """Wrap a type in a Pointer."""
    return PointerType(typ)


# ---------------------------------------------------------------------------
# Parameter and function extraction
# ---------------------------------------------------------------------------


def _make_param(
    arg: ASTNode,
    modifier: str,
    has_default: bool,
    default_node: ASTNode | None,
    mutated_params: set[str],
    known_classes: dict[str, str],
    errors: list[TypeCollectError],
    func_name: str,
) -> ParamInfo | None:
    """Build a ParamInfo from an AST arg node. Returns None on error."""
    param_name = get_str(arg, "arg")
    if not param_name:
        return None
    annotation = get_node(arg, "annotation")
    lineno = get_int(arg, "lineno")
    if not annotation:
        errors.append(
            TypeCollectError(
                lineno,
                0,
                "parameter '"
                + param_name
                + "' missing type annotation in "
                + func_name
                + "()",
            )
        )
        return None
    py_type = annotation_to_str(annotation)
    typ = py_type_to_type_dict(py_type, known_classes, errors, lineno, 0)
    if param_name in mutated_params and _is_slice_type(typ):
        typ = _wrap_pointer(typ)
    default_value: TypeNode | None = None
    if has_default and default_node is not None:
        default_value = _lower_default(default_node)
    return ParamInfo(
        name=param_name,
        typ=typ,
        py_type=py_type,
        has_default=has_default,
        default_value=default_value,
        modifier=modifier,
    )


def extract_func_info(
    node: ASTNode,
    known_classes: dict[str, str],
    errors: list[TypeCollectError],
    is_method: bool,
    receiver_type: str,
) -> FuncInfo | None:
    """Extract function signature information from a FunctionDef AST node."""
    func_name = get_str(node, "name")
    lineno = get_int(node, "lineno")
    mutated_params = detect_mutated_params(node)
    params: list[ParamInfo] = []
    args = get_node(node, "args")
    posonlyargs = get_nodes(args, "posonlyargs")
    regular_args = get_nodes(args, "args")
    kwonlyargs = get_nodes(args, "kwonlyargs")
    defaults = get_nodes(args, "defaults")
    kw_defaults = get_jlist(args, "kw_defaults")
    # Filter self from params
    non_self_posonly: list[ASTNode] = []
    for a in posonlyargs:
        if get_str(a, "arg") != "self":
            non_self_posonly.append(a)
    non_self_regular: list[ASTNode] = []
    for a in regular_args:
        if get_str(a, "arg") != "self":
            non_self_regular.append(a)
    n_positional = len(non_self_posonly) + len(non_self_regular)
    n_defaults = len(defaults)
    had_error = False
    for i, posonly_arg in enumerate(non_self_posonly):
        has_default = i >= n_positional - n_defaults
        default_node: ASTNode | None = None
        if has_default:
            default_idx = i - (n_positional - n_defaults)
            if default_idx >= 0 and default_idx < len(defaults):
                default_node = defaults[default_idx]
        p = _make_param(
            posonly_arg,
            "positional",
            has_default,
            default_node,
            mutated_params,
            known_classes,
            errors,
            func_name,
        )
        if p is not None:
            params.append(p)
        else:
            had_error = True
    for i, regular_arg in enumerate(non_self_regular):
        global_i = len(non_self_posonly) + i
        has_default = global_i >= n_positional - n_defaults
        default_node = None
        if has_default:
            default_idx = global_i - (n_positional - n_defaults)
            if default_idx >= 0 and default_idx < len(defaults):
                default_node = defaults[default_idx]
        p = _make_param(
            regular_arg,
            "pos_or_kw",
            has_default,
            default_node,
            mutated_params,
            known_classes,
            errors,
            func_name,
        )
        if p is not None:
            params.append(p)
        else:
            had_error = True
    for i, kw_arg in enumerate(kwonlyargs):
        has_default = False
        default_node = None
        if i < len(kw_defaults):
            kw_def = kw_defaults[i]
            if not isinstance(kw_def, JNull):
                has_default = True
                if isinstance(kw_def, JDict):
                    default_node = kw_def.entries
        p = _make_param(
            kw_arg,
            "keyword",
            has_default,
            default_node,
            mutated_params,
            known_classes,
            errors,
            func_name,
        )
        if p is not None:
            params.append(p)
        else:
            had_error = True
    returns = get_node(node, "returns")
    if not returns:
        if func_name == "__init__":
            returns_node: ASTNode = {
                "_type": JStr("Constant"),
                "value": JNull(),
            }
            py_return = annotation_to_str(returns_node)
            return_type = py_type_to_type_dict(
                py_return, known_classes, errors, lineno, 0
            )
        else:
            errors.append(
                TypeCollectError(
                    lineno,
                    0,
                    "function '" + func_name + "' missing return type annotation",
                )
            )
            return None
    else:
        py_return = annotation_to_str(returns)
        return_type = py_type_to_type_dict(py_return, known_classes, errors, lineno, 0)
    if had_error:
        return None
    return FuncInfo(
        name=func_name,
        params=params,
        return_type=return_type,
        return_py_type=py_return,
        is_method=is_method,
        receiver_type=receiver_type,
    )


# ---------------------------------------------------------------------------
# Signature collection entry point
# ---------------------------------------------------------------------------


# Methods excluded from method_to_structs dispatch mapping
_EXCLUDED_METHODS: set[str] = {
    "__init__",
    "__repr__",
    "to_sexp",
    "kind",
    "ToSexp",
    "GetKind",
}


def collect_signatures(
    tree: ASTNode,
    known_classes: dict[str, str],
    node_classes: set[str],
    type_aliases: dict[str, str] | None = None,
    class_bases: dict[str, list[str]] | None = None,
) -> SignatureResult:
    """Collect function and method signatures from the module AST."""
    while _TYPE_ALIASES:
        _TYPE_ALIASES.pop(list(_TYPE_ALIASES.keys())[0])
    while _EXPANDING_ALIASES:
        _EXPANDING_ALIASES.pop(list(_EXPANDING_ALIASES.keys())[0])
    while _CLASS_BASES:
        _CLASS_BASES.pop(list(_CLASS_BASES.keys())[0])
    if class_bases is not None:
        for cb_key in list(class_bases.keys()):
            _CLASS_BASES[cb_key] = class_bases[cb_key]
    if type_aliases is not None:
        for ta_key in list(type_aliases.keys()):
            _TYPE_ALIASES[ta_key] = type_aliases[ta_key]
    result = SignatureResult()
    body = get_nodes(tree, "body")
    for node in body:
        t = get_str(node, "_type")
        sf = get_str(node, "_source_file")
        if t == "FunctionDef":
            err_before = len(result._errors)
            info = extract_func_info(node, known_classes, result._errors, False, "")
            for err in result._errors[err_before:]:
                err.source_file = sf
            if info is not None:
                result.functions[info.name] = info
        elif t == "ClassDef":
            class_name = get_str(node, "name")
            class_body = get_nodes(node, "body")
            class_methods: dict[str, FuncInfo] = {}
            for stmt in class_body:
                if get_str(stmt, "_type") == "FunctionDef":
                    stmt_sf = get_str(stmt, "_source_file")
                    if not stmt_sf:
                        stmt_sf = sf
                    err_before = len(result._errors)
                    method_info = extract_func_info(
                        stmt, known_classes, result._errors, True, class_name
                    )
                    for err in result._errors[err_before:]:
                        err.source_file = stmt_sf
                    if method_info is not None:
                        class_methods[method_info.name] = method_info
            if class_methods:
                result.methods[class_name] = class_methods
            if class_name in node_classes:
                for mname in list(class_methods.keys()):
                    if mname not in _EXCLUDED_METHODS:
                        result.method_to_structs[mname] = class_name
    return result


# ---------------------------------------------------------------------------
# Field data classes
# ---------------------------------------------------------------------------


class FieldInfo:
    """A single class field."""

    def __init__(
        self,
        name: str,
        typ: TypeNode,
        py_name: str,
        has_default: bool,
        default: TypeNode | None,
    ) -> None:
        self.name: str = name
        self.typ: TypeNode = typ
        self.py_name: str = py_name
        self.has_default: bool = has_default
        self.default: TypeNode | None = default
        self.exclude_from_init: bool = False
        self.self_ref: bool = False

    def to_dict(self) -> JsonValue:
        """Serialize to a JsonValue dict for test assertions."""
        dv: JsonValue = JNull()
        if self.default is not None:
            dv = typenode_to_dict(self.default)
        return JDict(
            {
                "typ": typenode_to_dict(self.typ),
                "py_name": JStr(self.py_name),
                "has_default": JBool(self.has_default),
                "default": dv,
            }
        )


class ClassInfo:
    """Field and constructor info for a single class."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.fields: dict[str, FieldInfo] = {}
        self.field_order: list[str] = []
        self.init_params: list[str] = []
        self.param_defaults: dict[str, bool] = {}
        self.param_to_field: dict[str, str] = {}
        self.const_fields: dict[str, str] = {}
        self.is_dataclass: bool = False
        self.has_explicit_init: bool = False
        self.kw_only: bool = False
        self.needs_constructor: bool = False
        self.is_enum: bool = False
        self.enum_variants: list[str] = []

    def to_dict(self) -> JsonValue:
        """Serialize to a JsonValue dict for test assertions."""
        fields: dict[str, JsonValue] = {}
        fkeys = self.field_order if self.field_order else list(self.fields.keys())
        for fkey in fkeys:
            fields[fkey] = self.fields[fkey].to_dict()
        init_params_jv: list[JsonValue] = []
        for init_param in self.init_params:
            init_params_jv.append(JStr(init_param))
        ptf: dict[str, JsonValue] = {}
        for ptf_key in list(self.param_to_field.keys()):
            ptf[ptf_key] = JStr(self.param_to_field[ptf_key])
        cf: dict[str, JsonValue] = {}
        for cf_key in list(self.const_fields.keys()):
            cf[cf_key] = JStr(self.const_fields[cf_key])
        return JDict(
            {
                "fields": JDict(fields),
                "init_params": JList(init_params_jv),
                "param_to_field": JDict(ptf),
                "const_fields": JDict(cf),
                "is_dataclass": JBool(self.is_dataclass),
                "kw_only": JBool(self.kw_only),
                "needs_constructor": JBool(self.needs_constructor),
            }
        )


class FieldError:
    """An error found during field collection."""

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
            + ": [fields] "
            + self.message
        )


class FieldResult:
    """Result of field collection."""

    def __init__(self) -> None:
        self.classes: dict[str, ClassInfo] = {}
        self.field_to_structs: dict[str, list[str]] = {}
        self._errors: list[FieldError] = []

    def add_error(
        self, lineno: int, col: int, message: str, source_file: str = ""
    ) -> None:
        self._errors.append(FieldError(lineno, col, message, source_file))

    def errors(self) -> list[FieldError]:
        return self._errors

    def to_dict(self) -> JsonValue:
        """Serialize to nested JsonValue dicts for test assertions."""
        classes: dict[str, JsonValue] = {}
        for ckey in list(self.classes.keys()):
            classes[ckey] = self.classes[ckey].to_dict()
        return JDict({"classes": JDict(classes)})


# ---------------------------------------------------------------------------
# Field AST helpers
# ---------------------------------------------------------------------------


def _wrap_nodes(lst: list[ASTNode]) -> JList:
    """Wrap a list of ASTNode dicts into a JList of JDicts."""
    items: list[JsonValue] = []
    for item in lst:
        items.append(JDict(item))
    return JList(items)


# ---------------------------------------------------------------------------
# Field type helpers
# ---------------------------------------------------------------------------


def _value_refs_defaulted_param(
    node: ASTNode, param_has_default: dict[str, bool]
) -> bool:
    """Check if an expression tree references a parameter with a default."""
    if _is_type(node, ["Name"]):
        pname = get_str(node, "id")
        return param_has_default.get(pname, False)
    if _is_type(node, ["IfExp"]):
        body = get_node(node, "body")
        orelse = get_node(node, "orelse")
        if _value_refs_defaulted_param(body, param_has_default):
            return True
        if _value_refs_defaulted_param(orelse, param_has_default):
            return True
    return False


def _unwrap_field_type(typ: TypeNode) -> TypeNode:
    """Unwrap Pointer(StructRef(X)) -> StructRef(X) and Slice(byte) -> bytes."""
    if isinstance(typ, PointerType):
        if isinstance(typ.target, StructRef):
            return typ.target
    if isinstance(typ, SliceType):
        if isinstance(typ.element, PrimitiveType) and typ.element.kind == "byte":
            return PrimitiveType("bytes")
    return typ


def _type_kind_str(typ: TypeNode) -> str:
    """Short string for a type, used in error messages."""
    if isinstance(typ, PrimitiveType):
        if typ.kind == "string":
            return "str"
        return typ.kind
    if isinstance(typ, SliceType):
        return "list"
    if isinstance(typ, MapType):
        return "dict"
    if isinstance(typ, SetType):
        return "set"
    if isinstance(typ, TupleType):
        return "tuple"
    if isinstance(typ, OptionalType):
        return "Optional"
    if isinstance(typ, StructRef):
        return typ.name
    if isinstance(typ, InterfaceRef):
        return "interface"
    if isinstance(typ, FuncType):
        return "Callable"
    if isinstance(typ, PointerType):
        return "Pointer"
    return "unknown"


# ---------------------------------------------------------------------------
# Field default values
# ---------------------------------------------------------------------------


def _make_default_expr(node: ASTNode) -> TypeNode | None:
    """Convert a constant AST node to a default value TypeNode."""
    if not _is_type(node, ["Constant"]):
        return None
    v = node.get("value")
    if isinstance(v, JBool):
        return BoolLit(v.value)
    if isinstance(v, JInt):
        return IntLit(v.value)
    if isinstance(v, JStr):
        return StringLit(v.value)
    return None


# ---------------------------------------------------------------------------
# PascalCase -> kebab-case
# ---------------------------------------------------------------------------


def _pascal_to_kebab(name: str) -> str:
    """PascalCase to kebab-case: BinaryOp -> binary-op."""
    result: list[str] = []
    i = 0
    while i < len(name):
        ch = name[i]
        if ch.isupper() and i > 0:
            prev = name[i - 1]
            if prev.islower() or prev.isdigit():
                result.append("-")
            elif prev.isupper() and i + 1 < len(name) and name[i + 1].islower():
                result.append("-")
        result.append(ch)
        i += 1
    return "".join(result).lower()


# ---------------------------------------------------------------------------
# Dataclass detection
# ---------------------------------------------------------------------------


def _is_dataclass_class(node: ASTNode) -> tuple[bool, bool]:
    """Check decorator_list for @dataclass. Returns (is_dataclass, kw_only)."""
    dec_list = get_nodes(node, "decorator_list")
    for dec in dec_list:
        if _is_type(dec, ["Name"]) and get_str(dec, "id") == "dataclass":
            return (True, False)
        if _is_type(dec, ["Call"]):
            func = get_node(dec, "func")
            if _is_type(func, ["Name"]) and get_str(func, "id") == "dataclass":
                kw_only = False
                keywords = get_nodes(dec, "keywords")
                for kw in keywords:
                    if get_str(kw, "arg") == "kw_only":
                        val = get_node(kw, "value")
                        if _is_type(val, ["Constant"]):
                            v = val.get("value")
                            if isinstance(v, JBool) and v.value:
                                kw_only = True
                return (True, kw_only)
    return (False, False)


# ---------------------------------------------------------------------------
# field(default_factory=...) detection
# ---------------------------------------------------------------------------


def _call_refs_self(node: ASTNode) -> bool:
    """True if node is a Call whose arguments include 'self'."""
    if not _is_type(node, ["Call"]):
        return False
    for arg in get_nodes(node, "args"):
        if _is_type(arg, ["Name"]) and get_str(arg, "id") == "self":
            return True
    return False


def _is_field_call_default_factory(node: ASTNode) -> bool:
    """Detect field(default_factory=...)."""
    if not _is_type(node, ["Call"]):
        return False
    func = get_node(node, "func")
    if not (_is_type(func, ["Name"]) and get_str(func, "id") == "field"):
        return False
    keywords = get_nodes(node, "keywords")
    for kw in keywords:
        if get_str(kw, "arg") == "default_factory":
            return True
    return False


def _is_field_call_init_false(node: ASTNode) -> bool:
    """Detect field(init=False)."""
    if not _is_type(node, ["Call"]):
        return False
    func = get_node(node, "func")
    if not (_is_type(func, ["Name"]) and get_str(func, "id") == "field"):
        return False
    keywords = get_nodes(node, "keywords")
    for kw in keywords:
        if get_str(kw, "arg") == "init":
            val = get_node(kw, "value")
            if val and _is_type(val, ["Constant"]):
                v = val.get("value")
                if isinstance(v, JBool) and not v.value:
                    return True
    return False


# ---------------------------------------------------------------------------
# Conditional field assignment check
# ---------------------------------------------------------------------------


def _check_no_field_assign_in_block(block: list[ASTNode]) -> str | None:
    """Return field name if self.x = ... found inside block, else None."""
    wrapper: ASTNode = {"_type": JStr("_wrapper"), "body": _wrap_nodes(block)}
    all_nodes = _dict_walk(wrapper)
    for stmt in all_nodes:
        if _is_type(stmt, ["Assign"]):
            targets = get_nodes(stmt, "targets")
            for tgt in targets:
                if _is_type(tgt, ["Attribute"]):
                    val_node = get_node(tgt, "value")
                    if (
                        _is_type(val_node, ["Name"])
                        and get_str(val_node, "id") == "self"
                    ):
                        attr = get_str(tgt, "attr")
                        if attr:
                            return attr
        if _is_type(stmt, ["AnnAssign"]):
            target = get_node(stmt, "target")
            if _is_type(target, ["Attribute"]):
                val_node = get_node(target, "value")
                if _is_type(val_node, ["Name"]) and get_str(val_node, "id") == "self":
                    attr = get_str(target, "attr")
                    if attr:
                        return attr
    return None


# ---------------------------------------------------------------------------
# Outside-init field check
# ---------------------------------------------------------------------------


def _check_no_new_fields_outside_init(
    func: ASTNode, known_fields: set[str]
) -> str | None:
    """Return field name if method introduces new self.x not in known_fields."""
    all_nodes = _dict_walk(func)
    for stmt in all_nodes:
        if _is_type(stmt, ["Assign"]):
            targets = get_nodes(stmt, "targets")
            for tgt in targets:
                if _is_type(tgt, ["Attribute"]):
                    val_node = get_node(tgt, "value")
                    if (
                        _is_type(val_node, ["Name"])
                        and get_str(val_node, "id") == "self"
                    ):
                        fname = get_str(tgt, "attr")
                        if fname and fname not in known_fields:
                            return fname
        if _is_type(stmt, ["AnnAssign"]):
            target = get_node(stmt, "target")
            if _is_type(target, ["Attribute"]):
                val_node = get_node(target, "value")
                if _is_type(val_node, ["Name"]) and get_str(val_node, "id") == "self":
                    fname = get_str(target, "attr")
                    if fname and fname not in known_fields:
                        return fname
    return None


# ---------------------------------------------------------------------------
# Type inference from expressions
# ---------------------------------------------------------------------------


def _infer_type_from_value(
    node: ASTNode,
    param_types: dict[str, str],
    known_classes: dict[str, str],
    func_return_types: dict[str, str],
    errors: list[FieldError],
    lineno: int,
) -> TypeNode | None:
    """Infer a TypeNode from an expression AST node. Returns None if cannot infer."""
    if not isinstance(node, dict):
        return None
    t = get_str(node, "_type")
    if t == "Constant":
        v = node.get("value")
        if v is None or isinstance(v, JNull):
            return PrimitiveType("void")
        if isinstance(v, JBool):
            return PrimitiveType("bool")
        if isinstance(v, JInt):
            return PrimitiveType("int")
        if isinstance(v, JFloat):
            return PrimitiveType("float")
        if isinstance(v, JStr):
            return PrimitiveType("string")
    if t == "Name":
        name = get_str(node, "id")
        if name and name in param_types:
            py_type = param_types[name]
            sig_errors: list[TypeCollectError] = []
            typ = py_type_to_type_dict(py_type, known_classes, sig_errors, lineno, 0)
            return _unwrap_field_type(typ)
        if name:
            errors.append(
                FieldError(lineno, 0, "cannot infer type for field from '" + name + "'")
            )
        return None
    if t == "Call":
        func = get_node(node, "func")
        if _is_type(func, ["Name"]):
            func_name = get_str(func, "id")
            if func_name:
                if func_name in known_classes:
                    return StructRef(func_name)
                if func_name in func_return_types:
                    py_ret = func_return_types[func_name]
                    sig_errors: list[TypeCollectError] = []
                    typ = py_type_to_type_dict(
                        py_ret, known_classes, sig_errors, lineno, 0
                    )
                    return _unwrap_field_type(typ)
        return None
    if t == "BinOp":
        left = get_node(node, "left")
        right = get_node(node, "right")
        if has_key(left, "_type") and has_key(right, "_type"):
            left_t = _infer_type_from_value(
                left, param_types, known_classes, func_return_types, errors, lineno
            )
            if left_t is not None:
                return left_t
        return None
    if t == "UnaryOp":
        operand = get_node(node, "operand")
        if has_key(operand, "_type"):
            return _infer_type_from_value(
                operand, param_types, known_classes, func_return_types, errors, lineno
            )
    if t == "List":
        return SliceType(InterfaceRef("any"))
    if t == "Dict":
        return MapType(InterfaceRef("any"), InterfaceRef("any"))
    if t == "Set":
        return SetType(InterfaceRef("any"))
    if t == "Tuple":
        return TupleType([], False)
    return None


# ---------------------------------------------------------------------------
# Init field collection
# ---------------------------------------------------------------------------


def _collect_init_fields(
    init: ASTNode,
    info: ClassInfo,
    known_classes: dict[str, str],
    func_return_types: dict[str, str],
    errors: list[FieldError],
) -> None:
    """Collect fields assigned in __init__."""
    args = get_node(init, "args")
    param_types: dict[str, str] = {}
    args_list = get_nodes(args, "args")
    for arg in args_list:
        arg_name = get_str(arg, "arg")
        if arg_name and arg_name != "self":
            info.init_params.append(arg_name)
            annotation = get_node(arg, "annotation")
            if annotation:
                param_types[arg_name] = annotation_to_str(annotation)
    posonlyargs = get_nodes(args, "posonlyargs")
    for arg in posonlyargs:
        arg_name = get_str(arg, "arg")
        if arg_name and arg_name != "self":
            info.init_params.append(arg_name)
            annotation = get_node(arg, "annotation")
            if annotation:
                param_types[arg_name] = annotation_to_str(annotation)
    kwonlyargs = get_nodes(args, "kwonlyargs")
    for arg in kwonlyargs:
        arg_name = get_str(arg, "arg")
        if arg_name and arg_name != "self":
            info.init_params.append(arg_name)
            annotation = get_node(arg, "annotation")
            if annotation:
                param_types[arg_name] = annotation_to_str(annotation)
    # Build param_has_default map from defaults/kw_defaults
    param_has_default: dict[str, bool] = {}
    defaults = get_nodes(args, "defaults")
    kw_defaults = get_jlist(args, "kw_defaults")
    non_self_posonly: list[ASTNode] = []
    for a in posonlyargs:
        if get_str(a, "arg") != "self":
            non_self_posonly.append(a)
    non_self_regular: list[ASTNode] = []
    for a in args_list:
        if get_str(a, "arg") != "self":
            non_self_regular.append(a)
    n_positional = len(non_self_posonly) + len(non_self_regular)
    n_defaults = len(defaults)
    for i, posonly_arg in enumerate(non_self_posonly):
        pname = get_str(posonly_arg, "arg")
        if pname:
            param_has_default[pname] = i >= n_positional - n_defaults
    for i, regular_arg in enumerate(non_self_regular):
        pname = get_str(regular_arg, "arg")
        if pname:
            idx = len(non_self_posonly) + i
            param_has_default[pname] = idx >= n_positional - n_defaults
    for i, kw_arg in enumerate(kwonlyargs):
        pname = get_str(kw_arg, "arg")
        if pname and pname != "self":
            has_kw_def = i < len(kw_defaults) and not isinstance(kw_defaults[i], JNull)
            param_has_default[pname] = has_kw_def
    info.param_defaults = param_has_default
    has_computed_init = False
    body = get_nodes(init, "body")
    lineno = get_int(init, "lineno")
    for stmt in body:
        stmt_lineno = get_int(stmt, "lineno")
        if stmt_lineno == 0:
            stmt_lineno = lineno
        if _is_type(stmt, ["If", "For", "While"]):
            body_stmts = get_nodes(stmt, "body")
            orelse_stmts = get_nodes(stmt, "orelse")
            bad = _check_no_field_assign_in_block(body_stmts)
            if bad is None:
                bad = _check_no_field_assign_in_block(orelse_stmts)
            if bad is not None:
                errors.append(
                    FieldError(
                        stmt_lineno,
                        0,
                        "conditional field assignment not allowed: " + bad,
                    )
                )
                return
            continue
        if _is_type(stmt, ["AnnAssign"]):
            target = get_node(stmt, "target")
            if _is_type(target, ["Attribute"]):
                val_node = get_node(target, "value")
                if _is_type(val_node, ["Name"]) and get_str(val_node, "id") == "self":
                    field_name = get_str(target, "attr")
                    if field_name:
                        ann = get_node(stmt, "annotation")
                        py_type = annotation_to_str(ann)
                        sig_errors: list[TypeCollectError] = []
                        typ = py_type_to_type_dict(
                            py_type, known_classes, sig_errors, stmt_lineno, 0
                        )
                        typ = _unwrap_field_type(typ)
                        if field_name in info.fields:
                            existing_kind = _type_kind_str(info.fields[field_name].typ)
                            new_kind = _type_kind_str(typ)
                            if existing_kind != new_kind:
                                errors.append(
                                    FieldError(
                                        stmt_lineno,
                                        0,
                                        "field '"
                                        + field_name
                                        + "' declared as "
                                        + existing_kind
                                        + " but assigned "
                                        + new_kind,
                                    )
                                )
                                return
                        else:
                            ann_has_default = False
                            ann_val = stmt.get("value")
                            if ann_val is not None and not isinstance(ann_val, JNull):
                                if isinstance(ann_val, JDict):
                                    value_node = ann_val.entries
                                else:
                                    value_node: ASTNode = {}
                                if (
                                    _is_type(value_node, ["Name"])
                                    and get_str(value_node, "id") in param_types
                                ):
                                    param_name = get_str(value_node, "id")
                                    ann_has_default = param_has_default.get(
                                        param_name, False
                                    )
                                    info.param_to_field[param_name] = field_name
                                else:
                                    ann_has_default = True
                            if field_name not in info.fields:
                                info.field_order.append(field_name)
                            finfo_new = FieldInfo(
                                name=field_name,
                                typ=typ,
                                py_name=field_name,
                                has_default=ann_has_default,
                                default=None,
                            )
                            if ann_has_default and ann_val is not None:
                                vn = (
                                    ann_val.entries
                                    if isinstance(ann_val, JDict)
                                    else {}
                                )
                                if _call_refs_self(vn):
                                    finfo_new.self_ref = True
                            info.fields[field_name] = finfo_new
                        ann_val2 = stmt.get("value")
                        if ann_val2 is not None and not isinstance(ann_val2, JNull):
                            if isinstance(ann_val2, JDict):
                                value_node = ann_val2.entries
                            else:
                                value_node: ASTNode = {}
                            if not (
                                _is_type(value_node, ["Name"])
                                and get_str(value_node, "id") in param_types
                            ):
                                has_computed_init = True
            continue
        if _is_type(stmt, ["Assign"]):
            targets = get_nodes(stmt, "targets")
            for tgt in targets:
                if _is_type(tgt, ["Attribute"]):
                    val_node = get_node(tgt, "value")
                    if (
                        _is_type(val_node, ["Name"])
                        and get_str(val_node, "id") == "self"
                    ):
                        field_name = get_str(tgt, "attr")
                        if field_name:
                            value = get_node(stmt, "value")
                            is_simple_param = (
                                _is_type(value, ["Name"])
                                and get_str(value, "id") != ""
                                and get_str(value, "id") in param_types
                            )
                            is_const_str = _is_type(value, ["Constant"]) and isinstance(
                                value.get("value"), JStr
                            )
                            if is_simple_param:
                                param_name = get_str(value, "id")
                                info.param_to_field[param_name] = field_name
                            elif is_const_str:
                                v = value.get("value")
                                if isinstance(v, JStr):
                                    info.const_fields[field_name] = v.value
                            else:
                                has_computed_init = True
                            if field_name not in info.fields:
                                if is_simple_param:
                                    param_name = get_str(value, "id")
                                    if param_name in param_types:
                                        py_type = param_types[param_name]
                                        sig_errors2: list[TypeCollectError] = []
                                        typ = py_type_to_type_dict(
                                            py_type,
                                            known_classes,
                                            sig_errors2,
                                            stmt_lineno,
                                            0,
                                        )
                                        typ = _unwrap_field_type(typ)
                                        if field_name not in info.fields:
                                            info.field_order.append(field_name)
                                        info.fields[field_name] = FieldInfo(
                                            name=field_name,
                                            typ=typ,
                                            py_name=field_name,
                                            has_default=param_has_default.get(
                                                param_name, False
                                            ),
                                            default=None,
                                        )
                                elif is_const_str:
                                    if field_name not in info.fields:
                                        info.field_order.append(field_name)
                                    info.fields[field_name] = FieldInfo(
                                        name=field_name,
                                        typ=PrimitiveType("string"),
                                        py_name=field_name,
                                        has_default=True,
                                        default=None,
                                    )
                                else:
                                    inferred_typ = _infer_type_from_value(
                                        value,
                                        param_types,
                                        known_classes,
                                        func_return_types,
                                        errors,
                                        stmt_lineno,
                                    )
                                    if inferred_typ is not None:
                                        if field_name not in info.fields:
                                            info.field_order.append(field_name)
                                        info.fields[field_name] = FieldInfo(
                                            name=field_name,
                                            typ=_unwrap_field_type(inferred_typ),
                                            py_name=field_name,
                                            has_default=True,
                                            default=None,
                                        )
                            elif not is_simple_param:
                                inferred = _infer_type_from_value(
                                    value,
                                    param_types,
                                    known_classes,
                                    func_return_types,
                                    errors,
                                    stmt_lineno,
                                )
                                if inferred is not None:
                                    unwrapped = _unwrap_field_type(inferred)
                                    existing_kind = _type_kind_str(
                                        info.fields[field_name].typ
                                    )
                                    new_kind = _type_kind_str(unwrapped)
                                    if existing_kind != new_kind:
                                        errors.append(
                                            FieldError(
                                                stmt_lineno,
                                                0,
                                                "field '"
                                                + field_name
                                                + "' declared as "
                                                + existing_kind
                                                + " but assigned "
                                                + new_kind,
                                            )
                                        )
                                        return
    if not info.init_params:
        for fk in info.fields:
            info.fields[fk].has_default = True
    if info.init_params:
        info.needs_constructor = True
    elif has_computed_init:
        info.needs_constructor = True


# ---------------------------------------------------------------------------
# Class field collection
# ---------------------------------------------------------------------------


def _collect_class_fields(
    node: ASTNode,
    known_classes: dict[str, str],
    node_classes: set[str],
    hierarchy_roots: set[str],
    func_return_types: dict[str, str],
    result: FieldResult,
) -> None:
    """Collect fields from a single class body and __init__."""
    class_name = get_str(node, "name")
    lineno = get_int(node, "lineno")
    info = ClassInfo(class_name)
    # Check if this is an enum class
    bases = get_nodes(node, "bases")
    for base in bases:
        base_name = get_str(base, "id") if _is_type(base, ["Name"]) else ""
        if base_name == "StrEnum" or base_name == "IntEnum":
            info.is_enum = True
    if info.is_enum:
        body = get_nodes(node, "body")
        seen_variants: set[str] = set()
        for stmt in body:
            if _is_type(stmt, ["Assign"]):
                targets = get_nodes(stmt, "targets")
                if targets:
                    t = targets[0]
                    if _is_type(t, ["Name"]):
                        vname = get_str(t, "id")
                        if vname:
                            if vname in seen_variants:
                                result.add_error(
                                    lineno, 0, "duplicate variant '" + vname + "'"
                                )
                                return
                            seen_variants.add(vname)
                            info.enum_variants.append(vname)
        result.classes[class_name] = info
        return
    is_dc, kw_only = _is_dataclass_class(node)
    info.is_dataclass = is_dc
    info.kw_only = kw_only
    seen_fields: set[str] = set()
    body = get_nodes(node, "body")
    for stmt in body:
        if _is_type(stmt, ["AnnAssign"]):
            target = get_node(stmt, "target")
            if _is_type(target, ["Name"]):
                field_name = get_str(target, "id")
                if field_name:
                    if field_name in seen_fields:
                        result.add_error(
                            lineno, 0, "field '" + field_name + "' already declared"
                        )
                        return
                    seen_fields.add(field_name)
                    ann = get_node(stmt, "annotation")
                    py_type = annotation_to_str(ann)
                    sig_errors: list[TypeCollectError] = []
                    typ = py_type_to_type_dict(
                        py_type, known_classes, sig_errors, lineno, 0
                    )
                    typ = _unwrap_field_type(typ)
                    has_default = False
                    default_expr: TypeNode | None = None
                    v = stmt.get("value")
                    if v is not None and not isinstance(v, JNull):
                        if isinstance(v, JDict):
                            value_node = v.entries
                        else:
                            value_node: ASTNode = {}
                        if _is_field_call_default_factory(value_node):
                            result.add_error(
                                lineno, 0, "field(default_factory=...) not allowed"
                            )
                            return
                        exclude_from_init = _is_field_call_init_false(value_node)
                        has_default = True
                        default_expr = _make_default_expr(value_node)
                    else:
                        exclude_from_init = False
                    if field_name not in info.fields:
                        info.field_order.append(field_name)
                    fi = FieldInfo(
                        name=field_name,
                        typ=typ,
                        py_name=field_name,
                        has_default=has_default,
                        default=default_expr,
                    )
                    if exclude_from_init:
                        fi.exclude_from_init = True
                    info.fields[field_name] = fi
    has_init = False
    for stmt in body:
        if _is_type(stmt, ["FunctionDef"]) and get_str(stmt, "name") == "__init__":
            has_init = True
            info.has_explicit_init = True
            _collect_init_fields(
                stmt, info, known_classes, func_return_types, result._errors
            )
            if result._errors:
                return
    if is_dc and not has_init:
        fkeys = info.field_order if info.field_order else list(info.fields.keys())
        for fkey in fkeys:
            if info.fields[fkey].exclude_from_init:
                continue
            info.init_params.append(fkey)
            info.param_defaults[fkey] = info.fields[fkey].has_default
    known_field_set: set[str] = set()
    fkeys = info.field_order if info.field_order else list(info.fields.keys())
    for fkey in fkeys:
        known_field_set.add(fkey)
    for ckey in list(info.const_fields.keys()):
        known_field_set.add(ckey)
    for stmt in body:
        if _is_type(stmt, ["FunctionDef"]) and get_str(stmt, "name") != "__init__":
            bad = _check_no_new_fields_outside_init(stmt, known_field_set)
            if bad is not None:
                result.add_error(
                    lineno, 0, "field '" + bad + "' must be assigned in __init__"
                )
                return
    if has_init and class_name not in hierarchy_roots:
        if "kind" not in info.const_fields:
            kind_from_param = False
            for p in info.init_params:
                mapped = info.param_to_field.get(p)
                if mapped == "kind" or p == "kind":
                    kind_from_param = True
            if not kind_from_param:
                info.const_fields["kind"] = _pascal_to_kebab(class_name)
    if class_name in node_classes:
        fkeys = info.field_order if info.field_order else list(info.fields.keys())
        for fname in fkeys:
            if fname not in result.field_to_structs:
                result.field_to_structs[fname] = []
            if class_name not in result.field_to_structs[fname]:
                result.field_to_structs[fname].append(class_name)
    result.classes[class_name] = info


# ---------------------------------------------------------------------------
# Field collection entry point
# ---------------------------------------------------------------------------


def collect_fields(
    tree: ASTNode,
    known_classes: dict[str, str],
    node_classes: set[str],
    hierarchy_roots: set[str],
    sig_result: SignatureResult,
) -> FieldResult:
    """Collect field information from all classes in the module AST.

    Args:
        tree: The module AST dict.
        known_classes: Set of known class names from the name table.
        node_classes: Set of class names that are Node subclasses.
        hierarchy_roots: Set of class names that are hierarchy roots.
        sig_result: Signature result for function return types.
    """
    func_return_types: dict[str, str] = {}
    for fkey in list(sig_result.functions.keys()):
        func = sig_result.functions[fkey]
        func_return_types[fkey] = func.return_py_type
    result = FieldResult()
    body = get_nodes(tree, "body")
    for node in body:
        if _is_type(node, ["ClassDef"]):
            sf = get_str(node, "_source_file")
            err_before = len(result._errors)
            _collect_class_fields(
                node,
                known_classes,
                node_classes,
                hierarchy_roots,
                func_return_types,
                result,
            )
            for err in result._errors[err_before:]:
                err.source_file = sf
            if result._errors:
                return result
    return result


# ---------------------------------------------------------------------------
# Combined collection
# ---------------------------------------------------------------------------


class TypeCollectResult:
    """Combined result of signature and field collection."""

    def __init__(self) -> None:
        self.functions: dict[str, FuncInfo] = {}
        self.methods: dict[str, dict[str, FuncInfo]] = {}
        self.method_to_structs: dict[str, str] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.field_to_structs: dict[str, list[str]] = {}
        self._errors: list[TypeCollectError | FieldError] = []

    def errors(self) -> list[TypeCollectError | FieldError]:
        return self._errors

    def fields_to_dict(self) -> JsonValue:
        """Serialize fields portion to JsonValue."""
        classes: dict[str, JsonValue] = {}
        for ckey in list(self.classes.keys()):
            classes[ckey] = self.classes[ckey].to_dict()
        return JDict({"classes": JDict(classes)})


def collect_types(
    tree: ASTNode,
    known_classes: dict[str, str],
    node_classes: set[str],
    type_aliases: dict[str, str] | None,
    class_bases: dict[str, list[str]] | None,
    hierarchy_roots: set[str],
) -> TypeCollectResult:
    """Collect both signatures and fields from the module AST."""
    sig_result = collect_signatures(
        tree, known_classes, node_classes, type_aliases, class_bases
    )
    result = TypeCollectResult()
    result.functions = sig_result.functions
    result.methods = sig_result.methods
    result.method_to_structs = sig_result.method_to_structs
    sig_errors = sig_result.errors()
    if sig_errors:
        for sig_error in sig_errors:
            result._errors.append(sig_error)
        return result
    field_result = collect_fields(
        tree, known_classes, node_classes, hierarchy_roots, sig_result
    )
    result.classes = field_result.classes
    result.field_to_structs = field_result.field_to_structs
    field_errors = field_result.errors()
    if field_errors:
        for field_error in field_errors:
            result._errors.append(field_error)
    return result
