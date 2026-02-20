"""Phase 5: Function and method signature collection.

Collect function and method signatures. Parse Python type annotations into
TypeNode objects, classify parameter kinds, record default values, and detect
mutated parameters.

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from __future__ import annotations

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
# Data classes
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
        i = 0
        while i < len(self.params):
            param_dicts.append(self.params[i].to_dict())
            i += 1
        return JDict(
            {
                "params": JList(param_dicts),
                "return_type": typenode_to_dict(self.return_type),
                "return_py_type": JStr(self.return_py_type),
                "is_method": JBool(self.is_method),
                "receiver_type": JStr(self.receiver_type),
            }
        )


class SignatureError:
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
        if self.source_file != "":
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
        self._errors: list[SignatureError] = []

    def add_error(self, lineno: int, col: int, message: str) -> None:
        self._errors.append(SignatureError(lineno, col, message))

    def errors(self) -> list[SignatureError]:
        return self._errors

    def to_dict(self) -> JsonValue:
        """Serialize to nested JsonValue dicts for test assertions."""
        funcs: dict[str, JsonValue] = {}
        fkeys = list(self.functions.keys())
        i = 0
        while i < len(fkeys):
            name = fkeys[i]
            funcs[name] = self.functions[name].to_dict()
            i += 1
        meths: dict[str, JsonValue] = {}
        ckeys = list(self.methods.keys())
        i = 0
        while i < len(ckeys):
            cname = ckeys[i]
            class_methods: dict[str, JsonValue] = {}
            mkeys = list(self.methods[cname].keys())
            j = 0
            while j < len(mkeys):
                mname = mkeys[j]
                class_methods[mname] = self.methods[cname][mname].to_dict()
                j += 1
            meths[cname] = JDict(class_methods)
            i += 1
        result: dict[str, JsonValue] = {}
        if len(funcs) > 0:
            result["functions"] = JDict(funcs)
        if len(meths) > 0:
            result["methods"] = JDict(meths)
        return JDict(result)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _is_type(node: ASTNode, type_names: list[str]) -> bool:
    """Check if node is one of the given AST types."""
    t = get_str(node, "_type")
    i = 0
    while i < len(type_names):
        if t == type_names[i]:
            return True
        i += 1
    return False


def _dict_walk(node: ASTNode) -> list[ASTNode]:
    """Walk dict-based AST, returns list of all nodes."""
    result: list[ASTNode] = [node]
    keys = list(node.keys())
    i = 0
    while i < len(keys):
        key = keys[i]
        if not key.startswith("_"):
            value = node[key]
            if isinstance(value, JDict) and has_key(value.entries, "_type"):
                result = result + _dict_walk(value.entries)
            elif isinstance(value, JList):
                j = 0
                while j < len(value.items):
                    item = value.items[j]
                    if isinstance(item, JDict) and has_key(item.entries, "_type"):
                        result = result + _dict_walk(item.entries)
                    j += 1
        i += 1
    return result


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
        i = 0
        while i < len(elts):
            parts.append(annotation_to_str(elts[i]))
            i += 1
        return "[" + ", ".join(parts) + "]"
    if node_t == "Subscript":
        value_node = get_node(node, "value")
        base = annotation_to_str(value_node)
        slc = get_node(node, "slice")
        if get_str(slc, "_type") == "Tuple":
            elts = get_nodes(slc, "elts")
            parts: list[str] = []
            i = 0
            while i < len(elts):
                parts.append(annotation_to_str(elts[i]))
                i += 1
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
    i = 0
    while i < len(s):
        c = s[i]
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
        i += 1
    tail = "".join(current).strip()
    if tail != "":
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
    if tail != "":
        result.append(tail)
    return result


# Type alias expansions, populated by collect_signatures()
_TYPE_ALIASES: dict[str, str] = {}


# Primitive type mapping: Python name -> kind string
_PRIM_MAP: dict[str, str] = {
    "int": "int",
    "str": "string",
    "bool": "bool",
    "float": "float",
    "byte": "byte",
    "None": "void",
}


def py_type_to_type_dict(
    py_type: str,
    known_classes: set[str],
    errors: list[SignatureError],
    lineno: int,
    col: int,
) -> TypeNode:
    """Convert a Python type string to a TypeNode."""
    s = py_type.strip()
    if s == "":
        return InterfaceRef("any")
    # Expand type aliases
    if s in _TYPE_ALIASES:
        return py_type_to_type_dict(
            _TYPE_ALIASES[s], known_classes, errors, lineno, col
        )
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
    if s == "bytes" or s == "bytearray":
        return SliceType(PrimitiveType("byte"))
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
    if s in known_classes:
        return PointerType(StructRef(s))
    errors.append(SignatureError(lineno, col, "unknown type '" + s + "'"))
    return InterfaceRef("any")


def _resolve_subscript(
    base: str,
    inner: str,
    known_classes: set[str],
    errors: list[SignatureError],
    lineno: int,
    col: int,
) -> TypeNode:
    """Resolve a subscripted type like list[int], dict[str, int], etc."""
    args = _split_type_args(inner)
    if base == "list":
        if len(args) != 1:
            errors.append(
                SignatureError(
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
                SignatureError(
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
                SignatureError(
                    lineno,
                    col,
                    base + " requires 1 type argument, got " + str(len(args)),
                )
            )
            return InterfaceRef("any")
        elem = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        return SetType(elem)
    if base == "tuple":
        if len(args) == 0:
            errors.append(
                SignatureError(lineno, col, "tuple requires at least 1 type argument")
            )
            return InterfaceRef("any")
        # Check for variadic tuple: tuple[T, ...]
        if len(args) == 2 and args[1] == "...":
            elem = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
            return TupleType([elem], True)
        elems: list[TypeNode] = []
        i = 0
        while i < len(args):
            elems.append(
                py_type_to_type_dict(args[i], known_classes, errors, lineno, col)
            )
            i += 1
        return TupleType(elems, False)
    if base == "Optional":
        if len(args) != 1:
            errors.append(
                SignatureError(
                    lineno,
                    col,
                    "Optional requires 1 type argument, got " + str(len(args)),
                )
            )
            return InterfaceRef("any")
        inner_t = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        return OptionalType(inner_t)
    if base == "Union":
        return _resolve_union(args, known_classes, errors, lineno, col)
    if base == "Callable":
        if len(args) != 2:
            errors.append(
                SignatureError(
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
            if param_inner != "":
                param_parts = _split_type_args(param_inner)
                j = 0
                while j < len(param_parts):
                    param_types.append(
                        py_type_to_type_dict(
                            param_parts[j], known_classes, errors, lineno, col
                        )
                    )
                    j += 1
        ret = py_type_to_type_dict(args[1], known_classes, errors, lineno, col)
        return FuncType(param_types, ret)
    # Sequence/Iterable[T] -> Slice(T)
    if base == "Sequence" or base == "Iterable":
        if len(args) != 1:
            errors.append(
                SignatureError(
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
                SignatureError(
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
    errors.append(SignatureError(lineno, col, "unknown type '" + base + "'"))
    return InterfaceRef("any")


def _resolve_union(
    members: list[str],
    known_classes: set[str],
    errors: list[SignatureError],
    lineno: int,
    col: int,
) -> TypeNode:
    """Resolve a union type from its member strings."""
    # Deduplicate
    unique: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(members):
        m = members[i].strip()
        if m not in seen:
            unique.append(m)
            seen.add(m)
        i += 1
    # Single type after dedup
    if len(unique) == 1:
        return py_type_to_type_dict(unique[0], known_classes, errors, lineno, col)
    # Check for T | None -> Optional
    has_none = False
    non_none: list[str] = []
    i = 0
    while i < len(unique):
        if unique[i] == "None":
            has_none = True
        else:
            non_none.append(unique[i])
        i += 1
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
    known_classes: set[str],
    errors: list[SignatureError],
    lineno: int,
    col: int,
) -> TypeNode:
    """Resolve a union with no None members to an InterfaceRef."""
    return InterfaceRef("any")


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
    if len(args_node) == 0:
        return mutated
    param_names: set[str] = set()
    posonlyargs = get_nodes(args_node, "posonlyargs")
    regular_args = get_nodes(args_node, "args")
    kwonlyargs = get_nodes(args_node, "kwonlyargs")
    i = 0
    while i < len(posonlyargs):
        a = posonlyargs[i]
        name = get_str(a, "arg")
        if name != "" and name != "self":
            param_names.add(name)
        i += 1
    i = 0
    while i < len(regular_args):
        a = regular_args[i]
        name = get_str(a, "arg")
        if name != "" and name != "self":
            param_names.add(name)
        i += 1
    i = 0
    while i < len(kwonlyargs):
        a = kwonlyargs[i]
        name = get_str(a, "arg")
        if name != "" and name != "self":
            param_names.add(name)
        i += 1
    all_nodes = _dict_walk(node)
    i = 0
    while i < len(all_nodes):
        stmt = all_nodes[i]
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
                            if obj_id != "" and obj_id in param_names:
                                mutated.add(obj_id)
        if _is_type(stmt, ["Assign"]):
            targets = get_nodes(stmt, "targets")
            j = 0
            while j < len(targets):
                target = targets[j]
                if _is_type(target, ["Subscript"]):
                    obj = get_node(target, "value")
                    if _is_type(obj, ["Name"]):
                        obj_id = get_str(obj, "id")
                        if obj_id != "" and obj_id in param_names:
                            mutated.add(obj_id)
                j += 1
        i += 1
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
    known_classes: set[str],
    errors: list[SignatureError],
    func_name: str,
) -> ParamInfo | None:
    """Build a ParamInfo from an AST arg node. Returns None on error."""
    param_name = get_str(arg, "arg")
    if param_name == "":
        return None
    annotation = get_node(arg, "annotation")
    lineno = get_int(arg, "lineno")
    if len(annotation) == 0:
        errors.append(
            SignatureError(
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
    known_classes: set[str],
    errors: list[SignatureError],
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
    i = 0
    while i < len(posonlyargs):
        a = posonlyargs[i]
        if get_str(a, "arg") != "self":
            non_self_posonly.append(a)
        i += 1
    non_self_regular: list[ASTNode] = []
    i = 0
    while i < len(regular_args):
        a = regular_args[i]
        if get_str(a, "arg") != "self":
            non_self_regular.append(a)
        i += 1
    n_positional = len(non_self_posonly) + len(non_self_regular)
    n_defaults = len(defaults)
    had_error = False
    i = 0
    while i < len(non_self_posonly):
        has_default = i >= n_positional - n_defaults
        default_node: ASTNode | None = None
        if has_default:
            default_idx = i - (n_positional - n_defaults)
            if default_idx >= 0 and default_idx < len(defaults):
                default_node = defaults[default_idx]
        p = _make_param(
            non_self_posonly[i],
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
        i += 1
    i = 0
    while i < len(non_self_regular):
        global_i = len(non_self_posonly) + i
        has_default = global_i >= n_positional - n_defaults
        default_node = None
        if has_default:
            default_idx = global_i - (n_positional - n_defaults)
            if default_idx >= 0 and default_idx < len(defaults):
                default_node = defaults[default_idx]
        p = _make_param(
            non_self_regular[i],
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
        i += 1
    i = 0
    while i < len(kwonlyargs):
        has_default = False
        default_node = None
        if i < len(kw_defaults):
            kw_def = kw_defaults[i]
            if not isinstance(kw_def, JNull):
                has_default = True
                if isinstance(kw_def, JDict):
                    default_node = kw_def.entries
        p = _make_param(
            kwonlyargs[i],
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
        i += 1
    returns = get_node(node, "returns")
    if len(returns) == 0:
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
                SignatureError(
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
# Main entry point
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
    known_classes: set[str],
    node_classes: set[str],
    type_aliases: dict[str, str] | None = None,
) -> SignatureResult:
    """Collect function and method signatures from the module AST."""
    _TYPE_ALIASES.clear()
    if type_aliases is not None:
        ta_keys = list(type_aliases.keys())
        tai = 0
        while tai < len(ta_keys):
            _TYPE_ALIASES[ta_keys[tai]] = type_aliases[ta_keys[tai]]
            tai += 1
    result = SignatureResult()
    body = get_nodes(tree, "body")
    i = 0
    while i < len(body):
        node = body[i]
        t = get_str(node, "_type")
        sf = get_str(node, "_source_file")
        if t == "FunctionDef":
            err_before = len(result._errors)
            info = extract_func_info(node, known_classes, result._errors, False, "")
            ei = err_before
            while ei < len(result._errors):
                result._errors[ei].source_file = sf
                ei += 1
            if info is not None:
                result.functions[info.name] = info
        elif t == "ClassDef":
            class_name = get_str(node, "name")
            class_body = get_nodes(node, "body")
            class_methods: dict[str, FuncInfo] = {}
            j = 0
            while j < len(class_body):
                stmt = class_body[j]
                if get_str(stmt, "_type") == "FunctionDef":
                    stmt_sf = get_str(stmt, "_source_file")
                    if stmt_sf == "":
                        stmt_sf = sf
                    err_before = len(result._errors)
                    method_info = extract_func_info(
                        stmt, known_classes, result._errors, True, class_name
                    )
                    ei = err_before
                    while ei < len(result._errors):
                        result._errors[ei].source_file = stmt_sf
                        ei += 1
                    if method_info is not None:
                        class_methods[method_info.name] = method_info
                j += 1
            if len(class_methods) > 0:
                result.methods[class_name] = class_methods
            if class_name in node_classes:
                mkeys = list(class_methods.keys())
                j = 0
                while j < len(mkeys):
                    mname = mkeys[j]
                    if mname not in _EXCLUDED_METHODS:
                        result.method_to_structs[mname] = class_name
                    j += 1
        i += 1
    return result
