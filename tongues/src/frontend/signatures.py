"""Phase 5: Function and method signature collection.

Collect function and method signatures. Parse Python type annotations into
type dicts, classify parameter kinds, record default values, and detect
mutated parameters.

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from __future__ import annotations


# Type alias for AST dict nodes
ASTNode = dict[str, object]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class ParamInfo:
    """A single function/method parameter."""

    def __init__(
        self,
        name: str,
        typ: dict[str, object],
        py_type: str,
        has_default: bool,
        default_value: dict[str, object] | None,
        modifier: str,
    ) -> None:
        self.name: str = name
        self.typ: dict[str, object] = typ
        self.py_type: str = py_type
        self.has_default: bool = has_default
        self.default_value: dict[str, object] | None = default_value
        self.modifier: str = modifier

    def to_dict(self) -> dict[str, object]:
        """Serialize to a dict for test assertions."""
        d: dict[str, object] = {
            "name": self.name,
            "typ": self.typ,
            "py_type": self.py_type,
            "has_default": self.has_default,
            "default_value": self.default_value,
            "modifier": self.modifier,
        }
        return d


class FuncInfo:
    """Signature information for a function or method."""

    def __init__(
        self,
        name: str,
        params: list[ParamInfo],
        return_type: dict[str, object],
        return_py_type: str,
        is_method: bool,
        receiver_type: str,
    ) -> None:
        self.name: str = name
        self.params: list[ParamInfo] = params
        self.return_type: dict[str, object] = return_type
        self.return_py_type: str = return_py_type
        self.is_method: bool = is_method
        self.receiver_type: str = receiver_type

    def to_dict(self) -> dict[str, object]:
        """Serialize to a dict for test assertions."""
        param_dicts: list[object] = []
        i = 0
        while i < len(self.params):
            param_dicts.append(self.params[i].to_dict())
            i += 1
        d: dict[str, object] = {
            "params": param_dicts,
            "return_type": self.return_type,
            "return_py_type": self.return_py_type,
            "is_method": self.is_method,
            "receiver_type": self.receiver_type,
        }
        return d


class SignatureError:
    """An error found during signature collection."""

    def __init__(self, lineno: int, col: int, message: str) -> None:
        self.lineno: int = lineno
        self.col: int = col
        self.message: str = message

    def __repr__(self) -> str:
        return (
            "error:"
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

    def to_dict(self) -> dict[str, object]:
        """Serialize to nested dicts for test assertions."""
        funcs: dict[str, object] = {}
        fkeys = list(self.functions.keys())
        i = 0
        while i < len(fkeys):
            name = fkeys[i]
            funcs[name] = self.functions[name].to_dict()
            i += 1
        meths: dict[str, object] = {}
        ckeys = list(self.methods.keys())
        i = 0
        while i < len(ckeys):
            cname = ckeys[i]
            class_methods: dict[str, object] = {}
            mkeys = list(self.methods[cname].keys())
            j = 0
            while j < len(mkeys):
                mname = mkeys[j]
                class_methods[mname] = self.methods[cname][mname].to_dict()
                j += 1
            meths[cname] = class_methods
            i += 1
        result: dict[str, object] = {}
        if len(funcs) > 0:
            result["functions"] = funcs
        if len(meths) > 0:
            result["methods"] = meths
        return result


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _is_type(node: object, type_names: list[str]) -> bool:
    """Check if node is one of the given AST types."""
    if not isinstance(node, dict):
        return False
    t = node.get("_type")
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
            if isinstance(value, dict) and "_type" in value:
                result = result + _dict_walk(value)
            elif isinstance(value, list):
                j = 0
                while j < len(value):
                    item = value[j]
                    if isinstance(item, dict) and "_type" in item:
                        result = result + _dict_walk(item)
                    j += 1
        i += 1
    return result


# ---------------------------------------------------------------------------
# Annotation to string
# ---------------------------------------------------------------------------


def annotation_to_str(node: object) -> str:
    """Convert a type annotation AST node to its string representation."""
    if node is None:
        return ""
    if not isinstance(node, dict):
        return ""
    node_t = node.get("_type")
    if node_t == "Name":
        v = node.get("id", "")
        if isinstance(v, str):
            return v
        return ""
    if node_t == "Constant":
        v = node.get("value")
        if v is None:
            return "None"
        s_v = str(v)
        if s_v == "Ellipsis":
            return "..."
        return s_v
    if node_t == "List":
        elts = node.get("elts", [])
        if not isinstance(elts, list):
            return "[]"
        parts: list[str] = []
        i = 0
        while i < len(elts):
            parts.append(annotation_to_str(elts[i]))
            i += 1
        return "[" + ", ".join(parts) + "]"
    if node_t == "Subscript":
        base = annotation_to_str(node.get("value"))
        slc = node.get("slice", {})
        if isinstance(slc, dict) and slc.get("_type") == "Tuple":
            elts = slc.get("elts", [])
            if not isinstance(elts, list):
                return base + "[]"
            parts = []
            i = 0
            while i < len(elts):
                parts.append(annotation_to_str(elts[i]))
                i += 1
            return base + "[" + ", ".join(parts) + "]"
        return base + "[" + annotation_to_str(slc) + "]"
    if node_t == "BinOp":
        op = node.get("op", {})
        if isinstance(op, dict) and op.get("_type") == "BitOr":
            left = annotation_to_str(node.get("left"))
            right = annotation_to_str(node.get("right"))
            return left + " | " + right
    if node_t == "Attribute":
        v = node.get("attr", "")
        if isinstance(v, str):
            return v
        return ""
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
            current = []
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
            current = []
            i += 3
            continue
        else:
            current.append(c)
        i += 1
    tail = "".join(current).strip()
    if tail != "":
        result.append(tail)
    return result


# Primitive type mapping: Python name -> dict kind
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
) -> dict[str, object]:
    """Convert a Python type string to a type dict.

    Returns a type dict like {"kind": "int"} or {"_type": "Slice", "element": {...}}.
    """
    s = py_type.strip()
    if s == "":
        return {"_type": "InterfaceRef", "name": "any"}
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
        return {"kind": _PRIM_MAP[s]}
    # bytes -> Slice(byte)
    if s == "bytes" or s == "bytearray":
        return {"_type": "Slice", "element": {"kind": "byte"}}
    # object -> InterfaceRef("any") per spec
    if s == "object":
        return {"_type": "InterfaceRef", "name": "any"}
    # Known class -> Pointer(StructRef)
    if s in known_classes:
        return {"_type": "Pointer", "target": {"_type": "StructRef", "name": s}}
    errors.append(SignatureError(lineno, col, "unknown type '" + s + "'"))
    return {"_type": "InterfaceRef", "name": "any"}


def _resolve_subscript(
    base: str,
    inner: str,
    known_classes: set[str],
    errors: list[SignatureError],
    lineno: int,
    col: int,
) -> dict[str, object]:
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
            return {"_type": "InterfaceRef", "name": "any"}
        elem = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        return {"_type": "Slice", "element": elem}
    if base == "dict":
        if len(args) != 2:
            errors.append(
                SignatureError(
                    lineno,
                    col,
                    "dict requires 2 type arguments, got " + str(len(args)),
                )
            )
            return {"_type": "InterfaceRef", "name": "any"}
        key = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        val = py_type_to_type_dict(args[1], known_classes, errors, lineno, col)
        return {"_type": "Map", "key": key, "value": val}
    if base == "set" or base == "frozenset":
        if len(args) != 1:
            errors.append(
                SignatureError(
                    lineno,
                    col,
                    base + " requires 1 type argument, got " + str(len(args)),
                )
            )
            return {"_type": "InterfaceRef", "name": "any"}
        elem = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        return {"_type": "Set", "element": elem}
    if base == "tuple":
        if len(args) == 0:
            errors.append(
                SignatureError(lineno, col, "tuple requires at least 1 type argument")
            )
            return {"_type": "InterfaceRef", "name": "any"}
        # Check for variadic tuple: tuple[T, ...]
        if len(args) == 2 and args[1] == "...":
            elem = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
            return {"_type": "Tuple", "elements": [elem], "variadic": True}
        elems: list[object] = []
        i = 0
        while i < len(args):
            elems.append(
                py_type_to_type_dict(args[i], known_classes, errors, lineno, col)
            )
            i += 1
        return {"_type": "Tuple", "elements": elems, "variadic": False}
    if base == "Optional":
        if len(args) != 1:
            errors.append(
                SignatureError(
                    lineno,
                    col,
                    "Optional requires 1 type argument, got " + str(len(args)),
                )
            )
            return {"_type": "InterfaceRef", "name": "any"}
        inner_t = py_type_to_type_dict(args[0], known_classes, errors, lineno, col)
        return {"_type": "Optional", "inner": inner_t}
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
            return {"_type": "InterfaceRef", "name": "any"}
        # First arg is a list of param types like [int, str]
        param_str = args[0].strip()
        param_types: list[object] = []
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
        return {"_type": "FuncType", "params": param_types, "ret": ret}
    # Unknown subscript base
    errors.append(SignatureError(lineno, col, "unknown type '" + base + "'"))
    return {"_type": "InterfaceRef", "name": "any"}


def _resolve_union(
    members: list[str],
    known_classes: set[str],
    errors: list[SignatureError],
    lineno: int,
    col: int,
) -> dict[str, object]:
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
        return {"_type": "Optional", "inner": inner}
    if has_none and len(non_none) > 1:
        # Optional of a union — wrap the union in Optional
        inner = _resolve_non_none_union(non_none, known_classes, errors, lineno, col)
        return {"_type": "Optional", "inner": inner}
    # Non-None union -> InterfaceRef
    return _resolve_non_none_union(unique, known_classes, errors, lineno, col)


def _resolve_non_none_union(
    members: list[str],
    known_classes: set[str],
    errors: list[SignatureError],
    lineno: int,
    col: int,
) -> dict[str, object]:
    """Resolve a union with no None members to an InterfaceRef."""
    return {"_type": "InterfaceRef", "name": "any"}


# ---------------------------------------------------------------------------
# Default value lowering
# ---------------------------------------------------------------------------


def _lower_default(node: ASTNode) -> dict[str, object] | None:
    """Lower a default value AST node to an expression dict."""
    if not isinstance(node, dict):
        return None
    t = node.get("_type")
    if t == "Constant":
        v = node.get("value")
        if v is None:
            return {"_type": "NilLit"}
        if isinstance(v, bool):
            return {"_type": "BoolLit", "value": v}
        if isinstance(v, int):
            return {"_type": "IntLit", "value": v}
        if isinstance(v, float):
            return {"_type": "FloatLit", "value": v}
        if isinstance(v, str):
            return {"_type": "StringLit", "value": v}
    if t == "UnaryOp":
        op = node.get("op", {})
        if isinstance(op, dict) and op.get("_type") == "USub":
            operand = node.get("operand", {})
            if isinstance(operand, dict) and operand.get("_type") == "Constant":
                v = operand.get("value")
                if isinstance(v, int):
                    return {"_type": "IntLit", "value": -v}
                if isinstance(v, float):
                    return {"_type": "FloatLit", "value": -v}
    if t == "List":
        return {"_type": "ListLit", "elements": []}
    if t == "Dict":
        return {"_type": "MapLit", "entries": []}
    if t == "Set":
        return {"_type": "SetLit", "elements": []}
    if t == "Tuple":
        return {"_type": "TupleLit", "elements": []}
    return None


# ---------------------------------------------------------------------------
# Mutated parameter detection
# ---------------------------------------------------------------------------


def detect_mutated_params(node: ASTNode) -> set[str]:
    """Detect which parameters are mutated in the function body."""
    mutated: set[str] = set()
    args = node.get("args", {})
    if not isinstance(args, dict):
        return mutated
    # Collect all param names (excluding self)
    param_names: set[str] = set()
    posonlyargs = args.get("posonlyargs", [])
    regular_args = args.get("args", [])
    kwonlyargs = args.get("kwonlyargs", [])
    if isinstance(posonlyargs, list):
        i = 0
        while i < len(posonlyargs):
            a = posonlyargs[i]
            if isinstance(a, dict):
                name = a.get("arg")
                if isinstance(name, str) and name != "self":
                    param_names.add(name)
            i += 1
    if isinstance(regular_args, list):
        i = 0
        while i < len(regular_args):
            a = regular_args[i]
            if isinstance(a, dict):
                name = a.get("arg")
                if isinstance(name, str) and name != "self":
                    param_names.add(name)
            i += 1
    if isinstance(kwonlyargs, list):
        i = 0
        while i < len(kwonlyargs):
            a = kwonlyargs[i]
            if isinstance(a, dict):
                name = a.get("arg")
                if isinstance(name, str) and name != "self":
                    param_names.add(name)
            i += 1
    # Walk the function body looking for mutations
    all_nodes = _dict_walk(node)
    i = 0
    while i < len(all_nodes):
        stmt = all_nodes[i]
        # param.append(...), param.extend(...), param.clear(), param.pop()
        if _is_type(stmt, ["Expr"]):
            val = stmt.get("value")
            if isinstance(val, dict) and _is_type(val, ["Call"]):
                func = val.get("func")
                if isinstance(func, dict) and _is_type(func, ["Attribute"]):
                    attr = func.get("attr")
                    if attr in ("append", "extend", "clear", "pop"):
                        obj = func.get("value")
                        if isinstance(obj, dict) and _is_type(obj, ["Name"]):
                            obj_id = obj.get("id")
                            if isinstance(obj_id, str) and obj_id in param_names:
                                mutated.add(obj_id)
        # param[i] = ...
        if _is_type(stmt, ["Assign"]):
            targets = stmt.get("targets", [])
            if isinstance(targets, list):
                j = 0
                while j < len(targets):
                    target = targets[j]
                    if isinstance(target, dict) and _is_type(target, ["Subscript"]):
                        obj = target.get("value")
                        if isinstance(obj, dict) and _is_type(obj, ["Name"]):
                            obj_id = obj.get("id")
                            if isinstance(obj_id, str) and obj_id in param_names:
                                mutated.add(obj_id)
                    j += 1
        i += 1
    return mutated


# ---------------------------------------------------------------------------
# Type dict helpers
# ---------------------------------------------------------------------------


def _is_slice_type(typ: dict[str, object]) -> bool:
    """Check if a type dict is a Slice (list) type."""
    return typ.get("_type") == "Slice"


def _wrap_pointer(typ: dict[str, object]) -> dict[str, object]:
    """Wrap a type in a Pointer."""
    return {"_type": "Pointer", "target": typ}


# ---------------------------------------------------------------------------
# Parameter and function extraction
# ---------------------------------------------------------------------------


def _make_param(
    arg: ASTNode,
    modifier: str,
    has_default: bool,
    default_node: object,
    mutated_params: set[str],
    known_classes: set[str],
    errors: list[SignatureError],
    func_name: str,
) -> ParamInfo | None:
    """Build a ParamInfo from an AST arg node. Returns None on error."""
    if not isinstance(arg, dict):
        return None
    param_name = arg.get("arg")
    if not isinstance(param_name, str):
        return None
    annotation = arg.get("annotation")
    lineno_val = arg.get("lineno", 0)
    lineno = lineno_val if isinstance(lineno_val, int) else 0
    if annotation is None:
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
    # Wrap mutated list params in Pointer
    if param_name in mutated_params and _is_slice_type(typ):
        typ = _wrap_pointer(typ)
    default_value: dict[str, object] | None = None
    if has_default and default_node is not None and isinstance(default_node, dict):
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
    func_name = node.get("name", "")
    if not isinstance(func_name, str):
        func_name = ""
    lineno_val = node.get("lineno", 0)
    lineno = lineno_val if isinstance(lineno_val, int) else 0
    mutated_params = detect_mutated_params(node)
    params: list[ParamInfo] = []
    args = node.get("args", {})
    if not isinstance(args, dict):
        args = {}
    # Positional-only params (before /)
    posonlyargs = args.get("posonlyargs", [])
    if not isinstance(posonlyargs, list):
        posonlyargs = []
    # Regular params (between / and *)
    regular_args = args.get("args", [])
    if not isinstance(regular_args, list):
        regular_args = []
    # Keyword-only params (after *)
    kwonlyargs = args.get("kwonlyargs", [])
    if not isinstance(kwonlyargs, list):
        kwonlyargs = []
    # Defaults apply to the tail of posonlyargs + regular_args
    defaults = args.get("defaults", [])
    if not isinstance(defaults, list):
        defaults = []
    # kw_defaults is parallel to kwonlyargs (None entries for no default)
    kw_defaults = args.get("kw_defaults", [])
    if not isinstance(kw_defaults, list):
        kw_defaults = []
    # Filter self from params
    non_self_posonly: list[ASTNode] = []
    i = 0
    while i < len(posonlyargs):
        a = posonlyargs[i]
        if isinstance(a, dict) and a.get("arg") != "self":
            non_self_posonly.append(a)
        i += 1
    non_self_regular: list[ASTNode] = []
    i = 0
    while i < len(regular_args):
        a = regular_args[i]
        if isinstance(a, dict) and a.get("arg") != "self":
            non_self_regular.append(a)
        i += 1
    # defaults covers the tail of posonlyargs + regular_args combined
    n_positional = len(non_self_posonly) + len(non_self_regular)
    n_defaults = len(defaults)
    had_error = False
    # Positional-only params
    i = 0
    while i < len(non_self_posonly):
        has_default = i >= n_positional - n_defaults
        default_node: object = None
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
    # Regular params
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
    # Keyword-only params
    i = 0
    while i < len(kwonlyargs):
        has_default = False
        default_node = None
        if i < len(kw_defaults):
            kw_def = kw_defaults[i]
            if kw_def is not None:
                has_default = True
                default_node = kw_def
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
    # Return type
    returns = node.get("returns")
    if returns is None:
        errors.append(
            SignatureError(
                lineno,
                0,
                "function '" + func_name + "' missing return type annotation",
            )
        )
        return None
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
) -> SignatureResult:
    """Collect function and method signatures from the module AST.

    Args:
        tree: The module AST dict.
        known_classes: Set of known class names from the name table.
        node_classes: Set of class names that are Node subclasses.
    """
    result = SignatureResult()
    body = tree.get("body", [])
    if not isinstance(body, list):
        return result
    i = 0
    while i < len(body):
        node = body[i]
        if not isinstance(node, dict):
            i += 1
            continue
        t = node.get("_type")
        if t == "FunctionDef":
            info = extract_func_info(node, known_classes, result._errors, False, "")
            if info is not None:
                result.functions[info.name] = info
        elif t == "ClassDef":
            class_name = node.get("name", "")
            if not isinstance(class_name, str):
                class_name = ""
            class_body = node.get("body", [])
            if not isinstance(class_body, list):
                class_body = []
            class_methods: dict[str, FuncInfo] = {}
            j = 0
            while j < len(class_body):
                stmt = class_body[j]
                if isinstance(stmt, dict) and stmt.get("_type") == "FunctionDef":
                    method_info = extract_func_info(
                        stmt, known_classes, result._errors, True, class_name
                    )
                    if method_info is not None:
                        class_methods[method_info.name] = method_info
                j += 1
            if len(class_methods) > 0:
                result.methods[class_name] = class_methods
            # Build method-to-struct mapping for Node subclasses
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
