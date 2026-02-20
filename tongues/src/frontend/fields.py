"""Phase 6: Field type collection.

Collect field types from class definitions. Fields are declared in class bodies
(dataclass-style annotations) or assigned in __init__. Also determines
constructor parameters, constant discriminator fields, and auto-generated kind
values.

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
    BoolLit,
    IntLit,
    StringLit,
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
from .signatures import (
    SignatureResult,
    annotation_to_str,
    py_type_to_type_dict,
)


# ---------------------------------------------------------------------------
# Data classes
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
        self.init_params: list[str] = []
        self.param_to_field: dict[str, str] = {}
        self.const_fields: dict[str, str] = {}
        self.is_dataclass: bool = False
        self.kw_only: bool = False
        self.needs_constructor: bool = False

    def to_dict(self) -> JsonValue:
        """Serialize to a JsonValue dict for test assertions."""
        fields: dict[str, JsonValue] = {}
        fkeys = list(self.fields.keys())
        i = 0
        while i < len(fkeys):
            fields[fkeys[i]] = self.fields[fkeys[i]].to_dict()
            i += 1
        init_params_jv: list[JsonValue] = []
        j = 0
        while j < len(self.init_params):
            init_params_jv.append(JStr(self.init_params[j]))
            j += 1
        ptf: dict[str, JsonValue] = {}
        ptf_keys = list(self.param_to_field.keys())
        j = 0
        while j < len(ptf_keys):
            ptf[ptf_keys[j]] = JStr(self.param_to_field[ptf_keys[j]])
            j += 1
        cf: dict[str, JsonValue] = {}
        cf_keys = list(self.const_fields.keys())
        j = 0
        while j < len(cf_keys):
            cf[cf_keys[j]] = JStr(self.const_fields[cf_keys[j]])
            j += 1
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
        if self.source_file != "":
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
        ckeys = list(self.classes.keys())
        i = 0
        while i < len(ckeys):
            classes[ckeys[i]] = self.classes[ckeys[i]].to_dict()
            i += 1
        return JDict({"classes": JDict(classes)})


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


def _wrap_nodes(lst: list[ASTNode]) -> JList:
    """Wrap a list of ASTNode dicts into a JList of JDicts."""
    items: list[JsonValue] = []
    i = 0
    while i < len(lst):
        items.append(JDict(lst[i]))
        i += 1
    return JList(items)


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
# Type helpers
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
# Default values
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
    i = 0
    while i < len(dec_list):
        dec = dec_list[i]
        if _is_type(dec, ["Name"]) and get_str(dec, "id") == "dataclass":
            return (True, False)
        if _is_type(dec, ["Call"]):
            func = get_node(dec, "func")
            if _is_type(func, ["Name"]) and get_str(func, "id") == "dataclass":
                kw_only = False
                keywords = get_nodes(dec, "keywords")
                j = 0
                while j < len(keywords):
                    kw = keywords[j]
                    if get_str(kw, "arg") == "kw_only":
                        val = get_node(kw, "value")
                        if _is_type(val, ["Constant"]):
                            v = val.get("value")
                            if isinstance(v, JBool) and v.value:
                                kw_only = True
                    j += 1
                return (True, kw_only)
        i += 1
    return (False, False)


# ---------------------------------------------------------------------------
# field(default_factory=...) detection
# ---------------------------------------------------------------------------


def _is_field_call_default_factory(node: ASTNode) -> bool:
    """Detect field(default_factory=...)."""
    if not _is_type(node, ["Call"]):
        return False
    func = get_node(node, "func")
    if not (_is_type(func, ["Name"]) and get_str(func, "id") == "field"):
        return False
    keywords = get_nodes(node, "keywords")
    i = 0
    while i < len(keywords):
        kw = keywords[i]
        if get_str(kw, "arg") == "default_factory":
            return True
        i += 1
    return False


# ---------------------------------------------------------------------------
# Conditional field assignment check
# ---------------------------------------------------------------------------


def _check_no_field_assign_in_block(block: list[ASTNode]) -> str | None:
    """Return field name if self.x = ... found inside block, else None."""
    wrapper: ASTNode = {"_type": JStr("_wrapper"), "body": _wrap_nodes(block)}
    all_nodes = _dict_walk(wrapper)
    i = 0
    while i < len(all_nodes):
        stmt = all_nodes[i]
        if _is_type(stmt, ["Assign"]):
            targets = get_nodes(stmt, "targets")
            j = 0
            while j < len(targets):
                target = targets[j]
                if _is_type(target, ["Attribute"]):
                    val_node = get_node(target, "value")
                    if (
                        _is_type(val_node, ["Name"])
                        and get_str(val_node, "id") == "self"
                    ):
                        attr = get_str(target, "attr")
                        if attr != "":
                            return attr
                j += 1
        if _is_type(stmt, ["AnnAssign"]):
            target = get_node(stmt, "target")
            if _is_type(target, ["Attribute"]):
                val_node = get_node(target, "value")
                if _is_type(val_node, ["Name"]) and get_str(val_node, "id") == "self":
                    attr = get_str(target, "attr")
                    if attr != "":
                        return attr
        i += 1
    return None


# ---------------------------------------------------------------------------
# Outside-init field check
# ---------------------------------------------------------------------------


def _check_no_new_fields_outside_init(
    func: ASTNode, known_fields: set[str]
) -> str | None:
    """Return field name if method introduces new self.x not in known_fields."""
    all_nodes = _dict_walk(func)
    i = 0
    while i < len(all_nodes):
        stmt = all_nodes[i]
        if _is_type(stmt, ["Assign"]):
            targets = get_nodes(stmt, "targets")
            j = 0
            while j < len(targets):
                target = targets[j]
                if _is_type(target, ["Attribute"]):
                    val_node = get_node(target, "value")
                    if (
                        _is_type(val_node, ["Name"])
                        and get_str(val_node, "id") == "self"
                    ):
                        fname = get_str(target, "attr")
                        if fname != "" and fname not in known_fields:
                            return fname
                j += 1
        if _is_type(stmt, ["AnnAssign"]):
            target = get_node(stmt, "target")
            if _is_type(target, ["Attribute"]):
                val_node = get_node(target, "value")
                if _is_type(val_node, ["Name"]) and get_str(val_node, "id") == "self":
                    fname = get_str(target, "attr")
                    if fname != "" and fname not in known_fields:
                        return fname
        i += 1
    return None


# ---------------------------------------------------------------------------
# Type inference from expressions
# ---------------------------------------------------------------------------


def _infer_type_from_value(
    node: ASTNode,
    param_types: dict[str, str],
    known_classes: set[str],
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
        if name != "" and name in param_types:
            py_type = param_types[name]
            from .signatures import SignatureError

            sig_errors: list[SignatureError] = []
            typ = py_type_to_type_dict(py_type, known_classes, sig_errors, lineno, 0)
            return _unwrap_field_type(typ)
        if name != "":
            errors.append(
                FieldError(lineno, 0, "cannot infer type for field from '" + name + "'")
            )
        return None
    if t == "Call":
        func = get_node(node, "func")
        if _is_type(func, ["Name"]):
            func_name = get_str(func, "id")
            if func_name != "":
                if func_name in known_classes:
                    return StructRef(func_name)
                if func_name in func_return_types:
                    py_ret = func_return_types[func_name]
                    from .signatures import SignatureError

                    sig_errors: list[SignatureError] = []
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
    known_classes: set[str],
    func_return_types: dict[str, str],
    errors: list[FieldError],
) -> None:
    """Collect fields assigned in __init__."""
    args = get_node(init, "args")
    param_types: dict[str, str] = {}
    args_list = get_nodes(args, "args")
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        arg_name = get_str(arg, "arg")
        if arg_name != "" and arg_name != "self":
            info.init_params.append(arg_name)
            annotation = get_node(arg, "annotation")
            if len(annotation) > 0:
                param_types[arg_name] = annotation_to_str(annotation)
        i += 1
    posonlyargs = get_nodes(args, "posonlyargs")
    i = 0
    while i < len(posonlyargs):
        arg = posonlyargs[i]
        arg_name = get_str(arg, "arg")
        if arg_name != "" and arg_name != "self":
            info.init_params.append(arg_name)
            annotation = get_node(arg, "annotation")
            if len(annotation) > 0:
                param_types[arg_name] = annotation_to_str(annotation)
        i += 1
    kwonlyargs = get_nodes(args, "kwonlyargs")
    i = 0
    while i < len(kwonlyargs):
        arg = kwonlyargs[i]
        arg_name = get_str(arg, "arg")
        if arg_name != "" and arg_name != "self":
            info.init_params.append(arg_name)
            annotation = get_node(arg, "annotation")
            if len(annotation) > 0:
                param_types[arg_name] = annotation_to_str(annotation)
        i += 1
    # Build param_has_default map from defaults/kw_defaults
    param_has_default: dict[str, bool] = {}
    defaults = get_nodes(args, "defaults")
    kw_defaults = get_jlist(args, "kw_defaults")
    non_self_posonly: list[ASTNode] = []
    i = 0
    while i < len(posonlyargs):
        a = posonlyargs[i]
        if get_str(a, "arg") != "self":
            non_self_posonly.append(a)
        i += 1
    non_self_regular: list[ASTNode] = []
    i = 0
    while i < len(args_list):
        a = args_list[i]
        if get_str(a, "arg") != "self":
            non_self_regular.append(a)
        i += 1
    n_positional = len(non_self_posonly) + len(non_self_regular)
    n_defaults = len(defaults)
    i = 0
    while i < len(non_self_posonly):
        pname = get_str(non_self_posonly[i], "arg")
        if pname != "":
            param_has_default[pname] = i >= n_positional - n_defaults
        i += 1
    i = 0
    while i < len(non_self_regular):
        pname = get_str(non_self_regular[i], "arg")
        if pname != "":
            idx = len(non_self_posonly) + i
            param_has_default[pname] = idx >= n_positional - n_defaults
        i += 1
    i = 0
    while i < len(kwonlyargs):
        pname = get_str(kwonlyargs[i], "arg")
        if pname != "" and pname != "self":
            has_kw_def = i < len(kw_defaults) and not isinstance(kw_defaults[i], JNull)
            param_has_default[pname] = has_kw_def
        i += 1
    has_computed_init = False
    body = get_nodes(init, "body")
    lineno = get_int(init, "lineno")
    i = 0
    while i < len(body):
        stmt = body[i]
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
            i += 1
            continue
        if _is_type(stmt, ["AnnAssign"]):
            target = get_node(stmt, "target")
            if _is_type(target, ["Attribute"]):
                val_node = get_node(target, "value")
                if _is_type(val_node, ["Name"]) and get_str(val_node, "id") == "self":
                    field_name = get_str(target, "attr")
                    if field_name != "":
                        ann = get_node(stmt, "annotation")
                        py_type = annotation_to_str(ann)
                        from .signatures import SignatureError

                        sig_errors: list[SignatureError] = []
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
                            info.fields[field_name] = FieldInfo(
                                name=field_name,
                                typ=typ,
                                py_name=field_name,
                                has_default=ann_has_default,
                                default=None,
                            )
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
            i += 1
            continue
        if _is_type(stmt, ["Assign"]):
            targets = get_nodes(stmt, "targets")
            j = 0
            while j < len(targets):
                target = targets[j]
                if _is_type(target, ["Attribute"]):
                    val_node = get_node(target, "value")
                    if (
                        _is_type(val_node, ["Name"])
                        and get_str(val_node, "id") == "self"
                    ):
                        field_name = get_str(target, "attr")
                        if field_name != "":
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
                                        from .signatures import SignatureError

                                        sig_errors2: list[SignatureError] = []
                                        typ = py_type_to_type_dict(
                                            py_type,
                                            known_classes,
                                            sig_errors2,
                                            stmt_lineno,
                                            0,
                                        )
                                        typ = _unwrap_field_type(typ)
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
                j += 1
        i += 1
    if len(info.init_params) == 0:
        for fk in info.fields:
            info.fields[fk].has_default = True
    if len(info.init_params) > 0:
        info.needs_constructor = True
    elif has_computed_init:
        info.needs_constructor = True


# ---------------------------------------------------------------------------
# Class field collection
# ---------------------------------------------------------------------------


def _collect_class_fields(
    node: ASTNode,
    known_classes: set[str],
    node_classes: set[str],
    hierarchy_roots: set[str],
    func_return_types: dict[str, str],
    result: FieldResult,
) -> None:
    """Collect fields from a single class body and __init__."""
    class_name = get_str(node, "name")
    lineno = get_int(node, "lineno")
    info = ClassInfo(class_name)
    is_dc, kw_only = _is_dataclass_class(node)
    info.is_dataclass = is_dc
    info.kw_only = kw_only
    seen_fields: set[str] = set()
    body = get_nodes(node, "body")
    i = 0
    while i < len(body):
        stmt = body[i]
        if _is_type(stmt, ["AnnAssign"]):
            target = get_node(stmt, "target")
            if _is_type(target, ["Name"]):
                field_name = get_str(target, "id")
                if field_name != "":
                    if field_name in seen_fields:
                        result.add_error(
                            lineno, 0, "field '" + field_name + "' already declared"
                        )
                        return
                    seen_fields.add(field_name)
                    ann = get_node(stmt, "annotation")
                    py_type = annotation_to_str(ann)
                    from .signatures import SignatureError

                    sig_errors: list[SignatureError] = []
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
                        has_default = True
                        default_expr = _make_default_expr(value_node)
                    info.fields[field_name] = FieldInfo(
                        name=field_name,
                        typ=typ,
                        py_name=field_name,
                        has_default=has_default,
                        default=default_expr,
                    )
        i += 1
    has_init = False
    i = 0
    while i < len(body):
        stmt = body[i]
        if _is_type(stmt, ["FunctionDef"]) and get_str(stmt, "name") == "__init__":
            has_init = True
            _collect_init_fields(
                stmt, info, known_classes, func_return_types, result._errors
            )
            if len(result._errors) > 0:
                return
        i += 1
    if is_dc and not has_init:
        fkeys = list(info.fields.keys())
        j = 0
        while j < len(fkeys):
            info.init_params.append(fkeys[j])
            j += 1
    known_field_set: set[str] = set()
    fkeys = list(info.fields.keys())
    j = 0
    while j < len(fkeys):
        known_field_set.add(fkeys[j])
        j += 1
    ckeys = list(info.const_fields.keys())
    j = 0
    while j < len(ckeys):
        known_field_set.add(ckeys[j])
        j += 1
    i = 0
    while i < len(body):
        stmt = body[i]
        if _is_type(stmt, ["FunctionDef"]) and get_str(stmt, "name") != "__init__":
            bad = _check_no_new_fields_outside_init(stmt, known_field_set)
            if bad is not None:
                result.add_error(
                    lineno, 0, "field '" + bad + "' must be assigned in __init__"
                )
                return
        i += 1
    if has_init and class_name not in hierarchy_roots:
        if "kind" not in info.const_fields:
            kind_from_param = False
            j = 0
            while j < len(info.init_params):
                p = info.init_params[j]
                mapped = info.param_to_field.get(p)
                if mapped == "kind" or p == "kind":
                    kind_from_param = True
                j += 1
            if not kind_from_param:
                info.const_fields["kind"] = _pascal_to_kebab(class_name)
    if class_name in node_classes:
        fkeys = list(info.fields.keys())
        j = 0
        while j < len(fkeys):
            fname = fkeys[j]
            if fname not in result.field_to_structs:
                result.field_to_structs[fname] = []
            if class_name not in result.field_to_structs[fname]:
                result.field_to_structs[fname].append(class_name)
            j += 1
    result.classes[class_name] = info


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def collect_fields(
    tree: ASTNode,
    known_classes: set[str],
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
    fkeys = list(sig_result.functions.keys())
    i = 0
    while i < len(fkeys):
        func = sig_result.functions[fkeys[i]]
        func_return_types[fkeys[i]] = func.return_py_type
        i += 1
    result = FieldResult()
    body = get_nodes(tree, "body")
    i = 0
    while i < len(body):
        node = body[i]
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
            ei = err_before
            while ei < len(result._errors):
                result._errors[ei].source_file = sf
                ei += 1
            if len(result._errors) > 0:
                return result
        i += 1
    return result
