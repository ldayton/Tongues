"""Syntax verification, name binding, and derived class metadata."""

from typing import Callable

from .cfg import build_cfg, FlowGraph
from .typecollect import annotation_to_str
from .types import (
    JStr,
    JBool,
    JDict,
    JList,
    JNull,
    ASTNode,
    get_str,
    get_int,
    get_bool,
    get_node,
    get_nodes,
    get_jlist,
    has_key,
)


def get_children(node: ASTNode) -> list[ASTNode]:
    """Get all child nodes from a dict-based AST node."""
    children: list[ASTNode] = []
    keys = list(node.keys())
    i = 0
    while i < len(keys):
        key = keys[i]
        if key.startswith("_") or key in (
            "lineno",
            "col_offset",
            "end_lineno",
            "end_col_offset",
        ):
            i += 1
            continue
        val = node[key]
        if isinstance(val, JDict) and has_key(val.entries, "_type"):
            children.append(val.entries)
        elif isinstance(val, JList):
            j = 0
            while j < len(val.items):
                item = val.items[j]
                if isinstance(item, JDict) and has_key(item.entries, "_type"):
                    children.append(item.entries)
                j += 1
        i += 1
    return children


def walk(node: ASTNode, visitor: Callable[[ASTNode], None]) -> None:
    """Walk dict-based AST, calling visitor on each node."""
    visitor(node)
    children = get_children(node)
    i = 0
    while i < len(children):
        walk(children[i], visitor)
        i += 1


def _has_present(node: ASTNode, key: str) -> bool:
    """Check if key exists and is not JNull (for optional AST fields)."""
    v = node.get(key)
    if v is None:
        return False
    return not isinstance(v, JNull)


class Violation:
    """A subset violation with location and diagnostic."""

    def __init__(
        self,
        lineno: int,
        col: int,
        category: str,
        message: str,
        is_warning: bool,
        source_file: str = "",
    ):
        self.lineno: int = lineno
        self.col: int = col
        self.category: str = category
        self.message: str = message
        self.is_warning: bool = is_warning
        self.source_file: str = source_file

    def __repr__(self) -> str:
        prefix = "warning" if self.is_warning else "error"
        file_prefix = ""
        if self.source_file != "":
            file_prefix = self.source_file + ":"
        return (
            file_prefix
            + prefix
            + ":"
            + str(self.lineno)
            + ":"
            + str(self.col)
            + ": ["
            + self.category
            + "] "
            + self.message
        )


class VerifyResult:
    """Result of subset verification."""

    def __init__(self) -> None:
        self.violations: list[Violation] = []

    def add_error(
        self, lineno: int, col: int, category: str, message: str, source_file: str = ""
    ) -> None:
        self.violations.append(
            Violation(lineno, col, category, message, False, source_file)
        )

    def add_warning(
        self, lineno: int, col: int, category: str, message: str, source_file: str = ""
    ) -> None:
        self.violations.append(
            Violation(lineno, col, category, message, True, source_file)
        )

    def errors(self) -> list[Violation]:
        result: list[Violation] = []
        i = 0
        while i < len(self.violations):
            v = self.violations[i]
            if not v.is_warning:
                result.append(v)
            i += 1
        return result

    def warnings(self) -> list[Violation]:
        result: list[Violation] = []
        i = 0
        while i < len(self.violations):
            v = self.violations[i]
            if v.is_warning:
                result.append(v)
            i += 1
        return result

    def ok(self) -> bool:
        return len(self.errors()) == 0


# Builtins that are explicitly banned
BANNED_BUILTINS: set[str] = {
    "getattr",
    "setattr",
    "hasattr",
    "delattr",
    "type",
    "vars",
    "dir",
    "globals",
    "locals",
    "id",
    "callable",
    "eval",
    "exec",
    "compile",
    "__import__",
    "issubclass",
    "hash",
    "format",
    "memoryview",
    "iter",
    "next",
    "map",
    "filter",
    "input",
    "breakpoint",
    "help",
    "exit",
    "quit",
    "staticmethod",
    "classmethod",
    "property",
    "complex",
    "aiter",
    "anext",
    "TypeVar",
}

# Node types that are completely banned
BANNED_NODES: set[str] = {
    "AsyncFunctionDef",
    "AsyncFor",
    "AsyncWith",
    "Await",
    "Lambda",
    "Global",
    "Nonlocal",
    "TypeAlias",
    "TryStar",
}

# Functions that eagerly consume generator expressions
EAGER_CONSUMERS: set[str] = {
    "tuple",
    "list",
    "dict",
    "set",
    "frozenset",
    "any",
    "all",
    "sum",
    "min",
    "max",
    "sorted",
}

# Type names that are banned
BANNED_TYPE_NAMES: set[str] = {"Any"}

# Modules that can only be imported as `import X`, not `from X import ...`
IMPORT_ONLY_MODULES: set[str] = {"sys", "os"}

# Modules allowed in `from X import Y` (non-relative)
ALLOWED_FROM_MODULES: set[str] = {
    "typing",
    "dataclasses",
    "collections.abc",
    "__future__",
}

# Restricted builtin keyword arguments: {func_name: {banned_kwarg_names}}
RESTRICTED_KWARGS: dict[str, set[str]] = {
    "min": {"key", "default"},
    "max": {"key", "default"},
    "sorted": {"key"},
    "print": {"sep"},
}

# Bare collection types that need type parameters
BARE_COLLECTION_TYPES: set[str] = {"list", "dict", "set", "tuple", "frozenset"}

# Allowed dunder methods
ALLOWED_DUNDERS: set[str] = {"__init__", "__new__", "__repr__"}


def is_bare_collection(annotation: ASTNode | None) -> bool:
    """Check if annotation is a bare collection type without parameters."""
    if annotation is None:
        return False
    if get_str(annotation, "_type") != "Name":
        return False
    name_id = get_str(annotation, "id")
    if name_id == "":
        return False
    return name_id in BARE_COLLECTION_TYPES


def is_none_constant(node: ASTNode) -> bool:
    """Check if node is None constant."""
    if get_str(node, "_type") != "Constant":
        return False
    val = node.get("value")
    return isinstance(val, JNull)


def is_singleton_constant(node: ASTNode) -> bool:
    """Check if node is a None/True/False singleton constant."""
    if get_str(node, "_type") != "Constant":
        return False
    val = node.get("value")
    return isinstance(val, JNull) or isinstance(val, JBool)


def get_attr_name(node: ASTNode) -> str | None:
    """Get attr from Attribute node."""
    if get_str(node, "_type") == "Attribute":
        val = get_str(node, "attr")
        if val != "":
            return val
    return None


def is_allowed_dataclass_args(keywords: list[ASTNode]) -> bool:
    """Check if dataclass args are only eq=True, unsafe_hash=True, or kw_only=True."""
    allowed: set[str] = {"eq", "unsafe_hash", "kw_only"}
    i = 0
    while i < len(keywords):
        kw = keywords[i]
        arg = get_str(kw, "arg")
        if arg not in allowed:
            return False
        value = get_node(kw, "value")
        if get_str(value, "_type") != "Constant" or get_bool(value, "value") != True:  # noqa: E712
            return False
        i += 1
    return True


def collect_annotated_fields(class_node: ASTNode) -> set[str]:
    """Collect all field names with type annotations in a class (including in methods)."""
    fields: set[str] = set()
    nodes_to_visit: list[ASTNode] = [class_node]
    i = 0
    while i < len(nodes_to_visit):
        node = nodes_to_visit[i]
        node_type = get_str(node, "_type")
        if node_type == "AnnAssign":
            target = get_node(node, "target")
            target_type = get_str(target, "_type")
            # Class-level: x: int = 0
            if target_type == "Name":
                target_id = get_str(target, "id")
                if target_id != "":
                    fields.add(target_id)
            # Method-level: self.x: int = 0
            if target_type == "Attribute":
                target_value = get_node(target, "value")
                if get_name_id(target_value) == "self":
                    attr = get_str(target, "attr")
                    if attr != "":
                        fields.add(attr)
        # Add children to visit
        children = get_children(node)
        j = 0
        while j < len(children):
            nodes_to_visit.append(children[j])
            j += 1
        i += 1
    return fields


class Verifier:
    """Visitor that checks Tongues subset constraints."""

    def __init__(self) -> None:
        self.result: VerifyResult = VerifyResult()
        self.in_class: bool = False
        self.class_name: str = ""
        self.in_function: bool = False
        self.function_name: str = ""
        self.annotated_params: set[str] = set()
        self.annotated_fields: set[str] = set()
        self.annotated_locals: set[str] = set()
        # Context flags for eager iteration
        self.in_eager_consumer: bool = False
        self.in_for_iter: bool = False
        self.in_for_body: bool = False  # For structural recursion (yield allowed)
        self.in_file_open: bool = False  # Inside validated with-open block
        # Variables guarded by `if var:` condition (for tuple unpacking)
        self.guarded_vars: set[str] = set()

    def error(self, node: ASTNode, category: str, message: str) -> None:
        lineno = get_int(node, "lineno")
        col = get_int(node, "col_offset")
        source_file = get_str(node, "_source_file")
        self.result.add_error(lineno, col, category, message, source_file)

    def warning(self, node: ASTNode, category: str, message: str) -> None:
        lineno = get_int(node, "lineno")
        col = get_int(node, "col_offset")
        source_file = get_str(node, "_source_file")
        self.result.add_warning(lineno, col, category, message, source_file)

    def visit(self, node: ASTNode) -> None:
        """Dispatch to appropriate visit method."""
        node_type = get_str(node, "_type")
        # Check banned nodes first
        if node_type in BANNED_NODES:
            self.visit_banned_node(node, node_type)
            return
        # Explicit dispatch (self-hosting: no getattr)
        if node_type == "Module":
            self.visit_Module(node)
        elif node_type == "FunctionDef":
            self.visit_FunctionDef(node)
        elif node_type == "ClassDef":
            self.visit_ClassDef(node)
        elif node_type == "Call":
            self.visit_Call(node)
        elif node_type == "Compare":
            self.visit_Compare(node)
        elif node_type == "BoolOp":
            self.visit_BoolOp(node)
        elif node_type == "Assign":
            self.visit_Assign(node)
        elif node_type == "AnnAssign":
            self.visit_AnnAssign(node)
        elif node_type == "For":
            self.visit_For(node)
        elif node_type == "While":
            self.visit_While(node)
        elif node_type == "If":
            self.visit_If(node)
        elif node_type == "Try":
            self.visit_Try(node)
        elif node_type == "ExceptHandler":
            self.visit_ExceptHandler(node)
        elif node_type == "Import":
            self.visit_Import(node)
        elif node_type == "ImportFrom":
            self.visit_ImportFrom(node)
        elif node_type == "Attribute":
            self.visit_Attribute(node)
        elif node_type == "BinOp":
            self.visit_BinOp(node)
        elif node_type == "Delete":
            self.visit_Delete(node)
        elif node_type == "JoinedStr":
            self.visit_JoinedStr(node)
        elif node_type == "Constant":
            self.visit_Constant(node)
        elif node_type == "FormattedValue":
            self.visit_FormattedValue(node)
        elif node_type == "GeneratorExp":
            self.visit_GeneratorExp(node)
        elif node_type == "ListComp":
            self.visit_ListComp(node)
        elif node_type == "SetComp":
            self.visit_SetComp(node)
        elif node_type == "DictComp":
            self.visit_DictComp(node)
        elif node_type == "Name":
            self.visit_Name(node)
        elif node_type == "Yield":
            self.visit_Yield(node)
        elif node_type == "YieldFrom":
            self.visit_YieldFrom(node)
        elif node_type == "With":
            self.visit_With(node)
        elif node_type == "Match":
            pass
        else:
            # Unknown node types get a warning
            if node_type and not self.is_known_node(node_type):
                self.warning(node, "syntax", "unknown node type: " + node_type)
            # Still traverse children for all nodes
            children = get_children(node)
            i = 0
            while i < len(children):
                self.visit(children[i])
                i += 1

    def is_known_node(self, node_type: str) -> bool:
        """Check if node type is recognized."""
        known: set[str] = {
            "Module",
            "FunctionDef",
            "ClassDef",
            "Return",
            "Assign",
            "AnnAssign",
            "AugAssign",
            "For",
            "While",
            "If",
            "Raise",
            "Try",
            "ExceptHandler",
            "Import",
            "ImportFrom",
            "Pass",
            "Break",
            "Continue",
            "Expr",
            "BoolOp",
            "BinOp",
            "UnaryOp",
            "IfExp",
            "Dict",
            "Set",
            "ListComp",
            "SetComp",
            "DictComp",
            "GeneratorExp",
            "Compare",
            "Call",
            "JoinedStr",
            "FormattedValue",
            "Constant",
            "Attribute",
            "Subscript",
            "Starred",
            "Name",
            "List",
            "Tuple",
            "Slice",
            "And",
            "Or",
            "Add",
            "Sub",
            "Mult",
            "Div",
            "Mod",
            "Pow",
            "LShift",
            "RShift",
            "BitOr",
            "BitXor",
            "BitAnd",
            "FloorDiv",
            "Invert",
            "Not",
            "UAdd",
            "USub",
            "Eq",
            "NotEq",
            "Lt",
            "LtE",
            "Gt",
            "GtE",
            "Is",
            "IsNot",
            "In",
            "NotIn",
            "arg",
            "arguments",
            "keyword",
            "alias",
            "Match",
            "match_case",
            "MatchValue",
            "MatchSingleton",
            "MatchSequence",
            "MatchMapping",
            "MatchClass",
            "MatchStar",
            "MatchAs",
            "MatchOr",
            "Assert",
            "Yield",
            "YieldFrom",
            "Load",
            "Store",
            "Del",
        }
        return node_type in known

    def visit_banned_node(self, node: ASTNode, node_type: str) -> None:
        """Report error for banned node type."""
        category = "syntax"
        message = node_type + " is not allowed"
        if node_type in ("AsyncFunctionDef", "AsyncFor", "AsyncWith", "Await"):
            category = "async"
            message = "async/await is not allowed"
        elif node_type == "GeneratorExp":
            category = "generator"
            message = "generator expression only allowed in eager consumer"
        elif node_type == "With":
            category = "control"
            message = "with statement: use try/finally instead"
        elif node_type == "Lambda":
            category = "function"
            message = "lambda: use named function instead"
        elif node_type in ("Global", "Nonlocal"):
            category = "control"
            message = node_type.lower() + ": pass as parameter instead"
        self.error(node, category, message)

    def visit_Module(self, node: ASTNode) -> None:
        """Visit module - just traverse body."""
        body = get_nodes(node, "body")
        i = 0
        while i < len(body):
            self.visit(body[i])
            i += 1

    def visit_FunctionDef(self, node: ASTNode) -> None:
        """Check function definition constraints."""
        name = get_str(node, "name")
        # Check decorators
        decorators = get_nodes(node, "decorator_list")
        i = 0
        while i < len(decorators):
            dec = decorators[i]
            dec_name = get_name_id(dec)
            if dec_name is None:
                dec_name = get_attr_name(dec)
            if dec_name == "staticmethod":
                self.error(node, "function", "@staticmethod: use module-level function")
            elif dec_name == "classmethod":
                self.error(node, "function", "@classmethod: use module-level function")
            elif dec_name == "property":
                self.error(node, "function", "@property: use explicit getter method")
            else:
                self.error(node, "function", "decorators are not allowed")
            i += 1
        # Check nested function
        if self.in_function:
            self.error(
                node,
                "function",
                "nested function '" + name + "': define at module level",
            )
        # Check dunder methods
        if name.startswith("__") and name.endswith("__"):
            if name not in ALLOWED_DUNDERS:
                self.error(
                    node,
                    "function",
                    "dunder method "
                    + name
                    + ": only __init__/__new__/__repr__ allowed",
                )
        # Check *args and **kwargs
        args_node = get_node(node, "args")
        if _has_present(args_node, "vararg"):
            self.error(node, "function", "*args: use explicit parameters")
        if _has_present(args_node, "kwarg"):
            self.error(node, "function", "**kwargs: use explicit parameters")
        # Check return type (except __init__, __new__)
        if name not in ("__init__", "__new__"):
            if not _has_present(node, "returns"):
                self.error(node, "types", "missing type annotation for '" + name + "'")
        # Check return type bare collection and visit return annotation
        if _has_present(node, "returns"):
            returns = get_node(node, "returns")
            if is_bare_collection(returns):
                self.error(
                    node,
                    "types",
                    "bare "
                    + get_str(returns, "id")
                    + ": "
                    + name
                    + "() return needs type parameter",
                )
            self.visit(returns)
        # Check parameter types
        args_list = get_nodes(args_node, "args")
        # Also check keyword-only args
        kwonlyargs = get_nodes(args_node, "kwonlyargs")
        # Also check positional-only args
        posonlyargs = get_nodes(args_node, "posonlyargs")
        all_args: list[ASTNode] = []
        ai = 0
        while ai < len(posonlyargs):
            all_args.append(posonlyargs[ai])
            ai += 1
        ai = 0
        while ai < len(args_list):
            all_args.append(args_list[ai])
            ai += 1
        ai = 0
        while ai < len(kwonlyargs):
            all_args.append(kwonlyargs[ai])
            ai += 1
        old_annotated = self.annotated_params
        self.annotated_params = set()
        j = 0
        while j < len(all_args):
            arg = all_args[j]
            arg_name = get_str(arg, "arg")
            has_annotation = _has_present(arg, "annotation")
            # Skip self/cls first param
            if j == 0 and arg_name in ("self", "cls"):
                j += 1
                continue
            if not has_annotation:
                self.error(
                    node,
                    "types",
                    "missing type annotation for '" + arg_name + "'",
                )
            else:
                annotation = get_node(arg, "annotation")
                self.annotated_params.add(arg_name)
                self.visit(annotation)
                if is_bare_collection(annotation):
                    self.error(
                        node,
                        "types",
                        "bare "
                        + get_str(annotation, "id")
                        + ": "
                        + arg_name
                        + " needs type parameter",
                    )
            j += 1
        # Check mutable defaults
        defaults = get_nodes(args_node, "defaults")
        kw_defaults = get_jlist(args_node, "kw_defaults")
        all_defaults: list[ASTNode] = []
        k = 0
        while k < len(defaults):
            all_defaults.append(defaults[k])
            k += 1
        k = 0
        while k < len(kw_defaults):
            kw_d = kw_defaults[k]
            if isinstance(kw_d, JDict):
                all_defaults.append(kw_d.entries)
            k += 1
        k = 0
        while k < len(all_defaults):
            d = all_defaults[k]
            d_type = get_str(d, "_type")
            if d_type in ("List", "Dict", "Set"):
                self.error(
                    node,
                    "function",
                    "mutable default argument: use None and initialize in body",
                )
            elif d_type == "Lambda":
                self.error(node, "function", "lambda: use named function instead")
            k += 1
        # Visit body
        old_in_function = self.in_function
        old_function_name = self.function_name
        old_annotated_locals = self.annotated_locals
        self.in_function = True
        self.function_name = name
        self.annotated_locals = set()
        body = get_nodes(node, "body")
        m = 0
        while m < len(body):
            self.visit(body[m])
            m += 1
        self.in_function = old_in_function
        self.function_name = old_function_name
        self.annotated_params = old_annotated
        self.annotated_locals = old_annotated_locals

    def visit_ClassDef(self, node: ASTNode) -> None:
        """Check class definition constraints."""
        name = get_str(node, "name")
        # Check decorators - only @dataclass (no arguments) is allowed
        decorators = get_nodes(node, "decorator_list")
        i = 0
        while i < len(decorators):
            dec = decorators[i]
            dec_type = get_str(dec, "_type")
            if dec_type == "Name" and get_str(dec, "id") == "dataclass":
                pass  # @dataclass with no arguments is allowed
            elif dec_type == "Call":
                func = get_node(dec, "func")
                if len(func) > 0 and get_str(func, "id") == "dataclass":
                    keywords = get_nodes(dec, "keywords")
                    if not is_allowed_dataclass_args(keywords):
                        self.error(
                            node,
                            "class",
                            "@dataclass: only eq=True and unsafe_hash=True allowed",
                        )
                else:
                    self.error(node, "class", "class decorator not allowed")
            else:
                self.error(node, "class", "class decorator not allowed")
            i += 1
        # Check nested class
        if self.in_class:
            self.error(node, "class", "nested class: define at module level")
        # Check multiple inheritance (Exception doesn't count)
        bases = get_nodes(node, "bases")
        real_bases: list[ASTNode] = []
        j = 0
        while j < len(bases):
            b = bases[j]
            b_name = get_name_id(b)
            if b_name != "Exception":
                real_bases.append(b)
            j += 1
        if len(real_bases) > 1:
            self.error(node, "class", "multiple inheritance: use single base class")
        # Collect annotated fields (walk entire class including method bodies)
        old_fields = self.annotated_fields
        self.annotated_fields = collect_annotated_fields(node)
        body = get_nodes(node, "body")
        # Visit body
        old_in_class = self.in_class
        old_class_name = self.class_name
        self.in_class = True
        self.class_name = name
        m = 0
        while m < len(body):
            child = body[m]
            # Check nested class
            if get_str(child, "_type") == "ClassDef":
                self.error(child, "class", "nested class: define at module level")
            self.visit(child)
            m += 1
        self.in_class = old_in_class
        self.class_name = old_class_name
        self.annotated_fields = old_fields

    def visit_Call(self, node: ASTNode) -> None:
        """Check function call constraints."""
        func = get_node(node, "func")
        func_name = get_name_id(func)
        # Check banned builtins
        if func_name is not None and func_name in BANNED_BUILTINS:
            self.error(node, "builtin", func_name + "() is not allowed")
        # open() only allowed inside validated with-open
        if func_name == "open" and not self.in_file_open:
            self.error(node, "builtin", "open() only allowed in with-open idiom")
        # Check enumerate/zip only allowed in for-loop iter or eager consumer
        if (
            func_name is not None
            and func_name in ("enumerate", "zip")
            and not self.in_for_iter
            and not self.in_eager_consumer
        ):
            self.error(
                node,
                "builtin",
                func_name + "() only allowed in for-loop header or eager consumer",
            )
        # Check if this is an eager consumer (for generator expressions)
        is_eager = func_name is not None and func_name in EAGER_CONSUMERS
        # Also check for str.join method call
        if not is_eager and len(func) > 0 and get_str(func, "_type") == "Attribute":
            if get_str(func, "attr") == "join":
                is_eager = True
        # Check restricted keyword arguments (min/max key/default, sorted key, print sep)
        keywords = get_nodes(node, "keywords")
        if func_name is not None and func_name in RESTRICTED_KWARGS:
            banned_kwargs = RESTRICTED_KWARGS[func_name]
            j = 0
            while j < len(keywords):
                kw = keywords[j]
                kw_arg = get_str(kw, "arg")
                if kw_arg != "" and kw_arg in banned_kwargs:
                    self.error(
                        node,
                        "builtin",
                        func_name + "() does not allow " + kw_arg + "= argument",
                    )
                j += 1
        # Check print: only one positional argument
        args = get_nodes(node, "args")
        if func_name == "print" and len(args) > 1:
            self.error(
                node,
                "builtin",
                "print() takes one value; use f-string or str concatenation",
            )
        # Check field(default_factory=...)
        if func_name == "field":
            j = 0
            while j < len(keywords):
                kw = keywords[j]
                if get_str(kw, "arg") == "default_factory":
                    self.error(
                        node,
                        "class",
                        "field(default_factory=...) not allowed: use simple defaults",
                    )
                j += 1
        # Check *args in call
        i = 0
        while i < len(args):
            arg = args[i]
            if get_str(arg, "_type") == "Starred":
                self.error(
                    node, "expression", "*args in call: unpack arguments explicitly"
                )
                break
            i += 1
        # Check **kwargs in call
        j = 0
        while j < len(keywords):
            kw = keywords[j]
            if not _has_present(kw, "arg"):
                self.error(
                    node, "expression", "**kwargs in call: pass arguments explicitly"
                )
                break
            j += 1
        # Visit children
        self.visit(func)
        # Set eager consumer context when visiting args
        old_in_eager = self.in_eager_consumer
        if is_eager:
            self.in_eager_consumer = True
        k = 0
        while k < len(args):
            self.visit(args[k])
            k += 1
        self.in_eager_consumer = old_in_eager
        m = 0
        while m < len(keywords):
            kw = keywords[m]
            if _has_present(kw, "value"):
                self.visit(get_node(kw, "value"))
            m += 1

    def visit_Compare(self, node: ASTNode) -> None:
        """Check comparison constraints."""
        ops = get_nodes(node, "ops")
        comparators = get_nodes(node, "comparators")
        # Check is/is not — only allowed with None/True/False singletons
        left = get_node(node, "left")
        i = 0
        while i < len(ops):
            op = ops[i]
            comparator = comparators[i]
            op_type = get_str(op, "_type")
            if op_type in ("Is", "IsNot"):
                if not is_singleton_constant(left) and not is_singleton_constant(
                    comparator
                ):
                    self.error(
                        node,
                        "reflection",
                        "is/is not only allowed with None/True/False",
                    )
            left = comparator
            i += 1
        # Visit children
        self.visit(get_node(node, "left"))
        j = 0
        while j < len(comparators):
            self.visit(comparators[j])
            j += 1

    def visit_BoolOp(self, node: ASTNode) -> None:
        """Check boolean operation constraints."""
        values = get_nodes(node, "values")
        j = 0
        while j < len(values):
            self.visit(values[j])
            j += 1

    def visit_Assign(self, node: ASTNode) -> None:
        """Check assignment constraints."""
        targets = get_nodes(node, "targets")
        value = get_node(node, "value")
        # Check tuple unpack from variable (allowed if guarded by `if var:`)
        if len(targets) == 1:
            target = targets[0]
            if (
                get_str(target, "_type") == "Tuple"
                and get_str(value, "_type") == "Name"
            ):
                var_name = get_str(value, "id")
                if var_name not in self.guarded_vars:
                    self.error(
                        node,
                        "expression",
                        "tuple unpack from variable: unpack directly from call",
                    )
        # Visit children
        j = 0
        while j < len(targets):
            self.visit(targets[j])
            j += 1
        self.visit(value)

    def visit_AnnAssign(self, node: ASTNode) -> None:
        """Check annotated assignment constraints."""
        target = get_node(node, "target")
        # Track annotated local variables
        if self.in_function and get_str(target, "_type") == "Name":
            target_name = get_str(target, "id")
            if target_name != "":
                self.annotated_locals.add(target_name)
        # Check bare collection
        if _has_present(node, "annotation"):
            annotation = get_node(node, "annotation")
            if is_bare_collection(annotation):
                t_name = get_str(target, "id")
                if t_name == "":
                    t_name = "?"
                self.error(
                    node,
                    "types",
                    "bare "
                    + get_str(annotation, "id")
                    + ": "
                    + t_name
                    + " needs type parameter",
                )
            # Visit children
            self.visit(target)
            self.visit(annotation)
        else:
            self.visit(target)
        if _has_present(node, "value"):
            self.visit(get_node(node, "value"))

    def _is_valid_file_open(self, node: ASTNode) -> bool:
        """Check if a With node is the allowed with-open file I/O idiom."""
        items = get_nodes(node, "items")
        if len(items) != 1:
            return False
        item = items[0]
        ctx_expr = get_node(item, "context_expr")
        if not _has_present(item, "optional_vars"):
            return False
        opt_vars = get_node(item, "optional_vars")
        if get_name_id(get_node(ctx_expr, "func")) != "open":
            return False
        if get_name_id(opt_vars) is None:
            return False
        handle = get_name_id(opt_vars)
        args = get_nodes(ctx_expr, "args")
        if len(args) != 2:
            return False
        mode_node = args[1]
        if get_str(mode_node, "_type") != "Constant":
            return False
        mode = get_str(mode_node, "value")
        if mode not in ("rb", "w", "wb"):
            return False
        body = get_nodes(node, "body")
        if len(body) != 1:
            return False
        stmt = body[0]
        if mode == "rb":
            if get_str(stmt, "_type") != "Assign":
                return False
            val = get_node(stmt, "value")
            if get_str(val, "_type") != "Call":
                return False
            func = get_node(val, "func")
            if get_str(func, "_type") != "Attribute" or get_str(func, "attr") != "read":
                return False
            if get_name_id(get_node(func, "value")) != handle:
                return False
            if len(get_nodes(val, "args")) != 0:
                return False
        else:
            if get_str(stmt, "_type") != "Expr":
                return False
            call = get_node(stmt, "value")
            if get_str(call, "_type") != "Call":
                return False
            func = get_node(call, "func")
            if (
                get_str(func, "_type") != "Attribute"
                or get_str(func, "attr") != "write"
            ):
                return False
            if get_name_id(get_node(func, "value")) != handle:
                return False
            if len(get_nodes(call, "args")) != 1:
                return False
        return True

    def visit_With(self, node: ASTNode) -> None:
        """Only allow with-open file I/O idiom; reject all other with statements."""
        if not self._is_valid_file_open(node):
            self.error(
                node, "with", "with statement: only with-open file I/O is allowed"
            )
            return
        old = self.in_file_open
        self.in_file_open = True
        items = get_nodes(node, "items")
        ctx_expr = get_node(items[0], "context_expr")
        self.visit(ctx_expr)
        body = get_nodes(node, "body")
        i = 0
        while i < len(body):
            self.visit(body[i])
            i += 1
        self.in_file_open = old

    def visit_For(self, node: ASTNode) -> None:
        """Check for loop constraints."""
        # Check loop else
        orelse = get_nodes(node, "orelse")
        if len(orelse) > 0:
            self.error(node, "control", "loop else: use flag variable instead")
        # Visit children
        if _has_present(node, "target"):
            self.visit(get_node(node, "target"))
        if _has_present(node, "iter"):
            # Set context flag for enumerate/zip in for-loop iter
            old_in_for_iter = self.in_for_iter
            self.in_for_iter = True
            self.visit(get_node(node, "iter"))
            self.in_for_iter = old_in_for_iter
        body = get_nodes(node, "body")
        old_in_for_body = self.in_for_body
        self.in_for_body = True
        i = 0
        while i < len(body):
            self.visit(body[i])
            i += 1
        self.in_for_body = old_in_for_body
        j = 0
        while j < len(orelse):
            self.visit(orelse[j])
            j += 1

    def visit_While(self, node: ASTNode) -> None:
        """Check while loop constraints."""
        # Check loop else
        orelse = get_nodes(node, "orelse")
        if len(orelse) > 0:
            self.error(node, "control", "loop else: use flag variable instead")
        # Visit children
        if _has_present(node, "test"):
            self.visit(get_node(node, "test"))
        body = get_nodes(node, "body")
        i = 0
        while i < len(body):
            self.visit(body[i])
            i += 1
        j = 0
        while j < len(orelse):
            self.visit(orelse[j])
            j += 1

    def visit_Name(self, node: ASTNode) -> None:
        """Check for banned type names like Any."""
        name_id = get_str(node, "id")
        if name_id in BANNED_TYPE_NAMES:
            self.error(
                node,
                "types",
                name_id + " is not allowed: use object + isinstance() instead",
            )

    def visit_Yield(self, node: ASTNode) -> None:
        """Check yield - allowed only in for-loop body (structural recursion)."""
        if not self.in_for_body:
            self.error(
                node,
                "generator",
                "yield only allowed in for-loop body (structural recursion)",
            )
        # Visit the yielded value
        if _has_present(node, "value"):
            self.visit(get_node(node, "value"))

    def visit_YieldFrom(self, node: ASTNode) -> None:
        """Check yield from - allowed only in for-loop body (structural recursion)."""
        if not self.in_for_body:
            self.error(
                node,
                "generator",
                "yield from only allowed in for-loop body (structural recursion)",
            )
        # Visit the yielded value
        if _has_present(node, "value"):
            self.visit(get_node(node, "value"))

    def visit_If(self, node: ASTNode) -> None:
        """Check if statement and track guarded variables for tuple unpacking."""
        has_test = _has_present(node, "test")
        body = get_nodes(node, "body")
        orelse = get_nodes(node, "orelse")
        # Check if condition guards a variable for tuple unpacking
        # Patterns: `if var:`, `if var is not None:`, `if (var := call()) is not None:`
        guarded_var: str | None = None
        if has_test:
            test = get_node(node, "test")
            test_type = get_str(test, "_type")
            if test_type == "Name":
                # Simple: `if var:`
                guarded_var = get_str(test, "id")
            elif test_type == "NamedExpr":
                # Walrus: `if (var := call()):`
                target = get_node(test, "target")
                if get_str(target, "_type") == "Name":
                    guarded_var = get_str(target, "id")
            elif test_type == "Compare":
                # Check for `var is not None` or `(var := ...) is not None`
                left = get_node(test, "left")
                ops = get_nodes(test, "ops")
                comparators = get_nodes(test, "comparators")
                if len(ops) == 1 and len(comparators) == 1:
                    op = ops[0]
                    comp = comparators[0]
                    if get_str(op, "_type") == "IsNot" and is_none_constant(comp):
                        # Left side is the guarded expression
                        left_type = get_str(left, "_type")
                        if left_type == "Name":
                            guarded_var = get_str(left, "id")
                        elif left_type == "NamedExpr":
                            # Walrus operator: (var := call()) is not None
                            target = get_node(left, "target")
                            if get_str(target, "_type") == "Name":
                                guarded_var = get_str(target, "id")
        # Visit condition
        if has_test:
            self.visit(get_node(node, "test"))
        # Visit then-branch with guarded variable in scope
        if guarded_var is not None and guarded_var != "":
            self.guarded_vars.add(guarded_var)
        i = 0
        while i < len(body):
            self.visit(body[i])
            i += 1
        if guarded_var is not None and guarded_var != "":
            self.guarded_vars.discard(guarded_var)
        # Visit else-branch (no guarding)
        j = 0
        while j < len(orelse):
            self.visit(orelse[j])
            j += 1

    def visit_Try(self, node: ASTNode) -> None:
        """Check try statement constraints."""
        # Check try else
        orelse = get_nodes(node, "orelse")
        if len(orelse) > 0:
            self.error(node, "control", "try else: move else code after try block")
        # Visit children
        body = get_nodes(node, "body")
        i = 0
        while i < len(body):
            self.visit(body[i])
            i += 1
        handlers = get_nodes(node, "handlers")
        j = 0
        while j < len(handlers):
            self.visit(handlers[j])
            j += 1
        k = 0
        while k < len(orelse):
            self.visit(orelse[k])
            k += 1
        finalbody = get_nodes(node, "finalbody")
        m = 0
        while m < len(finalbody):
            self.visit(finalbody[m])
            m += 1

    def visit_ExceptHandler(self, node: ASTNode) -> None:
        """Check except handler constraints."""
        # Check bare except
        if not _has_present(node, "type"):
            self.error(node, "control", "bare except: specify exception type")
        else:
            self.visit(get_node(node, "type"))
        body = get_nodes(node, "body")
        i = 0
        while i < len(body):
            self.visit(body[i])
            i += 1

    def visit_Import(self, node: ASTNode) -> None:
        """Check import constraints. Only 'import sys/os' allowed."""
        names = get_nodes(node, "names")
        i = 0
        while i < len(names):
            alias = names[i]
            name = get_str(alias, "name")
            asname = get_str(alias, "asname")
            if asname != "":
                self.error(
                    node,
                    "import",
                    "import " + name + " as " + asname + ": module aliases not allowed",
                )
            elif name not in IMPORT_ONLY_MODULES:
                self.error(
                    node,
                    "import",
                    "import " + name + ": not allowed, code must be self-contained",
                )
            i += 1

    def visit_ImportFrom(self, node: ASTNode) -> None:
        """Check from-import syntax: no stars, no from sys/os, no banned modules."""
        # Check for star imports
        import_names = get_nodes(node, "names")
        i = 0
        while i < len(import_names):
            alias = import_names[i]
            if get_str(alias, "name") == "*":
                self.error(node, "import", "star import: import names explicitly")
                return
            i += 1
        level = get_int(node, "level")
        if level > 0:
            return
        module = get_str(node, "module")
        if module == "":
            return
        top_module = module.split(".")[0]
        # sys/os can only be used with `import X`, not `from X import ...`
        if top_module in IMPORT_ONLY_MODULES:
            self.error(
                node,
                "import",
                "from " + module + " import: use 'import " + top_module + "' instead",
            )
            return
        # Only allowed stdlib modules can be from-imported
        if (
            module not in ALLOWED_FROM_MODULES
            and top_module not in ALLOWED_FROM_MODULES
        ):
            self.error(
                node,
                "import",
                "import of '" + module + "' not allowed",
            )

    def visit_Attribute(self, node: ASTNode) -> None:
        """Check attribute access constraints."""
        attr = get_str(node, "attr")
        # Check __class__
        if attr == "__class__":
            self.error(node, "reflection", "__class__: use isinstance() instead")
        # Check __dict__
        if attr == "__dict__":
            self.error(node, "reflection", "__dict__: direct attribute access only")
        # Visit value
        if _has_present(node, "value"):
            self.visit(get_node(node, "value"))

    def visit_BinOp(self, node: ASTNode) -> None:
        """Visit binary operation children."""
        # Visit children
        if _has_present(node, "left"):
            self.visit(get_node(node, "left"))
        if _has_present(node, "right"):
            self.visit(get_node(node, "right"))

    def visit_Delete(self, node: ASTNode) -> None:
        """Check delete statement - banned."""
        self.error(node, "syntax", "del: reassign or let variable go out of scope")

    def visit_JoinedStr(self, node: ASTNode) -> None:
        """Visit f-string, check children."""
        values = get_nodes(node, "values")
        i = 0
        while i < len(values):
            self.visit(values[i])
            i += 1

    def visit_Constant(self, node: ASTNode) -> None:
        """Check constant values - reject invalid Unicode in strings."""
        raw = node.get("value")
        if isinstance(raw, JStr):
            value = raw.value
            i = 0
            while i < len(value):
                code = ord(value[i])
                # Reject surrogate code points (not valid Unicode scalar values)
                if 0xD800 <= code <= 0xDFFF:
                    hex_str = hex(code)[2:].upper().zfill(4)
                    self.error(
                        node,
                        "string",
                        "surrogate code point U+"
                        + hex_str
                        + " not allowed in string literal",
                    )
                i += 1

    def visit_FormattedValue(self, node: ASTNode) -> None:
        """Check f-string replacement field: {expr} only, no !conv or :spec."""
        conversion = get_int(node, "conversion")
        if conversion != -1:
            self.error(node, "syntax", "f-string !conversion not supported")
        if _has_present(node, "format_spec"):
            self.error(node, "syntax", "f-string :format_spec not supported")
        if _has_present(node, "value"):
            self.visit(get_node(node, "value"))

    def visit_GeneratorExp(self, node: ASTNode) -> None:
        """Check generator expression - only allowed in eager consumer context."""
        if not self.in_eager_consumer:
            self.error(
                node,
                "generator",
                "generator expression only allowed in eager consumer (tuple, list, any, all, etc.)",
            )
        # Visit children (elt, generators)
        if _has_present(node, "elt"):
            self.visit(get_node(node, "elt"))
        generators = get_nodes(node, "generators")
        i = 0
        while i < len(generators):
            gen = generators[i]
            if _has_present(gen, "target"):
                self.visit(get_node(gen, "target"))
            if _has_present(gen, "iter"):
                self.visit(get_node(gen, "iter"))
            ifs = get_nodes(gen, "ifs")
            j = 0
            while j < len(ifs):
                self.visit(ifs[j])
                j += 1
            i += 1

    def visit_ListComp(self, node: ASTNode) -> None:
        """List comprehensions are eager - set context for enumerate/zip in generators."""
        old_in_eager = self.in_eager_consumer
        self.in_eager_consumer = True
        if _has_present(node, "elt"):
            self.visit(get_node(node, "elt"))
        generators = get_nodes(node, "generators")
        i = 0
        while i < len(generators):
            gen = generators[i]
            if _has_present(gen, "target"):
                self.visit(get_node(gen, "target"))
            if _has_present(gen, "iter"):
                self.visit(get_node(gen, "iter"))
            ifs = get_nodes(gen, "ifs")
            j = 0
            while j < len(ifs):
                self.visit(ifs[j])
                j += 1
            i += 1
        self.in_eager_consumer = old_in_eager

    def visit_SetComp(self, node: ASTNode) -> None:
        """Set comprehensions are eager - set context for enumerate/zip in generators."""
        old_in_eager = self.in_eager_consumer
        self.in_eager_consumer = True
        if _has_present(node, "elt"):
            self.visit(get_node(node, "elt"))
        generators = get_nodes(node, "generators")
        i = 0
        while i < len(generators):
            gen = generators[i]
            if _has_present(gen, "target"):
                self.visit(get_node(gen, "target"))
            if _has_present(gen, "iter"):
                self.visit(get_node(gen, "iter"))
            ifs = get_nodes(gen, "ifs")
            j = 0
            while j < len(ifs):
                self.visit(ifs[j])
                j += 1
            i += 1
        self.in_eager_consumer = old_in_eager

    def visit_DictComp(self, node: ASTNode) -> None:
        """Dict comprehensions are eager - set context for enumerate/zip in generators."""
        old_in_eager = self.in_eager_consumer
        self.in_eager_consumer = True
        if _has_present(node, "key"):
            self.visit(get_node(node, "key"))
        if _has_present(node, "value"):
            self.visit(get_node(node, "value"))
        generators = get_nodes(node, "generators")
        i = 0
        while i < len(generators):
            gen = generators[i]
            if _has_present(gen, "target"):
                self.visit(get_node(gen, "target"))
            if _has_present(gen, "iter"):
                self.visit(get_node(gen, "iter"))
            ifs = get_nodes(gen, "ifs")
            j = 0
            while j < len(ifs):
                self.visit(ifs[j])
                j += 1
            i += 1
        self.in_eager_consumer = old_in_eager


def verify(ast_dict: ASTNode) -> VerifyResult:
    """Verify dict-based AST conforms to Tongues subset.

    Args:
        ast_dict: Dict-based AST from parse.py

    Returns:
        VerifyResult with any violations found
    """
    verifier = Verifier()
    verifier.visit(ast_dict)
    return verifier.result


class ImportInfo:
    """Information about an import statement."""

    def __init__(self, module: str, level: int, lineno: int, col: int):
        self.module: str = module
        self.level: int = level
        self.lineno: int = lineno
        self.col: int = col


def extract_imports(ast_dict: ASTNode) -> list[ImportInfo]:
    """Extract all from-imports from an AST."""
    result: list[ImportInfo] = []
    body = get_nodes(ast_dict, "body")
    i = 0
    while i < len(body):
        node = body[i]
        if get_str(node, "_type") == "ImportFrom":
            module = get_str(node, "module")
            level = get_int(node, "level")
            lineno = get_int(node, "lineno")
            if lineno == 0:
                lineno = 1
            col = get_int(node, "col_offset")
            if module == "" and level > 0:
                # from . import X, Y - each name is a module
                names = get_nodes(node, "names")
                j = 0
                while j < len(names):
                    name_node = names[j]
                    name = get_str(name_node, "name")
                    if name != "" and name != "*":
                        result.append(ImportInfo(name, level, lineno, col))
                    j += 1
            else:
                result.append(ImportInfo(module, level, lineno, col))
        i += 1
    return result


class ProjectVerifyResult:
    """Result of project-level verification."""

    def __init__(self) -> None:
        self.file_results: dict[str, VerifyResult] = {}
        self.unresolved_imports: list[tuple[str, ImportInfo]] = []

    def errors(self) -> list[str]:
        """Get all errors as formatted strings."""
        result: list[str] = []
        files = sorted(self.file_results.keys())
        i = 0
        while i < len(files):
            f = files[i]
            file_result = self.file_results[f]
            errs = file_result.errors()
            j = 0
            while j < len(errs):
                result.append(f + ": " + str(errs[j]))
                j += 1
            i += 1
        j = 0
        while j < len(self.unresolved_imports):
            file_path, imp = self.unresolved_imports[j]
            msg = file_path + ":" + str(imp.lineno) + ":" + str(imp.col)
            msg = msg + ": [import] unresolved import: " + imp.module
            result.append(msg)
            j += 1
        return result

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        if len(self.unresolved_imports) > 0:
            return True
        files = list(self.file_results.keys())
        i = 0
        while i < len(files):
            if not self.file_results[files[i]].ok():
                return True
            i += 1
        return False


# Allowed builtins
ALLOWED_BUILTINS: set[str] = {
    # Math
    "abs",
    "min",
    "max",
    "sum",
    "round",
    "divmod",
    "pow",
    # Conversion
    "int",
    "float",
    "str",
    "bool",
    "bytes",
    "bytearray",
    "chr",
    "ord",
    "complex",
    # Collections
    "list",
    "dict",
    "set",
    "tuple",
    "frozenset",
    "len",
    "sorted",
    # Type check
    "isinstance",
    # Iteration
    "range",
    "enumerate",
    "zip",
    "reversed",
    # Formatting
    "repr",
    "ascii",
    "bin",
    "hex",
    "oct",
    # Boolean
    "all",
    "any",
    # Other
    "slice",
    "super",
    "object",
    "Exception",
    "BaseException",
    "NotImplementedError",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "RuntimeError",
    "AssertionError",
    "OSError",
    "ZeroDivisionError",
    "OverflowError",
    "FileNotFoundError",
    "UnicodeDecodeError",
    "StopIteration",
    # I/O
    "open",
    # print is handled specially
    "print",
    # Type introspection (test-only)
    "reveal_type",
}


class NameInfo:
    """Information about a declared name."""

    def __init__(
        self,
        name: str,
        kind: str,
        scope: str,
        lineno: int,
        col: int,
        decl_class: str,
        decl_func: str,
        bases: list[str] | None = None,
    ):
        self.name: str = name
        self.kind: str = kind  # "class" | "function" | "variable" | "parameter" | "field" | "constant" | "type_alias" | "import" | "builtin"
        self.scope: str = scope  # "builtin" | "module" | "class" | "local"
        self.lineno: int = lineno
        self.col: int = col
        self.decl_class: str = decl_class  # Class name if field/method
        self.decl_func: str = decl_func  # Function name if local/param
        self.bases: list[str] = bases if bases is not None else []

    def __repr__(self) -> str:
        return "NameInfo(" + self.name + ", " + self.kind + ", " + self.scope + ")"


class NameTable:
    """Symbol table for resolved names."""

    def __init__(self) -> None:
        self.module_names: dict[str, NameInfo] = {}
        self.class_names: dict[str, dict[str, NameInfo]] = {}
        self.local_names: dict[tuple[str, str], dict[str, NameInfo]] = {}

    def add_module(self, info: NameInfo) -> None:
        self.module_names[info.name] = info

    def add_class_member(self, class_name: str, info: NameInfo) -> None:
        if class_name not in self.class_names:
            self.class_names[class_name] = {}
        self.class_names[class_name][info.name] = info

    def add_local(self, class_name: str, func_name: str, info: NameInfo) -> None:
        key: tuple[str, str] = (class_name, func_name)
        if key not in self.local_names:
            self.local_names[key] = {}
        self.local_names[key][info.name] = info

    def get_module(self, name: str) -> NameInfo | None:
        return self.module_names.get(name)

    def get_class_member(self, class_name: str, name: str) -> NameInfo | None:
        class_members = self.class_names.get(class_name)
        if class_members is None:
            return None
        return class_members.get(name)

    def get_local(self, class_name: str, func_name: str, name: str) -> NameInfo | None:
        key: tuple[str, str] = (class_name, func_name)
        local_scope = self.local_names.get(key)
        if local_scope is None:
            return None
        return local_scope.get(name)


class NameViolation:
    """A name resolution error with location."""

    def __init__(
        self,
        lineno: int,
        col: int,
        category: str,
        message: str,
        source_file: str = "",
    ):
        self.lineno: int = lineno
        self.col: int = col
        self.category: str = category
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
            + ": ["
            + self.category
            + "] "
            + self.message
        )


class NameResult:
    """Result of name resolution."""

    def __init__(self) -> None:
        self.table: NameTable = NameTable()
        self.violations: list[NameViolation] = []
        self.warnings: list[NameViolation] = []

    def add_error(
        self, lineno: int, col: int, category: str, message: str, source_file: str = ""
    ) -> None:
        self.violations.append(
            NameViolation(lineno, col, category, message, source_file)
        )

    def add_warning(
        self, lineno: int, col: int, category: str, message: str, source_file: str = ""
    ) -> None:
        self.warnings.append(NameViolation(lineno, col, category, message, source_file))

    def errors(self) -> list[NameViolation]:
        return self.violations

    def ok(self) -> bool:
        return len(self.violations) == 0


def get_name_id(node: ASTNode) -> str | None:
    """Get id from Name node."""
    if get_str(node, "_type") == "Name":
        val = node.get("id")
        if isinstance(val, JStr):
            return val.value
    return None


def is_all_caps(name: str) -> bool:
    """Check if name is ALL_CAPS (constant convention)."""
    if len(name) == 0:
        return False
    i = 0
    while i < len(name):
        c = name[i]
        if c != "_" and not c.isupper() and not c.isdigit():
            return False
        i += 1
    # Must have at least one letter
    j = 0
    has_letter = False
    while j < len(name):
        if name[j].isupper():
            has_letter = True
            break
        j += 1
    return has_letter


def is_type_alias(name: str, value: ASTNode) -> bool:
    """Check if this looks like a type alias (PascalCase = type expression)."""
    if len(name) == 0:
        return False
    # Must start with uppercase
    if not name[0].isupper():
        return False
    # Value should be a Subscript (like dict[str, object]) or Name (like int)
    value_type = get_str(value, "_type")
    if value_type == "Subscript" or value_type == "Name" or value_type == "BinOp":
        return True
    return False


class NameResolver:
    """Resolves names in a dict-based AST."""

    def __init__(self) -> None:
        self.result: NameResult = NameResult()

    def _register_module_name(self, stmt: ASTNode, info: NameInfo) -> bool:
        """Register a module-level name, erroring on duplicates. Returns True if registered."""
        existing = self.result.table.get_module(info.name)
        if existing is not None:
            self.error(
                stmt,
                "redefinition",
                "'" + info.name + "' already defined at line " + str(existing.lineno),
            )
            return False
        self.result.table.add_module(info)
        return True

    def _get_base_name(self, base: ASTNode) -> str:
        """Extract base class name from AST node."""
        if get_str(base, "_type") == "Name":
            return get_str(base, "id")
        if get_str(base, "_type") == "Attribute":
            return get_str(base, "attr")
        return ""

    def error(self, node: ASTNode, category: str, message: str) -> None:
        lineno = get_int(node, "lineno")
        col = get_int(node, "col_offset")
        source_file = get_str(node, "_source_file")
        self.result.add_error(lineno, col, category, message, source_file)

    def warning(self, node: ASTNode, category: str, message: str) -> None:
        lineno = get_int(node, "lineno")
        col = get_int(node, "col_offset")
        source_file = get_str(node, "_source_file")
        self.result.add_warning(lineno, col, category, message, source_file)

    def resolve(self, ast_dict: ASTNode) -> NameResult:
        """Main entry point: run all passes and return result."""
        self.pass1_module_names(ast_dict)
        self.validate_base_classes(ast_dict)
        self.pass2_class_names(ast_dict)
        self.pass3_locals_and_refs(ast_dict)
        return self.result

    def validate_base_classes(self, ast_dict: ASTNode) -> None:
        """Validate that all base class names resolve."""
        body = get_nodes(ast_dict, "body")
        i = 0
        while i < len(body):
            stmt = body[i]
            if get_str(stmt, "_type") == "ClassDef":
                bases = get_nodes(stmt, "bases")
                j = 0
                while j < len(bases):
                    base = bases[j]
                    base_name = self._get_base_name(base)
                    if base_name != "":
                        if not self.resolve_name(base_name, "", ""):
                            self.error(
                                base,
                                "undefined",
                                "name '" + base_name + "' is not defined",
                            )
                    j += 1
            i += 1

    def pass1_module_names(self, ast_dict: ASTNode) -> None:
        """Pass 1: Collect module-level names (classes, functions, constants)."""
        body = get_nodes(ast_dict, "body")
        i = 0
        while i < len(body):
            stmt = body[i]
            node_type = get_str(stmt, "_type")
            lineno = get_int(stmt, "lineno")
            col = get_int(stmt, "col_offset")
            if node_type == "ClassDef":
                name = get_str(stmt, "name")
                bases: list[str] = []
                base_nodes = get_nodes(stmt, "bases")
                bi = 0
                while bi < len(base_nodes):
                    base_name = self._get_base_name(base_nodes[bi])
                    if base_name != "":
                        bases.append(base_name)
                    bi += 1
                info = NameInfo(name, "class", "module", lineno, col, "", "", bases)
                self._register_module_name(stmt, info)
            elif node_type == "FunctionDef":
                name = get_str(stmt, "name")
                info = NameInfo(name, "function", "module", lineno, col, "", "")
                self._register_module_name(stmt, info)
            elif node_type == "Assign":
                targets = get_nodes(stmt, "targets")
                value = get_node(stmt, "value")
                j = 0
                while j < len(targets):
                    target = targets[j]
                    if get_str(target, "_type") == "Name":
                        name = get_str(target, "id")
                        if is_all_caps(name):
                            info = NameInfo(
                                name, "constant", "module", lineno, col, "", ""
                            )
                            self._register_module_name(stmt, info)
                        elif is_type_alias(name, value):
                            info = NameInfo(
                                name, "type_alias", "module", lineno, col, "", ""
                            )
                            self._register_module_name(stmt, info)
                    j += 1
            elif node_type == "AnnAssign":
                target = get_node(stmt, "target")
                if get_str(target, "_type") == "Name":
                    name = get_str(target, "id")
                    kind = "constant" if is_all_caps(name) else "variable"
                    info = NameInfo(name, kind, "module", lineno, col, "", "")
                    self._register_module_name(stmt, info)
            elif node_type == "Import":
                names_list = get_nodes(stmt, "names")
                j = 0
                while j < len(names_list):
                    alias = names_list[j]
                    asname_str = get_str(alias, "asname")
                    import_name = get_str(alias, "name")
                    bound_name = asname_str if asname_str != "" else import_name
                    if bound_name != "":
                        info = NameInfo(
                            bound_name, "import", "module", lineno, col, "", ""
                        )
                        self._register_module_name(stmt, info)
                    j += 1
            elif node_type == "ImportFrom":
                names_list = get_nodes(stmt, "names")
                j = 0
                while j < len(names_list):
                    alias = names_list[j]
                    asname_str = get_str(alias, "asname")
                    import_name = get_str(alias, "name")
                    bound_name = asname_str if asname_str != "" else import_name
                    if bound_name != "" and bound_name != "*":
                        info = NameInfo(
                            bound_name, "import", "module", lineno, col, "", ""
                        )
                        self._register_module_name(stmt, info)
                    j += 1
            elif node_type == "If":
                # Handle TYPE_CHECKING blocks - imports inside are module-level
                test = get_node(stmt, "test")
                if (
                    get_str(test, "_type") == "Name"
                    and get_str(test, "id") == "TYPE_CHECKING"
                ):
                    if_body = get_nodes(stmt, "body")
                    j = 0
                    while j < len(if_body):
                        if_stmt = if_body[j]
                        if get_str(if_stmt, "_type") == "ImportFrom":
                            if_names = get_nodes(if_stmt, "names")
                            k = 0
                            while k < len(if_names):
                                alias = if_names[k]
                                asname_str = get_str(alias, "asname")
                                import_name = get_str(alias, "name")
                                bound_name = (
                                    asname_str if asname_str != "" else import_name
                                )
                                if bound_name != "" and bound_name != "*":
                                    if_lineno = get_int(if_stmt, "lineno")
                                    if_col = get_int(if_stmt, "col_offset")
                                    info = NameInfo(
                                        bound_name,
                                        "import",
                                        "module",
                                        if_lineno,
                                        if_col,
                                        "",
                                        "",
                                    )
                                    self._register_module_name(if_stmt, info)
                                k += 1
                        j += 1
            i += 1

    def pass2_class_names(self, ast_dict: ASTNode) -> None:
        """Pass 2: Collect class-level names (methods, fields)."""
        body = get_nodes(ast_dict, "body")
        i = 0
        while i < len(body):
            stmt = body[i]
            if get_str(stmt, "_type") == "ClassDef":
                self.collect_class_members(stmt)
            i += 1

    def collect_class_members(self, class_node: ASTNode) -> None:
        """Collect all members of a class."""
        class_name = get_str(class_node, "name")
        body = get_nodes(class_node, "body")
        # First pass: collect methods and annotated fields
        i = 0
        while i < len(body):
            stmt = body[i]
            node_type = get_str(stmt, "_type")
            lineno = get_int(stmt, "lineno")
            col = get_int(stmt, "col_offset")
            if node_type == "FunctionDef":
                name = get_str(stmt, "name")
                info = NameInfo(name, "function", "class", lineno, col, class_name, "")
                self.result.table.add_class_member(class_name, info)
            elif node_type == "AnnAssign":
                target = get_node(stmt, "target")
                if get_str(target, "_type") == "Name":
                    name = get_str(target, "id")
                    info = NameInfo(name, "field", "class", lineno, col, class_name, "")
                    self.result.table.add_class_member(class_name, info)
            i += 1
        # Second pass: collect self.x assignments in __init__
        j = 0
        while j < len(body):
            stmt = body[j]
            if (
                get_str(stmt, "_type") == "FunctionDef"
                and get_str(stmt, "name") == "__init__"
            ):
                self.collect_init_fields(class_name, stmt)
            j += 1

    def collect_init_fields(self, class_name: str, init_node: ASTNode) -> None:
        """Collect self.x = ... fields from __init__."""
        body = get_nodes(init_node, "body")
        nodes_to_visit: list[ASTNode] = []
        i = 0
        while i < len(body):
            nodes_to_visit.append(body[i])
            i += 1
        j = 0
        while j < len(nodes_to_visit):
            node = nodes_to_visit[j]
            node_type = get_str(node, "_type")
            if node_type == "Assign":
                targets = get_nodes(node, "targets")
                k = 0
                while k < len(targets):
                    target = targets[k]
                    if get_str(target, "_type") == "Attribute":
                        value_node = get_node(target, "value")
                        if get_name_id(value_node) == "self":
                            attr = get_str(target, "attr")
                            existing = self.result.table.get_class_member(
                                class_name, attr
                            )
                            if existing is None:
                                lineno = get_int(node, "lineno")
                                col = get_int(node, "col_offset")
                                info = NameInfo(
                                    attr, "field", "class", lineno, col, class_name, ""
                                )
                                self.result.table.add_class_member(class_name, info)
                    k += 1
            elif node_type == "AnnAssign":
                target = get_node(node, "target")
                if get_str(target, "_type") == "Attribute":
                    value_node = get_node(target, "value")
                    if get_name_id(value_node) == "self":
                        attr = get_str(target, "attr")
                        existing = self.result.table.get_class_member(class_name, attr)
                        if existing is None:
                            lineno = get_int(node, "lineno")
                            col = get_int(node, "col_offset")
                            info = NameInfo(
                                attr, "field", "class", lineno, col, class_name, ""
                            )
                            self.result.table.add_class_member(class_name, info)
            # Add children for If, While, etc.
            children = get_children(node)
            m = 0
            while m < len(children):
                child = children[m]
                child_type = get_str(child, "_type")
                # Skip nested functions (shouldn't exist per Phase 3)
                if child_type != "FunctionDef":
                    nodes_to_visit.append(child)
                m += 1
            j += 1

    def pass3_locals_and_refs(self, ast_dict: ASTNode) -> None:
        """Pass 3: Collect locals and resolve all name references."""
        body = get_nodes(ast_dict, "body")
        i = 0
        while i < len(body):
            stmt = body[i]
            node_type = get_str(stmt, "_type")
            if node_type == "FunctionDef":
                self.process_function(stmt, "", get_str(stmt, "name"))
            elif node_type == "ClassDef":
                self.process_class(stmt)
            i += 1

    def process_class(self, class_node: ASTNode) -> None:
        """Process all methods in a class."""
        class_name = get_str(class_node, "name")
        body = get_nodes(class_node, "body")
        i = 0
        while i < len(body):
            stmt = body[i]
            if get_str(stmt, "_type") == "FunctionDef":
                func_name = get_str(stmt, "name")
                self.process_function(stmt, class_name, func_name)
            i += 1

    def process_function(
        self, func_node: ASTNode, class_name: str, func_name: str
    ) -> None:
        """Process a function: collect params/locals, then resolve references."""
        # Collect parameters
        args_node = get_node(func_node, "args")
        args_list = get_nodes(args_node, "args")
        i = 0
        while i < len(args_list):
            arg = args_list[i]
            arg_name = get_str(arg, "arg")
            lineno = get_int(arg, "lineno")
            col = get_int(arg, "col_offset")
            # Add self/cls without shadowing warning
            if i == 0 and arg_name in ("self", "cls"):
                info = NameInfo(
                    arg_name, "parameter", "local", lineno, col, class_name, func_name
                )
                self.result.table.add_local(class_name, func_name, info)
                i += 1
                continue
            # Warn if parameter shadows a builtin
            if arg_name in ALLOWED_BUILTINS:
                self.warning(
                    arg, "shadowing", "parameter '" + arg_name + "' shadows builtin"
                )
            info = NameInfo(
                arg_name, "parameter", "local", lineno, col, class_name, func_name
            )
            self.result.table.add_local(class_name, func_name, info)
            i += 1
        # Collect keyword-only parameters
        kw_list = get_nodes(args_node, "kwonlyargs")
        i = 0
        while i < len(kw_list):
            arg = kw_list[i]
            arg_name = get_str(arg, "arg")
            lineno = get_int(arg, "lineno")
            col = get_int(arg, "col_offset")
            if arg_name in ALLOWED_BUILTINS:
                self.warning(
                    arg, "shadowing", "parameter '" + arg_name + "' shadows builtin"
                )
            info = NameInfo(
                arg_name, "parameter", "local", lineno, col, class_name, func_name
            )
            self.result.table.add_local(class_name, func_name, info)
            i += 1
        # Collect local variables from body
        body = get_nodes(func_node, "body")
        self.collect_locals_from_body(body, class_name, func_name)
        # Resolve all Name references
        self.resolve_references_in_body(body, class_name, func_name)

    def collect_locals_from_body(
        self, body: list[ASTNode], class_name: str, func_name: str
    ) -> None:
        """Collect local variable names from function body."""
        nodes_to_visit: list[ASTNode] = []
        i = 0
        while i < len(body):
            nodes_to_visit.append(body[i])
            i += 1
        j = 0
        while j < len(nodes_to_visit):
            node = nodes_to_visit[j]
            node_type = get_str(node, "_type")
            if node_type == "Assign":
                targets = get_nodes(node, "targets")
                k = 0
                while k < len(targets):
                    self.collect_assign_target(targets[k], class_name, func_name, node)
                    k += 1
            elif node_type == "AnnAssign":
                target = get_node(node, "target")
                if get_str(target, "_type") == "Name":
                    name = get_str(target, "id")
                    existing = self.result.table.get_local(class_name, func_name, name)
                    if existing is None:
                        lineno = get_int(node, "lineno")
                        col = get_int(node, "col_offset")
                        info = NameInfo(
                            name,
                            "variable",
                            "local",
                            lineno,
                            col,
                            class_name,
                            func_name,
                        )
                        self.result.table.add_local(class_name, func_name, info)
            elif node_type == "For":
                target = get_node(node, "target")
                self.collect_assign_target(target, class_name, func_name, node)
            elif node_type == "ExceptHandler":
                exc_name = get_str(node, "name")
                if has_key(node, "name") and exc_name != "":
                    existing = self.result.table.get_local(
                        class_name, func_name, exc_name
                    )
                    if existing is None:
                        lineno = get_int(node, "lineno")
                        col = get_int(node, "col_offset")
                        info = NameInfo(
                            exc_name,
                            "variable",
                            "local",
                            lineno,
                            col,
                            class_name,
                            func_name,
                        )
                        self.result.table.add_local(class_name, func_name, info)
            elif node_type == "ImportFrom":
                # Register imported names in local scope
                names_list = get_nodes(node, "names")
                k = 0
                while k < len(names_list):
                    alias = names_list[k]
                    asname_str = get_str(alias, "asname")
                    import_name = get_str(alias, "name")
                    bound_name = asname_str if asname_str != "" else import_name
                    if bound_name != "" and bound_name != "*":
                        existing = self.result.table.get_local(
                            class_name, func_name, bound_name
                        )
                        if existing is None:
                            lineno = get_int(node, "lineno")
                            col = get_int(node, "col_offset")
                            info = NameInfo(
                                bound_name,
                                "import",
                                "local",
                                lineno,
                                col,
                                class_name,
                                func_name,
                            )
                            self.result.table.add_local(class_name, func_name, info)
                    k += 1
            elif node_type == "Match":
                # Collect pattern variables from match/case
                cases = get_nodes(node, "cases")
                k = 0
                while k < len(cases):
                    case_node = cases[k]
                    pattern = get_node(case_node, "pattern")
                    self.collect_pattern_names(pattern, class_name, func_name, node)
                    k += 1
            elif node_type == "With":
                items = get_nodes(node, "items")
                k = 0
                while k < len(items):
                    item = items[k]
                    if has_key(item, "optional_vars"):
                        opt_vars = get_node(item, "optional_vars")
                        self.collect_assign_target(
                            opt_vars, class_name, func_name, node
                        )
                    k += 1
            elif node_type == "NamedExpr":
                # Walrus operator: (x := expr)
                target = get_node(node, "target")
                if get_str(target, "_type") == "Name":
                    name = get_str(target, "id")
                    existing = self.result.table.get_local(class_name, func_name, name)
                    if existing is None:
                        lineno = get_int(node, "lineno")
                        col = get_int(node, "col_offset")
                        info = NameInfo(
                            name,
                            "variable",
                            "local",
                            lineno,
                            col,
                            class_name,
                            func_name,
                        )
                        self.result.table.add_local(class_name, func_name, info)
            # Add children (skip nested FunctionDef - shouldn't exist per Phase 3)
            children = get_children(node)
            m = 0
            while m < len(children):
                child = children[m]
                if get_str(child, "_type") != "FunctionDef":
                    nodes_to_visit.append(child)
                m += 1
            j += 1

    def collect_assign_target(
        self, target: ASTNode, class_name: str, func_name: str, stmt: ASTNode
    ) -> None:
        """Collect names from an assignment target."""
        target_type = get_str(target, "_type")
        if target_type == "Name":
            name = get_str(target, "id")
            existing = self.result.table.get_local(class_name, func_name, name)
            if existing is None:
                lineno = get_int(stmt, "lineno")
                col = get_int(stmt, "col_offset")
                info = NameInfo(
                    name, "variable", "local", lineno, col, class_name, func_name
                )
                self.result.table.add_local(class_name, func_name, info)
        elif target_type == "Tuple" or target_type == "List":
            elts = get_nodes(target, "elts")
            i = 0
            while i < len(elts):
                self.collect_assign_target(elts[i], class_name, func_name, stmt)
                i += 1
        # Attribute targets (self.x) are handled in pass2

    def collect_pattern_names(
        self, pattern: ASTNode, class_name: str, func_name: str, stmt: ASTNode
    ) -> None:
        """Collect names bound by a match pattern."""
        pattern_type = get_str(pattern, "_type")
        if pattern_type == "MatchAs":
            # MatchAs(pattern=inner, name=bound_name)
            name = get_str(pattern, "name")
            if has_key(pattern, "name") and name != "" and name != "_":
                existing = self.result.table.get_local(class_name, func_name, name)
                if existing is None:
                    lineno = get_int(pattern, "lineno")
                    col = get_int(pattern, "col_offset")
                    info = NameInfo(
                        name, "variable", "local", lineno, col, class_name, func_name
                    )
                    self.result.table.add_local(class_name, func_name, info)
            if has_key(pattern, "pattern"):
                inner = get_node(pattern, "pattern")
                self.collect_pattern_names(inner, class_name, func_name, stmt)
        elif pattern_type == "MatchClass":
            # MatchClass(cls=..., patterns=[], kwd_attrs=[], kwd_patterns=[])
            kwd_patterns = get_nodes(pattern, "kwd_patterns")
            i = 0
            while i < len(kwd_patterns):
                self.collect_pattern_names(kwd_patterns[i], class_name, func_name, stmt)
                i += 1
            # Positional patterns
            patterns = get_nodes(pattern, "patterns")
            i = 0
            while i < len(patterns):
                self.collect_pattern_names(patterns[i], class_name, func_name, stmt)
                i += 1
        elif pattern_type == "MatchMapping":
            # MatchMapping(keys=[], patterns=[], rest=name)
            patterns = get_nodes(pattern, "patterns")
            i = 0
            while i < len(patterns):
                self.collect_pattern_names(patterns[i], class_name, func_name, stmt)
                i += 1
            rest = get_str(pattern, "rest")
            if has_key(pattern, "rest") and rest != "":
                existing = self.result.table.get_local(class_name, func_name, rest)
                if existing is None:
                    lineno = get_int(pattern, "lineno")
                    col = get_int(pattern, "col_offset")
                    info = NameInfo(
                        rest, "variable", "local", lineno, col, class_name, func_name
                    )
                    self.result.table.add_local(class_name, func_name, info)
        elif pattern_type == "MatchSequence":
            # MatchSequence(patterns=[])
            patterns = get_nodes(pattern, "patterns")
            i = 0
            while i < len(patterns):
                self.collect_pattern_names(patterns[i], class_name, func_name, stmt)
                i += 1
        elif pattern_type == "MatchStar":
            # MatchStar(name=bound_name)
            name = get_str(pattern, "name")
            if has_key(pattern, "name") and name != "" and name != "_":
                existing = self.result.table.get_local(class_name, func_name, name)
                if existing is None:
                    lineno = get_int(pattern, "lineno")
                    col = get_int(pattern, "col_offset")
                    info = NameInfo(
                        name, "variable", "local", lineno, col, class_name, func_name
                    )
                    self.result.table.add_local(class_name, func_name, info)
        elif pattern_type == "MatchOr":
            # MatchOr(patterns=[]) - all alternatives should bind same names
            patterns = get_nodes(pattern, "patterns")
            if len(patterns) > 0:
                self.collect_pattern_names(patterns[0], class_name, func_name, stmt)

    def _collect_target_names(self, target: ASTNode, names: set[str]) -> None:
        """Extract variable names from an assignment target into a set."""
        target_type = get_str(target, "_type")
        if target_type == "Name":
            name = get_str(target, "id")
            if name != "":
                names.add(name)
        elif target_type == "Tuple" or target_type == "List":
            elts = get_nodes(target, "elts")
            i = 0
            while i < len(elts):
                self._collect_target_names(elts[i], names)
                i += 1

    def resolve_comprehension_refs(
        self,
        node: ASTNode,
        class_name: str,
        func_name: str,
        outer_comp_vars: set[str],
    ) -> None:
        """Resolve references inside a comprehension with its own scope."""
        comp_vars: set[str] = set()
        i = 0
        keys = list(outer_comp_vars)
        while i < len(keys):
            comp_vars.add(keys[i])
            i += 1
        generators = get_nodes(node, "generators")
        i = 0
        while i < len(generators):
            gen = generators[i]
            target = get_node(gen, "target")
            self._collect_target_names(target, comp_vars)
            i += 1
        # Walk all children except nested comprehensions (handled recursively)
        nodes_to_visit: list[ASTNode] = []
        children = get_children(node)
        i = 0
        while i < len(children):
            nodes_to_visit.append(children[i])
            i += 1
        j = 0
        while j < len(nodes_to_visit):
            child = nodes_to_visit[j]
            child_type = get_str(child, "_type")
            if child_type in ("ListComp", "SetComp", "DictComp", "GeneratorExp"):
                self.resolve_comprehension_refs(child, class_name, func_name, comp_vars)
                j += 1
                continue
            if child_type == "Name":
                ctx = get_node(child, "ctx")
                if get_str(ctx, "_type") == "Load":
                    name = get_str(child, "id")
                    if name not in comp_vars:
                        if not self.resolve_name(name, class_name, func_name):
                            self.error(
                                child,
                                "undefined",
                                "name '" + name + "' is not defined",
                            )
            grandchildren = get_children(child)
            m = 0
            while m < len(grandchildren):
                gc = grandchildren[m]
                if get_str(gc, "_type") != "FunctionDef":
                    nodes_to_visit.append(gc)
                m += 1
            j += 1

    def resolve_references_in_body(
        self, body: list[ASTNode], class_name: str, func_name: str
    ) -> None:
        """Walk body and resolve all Name nodes with ctx=Load."""
        nodes_to_visit: list[ASTNode] = []
        i = 0
        while i < len(body):
            nodes_to_visit.append(body[i])
            i += 1
        j = 0
        while j < len(nodes_to_visit):
            node = nodes_to_visit[j]
            node_type = get_str(node, "_type")
            if node_type in ("ListComp", "SetComp", "DictComp", "GeneratorExp"):
                self.resolve_comprehension_refs(node, class_name, func_name, set())
                j += 1
                continue
            if node_type == "Name":
                ctx = get_node(node, "ctx")
                ctx_type = get_str(ctx, "_type")
                if ctx_type == "Load":
                    name = get_str(node, "id")
                    if not self.resolve_name(name, class_name, func_name):
                        msg = "name '" + name + "' is not defined"
                        if func_name == "__init__":
                            msg += "; cannot infer type"
                        self.error(node, "undefined", msg)
            # Add children (skip nested FunctionDef)
            children = get_children(node)
            m = 0
            while m < len(children):
                child = children[m]
                if get_str(child, "_type") != "FunctionDef":
                    nodes_to_visit.append(child)
                m += 1
            j += 1

    def resolve_name(self, name: str, class_name: str, func_name: str) -> bool:
        """Try to resolve a name: local -> module -> builtin."""
        # Check local scope
        local_info = self.result.table.get_local(class_name, func_name, name)
        if local_info is not None:
            return True
        # Check module scope
        module_info = self.result.table.get_module(name)
        if module_info is not None:
            return True
        # Check builtin scope
        if name in ALLOWED_BUILTINS:
            return True
        return False


def resolve_names(ast_dict: ASTNode) -> NameResult:
    """Phase 4: Resolve all names in the AST.

    Args:
        ast_dict: Dict-based AST from parse.py (validated by verify)

    Returns:
        NameResult with symbol table and any violations found
    """
    resolver = NameResolver()
    return resolver.resolve(ast_dict)


class BindResult:
    """Combined result of subset verification and name resolution."""

    def __init__(self) -> None:
        self.subset_violations: list[Violation] = []
        self.subset_warnings: list[Violation] = []
        self.table: NameTable = NameTable()
        self.name_violations: list[NameViolation] = []
        self.name_warnings: list[NameViolation] = []
        self.known_classes: set[str] = set()
        self.node_classes: set[str] = set()
        self.class_bases: dict[str, list[str]] = {}
        self.type_aliases: dict[str, str] = {}
        self.class_source_files: dict[str, str] = {}
        self.flow_graphs: dict[str, FlowGraph] = {}

    def subset_ok(self) -> bool:
        return len(self.subset_violations) == 0

    def names_ok(self) -> bool:
        return len(self.name_violations) == 0

    def ok(self) -> bool:
        return self.subset_ok() and self.names_ok()


def _compute_derived(
    ast_dict: ASTNode,
    table: NameTable,
    result: BindResult,
) -> None:
    """Compute derived class metadata from the name table."""
    mkeys = list(table.module_names.keys())
    ki = 0
    while ki < len(mkeys):
        mname = mkeys[ki]
        info = table.module_names[mname]
        if info.kind == "class":
            result.known_classes.add(mname)
            bi = 0
            while bi < len(info.bases):
                base = info.bases[bi]
                if base == "Node" or base.endswith("Node"):
                    result.node_classes.add(mname)
                bi += 1
        ki += 1
    ki = 0
    while ki < len(mkeys):
        mname = mkeys[ki]
        info = table.module_names[mname]
        if info.kind == "class":
            result.class_bases[mname] = list(info.bases)
        ki += 1
    ta_body = get_nodes(ast_dict, "body")
    tai = 0
    while tai < len(ta_body):
        ta_stmt = ta_body[tai]
        if get_str(ta_stmt, "_type") == "Assign":
            ta_targets = get_nodes(ta_stmt, "targets")
            if len(ta_targets) == 1:
                ta_target = ta_targets[0]
                if get_str(ta_target, "_type") == "Name":
                    ta_name = get_str(ta_target, "id")
                    if ta_name != "":
                        ta_info = table.module_names.get(ta_name)
                        if ta_info is not None and ta_info.kind == "type_alias":
                            ta_value_v = ta_stmt.get("value")
                            ta_value: ASTNode | None = None
                            if isinstance(ta_value_v, JDict):
                                ta_value = ta_value_v.entries
                            ta_str = annotation_to_str(ta_value)
                            if ta_str != "":
                                result.type_aliases[ta_name] = ta_str
        tai += 1
    csf_body = get_nodes(ast_dict, "body")
    csf_i = 0
    while csf_i < len(csf_body):
        csf_node = csf_body[csf_i]
        if get_str(csf_node, "_type") == "ClassDef":
            csf_name = get_str(csf_node, "name")
            csf_sf = get_str(csf_node, "_source_file")
            if csf_name != "" and csf_sf != "":
                result.class_source_files[csf_name] = csf_sf
        csf_i += 1


def run_bind(ast_dict: ASTNode) -> BindResult:
    """Run subset verification, name resolution, and compute derived data."""
    result = BindResult()
    verify_result = verify(ast_dict)
    result.subset_violations = verify_result.errors()
    result.subset_warnings = verify_result.warnings()
    name_result = resolve_names(ast_dict)
    result.table = name_result.table
    result.name_violations = name_result.violations
    result.name_warnings = name_result.warnings
    if result.names_ok():
        _compute_derived(ast_dict, name_result.table, result)
    body = get_nodes(ast_dict, "body")
    graphs: dict[str, FlowGraph] = {}
    i = 0
    while i < len(body):
        node = body[i]
        t = get_str(node, "_type")
        if t == "FunctionDef":
            fname = get_str(node, "name")
            fn_body = get_nodes(node, "body")
            if fname != "" and len(fn_body) > 0:
                graphs["module::" + fname] = build_cfg(fn_body)
        if t == "ClassDef":
            cname = get_str(node, "name")
            cbody = get_nodes(node, "body")
            j = 0
            while j < len(cbody):
                m = cbody[j]
                if get_str(m, "_type") == "FunctionDef":
                    mname = get_str(m, "name")
                    m_body = get_nodes(m, "body")
                    if mname != "" and len(m_body) > 0:
                        graphs[cname + "::" + mname] = build_cfg(m_body)
                j += 1
        i += 1
    result.flow_graphs = graphs
    return result
