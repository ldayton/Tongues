"""Phase 3: Verify dict-based AST conforms to Tongues subset.

Validates the AST from Phase 2 against language constraints defined in spec.md.
Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from typing import Callable

from .types import (
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
    get_jlist,
    has_key,
)


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
        self.node_count: int = 0

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


# Allowed builtins from spec.md
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
    "chr",
    "ord",
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
    # Iteration (enumerate/zip only in for-loop headers)
    "range",
    "enumerate",
    "zip",
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
    # print is handled specially
    "print",
}

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


def is_constant(node: ASTNode) -> bool:
    """Check if node is a constant literal."""
    return get_str(node, "_type") == "Constant"


def is_obvious_literal(node: ASTNode) -> bool:
    """Check if node is a literal with obvious type."""
    if get_str(node, "_type") != "Constant":
        return False
    val = node.get("value")
    if val is None or isinstance(val, JNull):
        return False
    return isinstance(val, (JStr, JInt, JBool, JFloat))


def get_name_id(node: ASTNode) -> str | None:
    """Get id from Name node."""
    if get_str(node, "_type") == "Name":
        val = get_str(node, "id")
        if val != "":
            return val
    return None


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
        self.result.node_count += 1
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
            d = kw_defaults[k]
            if isinstance(d, JDict):
                all_defaults.append(d.entries)
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
            func_name in ("enumerate", "zip")
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
            if self.file_results[files[i]].has_errors():
                return True
            i += 1
        return False
