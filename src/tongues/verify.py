"""Verify Python code is compatible with the Tongues subset.

The Tongues subset is restricted Python that can be transpiled to any language.
Key constraint: everything knowable at runtime must be knowable at compile time.

This module is itself Tongues-compliant and can be transpiled.
"""

from typing import Any


# =============================================================================
# Violations
# =============================================================================


class Violation:
    """A single violation of the Tongues subset."""

    def __init__(
        self,
        file: str,
        line: int,
        col: int,
        category: str,
        message: str,
        is_warning: bool,
    ) -> None:
        self.file = file
        self.line = line
        self.col = col
        self.category = category
        self.message = message
        self.is_warning = is_warning

    def __repr__(self) -> str:
        if self.is_warning:
            level = "warning"
        else:
            level = "error"
        return (
            self.file
            + ":"
            + str(self.line)
            + ":"
            + str(self.col)
            + ": "
            + level
            + ": ["
            + self.category
            + "] "
            + self.message
        )


class VerifyResult:
    """Result of verification including violations and node statistics."""

    def __init__(
        self, violations: list[Violation], seen_nodes: dict[str, int]
    ) -> None:
        self.violations = violations
        self.seen_nodes = seen_nodes


# =============================================================================
# Rules by category
# =============================================================================


def get_banned_builtins() -> dict[str, str]:
    """Return dict of banned builtins with reasons."""
    return {
        "getattr": "requires runtime introspection",
        "setattr": "requires runtime introspection",
        "hasattr": "requires runtime introspection",
        "delattr": "requires runtime introspection",
        "type": "requires runtime introspection",
        "vars": "requires runtime introspection",
        "dir": "requires runtime introspection",
        "globals": "requires runtime introspection",
        "locals": "requires runtime introspection",
        "id": "requires runtime introspection",
        "callable": "requires runtime introspection",
        "eval": "requires runtime introspection",
        "exec": "requires runtime introspection",
        "compile": "requires runtime introspection",
        "__import__": "requires runtime introspection",
        "issubclass": "requires runtime class hierarchy",
        "hash": "implementation-specific",
        "format": "format protocol is Python-specific",
        "memoryview": "low-level memory access",
        "iter": "returns lazy iterator (use list)",
        "next": "returns lazy iterator (use list)",
        "aiter": "returns lazy iterator",
        "anext": "returns lazy iterator",
        "map": "returns lazy iterator (use list comprehension)",
        "filter": "returns lazy iterator (use list comprehension)",
        "zip": "returns lazy iterator (use indexed loop)",
        "enumerate": "returns lazy iterator (use range(len(...)))",
        "reversed": "returns lazy iterator (use slicing [::-1])",
        "open": "I/O is not portable",
        "input": "I/O is not portable",
        "print": "I/O is not portable",
        "breakpoint": "interactive-only",
        "help": "interactive-only",
        "exit": "interactive-only",
        "quit": "interactive-only",
        "copyright": "interactive-only",
        "credits": "interactive-only",
        "license": "interactive-only",
        "staticmethod": "use module-level function instead",
        "classmethod": "use module-level function instead",
        "property": "use explicit getter method instead",
    }


def get_allowed_builtins() -> set[str]:
    """Return set of allowed builtins."""
    return {
        "abs",
        "min",
        "max",
        "sum",
        "len",
        "range",
        "round",
        "divmod",
        "sorted",
        "pow",
        "all",
        "any",
        "int",
        "float",
        "str",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "slice",
        "ord",
        "chr",
        "bin",
        "hex",
        "oct",
        "repr",
        "ascii",
        "isinstance",
        "super",
        "object",
        "Exception",
        "BaseException",
    }


def get_reflection_dunders() -> set[str]:
    """Return set of reflection dunder attributes."""
    return {
        "__class__",
        "__dict__",
        "__name__",
        "__module__",
        "__bases__",
        "__mro__",
        "__subclasses__",
        "__qualname__",
        "__annotations__",
    }


def get_allowed_dunders() -> set[str]:
    """Return set of allowed dunder methods."""
    return {"__init__", "__new__", "__repr__"}


def get_bare_collections() -> set[str]:
    """Return set of bare collection type names."""
    return {"list", "dict", "set", "tuple"}


def get_allowed_nodes() -> set[str]:
    """Return set of allowed AST node types."""
    return {
        "Module",
        "Expression",
        "Interactive",
        "Constant",
        "List",
        "Tuple",
        "Dict",
        "Set",
        "Name",
        "Starred",
        "Expr",
        "UnaryOp",
        "BinOp",
        "BoolOp",
        "Compare",
        "IfExp",
        "Call",
        "Attribute",
        "Subscript",
        "Slice",
        "ListComp",
        "SetComp",
        "DictComp",
        "comprehension",
        "Assign",
        "AnnAssign",
        "AugAssign",
        "Return",
        "Pass",
        "Break",
        "Continue",
        "If",
        "For",
        "While",
        "Try",
        "Raise",
        "Assert",
        "FunctionDef",
        "ClassDef",
        "arg",
        "arguments",
        "keyword",
        "ExceptHandler",
        "Import",
        "ImportFrom",
        "alias",
        "Load",
        "Store",
        "Del",
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
        "Add",
        "Sub",
        "Mult",
        "Div",
        "FloorDiv",
        "Mod",
        "Pow",
        "MatMult",
        "LShift",
        "RShift",
        "BitOr",
        "BitXor",
        "BitAnd",
        "And",
        "Or",
        "Not",
        "Invert",
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
        "Delete",
        "withitem",
    }


def get_banned_nodes() -> dict[str, tuple[str, str]]:
    """Return dict of banned AST nodes with (category, message)."""
    return {
        "AsyncFunctionDef": ("async", "async has no portable equivalent"),
        "AsyncFor": ("async", "async for has no portable equivalent"),
        "AsyncWith": ("async", "async with has no portable equivalent"),
        "Await": ("async", "await has no portable equivalent"),
        "Yield": ("generator", "yield requires lazy evaluation (use list)"),
        "YieldFrom": ("generator", "yield from requires lazy evaluation (use list)"),
        "GeneratorExp": (
            "generator",
            "generator expression requires lazy evaluation (use list comprehension)",
        ),
        "With": (
            "control",
            "with statement requires context manager protocol (use try/finally)",
        ),
        "Lambda": ("function", "lambda is not supported (use named function)"),
        "Global": ("function", "global is not supported (pass as parameter)"),
        "Nonlocal": ("function", "nonlocal is not supported (pass as parameter)"),
        "NamedExpr": (
            "syntax",
            "walrus operator := complicates type inference (use separate assignment)",
        ),
        "TypeAlias": ("syntax", "type aliases are Python 3.12+ (use explicit types)"),
        "TryStar": ("syntax", "except* is Python 3.11+ (use regular except)"),
        "JoinedStr": (
            "syntax",
            "f-strings use Python-specific format protocol (use concatenation or %)",
        ),
        "FormattedValue": ("syntax", "f-strings use Python-specific format protocol"),
        "TypeVar": ("syntax", "type parameters are Python 3.12+"),
        "ParamSpec": ("syntax", "type parameters are Python 3.12+"),
        "TypeVarTuple": ("syntax", "type parameters are Python 3.12+"),
    }


# Module-level constants for tests to import
BANNED_BUILTINS: dict[str, str] = get_banned_builtins()
ALLOWED_BUILTINS: set[str] = get_allowed_builtins()
ALLOWED_NODES: set[str] = get_allowed_nodes()
BANNED_NODES: dict[str, tuple[str, str]] = get_banned_nodes()


# =============================================================================
# Helpers
# =============================================================================


def get_children(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Get immediate child nodes of a dict-based AST node."""
    children: list[dict[str, Any]] = []
    keys = list(node.keys())
    i = 0
    while i < len(keys):
        key = keys[i]
        if not key.startswith("_") and key not in (
            "lineno",
            "col_offset",
            "end_lineno",
            "end_col_offset",
        ):
            value = node[key]
            if isinstance(value, dict) and "_type" in value:
                children.append(value)
            elif isinstance(value, list):
                j = 0
                while j < len(value):
                    item = value[j]
                    if isinstance(item, dict) and "_type" in item:
                        children.append(item)
                    j = j + 1
        i = i + 1
    return children


def walk(node: dict[str, Any], result: list[dict[str, Any]]) -> None:
    """Recursively collect all descendant nodes in the dict-based AST."""
    result.append(node)
    children = get_children(node)
    i = 0
    while i < len(children):
        walk(children[i], result)
        i = i + 1


def is_bare_collection(annotation: dict[str, Any] | None) -> str | None:
    """Return collection name if annotation is bare (e.g., list instead of list[int])."""
    if annotation is None:
        return None
    if annotation.get("_type") == "Name":
        name = annotation.get("id")
        if name in get_bare_collections():
            return name
    return None


def is_obvious_literal(node: dict[str, Any]) -> bool:
    """Check if node is a literal with obvious type (str, int, bool, float)."""
    if node.get("_type") == "Constant":
        value = node.get("value")
        if value is None:
            return False
        if isinstance(value, (str, int, bool, float)):
            return True
    return False


# =============================================================================
# Verifier
# =============================================================================


class Verifier:
    """Dict-based AST visitor that collects Tongues subset violations."""

    def __init__(self, file: str, tree: dict[str, Any]) -> None:
        self.file = file
        self.violations: list[Violation] = []
        self.seen_nodes: dict[str, int] = {}
        self.function_stack: list[dict[str, Any]] = []
        self.defined_names: set[str] = set()
        self._collect_defined_names(tree)

    def _collect_defined_names(self, tree: dict[str, Any]) -> None:
        """Collect all user-defined names from the tree."""
        nodes: list[dict[str, Any]] = []
        walk(tree, nodes)
        i = 0
        while i < len(nodes):
            node = nodes[i]
            node_type = node.get("_type")
            if node_type in ("FunctionDef", "AsyncFunctionDef", "ClassDef"):
                name = node.get("name")
                if name is not None:
                    self.defined_names.add(name)
            elif node_type == "Assign":
                targets = node.get("targets")
                if targets is not None:
                    j = 0
                    while j < len(targets):
                        target = targets[j]
                        if isinstance(target, dict) and target.get("_type") == "Name":
                            name = target.get("id")
                            if name is not None:
                                self.defined_names.add(name)
                        j = j + 1
            elif node_type == "AnnAssign":
                target = node.get("target")
                if isinstance(target, dict) and target.get("_type") == "Name":
                    name = target.get("id")
                    if name is not None:
                        self.defined_names.add(name)
            i = i + 1

    def visit(self, node: dict[str, Any]) -> None:
        """Visit a node, checking if it's allowed/banned/unknown."""
        node_type = node.get("_type", "")
        self.seen_nodes[node_type] = self.seen_nodes.get(node_type, 0) + 1
        banned_nodes = get_banned_nodes()
        allowed_nodes = get_allowed_nodes()
        if node_type in banned_nodes:
            cat_msg = banned_nodes[node_type]
            self.add(node, cat_msg[0], cat_msg[1], False)
        elif node_type not in allowed_nodes:
            self.add(node, "syntax", node_type + ": unknown AST node", True)
        self._dispatch_visit(node, node_type)

    def _dispatch_visit(self, node: dict[str, Any], node_type: str) -> None:
        """Dispatch to specific visitor method."""
        if node_type == "FunctionDef":
            self.visit_FunctionDef(node)
        elif node_type == "ClassDef":
            self.visit_ClassDef(node)
        elif node_type == "For":
            self.visit_For(node)
        elif node_type == "While":
            self.visit_While(node)
        elif node_type == "ExceptHandler":
            self.visit_ExceptHandler(node)
        elif node_type == "Import":
            self.visit_Import(node)
        elif node_type == "ImportFrom":
            self.visit_ImportFrom(node)
        elif node_type == "Delete":
            self.visit_Delete(node)
        elif node_type == "Assign":
            self.visit_Assign(node)
        elif node_type == "AnnAssign":
            self.visit_AnnAssign(node)
        elif node_type == "Call":
            self.visit_Call(node)
        elif node_type == "Attribute":
            self.visit_Attribute(node)
        elif node_type == "Compare":
            self.visit_Compare(node)
        elif node_type == "BoolOp":
            self.visit_BoolOp(node)
        else:
            self.generic_visit(node)

    def generic_visit(self, node: dict[str, Any]) -> None:
        """Visit all children of a node."""
        children = get_children(node)
        i = 0
        while i < len(children):
            self.visit(children[i])
            i = i + 1

    def add(
        self,
        node: dict[str, Any],
        category: str,
        message: str,
        is_warning: bool,
    ) -> None:
        """Record a violation."""
        line = node.get("lineno")
        if line is None:
            line = 0
        col = node.get("col_offset")
        if col is None:
            col = 0
        self.violations.append(
            Violation(
                file=self.file,
                line=line,
                col=col,
                category=category,
                message=message,
                is_warning=is_warning,
            )
        )

    # -------------------------------------------------------------------------
    # Reflection
    # -------------------------------------------------------------------------

    def check_banned_call(self, node: dict[str, Any]) -> None:
        """Check for banned built-in function calls."""
        func = node.get("func")
        if isinstance(func, dict) and func.get("_type") == "Name":
            name = func.get("id")
            banned = get_banned_builtins()
            allowed = get_allowed_builtins()
            if name in banned:
                msg = banned[name]
                self.add(node, "builtin", name + "(): " + msg, False)
            elif name not in allowed and name not in self.defined_names:
                if name is not None and len(name) > 0:
                    first_char = name[0]
                    if first_char.islower() and not name.startswith("_"):
                        self.add(node, "builtin", name + "(): unknown builtin", True)

    def check_reflection_attribute(self, node: dict[str, Any]) -> None:
        """Check for dunder attribute access."""
        attr = node.get("attr")
        if attr in get_reflection_dunders():
            self.add(node, "reflection", "." + attr + " requires runtime introspection", False)

    # -------------------------------------------------------------------------
    # Control flow
    # -------------------------------------------------------------------------

    def visit_For(self, node: dict[str, Any]) -> None:
        if node.get("orelse"):
            self.add(node, "control", "loop else is not supported (use flag variable)", False)
        self.generic_visit(node)

    def visit_While(self, node: dict[str, Any]) -> None:
        if node.get("orelse"):
            self.add(node, "control", "loop else is not supported (use flag variable)", False)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: dict[str, Any]) -> None:
        if node.get("type") is None:
            self.add(
                node, "control", "bare except is not supported (specify exception type)", False
            )
        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # Functions
    # -------------------------------------------------------------------------

    def visit_FunctionDef(self, node: dict[str, Any]) -> None:
        name = node.get("name", "")
        if len(self.function_stack) > 0:
            self.add(node, "function", "nested function '" + name + "' (move to module level)", False)
        args = node.get("args")
        if args is None:
            args = {}
        if args.get("kwarg"):
            self.add(node, "function", "**kwargs is not supported (use explicit parameters)", False)
        if name.startswith("__") and name.endswith("__"):
            allowed_dunders = get_allowed_dunders()
            if name not in allowed_dunders:
                self.add(node, "function", name + " has no portable equivalent", False)
        decorator_list = node.get("decorator_list")
        if decorator_list is None:
            decorator_list = []
        j = 0
        while j < len(decorator_list):
            dec = decorator_list[j]
            dec_name = None
            if isinstance(dec, dict):
                if dec.get("_type") == "Name":
                    dec_name = dec.get("id")
                elif dec.get("_type") == "Attribute":
                    dec_name = dec.get("attr")
            if dec_name == "staticmethod":
                self.add(
                    dec,
                    "function",
                    "@staticmethod is not supported (use module-level function)",
                    False,
                )
            elif dec_name == "classmethod":
                self.add(
                    dec,
                    "function",
                    "@classmethod is not supported (use module-level function)",
                    False,
                )
            elif dec_name == "property":
                self.add(dec, "function", "@property is not supported (use explicit getter)", False)
            else:
                self.add(dec, "function", "decorators require runtime modification", False)
            j = j + 1
        defaults = args.get("defaults")
        if defaults is None:
            defaults = []
        kw_defaults = args.get("kw_defaults")
        if kw_defaults is None:
            kw_defaults = []
        all_defaults = defaults + kw_defaults
        k = 0
        while k < len(all_defaults):
            default = all_defaults[k]
            if default is not None and isinstance(default, dict):
                if default.get("_type") in ("List", "Dict", "Set"):
                    self.add(
                        node,
                        "function",
                        "mutable default argument is not supported (use None)",
                        False,
                    )
                    break
            k = k + 1
        self.check_function_annotations(node)
        self.function_stack.append(node)
        self.generic_visit(node)
        self.function_stack.pop()

    # -------------------------------------------------------------------------
    # Classes
    # -------------------------------------------------------------------------

    def visit_ClassDef(self, node: dict[str, Any]) -> None:
        bases = node.get("bases")
        if bases is None:
            bases = []
        real_bases: list[dict[str, Any]] = []
        i = 0
        while i < len(bases):
            b = bases[i]
            is_exception = (
                isinstance(b, dict)
                and b.get("_type") == "Name"
                and b.get("id") == "Exception"
            )
            if not is_exception:
                real_bases.append(b)
            i = i + 1
        if len(real_bases) > 1:
            self.add(
                node,
                "class",
                "multiple inheritance has no portable equivalent (use composition)",
                False,
            )
        decorator_list = node.get("decorator_list")
        if decorator_list is None:
            decorator_list = []
        j = 0
        while j < len(decorator_list):
            dec = decorator_list[j]
            self.add(dec, "class", "class decorators require runtime modification", False)
            j = j + 1
        body = node.get("body")
        if body is None:
            body = []
        k = 0
        while k < len(body):
            item = body[k]
            if isinstance(item, dict) and item.get("_type") == "ClassDef":
                self.add(item, "class", "nested class (move to module level)", False)
            k = k + 1
        self.check_unannotated_fields(node)
        self.generic_visit(node)

    def check_unannotated_fields(self, class_node: dict[str, Any]) -> None:
        """Check for self.x = val assignments that need type annotations."""
        annotated_fields: set[str] = set()
        nodes: list[dict[str, Any]] = []
        walk(class_node, nodes)
        i = 0
        while i < len(nodes):
            item = nodes[i]
            if item.get("_type") == "AnnAssign":
                target = item.get("target")
                if isinstance(target, dict):
                    if target.get("_type") == "Name":
                        name = target.get("id")
                        if name is not None:
                            annotated_fields.add(name)
                    elif target.get("_type") == "Attribute":
                        value = target.get("value")
                        if (
                            isinstance(value, dict)
                            and value.get("_type") == "Name"
                            and value.get("id") == "self"
                        ):
                            attr = target.get("attr")
                            if attr is not None:
                                annotated_fields.add(attr)
            i = i + 1
        body = class_node.get("body")
        if body is None:
            body = []
        j = 0
        while j < len(body):
            item = body[j]
            if not isinstance(item, dict) or item.get("_type") != "FunctionDef":
                j = j + 1
                continue
            args = item.get("args")
            if args is None:
                args = {}
            arg_list = args.get("args")
            if arg_list is None:
                arg_list = []
            annotated_params: set[str] = set()
            k = 0
            while k < len(arg_list):
                arg = arg_list[k]
                if isinstance(arg, dict) and arg.get("annotation") is not None:
                    arg_name = arg.get("arg")
                    if arg_name is not None:
                        annotated_params.add(arg_name)
                k = k + 1
            method_nodes: list[dict[str, Any]] = []
            walk(item, method_nodes)
            m = 0
            while m < len(method_nodes):
                stmt = method_nodes[m]
                if stmt.get("_type") != "Assign":
                    m = m + 1
                    continue
                targets = stmt.get("targets")
                if targets is None:
                    targets = []
                n = 0
                while n < len(targets):
                    target = targets[n]
                    if not isinstance(target, dict) or target.get("_type") != "Attribute":
                        n = n + 1
                        continue
                    target_value = target.get("value")
                    if not isinstance(target_value, dict) or target_value.get("_type") != "Name":
                        n = n + 1
                        continue
                    if target_value.get("id") != "self":
                        n = n + 1
                        continue
                    field = target.get("attr")
                    if field is None or field in annotated_fields:
                        n = n + 1
                        continue
                    stmt_value = stmt.get("value")
                    if isinstance(stmt_value, dict):
                        if (
                            stmt_value.get("_type") == "Name"
                            and stmt_value.get("id") in annotated_params
                        ):
                            annotated_fields.add(field)
                            n = n + 1
                            continue
                        if is_obvious_literal(stmt_value):
                            annotated_fields.add(field)
                            n = n + 1
                            continue
                    self.add(stmt, "types", "self." + field + " needs type annotation", False)
                    annotated_fields.add(field)
                    n = n + 1
                m = m + 1
            j = j + 1

    # -------------------------------------------------------------------------
    # Imports
    # -------------------------------------------------------------------------

    def visit_Import(self, node: dict[str, Any]) -> None:
        self.add(node, "import", "import is not supported (code must be self-contained)", False)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: dict[str, Any]) -> None:
        allowed = {"__future__", "typing", "collections.abc"}
        module = node.get("module")
        if module not in allowed:
            msg = "from " + str(module) + " import (code must be self-contained)"
            self.add(node, "import", msg, False)
        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # Statements
    # -------------------------------------------------------------------------

    def visit_Delete(self, node: dict[str, Any]) -> None:
        targets = node.get("targets")
        if targets is None:
            targets = []
        i = 0
        while i < len(targets):
            target = targets[i]
            if not isinstance(target, dict):
                i = i + 1
                continue
            target_type = target.get("_type")
            if target_type == "Subscript":
                i = i + 1
                continue
            if target_type == "Name":
                name = target.get("id")
                self.add(
                    node,
                    "statement",
                    "del " + str(name) + " (unbinding variables is not portable)",
                    False,
                )
            else:
                self.add(node, "statement", "del is not supported for this target", False)
            i = i + 1
        self.generic_visit(node)

    def visit_Assign(self, node: dict[str, Any]) -> None:
        targets = node.get("targets")
        if targets is None:
            targets = []
        if len(targets) == 1:
            target = targets[0]
            if isinstance(target, dict) and target.get("_type") == "Tuple":
                value = node.get("value")
                if isinstance(value, dict) and value.get("_type") == "Name":
                    self.add(
                        node,
                        "expression",
                        "tuple unpack from variable (unpack directly from function call)",
                        False,
                    )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: dict[str, Any]) -> None:
        annotation = node.get("annotation")
        bare = is_bare_collection(annotation)
        if bare is not None:
            target = node.get("target")
            name = "?"
            if isinstance(target, dict) and target.get("_type") == "Name":
                name = target.get("id", "?")
            self.add(node, "types", "bare " + bare + ": " + name + " needs type parameter", False)
        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # Expressions
    # -------------------------------------------------------------------------

    def visit_Call(self, node: dict[str, Any]) -> None:
        self.check_banned_call(node)
        keywords = node.get("keywords")
        if keywords is None:
            keywords = []
        i = 0
        while i < len(keywords):
            kw = keywords[i]
            if isinstance(kw, dict) and kw.get("arg") is None:
                self.add(node, "expression", "**kwargs in call (pass arguments explicitly)", False)
                break
            i = i + 1
        args = node.get("args")
        if args is None:
            args = []
        j = 0
        while j < len(args):
            arg = args[j]
            if isinstance(arg, dict) and arg.get("_type") == "Starred":
                self.add(node, "expression", "*args in call (pass arguments explicitly)", False)
                break
            j = j + 1
        self.generic_visit(node)

    def visit_Attribute(self, node: dict[str, Any]) -> None:
        self.check_reflection_attribute(node)
        self.generic_visit(node)

    def visit_Compare(self, node: dict[str, Any]) -> None:
        ops = node.get("ops")
        if ops is None:
            ops = []
        comparators = node.get("comparators")
        if comparators is None:
            comparators = []
        i = 0
        while i < len(ops):
            op = ops[i]
            if isinstance(op, dict) and op.get("_type") in ("Is", "IsNot"):
                if i == 0:
                    left = node.get("left")
                else:
                    left = comparators[i - 1]
                if i < len(comparators):
                    right = comparators[i]
                else:
                    right = None
                left_none = (
                    isinstance(left, dict)
                    and left.get("_type") == "Constant"
                    and left.get("value") is None
                )
                right_none = (
                    isinstance(right, dict)
                    and right.get("_type") == "Constant"
                    and right.get("value") is None
                )
                if not left_none and not right_none:
                    self.add(node, "expression", "is/is not only for None (use == for values)", False)
            i = i + 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: dict[str, Any]) -> None:
        op = node.get("op")
        if isinstance(op, dict) and op.get("_type") == "Or":
            values = node.get("values")
            if values is None:
                values = []
            i = 1
            while i < len(values):
                val = values[i]
                if isinstance(val, dict):
                    if val.get("_type") in ("List", "Dict", "Set"):
                        self.add(
                            node,
                            "expression",
                            "or-default pattern is not supported (use if/else)",
                            False,
                        )
                        break
                    if val.get("_type") == "Constant" and val.get("value") in (0, "", False):
                        self.add(
                            node,
                            "expression",
                            "or-default pattern is not supported (use if/else)",
                            False,
                        )
                        break
                i = i + 1
        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # Type annotations
    # -------------------------------------------------------------------------

    def check_function_annotations(self, node: dict[str, Any]) -> None:
        """Check for missing or bare type annotations."""
        name = node.get("name", "")
        if node.get("returns") is None and name not in ("__init__", "__new__"):
            self.add(node, "types", "missing return type: def " + name + "() -> ...", False)
        bare = is_bare_collection(node.get("returns"))
        if bare is not None:
            self.add(
                node, "types", "bare " + bare + ": " + name + "() return needs type parameter", False
            )
        args = node.get("args")
        if args is None:
            args = {}
        arg_list = args.get("args")
        if arg_list is None:
            arg_list = []
        i = 0
        while i < len(arg_list):
            arg = arg_list[i]
            if not isinstance(arg, dict):
                i = i + 1
                continue
            arg_name = arg.get("arg", "")
            if arg_name in ("self", "cls") and i == 0:
                i = i + 1
                continue
            if arg.get("annotation") is None:
                self.add(
                    node, "types", "missing param type: " + arg_name + " in " + name + "()", False
                )
            else:
                bare = is_bare_collection(arg.get("annotation"))
                if bare is not None:
                    self.add(
                        node,
                        "types",
                        "bare " + bare + ": " + arg_name + " in " + name + "() needs type parameter",
                        False,
                    )
            i = i + 1


# =============================================================================
# Public API
# =============================================================================


def verify_ast(tree: dict[str, Any], file: str | None = None) -> VerifyResult:
    """Verify a dict-based AST against the Tongues subset."""
    if file is None:
        file = "<stdin>"
    verifier = Verifier(file, tree)
    verifier.visit(tree)
    return VerifyResult(violations=verifier.violations, seen_nodes=verifier.seen_nodes)
