"""Verify Python code is compatible with the Tongues subset.

The Tongues subset is restricted Python that can be transpiled to any language.
Key constraint: everything knowable at runtime must be knowable at compile time.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# Violations
# =============================================================================


@dataclass
class Violation:
    """A single violation of the Tongues subset."""

    file: Path
    line: int
    col: int
    category: str
    message: str
    is_warning: bool = False

    def __str__(self) -> str:
        level = "warning" if self.is_warning else "error"
        return f"{self.file}:{self.line}:{self.col}: {level}: [{self.category}] {self.message}"


# =============================================================================
# Rules by category
# =============================================================================

# Banned built-in functions by category
BANNED_BUILTINS: dict[str, str] = {
    # Reflection: inspecting types/attributes at runtime
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
    # Lazy iterators (generators are banned)
    "iter": "returns lazy iterator (use list)",
    "next": "returns lazy iterator (use list)",
    "aiter": "returns lazy iterator",
    "anext": "returns lazy iterator",
    "map": "returns lazy iterator (use list comprehension)",
    "filter": "returns lazy iterator (use list comprehension)",
    "zip": "returns lazy iterator (use indexed loop)",
    "enumerate": "returns lazy iterator (use range(len(...)))",
    "reversed": "returns lazy iterator (use slicing [::-1])",
    # I/O
    "open": "I/O is not portable",
    "input": "I/O is not portable",
    "print": "I/O is not portable",
    # Interactive
    "breakpoint": "interactive-only",
    "help": "interactive-only",
    "exit": "interactive-only",
    "quit": "interactive-only",
    "copyright": "interactive-only",
    "credits": "interactive-only",
    "license": "interactive-only",
    # Decorators (handled separately, but ban as function calls too)
    "staticmethod": "use module-level function instead",
    "classmethod": "use module-level function instead",
    "property": "use explicit getter method instead",
}

# Allowed built-in functions
ALLOWED_BUILTINS = frozenset({
    # Math/utility
    "abs", "min", "max", "sum", "len", "range", "round", "divmod", "sorted", "pow",
    "all", "any",
    # Type conversion
    "int", "float", "str", "bool", "bytes", "bytearray", "complex",
    # Collections
    "list", "dict", "set", "tuple", "frozenset", "slice",
    # Character/number conversion
    "ord", "chr", "bin", "hex", "oct",
    # String conversion
    "repr", "ascii",
    # Type checking (needed for union types)
    "isinstance",
    # OOP
    "super", "object",
    # Exceptions (used as base classes, not called)
    "Exception", "BaseException",
})

REFLECTION_DUNDERS = frozenset({
    "__class__", "__dict__", "__name__", "__module__",
    "__bases__", "__mro__", "__subclasses__",
    "__qualname__", "__annotations__",
})

# Allowed dunders (everything else is banned)
ALLOWED_DUNDERS = frozenset({
    "__init__", "__new__", "__repr__",
})

# Bare collection types that need type parameters
BARE_COLLECTIONS = frozenset({"list", "dict", "set", "tuple"})

# AST node classification
# Allowed: these constructs are fully supported
ALLOWED_NODES = frozenset({
    # Module structure
    "Module", "Expression", "Interactive",
    # Literals
    "Constant", "List", "Tuple", "Dict", "Set",
    # Variables
    "Name", "Starred",
    # Expressions
    "Expr", "UnaryOp", "BinOp", "BoolOp", "Compare", "IfExp",
    "Call", "Attribute", "Subscript", "Slice",
    # Comprehensions (eager, not lazy)
    "ListComp", "SetComp", "DictComp", "comprehension",
    # Statements
    "Assign", "AnnAssign", "AugAssign", "Return", "Pass", "Break", "Continue",
    "If", "For", "While", "Try", "Raise", "Assert",
    # Definitions
    "FunctionDef", "ClassDef", "arg", "arguments", "keyword",
    # Exception handling
    "ExceptHandler",
    # Import (checked separately)
    "Import", "ImportFrom", "alias",
    # Context expressions
    "Load", "Store", "Del",
    # Match (allowed, transpiler expands)
    "Match", "match_case", "MatchValue", "MatchSingleton", "MatchSequence",
    "MatchMapping", "MatchClass", "MatchStar", "MatchAs", "MatchOr",
    # Operators (not nodes themselves, but part of expressions)
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow", "MatMult",
    "LShift", "RShift", "BitOr", "BitXor", "BitAnd",
    "And", "Or", "Not", "Invert", "UAdd", "USub",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
    # Delete (checked separately)
    "Delete",
    # With item (part of banned With, but node itself is fine)
    "withitem",
})

# Banned: these constructs are not supported with specific reasons
# Format: node_type -> (category, message)
BANNED_NODES: dict[str, tuple[str, str]] = {
    # Async
    "AsyncFunctionDef": ("async", "async has no portable equivalent"),
    "AsyncFor": ("async", "async for has no portable equivalent"),
    "AsyncWith": ("async", "async with has no portable equivalent"),
    "Await": ("async", "await has no portable equivalent"),
    # Generators
    "Yield": ("generator", "yield requires lazy evaluation (use list)"),
    "YieldFrom": ("generator", "yield from requires lazy evaluation (use list)"),
    "GeneratorExp": ("generator", "generator expression requires lazy evaluation (use list comprehension)"),
    # Control flow
    "With": ("control", "with statement requires context manager protocol (use try/finally)"),
    # Functions
    "Lambda": ("function", "lambda is not supported (use named function)"),
    "Global": ("function", "global is not supported (pass as parameter)"),
    "Nonlocal": ("function", "nonlocal is not supported (pass as parameter)"),
    # Modern Python features
    "NamedExpr": ("syntax", "walrus operator := complicates type inference (use separate assignment)"),
    "TypeAlias": ("syntax", "type aliases are Python 3.12+ (use explicit types)"),
    "TryStar": ("syntax", "except* is Python 3.11+ (use regular except)"),
    # F-strings (format protocol)
    "JoinedStr": ("syntax", "f-strings use Python-specific format protocol (use concatenation or %)"),
    "FormattedValue": ("syntax", "f-strings use Python-specific format protocol"),
    # Type parameters (Python 3.12+)
    "TypeVar": ("syntax", "type parameters are Python 3.12+"),
    "ParamSpec": ("syntax", "type parameters are Python 3.12+"),
    "TypeVarTuple": ("syntax", "type parameters are Python 3.12+"),
}



# =============================================================================
# Helpers
# =============================================================================


def is_bare_collection(annotation: ast.expr | None) -> str | None:
    """Return collection name if annotation is bare (e.g., list instead of list[int])."""
    if isinstance(annotation, ast.Name) and annotation.id in BARE_COLLECTIONS:
        return annotation.id
    return None


def is_obvious_literal(node: ast.expr) -> bool:
    """Check if node is a literal with obvious type (str, int, bool, float)."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, int, bool, float)) and node.value is not None
    return False


# =============================================================================
# Verifier
# =============================================================================


class Verifier(ast.NodeVisitor):
    """AST visitor that collects Tongues subset violations."""

    def __init__(self, file: Path, source: str, tree: ast.AST, verbose: bool = False) -> None:
        self.file = file
        self.source = source
        self.verbose = verbose
        self.violations: list[Violation] = []
        self.seen_nodes: dict[str, int] = {}  # Track node types seen
        self.function_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        # Collect all user-defined names (functions, classes, variables)
        self.defined_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.defined_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.defined_names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                self.defined_names.add(node.target.id)

    def visit(self, node: ast.AST) -> None:
        """Visit a node, checking if it's allowed/banned/unknown."""
        node_type = node.__class__.__name__
        self.seen_nodes[node_type] = self.seen_nodes.get(node_type, 0) + 1
        # Check if banned (with specific message)
        if node_type in BANNED_NODES:
            category, message = BANNED_NODES[node_type]
            self.add(node, category, message)
        # Check if unknown (not in allowed or banned)
        elif node_type not in ALLOWED_NODES:
            self.add(node, "syntax", f"{node_type}: unknown AST node", is_warning=True)
        # Call the specific visitor if it exists (for additional checks)
        visitor = getattr(self, f"visit_{node_type}", None)
        if visitor:
            visitor(node)
        else:
            self.generic_visit(node)

    def add(self, node: ast.AST, category: str, message: str, is_warning: bool = False) -> None:
        """Record a violation."""
        self.violations.append(Violation(
            file=self.file,
            line=getattr(node, "lineno", 0),
            col=getattr(node, "col_offset", 0),
            category=category,
            message=message,
            is_warning=is_warning,
        ))

    # -------------------------------------------------------------------------
    # Reflection
    # -------------------------------------------------------------------------

    def check_banned_call(self, node: ast.Call) -> None:
        """Check for banned built-in function calls."""
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in BANNED_BUILTINS:
                msg = BANNED_BUILTINS[name]
                self.add(node, "builtin", f"{name}(): {msg}")
            elif name not in ALLOWED_BUILTINS and name not in self.defined_names:
                # Warn on unknown names not defined in this file
                # Skip uppercase (user-defined classes) and _private names
                if name[0].islower() and not name.startswith("_"):
                    self.add(node, "builtin", f"{name}(): unknown builtin", is_warning=True)

    def check_reflection_attribute(self, node: ast.Attribute) -> None:
        """Check for dunder attribute access."""
        if node.attr in REFLECTION_DUNDERS:
            self.add(node, "reflection", f".{node.attr} requires runtime introspection")

    # -------------------------------------------------------------------------
    # Control flow
    # -------------------------------------------------------------------------


    def visit_For(self, node: ast.For) -> None:
        if node.orelse:
            self.add(node, "control", "loop else is not supported (use flag variable)")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if node.orelse:
            self.add(node, "control", "loop else is not supported (use flag variable)")
        self.generic_visit(node)


    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.add(node, "control", "bare except is not supported (specify exception type)")
        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # Functions
    # -------------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check for nested function
        if self.function_stack:
            self.add(node, "function", f"nested function '{node.name}' (move to module level)")
        # Check for **kwargs
        if node.args.kwarg:
            self.add(node, "function", "**kwargs is not supported (use explicit parameters)")
        # Check for dunder methods
        if node.name.startswith("__") and node.name.endswith("__"):
            if node.name not in ALLOWED_DUNDERS:
                self.add(node, "function", f"{node.name} has no portable equivalent")
        # Check for decorators
        for dec in node.decorator_list:
            dec_name = None
            if isinstance(dec, ast.Name):
                dec_name = dec.id
            elif isinstance(dec, ast.Attribute):
                dec_name = dec.attr
            if dec_name == "staticmethod":
                self.add(dec, "function", "@staticmethod is not supported (use module-level function)")
            elif dec_name == "classmethod":
                self.add(dec, "function", "@classmethod is not supported (use module-level function)")
            elif dec_name == "property":
                self.add(dec, "function", "@property is not supported (use explicit getter)")
            else:
                self.add(dec, "function", "decorators require runtime modification")
        # Check for mutable default arguments
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.add(node, "function", "mutable default argument is not supported (use None)")
                break
        # Check type annotations
        self.check_function_annotations(node)
        # Visit body with function on stack
        self.function_stack.append(node)
        self.generic_visit(node)
        self.function_stack.pop()

    # -------------------------------------------------------------------------
    # Classes
    # -------------------------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Check for multiple inheritance (excluding Exception)
        real_bases = [b for b in node.bases
                      if not (isinstance(b, ast.Name) and b.id == "Exception")]
        if len(real_bases) > 1:
            self.add(node, "class", "multiple inheritance has no portable equivalent (use composition)")
        # Check for class decorators
        for dec in node.decorator_list:
            self.add(dec, "class", "class decorators require runtime modification")
        # Check for nested classes
        for item in node.body:
            if isinstance(item, ast.ClassDef):
                self.add(item, "class", "nested class (move to module level)")
        # Check for unannotated field assignments
        self.check_unannotated_fields(node)
        self.generic_visit(node)

    def check_unannotated_fields(self, class_node: ast.ClassDef) -> None:
        """Check for self.x = val assignments that need type annotations."""
        annotated_fields: set[str] = set()
        # Collect annotated fields
        for item in ast.walk(class_node):
            if isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    annotated_fields.add(item.target.id)
                elif isinstance(item.target, ast.Attribute):
                    if isinstance(item.target.value, ast.Name) and item.target.value.id == "self":
                        annotated_fields.add(item.target.attr)
        # Check methods for unannotated field assignments
        for item in class_node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            annotated_params = {arg.arg for arg in item.args.args if arg.annotation is not None}
            for stmt in ast.walk(item):
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if not isinstance(target, ast.Attribute):
                        continue
                    if not isinstance(target.value, ast.Name) or target.value.id != "self":
                        continue
                    field = target.attr
                    if field in annotated_fields:
                        continue
                    # Skip if value is annotated param or obvious literal
                    if isinstance(stmt.value, ast.Name) and stmt.value.id in annotated_params:
                        annotated_fields.add(field)
                        continue
                    if is_obvious_literal(stmt.value):
                        annotated_fields.add(field)
                        continue
                    self.add(stmt, "types", f"self.{field} needs type annotation")
                    annotated_fields.add(field)

    # -------------------------------------------------------------------------
    # Imports
    # -------------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        self.add(node, "import", "import is not supported (code must be self-contained)")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        allowed = {"__future__", "typing", "collections.abc"}
        if node.module not in allowed:
            self.add(node, "import", f"from {node.module} import (code must be self-contained)")
        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # Statements
    # -------------------------------------------------------------------------

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                continue  # del d[key] is fine (equivalent to .pop())
            if isinstance(target, ast.Name):
                self.add(node, "statement", f"del {target.id} (unbinding variables is not portable)")
            else:
                self.add(node, "statement", "del is not supported for this target")
        self.generic_visit(node)


    def visit_Assign(self, node: ast.Assign) -> None:
        # Check for tuple unpack from variable
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            if isinstance(node.value, ast.Name):
                self.add(node, "expression", "tuple unpack from variable (unpack directly from function call)")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Check for bare collection types
        if bare := is_bare_collection(node.annotation):
            name = node.target.id if isinstance(node.target, ast.Name) else "?"
            self.add(node, "types", f"bare {bare}: {name} needs type parameter")
        self.generic_visit(node)

    # -------------------------------------------------------------------------
    # Expressions
    # -------------------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        self.check_banned_call(node)
        # Check for **kwargs in call
        for kw in node.keywords:
            if kw.arg is None:
                self.add(node, "expression", "**kwargs in call (pass arguments explicitly)")
                break
        # Check for *args in call
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                self.add(node, "expression", "*args in call (pass arguments explicitly)")
                break
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.check_reflection_attribute(node)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # Check for is/is not with non-None
        for i, op in enumerate(node.ops):
            if isinstance(op, (ast.Is, ast.IsNot)):
                left = node.left if i == 0 else node.comparators[i - 1]
                right = node.comparators[i]
                left_none = isinstance(left, ast.Constant) and left.value is None
                right_none = isinstance(right, ast.Constant) and right.value is None
                if not left_none and not right_none:
                    self.add(node, "expression", "is/is not only for None (use == for values)")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Check for or-default pattern: x or []
        if isinstance(node.op, ast.Or):
            for val in node.values[1:]:
                if isinstance(val, (ast.List, ast.Dict, ast.Set)):
                    self.add(node, "expression", "or-default pattern is not supported (use if/else)")
                    break
                if isinstance(val, ast.Constant) and val.value in (0, "", False):
                    self.add(node, "expression", "or-default pattern is not supported (use if/else)")
                    break
        self.generic_visit(node)


    # -------------------------------------------------------------------------
    # Type annotations
    # -------------------------------------------------------------------------

    def check_function_annotations(self, node: ast.FunctionDef) -> None:
        """Check for missing or bare type annotations."""
        # Missing return type (skip __init__, __new__)
        if node.returns is None and node.name not in ("__init__", "__new__"):
            self.add(node, "types", f"missing return type: def {node.name}() -> ...")
        # Bare return type
        if bare := is_bare_collection(node.returns):
            self.add(node, "types", f"bare {bare}: {node.name}() return needs type parameter")
        # Check parameters
        for i, arg in enumerate(node.args.args):
            if arg.arg in ("self", "cls") and i == 0:
                continue
            if arg.annotation is None:
                self.add(node, "types", f"missing param type: {arg.arg} in {node.name}()")
            elif bare := is_bare_collection(arg.annotation):
                self.add(node, "types", f"bare {bare}: {arg.arg} in {node.name}() needs type parameter")


# =============================================================================
# Public API
# =============================================================================


@dataclass
class VerifyResult:
    """Result of verification including violations and node statistics."""
    violations: list[Violation]
    seen_nodes: dict[str, int]


def verify_source(source: str, file: Path | None = None) -> VerifyResult:
    """Verify source code against the Tongues subset."""
    file = file or Path("<stdin>")
    tree = ast.parse(source, filename=str(file))
    verifier = Verifier(file, source, tree)
    verifier.visit(tree)
    return VerifyResult(violations=verifier.violations, seen_nodes=verifier.seen_nodes)


def verify_file(path: Path) -> VerifyResult:
    """Verify a file against the Tongues subset."""
    source = path.read_text()
    return verify_source(source, path)
