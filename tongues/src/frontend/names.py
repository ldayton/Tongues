"""Phase 4: Scope analysis and name binding.

Builds a symbol table mapping names to their declarations. Validates that all
referenced names resolve. Since Phase 3 guarantees no nested functions and no
global/nonlocal, scoping is simple: local → module → builtin.

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from typing import Callable

from .types import (
    ASTNode,
    JDict,
    JList,
    JStr,
    get_int,
    get_node,
    get_nodes,
    get_str,
    has_key,
)


# Allowed builtins (copied from subset.py)
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
        self.current_class: str = ""
        self.current_func: str = ""

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
        self.current_class = class_name
        self.current_func = func_name
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
        self.current_class = ""
        self.current_func = ""

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
        """Try to resolve a name: local → module → builtin."""
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
