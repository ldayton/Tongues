"""Phase 7: Class hierarchy analysis.

Build the class hierarchy and classify structs. Detects the hierarchy root,
marks node subclasses and exception subclasses, validates no cycles exist.

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from __future__ import annotations

from .types import JsonValue, JStr, JList, JDict, JNull


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class HierarchyError:
    """An error found during hierarchy analysis."""

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
            + ": [hierarchy] "
            + self.message
        )


class HierarchyResult:
    """Result of hierarchy analysis."""

    def __init__(self) -> None:
        self.hierarchy_root: str | None = None
        self.hierarchy_roots: list[str] = []
        self.node_types: list[str] = []
        self.exception_types: list[str] = []
        self.ancestors: dict[str, list[str]] = {}
        self._errors: list[HierarchyError] = []

    def add_error(
        self, lineno: int, col: int, message: str, source_file: str = ""
    ) -> None:
        self._errors.append(HierarchyError(lineno, col, message, source_file))

    def errors(self) -> list[HierarchyError]:
        return self._errors

    def is_hierarchy_root(self, name: str) -> bool:
        """Check if name is any hierarchy root."""
        i = 0
        while i < len(self.hierarchy_roots):
            if self.hierarchy_roots[i] == name:
                return True
            i += 1
        return False

    def root_of(self, name: str) -> str | None:
        """Find the nearest hierarchy root ancestor (or self if root)."""
        if self.is_hierarchy_root(name):
            return name
        cur = name
        while True:
            ancestors = self.ancestors.get(cur)
            if ancestors is None or len(ancestors) == 0:
                return None
            parent = ancestors[0]
            if self.is_hierarchy_root(parent):
                return parent
            cur = parent

    def is_node(self, name: str) -> bool:
        """Check if name is a node type."""
        i = 0
        while i < len(self.node_types):
            if self.node_types[i] == name:
                return True
            i += 1
        return False

    def is_exception(self, name: str) -> bool:
        """Check if name is an exception type."""
        i = 0
        while i < len(self.exception_types):
            if self.exception_types[i] == name:
                return True
            i += 1
        return False

    def to_dict(self) -> JsonValue:
        """Serialize to nested JsonValue dicts for test assertions."""
        root_jv: JsonValue = JNull()
        if self.hierarchy_root is not None:
            root_jv = JStr(self.hierarchy_root)
        node_types_jv: list[JsonValue] = []
        i = 0
        while i < len(self.node_types):
            node_types_jv.append(JStr(self.node_types[i]))
            i += 1
        exception_types_jv: list[JsonValue] = []
        i = 0
        while i < len(self.exception_types):
            exception_types_jv.append(JStr(self.exception_types[i]))
            i += 1
        ancestors: dict[str, JsonValue] = {}
        akeys = list(self.ancestors.keys())
        i = 0
        while i < len(akeys):
            ancestor_list: list[JsonValue] = []
            j = 0
            while j < len(self.ancestors[akeys[i]]):
                ancestor_list.append(JStr(self.ancestors[akeys[i]][j]))
                j += 1
            ancestors[akeys[i]] = JList(ancestor_list)
            i += 1
        return JDict(
            {
                "root": root_jv,
                "node_types": JList(node_types_jv),
                "exception_types": JList(exception_types_jv),
                "ancestors": JDict(ancestors),
            }
        )


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def _detect_cycles(
    class_bases: dict[str, list[str]],
    errors: list[HierarchyError],
    class_source_files: dict[str, str],
) -> bool:
    """Check for cycles in the inheritance graph. Returns True if cycle found."""
    ckeys = list(class_bases.keys())
    i = 0
    while i < len(ckeys):
        name = ckeys[i]
        visited: set[str] = set()
        current: str | None = name
        while current is not None:
            assert current is not None
            if current in visited:
                sf = class_source_files.get(name, "")
                errors.append(HierarchyError(0, 0, "cycle in inheritance: " + name, sf))
                return True
            visited.add(current)
            bases = class_bases.get(current)
            if bases is not None and len(bases) > 0:
                current = bases[0]
            else:
                current = None
        i += 1
    return False


# ---------------------------------------------------------------------------
# Exception subclass detection
# ---------------------------------------------------------------------------


def _is_exception_subclass(
    name: str,
    class_bases: dict[str, list[str]],
    cache: dict[str, bool],
) -> bool:
    """Check if a class is an Exception subclass (directly or transitively)."""
    if name == "Exception":
        return True
    if name in cache:
        return cache[name]
    bases = class_bases.get(name)
    if bases is None or len(bases) == 0:
        cache[name] = False
        return False
    i = 0
    while i < len(bases):
        if _is_exception_subclass(bases[i], class_bases, cache):
            cache[name] = True
            return True
        i += 1
    cache[name] = False
    return False


# ---------------------------------------------------------------------------
# Hierarchy root detection
# ---------------------------------------------------------------------------


def _find_hierarchy_roots(
    known_classes: set[str],
    class_bases: dict[str, list[str]],
    exception_cache: dict[str, bool],
) -> list[str]:
    """Find all roots of class hierarchies.

    A class is a root if it has no base classes, is used as a base by
    at least one other class, and is a known class. Exception classes
    and their subclasses are excluded.
    """
    # Find all classes used as a base
    used_as_base: set[str] = set()
    ckeys = list(class_bases.keys())
    i = 0
    while i < len(ckeys):
        name = ckeys[i]
        bases = class_bases[name]
        j = 0
        while j < len(bases):
            base = bases[j]
            if base != "Exception" and not _is_exception_subclass(
                base, class_bases, exception_cache
            ):
                used_as_base.add(base)
            j += 1
        i += 1
    # Find roots: used as base and in known_classes
    roots: list[str] = []
    ukeys = list(used_as_base)
    i = 0
    while i < len(ukeys):
        name = ukeys[i]
        if name in known_classes:
            roots.append(name)
        i += 1
    return roots


# ---------------------------------------------------------------------------
# Node subclass detection
# ---------------------------------------------------------------------------


def _is_node_subclass(
    name: str,
    hierarchy_root: str,
    class_bases: dict[str, list[str]],
    cache: dict[str, bool],
) -> bool:
    """Check if a class is a node subclass (transitively inherits from root)."""
    if name == hierarchy_root:
        return True
    if name in cache:
        return cache[name]
    bases = class_bases.get(name)
    if bases is None or len(bases) == 0:
        cache[name] = False
        return False
    i = 0
    while i < len(bases):
        if _is_node_subclass(bases[i], hierarchy_root, class_bases, cache):
            cache[name] = True
            return True
        i += 1
    cache[name] = False
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_hierarchy(
    known_classes: set[str],
    class_bases: dict[str, list[str]],
    class_source_files: dict[str, str] | None = None,
) -> HierarchyResult:
    """Build the class hierarchy and classify structs.

    Args:
        known_classes: Set of known class names.
        class_bases: Dict mapping class name to list of base class names.
        class_source_files: Optional mapping of class names to source file paths.
    """
    if class_source_files is None:
        class_source_files = {}
    result = HierarchyResult()
    # Validate base classes exist
    ckeys = list(class_bases.keys())
    i = 0
    while i < len(ckeys):
        name = ckeys[i]
        bases = class_bases[name]
        sf = class_source_files.get(name, "")
        j = 0
        while j < len(bases):
            base = bases[j]
            if base != "Exception" and base not in known_classes:
                result.add_error(0, 0, "'" + base + "' is not defined", sf)
                return result
            j += 1
        i += 1
    # Detect cycles
    if _detect_cycles(class_bases, result._errors, class_source_files):
        return result
    # Build ancestor lists (direct bases only)
    i = 0
    while i < len(ckeys):
        name = ckeys[i]
        bases = class_bases.get(name, [])
        ancestors: list[str] = []
        j = 0
        while j < len(bases):
            if bases[j] != "Exception":
                ancestors.append(bases[j])
            j += 1
        result.ancestors[name] = ancestors
        i += 1
    # Detect exception subclasses
    exception_cache: dict[str, bool] = {}
    i = 0
    while i < len(ckeys):
        name = ckeys[i]
        if _is_exception_subclass(name, class_bases, exception_cache):
            result.exception_types.append(name)
        i += 1
    # Find hierarchy roots
    result.hierarchy_roots = _find_hierarchy_roots(
        known_classes, class_bases, exception_cache
    )
    if len(result.hierarchy_roots) == 1:
        result.hierarchy_root = result.hierarchy_roots[0]
    # Classify node types
    ri = 0
    while ri < len(result.hierarchy_roots):
        root = result.hierarchy_roots[ri]
        node_cache: dict[str, bool] = {}
        i = 0
        while i < len(ckeys):
            name = ckeys[i]
            if _is_node_subclass(name, root, class_bases, node_cache):
                if not result.is_node(name):
                    result.node_types.append(name)
            i += 1
        ri += 1
    return result
