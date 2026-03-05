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
        if self.source_file:
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
        return name in self.hierarchy_roots

    def root_of(self, name: str) -> str | None:
        """Find the nearest hierarchy root ancestor (or self if root)."""
        if self.is_hierarchy_root(name):
            return name
        cur = name
        while True:
            cur_ancestors = self.ancestors.get(cur)
            if cur_ancestors is None or not cur_ancestors:
                return None
            parent = cur_ancestors[0]
            if self.is_hierarchy_root(parent):
                return parent
            cur = parent

    def is_node(self, name: str) -> bool:
        """Check if name is a node type."""
        return name in self.node_types

    def is_exception(self, name: str) -> bool:
        """Check if name is an exception type."""
        return name in self.exception_types

    def to_dict(self) -> JsonValue:
        """Serialize to nested JsonValue dicts for test assertions."""
        root_jv: JsonValue = JNull()
        if self.hierarchy_root is not None:
            root_jv = JStr(self.hierarchy_root)
        node_types_jv: list[JsonValue] = []
        for nt in self.node_types:
            node_types_jv.append(JStr(nt))
        exception_types_jv: list[JsonValue] = []
        for et in self.exception_types:
            exception_types_jv.append(JStr(et))
        ancestors: dict[str, JsonValue] = {}
        for akey in self.ancestors:
            ancestor_list: list[JsonValue] = []
            for anc in self.ancestors[akey]:
                ancestor_list.append(JStr(anc))
            ancestors[akey] = JList(ancestor_list)
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
    for name in class_bases:
        visited: set[str] = set()
        cur: str = name
        while True:
            if cur in visited:
                sf = class_source_files.get(name, "")
                errors.append(HierarchyError(0, 0, "cycle in inheritance: " + name, sf))
                return True
            visited.add(cur)
            bases = class_bases.get(cur)
            if bases is not None and bases:
                cur = bases[0]
            else:
                break
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
    if bases is None or not bases:
        cache[name] = False
        return False
    for base in bases:
        if _is_exception_subclass(base, class_bases, cache):
            cache[name] = True
            return True
    cache[name] = False
    return False


# ---------------------------------------------------------------------------
# Hierarchy root detection
# ---------------------------------------------------------------------------


def _find_hierarchy_roots(
    known_classes: dict[str, str],
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
    for name in class_bases:
        bases = class_bases[name]
        for base in bases:
            if base != "Exception" and not _is_exception_subclass(
                base, class_bases, exception_cache
            ):
                used_as_base.add(base)
    # Find roots: used as base and in known_classes
    roots: list[str] = []
    for name in used_as_base:
        if name in known_classes:
            roots.append(name)
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
    if bases is None or not bases:
        cache[name] = False
        return False
    for base in bases:
        if _is_node_subclass(base, hierarchy_root, class_bases, cache):
            cache[name] = True
            return True
    cache[name] = False
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_hierarchy(
    known_classes: dict[str, str],
    class_bases: dict[str, list[str]],
    class_source_files: dict[str, str] | None = None,
) -> HierarchyResult:
    """Build the class hierarchy and classify structs.

    Args:
        known_classes: Set of known class names.
        class_bases: Dict mapping class name to list of base class names.
        class_source_files: Optional mapping of class names to source file paths.
    """
    src_files: dict[str, str] = (
        class_source_files if class_source_files is not None else {}
    )
    result = HierarchyResult()
    # Validate base classes exist
    for name in class_bases:
        bases = class_bases[name]
        sf = src_files.get(name, "")
        for base in bases:
            if base != "Exception" and base not in known_classes:
                result.add_error(0, 0, "'" + base + "' is not defined", sf)
                return result
    # Detect cycles
    if _detect_cycles(class_bases, result._errors, src_files):
        return result
    # Build ancestor lists (direct bases only)
    for name in class_bases:
        bases = class_bases.get(name, [])
        ancestors: list[str] = []
        for base in bases:
            if base != "Exception":
                ancestors.append(base)
        result.ancestors[name] = ancestors
    # Detect exception subclasses
    exception_cache: dict[str, bool] = {}
    for name in class_bases:
        if _is_exception_subclass(name, class_bases, exception_cache):
            result.exception_types.append(name)
    # Find hierarchy roots
    result.hierarchy_roots = _find_hierarchy_roots(
        known_classes, class_bases, exception_cache
    )
    if len(result.hierarchy_roots) == 1:
        result.hierarchy_root = result.hierarchy_roots[0]
    # Classify node types
    for root in result.hierarchy_roots:
        node_cache: dict[str, bool] = {}
        for name in class_bases:
            if _is_node_subclass(name, root, class_bases, node_cache):
                if not result.is_node(name):
                    result.node_types.append(name)
    return result
