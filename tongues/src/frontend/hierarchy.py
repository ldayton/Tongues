"""Phase 7: Class hierarchy analysis.

Build the class hierarchy and classify structs. Detects the hierarchy root,
marks node subclasses and exception subclasses, validates no cycles exist.

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from __future__ import annotations


# Type alias for AST dict nodes
ASTNode = dict[str, object]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class HierarchyError:
    """An error found during hierarchy analysis."""

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
            + ": [hierarchy] "
            + self.message
        )


class HierarchyResult:
    """Result of hierarchy analysis."""

    def __init__(self) -> None:
        self.hierarchy_root: str | None = None
        self.node_types: list[str] = []
        self.exception_types: list[str] = []
        self.ancestors: dict[str, list[str]] = {}
        self._errors: list[HierarchyError] = []

    def add_error(self, lineno: int, col: int, message: str) -> None:
        self._errors.append(HierarchyError(lineno, col, message))

    def errors(self) -> list[HierarchyError]:
        return self._errors

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

    def to_dict(self) -> dict[str, object]:
        """Serialize to nested dicts for test assertions."""
        d: dict[str, object] = {
            "root": self.hierarchy_root,
            "node_types": list(self.node_types),
            "exception_types": list(self.exception_types),
        }
        ancestors: dict[str, object] = {}
        akeys = list(self.ancestors.keys())
        i = 0
        while i < len(akeys):
            ancestors[akeys[i]] = list(self.ancestors[akeys[i]])
            i += 1
        d["ancestors"] = ancestors
        return d


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def _detect_cycles(
    class_bases: dict[str, list[str]],
    errors: list[HierarchyError],
) -> bool:
    """Check for cycles in the inheritance graph. Returns True if cycle found."""
    ckeys = list(class_bases.keys())
    i = 0
    while i < len(ckeys):
        name = ckeys[i]
        visited: set[str] = set()
        current: str | None = name
        while current is not None:
            if current in visited:
                errors.append(HierarchyError(0, 0, "cycle in inheritance: " + name))
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


def _find_hierarchy_root(
    known_classes: set[str],
    class_bases: dict[str, list[str]],
    exception_cache: dict[str, bool],
) -> str | None:
    """Find the single root of the class hierarchy.

    A class is the root if it has no base classes, is used as a base by
    at least one other class, and is the only such root. Exception classes
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
    # Find roots: used as base, no base themselves, in known_classes
    roots: list[str] = []
    ukeys = list(used_as_base)
    i = 0
    while i < len(ukeys):
        name = ukeys[i]
        if name in known_classes:
            bases = class_bases.get(name)
            if bases is None or len(bases) == 0:
                roots.append(name)
        i += 1
    if len(roots) == 1:
        return roots[0]
    return None


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
) -> HierarchyResult:
    """Build the class hierarchy and classify structs.

    Args:
        known_classes: Set of known class names.
        class_bases: Dict mapping class name to list of base class names.
    """
    result = HierarchyResult()
    # Validate base classes exist
    ckeys = list(class_bases.keys())
    i = 0
    while i < len(ckeys):
        name = ckeys[i]
        bases = class_bases[name]
        j = 0
        while j < len(bases):
            base = bases[j]
            if base != "Exception" and base not in known_classes:
                result.add_error(0, 0, "'" + base + "' is not defined")
                return result
            j += 1
        i += 1
    # Detect cycles
    if _detect_cycles(class_bases, result._errors):
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
    # Find hierarchy root
    result.hierarchy_root = _find_hierarchy_root(
        known_classes, class_bases, exception_cache
    )
    # Classify node types
    if result.hierarchy_root is not None:
        node_cache: dict[str, bool] = {}
        i = 0
        while i < len(ckeys):
            name = ckeys[i]
            if _is_node_subclass(name, result.hierarchy_root, class_bases, node_cache):
                result.node_types.append(name)
            i += 1
    return result
