"""Tests for the Tongues verifier."""

import ast
import builtins

import pytest
from bootstrap.cli import verify_source
from tongues.verify import ALLOWED_BUILTINS, BANNED_BUILTINS, ALLOWED_NODES, BANNED_NODES


def violations(source: str) -> list[tuple[str, str]]:
    """Return (category, message) pairs for all violations."""
    return [(v.category, v.message) for v in verify_source(source).violations]


def categories(source: str) -> list[str]:
    """Return just the categories of violations."""
    return [v.category for v in verify_source(source).violations]


def errors(source: str) -> list[str]:
    """Return categories of errors only."""
    return [v.category for v in verify_source(source).violations if not v.is_warning]


def warnings(source: str) -> list[str]:
    """Return categories of warnings only."""
    return [v.category for v in verify_source(source).violations if v.is_warning]


def seen_nodes(source: str) -> dict[str, int]:
    """Return the AST nodes seen during verification."""
    return verify_source(source).seen_nodes


# =============================================================================
# Reflection
# =============================================================================


@pytest.mark.parametrize("func", [
    # Reflection
    "getattr", "setattr", "hasattr", "delattr",
    "type", "vars", "dir", "globals", "locals",
    "id", "callable", "eval", "exec", "compile",
    "issubclass", "hash", "format", "memoryview",
    # Lazy iterators
    "iter", "next", "aiter", "anext",
    "map", "filter", "zip", "enumerate", "reversed",
    # I/O
    "open", "input", "print",
    # Interactive
    "breakpoint", "help", "exit", "quit",
])
def test_banned_builtins(func: str) -> None:
    assert "builtin" in errors(f"x = {func}(y)")


@pytest.mark.parametrize("func", [
    "abs", "min", "max", "sum", "len", "range", "round", "divmod", "sorted", "pow",
    "all", "any", "int", "float", "str", "bool", "list", "dict", "set", "tuple",
    "ord", "chr", "bin", "hex", "oct", "repr", "isinstance", "super",
])
def test_allowed_builtins(func: str) -> None:
    assert "builtin" not in categories(f"x = {func}(y)")


def test_unknown_builtin_warns() -> None:
    src = "def f() -> None:\n someunknownfunc()"
    assert "builtin" in warnings(src)
    assert "builtin" not in errors(src)


def test_user_defined_no_warning() -> None:
    src = "def my_func() -> None: pass\ndef f() -> None:\n my_func()"
    assert "builtin" not in warnings(src)


@pytest.mark.parametrize("dunder", [
    "__class__", "__dict__", "__name__", "__module__",
    "__bases__", "__mro__", "__subclasses__",
])
def test_reflection_dunders(dunder: str) -> None:
    assert "reflection" in categories(f"x = obj.{dunder}")


# =============================================================================
# Async
# =============================================================================


def test_async_def() -> None:
    assert "async" in categories("async def f() -> None: pass")


# =============================================================================
# Generators
# =============================================================================


def test_yield() -> None:
    src = "def f() -> int:\n yield 1"
    assert "generator" in categories(src)


def test_yield_from() -> None:
    src = "def f() -> int:\n yield from [1, 2]"
    assert "generator" in categories(src)


def test_generator_expression() -> None:
    src = "def f() -> int:\n return sum(x for x in [1, 2])"
    assert "generator" in categories(src)


# =============================================================================
# Control flow
# =============================================================================


def test_with_statement() -> None:
    src = "def f() -> None:\n with open('x'): pass"
    assert "control" in categories(src)


def test_loop_else() -> None:
    src = "def f() -> None:\n for x in []: pass\n else: pass"
    assert "control" in categories(src)


def test_bare_except() -> None:
    src = "def f() -> None:\n try: pass\n except: pass"
    assert "control" in categories(src)


# =============================================================================
# Functions
# =============================================================================


def test_lambda() -> None:
    assert "function" in categories("f = lambda x: x")


def test_global() -> None:
    src = "x = 1\ndef f() -> None:\n global x"
    assert "function" in categories(src)


def test_nonlocal() -> None:
    # Note: nonlocal requires nested function which is also banned
    src = "def f() -> None:\n x = 1\n def g() -> None:\n  nonlocal x"
    cats = categories(src)
    assert "function" in cats


def test_nested_function() -> None:
    src = "def f() -> None:\n def g() -> None: pass"
    assert "function" in categories(src)


def test_kwargs_param() -> None:
    src = "def f(**kwargs) -> None: pass"
    assert "function" in categories(src)


def test_staticmethod() -> None:
    src = "class C:\n @staticmethod\n def f() -> None: pass"
    assert "function" in categories(src)


def test_classmethod() -> None:
    src = "class C:\n @classmethod\n def f(cls) -> None: pass"
    assert "function" in categories(src)


def test_property() -> None:
    src = "class C:\n @property\n def x(self) -> int: return 1"
    assert "function" in categories(src)


def test_decorator() -> None:
    src = "@decorator\ndef f() -> None: pass"
    assert "function" in categories(src)


def test_mutable_default() -> None:
    src = "def f(x: list[int] = []) -> None: pass"
    assert "function" in categories(src)


def test_banned_dunder() -> None:
    src = "class C:\n def __eq__(self, other: object) -> bool: return True"
    assert "function" in categories(src)


def test_allowed_dunders() -> None:
    src = "class C:\n def __init__(self) -> None: pass\n def __repr__(self) -> str: return ''"
    assert "function" not in categories(src)


# =============================================================================
# Classes
# =============================================================================


def test_multiple_inheritance() -> None:
    src = "class C(A, B): pass"
    assert "class" in categories(src)


def test_exception_inheritance_ok() -> None:
    src = "class MyError(Exception): pass"
    assert "class" not in categories(src)


def test_class_decorator() -> None:
    src = "@dataclass\nclass C: pass"
    assert "class" in categories(src)


def test_nested_class() -> None:
    src = "class Outer:\n class Inner: pass"
    assert "class" in categories(src)


def test_unannotated_field() -> None:
    src = "class C:\n def __init__(self) -> None:\n  self.x = some_func()"
    assert "types" in categories(src)


def test_annotated_field_ok() -> None:
    src = "class C:\n x: int\n def __init__(self) -> None:\n  self.x = 1"
    assert "types" not in categories(src)


def test_field_from_annotated_param_ok() -> None:
    src = "class C:\n def __init__(self, x: int) -> None:\n  self.x = x"
    assert "types" not in categories(src)


def test_field_literal_ok() -> None:
    src = "class C:\n def __init__(self) -> None:\n  self.x = 42"
    assert "types" not in categories(src)


# =============================================================================
# Imports
# =============================================================================


def test_import() -> None:
    assert "import" in categories("import os")


def test_from_import_banned() -> None:
    assert "import" in categories("from os import path")


def test_from_typing_ok() -> None:
    assert "import" not in categories("from typing import List")


def test_from_future_ok() -> None:
    assert "import" not in categories("from __future__ import annotations")


def test_from_collections_abc_ok() -> None:
    assert "import" not in categories("from collections.abc import Iterator")


# =============================================================================
# Statements
# =============================================================================


def test_del_variable() -> None:
    src = "def f() -> None:\n x = 1\n del x"
    assert "statement" in categories(src)


def test_del_subscript_ok() -> None:
    src = "def f(d: dict[str, int]) -> None:\n del d['key']"
    assert "statement" not in categories(src)


def test_tuple_unpack_from_variable() -> None:
    src = "def f(t: tuple[int, int]) -> None:\n a, b = t"
    assert "expression" in categories(src)


def test_tuple_unpack_from_call_ok() -> None:
    src = "def f() -> None:\n a, b = divmod(10, 3)"
    assert "expression" not in categories(src)


# =============================================================================
# Expressions
# =============================================================================


def test_kwargs_in_call() -> None:
    src = "def f() -> None:\n g(**d)"
    assert "expression" in categories(src)


def test_args_in_call() -> None:
    src = "def f() -> None:\n g(*args)"
    assert "expression" in categories(src)


def test_is_with_non_none() -> None:
    src = "def f(x: int, y: int) -> bool:\n return x is y"
    assert "expression" in categories(src)


def test_is_none_ok() -> None:
    src = "def f(x: int | None) -> bool:\n return x is None"
    assert "expression" not in categories(src)


def test_or_default() -> None:
    src = "def f(x: list[int] | None) -> list[int]:\n return x or []"
    assert "expression" in categories(src)


# =============================================================================
# Types
# =============================================================================


def test_missing_return_type() -> None:
    assert "types" in categories("def f(): pass")


def test_missing_param_type() -> None:
    assert "types" in categories("def f(x) -> None: pass")


def test_bare_collection_annotation() -> None:
    src = "def f() -> None:\n x: list = []"
    assert "types" in categories(src)


def test_bare_list_return() -> None:
    assert "types" in categories("def f() -> list: pass")


def test_bare_list_param() -> None:
    assert "types" in categories("def f(x: list) -> None: pass")


def test_parameterized_list_ok() -> None:
    src = "def f(x: list[int]) -> list[str]:\n y: list[bool] = []\n return []"
    # Should only have the missing return value issue, not bare list
    cats = categories(src)
    assert cats.count("types") == 0


# =============================================================================
# Builtin coverage
# =============================================================================


def test_builtin_lists_comprehensive() -> None:
    """Ensure every callable builtin is either allowed or banned."""
    # Get all builtin names
    all_builtins = set(dir(builtins))

    # Exclude non-callable constants
    constants = {"True", "False", "None", "Ellipsis", "NotImplemented", "__debug__"}

    # Exclude private/internal names
    private = {name for name in all_builtins if name.startswith("_") and name != "__import__"}

    # Exclude exception classes (they inherit from BaseException)
    exceptions = set()
    for name in all_builtins:
        obj = getattr(builtins, name, None)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            exceptions.add(name)

    # The builtins we need to categorize
    to_categorize = all_builtins - constants - private - exceptions

    # Check coverage
    categorized = set(ALLOWED_BUILTINS) | set(BANNED_BUILTINS)
    uncategorized = to_categorize - categorized

    assert not uncategorized, f"Uncategorized builtins: {sorted(uncategorized)}"


def test_no_overlap_between_allowed_and_banned() -> None:
    """Ensure no builtin is both allowed and banned."""
    overlap = set(ALLOWED_BUILTINS) & set(BANNED_BUILTINS)
    assert not overlap, f"Builtins in both lists: {sorted(overlap)}"


# =============================================================================
# AST node coverage
# =============================================================================


def test_ast_node_lists_comprehensive() -> None:
    """Ensure every AST node type is either allowed or banned."""
    # Get all AST node classes
    all_ast_nodes = set()
    for name in dir(ast):
        obj = getattr(ast, name)
        if isinstance(obj, type) and issubclass(obj, ast.AST) and obj is not ast.AST:
            all_ast_nodes.add(name)

    # Exclude internal/deprecated/abstract nodes
    internal = {
        "AST",  # Base class
        "Index", "ExtSlice", "Suite",  # Deprecated
        "AugLoad", "AugStore", "Param",  # Context only
        "FunctionType",  # Type comments
        "TypeIgnore",  # Type comments
        "NodeVisitor", "NodeTransformer",  # Not nodes
        "Interpolation", "TemplateStr",  # 3.14+ t-strings
        # Abstract base classes (lowercase)
        "boolop", "cmpop", "excepthandler", "expr", "expr_context",
        "mod", "operator", "pattern", "slice", "stmt", "type_ignore",
        "type_param", "unaryop",
        # Internal
        "_ast_Ellipsis",
    }

    to_categorize = all_ast_nodes - internal
    categorized = set(ALLOWED_NODES) | set(BANNED_NODES.keys())
    uncategorized = to_categorize - categorized

    assert not uncategorized, f"Uncategorized AST nodes: {sorted(uncategorized)}"


def test_no_overlap_between_allowed_and_banned_nodes() -> None:
    """Ensure no AST node is both allowed and banned."""
    overlap = set(ALLOWED_NODES) & set(BANNED_NODES.keys())
    assert not overlap, f"AST nodes in both lists: {sorted(overlap)}"


# =============================================================================
# Clean code should pass
# =============================================================================


def test_clean_class() -> None:
    src = """
class Point:
    x: int
    y: int

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def distance(self, other: "Point") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5
"""
    assert violations(src) == []


def test_fully_clean() -> None:
    src = """
from typing import Optional

def factorial(n: int) -> int:
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result = result * i
    return result

def is_even(n: int) -> bool:
    return n % 2 == 0
"""
    assert violations(src) == []
