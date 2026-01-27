"""AST parsing utilities for the bootstrap layer.

This module handles Python-specific AST parsing that cannot be expressed
in the Tongues subset. It provides dict-based AST that can be passed to
the Tongues-compliant verifier.
"""

import ast
import base64
from pathlib import Path


def ast_to_dict(node: ast.AST, include_locations: bool = True) -> dict:
    """Convert an AST node to a dictionary representation."""
    result = {"_type": node.__class__.__name__}
    for field, value in ast.iter_fields(node):
        result[field] = convert_value(value, include_locations)
    if include_locations:
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(node, attr):
                result[attr] = getattr(node, attr)
    return result


def convert_value(value, include_locations: bool = True):
    """Convert a value from an AST field to its dict representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"_bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, complex):
        return {"_complex": [value.real, value.imag]}
    if value is ...:
        return {"_ellipsis": True}
    if isinstance(value, list):
        return [convert_value(item, include_locations) for item in value]
    if isinstance(value, ast.AST):
        return ast_to_dict(value, include_locations)
    raise TypeError(f"Unexpected AST value type: {type(value).__name__}")


def parse_source(source: str, filename: str = "<string>") -> dict:
    """Parse Python source and return dict-based AST."""
    tree = ast.parse(source, filename=filename)
    return ast_to_dict(tree)


def parse_file(path: Path) -> dict:
    """Parse a Python file and return dict-based AST."""
    source = path.read_text(encoding="utf-8")
    return ast_to_dict(ast.parse(source, filename=str(path)))


def load_bootstrap(path: Path) -> dict:
    """Load a bootstrap output module and return its AST dict."""
    content = path.read_text(encoding="utf-8")
    namespace: dict = {}
    exec(content, namespace)
    return namespace["AST"]


def walk(node: dict):
    """Recursively yield all descendant nodes in the dict-based AST."""
    yield node
    for key, value in node.items():
        if key.startswith("_") or key in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            continue
        if isinstance(value, dict) and "_type" in value:
            yield from walk(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "_type" in item:
                    yield from walk(item)


def get_children(node: dict) -> list[dict]:
    """Get immediate child nodes of a dict-based AST node."""
    children = []
    for key, value in node.items():
        if key.startswith("_") or key in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            continue
        if isinstance(value, dict) and "_type" in value:
            children.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "_type" in item:
                    children.append(item)
    return children
