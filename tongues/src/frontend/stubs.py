"""Stub types for stdlib modules used by Tongues source."""

from __future__ import annotations

from .types import (
    TypeNode,
    FuncType,
    SliceType,
    MapType,
    OptionalType,
    PrimitiveType,
    INT_TYPE,
    STR_TYPE,
    BOOL_TYPE,
    BYTES_TYPE,
    VOID_TYPE,
)

# Synthetic types for IO objects (sys.stdin, sys.stdout, sys.stderr)
TEXTIO_TYPE = PrimitiveType("TextIO")
BYTESIO_TYPE = PrimitiveType("BytesIO")

# Module types - used when a module is referenced as a value (not just attribute access)
MODULE_TYPES: dict[str, TypeNode] = {
    "sys": PrimitiveType("module:sys"),
    "os": PrimitiveType("module:os"),
}


def lookup_module_type(name: str) -> TypeNode | None:
    """Look up a module type by name."""
    return MODULE_TYPES.get(name)

# Module-level attributes: MODULE_ATTRS[module][attr] -> TypeNode
MODULE_ATTRS: dict[str, dict[str, TypeNode]] = {
    "sys": {
        "argv": SliceType(STR_TYPE),
        "stdin": TEXTIO_TYPE,
        "stdout": TEXTIO_TYPE,
        "stderr": TEXTIO_TYPE,
    },
    "os": {
        "environ": MapType(STR_TYPE, STR_TYPE),
    },
}

# Module-level functions: MODULE_FUNCS[module][func] -> FuncType
MODULE_FUNCS: dict[str, dict[str, FuncType]] = {
    "sys": {
        "exit": FuncType([INT_TYPE], VOID_TYPE),
    },
    "os": {
        "getenv": FuncType([STR_TYPE], OptionalType(STR_TYPE)),
    },
}

# Stub type methods: STUB_METHODS[type_name][method] -> FuncType
STUB_METHODS: dict[str, dict[str, FuncType]] = {
    "TextIO": {
        "read": FuncType([], STR_TYPE),
        "readline": FuncType([], STR_TYPE),
        "write": FuncType([STR_TYPE], INT_TYPE),
        "isatty": FuncType([], BOOL_TYPE),
    },
    "BytesIO": {
        "read": FuncType([], BYTES_TYPE),
        "write": FuncType([BYTES_TYPE], INT_TYPE),
    },
}

# Stub type attributes: STUB_ATTRS[type_name][attr] -> TypeNode
STUB_ATTRS: dict[str, dict[str, TypeNode]] = {
    "TextIO": {
        "buffer": BYTESIO_TYPE,
    },
}


def lookup_module_attr(module: str, attr: str) -> TypeNode | None:
    """Look up a module-level attribute type."""
    mod = MODULE_ATTRS.get(module)
    if mod is not None:
        return mod.get(attr)
    return None


def lookup_module_func(module: str, func: str) -> FuncType | None:
    """Look up a module-level function type."""
    mod = MODULE_FUNCS.get(module)
    if mod is not None:
        return mod.get(func)
    return None


def lookup_stub_method(type_name: str, method: str) -> FuncType | None:
    """Look up a method on a stub type."""
    stub = STUB_METHODS.get(type_name)
    if stub is not None:
        return stub.get(method)
    return None


def lookup_stub_attr(type_name: str, attr: str) -> TypeNode | None:
    """Look up an attribute on a stub type."""
    stub = STUB_ATTRS.get(type_name)
    if stub is not None:
        return stub.get(attr)
    return None
