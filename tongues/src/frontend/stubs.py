"""Stub types for stdlib modules used by Tongues source."""

from __future__ import annotations

from .types import (
    TypeNode,
    FuncType,
    SliceType,
    MapType,
    OptionalType,
    PrimitiveType,
)

# Define primitive types locally to avoid cross-module constant references
# which can fail during self-transpile checker
_INT = PrimitiveType("int")
_STR = PrimitiveType("string")
_BOOL = PrimitiveType("bool")
_BYTES = PrimitiveType("bytes")
_VOID = PrimitiveType("void")

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
        "argv": SliceType(_STR),
        "stdin": TEXTIO_TYPE,
        "stdout": TEXTIO_TYPE,
        "stderr": TEXTIO_TYPE,
    },
    "os": {
        "environ": MapType(_STR, _STR),
    },
}

# Module-level functions: MODULE_FUNCS[module][func] -> FuncType
MODULE_FUNCS: dict[str, dict[str, FuncType]] = {
    "sys": {
        "exit": FuncType([_INT], _VOID),
    },
    "os": {
        "getenv": FuncType([_STR], OptionalType(_STR)),
    },
}

# Stub type methods: STUB_METHODS[type_name][method] -> FuncType
STUB_METHODS: dict[str, dict[str, FuncType]] = {
    "TextIO": {
        "read": FuncType([], _STR),
        "readline": FuncType([], _STR),
        "write": FuncType([_STR], _INT),
        "isatty": FuncType([], _BOOL),
    },
    "BytesIO": {
        "read": FuncType([], _BYTES),
        "write": FuncType([_BYTES], _INT),
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
