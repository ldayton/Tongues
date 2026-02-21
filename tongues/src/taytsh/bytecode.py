"""Bytecode definitions for the Taytsh VM."""

from __future__ import annotations

from dataclasses import dataclass

from .check import Type


# ============================================================
# OPCODES
# ============================================================

# Constants
OP_CONST: int = 0
OP_TRUE: int = 1
OP_FALSE: int = 2
OP_NIL: int = 3
OP_INT_ZERO: int = 4
OP_INT_ONE: int = 5
OP_EXTENDED_ARG: int = 6

# Variables
OP_LOAD_LOCAL: int = 10
OP_STORE_LOCAL: int = 11
OP_LOAD_GLOBAL: int = 12
OP_LOAD_BUILTIN: int = 13
OP_LOAD_CAPTURE: int = 14

# Stack
OP_POP: int = 20
OP_DUP: int = 21
OP_ROT_TWO: int = 22

# Arithmetic — type-specialized
OP_ADD_INT: int = 30
OP_ADD_FLOAT: int = 31
OP_ADD_STRING: int = 32
OP_ADD_BYTE: int = 33
OP_SUB_INT: int = 34
OP_SUB_FLOAT: int = 35
OP_SUB_BYTE: int = 36
OP_MUL_INT: int = 37
OP_MUL_FLOAT: int = 38
OP_MUL_BYTE: int = 39
OP_DIV_INT: int = 40
OP_DIV_FLOAT: int = 41
OP_MOD_INT: int = 42
OP_MOD_FLOAT: int = 43
OP_NEG_INT: int = 44
OP_NEG_FLOAT: int = 45
OP_BIT_AND: int = 46
OP_BIT_OR: int = 47
OP_BIT_XOR: int = 48
OP_BIT_NOT: int = 49
OP_SHIFT_LEFT: int = 50
OP_SHIFT_RIGHT: int = 51
OP_SHIFT_RIGHT_UNSIGNED: int = 52

# Comparison — arg encodes LT=0/LE=1/GT=2/GE=3
OP_EQ: int = 60
OP_NE: int = 61
OP_CMP_INT: int = 62
OP_CMP_FLOAT: int = 63
OP_CMP_STRING: int = 64
OP_CMP_BYTE: int = 65
OP_CMP_RUNE: int = 66

CMP_LT: int = 0
CMP_LE: int = 1
CMP_GT: int = 2
CMP_GE: int = 3

# Logic
OP_NOT: int = 70

# Control flow
OP_JUMP: int = 80
OP_JUMP_IF_FALSE: int = 81
OP_JUMP_IF_TRUE: int = 82
OP_JUMP_BACK: int = 83
OP_RETURN: int = 84
OP_RETURN_VOID: int = 85
OP_CALL: int = 86
OP_CALL_METHOD: int = 87

# Collections
OP_BUILD_LIST: int = 90
OP_BUILD_MAP: int = 91
OP_BUILD_SET: int = 92
OP_BUILD_TUPLE: int = 93
OP_INDEX: int = 94
OP_STORE_INDEX: int = 95
OP_SLICE: int = 96
OP_TUPLE_ACCESS: int = 97

# Structs/Enums
OP_BUILD_STRUCT: int = 100
OP_GET_FIELD: int = 101
OP_SET_FIELD: int = 102
OP_LOAD_ENUM: int = 103

# Exceptions
OP_PUSH_HANDLER: int = 110
OP_POP_HANDLER: int = 111
OP_THROW: int = 112
OP_PUSH_FINALLY: int = 113

# Iteration
OP_GET_ITER: int = 120
OP_FOR_ITER: int = 121
OP_UNPACK: int = 122

# Type tests
OP_IS_TYPE: int = 130
OP_MATCH_TYPE: int = 131

# Builtins
OP_CALL_BUILTIN: int = 140


# ============================================================
# VM VALUES
# ============================================================


@dataclass
class Val:
    """Base for all VM values."""


@dataclass
class VNil(Val):
    pass


@dataclass
class VBool(Val):
    value: bool


@dataclass
class VInt(Val):
    value: int


@dataclass
class VFloat(Val):
    value: float


@dataclass
class VByte(Val):
    value: int


@dataclass
class VBytes(Val):
    value: bytes


@dataclass
class VStr(Val):
    value: str


@dataclass
class VRune(Val):
    value: str


@dataclass
class VList(Val):
    items: list[Val]


@dataclass
class VMap(Val):
    keys: list[Val]
    values: list[Val]


@dataclass
class VSet(Val):
    items: list[Val]


@dataclass
class VTuple(Val):
    items: list[Val]


@dataclass
class VEnum(Val):
    enum_name: str
    variant: str


@dataclass
class VStruct(Val):
    type_name: str
    field_names: list[str]
    field_values: list[Val]


@dataclass
class VFunc(Val):
    code_index: int
    captures: list[Val]


# ============================================================
# CODE OBJECTS
# ============================================================


@dataclass
class CodeObject:
    name: str
    param_count: int
    local_count: int
    code: list[int]
    constants: list[Val]
    lines: list[int]
    local_names: list[str]


@dataclass
class StructDef:
    name: str
    field_names: list[str]
    field_types: list[Type]
    parent: str | None
    method_names: list[str]
    method_indices: list[int]


@dataclass
class EnumDef:
    name: str
    variants: list[str]


@dataclass
class InterfaceDef:
    name: str
    variant_names: list[str]


@dataclass
class CompiledModule:
    code_objects: list[CodeObject]
    global_names: list[str]
    struct_defs: list[StructDef]
    enum_defs: list[EnumDef]
    interface_defs: list[InterfaceDef]
    entry_index: int
