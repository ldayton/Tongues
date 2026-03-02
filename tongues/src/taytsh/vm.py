"""Stack-based bytecode VM for Taytsh."""

from __future__ import annotations

from dataclasses import dataclass

from .ast import TModule
from .bytecode import (
    CMP_GE,
    CMP_GT,
    CMP_LE,
    CMP_LT,
    CodeObject,
    CompiledModule,
    InterfaceDef,
    StructDef,
    OP_ADD_BYTE,
    OP_ADD_FLOAT,
    OP_ADD_INT,
    OP_ADD_STRING,
    OP_BIT_AND,
    OP_BIT_NOT,
    OP_BIT_OR,
    OP_BIT_XOR,
    OP_BUILD_LIST,
    OP_BUILD_MAP,
    OP_BUILD_SET,
    OP_BUILD_STRUCT,
    OP_BUILD_TUPLE,
    OP_CALL,
    OP_CALL_BUILTIN,
    OP_CALL_METHOD,
    OP_CMP_BYTE,
    OP_CMP_FLOAT,
    OP_CMP_INT,
    OP_CMP_RUNE,
    OP_CMP_STRING,
    OP_CONST,
    OP_DIV_FLOAT,
    OP_DIV_INT,
    OP_DUP,
    OP_EQ,
    OP_EXTENDED_ARG,
    OP_FALSE,
    OP_FOR_ITER,
    OP_GET_FIELD,
    OP_GET_ITER,
    OP_INDEX,
    OP_IS_TYPE,
    OP_JUMP,
    OP_JUMP_BACK,
    OP_JUMP_IF_FALSE,
    OP_JUMP_IF_TRUE,
    OP_LOAD_BUILTIN,
    OP_LOAD_CAPTURE,
    OP_LOAD_ENUM,
    OP_LOAD_GLOBAL,
    OP_LOAD_LOCAL,
    OP_STORE_GLOBAL,
    OP_MATCH_TYPE,
    OP_MOD_FLOAT,
    OP_MOD_INT,
    OP_MUL_BYTE,
    OP_MUL_FLOAT,
    OP_MUL_INT,
    OP_NE,
    OP_NEG_FLOAT,
    OP_NEG_INT,
    OP_NIL,
    OP_NOT,
    OP_POP,
    OP_POP_HANDLER,
    OP_PUSH_FINALLY,
    OP_PUSH_HANDLER,
    OP_RETURN,
    OP_RETURN_VOID,
    OP_ROT_TWO,
    OP_SET_FIELD,
    OP_SHIFT_LEFT,
    OP_SHIFT_RIGHT,
    OP_SHIFT_RIGHT_UNSIGNED,
    OP_SLICE,
    OP_STORE_INDEX,
    OP_STORE_LOCAL,
    OP_SUB_BYTE,
    OP_SUB_FLOAT,
    OP_SUB_INT,
    OP_THROW,
    OP_TRUE,
    OP_TUPLE_ACCESS,
    OP_UNPACK,
    OP_INT_ZERO,
    OP_INT_ONE,
    Val,
    VBool,
    VByte,
    VBytes,
    VEnum,
    VFloat,
    VFunc,
    VInt,
    VList,
    VMap,
    VNil,
    VRune,
    VSet,
    VStr,
    VStruct,
    VTuple,
)
from .compiler import BUILTIN_TABLE, compile_module


# ============================================================
# VM DATA STRUCTURES
# ============================================================


@dataclass
class Frame:
    code: CodeObject
    ip: int
    bp: int


HANDLER_CATCH: int = 0
HANDLER_FINALLY: int = 1


@dataclass
class Handler:
    frame_depth: int
    catch_ip: int
    stack_depth: int
    kind: int


@dataclass
class VMResult:
    exit_code: int
    stdout: str
    stderr: str


# ============================================================
# SENTINEL VALUES
# ============================================================

_NONE_VAL: VNil = VNil()
_TRUE_VAL: VBool = VBool(True)
_FALSE_VAL: VBool = VBool(False)
_ZERO_INT: VInt = VInt(0)
_ONE_INT: VInt = VInt(1)


# ============================================================
# VM EXCEPTIONS
# ============================================================


class _VMExit(Exception):
    def __init__(self, code: int) -> None:
        self.code: int = code


class _VMThrow(Exception):
    def __init__(self, value: Val) -> None:
        self.value: Val = value


# ============================================================
# HELPERS
# ============================================================


def _isnan(x: float) -> bool:
    return x != x


def _isinf(x: float) -> bool:
    return x == float("inf") or x == float("-inf")


def _fmod(a: float, b: float) -> float:
    """IEEE 754 remainder (truncation semantics, sign follows dividend)."""
    result = a % b
    # Python % uses floor division; correct sign to follow dividend
    if result != 0.0 and (result < 0) != (a < 0):
        result -= b
    return result


def _val_to_string(v: Val) -> str:
    if isinstance(v, VStr):
        return v.value
    if isinstance(v, VInt):
        return str(v.value)
    if isinstance(v, VFloat):
        f = v.value
        if _isnan(f):
            return "NaN"
        if f == float("inf"):
            return "Inf"
        if f == float("-inf"):
            return "-Inf"
        if f == 0.0 and str(f) == "-0.0":
            return "-0.0"
        if f == int(f) and not _isinf(f):
            return str(int(f)) + ".0"
        return str(f)
    if isinstance(v, VBool):
        return "true" if v.value else "false"
    if isinstance(v, VNil):
        return "nil"
    if isinstance(v, VByte):
        return str(v.value)
    if isinstance(v, VRune):
        return v.value
    if isinstance(v, VBytes):
        hex_parts: list[str] = []
        for b in v.value:
            h = hex(b)[2:]
            if len(h) < 2:
                h = "0" + h
            hex_parts.append("\\x" + h)
        hex_chars = "".join(hex_parts)
        return 'b"' + hex_chars + '"'
    if isinstance(v, VList):
        parts: list[str] = []
        for item in v.items:
            parts.append(_val_to_string_quoted(item))
        return "[" + ", ".join(parts) + "]"
    if isinstance(v, VMap):
        map_keys: list[Val] = list(v.keys)
        map_vals: list[Val] = list(v.values)
        _sort_key_pairs(map_keys, map_vals)
        parts2: list[str] = []
        mi = 0
        while mi < len(map_keys):
            ks = _val_to_string_quoted(map_keys[mi])
            vs2 = _val_to_string_quoted(map_vals[mi])
            parts2.append(ks + ": " + vs2)
            mi += 1
        return "{" + ", ".join(parts2) + "}"
    if isinstance(v, VSet):
        sorted_items = list(v.items)
        _sort_vals(sorted_items)
        parts3: list[str] = []
        for item in sorted_items:
            parts3.append(_val_to_string_quoted(item))
        return "{" + ", ".join(parts3) + "}"
    if isinstance(v, VTuple):
        parts4: list[str] = []
        for item in v.items:
            parts4.append(_val_to_string_quoted(item))
        return "(" + ", ".join(parts4) + ")"
    if isinstance(v, VEnum):
        return v.enum_name + "." + v.variant
    if isinstance(v, VStruct):
        parts5: list[str] = []
        i2 = 0
        while i2 < len(v.field_names):
            parts5.append(v.field_names[i2] + ": " + _val_to_string(v.field_values[i2]))
            i2 += 1
        return v.type_name + "{" + ", ".join(parts5) + "}"
    if isinstance(v, VFunc):
        return "fn" + v.type_sig if v.type_sig else "<fn>"
    return "<unknown>"


def _val_to_string_quoted(v: Val) -> str:
    """Like _val_to_string but quotes strings and runes."""
    if isinstance(v, VStr):
        return '"' + v.value + '"'
    if isinstance(v, VRune):
        return "'" + v.value + "'"
    return _val_to_string(v)


def _sort_key(v: Val) -> tuple[int, float, str]:
    if isinstance(v, VInt):
        return (0, float(v.value), "")
    if isinstance(v, VFloat):
        return (1, v.value, "")
    if isinstance(v, VStr):
        return (2, 0.0, v.value)
    if isinstance(v, VByte):
        return (3, float(v.value), "")
    if isinstance(v, VBool):
        return (4, 1.0 if v.value else 0.0, "")
    return (9, 0.0, "")


def _sort_key_pairs(keys: list[Val], vals: list[Val]) -> None:
    """Insertion sort for parallel key/value lists by key sort order."""
    i = 1
    while i < len(keys):
        kk = keys[i]
        vv = vals[i]
        sk = _sort_key(kk)
        j = i - 1
        while j >= 0 and _sort_key(keys[j]) > sk:
            keys[j + 1] = keys[j]
            vals[j + 1] = vals[j]
            j -= 1
        keys[j + 1] = kk
        vals[j + 1] = vv
        i += 1


def _read_file_bytes(path: str) -> VBytes:
    with open(path, "rb") as f:
        data = f.read()
    return VBytes(data)


def _write_file_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def _extract_str_or_rune(v: Val) -> str | None:
    if isinstance(v, VStr):
        return v.value
    if isinstance(v, VRune):
        return v.value
    return None


def _val_eq(a: Val, b: Val) -> bool:
    if isinstance(a, VNil) and isinstance(b, VNil):
        return True
    if isinstance(a, VBool) and isinstance(b, VBool):
        return a.value == b.value
    if isinstance(a, VInt) and isinstance(b, VInt):
        return a.value == b.value
    if isinstance(a, VFloat) and isinstance(b, VFloat):
        return a.value == b.value
    if isinstance(a, VByte) and isinstance(b, VByte):
        return a.value == b.value
    if isinstance(a, VByte) and isinstance(b, VInt):
        return a.value == b.value
    if isinstance(a, VInt) and isinstance(b, VByte):
        return a.value == b.value
    if isinstance(a, VStr) and isinstance(b, VStr):
        return a.value == b.value
    if isinstance(a, VRune) and isinstance(b, VRune):
        return a.value == b.value
    if isinstance(a, VBytes) and isinstance(b, VBytes):
        return a.value == b.value
    if isinstance(a, VEnum) and isinstance(b, VEnum):
        return a.enum_name == b.enum_name and a.variant == b.variant
    if isinstance(a, VList) and isinstance(b, VList):
        if len(a.items) != len(b.items):
            return False
        i = 0
        while i < len(a.items):
            if not _val_eq(a.items[i], b.items[i]):
                return False
            i += 1
        return True
    if isinstance(a, VTuple) and isinstance(b, VTuple):
        if len(a.items) != len(b.items):
            return False
        i = 0
        while i < len(a.items):
            if not _val_eq(a.items[i], b.items[i]):
                return False
            i += 1
        return True
    if isinstance(a, VMap) and isinstance(b, VMap):
        if len(a.keys) != len(b.keys):
            return False
        i = 0
        while i < len(a.keys):
            found = False
            j = 0
            while j < len(b.keys):
                if _val_eq(a.keys[i], b.keys[j]):
                    if _val_eq(a.values[i], b.values[j]):
                        found = True
                    break
                j += 1
            if not found:
                return False
            i += 1
        return True
    if isinstance(a, VSet) and isinstance(b, VSet):
        if len(a.items) != len(b.items):
            return False
        i = 0
        while i < len(a.items):
            found = False
            j = 0
            while j < len(b.items):
                if _val_eq(a.items[i], b.items[j]):
                    found = True
                    break
                j += 1
            if not found:
                return False
            i += 1
        return True
    if isinstance(a, VStruct) and isinstance(b, VStruct):
        if a.type_name != b.type_name:
            return False
        i = 0
        while i < len(a.field_values):
            if not _val_eq(a.field_values[i], b.field_values[i]):
                return False
            i += 1
        return True
    if isinstance(a, VNil) or isinstance(b, VNil):
        return False
    return False


def _is_truthy(v: Val) -> bool:
    if isinstance(v, VBool):
        return v.value
    if isinstance(v, VNil):
        return False
    return True


def _make_error_struct(type_name: str, message: str) -> VStruct:
    return VStruct(
        type_name=type_name,
        field_names=["message"],
        field_values=[VStr(message)],
    )


def _floor(x: float) -> int:
    i = int(x)
    if x < 0 and x != i:
        return i - 1
    return i


def _ceil(x: float) -> int:
    i = int(x)
    if x > 0 and x != i:
        return i + 1
    return i


def _sqrt(x: float) -> float:
    return x**0.5


_INT64_MIN: int = -(2**63)
_INT64_MAX: int = 2**63 - 1


def _wrapping_add(a: int, b: int) -> int:
    r = a + b
    r = ((r - _INT64_MIN) % (2**64)) + _INT64_MIN
    return r


def _wrapping_sub(a: int, b: int) -> int:
    r = a - b
    r = ((r - _INT64_MIN) % (2**64)) + _INT64_MIN
    return r


def _wrapping_mul(a: int, b: int) -> int:
    r = a * b
    r = ((r - _INT64_MIN) % (2**64)) + _INT64_MIN
    return r


# ============================================================
# BUILTINS
# ============================================================


class _BuiltinDispatch:
    """Registry of builtin implementations keyed by index."""

    def __init__(self, vm: VM) -> None:
        self.vm: VM = vm
        self._table: dict[str, int] = {}
        i = 0
        while i < len(BUILTIN_TABLE):
            self._table[BUILTIN_TABLE[i]] = i
            i += 1

    def call(self, idx: int, args: list[Val]) -> Val:
        name = BUILTIN_TABLE[idx]
        if name == "Assert":
            return self._assert(args)
        if name == "WritelnOut":
            return self._writeln_out(args)
        if name == "WritelnErr":
            return self._writeln_err(args)
        if name == "WriteOut":
            return self._write_out(args)
        if name == "WriteErr":
            return self._write_err(args)
        if name == "ToString":
            return self._to_string(args)
        if name == "Len":
            return self._len(args)
        if name == "Concat":
            return self._concat(args)
        if name == "Append":
            return self._append(args)
        if name == "Insert":
            return self._insert(args)
        if name == "Pop":
            return self._pop(args)
        if name == "RemoveAt":
            return self._remove_at(args)
        if name == "IndexOf":
            return self._index_of(args)
        if name == "Reversed":
            return self._reversed(args)
        if name == "Sorted":
            return self._sorted(args)
        if name == "Map":
            return self._map_new(args)
        if name == "Get":
            return self._map_get(args)
        if name == "Delete":
            return self._map_delete(args)
        if name == "Keys":
            return self._map_keys(args)
        if name == "Values":
            return self._map_values(args)
        if name == "Items":
            return self._map_items(args)
        if name == "Merge":
            return self._map_merge(args)
        if name == "PopItem":
            return self._map_pop_item(args)
        if name == "MapFromKeys":
            return self._map_from_keys(args)
        if name == "MapFromPairs":
            return self._map_from_pairs(args)
        if name == "Set":
            return self._set_new(args)
        if name == "Add":
            return self._set_add(args)
        if name == "Remove":
            return self._set_remove(args)
        if name == "Union":
            return self._set_union(args)
        if name == "Intersection":
            return self._set_intersection(args)
        if name == "Difference":
            return self._set_difference(args)
        if name == "SetFromList":
            return self._set_from_list(args)
        if name == "ListFrom":
            return self._list_from(args)
        if name == "Abs":
            return self._abs(args)
        if name == "Min":
            return self._min(args)
        if name == "Max":
            return self._max(args)
        if name == "Sum":
            return self._sum(args)
        if name == "Pow":
            return self._pow(args)
        if name == "Round":
            return self._round(args)
        if name == "Floor":
            return self._floor(args)
        if name == "Ceil":
            return self._ceil(args)
        if name == "DivMod":
            return self._divmod(args)
        if name == "Sqrt":
            return self._sqrt(args)
        if name == "IsNaN":
            return self._isnan(args)
        if name == "IsInf":
            return self._isinf(args)
        if name == "IntToFloat":
            return self._int_to_float(args)
        if name == "FloatToInt":
            return self._float_to_int(args)
        if name == "ByteToInt":
            return self._byte_to_int(args)
        if name == "IntToByte":
            return self._int_to_byte(args)
        if name == "RuneFromInt":
            return self._rune_from_int(args)
        if name == "RuneToInt":
            return self._rune_to_int(args)
        if name == "ParseInt":
            return self._parse_int(args)
        if name == "ParseFloat":
            return self._parse_float(args)
        if name == "FormatInt":
            return self._format_int(args)
        if name == "Upper":
            return self._upper(args)
        if name == "Lower":
            return self._lower(args)
        if name == "Trim":
            return self._trim(args)
        if name == "TrimStart":
            return self._trim_start(args)
        if name == "TrimEnd":
            return self._trim_end(args)
        if name == "Split":
            return self._split(args)
        if name == "SplitN":
            return self._split_n(args)
        if name == "SplitWhitespace":
            return self._split_whitespace(args)
        if name == "Join":
            return self._join(args)
        if name == "Find":
            return self._find(args)
        if name == "RFind":
            return self._rfind(args)
        if name == "Count":
            return self._count(args)
        if name == "Contains":
            return self._contains(args)
        if name == "Replace":
            return self._replace(args)
        if name == "ReplaceCount":
            return self._replace_count(args)
        if name == "Repeat":
            return self._repeat(args)
        if name == "Reverse":
            return self._reverse(args)
        if name == "StartsWith":
            return self._starts_with(args)
        if name == "EndsWith":
            return self._ends_with(args)
        if name == "IsDigit":
            return self._is_digit(args)
        if name == "IsAlpha":
            return self._is_alpha(args)
        if name == "IsAlnum":
            return self._is_alnum(args)
        if name == "IsSpace":
            return self._is_space(args)
        if name == "IsUpper":
            return self._is_upper(args)
        if name == "IsLower":
            return self._is_lower(args)
        if name == "Format":
            return self._format(args)
        if name == "Exit":
            return self._exit(args)
        if name == "Unwrap":
            return self._unwrap(args)
        if name == "IsNil":
            return self._is_nil(args)
        if name == "IsType":
            return self._is_type(args)
        if name == "FloorDiv":
            return self._floor_div(args)
        if name == "PythonMod":
            return self._python_mod(args)
        if name == "WrappingAdd":
            return self._wrapping_add(args)
        if name == "WrappingSub":
            return self._wrapping_sub(args)
        if name == "WrappingMul":
            return self._wrapping_mul(args)
        if name == "Bytes":
            return self._bytes_new(args)
        if name == "BytesFrom":
            return self._bytes_from(args)
        if name == "RangeList":
            return self._range_list(args)
        if name == "ListCompare":
            return self._list_compare(args)
        if name == "Zip":
            return self._zip(args)
        if name == "Chars":
            return self._chars(args)
        if name == "ReplaceSlice":
            return self._replace_slice(args)
        if name == "Encode":
            return self._encode(args)
        if name == "Decode":
            return self._decode(args)
        if name == "ReadLine":
            return self._read_line(args)
        if name == "ReadAll":
            return self._read_all(args)
        if name == "ReadBytes":
            return self._read_bytes(args)
        if name == "ReadBytesN":
            return self._read_bytes_n(args)
        if name == "ReadFile":
            return self._read_file(args)
        if name == "ReadFileBytes":
            return self._read_file_bytes(args)
        if name == "WriteFile":
            return self._write_file(args)
        if name == "Args":
            return self._args(args)
        if name == "GetEnv":
            return self._get_env(args)
        raise _VMThrow(_make_error_struct("RuntimeError", "unknown builtin: " + name))

    # ── I/O ───────────────────────────────────────────────────

    def _assert(self, args: list[Val]) -> Val:
        if len(args) == 0:
            raise _VMThrow(_make_error_struct("AssertError", "assertion failed"))
        cond = args[0]
        if isinstance(cond, VBool) and cond.value:
            return _NONE_VAL
        msg = "assertion failed"
        if len(args) > 1 and isinstance(args[1], VStr):
            msg = args[1].value
        raise _VMThrow(_make_error_struct("AssertError", msg))

    def _writeln_out(self, args: list[Val]) -> Val:
        s = _val_to_string(args[0]) if len(args) > 0 else ""
        self.vm.stdout_buf.append(s)
        self.vm.stdout_buf.append("\n")
        return _NONE_VAL

    def _writeln_err(self, args: list[Val]) -> Val:
        s = _val_to_string(args[0]) if len(args) > 0 else ""
        self.vm.stderr_buf.append(s)
        self.vm.stderr_buf.append("\n")
        return _NONE_VAL

    def _write_out(self, args: list[Val]) -> Val:
        s = _val_to_string(args[0]) if len(args) > 0 else ""
        self.vm.stdout_buf.append(s)
        return _NONE_VAL

    def _write_err(self, args: list[Val]) -> Val:
        s = _val_to_string(args[0]) if len(args) > 0 else ""
        self.vm.stderr_buf.append(s)
        return _NONE_VAL

    def _to_string(self, args: list[Val]) -> Val:
        if len(args) == 0:
            return VStr("")
        v = args[0]
        if isinstance(v, VStruct):
            method_idx = self.vm._find_method(v.type_name, "ToString")
            if method_idx is not None:
                return self.vm._call_method_sync(v, method_idx, [])
        return VStr(_val_to_string(v))

    def _exit(self, args: list[Val]) -> Val:
        code = 0
        if len(args) > 0 and isinstance(args[0], VInt):
            code = args[0].value
        raise _VMExit(code)

    # ── String operations ─────────────────────────────────────

    def _len(self, args: list[Val]) -> Val:
        v = args[0]
        if isinstance(v, VStr):
            return VInt(len(v.value))
        if isinstance(v, VList):
            return VInt(len(v.items))
        if isinstance(v, VMap):
            return VInt(len(v.keys))
        if isinstance(v, VSet):
            return VInt(len(v.items))
        if isinstance(v, VBytes):
            return VInt(len(v.value))
        return VInt(0)

    def _concat(self, args: list[Val]) -> Val:
        if len(args) < 2:
            return VStr("")
        a = args[0]
        b = args[1]
        if isinstance(a, VStr) and isinstance(b, VStr):
            return VStr(a.value + b.value)
        if isinstance(a, VList) and isinstance(b, VList):
            return VList(list(a.items) + list(b.items))
        if isinstance(a, VBytes) and isinstance(b, VBytes):
            return VBytes(a.value + b.value)
        return VStr(_val_to_string(a) + _val_to_string(b))

    def _upper(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            return VStr(args[0].value.upper())
        return args[0]

    def _lower(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            return VStr(args[0].value.lower())
        return args[0]

    def _trim(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            if len(args) > 1 and isinstance(args[1], VStr):
                return VStr(args[0].value.strip(args[1].value))
            return VStr(args[0].value.strip())
        return args[0]

    def _trim_start(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            if len(args) > 1 and isinstance(args[1], VStr):
                return VStr(args[0].value.lstrip(args[1].value))
            return VStr(args[0].value.lstrip())
        return args[0]

    def _trim_end(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            if len(args) > 1 and isinstance(args[1], VStr):
                return VStr(args[0].value.rstrip(args[1].value))
            return VStr(args[0].value.rstrip())
        return args[0]

    def _split(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr) and isinstance(args[1], VStr):
            if args[1].value == "":
                raise _VMThrow(_make_error_struct("ValueError", "empty separator"))
            parts = args[0].value.split(args[1].value)
            items: list[Val] = []
            pi = 0
            while pi < len(parts):
                items.append(VStr(parts[pi]))
                pi += 1
            return VList(items)
        return VList([])

    def _split_n(self, args: list[Val]) -> Val:
        if (
            isinstance(args[0], VStr)
            and isinstance(args[1], VStr)
            and isinstance(args[2], VInt)
        ):
            n = args[2].value
            if n <= 0:
                raise _VMThrow(
                    _make_error_struct("ValueError", "SplitN max must be > 0")
                )
            parts = args[0].value.split(args[1].value, n - 1)
            items2: list[Val] = []
            pi2 = 0
            while pi2 < len(parts):
                items2.append(VStr(parts[pi2]))
                pi2 += 1
            return VList(items2)
        return VList([])

    def _split_whitespace(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            parts = args[0].value.split()
            items3: list[Val] = []
            pi3 = 0
            while pi3 < len(parts):
                items3.append(VStr(parts[pi3]))
                pi3 += 1
            return VList(items3)
        return VList([])

    def _join(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr) and isinstance(args[1], VList):
            parts: list[str] = []
            for item in args[1].items:
                parts.append(_val_to_string(item))
            return VStr(args[0].value.join(parts))
        return VStr("")

    def _find(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr) and isinstance(args[1], VStr):
            return VInt(args[0].value.find(args[1].value))
        return VInt(-1)

    def _rfind(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr) and isinstance(args[1], VStr):
            return VInt(args[0].value.rfind(args[1].value))
        return VInt(-1)

    def _count(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr) and isinstance(args[1], VStr):
            return VInt(args[0].value.count(args[1].value))
        return VInt(0)

    def _contains(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr) and isinstance(args[1], VStr):
            return VBool(args[1].value in args[0].value)
        if isinstance(args[0], VList):
            for item in args[0].items:
                if _val_eq(item, args[1]):
                    return _TRUE_VAL
            return _FALSE_VAL
        if isinstance(args[0], VSet):
            for item in args[0].items:
                if _val_eq(item, args[1]):
                    return _TRUE_VAL
            return _FALSE_VAL
        if isinstance(args[0], VMap):
            for k in args[0].keys:
                if _val_eq(k, args[1]):
                    return _TRUE_VAL
            return _FALSE_VAL
        return _FALSE_VAL

    def _replace(self, args: list[Val]) -> Val:
        if (
            isinstance(args[0], VStr)
            and isinstance(args[1], VStr)
            and isinstance(args[2], VStr)
        ):
            return VStr(args[0].value.replace(args[1].value, args[2].value))
        return args[0]

    def _replace_count(self, args: list[Val]) -> Val:
        if (
            len(args) >= 4
            and isinstance(args[0], VStr)
            and isinstance(args[1], VStr)
            and isinstance(args[2], VStr)
            and isinstance(args[3], VInt)
        ):
            return VStr(
                args[0].value.replace(args[1].value, args[2].value, args[3].value)
            )
        return args[0]

    def _repeat(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr) and isinstance(args[1], VInt):
            rparts: list[str] = []
            ri = 0
            while ri < args[1].value:
                rparts.append(args[0].value)
                ri += 1
            return VStr("".join(rparts))
        if isinstance(args[0], VList) and isinstance(args[1], VInt):
            n = args[1].value
            if n <= 0:
                return VList([])
            result: list[Val] = []
            i = 0
            while i < n:
                result.extend(args[0].items)
                i += 1
            return VList(result)
        return VStr("")

    def _reverse(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            return VStr(args[0].value[::-1])
        return args[0]

    def _starts_with(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr) and isinstance(args[1], VStr):
            return VBool(args[0].value.startswith(args[1].value))
        return _FALSE_VAL

    def _ends_with(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr) and isinstance(args[1], VStr):
            return VBool(args[0].value.endswith(args[1].value))
        return _FALSE_VAL

    def _is_digit(self, args: list[Val]) -> Val:
        s = _extract_str_or_rune(args[0])
        return VBool(len(s) > 0 and s.isdigit()) if s is not None else _FALSE_VAL

    def _is_alpha(self, args: list[Val]) -> Val:
        s = _extract_str_or_rune(args[0])
        return VBool(len(s) > 0 and s.isalpha()) if s is not None else _FALSE_VAL

    def _is_alnum(self, args: list[Val]) -> Val:
        s = _extract_str_or_rune(args[0])
        return VBool(len(s) > 0 and s.isalnum()) if s is not None else _FALSE_VAL

    def _is_space(self, args: list[Val]) -> Val:
        s = _extract_str_or_rune(args[0])
        return VBool(len(s) > 0 and s.isspace()) if s is not None else _FALSE_VAL

    def _is_upper(self, args: list[Val]) -> Val:
        s = _extract_str_or_rune(args[0])
        return VBool(len(s) > 0 and s.isupper()) if s is not None else _FALSE_VAL

    def _is_lower(self, args: list[Val]) -> Val:
        s = _extract_str_or_rune(args[0])
        return VBool(len(s) > 0 and s.islower()) if s is not None else _FALSE_VAL

    def _format(self, args: list[Val]) -> Val:
        if len(args) < 1:
            return VStr("")
        a0 = args[0]
        if not isinstance(a0, VStr):
            return VStr("")
        template = a0.value
        result: list[str] = []
        ai = 1
        i = 0
        while i < len(template):
            if template[i] == "{" and i + 1 < len(template) and template[i + 1] == "}":
                if ai < len(args):
                    result.append(_val_to_string(args[ai]))
                    ai += 1
                i += 2
            else:
                result.append(template[i])
                i += 1
        return VStr("".join(result))

    # ── Numeric ───────────────────────────────────────────────

    def _abs(self, args: list[Val]) -> Val:
        v = args[0]
        if isinstance(v, VInt):
            return VInt(v.value if v.value >= 0 else -v.value)
        if isinstance(v, VFloat):
            return VFloat(v.value if v.value >= 0 else -v.value)
        return v

    def _min(self, args: list[Val]) -> Val:
        a = args[0]
        if len(args) == 1:
            items: list[Val] | None = None
            if isinstance(a, VList):
                items = a.items
            elif isinstance(a, VTuple):
                items = a.items
            if items is not None:
                if len(items) == 0:
                    raise _VMThrow(
                        _make_error_struct(
                            "ValueError", "min() arg is an empty sequence"
                        )
                    )
                best = items[0]
                i = 1
                while i < len(items):
                    if _val_compare(items[i], best) < 0:
                        best = items[i]
                    i += 1
                return best
        b = args[1]
        if isinstance(a, VInt) and isinstance(b, VInt):
            return a if a.value <= b.value else b
        if isinstance(a, VFloat) and isinstance(b, VFloat):
            if _isnan(a.value) or _isnan(b.value):
                return VFloat(float("nan"))
            return a if a.value <= b.value else b
        return a

    def _max(self, args: list[Val]) -> Val:
        a = args[0]
        if len(args) == 1:
            items: list[Val] | None = None
            if isinstance(a, VList):
                items = a.items
            elif isinstance(a, VTuple):
                items = a.items
            if items is not None:
                if len(items) == 0:
                    raise _VMThrow(
                        _make_error_struct(
                            "ValueError", "max() arg is an empty sequence"
                        )
                    )
                best = items[0]
                i = 1
                while i < len(items):
                    if _val_compare(items[i], best) > 0:
                        best = items[i]
                    i += 1
                return best
        b = args[1]
        if isinstance(a, VInt) and isinstance(b, VInt):
            return a if a.value >= b.value else b
        if isinstance(a, VFloat) and isinstance(b, VFloat):
            if _isnan(a.value) or _isnan(b.value):
                return VFloat(float("nan"))
            return a if a.value >= b.value else b
        return a

    def _sum(self, args: list[Val]) -> Val:
        lst = args[0]
        if not isinstance(lst, VList):
            return _ZERO_INT
        if len(lst.items) == 0:
            return _ZERO_INT
        first = lst.items[0] if len(lst.items) > 0 else None
        if first is not None and isinstance(first, VFloat):
            total = 0.0
            for item in lst.items:
                if isinstance(item, VFloat):
                    total += item.value
            return VFloat(total)
        total_i = 0
        for item in lst.items:
            if isinstance(item, VInt):
                total_i += item.value
        return VInt(total_i)

    def _pow(self, args: list[Val]) -> Val:
        a = args[0]
        b = args[1]
        if isinstance(a, VInt) and isinstance(b, VInt):
            if b.value < 0:
                raise _VMThrow(
                    _make_error_struct("ValueError", "negative exponent in integer Pow")
                )
            r = a.value**b.value
            if r > _INT64_MAX or r < _INT64_MIN:
                raise _VMThrow(_make_error_struct("OverflowError", "integer overflow"))
            return VInt(r)
        if isinstance(a, VFloat) and isinstance(b, VFloat):
            return VFloat(a.value**b.value)
        return _ZERO_INT

    def _round(self, args: list[Val]) -> Val:
        v = args[0]
        if isinstance(v, VFloat):
            f = v.value
            if _isnan(f):
                raise _VMThrow(_make_error_struct("ValueError", "cannot round NaN"))
            if _isinf(f):
                raise _VMThrow(
                    _make_error_struct("ValueError", "cannot round Infinity")
                )
            # Round half away from zero
            if f >= 0:
                return VInt(int(f + 0.5))
            else:
                return VInt(-int(-f + 0.5))
        return _ZERO_INT

    def _floor(self, args: list[Val]) -> Val:
        v = args[0]
        if isinstance(v, VFloat):
            f = v.value
            if _isnan(f):
                raise _VMThrow(_make_error_struct("ValueError", "cannot floor NaN"))
            if _isinf(f):
                raise _VMThrow(
                    _make_error_struct("ValueError", "cannot floor Infinity")
                )
            return VInt(_floor(f))
        return _ZERO_INT

    def _ceil(self, args: list[Val]) -> Val:
        v = args[0]
        if isinstance(v, VFloat):
            f = v.value
            if _isnan(f):
                raise _VMThrow(_make_error_struct("ValueError", "cannot ceil NaN"))
            if _isinf(f):
                raise _VMThrow(_make_error_struct("ValueError", "cannot ceil Infinity"))
            return VInt(_ceil(f))
        return _ZERO_INT

    def _divmod(self, args: list[Val]) -> Val:
        a = args[0]
        b = args[1]
        if isinstance(a, VInt) and isinstance(b, VInt):
            if b.value == 0:
                raise _VMThrow(
                    _make_error_struct("ZeroDivisionError", "division by zero")
                )
            q2 = (
                a.value // b.value
                if (a.value >= 0 and b.value > 0) or (a.value <= 0 and b.value < 0)
                else -(abs(a.value) // abs(b.value))
            )
            # Truncation toward zero
            if a.value >= 0:
                q2 = a.value // b.value
            else:
                q2 = -((-a.value) // b.value)
            r = a.value - q2 * b.value
            return VTuple([VInt(q2), VInt(r)])
        return VTuple([_ZERO_INT, _ZERO_INT])

    def _sqrt(self, args: list[Val]) -> Val:
        v = args[0]
        if isinstance(v, VFloat):
            return VFloat(_sqrt(v.value))
        return VFloat(0.0)

    def _isnan(self, args: list[Val]) -> Val:
        v = args[0]
        if isinstance(v, VFloat):
            return VBool(_isnan(v.value))
        return _FALSE_VAL

    def _isinf(self, args: list[Val]) -> Val:
        v = args[0]
        if isinstance(v, VFloat):
            return VBool(_isinf(v.value))
        return _FALSE_VAL

    # ── Conversions ───────────────────────────────────────────

    def _int_to_float(self, args: list[Val]) -> Val:
        if isinstance(args[0], VInt):
            val: int = args[0].value
            return VFloat(float(val))
        return VFloat(0.0)

    def _float_to_int(self, args: list[Val]) -> Val:
        if isinstance(args[0], VFloat):
            f = args[0].value
            if _isnan(f):
                raise _VMThrow(
                    _make_error_struct("ValueError", "cannot convert NaN to int")
                )
            if _isinf(f):
                raise _VMThrow(
                    _make_error_struct("ValueError", "cannot convert Infinity to int")
                )
            return VInt(int(f))
        return _ZERO_INT

    def _byte_to_int(self, args: list[Val]) -> Val:
        if isinstance(args[0], VByte):
            return VInt(args[0].value)
        return _ZERO_INT

    def _int_to_byte(self, args: list[Val]) -> Val:
        if isinstance(args[0], VInt):
            return VByte(args[0].value & 0xFF)
        return VByte(0)

    def _rune_from_int(self, args: list[Val]) -> Val:
        if isinstance(args[0], VInt):
            return VRune(chr(args[0].value))
        return VRune("\x00")

    def _rune_to_int(self, args: list[Val]) -> Val:
        if isinstance(args[0], VRune):
            return VInt(ord(args[0].value[0]))
        return _ZERO_INT

    def _parse_int(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            s = args[0].value.strip()
            base = 10
            if len(args) > 1 and isinstance(args[1], VInt):
                base = args[1].value
            try:
                return VInt(int(s, base))
            except (ValueError, OverflowError):
                raise _VMThrow(
                    _make_error_struct("ValueError", "invalid integer: " + s)
                )
        return _ZERO_INT

    def _parse_float(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            s = args[0].value.strip()
            try:
                return VFloat(float(s))
            except ValueError:
                raise _VMThrow(_make_error_struct("ValueError", "invalid float: " + s))
        return VFloat(0.0)

    def _format_int(self, args: list[Val]) -> Val:
        if isinstance(args[0], VInt) and isinstance(args[1], VInt):
            n = args[0].value
            base = args[1].value
            if base == 16:
                if n < 0:
                    return VStr("-" + hex(-n)[2:])
                return VStr(hex(n)[2:])
            if base == 8:
                if n < 0:
                    return VStr("-" + oct(-n)[2:])
                return VStr(oct(n)[2:])
            if base == 2:
                if n < 0:
                    return VStr("-" + bin(-n)[2:])
                return VStr(bin(n)[2:])
            return VStr(str(n))
        return VStr("")

    # ── List operations ───────────────────────────────────────

    def _append(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList):
            args[0].items.append(args[1])
        return _NONE_VAL

    def _insert(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList) and isinstance(args[1], VInt):
            args[0].items.insert(args[1].value, args[2])
        return _NONE_VAL

    def _pop(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList):
            if len(args[0].items) == 0:
                raise _VMThrow(_make_error_struct("IndexError", "pop from empty list"))
            return args[0].items.pop()
        return _NONE_VAL

    def _remove_at(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList) and isinstance(args[1], VInt):
            idx = args[1].value
            if idx < 0 or idx >= len(args[0].items):
                raise _VMThrow(_make_error_struct("IndexError", "index out of range"))
            args[0].items.pop(idx)
        return _NONE_VAL

    def _index_of(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList):
            i = 0
            while i < len(args[0].items):
                if _val_eq(args[0].items[i], args[1]):
                    return VInt(i)
                i += 1
        return VInt(-1)

    def _reversed(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList):
            return VList(list(reversed(args[0].items)))
        return VList([])

    def _sorted(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList):
            items = list(args[0].items)
            for item in items:
                if isinstance(item, VFloat) and _isnan(item.value):
                    raise _VMThrow(
                        _make_error_struct(
                            "ValueError", "cannot sort list containing NaN"
                        )
                    )
            _sort_vals(items)
            return VList(items)
        return VList([])

    # ── Map operations ────────────────────────────────────────

    def _map_new(self, args: list[Val]) -> Val:
        return VMap([], [])

    def _map_get(self, args: list[Val]) -> Val:
        if isinstance(args[0], VMap):
            key = args[1]
            i = 0
            while i < len(args[0].keys):
                if _val_eq(args[0].keys[i], key):
                    return args[0].values[i]
                i += 1
            if len(args) > 2:
                return args[2]
            return _NONE_VAL
        return _NONE_VAL

    def _map_delete(self, args: list[Val]) -> Val:
        if isinstance(args[0], VMap):
            key = args[1]
            i = 0
            while i < len(args[0].keys):
                if _val_eq(args[0].keys[i], key):
                    args[0].keys.pop(i)
                    args[0].values.pop(i)
                    return _NONE_VAL
                i += 1
        return _NONE_VAL

    def _map_keys(self, args: list[Val]) -> Val:
        if isinstance(args[0], VMap):
            return VList(list(args[0].keys))
        return VList([])

    def _map_values(self, args: list[Val]) -> Val:
        if isinstance(args[0], VMap):
            return VList(list(args[0].values))
        return VList([])

    def _map_items(self, args: list[Val]) -> Val:
        if isinstance(args[0], VMap):
            result: list[Val] = []
            i = 0
            while i < len(args[0].keys):
                result.append(VTuple([args[0].keys[i], args[0].values[i]]))
                i += 1
            return VList(result)
        return VList([])

    def _map_merge(self, args: list[Val]) -> Val:
        if isinstance(args[0], VMap) and isinstance(args[1], VMap):
            keys: list[Val] = list(args[0].keys)
            values: list[Val] = list(args[0].values)
            i = 0
            while i < len(args[1].keys):
                k2 = args[1].keys[i]
                found = False
                j = 0
                while j < len(keys):
                    if _val_eq(keys[j], k2):
                        values[j] = args[1].values[i]
                        found = True
                        break
                    j += 1
                if not found:
                    keys.append(k2)
                    values.append(args[1].values[i])
                i += 1
            return VMap(keys, values)
        return _NONE_VAL

    def _map_pop_item(self, args: list[Val]) -> Val:
        if isinstance(args[0], VMap):
            if len(args[0].keys) == 0:
                raise _VMThrow(_make_error_struct("KeyError", "pop from empty map"))
            pk = args[0].keys.pop()
            pv = args[0].values.pop()
            return VTuple([pk, pv])
        return _NONE_VAL

    def _map_from_keys(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList):
            keys = args[0].items
            val = args[1]
            ks: list[Val] = []
            vs: list[Val] = []
            for k in keys:
                ks.append(k)
                vs.append(val)
            return VMap(ks, vs)
        return VMap([], [])

    def _map_from_pairs(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList):
            ks: list[Val] = []
            vs: list[Val] = []
            for pair in args[0].items:
                if isinstance(pair, VTuple) and len(pair.items) >= 2:
                    ks.append(pair.items[0])
                    vs.append(pair.items[1])
            return VMap(ks, vs)
        return VMap([], [])

    # ── Set operations ────────────────────────────────────────

    def _set_new(self, args: list[Val]) -> Val:
        return VSet([])

    def _set_add(self, args: list[Val]) -> Val:
        if isinstance(args[0], VSet):
            val = args[1]
            for existing in args[0].items:
                if _val_eq(existing, val):
                    return _NONE_VAL
            args[0].items.append(val)
        return _NONE_VAL

    def _set_remove(self, args: list[Val]) -> Val:
        if isinstance(args[0], VSet):
            val = args[1]
            i = 0
            while i < len(args[0].items):
                if _val_eq(args[0].items[i], val):
                    args[0].items.pop(i)
                    return _NONE_VAL
                i += 1
        return _NONE_VAL

    def _set_union(self, args: list[Val]) -> Val:
        if isinstance(args[0], VSet) and isinstance(args[1], VSet):
            result = list(args[0].items)
            for item in args[1].items:
                found = False
                for r in result:
                    if _val_eq(r, item):
                        found = True
                        break
                if not found:
                    result.append(item)
            return VSet(result)
        return VSet([])

    def _set_intersection(self, args: list[Val]) -> Val:
        if isinstance(args[0], VSet) and isinstance(args[1], VSet):
            result: list[Val] = []
            for item in args[0].items:
                for item2 in args[1].items:
                    if _val_eq(item, item2):
                        result.append(item)
                        break
            return VSet(result)
        return VSet([])

    def _set_difference(self, args: list[Val]) -> Val:
        if isinstance(args[0], VSet) and isinstance(args[1], VSet):
            result: list[Val] = []
            for item in args[0].items:
                found = False
                for item2 in args[1].items:
                    if _val_eq(item, item2):
                        found = True
                        break
                if not found:
                    result.append(item)
            return VSet(result)
        return VSet([])

    def _set_from_list(self, args: list[Val]) -> Val:
        if isinstance(args[0], VSet):
            return args[0]
        if isinstance(args[0], VList):
            result: list[Val] = []
            for item in args[0].items:
                found = False
                for r in result:
                    if _val_eq(r, item):
                        found = True
                        break
                if not found:
                    result.append(item)
            return VSet(result)
        return VSet([])

    def _list_from(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList):
            return VList(list(args[0].items))
        if isinstance(args[0], VSet):
            return VList(list(args[0].items))
        return VList([])

    # ── Misc builtins ─────────────────────────────────────────

    def _unwrap(self, args: list[Val]) -> Val:
        v = args[0]
        if isinstance(v, VNil):
            msg = "unwrap called on nil"
            if len(args) > 1 and isinstance(args[1], VStr):
                msg = args[1].value
            raise _VMThrow(_make_error_struct("NilError", msg))
        return v

    def _is_nil(self, args: list[Val]) -> Val:
        return VBool(isinstance(args[0], VNil))

    def _is_type(self, args: list[Val]) -> Val:
        if len(args) < 2:
            return _FALSE_VAL
        v = args[0]
        if isinstance(args[1], VStr):
            type_name = args[1].value
            return VBool(_val_is_type(v, type_name, self.vm.module.interface_defs))
        return _FALSE_VAL

    def _floor_div(self, args: list[Val]) -> Val:
        if isinstance(args[0], VInt) and isinstance(args[1], VInt):
            if args[1].value == 0:
                raise _VMThrow(
                    _make_error_struct("ZeroDivisionError", "division by zero")
                )
            return VInt(args[0].value // args[1].value)
        return _ZERO_INT

    def _python_mod(self, args: list[Val]) -> Val:
        if isinstance(args[0], VInt) and isinstance(args[1], VInt):
            if args[1].value == 0:
                raise _VMThrow(
                    _make_error_struct("ZeroDivisionError", "division by zero")
                )
            return VInt(args[0].value % args[1].value)
        return _ZERO_INT

    def _wrapping_add(self, args: list[Val]) -> Val:
        if isinstance(args[0], VInt) and isinstance(args[1], VInt):
            return VInt(_wrapping_add(args[0].value, args[1].value))
        return _ZERO_INT

    def _wrapping_sub(self, args: list[Val]) -> Val:
        if isinstance(args[0], VInt) and isinstance(args[1], VInt):
            return VInt(_wrapping_sub(args[0].value, args[1].value))
        return _ZERO_INT

    def _wrapping_mul(self, args: list[Val]) -> Val:
        if isinstance(args[0], VInt) and isinstance(args[1], VInt):
            return VInt(_wrapping_mul(args[0].value, args[1].value))
        return _ZERO_INT

    def _bytes_new(self, args: list[Val]) -> Val:
        if isinstance(args[0], VInt):
            return VBytes(bytes(args[0].value))
        return VBytes(b"")

    def _bytes_from(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList):
            bs: list[int] = []
            for item in args[0].items:
                if isinstance(item, VByte):
                    bs.append(item.value)
                elif isinstance(item, VInt):
                    bs.append(item.value & 0xFF)
            return VBytes(bytes(bs))
        return VBytes(b"")

    def _range_list(self, args: list[Val]) -> Val:
        if len(args) == 1 and isinstance(args[0], VInt):
            r1: list[Val] = []
            ri1 = 0
            end1 = args[0].value
            while ri1 < end1:
                r1.append(VInt(ri1))
                ri1 += 1
            return VList(r1)
        if len(args) == 2 and isinstance(args[0], VInt) and isinstance(args[1], VInt):
            r2: list[Val] = []
            ri2 = args[0].value
            end2 = args[1].value
            while ri2 < end2:
                r2.append(VInt(ri2))
                ri2 += 1
            return VList(r2)
        if (
            len(args) >= 3
            and isinstance(args[0], VInt)
            and isinstance(args[1], VInt)
            and isinstance(args[2], VInt)
        ):
            r3: list[Val] = []
            ri3 = args[0].value
            end3 = args[1].value
            step3 = args[2].value
            if step3 > 0:
                while ri3 < end3:
                    r3.append(VInt(ri3))
                    ri3 += step3
            elif step3 < 0:
                while ri3 > end3:
                    r3.append(VInt(ri3))
                    ri3 += step3
            return VList(r3)
        return VList([])

    def _list_compare(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList) and isinstance(args[1], VList):
            a = args[0].items
            b = args[1].items
            i = 0
            while i < len(a) and i < len(b):
                c = _val_compare(a[i], b[i])
                if c != 0:
                    return VInt(c)
                i += 1
            if len(a) < len(b):
                return VInt(-1)
            if len(a) > len(b):
                return VInt(1)
            return _ZERO_INT
        return _ZERO_INT

    def _zip(self, args: list[Val]) -> Val:
        if isinstance(args[0], VList) and isinstance(args[1], VList):
            a = args[0].items
            b = args[1].items
            n = min(len(a), len(b))
            result: list[Val] = []
            i = 0
            while i < n:
                result.append(VTuple([a[i], b[i]]))
                i += 1
            return VList(result)
        return VList([])

    def _chars(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            cv: list[Val] = []
            ci = 0
            while ci < len(args[0].value):
                cv.append(VRune(args[0].value[ci : ci + 1]))
                ci += 1
            return VList(cv)
        return VList([])

    def _replace_slice(self, args: list[Val]) -> Val:
        if (
            isinstance(args[0], VList)
            and isinstance(args[1], VInt)
            and isinstance(args[2], VInt)
            and isinstance(args[3], VList)
        ):
            lo = args[1].value
            hi = args[2].value
            args[0].items[lo:hi] = args[3].items
        return _NONE_VAL

    def _encode(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            return VBytes(args[0].value.encode("utf-8"))
        return VBytes(b"")

    def _decode(self, args: list[Val]) -> Val:
        if isinstance(args[0], VBytes):
            try:
                return VStr(args[0].value.decode("utf-8"))
            except UnicodeDecodeError:
                raise _VMThrow(_make_error_struct("ValueError", "invalid utf-8"))
        return VStr("")

    def _read_line(self, args: list[Val]) -> Val:
        data = self.vm.stdin_data
        pos = self.vm.stdin_pos
        if pos >= len(data):
            return VStr("")
        end = data[pos:].find(b"\n")
        if end != -1:
            end = end + pos
        if end == -1:
            end = len(data)
        else:
            end += 1
        line = data[pos:end].decode("utf-8", errors="replace")
        self.vm.stdin_pos = end
        if line.endswith("\n"):
            line = line[:-1]
        return VStr(line)

    def _read_all(self, args: list[Val]) -> Val:
        data = self.vm.stdin_data
        pos = self.vm.stdin_pos
        rest = data[pos:].decode("utf-8", errors="replace")
        self.vm.stdin_pos = len(data)
        return VStr(rest)

    def _read_bytes(self, args: list[Val]) -> Val:
        pos = self.vm.stdin_pos
        self.vm.stdin_pos = len(self.vm.stdin_data)
        return VBytes(self.vm.stdin_data[pos:])

    def _read_bytes_n(self, args: list[Val]) -> Val:
        if isinstance(args[0], VInt):
            return self._do_read_bytes_n(args[0].value)
        return VBytes(b"")

    def _do_read_bytes_n(self, n: int) -> Val:
        data = self.vm.stdin_data
        pos = self.vm.stdin_pos
        end = pos + n
        if end > len(data):
            end = len(data)
        chunk = data[pos:end]
        self.vm.stdin_pos = end
        return VBytes(chunk)

    def _read_file(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            try:
                with open(args[0].value, "rb") as f:
                    data = f.read()
                try:
                    return VStr(data.decode("utf-8"))
                except UnicodeDecodeError as e:
                    raise _VMThrow(_make_error_struct("ValueError", str(e)))
            except OSError as e:
                raise _VMThrow(_make_error_struct("IOError", str(e)))
        return VStr("")

    def _read_file_bytes(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            try:
                return _read_file_bytes(args[0].value)
            except OSError as e:
                raise _VMThrow(_make_error_struct("IOError", str(e)))
        return VBytes(b"")

    def _write_file(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr) and isinstance(args[1], VBytes):
            try:
                _write_file_bytes(args[0].value, args[1].value)
            except OSError as e:
                raise _VMThrow(_make_error_struct("IOError", str(e)))
        return _NONE_VAL

    def _args(self, args: list[Val]) -> Val:
        items_a: list[Val] = []
        ai = 0
        while ai < len(self.vm.program_args):
            items_a.append(VStr(self.vm.program_args[ai]))
            ai += 1
        return VList(items_a)

    def _get_env(self, args: list[Val]) -> Val:
        if isinstance(args[0], VStr):
            val = self.vm.env_vars.get(args[0].value)
            if val is not None:
                return VStr(val)
        return _NONE_VAL


def _val_is_type(
    v: Val, type_name: str, interface_defs: list[InterfaceDef] | None = None
) -> bool:
    if isinstance(v, VStruct):
        if v.type_name == type_name:
            return True
        if interface_defs is not None:
            for idef in interface_defs:
                if idef.name == type_name and v.type_name in idef.variant_names:
                    return True
        return False
    if isinstance(v, VEnum):
        return v.enum_name == type_name
    if isinstance(v, VInt):
        return type_name == "int"
    if isinstance(v, VFloat):
        return type_name == "float"
    if isinstance(v, VBool):
        return type_name == "bool"
    if isinstance(v, VStr):
        return type_name == "string"
    if isinstance(v, VByte):
        return type_name == "byte"
    if isinstance(v, VBytes):
        return type_name == "bytes"
    if isinstance(v, VRune):
        return type_name == "rune"
    if isinstance(v, VNil):
        return type_name == "nil"
    if isinstance(v, VList):
        return type_name == "list"
    if isinstance(v, VMap):
        return type_name == "map"
    if isinstance(v, VSet):
        return type_name == "set"
    if isinstance(v, VTuple):
        return type_name == "tuple"
    return False


def _val_compare(a: Val, b: Val) -> int:
    if isinstance(a, VInt) and isinstance(b, VInt):
        if a.value < b.value:
            return -1
        if a.value > b.value:
            return 1
        return 0
    if isinstance(a, VFloat) and isinstance(b, VFloat):
        if a.value < b.value:
            return -1
        if a.value > b.value:
            return 1
        return 0
    if isinstance(a, VStr) and isinstance(b, VStr):
        if a.value < b.value:
            return -1
        if a.value > b.value:
            return 1
        return 0
    if isinstance(a, VByte) and isinstance(b, VByte):
        if a.value < b.value:
            return -1
        if a.value > b.value:
            return 1
        return 0
    return 0


def _sort_vals(items: list[Val]) -> None:
    """Insertion sort for Val lists."""
    i = 1
    while i < len(items):
        key = items[i]
        j = i - 1
        while j >= 0 and _val_compare(items[j], key) > 0:
            items[j + 1] = items[j]
            j -= 1
        items[j + 1] = key
        i += 1


# ============================================================
# VM
# ============================================================


class VM:
    def __init__(
        self,
        module: CompiledModule,
    ) -> None:
        self.module: CompiledModule = module
        self.stack: list[Val] = []
        self.frames: list[Frame] = []
        self.handlers: list[Handler] = []
        self.globals: list[Val] = []
        self.stdout_buf: list[str] = []
        self.stderr_buf: list[str] = []
        self.stdin_data: bytes = b""
        self.stdin_pos: int = 0
        self.program_args: list[str] = []
        self.env_vars: dict[str, str] = {}
        self.builtins: _BuiltinDispatch = _BuiltinDispatch(self)
        self.pending_exception: Val | None = None
        # Initialize globals — one slot per function
        self._init_globals()

    def _init_globals(self) -> None:
        # Map function names to their code object indices
        self.globals = []
        i = 0
        while i < len(self.module.global_names):
            name = self.module.global_names[i]
            # Find the code object for this function
            found = False
            j = 0
            while j < len(self.module.code_objects):
                co = self.module.code_objects[j]
                if co.name == name:
                    self.globals.append(VFunc(j, co.type_sig))
                    found = True
                    break
                j += 1
            if not found:
                self.globals.append(_NONE_VAL)
            i += 1

    def run(self) -> VMResult:
        if self.module.entry_index < 0:
            return VMResult(1, "", "no Main function")
        try:
            # Run top-level let initializers
            if self.module.init_index >= 0:
                init_code = self.module.code_objects[self.module.init_index]
                init_frame = Frame(code=init_code, ip=0, bp=len(self.stack))
                self.frames.append(init_frame)
                self._dispatch()
            # Run Main
            entry_code = self.module.code_objects[self.module.entry_index]
            frame = Frame(code=entry_code, ip=0, bp=len(self.stack))
            i = 0
            while i < entry_code.local_count:
                self.stack.append(_NONE_VAL)
                i += 1
            self.frames.append(frame)
            self._dispatch()
            return VMResult(0, "".join(self.stdout_buf), "".join(self.stderr_buf))
        except _VMExit as e:
            return VMResult(e.code, "".join(self.stdout_buf), "".join(self.stderr_buf))
        except _VMThrow as t:
            msg = _val_to_string(t.value)
            self.stderr_buf.append(msg)
            self.stderr_buf.append("\n")
            return VMResult(1, "".join(self.stdout_buf), "".join(self.stderr_buf))

    def _dispatch(self) -> None:
        while len(self.frames) > 0:
            try:
                self._dispatch_one()
            except _VMThrow as t:
                if len(self.handlers) > 0:
                    self._throw(t.value)
                else:
                    raise t

    def _dispatch_one(self) -> None:
        while len(self.frames) > 0:
            frame = self.frames[-1]
            code = frame.code.code
            if frame.ip >= len(code):
                self._do_return_void()
                continue
            op = code[frame.ip]
            arg = code[frame.ip + 1]
            frame.ip += 2
            # Handle EXTENDED_ARG
            if op == OP_EXTENDED_ARG:
                next_op = code[frame.ip]
                next_arg = code[frame.ip + 1]
                frame.ip += 2
                arg = (arg << 8) | next_arg
                op = next_op
                # Could chain more
                if op == OP_EXTENDED_ARG:
                    next_op2 = code[frame.ip]
                    next_arg2 = code[frame.ip + 1]
                    frame.ip += 2
                    arg = (arg << 8) | next_arg2
                    op = next_op2
            # ── Constants ─────────────────────────────────
            if op == OP_CONST:
                self.stack.append(frame.code.constants[arg])
            elif op == OP_TRUE:
                self.stack.append(_TRUE_VAL)
            elif op == OP_FALSE:
                self.stack.append(_FALSE_VAL)
            elif op == OP_NIL:
                self.stack.append(_NONE_VAL)
            elif op == OP_INT_ZERO:
                self.stack.append(_ZERO_INT)
            elif op == OP_INT_ONE:
                self.stack.append(_ONE_INT)
            # ── Variables ─────────────────────────────────
            elif op == OP_LOAD_LOCAL:
                self.stack.append(self.stack[frame.bp + arg])
            elif op == OP_STORE_LOCAL:
                self.stack[frame.bp + arg] = self.stack[-1]
                self.stack.pop()
            elif op == OP_LOAD_GLOBAL:
                self.stack.append(self.globals[arg])
            elif op == OP_STORE_GLOBAL:
                self.globals[arg] = self.stack.pop()
            elif op == OP_LOAD_BUILTIN:
                self.stack.append(VInt(arg))  # placeholder — builtins resolved at call
            elif op == OP_LOAD_CAPTURE:
                self.stack.append(_NONE_VAL)
            # ── Stack ─────────────────────────────────────
            elif op == OP_POP:
                if len(self.stack) > frame.bp:
                    self.stack.pop()
            elif op == OP_DUP:
                self.stack.append(self.stack[-1])
            elif op == OP_ROT_TWO:
                a = self.stack[-1]
                b = self.stack[-2]
                self.stack[-1] = b
                self.stack[-2] = a
            # ── Arithmetic ────────────────────────────────
            elif op == OP_ADD_INT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VInt) and isinstance(b, VInt):
                    r = a.value + b.value
                    if r > _INT64_MAX or r < _INT64_MIN:
                        self._throw(
                            _make_error_struct("OverflowError", "integer overflow")
                        )
                        continue
                    self.stack.append(VInt(r))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_ADD_FLOAT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VFloat) and isinstance(b, VFloat):
                    self.stack.append(VFloat(a.value + b.value))
                else:
                    self.stack.append(VFloat(0.0))
            elif op == OP_ADD_STRING:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VStr) and isinstance(b, VStr):
                    self.stack.append(VStr(a.value + b.value))
                else:
                    self.stack.append(VStr(""))
            elif op == OP_ADD_BYTE:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VByte) and isinstance(b, VByte):
                    self.stack.append(VByte((a.value + b.value) & 0xFF))
                else:
                    self.stack.append(VByte(0))
            elif op == OP_SUB_INT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VInt) and isinstance(b, VInt):
                    r = a.value - b.value
                    if r > _INT64_MAX or r < _INT64_MIN:
                        self._throw(
                            _make_error_struct("OverflowError", "integer overflow")
                        )
                        continue
                    self.stack.append(VInt(r))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_SUB_FLOAT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VFloat) and isinstance(b, VFloat):
                    self.stack.append(VFloat(a.value - b.value))
                else:
                    self.stack.append(VFloat(0.0))
            elif op == OP_SUB_BYTE:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VByte) and isinstance(b, VByte):
                    self.stack.append(VByte((a.value - b.value) & 0xFF))
                else:
                    self.stack.append(VByte(0))
            elif op == OP_MUL_INT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VInt) and isinstance(b, VInt):
                    r = a.value * b.value
                    if r > _INT64_MAX or r < _INT64_MIN:
                        self._throw(
                            _make_error_struct("OverflowError", "integer overflow")
                        )
                        continue
                    self.stack.append(VInt(r))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_MUL_FLOAT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VFloat) and isinstance(b, VFloat):
                    self.stack.append(VFloat(a.value * b.value))
                else:
                    self.stack.append(VFloat(0.0))
            elif op == OP_MUL_BYTE:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VByte) and isinstance(b, VByte):
                    self.stack.append(VByte((a.value * b.value) & 0xFF))
                else:
                    self.stack.append(VByte(0))
            elif op == OP_DIV_INT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VByte) and isinstance(b, VByte):
                    if b.value == 0:
                        self._throw(
                            _make_error_struct(
                                "ZeroDivisionError", "integer division by zero"
                            )
                        )
                        continue
                    self.stack.append(VByte(a.value // b.value))
                elif isinstance(a, VInt) and isinstance(b, VInt):
                    if b.value == 0:
                        self._throw(
                            _make_error_struct(
                                "ZeroDivisionError", "integer division by zero"
                            )
                        )
                        continue
                    # Truncate toward zero
                    if (a.value < 0) != (b.value < 0) and a.value % b.value != 0:
                        self.stack.append(VInt(-(abs(a.value) // abs(b.value))))
                    else:
                        self.stack.append(VInt(a.value // b.value))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_DIV_FLOAT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VFloat) and isinstance(b, VFloat):
                    if b.value == 0.0:
                        if a.value == 0.0:
                            self.stack.append(VFloat(float("nan")))
                        elif a.value > 0:
                            self.stack.append(VFloat(float("inf")))
                        else:
                            self.stack.append(VFloat(float("-inf")))
                    else:
                        self.stack.append(VFloat(a.value / b.value))
                else:
                    self.stack.append(VFloat(0.0))
            elif op == OP_MOD_INT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VByte) and isinstance(b, VByte):
                    if b.value == 0:
                        self._throw(
                            _make_error_struct(
                                "ZeroDivisionError", "integer modulo by zero"
                            )
                        )
                        continue
                    self.stack.append(VByte(a.value % b.value))
                elif isinstance(a, VInt) and isinstance(b, VInt):
                    if b.value == 0:
                        self._throw(
                            _make_error_struct(
                                "ZeroDivisionError", "integer modulo by zero"
                            )
                        )
                        continue
                    # Remainder follows dividend sign (truncation toward zero)
                    if a.value == 0:
                        self.stack.append(_ZERO_INT)
                    else:
                        r = abs(a.value) % abs(b.value)
                        if a.value < 0:
                            r = -r
                        self.stack.append(VInt(r))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_MOD_FLOAT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VFloat) and isinstance(b, VFloat):
                    if b.value == 0.0:
                        self._throw(
                            _make_error_struct(
                                "ZeroDivisionError", "float modulo by zero"
                            )
                        )
                        continue
                    self.stack.append(VFloat(_fmod(a.value, b.value)))
                else:
                    self.stack.append(VFloat(0.0))
            elif op == OP_NEG_INT:
                v = self.stack.pop()
                if isinstance(v, VByte):
                    self.stack.append(VByte((-v.value) & 0xFF))
                elif isinstance(v, VInt):
                    r = -v.value
                    if r > _INT64_MAX:
                        self._throw(
                            _make_error_struct("OverflowError", "integer overflow")
                        )
                        continue
                    self.stack.append(VInt(r))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_NEG_FLOAT:
                v = self.stack.pop()
                if isinstance(v, VFloat):
                    self.stack.append(VFloat(-v.value))
                else:
                    self.stack.append(VFloat(0.0))
            elif op == OP_BIT_AND:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VByte) and isinstance(b, VByte):
                    self.stack.append(VByte(a.value & b.value))
                elif isinstance(a, VInt) and isinstance(b, VInt):
                    self.stack.append(VInt(a.value & b.value))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_BIT_OR:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VByte) and isinstance(b, VByte):
                    self.stack.append(VByte(a.value | b.value))
                elif isinstance(a, VInt) and isinstance(b, VInt):
                    self.stack.append(VInt(a.value | b.value))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_BIT_XOR:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VByte) and isinstance(b, VByte):
                    self.stack.append(VByte(a.value ^ b.value))
                elif isinstance(a, VInt) and isinstance(b, VInt):
                    self.stack.append(VInt(a.value ^ b.value))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_BIT_NOT:
                v = self.stack.pop()
                if isinstance(v, VByte):
                    self.stack.append(VByte((~v.value) & 0xFF))
                elif isinstance(v, VInt):
                    self.stack.append(VInt(~v.value))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_SHIFT_LEFT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VInt) and isinstance(b, VInt):
                    if b.value < 0 or b.value >= 64:
                        self._throw(
                            _make_error_struct(
                                "OverflowError", "shift amount out of range"
                            )
                        )
                        continue
                    r = a.value << b.value
                    r = ((r - _INT64_MIN) % (2**64)) + _INT64_MIN
                    self.stack.append(VInt(r))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_SHIFT_RIGHT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VInt) and isinstance(b, VInt):
                    if b.value < 0 or b.value >= 64:
                        self._throw(
                            _make_error_struct(
                                "OverflowError", "shift amount out of range"
                            )
                        )
                        continue
                    self.stack.append(VInt(a.value >> b.value))
                else:
                    self.stack.append(_ZERO_INT)
            elif op == OP_SHIFT_RIGHT_UNSIGNED:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VInt) and isinstance(b, VInt):
                    if b.value < 0 or b.value >= 64:
                        self._throw(
                            _make_error_struct(
                                "OverflowError", "shift amount out of range"
                            )
                        )
                        continue
                    mask = (1 << 64) - 1
                    ua = a.value & mask
                    self.stack.append(VInt(ua >> b.value))
                else:
                    self.stack.append(_ZERO_INT)
            # ── Comparison ────────────────────────────────
            elif op == OP_EQ:
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(_TRUE_VAL if _val_eq(a, b) else _FALSE_VAL)
            elif op == OP_NE:
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(_FALSE_VAL if _val_eq(a, b) else _TRUE_VAL)
            elif op == OP_CMP_INT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VInt) and isinstance(b, VInt):
                    self.stack.append(_cmp_result(a.value, b.value, arg))
                else:
                    self.stack.append(_FALSE_VAL)
            elif op == OP_CMP_FLOAT:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VFloat) and isinstance(b, VFloat):
                    self.stack.append(_cmp_result_float(a.value, b.value, arg))
                else:
                    self.stack.append(_FALSE_VAL)
            elif op == OP_CMP_STRING:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VStr) and isinstance(b, VStr):
                    self.stack.append(_cmp_result_str(a.value, b.value, arg))
                else:
                    self.stack.append(_FALSE_VAL)
            elif op == OP_CMP_BYTE:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VByte) and isinstance(b, VByte):
                    self.stack.append(_cmp_result(a.value, b.value, arg))
                else:
                    self.stack.append(_FALSE_VAL)
            elif op == OP_CMP_RUNE:
                b = self.stack.pop()
                a = self.stack.pop()
                if isinstance(a, VRune) and isinstance(b, VRune):
                    self.stack.append(_cmp_result_str(a.value, b.value, arg))
                else:
                    self.stack.append(_FALSE_VAL)
            # ── Logic ─────────────────────────────────────
            elif op == OP_NOT:
                v = self.stack.pop()
                self.stack.append(_FALSE_VAL if _is_truthy(v) else _TRUE_VAL)
            # ── Control flow ──────────────────────────────
            elif op == OP_JUMP:
                frame.ip += arg
            elif op == OP_JUMP_IF_FALSE:
                v = self.stack.pop()
                if not _is_truthy(v):
                    frame.ip += arg
            elif op == OP_JUMP_IF_TRUE:
                v = self.stack.pop()
                if _is_truthy(v):
                    frame.ip += arg
            elif op == OP_JUMP_BACK:
                frame.ip -= arg
            elif op == OP_RETURN:
                ret_val = self.stack.pop()
                self._do_return(ret_val)
            elif op == OP_RETURN_VOID:
                self._do_return_void()
            elif op == OP_CALL:
                self._do_call(arg)
            elif op == OP_CALL_METHOD:
                self._do_call_method(frame, arg)
            elif op == OP_CALL_BUILTIN:
                self._do_call_builtin(arg)
            # ── Collections ───────────────────────────────
            elif op == OP_BUILD_LIST:
                items: list[Val] = []
                i = 0
                while i < arg:
                    items.append(_NONE_VAL)
                    i += 1
                i = arg - 1
                while i >= 0:
                    items[i] = self.stack.pop()
                    i -= 1
                self.stack.append(VList(items))
            elif op == OP_BUILD_MAP:
                ks: list[Val] = []
                vs: list[Val] = []
                pairs: list[tuple[Val, Val]] = []
                i = 0
                while i < arg:
                    pairs.append((_NONE_VAL, _NONE_VAL))
                    i += 1
                i = arg - 1
                while i >= 0:
                    v = self.stack.pop()
                    k = self.stack.pop()
                    pairs[i] = (k, v)
                    i -= 1
                pi = 0
                while pi < len(pairs):
                    ks.append(pairs[pi][0])
                    vs.append(pairs[pi][1])
                    pi += 1
                self.stack.append(VMap(ks, vs))
            elif op == OP_BUILD_SET:
                raw: list[Val] = []
                i = 0
                while i < arg:
                    raw.append(_NONE_VAL)
                    i += 1
                i = arg - 1
                while i >= 0:
                    raw[i] = self.stack.pop()
                    i -= 1
                items2: list[Val] = []
                for item in raw:
                    dup = False
                    for existing in items2:
                        if _val_eq(item, existing):
                            dup = True
                            break
                    if not dup:
                        items2.append(item)
                self.stack.append(VSet(items2))
            elif op == OP_BUILD_TUPLE:
                items3: list[Val] = []
                i = 0
                while i < arg:
                    items3.append(_NONE_VAL)
                    i += 1
                i = arg - 1
                while i >= 0:
                    items3[i] = self.stack.pop()
                    i -= 1
                self.stack.append(VTuple(items3))
            elif op == OP_INDEX:
                idx_val = self.stack.pop()
                obj = self.stack.pop()
                self._do_index(obj, idx_val)
            elif op == OP_STORE_INDEX:
                val = self.stack.pop()
                idx_val = self.stack.pop()
                obj = self.stack.pop()
                self._do_store_index(obj, idx_val, val)
            elif op == OP_SLICE:
                high = self.stack.pop()
                low = self.stack.pop()
                obj = self.stack.pop()
                self._do_slice(obj, low, high)
            elif op == OP_TUPLE_ACCESS:
                obj = self.stack.pop()
                if isinstance(obj, VTuple) and arg < len(obj.items):
                    self.stack.append(obj.items[arg])
                else:
                    self.stack.append(_NONE_VAL)
            # ── Structs/Enums ─────────────────────────────
            elif op == OP_BUILD_STRUCT:
                self._do_build_struct(frame, arg)
            elif op == OP_GET_FIELD:
                self._do_get_field(frame, arg)
            elif op == OP_SET_FIELD:
                self._do_set_field(frame, arg)
            elif op == OP_LOAD_ENUM:
                enum_name_val = frame.code.constants[arg]
                variant_val = frame.code.constants[arg + 1]
                if isinstance(enum_name_val, VStr) and isinstance(variant_val, VStr):
                    self.stack.append(VEnum(enum_name_val.value, variant_val.value))
                else:
                    self.stack.append(_NONE_VAL)
            # ── Exceptions ────────────────────────────────
            elif op == OP_PUSH_HANDLER:
                handler = Handler(
                    frame_depth=len(self.frames) - 1,
                    catch_ip=frame.ip + arg,
                    stack_depth=len(self.stack),
                    kind=HANDLER_CATCH,
                )
                self.handlers.append(handler)
            elif op == OP_POP_HANDLER:
                if len(self.handlers) > 0:
                    self.handlers.pop()
            elif op == OP_THROW:
                val = self.stack.pop()
                if isinstance(val, VNil):
                    pend = self.pending_exception
                    self.pending_exception = None
                    if pend is not None:
                        self._throw(pend)
                    else:
                        self._throw(val)
                else:
                    self.pending_exception = None
                    self._throw(val)
            elif op == OP_PUSH_FINALLY:
                handler = Handler(
                    frame_depth=len(self.frames) - 1,
                    catch_ip=frame.ip + arg,
                    stack_depth=len(self.stack),
                    kind=HANDLER_FINALLY,
                )
                self.handlers.append(handler)
            # ── Iteration ─────────────────────────────────
            elif op == OP_GET_ITER:
                self._do_get_iter(arg)
            elif op == OP_FOR_ITER:
                self._do_for_iter(frame, arg)
            elif op == OP_UNPACK:
                self._do_unpack(arg)
            # ── Type tests ────────────────────────────────
            elif op == OP_IS_TYPE:
                type_name_val = frame.code.constants[arg]
                val = self.stack.pop()
                if isinstance(type_name_val, VStr):
                    self.stack.append(
                        _TRUE_VAL
                        if _val_is_type(
                            val, type_name_val.value, self.module.interface_defs
                        )
                        else _FALSE_VAL
                    )
                else:
                    self.stack.append(_FALSE_VAL)
            elif op == OP_MATCH_TYPE:
                type_name_val = frame.code.constants[arg]
                val = self.stack[-1]  # peek, don't pop
                if isinstance(type_name_val, VStr):
                    tn = type_name_val.value
                    if "." in tn:
                        # Enum variant match
                        if isinstance(val, VEnum):
                            self.stack.append(
                                _TRUE_VAL
                                if (val.enum_name + "." + val.variant) == tn
                                else _FALSE_VAL
                            )
                        else:
                            self.stack.append(_FALSE_VAL)
                    elif tn == "nil":
                        self.stack.append(
                            _TRUE_VAL if isinstance(val, VNil) else _FALSE_VAL
                        )
                    else:
                        self.stack.append(
                            _TRUE_VAL
                            if _val_is_type(val, tn, self.module.interface_defs)
                            else _FALSE_VAL
                        )
                else:
                    self.stack.append(_FALSE_VAL)

    def _throw(self, val: Val) -> None:
        """Throw an exception, unwinding to the nearest handler."""
        while len(self.handlers) > 0:
            handler = self.handlers.pop()
            # Unwind frames
            while len(self.frames) > handler.frame_depth + 1:
                self.frames.pop()
            # Restore stack
            while len(self.stack) > handler.stack_depth:
                self.stack.pop()
            if handler.kind == HANDLER_CATCH:
                # Push exception value for catch handler
                self.stack.append(val)
                self.frames[-1].ip = handler.catch_ip
                return
            elif handler.kind == HANDLER_FINALLY:
                self.frames[-1].ip = handler.catch_ip
                self.pending_exception = val
                return
        # No handler found — propagate as Python exception
        raise _VMThrow(val)

    def _do_return(self, ret_val: Val) -> None:
        frame = self.frames.pop()
        # Pop locals and args
        while len(self.stack) > frame.bp:
            self.stack.pop()
        self.stack.append(ret_val)

    def _do_return_void(self) -> None:
        frame = self.frames.pop()
        while len(self.stack) > frame.bp:
            self.stack.pop()
        self.stack.append(_NONE_VAL)

    def _do_call(self, argc: int) -> None:
        # Stack: [func, arg0, arg1, ..., argN-1]
        # Pop args
        args: list[Val] = []
        i = 0
        while i < argc:
            args.append(_NONE_VAL)
            i += 1
        i = argc - 1
        while i >= 0:
            args[i] = self.stack.pop()
            i -= 1
        func_val = self.stack.pop()
        if isinstance(func_val, VFunc):
            code = self.module.code_objects[func_val.code_index]
            bp = len(self.stack)
            # Push args as locals
            for a in args:
                self.stack.append(a)
            # Pad remaining locals
            i2 = argc
            while i2 < code.local_count:
                self.stack.append(_NONE_VAL)
                i2 += 1
            self.frames.append(Frame(code=code, ip=0, bp=bp))
        elif isinstance(func_val, VInt):
            # This is a builtin index from LOAD_BUILTIN
            result = self.builtins.call(func_val.value, args)
            self.stack.append(result)
        else:
            raise _VMThrow(_make_error_struct("TypeError", "not callable"))

    def _do_call_method(self, frame: Frame, const_idx: int) -> None:
        """Call a method. Stack: [..., obj, arg0, ..., argN-1]. Method name in constants.
        Followed by trailing (0, argc) pair encoding the argument count."""
        method_name_val = frame.code.constants[const_idx]
        if not isinstance(method_name_val, VStr):
            return
        method_name = method_name_val.value
        _skip = frame.code.code[frame.ip]
        argc = frame.code.code[frame.ip + 1]
        frame.ip += 2
        # Pop args
        args: list[Val] = []
        i = 0
        while i < argc:
            args.append(_NONE_VAL)
            i += 1
        i = argc - 1
        while i >= 0:
            args[i] = self.stack.pop()
            i -= 1
        # Pop receiver (self)
        obj = self.stack.pop()
        if isinstance(obj, VStruct):
            # Look up method in struct defs
            all_sdefs: list[StructDef] = self.module.struct_defs
            for sd in all_sdefs:
                if sd.name == obj.type_name:
                    mi = 0
                    while mi < len(sd.method_names):
                        if sd.method_names[mi] == method_name:
                            code = self.module.code_objects[sd.method_indices[mi]]
                            bp = len(self.stack)
                            # Push 'this' as local[0], then args
                            self.stack.append(obj)
                            for a in args:
                                self.stack.append(a)
                            # Pad remaining locals
                            filled = 1 + argc
                            while filled < code.local_count:
                                self.stack.append(_NONE_VAL)
                                filled += 1
                            self.frames.append(Frame(code=code, ip=0, bp=bp))
                            return
                        mi += 1
                    # Check parent struct
                    parent = sd.parent
                    while parent is not None:
                        found_parent = False
                        for psd in all_sdefs:
                            if psd.name == parent:
                                found_parent = True
                                pi = 0
                                while pi < len(psd.method_names):
                                    if psd.method_names[pi] == method_name:
                                        code = self.module.code_objects[
                                            psd.method_indices[pi]
                                        ]
                                        bp = len(self.stack)
                                        self.stack.append(obj)
                                        for a in args:
                                            self.stack.append(a)
                                        filled = 1 + argc
                                        while filled < code.local_count:
                                            self.stack.append(_NONE_VAL)
                                            filled += 1
                                        self.frames.append(
                                            Frame(code=code, ip=0, bp=bp)
                                        )
                                        return
                                    pi += 1
                                parent = psd.parent
                                break
                        if not found_parent:
                            parent = None
                    break
        self._throw(
            _make_error_struct(
                "TypeError", "no method '" + method_name + "' on " + _val_to_string(obj)
            )
        )

    def _find_method(self, type_name: str, method_name: str) -> int | None:
        """Find a method index for a struct type. Returns code object index or None."""
        sdefs: list[StructDef] = self.module.struct_defs
        for sd in sdefs:
            if sd.name == type_name:
                mi = 0
                while mi < len(sd.method_names):
                    if sd.method_names[mi] == method_name:
                        return sd.method_indices[mi]
                    mi += 1
                parent = sd.parent
                while parent is not None:
                    found_parent = False
                    for psd in sdefs:
                        if psd.name == parent:
                            found_parent = True
                            pi = 0
                            while pi < len(psd.method_names):
                                if psd.method_names[pi] == method_name:
                                    return psd.method_indices[pi]
                                pi += 1
                            parent = psd.parent
                            break
                    if not found_parent:
                        parent = None
                break
        return None

    def _call_method_sync(self, obj: Val, code_idx: int, args: list[Val]) -> Val:
        """Synchronously call a method and return its result."""
        code = self.module.code_objects[code_idx]
        bp = len(self.stack)
        saved_frames = len(self.frames)
        self.stack.append(obj)
        for a in args:
            self.stack.append(a)
        filled = 1 + len(args)
        while filled < code.local_count:
            self.stack.append(_NONE_VAL)
            filled += 1
        self.frames.append(Frame(code=code, ip=0, bp=bp))
        # Run until this frame returns
        while len(self.frames) > saved_frames:
            self._dispatch_one()
        if len(self.stack) > bp:
            return self.stack.pop()
        return _NONE_VAL

    def _do_call_builtin(self, idx: int) -> None:
        """Call a builtin. Followed by trailing (0, argc) pair encoding arg count."""
        frame = self.frames[-1]
        _skip2 = frame.code.code[frame.ip]
        argc = frame.code.code[frame.ip + 1]
        frame.ip += 2
        # Pop args
        args_list: list[Val] = []
        i = 0
        while i < argc:
            args_list.append(_NONE_VAL)
            i += 1
        i = argc - 1
        while i >= 0:
            args_list[i] = self.stack.pop()
            i -= 1
        try:
            result = self.builtins.call(idx, args_list)
            self.stack.append(result)
        except _VMThrow:
            raise
        except _VMExit:
            raise

    def _do_index(self, obj: Val, idx: Val) -> None:
        if isinstance(obj, VList):
            if isinstance(idx, VInt):
                i = idx.value
                if i < 0:
                    i += len(obj.items)
                if i < 0 or i >= len(obj.items):
                    self._throw(
                        _make_error_struct("IndexError", "list index out of range")
                    )
                    return
                self.stack.append(obj.items[i])
                return
        if isinstance(obj, VMap):
            i = 0
            while i < len(obj.keys):
                if _val_eq(obj.keys[i], idx):
                    self.stack.append(obj.values[i])
                    return
                i += 1
            self._throw(
                _make_error_struct("KeyError", "key not found: " + _val_to_string(idx))
            )
            return
        if isinstance(obj, VStr):
            if isinstance(idx, VInt):
                i = idx.value
                if i < 0:
                    i += len(obj.value)
                if i < 0 or i >= len(obj.value):
                    self._throw(
                        _make_error_struct("IndexError", "string index out of range")
                    )
                    return
                self.stack.append(VRune(obj.value[i]))
                return
        if isinstance(obj, VBytes):
            if isinstance(idx, VInt):
                i = idx.value
                if i < 0:
                    i += len(obj.value)
                if i < 0 or i >= len(obj.value):
                    self._throw(
                        _make_error_struct("IndexError", "bytes index out of range")
                    )
                    return
                self.stack.append(VByte(int(obj.value[i])))
                return
        if isinstance(obj, VTuple):
            if isinstance(idx, VInt):
                i = idx.value
                if i < 0 or i >= len(obj.items):
                    self._throw(
                        _make_error_struct("IndexError", "tuple index out of range")
                    )
                    return
                self.stack.append(obj.items[i])
                return
        self.stack.append(_NONE_VAL)

    def _do_store_index(self, obj: Val, idx: Val, val: Val) -> None:
        if isinstance(obj, VList) and isinstance(idx, VInt):
            i = idx.value
            if i < 0:
                i += len(obj.items)
            if 0 <= i < len(obj.items):
                obj.items[i] = val
            return
        if isinstance(obj, VMap):
            i = 0
            while i < len(obj.keys):
                if _val_eq(obj.keys[i], idx):
                    obj.values[i] = val
                    return
                i += 1
            obj.keys.append(idx)
            obj.values.append(val)

    def _do_slice(self, obj: Val, low: Val, high: Val) -> None:
        if isinstance(obj, VList) and isinstance(low, VInt) and isinstance(high, VInt):
            lo = low.value
            hi = high.value
            n = len(obj.items)
            if lo < 0 or hi > n or lo > hi:
                self._throw(_make_error_struct("IndexError", "slice out of range"))
                return
            self.stack.append(VList(list(obj.items[lo:hi])))
            return
        if isinstance(obj, VStr) and isinstance(low, VInt) and isinstance(high, VInt):
            lo = low.value
            hi = high.value
            n = len(obj.value)
            if lo < 0 or hi > n or lo > hi:
                self._throw(_make_error_struct("IndexError", "slice out of range"))
                return
            self.stack.append(VStr(obj.value[lo:hi]))
            return
        if isinstance(obj, VBytes) and isinstance(low, VInt) and isinstance(high, VInt):
            lo = low.value
            hi = high.value
            n = len(obj.value)
            if lo < 0 or hi > n or lo > hi:
                self._throw(_make_error_struct("IndexError", "slice out of range"))
                return
            self.stack.append(VBytes(obj.value[lo:hi]))
            return
        self.stack.append(_NONE_VAL)

    def _do_build_struct(self, frame: Frame, arg: int) -> None:
        # arg is either a struct_defs index or a constants index for error structs
        if arg < len(self.module.struct_defs):
            sd = self.module.struct_defs[arg]
            nfields = len(sd.field_names)
            vals: list[Val] = []
            i = 0
            while i < nfields:
                vals.append(_NONE_VAL)
                i += 1
            i = nfields - 1
            while i >= 0:
                vals[i] = self.stack.pop()
                i -= 1
            self.stack.append(VStruct(sd.name, list(sd.field_names), vals))
        else:
            # Error struct: arg is constant index containing type name
            type_name_val = frame.code.constants[arg]
            if isinstance(type_name_val, VStr):
                message = self.stack.pop()
                self.stack.append(
                    VStruct(
                        type_name_val.value,
                        ["message"],
                        [message],
                    )
                )
            else:
                self.stack.append(_NONE_VAL)

    def _do_get_field(self, frame: Frame, const_idx: int) -> None:
        field_name_val = frame.code.constants[const_idx]
        obj = self.stack.pop()
        if isinstance(obj, VStruct) and isinstance(field_name_val, VStr):
            fname = field_name_val.value
            i = 0
            while i < len(obj.field_names):
                if obj.field_names[i] == fname:
                    self.stack.append(obj.field_values[i])
                    return
                i += 1
        self.stack.append(_NONE_VAL)

    def _do_set_field(self, frame: Frame, const_idx: int) -> None:
        field_name_val = frame.code.constants[const_idx]
        val = self.stack.pop()
        obj = self.stack.pop()
        if isinstance(obj, VStruct) and isinstance(field_name_val, VStr):
            fname = field_name_val.value
            i = 0
            while i < len(obj.field_names):
                if obj.field_names[i] == fname:
                    obj.field_values[i] = val
                    return
                i += 1

    def _do_get_iter(self, argc: int) -> None:
        """Set up range iteration. Stack has range args (1-3)."""
        if argc == 1:
            end = self.stack.pop()
            if isinstance(end, VInt):
                # Push: current, end, step
                self.stack.append(_ZERO_INT)
                self.stack.append(end)
                self.stack.append(_ONE_INT)
            else:
                self.stack.append(_ZERO_INT)
                self.stack.append(_ZERO_INT)
                self.stack.append(_ONE_INT)
        elif argc == 2:
            end = self.stack.pop()
            start = self.stack.pop()
            if isinstance(start, VInt) and isinstance(end, VInt):
                self.stack.append(start)
                self.stack.append(end)
                self.stack.append(_ONE_INT)
            else:
                self.stack.append(_ZERO_INT)
                self.stack.append(_ZERO_INT)
                self.stack.append(_ONE_INT)
        elif argc == 3:
            step = self.stack.pop()
            end = self.stack.pop()
            start = self.stack.pop()
            if (
                isinstance(start, VInt)
                and isinstance(end, VInt)
                and isinstance(step, VInt)
            ):
                self.stack.append(start)
                self.stack.append(end)
                self.stack.append(step)
            else:
                self.stack.append(_ZERO_INT)
                self.stack.append(_ZERO_INT)
                self.stack.append(_ONE_INT)

    def _do_for_iter(self, frame: Frame, jump_offset: int) -> None:
        """Check if range iteration should continue.
        Stack top: step, end, current (from top).
        If in range: push current value, advance current, continue.
        If done: jump to offset."""
        # Peek at the top 3 values
        sp = len(self.stack)
        step = self.stack[sp - 1]
        end = self.stack[sp - 2]
        current = self.stack[sp - 3]
        if (
            isinstance(current, VInt)
            and isinstance(end, VInt)
            and isinstance(step, VInt)
        ):
            if step.value > 0:
                if current.value >= end.value:
                    frame.ip += jump_offset
                    return
            elif step.value < 0:
                if current.value <= end.value:
                    frame.ip += jump_offset
                    return
            else:
                frame.ip += jump_offset
                return
            # Push current value for the loop variable
            self.stack.append(current)
            # Advance current
            self.stack[sp - 3] = VInt(current.value + step.value)
        else:
            # Collection iteration — always pushes 2 values: index/key, then element/value
            # Stack: index_counter, collection (from top)
            idx = self.stack[sp - 1]
            collection = self.stack[sp - 2]
            if isinstance(idx, VInt):
                if isinstance(collection, VList):
                    if idx.value >= len(collection.items):
                        frame.ip += jump_offset
                        return
                    self.stack.append(VInt(idx.value))
                    self.stack.append(collection.items[idx.value])
                    self.stack[sp - 1] = VInt(idx.value + 1)
                elif isinstance(collection, VStr):
                    if idx.value >= len(collection.value):
                        frame.ip += jump_offset
                        return
                    self.stack.append(VInt(idx.value))
                    self.stack.append(VRune(collection.value[idx.value]))
                    self.stack[sp - 1] = VInt(idx.value + 1)
                elif isinstance(collection, VMap):
                    if idx.value >= len(collection.keys):
                        frame.ip += jump_offset
                        return
                    self.stack.append(collection.keys[idx.value])
                    self.stack.append(collection.values[idx.value])
                    self.stack[sp - 1] = VInt(idx.value + 1)
                elif isinstance(collection, VSet):
                    if idx.value >= len(collection.items):
                        frame.ip += jump_offset
                        return
                    self.stack.append(VInt(idx.value))
                    self.stack.append(collection.items[idx.value])
                    self.stack[sp - 1] = VInt(idx.value + 1)
                elif isinstance(collection, VBytes):
                    if idx.value >= len(collection.value):
                        frame.ip += jump_offset
                        return
                    self.stack.append(VInt(idx.value))
                    self.stack.append(VByte(int(collection.value[idx.value])))
                    self.stack[sp - 1] = VInt(idx.value + 1)
                else:
                    frame.ip += jump_offset

    def _do_unpack(self, n: int) -> None:
        val = self.stack.pop()
        if isinstance(val, VTuple):
            # Push elements in order (first at top... actually first first)
            i = 0
            while i < n and i < len(val.items):
                self.stack.append(val.items[i])
                i += 1
            while i < n:
                self.stack.append(_NONE_VAL)
                i += 1
        elif isinstance(val, VList):
            i = 0
            while i < n and i < len(val.items):
                self.stack.append(val.items[i])
                i += 1
            while i < n:
                self.stack.append(_NONE_VAL)
                i += 1
        else:
            i = 0
            while i < n:
                self.stack.append(_NONE_VAL)
                i += 1


# ============================================================
# COMPARISON HELPERS
# ============================================================


def _cmp_result(a: int, b: int, kind: int) -> Val:
    if kind == CMP_LT:
        return _TRUE_VAL if a < b else _FALSE_VAL
    if kind == CMP_LE:
        return _TRUE_VAL if a <= b else _FALSE_VAL
    if kind == CMP_GT:
        return _TRUE_VAL if a > b else _FALSE_VAL
    if kind == CMP_GE:
        return _TRUE_VAL if a >= b else _FALSE_VAL
    return _FALSE_VAL


def _cmp_result_float(a: float, b: float, kind: int) -> Val:
    if kind == CMP_LT:
        return _TRUE_VAL if a < b else _FALSE_VAL
    if kind == CMP_LE:
        return _TRUE_VAL if a <= b else _FALSE_VAL
    if kind == CMP_GT:
        return _TRUE_VAL if a > b else _FALSE_VAL
    if kind == CMP_GE:
        return _TRUE_VAL if a >= b else _FALSE_VAL
    return _FALSE_VAL


def _cmp_result_str(a: str, b: str, kind: int) -> Val:
    if kind == CMP_LT:
        return _TRUE_VAL if a < b else _FALSE_VAL
    if kind == CMP_LE:
        return _TRUE_VAL if a <= b else _FALSE_VAL
    if kind == CMP_GT:
        return _TRUE_VAL if a > b else _FALSE_VAL
    if kind == CMP_GE:
        return _TRUE_VAL if a >= b else _FALSE_VAL
    return _FALSE_VAL


# ============================================================
# PUBLIC API
# ============================================================


def vm_run(
    module: TModule,
    *,
    stdin: bytes = b"",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> VMResult:
    """Compile and run a Taytsh module through the bytecode VM."""
    compiled = compile_module(module)
    vm = VM(compiled)
    vm.stdin_data = stdin
    if args is not None:
        vm.program_args = args
    if env is not None:
        vm.env_vars = env
    return vm.run()
