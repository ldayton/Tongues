"""Java backend: Taytsh AST → Java source code."""

from __future__ import annotations


from .util import (
    Emitter,
    _check_float_expr,
    _check_float_list,
    _check_int_expr,
    _emit_line,
    _emit_output,
    collect_builtin_calls,
    escape_string,
)


def _escape_java_string(value: str) -> str:
    """Escape a string for Java, converting non-Java escape sequences."""
    s = escape_string(value)
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            nc = s[i + 1]
            if nc == "U" and i + 10 <= len(s):
                cp = int(s[i + 2 : i + 10], 16)
                hi = 0xD800 + ((cp - 0x10000) >> 10)
                lo = 0xDC00 + ((cp - 0x10000) & 0x3FF)
                out.append("\\u" + hex(hi)[2:].zfill(4) + "\\u" + hex(lo)[2:].zfill(4))
                i += 10
            elif nc == "v":
                out.append("\\u000b")
                i += 2
            elif nc == "x" and i + 4 <= len(s) and s[i + 2 : i + 4] == "00":
                out.append("\\u0000")
                i += 4
            elif nc == "\\":
                out.append("\\\\")
                i += 2
            else:
                out.append(s[i])
                i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _decimal_sub(a: str, b: str) -> str:
    """Subtract b from a (a >= b, positive decimal strings)."""
    result: list[str] = []
    borrow = 0
    ai = len(a) - 1
    bi = len(b) - 1
    while ai >= 0:
        da = ord(a[ai]) - 48
        db = (ord(b[bi]) - 48) if bi >= 0 else 0
        diff = da - db - borrow
        if diff < 0:
            diff += 10
            borrow = 1
        else:
            borrow = 0
        result.append(chr(48 + diff))
        ai -= 1
        bi -= 1
    result.reverse()
    i = 0
    while i < len(result) - 1 and result[i] == "0":
        i += 1
    return "".join(result[i:])


def _escape_java_char(value: str) -> str:
    """Escape a single character for a Java char literal."""
    if value == "'":
        return "\\'"
    if value == '"':
        return '"'
    s = _escape_java_string(value)
    if s.startswith("\\u") and len(s) > 6:
        return s[:6]
    return s


JAVA_STRICT_INT_BINARY: dict[str, str] = {
    "+": "Math.addExact",
    "-": "Math.subtractExact",
    "*": "Math.multiplyExact",
    "/": "checkedDiv",
    "%": "checkedRem",
    "<<": "checkedShl",
    ">>": "checkedShr",
    ">>>": ">>>",
}

JAVA_STRICT_INT_COMPOUND: dict[str, str] = {
    "+=": "Math.addExact",
    "-=": "Math.subtractExact",
    "*=": "Math.multiplyExact",
}
from ..taytsh.ast import (
    Ann,
    TArg,
    TAssignStmt,
    TBinaryOp,
    TBoolLit,
    TBreakStmt,
    TByteLit,
    TBytesLit,
    TCall,
    TCatch,
    TContinueStmt,
    TEnumDecl,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFieldDecl,
    TFnDecl,
    TFnLit,
    TFloatLit,
    TForStmt,
    TFuncType,
    TIdentType,
    TIfStmt,
    TIndex,
    TIntLit,
    TInterfaceDecl,
    TLetStmt,
    TListLit,
    TListType,
    TMapLit,
    TMapType,
    TMatchStmt,
    TModule,
    TModuleItem,
    TNilLit,
    TOpAssignStmt,
    TOptionalType,
    TParam,
    TPatternEnum,
    TPatternNil,
    TPatternType,
    TPrimitive,
    TRange,
    TReturnStmt,
    TRuneLit,
    TSetLit,
    TSetType,
    TSlice,
    TStmt,
    TStringLit,
    TStructDecl,
    TTernary,
    TThrowStmt,
    TTupleAccess,
    TTupleAssignStmt,
    TTupleLit,
    TTupleType,
    TTryStmt,
    TType,
    TUnaryOp,
    TUnionType,
    TVar,
    TWhileStmt,
)
from ..taytsh.check import (
    BUILTIN_NAMES,
    BUILTIN_STRUCTS,
)

# ============================================================
# JAVA RESERVED WORDS
# ============================================================

_ISTYPE_MAP: dict[str, str] = {
    "dict": "HashMap",
    "list": "ArrayList",
    "set": "HashSet",
    "str": "String",
    "int": "Integer",
    "float": "Double",
    "bool": "Boolean",
    "AssertError": "AssertionError",
    "tuple": "List",
}

_JAVA_RESERVED = frozenset(
    {
        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extends",
        "final",
        "finally",
        "float",
        "for",
        "goto",
        "if",
        "implements",
        "import",
        "instanceof",
        "int",
        "interface",
        "long",
        "native",
        "new",
        "null",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "short",
        "static",
        "strictfp",
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "transient",
        "try",
        "void",
        "volatile",
        "while",
        "yield",
        # Java 9+ keyword
        "_",
        # Common clashes
        "System",
        "String",
        "Object",
        "Math",
        "Integer",
        "Double",
        "Boolean",
        "Character",
        "Arrays",
        "List",
        "Map",
        "Set",
        "HashMap",
        "HashSet",
        "ArrayList",
        "Collections",
        "Collectors",
        "IntStream",
        "Path",
        "Files",
        "Comparator",
        "StandardCharsets",
    }
)


def _safe_name(name: str) -> str:
    if name in _JAVA_RESERVED:
        return name + "_"
    return name


def _restore_name(name: str, annotations: Ann) -> str:
    key = "name.original." + name
    if key in annotations:
        return _safe_name(annotations[key])
    return _safe_name(name)


def _binding_name(expr_str: str) -> str:
    """Make a safe Java identifier for a pattern binding alias."""
    _KEEP = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    out: list[str] = []
    for ch in expr_str:
        if ch in _KEEP:
            out.append(ch)
        elif ch == ".":
            out.append("_")
    return "".join(out) + "__"


def _lower1(s: str) -> str:
    """Lowercase first character."""
    if not s:
        return s
    return s[0].lower() + s[1:]


_EXCEPTION_MAP: dict[str, str] = {
    "Exception": "RuntimeException",
    "BaseException": "RuntimeException",
    "RuntimeError": "RuntimeException",
}


_JAVA_STDLIB_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "RuntimeException",
        "Exception",
        "ArithmeticException",
        "NumberFormatException",
        "IndexOutOfBoundsException",
        "ClassCastException",
        "UnsupportedOperationException",
        "IllegalArgumentException",
        "NullPointerException",
        "IllegalStateException",
        "AssertionError",
    }
)


# ============================================================
# OPERATOR MAPS
# ============================================================

_PRECEDENCE: dict[str, int] = {
    "||": 1,
    "&&": 2,
    "|": 3,
    "^": 4,
    "&": 5,
    "==": 6,
    "!=": 6,
    "<": 7,
    ">": 7,
    "<=": 7,
    ">=": 7,
    "<<": 8,
    ">>": 8,
    ">>>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}
# NOTE: higher number = higher precedence (tighter binding).
# Java order (low→high): || && | ^ & == != < > <= >= << >> >>> + - * / %

_CMP_OPS = frozenset(["==", "!=", "<", ">", "<=", ">="])


def _needs_parens(child_op: str, parent_op: str, is_left: bool) -> bool:
    child_prec = _PRECEDENCE.get(child_op, 0)
    parent_prec = _PRECEDENCE.get(parent_op, 0)
    if child_prec < parent_prec:
        return True
    if not is_left and child_prec == parent_prec:
        return True
    if child_op in _CMP_OPS and parent_op in _CMP_OPS:
        return True
    return False


# ============================================================
# IMPORT SCANNING
# ============================================================

_FS_IMPORTS = frozenset({"ReadFile", "ReadFileBytes", "WriteFile"})
_IO_IMPORTS = frozenset(
    {
        "ReadAll",
        "ReadLine",
        "ReadBytes",
        "ReadBytesN",
        "WriteOut",
        "WriteErr",
        "WritelnErr",
    }
)
_SYSTEM_IMPORTS = frozenset({"Args", "GetEnv", "Exit"})


def _scan_imports(module: TModule) -> dict[str, bool]:
    """Scan module for import requirements."""
    raise NotImplementedError


# ============================================================
# TYPE EMISSION
# ============================================================

# Java needs explicit types. This maps Taytsh types to Java types:
#   int -> long
#   float -> double
#   bool -> boolean
#   str -> String
#   rune -> char
#   byte -> int (unsigned 0-255)
#   bytes -> byte[]
#   list[T] -> List<T> (ArrayList)
#   map[K, V] -> HashMap<K, V>
#   set[T] -> HashSet<T>
#   T? -> T (nullable, boxed if primitive)
#   tuple[A, B] -> Object[]
#   func(A) -> B -> Function<A, B> / IntFunction / etc.


# ============================================================
# STRICT MATH HELPERS
# ============================================================

_STRICT_MATH_HELPERS: dict[str, str] = {
    # Methods that need helper functions emitted in the preamble.
    # Math.addExact etc. are built-in to Java, but shift checks are not.
    "checkedShl": ...,  # TODO
    "checkedShr": ...,  # TODO
    "checkedDiv": ...,  # TODO
    "checkedRem": ...,  # TODO
    "checkedPow": ...,  # TODO
    "strictFmod": ...,  # TODO
    "strictMinF64": ...,  # TODO
    "strictMaxF64": ...,  # TODO
    "strictSortedF64": ...,  # TODO
}


# ============================================================
# EMITTER
# ============================================================


class _JavaEmitter(Emitter):
    def __init__(
        self,
        struct_names: set[str],
        struct_fields: dict[str, list[str]],
        strict_math: bool = False,
        strict_tostring: bool = False,
    ) -> None:
        self.struct_names = struct_names
        self.struct_fields = struct_fields
        self.strict_math = strict_math
        self.strict_tostring = strict_tostring
        self.indent: int = 0
        self.lines: list[str] = []
        self.self_name: str | None = None
        self.var_types: dict[str, TType] = {}
        self.module_let_names: set[str] = set()
        self._needs_read_all: bool = False
        self._module_decls: list[TModuleItem] = []
        self._current_struct: str = ""
        self._struct_method_names: dict[str, set[str]] = {}
        self._struct_field_decls: dict[str, list[TFieldDecl]] = {}
        self._enum_names: set[str] = set()
        self._interface_names: set[str] = set()
        self._error_struct_names: set[str] = set()
        self.fn_names: set[str] = set()
        self._needs_strict_helpers: set[str] = set()
        self._needs_replace_count: bool = False
        self._needs_to_byte_array: bool = False
        self._needs_replace_slice: bool = False
        self._needs_concat_lists: bool = False
        self._needs_concat_bytes: bool = False
        self._needs_union_sets: bool = False
        self._needs_intersect_sets: bool = False
        self._needs_difference_sets: bool = False
        self._needs_merge_maps: bool = False
        self._needs_repeat_list: bool = False
        self._needs_repeat_bytes: bool = False
        self._needs_list_compare: bool = False
        self._needs_zfill: bool = False
        self._needs_bytes_helpers: bool = False
        self._needs_decode_utf8: bool = False
        self._needs_pop_item: bool = False
        self._needs_set_pop: bool = False
        self._needs_hex_helper: bool = False
        self._needs_string_pos_helpers: bool = False
        self._needs_argv: bool = False
        self._needs_throwing_runnable: bool = False
        self._ret_is_void: bool = True
        self._ret_type: str = "void"
        self._wide_vars: set[str] = set()
        self._var_aliases: dict[str, str] = {}
        self._narrowed_types: dict[str, str] = {}
        self._interface_methods: dict[str, list[TFnDecl]] = {}
        self._interface_fields: dict[str, list[TFieldDecl]] = {}
        self._tmp_counter: int = 0
        self._mutable_vars: set[str] = set()

    def _line(self, text: str = "") -> None:
        _emit_line(self.lines, self.indent, text)

    def output(self) -> str:
        return _emit_output(self.lines)

    def _is_int_expr(self, expr: TExpr) -> bool:
        return _check_int_expr(expr, self.var_types)

    def _is_float_expr(self, expr: TExpr) -> bool:
        return _check_float_expr(expr, self.var_types)

    def _is_float_list(self, expr: TExpr) -> bool:
        return _check_float_list(expr, self.var_types)

    # ── Type emission ────────────────────────────────────────

    def _type(self, typ: TType) -> str:
        """Emit a Java type from a Taytsh type node."""
        if isinstance(typ, TPrimitive):
            if typ.kind == "int":
                if self.strict_math:
                    return "long"
                return "int"
            if typ.kind == "float":
                return "double"
            if typ.kind == "bool":
                return "boolean"
            if typ.kind == "string":
                return "String"
            if typ.kind == "rune":
                return "char"
            if typ.kind == "byte":
                return "int"
            if typ.kind == "bytes":
                return "byte[]"
            if typ.kind == "nil" or typ.kind == "void":
                return "void"
        if isinstance(typ, TIdentType):
            mapped = _ISTYPE_MAP.get(typ.name)
            if mapped is not None:
                return mapped
            return typ.name
        if isinstance(typ, TListType):
            return "List<" + self._boxed_type(typ.element) + ">"
        if isinstance(typ, TMapType):
            return (
                "HashMap<"
                + self._boxed_type(typ.key)
                + ", "
                + self._boxed_type(typ.value)
                + ">"
            )
        if isinstance(typ, TSetType):
            return "HashSet<" + self._boxed_type(typ.element) + ">"
        if isinstance(typ, TOptionalType):
            return self._boxed_type(typ.inner)
        if isinstance(typ, TTupleType):
            tup: TTupleType = typ
            elem_type = self._tuple_element_boxed(tup)
            return "List<" + elem_type + ">"
        if isinstance(typ, TFuncType):
            return "Function"
        if isinstance(typ, TUnionType):
            non_nil = [
                m
                for m in typ.members
                if not (isinstance(m, TPrimitive) and m.kind == "nil")
            ]
            if len(non_nil) == 1:
                return self._boxed_type(non_nil[0])
            return "Object"
        return "Object"

    def _boxed_type(self, typ: TType) -> str:
        """Emit boxed Java type (Integer instead of int, etc.)."""
        if isinstance(typ, TPrimitive):
            if typ.kind == "int":
                if self.strict_math:
                    return "Long"
                return "Integer"
            if typ.kind == "float":
                return "Double"
            if typ.kind == "bool":
                return "Boolean"
            if typ.kind == "string":
                return "String"
            if typ.kind == "rune":
                return "Character"
            if typ.kind == "byte":
                return "Integer"
            if typ.kind == "bytes":
                return "byte[]"
            if typ.kind == "nil" or typ.kind == "void":
                return "Object"
        if isinstance(typ, TTupleType):
            tup2: TTupleType = typ
            elem_type2 = self._tuple_element_boxed(tup2)
            return "List<" + elem_type2 + ">"
        return self._type(typ)

    def _tuple_element_boxed(self, typ: TTupleType) -> str:
        """Return boxed element type if all tuple elements share the same type."""
        if len(typ.elements) == 0:
            return "Object"
        non_nil: list[TType] = [
            e
            for e in typ.elements
            if not (isinstance(e, TPrimitive) and e.kind == "nil")
        ]
        if len(non_nil) == 0:
            return "Object"
        first = self._boxed_type(non_nil[0])
        for e in non_nil[1:]:
            if self._boxed_type(e) != first:
                return "Object"
        return first

    def _boxed_from_ann(self, ann: str) -> str:
        """Map a type annotation string to a boxed Java type."""
        m: dict[str, str] = {
            "int": "Integer",
            "float": "Double",
            "bool": "Boolean",
            "string": "String",
            "rune": "Character",
            "byte": "Integer",
        }
        return m.get(ann, "")

    def _parse_map_type_ann(self, ann_str: str) -> tuple[str, str] | None:
        """Parse 'map[K, V]' annotation and return (BoxedK, BoxedV) or None."""
        if not ann_str.startswith("map[") or not ann_str.endswith("]"):
            return None
        inner = ann_str[4:-1]
        depth = 0
        comma_pos = -1
        for i in range(len(inner)):
            c = inner[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            elif c == "," and depth == 0:
                comma_pos = i
                break
        if comma_pos < 0:
            return None
        k = inner[:comma_pos].strip()
        v = inner[comma_pos + 1 :].strip()
        bk = self._boxed_from_ann(k)
        bv = self._boxed_from_ann(v)
        if not bk or not bv:
            return None
        return (bk, bv)

    def _map_lit_coerce(self, expr: TExpr, expected_type: str) -> str:
        """Emit a map literal key or value, coercing int→double when needed."""
        if expected_type == "Double":
            if isinstance(expr, TIntLit):
                ilit: TIntLit = expr
                return str(ilit.value) + ".0"
        return self._expr(expr)

    def _tuple_witness_from_ann(self, ann: str) -> str:
        """Get type witness for a tuple literal, e.g. '<Integer>' or '' (none)."""
        inner = ""
        if ann.startswith("tuple[") and ann.endswith("]"):
            inner = ann[6:-1]
        elif ann.startswith("(") and ann.endswith(")"):
            inner = ann[1:-1]
        else:
            return ""
        if inner.endswith(", ..."):
            inner = inner[:-5]
        parts = self._split_ann_top_level(inner, ", ")
        if not parts or (len(parts) == 1 and parts[0] == ""):
            return ""
        non_nil = [p for p in parts if p != "nil"]
        if len(non_nil) == 0:
            return ""
        first_j = self._java_boxed_from_ann(non_nil[0])
        if first_j is None:
            return ""
        for p in non_nil[1:]:
            pj = self._java_boxed_from_ann(p)
            if pj != first_j:
                return ""
        return "<" + first_j + ">"

    def _expr_type_ann(self, expr: TExpr) -> str:
        """Try to determine the type annotation string for an expression."""
        ann = expr.annotations.get("type", "")
        if ann:
            return ann
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            if typ is not None:
                return self._type_to_ann(typ)
        if isinstance(expr, TFieldAccess):
            owner_ann = self._expr_type_ann(expr.obj)
            if owner_ann and owner_ann in self.struct_names:
                fdecls = self._struct_field_decls.get(owner_ann, [])
                for fd in fdecls:
                    if fd.name == expr.field:
                        return self._type_to_ann(fd.typ)
        if isinstance(expr, TListLit):
            return "list["
        if isinstance(expr, TBytesLit):
            return "bytes"
        return ""

    def _type_to_ann(self, typ: TType) -> str:
        """Convert a TType to an annotation string for cast lookups."""
        if isinstance(typ, TPrimitive):
            return typ.kind
        if isinstance(typ, TIdentType):
            return typ.name
        if isinstance(typ, TListType):
            return "list[" + self._type_to_ann(typ.element) + "]"
        if isinstance(typ, TMapType):
            return (
                "map["
                + self._type_to_ann(typ.key)
                + ", "
                + self._type_to_ann(typ.value)
                + "]"
            )
        if isinstance(typ, TSetType):
            return "set[" + self._type_to_ann(typ.element) + "]"
        return ""

    def _raw_binding_value_cast(self, expr: TExpr) -> str | None:
        """Return cast type for .get() on a raw-typed pattern binding, or None."""
        if not isinstance(expr, TVar):
            return None
        n = _restore_name(expr.name, expr.annotations)
        alias = self._var_aliases.get(n)
        if alias is None:
            return None
        if alias not in self._narrowed_types:
            return None
        typ = self.var_types.get(expr.name)
        if isinstance(typ, TMapType):
            return self._boxed_type(typ.value)
        if isinstance(typ, TListType):
            return self._boxed_type(typ.element)
        return None

    def _is_raw_binding_var(self, expr: TExpr) -> bool:
        """Check if expr is a TVar that resolved to a raw pattern binding."""
        if not isinstance(expr, TVar):
            return False
        n = _restore_name(expr.name, expr.annotations)
        alias = self._var_aliases.get(n)
        return alias is not None and alias in self._narrowed_types

    def _iterable_tuple_types(self, iterable: TExpr) -> list[str]:
        """Get the Java types for each element of a tuple iterable (list[tuple[...]])."""
        typ: TType | None = None
        if isinstance(iterable, TFieldAccess):
            owner_ann = self._expr_type_ann(iterable.obj)
            if owner_ann and owner_ann in self.struct_names:
                fdecls = self._struct_field_decls.get(owner_ann, [])
                for fd in fdecls:
                    if fd.name == iterable.field:
                        typ = fd.typ
                        break
        elif isinstance(iterable, TVar):
            typ = self.var_types.get(iterable.name)
        if typ is not None:
            if not isinstance(typ, TListType):
                return []
            elem = typ.element
            if not isinstance(elem, TTupleType):
                return []
            return [self._type(et) for et in elem.elements]
        ann = iterable.annotations.get("type", "")
        if ann.startswith("list[") and ann.endswith("]"):
            inner = ann[5:-1]
            if inner.startswith("(") and inner.endswith(")"):
                parts = self._split_ann_top_level(inner[1:-1], ", ")
                result: list[str] = []
                for p in parts:
                    j = self._java_boxed_from_ann(p)
                    if j is None:
                        return []
                    result.append(j)
                return result
        return []

    def _tuple_cast_type(self, ann: str) -> str | None:
        """Return the boxed Java cast type for extracting from Object[]."""
        _BOXED_CAST: dict[str, str] = {
            "int": "Integer",
            "float": "Double",
            "bool": "Boolean",
            "string": "String",
            "rune": "Character",
            "byte": "Integer",
            "bytes": "byte[]",
        }
        if ann in _BOXED_CAST:
            return _BOXED_CAST[ann]
        if ann.startswith("list["):
            inner = ann[5:-1]
            inner_j = self._java_boxed_from_ann(inner)
            if inner_j is not None:
                return "ArrayList<" + inner_j + ">"
            return "ArrayList"
        if ann.startswith("tuple[") or (ann.startswith("(") and ann.endswith(")")):
            j = self._java_type_from_ann(ann)
            if j is not None:
                return j
            return "List<Object>"
        if ann.startswith("map["):
            inner = ann[4:-1]
            comma = self._find_top_level_comma(inner)
            if comma >= 0:
                k = inner[:comma].strip()
                v = inner[comma + 1 :].strip()
                kj = self._java_boxed_from_ann(k)
                vj = self._java_boxed_from_ann(v)
                if kj is not None and vj is not None:
                    return "HashMap<" + kj + ", " + vj + ">"
            return "HashMap"
        if ann.startswith("set["):
            inner = ann[4:-1]
            inner_j = self._java_boxed_from_ann(inner)
            if inner_j is not None:
                return "HashSet<" + inner_j + ">"
            return "HashSet"
        result = self._java_type_for_ann(ann)
        if result is not None:
            return result
        return None

    def _java_type_for_ann(self, ann: str) -> str | None:
        """Convert a type annotation string to a Java type for casting."""
        _PRIM: dict[str, str] = {
            "int": "int",
            "float": "double",
            "bool": "boolean",
            "string": "String",
            "rune": "char",
            "byte": "int",
            "bytes": "byte[]",
        }
        if ann in _PRIM:
            return None
        if ann in self.struct_names:
            return _safe_name(ann)
        if ann in self._enum_names:
            return ann
        if ann.startswith("list["):
            return "List<" + self._boxed_ann(ann[5:-1]) + ">"
        if ann.startswith("set["):
            return "HashSet<" + self._boxed_ann(ann[4:-1]) + ">"
        if ann.startswith("map["):
            inner = ann[4:-1]
            comma = inner.find(", ")
            if comma >= 0:
                k = inner[:comma]
                v = inner[comma + 2 :]
                return "HashMap<" + self._boxed_ann(k) + ", " + self._boxed_ann(v) + ">"
        return None

    def _boxed_ann(self, ann: str) -> str:
        """Convert a type annotation to a boxed Java type name."""
        _BOXED: dict[str, str] = {
            "int": "Integer",
            "float": "Double",
            "bool": "Boolean",
            "string": "String",
            "rune": "Character",
            "byte": "Integer",
        }
        if ann in _BOXED:
            return _BOXED[ann]
        if ann in self.struct_names or ann in self._enum_names:
            return _safe_name(ann)
        return "Object"

    def _zero_value(self, typ: TType) -> str:
        if isinstance(typ, TPrimitive):
            if typ.kind in ("int", "byte"):
                return "0"
            if typ.kind == "rune":
                return "'\\0'"
            if typ.kind == "float":
                return "0.0"
            if typ.kind == "bool":
                return "false"
            if typ.kind == "string":
                return '""'
            if typ.kind == "bytes":
                return "new byte[0]"
        return "null"

    def _field_default(self, fld: TFieldDecl, in_body: bool = False) -> str:
        """Return the Java default value for a field with has_default=True."""
        if fld.default_expr is not None and not fld.self_ref:
            return self._expr(fld.default_expr)
        typ = fld.typ
        if isinstance(typ, TListType):
            return "new ArrayList<>()"
        if isinstance(typ, TMapType):
            return "new HashMap<>()"
        if isinstance(typ, TSetType):
            return "new HashSet<>()"
        if (
            fld.has_default
            and isinstance(typ, TIdentType)
            and typ.name in self.struct_names
        ):
            if fld.self_ref:
                return "new " + _safe_name(typ.name) + "(this)" if in_body else "null"
            target_fields = self._struct_field_decls.get(typ.name, [])
            has_noarg = len(target_fields) == 0 or all(
                f.has_default for f in target_fields
            )
            if has_noarg:
                return "new " + _safe_name(typ.name) + "()"
        return self._field_zero(typ)

    def _field_zero(self, typ: TType) -> str:
        """Return the Java zero/default value for a type."""
        if isinstance(typ, TPrimitive):
            if typ.kind in ("int", "byte"):
                return "0"
            if typ.kind == "rune":
                return "'\\0'"
            if typ.kind == "float":
                return "0.0"
            if typ.kind == "bool":
                return "false"
            if typ.kind == "string":
                return '""'
        if isinstance(typ, TIdentType):
            n = typ.name
            if n == "int":
                return "0"
            if n == "float":
                return "0.0"
            if n == "bool":
                return "false"
            if n == "string":
                return '""'
        return "null"

    def _emit_builtin_exception_stubs(self, module: TModule) -> None:
        thrown = _collect_thrown_types(module)
        if _module_uses_builtin(module, "Decode"):
            thrown.add("UnicodeDecodeError")
        defined: set[str] = set()
        for decl in module.decls:
            if isinstance(decl, TStructDecl):
                defined.add(decl.name)
            elif isinstance(decl, TInterfaceDecl):
                defined.add(decl.name)
        mapped = set(_EXCEPTION_MAP.values())
        # Ensure ValueError is emitted before UnicodeDecodeError (its subclass)
        reordered: list[str] = []
        seen: set[str] = set()
        for name in sorted(thrown - defined):
            if name == "UnicodeDecodeError" and "ValueError" not in seen:
                reordered.append("ValueError")
                seen.add("ValueError")
            if name not in seen:
                reordered.append(name)
                seen.add(name)
        for name in reordered:
            java_name = _EXCEPTION_MAP.get(name, name)
            if java_name in mapped and name in _EXCEPTION_MAP:
                continue
            if java_name in _JAVA_STDLIB_EXCEPTIONS:
                continue
            if name in self.fn_names:
                continue
            self._line()
            exc_parent = (
                "ValueError" if name == "UnicodeDecodeError" else "RuntimeException"
            )
            self._line("static class " + java_name + " extends " + exc_parent + " {")
            self.indent += 1
            self._line(java_name + "(String message) { super(message); }")
            self._line(java_name + "() { super(); }")
            self.indent -= 1
            self._line("}")

    def _collect_interface_members(self, module: TModule) -> None:
        """Find methods and fields shared by all implementors of each interface."""
        implementors: dict[str, list[TStructDecl]] = {}
        for decl in module.decls:
            if isinstance(decl, TStructDecl) and decl.parent is not None:
                if decl.parent not in implementors:
                    implementors[decl.parent] = []
                implementors[decl.parent].append(decl)
        for iface_name, structs in implementors.items():
            if not structs:
                continue
            first_methods: dict[str, TFnDecl] = {}
            for m in structs[0].methods:
                first_methods[m.name] = m
            common_methods: set[str] = set(first_methods.keys())
            first_fields: dict[str, TFieldDecl] = {}
            for f in structs[0].fields:
                first_fields[f.name] = f
            common_fields: set[str] = set(first_fields.keys())
            for s in structs[1:]:
                other_methods: set[str] = set()
                for m in s.methods:
                    other_methods.add(m.name)
                common_methods &= other_methods
                other_fields: set[str] = set()
                for f in s.fields:
                    other_fields.add(f.name)
                common_fields &= other_fields
            result_m: list[TFnDecl] = []
            for mname in sorted(common_methods):
                result_m.append(first_methods[mname])
            if result_m:
                self._interface_methods[iface_name] = result_m
            result_f: list[TFieldDecl] = []
            for fname in sorted(common_fields):
                result_f.append(first_fields[fname])
            if result_f:
                self._interface_fields[iface_name] = result_f

    # ── Module ───────────────────────────────────────────────

    def emit_module(self, module: TModule) -> None:
        self._module_decls = module.decls
        for decl in module.decls:
            if isinstance(decl, TLetStmt):
                self.module_let_names.add(decl.name)
            if isinstance(decl, TFnDecl):
                self.fn_names.add(decl.name)
            if isinstance(decl, TStructDecl):
                mset: set[str] = set()
                for m in decl.methods:
                    self.fn_names.add(m.name)
                    mset.add(m.name)
                if decl.parent is not None:
                    self._struct_method_names[decl.name] = (
                        mset | self._struct_method_names.get(decl.parent, set())
                    )
                else:
                    self._struct_method_names[decl.name] = mset
        self._collect_interface_members(module)
        all_builtins: set[str] = set()
        for decl in module.decls:
            if isinstance(decl, TFnDecl):
                all_builtins |= collect_builtin_calls(decl.body)
            elif isinstance(decl, TStructDecl):
                for m in decl.methods:
                    all_builtins |= collect_builtin_calls(m.body)
            elif isinstance(decl, TStmt):
                all_builtins |= collect_builtin_calls([decl])
        self._needs_read_all = "ReadAll" in all_builtins
        for decl in module.decls:
            if isinstance(decl, TLetStmt):
                if decl.annotations.get("intwidth.wide") == "true":
                    self._wide_vars.add(decl.name)
        self._emit_imports(module)
        self._line("public class Main {")
        self.indent += 1
        for decl in module.decls:
            match decl:
                case TEnumDecl():
                    self._line()
                    self._emit_enum(decl)
                case TStructDecl():
                    self._line()
                    self._emit_struct(decl)
                case TInterfaceDecl():
                    self._line()
                    self._emit_interface(decl)
        self._emit_builtin_exception_stubs(module)
        self._line()
        self._line("static class SystemExitException extends RuntimeException {")
        self.indent += 1
        self._line("final int code;")
        self._line("SystemExitException(int code) { this.code = code; }")
        self.indent -= 1
        self._line("}")
        self._line()
        self._line(
            "static int doExit(int code) { throw new SystemExitException(code); }"
        )
        self._line()
        self._line("static int parseIntAuto(String s, int radix) {")
        self.indent += 1
        self._line("return (int) parseLongAuto(s, radix);")
        self.indent -= 1
        self._line("}")
        self._line()
        self._line("static long parseLongAuto(String s, int radix) {")
        self.indent += 1
        self._line("if (radix != 0) {")
        self.indent += 1
        self._line(
            'if (radix == 16 && (s.startsWith("0x") || s.startsWith("0X"))) s = s.substring(2);'
        )
        self._line(
            'if (radix == 2 && (s.startsWith("0b") || s.startsWith("0B"))) s = s.substring(2);'
        )
        self._line(
            'if (radix == 8 && (s.startsWith("0o") || s.startsWith("0O"))) s = s.substring(2);'
        )
        self._line("try { return Long.parseLong(s, radix); }")
        self._line(
            "catch (NumberFormatException e) { return Long.parseUnsignedLong(s, radix); }"
        )
        self.indent -= 1
        self._line("}")
        self._line('if (s.startsWith("0x") || s.startsWith("0X")) {')
        self.indent += 1
        self._line("try { return Long.parseLong(s.substring(2), 16); }")
        self._line(
            "catch (NumberFormatException e) { return Long.parseUnsignedLong(s.substring(2), 16); }"
        )
        self.indent -= 1
        self._line("}")
        self._line('if (s.startsWith("0b") || s.startsWith("0B")) {')
        self.indent += 1
        self._line("try { return Long.parseLong(s.substring(2), 2); }")
        self._line(
            "catch (NumberFormatException e) { return Long.parseUnsignedLong(s.substring(2), 2); }"
        )
        self.indent -= 1
        self._line("}")
        self._line('if (s.startsWith("0o") || s.startsWith("0O")) {')
        self.indent += 1
        self._line("try { return Long.parseLong(s.substring(2), 8); }")
        self._line(
            "catch (NumberFormatException e) { return Long.parseUnsignedLong(s.substring(2), 8); }"
        )
        self.indent -= 1
        self._line("}")
        self._line("return Long.parseLong(s, 10);")
        self.indent -= 1
        self._line("}")
        for decl in module.decls:
            if isinstance(decl, TLetStmt):
                self._line()
                self._emit_module_let(decl)
        for decl in module.decls:
            if isinstance(decl, TFnDecl):
                self._line()
                self._emit_fn(decl)
        top_stmts: list[TStmt] = []
        for decl in module.decls:
            if isinstance(decl, TStmt) and not isinstance(decl, TLetStmt):
                top_stmts.append(decl)
        if top_stmts:
            self._line()
            self._line("public static void main(String[] args) throws Exception {")
            self.indent += 1
            if self._needs_argv:
                self._line("_argv = Arrays.asList(args);")
            self._line("try {")
            self.indent += 1
            if self._needs_read_all:
                self._line(
                    "String input = new String(System.in.readAllBytes(), StandardCharsets.UTF_8);"
                )
            self._emit_stmts(top_stmts)
            self.indent -= 1
            self._line(
                '} catch (SystemExitException e) { if (System.getProperty("tongues.test") == null) System.exit(e.code); else throw e; }'
            )
            self.indent -= 1
            self._line("}")
        if self._needs_argv:
            self._line("static List<String> _argv = new ArrayList<>();")
        if self._needs_replace_slice:
            self._line()
            self._line(
                "static <T> void replaceSlice(List<T> xs, int lo, int hi, List<T> vals) {"
            )
            self.indent += 1
            self._line("xs.subList(lo, hi).clear();")
            self._line("xs.addAll(lo, vals);")
            self.indent -= 1
            self._line("}")
        if self._needs_list_compare:
            self._line()
            self._line('@SuppressWarnings("unchecked")')
            self._line(
                "static <T extends Comparable<T>> int _listCompare(List<T> a, List<T> b) {"
            )
            self.indent += 1
            self._line("int n = Math.min(a.size(), b.size());")
            self._line("for (int i = 0; i < n; i++) {")
            self.indent += 1
            self._line("int c = ((Comparable<T>) a.get(i)).compareTo(b.get(i));")
            self._line("if (c != 0) return c;")
            self.indent -= 1
            self._line("}")
            self._line("return Integer.compare(a.size(), b.size());")
            self.indent -= 1
            self._line("}")
        if self._needs_to_byte_array:
            self._line()
            self._line("static byte[] toByteArray(List<Integer> xs) {")
            self.indent += 1
            self._line("byte[] result = new byte[xs.size()];")
            self._line(
                "for (int i = 0; i < xs.size(); i++) result[i] = xs.get(i).byteValue();"
            )
            self._line("return result;")
            self.indent -= 1
            self._line("}")
        if self._needs_concat_lists:
            self._line()
            self._line("static <T> List<T> _concatLists(List<T> a, List<T> b) {")
            self.indent += 1
            self._line("List<T> r = new ArrayList<>(a);")
            self._line("r.addAll(b);")
            self._line("return r;")
            self.indent -= 1
            self._line("}")
        if self._needs_concat_bytes:
            self._line()
            self._line("static byte[] _concatBytes(byte[] a, byte[] b) {")
            self.indent += 1
            self._line("byte[] r = new byte[a.length + b.length];")
            self._line("System.arraycopy(a, 0, r, 0, a.length);")
            self._line("System.arraycopy(b, 0, r, a.length, b.length);")
            self._line("return r;")
            self.indent -= 1
            self._line("}")
        if self._needs_union_sets:
            self._line()
            self._line("static <T> HashSet<T> _unionSets(HashSet<T> a, HashSet<T> b) {")
            self.indent += 1
            self._line("HashSet<T> r = new HashSet<>(a);")
            self._line("r.addAll(b);")
            self._line("return r;")
            self.indent -= 1
            self._line("}")
        if self._needs_intersect_sets:
            self._line()
            self._line(
                "static <T> HashSet<T> _intersectSets(HashSet<T> a, HashSet<T> b) {"
            )
            self.indent += 1
            self._line("HashSet<T> r = new HashSet<>(a);")
            self._line("r.retainAll(b);")
            self._line("return r;")
            self.indent -= 1
            self._line("}")
        if self._needs_difference_sets:
            self._line()
            self._line(
                "static <T> HashSet<T> _differenceSets(HashSet<T> a, HashSet<T> b) {"
            )
            self.indent += 1
            self._line("HashSet<T> r = new HashSet<>(a);")
            self._line("r.removeAll(b);")
            self._line("return r;")
            self.indent -= 1
            self._line("}")
        if self._needs_merge_maps:
            self._line()
            self._line(
                "static <K, V> HashMap<K, V> _mergeMaps(HashMap<K, V> a, HashMap<K, V> b) {"
            )
            self.indent += 1
            self._line("HashMap<K, V> r = new HashMap<>(a);")
            self._line("r.putAll(b);")
            self._line("return r;")
            self.indent -= 1
            self._line("}")
        if self._needs_repeat_list:
            self._line()
            self._line("static <T> List<T> _repeatList(List<T> xs, int n) {")
            self.indent += 1
            self._line("List<T> r = new ArrayList<>();")
            self._line("for (int i = 0; i < n; i++) r.addAll(xs);")
            self._line("return r;")
            self.indent -= 1
            self._line("}")
        if self._needs_repeat_bytes:
            self._line()
            self._line("static byte[] _repeatBytes(byte[] xs, int n) {")
            self.indent += 1
            self._line("byte[] r = new byte[xs.length * n];")
            self._line(
                "for (int i = 0; i < n; i++) System.arraycopy(xs, 0, r, i * xs.length, xs.length);"
            )
            self._line("return r;")
            self.indent -= 1
            self._line("}")
        if self._needs_zfill:
            self._line()
            self._line("static String _zfill(String s, int width) {")
            self.indent += 1
            self._line("if (s.length() >= width) return s;")
            self._line('return "0".repeat(width - s.length()) + s;')
            self.indent -= 1
            self._line("}")
        if self._needs_replace_count:
            self._line()
            self._line(
                "static String replaceCount(String s, String old, String rep, int count) {"
            )
            self.indent += 1
            self._line("StringBuilder sb = new StringBuilder();")
            self._line("int start = 0;")
            self._line("int n = 0;")
            self._line("while (n < count) {")
            self.indent += 1
            self._line("int idx = s.indexOf(old, start);")
            self._line("if (idx < 0) break;")
            self._line("sb.append(s, start, idx);")
            self._line("sb.append(rep);")
            self._line("start = idx + old.length();")
            self._line("n++;")
            self.indent -= 1
            self._line("}")
            self._line("sb.append(s.substring(start));")
            self._line("return sb.toString();")
            self.indent -= 1
            self._line("}")
        if self._needs_bytes_helpers:
            self._emit_bytes_helpers()
        if self._needs_string_pos_helpers:
            self._line()
            self._line(
                "static int _findInRange(String s, String sub, int start, int end) {"
            )
            self.indent += 1
            self._line("int i = s.substring(start, end).indexOf(sub);")
            self._line("return i == -1 ? -1 : i + start;")
            self.indent -= 1
            self._line("}")
            self._line()
            self._line("static int _rfindFrom(String s, String sub, int start) {")
            self.indent += 1
            self._line("int i = s.substring(start).lastIndexOf(sub);")
            self._line("return i == -1 ? -1 : i + start;")
            self.indent -= 1
            self._line("}")
            self._line()
            self._line(
                "static int _rfindInRange(String s, String sub, int start, int end) {"
            )
            self.indent += 1
            self._line("int i = s.substring(start, end).lastIndexOf(sub);")
            self._line("return i == -1 ? -1 : i + start;")
            self.indent -= 1
            self._line("}")
        if self._needs_hex_helper:
            self._line()
            self._line("static String _bytesHex(byte[] data) {")
            self.indent += 1
            self._line("StringBuilder sb = new StringBuilder();")
            self._line(
                'for (byte b : data) sb.append(String.format("%02x", b & 0xFF));'
            )
            self._line("return sb.toString();")
            self.indent -= 1
            self._line("}")
        if self._needs_pop_item:
            self._line()
            self._line('@SuppressWarnings("unchecked")')
            self._line("static <K, V> List<Object> _popItem(HashMap<K, V> m) {")
            self.indent += 1
            self._line("var it = m.entrySet().iterator();")
            self._line("Map.Entry<K, V> last = null;")
            self._line("while (it.hasNext()) last = it.next();")
            self._line("m.remove(last.getKey());")
            self._line("return Arrays.asList(last.getKey(), last.getValue());")
            self.indent -= 1
            self._line("}")
        if self._needs_set_pop:
            self._line()
            self._line("static <T> T _setPop(HashSet<T> s) {")
            self.indent += 1
            self._line("var it = s.iterator();")
            self._line("T val = it.next();")
            self._line("it.remove();")
            self._line("return val;")
            self.indent -= 1
            self._line("}")
        if self._needs_throwing_runnable:
            self._line()
            self._line("@FunctionalInterface")
            self._line("interface ThrowingRunnable {")
            self.indent += 1
            self._line("void run() throws Exception;")
            self.indent -= 1
            self._line("}")
        if self._needs_decode_utf8:
            self._line()
            self._line("static String _decodeUtf8(byte[] b) {")
            self.indent += 1
            self._line("try {")
            self.indent += 1
            self._line("return java.nio.charset.StandardCharsets.UTF_8.newDecoder()")
            self._line(
                "    .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)"
            )
            self._line(
                "    .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)"
            )
            self._line("    .decode(java.nio.ByteBuffer.wrap(b)).toString();")
            self.indent -= 1
            self._line("} catch (java.nio.charset.CharacterCodingException e) {")
            self.indent += 1
            self._line("throw new UnicodeDecodeError(e.getMessage());")
            self.indent -= 1
            self._line("}")
            self.indent -= 1
            self._line("}")
        self.indent -= 1
        self._line("}")

    def _emit_bytes_helpers(self) -> None:
        helpers = [
            "static int _bytesIndexOf(byte[] data, byte[] pat) {",
            "    for (int i = 0; i <= data.length - pat.length; i++) {",
            "        boolean m = true;",
            "        for (int j = 0; j < pat.length; j++) { if (data[i+j] != pat[j]) { m = false; break; } }",
            "        if (m) return i;",
            "    }",
            "    return -1;",
            "}",
            "static int _bytesLastIndexOf(byte[] data, byte[] pat) {",
            "    for (int i = data.length - pat.length; i >= 0; i--) {",
            "        boolean m = true;",
            "        for (int j = 0; j < pat.length; j++) { if (data[i+j] != pat[j]) { m = false; break; } }",
            "        if (m) return i;",
            "    }",
            "    return -1;",
            "}",
            "static byte[] _bytesReplace(byte[] data, byte[] old, byte[] rep) {",
            "    java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();",
            "    int i = 0;",
            "    while (i <= data.length - old.length) {",
            "        int idx = _bytesIndexOf(Arrays.copyOfRange(data, i, data.length), old);",
            "        if (idx < 0) break;",
            "        out.write(data, i, idx);",
            "        out.write(rep, 0, rep.length);",
            "        i += idx + old.length;",
            "    }",
            "    out.write(data, i, data.length - i);",
            "    return out.toByteArray();",
            "}",
            "static List<byte[]> _bytesSplit(byte[] data, byte[] sep) {",
            "    List<byte[]> result = new ArrayList<>();",
            "    int start = 0;",
            "    while (start <= data.length) {",
            "        int idx = _bytesIndexOf(Arrays.copyOfRange(data, start, data.length), sep);",
            "        if (idx < 0) { result.add(Arrays.copyOfRange(data, start, data.length)); break; }",
            "        result.add(Arrays.copyOfRange(data, start, start + idx));",
            "        start += idx + sep.length;",
            "    }",
            "    return result;",
            "}",
            "static List<byte[]> _bytesSplitN(byte[] data, byte[] sep, int n) {",
            "    List<byte[]> result = new ArrayList<>();",
            "    int start = 0;",
            "    while (start <= data.length && result.size() < n - 1) {",
            "        int idx = _bytesIndexOf(Arrays.copyOfRange(data, start, data.length), sep);",
            "        if (idx < 0) break;",
            "        result.add(Arrays.copyOfRange(data, start, start + idx));",
            "        start += idx + sep.length;",
            "    }",
            "    result.add(Arrays.copyOfRange(data, start, data.length));",
            "    return result;",
            "}",
            "static byte[] _bytesJoin(byte[] sep, List<byte[]> parts) {",
            "    java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();",
            "    for (int i = 0; i < parts.size(); i++) {",
            "        if (i > 0) out.write(sep, 0, sep.length);",
            "        out.write(parts.get(i), 0, parts.get(i).length);",
            "    }",
            "    return out.toByteArray();",
            "}",
            "static byte[] _bytesUpper(byte[] data) {",
            "    byte[] r = Arrays.copyOf(data, data.length);",
            "    for (int i = 0; i < r.length; i++) if (r[i] >= 97 && r[i] <= 122) r[i] -= 32;",
            "    return r;",
            "}",
            "static byte[] _bytesLower(byte[] data) {",
            "    byte[] r = Arrays.copyOf(data, data.length);",
            "    for (int i = 0; i < r.length; i++) if (r[i] >= 65 && r[i] <= 90) r[i] += 32;",
            "    return r;",
            "}",
            "static boolean _bytesStartsWith(byte[] data, byte[] prefix) {",
            "    if (prefix.length > data.length) return false;",
            "    return Arrays.equals(data, 0, prefix.length, prefix, 0, prefix.length);",
            "}",
            "static boolean _bytesEndsWith(byte[] data, byte[] suffix) {",
            "    if (suffix.length > data.length) return false;",
            "    return Arrays.equals(data, data.length - suffix.length, data.length, suffix, 0, suffix.length);",
            "}",
            "static boolean _bytesContains(byte[] data, byte[] pat) {",
            "    return _bytesIndexOf(data, pat) >= 0;",
            "}",
            "static int _bytesCount(byte[] data, byte[] pat) {",
            "    int count = 0; int start = 0;",
            "    while (start <= data.length - pat.length) {",
            "        int idx = _bytesIndexOf(Arrays.copyOfRange(data, start, data.length), pat);",
            "        if (idx < 0) break;",
            "        count++; start += idx + pat.length;",
            "    }",
            "    return count;",
            "}",
            "static byte[] _bytesTrim(byte[] data) {",
            "    int s = 0, e = data.length;",
            "    while (s < e && (data[s] & 0xFF) <= 32) s++;",
            "    while (e > s && (data[e-1] & 0xFF) <= 32) e--;",
            "    return Arrays.copyOfRange(data, s, e);",
            "}",
            "static byte[] _bytesTrimStart(byte[] data) {",
            "    int s = 0;",
            "    while (s < data.length && (data[s] & 0xFF) <= 32) s++;",
            "    return Arrays.copyOfRange(data, s, data.length);",
            "}",
            "static byte[] _bytesTrimEnd(byte[] data) {",
            "    int e = data.length;",
            "    while (e > 0 && (data[e-1] & 0xFF) <= 32) e--;",
            "    return Arrays.copyOfRange(data, 0, e);",
            "}",
        ]
        for line in helpers:
            self._line()
            self._line(line)

    # ── Imports ──────────────────────────────────────────────

    def _emit_imports(self, module: TModule) -> None:
        self._line("import java.util.*;")
        self._line("import java.util.stream.*;")
        self._line("import java.util.function.*;")
        self._line("import java.util.regex.Pattern;")
        self._line("import java.io.*;")
        self._line("import java.nio.file.*;")
        self._line("import java.nio.charset.*;")
        self._line()

    # ── Declarations ─────────────────────────────────────────

    def _emit_enum(self, decl: TEnumDecl) -> None:
        name = _safe_name(decl.name)
        variants = ", ".join(decl.variants)
        self._line("enum " + name + " { " + variants + " }")

    def _emit_struct(self, decl: TStructDecl) -> None:
        name = _safe_name(decl.name)
        parent = decl.parent
        if parent == "Error" or decl.name in self._error_struct_names:
            self._emit_error_struct(decl)
            return
        extends = ""
        # Collect parent field names to avoid re-declaring inherited fields
        parent_field_names: set[str] = set()
        if parent is not None:
            extends = " extends " + _safe_name(parent)
            # Check _struct_field_decls for parent's declared fields
            parent_fields = self._struct_field_decls.get(parent, [])
            for pf in parent_fields:
                parent_field_names.add(pf.name)
            # Also check _interface_fields for common fields hoisted to parent
            interface_fields = self._interface_fields.get(parent, [])
            for pf in interface_fields:
                parent_field_names.add(pf.name)
        self._line("static class " + name + extends + " {")
        self.indent += 1
        # Only declare fields that are NOT inherited from parent
        for f in decl.fields:
            if f.name not in parent_field_names:
                self._line(
                    "public " + self._type(f.typ) + " " + _safe_name(f.name) + ";"
                )
        if decl.fields:
            param_fields = [f for f in decl.fields if not f.body_computed]
            body_fields = [f for f in decl.fields if f.body_computed]
            old_self = self.self_name
            self.self_name = "this"
            params = ", ".join(
                self._type(f.typ) + " " + _safe_name(f.name) for f in param_fields
            )
            self._line("public " + name + "(" + params + ") {")
            self.indent += 1
            for f in param_fields:
                safe = _safe_name(f.name)
                self._line("this." + safe + " = " + safe + ";")
            for f in body_fields:
                safe = _safe_name(f.name)
                if f.default_expr is not None:
                    self._line(
                        "this." + safe + " = " + self._expr(f.default_expr) + ";"
                    )
            self.indent -= 1
            self._line("}")
            required = [f for f in param_fields if not f.has_default]
            defaulted = [f for f in param_fields if f.has_default]
            if defaulted:
                for i in range(len(defaulted)):
                    used = required + defaulted[:i]
                    used_names: set[str] = set()
                    for f in used:
                        used_names.add(f.name)
                    if used:
                        rparams = ", ".join(
                            self._type(f.typ) + " " + _safe_name(f.name) for f in used
                        )
                        self._line("public " + name + "(" + rparams + ") {")
                    else:
                        self._line("public " + name + "() {")
                    self.indent += 1
                    for f in param_fields:
                        safe = _safe_name(f.name)
                        if f.name in used_names:
                            self._line("this." + safe + " = " + safe + ";")
                        else:
                            self._line(
                                "this."
                                + safe
                                + " = "
                                + self._field_default(f, in_body=True)
                                + ";"
                            )
                    for f in body_fields:
                        safe = _safe_name(f.name)
                        if f.default_expr is not None:
                            self._line(
                                "this."
                                + safe
                                + " = "
                                + self._expr(f.default_expr)
                                + ";"
                            )
                    self.indent -= 1
                    self._line("}")
            self.self_name = old_self
        if decl.fields:
            self._line()
            self._emit_struct_equals(name, decl.fields)
            self._line()
            self._emit_struct_hashCode(name, decl.fields)
        old_struct = self._current_struct
        self._current_struct = decl.name
        for m in decl.methods:
            self._line()
            self._emit_method(m)
        self._current_struct = old_struct
        self.indent -= 1
        self._line("}")

    def _emit_struct_equals(self, name: str, fields: list[TFieldDecl]) -> None:
        self._line("@Override")
        self._line("public boolean equals(Object _o) {")
        self.indent += 1
        self._line("if (this == _o) return true;")
        self._line("if (!(_o instanceof " + name + ")) return false;")
        self._line(name + " _that = (" + name + ") _o;")
        for f in fields:
            safe = _safe_name(f.name)
            jtype = self._type(f.typ)
            if jtype in ("int", "long", "double", "boolean", "byte", "char"):
                self._line("if (this." + safe + " != _that." + safe + ") return false;")
            else:
                self._line(
                    "if (!Objects.equals(this."
                    + safe
                    + ", _that."
                    + safe
                    + ")) return false;"
                )
        self._line("return true;")
        self.indent -= 1
        self._line("}")

    def _emit_struct_hashCode(self, name: str, fields: list[TFieldDecl]) -> None:
        self._line("@Override")
        self._line("public int hashCode() {")
        self.indent += 1
        self._line(
            "return Objects.hash("
            + ", ".join(_safe_name(f.name) for f in fields)
            + ");"
        )
        self.indent -= 1
        self._line("}")

    def _emit_error_struct(self, decl: TStructDecl) -> None:
        name = _safe_name(decl.name)
        self._line("static class " + name + " extends Exception {")
        self.indent += 1
        param_fields = [f for f in decl.fields if not f.body_computed]
        body_fields = [f for f in decl.fields if f.body_computed]
        has_msg = any(f.name == "message" for f in param_fields)
        extra_fields = [f for f in param_fields if f.name != "message"]
        for f in extra_fields:
            self._line("public " + self._type(f.typ) + " " + _safe_name(f.name) + ";")
        for f in body_fields:
            self._line("public " + self._type(f.typ) + " " + _safe_name(f.name) + ";")
        params = ", ".join(
            self._type(f.typ) + " " + _safe_name(f.name) for f in param_fields
        )
        self._line("public " + name + "(" + params + ") {")
        self.indent += 1
        if has_msg:
            self._line("super(message);")
        for f in extra_fields:
            safe = _safe_name(f.name)
            self._line("this." + safe + " = " + safe + ";")
        old_self = self.self_name
        self.self_name = "this"
        for f in body_fields:
            safe = _safe_name(f.name)
            if f.default_expr is not None:
                self._line("this." + safe + " = " + self._expr(f.default_expr) + ";")
        self.self_name = old_self
        self.indent -= 1
        self._line("}")
        self.indent -= 1
        self._line("}")

    def _emit_interface(self, decl: TInterfaceDecl) -> None:
        name = _safe_name(decl.name)
        parent = decl.annotations.get("_parent_interface", "")
        extends = ""
        # Collect parent field names to avoid re-declaring inherited fields
        parent_field_names: set[str] = set()
        if parent:
            extends = " extends " + _safe_name(parent)
            # Get fields from parent interface
            if parent in self._interface_fields:
                for pf in self._interface_fields[parent]:
                    parent_field_names.add(pf.name)
            # Also check struct_field_decls for parent fields
            parent_decl_fields = self._struct_field_decls.get(parent, [])
            for pf in parent_decl_fields:
                parent_field_names.add(pf.name)
        self._line("static class " + name + extends + " {")
        self.indent += 1
        declared_fields: set[str] = set()
        # Only declare fields NOT inherited from parent
        for f in decl.fields:
            if f.name not in parent_field_names:
                self._line(
                    "public " + self._type(f.typ) + " " + _safe_name(f.name) + ";"
                )
            declared_fields.add(f.name)
        for f in self._interface_fields.get(decl.name, []):
            if f.name not in declared_fields and f.name not in parent_field_names:
                self._line(
                    "public " + self._type(f.typ) + " " + _safe_name(f.name) + ";"
                )
        parent_methods: set[str] = set()
        if parent and parent in self._interface_methods:
            for pm in self._interface_methods[parent]:
                parent_methods.add(pm.name)
        methods = self._interface_methods.get(decl.name, [])
        for m in methods:
            if m.name in parent_methods:
                continue
            ret = "void"
            if m.ret is not None:
                ret = self._type(m.ret)
            params = m.params[1:]
            param_str = ", ".join(
                self._type(p.typ) + " " + _safe_name(p.name)
                for p in params
                if p.typ is not None
            )
            self._line(
                ret
                + " "
                + _safe_name(m.name)
                + "("
                + param_str
                + ") throws Exception { throw new RuntimeException(); }"
            )
        self.indent -= 1
        self._line("}")

    def _emit_fn(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        old_ret = self._ret_is_void
        old_ret_type = self._ret_type
        self._tmp_counter = 0
        self._mutable_vars = set()
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
        if decl.name == "Main":
            self._ret_is_void = True
            self._line("public static void main(String[] args) throws Exception {")
            self.indent += 1
            if self._needs_argv:
                self._line("_argv = Arrays.asList(args);")
            self._line("try {")
        else:
            ret = "void"
            if decl.ret is not None:
                ret = self._type(decl.ret)
            if decl.annotations.get("intwidth.wide_return") == "true" and ret == "int":
                ret = "long"
            self._ret_is_void = ret == "void"
            self._ret_type = ret
            params = self._params(decl.params)
            fname = _lower1(_safe_name(decl.name))
            self._line(
                "public static "
                + ret
                + " "
                + fname
                + "("
                + params
                + ") throws Exception {"
            )
        self.indent += 1
        if decl.name == "Main":
            if self._needs_read_all:
                self._line(
                    "String input = new String(System.in.readAllBytes(), StandardCharsets.UTF_8);"
                )
            self._emit_stmts(decl.body)
            self.indent -= 1
            self._line(
                '} catch (SystemExitException e) { if (System.getProperty("tongues.test") == null) System.exit(e.code); else throw e; }'
            )
            self.indent -= 1
            self._line("}")
        else:
            self._emit_stmts(decl.body)
            self.indent -= 1
            self._line("}")
            self._emit_fn_overloads(decl, is_static=True)
        self.var_types = old_var_types
        self._ret_is_void = old_ret
        self._ret_type = old_ret_type

    def _emit_fn_overloads(self, decl: TFnDecl, is_static: bool) -> None:
        """Emit overloaded versions for functions with default parameters."""
        params = decl.params if is_static else decl.params[1:]
        has_defaults = any(p.has_default for p in params)
        if not has_defaults:
            return
        required = [p for p in params if not p.has_default]
        defaulted = [p for p in params if p.has_default]
        ret = "void"
        if decl.ret is not None:
            ret = self._type(decl.ret)
        if decl.annotations.get("intwidth.wide_return") == "true" and ret == "int":
            ret = "long"
        fname = _lower1(_safe_name(decl.name))
        prefix = "public static " if is_static else ""
        for i in range(len(defaulted)):
            used = required + defaulted[:i]
            pstr = self._params(used)
            self._line()
            self._line(prefix + ret + " " + fname + "(" + pstr + ") throws Exception {")
            self.indent += 1
            args: list[str] = []
            for p in used:
                args.append(_restore_name(p.name, p.annotations))
            for p in defaulted[i:]:
                args.append(self._param_zero(p))
            call = fname + "(" + ", ".join(args) + ")"
            if ret == "void":
                self._line(call + ";")
            else:
                self._line("return " + call + ";")
            self.indent -= 1
            self._line("}")

    def _param_zero(self, param: TParam) -> str:
        """Return the Java zero/default value for a parameter."""
        if param.typ is None:
            return "null"
        return self._field_zero(param.typ)

    def _emit_method(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        old_self = self.self_name
        old_ret = self._ret_is_void
        old_ret_type = self._ret_type
        self._tmp_counter = 0
        self._mutable_vars = set()
        self.self_name = decl.params[0].name if decl.params else None
        method_params = decl.params[1:]
        for p in method_params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
        ret = "void"
        if decl.ret is not None:
            ret = self._type(decl.ret)
        if decl.annotations.get("intwidth.wide_return") == "true" and ret == "int":
            ret = "long"
        self._ret_is_void = ret == "void"
        self._ret_type = ret
        params = self._params(method_params)
        fname = _safe_name(decl.name).lower()
        # Python's __repr__ becomes Java's toString
        if fname == "__repr__":
            fname = "toString"
            # toString must be public and cannot throw checked exceptions
            self._line("public " + ret + " " + fname + "(" + params + ") {")
        else:
            self._line(ret + " " + fname + "(" + params + ") throws Exception {")
        self.indent += 1
        self._emit_stmts(decl.body)
        self.indent -= 1
        self._line("}")
        self._emit_fn_overloads(decl, is_static=False)
        self.self_name = old_self
        self.var_types = old_var_types
        self._ret_is_void = old_ret
        self._ret_type = old_ret_type

    def _params(self, params: list[TParam]) -> str:
        parts: list[str] = []
        for p in params:
            if p.typ is None:
                continue
            jtype = self._type(p.typ)
            if p.annotations.get("intwidth.wide") == "true" and jtype == "int":
                jtype = "long"
            name = _restore_name(p.name, p.annotations)
            parts.append(jtype + " " + name)
        return ", ".join(parts)

    # ── Statements ───────────────────────────────────────────

    _TYPE_ANN_TO_JAVA: dict[str, str] = {
        "int": "int",
        "float": "double",
        "bool": "boolean",
        "string": "String",
        "bytes": "byte[]",
        "rune": "char",
    }

    def _java_type_from_ann(self, ann: str) -> str | None:
        """Convert a checker type annotation string to a Java type."""
        if len(ann) == 0:
            return None
        ann = ann.strip()
        if ann == "nil":
            return None
        j = self._TYPE_ANN_TO_JAVA.get(ann)
        if j is not None:
            return j
        if ann in self.struct_names:
            return ann
        if ann.startswith("list[") and ann.endswith("]"):
            inner = ann[5:-1]
            inner_j = self._java_boxed_from_ann(inner)
            if inner_j is not None:
                return "List<" + inner_j + ">"
            return None
        if ann.startswith("map[") and ann.endswith("]"):
            inner = ann[4:-1]
            comma = self._find_top_level_comma(inner)
            if comma < 0:
                return None
            key_ann = inner[:comma].strip()
            val_ann = inner[comma + 1 :].strip()
            key_j = self._java_boxed_from_ann(key_ann)
            val_j = self._java_boxed_from_ann(val_ann)
            if key_j is not None and val_j is not None:
                return "HashMap<" + key_j + ", " + val_j + ">"
            return None
        if ann.startswith("set[") and ann.endswith("]"):
            inner = ann[4:-1]
            inner_j = self._java_boxed_from_ann(inner)
            if inner_j is not None:
                return "HashSet<" + inner_j + ">"
            return None
        if " | " in ann:
            parts = [p.strip() for p in ann.split(" | ")]
            non_nil = [p for p in parts if p != "nil"]
            if len(non_nil) == 1:
                return self._java_type_from_ann(non_nil[0])
            return None
        if ann.startswith("tuple[") and ann.endswith("]"):
            inner = ann[6:-1]
            if inner.endswith(", ..."):
                inner = inner[:-5]
            parts = self._split_ann_top_level(inner, ", ")
            if not parts or (len(parts) == 1 and parts[0] == ""):
                return "List<Object>"
            first_j = self._java_boxed_from_ann(parts[0])
            if first_j is None:
                return "List<Object>"
            all_same = True
            for p in parts[1:]:
                pj = self._java_boxed_from_ann(p)
                if pj != first_j:
                    all_same = False
                    break
            if all_same:
                return "List<" + first_j + ">"
            return "List<Object>"
        if ann.startswith("(") and ann.endswith(")"):
            inner = ann[1:-1]
            if inner.endswith(", ..."):
                inner = inner[:-5]
            parts = self._split_ann_top_level(inner, ", ")
            if not parts or (len(parts) == 1 and parts[0] == ""):
                return "List<Object>"
            first_j = self._java_boxed_from_ann(parts[0])
            if first_j is None:
                return "List<Object>"
            all_same = True
            for p in parts[1:]:
                pj = self._java_boxed_from_ann(p)
                if pj != first_j:
                    all_same = False
                    break
            if all_same:
                return "List<" + first_j + ">"
            return "List<Object>"
        return None

    def _java_boxed_from_ann(self, ann: str) -> str | None:
        """Convert annotation to boxed Java type for generics."""
        _BOXED = {
            "int": "Integer",
            "float": "Double",
            "bool": "Boolean",
            "string": "String",
            "rune": "Character",
            "bytes": "byte[]",
        }
        b = _BOXED.get(ann)
        if b is not None:
            return b
        if " | " in ann:
            parts = self._split_ann_top_level(ann, " | ")
            non_nil = [p for p in parts if p != "nil"]
            if len(non_nil) == 1:
                return self._java_boxed_from_ann(non_nil[0])
            return None
        j = self._java_type_from_ann(ann)
        return j

    def _find_top_level_comma(self, s: str) -> int:
        """Find first comma at bracket depth 0."""
        depth = 0
        for i in range(len(s)):
            c = s[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            elif c == "," and depth == 0:
                return i
        return -1

    def _split_ann_top_level(self, s: str, sep: str) -> list[str]:
        """Split annotation string on sep at bracket/paren depth 0."""
        parts: list[str] = []
        depth = 0
        start = 0
        i = 0
        sep_len = len(sep)
        while i < len(s):
            c = s[i]
            if c in ("[", "("):
                depth += 1
            elif c in ("]", ")"):
                depth -= 1
            elif depth == 0 and s[i : i + sep_len] == sep:
                parts.append(s[start:i].strip())
                i += sep_len
                start = i
                continue
            i += 1
        parts.append(s[start:].strip())
        return parts

    def _emit_stmts(self, stmts: list[TStmt]) -> None:
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            if (
                i + 1 < len(stmts)
                and isinstance(stmt, TLetStmt)
                and isinstance(stmts[i + 1], TForStmt)
            ):
                next_stmt = stmts[i + 1]
                assert isinstance(next_stmt, TForStmt)
                prov = next_stmt.annotations.get("provenance", "")
                if prov in ("any_call", "all_call"):
                    result = self._try_any_all(stmt, next_stmt, prov)
                    if result is not None:
                        lhs, rhs = result
                        folded = self._fold_any_all(stmts, i, stmt.name, rhs)
                        if folded is not None:
                            self._line(folded)
                            i += 3
                            continue
                        self._line(lhs + " = " + rhs + ";")
                        i += 2
                        continue
                if prov in (
                    "list_comprehension",
                    "dict_comprehension",
                    "set_comprehension",
                    "step_slice",
                ):
                    stream = self._try_comprehension_stream(stmt, next_stmt, prov)
                    if stream is not None:
                        self._line(stream)
                        i += 2
                        continue
            guard: tuple[str, str] | None = None
            if isinstance(stmt, TIfStmt) and i + 1 < len(stmts):
                guard = self._detect_isinstance_guard(stmt)
            if guard is None and i + 1 < len(stmts):
                guard = self._detect_assert_isinstance(stmt)
            if guard is not None:
                self._emit_stmt(stmt)
                var_name, type_name = guard
                if self._instanceof_is_redundant(var_name, type_name):
                    self._emit_stmts(stmts[i + 1 :])
                    return
                cast_name = _binding_name(var_name + "_" + type_name)
                self._line(
                    type_name
                    + " "
                    + cast_name
                    + " = ("
                    + type_name
                    + ") "
                    + var_name
                    + ";"
                )
                old_aliases = self._var_aliases.copy()
                old_narrowed = self._narrowed_types.copy()
                self._var_aliases[var_name] = cast_name
                self._narrowed_types[cast_name] = type_name
                self._emit_stmts(stmts[i + 1 :])
                self._var_aliases = old_aliases
                self._narrowed_types = old_narrowed
                return
            self._emit_stmt(stmt)
            i += 1

    def _detect_isinstance_guard(self, stmt: TIfStmt) -> tuple[str, str] | None:
        """Detect `if not isinstance(x, T): <early exit>` guard pattern."""
        if stmt.else_body is not None:
            return None
        if not stmt.then_body:
            return None
        last = stmt.then_body[-1]
        if not isinstance(last, (TReturnStmt, TThrowStmt, TContinueStmt, TBreakStmt)):
            return None
        cond = stmt.cond
        if not isinstance(cond, TUnaryOp):
            return None
        if cond.op not in ("not", "!"):
            return None
        return self._isinstance_info(cond.operand)

    def _detect_assert_isinstance(self, stmt: TStmt) -> tuple[str, str] | None:
        """Detect `assert isinstance(x, T)` pattern."""
        if not isinstance(stmt, TExprStmt):
            return None
        expr = stmt.expr
        if not isinstance(expr, TCall):
            return None
        if not isinstance(expr.func, TVar) or expr.func.name != "Assert":
            return None
        if not expr.args:
            return None
        return self._isinstance_info(expr.args[0].value)

    def _emit_stmt(self, stmt: TStmt) -> None:
        match stmt:
            case TLetStmt():
                self._emit_let(stmt)
            case TAssignStmt():
                if isinstance(stmt.target, TIndex):
                    self._emit_index_assign(stmt.target, stmt.value)
                elif (
                    isinstance(stmt.target, TVar)
                    and self._var_aliases.get(stmt.target.name, stmt.target.name)
                    != stmt.target.name
                ):
                    target = _safe_name(stmt.target.name)
                    self._line(target + " = " + self._expr(stmt.value) + ";")
                else:
                    val = self._expr(stmt.value)
                    if (
                        isinstance(stmt.target, TVar)
                        and stmt.target.name not in self._wide_vars
                        and self._is_int_expr(stmt.target)
                        and self._yields_long(stmt.value)
                    ):
                        val = "(int)(" + val + ")"
                    self._line(self._expr(stmt.target) + " = " + val + ";")
            case TTupleAssignStmt():
                self._emit_tuple_assign(stmt)
            case TOpAssignStmt():
                if stmt.op == "+=" and self._is_list_expr(stmt.target):
                    self._line(
                        self._expr(stmt.target)
                        + ".addAll("
                        + self._expr(stmt.value)
                        + ");"
                    )
                elif stmt.op == "+=" and self._is_bytes_expr(stmt.target):
                    self._needs_concat_bytes = True
                    rhs_val = self._expr(stmt.value)
                    if self._is_string_expr(stmt.value):
                        rhs_val = "(" + rhs_val + ").getBytes(StandardCharsets.UTF_8)"
                    self._line(
                        self._expr(stmt.target)
                        + " = _concatBytes("
                        + self._expr(stmt.target)
                        + ", "
                        + rhs_val
                        + ");"
                    )
                elif (
                    self.strict_math
                    and stmt.op in JAVA_STRICT_INT_COMPOUND
                    and self._is_int_expr(stmt.target)
                ):
                    fn = JAVA_STRICT_INT_COMPOUND[stmt.op]
                    self._line(
                        self._expr(stmt.target)
                        + " = "
                        + fn
                        + "("
                        + self._expr(stmt.target)
                        + ", "
                        + self._expr(stmt.value)
                        + ");"
                    )
                elif isinstance(stmt.target, TIndex):
                    obj = self._expr(stmt.target.obj)
                    idx = self._expr(stmt.target.index)
                    val = self._expr(stmt.value)
                    op_char = stmt.op[0]
                    ann_t = stmt.target.obj.annotations.get("type", "")
                    if ann_t.startswith("map[") or self._is_map_type(stmt.target.obj):
                        self._line(
                            obj
                            + ".put("
                            + idx
                            + ", "
                            + obj
                            + ".get("
                            + idx
                            + ") "
                            + op_char
                            + " "
                            + val
                            + ");"
                        )
                    else:
                        self._line(
                            obj
                            + ".set("
                            + idx
                            + ", "
                            + obj
                            + ".get("
                            + idx
                            + ") "
                            + op_char
                            + " "
                            + val
                            + ");"
                        )
                else:
                    self._line(
                        self._expr(stmt.target)
                        + " "
                        + stmt.op
                        + " "
                        + self._expr(stmt.value)
                        + ";"
                    )
            case TExprStmt():
                self._emit_expr_stmt(stmt)
            case TReturnStmt():
                if stmt.value is not None:
                    prov = stmt.value.annotations.get("provenance", "")
                    if prov in ("partition", "rpartition") and isinstance(
                        stmt.value, TTernary
                    ):
                        self._emit_partition_return_impl(stmt.value, prov)
                    else:
                        val = self._expr(stmt.value)
                        if self._ret_type == "int" and self._yields_long(stmt.value):
                            val = "(int)(" + val + ")"
                        self._line("return " + val + ";")
                else:
                    if self._ret_is_void:
                        self._line("return;")
                    else:
                        self._line("return null;")
            case TThrowStmt():
                throw_val = stmt.expr
                if isinstance(throw_val, TVar):
                    vname = _safe_name(throw_val.name)
                    if vname in self.struct_names or vname in _EXCEPTION_MAP:
                        exc_name = _EXCEPTION_MAP.get(throw_val.name, vname)
                        self._line("throw new " + exc_name + "();")
                    else:
                        self._line("throw " + self._expr(throw_val) + ";")
                elif isinstance(throw_val, TCall):
                    throw_func = throw_val.func
                    if isinstance(throw_func, TVar):
                        fname = throw_func.name
                        exc_name = _EXCEPTION_MAP.get(fname, _safe_name(fname))
                        exc_args = ", ".join(
                            self._expr(a.value) for a in throw_val.args
                        )
                        if fname in self.fn_names and (
                            fname not in self.struct_names
                            or fname not in self._struct_field_decls
                            or len(throw_val.args)
                            != len(self._struct_field_decls[fname])
                        ):
                            self._line("throw " + exc_name + "(" + exc_args + ");")
                        else:
                            self._line("throw new " + exc_name + "(" + exc_args + ");")
                    else:
                        self._line("throw " + self._expr(stmt.expr) + ";")
                else:
                    self._line("throw " + self._expr(stmt.expr) + ";")
            case TBreakStmt():
                self._line("break;")
            case TContinueStmt():
                self._line("continue;")
            case TIfStmt():
                self._emit_if(stmt)
            case TWhileStmt():
                self._emit_while(stmt)
            case TForStmt():
                self._emit_for(stmt)
            case TTryStmt():
                self._emit_try(stmt)
            case TMatchStmt():
                self._emit_match(stmt)

    def _emit_module_let(self, stmt: TLetStmt) -> None:
        safe = _restore_name(stmt.name, stmt.annotations)
        self.var_types[stmt.name] = stmt.typ
        jtype = self._type(stmt.typ)
        if stmt.annotations.get("intwidth.wide") == "true" and jtype == "int":
            jtype = "long"
            self._wide_vars.add(stmt.name)
        is_const = stmt.annotations.get("scope.is_const") == "true"
        qual = "static final " if is_const else "static "
        if stmt.value is not None:
            # Static field initializers can't call exception-throwing functions;
            # final also can't be used with the try/catch static block pattern
            if self._expr_contains_call(stmt.value):
                self._line("static " + jtype + " " + safe + ";")
                self._line("static {")
                self.indent += 1
                self._line("try {")
                self.indent += 1
                self._line(safe + " = " + self._expr(stmt.value) + ";")
                self.indent -= 1
                self._line("} catch (Exception e) { throw new RuntimeException(e); }")
                self.indent -= 1
                self._line("}")
            else:
                self._line(
                    qual + jtype + " " + safe + " = " + self._expr(stmt.value) + ";"
                )
        else:
            self._line(qual + jtype + " " + safe + ";")

    def _emit_let(self, stmt: TLetStmt) -> None:
        safe = _restore_name(stmt.name, stmt.annotations)
        self.var_types[stmt.name] = stmt.typ
        if stmt.annotations.get("scope.is_const") != "true":
            self._mutable_vars.add(stmt.name)
        unused = stmt.annotations.get("liveness.initial_value_unused") == "true"
        jtype = self._type(stmt.typ)
        if jtype == "void":
            jtype = "Object"
        # Fallback: if type is error/Object, try to infer from value annotation
        if jtype == "Object" and stmt.value is not None:
            val_ann = stmt.value.annotations.get("type", "")
            if val_ann == "string":
                jtype = "String"
                self.var_types[stmt.name] = TPrimitive(stmt.pos, "string")
            elif val_ann == "bytes":
                jtype = "byte[]"
                self.var_types[stmt.name] = TPrimitive(stmt.pos, "bytes")
            elif val_ann == "int":
                jtype = "int"
                self.var_types[stmt.name] = TPrimitive(stmt.pos, "int")
            elif val_ann == "float":
                jtype = "double"
                self.var_types[stmt.name] = TPrimitive(stmt.pos, "float")
            elif val_ann == "bool":
                jtype = "boolean"
                self.var_types[stmt.name] = TPrimitive(stmt.pos, "bool")
        if stmt.annotations.get("intwidth.wide") == "true" and jtype == "int":
            jtype = "long"
        if stmt.value is not None and not unused:
            if (
                isinstance(stmt.value, TCall)
                and isinstance(stmt.value.func, TVar)
                and stmt.value.func.name == "Concat"
                and stmt.value.annotations.get("provenance") == "star_unpack"
            ):
                self._emit_star_unpack_let(safe, jtype, stmt.value)
                return
            if (
                isinstance(stmt.value, TCall)
                and isinstance(stmt.value.func, TVar)
                and stmt.value.func.name == "Concat"
                and self._is_bytes_expr(stmt.value.args[0].value)
            ):
                a = self._expr(stmt.value.args[0].value)
                b = self._expr(stmt.value.args[1].value)
                self._line(
                    "byte[] "
                    + safe
                    + " = new byte["
                    + a
                    + ".length + "
                    + b
                    + ".length];"
                )
                self._line(
                    "System.arraycopy(" + a + ", 0, " + safe + ", 0, " + a + ".length);"
                )
                self._line(
                    "System.arraycopy("
                    + b
                    + ", 0, "
                    + safe
                    + ", "
                    + a
                    + ".length, "
                    + b
                    + ".length);"
                )
                return
            val = self._expr(stmt.value)
            if jtype == "int" and self._yields_long(stmt.value):
                val = "(int)(" + val + ")"
            self._line(jtype + " " + safe + " = " + val + ";")
        else:
            self._line(jtype + " " + safe + ";")

    def _emit_tuple_assign(self, stmt: TTupleAssignStmt) -> None:
        unused_str = stmt.annotations.get("liveness.tuple_unused_indices", "")
        unused_indices: set[int] = set()
        if unused_str:
            for s in unused_str.split(","):
                if s:
                    unused_indices.add(int(s))
        if self._is_divmod_call(stmt.value):
            self._emit_divmod_assign(stmt, unused_indices)
            return
        tmp = "__tmp"
        self._tmp_counter += 1
        self._line("{")
        self.indent += 1
        self._line("var " + tmp + " = " + self._expr(stmt.value) + ";")
        for i, t in enumerate(stmt.targets):
            if i in unused_indices:
                continue
            rhs = tmp + ".get(" + str(i) + ")"
            type_ann = t.annotations.get("type", "")
            if type_ann == "" and isinstance(t, TVar):
                vtype = self.var_types.get(t.name)
                if vtype is not None:
                    type_ann = self._type_to_ann(vtype)
            if type_ann:
                cast = self._tuple_cast_type(type_ann)
                if cast is not None:
                    rhs = "(" + cast + ") " + rhs
                elif isinstance(t, TVar):
                    vtype2 = self.var_types.get(t.name)
                    if vtype2 is not None:
                        jtype = self._type(vtype2)
                        if jtype != "Object" and jtype != "var":
                            rhs = "(" + jtype + ") " + rhs
            elif isinstance(t, TVar):
                vtype = self.var_types.get(t.name)
                if vtype is not None:
                    jtype = self._type(vtype)
                    if jtype != "Object" and jtype != "var":
                        rhs = "(" + jtype + ") " + rhs
            self._line(self._expr(t) + " = " + rhs + ";")
        self.indent -= 1
        self._line("}")

    def _emit_expr_stmt(self, stmt: TExprStmt) -> None:
        expr = stmt.expr
        if isinstance(expr, TStringLit):
            return
        if (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Assert"
        ):
            msg = self._a(expr.args, 1) if len(expr.args) > 1 else '"assertion failed"'
            self._line(
                "if (!("
                + self._a(expr.args, 0)
                + ")) { throw new AssertionError("
                + msg
                + "); }"
            )
            return
        if (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "RemoveAt"
            and stmt.annotations.get("provenance") == "del_subscript"
        ):
            self._line(
                self._expr(expr.args[0].value)
                + ".remove("
                + self._expr(expr.args[1].value)
                + ");"
            )
            return
        self._line(self._expr(expr) + ";")

    def _isinstance_info(self, cond: TExpr) -> tuple[str, str] | None:
        """If cond is IsType(var_or_field, Type), return (expr_str, type_name)."""
        if not isinstance(cond, TCall):
            return None
        func = cond.func
        if not isinstance(func, TVar) or func.name != "IsType":
            return None
        if len(cond.args) < 2:
            return None
        var_expr = cond.args[0].value
        if not isinstance(var_expr, (TVar, TFieldAccess)):
            return None
        type_arg = cond.args[1].value
        var_name = self._expr(var_expr)
        if isinstance(type_arg, TStringLit):
            tn = type_arg.value
            type_name = _ISTYPE_MAP.get(tn, tn)
        else:
            type_name = self._expr(type_arg)
        return (var_name, type_name)

    def _isinstance_info_checked(self, cond: TExpr) -> tuple[str, str]:
        """Like _isinstance_info but asserts the result is not None."""
        result = self._isinstance_info(cond)
        assert result is not None
        return result

    def _instanceof_is_redundant(self, var_name: str, type_name_str: str) -> bool:
        """Check if var already has the target type (instanceof would erase generics)."""
        var_type = self.var_types.get(var_name)
        if var_type is None:
            return False
        var_java = self._type(var_type)
        raw = var_java.split("<")[0] if "<" in var_java else var_java
        return raw == type_name_str

    def _collect_isinstance_checks(self, cond: TExpr) -> list[tuple[str, str]]:
        """Collect all IsType checks from a condition (including &&-joined)."""
        if isinstance(cond, TBinaryOp) and cond.op in ("&&", "and"):
            return self._collect_isinstance_checks(
                cond.left
            ) + self._collect_isinstance_checks(cond.right)
        info = self._isinstance_info(cond)
        if info is not None:
            return [info]
        return []

    def _collect_isinstance_raw(self, cond: TExpr) -> list[TExpr]:
        """Collect raw IsType call expressions from a compound && condition."""
        if isinstance(cond, TBinaryOp) and cond.op in ("&&", "and"):
            return self._collect_isinstance_raw(
                cond.left
            ) + self._collect_isinstance_raw(cond.right)
        if self._isinstance_info(cond) is not None:
            return [cond]
        return []

    def _emit_if_with_bindings(
        self, cond_expr: TExpr, cond_str: str, body: list[TStmt]
    ) -> None:
        raw_checks = self._collect_isinstance_raw(cond_expr)
        old_aliases = self._var_aliases.copy()
        old_narrowed = self._narrowed_types.copy()
        if raw_checks:
            parts: list[str] = []
            for check_expr in raw_checks:
                var_name, type_name = self._isinstance_info_checked(check_expr)
                if self._instanceof_is_redundant(var_name, type_name):
                    self._var_aliases[var_name] = var_name
                    continue
                cast_name = _binding_name(var_name + "_" + type_name)
                parts.append(var_name + " instanceof " + type_name + " " + cast_name)
                self._var_aliases[var_name] = cast_name
                self._narrowed_types[cast_name] = type_name
            remaining = self._strip_isinstance_parts(cond_expr)
            if remaining is not None:
                parts.append(self._maybe_paren(remaining, "&&", False))
            if parts:
                self._line("if (" + " && ".join(parts) + ") {")
            else:
                self._line("if (" + cond_str + ") {")
        else:
            self._line("if (" + cond_str + ") {")
        self.indent += 1
        self._emit_stmts(body)
        self.indent -= 1
        self._var_aliases = old_aliases
        self._narrowed_types = old_narrowed

    def _strip_isinstance_parts(self, cond: TExpr) -> TExpr | None:
        """Return the non-isinstance portion of a compound && condition."""
        if isinstance(cond, TBinaryOp) and cond.op in ("&&", "and"):
            left = self._strip_isinstance_parts(cond.left)
            right = self._strip_isinstance_parts(cond.right)
            if left is None and right is None:
                return None
            if left is None:
                return right
            if right is None:
                return left
            return TBinaryOp(cond.pos, cond.annotations, cond.op, left, right)
        if self._isinstance_info(cond) is not None:
            return None
        return cond

    def _emit_if(self, stmt: TIfStmt) -> None:
        prov = stmt.annotations.get("provenance", "")
        truth = self._truthiness_expr(stmt.cond, raised=prov == "truthiness")
        cond = truth if truth is not None else self._expr(stmt.cond)
        self._emit_if_with_bindings(stmt.cond, cond, stmt.then_body)
        neg_info = self._detect_negated_isinstance(stmt.cond)
        if neg_info is not None and stmt.else_body is not None:
            var_name = neg_info[0]
            type_name = neg_info[1]
            cast_name = _binding_name(var_name + "_" + type_name)
            old_aliases = self._var_aliases.copy()
            old_narrowed = self._narrowed_types.copy()
            self._var_aliases[var_name] = cast_name
            self._narrowed_types[cast_name] = type_name
            self._line("} else {")
            self.indent += 1
            self._line(
                type_name + " " + cast_name + " = (" + type_name + ") " + var_name + ";"
            )
            self._emit_stmts(stmt.else_body)
            self.indent -= 1
            self._line("}")
            self._var_aliases = old_aliases
            self._narrowed_types = old_narrowed
        else:
            self._emit_else_body(stmt.else_body)

    def _emit_else_body(self, else_body: list[TStmt] | None) -> None:
        if else_body is None:
            self._line("}")
            return
        if len(else_body) == 1 and isinstance(else_body[0], TIfStmt):
            elif_stmt = else_body[0]
            assert isinstance(elif_stmt, TIfStmt)
            prov = elif_stmt.annotations.get("provenance", "")
            truth = self._truthiness_expr(elif_stmt.cond, raised=prov == "truthiness")
            cond = truth if truth is not None else self._expr(elif_stmt.cond)
            raw_checks = self._collect_isinstance_raw(elif_stmt.cond)
            old_aliases = self._var_aliases.copy()
            old_narrowed = self._narrowed_types.copy()
            if raw_checks:
                parts_list: list[str] = []
                for check_expr in raw_checks:
                    var_name, type_name = self._isinstance_info_checked(check_expr)
                    cast_name = _binding_name(var_name + "_" + type_name)
                    parts_list.append(
                        var_name + " instanceof " + type_name + " " + cast_name
                    )
                    self._var_aliases[var_name] = cast_name
                    self._narrowed_types[cast_name] = type_name
                remaining = self._strip_isinstance_parts(elif_stmt.cond)
                if remaining is not None:
                    parts_list.append(self._maybe_paren(remaining, "&&", False))
                self._line("} else if (" + " && ".join(parts_list) + ") {")
            else:
                self._line("} else if (" + cond + ") {")
            self.indent += 1
            self._emit_stmts(elif_stmt.then_body)
            self.indent -= 1
            self._var_aliases = old_aliases
            self._narrowed_types = old_narrowed
            self._emit_else_body(elif_stmt.else_body)
        else:
            self._line("} else {")
            self.indent += 1
            self._emit_stmts(else_body)
            self.indent -= 1
            self._line("}")

    def _emit_while(self, stmt: TWhileStmt) -> None:
        self._line("while (" + self._expr(stmt.cond) + ") {")
        self.indent += 1
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _emit_for(self, stmt: TForStmt) -> None:
        ann = stmt.annotations
        binding = [_restore_name(b, ann) for b in stmt.binding]
        if isinstance(stmt.iterable, TListLit):
            lit: TListLit = stmt.iterable
            if len(lit.elements) == 0:
                return
        if isinstance(stmt.iterable, TTupleLit):
            tlit: TTupleLit = stmt.iterable
            if len(tlit.elements) == 0:
                return
        if isinstance(stmt.iterable, TRange):
            self._emit_for_range(binding[0], stmt.iterable, stmt.body, ann)
            return
        if self._is_enumerate_for(stmt):
            idx = binding[0]
            val = binding[1]
            iterable = self._expr(stmt.iterable)
            idx_is_reused = ann.get(f"scope.binder.{idx}.is_reuse") == "true"
            idx_decl = idx if idx_is_reused else "int " + idx
            self._line(
                "for ("
                + idx_decl
                + " = 0; "
                + idx
                + " < "
                + iterable
                + ".size(); "
                + idx
                + "++) {"
            )
            self.indent += 1
            self._line("var " + val + " = " + iterable + ".get(" + idx + ");")
            self._emit_stmts(stmt.body)
            self.indent -= 1
            self._line("}")
            return
        if (
            isinstance(stmt.iterable, TCall)
            and isinstance(stmt.iterable.func, TVar)
            and stmt.iterable.func.name == "Reversed"
        ):
            inner_expr = stmt.iterable.args[0].value
            inner = self._expr(inner_expr)
            if self._is_bytes_expr(inner_expr):
                tmp = "__rev_bytes_" + str(self._tmp_counter)
                self._tmp_counter += 1
                self._line("byte[] " + tmp + " = " + inner + ";")
                self._line("for (int __i = " + tmp + ".length - 1; __i >= 0; __i--) {")
                self.indent += 1
                self._line("var " + binding[0] + " = " + tmp + "[__i];")
                self._emit_stmts(stmt.body)
                self.indent -= 1
                self._line("}")
                return
            self._line("for (var " + binding[0] + " : " + inner + ".reversed()) {")
            self.indent += 1
            self._emit_stmts(stmt.body)
            self.indent -= 1
            self._line("}")
            return
        if (
            isinstance(stmt.iterable, TCall)
            and isinstance(stmt.iterable.func, TVar)
            and stmt.iterable.func.name == "Zip"
        ):
            self._emit_for_zip_impl(stmt, binding)
            return
        iterable_expr = self._expr(stmt.iterable)
        iter_type = ann.get("iter_type", "")
        iter_var_type = (
            self.var_types.get(stmt.iterable.name)
            if isinstance(stmt.iterable, TVar)
            else None
        )
        is_string = (
            iter_type == "string"
            or isinstance(stmt.iterable, TStringLit)
            or (
                isinstance(iter_var_type, TPrimitive) and iter_var_type.kind == "string"
            )
            or self._is_string_expr(stmt.iterable)
        )
        is_map = self._is_map_type(stmt.iterable)
        is_items = (
            isinstance(stmt.iterable, TCall)
            and isinstance(stmt.iterable.func, TVar)
            and stmt.iterable.func.name == "Items"
        )
        if is_items and len(binding) == 2 and isinstance(stmt.iterable, TCall):
            items_call: TCall = stmt.iterable
            map_obj = self._expr(items_call.args[0].value)
            entry_var = "__entry" + str(self._tmp_counter)
            self._tmp_counter += 1
            self._line("for (var " + entry_var + " : " + map_obj + ".entrySet()) {")
            self.indent += 1
            self._line("var " + binding[0] + " = " + entry_var + ".getKey();")
            self._line("var " + binding[1] + " = " + entry_var + ".getValue();")
            self._emit_stmts(stmt.body)
            self.indent -= 1
            self._line("}")
            return
        if len(binding) >= 2 and not is_map and not is_string:
            tuple_types = self._iterable_tuple_types(stmt.iterable)
            entry_var = "__entry" + str(self._tmp_counter)
            self._tmp_counter += 1
            self._line("for (var " + entry_var + " : " + iterable_expr + ") {")
            self.indent += 1
            for bi in range(len(binding)):
                bname = binding[bi]
                rhs = entry_var + ".get(" + str(bi) + ")"
                btype = self.var_types.get(bname)
                if btype is not None:
                    jtype = self._type(btype)
                    if jtype != "Object":
                        rhs = "(" + jtype + ") " + rhs
                elif bi < len(tuple_types) and tuple_types[bi] != "Object":
                    rhs = "(" + tuple_types[bi] + ") " + rhs
                self._line("var " + bname + " = " + rhs + ";")
            self._emit_stmts(stmt.body)
            self.indent -= 1
            self._line("}")
            return
        if is_string:
            self._line(
                "for (char " + binding[0] + " : " + iterable_expr + ".toCharArray()) {"
            )
        elif is_map and len(binding) == 2:
            entry_var = "__entry" + str(self._tmp_counter)
            self._tmp_counter += 1
            self._line(
                "for (var " + entry_var + " : " + iterable_expr + ".entrySet()) {"
            )
            self.indent += 1
            self._line("var " + binding[0] + " = " + entry_var + ".getKey();")
            self._line("var " + binding[1] + " = " + entry_var + ".getValue();")
            self._emit_stmts(stmt.body)
            self.indent -= 1
            self._line("}")
            return
        elif is_map and len(binding) == 1:
            self._line(
                "for (var " + binding[0] + " : " + iterable_expr + ".keySet()) {"
            )
        else:
            loop_var = binding[0]
            elem_type = self._for_elem_type(stmt.iterable)
            if loop_var in self.var_types:
                tmp = "__loop_" + str(self._tmp_counter)
                self._tmp_counter += 1
                self._line(
                    "for (" + elem_type + " " + tmp + " : " + iterable_expr + ") {"
                )
                self.indent += 1
                self._line(loop_var + " = " + tmp + ";")
                self._emit_stmts(stmt.body)
                self.indent -= 1
                self._line("}")
                return
            self._line(
                "for (" + elem_type + " " + loop_var + " : " + iterable_expr + ") {"
            )
        self.indent += 1
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _try_any_all(
        self,
        let_stmt: TLetStmt,
        for_stmt: TForStmt,
        prov: str,
    ) -> tuple[str, str] | None:
        """Try to emit any/all as stream expression. Returns (lhs, rhs) or None."""
        # Dict items with 2 bindings: rewrite as entrySet().stream() with entry accessors
        is_items = self._is_items_call(for_stmt.iterable)
        is_map = self._is_map_type(for_stmt.iterable)
        if len(for_stmt.binding) == 2 and (is_items or is_map):
            return self._try_any_all_items(let_stmt, for_stmt, prov)
        if len(for_stmt.binding) > 1:
            return None
        acc = _safe_name(let_stmt.name)
        binder = _safe_name(for_stmt.binding[0])
        iterable = self._expr(for_stmt.iterable)
        func = "anyMatch" if prov == "any_call" else "allMatch"
        body = for_stmt.body
        if len(body) != 1:
            return None
        outer_if = body[0]
        if not isinstance(outer_if, TIfStmt):
            return None
        iter_ann = self._expr_type_ann(for_stmt.iterable)
        if iter_ann == "string":
            return None
        stream_src = iterable + ".stream()"
        mutable_vars = self._mutable_vars
        if (
            len(outer_if.then_body) == 2
            and isinstance(outer_if.then_body[0], TAssignStmt)
            and isinstance(outer_if.then_body[1], TBreakStmt)
        ):
            cond = (
                self._strip_not(outer_if.cond) if prov == "all_call" else outer_if.cond
            )
            # Lambdas can't throw checked exceptions, so fall back to loop form
            if self._expr_contains_call(cond):
                return None
            if self._expr_captures_mutable(cond, binder, mutable_vars):
                return None
            cond_s = self._expr(cond)
            lhs = "boolean " + acc
            rhs = stream_src + "." + func + "(" + binder + " -> " + cond_s + ")"
            return (lhs, rhs)
        if len(outer_if.then_body) == 1 and isinstance(outer_if.then_body[0], TIfStmt):
            inner_if = outer_if.then_body[0]
            if not isinstance(inner_if, TIfStmt):
                return None
            if (
                len(inner_if.then_body) == 2
                and isinstance(inner_if.then_body[0], TAssignStmt)
                and isinstance(inner_if.then_body[1], TBreakStmt)
            ):
                cond = (
                    self._strip_not(inner_if.cond)
                    if prov == "all_call"
                    else inner_if.cond
                )
                # Lambdas can't throw checked exceptions, so fall back to loop form
                if self._expr_contains_call(outer_if.cond) or self._expr_contains_call(
                    cond
                ):
                    return None
                if self._expr_captures_mutable(
                    outer_if.cond, binder, mutable_vars
                ) or self._expr_captures_mutable(cond, binder, mutable_vars):
                    return None
                filter_s = self._expr(outer_if.cond)
                cond_s = self._expr(cond)
                lhs = "boolean " + acc
                rhs = (
                    stream_src
                    + ".filter("
                    + binder
                    + " -> "
                    + filter_s
                    + ")."
                    + func
                    + "("
                    + binder
                    + " -> "
                    + cond_s
                    + ")"
                )
                return (lhs, rhs)
        return None

    def _is_items_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Items"
        )

    def _try_any_all_items(
        self, let_stmt: TLetStmt, for_stmt: TForStmt, prov: str
    ) -> tuple[str, str] | None:
        """Emit any/all over dict items as entrySet().stream() with entry accessors."""
        body = for_stmt.body
        if len(body) != 1:
            return None
        first = body[0]
        if not isinstance(first, TIfStmt):
            return None
        outer_if = first
        if not (
            len(outer_if.then_body) == 2
            and isinstance(outer_if.then_body[0], TAssignStmt)
            and isinstance(outer_if.then_body[1], TBreakStmt)
        ):
            return None
        cond = self._strip_not(outer_if.cond) if prov == "all_call" else outer_if.cond
        if self._expr_contains_call(cond):
            return None
        if (
            isinstance(for_stmt.iterable, TCall)
            and isinstance(for_stmt.iterable.func, TVar)
            and for_stmt.iterable.func.name == "Items"
        ):
            map_obj = self._expr(for_stmt.iterable.args[0].value)
        else:
            map_obj = self._expr(for_stmt.iterable)
        acc = _safe_name(let_stmt.name)
        key_name = _safe_name(for_stmt.binding[0])
        val_name = _safe_name(for_stmt.binding[1])
        entry = "e"
        old_aliases = self._var_aliases.copy()
        self._var_aliases[key_name] = entry + ".getKey()"
        self._var_aliases[val_name] = entry + ".getValue()"
        cond_s = self._expr(cond)
        self._var_aliases = old_aliases
        func = "anyMatch" if prov == "any_call" else "allMatch"
        lhs = "boolean " + acc
        rhs = (
            map_obj
            + ".entrySet().stream()."
            + func
            + "("
            + entry
            + " -> "
            + cond_s
            + ")"
        )
        return (lhs, rhs)

    def _fold_any_all(
        self, stmts: list[TStmt], i: int, temp_name: str, rhs: str
    ) -> str | None:
        """Fold temp variable into the final assignment or return."""
        if i + 2 >= len(stmts):
            return None
        third = stmts[i + 2]
        if isinstance(third, TLetStmt) and isinstance(third.value, TVar):
            if third.value.name == temp_name:
                real = _safe_name(third.name)
                jtype = self._type(third.typ)
                return jtype + " " + real + " = " + rhs + ";"
        if isinstance(third, TReturnStmt) and isinstance(third.value, TVar):
            if third.value.name == temp_name:
                return "return " + rhs + ";"
        if isinstance(third, TAssignStmt) and isinstance(third.value, TVar):
            if third.value.name == temp_name and isinstance(third.target, TVar):
                real = _safe_name(third.target.name)
                return real + " = " + rhs + ";"
        return None

    def _try_comprehension_stream(
        self, let_stmt: TLetStmt, for_stmt: TForStmt, prov: str
    ) -> str | None:
        """Try to emit list/set/dict comprehension as stream expression."""
        acc = _safe_name(let_stmt.name)
        binder = _safe_name(for_stmt.binding[0])
        iterable = self._expr(for_stmt.iterable)
        body = for_stmt.body
        type_str = self._let_type(let_stmt)
        if prov == "list_comprehension":
            append = self._extract_append(body)
            if append is not None:
                map_expr = self._expr(append)
                return (
                    type_str
                    + " "
                    + acc
                    + " = "
                    + iterable
                    + ".stream().map("
                    + binder
                    + " -> "
                    + map_expr
                    + ").collect(Collectors.toList());"
                )
            filtered = self._extract_filtered_append(body)
            if filtered is not None:
                filter_cond, val_expr = filtered
                filter_s = self._expr(filter_cond)
                val_s = self._expr(val_expr)
                if val_s == binder:
                    return (
                        type_str
                        + " "
                        + acc
                        + " = "
                        + iterable
                        + ".stream().filter("
                        + binder
                        + " -> "
                        + filter_s
                        + ").collect(Collectors.toList());"
                    )
                return (
                    type_str
                    + " "
                    + acc
                    + " = "
                    + iterable
                    + ".stream().filter("
                    + binder
                    + " -> "
                    + filter_s
                    + ").map("
                    + binder
                    + " -> "
                    + val_s
                    + ").collect(Collectors.toList());"
                )
        if prov == "set_comprehension":
            append = self._extract_set_add(body)
            if append is not None:
                val_s2 = self._expr(append)
                if val_s2 == binder:
                    return type_str + " " + acc + " = new HashSet<>(" + iterable + ");"
                return (
                    type_str
                    + " "
                    + acc
                    + " = "
                    + iterable
                    + ".stream().map("
                    + binder
                    + " -> "
                    + val_s2
                    + ").collect(Collectors.toCollection(HashSet::new));"
                )
        if prov == "dict_comprehension":
            return self._try_dict_comprehension(let_stmt, for_stmt)
        if prov == "step_slice":
            return self._try_step_slice(let_stmt, for_stmt)
        return None

    def _try_dict_comprehension(
        self, let_stmt: TLetStmt, for_stmt: TForStmt
    ) -> str | None:
        """Emit dict comprehension as IntStream.range().collect(toMap(...))."""
        acc = _safe_name(let_stmt.name)
        type_str = self._let_type(let_stmt)
        body = for_stmt.body
        if len(body) != 1:
            return None
        assign = body[0]
        if not isinstance(assign, TAssignStmt):
            return None
        if not isinstance(assign.target, TIndex):
            return None
        binding = [_safe_name(b) for b in for_stmt.binding]
        if len(binding) != 2:
            return None
        idx_var = binding[0]
        val_var = binding[1]
        iterable = self._expr(for_stmt.iterable)
        key_expr = self._expr(assign.target.index)
        val_expr = self._expr(assign.value)
        key_fn: str = ""
        if key_expr == val_var:
            key_fn = iterable + "::get"
        else:
            key_fn = val_var + " -> " + key_expr
        val_fn = idx_var + " -> " + val_expr
        return (
            type_str
            + " "
            + acc
            + " = IntStream.range(0, "
            + iterable
            + ".size()).boxed().collect(Collectors.toMap("
            + key_fn
            + ", "
            + val_fn
            + ", (a, b) -> b, HashMap::new));"
        )

    def _try_step_slice(self, let_stmt: TLetStmt, for_stmt: TForStmt) -> str | None:
        """Emit step-slice as IntStream expression."""
        if not isinstance(for_stmt.iterable, TRange):
            return None
        rng = for_stmt.iterable
        start_val = self._static_int(rng.args[0])
        step_val_opt = self._static_int(rng.args[2]) if len(rng.args) >= 3 else None
        if step_val_opt is None:
            return None
        step_val: int = step_val_opt
        acc = _safe_name(let_stmt.name)
        type_str = self._let_type(let_stmt)
        size_expr = self._expr(rng.args[1])
        is_string = (
            isinstance(let_stmt.typ, TPrimitive) and let_stmt.typ.kind == "string"
        )
        if start_val is None:
            return None
        if start_val == 0:
            filter_expr = "i -> i % " + str(step_val) + " == 0"
        elif start_val < step_val:
            filter_expr = "i -> i % " + str(step_val) + " == " + str(start_val)
        else:
            filter_expr = (
                "i -> i >= "
                + str(start_val)
                + " && (i - "
                + str(start_val)
                + ") % "
                + str(step_val)
                + " == 0"
            )
        if is_string:
            src = self._find_string_source(for_stmt.body)
            if src is not None:
                return (
                    type_str
                    + " "
                    + acc
                    + " = IntStream.range(0, "
                    + size_expr
                    + ").filter("
                    + filter_expr
                    + ").mapToObj(i -> String.valueOf("
                    + src
                    + ".charAt(i))).collect(Collectors.joining());"
                )
        body = for_stmt.body
        append = self._extract_append(body)
        if append is None:
            return None
        if isinstance(append, TIndex) and isinstance(append.obj, TVar):
            src_name = self._expr(append.obj)
            return (
                type_str
                + " "
                + acc
                + " = IntStream.range(0, "
                + size_expr
                + ").filter("
                + filter_expr
                + ").mapToObj("
                + src_name
                + "::get).collect(Collectors.toList());"
            )
        return None

    def _find_string_source(self, body: list[TStmt]) -> str | None:
        """Find the source string var in: r = Concat(r, ToString(s[__i]))."""
        if len(body) != 1:
            return None
        stmt0 = body[0]
        if not isinstance(stmt0, TAssignStmt):
            return None
        val = stmt0.value
        if not isinstance(val, TCall):
            return None
        val_func = val.func
        if not isinstance(val_func, TVar) or val_func.name != "Concat":
            return None
        to_str = val.args[1].value
        if not isinstance(to_str, TCall):
            return None
        to_str_func = to_str.func
        if not isinstance(to_str_func, TVar) or to_str_func.name != "ToString":
            return None
        idx = to_str.args[0].value
        if isinstance(idx, TIndex) and isinstance(idx.obj, TVar):
            return self._expr(idx.obj)
        return None

    def _extract_append(self, body: list[TStmt]) -> TExpr | None:
        """Extract the value from Append(acc, value) in a single-stmt body."""
        if len(body) != 1:
            return None
        stmt0 = body[0]
        if not isinstance(stmt0, TExprStmt):
            return None
        call = stmt0.expr
        if (
            isinstance(call, TCall)
            and isinstance(call.func, TVar)
            and call.func.name == "Append"
            and len(call.args) == 2
        ):
            return call.args[1].value
        return None

    def _extract_set_add(self, body: list[TStmt]) -> TExpr | None:
        """Extract the value from Add(acc, value) in a single-stmt body."""
        if len(body) != 1:
            return None
        stmt0 = body[0]
        if not isinstance(stmt0, TExprStmt):
            return None
        call = stmt0.expr
        if (
            isinstance(call, TCall)
            and isinstance(call.func, TVar)
            and call.func.name == "Add"
            and len(call.args) == 2
        ):
            return call.args[1].value
        return None

    def _extract_filtered_append(self, body: list[TStmt]) -> tuple[TExpr, TExpr] | None:
        """Extract (filter_cond, value) from: if cond { Append(acc, val) }."""
        if len(body) != 1:
            return None
        if_stmt = body[0]
        if not isinstance(if_stmt, TIfStmt):
            return None
        append = self._extract_append(if_stmt.then_body)
        if append is not None:
            return (if_stmt.cond, append)
        return None

    def _let_type(self, stmt: TLetStmt) -> str:
        if stmt.typ is not None:
            return self._type(stmt.typ)
        return "var"

    def _emit_for_range(
        self,
        var_name: str,
        iterable: TRange,
        body: list[TStmt],
        ann: Ann,
    ) -> None:
        itype = "long" if ann.get("intwidth.wide") == "true" else "int"
        is_reused = ann.get(f"scope.binder.{var_name}.is_reuse") == "true"
        decl = var_name if is_reused else itype + " " + var_name
        nargs = len(iterable.args)
        if nargs == 1:
            high = self._expr(iterable.args[0])
            self._line(
                "for ("
                + decl
                + " = 0; "
                + var_name
                + " < "
                + high
                + "; "
                + var_name
                + "++) {"
            )
        elif nargs == 2:
            low = self._expr(iterable.args[0])
            high = self._expr(iterable.args[1])
            self._line(
                "for ("
                + decl
                + " = "
                + low
                + "; "
                + var_name
                + " < "
                + high
                + "; "
                + var_name
                + "++) {"
            )
        else:
            low = self._expr(iterable.args[0])
            high = self._expr(iterable.args[1])
            step = iterable.args[2]
            step_val = self._static_int(step)
            if step_val is None:
                step_str = self._expr(step)
                self._line(
                    "for ("
                    + decl
                    + " = "
                    + low
                    + "; "
                    + var_name
                    + " < "
                    + high
                    + "; "
                    + var_name
                    + " += "
                    + step_str
                    + ") {"
                )
            elif step_val == -1:
                prov = ann.get("provenance", "")
                if prov == "reversed_range":
                    low_val = self._static_int(iterable.args[0])
                    high_val = self._static_int(iterable.args[1])
                    if low_val is not None and high_val is not None:
                        self._line(
                            "for ("
                            + decl
                            + " = "
                            + str(low_val)
                            + "; "
                            + var_name
                            + " >= "
                            + str(high_val + 1)
                            + "; "
                            + var_name
                            + "--) {"
                        )
                    else:
                        self._line(
                            "for ("
                            + decl
                            + " = "
                            + low
                            + "; "
                            + var_name
                            + " > "
                            + high
                            + "; "
                            + var_name
                            + "--) {"
                        )
                else:
                    self._line(
                        "for ("
                        + decl
                        + " = "
                        + low
                        + "; "
                        + var_name
                        + " > "
                        + high
                        + "; "
                        + var_name
                        + "--) {"
                    )
            elif step_val == 1:
                self._line(
                    "for ("
                    + decl
                    + " = "
                    + low
                    + "; "
                    + var_name
                    + " < "
                    + high
                    + "; "
                    + var_name
                    + "++) {"
                )
            elif step_val < 0:
                step_str = self._expr(step)
                self._line(
                    "for ("
                    + decl
                    + " = "
                    + low
                    + "; "
                    + var_name
                    + " > "
                    + high
                    + "; "
                    + var_name
                    + " += "
                    + step_str
                    + ") {"
                )
            else:
                step_str = self._expr(step)
                self._line(
                    "for ("
                    + decl
                    + " = "
                    + low
                    + "; "
                    + var_name
                    + " < "
                    + high
                    + "; "
                    + var_name
                    + " += "
                    + step_str
                    + ") {"
                )
        self.indent += 1
        self._emit_stmts(body)
        self.indent -= 1
        self._line("}")

    def _emit_try(self, stmt: TTryStmt) -> None:
        self._line("try {")
        self.indent += 1
        self._emit_stmts(stmt.body)
        self.indent -= 1
        for catch in stmt.catches:
            self._emit_catch(catch)
        if stmt.finally_body is not None:
            self._line("} finally {")
            self.indent += 1
            self._emit_stmts(stmt.finally_body)
            self.indent -= 1
        self._line("}")

    def _emit_catch(self, catch: TCatch) -> None:
        unused = catch.annotations.get("liveness.catch_var_unused") == "true"
        name = "_" + _safe_name(catch.name) if unused else _safe_name(catch.name)
        if not catch.types:
            type_str = "Exception"
        elif len(catch.types) == 1:
            type_str = self._type(catch.types[0])
        else:
            type_str = " | ".join(self._type(t) for t in catch.types)
        self._line("} catch (" + type_str + " " + name + ") {")
        self.indent += 1
        self._emit_stmts(catch.body)
        self.indent -= 1

    def _match_needs_object_cast(self, stmt: TMatchStmt, expr_str: str) -> bool:
        """Check if match expression type is incompatible with case types."""
        _COLLECTION_TYPES = {"ArrayList", "HashMap", "HashSet"}
        if not isinstance(stmt.expr, TVar):
            return False
        expr_type = self.var_types.get(stmt.expr.name)
        if expr_type is None:
            return False
        expr_java = self._type(expr_type)
        expr_raw = expr_java.split("<")[0] if "<" in expr_java else expr_java
        if expr_raw == "Object":
            return False
        for case in stmt.cases:
            if isinstance(case.pattern, TPatternType):
                case_type = self._type(case.pattern.type_name)
                case_raw = case_type.split("<")[0] if "<" in case_type else case_type
                if case_raw in _COLLECTION_TYPES and case_raw != expr_raw:
                    return True
                if expr_raw not in self.struct_names and (
                    case_raw in self.struct_names
                ):
                    return True
        return False

    def _emit_match(self, stmt: TMatchStmt) -> None:
        expr_str = self._expr(stmt.expr)
        if self._match_needs_object_cast(stmt, expr_str):
            expr_str = "((Object) " + expr_str + ")"
        if stmt.cases and isinstance(stmt.cases[0].pattern, TPatternEnum):
            self._emit_match_enum(stmt, expr_str)
        elif stmt.cases and isinstance(stmt.cases[0].pattern, TPatternType):
            pat0 = stmt.cases[0].pattern
            assert isinstance(pat0, TPatternType)
            first_type = pat0.type_name
            if (
                isinstance(first_type, TIdentType)
                and first_type.name in self.struct_names
            ):
                if any(isinstance(d, TInterfaceDecl) for d in self._module_decls):
                    self._emit_match_switch(stmt, expr_str)
                    return
            self._emit_match_instanceof(stmt, expr_str)
        else:
            self._emit_match_instanceof(stmt, expr_str)

    def _emit_match_enum(self, stmt: TMatchStmt, expr_str: str) -> None:
        self._line("switch (" + expr_str + ") {")
        self.indent += 1
        for case in stmt.cases:
            pat = case.pattern
            if isinstance(pat, TPatternEnum):
                self._emit_switch_case("case " + pat.variant, case.body)
        if stmt.default:
            self._emit_switch_case("default", stmt.default.body)
        self.indent -= 1
        self._line("}")

    def _emit_switch_case(self, header: str, body: list[TStmt]) -> None:
        if len(body) == 1:
            old_lines = self.lines
            self.lines = []
            self._emit_stmts(body)
            inner = " ".join(l.strip() for l in self.lines)
            self.lines = old_lines
            self._line(header + " -> { " + inner + " }")
        else:
            self._line(header + " -> {")
            self.indent += 1
            self._emit_stmts(body)
            self.indent -= 1
            self._line("}")

    def _emit_match_switch(self, stmt: TMatchStmt, expr_str: str) -> None:
        plain = (
            _safe_name(stmt.expr.name)
            if isinstance(stmt.expr, TVar) and stmt.expr.name != self.self_name
            else None
        )
        narrowed = self._narrowed_types.get(expr_str, "")
        if len(narrowed) == 0 and plain is not None:
            narrowed = self._narrowed_types.get(plain, "")
        if len(narrowed) > 0:
            for case in stmt.cases:
                pat = case.pattern
                if isinstance(pat, TPatternType):
                    type_name = self._type(pat.type_name)
                    if type_name == narrowed:
                        binding = _safe_name(pat.name)
                        old_aliases = self._var_aliases.copy()
                        self._var_aliases[expr_str] = binding
                        if plain is not None and plain != expr_str:
                            self._var_aliases[plain] = binding
                        self._line("var " + binding + " = " + expr_str + ";")
                        self._emit_stmts(case.body)
                        self._var_aliases = old_aliases
                        return
            if stmt.default:
                self._emit_stmts(stmt.default.body)
            return
        self._line("switch (" + expr_str + ") {")
        self.indent += 1
        for case in stmt.cases:
            pat = case.pattern
            if isinstance(pat, TPatternType):
                type_name = self._type(pat.type_name)
                binding = _safe_name(pat.name)
                old_aliases = self._var_aliases.copy()
                old_narrowed = self._narrowed_types.copy()
                self._var_aliases[expr_str] = binding
                self._narrowed_types[binding] = type_name
                if plain is not None and plain != expr_str:
                    self._var_aliases[plain] = binding
                    self._narrowed_types[plain] = type_name
                self._emit_switch_case("case " + type_name + " " + binding, case.body)
                self._var_aliases = old_aliases
                self._narrowed_types = old_narrowed
        if stmt.default:
            self._emit_switch_case("case null, default", stmt.default.body)
        else:
            # Handle null to avoid NPE on null switch values
            self._line("case null, default -> {}")
        self.indent -= 1
        self._line("}")

    def _emit_match_instanceof(self, stmt: TMatchStmt, expr_str: str) -> None:
        plain = (
            _safe_name(stmt.expr.name)
            if isinstance(stmt.expr, TVar) and stmt.expr.name != self.self_name
            else None
        )
        first = True
        for case in stmt.cases:
            pat = case.pattern
            if isinstance(pat, TPatternType):
                type_name = self._boxed_type(pat.type_name)
                binding = _safe_name(pat.name)
                if first:
                    self._line(
                        "if ("
                        + expr_str
                        + " instanceof "
                        + type_name
                        + " "
                        + binding
                        + ") {"
                    )
                else:
                    self._line(
                        "} else if ("
                        + expr_str
                        + " instanceof "
                        + type_name
                        + " "
                        + binding
                        + ") {"
                    )
            elif isinstance(pat, TPatternNil):
                if first:
                    self._line("if (" + expr_str + " == null) {")
                else:
                    self._line("} else if (" + expr_str + " == null) {")
            first = False
            old_aliases = self._var_aliases.copy()
            old_narrowed = self._narrowed_types.copy()
            if isinstance(pat, TPatternType):
                binding_name = _safe_name(pat.name)
                self._var_aliases[expr_str] = binding_name
                self._narrowed_types[binding_name] = self._boxed_type(pat.type_name)
                if plain is not None and plain != expr_str:
                    self._var_aliases[plain] = binding_name
                    self._narrowed_types[plain] = self._narrowed_types[binding_name]
            self.indent += 1
            self._emit_stmts(case.body)
            self.indent -= 1
            self._var_aliases = old_aliases
            self._narrowed_types = old_narrowed
        if stmt.default:
            binding = stmt.default.name
            if first:
                self._line("{")
            else:
                self._line("} else {")
            self.indent += 1
            if binding is not None:
                self._line("var " + _safe_name(binding) + " = " + expr_str + ";")
            self._emit_stmts(stmt.default.body)
            self.indent -= 1
            self._line("}")
        elif not first:
            self._line("}")

    def _emit_partition_return_impl(self, expr: TTernary, prov: str) -> None:
        """Emit partition/rpartition as temp var + ternary return."""
        cond = expr.cond
        if not isinstance(cond, TBinaryOp):
            self._line("return " + self._expr(expr) + ";")
            return
        cond_bin: TBinaryOp = cond
        if cond_bin.op != ">=":
            self._line("return " + self._expr(expr) + ";")
            return
        find_call = cond_bin.left
        if not isinstance(find_call, TCall):
            self._line("return " + self._expr(expr) + ";")
            return
        fc_func = find_call.func
        if not isinstance(fc_func, TVar):
            self._line("return " + self._expr(expr) + ";")
            return
        method = "indexOf" if fc_func.name == "Find" else "lastIndexOf"
        s_arg = self._expr(find_call.args[0].value)
        sep_arg = self._expr(find_call.args[1].value)
        self._line("int __idx = " + s_arg + "." + method + "(" + sep_arg + ");")
        true_str = (
            "Arrays.asList("
            + s_arg
            + ".substring(0, __idx), "
            + sep_arg
            + ", "
            + s_arg
            + ".substring(__idx + "
            + sep_arg
            + ".length()))"
        )
        false_str: str = ""
        if prov == "partition":
            false_str = "Arrays.asList(" + s_arg + ', "", "")'
        else:
            false_str = 'Arrays.asList("", "", ' + s_arg + ")"
        self._line("return __idx >= 0 ? " + true_str + " : " + false_str + ";")

    def _emit_star_unpack_let(self, safe: str, jtype: str, call: TCall) -> None:
        """Emit star-unpack Concat as ArrayList + addAll/add."""
        parts: list[TExpr] = []
        self._flatten_concat(call, parts)
        first = True
        for part in parts:
            if isinstance(part, TListLit) and len(part.elements) == 1:
                elem = self._expr(part.elements[0])
                if first:
                    self._line(jtype + " " + safe + " = new ArrayList<>();")
                    first = False
                self._line(safe + ".add(" + elem + ");")
            else:
                src = self._expr(part)
                if first:
                    self._line(jtype + " " + safe + " = new ArrayList<>(" + src + ");")
                    first = False
                else:
                    self._line(safe + ".addAll(" + src + ");")

    def _flatten_concat(self, expr: TExpr, parts: list[TExpr]) -> None:
        """Flatten nested Concat(Concat(a, b), c) into [a, b, c]."""
        if (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Concat"
            and expr.annotations.get("provenance") == "star_unpack"
        ):
            self._flatten_concat(expr.args[0].value, parts)
            self._flatten_concat(expr.args[1].value, parts)
        else:
            parts.append(expr)

    # ── Comprehension / any / all ────────────────────────────

    def _emit_any_all(
        self,
        let_stmt: TLetStmt,
        for_stmt: TForStmt,
        is_any: bool,
    ) -> None:
        """Emit xs.stream().anyMatch/allMatch (provenance) or loop form."""
        raise NotImplementedError

    # ── Expressions ──────────────────────────────────────────

    def _expr(self, expr: TExpr) -> str:
        if isinstance(expr, TIntLit):
            return self._int_lit(expr)
        if isinstance(expr, TFloatLit):
            return expr.raw
        if isinstance(expr, TStringLit):
            return '"' + _escape_java_string(expr.value) + '"'
        if isinstance(expr, TBoolLit):
            return "true" if expr.value else "false"
        if isinstance(expr, TNilLit):
            return "null"
        if isinstance(expr, TByteLit):
            return expr.raw
        if isinstance(expr, TBytesLit):
            return self._bytes_lit(expr)
        if isinstance(expr, TRuneLit):
            return "'" + _escape_java_char(expr.value) + "'"
        if isinstance(expr, TVar):
            if expr.name == self.self_name:
                return "this"
            n = _restore_name(expr.name, expr.annotations)
            if n in self._var_aliases:
                return self._var_aliases[n]
            if expr.name in self.fn_names:
                return _lower1(n)
            narrowed = expr.annotations.get("scope.narrowed_type", "")
            if narrowed and narrowed in self.struct_names:
                result = "((" + _safe_name(narrowed) + ") " + n + ")"
                if result in self._var_aliases:
                    return self._var_aliases[result]
                return result
            return n
        if isinstance(expr, TFieldAccess):
            obj = self._expr(expr.obj)
            field = _safe_name(expr.field)
            obj_ann = self._expr_type_ann(expr.obj)
            if field == "message" and obj_ann not in self.struct_names:
                result = obj + ".getMessage()"
            else:
                result = obj + "." + field
            if result in self._var_aliases:
                return self._var_aliases[result]
            return result
        if isinstance(expr, TTupleAccess):
            raw = self._expr(expr.obj) + ".get(" + str(expr.index) + ")"
            type_ann = expr.annotations.get("type", "")
            if type_ann:
                cast_type = self._tuple_cast_type(type_ann)
                if cast_type is not None:
                    return "((" + cast_type + ") " + raw + ")"
            return raw
        if isinstance(expr, TIndex):
            if expr.annotations.get("provenance") == "negative_index":
                neg = self._negative_index(expr)
                if neg is not None:
                    return neg
            return self._index_expr(expr)
        if isinstance(expr, TSlice):
            return self._slice(expr)
        if isinstance(expr, TBinaryOp):
            return self._binary(expr)
        if isinstance(expr, TUnaryOp):
            return self._unary(expr)
        if isinstance(expr, TTernary):
            return self._ternary(expr)
        if isinstance(expr, TListLit):
            if not expr.elements:
                ann = expr.annotations.get("type", "")
                if ann.startswith("list["):
                    inner = ann[5:-1]
                    boxed = self._boxed_from_ann(inner)
                    if boxed != "":
                        return "new ArrayList<" + boxed + ">()"
                return "new ArrayList<>()"
            has_tuple = any(self._is_tuple_expr(e) for e in expr.elements)
            elems = self._join_exprs(expr.elements, ", ")
            if has_tuple:
                return "new ArrayList<>(Arrays.asList(" + elems + "))"
            # Single-element list with array type needs special handling
            if len(expr.elements) == 1 and self._type_is_array(expr.elements[0]):
                return "new ArrayList<>(Collections.singletonList(" + elems + "))"
            return "new ArrayList<>(List.of(" + elems + "))"
        if isinstance(expr, TMapLit):
            mlit: TMapLit = expr
            if not mlit.entries:
                return "new HashMap<>()"
            map_ann = mlit.annotations.get("type", "")
            parsed_map = self._parse_map_type_ann(map_ann)
            key_coerce = ""
            val_coerce = ""
            if parsed_map is not None:
                key_coerce = parsed_map[0]
                val_coerce = parsed_map[1]
            if len(mlit.entries) <= 10:
                pair_strs: list[str] = []
                for k, v in mlit.entries:
                    k_s = self._map_lit_coerce(k, key_coerce)
                    v_s = self._map_lit_coerce(v, val_coerce)
                    pair_strs.append(k_s + ", " + v_s)
                return "new HashMap<>(Map.of(" + ", ".join(pair_strs) + "))"
            entry_strs: list[str] = []
            for k, v in mlit.entries:
                k_s = self._map_lit_coerce(k, key_coerce)
                v_s = self._map_lit_coerce(v, val_coerce)
                entry_strs.append("Map.entry(" + k_s + ", " + v_s + ")")
            return "new HashMap<>(Map.ofEntries(" + ", ".join(entry_strs) + "))"
        if isinstance(expr, TSetLit):
            if not expr.elements:
                return "new HashSet<>()"
            elems = self._join_exprs(expr.elements, ", ")
            return "new HashSet<>(Set.of(" + elems + "))"
        if isinstance(expr, TTupleLit):
            elems = self._join_exprs(expr.elements, ", ")
            return "Arrays.asList(" + elems + ")"
        if isinstance(expr, TFnLit):
            return self._fn_lit(expr)
        if isinstance(expr, TCall):
            return self._call(expr)
        return "null"

    def _int_lit(self, expr: TIntLit) -> str:
        v: int = expr.value
        raw: str = expr.raw
        # Note: raw is always decimal (converted by _int_to_decimal in parser)
        # Check for unsigned 64-bit values (> Long.MAX_VALUE in Java terms)
        # Detect by decimal string: > MAX_LONG if 20+ digits or 19 digits and > "9223372036854775807"
        is_unsigned_64: bool = False
        if not raw.startswith("-"):
            if len(raw) >= 20:
                is_unsigned_64 = True
            elif len(raw) == 19 and raw > "9223372036854775807":
                is_unsigned_64 = True
        if is_unsigned_64:
            # Compute signed = unsigned - 2^64 using string arithmetic so it
            # works correctly in Java where int may truncate large values and
            # 1<<64 is subject to modular shift.
            signed_str = "-" + _decimal_sub("18446744073709551616", raw)
            if self.strict_math:
                return signed_str + "L"
            return "(int) " + signed_str + "L"
        if raw.startswith(("0x", "0X", "0o", "0O", "0b", "0B")):
            if v > 2147483647 or v < -2147483648:
                if self.strict_math:
                    return raw + "L"
                return "(int) " + raw + "L"
            return raw
        # Check if decimal literal exceeds int range using raw string
        # (to handle cases where v overflowed in Java)
        needs_long: bool = False
        if raw.startswith("-"):
            # Negative: > 10 chars or 11 chars and > "2147483648"
            if len(raw) > 11:
                needs_long = True
            elif len(raw) == 11 and raw[1:] > "2147483648":
                needs_long = True
        else:
            # Positive: > 10 chars or 10 chars and > "2147483647"
            if len(raw) > 10:
                needs_long = True
            elif len(raw) == 10 and raw > "2147483647":
                needs_long = True
        if needs_long:
            if self.strict_math:
                return raw + "L"
            return "(int) " + raw + "L"
        return str(v)

    def _yields_long(self, expr: TExpr) -> bool:
        """Check if an expression would produce a long value in emitted Java."""
        if isinstance(expr, TIntLit):
            # Use raw string to detect: avoids issues with Java overflow
            raw: str = expr.raw
            if raw.startswith("-"):
                # Negative: needs long if > 11 chars or 11 chars and > "2147483648"
                if len(raw) > 11:
                    return True
                if len(raw) == 11 and raw[1:] > "2147483648":
                    return True
            else:
                # Positive: needs long if > 10 chars or 10 chars and > "2147483647"
                if len(raw) > 10:
                    return True
                if len(raw) == 10 and raw > "2147483647":
                    return True
            return False
        if isinstance(expr, TVar):
            return expr.name in self._wide_vars
        if isinstance(expr, TBinaryOp):
            if (
                expr.op == "<<"
                and isinstance(expr.right, TIntLit)
                and expr.right.value >= 31
                and isinstance(expr.left, TIntLit)
            ):
                return True
            return self._yields_long(expr.left) or self._yields_long(expr.right)
        if isinstance(expr, TUnaryOp):
            return self._yields_long(expr.operand)
        if isinstance(expr, TTernary):
            return self._yields_long(expr.then_expr) or self._yields_long(
                expr.else_expr
            )
        return False

    def _bytes_lit(self, expr: TBytesLit) -> str:
        elems = ", ".join("(byte) 0x" + hex(b)[2:].zfill(2) for b in expr.value)
        return "new byte[]{" + elems + "}"

    def _slice(self, expr: TSlice) -> str:
        obj = self._expr(expr.obj)
        lo = (
            self._expr(expr.low)
            if not isinstance(expr.low, TIntLit) or expr.low.value != 0
            else "0"
        )
        hi_expr = expr.high
        ann = expr.obj.annotations.get("type", "")
        hi_is_len = (
            isinstance(hi_expr, TCall)
            and isinstance(hi_expr.func, TVar)
            and hi_expr.func.name == "Len"
        )
        if ann == "bytes" or self._is_bytes_expr(expr.obj):
            hi = obj + ".length" if hi_is_len else self._expr(hi_expr)
            return "Arrays.copyOfRange(" + obj + ", " + lo + ", " + hi + ")"
        if ann == "string" or self._is_string_expr(expr.obj):
            if hi_is_len:
                return obj + ".substring(" + lo + ")"
            hi = self._expr(hi_expr)
            return (
                obj
                + ".substring("
                + lo
                + ", Math.min("
                + hi
                + ", "
                + obj
                + ".length()))"
            )
        if hi_is_len:
            if lo == "0":
                return "new ArrayList<>(" + obj + ")"
            return (
                "new ArrayList<>("
                + obj
                + ".subList(Math.min("
                + lo
                + ", "
                + obj
                + ".size()), "
                + obj
                + ".size()))"
            )
        hi = self._expr(hi_expr)
        if lo == "0":
            return (
                "new ArrayList<>("
                + obj
                + ".subList(0, Math.min("
                + hi
                + ", "
                + obj
                + ".size())))"
            )
        return (
            "new ArrayList<>("
            + obj
            + ".subList(Math.min("
            + lo
            + ", "
            + obj
            + ".size()), Math.min("
            + hi
            + ", "
            + obj
            + ".size())))"
        )

    def _emit_index_assign(self, target: TIndex, value: TExpr) -> None:
        obj = self._expr(target.obj)
        idx = self._expr(target.index)
        val = self._expr(value)
        ann = target.obj.annotations.get("type", "")
        if ann.startswith("map[") or self._is_map_type(target.obj):
            self._line(obj + ".put(" + idx + ", " + val + ");")
        elif ann == "bytes" or self._is_bytes_expr(target.obj):
            self._line(obj + "[" + idx + "] = " + val + ";")
        else:
            self._line(obj + ".set(" + idx + ", " + val + ");")

    def _index_expr(self, expr: TIndex) -> str:
        obj = self._expr(expr.obj)
        idx = self._expr(expr.index)
        ann = expr.obj.annotations.get("type", "")
        if ann.startswith("map[") or self._is_map_type(expr.obj):
            result = obj + ".get(" + idx + ")"
            cast = self._raw_binding_value_cast(expr.obj)
            if cast is not None:
                return "(" + cast + ") " + result
            return result
        if ann == "bytes" or self._is_bytes_expr(expr.obj):
            return obj + "[" + idx + "]"
        if ann == "string" or self._is_string_expr(expr.obj):
            return obj + ".charAt(" + idx + ")"
        result = obj + ".get(" + idx + ")"
        cast = self._raw_binding_value_cast(expr.obj)
        if cast is not None:
            return "(" + cast + ") " + result
        return result

    def _is_string_expr(self, expr: TExpr) -> bool:
        ann = expr.annotations.get("type", "")
        if ann == "string":
            return True
        if isinstance(expr, TStringLit):
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "string"
        if isinstance(expr, TCall):
            if isinstance(expr.func, TVar):
                if expr.func.name in ("ToString", "FormatInt", "oct", "bin", "hex"):
                    return True
                if expr.func.name == "Concat" and len(expr.args) > 0:
                    if self._is_string_expr(expr.args[0].value):
                        return True
                    if len(expr.args) > 1 and self._is_string_expr(expr.args[1].value):
                        return True
            if isinstance(expr.func, TFieldAccess) and expr.func.field == "to_string":
                return True
        if isinstance(expr, TTernary):
            prov = expr.annotations.get("provenance", "")
            if prov == "removeprefix" or prov == "removesuffix":
                return True
        return False

    def _is_nullable_string_expr(self, expr: TExpr) -> bool:
        """Check if expr is an optional string (String | None)."""
        ann = expr.annotations.get("type", "")
        if ann in ("string | nil", "nil | string"):
            return True
        return False

    def _is_rune_expr(self, expr: TExpr) -> bool:
        ann = expr.annotations.get("type", "")
        if ann == "rune":
            return True
        if isinstance(expr, TRuneLit):
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "rune"
        return False

    def _is_tuple_expr(self, expr: TExpr) -> bool:
        if isinstance(expr, TTupleLit):
            return True
        ann = expr.annotations.get("type", "")
        if ann.startswith("tuple[") or ann.startswith("("):
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TTupleType)
        return False

    def _type_is_array(self, expr: TExpr) -> bool:
        """Check if expression has an array type (tuple or bytes → Object[]/byte[])."""
        if isinstance(expr, (TTupleLit, TBytesLit)):
            return True
        ann = expr.annotations.get("type", "")
        if ann.startswith("tuple[") or ann == "bytes":
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            if isinstance(typ, TTupleType):
                return True
            if isinstance(typ, TOptionalType) and isinstance(typ.inner, TTupleType):
                return True
        return False

    def _expr_captures_mutable(
        self, expr: TExpr, binder: str, mutable_vars: set[str]
    ) -> bool:
        """Check if expr references any mutable variable other than the binder."""
        if isinstance(expr, TVar):
            return expr.name != binder and expr.name in mutable_vars
        if isinstance(expr, TCall):
            if any(
                self._expr_captures_mutable(a.value, binder, mutable_vars)
                for a in expr.args
            ):
                return True
            if isinstance(expr.func, TFieldAccess):
                return self._expr_captures_mutable(expr.func.obj, binder, mutable_vars)
            return False
        if isinstance(expr, TBinaryOp):
            return self._expr_captures_mutable(
                expr.left, binder, mutable_vars
            ) or self._expr_captures_mutable(expr.right, binder, mutable_vars)
        if isinstance(expr, TUnaryOp):
            return self._expr_captures_mutable(expr.operand, binder, mutable_vars)
        if isinstance(expr, TIndex):
            return self._expr_captures_mutable(
                expr.obj, binder, mutable_vars
            ) or self._expr_captures_mutable(expr.index, binder, mutable_vars)
        if isinstance(expr, TFieldAccess):
            return self._expr_captures_mutable(expr.obj, binder, mutable_vars)
        if isinstance(expr, TTernary):
            return (
                self._expr_captures_mutable(expr.cond, binder, mutable_vars)
                or self._expr_captures_mutable(expr.then_expr, binder, mutable_vars)
                or self._expr_captures_mutable(expr.else_expr, binder, mutable_vars)
            )
        return False

    def _expr_contains_call(self, expr: TExpr) -> bool:
        """Check if an expression contains a user-defined function call (recursively).

        Builtin calls (not in fn_names) are safe in Java lambdas since they
        don't throw checked exceptions. Only user-defined functions and method
        calls are flagged.
        """
        if isinstance(expr, TCall):
            if isinstance(expr.func, TVar) and expr.func.name not in self.fn_names:
                return any(self._expr_contains_call(a.value) for a in expr.args)
            return True
        if isinstance(expr, TBinaryOp):
            return self._expr_contains_call(expr.left) or self._expr_contains_call(
                expr.right
            )
        if isinstance(expr, TUnaryOp):
            return self._expr_contains_call(expr.operand)
        if isinstance(expr, TIndex):
            return self._expr_contains_call(expr.obj) or self._expr_contains_call(
                expr.index
            )
        if isinstance(expr, TFieldAccess):
            return self._expr_contains_call(expr.obj)
        if isinstance(expr, TTernary):
            return (
                self._expr_contains_call(expr.cond)
                or self._expr_contains_call(expr.then_expr)
                or self._expr_contains_call(expr.else_expr)
            )
        if isinstance(expr, TListLit):
            for e in expr.elements:
                if self._expr_contains_call(e):
                    return True
        if isinstance(expr, TTupleLit):
            for e in expr.elements:
                if self._expr_contains_call(e):
                    return True
        return False

    def _string_eq(
        self, str_expr: TExpr, other: TExpr, op: str, use_objects_equals: bool = False
    ) -> str | None:
        """Emit idiomatic equality for a string-typed expression.

        Optimizes ToString(rune_expr) == "x" → rune_expr == 'x'.
        Falls back to .equals() for string comparisons.
        Uses Objects.equals() for nullable strings.
        """
        char_op = "==" if op == "==" else "!="
        rune_inner = self._unwrap_tostring_rune(str_expr)
        if rune_inner is not None:
            if isinstance(other, TStringLit) and len(other.value) == 1:
                return (
                    self._expr(rune_inner)
                    + " "
                    + char_op
                    + " '"
                    + _escape_java_char(other.value)
                    + "'"
                )
            rune_other = self._unwrap_tostring_rune(other)
            if rune_other is not None:
                return (
                    self._expr(rune_inner)
                    + " "
                    + char_op
                    + " "
                    + self._expr(rune_other)
                )
        a = self._expr(str_expr)
        b = self._expr(other)
        if self._is_concat_expr(str_expr) or isinstance(str_expr, TTernary):
            a = "(" + a + ")"
        # Use simple != null or == null when comparing to nil
        if isinstance(other, TNilLit):
            if op == "==":
                return a + " == null"
            return a + " != null"
        if use_objects_equals:
            if op == "==":
                return "Objects.equals(" + a + ", " + b + ")"
            return "!Objects.equals(" + a + ", " + b + ")"
        if op == "==":
            return a + ".equals(" + b + ")"
        return "!" + a + ".equals(" + b + ")"

    def _is_concat_expr(self, expr: TExpr) -> bool:
        """True if the expression emits inline string concatenation with +."""
        if not isinstance(expr, TCall) or not isinstance(expr.func, TVar):
            return False
        return expr.func.name in ("Concat", "ToRepr")

    def _unwrap_tostring_rune(self, expr: TExpr) -> TExpr | None:
        """If expr is ToString(rune_expr), return the rune_expr."""
        if (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "ToString"
            and len(expr.args) == 1
        ):
            inner = expr.args[0].value
            if self._is_rune_expr(inner):
                return inner
            if isinstance(inner, TIndex):
                inner_type = inner.obj.annotations.get("type", "")
                if inner_type == "string" or self._is_string_expr(inner.obj):
                    return inner
        return None

    def _negative_index(self, expr: TIndex) -> str | None:
        """Detect x[Len(x)-n] pattern, emit xs.getLast() or xs.get(xs.size()-n)."""
        idx = expr.index
        if isinstance(idx, TBinaryOp) and idx.op == "-":
            if (
                isinstance(idx.left, TCall)
                and isinstance(idx.left.func, TVar)
                and idx.left.func.name == "Len"
            ):
                n = self._static_int(idx.right)
                obj = self._expr(expr.obj)
                if n == 1:
                    return obj + ".getLast()"
                return obj + ".get(" + obj + ".size() - " + self._expr(idx.right) + ")"
        return None

    def _binary(self, expr: TBinaryOp) -> str:
        op = expr.op
        if (
            op == "<<"
            and isinstance(expr.right, TIntLit)
            and expr.right.value >= 31
            and isinstance(expr.left, TIntLit)
        ):
            return str(expr.left.value) + "L << " + str(expr.right.value)
        if (
            op == "/"
            and isinstance(expr.left, TFloatLit)
            and expr.left.value == 0.0
            and isinstance(expr.right, TFloatLit)
            and expr.right.value == 0.0
        ):
            return "Double.NaN"
        if self.strict_math:
            if op in JAVA_STRICT_INT_BINARY and self._is_int_expr(expr.left):
                fn = JAVA_STRICT_INT_BINARY[op]
                if fn == ">>>":
                    return self._expr(expr.left) + " >>> " + self._expr(expr.right)
                return (
                    fn
                    + "("
                    + self._expr(expr.left)
                    + ", "
                    + self._expr(expr.right)
                    + ")"
                )
            if op == "%" and self._is_float_expr(expr.left):
                return (
                    "strictFmod("
                    + self._expr(expr.left)
                    + ", "
                    + self._expr(expr.right)
                    + ")"
                )
        if op in ("<", "<=", ">", ">=") and (
            self._is_tuple_expr(expr.left) or self._is_list_expr(expr.left)
        ):
            self._needs_list_compare = True
            return (
                "_listCompare("
                + self._expr(expr.left)
                + ", "
                + self._expr(expr.right)
                + ") "
                + op
                + " 0"
            )
        if op in ("<", "<=", ">", ">=") and self._is_string_expr(expr.left):
            return (
                self._expr(expr.left)
                + ".compareTo("
                + self._expr(expr.right)
                + ") "
                + op
                + " 0"
            )
        if op in ("==", "!=") and self._is_string_expr(expr.left):
            result = self._string_eq(expr.left, expr.right, op)
            if result is not None:
                return result
        if op in ("==", "!=") and self._is_string_expr(expr.right):
            result = self._string_eq(expr.right, expr.left, op)
            if result is not None:
                return result
        # Handle nullable strings with Objects.equals()
        if op in ("==", "!=") and self._is_nullable_string_expr(expr.left):
            result = self._string_eq(expr.left, expr.right, op, use_objects_equals=True)
            if result is not None:
                return result
        if op in ("==", "!=") and self._is_nullable_string_expr(expr.right):
            result = self._string_eq(expr.right, expr.left, op, use_objects_equals=True)
            if result is not None:
                return result
        # Handle nullable boolean comparisons with Boolean.TRUE.equals() to avoid NPE
        if op in ("==", "!=") and isinstance(expr.right, TBoolLit):
            left_ann = expr.left.annotations.get("type", "")
            # If left side is optional bool, use Boolean.TRUE/FALSE.equals()
            if "nil" in left_ann and "bool" in left_ann:
                left_str = self._expr(expr.left)
                bool_class = "Boolean.TRUE" if expr.right.value else "Boolean.FALSE"
                if op == "==":
                    return bool_class + ".equals(" + left_str + ")"
                return "!" + bool_class + ".equals(" + left_str + ")"
        if op in ("==", "!=") and isinstance(expr.left, TBoolLit):
            right_ann = expr.right.annotations.get("type", "")
            if "nil" in right_ann and "bool" in right_ann:
                right_str = self._expr(expr.right)
                bool_class = "Boolean.TRUE" if expr.left.value else "Boolean.FALSE"
                if op == "==":
                    return bool_class + ".equals(" + right_str + ")"
                return "!" + bool_class + ".equals(" + right_str + ")"
        # Handle nullable int comparisons with Integer.valueOf().equals() to avoid NPE
        if op in ("==", "!=") and isinstance(expr.right, TIntLit):
            left_ann = expr.left.annotations.get("type", "")
            if "nil" in left_ann and "int" in left_ann:
                left_str = self._expr(expr.left)
                val_str = str(expr.right.value)
                if op == "==":
                    return "Integer.valueOf(" + val_str + ").equals(" + left_str + ")"
                return "!Integer.valueOf(" + val_str + ").equals(" + left_str + ")"
        if op in ("==", "!=") and isinstance(expr.left, TIntLit):
            right_ann = expr.right.annotations.get("type", "")
            if "nil" in right_ann and "int" in right_ann:
                right_str = self._expr(expr.right)
                val_str = str(expr.left.value)
                if op == "==":
                    return "Integer.valueOf(" + val_str + ").equals(" + right_str + ")"
                return "!Integer.valueOf(" + val_str + ").equals(" + right_str + ")"
        if op in ("&&", "and"):
            checks = self._collect_isinstance_raw(expr.left)
            if checks:
                old_aliases = self._var_aliases.copy()
                parts: list[str] = []
                for check_expr in checks:
                    info = self._isinstance_info(check_expr)
                    if info is not None:
                        var_name = info[0]
                        cast_name = _binding_name(var_name + "_" + info[1])
                        parts.append(
                            var_name + " instanceof " + info[1] + " " + cast_name
                        )
                        self._var_aliases[var_name] = cast_name
                remaining = self._strip_isinstance_parts(expr.left)
                if remaining is not None:
                    parts.append(self._expr(remaining))
                right = self._maybe_paren(expr.right, "&&", False)
                self._var_aliases = old_aliases
                return " && ".join(parts) + " && " + right
        if op in ("||", "or"):
            neg_info = self._detect_negated_isinstance(expr.left)
            if neg_info is not None:
                var_name, type_name_str = neg_info
                cast_name = _binding_name(var_name + "_" + type_name_str)
                old_aliases = self._var_aliases.copy()
                self._var_aliases[var_name] = cast_name
                right = self._expr(expr.right)
                self._var_aliases = old_aliases
                return (
                    "!("
                    + var_name
                    + " instanceof "
                    + type_name_str
                    + " "
                    + cast_name
                    + ")"
                    + " || "
                    + right
                )
        if op in ("|", "&", "^") and self._is_bool_expr(expr.right):
            return (
                self._maybe_paren(expr.left, op, True)
                + " "
                + op
                + " ("
                + self._expr(expr.right)
                + " ? 1 : 0)"
            )
        if op in ("==", "!=") and not self._is_primitive_eq(expr.left, expr.right):
            left = self._expr(expr.left)
            right = self._expr(expr.right)
            if op == "==":
                return "Objects.equals(" + left + ", " + right + ")"
            return "!Objects.equals(" + left + ", " + right + ")"
        left = self._maybe_paren(expr.left, op, True)
        right = self._maybe_paren(expr.right, op, False)
        return left + " " + op + " " + right

    def _is_primitive_eq(self, left: TExpr, right: TExpr) -> bool:
        """True if == / != can use Java's == (both sides are primitives or null)."""
        if isinstance(left, TNilLit) or isinstance(right, TNilLit):
            return True
        if self._is_int_expr(left) or self._is_int_expr(right):
            return True
        if self._is_float_expr(left) or self._is_float_expr(right):
            return True
        if self._is_bool_expr(left) or self._is_bool_expr(right):
            return True
        if self._is_rune_expr(left) or self._is_rune_expr(right):
            return True
        left_ann = left.annotations.get("type", "")
        right_ann = right.annotations.get("type", "")
        for ann in (left_ann, right_ann):
            if ann in ("int", "float", "bool", "byte", "rune"):
                return True
            if ann in self._enum_names:
                return True
        return False

    def _detect_negated_isinstance(self, expr: TExpr) -> tuple[str, str] | None:
        """Detect `not isinstance(x, T)` and return (var_name, type_name)."""
        if isinstance(expr, TUnaryOp) and expr.op in ("not", "!"):
            return self._isinstance_info(expr.operand)
        return None

    def _unary(self, expr: TUnaryOp) -> str:
        if expr.op in ("not", "!"):
            return self._unary_not(expr.operand)
        operand = self._expr(expr.operand)
        if isinstance(expr.operand, (TBinaryOp, TTernary)):
            operand = "(" + operand + ")"
        elif (
            expr.op == "-"
            and isinstance(expr.operand, TUnaryOp)
            and expr.operand.op == "-"
        ):
            operand = "(" + operand + ")"
        if self.strict_math and expr.op == "-" and self._is_int_expr(expr.operand):
            return "Math.negateExact(" + operand + ")"
        return expr.op + operand

    def _unary_not(self, operand: TExpr) -> str:
        """Emit `!operand` with Java-correct truthiness."""
        if isinstance(operand, TCall) and isinstance(operand.func, TVar):
            fn = operand.func.name
            if fn == "IsNil":
                return self._a(operand.args, 0) + " != null"
            if fn == "IsType":
                info = self._isinstance_info(operand)
                if info is not None:
                    return "!(" + info[0] + " instanceof " + info[1] + ")"
        rendered = self._expr(operand)
        if " instanceof " in rendered:
            return "!(" + rendered + ")"
        if isinstance(operand, (TVar, TBoolLit, TFieldAccess, TIndex, TCall)):
            return "!" + rendered
        return "!(" + rendered + ")"

    def _ternary(self, expr: TTernary) -> str:
        prov = expr.annotations.get("provenance", "")
        if prov == "removeprefix" or prov == "removesuffix":
            return self._remove_affix_ternary(expr, prov)
        else_str = self._expr(expr.else_expr)
        if isinstance(expr.else_expr, TTernary):
            else_str = "(" + else_str + ")"
        raw_checks = self._collect_isinstance_raw(expr.cond)
        if raw_checks:
            parts: list[str] = []
            old_aliases = self._var_aliases.copy()
            for check_expr in raw_checks:
                var_name, type_name_str = self._isinstance_info_checked(check_expr)
                if self._instanceof_is_redundant(var_name, type_name_str):
                    self._var_aliases[var_name] = var_name
                    continue
                cast_name = _binding_name(var_name + "_" + type_name_str)
                parts.append(
                    var_name + " instanceof " + type_name_str + " " + cast_name
                )
                self._var_aliases[var_name] = cast_name
            remaining = self._strip_isinstance_parts(expr.cond)
            if remaining is not None:
                parts.append(self._maybe_paren(remaining, "&&", False))
            cond_str = " && ".join(parts) if parts else "true"
            then_str = self._expr(expr.then_expr)
            self._var_aliases = old_aliases
            return cond_str + " ? " + then_str + " : " + else_str
        return (
            self._expr(expr.cond)
            + " ? "
            + self._expr(expr.then_expr)
            + " : "
            + else_str
        )

    def _none_coalesce(self, expr: TTernary) -> str:
        raise NotImplementedError

    def _partition_ternary(self, expr: TTernary, prov: str) -> str:
        raise NotImplementedError

    def _remove_affix_ternary(self, expr: TTernary, prov: str) -> str:
        """Emit removeprefix/removesuffix as proper Java ternary with string results."""
        rx_cond = expr.cond
        if isinstance(rx_cond, TCall) and isinstance(rx_cond.func, TVar):
            obj_s = self._expr(rx_cond.args[0].value)
            arg_s = self._expr(rx_cond.args[1].value)
            if prov == "removeprefix":
                return (
                    obj_s
                    + ".startsWith("
                    + arg_s
                    + ") ? "
                    + obj_s
                    + ".substring("
                    + arg_s
                    + ".length()) : "
                    + obj_s
                )
            else:  # removesuffix
                return (
                    obj_s
                    + ".endsWith("
                    + arg_s
                    + ") ? "
                    + obj_s
                    + ".substring(0, "
                    + obj_s
                    + ".length() - "
                    + arg_s
                    + ".length()) : "
                    + obj_s
                )
        # Fallback to standard ternary
        return (
            self._expr(expr.cond)
            + " ? "
            + self._expr(expr.then_expr)
            + " : "
            + self._expr(expr.else_expr)
        )

    def _maybe_paren(self, expr: TExpr, parent_op: str, is_left: bool) -> str:
        s = self._expr(expr)
        if isinstance(expr, TBinaryOp) and _needs_parens(expr.op, parent_op, is_left):
            return "(" + s + ")"
        if isinstance(expr, TTernary):
            return "(" + s + ")"
        return s

    def _fn_lit(self, expr: TFnLit) -> str:
        """Emit lambda: (x) -> expr  or  (x) -> { stmts }"""
        params = ", ".join(_safe_name(p.name) for p in expr.params)
        if len(expr.body) == 1:
            stmt0 = expr.body[0]
            if isinstance(stmt0, TReturnStmt) and stmt0.value is not None:
                return "(" + params + ") -> " + self._expr(stmt0.value)
            if (
                isinstance(stmt0, TExprStmt)
                and expr.annotations.get("fn_lit.arrow") == "true"
            ):
                return "(" + params + ") -> " + self._expr(stmt0.expr)
        lines_buf: list[str] = []
        old_lines = self.lines
        self.lines = lines_buf
        self._emit_stmts(expr.body)
        self.lines = old_lines
        pad = "    " * self.indent
        result = "(" + params + ") -> {\n"
        for line in lines_buf:
            result += pad + "    " + line.strip() + "\n"
        result += pad + "}"
        return result

    def _call(self, expr: TCall) -> str:
        func = expr.func
        args = expr.args
        if isinstance(func, TVar) and func.name == "not" and len(args) == 1:
            operand = args[0].value
            inner = self._expr(operand)
            if isinstance(operand, (TBinaryOp, TCall, TUnaryOp)):
                return "!(" + inner + ")"
            return "!" + inner
        if isinstance(func, TVar) and func.name in BUILTIN_NAMES:
            return self._builtin_call(func.name, args, expr.annotations)
        if isinstance(func, TVar) and func.name in self.struct_names:
            if func.name in self.fn_names:
                fields = self._struct_field_decls.get(func.name, [])
                if len(args) != len(fields):
                    fn_name = self._expr(func)
                    arg_strs = self._join_args(args, ", ")
                    return fn_name + "(" + arg_strs + ")"
            return self._struct_call(func.name, args)
        if isinstance(func, TVar) and func.name == "oct" and len(args) == 1:
            return '"0o" + Integer.toOctalString(' + self._a(args, 0) + ")"
        if isinstance(func, TVar) and func.name == "bin" and len(args) == 1:
            return '"0b" + Integer.toBinaryString(' + self._a(args, 0) + ")"
        if isinstance(func, TVar) and func.name == "hex" and len(args) == 1:
            return '"0x" + Integer.toHexString(' + self._a(args, 0) + ")"
        if isinstance(func, TFieldAccess):
            return self._method_call(func, args)
        fn_name = self._expr(func)
        arg_strs = self._join_args(args, ", ")
        if isinstance(func, TVar):
            typ = self.var_types.get(func.name)
            if isinstance(typ, TFuncType):
                return fn_name + ".apply(" + arg_strs + ")"
            if self._current_struct:
                methods = self._struct_method_names.get(self._current_struct, set())
                if func.name in methods:
                    fn_name = "Main." + fn_name
        return fn_name + "(" + arg_strs + ")"

    def _struct_call(self, name: str, args: list[TArg]) -> str:
        """Emit: new StructName(args)"""
        safe = _EXCEPTION_MAP.get(name, _safe_name(name))
        if name in self._struct_field_decls and any(a.name is not None for a in args):
            field_order = [f.name for f in self._struct_field_decls[name]]
            ordered = self._reorder_named_args(args, field_order)
            arg_strs = ", ".join(self._expr(a.value) for a in ordered)
        else:
            arg_strs = self._join_args(args, ", ")
        return "new " + safe + "(" + arg_strs + ")"

    def _reorder_named_args(
        self, args: list[TArg], field_order: list[str]
    ) -> list[TArg]:
        named = {a.name: a for a in args if a.name}
        if not named:
            return args
        result: list[TArg] = []
        for fname in field_order:
            if fname in named:
                result.append(named[fname])
        return result

    def _method_call(self, func: TFieldAccess, args: list[TArg]) -> str:
        obj = self._expr(func.obj)
        method = _safe_name(func.field).lower()
        if method == "split":
            # Java String.split takes a regex, Python str.split takes a literal
            # Escape regex special characters in the separator
            sep = self._a(args, 0)
            return "List.of(" + obj + ".split(Pattern.quote(" + sep + ")))"
        if method == "zfill":
            self._needs_zfill = True
            return "_zfill(" + obj + ", " + self._a(args, 0) + ")"
        if method == "keys":
            if isinstance(func.obj, TVar) and self._is_raw_binding_var(func.obj):
                typ = self.var_types.get(func.obj.name)
                if isinstance(typ, TMapType):
                    key_type = self._boxed_type(typ.key)
                    return "new ArrayList<" + key_type + ">(" + obj + ".keySet())"
            return "new ArrayList<>(" + obj + ".keySet())"
        if method == "find" and self._is_bytes_expr(func.obj):
            self._needs_bytes_helpers = True
            return "_bytesIndexOf(" + obj + ", " + self._join_args(args, ", ") + ")"
        if method == "decode" and self._is_bytes_expr(func.obj):
            return "new String(" + obj + ", StandardCharsets.UTF_8)"
        if method == "hex" and self._is_bytes_expr(func.obj):
            self._needs_hex_helper = True
            return "_bytesHex(" + obj + ")"
        if method == "get" and len(args) == 2:
            return obj + ".getOrDefault(" + self._join_args(args, ", ") + ")"
        if method == "startswith":
            method = "startsWith"
        elif method == "endswith":
            method = "endsWith"
        elif method == "replaceall":
            method = "replaceAll"
        elif method == "append":
            method = "add"
        if method in ("size", "length") and self._is_bytes_expr(func.obj):
            return obj + ".length"
        arg_strs = self._join_args(args, ", ")
        return obj + "." + method + "(" + arg_strs + ")"

    def _builtin_call(self, name: str, args: list[TArg], ann: Ann | None = None) -> str:
        if ann is None:
            ann = {}
        if name == "WritelnOut":
            return "System.out.println(" + self._a(args, 0) + ")"
        if name == "WriteOut":
            return "System.out.print(" + self._a(args, 0) + ")"
        if name == "WritelnErr":
            return "System.err.println(" + self._a(args, 0) + ")"
        if name == "WriteErr":
            return "System.err.print(" + self._a(args, 0) + ")"
        if name == "ToString":
            return "String.valueOf(" + self._a(args, 0) + ")"
        if name == "ParseInt":
            return "parseIntAuto(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "ReadAll":
            return "input"
        if name == "Unwrap":
            return self._a(args, 0)
        if name == "DivMod":
            return self._a(args, 0) + " / " + self._a(args, 1)
        if name == "Format":
            return self._format_call(args)
        if name == "RuneToInt":
            arg = args[0].value
            if isinstance(arg, TRuneLit):
                return "(int) " + self._a(args, 0)
            return "(int) (" + self._a(args, 0) + ")"
        if name == "RuneFromInt":
            return "(char) (" + self._a(args, 0) + ")"
        if name == "Len":
            return self._len_expr(args[0].value)
        if name == "Abs":
            return "Math.abs(" + self._a(args, 0) + ")"
        if name == "Min":
            if len(args) == 1:
                return "Collections.min(" + self._a(args, 0) + ")"
            if len(args) == 2 and isinstance(args[1].value, TFnLit):
                return (
                    "Collections.min("
                    + self._a(args, 0)
                    + ", "
                    + self._comparator(args[1].value)
                    + ")"
                )
            if (
                self.strict_math
                and len(args) == 2
                and self._is_float_expr(args[0].value)
            ):
                return (
                    "strictMinF64(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            return "Math.min(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Max":
            if len(args) == 1:
                return "Collections.max(" + self._a(args, 0) + ")"
            if len(args) == 2 and isinstance(args[1].value, TFnLit):
                return (
                    "Collections.max("
                    + self._a(args, 0)
                    + ", "
                    + self._comparator(args[1].value)
                    + ")"
                )
            if (
                self.strict_math
                and len(args) == 2
                and self._is_float_expr(args[0].value)
            ):
                return (
                    "strictMaxF64(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            return "Math.max(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Pow":
            if self.strict_math and self._is_int_expr(args[0].value):
                return "checkedPow(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            cast = "(long)" if self.strict_math else "(int)"
            return (
                cast + " Math.pow(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            )
        if name == "Sorted" and self.strict_math and self._is_float_list(args[0].value):
            return "strictSortedF64(" + self._a(args, 0) + ")"
        if name == "Exit":
            return "doExit(" + self._a(args, 0) + ")"
        if name == "IntToFloat":
            return self._a(args, 0)
        if name == "FloatToInt":
            arg = args[0].value
            if isinstance(arg, (TBinaryOp, TUnaryOp)):
                return "(int)(" + self._a(args, 0) + ")"
            return "(int) " + self._a(args, 0)
        if name == "ByteToInt":
            return "(" + self._a(args, 0) + " & 0xFF)"
        if name == "IntToByte":
            return self._a(args, 0)
        if name == "IsNil":
            return self._a(args, 0) + " == null"
        if name == "IsNaN":
            return "Double.isNaN(" + self._a(args, 0) + ")"
        if name == "WrappingAdd":
            return self._a(args, 0) + " + " + self._a(args, 1)
        if name == "WrappingSub":
            return self._a(args, 0) + " - " + self._a(args, 1)
        if name == "WrappingMul":
            return self._a(args, 0) + " * " + self._a(args, 1)
        if name == "Sqrt":
            return "Math.sqrt(" + self._a(args, 0) + ")"
        if name == "Floor":
            return "(int) Math.floor(" + self._a(args, 0) + ")"
        if name == "Ceil":
            return "(int) Math.ceil(" + self._a(args, 0) + ")"
        if name == "Round":
            return "(int) Math.round(" + self._a(args, 0) + ")"
        if name == "FloorDiv":
            if self._is_float_expr(args[0].value):
                left = self._maybe_paren(args[0].value, "/", is_left=True)
                right = self._maybe_paren(args[1].value, "/", is_left=False)
                return "Math.floor(" + left + " / " + right + ")"
            return "Math.floorDiv(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "PythonMod":
            if self._is_float_expr(args[0].value):
                a = self._maybe_paren(args[0].value, "/", is_left=True)
                b = self._maybe_paren(args[1].value, "/", is_left=False)
                return "(" + a + " - Math.floor(" + a + " / " + b + ") * " + b + ")"
            return "Math.floorMod(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "IsInf":
            return "Double.isInfinite(" + self._a(args, 0) + ")"
        if name == "Append":
            return self._a(args, 0) + ".add(" + self._a(args, 1) + ")"
        if name == "Insert":
            return (
                self._a(args, 0)
                + ".add("
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ")"
            )
        if name == "Pop":
            if self._is_set_expr(args[0].value):
                self._needs_set_pop = True
                return "_setPop(" + self._a(args, 0) + ")"
            return self._a(args, 0) + ".removeLast()"
        if name == "PopItem":
            self._needs_pop_item = True
            return "_popItem(" + self._a(args, 0) + ")"
        if name == "RemoveAt":
            idx_val = self._static_int(args[1].value)
            if idx_val == 0:
                return self._a(args, 0) + ".removeFirst()"
            return self._a(args, 0) + ".remove(" + self._a(args, 1) + ")"
        if name == "Get":
            if len(args) == 3:
                return (
                    self._a(args, 0)
                    + ".getOrDefault("
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ")"
                )
            return self._a(args, 0) + ".get(" + self._a(args, 1) + ")"
        if name == "Add":
            return self._a(args, 0) + ".add(" + self._a(args, 1) + ")"
        if name == "Delete":
            return self._a(args, 0) + ".remove(" + self._a(args, 1) + ")"
        if name == "Contains":
            obj = args[0].value
            if self._is_map_type(obj):
                return self._a(args, 0) + ".containsKey(" + self._a(args, 1) + ")"
            if self._is_bytes_expr(obj):
                self._needs_bytes_helpers = True
                return (
                    "_bytesContains(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            type_ann = obj.annotations.get("type", "")
            if type_ann == "string" or self._is_string_expr(obj):
                arg1 = args[1].value
                a1_ann = self._expr_type_ann(arg1)
                if a1_ann == "rune":
                    return (
                        self._a(args, 0)
                        + ".contains(String.valueOf("
                        + self._a(args, 1)
                        + "))"
                    )
                return self._a(args, 0) + ".contains(" + self._a(args, 1) + ")"
            return self._a(args, 0) + ".contains(" + self._a(args, 1) + ")"
        if name == "Keys":
            return "new ArrayList<>(" + self._a(args, 0) + ".keySet())"
        if name == "IndexOf":
            return self._a(args, 0) + ".indexOf(" + self._a(args, 1) + ")"
        if name == "Concat":
            a0_str = self._is_string_expr(args[0].value)
            a1_str = len(args) > 1 and self._is_string_expr(args[1].value)
            a0_bytes = self._is_bytes_expr(args[0].value)
            a1_bytes = len(args) > 1 and self._is_bytes_expr(args[1].value)
            if a0_str and a1_bytes:
                self._needs_concat_bytes = True
                lhs = "(" + self._a(args, 0) + ").getBytes(StandardCharsets.UTF_8)"
                return "_concatBytes(" + lhs + ", " + self._a(args, 1) + ")"
            if a0_str or a1_str:
                lhs = self._a(args, 0)
                rhs = self._a(args, 1)
                if isinstance(args[0].value, TTernary):
                    lhs = "(" + lhs + ")"
                if isinstance(args[1].value, TTernary):
                    rhs = "(" + rhs + ")"
                return lhs + " + " + rhs
            if a0_bytes or a1_bytes:
                self._needs_concat_bytes = True
                lhs = self._a(args, 0)
                rhs = self._a(args, 1)
                if a0_str:
                    lhs = "(" + lhs + ").getBytes(StandardCharsets.UTF_8)"
                if a1_str:
                    rhs = "(" + rhs + ").getBytes(StandardCharsets.UTF_8)"
                return "_concatBytes(" + lhs + ", " + rhs + ")"
            self._needs_concat_lists = True
            return "_concatLists(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Merge":
            self._needs_merge_maps = True
            return "_mergeMaps(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Union":
            self._needs_union_sets = True
            return "_unionSets(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Intersection":
            self._needs_intersect_sets = True
            return "_intersectSets(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Difference":
            self._needs_difference_sets = True
            return "_differenceSets(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "SetFromList":
            inner = args[0].value
            if (
                isinstance(inner, TCall)
                and isinstance(inner.func, TVar)
                and inner.func.name == "Keys"
            ):
                return "new HashSet<>(" + self._expr(inner.args[0].value) + ".keySet())"
            return "new HashSet<>(" + self._a(args, 0) + ")"
        if name == "ListFrom":
            inner = args[0].value
            if (
                isinstance(inner, TCall)
                and isinstance(inner.func, TVar)
                and inner.func.name == "Keys"
            ):
                return (
                    "new ArrayList<>(" + self._expr(inner.args[0].value) + ".keySet())"
                )
            if (
                isinstance(inner, TCall)
                and isinstance(inner.func, TVar)
                and inner.func.name == "Values"
            ):
                return (
                    "new ArrayList<>(" + self._expr(inner.args[0].value) + ".values())"
                )
            if self._is_bytes_expr(inner):
                src = self._expr(inner)
                return (
                    "IntStream.range(0, "
                    + src
                    + ".length).map(i -> "
                    + src
                    + "[i] & 0xFF).boxed().collect(Collectors.toList())"
                )
            return "new ArrayList<>(" + self._a(args, 0) + ")"
        if name == "Sorted":
            if len(args) == 2 and isinstance(args[1].value, TFnLit):
                return (
                    self._a(args, 0)
                    + ".stream().sorted("
                    + self._comparator(args[1].value)
                    + ").collect(Collectors.toList())"
                )
            return self._a(args, 0) + ".stream().sorted().collect(Collectors.toList())"
        if name == "Reversed":
            return "new ArrayList<>(" + self._a(args, 0) + ".reversed())"
        if name == "Reverse":
            type_ann = args[0].value.annotations.get("type", "")
            if type_ann == "string" or self._is_string_expr(args[0].value):
                return (
                    "new StringBuilder(" + self._a(args, 0) + ").reverse().toString()"
                )
            return "java.util.Collections.reverse(" + self._a(args, 0) + ")"
        if name == "Sum":
            first = args[0].value
            if isinstance(first, TListLit):
                slit: TListLit = first
                if len(slit.elements) == 0:
                    return "0"
            return self._a(args, 0) + ".stream().mapToInt(Integer::intValue).sum()"
        if name == "Map":
            return "new HashMap<>()"
        if name == "Set":
            return "new HashSet<>()"
        if name == "Zip":
            a = self._a(args, 0)
            b = self._a(args, 1)
            return (
                "IntStream.range(0, Math.min("
                + a
                + ".size(), "
                + b
                + ".size())).mapToObj(i -> Arrays.<Object>asList("
                + a
                + ".get(i), "
                + b
                + ".get(i))).collect(Collectors.toList())"
            )
        if name == "Repeat":
            first = args[0].value
            type_ann = first.annotations.get("type", "")
            if type_ann == "string" or self._is_string_expr(first):
                return self._a(args, 0) + ".repeat(" + self._a(args, 1) + ")"
            if type_ann == "bytes" or self._is_bytes_expr(first):
                self._needs_repeat_bytes = True
                return (
                    "_repeatBytes(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            if isinstance(first, TListLit) and len(first.elements) == 1:
                elem = self._expr(first.elements[0])
                return (
                    "new ArrayList<>(Collections.nCopies("
                    + self._a(args, 1)
                    + ", "
                    + elem
                    + "))"
                )
            self._needs_repeat_list = True
            return "_repeatList(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "RangeList":
            if len(args) == 1:
                return (
                    "IntStream.range(0, "
                    + self._a(args, 0)
                    + ").boxed().collect(Collectors.toList())"
                )
            if len(args) == 3:
                step_val = self._static_int(args[2].value)
                if step_val == 1:
                    return (
                        "IntStream.range("
                        + self._a(args, 0)
                        + ", "
                        + self._a(args, 1)
                        + ").boxed().collect(Collectors.toList())"
                    )
                return (
                    "IntStream.iterate("
                    + self._a(args, 0)
                    + ", i -> i < "
                    + self._a(args, 1)
                    + ", i -> i + "
                    + self._a(args, 2)
                    + ").boxed().collect(Collectors.toList())"
                )
            return (
                "IntStream.range("
                + self._a(args, 0)
                + ", "
                + self._a(args, 1)
                + ").boxed().collect(Collectors.toList())"
            )
        if name == "Split":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return "_bytesSplit(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            # Skip Pattern.quote for simple literals without regex metacharacters
            sep_arg = args[1].value
            if isinstance(sep_arg, TStringLit) and not any(
                c in sep_arg.value for c in r"\.[]{}()*+-?^$|"
            ):
                return (
                    "List.of(" + self._a(args, 0) + ".split(" + self._a(args, 1) + "))"
                )
            # Use Pattern.quote to escape regex special characters
            return (
                "List.of("
                + self._a(args, 0)
                + ".split(Pattern.quote("
                + self._a(args, 1)
                + ")))"
            )
        if name == "SplitN":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return (
                    "_bytesSplitN("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ")"
                )
            return (
                "new ArrayList<>(List.of("
                + self._a(args, 0)
                + ".split(Pattern.quote("
                + self._a(args, 1)
                + "), "
                + self._a(args, 2)
                + ")))"
            )
        if name == "Join":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return "_bytesJoin(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            return "String.join(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Replace":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return (
                    "_bytesReplace("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ")"
                )
            return (
                self._a(args, 0)
                + ".replace("
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ")"
            )
        if name == "ReplaceCount":
            self._needs_replace_count = True
            return (
                "replaceCount("
                + self._a(args, 0)
                + ", "
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ", "
                + self._a(args, 3)
                + ")"
            )
        if name == "Upper":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return "_bytesUpper(" + self._a(args, 0) + ")"
            return self._a(args, 0) + ".toUpperCase()"
        if name == "Lower":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return "_bytesLower(" + self._a(args, 0) + ")"
            return self._a(args, 0) + ".toLowerCase()"
        if name == "StartsWith":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return (
                    "_bytesStartsWith("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ")"
                )
            if len(args) == 4:
                return (
                    self._a(args, 0)
                    + ".substring("
                    + self._a(args, 2)
                    + ", "
                    + self._a(args, 3)
                    + ").startsWith("
                    + self._a(args, 1)
                    + ")"
                )
            if len(args) == 3:
                return (
                    self._a(args, 0)
                    + ".startsWith("
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ")"
                )
            return self._a(args, 0) + ".startsWith(" + self._a(args, 1) + ")"
        if name == "EndsWith":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return (
                    "_bytesEndsWith(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            if len(args) >= 3:
                s = self._a(args, 0) + ".substring(" + self._a(args, 2)
                if len(args) == 4:
                    s += ", " + self._a(args, 3)
                s += ")"
                return s + ".endsWith(" + self._a(args, 1) + ")"
            return self._a(args, 0) + ".endsWith(" + self._a(args, 1) + ")"
        if name == "Trim":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return "_bytesTrim(" + self._a(args, 0) + ")"
            return self._emit_trim(args, "both")
        if name == "TrimStart":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return "_bytesTrimStart(" + self._a(args, 0) + ")"
            return self._emit_trim(args, "start")
        if name == "TrimEnd":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return "_bytesTrimEnd(" + self._a(args, 0) + ")"
            return self._emit_trim(args, "end")
        if name == "Find":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return (
                    "_bytesIndexOf(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            if len(args) == 4:
                self._needs_string_pos_helpers = True
                return (
                    "_findInRange("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ", "
                    + self._a(args, 3)
                    + ")"
                )
            if len(args) == 3:
                return (
                    self._a(args, 0)
                    + ".indexOf("
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ")"
                )
            return self._a(args, 0) + ".indexOf(" + self._a(args, 1) + ")"
        if name == "RFind":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return (
                    "_bytesLastIndexOf("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ")"
                )
            if len(args) == 4:
                self._needs_string_pos_helpers = True
                return (
                    "_rfindInRange("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ", "
                    + self._a(args, 3)
                    + ")"
                )
            if len(args) == 3:
                self._needs_string_pos_helpers = True
                return (
                    "_rfindFrom("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ")"
                )
            return self._a(args, 0) + ".lastIndexOf(" + self._a(args, 1) + ")"
        if name == "Count":
            if self._is_bytes_expr(args[0].value):
                self._needs_bytes_helpers = True
                return "_bytesCount(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            if self._is_list_expr(args[0].value):
                return (
                    "Collections.frequency("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ")"
                )
            subj = self._a(args, 0)
            if len(args) >= 3:
                subj = subj + ".substring(" + self._a(args, 2)
                if len(args) == 4:
                    subj += ", " + self._a(args, 3)
                subj += ")"
            # Skip Pattern.quote for simple literals without regex metacharacters
            sep_arg = args[1].value
            if isinstance(sep_arg, TStringLit) and not any(
                c in sep_arg.value for c in r"\.[]{}()*+-?^$|"
            ):
                return subj + ".split(" + self._a(args, 1) + ", -1).length - 1"
            return (
                subj + ".split(Pattern.quote(" + self._a(args, 1) + "), -1).length - 1"
            )
        if name == "FormatInt":
            return (
                "Integer.toString(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            )
        if name == "ParseFloat":
            a = self._a(args, 0)
            return (
                "Double.parseDouble("
                + a
                + '.replace("inf", "Infinity")'
                + '.replace("nan", "NaN"))'
            )
        if name == "Assert":
            raise NotImplementedError("builtin: Assert")
        if name == "Args":
            self._needs_argv = True
            return "new ArrayList<String>(_argv)"
        if name == "GetEnv":
            return "System.getenv(" + self._a(args, 0) + ")"
        if name == "ReadLine":
            return "new java.io.BufferedReader(new java.io.InputStreamReader(System.in)).readLine()"
        if name == "ReadFile":
            return "Files.readString(Path.of(" + self._a(args, 0) + "))"
        if name == "ReadFileBytes":
            return "Files.readAllBytes(Path.of(" + self._a(args, 0) + "))"
        if name == "WriteFile":
            if self._is_bytes_expr(args[1].value):
                return (
                    "Files.write(Path.of("
                    + self._a(args, 0)
                    + "), "
                    + self._a(args, 1)
                    + ")"
                )
            return (
                "Files.writeString(Path.of("
                + self._a(args, 0)
                + "), "
                + self._a(args, 1)
                + ")"
            )
        if name == "ReadBytes":
            return "System.in.readAllBytes()"
        if name == "ReadBytesN":
            return "System.in.readNBytes(" + self._a(args, 0) + ")"
        if name == "Encode":
            inner = args[0].value
            if isinstance(inner, (TBinaryOp, TCall)):
                return "(" + self._a(args, 0) + ").getBytes(StandardCharsets.UTF_8)"
            return self._a(args, 0) + ".getBytes(StandardCharsets.UTF_8)"
        if name == "Decode":
            self._needs_decode_utf8 = True
            return "_decodeUtf8(" + self._a(args, 0) + ")"
        if name == "Bytes":
            return "new byte[" + self._a(args, 0) + "]"
        if name == "IsDigit":
            if self._is_rune_expr(args[0].value):
                return "Character.isDigit(" + self._a(args, 0) + ")"
            return self._a(args, 0) + ".chars().allMatch(Character::isDigit)"
        if name == "IsAlpha":
            if self._is_rune_expr(args[0].value):
                return "Character.isLetter(" + self._a(args, 0) + ")"
            return self._a(args, 0) + ".chars().allMatch(Character::isLetter)"
        if name == "IsAlphanumeric":
            if self._is_rune_expr(args[0].value):
                return "Character.isLetterOrDigit(" + self._a(args, 0) + ")"
            return self._a(args, 0) + ".chars().allMatch(Character::isLetterOrDigit)"
        if name == "IsUpper":
            if self._is_rune_expr(args[0].value):
                return "Character.isUpperCase(" + self._a(args, 0) + ")"
            return self._a(args, 0) + ".chars().allMatch(Character::isUpperCase)"
        if name == "IsLower":
            if self._is_rune_expr(args[0].value):
                return "Character.isLowerCase(" + self._a(args, 0) + ")"
            return self._a(args, 0) + ".chars().allMatch(Character::isLowerCase)"
        if name == "IsSpace":
            if self._is_rune_expr(args[0].value):
                return "Character.isWhitespace(" + self._a(args, 0) + ")"
            return self._a(args, 0) + ".chars().allMatch(Character::isWhitespace)"
        if name == "IsType":
            type_arg = args[1].value
            if isinstance(type_arg, TStringLit):
                tn = type_arg.value
                arg_ann = args[0].value.annotations.get("type", "")
                if arg_ann == tn and tn in ("int", "float", "bool", "byte", "rune"):
                    return "true"
                tn = _ISTYPE_MAP.get(tn, tn)
                return self._a(args, 0) + " instanceof " + tn
            return self._a(args, 0) + " instanceof " + self._expr(type_arg)
        if name == "Values":
            return "new ArrayList<>(" + self._a(args, 0) + ".values())"
        if name == "Items":
            items_obj = self._a(args, 0)
            return (
                items_obj
                + ".entrySet().stream().map(e -> Arrays.<Object>asList(e.getKey(), e.getValue())).collect(Collectors.toList())"
            )
        if name == "Remove":
            return self._a(args, 0) + ".remove(" + self._a(args, 1) + ")"
        if name == "SplitWhitespace":
            return "List.of(" + self._a(args, 0) + '.trim().split("\\\\s+"))'
        if name == "IsAlnum":
            if self._is_rune_expr(args[0].value):
                return "Character.isLetterOrDigit(" + self._a(args, 0) + ")"
            return self._a(args, 0) + ".chars().allMatch(Character::isLetterOrDigit)"
        if name == "BytesFrom":
            self._needs_to_byte_array = True
            return "toByteArray(" + self._a(args, 0) + ")"
        if name == "ToRepr":
            if self._is_string_expr(args[0].value):
                return '"\\"" + ' + self._a(args, 0) + ' + "\\""'
            return "String.valueOf(" + self._a(args, 0) + ")"
        if name == "ReplaceSlice":
            self._needs_replace_slice = True
            return (
                "replaceSlice("
                + self._a(args, 0)
                + ", "
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ", "
                + self._a(args, 3)
                + ")"
            )
        if name == "ListCompare":
            self._needs_list_compare = True
            return "_listCompare(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "MapFromKeys":
            keys = self._a(args, 0)
            default = self._a(args, 1)
            return (
                keys
                + ".stream().collect(Collectors.toMap(k -> k, k -> "
                + default
                + ", (a, b) -> b, LinkedHashMap::new))"
            )
        if name == "MapFromPairs":
            pairs = self._a(args, 0)
            key_cast = ""
            val_cast = ""
            if ann is not None:
                map_ann = ann.get("type", "")
                parsed = self._parse_map_type_ann(map_ann)
                if parsed is not None:
                    key_cast = "(" + parsed[0] + ") "
                    val_cast = "(" + parsed[1] + ") "
            return (
                pairs
                + ".stream().collect(Collectors.toMap(p -> "
                + key_cast
                + "p.get(0), p -> "
                + val_cast
                + "p.get(1), (a, b) -> b, LinkedHashMap::new))"
            )
        raise NotImplementedError("builtin: " + name)

    def _star_unpack(self, expr: TCall) -> str:
        """Emit list concatenation via addAll pattern."""
        raise NotImplementedError

    def _flatten_star_unpack(self, expr: TExpr, parts: list[TExpr]) -> None:
        raise NotImplementedError

    # ── Truthiness / Len / helpers ───────────────────────────

    def _truthiness_expr(self, cond: TExpr, *, raised: bool = False) -> str | None:
        """Emit idiomatic truthiness checks.
        - s != "" always becomes !s.isEmpty()
        - Len(x) > 0 becomes !x.isEmpty() only when raised (provenance=truthiness)
        """
        if isinstance(cond, TBinaryOp):
            if (
                raised
                and cond.op == ">"
                and isinstance(cond.right, TIntLit)
                and cond.right.value == 0
                and isinstance(cond.left, TCall)
                and isinstance(cond.left.func, TVar)
                and cond.left.func.name == "Len"
            ):
                inner = cond.left.args[0].value
                return "!" + self._expr(inner) + ".isEmpty()"
            if (
                cond.op == "!="
                and isinstance(cond.right, TStringLit)
                and not cond.right.value
            ):
                return "!" + self._expr(cond.left) + ".isEmpty()"
        # Handle bare string variable as condition (if lowering didn't transform it)
        if self._is_string_expr(cond):
            return "!" + self._expr(cond) + ".isEmpty()"
        return None

    def _len_expr(self, expr: TExpr) -> str:
        """Emit .size() / .length() / .length as appropriate."""
        e = self._expr(expr)
        ann = expr.annotations.get("type", "")
        if ann == "string":
            return e + ".length()"
        if ann == "bytes":
            return e + ".length"
        if ann.startswith("list[") or ann.startswith("map[") or ann.startswith("set["):
            return e + ".size()"
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            if isinstance(typ, (TListType, TMapType, TSetType)):
                return e + ".size()"
            if isinstance(typ, TIdentType) and typ.name == "bytes":
                return e + ".length"
            if isinstance(typ, TPrimitive) and typ.kind == "bytes":
                return e + ".length"
            if isinstance(typ, TPrimitive) and typ.kind == "string":
                return e + ".length()"
        if isinstance(expr, TStringLit):
            return e + ".length()"
        if self._is_bytes_expr(expr):
            return e + ".length"
        return e + ".size()"

    def _is_map_type(self, expr: TExpr) -> bool:
        ann = expr.annotations.get("type", "")
        if ann.startswith("map["):
            return True
        if isinstance(expr, TMapLit):
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TMapType)
        return False

    def _is_set_type(self, expr: TExpr) -> bool:
        raise NotImplementedError

    def _is_bool_expr(self, expr: TExpr) -> bool:
        """Check if expression evaluates to boolean in Java (comparison/logical)."""
        if isinstance(expr, TBinaryOp):
            return expr.op in (
                "==",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
                "&&",
                "||",
                "and",
                "or",
            )
        if isinstance(expr, TBoolLit):
            return True
        return False

    def _is_list_expr(self, expr: TExpr) -> bool:
        ann = self._expr_type_ann(expr)
        return ann.startswith("list[")

    def _is_bytes_expr(self, expr: TExpr) -> bool:
        ann = self._expr_type_ann(expr)
        return ann == "bytes"

    def _is_set_expr(self, expr: TExpr) -> bool:
        ann = self._expr_type_ann(expr)
        return ann.startswith("set[")

    def _is_int_list(self, expr: TExpr) -> bool:
        raise NotImplementedError

    def _is_isinstance_call(self, expr: TExpr) -> bool:
        raise NotImplementedError

    def _is_isnil_call(self, expr: TExpr) -> bool:
        raise NotImplementedError

    def _is_divmod_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "DivMod"
        )

    def _has_tuple_key(self, map_expr: TExpr) -> bool:
        raise NotImplementedError

    def _map_key(self, key_expr: TExpr) -> str:
        raise NotImplementedError

    def _map_key_for(self, map_expr: TExpr, key_expr: TExpr) -> str:
        raise NotImplementedError

    def _get_field_type(self, struct_name: str, field: str) -> TType | None:
        raise NotImplementedError

    def _pattern_type_name(self, typ: TType) -> str:
        raise NotImplementedError

    # ── Sorted / Min / Max with key ──────────────────────────

    def _sorted_with_key(self, collection: TExpr, key_fn: TFnLit) -> str:
        """Emit: xs.stream().sorted(Comparator.comparingInt(lambda)).collect(...)"""
        raise NotImplementedError

    def _min_max_key_cmp(self, key_fn: TFnLit, cmp_op: str) -> str:
        """Emit: Collections.min/max(xs, Comparator.comparingInt(...))"""
        raise NotImplementedError

    # ── Emit helpers ─────────────────────────────────────────

    def _emit_field_assign(self, fld: TFieldDecl, safe: str) -> None:
        raise NotImplementedError

    def _needs_null_guard(self, fld: TFieldDecl) -> bool:
        raise NotImplementedError

    def _static_int(self, expr: TExpr) -> int | None:
        if isinstance(expr, TIntLit):
            return expr.value
        if (
            isinstance(expr, TUnaryOp)
            and expr.op == "-"
            and isinstance(expr.operand, TIntLit)
        ):
            return -expr.operand.value
        if (
            isinstance(expr, TBinaryOp)
            and expr.op == "-"
            and isinstance(expr.left, TIntLit)
            and isinstance(expr.right, TIntLit)
        ):
            return expr.left.value - expr.right.value
        return None

    def _flatten_isinstance_tuple(self, expr: TExpr, types: list[str]) -> str | None:
        raise NotImplementedError

    def _emit_partition_return(self, expr: TTernary) -> None:
        raise NotImplementedError

    def _emit_divmod_assign(self, stmt: TTupleAssignStmt, unused: set[int]) -> None:
        assert isinstance(stmt.value, TCall)
        call: TCall = stmt.value
        a = self._maybe_paren(call.args[0].value, "/", True)
        b = self._maybe_paren(call.args[1].value, "/", False)
        q = self._expr(stmt.targets[0])
        r = self._expr(stmt.targets[1])
        if 0 not in unused:
            self._line(q + " = " + a + " / " + b + ";")
        if 1 not in unused:
            self._line(r + " = " + a + " - " + q + " * " + b + ";")

    def _escape_regex_class(self, s: str) -> str:
        raise NotImplementedError

    def _emit_trim(self, args: list[TArg], mode: str) -> str:
        s = self._a(args, 0)
        if len(args) < 2:
            if mode == "both":
                return s + ".strip()"
            if mode == "start":
                return s + ".stripLeading()"
            return s + ".stripTrailing()"
        chars_expr = args[1].value
        if isinstance(chars_expr, TStringLit) and chars_expr.value == " \t\n\r\x0b\x0c":
            if mode == "both":
                return s + ".strip()"
            if mode == "start":
                return s + ".stripLeading()"
            return s + ".stripTrailing()"
        if isinstance(chars_expr, TStringLit):
            parts: list[str] = []
            for ch in chars_expr.value:
                if ch == "\\":
                    parts.append("\\\\\\\\")
                elif ch in ("[", "]", "^", "-"):
                    parts.append("\\\\" + ch)
                elif ch == '"':
                    parts.append('\\"')
                elif ch == "\n":
                    parts.append("\\n")
                elif ch == "\t":
                    parts.append("\\t")
                elif ch == "\r":
                    parts.append("\\r")
                elif ord(ch) < 32 or ord(ch) > 126:
                    h = hex(ord(ch))[2:]
                    while len(h) < 4:
                        h = "0" + h
                    parts.append("\\\\u" + h)
                else:
                    parts.append(ch)
            esc = "".join(parts)
            if mode == "both":
                return s + '.replaceAll("^[' + esc + "]+|[" + esc + ']+$", "")'
            if mode == "start":
                return s + '.replaceAll("^[' + esc + ']+", "")'
            return s + '.replaceAll("[' + esc + ']+$", "")'
        chars = self._a(args, 1)
        if mode == "both":
            return (
                s
                + '.replaceAll("^[" + '
                + chars
                + ' + "]+|[" + '
                + chars
                + ' + "]+$", "")'
            )
        if mode == "start":
            return s + '.replaceAll("^[" + ' + chars + ' + "]+", "")'
        return s + '.replaceAll("[" + ' + chars + ' + "]+$", "")'

    def _format_int(self, args: list[TArg]) -> str:
        raise NotImplementedError

    def _format_call(self, args: list[TArg]) -> str:
        template_expr = args[0].value
        if not isinstance(template_expr, TStringLit):
            return "String.format(" + self._join_args(args, ", ") + ")"
        template = template_expr.value
        fmt_args = args[1:]
        parts: list[str] = []
        remaining = template
        arg_idx = 0
        while "{}" in remaining and arg_idx < len(fmt_args):
            split_parts = remaining.split("{}", 1)
            before = split_parts[0]
            remaining = split_parts[1]
            if before:
                parts.append('"' + _escape_java_string(before) + '"')
            parts.append(self._expr(fmt_args[arg_idx].value))
            arg_idx += 1
        if remaining:
            parts.append('"' + _escape_java_string(remaining) + '"')
        if not parts:
            return '""'
        return " + ".join(parts)

    def _for_elem_type(self, iterable: TExpr) -> str:
        """Resolve the element type of a for-each iterable."""
        typ: TType | None = None
        if isinstance(iterable, TVar):
            typ = self.var_types.get(iterable.name)
        if isinstance(typ, TListType):
            return self._type(typ.element)
        if isinstance(typ, TSetType):
            return self._type(typ.element)
        return "var"

    def _comparator(self, fn_lit: TExpr) -> str:
        if not isinstance(fn_lit, TFnLit):
            return ""
        param = _safe_name(fn_lit.params[0].name)
        body = fn_lit.body
        inner: TExpr | None = None
        if len(body) == 1:
            stmt0 = body[0]
            if isinstance(stmt0, TReturnStmt) and stmt0.value is not None:
                inner = stmt0.value
            elif isinstance(stmt0, TExprStmt):
                inner = stmt0.expr
        if inner is not None:
            ref = self._try_method_ref(fn_lit.params[0], inner)
            if ref is not None:
                return "Comparator.comparingInt(" + ref + ")"
            return "Comparator.comparingInt(" + param + " -> " + self._expr(inner) + ")"
        return "Comparator.comparingInt(" + self._fn_lit(fn_lit) + ")"

    def _try_method_ref(self, param: TParam, body_expr: TExpr) -> str | None:
        """Try to emit a method reference like String::length."""
        if not isinstance(body_expr, TCall):
            return None
        be_func = body_expr.func
        if not isinstance(be_func, TVar):
            return None
        if be_func.name != "Len" or len(body_expr.args) != 1:
            return None
        arg = body_expr.args[0].value
        if not (isinstance(arg, TVar) and arg.name == param.name):
            return None
        if (
            param.typ is not None
            and isinstance(param.typ, TPrimitive)
            and param.typ.kind == "string"
        ):
            return "String::length"
        return None

    def _join_args(self, args: list[TArg], sep: str) -> str:
        return sep.join(self._expr(a.value) for a in args)

    def _join_exprs(self, exprs: list[TExpr], sep: str) -> str:
        return sep.join(self._expr(e) for e in exprs)

    def _fold_temp_assign(
        self,
        stmts: list[TStmt],
        i: int,
    ) -> bool:
        raise NotImplementedError

    def _strip_not(self, expr: TExpr) -> TExpr:
        if isinstance(expr, TUnaryOp) and expr.op in ("!", "not"):
            return expr.operand
        if (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "not"
            and len(expr.args) == 1
        ):
            return expr.args[0].value
        return expr

    # ── Zip / Reversed / Map iteration ───────────────────────

    def _emit_for_zip_impl(self, stmt: TForStmt, binding: list[str]) -> None:
        assert isinstance(stmt.iterable, TCall)
        zip_args = stmt.iterable.args
        sources = [self._expr(a.value) for a in zip_args]
        size_exprs: list[str] = []
        for i, a in enumerate(zip_args):
            if self._is_bytes_expr(a.value):
                size_exprs.append(sources[i] + ".length")
            else:
                size_exprs.append(sources[i] + ".size()")
        if len(size_exprs) == 1:
            limit = size_exprs[0]
        else:
            limit = size_exprs[-1]
            for s in reversed(size_exprs[:-1]):
                limit = "Math.min(" + s + ", " + limit + ")"
        self._line("for (int __i = 0; __i < " + limit + "; __i++) {")
        self.indent += 1
        for i, b in enumerate(binding):
            ann = stmt.annotations
            elem_type = ann.get("zip_type_" + str(i), "")
            type_str = self._zip_elem_type(zip_args[i].value, elem_type)
            if self._is_bytes_expr(zip_args[i].value):
                self._line(type_str + " " + b + " = " + sources[i] + "[__i];")
            else:
                self._line(type_str + " " + b + " = " + sources[i] + ".get(__i);")
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _zip_elem_type(self, source: TExpr, type_hint: str) -> str:
        if self._is_bytes_expr(source):
            return "int"
        if type_hint == "int":
            return "int"
        if type_hint == "float":
            return "double"
        if type_hint == "bool":
            return "boolean"
        if type_hint == "string":
            return "String"
        result = self._type_from_ann(source)
        if result is not None:
            return result
        return "var"

    def _type_from_ann(self, expr: TExpr) -> str | None:
        ann = expr.annotations.get("type", "")
        if ann.startswith("list["):
            inner = ann[5:-1]
            m = {"int": "int", "float": "double", "bool": "boolean", "string": "String"}
            return m.get(inner)
        return None

    def _is_zip_for(self, stmt: TForStmt) -> bool:
        raise NotImplementedError

    def _emit_for_zip(self, stmt: TForStmt, binding: list[str], ann: Ann) -> list[str]:
        raise NotImplementedError

    def _for_iterable(self, iterable: TExpr) -> str:
        raise NotImplementedError

    def _emit_for_keys(self, stmt: TForStmt, binding: list[str], ann: Ann) -> None:
        raise NotImplementedError

    def _is_map_for(self, stmt: TForStmt) -> bool:
        raise NotImplementedError

    def _is_builtin_call(self, expr: TExpr, name: str) -> bool:
        raise NotImplementedError

    def _step_slice_source(
        self, stmt: TStmt, acc_name: str
    ) -> tuple[bool, TExpr | None]:
        """Extract (is_string, source_obj) from a step_slice loop body."""
        if isinstance(stmt, TExprStmt):
            call = stmt.expr
            if isinstance(call, TCall) and self._is_append_to(call, acc_name):
                elem = call.args[1].value
                if isinstance(elem, TIndex):
                    return False, elem.obj
        if isinstance(stmt, TAssignStmt) and isinstance(stmt.target, TVar):
            if stmt.target.name == acc_name and isinstance(stmt.value, TCall):
                if (
                    isinstance(stmt.value.func, TVar)
                    and stmt.value.func.name == "Concat"
                ):
                    second = stmt.value.args[1].value
                    if (
                        isinstance(second, TCall)
                        and isinstance(second.func, TVar)
                        and second.func.name == "ToString"
                    ):
                        inner = second.args[0].value
                        if isinstance(inner, TIndex):
                            return True, inner.obj
        return False, None

    def _step_slice_is_string(self, stmt: TStmt, acc_name: str) -> bool:
        is_string, _ = self._step_slice_source(stmt, acc_name)
        return is_string

    def _is_len_of(self, expr: TExpr, obj: TExpr) -> bool:
        raise NotImplementedError


# ============================================================
# PUBLIC API
# ============================================================


def _visit_thrown_types(stmts: list[TStmt], names: set[str]) -> None:
    for stmt in stmts:
        if isinstance(stmt, TThrowStmt):
            throw_expr = stmt.expr
            if isinstance(throw_expr, TCall):
                throw_func = throw_expr.func
                if isinstance(throw_func, TVar):
                    names.add(throw_func.name)
        elif isinstance(stmt, TTryStmt):
            _visit_thrown_types(stmt.body, names)
            for catch in stmt.catches:
                for t in catch.types:
                    if isinstance(t, TIdentType):
                        names.add(t.name)
                _visit_thrown_types(catch.body, names)
            if stmt.finally_body is not None:
                _visit_thrown_types(stmt.finally_body, names)
        elif isinstance(stmt, TIfStmt):
            _visit_thrown_types(stmt.then_body, names)
            if stmt.else_body is not None:
                _visit_thrown_types(stmt.else_body, names)
        elif isinstance(stmt, TWhileStmt):
            _visit_thrown_types(stmt.body, names)
        elif isinstance(stmt, TForStmt):
            _visit_thrown_types(stmt.body, names)
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                _visit_thrown_types(case.body, names)
            if stmt.default is not None:
                _visit_thrown_types(stmt.default.body, names)


def _expr_uses_builtin(expr: TExpr, name: str) -> bool:
    """Check if an expression tree contains a call to the given builtin."""
    if isinstance(expr, TCall):
        if isinstance(expr.func, TVar) and expr.func.name == name:
            return True
        if _expr_uses_builtin(expr.func, name):
            return True
        for arg in expr.args:
            if _expr_uses_builtin(arg.value, name):
                return True
    if isinstance(expr, TBinaryOp):
        return _expr_uses_builtin(expr.left, name) or _expr_uses_builtin(
            expr.right, name
        )
    if isinstance(expr, TUnaryOp):
        return _expr_uses_builtin(expr.operand, name)
    if isinstance(expr, TIndex):
        return _expr_uses_builtin(expr.obj, name) or _expr_uses_builtin(
            expr.index, name
        )
    if isinstance(expr, TTernary):
        return (
            _expr_uses_builtin(expr.cond, name)
            or _expr_uses_builtin(expr.then_expr, name)
            or _expr_uses_builtin(expr.else_expr, name)
        )
    return False


def _stmts_use_builtin(stmts: list[TStmt], name: str) -> bool:
    """Check if any statement in a list uses the given builtin."""
    for stmt in stmts:
        if isinstance(stmt, TExprStmt) and _expr_uses_builtin(stmt.expr, name):
            return True
        if (
            isinstance(stmt, TLetStmt)
            and stmt.value is not None
            and _expr_uses_builtin(stmt.value, name)
        ):
            return True
        if isinstance(stmt, TAssignStmt) and _expr_uses_builtin(stmt.value, name):
            return True
        if (
            isinstance(stmt, TReturnStmt)
            and stmt.value is not None
            and _expr_uses_builtin(stmt.value, name)
        ):
            return True
        if isinstance(stmt, TIfStmt):
            if _stmts_use_builtin(stmt.then_body, name):
                return True
            if stmt.else_body is not None and _stmts_use_builtin(stmt.else_body, name):
                return True
        if isinstance(stmt, TForStmt) and _stmts_use_builtin(stmt.body, name):
            return True
        if isinstance(stmt, TWhileStmt) and _stmts_use_builtin(stmt.body, name):
            return True
        if isinstance(stmt, TTryStmt):
            if _stmts_use_builtin(stmt.body, name):
                return True
            for catch in stmt.catches:
                if _stmts_use_builtin(catch.body, name):
                    return True
            if stmt.finally_body is not None and _stmts_use_builtin(
                stmt.finally_body, name
            ):
                return True
        if isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                if _stmts_use_builtin(case.body, name):
                    return True
            if stmt.default is not None and _stmts_use_builtin(stmt.default.body, name):
                return True
    return False


def _module_uses_builtin(module: TModule, name: str) -> bool:
    """Check if any function in the module uses the given builtin."""
    for decl in module.decls:
        if isinstance(decl, TFnDecl) and _stmts_use_builtin(decl.body, name):
            return True
        if isinstance(decl, TStructDecl):
            for m in decl.methods:
                if _stmts_use_builtin(m.body, name):
                    return True
    return False


def _collect_thrown_types(module: TModule) -> set[str]:
    """Find struct names used in throw or catch statements."""
    names: set[str] = set()
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            _visit_thrown_types(decl.body, names)
        if isinstance(decl, TStructDecl):
            for m in decl.methods:
                _visit_thrown_types(m.body, names)
    return names


def emit_java(module: TModule) -> str:
    struct_names: set[str] = set(BUILTIN_STRUCTS.keys())
    struct_fields: dict[str, list[str]] = {}
    struct_field_decls: dict[str, list[TFieldDecl]] = {}
    enum_names: set[str] = set()
    interface_names: set[str] = set()
    error_struct_names: set[str] = set()
    for decl in module.decls:
        if isinstance(decl, TStructDecl):
            struct_names.add(decl.name)
            fnames: list[str] = []
            for f in decl.fields:
                fnames.append(_safe_name(f.name))
            struct_fields[decl.name] = fnames
            struct_field_decls[decl.name] = decl.fields
            if decl.parent == "Error":
                error_struct_names.add(decl.name)
        elif isinstance(decl, TInterfaceDecl):
            struct_names.add(decl.name)
            interface_names.add(decl.name)
            if decl.fields:
                ifnames: list[str] = []
                for f in decl.fields:
                    ifnames.append(_safe_name(f.name))
                struct_fields[decl.name] = ifnames
                struct_field_decls[decl.name] = decl.fields
        elif isinstance(decl, TEnumDecl):
            enum_names.add(decl.name)
    user_struct_names: set[str] = set()
    for d in module.decls:
        if isinstance(d, TStructDecl):
            user_struct_names.add(d.name)
    error_struct_names |= _collect_thrown_types(module) & user_struct_names
    emitter = _JavaEmitter(
        struct_names, struct_fields, module.strict_math, module.strict_tostring
    )
    emitter._struct_field_decls = struct_field_decls
    emitter._enum_names = enum_names
    emitter._interface_names = interface_names
    emitter._error_struct_names = error_struct_names
    emitter.emit_module(module)
    return emitter.output()
