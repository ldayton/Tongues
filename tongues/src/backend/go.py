"""Go backend: Taytsh AST → Go source code."""

from __future__ import annotations

from .util import (
    STRICT_INT_BINARY,
    STRICT_INT_COMPOUND,
    Emitter,
    collect_builtin_calls,
    escape_string,
)
from ..taytsh.ast import (
    Ann,
    Pos,
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
    TMatchCase,
    TMatchStmt,
    TModule,
    TNilLit,
    TOpAssignStmt,
    TOptionalType,
    TParam,
    TPattern,
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
    TTryStmt,
    TTupleType,
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
# GO RESERVED WORDS
# ============================================================

_GO_RESERVED = frozenset(
    {
        "break",
        "case",
        "chan",
        "const",
        "continue",
        "default",
        "defer",
        "else",
        "fallthrough",
        "for",
        "func",
        "go",
        "goto",
        "if",
        "import",
        "interface",
        "map",
        "package",
        "range",
        "return",
        "select",
        "struct",
        "switch",
        "type",
        "var",
        "true",
        "false",
        "nil",
        "len",
        "cap",
        "make",
        "new",
        "append",
        "delete",
        "copy",
        "min",
        "max",
        "close",
        "panic",
        "recover",
        "print",
        "println",
        "error",
        "string",
        "int",
        "float64",
        "bool",
        "byte",
        "rune",
        "any",
        "iota",
    }
)


def _safe_name(name: str) -> str:
    if name in _GO_RESERVED:
        return name + "_"
    return name


def _to_lower_camel(name: str) -> str:
    """Convert PascalCase to lowerCamelCase."""
    if not name or name[0].islower():
        return name
    return name[0].lower() + name[1:]


def _split_tuple_ann(inner: str) -> list[str]:
    """Split tuple annotation by top-level commas, respecting brackets."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i in range(len(inner)):
        ch = inner[i]
        if ch in ("[", "("):
            depth += 1
        elif ch in ("]", ")"):
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[start:i].strip())
            start = i + 1
    parts.append(inner[start:].strip())
    return parts


def _restore_name(name: str, annotations: Ann) -> str:
    key = "name.original." + name
    if key in annotations:
        return _safe_name(annotations[key])
    return _safe_name(name)


# ============================================================
# OPERATOR MAPS
# ============================================================

_PRECEDENCE: dict[str, int] = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 3,
    ">": 3,
    "<=": 3,
    ">=": 3,
    "|": 4,
    "^": 5,
    "&": 6,
    "<<": 7,
    ">>": 7,
    "+": 8,
    "-": 8,
    "*": 9,
    "/": 9,
    "%": 9,
}

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

_FMT_BUILTINS = frozenset(
    {
        "WritelnOut",
        "WriteOut",
        "WritelnErr",
        "WriteErr",
        "Format",
        "ToString",
    }
)

_OS_BUILTINS = frozenset(
    {
        "ReadFile",
        "ReadFileBytes",
        "WriteFile",
        "Args",
        "Exit",
        "GetEnv",
    }
)

_STRCONV_BUILTINS = frozenset(
    {
        "ToString",
        "ParseInt",
        "ParseFloat",
        "FormatInt",
    }
)

_STRINGS_BUILTINS = frozenset(
    {
        "Split",
        "SplitN",
        "Join",
        "Find",
        "RFind",
        "Replace",
        "ReplaceCount",
        "Upper",
        "Lower",
        "Trim",
        "TrimStart",
        "TrimEnd",
        "Count",
        "StartsWith",
        "EndsWith",
        "Repeat",
        "Contains",
        "SplitWhitespace",
    }
)

_MATH_BUILTINS = frozenset(
    {
        "Sqrt",
        "Floor",
        "Ceil",
        "Abs",
        "IsNaN",
        "IsInf",
        "Pow",
    }
)

_SLICES_BUILTINS = frozenset(
    {
        "Sorted",
        "SortBy",
        "Reversed",
        "Reverse",
        "IndexOf",
        "Contains",
        "RemoveAt",
        "Insert",
        "MinBy",
        "MaxBy",
    }
)

_UNICODE_BUILTINS = frozenset(
    {
        "IsDigit",
        "IsAlpha",
        "IsAlnum",
        "IsSpace",
        "IsUpper",
        "IsLower",
    }
)

_IO_BUILTINS = frozenset({"ReadAll"})

_BUFIO_BUILTINS = frozenset({"ReadLine"})

_ERRORS_BUILTINS = frozenset({"Exception"})


def _scan_imports(module: TModule) -> set[str]:
    """Scan module to determine which Go packages are needed."""
    all_builtins: set[str] = set()
    has_try = False
    has_format_call = False
    has_tostring = False

    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            all_builtins |= collect_builtin_calls(decl.body)
            has_try = has_try or _has_try_stmt(decl.body)
        elif isinstance(decl, TStructDecl):
            for m in decl.methods:
                all_builtins |= collect_builtin_calls(m.body)
                has_try = has_try or _has_try_stmt(m.body)
        elif isinstance(decl, TStmt):
            all_builtins |= collect_builtin_calls([decl])

    pkgs: set[str] = set()
    if all_builtins & _FMT_BUILTINS:
        pkgs.add("fmt")
    if all_builtins & _OS_BUILTINS:
        pkgs.add("os")
    if all_builtins & _STRCONV_BUILTINS:
        pkgs.add("strconv")
    if all_builtins & _STRINGS_BUILTINS:
        pkgs.add("strings")
    if all_builtins & _MATH_BUILTINS:
        pkgs.add("math")
    if all_builtins & _SLICES_BUILTINS:
        pkgs.add("slices")
    if all_builtins & _UNICODE_BUILTINS:
        pkgs.add("unicode")
    if all_builtins & _IO_BUILTINS:
        pkgs.add("io")
        pkgs.add("os")
    if all_builtins & _BUFIO_BUILTINS:
        pkgs.add("bufio")
        pkgs.add("os")
    if "Exception" in all_builtins:
        pkgs.add("errors")
    if "ReadAll" in all_builtins:
        pkgs.add("io")
        pkgs.add("os")
    if _has_bytes_compare(module):
        pkgs.add("bytes")
    return pkgs


def _has_bytes_compare(module: TModule) -> bool:
    """Check if module has byte slice comparisons needing bytes.Equal."""
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            if _stmts_have_bytes_compare(decl.body):
                return True
        elif isinstance(decl, TStructDecl):
            for m in decl.methods:
                if _stmts_have_bytes_compare(m.body):
                    return True
    return False


def _stmts_have_bytes_compare(stmts: list[TStmt]) -> bool:
    for s in stmts:
        if isinstance(s, TLetStmt) and s.value is not None:
            if _expr_has_bytes_compare(s.value):
                return True
        elif isinstance(s, TReturnStmt) and s.value is not None:
            if _expr_has_bytes_compare(s.value):
                return True
        elif isinstance(s, TExprStmt):
            if _expr_has_bytes_compare(s.expr):
                return True
        elif isinstance(s, TIfStmt):
            if _expr_has_bytes_compare(s.cond):
                return True
            if _stmts_have_bytes_compare(s.then_body):
                return True
            if s.else_body is not None and _stmts_have_bytes_compare(s.else_body):
                return True
        elif isinstance(s, TWhileStmt):
            if _stmts_have_bytes_compare(s.body):
                return True
        elif isinstance(s, TForStmt):
            if _stmts_have_bytes_compare(s.body):
                return True
    return False


def _expr_has_bytes_compare(expr: TExpr) -> bool:
    if isinstance(expr, TBinaryOp):
        if expr.op in ("==", "!="):
            if isinstance(expr.left, TBytesLit) or isinstance(expr.right, TBytesLit):
                return True
        return _expr_has_bytes_compare(expr.left) or _expr_has_bytes_compare(expr.right)
    return False


def _stmts_ref_var(stmts: list[TStmt], name: str) -> bool:
    """Check if any statement in the list references a variable by name."""
    for s in stmts:
        if isinstance(s, TLetStmt):
            if s.value is not None and _expr_ref_var(s.value, name):
                return True
        elif isinstance(s, TReturnStmt):
            if s.value is not None and _expr_ref_var(s.value, name):
                return True
        elif isinstance(s, TExprStmt):
            if _expr_ref_var(s.expr, name):
                return True
        elif isinstance(s, TAssignStmt):
            if _expr_ref_var(s.value, name):
                return True
            if isinstance(s.target, TVar) and s.target.name == name:
                return True
            if _expr_ref_var(s.target, name):
                return True
        elif isinstance(s, TIfStmt):
            if _expr_ref_var(s.cond, name):
                return True
            if _stmts_ref_var(s.then_body, name):
                return True
            if s.else_body is not None and _stmts_ref_var(s.else_body, name):
                return True
        elif isinstance(s, TForStmt):
            if _expr_ref_var(s.iterable, name):
                return True
            if _stmts_ref_var(s.body, name):
                return True
        elif isinstance(s, TWhileStmt):
            if _expr_ref_var(s.cond, name):
                return True
            if _stmts_ref_var(s.body, name):
                return True
        elif isinstance(s, TThrowStmt):
            if _expr_ref_var(s.expr, name):
                return True
        elif isinstance(s, TMatchStmt):
            if _expr_ref_var(s.expr, name):
                return True
            for c in s.cases:
                if _stmts_ref_var(c.body, name):
                    return True
            if s.default is not None and _stmts_ref_var(s.default.body, name):
                return True
        elif isinstance(s, TTryStmt):
            if _stmts_ref_var(s.body, name):
                return True
            for c in s.catches:
                if _stmts_ref_var(c.body, name):
                    return True
            if s.finally_body is not None and _stmts_ref_var(s.finally_body, name):
                return True
        elif isinstance(s, TOpAssignStmt):
            if _expr_ref_var(s.target, name) or _expr_ref_var(s.value, name):
                return True
        elif isinstance(s, TTupleAssignStmt):
            if _expr_ref_var(s.value, name):
                return True
            for t in s.targets:
                if _expr_ref_var(t, name):
                    return True
    return False


def _expr_ref_var(expr: TExpr, name: str) -> bool:
    if isinstance(expr, TVar):
        return expr.name == name
    if isinstance(expr, TFieldAccess):
        return _expr_ref_var(expr.obj, name)
    if isinstance(expr, TCall):
        if _expr_ref_var(expr.func, name):
            return True
        for a in expr.args:
            if _expr_ref_var(a.value, name):
                return True
    if isinstance(expr, TBinaryOp):
        return _expr_ref_var(expr.left, name) or _expr_ref_var(expr.right, name)
    if isinstance(expr, TUnaryOp):
        return _expr_ref_var(expr.operand, name)
    if isinstance(expr, TIndex):
        return _expr_ref_var(expr.obj, name) or _expr_ref_var(expr.index, name)
    if isinstance(expr, TListLit):
        for e in expr.elements:
            if _expr_ref_var(e, name):
                return True
    if isinstance(expr, TSlice):
        if _expr_ref_var(expr.obj, name):
            return True
    if isinstance(expr, TTernary):
        return (
            _expr_ref_var(expr.cond, name)
            or _expr_ref_var(expr.then_expr, name)
            or _expr_ref_var(expr.else_expr, name)
        )
    if isinstance(expr, TTupleAccess):
        return _expr_ref_var(expr.obj, name)
    if isinstance(expr, TMapLit):
        for k, v in expr.entries:
            if _expr_ref_var(k, name) or _expr_ref_var(v, name):
                return True
    if isinstance(expr, (TSetLit, TTupleLit)):
        for e in expr.elements:
            if _expr_ref_var(e, name):
                return True
    if isinstance(expr, TRange):
        for a in expr.args:
            if _expr_ref_var(a, name):
                return True
    if isinstance(expr, TFnLit):
        return _stmts_ref_var(expr.body, name)
    return False


def _has_try_stmt(stmts: list[TStmt]) -> bool:
    for s in stmts:
        if isinstance(s, TTryStmt):
            return True
        if isinstance(s, TIfStmt):
            if _has_try_stmt(s.then_body):
                return True
            if s.else_body is not None and _has_try_stmt(s.else_body):
                return True
        if isinstance(s, (TWhileStmt, TForStmt)):
            if _has_try_stmt(s.body):
                return True
        if isinstance(s, TMatchStmt):
            for c in s.cases:
                if _has_try_stmt(c.body):
                    return True
            if s.default and _has_try_stmt(s.default.body):
                return True
    return False


# ============================================================
# STRICT MATH HELPERS
# ============================================================

_STRICT_MATH_HELPERS: dict[str, str] = {
    "checked_add_i64": """\
func checkedAddI64(a, b int) int {
\tif (b > 0 && a > 9223372036854775807-b) || (b < 0 && a < -9223372036854775808-b) {
\t\tpanic("integer overflow")
\t}
\treturn a + b
}""",
    "checked_sub_i64": """\
func checkedSubI64(a, b int) int {
\tif (b < 0 && a > 9223372036854775807+b) || (b > 0 && a < -9223372036854775808+b) {
\t\tpanic("integer overflow")
\t}
\treturn a - b
}""",
    "checked_mul_i64": """\
func checkedMulI64(a, b int) int {
\tif a == 0 || b == 0 {
\t\treturn 0
\t}
\tr := a * b
\tif r/a != b {
\t\tpanic("integer overflow")
\t}
\treturn r
}""",
    "checked_div_i64": """\
func checkedDivI64(a, b int) int {
\tif b == 0 {
\t\tpanic("division by zero")
\t}
\tif a == -9223372036854775808 && b == -1 {
\t\tpanic("integer overflow")
\t}
\treturn a / b
}""",
    "checked_rem_i64": """\
func checkedRemI64(a, b int) int {
\tif b == 0 {
\t\tpanic("division by zero")
\t}
\treturn a % b
}""",
    "checked_neg_i64": """\
func checkedNegI64(a int) int {
\tif a == -9223372036854775808 {
\t\tpanic("integer overflow")
\t}
\treturn -a
}""",
    "checked_shl_i64": """\
func checkedShlI64(a, b int) int {
\tif b < 0 || b > 63 {
\t\tpanic("shift out of range")
\t}
\tr := a << uint(b)
\tif r >> uint(b) != a {
\t\tpanic("integer overflow")
\t}
\treturn r
}""",
    "checked_shr_i64": """\
func checkedShrI64(a, b int) int {
\tif b < 0 || b > 63 {
\t\tpanic("shift out of range")
\t}
\treturn a >> uint(b)
}""",
    "logical_shr_i64": """\
func logicalShrI64(a, b int) int {
\tif b < 0 || b > 63 {
\t\tpanic("shift out of range")
\t}
\treturn int(uint64(a) >> uint(b))
}""",
    "checked_pow_i64": """\
func checkedPowI64(a, b int) int {
\tif b < 0 {
\t\tpanic("negative exponent")
\t}
\tr := 1
\tbase := a
\tfor i := 0; i < b; i++ {
\t\told := r
\t\tr *= base
\t\tif base != 0 && r/base != old {
\t\t\tpanic("integer overflow")
\t\t}
\t}
\treturn r
}""",
    "wrapping_add": """\
func wrappingAdd(a, b int) int {
\treturn int(int64(a) + int64(b))
}""",
    "wrapping_sub": """\
func wrappingSub(a, b int) int {
\treturn int(int64(a) - int64(b))
}""",
    "wrapping_mul": """\
func wrappingMul(a, b int) int {
\treturn int(int64(a) * int64(b))
}""",
    "strict_fmod": """\
func strictFmod(a, b float64) float64 {
\tif b == 0.0 {
\t\tpanic("float modulo by zero")
\t}
\treturn math.Remainder(a, b) + b*0
}""",
    "strict_min_f64": """\
func strictMinF64(a, b float64) float64 {
\tif math.IsNaN(a) || math.IsNaN(b) {
\t\treturn math.NaN()
\t}
\treturn math.Min(a, b)
}""",
    "strict_max_f64": """\
func strictMaxF64(a, b float64) float64 {
\tif math.IsNaN(a) || math.IsNaN(b) {
\t\treturn math.NaN()
\t}
\treturn math.Max(a, b)
}""",
    "strict_sorted_f64": """\
func strictSortedF64(xs []float64) []float64 {
\tfor _, x := range xs {
\t\tif math.IsNaN(x) {
\t\t\tpanic("NaN in sort")
\t\t}
\t}
\tout := make([]float64, len(xs))
\tcopy(out, xs)
\tslices.Sort(out)
\treturn out
}""",
}

# Map from util.py helper names to Go function names
_STRICT_GO_BINARY: dict[str, str] = {
    "+": "checkedAddI64",
    "-": "checkedSubI64",
    "*": "checkedMulI64",
    "/": "checkedDivI64",
    "%": "checkedRemI64",
    "<<": "checkedShlI64",
    ">>": "checkedShrI64",
    ">>>": "logicalShrI64",
}

_STRICT_GO_COMPOUND: dict[str, str] = {
    "+=": "checkedAddI64",
    "-=": "checkedSubI64",
    "*=": "checkedMulI64",
}


# ============================================================
# EMITTER
# ============================================================


class _GoEmitter(Emitter):
    def __init__(
        self,
        struct_names: set[str],
        struct_fields: dict[str, list[str]],
        struct_field_types: dict[str, list[TFieldDecl]],
        strict_math: bool = False,
        strict_tostring: bool = False,
    ) -> None:
        self.struct_names = struct_names
        self.struct_fields = struct_fields
        self.struct_field_types = struct_field_types
        self.strict_math = strict_math
        self.strict_tostring = strict_tostring
        self.indent: int = 0
        self.lines: list[str] = []
        self.self_name: str | None = None
        self.var_types: dict[str, TType] = {}
        self.module_let_names: set[str] = set()
        self._current_struct: str = ""
        self.fn_names: set[str] = set()
        self._used_strict_helpers: set[str] = set()
        self._enum_names: set[str] = set()
        self._enum_type_map: dict[str, str] = {}
        self._interface_children: dict[str, list[str]] = {}
        self._interface_names: set[str] = set()
        self._interface_parent: dict[str, str] = {}
        self._interface_common_fields: dict[str, list[TFieldDecl]] = {}
        self._error_structs: set[str] = set()
        self._pointer_structs: set[str] = set()
        self._var_aliases: dict[str, str] = {}
        self._tuple_unpack_counter: int = 0
        self._need_reverse_string: bool = False
        self._in_func: bool = False
        self._current_ret_type: TType | None = None
        self._try_action_var: str | None = None
        self._try_result_var: str | None = None
        self._fn_params: dict[str, list[TParam]] = {}
        self._interface_methods: dict[str, list[TFnDecl]] = {}
        self._suppress_deref: bool = False

    def _line(self, text: str = "") -> None:
        if text:
            self.lines.append("\t" * self.indent + text)
        else:
            self.lines.append("")

    # ── Module ────────────────────────────────────────────────

    def emit_module(self, module: TModule) -> None:
        for decl in module.decls:
            if isinstance(decl, TLetStmt):
                self.module_let_names.add(decl.name)
            if isinstance(decl, TFnDecl):
                self.fn_names.add(decl.name)
                self._fn_params[decl.name] = decl.params
            if isinstance(decl, TStructDecl):
                for m in decl.methods:
                    self.fn_names.add(m.name)
                    self._fn_params[decl.name + "." + m.name] = m.params
                is_error = decl.name in BUILTIN_STRUCTS
                if not is_error and decl.parent is not None:
                    if decl.parent in BUILTIN_STRUCTS:
                        is_error = True
                if not is_error and decl.annotations.get("_is_exception") is not None:
                    is_error = True
                if is_error:
                    self._error_structs.add(decl.name)
            if isinstance(decl, TEnumDecl):
                self._enum_names.add(decl.name)
                for v in decl.variants:
                    self._enum_type_map[decl.name + v] = decl.name
            if isinstance(decl, TInterfaceDecl):
                self._interface_names.add(decl.name)
                self._interface_children[decl.name] = []
        # Build interface parent chain from _parent_interface annotations
        self._interface_parent: dict[str, str] = {}
        for decl in module.decls:
            if isinstance(decl, TInterfaceDecl):
                p = decl.annotations.get("_parent_interface", "")
                if p != "" and p in self._interface_names:
                    self._interface_parent[decl.name] = p
        for decl in module.decls:
            if isinstance(decl, TStructDecl) and decl.parent is not None:
                if decl.parent in self._interface_children:
                    self._interface_children[decl.parent].append(decl.name)
                    self._pointer_structs.add(decl.name)
        self._pointer_structs.update(self._error_structs)

        # Scan child structs to find common methods for interfaces
        self._interface_methods: dict[str, list[TFnDecl]] = {}
        for iface_name in self._interface_children:
            children = self._interface_children[iface_name]
            if not children:
                continue
            # Collect methods per child struct
            child_methods: dict[str, list[TFnDecl]] = {}
            for decl in module.decls:
                if isinstance(decl, TStructDecl) and decl.name in children:
                    for m in decl.methods:
                        if m.name not in child_methods:
                            child_methods[m.name] = []
                        child_methods[m.name].append(m)
            # Methods present in ALL children
            common: list[TFnDecl] = []
            for mname in child_methods:
                if len(child_methods[mname]) == len(children):
                    common.append(child_methods[mname][0])
            self._interface_methods[iface_name] = common

        # Scan child structs to find common fields for interface accessors
        self._interface_common_fields: dict[str, list[TFieldDecl]] = {}
        for iface_name in self._interface_children:
            children = self._interface_children[iface_name]
            if not children:
                continue
            field_counts: dict[str, TFieldDecl] = {}
            field_present: dict[str, int] = {}
            for decl in module.decls:
                if isinstance(decl, TStructDecl) and decl.name in children:
                    for fld in decl.fields:
                        if fld.name not in field_counts:
                            field_counts[fld.name] = fld
                            field_present[fld.name] = 0
                        field_present[fld.name] += 1
            common_fields: list[TFieldDecl] = []
            for fname in field_counts:
                if field_present[fname] == len(children):
                    common_fields.append(field_counts[fname])
            self._interface_common_fields[iface_name] = common_fields

        # Collect error struct refs for built-in error types
        declared_structs: set[str] = set()
        for decl in module.decls:
            if isinstance(decl, TStructDecl):
                declared_structs.add(decl.name)
        builtin_error_refs, all_error_refs = _collect_error_refs(module)
        referenced_errors: list[str] = sorted(
            list(builtin_error_refs - declared_structs)
        )
        # Mark user-defined structs used in throw/catch as error structs
        for name in all_error_refs:
            if name in declared_structs:
                self._error_structs.add(name)

        # Emit package + imports
        self._line("package main")
        self._line()
        pkgs = _scan_imports(module)
        if referenced_errors:
            pkgs.add("errors")
        if pkgs:
            if len(pkgs) == 1:
                self._line('import "' + sorted(pkgs)[0] + '"')
            else:
                self._line("import (")
                self.indent += 1
                for p in sorted(pkgs):
                    self._line('"' + p + '"')
                self.indent -= 1
                self._line(")")
            self._line()

        # Emit referenced built-in error structs
        for ename in referenced_errors:
            self._line("type " + ename + " struct {")
            self.indent += 1
            self._line("Message string")
            self.indent -= 1
            self._line("}")
            self._line()
            self._line("func (e *" + ename + ") Error() string {")
            self.indent += 1
            self._line("return e.Message")
            self.indent -= 1
            self._line("}")
            self._line()
            self._error_structs.add(ename)

        # Emit declarations
        need_blank = False
        for decl in module.decls:
            if isinstance(decl, TInterfaceDecl):
                if need_blank:
                    self._line()
                self._emit_interface(decl)
                need_blank = True
                continue
            if need_blank:
                self._line()
            match decl:
                case TStructDecl():
                    self._emit_struct(decl)
                case TEnumDecl():
                    self._emit_enum(decl)
                case TLetStmt():
                    self._emit_let(decl)
                case TFnDecl():
                    self._emit_fn(decl)
            need_blank = True
        if self._need_reverse_string:
            self._line()
            self._line("func reverseString(s string) string {")
            self.indent += 1
            self._line("runes := []rune(s)")
            self._line("for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {")
            self.indent += 1
            self._line("runes[i], runes[j] = runes[j], runes[i]")
            self.indent -= 1
            self._line("}")
            self._line("return string(runes)")
            self.indent -= 1
            self._line("}")

    # ── Enum ──────────────────────────────────────────────────

    def _emit_enum(self, decl: TEnumDecl) -> None:
        self._line("type " + decl.name + " int")
        self._line()
        self._line("const (")
        self.indent += 1
        for i, variant in enumerate(decl.variants):
            if i == 0:
                self._line(decl.name + variant + " " + decl.name + " = iota")
            else:
                self._line(decl.name + variant)
        self.indent -= 1
        self._line(")")

    # ── Struct ────────────────────────────────────────────────

    def _emit_struct(self, decl: TStructDecl) -> None:
        old_struct = self._current_struct
        self._current_struct = decl.name
        is_error = decl.name in self._error_structs
        self._line("type " + decl.name + " struct {")
        self.indent += 1
        if decl.parent is not None:
            if decl.parent not in self._interface_names:
                self._line(decl.parent)
        for fld in decl.fields:
            fname = fld.name[0].upper() + fld.name[1:] if fld.name else fld.name
            self._line(fname + " " + self._type(fld.typ))
        self.indent -= 1
        self._line("}")
        if decl.parent is not None:
            if decl.parent in self._interface_names:
                # Emit marker for direct parent and all ancestor interfaces
                iface = decl.parent
                while iface != "":
                    self._line()
                    self._line("func (self " + decl.name + ") is" + iface + "() {}")
                    iface = self._interface_parent.get(iface, "")
        if decl.parent is not None and decl.parent in self._interface_names:
            common_fields = self._interface_common_fields.get(decl.parent, [])
            for fld in common_fields:
                fname = fld.name[0].upper() + fld.name[1:] if fld.name else fld.name
                self._line()
                self._line(
                    "func (self *"
                    + decl.name
                    + ") Get"
                    + fname
                    + "() "
                    + self._type(fld.typ)
                    + " {"
                )
                self.indent += 1
                self._line("return self." + fname)
                self.indent -= 1
                self._line("}")
        if is_error:
            self._line()
            msg_field: TFieldDecl | None = None
            for fld in decl.fields:
                if fld.name in ("message", "msg"):
                    msg_field = fld
                    break
            if isinstance(msg_field, TFieldDecl):
                fname = msg_field.name[0].upper() + msg_field.name[1:]
                self._line("func (e *" + decl.name + ") Error() string {")
                self.indent += 1
                self._line("return e." + fname)
                self.indent -= 1
                self._line("}")
            else:
                self._line("func (e *" + decl.name + ") Error() string {")
                self.indent += 1
                self._line('return "' + decl.name + '"')
                self.indent -= 1
                self._line("}")
        for method in decl.methods:
            self._line()
            self._emit_method(method, decl.name)
        self._current_struct = old_struct

    def _emit_interface(self, decl: TInterfaceDecl) -> None:
        self._line("type " + decl.name + " interface {")
        self.indent += 1
        parent_iface = decl.annotations.get("_parent_interface", "")
        if parent_iface != "" and parent_iface in self._interface_names:
            self._line(parent_iface)
        self._line("is" + decl.name + "()")
        for fld in decl.fields:
            if isinstance(fld.typ, TFuncType):
                mname = fld.name
                fn_params = fld.typ.params
                if len(fn_params) > 0:
                    param_types: list[str] = []
                    i = 0
                    while i < len(fn_params) - 1:
                        param_types.append(self._type(fn_params[i]))
                        i += 1
                    ret = self._type(fn_params[len(fn_params) - 1])
                    sig = mname + "(" + ", ".join(param_types) + ")"
                    if ret != "":
                        sig += " " + ret
                    self._line(sig)
                else:
                    self._line(mname + "()")
        emitted: set[str] = set()
        for fld in decl.fields:
            emitted.add(fld.name)
        common_methods = self._interface_methods.get(decl.name, [])
        for m in common_methods:
            if m.name in emitted:
                continue
            param_types_m: list[str] = []
            for p in m.params:
                if p.typ is None:
                    continue
                param_types_m.append(self._type(p.typ))
            ret_m = self._return_type(m)
            sig = m.name + "(" + ", ".join(param_types_m) + ")"
            if ret_m:
                sig += " " + ret_m
            self._line(sig)
            emitted.add(m.name)
        common_fields = self._interface_common_fields.get(decl.name, [])
        for fld in common_fields:
            fname = fld.name[0].upper() + fld.name[1:] if fld.name else fld.name
            self._line("Get" + fname + "() " + self._type(fld.typ))
        self.indent -= 1
        self._line("}")

    # ── Function / Method ─────────────────────────────────────

    def _emit_fn(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        old_in_func = self._in_func
        old_ret_type = self._current_ret_type
        self._in_func = True
        self._current_ret_type = decl.ret
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
        params = self._params(decl.params)
        ret = self._return_type(decl)
        if decl.name == "Main":
            self._line("func main() {")
        else:
            fname = _safe_name(decl.name)
            sig = "func " + fname + "(" + params + ")"
            if ret:
                sig += " " + ret
            self._line(sig + " {")
        self.indent += 1
        self._emit_stmts(decl.body)
        self.indent -= 1
        self._line("}")
        self.var_types = old_var_types
        self._in_func = old_in_func
        self._current_ret_type = old_ret_type

    def _emit_method(self, decl: TFnDecl, struct_name: str) -> None:
        old_var_types = self.var_types.copy()
        old_in_func = self._in_func
        old_ret_type = self._current_ret_type
        self._in_func = True
        self._current_ret_type = decl.ret
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
            elif p.typ is None and struct_name:
                self.var_types[p.name] = TIdentType(pos=decl.pos, name=struct_name)
        params_list = [p for p in decl.params if p.typ is not None]
        params = self._params(params_list)
        ret = self._return_type(decl)
        old_self = self.self_name
        if decl.params and decl.params[0].typ is None:
            self.self_name = decl.params[0].name
        method_name = decl.name
        if method_name in ("__repr__", "__str__"):
            method_name = "String"
        receiver = "self *" + struct_name
        sig = "func (" + receiver + ") " + method_name + "(" + params + ")"
        if ret:
            sig += " " + ret
        self._line(sig + " {")
        self.indent += 1
        # Marker method for interface membership
        self._emit_stmts(decl.body)
        self.indent -= 1
        self._line("}")
        self.self_name = old_self
        self.var_types = old_var_types
        self._in_func = old_in_func
        self._current_ret_type = old_ret_type

    def _params(self, params: list[TParam]) -> str:
        parts: list[str] = []
        for p in params:
            if p.typ is None:
                continue
            s = _restore_name(p.name, p.annotations) + " " + self._type(p.typ)
            parts.append(s)
        return ", ".join(parts)

    def _is_opt_prim_type(self, typ: TType | None) -> bool:
        return isinstance(typ, TOptionalType) and isinstance(typ.inner, TPrimitive)

    def _opt_prim_inner_go(self, typ: TType | None) -> str:
        """Get Go type string for inner type of an optional primitive."""
        if isinstance(typ, TOptionalType):
            return self._type(typ.inner)
        return "interface{}"

    def _is_go_ref_type(self, typ: TType | None) -> bool:
        """Check if a type is a reference type in Go (no deref needed for Unwrap)."""
        if typ is None:
            return False
        if isinstance(typ, (TListType, TMapType, TSetType)):
            return True
        if isinstance(typ, TIdentType):
            return True
        if isinstance(typ, TOptionalType):
            return self._is_go_ref_type(typ.inner)
        if isinstance(typ, TUnionType):
            return True
        return False

    def _needs_ptr_wrap_return(self, value: TExpr) -> bool:
        """Check if a return value needs &-wrapping for optional primitive returns."""
        if isinstance(value, TNilLit):
            return False
        rt = self._current_ret_type
        if not (isinstance(rt, TOptionalType) and isinstance(rt.inner, TPrimitive)):
            return False
        # Don't wrap if the value already produces a pointer type
        ann = value.annotations.get("type", "")
        if ann.endswith("?") or "nil" in ann.split(" | "):
            return False
        # Don't wrap field access on optional primitive fields (already dereffed)
        if isinstance(value, TFieldAccess) and self._is_optional_primitive_field(value):
            return False
        return True

    def _is_optional_return_match(self, value: TExpr) -> bool:
        """Check if value is an optional primitive var matching the return type."""
        rt = self._current_ret_type
        if not isinstance(rt, TOptionalType):
            return False
        rt_inner = rt.inner
        if not isinstance(rt_inner, TPrimitive):
            return False
        if not isinstance(value, TVar):
            return False
        vt = self.var_types.get(value.name)
        if not isinstance(vt, TOptionalType):
            return False
        vt_inner = vt.inner
        if not isinstance(vt_inner, TPrimitive):
            return False
        return vt_inner.kind == rt_inner.kind

    def _needs_ptr_wrap_assign(self, target: TExpr, value: TExpr) -> bool:
        """Check if an assignment target is an optional primitive field needing &-wrap."""
        if not isinstance(target, TFieldAccess):
            return False
        if not self._is_optional_primitive_field(target):
            return False
        if isinstance(value, TNilLit):
            return False
        ann = value.annotations.get("type", "")
        if ann.endswith("?") or "nil" in ann.split(" | "):
            return False
        if isinstance(value, TFieldAccess) and self._is_optional_primitive_field(value):
            return False
        return True

    def _expected_type_ann(self, expr: TExpr) -> str:
        return expr.annotations.get("expected_type", "")

    def _needs_ptr_wrap_from_expected(self, expr: TExpr) -> bool:
        """Check if expected_type is optional primitive but actual type is non-optional."""
        expected = self._expected_type_ann(expr)
        if not expected.endswith("?"):
            return False
        inner = expected[:-1]
        if inner not in ("int", "float", "string", "bool"):
            return False
        actual = expr.annotations.get("type", "")
        if actual.endswith("?") or "nil" in actual.split(" | "):
            return False
        if isinstance(expr, TNilLit):
            return False
        return True

    def _needs_deref_from_expected(self, expr: TExpr) -> bool:
        """Check if expected_type is non-optional but actual type is optional primitive."""
        expected = self._expected_type_ann(expr)
        if not expected or expected.endswith("?"):
            return False
        actual = expr.annotations.get("type", "")
        if not actual.endswith("?"):
            return False
        inner = actual[:-1]
        if inner not in ("int", "float", "string", "bool"):
            return False
        return True

    def _expr_preserve_ptr(self, expr: TExpr) -> str:
        """Emit expression WITHOUT auto-deref for optional primitives."""
        old = self._suppress_deref
        self._suppress_deref = True
        result = self._expr(expr)
        self._suppress_deref = old
        return result

    def _expr_no_narrowing(self, expr: TExpr) -> str:
        """Emit expression without TVar narrowing type assertion."""
        if isinstance(expr, TVar):
            if expr.name == self.self_name:
                return "self"
            name = _restore_name(expr.name, expr.annotations)
            name = self._var_aliases.get(name, name)
            if self._needs_deref(expr):
                return "*" + name
            return name
        return self._expr(expr)

    def _field_access_no_deref(self, expr: TFieldAccess) -> str:
        """Emit field access without optional primitive deref."""
        obj_s = self._expr(expr.obj)
        fname = expr.field[0].upper() + expr.field[1:] if expr.field else expr.field
        return obj_s + "." + fname

    def _return_type(self, decl: TFnDecl) -> str:
        if decl.ret is None:
            return ""
        if isinstance(decl.ret, TPrimitive) and decl.ret.kind == "void":
            return ""
        return self._type(decl.ret)

    def _type(self, typ: TType) -> str:
        if isinstance(typ, TPrimitive):
            k = typ.kind
            if k == "int":
                return "int"
            if k == "float":
                return "float64"
            if k == "string":
                return "string"
            if k == "bool":
                return "bool"
            if k == "rune":
                return "rune"
            if k == "byte":
                return "byte"
            if k == "bytes":
                return "[]byte"
            if k == "void":
                return ""
            return k
        if isinstance(typ, TListType):
            return "[]" + self._type(typ.element)
        if isinstance(typ, TMapType):
            return "map[" + self._type(typ.key) + "]" + self._type(typ.value)
        if isinstance(typ, TSetType):
            return "map[" + self._type(typ.element) + "]bool"
        if isinstance(typ, TTupleType):
            if typ.elements:
                types = [self._type(e) for e in typ.elements]
                if all(t == types[0] for t in types[1:]):
                    return "[" + str(len(typ.elements)) + "]" + types[0]
            return "[" + str(len(typ.elements)) + "]any"
        if isinstance(typ, TOptionalType):
            inner = typ.inner
            if isinstance(inner, TPrimitive):
                return "*" + self._type(inner)
            return self._type(inner)
        if isinstance(typ, TFuncType):
            param_types = typ.params[:-1]
            ret_type = typ.params[-1] if typ.params else None
            params_str = ", ".join(self._type(t) for t in param_types)
            ret_str = ""
            if ret_type is not None:
                r = self._type(ret_type)
                if r:
                    ret_str = " " + r
            return "func(" + params_str + ")" + ret_str
        if isinstance(typ, TIdentType):
            if typ.name in self._interface_names:
                return typ.name
            return "*" + typ.name
        if isinstance(typ, TUnionType):
            return "any"
        return "any"

    def _zero_value(self, typ: TType) -> str:
        if isinstance(typ, TPrimitive):
            if typ.kind in ("int", "byte"):
                return "0"
            if typ.kind == "float":
                return "0.0"
            if typ.kind == "bool":
                return "false"
            if typ.kind in ("string", "rune"):
                return '""'
            if typ.kind == "bytes":
                return "[]byte{}"
        if isinstance(typ, TOptionalType):
            return "nil"
        if isinstance(typ, TIdentType):
            return "nil"
        if isinstance(typ, TListType):
            return self._type(typ) + "{}"
        if isinstance(typ, TMapType):
            return self._type(typ) + "{}"
        if isinstance(typ, TSetType):
            return self._type(typ) + "{}"
        if isinstance(typ, TTupleType):
            return self._type(typ) + "{}"
        return "nil"

    # ── Statements ────────────────────────────────────────────

    def _emit_stmts(self, stmts: list[TStmt]) -> None:
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            # Look ahead: merge let declarations with following tuple assign (DivMod)
            if isinstance(stmt, TLetStmt):
                merged = self._try_merge_let_tuple(stmts, i)
                if merged > 0:
                    i += merged
                    continue
                if i + 1 < len(stmts):
                    next_stmt = stmts[i + 1]
                    if isinstance(next_stmt, TForStmt):
                        prov = next_stmt.annotations.get("provenance", "")
                        if prov in (
                            "list_comprehension",
                            "dict_comprehension",
                            "set_comprehension",
                        ):
                            self._emit_let(stmt)
                            self._emit_for(next_stmt)
                            i += 2
                            continue
                        if prov in ("any_call", "all_call"):
                            result = self._emit_any_all(stmts, i, stmt, next_stmt, prov)
                            if result > 0:
                                i += result
                                continue
                    # Merge let + assign when initial value is unused
                    if (
                        isinstance(next_stmt, TAssignStmt)
                        and isinstance(next_stmt.target, TVar)
                        and next_stmt.target.name == stmt.name
                        and stmt.annotations.get("liveness.initial_value_unused")
                        == "true"
                    ):
                        self.var_types[stmt.name] = stmt.typ
                        safe = _restore_name(stmt.name, stmt.annotations)
                        # Check for ParseInt multi-return pattern
                        val = next_stmt.value
                        if (
                            isinstance(val, TCall)
                            and isinstance(val.func, TVar)
                            and val.func.name == "ParseInt"
                        ):
                            self._line(
                                safe + ", _ := " + self._parseint_direct(val.args)
                            )
                            i += 2
                            continue
                        self._line(safe + " := " + self._expr(next_stmt.value))
                        i += 2
                        continue
            if isinstance(stmt, TLetStmt):
                self._emit_let(stmt, remaining=stmts[i + 1 :])
            else:
                self._emit_stmt(stmt)
            i += 1

    def _try_merge_let_tuple(self, stmts: list[TStmt], i: int) -> int:
        """Try to merge consecutive let declarations with a following tuple assign."""
        stmt = stmts[i]
        if not isinstance(stmt, TLetStmt):
            return 0
        # Look for pattern: let q; let r; q, r = DivMod(a, b)
        j = i + 1
        let_names: list[str] = [stmt.name]
        let_stmts: list[TLetStmt] = [stmt]
        while j < len(stmts):
            s_j = stmts[j]
            if not isinstance(s_j, TLetStmt):
                break
            let_names.append(s_j.name)
            let_stmts.append(s_j)
            j += 1
        if j >= len(stmts):
            return 0
        next_s = stmts[j]
        if not isinstance(next_s, TTupleAssignStmt):
            return 0
        if not self._is_divmod_call(next_s.value):
            return 0
        # Check targets match the let names
        target_names: list[str] = []
        for t in next_s.targets:
            if isinstance(t, TVar):
                target_names.append(t.name)
            else:
                return 0
        if not set(target_names).issubset(set(let_names)):
            return 0
        # Register var types
        for ls in let_stmts:
            self.var_types[ls.name] = ls.typ
        # Emit DivMod with :=
        unused_str = next_s.annotations.get("liveness.tuple_unused_indices", "")
        unused_indices: set[int] = set()
        if unused_str:
            for s in unused_str.split(","):
                if s:
                    unused_indices.add(int(s))
        call = next_s.value
        assert isinstance(call, TCall)
        a = self._expr(call.args[0].value)
        b = self._expr(call.args[1].value)
        q_name = (
            _restore_name(target_names[0], next_s.annotations) if target_names else "q"
        )
        if 1 in unused_indices:
            self._line(q_name + " := " + a + " / " + b)
        else:
            r_name = (
                _restore_name(target_names[1], next_s.annotations)
                if len(target_names) > 1
                else "r"
            )
            self._line(q_name + " := " + a + " / " + b)
            self._line(r_name + " := " + a + " - " + q_name + "*" + b)
        return j - i + 1

    def _emit_stmt(self, stmt: TStmt) -> None:
        match stmt:
            case TLetStmt():
                self._emit_let(stmt)
            case TAssignStmt():
                val_s = self._expr(stmt.value)
                if self._is_empty_map_or_set_call(stmt.value):
                    inferred = self._infer_list_type(stmt.target)
                    if inferred is not None:
                        val_s = self._type(inferred) + "{}"
                if isinstance(
                    stmt.target, TFieldAccess
                ) and self._needs_ptr_wrap_assign(stmt.target, stmt.value):
                    tgt_s = self._field_access_no_deref(stmt.target)
                    self._line("__atmp := " + val_s)
                    self._line(tgt_s + " = &__atmp")
                elif self._needs_ptr_wrap_from_expected(stmt.value):
                    self._line("__atmp := " + val_s)
                    self._line(self._expr(stmt.target) + " = &__atmp")
                elif self._needs_deref_from_expected(stmt.value):
                    self._line(self._expr(stmt.target) + " = *" + val_s)
                elif isinstance(stmt.target, TVar):
                    vt = self.var_types.get(stmt.target.name)
                    if (
                        isinstance(vt, TOptionalType)
                        and isinstance(vt.inner, TPrimitive)
                        and not isinstance(stmt.value, TNilLit)
                    ):
                        val_ann = stmt.value.annotations.get("type", "")
                        tgt_s = _restore_name(stmt.target.name, stmt.target.annotations)
                        tgt_s = self._var_aliases.get(tgt_s, tgt_s)
                        if not val_ann.endswith("?") and "nil" not in val_ann.split(
                            " | "
                        ):
                            self._line("__atmp := " + val_s)
                            self._line(tgt_s + " = &__atmp")
                        else:
                            self._line(
                                tgt_s + " = " + self._expr_preserve_ptr(stmt.value)
                            )
                    else:
                        self._line(self._expr(stmt.target) + " = " + val_s)
                else:
                    self._line(self._expr(stmt.target) + " = " + val_s)
            case TTupleAssignStmt():
                self._emit_tuple_assign(stmt)
            case TOpAssignStmt():
                if (
                    self.strict_math
                    and stmt.op in STRICT_INT_COMPOUND
                    and self._is_int_expr(stmt.target)
                ):
                    fn = _STRICT_GO_COMPOUND[stmt.op]
                    self._used_strict_helpers.add(STRICT_INT_COMPOUND[stmt.op])
                    self._line(
                        self._expr(stmt.target)
                        + " = "
                        + fn
                        + "("
                        + self._expr(stmt.target)
                        + ", "
                        + self._expr(stmt.value)
                        + ")"
                    )
                else:
                    self._line(
                        self._expr(stmt.target)
                        + " "
                        + stmt.op
                        + " "
                        + self._expr(stmt.value)
                    )
            case TExprStmt():
                self._emit_expr_stmt(stmt)
            case TReturnStmt():
                if stmt.value is not None:
                    if self._try_result_var is not None:
                        self._line(
                            self._try_result_var + " = " + self._expr(stmt.value)
                        )
                        self._line("return")
                    elif isinstance(stmt.value, TTernary):
                        self._emit_ternary_return(stmt.value)
                    elif isinstance(
                        stmt.value, TBinaryOp
                    ) and self._is_isinstance_tuple_expr(stmt.value):
                        self._emit_isinstance_tuple_return(stmt.value)
                    elif self._needs_ptr_wrap_return(stmt.value):
                        self._line("__retv := " + self._expr(stmt.value))
                        self._line("return &__retv")
                    elif (
                        self._current_ret_type is not None
                        and self._is_empty_collection_return(stmt.value)
                    ):
                        self._line(
                            "return " + self._type(self._current_ret_type) + "{}"
                        )
                    elif isinstance(stmt.value, TNilLit):
                        ret_t = self._current_ret_type
                        if isinstance(ret_t, TOptionalType) and isinstance(
                            ret_t.inner, TTupleType
                        ):
                            self._line("return " + self._type(ret_t.inner) + "{}")
                        elif isinstance(ret_t, TTupleType):
                            self._line("return " + self._type(ret_t) + "{}")
                        else:
                            self._line("return nil")
                    elif isinstance(
                        stmt.value, TVar
                    ) and self._is_optional_return_match(stmt.value):
                        vname = _restore_name(stmt.value.name, stmt.value.annotations)
                        vname = self._var_aliases.get(vname, vname)
                        self._line("return " + vname)
                    else:
                        self._line("return " + self._expr(stmt.value))
                else:
                    self._line("return")
            case TThrowStmt():
                if (
                    isinstance(stmt.expr, TVar)
                    and stmt.expr.name in self._error_structs
                ):
                    self._line("panic(&" + stmt.expr.name + "{})")
                else:
                    self._line("panic(" + self._expr(stmt.expr) + ")")
            case TBreakStmt():
                if self._try_action_var is not None:
                    self._line(self._try_action_var + " = 2")
                    self._line("return")
                else:
                    self._line("break")
            case TContinueStmt():
                if self._try_action_var is not None:
                    self._line(self._try_action_var + " = 1")
                    self._line("return")
                else:
                    self._line("continue")
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

    def _emit_let(self, stmt: TLetStmt, remaining: list[TStmt] | None = None) -> None:
        safe = _restore_name(stmt.name, stmt.annotations)
        self.var_types[stmt.name] = stmt.typ
        if remaining is not None and not _stmts_ref_var(remaining, stmt.name):
            if stmt.value is not None:
                self._line("_ = " + self._expr(stmt.value))
            return
        unused = stmt.annotations.get("liveness.initial_value_unused") == "true"
        if stmt.value is not None and not unused:
            val = stmt.value
            if isinstance(val, TCall) and self._is_get_default_call(val):
                self._emit_get_default_let(safe, val)
                return
            if isinstance(val, TCall) and self._is_pop_call(val):
                self._emit_pop_let(safe, val)
                return
            if isinstance(val, TCall) and self._is_parseint_call(val):
                self._emit_parseint_let(safe, val)
                return
            if isinstance(val, TCall) and self._is_parsefloat_call(val):
                self._emit_parsefloat_let(safe, val)
                return
            if isinstance(val, TCall) and self._is_readbytesn_call(val):
                self._emit_readbytesn_let(safe, val)
                return
            if isinstance(val, TCall) and self._is_sorted_call(val):
                self._emit_sorted_let(safe, val)
                return
            if isinstance(val, TCall) and self._is_reversed_call(val):
                self._emit_reversed_let(safe, val)
                return
            if self._is_empty_map_or_set_call(val):
                go_type = self._type(stmt.typ)
                if self._in_func:
                    self._line(safe + " := " + go_type + "{}")
                else:
                    self._line("var " + safe + " = " + go_type + "{}")
                return
            if isinstance(val, TListLit) and not val.elements:
                go_type = self._type(stmt.typ)
                if self._in_func:
                    self._line(safe + " := " + go_type + "{}")
                else:
                    self._line("var " + safe + " = " + go_type + "{}")
                return
            if isinstance(val, TNilLit):
                self._line("var " + safe + " " + self._type(stmt.typ))
                return
            is_opt_prim = isinstance(stmt.typ, TOptionalType) and isinstance(
                stmt.typ.inner, TPrimitive
            )
            val_s = self._expr_preserve_ptr(val) if is_opt_prim else self._expr(val)
            if isinstance(val, TIndex) and isinstance(val.obj, TVar):
                obj_vt = self.var_types.get(val.obj.name)
                if isinstance(obj_vt, TTupleType):
                    go_type = self._type(stmt.typ)
                    if go_type != "any" and go_type != self._type(obj_vt):
                        val_s = val_s + ".(" + go_type + ")"
            if self._in_func:
                self._line(safe + " := " + val_s)
            else:
                self._line("var " + safe + " = " + val_s)
        else:
            self._line("var " + safe + " " + self._type(stmt.typ))

    def _resolve_field_type(self, expr: TFieldAccess) -> TType | None:
        """Resolve the declared type of a field access through struct types."""
        obj_type: TType | None = None
        if isinstance(expr.obj, TVar):
            obj_type = self.var_types.get(expr.obj.name)
        elif isinstance(expr.obj, TFieldAccess):
            obj_type = self._resolve_field_type(expr.obj)
        if isinstance(obj_type, TIdentType):
            for fd in self.struct_field_types.get(obj_type.name, []):
                if fd.name == expr.field:
                    return fd.typ
        return None

    def _is_interface_field_access(self, expr: TFieldAccess) -> bool:
        """Check if this is a field access on an interface-typed variable."""
        obj_type_name = self._obj_interface_type(expr.obj)
        if obj_type_name is not None:
            common = self._interface_common_fields.get(obj_type_name, [])
            for fld in common:
                if fld.name == expr.field:
                    return True
        return False

    def _obj_interface_type(self, obj: TExpr) -> str | None:
        """Return interface type name if obj is interface-typed, else None."""
        if isinstance(obj, TVar):
            vt = self.var_types.get(obj.name)
            iname = self._extract_interface_name(vt)
            if iname is not None:
                return iname
            ann = obj.annotations.get("type", "")
            if ann in self._interface_names:
                return ann
            if ann.endswith("?"):
                base = ann[:-1]
                if base in self._interface_names:
                    return base
        if isinstance(obj, TFieldAccess):
            ann = obj.annotations.get("type", "")
            if ann in self._interface_names:
                return ann
            if ann.endswith("?"):
                base = ann[:-1]
                if base in self._interface_names:
                    return base
            ft = self._resolve_field_type(obj)
            iname = self._extract_interface_name(ft)
            if iname is not None:
                return iname
        return None

    def _extract_interface_name(self, vt: TType | None) -> str | None:
        """Extract interface name from a type, handling optionals and unions."""
        if isinstance(vt, TIdentType) and vt.name in self._interface_names:
            return vt.name
        if isinstance(vt, TOptionalType):
            return self._extract_interface_name(vt.inner)
        if isinstance(vt, TUnionType):
            for m in vt.members:
                n = self._extract_interface_name(m)
                if n is not None:
                    return n
        return None

    def _is_empty_collection_return(self, expr: TExpr) -> bool:
        """Check if expr is an empty Map()/Set()/List()/list that needs ret type."""
        if self._current_ret_type is None:
            return False
        if self._is_empty_map_or_set_call(expr):
            return True
        if isinstance(expr, TListLit) and not expr.elements:
            return True
        if self._is_empty_list_call(expr):
            return True
        return False

    def _is_empty_map_or_set_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name in ("Map", "Set")
            and not expr.args
        )

    def _is_empty_list_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "List"
            and not expr.args
        )

    def _append_empty_collection(self, list_expr: TExpr, val_expr: TExpr) -> str:
        """Infer the correct type for an empty Map()/Set() appended to a list."""
        list_type = self._infer_list_type(list_expr)
        if isinstance(list_type, TListType) and isinstance(list_type.element, TMapType):
            kt = self._type(list_type.element.key)
            vt_s = self._type(list_type.element.value)
            return "map[" + kt + "]" + vt_s + "{}"
        if isinstance(val_expr, TCall) and isinstance(val_expr.func, TVar):
            if val_expr.func.name == "Map":
                return self._empty_map_call(val_expr.args)
            return self._empty_set_call(val_expr.args)
        return self._expr(val_expr)

    def _infer_list_type(self, expr: TExpr) -> TType | None:
        if isinstance(expr, TVar):
            return self.var_types.get(expr.name)
        if isinstance(expr, TFieldAccess):
            obj = expr.obj
            if isinstance(obj, TVar):
                obj_type = self.var_types.get(obj.name)
                if isinstance(obj_type, TIdentType):
                    field_decls = self.struct_field_types.get(obj_type.name, [])
                    for fd in field_decls:
                        if fd.name == expr.field:
                            return fd.typ
        return None

    def _is_get_default_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Get"
            and len(expr.args) == 3
        )

    def _emit_get_default_let(self, var_name: str, expr: TCall) -> None:
        map_arg = self._expr(expr.args[0].value)
        key = self._expr(expr.args[1].value)
        default = self._expr(expr.args[2].value)
        self._line(
            var_name
            + ", ok := "
            + map_arg
            + "["
            + key
            + "]; if !ok { "
            + var_name
            + " = "
            + default
            + " }"
        )

    def _is_pop_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Pop"
        )

    def _emit_pop_let(self, var_name: str, expr: TCall) -> None:
        obj = self._expr(expr.args[0].value)
        self._line(var_name + " := " + obj + "[len(" + obj + ")-1]")
        self._line(obj + " = " + obj + "[:len(" + obj + ")-1]")

    def _is_parseint_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "ParseInt"
        )

    def _emit_parseint_let(self, var_name: str, expr: TCall) -> None:
        s = self._expr(expr.args[0].value)
        base = self._expr(expr.args[1].value)
        self._line(var_name + ", _ := strconv.ParseInt(" + s + ", " + base + ", 64)")

    def _is_parsefloat_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "ParseFloat"
        )

    def _emit_parsefloat_let(self, var_name: str, expr: TCall) -> None:
        s = self._expr(expr.args[0].value)
        self._line(var_name + ", _ := strconv.ParseFloat(" + s + ", 64)")

    def _is_readbytesn_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "ReadBytesN"
        )

    def _emit_readbytesn_let(self, var_name: str, expr: TCall) -> None:
        n = self._expr(expr.args[0].value)
        self._line(var_name + " := make([]byte, " + n + ")")
        self._line("io.ReadFull(os.Stdin, " + var_name + ")")

    def _is_sorted_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Sorted"
            and not self._is_string_expr(expr.args[0].value)
        )

    def _emit_sorted_let(self, var_name: str, expr: TCall) -> None:
        if self.strict_math and self._is_float_list(expr.args[0].value):
            self._used_strict_helpers.add("strict_sorted_f64")
            self._line(
                var_name + " := strictSortedF64(" + self._expr(expr.args[0].value) + ")"
            )
            return
        arg = expr.args[0].value
        is_set = self._is_set_type(arg)
        if (
            not is_set
            and isinstance(arg, TCall)
            and isinstance(arg.func, TVar)
            and arg.func.name == "ListFrom"
            and arg.args
        ):
            is_set = self._is_set_type(arg.args[0].value)
            if is_set:
                arg = arg.args[0].value
        a = self._expr(arg)
        if is_set:
            et = self._infer_elem_type(arg)
            self._line(var_name + " := make([]" + et + ", 0, len(" + a + "))")
            self._line(
                "for k := range "
                + a
                + " { "
                + var_name
                + " = append("
                + var_name
                + ", k) }"
            )
        else:
            self._line(var_name + " := slices.Clone(" + a + ")")
        if len(expr.args) >= 2 and isinstance(expr.args[1].value, TFnLit):
            key_fn = self._expr(expr.args[1].value)
            et = self._infer_elem_type(arg)
            self._line(
                "slices.SortFunc("
                + var_name
                + ", func(a, b "
                + et
                + ") int { return "
                + key_fn
                + "(a) - "
                + key_fn
                + "(b) })"
            )
        else:
            self._line("slices.Sort(" + var_name + ")")

    def _is_reversed_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name in ("Reversed", "Reverse")
            and not self._is_string_expr(expr.args[0].value)
            and expr.annotations.get("provenance") != "reversed_slice"
        )

    def _emit_reversed_let(self, var_name: str, expr: TCall) -> None:
        a = self._expr(expr.args[0].value)
        self._line(var_name + " := slices.Clone(" + a + ")")
        self._line("slices.Reverse(" + var_name + ")")

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
        n = self._tuple_unpack_counter
        self._tuple_unpack_counter += 1
        tmp = "__tup" + str(n)
        self._line(tmp + " := " + self._expr(stmt.value))
        tuple_elem_type = self._tuple_go_elem_type(stmt.value)
        for i, t in enumerate(stmt.targets):
            if i in unused_indices:
                continue
            target_s = self._expr(t)
            elem = tmp + "[" + str(i) + "]"
            go_type = ""
            if isinstance(t, TVar):
                vt = self.var_types.get(t.name)
                if vt is not None:
                    go_type = self._type(vt)
            if go_type and go_type != "any" and tuple_elem_type == "any":
                self._line(target_s + " = " + elem + ".(" + go_type + ")")
            else:
                self._line(target_s + " = " + elem)

    def _tuple_go_elem_type(self, expr: TExpr) -> str:
        """Return the Go element type if tuple is homogeneous, else 'any'."""
        if isinstance(expr, TTupleLit) and expr.elements:
            types = [self._infer_go_type(e) for e in expr.elements]
            if types[0] != "any" and all(t == types[0] for t in types[1:]):
                return types[0]
        tup = self._resolve_tuple_type(expr)
        if tup is not None and tup.elements:
            types = [self._type(e) for e in tup.elements]
            if all(t == types[0] for t in types[1:]):
                return types[0]
        return "any"

    def _resolve_tuple_type(self, expr: TExpr) -> TTupleType | None:
        """Try to resolve a TTupleType for an expression."""
        if isinstance(expr, TVar):
            vt = self.var_types.get(expr.name)
            if isinstance(vt, TTupleType):
                return vt
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            ft = self.var_types.get(expr.func.name)
            if isinstance(ft, TFuncType) and ft.params:
                ret = ft.params[-1]
                if isinstance(ret, TTupleType):
                    return ret
        # Fallback: parse type annotation
        ann = expr.annotations.get("type", "")
        if ann.startswith("(") and ann.endswith(")"):
            parts = _split_tuple_ann(ann[1:-1])
            go_types = [self._ann_type_to_go(p) for p in parts]
            if all(t != "any" for t in go_types):
                elems: list[TType] = []
                for p in parts:
                    elems.append(TPrimitive(pos=expr.pos, kind=p))
                return TTupleType(pos=expr.pos, elements=elems)
        return None

    def _is_divmod_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "DivMod"
        )

    def _emit_divmod_assign(self, stmt: TTupleAssignStmt, unused: set[int]) -> None:
        call = stmt.value
        assert isinstance(call, TCall)
        a = self._expr(call.args[0].value)
        b = self._expr(call.args[1].value)
        q_target = self._expr(stmt.targets[0])
        if 1 in unused:
            self._line(q_target + " = " + a + " / " + b)
        else:
            r_target = self._expr(stmt.targets[1])
            self._line(q_target + " = " + a + " / " + b)
            self._line(r_target + " = " + a + " - " + q_target + "*" + b)

    def _emit_expr_stmt(self, stmt: TExprStmt) -> None:
        expr = stmt.expr
        # Skip docstrings — bare string/nil literals are no-ops in Go
        if isinstance(expr, TStringLit) or isinstance(expr, TNilLit):
            return
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            name = expr.func.name
            if name == "Assert":
                args = expr.args
                inner = args[0].value
                if (
                    isinstance(inner, TCall)
                    and isinstance(inner.func, TVar)
                    and inner.func.name == "IsInstance"
                    and isinstance(inner.args[0].value, TVar)
                ):
                    obj_var = inner.args[0].value
                    assert isinstance(obj_var, TVar)
                    type_arg = inner.args[1].value
                    type_name = ""
                    if isinstance(type_arg, TVar):
                        type_name = type_arg.name
                    elif isinstance(type_arg, TStringLit):
                        type_name = type_arg.value
                    if type_name and type_name not in self._interface_names:
                        vt = self.var_types.get(obj_var.name)
                        go_t = self._type(vt).lstrip("*") if vt is not None else "any"
                        if go_t in self._interface_names or go_t == "any":
                            obj_go = _restore_name(obj_var.name, obj_var.annotations)
                            obj_go = self._var_aliases.get(obj_go, obj_go)
                            alias = obj_go + "_assert"
                            self._line(
                                alias + " := " + obj_go + ".(*" + type_name + ")"
                            )
                            self._var_aliases[obj_go] = alias
                            self.var_types[obj_var.name] = TIdentType(
                                pos=obj_var.pos,
                                name=type_name,
                            )
                            return
                cond = self._expr(args[0].value)
                if len(args) > 1:
                    msg = self._expr(args[1].value)
                else:
                    msg = '"assertion failed"'
                self._line("if !(" + cond + ") { panic(" + msg + ") }")
                return
            if name == "Delete":
                args = expr.args
                self._line(
                    "delete("
                    + self._expr(args[0].value)
                    + ", "
                    + self._expr(args[1].value)
                    + ")"
                )
                return
            if name == "RemoveAt":
                args = expr.args
                obj = self._expr(args[0].value)
                idx = self._expr(args[1].value)
                static_idx = self._static_int(args[1].value)
                if static_idx is not None:
                    end = str(static_idx + 1)
                else:
                    end = idx + "+1"
                self._line(
                    obj + " = slices.Delete(" + obj + ", " + idx + ", " + end + ")"
                )
                return
            if name == "Append":
                args = expr.args
                obj = self._expr(args[0].value)
                val_expr = args[1].value
                if self._is_empty_map_or_set_call(val_expr):
                    val = self._append_empty_collection(args[0].value, val_expr)
                else:
                    val = self._expr(val_expr)
                self._line(obj + " = append(" + obj + ", " + val + ")")
                return
            if name == "Insert":
                args = expr.args
                obj = self._expr(args[0].value)
                idx = self._expr(args[1].value)
                val = self._expr(args[2].value)
                self._line(
                    obj + " = slices.Insert(" + obj + ", " + idx + ", " + val + ")"
                )
                return
            if name == "Add":
                args = expr.args
                obj = self._expr(args[0].value)
                val = self._expr(args[1].value)
                self._line(obj + "[" + val + "] = true")
                return
            if name == "Remove":
                args = expr.args
                obj = self._expr(args[0].value)
                val = self._expr(args[1].value)
                self._line("delete(" + obj + ", " + val + ")")
                return
            if name == "Pop":
                args = expr.args
                obj = self._expr(args[0].value)
                self._line(obj + " = " + obj + "[:len(" + obj + ")-1]")
                return
        self._line(self._expr(expr))

    def _emit_if(self, stmt: TIfStmt) -> None:
        cond = stmt.cond
        if self._is_map_contains_call(cond):
            call = cond
            assert isinstance(call, TCall)
            m = self._expr(call.args[0].value)
            k = self._expr(call.args[1].value)
            self._line("if _, ok := " + m + "[" + k + "]; ok {")
        else:
            self._line("if " + self._expr(cond) + " {")
        self.indent += 1
        self._emit_stmts(stmt.then_body)
        self.indent -= 1
        self._emit_else_body(stmt.else_body)

    def _is_map_contains_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Contains"
            and len(expr.args) >= 2
            and self._is_map_type(expr.args[0].value)
        )

    def _emit_else_body(self, else_body: list[TStmt] | None) -> None:
        if else_body is None or not else_body:
            self._line("}")
            return
        elif_stmt: TStmt | None = None
        if len(else_body) == 1:
            elif_stmt = else_body[0]
        if isinstance(elif_stmt, TIfStmt):
            self._line("} else if " + self._expr(elif_stmt.cond) + " {")
            self.indent += 1
            self._emit_stmts(elif_stmt.then_body)
            self.indent -= 1
            self._emit_else_body(elif_stmt.else_body)
        else:
            self._line("} else {")
            self.indent += 1
            self._emit_stmts(else_body)
            self.indent -= 1
            self._line("}")

    def _emit_while(self, stmt: TWhileStmt) -> None:
        if isinstance(stmt.cond, TBoolLit) and stmt.cond.value:
            self._line("for {")
        else:
            self._line("for " + self._expr(stmt.cond) + " {")
        self.indent += 1
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _emit_for(self, stmt: TForStmt) -> None:
        binding = stmt.binding
        ann = stmt.annotations
        prov = ann.get("provenance", "")
        if isinstance(stmt.iterable, TRange):
            self._emit_for_range(stmt, binding, ann, prov)
            return
        if self._is_zip_for(stmt):
            self._emit_for_zip(stmt, binding, ann)
            return
        if self._is_reversed_for(stmt):
            self._emit_for_reversed(stmt, binding, ann)
            return
        if len(binding) == 1:
            b = _restore_name(binding[0], ann)
            iterable = stmt.iterable
            if self._is_map_type(iterable):
                self._line("for " + b + " := range " + self._expr(iterable) + " {")
                self._track_for_binding_type(binding[0], iterable, "key")
            elif self._is_set_type(iterable):
                self._line("for " + b + " := range " + self._expr(iterable) + " {")
                self._track_for_binding_type(binding[0], iterable, "key")
            else:
                self._line("for _, " + b + " := range " + self._expr(iterable) + " {")
                self._track_for_binding_type(binding[0], iterable, "elem")
        elif len(binding) == 2:
            b0 = _restore_name(binding[0], ann)
            b1 = _restore_name(binding[1], ann)
            if not _stmts_ref_var(stmt.body, binding[0]):
                b0 = "_"
            if not _stmts_ref_var(stmt.body, binding[1]):
                b1 = "_"
            iter_is_map = self._is_map_for(stmt)
            is_enumerate = self._is_enumerate_for(stmt)
            if iter_is_map:
                self._line(
                    "for "
                    + b0
                    + ", "
                    + b1
                    + " := range "
                    + self._expr(stmt.iterable)
                    + " {"
                )
                self._track_for_binding_type(binding[0], stmt.iterable, "key")
                self._track_for_binding_type(binding[1], stmt.iterable, "val")
            elif is_enumerate:
                self._line(
                    "for "
                    + b0
                    + ", "
                    + b1
                    + " := range "
                    + self._expr(stmt.iterable)
                    + " {"
                )
                self._track_for_binding_type(binding[1], stmt.iterable, "elem")
            else:
                self._emit_for_tuple_unpack(stmt, binding, ann)
                return
        else:
            self._emit_for_tuple_unpack(stmt, binding, ann)
            return
        self.indent += 1
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _resolve_expr_type(self, expr: TExpr) -> TType | None:
        """Resolve TType for an expression (var, field access)."""
        if isinstance(expr, TVar):
            return self.var_types.get(expr.name)
        if isinstance(expr, TFieldAccess):
            return self._resolve_field_type(expr)
        return None

    def _track_for_binding_type(
        self, var_name: str, iterable: TExpr, role: str
    ) -> None:
        """Track var_types for for-loop binding variables."""
        vt = self._resolve_expr_type(iterable)
        if vt is None:
            return
        if role == "elem":
            if isinstance(vt, TListType):
                self.var_types[var_name] = vt.element
        elif role == "key":
            if isinstance(vt, TMapType):
                self.var_types[var_name] = vt.key
            elif isinstance(vt, TSetType):
                self.var_types[var_name] = vt.element
        elif role == "val":
            if isinstance(vt, TMapType):
                self.var_types[var_name] = vt.value

    def _emit_for_tuple_unpack(
        self, stmt: TForStmt, binding: list[str], ann: Ann
    ) -> None:
        iter_s = self._expr(stmt.iterable)
        self._line("for _, __unpack := range " + iter_s + " {")
        self.indent += 1
        # Get tuple element types from iterable's list element type
        elem_types: list[TType] | None = None
        vt = self._resolve_expr_type(stmt.iterable)
        if isinstance(vt, TListType):
            if isinstance(vt.element, TTupleType):
                elem_types = vt.element.elements
        homogeneous = False
        if elem_types is not None and len(elem_types) > 0:
            go_types = [self._type(e) for e in elem_types]
            homogeneous = all(t == go_types[0] for t in go_types[1:])
        i = 0
        while i < len(binding):
            name = _restore_name(binding[i], ann)
            if name != "_" and not _stmts_ref_var(stmt.body, binding[i]):
                name = "_"
            idx = str(i)
            op = "=" if name == "_" else ":="
            if elem_types is not None and i < len(elem_types):
                if homogeneous:
                    self._line(name + " " + op + " __unpack[" + idx + "]")
                else:
                    go_t = self._type(elem_types[i])
                    self._line(
                        name + " " + op + " __unpack[" + idx + "].(" + go_t + ")"
                    )
                if name != "_":
                    self.var_types[binding[i]] = elem_types[i]
            else:
                self._line(name + " " + op + " __unpack[" + idx + "]")
            i += 1
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _emit_for_range(
        self,
        stmt: TForStmt,
        binding: list[str],
        ann: Ann,
        prov: str,
    ) -> None:
        binder_parts: list[str] = []
        for b in binding:
            binder_parts.append(_restore_name(b, ann))
        binder = binder_parts[0] if binder_parts else "_"
        assert isinstance(stmt.iterable, TRange)
        args = stmt.iterable.args
        if prov == "reversed_range" and len(args) == 3:
            rargs = args
            start = self._expr(rargs[0])
            end_val = self._static_int(rargs[1])
            step_val = self._static_int(rargs[2])
            if step_val is not None:
                if step_val == -1:
                    if end_val is not None:
                        cmp_val = str(end_val + 1)
                        cmp_op = " >= "
                    else:
                        cmp_val = self._expr(rargs[1])
                        cmp_op = " > "
                    self._line(
                        "for "
                        + binder
                        + " := "
                        + start
                        + "; "
                        + binder
                        + cmp_op
                        + cmp_val
                        + "; "
                        + binder
                        + "-- {"
                    )
                elif step_val < 0:
                    self._line(
                        "for "
                        + binder
                        + " := "
                        + start
                        + "; "
                        + binder
                        + " > "
                        + self._expr(rargs[1])
                        + "; "
                        + binder
                        + " += "
                        + self._expr(rargs[2])
                        + " {"
                    )
                else:
                    self._emit_for_range_std(binder, args)
            else:
                self._emit_for_range_std(binder, args)
        elif prov == "reversed_for" and len(args) >= 2:
            start = self._expr(args[0])
            end = self._expr(args[1])
            self._line(
                "for "
                + binder
                + " := "
                + start
                + "; "
                + binder
                + " >= "
                + end
                + "; "
                + binder
                + "-- {"
            )
        else:
            self._emit_for_range_std(binder, args)
        self.indent += 1
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _emit_for_range_std(self, binder: str, args: list[TExpr]) -> None:
        if len(args) == 1:
            end = self._expr(args[0])
            self._line(
                "for "
                + binder
                + " := 0; "
                + binder
                + " < "
                + end
                + "; "
                + binder
                + "++ {"
            )
        elif len(args) == 2:
            start = self._expr(args[0])
            end = self._expr(args[1])
            self._line(
                "for "
                + binder
                + " := "
                + start
                + "; "
                + binder
                + " < "
                + end
                + "; "
                + binder
                + "++ {"
            )
        elif len(args) == 3:
            start = self._expr(args[0])
            end = self._expr(args[1])
            step_val = self._static_int(args[2])
            if step_val is not None:
                if step_val == -1:
                    self._line(
                        "for "
                        + binder
                        + " := "
                        + start
                        + "; "
                        + binder
                        + " > "
                        + end
                        + "; "
                        + binder
                        + "-- {"
                    )
                elif step_val < 0:
                    self._line(
                        "for "
                        + binder
                        + " := "
                        + start
                        + "; "
                        + binder
                        + " > "
                        + end
                        + "; "
                        + binder
                        + " += "
                        + self._expr(args[2])
                        + " {"
                    )
                else:
                    self._line(
                        "for "
                        + binder
                        + " := "
                        + start
                        + "; "
                        + binder
                        + " < "
                        + end
                        + "; "
                        + binder
                        + " += "
                        + self._expr(args[2])
                        + " {"
                    )
            else:
                step = self._expr(args[2])
                self._line(
                    "for "
                    + binder
                    + " := "
                    + start
                    + "; "
                    + binder
                    + " < "
                    + end
                    + "; "
                    + binder
                    + " += "
                    + step
                    + " {"
                )

    def _is_zip_for(self, stmt: TForStmt) -> bool:
        return not isinstance(stmt.iterable, TRange) and self._is_builtin_call(
            stmt.iterable, "Zip"
        )

    def _emit_for_zip(self, stmt: TForStmt, binding: list[str], ann: Ann) -> None:
        assert isinstance(stmt.iterable, TCall)
        zip_args = stmt.iterable.args
        arr_exprs = [self._expr(a.value) for a in zip_args]
        min_parts = ["len(" + e + ")" for e in arr_exprs]
        min_expr = "min(" + ", ".join(min_parts) + ")"
        self._line("for __i := 0; __i < " + min_expr + "; __i++ {")
        self.indent += 1
        for j, b in enumerate(binding):
            bname = _restore_name(b, ann)
            self._line(bname + " := " + arr_exprs[j] + "[__i]")
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _is_reversed_for(self, stmt: TForStmt) -> bool:
        if not isinstance(stmt.iterable, TCall):
            return False
        if not isinstance(stmt.iterable.func, TVar):
            return False
        return stmt.iterable.func.name in ("Reversed", "Reverse")

    def _emit_for_reversed(self, stmt: TForStmt, binding: list[str], ann: Ann) -> None:
        assert isinstance(stmt.iterable, TCall)
        inner = stmt.iterable.args[0].value
        b = _restore_name(binding[0], ann)
        arr = self._expr(inner)
        self._line("for i := len(" + arr + ") - 1; i >= 0; i-- {")
        self.indent += 1
        self._line(b + " := " + arr + "[i]")
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _is_builtin_call(self, expr: TExpr, name: str) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == name
        )

    def _is_map_type(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann.startswith("map[")
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TMapType)
        if isinstance(expr, TFieldAccess):
            return isinstance(self._resolve_field_type(expr), TMapType)
        return False

    def _is_set_type(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann.startswith("set[")
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TSetType)
        if isinstance(expr, TFieldAccess):
            return isinstance(self._resolve_field_type(expr), TSetType)
        return False

    def _is_map_for(self, stmt: TForStmt) -> bool:
        if stmt.annotations.get("for.items") == "true":
            return True
        return not isinstance(stmt.iterable, TRange) and self._is_map_type(
            stmt.iterable
        )

    def _is_list_expr(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann.startswith("list[")
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TListType)
        if isinstance(expr, TFieldAccess):
            return isinstance(self._resolve_field_type(expr), TListType)
        return isinstance(expr, TListLit)

    def _needs_deref(self, expr: TVar) -> bool:
        """Check if a variable needs optional dereference (*x)."""
        if self._suppress_deref:
            return False
        typ = self.var_types.get(expr.name)
        if not isinstance(typ, TOptionalType):
            return False
        if self._is_go_ref_type(typ.inner):
            return False
        # Check if narrowed via scope annotation
        narrowed = expr.annotations.get("scope.narrowed_type", "")
        if narrowed:
            return True
        # Check if type annotation shows non-optional (narrowed)
        ann_type = expr.annotations.get("type", "")
        if ann_type and not ann_type.endswith("?"):
            return True
        return False

    def _is_bytes_expr(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann == "bytes":
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "bytes"
        return False

    def _field_access_raw(self, expr: TFieldAccess) -> str:
        """Emit a field access without optional primitive deref."""
        obj_s = self._expr(expr.obj)
        fname = expr.field[0].upper() + expr.field[1:] if expr.field else expr.field
        return obj_s + "." + fname

    def _is_optional_primitive_field(self, expr: TFieldAccess) -> bool:
        """Check if field access returns an optional primitive (needs deref in Go)."""
        if self._suppress_deref:
            return False
        if not isinstance(expr.obj, TVar):
            return False
        obj_type = self.var_types.get(expr.obj.name)
        if isinstance(obj_type, TIdentType):
            struct_name = obj_type.name
        elif isinstance(obj_type, TOptionalType) and isinstance(
            obj_type.inner, TIdentType
        ):
            struct_name = obj_type.inner.name
        else:
            ann = expr.annotations.get("type", "")
            if ann.endswith("?"):
                ann_inner = ann[:-1]
                if ann_inner in ("string", "int", "float", "bool", "byte", "rune"):
                    return True
            return False
        fields = self.struct_field_types.get(struct_name, [])
        for f in fields:
            if f.name == expr.field and isinstance(f.typ, TOptionalType):
                inner = f.typ.inner
                if isinstance(inner, TPrimitive) and inner.kind in (
                    "string",
                    "int",
                    "float",
                    "bool",
                    "byte",
                    "rune",
                ):
                    return True
        return False

    def _is_string_expr(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann == "string":
            return True
        if isinstance(expr, TStringLit):
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "string"
        return False

    def _infer_elem_type(self, expr: TExpr) -> str:
        ann: str = expr.annotations.get("type", "")
        if ann.startswith("list[") and ann.endswith("]"):
            inner = ann[5:-1]
            return self._ann_type_to_go(inner)
        if ann.startswith("set[") and ann.endswith("]"):
            inner = ann[4:-1]
            return self._ann_type_to_go(inner)
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            if isinstance(typ, TListType):
                return self._type(typ.element)
            if isinstance(typ, TSetType):
                return self._type(typ.element)
        if isinstance(expr, TFieldAccess):
            resolved = self._resolve_field_type(expr)
            if isinstance(resolved, TListType):
                return self._type(resolved.element)
            if isinstance(resolved, TSetType):
                return self._type(resolved.element)
        if isinstance(expr, TListLit) and expr.elements:
            return self._infer_go_type(expr.elements[0])
        return "any"

    def _infer_map_key_type(self, expr: TExpr) -> str:
        ann: str = expr.annotations.get("type", "")
        if ann.startswith("map["):
            comma = ann.index(",")
            return self._ann_type_to_go(ann[4:comma])
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            if isinstance(typ, TMapType):
                return self._type(typ.key)
        if isinstance(expr, TFieldAccess):
            resolved = self._resolve_field_type(expr)
            if isinstance(resolved, TMapType):
                return self._type(resolved.key)
        return "any"

    def _infer_map_val_type(self, expr: TExpr) -> str:
        ann: str = expr.annotations.get("type", "")
        if ann.startswith("map["):
            comma = ann.index(",")
            inner = ann[comma + 2 : -1]
            return self._ann_type_to_go(inner)
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            if isinstance(typ, TMapType):
                return self._type(typ.value)
        if isinstance(expr, TFieldAccess):
            resolved = self._resolve_field_type(expr)
            if isinstance(resolved, TMapType):
                return self._type(resolved.value)
        return "any"

    def _empty_map_call(self, args: list[TArg], ann: Ann | None = None) -> str:
        if len(args) >= 2:
            kt = self._infer_go_type(args[0].value)
            vt = self._infer_go_type(args[1].value)
            return "map[" + kt + "]" + vt + "{}"
        if ann is not None:
            # Try expected_type first (from checker context), then actual type
            for key in ("expected_type", "type"):
                type_ann = ann.get(key, "")
                if type_ann.startswith("map["):
                    go_t = self._ann_type_to_go(type_ann)
                    if go_t != "any":
                        return go_t + "{}"
        return "map[string]any{}"

    def _empty_set_call(self, args: list[TArg], ann: Ann | None = None) -> str:
        if len(args) >= 1:
            kt = self._infer_go_type(args[0].value)
            return "map[" + kt + "]bool{}"
        if ann is not None:
            for key in ("expected_type", "type"):
                type_ann = ann.get(key, "")
                if type_ann.startswith("set["):
                    go_t = self._ann_type_to_go(type_ann)
                    if go_t != "any":
                        return go_t + "{}"
        return "map[string]bool{}"

    # ── Try / Catch ───────────────────────────────────────────

    def _has_loop_control(self, stmts: list[TStmt]) -> bool:
        """Check if stmts contain continue/break (not inside nested loops/fns)."""
        for s in stmts:
            if isinstance(s, TContinueStmt) or isinstance(s, TBreakStmt):
                return True
            if isinstance(s, TIfStmt):
                if self._has_loop_control(s.then_body):
                    return True
                if s.else_body is not None:
                    if self._has_loop_control(s.else_body):
                        return True
            if isinstance(s, TTryStmt):
                if self._has_loop_control(s.body):
                    return True
                for c in s.catches:
                    if self._has_loop_control(c.body):
                        return True
        return False

    def _has_return_value(self, stmts: list[TStmt]) -> bool:
        """Check if stmts contain return statements with values."""
        for s in stmts:
            if isinstance(s, TReturnStmt) and s.value is not None:
                return True
            if isinstance(s, TIfStmt):
                if self._has_return_value(s.then_body):
                    return True
                if s.else_body is not None and self._has_return_value(s.else_body):
                    return True
            if isinstance(s, TMatchStmt):
                for c in s.cases:
                    if self._has_return_value(c.body):
                        return True
                if s.default is not None and self._has_return_value(s.default.body):
                    return True
        return False

    def _emit_try(self, stmt: TTryStmt) -> None:
        has_finally = stmt.finally_body is not None and len(stmt.finally_body) > 0
        # Check if body or catches contain continue/break
        needs_action = self._has_loop_control(stmt.body)
        if not needs_action and stmt.catches:
            for c in stmt.catches:
                if self._has_loop_control(c.body):
                    needs_action = True
                    break
        old_try_action = self._try_action_var
        action_var: str | None = None
        if needs_action:
            if old_try_action is not None:
                action_var = old_try_action
            else:
                action_var = "__tryAction"
                self._line(action_var + " := 0")
            self._try_action_var = action_var
        # Check if body or catches contain return statements with values
        needs_result = self._has_return_value(stmt.body)
        if not needs_result and stmt.catches:
            for c in stmt.catches:
                if self._has_return_value(c.body):
                    needs_result = True
                    break
        old_try_result = self._try_result_var
        if needs_result and self._current_ret_type is not None:
            ret_type_s = self._type(self._current_ret_type)
            self._line("var __tryResult " + ret_type_s)
            self._try_result_var = "__tryResult"
        if has_finally and stmt.finally_body is not None:
            self._line("func() {")
            self.indent += 1
            self._line("defer func() {")
            self.indent += 1
            self._emit_stmts(stmt.finally_body)
            self.indent -= 1
            self._line("}()")
        if stmt.catches:
            compact = self._try_compact_catch(stmt.catches)
            self._line("func() {")
            self.indent += 1
            if compact is not None:
                self._line("defer func() { if recover() != nil { " + compact + " } }()")
            else:
                self._line("defer func() {")
                self.indent += 1
                self._line("if r := recover(); r != nil {")
                self.indent += 1
                self._emit_catch_body(stmt.catches)
                self.indent -= 1
                self._line("}")
                self.indent -= 1
                self._line("}()")
            self._emit_stmts(stmt.body)
            self._emit_try_unused_vars(stmt.body)
            self.indent -= 1
            self._line("}()")
        else:
            self._emit_stmts(stmt.body)
        if has_finally:
            if action_var is not None:
                self._line("if " + action_var + " != 0 { return }")
            self.indent -= 1
            self._line("}()")
        self._try_action_var = old_try_action
        self._try_result_var = old_try_result
        if action_var is not None and old_try_action is None:
            self._line("if " + action_var + " == 1 {")
            self.indent += 1
            self._line("continue")
            self.indent -= 1
            self._line("} else if " + action_var + " == 2 {")
            self.indent += 1
            self._line("break")
            self.indent -= 1
            self._line("}")
        if needs_result and self._current_ret_type is not None:
            self._line("return __tryResult")

    def _emit_try_unused_vars(self, body: list[TStmt]) -> None:
        """Emit _ = x for let variables in try body that are never read."""
        let_names: list[str] = []
        for i, s in enumerate(body):
            if isinstance(s, TLetStmt):
                # Only include vars that were actually declared (not suppressed)
                if _stmts_ref_var(body[i + 1 :], s.name):
                    let_names.append(s.name)
        if not let_names:
            return
        used: set[str] = set()
        self._scan_var_refs(body, used)
        for name in let_names:
            safe = _safe_name(name)
            if name not in used:
                self._line("_ = " + safe)

    def _scan_var_refs(self, stmts: list[TStmt], used: set[str]) -> None:
        """Collect variable names referenced in expressions (not in let targets)."""
        for s in stmts:
            if isinstance(s, TLetStmt):
                if s.value is not None:
                    self._scan_expr_refs(s.value, used)
            elif isinstance(s, TAssignStmt):
                self._scan_expr_refs(s.value, used)
                self._scan_expr_refs(s.target, used)
            elif isinstance(s, TExprStmt):
                self._scan_expr_refs(s.expr, used)
            elif isinstance(s, TReturnStmt):
                if s.value is not None:
                    self._scan_expr_refs(s.value, used)

    def _scan_expr_refs(self, expr: TExpr, used: set[str]) -> None:
        if isinstance(expr, TVar):
            used.add(expr.name)
        elif isinstance(expr, TCall):
            self._scan_expr_refs(expr.func, used)
            for a in expr.args:
                self._scan_expr_refs(a.value, used)
        elif isinstance(expr, TBinaryOp):
            self._scan_expr_refs(expr.left, used)
            self._scan_expr_refs(expr.right, used)
        elif isinstance(expr, TUnaryOp):
            self._scan_expr_refs(expr.operand, used)
        elif isinstance(expr, TIndex):
            self._scan_expr_refs(expr.obj, used)
            self._scan_expr_refs(expr.index, used)
        elif isinstance(expr, TFieldAccess):
            self._scan_expr_refs(expr.obj, used)

    def _try_compact_catch(self, catches: list[TCatch]) -> str | None:
        """Return compact single-statement catch body, or None if not applicable."""
        if len(catches) != 1:
            return None
        catch = catches[0]
        if catch.types:
            return None
        unused = catch.annotations.get("liveness.catch_var_unused") == "true"
        if not unused:
            return None
        if len(catch.body) != 1:
            return None
        stmt = catch.body[0]
        if isinstance(stmt, TAssignStmt):
            return self._expr(stmt.target) + " = " + self._expr(stmt.value)
        return None

    def _emit_catch_body(self, catches: list[TCatch]) -> None:
        if len(catches) == 1:
            catch = catches[0]
            unused = catch.annotations.get("liveness.catch_var_unused") == "true"
            if not catch.types:
                # catch-all
                if not unused:
                    cname = _restore_name(catch.name, catch.annotations)
                    self._line(cname + " := r")
                    self._line("_ = " + cname)
                self._emit_stmts(catch.body)
            else:
                self._emit_typed_catches(catches)
        else:
            self._emit_typed_catches(catches)

    def _emit_typed_catches(self, catches: list[TCatch]) -> None:
        # If any catch uses the variable, keep bindings for all
        all_unused = all(
            c.annotations.get("liveness.catch_var_unused") == "true" for c in catches
        )
        first = True
        opened_if = False
        for catch in catches:
            unused = all_unused
            types = catch.types
            if not types:
                if first:
                    self._emit_stmts(catch.body)
                else:
                    if opened_if:
                        # default case after typed if/else-if catches
                        self._line("} else {")
                    else:
                        self._line("{")
                    self.indent += 1
                    if not unused:
                        cname = _restore_name(catch.name, catch.annotations)
                        self._line(cname + " := r")
                    self._emit_stmts(catch.body)
                    self.indent -= 1
                    self._line("}")
                    opened_if = False
                continue
            cname = _restore_name(catch.name, catch.annotations) if not unused else "_"
            type_checks: list[str] = []
            for t in types:
                if isinstance(t, TIdentType):
                    type_checks.append(t.name)
            if len(type_checks) == 1:
                tname = type_checks[0]
                kw = "if" if first else "} else if"
                if unused:
                    self._line(kw + " _, ok := r.(*" + tname + "); ok {")
                else:
                    self._line(kw + " " + cname + ", ok := r.(*" + tname + "); ok {")
                self.indent += 1
                self._emit_stmts(catch.body)
                self.indent -= 1
                opened_if = True
            else:
                # Union catch: switch r.(type) { case *A, *B: ... }
                if opened_if:
                    self._line("}")
                    opened_if = False
                self._line("switch r.(type) {")
                cases = ", ".join("*" + t for t in type_checks)
                self._line("case " + cases + ":")
                self.indent += 1
                if not unused:
                    self._line(cname + " := r")
                self._emit_stmts(catch.body)
                self.indent -= 1
                self._line("}")
            first = False
        if opened_if:
            self._line("}")

    # ── Match ─────────────────────────────────────────────────

    def _emit_match(self, stmt: TMatchStmt) -> None:
        expr_s = self._expr_no_narrowing(stmt.expr)
        if expr_s.startswith("*"):
            expr_s = "(" + expr_s + ")"
        is_enum = self._is_enum_match(stmt)
        is_optional = self._is_optional_match(stmt)
        if is_optional:
            self._emit_optional_match(stmt, expr_s)
        elif is_enum:
            self._line("switch " + expr_s + " {")
            for case in stmt.cases:
                self._emit_match_case_enum(case, stmt)
            if stmt.default is not None:
                self._line("default:")
                self.indent += 1
                self._emit_stmts(stmt.default.body)
                self.indent -= 1
            self._line("}")
        else:
            # Type switch
            # If scrutinee is already a concrete type, skip the switch
            if isinstance(stmt.expr, TVar):
                vt = self.var_types.get(stmt.expr.name)
                if vt is not None:
                    go_t = self._type(vt).lstrip("*")
                    if go_t not in self._interface_names and go_t != "any":
                        # Concrete type — emit matching case body directly
                        for c in stmt.cases:
                            if isinstance(c.pattern, TPatternType):
                                if (
                                    c.pattern.name is not None
                                    and c.pattern.annotations.get(
                                        "liveness.match_var_unused"
                                    )
                                    != "true"
                                ):
                                    safe = _restore_name(
                                        c.pattern.name, c.pattern.annotations
                                    )
                                    self._line(safe + " := " + self._expr(stmt.expr))
                                self._emit_stmts(c.body)
                                return
                        if stmt.default is not None:
                            self._emit_stmts(stmt.default.body)
                        return
            switch_var = self._find_switch_var(stmt)
            # Check if any case needs the binding
            any_used = False
            for c in stmt.cases:
                if (
                    isinstance(c.pattern, TPatternType)
                    and c.pattern.name is not None
                    and c.pattern.annotations.get("liveness.match_var_unused") != "true"
                ):
                    any_used = True
                    break
            has_default_binding = (
                stmt.default is not None and stmt.default.name is not None
            )
            needs_capture = any_used or has_default_binding
            if not needs_capture and isinstance(stmt.expr, TVar):
                scrutinee_name = stmt.expr.name
                for c in stmt.cases:
                    if _stmts_ref_var(c.body, scrutinee_name):
                        needs_capture = True
                        break
            if needs_capture:
                self._line("switch " + switch_var + " := " + expr_s + ".(type) {")
            else:
                self._line("switch " + expr_s + ".(type) {")
            for case in stmt.cases:
                self._emit_match_case_type(case, stmt, not any_used, switch_var, expr_s)
            if stmt.default is not None:
                if stmt.default.name is not None:
                    dname = _restore_name(stmt.default.name, stmt.default.annotations)
                    self._line("default:")
                    self.indent += 1
                    self._line(dname + " := " + switch_var)
                    self._emit_stmts(stmt.default.body)
                    self.indent -= 1
                else:
                    self._line("default:")
                    self.indent += 1
                    self._emit_stmts(stmt.default.body)
                    self.indent -= 1
            self._line("}")

    def _is_enum_match(self, stmt: TMatchStmt) -> bool:
        for case in stmt.cases:
            if case.pattern is not None:
                if isinstance(case.pattern, TPatternEnum):
                    return True
                if isinstance(case.pattern, TPatternNil):
                    return False
                if isinstance(case.pattern, TPatternType):
                    return False
        return False

    def _is_optional_match(self, stmt: TMatchStmt) -> bool:
        has_nil = False
        has_type = False
        for case in stmt.cases:
            if case.pattern is not None:
                if isinstance(case.pattern, TPatternNil):
                    has_nil = True
                elif isinstance(case.pattern, TPatternType):
                    has_type = True
        return has_nil and has_type and len(stmt.cases) == 2

    def _emit_optional_match(self, stmt: TMatchStmt, expr_s: str) -> None:
        nil_case = None
        type_case = None
        for case in stmt.cases:
            if isinstance(case.pattern, TPatternNil):
                nil_case = case
            elif isinstance(case.pattern, TPatternType):
                type_case = case
        if not isinstance(nil_case, TMatchCase) or not isinstance(
            type_case, TMatchCase
        ):
            return
        # Get the raw variable name without dereference
        raw_expr = expr_s
        if isinstance(stmt.expr, TVar):
            raw_expr = _restore_name(stmt.expr.name, stmt.expr.annotations)
        self._line("if " + raw_expr + " == nil {")
        self.indent += 1
        self._emit_stmts(nil_case.body)
        self.indent -= 1
        self._line("} else {")
        self.indent += 1
        if isinstance(type_case.pattern, TPatternType):
            if type_case.pattern.name is not None:
                bname = _restore_name(type_case.pattern.name, type_case.annotations)
                self._line(bname + " := *" + raw_expr)
        self._emit_stmts(type_case.body)
        self.indent -= 1
        self._line("}")

    def _find_switch_var(self, stmt: TMatchStmt) -> str:
        # Use the first case's binding name as the switch variable
        for case in stmt.cases:
            if isinstance(case.pattern, TPatternType) and case.pattern.name:
                name = _restore_name(case.pattern.name, case.annotations)
                if name not in self.struct_names and name not in self._interface_names:
                    return name
        # Fall back to default binding
        if stmt.default is not None:
            dname = stmt.default.name
            if dname is not None:
                name = _restore_name(dname, stmt.default.annotations)
                if name not in self.struct_names and name not in self._interface_names:
                    return name
        return "__v"

    def _emit_match_case_enum(self, case: TMatchCase, stmt: TMatchStmt) -> None:
        pat = case.pattern
        if isinstance(pat, TPatternEnum):
            enum_name = pat.enum_name
            variant = pat.variant
            self._line("case " + enum_name + variant + ":")
        else:
            self._line("case " + self._expr_pattern(pat) + ":")
        self.indent += 1
        self._emit_stmts(case.body)
        self.indent -= 1

    def _emit_match_case_type(
        self,
        case: TMatchCase,
        stmt: TMatchStmt,
        var_unused: bool,
        switch_var: str,
        expr_s: str,
    ) -> None:
        pat = case.pattern
        if isinstance(pat, TPatternType):
            tname = pat.type_name
            if isinstance(tname, TPrimitive):
                type_str = self._type(tname)
                self._line("case " + type_str + ":")
            elif isinstance(tname, TIdentType):
                type_str = tname.name
                ptr = "" if type_str in self._interface_names else "*"
                self._line("case " + ptr + type_str + ":")
            else:
                type_str = self._type(tname)
                self._line("case " + type_str + ":")
            self.indent += 1
            pat_unused = pat.annotations.get("liveness.match_var_unused") == "true"
            aliases_added: list[str] = []
            # Determine the concrete type for this case arm
            case_type: TType | None = None
            if isinstance(tname, TIdentType):
                case_type = tname
            saved_var_types: list[tuple[str, TType | None]] = []
            if pat.name is not None and not pat_unused:
                bname = _restore_name(pat.name, case.annotations)
                if bname != switch_var:
                    if bname in self.struct_names or bname in self._interface_names:
                        self._var_aliases[bname] = switch_var
                        aliases_added.append(bname)
                    else:
                        self._line(bname + " := " + switch_var)
                        if case_type is not None:
                            saved_var_types.append(
                                (pat.name, self.var_types.get(pat.name))
                            )
                            self.var_types[pat.name] = case_type
            if isinstance(stmt.expr, TVar):
                scrutinee = _restore_name(stmt.expr.name, stmt.expr.annotations)
                if scrutinee != switch_var and _stmts_ref_var(
                    case.body, stmt.expr.name
                ):
                    self._line(scrutinee + " := " + switch_var)
                if case_type is not None:
                    saved_var_types.append(
                        (stmt.expr.name, self.var_types.get(stmt.expr.name))
                    )
                    self.var_types[stmt.expr.name] = case_type
            self._emit_stmts(case.body)
            for alias in aliases_added:
                del self._var_aliases[alias]
            for vname, old_type in saved_var_types:
                if old_type is None:
                    self.var_types.pop(vname, None)
                else:
                    self.var_types[vname] = old_type
            self.indent -= 1
        elif isinstance(pat, TPatternNil):
            self._line("case nil:")
            self.indent += 1
            self._emit_stmts(case.body)
            self.indent -= 1
        else:
            self._line("default:")
            self.indent += 1
            self._emit_stmts(case.body)
            self.indent -= 1

    def _expr_pattern(self, pat: TPattern) -> str:
        return "default"

    # ── Ternary ───────────────────────────────────────────────

    def _is_isinstance_tuple_expr(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TBinaryOp)
            and expr.op == "||"
            and expr.annotations.get("provenance") == "isinstance_tuple"
        )

    def _emit_isinstance_tuple_return(self, expr: TBinaryOp) -> None:
        types: list[str] = []
        obj = self._flatten_isinstance_tuple(expr, types)
        if obj is not None and types:
            cases = ", ".join("*" + t for t in types)
            self._line("switch " + obj + ".(type) {")
            self._line("case " + cases + ":")
            self.indent += 1
            self._line("return true")
            self.indent -= 1
            self._line("}")
            self._line("return false")
        else:
            self._line("return " + self._expr(expr))

    def _emit_ternary_return(self, expr: TTernary) -> None:
        """Go has no ternary — emit as if/else return."""
        prov = expr.annotations.get("provenance", "")
        if prov == "partition" or prov == "rpartition":
            self._emit_partition_return(expr)
            return
        if prov == "none_coalesce":
            self._emit_none_coalesce_return(expr)
            return
        if prov == "removeprefix":
            self._emit_cut_prefix_return(expr)
            return
        if prov == "removesuffix":
            self._emit_cut_suffix_return(expr)
            return
        self._line("if " + self._expr(expr.cond) + " {")
        self.indent += 1
        if isinstance(expr.then_expr, TTernary):
            self._emit_ternary_return(expr.then_expr)
        else:
            self._line("return " + self._expr(expr.then_expr))
        self.indent -= 1
        self._line("} else {")
        self.indent += 1
        if isinstance(expr.else_expr, TTernary):
            self._emit_ternary_return(expr.else_expr)
        else:
            self._line("return " + self._expr(expr.else_expr))
        self.indent -= 1
        self._line("}")

    def _emit_ternary_stmt(self, expr: TTernary, target: str) -> None:
        """Emit ternary as if/else assignment."""
        self._line("if " + self._expr(expr.cond) + " {")
        self.indent += 1
        if isinstance(expr.then_expr, TTernary):
            self._emit_ternary_stmt(expr.then_expr, target)
        else:
            self._line(target + " = " + self._expr(expr.then_expr))
        self.indent -= 1
        self._line("} else {")
        self.indent += 1
        if isinstance(expr.else_expr, TTernary):
            self._emit_ternary_stmt(expr.else_expr, target)
        else:
            self._line(target + " = " + self._expr(expr.else_expr))
        self.indent -= 1
        self._line("}")

    def _emit_none_coalesce_return(self, expr: TTernary) -> None:
        cond = expr.cond
        if (
            isinstance(cond, TBinaryOp)
            and cond.op == "!="
            and isinstance(cond.right, TNilLit)
        ):
            var_expr = cond.left
            if isinstance(var_expr, TVar):
                name = _restore_name(var_expr.name, var_expr.annotations)
                self._line("if " + name + " != nil { return *" + name + " }")
                self._line("return " + self._expr(expr.else_expr))
                return
        self._line("if " + self._expr(expr.cond) + " {")
        self.indent += 1
        self._line("return " + self._expr(expr.then_expr))
        self.indent -= 1
        self._line("}")
        self._line("return " + self._expr(expr.else_expr))

    def _emit_cut_prefix_return(self, expr: TTernary) -> None:
        cond = expr.cond
        if (
            isinstance(cond, TCall)
            and isinstance(cond.func, TVar)
            and cond.func.name == "StartsWith"
        ):
            s = self._expr(cond.args[0].value)
            p = self._expr(cond.args[1].value)
            self._line(
                "if __cut, __ok := strings.CutPrefix(" + s + ", " + p + "); __ok {"
            )
            self.indent += 1
            self._line("return __cut")
            self.indent -= 1
            self._line("}")
            self._line("return " + s)
            return
        self._emit_ternary_return(
            TTernary(
                pos=expr.pos,
                cond=cond,
                then_expr=expr.then_expr,
                else_expr=expr.else_expr,
                annotations={},
            )
        )

    def _emit_cut_suffix_return(self, expr: TTernary) -> None:
        cond = expr.cond
        if (
            isinstance(cond, TCall)
            and isinstance(cond.func, TVar)
            and cond.func.name == "EndsWith"
        ):
            s = self._expr(cond.args[0].value)
            p = self._expr(cond.args[1].value)
            self._line(
                "if __cut, __ok := strings.CutSuffix(" + s + ", " + p + "); __ok {"
            )
            self.indent += 1
            self._line("return __cut")
            self.indent -= 1
            self._line("}")
            self._line("return " + s)
            return
        self._emit_ternary_return(
            TTernary(
                pos=expr.pos,
                cond=cond,
                then_expr=expr.then_expr,
                else_expr=expr.else_expr,
                annotations={},
            )
        )

    def _emit_partition_return(self, expr: TTernary) -> None:
        prov = expr.annotations.get("provenance", "")
        cond = expr.cond
        if isinstance(cond, TBinaryOp) and isinstance(cond.left, TCall):
            call = cond.left
            obj_s = self._expr(call.args[0].value)
            sep_s = self._expr(call.args[1].value)
            method = "strings.Index" if prov == "partition" else "strings.LastIndex"
            self._line("__idx := " + method + "(" + obj_s + ", " + sep_s + ")")
            self._line("if __idx >= 0 {")
            self.indent += 1
            self._line(
                "return [3]any{"
                + obj_s
                + "[:__idx], "
                + sep_s
                + ", "
                + obj_s
                + "[__idx+len("
                + sep_s
                + "):]}"
            )
            self.indent -= 1
            self._line("} else {")
            self.indent += 1
            if prov == "partition":
                self._line("return [3]any{" + obj_s + ', "", ""}')
            else:
                self._line('return [3]any{"", "", ' + obj_s + "}")
            self.indent -= 1
            self._line("}")
            return
        self._emit_ternary_return(expr)

    # ── any_call / all_call ───────────────────────────────────

    def _emit_any_all(
        self,
        stmts: list[TStmt],
        i: int,
        let_stmt: TLetStmt,
        for_stmt: TForStmt,
        prov: str,
    ) -> int:
        # For Go, emit as loop (no array methods like .some/.every)
        self._emit_let(let_stmt)
        self._emit_for(for_stmt)
        return 2

    # ── Expressions ───────────────────────────────────────────

    def _expr(self, expr: TExpr) -> str:
        if isinstance(expr, TIntLit):
            return self._int_lit(expr)
        if isinstance(expr, TFloatLit):
            if expr.raw == "0.0 / 0.0":
                return "math.NaN()"
            if expr.raw == "1.0 / 0.0":
                return "math.Inf(1)"
            if expr.raw == "-1.0 / 0.0":
                return "math.Inf(-1)"
            return expr.raw
        if isinstance(expr, TStringLit):
            return '"' + escape_string(expr.value) + '"'
        if isinstance(expr, TBoolLit):
            return "true" if expr.value else "false"
        if isinstance(expr, TNilLit):
            return "nil"
        if isinstance(expr, TByteLit):
            return expr.raw
        if isinstance(expr, TBytesLit):
            return self._bytes_lit(expr)
        if isinstance(expr, TRuneLit):
            v = expr.value
            if v == "'":
                return "'\\''"
            return "'" + escape_string(v).replace('\\"', '"') + "'"
        if isinstance(expr, TVar):
            if expr.name == self.self_name:
                return "self"
            name = _restore_name(expr.name, expr.annotations)
            name = self._var_aliases.get(name, name)
            if self._needs_deref(expr):
                return "*" + name
            narrowed = expr.annotations.get("scope.narrowed_type", "")
            if (
                narrowed
                and narrowed in self.struct_names
                and narrowed not in self._interface_names
            ):
                vt = self.var_types.get(expr.name)
                if vt is not None:
                    go_t = self._type(vt).lstrip("*")
                    if go_t in self._interface_names or go_t == "any":
                        return name + ".(*" + narrowed + ")"
            return name
        if isinstance(expr, TFieldAccess):
            # Enum variant access: Color.Red → ColorRed
            if isinstance(expr.obj, TVar) and expr.obj.name in self._enum_names:
                return expr.obj.name + expr.field
            # Type assertion for narrowed interface variables or field paths
            narrowed = expr.obj.annotations.get("scope.narrowed_type", "")
            if (
                narrowed != ""
                and narrowed in self.struct_names
                and narrowed not in self._interface_names
            ):
                obj_s = self._expr_no_narrowing(expr.obj)
                fname = (
                    expr.field[0].upper() + expr.field[1:] if expr.field else expr.field
                )
                # Skip assertion if variable is already the concrete type
                if isinstance(expr.obj, TVar):
                    vt = self.var_types.get(expr.obj.name)
                    if vt is not None:
                        go_t = self._type(vt).lstrip("*")
                        if go_t == narrowed or (
                            go_t not in self._interface_names and go_t != "any"
                        ):
                            return obj_s + "." + fname
                return obj_s + ".(*" + narrowed + ")." + fname
            # Interface common field accessor
            if self._is_interface_field_access(expr):
                obj_s = self._expr(expr.obj)
                fname = (
                    expr.field[0].upper() + expr.field[1:] if expr.field else expr.field
                )
                return obj_s + ".Get" + fname + "()"
            obj_s = self._expr(expr.obj)
            fname = expr.field[0].upper() + expr.field[1:] if expr.field else expr.field
            result = obj_s + "." + fname
            if self._is_optional_primitive_field(expr):
                return "*" + result
            return result
        if isinstance(expr, TTupleAccess):
            return self._expr(expr.obj) + "[" + str(expr.index) + "]"
        if isinstance(expr, TIndex):
            neg = self._negative_index(expr)
            if neg is not None:
                return self._expr(expr.obj) + "[" + neg + "]"
            return self._expr(expr.obj) + "[" + self._expr(expr.index) + "]"
        if isinstance(expr, TSlice):
            return self._slice(expr)
        if isinstance(expr, TBinaryOp):
            return self._binary(expr)
        if isinstance(expr, TUnaryOp):
            return self._unary(expr)
        if isinstance(expr, TTernary):
            return self._ternary(expr)
        if isinstance(expr, TListLit):
            elem_type = self._infer_list_elem_type(expr)
            elems = self._join_exprs(expr.elements, ", ")
            return elem_type + "{" + elems + "}"
        if isinstance(expr, TMapLit):
            return self._map_lit(expr)
        if isinstance(expr, TSetLit):
            return self._set_lit(expr)
        if isinstance(expr, TTupleLit):
            elems = self._join_exprs(expr.elements, ", ")
            elem_type = "any"
            if expr.elements:
                types = [self._infer_go_type(e) for e in expr.elements]
                if types[0] and all(t == types[0] for t in types[1:]):
                    elem_type = types[0]
            return "[" + str(len(expr.elements)) + "]" + elem_type + "{" + elems + "}"
        if isinstance(expr, TFnLit):
            return self._fn_lit(expr)
        if isinstance(expr, TCall):
            return self._call(expr)
        raise NotImplementedError("unknown expression")

    def _int_lit(self, expr: TIntLit) -> str:
        raw = expr.raw
        if raw.startswith(("0x", "0X", "0o", "0O", "0b", "0B")):
            return raw
        v = expr.value
        if v > (1 << 63) - 1:
            return str(v - (1 << 64))
        return str(v)

    def _bytes_lit(self, expr: TBytesLit) -> str:
        parts: list[str] = []
        for b in expr.value:
            parts.append(str(b))
        return "[]byte{" + ", ".join(parts) + "}"

    def _infer_list_elem_type(self, expr: TListLit) -> str:
        for key in ("expected_type", "type"):
            ann = expr.annotations.get(key, "")
            if ann.startswith("list["):
                inner = ann[5:-1]
                if inner in self._interface_names:
                    return "[]" + inner
                if inner in self.struct_names:
                    return "[]*" + inner
                go_inner = self._ann_type_to_go(inner)
                if go_inner != "any":
                    return "[]" + go_inner
        if expr.elements:
            first = expr.elements[0]
            if isinstance(first, TIntLit):
                return "[]int"
            if isinstance(first, TFloatLit):
                return "[]float64"
            if isinstance(first, TStringLit):
                return "[]string"
            if isinstance(first, TBoolLit):
                return "[]bool"
            if isinstance(first, TVar):
                vt = self.var_types.get(first.name)
                if isinstance(vt, TIdentType):
                    for iface in self._interface_children:
                        if vt.name in self._interface_children[iface]:
                            return "[]" + iface
                    if vt.name in self.struct_names:
                        return "[]*" + vt.name
            if isinstance(first, TCall):
                if isinstance(first.func, TVar):
                    fname = first.func.name
                    for iface in self._interface_children:
                        if fname in self._interface_children[iface]:
                            return "[]" + iface
                    if fname in self.struct_names:
                        return "[]*" + fname
        return "[]any"

    def _type_str_to_go(self, s: str) -> str:
        if s == "int":
            return "[]int"
        if s == "float":
            return "[]float64"
        if s == "string":
            return "[]string"
        if s == "bool":
            return "[]bool"
        if s == "rune":
            return "[]rune"
        if s == "byte":
            return "[]byte"
        if s.startswith("list["):
            return "[]" + self._type_str_to_go(s[5:-1])
        return "[]any"

    def _map_lit(self, expr: TMapLit) -> str:
        ann = expr.annotations.get("type", "")
        if ann:
            go_type = self._ann_type_to_go(ann)
        elif expr.entries:
            k0, v0 = expr.entries[0]
            kt = self._infer_go_type(k0)
            vt = self._infer_go_type(v0)
            go_type = "map[" + kt + "]" + vt
        else:
            go_type = "map[string]any"
        if not expr.entries:
            return go_type + "{}"
        pairs: list[str] = []
        for k, v in expr.entries:
            pairs.append(self._expr(k) + ": " + self._expr(v))
        return go_type + "{" + ", ".join(pairs) + "}"

    def _set_lit(self, expr: TSetLit) -> str:
        ann = expr.annotations.get("type", "")
        if ann:
            go_type = self._ann_type_to_go(ann)
        elif expr.elements:
            et = self._infer_go_type(expr.elements[0])
            go_type = "map[" + et + "]bool"
        else:
            go_type = "map[any]bool"
        if not expr.elements:
            return go_type + "{}"
        pairs: list[str] = []
        for e in expr.elements:
            pairs.append(self._expr(e) + ": true")
        return go_type + "{" + ", ".join(pairs) + "}"

    def _ann_type_to_go(self, ann: str) -> str:
        if ann == "int":
            return "int"
        if ann == "float":
            return "float64"
        if ann == "string":
            return "string"
        if ann == "bool":
            return "bool"
        if ann.startswith("map["):
            # Parse map[K, V] or map[K]V
            depth = 0
            comma_pos = -1
            i = 4
            while i < len(ann):
                ch = ann[i]
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    if depth == 0:
                        if comma_pos > 0:
                            key_part = ann[4:comma_pos].strip()
                            val_part = ann[comma_pos + 1 : i].strip()
                            kt = self._ann_type_to_go(key_part)
                            vt = self._ann_type_to_go(val_part)
                            return "map[" + kt + "]" + vt
                        key_part = ann[4:i]
                        val_part = ann[i + 1 :].strip()
                        kt = self._ann_type_to_go(key_part)
                        vt = self._ann_type_to_go(val_part)
                        return "map[" + kt + "]" + vt
                    depth -= 1
                elif ch == "," and depth == 0:
                    comma_pos = i
                i += 1
        if ann.startswith("set["):
            inner = ann[4:-1]
            return "map[" + self._ann_type_to_go(inner) + "]bool"
        if ann.startswith("list["):
            inner = ann[5:-1]
            return "[]" + self._ann_type_to_go(inner)
        if ann.startswith("tuple[") and ann.endswith("]"):
            parts = _split_tuple_ann(ann[6:-1])
            go_parts = [self._ann_type_to_go(p) for p in parts]
            if all(t == go_parts[0] for t in go_parts[1:]):
                return "[" + str(len(parts)) + "]" + go_parts[0]
            return "[" + str(len(parts)) + "]any"
        if ann.startswith("(") and ann.endswith(")"):
            parts = _split_tuple_ann(ann[1:-1])
            go_parts = [self._ann_type_to_go(p) for p in parts]
            if all(t == go_parts[0] for t in go_parts[1:]):
                return "[" + str(len(parts)) + "]" + go_parts[0]
            return "[" + str(len(parts)) + "]any"
        if ann in self._interface_names or ann in self.struct_names:
            return ann
        return "any"

    def _infer_go_type(self, expr: TExpr) -> str:
        ann = expr.annotations.get("type", "")
        if ann:
            return self._ann_type_to_go(ann)
        if isinstance(expr, TIntLit):
            return "int"
        if isinstance(expr, TFloatLit):
            return "float64"
        if isinstance(expr, TStringLit):
            return "string"
        if isinstance(expr, TBoolLit):
            return "bool"
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            if typ is not None:
                return self._type(typ)
        return "any"

    def _is_len_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Len"
        )

    def _is_zero(self, expr: TExpr) -> bool:
        return isinstance(expr, TIntLit) and expr.value == 0

    def _slice(self, expr: TSlice) -> str:
        obj = self._expr(expr.obj)
        prov = expr.annotations.get("provenance", "")
        low = self._expr(expr.low)
        high = self._expr(expr.high)
        if self._is_len_call(expr.high):
            return obj + "[" + low + ":]"
        if prov == "open_start" and self._is_zero(expr.low):
            return obj + "[:" + high + "]"
        return obj + "[" + low + ":" + high + "]"

    def _negative_index(self, expr: TIndex) -> str | None:
        idx = expr.index
        if isinstance(idx, TBinaryOp) and idx.op == "-":
            if (
                isinstance(idx.left, TCall)
                and isinstance(idx.left.func, TVar)
                and idx.left.func.name == "Len"
            ):
                return (
                    "len("
                    + self._expr(idx.left.args[0].value)
                    + ")-"
                    + self._expr(idx.right)
                )
        return None

    def _binary(self, expr: TBinaryOp) -> str:
        op = expr.op
        # Detect NaN/Inf special float division patterns
        if op == "/" and isinstance(expr.right, TFloatLit) and expr.right.raw == "0.0":
            if isinstance(expr.left, TFloatLit):
                if expr.left.raw == "0.0":
                    return "math.NaN()"
                if expr.left.raw == "1.0":
                    return "math.Inf(1)"
            if isinstance(expr.left, TUnaryOp) and expr.left.op == "-":
                inner = expr.left.operand
                if isinstance(inner, TFloatLit) and inner.raw == "1.0":
                    return "math.Inf(-1)"
        if self.strict_math:
            if op in STRICT_INT_BINARY and self._is_int_expr(expr.left):
                fn = _STRICT_GO_BINARY[op]
                self._used_strict_helpers.add(STRICT_INT_BINARY[op])
                return (
                    fn
                    + "("
                    + self._expr(expr.left)
                    + ", "
                    + self._expr(expr.right)
                    + ")"
                )
            if op == "%" and self._is_float_expr(expr.left):
                self._used_strict_helpers.add("strict_fmod")
                return (
                    "strictFmod("
                    + self._expr(expr.left)
                    + ", "
                    + self._expr(expr.right)
                    + ")"
                )
        # isinstance tuple
        if op == "||" and expr.annotations.get("provenance") == "isinstance_tuple":
            types: list[str] = []
            obj = self._flatten_isinstance_tuple(expr, types)
            if obj is not None and types:
                cases = ", ".join("*" + t for t in types)
                return (
                    "func() bool { switch "
                    + obj
                    + ".(type) { case "
                    + cases
                    + ": return true }; return false }()"
                )
        # bytes comparisons: use bytes.Equal
        if op in ("==", "!=") and (
            isinstance(expr.left, TBytesLit) or isinstance(expr.right, TBytesLit)
        ):
            left_str = self._expr(expr.left)
            right_str = self._expr(expr.right)
            eq = "bytes.Equal(" + left_str + ", " + right_str + ")"
            if op == "!=":
                return "!" + eq
            return eq
        # nil comparisons: don't deref the variable or field
        if op in ("==", "!=") and isinstance(expr.right, TNilLit):
            if isinstance(expr.left, TVar):
                name = _restore_name(expr.left.name, expr.left.annotations)
                return name + " " + op + " nil"
            if isinstance(expr.left, TFieldAccess):
                return self._field_access_raw(expr.left) + " " + op + " nil"
        if op in ("==", "!=") and isinstance(expr.left, TNilLit):
            if isinstance(expr.right, TVar):
                name = _restore_name(expr.right.name, expr.right.annotations)
                return "nil " + op + " " + name
            if isinstance(expr.right, TFieldAccess):
                return "nil " + op + " " + self._field_access_raw(expr.right)
        left_str = self._maybe_paren(expr.left, op, is_left=True)
        right_str = self._maybe_paren(expr.right, op, is_left=False)
        return left_str + " " + op + " " + right_str

    def _unary(self, expr: TUnaryOp) -> str:
        op = expr.op
        if self.strict_math and op == "-" and self._is_int_expr(expr.operand):
            self._used_strict_helpers.add("checked_neg_i64")
            return "checkedNegI64(" + self._expr(expr.operand) + ")"
        if op == "~":
            return "^" + self._expr(expr.operand)
        if op == "!":
            if isinstance(expr.operand, (TBinaryOp,)):
                if expr.operand.op in ("&&", "||"):
                    return "!(" + self._expr(expr.operand) + ")"
                if expr.operand.op in (">", "<", ">=", "<=", "==", "!="):
                    return "!(" + self._expr(expr.operand) + ")"
                return "!" + self._expr(expr.operand)
            if isinstance(expr.operand, (TTernary,)):
                return "!(" + self._expr(expr.operand) + ")"
            if self._is_isnil_call(expr.operand) and isinstance(expr.operand, TCall):
                # !IsNil(x) → x != nil
                nil_arg = expr.operand.args[0].value
                if isinstance(nil_arg, TVar):
                    raw = _restore_name(nil_arg.name, nil_arg.annotations)
                    return raw + " != nil"
                if isinstance(nil_arg, TFieldAccess):
                    return self._field_access_raw(nil_arg) + " != nil"
                return self._a(expr.operand.args, 0) + " != nil"
            return "!" + self._expr(expr.operand)
        if op == "*" and isinstance(expr.operand, TVar):
            unwrap_vt = self.var_types.get(expr.operand.name)
            if self._is_go_ref_type(unwrap_vt):
                return _restore_name(expr.operand.name, expr.operand.annotations)
        if isinstance(expr.operand, (TBinaryOp, TTernary)):
            return op + "(" + self._expr(expr.operand) + ")"
        inner = self._expr(expr.operand)
        if op == "-" and inner.startswith("-"):
            return "-(" + inner + ")"
        return op + inner

    def _ternary(self, expr: TTernary) -> str:
        prov = expr.annotations.get("provenance", "")
        if prov == "none_coalesce":
            return self._none_coalesce(expr)
        # Go has no ternary operator — this should not normally be reached
        # as ternaries should be handled at the statement level
        return self._expr(expr.then_expr)

    def _none_coalesce(self, expr: TTernary) -> str:
        cond = expr.cond
        if isinstance(cond, TBinaryOp) and cond.op == "!=":
            if isinstance(cond.right, TNilLit):
                return self._expr(cond.left)
            if isinstance(cond.left, TNilLit):
                return self._expr(cond.right)
        return self._expr(expr.then_expr)

    def _flatten_isinstance_tuple(
        self, expr: TBinaryOp, types: list[str]
    ) -> str | None:
        obj: str | None = None
        if isinstance(expr.left, TBinaryOp) and expr.left.op == "||":
            obj = self._flatten_isinstance_tuple(expr.left, types)
        elif isinstance(expr.left, TCall) and self._is_isinstance_call(expr.left):
            obj = self._expr(expr.left.args[0].value)
            types.append(self._type_name_from_arg(expr.left.args[1].value))
        else:
            return None
        if isinstance(expr.right, TCall) and self._is_isinstance_call(expr.right):
            types.append(self._type_name_from_arg(expr.right.args[1].value))
        return obj

    def _type_name_from_arg(self, expr: TExpr) -> str:
        if isinstance(expr, TStringLit):
            return expr.value
        return self._expr(expr)

    def _is_isinstance_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name in ("IsInstance", "IsType")
        )

    def _is_isnil_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "IsNil"
        )

    def _maybe_paren(self, expr: TExpr, parent_op: str, is_left: bool) -> str:
        if isinstance(expr, TBinaryOp):
            if _needs_parens(expr.op, parent_op, is_left):
                return "(" + self._expr(expr) + ")"
        elif isinstance(expr, TTernary):
            return "(" + self._expr(expr) + ")"
        elif isinstance(expr, TUnaryOp):
            if expr.op == "!" and parent_op in _CMP_OPS:
                return "(" + self._expr(expr) + ")"
        return self._expr(expr)

    def _fn_lit(self, expr: TFnLit) -> str:
        params_parts: list[str] = []
        for p in expr.params:
            if p.typ is not None:
                params_parts.append(
                    _restore_name(p.name, p.annotations) + " " + self._type(p.typ)
                )
        params = ", ".join(params_parts)
        ret_type = ""
        if expr.ret is not None:
            r = self._type(expr.ret)
            if r:
                ret_type = " " + r
        first = expr.body[0] if expr.body else None
        if isinstance(first, TReturnStmt) and len(expr.body) == 1:
            fv = first.value
            if fv is not None:
                if not isinstance(fv, TTernary):
                    return (
                        "func("
                        + params
                        + ")"
                        + ret_type
                        + " { return "
                        + self._expr(fv)
                        + " }"
                    )
        # Arrow expression body
        if (
            expr.annotations.get("fn_lit.arrow") == "true"
            and isinstance(first, TExprStmt)
            and len(expr.body) == 1
        ):
            if ret_type.strip():
                return (
                    "func("
                    + params
                    + ")"
                    + ret_type
                    + " { return "
                    + self._expr(first.expr)
                    + " }"
                )
            return (
                "func("
                + params
                + ")"
                + ret_type
                + " { "
                + self._expr(first.expr)
                + " }"
            )
        old_lines = self.lines
        old_indent = self.indent
        self.lines = []
        self.indent = 0
        for s in expr.body:
            self._emit_stmt(s)
        body_lines = self.lines
        self.lines = old_lines
        self.indent = old_indent
        result = "func(" + params + ")" + ret_type + " {\n"
        for bl in body_lines:
            result += "\t" * self.indent + "\t" + bl + "\n"
        result += "\t" * self.indent + "}"
        return result

    # ── Calls ─────────────────────────────────────────────────

    def _call(self, expr: TCall) -> str:
        func = expr.func
        args = expr.args
        # Star unpack
        if (
            isinstance(func, TVar)
            and func.name == "Concat"
            and expr.annotations.get("provenance") == "star_unpack"
        ):
            return self._star_unpack(expr)
        # Reversed/Reverse with reversed_slice
        if (
            isinstance(func, TVar)
            and func.name in ("Reversed", "Reverse")
            and expr.annotations.get("provenance") == "reversed_slice"
        ):
            inner = args[0].value
            ann_type = inner.annotations.get("type", "")
            if ann_type == "string":
                a = self._a(args, 0)
                self._need_reverse_string = True
                return "reverseString(" + a + ")"
            a = self._a(args, 0)
            return (
                "slices.Clone(func() []any { s := slices.Clone("
                + a
                + "); slices.Reverse(s); return s }())"
            )
        # list_repetition
        if (
            isinstance(func, TVar)
            and func.name == "Repeat"
            and expr.annotations.get("provenance") == "list_repetition"
        ):
            count = self._a(args, 1)
            elem = args[0].value
            if isinstance(elem, TListLit) and len(elem.elements) == 1:
                et = self._infer_go_type(elem.elements[0])
                return "make([]" + et + ", " + count + ")"
            return "make([]any, " + count + ")"
        # ListFrom/SetFromList with dict_keys provenance
        if (
            isinstance(func, TVar)
            and func.name in ("ListFrom", "SetFromList")
            and expr.annotations.get("provenance") == "dict_keys"
        ):
            inner = args[0].value
            if isinstance(inner, TCall):
                dict_expr = self._expr(inner.args[0].value)
                kt = self._infer_map_key_type(inner.args[0].value)
                if func.name == "ListFrom":
                    return (
                        "func() []"
                        + kt
                        + " { var r []"
                        + kt
                        + "; for k := range "
                        + dict_expr
                        + " { r = append(r, k) }; return r }()"
                    )
                else:
                    return (
                        "func() map["
                        + kt
                        + "]bool { s := map["
                        + kt
                        + "]bool{}"
                        + "; for k := range "
                        + dict_expr
                        + " { s[k] = true }; return s }()"
                    )
        # Empty List() with expected_type
        if isinstance(func, TVar) and func.name == "List" and not args:
            exp = expr.annotations.get("expected_type", "")
            if exp.startswith("list["):
                go_t = self._ann_type_to_go(exp)
                if go_t != "any":
                    return go_t + "{}"
            return "[]any{}"
        # Builtin call
        if isinstance(func, TVar) and func.name in BUILTIN_NAMES:
            return self._builtin_call(func.name, args, expr.annotations)
        # Struct constructor
        if isinstance(func, TVar) and func.name in self.struct_names:
            return self._struct_call(func.name, args)
        # Method call
        if isinstance(func, TFieldAccess):
            return self._method_call(func, args)
        # Regular call
        fn_name = self._expr(func)
        args = self._fill_default_args(func, args)
        params = self._lookup_params(func)
        arg_strs = self._join_args(args, ", ", params)
        return fn_name + "(" + arg_strs + ")"

    def _star_unpack(self, expr: TCall) -> str:
        parts: list[TExpr] = []
        self._flatten_star_unpack(expr, parts)
        # In Go: append(a, b...)
        if len(parts) == 2 and isinstance(parts[0], TListLit):
            lits = ", ".join(self._expr(e) for e in parts[0].elements)
            et = (
                self._infer_go_type(parts[0].elements[0])
                if parts[0].elements
                else "any"
            )
            return "append([]" + et + "{" + lits + "}, " + self._expr(parts[1]) + "...)"
        if len(parts) >= 2:
            result = self._expr(parts[0])
            for p in parts[1:]:
                if isinstance(p, TListLit):
                    for elem in p.elements:
                        result = "append(" + result + ", " + self._expr(elem) + ")"
                else:
                    result = "append(" + result + ", " + self._expr(p) + "...)"
            return result
        return self._expr(parts[0]) if parts else "nil"

    def _flatten_star_unpack(self, expr: TExpr, parts: list[TExpr]) -> None:
        if (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Concat"
            and expr.annotations.get("provenance") == "star_unpack"
        ):
            self._flatten_star_unpack(expr.args[0].value, parts)
            parts.append(expr.args[1].value)
        else:
            parts.append(expr)

    def _struct_call(self, name: str, args: list[TArg]) -> str:
        if name == "Exception":
            if args:
                return "errors.New(" + self._expr(args[0].value) + ")"
            return 'errors.New("")'
        prefix = "&" if name not in self._interface_names else ""
        has_named = any(a.name is not None for a in args)
        if has_named:
            ordered = self.struct_fields.get(name, [])
            if ordered:
                named: dict[str, str] = {}
                for a in args:
                    if a.name is not None:
                        k = a.name[0].upper() + a.name[1:]
                        named[k] = self._struct_field_expr(name, a.name, a.value)
                pairs: list[str] = []
                for f in ordered:
                    uf = f[0].upper() + f[1:]
                    if uf in named:
                        pairs.append(uf + ": " + named[uf])
                return prefix + name + "{" + ", ".join(pairs) + "}"
        if not args:
            return prefix + name + "{}"
        # Positional args: map to field names
        ordered = self.struct_fields.get(name, [])
        if ordered:
            pairs: list[str] = []
            for i, a in enumerate(args):
                if i < len(ordered):
                    fname = ordered[i]
                    uf = fname[0].upper() + fname[1:]
                    pairs.append(
                        uf + ": " + self._struct_field_expr(name, fname, a.value)
                    )
                else:
                    pairs.append(self._expr(a.value))
            return prefix + name + "{" + ", ".join(pairs) + "}"
        parts: list[str] = []
        for a in args:
            parts.append(self._expr(a.value))
        return prefix + name + "{" + ", ".join(parts) + "}"

    def _struct_field_expr(
        self, struct_name: str, field_name: str, value: TExpr
    ) -> str:
        """Emit an expression for a struct field, using field type for list/map inference."""
        field_decls = self.struct_field_types.get(struct_name, [])
        if isinstance(value, TListLit):
            for fd in field_decls:
                if fd.name == field_name:
                    if isinstance(fd.typ, TListType):
                        elem_go = self._type(fd.typ.element)
                        elems = self._join_exprs(value.elements, ", ")
                        return "[]" + elem_go + "{" + elems + "}"
                    break
        is_empty_map_call = (
            isinstance(value, TCall)
            and isinstance(value.func, TVar)
            and value.func.name == "Map"
            and not value.args
        )
        if isinstance(value, TMapLit) or is_empty_map_call:
            for fd in field_decls:
                if fd.name == field_name:
                    if isinstance(fd.typ, TMapType):
                        kt = self._type(fd.typ.key)
                        vt = self._type(fd.typ.value)
                        if is_empty_map_call or (
                            isinstance(value, TMapLit) and not value.entries
                        ):
                            return "map[" + kt + "]" + vt + "{}"
                        if isinstance(value, TMapLit) and value.entries:
                            pairs: list[str] = []
                            for k, v in value.entries:
                                pairs.append(self._expr(k) + ": " + self._expr(v))
                            return "map[" + kt + "]" + vt + "{" + ", ".join(pairs) + "}"
                    break
        for fd in field_decls:
            if fd.name == field_name:
                if isinstance(fd.typ, TOptionalType) and isinstance(
                    fd.typ.inner, TPrimitive
                ):
                    val_s = self._expr_preserve_ptr(value)
                    val_ann = value.annotations.get("type", "")
                    val_is_opt = val_ann.endswith("?") or isinstance(value, TNilLit)
                    if not val_is_opt and isinstance(value, TVar):
                        val_vt = self.var_types.get(value.name)
                        if isinstance(val_vt, TOptionalType):
                            val_is_opt = True
                    if not val_is_opt and isinstance(value, TFieldAccess):
                        ft = self._resolve_field_type(value)
                        if isinstance(ft, TOptionalType):
                            val_is_opt = True
                    if not val_is_opt:
                        inner_go = self._type(fd.typ.inner)
                        return (
                            "func() *"
                            + inner_go
                            + " { v := "
                            + val_s
                            + "; return &v }()"
                        )
                    return val_s
                break
        return self._expr(value)

    def _method_call(self, func: TFieldAccess, args: list[TArg]) -> str:
        obj_str = self._expr(func.obj)
        if isinstance(func.obj, (TBinaryOp, TUnaryOp, TTernary)):
            obj_str = "(" + obj_str + ")"
        field = func.field
        if field == "hex" and not args:
            return 'fmt.Sprintf("%x", ' + obj_str + ")"
        args = self._fill_default_args(func, args)
        params = self._lookup_params(func)
        arg_strs = self._join_args(args, ", ", params)
        return obj_str + "." + field + "(" + arg_strs + ")"

    def _builtin_call(self, name: str, args: list[TArg], ann: Ann | None = None) -> str:
        if ann is None:
            ann = {}
        if name == "FloorDiv":
            return (
                "int(math.Floor(float64("
                + self._a(args, 0)
                + ") / float64("
                + self._a(args, 1)
                + ")))"
            )
        if name == "PythonMod":
            a = self._a(args, 0)
            b = self._a(args, 1)
            return "((" + a + " % " + b + ") + " + b + ") % " + b
        if name == "Append":
            obj = self._a(args, 0)
            val = self._a(args, 1)
            return "append(" + obj + ", " + val + ")"
        if name == "Insert":
            return (
                "slices.Insert("
                + self._a(args, 0)
                + ", "
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ")"
            )
        if name == "Pop":
            obj = self._a(args, 0)
            return obj + "[len(" + obj + ")-1]"
        if name == "RemoveAt":
            return (
                "slices.Delete("
                + self._a(args, 0)
                + ", "
                + self._a(args, 1)
                + ", "
                + self._a(args, 1)
                + "+1)"
            )
        if name == "IndexOf":
            return "slices.Index(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Upper":
            return "strings.ToUpper(" + self._a(args, 0) + ")"
        if name == "Lower":
            return "strings.ToLower(" + self._a(args, 0) + ")"
        if name == "Trim":
            if len(args) == 1:
                return "strings.TrimSpace(" + self._a(args, 0) + ")"
            return "strings.Trim(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "TrimStart":
            if len(args) == 1:
                return "strings.TrimLeft(" + self._a(args, 0) + ', " \\t\\n\\r")'
            return (
                "strings.TrimLeft(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            )
        if name == "TrimEnd":
            if len(args) == 1:
                return "strings.TrimRight(" + self._a(args, 0) + ', " \\t\\n\\r")'
            return (
                "strings.TrimRight(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            )
        if name == "Split":
            return "strings.Split(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "SplitN":
            return (
                "strings.SplitN("
                + self._a(args, 0)
                + ", "
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ")"
            )
        if name == "SplitWhitespace":
            return "strings.Fields(" + self._a(args, 0) + ")"
        if name == "Join":
            return "strings.Join(" + self._a(args, 1) + ", " + self._a(args, 0) + ")"
        if name == "Find":
            return "strings.Index(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "RFind":
            return (
                "strings.LastIndex(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            )
        if name == "Count":
            return "strings.Count(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Replace":
            return (
                "strings.ReplaceAll("
                + self._a(args, 0)
                + ", "
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ")"
            )
        if name == "ReplaceCount":
            return (
                "strings.Replace("
                + self._a(args, 0)
                + ", "
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ", 1)"
            )
        if name == "StartsWith":
            return (
                "strings.HasPrefix(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            )
        if name == "EndsWith":
            return (
                "strings.HasSuffix(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            )
        if name == "Repeat":
            if self._is_string_expr(args[0].value):
                return (
                    "strings.Repeat(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            first = args[0].value
            if isinstance(first, TListLit) and first.elements:
                et = self._infer_go_type(first.elements[0])
                return "make([]" + et + ", " + self._a(args, 1) + ")"
            return "make([]any, " + self._a(args, 1) + ")"
        if name == "Contains":
            first_arg = args[0].value
            if self._is_set_type(first_arg):
                return self._a(args, 0) + "[" + self._a(args, 1) + "]"
            if self._is_list_expr(first_arg):
                return (
                    "slices.Contains("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ")"
                )
            if self._is_map_type(first_arg):
                return (
                    "func() bool { _, ok := "
                    + self._a(args, 0)
                    + "["
                    + self._a(args, 1)
                    + "]; return ok }()"
                )
            return (
                "strings.Contains(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            )
        if name == "IsDigit":
            return "unicode.IsDigit(rune(" + self._a(args, 0) + "[0]))"
        if name == "IsAlpha":
            return "unicode.IsLetter(rune(" + self._a(args, 0) + "[0]))"
        if name == "IsAlnum":
            a = self._a(args, 0)
            return (
                "(unicode.IsLetter(rune("
                + a
                + "[0])) || unicode.IsDigit(rune("
                + a
                + "[0])))"
            )
        if name == "IsSpace":
            return "unicode.IsSpace(rune(" + self._a(args, 0) + "[0]))"
        if name == "IsUpper":
            return "unicode.IsUpper(rune(" + self._a(args, 0) + "[0]))"
        if name == "IsLower":
            return "unicode.IsLower(rune(" + self._a(args, 0) + "[0]))"
        if name == "Encode":
            return "[]byte(" + self._a(args, 0) + ")"
        if name == "Decode":
            return "string(" + self._a(args, 0) + ")"
        if name == "Add":
            return self._a(args, 0) + "[" + self._a(args, 1) + "] = true"
        if name == "Remove":
            return "delete(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Get":
            map_arg = self._a(args, 0)
            key = self._a(args, 1)
            if len(args) == 3:
                default = self._a(args, 2)
                return (
                    "func() "
                    + self._infer_map_val_type(args[0].value)
                    + " { v, ok := "
                    + map_arg
                    + "["
                    + key
                    + "]; if !ok { return "
                    + default
                    + " }; return v }()"
                )
            return map_arg + "[" + key + "]"
        if name == "Delete":
            return "delete(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Keys":
            a = self._a(args, 0)
            kt = self._infer_map_key_type(args[0].value)
            return (
                "func() []"
                + kt
                + " { var r []"
                + kt
                + "; for k := range "
                + a
                + " { r = append(r, k) }; return r }()"
            )
        if name == "Values":
            a = self._a(args, 0)
            vt = self._infer_map_val_type(args[0].value)
            return (
                "func() []"
                + vt
                + " { var r []"
                + vt
                + "; for _, v := range "
                + a
                + " { r = append(r, v) }; return r }()"
            )
        if name == "Items":
            a = self._a(args, 0)
            return (
                "func() [][2]any { var r [][2]any"
                + "; for k, v := range "
                + a
                + " { r = append(r, [2]any{k, v}) }; return r }()"
            )
        if name == "Merge":
            a = self._a(args, 0)
            b = self._a(args, 1)
            gt = self._infer_go_type(args[0].value)
            if self._is_empty_map_or_set_call(args[1].value) and gt != "any":
                b = gt + "{}"
            return (
                "func() "
                + gt
                + " { r := "
                + gt
                + "{}; for k, v := range "
                + a
                + " { r[k] = v }; for k, v := range "
                + b
                + " { r[k] = v }; return r }()"
            )
        if name == "Union":
            a = self._a(args, 0)
            b = self._a(args, 1)
            gt = self._infer_go_type(args[0].value)
            if self._is_empty_map_or_set_call(args[1].value) and gt != "any":
                b = gt + "{}"
            return (
                "func() "
                + gt
                + " { r := "
                + gt
                + "{}; for k := range "
                + a
                + " { r[k] = true }; for k := range "
                + b
                + " { r[k] = true }; return r }()"
            )
        if name == "Intersection":
            a = self._a(args, 0)
            b = self._a(args, 1)
            gt = self._infer_go_type(args[0].value)
            return (
                "func() "
                + gt
                + " { r := "
                + gt
                + "{}; for k := range "
                + a
                + " { if "
                + b
                + "[k] { r[k] = true } }; return r }()"
            )
        if name == "Difference":
            a = self._a(args, 0)
            b = self._a(args, 1)
            gt = self._infer_go_type(args[0].value)
            return (
                "func() "
                + gt
                + " { r := "
                + gt
                + "{}; for k := range "
                + a
                + " { if !"
                + b
                + "[k] { r[k] = true } }; return r }()"
            )
        if name == "Sorted":
            if self.strict_math and self._is_float_list(args[0].value):
                self._used_strict_helpers.add("strict_sorted_f64")
                return "strictSortedF64(" + self._a(args, 0) + ")"
            arg = args[0].value
            is_set = self._is_set_type(arg)
            if (
                not is_set
                and isinstance(arg, TCall)
                and isinstance(arg.func, TVar)
                and arg.func.name == "ListFrom"
                and arg.args
            ):
                is_set = self._is_set_type(arg.args[0].value)
                if is_set:
                    arg = arg.args[0].value
            a = self._expr(arg)
            et = self._infer_elem_type(arg)
            if is_set:
                clone_s = (
                    "make([]"
                    + et
                    + ", 0, len("
                    + a
                    + ")); for k := range "
                    + a
                    + " { s = append(s, k) }"
                )
            else:
                clone_s = "slices.Clone(" + a + ")"
            if is_set:
                ret_type = "[]" + et
            else:
                ret_type = self._infer_go_type(arg)
            if len(args) >= 2:
                key_fn = self._expr(args[1].value)
                return (
                    "func() "
                    + ret_type
                    + " { s := "
                    + clone_s
                    + "; slices.SortFunc(s, func(a, b "
                    + et
                    + ") int { return "
                    + key_fn
                    + "(a) - "
                    + key_fn
                    + "(b) }); return s }()"
                )
            return (
                "func() "
                + ret_type
                + " { s := "
                + clone_s
                + "; slices.Sort(s); return s }()"
            )
        if name == "SortBy":
            return (
                "slices.SortFunc("
                + self._a(args, 0)
                + ", func(a, b "
                + self._infer_elem_type(args[0].value)
                + ") int { return "
                + self._expr(args[1].value)
                + "(a) - "
                + self._expr(args[1].value)
                + "(b) })"
            )
        if name == "Reversed":
            a = self._a(args, 0)
            if self._is_string_expr(args[0].value):
                self._need_reverse_string = True
                return "reverseString(" + a + ")"
            return (
                "func() "
                + self._infer_go_type(args[0].value)
                + " { s := slices.Clone("
                + a
                + "); slices.Reverse(s); return s }()"
            )
        if name == "Reverse":
            a = self._a(args, 0)
            if self._is_string_expr(args[0].value):
                self._need_reverse_string = True
                return "reverseString(" + a + ")"
            return (
                "func() "
                + self._infer_go_type(args[0].value)
                + " { s := slices.Clone("
                + a
                + "); slices.Reverse(s); return s }()"
            )
        if name == "Sum":
            a = self._a(args, 0)
            return (
                "func() int { s := 0; for _, v := range "
                + a
                + " { s += v }; return s }()"
            )
        if name == "MinBy":
            if len(args) == 2 and isinstance(args[1].value, TFnLit):
                key_fn = self._expr(args[1].value)
                return (
                    "slices.MinFunc("
                    + self._a(args, 0)
                    + ", func(a, b "
                    + self._infer_elem_type(args[0].value)
                    + ") int { return "
                    + key_fn
                    + "(a) - "
                    + key_fn
                    + "(b) })"
                )
            if self.strict_math and self._is_float_expr(args[0].value):
                self._used_strict_helpers.add("strict_min_f64")
                return (
                    "strictMinF64(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            return "min(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "MaxBy":
            if len(args) == 2 and isinstance(args[1].value, TFnLit):
                key_fn = self._expr(args[1].value)
                return (
                    "slices.MaxFunc("
                    + self._a(args, 0)
                    + ", func(a, b "
                    + self._infer_elem_type(args[0].value)
                    + ") int { return "
                    + key_fn
                    + "(a) - "
                    + key_fn
                    + "(b) })"
                )
            if self.strict_math and self._is_float_expr(args[0].value):
                self._used_strict_helpers.add("strict_max_f64")
                return (
                    "strictMaxF64(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            return "max(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Len":
            return "len(" + self._a(args, 0) + ")"
        if name == "Abs":
            if self._is_int_expr(args[0].value):
                a = self._a(args, 0)
                return (
                    "func() int { if "
                    + a
                    + " < 0 { return -"
                    + a
                    + " }; return "
                    + a
                    + " }()"
                )
            return "math.Abs(" + self._a(args, 0) + ")"
        if name == "Min":
            if len(args) == 2 and isinstance(args[1].value, TFnLit):
                key_fn = self._expr(args[1].value)
                return (
                    "slices.MinFunc("
                    + self._a(args, 0)
                    + ", func(a, b "
                    + self._infer_elem_type(args[0].value)
                    + ") int { return "
                    + key_fn
                    + "(a) - "
                    + key_fn
                    + "(b) })"
                )
            if self.strict_math and self._is_float_expr(args[0].value):
                self._used_strict_helpers.add("strict_min_f64")
                return (
                    "strictMinF64(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            return "min(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Max":
            if len(args) == 2 and isinstance(args[1].value, TFnLit):
                key_fn = self._expr(args[1].value)
                return (
                    "slices.MaxFunc("
                    + self._a(args, 0)
                    + ", func(a, b "
                    + self._infer_elem_type(args[0].value)
                    + ") int { return "
                    + key_fn
                    + "(a) - "
                    + key_fn
                    + "(b) })"
                )
            if self.strict_math and self._is_float_expr(args[0].value):
                self._used_strict_helpers.add("strict_max_f64")
                return (
                    "strictMaxF64(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            return "max(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Sqrt":
            return "math.Sqrt(" + self._a(args, 0) + ")"
        if name == "Floor":
            return "int(math.Floor(" + self._a(args, 0) + "))"
        if name == "Ceil":
            return "int(math.Ceil(" + self._a(args, 0) + "))"
        if name == "IsNaN":
            return "math.IsNaN(" + self._a(args, 0) + ")"
        if name == "IsInf":
            return "math.IsInf(" + self._a(args, 0) + ", 0)"
        if name == "Pow":
            if self.strict_math and self._is_int_expr(args[0].value):
                self._used_strict_helpers.add("checked_pow_i64")
                return (
                    "checkedPowI64(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            return "math.Pow(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "ToString":
            return self._tostring_call(args)
        if name == "WritelnOut":
            return "fmt.Println(" + self._a(args, 0) + ")"
        if name == "WriteOut":
            return "fmt.Print(" + self._a(args, 0) + ")"
        if name == "WritelnErr":
            return "fmt.Fprintln(os.Stderr, " + self._a(args, 0) + ")"
        if name == "WriteErr":
            return "fmt.Fprint(os.Stderr, " + self._a(args, 0) + ")"
        if name == "ReadAll":
            return 'func() string { b, _ := io.ReadAll(os.Stdin); return strings.TrimRight(string(b), "\\n") }()'
        if name == "ReadFile":
            return (
                "func() string { b, _ := os.ReadFile("
                + self._a(args, 0)
                + "); return string(b) }()"
            )
        if name == "ReadFileBytes":
            return (
                "func() []byte { b, _ := os.ReadFile("
                + self._a(args, 0)
                + "); return b }()"
            )
        if name == "WriteFile":
            return (
                "os.WriteFile("
                + self._a(args, 0)
                + ", []byte("
                + self._a(args, 1)
                + "), 0644)"
            )
        if name == "Args":
            return "os.Args[1:]"
        if name == "Exit":
            return "os.Exit(" + self._a(args, 0) + ")"
        if name == "GetEnv":
            return (
                "func() *string { v, ok := os.LookupEnv("
                + self._a(args, 0)
                + "); if ok { return &v }; return nil }()"
            )
        if name == "ReadLine":
            return "func() string { r := bufio.NewReader(os.Stdin); s, _ := r.ReadString('\\n'); return strings.TrimRight(s, \"\\n\") }()"
        if name == "Format":
            return self._format_call(args)
        if name == "ParseInt":
            return self._parseint_call(args)
        if name == "ParseFloat":
            return self._parsefloat_call(args)
        if name == "FormatInt":
            return self._formatint_call(args)
        if name == "RuneToInt":
            return "int(" + self._a(args, 0) + ")"
        if name == "RuneFromInt":
            return "rune(" + self._a(args, 0) + ")"
        if name == "Unwrap":
            arg = args[0].value
            if isinstance(arg, TVar):
                unwrap_vt = self.var_types.get(arg.name)
                if self._is_go_ref_type(unwrap_vt):
                    return _restore_name(arg.name, arg.annotations)
                raw = _restore_name(arg.name, arg.annotations)
                return "*" + raw
            return "*" + self._a(args, 0)
        if name == "IsInstance":
            return self._isinstance_call(args)
        if name == "IsNil":
            arg = args[0].value
            if isinstance(arg, TVar):
                raw = _restore_name(arg.name, arg.annotations)
                return raw + " == nil"
            if isinstance(arg, TFieldAccess):
                return self._field_access_raw(arg) + " == nil"
            return self._a(args, 0) + " == nil"
        if name == "DivMod":
            a = self._a(args, 0)
            b = self._a(args, 1)
            return "[2]int{" + a + " / " + b + ", " + a + " % " + b + "}"
        if name == "RangeList":
            return self._range_list_call(args)
        if name == "ListFrom":
            return self._listfrom_call(args)
        if name == "SetFromList":
            return self._setfromlist_call(args)
        if name == "Bytes":
            return "[]byte(" + self._a(args, 0) + ")"
        if name == "BytesFrom":
            return "[]byte{" + self._a(args, 0) + "}"
        if name == "ReadBytes":
            return "[]byte{}"
        if name == "ReadBytesN":
            return "make([]byte, " + self._a(args, 0) + ")"
        if name == "Zip":
            return self._zip_call(args)
        if name == "Enumerate":
            return self._a(args, 0)
        if name == "WrappingAdd":
            self._used_strict_helpers.add("wrapping_add")
            return "wrappingAdd(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "WrappingSub":
            self._used_strict_helpers.add("wrapping_sub")
            return "wrappingSub(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "WrappingMul":
            self._used_strict_helpers.add("wrapping_mul")
            return "wrappingMul(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Round":
            return "int(math.Round(" + self._a(args, 0) + "))"
        if name == "IntToFloat":
            return "float64(" + self._a(args, 0) + ")"
        if name == "FloatToInt":
            return "int(" + self._a(args, 0) + ")"
        if name == "IntToByte":
            return "byte(" + self._a(args, 0) + ")"
        if name == "ByteToInt":
            return "int(" + self._a(args, 0) + ")"
        if name == "Concat":
            if self._is_bytes_expr(args[0].value):
                return "append(" + self._a(args, 0) + ", " + self._a(args, 1) + "...)"
            if self._is_list_expr(args[0].value):
                return "append(" + self._a(args, 0) + ", " + self._a(args, 1) + "...)"
            return self._a(args, 0) + " + " + self._a(args, 1)
        if name == "ConcatBytes":
            return "append(" + self._a(args, 0) + ", " + self._a(args, 1) + "...)"
        if name == "NaN":
            return "math.NaN()"
        if name == "Inf":
            return "math.Inf(1)"
        if name == "Map":
            return self._empty_map_call(args, ann)
        if name == "Set":
            return self._empty_set_call(args, ann)
        if name == "IsType":
            return self._isinstance_call(args)
        if name == "CutPrefix":
            return (
                "strings.CutPrefix(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            )
        if name == "CutSuffix":
            return (
                "strings.CutSuffix(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            )
        # Fallback
        arg_strs = self._join_args(args, ", ")
        return name + "(" + arg_strs + ")"

    def _isinstance_call(self, args: list[TArg]) -> str:
        obj = self._expr_no_narrowing(args[0].value)
        type_arg = args[1].value
        type_name = ""
        if isinstance(type_arg, TVar):
            type_name = type_arg.name
        elif isinstance(type_arg, TStringLit):
            type_name = type_arg.value
        if type_name:
            if type_name not in self._interface_names:
                # Check if the object is already a concrete (non-interface) type
                obj_expr = args[0].value
                obj_is_concrete = False
                if isinstance(obj_expr, TVar):
                    vt = self.var_types.get(obj_expr.name)
                    if vt is not None:
                        go_t = self._type(vt).lstrip("*")
                        if go_t not in self._interface_names and go_t != "any":
                            obj_is_concrete = True
                if obj_is_concrete:
                    return "true"
                return (
                    "func() bool { _, ok := "
                    + obj
                    + ".(*"
                    + type_name
                    + "); return ok }()"
                )
            return (
                "func() bool { _, ok := " + obj + ".(" + type_name + "); return ok }()"
            )
        return obj + " != nil"

    def _tostring_call(self, args: list[TArg]) -> str:
        arg = args[0].value
        if self._is_int_expr(arg):
            return "strconv.Itoa(" + self._a(args, 0) + ")"
        if self._is_float_expr(arg):
            return "strconv.FormatFloat(" + self._a(args, 0) + ", 'f', -1, 64)"
        ann = arg.annotations.get("type", "")
        if ann == "bool":
            return "strconv.FormatBool(" + self._a(args, 0) + ")"
        if isinstance(arg, TBoolLit):
            return "strconv.FormatBool(" + self._a(args, 0) + ")"
        if isinstance(arg, TVar):
            typ = self.var_types.get(arg.name)
            if isinstance(typ, TPrimitive):
                if typ.kind == "bool":
                    return "strconv.FormatBool(" + self._a(args, 0) + ")"
                if typ.kind == "rune":
                    return "string(" + self._a(args, 0) + ")"
        return "fmt.Sprint(" + self._a(args, 0) + ")"

    def _format_call(self, args: list[TArg]) -> str:
        template_expr = args[0].value
        if not isinstance(template_expr, TStringLit):
            arg_strs = self._join_args(args, ", ")
            return "fmt.Sprintf(" + arg_strs + ")"
        template = template_expr.value
        fmt_args = args[1:]
        # Replace {} with format verbs
        result_parts: list[str] = []
        go_args: list[str] = []
        arg_idx = 0
        i = 0
        while i < len(template):
            if template[i] == "{" and i + 1 < len(template) and template[i + 1] == "}":
                if arg_idx < len(fmt_args):
                    fa = fmt_args[arg_idx]
                    verb = self._format_verb(fa.value)
                    result_parts.append(verb)
                    go_args.append(self._expr(fa.value))
                    arg_idx += 1
                i += 2
            else:
                c = template[i]
                if c == "%":
                    result_parts.append("%%")
                else:
                    result_parts.append(c)
                i += 1
        fmt_str = '"' + escape_string("".join(result_parts)) + '"'
        if go_args:
            return "fmt.Sprintf(" + fmt_str + ", " + ", ".join(go_args) + ")"
        return fmt_str

    def _format_verb(self, expr: TExpr) -> str:
        if self._is_int_expr(expr):
            return "%d"
        if self._is_float_expr(expr):
            return "%f"
        if self._is_string_expr(expr):
            return "%s"
        ann = expr.annotations.get("type", "")
        if ann == "string":
            return "%s"
        return "%s"

    def _parseint_call(self, args: list[TArg]) -> str:
        s = self._a(args, 0)
        base = self._a(args, 1)
        # If used in a let := context, the caller will handle multi-return
        return (
            "func() int { v, _ := strconv.ParseInt("
            + s
            + ", "
            + base
            + ", 64); return int(v) }()"
        )

    def _parseint_direct(self, args: list[TArg]) -> str:
        """Return the raw strconv.ParseInt call for use with multi-return := pattern."""
        s = self._a(args, 0)
        base = self._a(args, 1)
        return "strconv.ParseInt(" + s + ", " + base + ", 64)"

    def _parsefloat_call(self, args: list[TArg]) -> str:
        s = self._a(args, 0)
        return (
            "func() float64 { v, _ := strconv.ParseFloat(" + s + ", 64); return v }()"
        )

    def _formatint_call(self, args: list[TArg]) -> str:
        n = self._a(args, 0)
        base = self._a(args, 1)
        return "strconv.FormatInt(" + n + ", " + base + ")"

    def _range_list_call(self, args: list[TArg]) -> str:
        start = self._a(args, 0) if len(args) >= 2 else "0"
        end = self._a(args, 1) if len(args) >= 2 else self._a(args, 0)
        step_val = self._static_int(args[2].value) if len(args) >= 3 else 1
        if step_val == 1:
            incr = "; i++ {"
        else:
            step_s = self._a(args, 2) if len(args) >= 3 else "1"
            incr = "; i += " + step_s + " {"
        return (
            "func() []int { var r []int; for i := "
            + start
            + "; i < "
            + end
            + incr
            + " r = append(r, i) }; return r }()"
        )

    def _listfrom_call(self, args: list[TArg]) -> str:
        a = self._a(args, 0)
        arg = args[0].value
        if self._is_bytes_expr(arg):
            return (
                "func() []int { var r []int; for _, b := range "
                + a
                + " { r = append(r, int(b)) }; return r }()"
            )
        if self._is_map_type(arg):
            kt = self._infer_map_key_type(arg)
            return (
                "func() []"
                + kt
                + " { var r []"
                + kt
                + "; for k := range "
                + a
                + " { r = append(r, k) }; return r }()"
            )
        return a

    def _setfromlist_call(self, args: list[TArg]) -> str:
        a = self._a(args, 0)
        arg = args[0].value
        if self._is_map_type(arg) or self._is_set_type(arg):
            kt = (
                self._infer_map_key_type(arg)
                if self._is_map_type(arg)
                else self._infer_elem_type(arg)
            )
            return (
                "func() map["
                + kt
                + "]bool { r := map["
                + kt
                + "]bool{}; for k := range "
                + a
                + " { r[k] = true }; return r }()"
            )
        et = self._infer_elem_type(arg)
        return (
            "func() map["
            + et
            + "]bool { r := map["
            + et
            + "]bool{}; for _, v := range "
            + a
            + " { r[v] = true }; return r }()"
        )

    def _zip_call(self, args: list[TArg]) -> str:
        arr_exprs = [self._a(args, i) for i in range(len(args))]
        return "__zip(" + ", ".join(arr_exprs) + ")"

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

    def _lookup_params(self, func: TExpr) -> list[TParam] | None:
        key = self._fn_key(func)
        if key == "":
            return None
        return self._fn_params.get(key)

    def _fn_key(self, func: TExpr) -> str:
        if isinstance(func, TFieldAccess):
            if isinstance(func.obj, TVar):
                obj_type = self.var_types.get(func.obj.name)
                if isinstance(obj_type, TIdentType):
                    return obj_type.name + "." + func.field
            ann_type = func.obj.annotations.get("type", "")
            if ann_type != "":
                return ann_type + "." + func.field
        elif isinstance(func, TVar):
            return func.name
        return ""

    def _fill_default_args(self, func: TExpr, args: list[TArg]) -> list[TArg]:
        """Fill in zero-value args for params with defaults when caller omits them."""
        key = self._fn_key(func)
        if key == "":
            return args
        params = self._fn_params.get(key)
        if params is None:
            return args
        non_self_params = [p for p in params if p.typ is not None]
        if len(args) >= len(non_self_params):
            return args
        filled = list(args)
        for p in non_self_params[len(args) :]:
            if p.has_default and p.typ is not None:
                filled.append(
                    TArg(
                        pos=p.pos, name=None, value=self._zero_value_expr(p.typ, p.pos)
                    )
                )
            else:
                break
        return filled

    def _zero_value_expr(self, typ: TType, pos: Pos) -> TExpr:
        """Return a zero-value expression for the given type."""
        if isinstance(typ, TPrimitive):
            if typ.kind == "string":
                return TStringLit(pos=pos, value="", annotations={})
            if typ.kind == "int":
                return TIntLit(pos=pos, value=0, raw="0", annotations={})
            if typ.kind == "float":
                return TFloatLit(pos=pos, value=0.0, raw="0.0", annotations={})
            if typ.kind == "bool":
                return TBoolLit(pos=pos, value=False, annotations={})
        return TNilLit(pos=pos, annotations={})

    def _join_args(
        self, args: list[TArg], sep: str, params: list[TParam] | None = None
    ) -> str:
        parts: list[str] = []
        non_self: list[TParam] | None = None
        if params is not None:
            non_self = [p for p in params if p.typ is not None]
        for idx, a in enumerate(args):
            ptype = (
                non_self[idx].typ
                if non_self is not None and idx < len(non_self)
                else None
            )
            if (
                isinstance(a.value, TListLit)
                and not a.value.elements
                and ptype is not None
            ):
                parts.append(self._type(ptype) + "{}")
            elif ptype is not None and self._is_opt_prim_type(ptype):
                val_ann = a.value.annotations.get("type", "")
                val_is_opt = val_ann.endswith("?") or isinstance(a.value, TNilLit)
                if not val_is_opt and isinstance(a.value, TVar):
                    vt = self.var_types.get(a.value.name)
                    if isinstance(vt, TOptionalType):
                        val_is_opt = True
                if not val_is_opt and isinstance(a.value, TFieldAccess):
                    ft = self._resolve_field_type(a.value)
                    if isinstance(ft, TOptionalType):
                        val_is_opt = True
                if not val_is_opt:
                    inner_go = self._opt_prim_inner_go(ptype)
                    parts.append(
                        "func() *"
                        + inner_go
                        + " { v := "
                        + self._expr(a.value)
                        + "; return &v }()"
                    )
                else:
                    parts.append(self._expr_preserve_ptr(a.value))
            elif ptype is not None and not isinstance(ptype, TOptionalType):
                val_ann = a.value.annotations.get("type", "")
                if val_ann.endswith("?") and val_ann[:-1] in (
                    "string",
                    "int",
                    "float",
                    "bool",
                ):
                    parts.append("*" + self._expr(a.value))
                elif self._needs_deref_from_expected(a.value):
                    parts.append("*" + self._expr(a.value))
                elif self._needs_ptr_wrap_from_expected(a.value):
                    parts.append(
                        "func() *"
                        + self._ann_type_to_go(self._expected_type_ann(a.value)[:-1])
                        + " { v := "
                        + self._expr(a.value)
                        + "; return &v }()"
                    )
                else:
                    parts.append(self._expr(a.value))
            elif self._needs_deref_from_expected(a.value):
                parts.append("*" + self._expr(a.value))
            elif self._needs_ptr_wrap_from_expected(a.value):
                parts.append(
                    "func() *"
                    + self._ann_type_to_go(self._expected_type_ann(a.value)[:-1])
                    + " { v := "
                    + self._expr(a.value)
                    + "; return &v }()"
                )
            else:
                parts.append(self._expr(a.value))
        return sep.join(parts)

    def _join_exprs(self, exprs: list[TExpr], sep: str) -> str:
        parts: list[str] = []
        for e in exprs:
            parts.append(self._expr(e))
        return sep.join(parts)


# ── Error reference collection ────────────────────────────


def _collect_error_refs(module: TModule) -> tuple[set[str], set[str]]:
    """Returns (builtin_error_refs, all_thrown_or_caught_names)."""
    builtin_refs: set[str] = set()
    all_refs: set[str] = set()
    builtin_names = set(BUILTIN_STRUCTS.keys())
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            _walk_error_refs_stmts(decl.body, builtin_refs, all_refs, builtin_names)
        elif isinstance(decl, TStructDecl):
            for m in decl.methods:
                _walk_error_refs_stmts(m.body, builtin_refs, all_refs, builtin_names)
        elif isinstance(decl, TStmt):
            _walk_error_refs_stmt(decl, builtin_refs, all_refs, builtin_names)
    return builtin_refs, all_refs


def _walk_error_refs_stmt(
    stmt: TStmt,
    builtin_refs: set[str],
    all_refs: set[str],
    builtin_names: set[str],
) -> None:
    if isinstance(stmt, TTryStmt):
        _walk_error_refs_stmts(stmt.body, builtin_refs, all_refs, builtin_names)
        for c in stmt.catches:
            for t in c.types:
                if isinstance(t, TIdentType):
                    all_refs.add(t.name)
                    if t.name in builtin_names:
                        builtin_refs.add(t.name)
            _walk_error_refs_stmts(c.body, builtin_refs, all_refs, builtin_names)
        if stmt.finally_body is not None:
            _walk_error_refs_stmts(
                stmt.finally_body, builtin_refs, all_refs, builtin_names
            )
    elif isinstance(stmt, TIfStmt):
        _walk_error_refs_stmts(stmt.then_body, builtin_refs, all_refs, builtin_names)
        if stmt.else_body is not None:
            _walk_error_refs_stmts(
                stmt.else_body, builtin_refs, all_refs, builtin_names
            )
    elif isinstance(stmt, (TWhileStmt, TForStmt)):
        _walk_error_refs_stmts(stmt.body, builtin_refs, all_refs, builtin_names)
    elif isinstance(stmt, TMatchStmt):
        for case in stmt.cases:
            _walk_error_refs_stmts(case.body, builtin_refs, all_refs, builtin_names)
        if stmt.default is not None:
            _walk_error_refs_stmts(
                stmt.default.body, builtin_refs, all_refs, builtin_names
            )
    if isinstance(stmt, TThrowStmt):
        if isinstance(stmt.expr, TCall):
            if isinstance(stmt.expr.func, TVar):
                name = stmt.expr.func.name
                all_refs.add(name)
                if name in builtin_names:
                    builtin_refs.add(name)
    elif isinstance(stmt, TLetStmt):
        val = stmt.value
        if val is not None:
            if isinstance(val, TCall):
                if isinstance(val.func, TVar):
                    if val.func.name in builtin_names:
                        builtin_refs.add(val.func.name)
    elif isinstance(stmt, TAssignStmt):
        val = stmt.value
        if isinstance(val, TCall):
            if isinstance(val.func, TVar):
                if val.func.name in builtin_names:
                    builtin_refs.add(val.func.name)
    elif isinstance(stmt, TReturnStmt):
        val = stmt.value
        if val is not None:
            if isinstance(val, TCall):
                if isinstance(val.func, TVar):
                    if val.func.name in builtin_names:
                        builtin_refs.add(val.func.name)


def _walk_error_refs_stmts(
    stmts: list[TStmt],
    builtin_refs: set[str],
    all_refs: set[str],
    builtin_names: set[str],
) -> None:
    for stmt in stmts:
        _walk_error_refs_stmt(stmt, builtin_refs, all_refs, builtin_names)


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================


def emit_go(module: TModule) -> str:
    struct_names: set[str] = set(BUILTIN_STRUCTS.keys())
    struct_fields: dict[str, list[str]] = {}
    struct_field_types: dict[str, list[TFieldDecl]] = {}
    for bname, bfields in BUILTIN_STRUCTS.items():
        struct_fields[bname] = list(bfields.keys())
    for decl in module.decls:
        match decl:
            case TStructDecl():
                struct_names.add(decl.name)
                fnames: list[str] = []
                for f in decl.fields:
                    fnames.append(f.name)
                struct_fields[decl.name] = fnames
                struct_field_types[decl.name] = decl.fields
            case TInterfaceDecl():
                struct_names.add(decl.name)
                if decl.fields:
                    ifnames: list[str] = []
                    for f in decl.fields:
                        ifnames.append(f.name)
                    struct_fields[decl.name] = ifnames
                    struct_field_types[decl.name] = decl.fields
    emitter = _GoEmitter(
        struct_names,
        struct_fields,
        struct_field_types,
        module.strict_math,
        module.strict_tostring,
    )
    emitter.emit_module(module)
    return emitter.output()
