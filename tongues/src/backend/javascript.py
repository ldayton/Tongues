"""JavaScript backend: Taytsh AST → JavaScript source code."""

from __future__ import annotations

from .ordering import order_decls
from .util import (
    STRICT_INT_BINARY,
    STRICT_INT_COMPOUND,
    Emitter,
    collect_builtin_calls,
    escape_string,
)
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
    TDecl,
    TDefault,
    TEnumDecl,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFieldDecl,
    TFnDecl,
    TFnLit,
    TFloatLit,
    TForStmt,
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
    TTryStmt,
    TType,
    TUnaryOp,
    TVar,
    TWhileStmt,
)
from ..taytsh.check import (
    BUILTIN_NAMES,
    BUILTIN_STRUCTS,
)

# ============================================================
# JAVASCRIPT RESERVED WORDS
# ============================================================

_JS_RESERVED = frozenset(
    {
        "abstract",
        "arguments",
        "await",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "debugger",
        "default",
        "delete",
        "do",
        "double",
        "else",
        "enum",
        "eval",
        "export",
        "extends",
        "false",
        "final",
        "finally",
        "float",
        "for",
        "function",
        "goto",
        "if",
        "implements",
        "import",
        "in",
        "instanceof",
        "int",
        "interface",
        "let",
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
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "transient",
        "true",
        "try",
        "typeof",
        "var",
        "void",
        "volatile",
        "while",
        "with",
        "yield",
        "undefined",
        "NaN",
        "Infinity",
        "console",
        "process",
        "Buffer",
        "Math",
        "Number",
        "String",
        "Array",
        "Object",
        "Set",
        "Map",
        "Error",
        "TypeError",
        "RangeError",
    }
)


def _safe_name(name: str) -> str:
    if name in _JS_RESERVED:
        return name + "_"
    return name


def _to_lower_camel(name: str) -> str:
    """Convert PascalCase to lowerCamelCase."""
    if not name or name[0].islower():
        return name
    return name[0].lower() + name[1:]


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
    "===": 3,
    "!==": 3,
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
    ">>>": 7,
    "+": 8,
    "-": 8,
    "*": 9,
    "/": 9,
    "%": 9,
    "**": 11,
}

_CMP_OPS = frozenset(["===", "!==", "==", "!=", "<", ">", "<=", ">="])


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

_FS_BUILTINS = frozenset(
    {"ReadAll", "ReadFile", "ReadFileBytes", "WriteFile", "ReadBytes"}
)

_PROCESS_BUILTINS = frozenset(
    {
        "WriteOut",
        "WriteErr",
        "WritelnErr",
        "ReadBytesN",
        "Args",
        "Exit",
    }
)

_BUFFER_BUILTINS = frozenset({"Encode", "Decode", "Bytes", "BytesFrom"})


def _scan_imports(
    module: TModule,
) -> tuple[bool, bool, bool]:
    """Return (needs_fs, needs_process, needs_buffer)."""
    needs_fs = False
    needs_process = False
    needs_buffer = False
    for decl in module.decls:
        if isinstance(decl, (TFnDecl, TStructDecl)):
            r_fs, r_proc, r_buf = _scan_decl_builtins(decl)
            if r_fs:
                needs_fs = True
            if r_proc:
                needs_process = True
            if r_buf:
                needs_buffer = True
    for decl in module.decls:
        if isinstance(decl, TStmt):
            for name in collect_builtin_calls([decl]):
                if name in _FS_BUILTINS:
                    needs_fs = True
                if name in _PROCESS_BUILTINS:
                    needs_process = True
                if name in _BUFFER_BUILTINS:
                    needs_buffer = True
    return needs_fs, needs_process, needs_buffer


def _scan_decl_builtins(decl: TDecl) -> tuple[bool, bool, bool]:
    needs_fs = False
    needs_process = False
    needs_buffer = False
    stmts: list[TStmt] = []
    if isinstance(decl, TFnDecl):
        stmts = decl.body
    elif isinstance(decl, TStructDecl):
        for m in decl.methods:
            r = _scan_decl_builtins(m)
            if r[0]:
                needs_fs = True
            if r[1]:
                needs_process = True
            if r[2]:
                needs_buffer = True
        return needs_fs, needs_process, needs_buffer
    for name in collect_builtin_calls(stmts):
        if name in _FS_BUILTINS:
            needs_fs = True
        if name in _PROCESS_BUILTINS:
            needs_process = True
        if name in _BUFFER_BUILTINS:
            needs_buffer = True
    return needs_fs, needs_process, needs_buffer


# ============================================================
# STRICT MATH HELPERS
# ============================================================

_STRICT_MATH_PREAMBLE = """\
const _I64_MIN = -9223372036854775808n;
const _I64_MAX = 9223372036854775807n;
function _check_i64(v) { if (v < _I64_MIN || v > _I64_MAX) throw new RangeError("integer overflow"); return Number(v); }
function checked_add_i64(a, b) { return _check_i64(BigInt(a) + BigInt(b)); }
function checked_sub_i64(a, b) { return _check_i64(BigInt(a) - BigInt(b)); }
function checked_mul_i64(a, b) { return _check_i64(BigInt(a) * BigInt(b)); }
function checked_div_i64(a, b) { if (b === 0) throw new RangeError("division by zero"); let q = BigInt(a) / BigInt(b); return _check_i64(q); }
function checked_rem_i64(a, b) { if (b === 0) throw new RangeError("division by zero"); return Number(BigInt(a) % BigInt(b)); }
function checked_neg_i64(a) { return _check_i64(-BigInt(a)); }
function checked_shl_i64(a, b) { if (b < 0 || b > 63) throw new RangeError("shift out of range"); return _check_i64(BigInt(a) << BigInt(b)); }
function checked_shr_i64(a, b) { if (b < 0 || b > 63) throw new RangeError("shift out of range"); return Number(BigInt(a) >> BigInt(b)); }
function logical_shr_i64(a, b) { if (b < 0 || b > 63) throw new RangeError("shift out of range"); let v = BigInt(a) & 0xFFFFFFFFFFFFFFFFn; return Number((v >> BigInt(b)) & 0xFFFFFFFFFFFFFFFFn); }
function checked_pow_i64(a, b) { if (b < 0) throw new RangeError("negative exponent"); let r = 1n; let base = BigInt(a); for (let i = 0; i < b; i++) { r *= base; if (r < _I64_MIN || r > _I64_MAX) throw new RangeError("integer overflow"); } return Number(r); }
function wrappingAdd(a, b) { let v = BigInt(a) + BigInt(b); v = ((v + 9223372036854775808n) % 18446744073709551616n + 18446744073709551616n) % 18446744073709551616n - 9223372036854775808n; return Number(v); }
function wrappingSub(a, b) { let v = BigInt(a) - BigInt(b); v = ((v + 9223372036854775808n) % 18446744073709551616n + 18446744073709551616n) % 18446744073709551616n - 9223372036854775808n; return Number(v); }
function wrappingMul(a, b) { let v = BigInt(a) * BigInt(b); v = ((v + 9223372036854775808n) % 18446744073709551616n + 18446744073709551616n) % 18446744073709551616n - 9223372036854775808n; return Number(v); }
function strict_fmod(a, b) { if (b === 0.0) throw new RangeError("float modulo by zero"); return a % b; }
function strict_min_f64(a, b) { if (Number.isNaN(a) || Number.isNaN(b)) return NaN; return Math.min(a, b); }
function strict_max_f64(a, b) { if (Number.isNaN(a) || Number.isNaN(b)) return NaN; return Math.max(a, b); }
function strict_sorted_f64(xs) { for (const x of xs) { if (Number.isNaN(x)) throw new RangeError("NaN in sort"); } return [...xs].sort((a, b) => a - b); }
"""


# ============================================================
# EMITTER
# ============================================================


class _JavaScriptEmitter(Emitter):
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
        self._current_struct: str = ""
        self._needs_read_all: bool = False
        self.fn_names: set[str] = set()

    # ── Module ────────────────────────────────────────────────

    def emit_module(self, module: TModule) -> None:
        for decl in module.decls:
            if isinstance(decl, TLetStmt):
                self.module_let_names.add(decl.name)
            if isinstance(decl, TFnDecl):
                self.fn_names.add(decl.name)
            if isinstance(decl, TStructDecl):
                for m in decl.methods:
                    self.fn_names.add(m.name)
        needs_fs, needs_process, needs_buffer = _scan_imports(module)
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
        if self._needs_read_all:
            needs_fs = True
        imports: list[str] = []
        if needs_fs:
            imports.append('const fs = require("fs");')
        if needs_process:
            imports.append('const process = require("process");')
        if needs_buffer:
            imports.append('const { Buffer } = require("buffer");')
        if imports:
            for line in imports:
                self._line(line)
            self._line()
        if self._needs_read_all:
            self._line('const input = fs.readFileSync("/dev/stdin", "utf-8");')
            self._line()
        if self.strict_math:
            for pline in _STRICT_MATH_PREAMBLE.strip().split("\n"):
                self._line(pline)
            self._line()
        need_blank = False
        for decl in order_decls(module.decls):
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

    # ── Enum ──────────────────────────────────────────────────

    def _emit_enum(self, decl: TEnumDecl) -> None:
        parts: list[str] = []
        for i, variant in enumerate(decl.variants):
            parts.append(variant + ": " + str(i))
        self._line(
            "const " + decl.name + " = Object.freeze({" + ", ".join(parts) + "});"
        )

    # ── Struct ────────────────────────────────────────────────

    def _emit_struct(self, decl: TStructDecl) -> None:
        is_error = decl.name in BUILTIN_STRUCTS
        if not is_error and decl.parent is not None:
            if decl.parent in BUILTIN_STRUCTS:
                is_error = True
        if not is_error and decl.annotations.get("_is_exception") is not None:
            is_error = True
        if is_error:
            self._emit_error_struct(decl)
        else:
            self._emit_data_struct(decl)

    def _emit_error_struct(self, decl: TStructDecl) -> None:
        parent = "Error"
        if decl.parent is not None and decl.parent not in BUILTIN_STRUCTS:
            parent = decl.parent
        self._line("class " + decl.name + " extends " + parent + " {")
        self.indent += 1
        params: list[str] = []
        for fld in decl.fields:
            params.append(_safe_name(fld.name))
        self._line("constructor(" + ", ".join(params) + ") {")
        self.indent += 1
        msg_field: TFieldDecl | None = None
        for fld in decl.fields:
            if fld.name in ("message", "msg"):
                msg_field = fld
                break
        if msg_field is not None:
            self._line("super(" + _safe_name(msg_field.name) + ");")
        else:
            self._line("super();")
        for fld in decl.fields:
            safe = _safe_name(fld.name)
            self._line("this." + safe + " = " + safe + ";")
        self.indent -= 1
        self._line("}")
        for method in decl.methods:
            self._line()
            self._emit_method(method)
        self.indent -= 1
        self._line("}")

    def _emit_data_struct(self, decl: TStructDecl) -> None:
        bases: list[str] = []
        if decl.parent is not None:
            bases.append(decl.parent)
        if bases:
            self._line("class " + decl.name + " extends " + bases[0] + " {")
        else:
            self._line("class " + decl.name + " {")
        self.indent += 1
        params: list[str] = []
        for fld in decl.fields:
            params.append(_safe_name(fld.name))
        if decl.fields or decl.parent is not None:
            self._line("constructor(" + ", ".join(params) + ") {")
            self.indent += 1
            if decl.parent is not None:
                self._line("super();")
            for fld in decl.fields:
                safe = _safe_name(fld.name)
                self._line("this." + safe + " = " + safe + ";")
            self.indent -= 1
            self._line("}")
        for i, method in enumerate(decl.methods):
            if i > 0 or decl.fields or decl.parent is not None:
                self._line()
            self._emit_method(method)
        if not decl.fields and not decl.methods and decl.parent is None:
            pass
        self.indent -= 1
        self._line("}")

    def _emit_interface(self, decl: TInterfaceDecl) -> None:
        self._line("class " + decl.name + " {")
        self.indent += 1
        if decl.fields:
            params: list[str] = []
            for fld in decl.fields:
                params.append(_safe_name(fld.name))
            self._line("constructor(" + ", ".join(params) + ") {")
            self.indent += 1
            for fld in decl.fields:
                safe = _safe_name(fld.name)
                self._line("this." + safe + " = " + safe + ";")
            self.indent -= 1
            self._line("}")
        self.indent -= 1
        self._line("}")

    # ── Function / Method ─────────────────────────────────────

    def _emit_fn(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
        params = self._params(decl.params)
        fname = (
            "main" if decl.name == "Main" else _safe_name(_to_lower_camel(decl.name))
        )
        self._line("function " + fname + "(" + params + ") {")
        self.indent += 1
        self._emit_stmts(decl.body)
        self.indent -= 1
        self._line("}")
        self.var_types = old_var_types

    def _emit_method(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
        params = self._params(decl.params)
        self._line(_safe_name(_to_lower_camel(decl.name)) + "(" + params + ") {")
        self.indent += 1
        old_self = self.self_name
        if decl.params and decl.params[0].typ is None:
            self.self_name = decl.params[0].name
        self._emit_stmts(decl.body)
        self.self_name = old_self
        self.indent -= 1
        self._line("}")
        self.var_types = old_var_types

    def _params(self, params: list[TParam]) -> str:
        parts: list[str] = []
        for p in params:
            if p.typ is None:
                continue
            s = _restore_name(p.name, p.annotations)
            if p.has_default:
                s = s + " = " + self._zero_value(p.typ)
            parts.append(s)
        return ", ".join(parts)

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
                return "Buffer.alloc(0)"
        return "null"

    # ── Statements ────────────────────────────────────────────

    def _emit_stmts(self, stmts: list[TStmt]) -> None:
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            if isinstance(stmt, TLetStmt) and i + 1 < len(stmts):
                next_stmt = stmts[i + 1]
                if isinstance(next_stmt, TForStmt):
                    prov = next_stmt.annotations.get("provenance", "")
                    if prov in (
                        "list_comprehension",
                        "dict_comprehension",
                        "set_comprehension",
                    ):
                        lc = self._try_comprehension(stmt, next_stmt, prov)
                        if lc is not None:
                            self._line(lc)
                            i += 2
                            continue
                    if prov == "step_slice":
                        ss = self._try_step_slice(stmt, next_stmt)
                        if ss is not None:
                            self._line(ss)
                            i += 2
                            continue
                    if prov in ("any_call", "all_call"):
                        result = self._emit_any_all(stmts, i, stmt, next_stmt, prov)
                        if result > 0:
                            i += result
                            continue
            self._emit_stmt(stmt)
            i += 1

    def _try_comprehension(
        self, let_stmt: TLetStmt, for_stmt: TForStmt, prov: str
    ) -> str | None:
        acc = _restore_name(let_stmt.name, let_stmt.annotations)
        binding = for_stmt.binding
        binder_parts: list[str] = []
        for b in binding:
            binder_parts.append(_restore_name(b, for_stmt.annotations))
        binders = ", ".join(binder_parts)
        if isinstance(for_stmt.iterable, TRange):
            return None
        iterable = self._expr(for_stmt.iterable)
        iter_is_map = self._is_map_for(for_stmt)
        is_enumerate = self._is_enumerate_for(for_stmt)
        body = for_stmt.body
        if prov == "list_comprehension":
            if len(body) == 1 and isinstance(body[0], TExprStmt):
                call = body[0].expr
                if isinstance(call, TCall) and self._is_append_to(call, let_stmt.name):
                    val = self._expr(call.args[1].value)
                    if is_enumerate:
                        return (
                            "let "
                            + acc
                            + " = "
                            + iterable
                            + ".map(("
                            + binder_parts[1]
                            + ", "
                            + binder_parts[0]
                            + ") => "
                            + val
                            + ");"
                        )
                    return (
                        "let "
                        + acc
                        + " = "
                        + iterable
                        + ".map("
                        + binders
                        + " => "
                        + val
                        + ");"
                    )
            if len(body) == 1:
                if_stmt = body[0]
                if isinstance(if_stmt, TIfStmt) and len(if_stmt.then_body) == 1:
                    then_first = if_stmt.then_body[0]
                    if isinstance(then_first, TExprStmt) and isinstance(
                        then_first.expr, TCall
                    ):
                        call = then_first.expr
                        if isinstance(call, TCall) and self._is_append_to(
                            call, let_stmt.name
                        ):
                            val = self._expr(call.args[1].value)
                            guard = self._expr(if_stmt.cond)
                            return (
                                "let "
                                + acc
                                + " = "
                                + iterable
                                + ".filter("
                                + binders
                                + " => "
                                + guard
                                + ");"
                            )
        elif prov == "dict_comprehension":
            if len(body) == 1 and isinstance(body[0], TAssignStmt):
                target = body[0].target
                if isinstance(target, TIndex):
                    key = self._expr(target.index)
                    val = self._expr(body[0].value)
                    if is_enumerate:
                        return (
                            "let "
                            + acc
                            + " = new Map("
                            + iterable
                            + ".map(("
                            + binder_parts[1]
                            + ", "
                            + binder_parts[0]
                            + ") => ["
                            + key
                            + ", "
                            + val
                            + "]));"
                        )
                    return (
                        "let "
                        + acc
                        + " = new Map("
                        + iterable
                        + ".map(("
                        + binders
                        + ") => ["
                        + key
                        + ", "
                        + val
                        + "]));"
                    )
        elif prov == "set_comprehension":
            if len(body) == 1 and isinstance(body[0], TExprStmt):
                call = body[0].expr
                if isinstance(call, TCall) and self._is_add_to(call, let_stmt.name):
                    val = self._expr(call.args[1].value)
                    if val == binders:
                        return "let " + acc + " = new Set(" + iterable + ");"
                    return (
                        "let "
                        + acc
                        + " = new Set("
                        + iterable
                        + ".map("
                        + binders
                        + " => "
                        + val
                        + "));"
                    )
        return None

    def _try_step_slice(self, let_stmt: TLetStmt, for_stmt: TForStmt) -> str | None:
        if not isinstance(for_stmt.iterable, TRange):
            return None
        range_args = for_stmt.iterable.args
        if len(range_args) != 3:
            return None
        body = for_stmt.body
        if len(body) != 1:
            return None
        src_obj = self._step_slice_source(body[0], let_stmt.name)
        if src_obj is None:
            return None
        src = self._expr(src_obj)
        acc = _restore_name(let_stmt.name, let_stmt.annotations)
        start_expr = range_args[0]
        step_expr = range_args[2]
        start_val = self._static_int(start_expr)
        step_val = self._static_int(step_expr)
        is_string = self._step_slice_is_string(body[0], let_stmt.name)
        spread = "[..." + src + "]" if is_string else src
        suffix = '.join("")' if is_string else ""
        if start_val is not None and step_val is not None:
            if start_val == 0:
                return (
                    "let "
                    + acc
                    + " = "
                    + spread
                    + ".filter((_, i) => i % "
                    + str(step_val)
                    + " === 0)"
                    + suffix
                    + ";"
                )
            elif start_val < step_val:
                return (
                    "let "
                    + acc
                    + " = "
                    + spread
                    + ".filter((_, i) => i % "
                    + str(step_val)
                    + " === "
                    + str(start_val)
                    + ")"
                    + suffix
                    + ";"
                )
            else:
                return (
                    "let "
                    + acc
                    + " = "
                    + spread
                    + ".filter((_, i) => i >= "
                    + str(start_val)
                    + " && (i - "
                    + str(start_val)
                    + ") % "
                    + str(step_val)
                    + " === 0)"
                    + suffix
                    + ";"
                )
        return None

    def _step_slice_source(self, stmt: TStmt, acc_name: str) -> TExpr | None:
        if isinstance(stmt, TExprStmt):
            call = stmt.expr
            if isinstance(call, TCall) and self._is_append_to(call, acc_name):
                elem = call.args[1].value
                if isinstance(elem, TIndex):
                    return elem.obj
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
                            return inner.obj
        return None

    def _step_slice_is_string(self, stmt: TStmt, acc_name: str) -> bool:
        if isinstance(stmt, TAssignStmt) and isinstance(stmt.target, TVar):
            if stmt.target.name == acc_name and isinstance(stmt.value, TCall):
                if (
                    isinstance(stmt.value.func, TVar)
                    and stmt.value.func.name == "Concat"
                ):
                    return True
        return False

    def _is_len_of(self, expr: TExpr, obj: TExpr) -> bool:
        if not isinstance(expr, TCall) or not isinstance(expr.func, TVar):
            return False
        if expr.func.name != "Len" or len(expr.args) != 1:
            return False
        arg = expr.args[0].value
        return isinstance(arg, TVar) and isinstance(obj, TVar) and arg.name == obj.name

    def _emit_any_all(
        self,
        stmts: list[TStmt],
        i: int,
        let_stmt: TLetStmt,
        for_stmt: TForStmt,
        prov: str,
    ) -> int:
        aa = self._try_any_all(let_stmt, for_stmt, prov)
        if aa:
            lhs, rhs = aa
            folded = self._fold_temp_assign(stmts, i, let_stmt.name, rhs)
            if folded is not None:
                self._line(folded)
                return 3
            self._line(lhs + " = " + rhs + ";")
            return 2
        return 0

    def _try_any_all(
        self, let_stmt: TLetStmt, for_stmt: TForStmt, prov: str
    ) -> tuple[str, str] | None:
        acc = _restore_name(let_stmt.name, let_stmt.annotations)
        binding = for_stmt.binding
        binder_parts: list[str] = []
        for b in binding:
            binder_parts.append(_restore_name(b, for_stmt.annotations))
        binders = ", ".join(binder_parts)
        if isinstance(for_stmt.iterable, TRange):
            return None
        iterable = self._expr(for_stmt.iterable)
        func = "some" if prov == "any_call" else "every"
        body = for_stmt.body
        if len(body) != 1:
            return None
        outer_if = body[0]
        if not isinstance(outer_if, TIfStmt):
            return None
        if (
            len(outer_if.then_body) == 2
            and isinstance(outer_if.then_body[0], TAssignStmt)
            and isinstance(outer_if.then_body[1], TBreakStmt)
        ):
            cond = (
                self._strip_not(outer_if.cond) if prov == "all_call" else outer_if.cond
            )
            cond_s = self._expr(cond)
            return (
                "let " + acc,
                iterable + "." + func + "(" + binders + " => " + cond_s + ")",
            )
        if len(outer_if.then_body) == 1:
            inner_if = outer_if.then_body[0]
            if (
                isinstance(inner_if, TIfStmt)
                and len(inner_if.then_body) == 2
                and isinstance(inner_if.then_body[0], TAssignStmt)
                and isinstance(inner_if.then_body[1], TBreakStmt)
            ):
                filter_s = self._expr(outer_if.cond)
                cond = (
                    self._strip_not(inner_if.cond)
                    if prov == "all_call"
                    else inner_if.cond
                )
                cond_s = self._expr(cond)
                return (
                    "let " + acc,
                    iterable
                    + "."
                    + func
                    + "("
                    + binders
                    + " => "
                    + filter_s
                    + " && "
                    + cond_s
                    + ")",
                )
        return None

    def _strip_not(self, expr: TExpr) -> TExpr:
        if isinstance(expr, TUnaryOp) and expr.op == "!":
            return expr.operand
        return expr

    def _fold_temp_assign(
        self, stmts: list[TStmt], i: int, temp_name: str, rhs: str
    ) -> str | None:
        if i + 2 >= len(stmts):
            return None
        third = stmts[i + 2]
        if isinstance(third, TLetStmt) and isinstance(third.value, TVar):
            if third.value.name == temp_name:
                real = _restore_name(third.name, third.annotations)
                return "let " + real + " = " + rhs + ";"
        if isinstance(third, TAssignStmt) and isinstance(third.value, TVar):
            if third.value.name == temp_name and isinstance(third.target, TVar):
                real = _restore_name(third.target.name, third.target.annotations)
                return real + " = " + rhs + ";"
        return None

    def _emit_stmt(self, stmt: TStmt) -> None:
        match stmt:
            case TLetStmt():
                self._emit_let(stmt)
            case TAssignStmt():
                if isinstance(stmt.target, TIndex) and self._is_map_type(
                    stmt.target.obj
                ):
                    self._line(
                        self._expr(stmt.target.obj)
                        + ".set("
                        + self._expr(stmt.target.index)
                        + ", "
                        + self._expr(stmt.value)
                        + ");"
                    )
                else:
                    self._line(
                        self._expr(stmt.target) + " = " + self._expr(stmt.value) + ";"
                    )
            case TTupleAssignStmt():
                self._emit_tuple_assign(stmt)
            case TOpAssignStmt():
                if (
                    self.strict_math
                    and stmt.op in STRICT_INT_COMPOUND
                    and self._is_int_expr(stmt.target)
                ):
                    fn = STRICT_INT_COMPOUND[stmt.op]
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
                    if isinstance(stmt.value, TTernary) and stmt.value.annotations.get(
                        "provenance"
                    ) in ("partition", "rpartition"):
                        self._emit_partition_return(stmt.value)
                    else:
                        self._line("return " + self._expr(stmt.value) + ";")
                else:
                    self._line("return;")
            case TThrowStmt():
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

    def _emit_let(self, stmt: TLetStmt) -> None:
        safe = _restore_name(stmt.name, stmt.annotations)
        self.var_types[stmt.name] = stmt.typ
        unused = stmt.annotations.get("liveness.initial_value_unused") == "true"
        if stmt.value is not None and not unused:
            self._line("let " + safe + " = " + self._expr(stmt.value) + ";")
        else:
            self._line("let " + safe + ";")

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
        parts: list[str] = []
        for i, t in enumerate(stmt.targets):
            if i in unused_indices:
                parts.append("_")
            else:
                parts.append(self._expr(t))
        self._line("[" + ", ".join(parts) + "] = " + self._expr(stmt.value) + ";")

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
            self._line(q_target + " = Math.trunc(" + a + " / " + b + ");")
        else:
            r_target = self._expr(stmt.targets[1])
            self._line(q_target + " = Math.trunc(" + a + " / " + b + ");")
            self._line(r_target + " = " + a + " - " + q_target + " * " + b + ";")

    def _emit_partition_return(self, expr: TTernary) -> None:
        prov = expr.annotations.get("provenance", "")
        cond = expr.cond
        if isinstance(cond, TBinaryOp) and isinstance(cond.left, TCall):
            call = cond.left
            obj_s = self._expr(call.args[0].value)
            sep_s = self._expr(call.args[1].value)
            method = "indexOf" if prov == "partition" else "lastIndexOf"
            self._line("let __idx = " + obj_s + "." + method + "(" + sep_s + ");")
            self._line("if (__idx >= 0) {")
            self.indent += 1
            self._line(
                "return ["
                + obj_s
                + ".slice(0, __idx), "
                + sep_s
                + ", "
                + obj_s
                + ".slice(__idx + "
                + sep_s
                + ".length)];"
            )
            self.indent -= 1
            self._line("} else {")
            self.indent += 1
            if prov == "partition":
                self._line("return [" + obj_s + ', "", ""];')
            else:
                self._line('return ["", "", ' + obj_s + "];")
            self.indent -= 1
            self._line("}")
            return
        self._line("return " + self._expr(expr) + ";")

    def _emit_expr_stmt(self, stmt: TExprStmt) -> None:
        expr = stmt.expr
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            name = expr.func.name
            if name == "Assert":
                args = expr.args
                cond = self._expr(args[0].value)
                if len(args) > 1:
                    msg = self._expr(args[1].value)
                else:
                    msg = '"assertion failed"'
                self._line("if (!(" + cond + ")) { throw new Error(" + msg + "); }")
                return
            if name == "Delete":
                args = expr.args
                if stmt.annotations.get("provenance") == "del_subscript":
                    self._line(
                        self._expr(args[0].value)
                        + ".delete("
                        + self._expr(args[1].value)
                        + ");"
                    )
                else:
                    self._line(
                        self._expr(args[0].value)
                        + ".delete("
                        + self._expr(args[1].value)
                        + ");"
                    )
                return
            if name == "RemoveAt":
                args = expr.args
                if stmt.annotations.get("provenance") == "del_subscript":
                    self._line(
                        self._expr(args[0].value)
                        + ".splice("
                        + self._expr(args[1].value)
                        + ", 1);"
                    )
                else:
                    self._line(
                        self._expr(args[0].value)
                        + ".splice("
                        + self._expr(args[1].value)
                        + ", 1);"
                    )
                return
        # Map index assignment: m[k] = v → m.set(k, v)
        if isinstance(expr, TCall):
            pass
        self._line(self._expr(expr) + ";")

    def _emit_if(self, stmt: TIfStmt) -> None:
        prov = stmt.annotations.get("provenance", "")
        if prov == "truthiness":
            truth = self._truthiness_expr(stmt.cond)
            if truth is not None:
                self._line("if (" + truth + ") {")
            else:
                self._line("if (" + self._expr(stmt.cond) + ") {")
        else:
            self._line("if (" + self._expr(stmt.cond) + ") {")
        self.indent += 1
        self._emit_stmts(stmt.then_body)
        self.indent -= 1
        self._emit_else_body(stmt.else_body)

    def _truthiness_expr(self, cond: TExpr) -> str | None:
        if isinstance(cond, TBinaryOp):
            if (
                cond.op == ">"
                and isinstance(cond.right, TIntLit)
                and cond.right.value == 0
                and isinstance(cond.left, TCall)
                and isinstance(cond.left.func, TVar)
                and cond.left.func.name == "Len"
            ):
                inner = cond.left.args[0].value
                len_str = self._len_expr(inner)
                return len_str
            if (
                cond.op == "!="
                and isinstance(cond.right, TStringLit)
                and not cond.right.value
            ):
                return self._expr(cond.left)
        return None

    def _len_expr(self, expr: TExpr) -> str:
        """Emit the length/size property for truthiness checks."""
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            if isinstance(typ, (TMapType, TSetType)):
                return self._expr(expr) + ".size"
        ann = expr.annotations.get("type", "")
        if ann.startswith("map[") or ann.startswith("set["):
            return self._expr(expr) + ".size"
        return self._expr(expr) + ".length"

    def _emit_else_body(self, else_body: list[TStmt] | None) -> None:
        if else_body is None or not else_body:
            self._line("}")
            return
        elif_stmt: TStmt | None = None
        if len(else_body) == 1:
            elif_stmt = else_body[0]
        if isinstance(elif_stmt, TIfStmt):
            prov = elif_stmt.annotations.get("provenance", "")
            if prov == "truthiness":
                truth = self._truthiness_expr(elif_stmt.cond)
                if truth is not None:
                    self._line("} else if (" + truth + ") {")
                else:
                    self._line("} else if (" + self._expr(elif_stmt.cond) + ") {")
            else:
                self._line("} else if (" + self._expr(elif_stmt.cond) + ") {")
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
        prov = stmt.annotations.get("provenance", "")
        if prov == "truthiness":
            truth = self._truthiness_expr(stmt.cond)
            if truth is not None:
                self._line("while (" + truth + ") {")
            else:
                self._line("while (" + self._expr(stmt.cond) + ") {")
        else:
            self._line("while (" + self._expr(stmt.cond) + ") {")
        self.indent += 1
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _is_builtin_call(self, expr: TExpr, name: str) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == name
        )

    def _emit_for(self, stmt: TForStmt) -> None:
        binding = stmt.binding
        ann = stmt.annotations
        prov = ann.get("provenance", "")
        if isinstance(stmt.iterable, TRange):
            self._emit_for_range(stmt, binding, ann, prov)
        elif self._is_zip_for(stmt):
            self._emit_for_zip(stmt, binding, ann)
        elif len(binding) == 1:
            b = _restore_name(binding[0], ann)
            iterable = self._for_iterable(stmt.iterable)
            if self._is_map_type(stmt.iterable):
                self._line(
                    "for (const "
                    + b
                    + " of "
                    + self._expr(stmt.iterable)
                    + ".keys()) {"
                )
            else:
                self._line("for (const " + b + " of " + iterable + ") {")
        elif len(binding) == 2:
            iter_is_map = self._is_map_for(stmt)
            is_enumerate = self._is_enumerate_for(stmt)
            b0 = _restore_name(binding[0], ann)
            b1 = _restore_name(binding[1], ann)
            if iter_is_map:
                self._line(
                    "for (const ["
                    + b0
                    + ", "
                    + b1
                    + "] of "
                    + self._expr(stmt.iterable)
                    + ") {"
                )
            elif is_enumerate:
                self._line(
                    "for (const ["
                    + b0
                    + ", "
                    + b1
                    + "] of "
                    + self._expr(stmt.iterable)
                    + ".entries()) {"
                )
            else:
                binder_parts: list[str] = []
                for b in binding:
                    binder_parts.append(_restore_name(b, ann))
                binders = ", ".join(binder_parts)
                self._line(
                    "for (const ["
                    + binders
                    + "] of "
                    + self._for_iterable(stmt.iterable)
                    + ") {"
                )
        else:
            binder_parts: list[str] = []
            for b in binding:
                binder_parts.append(_restore_name(b, ann))
            binders = ", ".join(binder_parts)
            self._line(
                "for (const ["
                + binders
                + "] of "
                + self._for_iterable(stmt.iterable)
                + ") {"
            )
        self.indent += 1
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
            end_val = self._static_int(rargs[1])
            start_val = self._static_int(rargs[0])
            if end_val is not None and start_val is not None:
                low = end_val + 1 if end_val >= 0 else end_val + 1
                self._line(
                    "for (let "
                    + binder
                    + " = "
                    + str(start_val)
                    + "; "
                    + binder
                    + " >= "
                    + str(low)
                    + "; "
                    + binder
                    + "--) {"
                )
                return
        if len(args) == 1:
            end = self._expr(args[0])
            self._line(
                "for (let "
                + binder
                + " = 0; "
                + binder
                + " < "
                + end
                + "; "
                + binder
                + "++) {"
            )
        elif len(args) == 2:
            start = self._expr(args[0])
            end = self._expr(args[1])
            self._line(
                "for (let "
                + binder
                + " = "
                + start
                + "; "
                + binder
                + " < "
                + end
                + "; "
                + binder
                + "++) {"
            )
        elif len(args) == 3:
            start = self._expr(args[0])
            end = self._expr(args[1])
            step_val = self._static_int(args[2])
            if step_val is not None:
                if step_val == -1:
                    self._line(
                        "for (let "
                        + binder
                        + " = "
                        + start
                        + "; "
                        + binder
                        + " > "
                        + end
                        + "; "
                        + binder
                        + "--) {"
                    )
                elif step_val < 0:
                    self._line(
                        "for (let "
                        + binder
                        + " = "
                        + start
                        + "; "
                        + binder
                        + " > "
                        + end
                        + "; "
                        + binder
                        + " += "
                        + self._expr(args[2])
                        + ") {"
                    )
                else:
                    step = self._expr(args[2])
                    self._line(
                        "for (let "
                        + binder
                        + " = "
                        + start
                        + "; "
                        + binder
                        + " < "
                        + end
                        + "; "
                        + binder
                        + " += "
                        + step
                        + ") {"
                    )
            else:
                step = self._expr(args[2])
                self._line(
                    "for (let "
                    + binder
                    + " = "
                    + start
                    + "; "
                    + binder
                    + " < "
                    + end
                    + "; "
                    + binder
                    + " += "
                    + step
                    + ") {"
                )

    def _is_zip_for(self, stmt: TForStmt) -> bool:
        return not isinstance(stmt.iterable, TRange) and self._is_builtin_call(
            stmt.iterable, "Zip"
        )

    def _emit_for_zip(self, stmt: TForStmt, binding: list[str], ann: Ann) -> None:
        assert isinstance(stmt.iterable, TCall)
        zip_args = stmt.iterable.args
        arr_exprs = [self._expr(a.value) for a in zip_args]
        min_parts = [e + ".length" for e in arr_exprs]
        min_expr = "Math.min(" + ", ".join(min_parts) + ")"
        self._line("for (let __i = 0; __i < " + min_expr + "; __i++) {")
        self.indent += 1
        for j, b in enumerate(binding):
            bname = _restore_name(b, ann)
            self._line("const " + bname + " = " + arr_exprs[j] + "[__i];")

    def _for_iterable(self, iterable: TExpr) -> str:
        if self._is_builtin_call(iterable, "Reversed") and isinstance(iterable, TCall):
            return "[..." + self._expr(iterable.args[0].value) + "].reverse()"
        if self._is_builtin_call(iterable, "Zip") and isinstance(iterable, TCall):
            return self._expr(iterable)
        return self._expr(iterable)

    def _emit_for_keys(self, stmt: TForStmt, binding: list[str], ann: Ann) -> None:
        b = _restore_name(binding[0], ann)
        self._line(
            "for (const " + b + " of " + self._expr(stmt.iterable) + ".keys()) {"
        )

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

    def _is_map_type(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann.startswith("map[")
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TMapType)
        return False

    def _is_set_type(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann.startswith("set[")
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TSetType)
        return False

    def _is_bytes_expr(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann == "bytes":
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "bytes"
        return False

    def _is_map_for(self, stmt: TForStmt) -> bool:
        if stmt.annotations.get("for.items") == "true":
            return True
        return not isinstance(stmt.iterable, TRange) and self._is_map_type(
            stmt.iterable
        )

    # ── Try / Catch ───────────────────────────────────────────

    def _emit_try(self, stmt: TTryStmt) -> None:
        self._line("try {")
        self.indent += 1
        self._emit_stmts(stmt.body)
        self.indent -= 1
        if stmt.catches:
            all_unused = all(
                c.annotations.get("liveness.catch_var_unused") == "true"
                for c in stmt.catches
            )
            if len(stmt.catches) == 1:
                catch = stmt.catches[0]
                unused = catch.annotations.get("liveness.catch_var_unused") == "true"
                has_types = bool(catch.types)
                if has_types and not unused:
                    cname = _restore_name(catch.name, catch.annotations)
                    self._line("} catch (" + cname + ") {")
                    self.indent += 1
                    self._emit_multi_catch(stmt.catches, cname)
                    self.indent -= 1
                elif has_types and unused:
                    self._line("} catch {")
                    self.indent += 1
                    self._emit_stmts(catch.body)
                    self.indent -= 1
                elif unused:
                    self._line("} catch (_) {")
                    self.indent += 1
                    self._emit_stmts(catch.body)
                    self.indent -= 1
                else:
                    cname = _restore_name(catch.name, catch.annotations)
                    self._line("} catch (" + cname + ") {")
                    self.indent += 1
                    self._emit_stmts(catch.body)
                    self.indent -= 1
            else:
                catch_name = "e"
                for c in stmt.catches:
                    if c.annotations.get("liveness.catch_var_unused") != "true":
                        catch_name = _restore_name(c.name, c.annotations)
                        break
                if all_unused:
                    self._line("} catch (_) {")
                else:
                    self._line("} catch (" + catch_name + ") {")
                self.indent += 1
                self._emit_multi_catch(stmt.catches, catch_name)
                self.indent -= 1
        if stmt.finally_body is not None:
            self._line("} finally {")
            self.indent += 1
            self._emit_stmts(stmt.finally_body)
            self.indent -= 1
        self._line("}")

    def _emit_multi_catch(self, catches: list[TCatch], catch_name: str) -> None:
        first = True
        has_typed = False
        has_untyped = False
        for catch in catches:
            if catch.types:
                has_typed = True
            else:
                has_untyped = True
        for catch in catches:
            if catch.types:
                types: list[str] = []
                for t in catch.types:
                    if isinstance(t, TIdentType):
                        types.append(t.name)
                    else:
                        types.append("Error")
                cond = " || ".join(catch_name + " instanceof " + tn for tn in types)
                keyword = "if" if first else "} else if"
                self._line(keyword + " (" + cond + ") {")
                self.indent += 1
                self._emit_stmts(catch.body)
                self.indent -= 1
                first = False
            else:
                if first:
                    self._emit_stmts(catch.body)
                else:
                    self._line("} else {")
                    self.indent += 1
                    self._emit_stmts(catch.body)
                    self.indent -= 1
                first = False
        if has_typed:
            self._line("}")

    def _emit_catch(self, catch: TCatch) -> None:
        unused = catch.annotations.get("liveness.catch_var_unused") == "true"
        if unused:
            self._line("} catch (_) {")
        else:
            cname = _restore_name(catch.name, catch.annotations)
            self._line("} catch (" + cname + ") {")
        self.indent += 1
        self._emit_stmts(catch.body)
        self.indent -= 1

    # ── Match ─────────────────────────────────────────────────

    def _emit_match(self, stmt: TMatchStmt) -> None:
        expr_str = self._expr(stmt.expr)
        first = True
        for case in stmt.cases:
            self._emit_match_case(case, expr_str, first)
            first = False
        if stmt.default is not None:
            self._emit_match_default(stmt.default, expr_str, first)

    def _emit_match_case(self, case: TMatchCase, expr_str: str, first: bool) -> None:
        pat = case.pattern
        keyword = "if" if first else "} else if"
        if isinstance(pat, TPatternType):
            type_name = self._pattern_type_name(pat.type_name)
            prim = self._js_typeof(type_name)
            if prim is not None:
                self._line(keyword + " (typeof " + expr_str + ' === "' + prim + '") {')
            else:
                self._line(
                    keyword + " (" + expr_str + " instanceof " + type_name + ") {"
                )
            self.indent += 1
            unused = pat.annotations.get("liveness.match_var_unused") == "true"
            if not unused:
                self._line("let " + _safe_name(pat.name) + " = " + expr_str + ";")
            self._emit_stmts(case.body)
            self.indent -= 1
        elif isinstance(pat, TPatternEnum):
            self._line(
                keyword
                + " ("
                + expr_str
                + " === "
                + pat.enum_name
                + "."
                + pat.variant
                + ") {"
            )
            self.indent += 1
            self._emit_stmts(case.body)
            self.indent -= 1
        elif isinstance(pat, TPatternNil):
            self._line(keyword + " (" + expr_str + " === null) {")
            self.indent += 1
            self._emit_stmts(case.body)
            self.indent -= 1

    def _js_typeof(self, type_name: str) -> str | None:
        if type_name == "int" or type_name == "float":
            return "number"
        if type_name == "string":
            return "string"
        if type_name == "bool":
            return "boolean"
        return None

    def _emit_match_default(
        self, default: TDefault, expr_str: str, first: bool
    ) -> None:
        if first:
            pass
        else:
            self._line("} else {")
        self.indent += 1
        if default.name is not None:
            unused = default.annotations.get("liveness.match_var_unused") == "true"
            if not unused:
                self._line("let " + _safe_name(default.name) + " = " + expr_str + ";")
        self._emit_stmts(default.body)
        self.indent -= 1
        self._line("}")

    def _pattern_type_name(self, typ: TType) -> str:
        if isinstance(typ, TIdentType):
            return typ.name
        if isinstance(typ, TPrimitive):
            return typ.kind
        return "Object"

    # ── Expressions ───────────────────────────────────────────

    def _expr(self, expr: TExpr) -> str:
        if isinstance(expr, TIntLit):
            return self._int_lit(expr)
        if isinstance(expr, TFloatLit):
            return expr.raw
        if isinstance(expr, TStringLit):
            return '"' + escape_string(expr.value) + '"'
        if isinstance(expr, TBoolLit):
            return "true" if expr.value else "false"
        if isinstance(expr, TNilLit):
            return "null"
        if isinstance(expr, TByteLit):
            return expr.raw
        if isinstance(expr, TBytesLit):
            return self._bytes_lit(expr)
        if isinstance(expr, TRuneLit):
            return '"' + escape_string(expr.value) + '"'
        if isinstance(expr, TVar):
            if expr.name == self.self_name:
                return "this"
            if (
                expr.name in self.fn_names
                and expr.name not in BUILTIN_NAMES
                and expr.name not in self.struct_names
            ):
                return _safe_name(
                    _to_lower_camel(_restore_name(expr.name, expr.annotations))
                )
            return _restore_name(expr.name, expr.annotations)
        if isinstance(expr, TFieldAccess):
            return self._expr(expr.obj) + "." + _safe_name(expr.field)
        if isinstance(expr, TTupleAccess):
            return self._expr(expr.obj) + "[" + str(expr.index) + "]"
        if isinstance(expr, TIndex):
            if expr.annotations.get("provenance") == "negative_index":
                neg = self._negative_index(expr)
                if neg is not None:
                    return self._expr(expr.obj) + "[" + neg + "]"
            if self._is_map_type(expr.obj):
                return self._expr(expr.obj) + ".get(" + self._expr(expr.index) + ")"
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
            elems = self._join_exprs(expr.elements, ", ")
            return "[" + elems + "]"
        if isinstance(expr, TMapLit):
            if not expr.entries:
                return "new Map()"
            pairs = ", ".join(
                "[" + self._expr(k) + ", " + self._expr(v) + "]"
                for k, v in expr.entries
            )
            return "new Map([" + pairs + "])"
        if isinstance(expr, TSetLit):
            if not expr.elements:
                return "new Set()"
            elems = self._join_exprs(expr.elements, ", ")
            return "new Set([" + elems + "])"
        if isinstance(expr, TTupleLit):
            elems = self._join_exprs(expr.elements, ", ")
            return "[" + elems + "]"
        if isinstance(expr, TFnLit):
            return self._fn_lit(expr)
        if isinstance(expr, TCall):
            return self._call(expr)
        raise NotImplementedError("unknown expression")

    def _int_lit(self, expr: TIntLit) -> str:
        raw = expr.raw
        if raw.startswith(("0x", "0X", "0o", "0O", "0b", "0B")):
            return raw
        return str(expr.value)

    def _bytes_lit(self, expr: TBytesLit) -> str:
        parts: list[str] = []
        for b in expr.value:
            h = hex(b)[2:]
            if len(h) == 1:
                h = "0" + h
            parts.append("\\x" + h)
        return 'Buffer.from("' + "".join(parts) + '")'

    def _slice(self, expr: TSlice) -> str:
        obj = self._expr(expr.obj)
        prov = expr.annotations.get("provenance", "")
        low = self._expr(expr.low)
        high = self._expr(expr.high)
        if prov == "open_end" and self._is_len_call(expr.high):
            high = ""
        if prov == "open_start" and self._is_zero(expr.low):
            low = "0"
        if high == "":
            return obj + ".slice(" + low + ")"
        return obj + ".slice(" + low + ", " + high + ")"

    def _negative_index(self, expr: TIndex) -> str | None:
        idx = expr.index
        if isinstance(idx, TBinaryOp) and idx.op == "-":
            if (
                isinstance(idx.left, TCall)
                and isinstance(idx.left.func, TVar)
                and idx.left.func.name == "Len"
            ):
                return (
                    self._expr(idx.left.args[0].value)
                    + ".length - "
                    + self._expr(idx.right)
                )
        return None

    def _binary(self, expr: TBinaryOp) -> str:
        op = expr.op
        if (
            op == "/"
            and isinstance(expr.left, TFloatLit)
            and expr.left.value == 0.0
            and isinstance(expr.right, TFloatLit)
            and expr.right.value == 0.0
        ):
            return "NaN"
        if self.strict_math:
            if op in STRICT_INT_BINARY and self._is_int_expr(expr.left):
                fn = STRICT_INT_BINARY[op]
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
                    "strict_fmod("
                    + self._expr(expr.left)
                    + ", "
                    + self._expr(expr.right)
                    + ")"
                )
        # isinstance tuple
        if op == "||" and expr.annotations.get("provenance") == "isinstance_tuple":
            types: list[str] = []
            obj = self._flatten_isinstance_tuple(expr, types)
            if obj is not None:
                parts = [obj + " instanceof " + t for t in types]
                return " || ".join(parts)
        # chained comparison: keep desugared form for JS
        if op == "&&" and expr.annotations.get("provenance") == "chained_comparison":
            left_str = self._maybe_paren(expr.left, op, is_left=True)
            right_str = self._maybe_paren(expr.right, op, is_left=False)
            return left_str + " && " + right_str
        # nil comparisons
        if op == "==" and isinstance(expr.right, TNilLit):
            return self._maybe_paren(expr.left, op, is_left=True) + " === null"
        if op == "!=" and isinstance(expr.right, TNilLit):
            return self._maybe_paren(expr.left, op, is_left=True) + " !== null"
        if op == "==" and isinstance(expr.left, TNilLit):
            return self._maybe_paren(expr.right, op, is_left=False) + " === null"
        if op == "!=" and isinstance(expr.left, TNilLit):
            return self._maybe_paren(expr.right, op, is_left=False) + " !== null"
        js_op = op
        if op == "==":
            js_op = "==="
        elif op == "!=":
            js_op = "!=="
        left_str = self._maybe_paren(expr.left, js_op, is_left=True)
        right_str = self._maybe_paren(expr.right, js_op, is_left=False)
        return left_str + " " + js_op + " " + right_str

    def _unary(self, expr: TUnaryOp) -> str:
        op = expr.op
        if self.strict_math and op == "-" and self._is_int_expr(expr.operand):
            return "checked_neg_i64(" + self._expr(expr.operand) + ")"
        if op == "!":
            if (
                isinstance(expr.operand, TCall)
                and isinstance(expr.operand.func, TVar)
                and expr.operand.func.name == "Contains"
                and expr.operand.annotations.get("provenance") == "not_in_operator"
            ):
                return (
                    "!"
                    + self._a(expr.operand.args, 0)
                    + ".includes("
                    + self._a(expr.operand.args, 1)
                    + ")"
                )
            if isinstance(expr.operand, (TBinaryOp,)):
                if expr.operand.op in ("&&", "||"):
                    return "!(" + self._expr(expr.operand) + ")"
                if expr.operand.op in (">", "<", ">=", "<=", "==", "!="):
                    return "!(" + self._expr(expr.operand) + ")"
                return "!" + self._expr(expr.operand)
            if isinstance(expr.operand, (TTernary,)):
                return "!(" + self._expr(expr.operand) + ")"
            return "!" + self._expr(expr.operand)
        if isinstance(expr.operand, (TBinaryOp, TTernary)):
            return op + "(" + self._expr(expr.operand) + ")"
        return op + self._expr(expr.operand)

    def _ternary(self, expr: TTernary) -> str:
        prov = expr.annotations.get("provenance", "")
        if prov == "none_coalesce":
            return self._none_coalesce(expr)
        if prov == "partition" or prov == "rpartition":
            return self._partition_ternary(expr, prov)
        if prov == "removeprefix" or prov == "removesuffix":
            return self._remove_affix_ternary(expr, prov)
        cond = self._expr(expr.cond)
        then = self._expr(expr.then_expr)
        if isinstance(expr.then_expr, TTernary):
            then = "(" + then + ")"
        else_s = self._expr(expr.else_expr)
        if isinstance(expr.else_expr, TTernary):
            else_s = "(" + else_s + ")"
        return cond + " ? " + then + " : " + else_s

    def _none_coalesce(self, expr: TTernary) -> str:
        cond = expr.cond
        if isinstance(cond, TBinaryOp) and cond.op == "!=":
            if isinstance(cond.right, TNilLit):
                return self._expr(cond.left) + " ?? " + self._expr(expr.else_expr)
            if isinstance(cond.left, TNilLit):
                return self._expr(cond.right) + " ?? " + self._expr(expr.else_expr)
        return self._expr(expr.then_expr) + " ?? " + self._expr(expr.else_expr)

    def _partition_ternary(self, expr: TTernary, prov: str) -> str:
        pt_cond = expr.cond
        if isinstance(pt_cond, TBinaryOp) and isinstance(pt_cond.left, TCall):
            pt_call = pt_cond.left
            obj_s = self._expr(pt_call.args[0].value)
            sep_s = self._expr(pt_call.args[1].value)
            method = "indexOf" if prov == "partition" else "lastIndexOf"
            return obj_s + "." + method + "(" + sep_s + ")"
        return (
            self._expr(expr.then_expr)
            + " ? "
            + self._expr(expr.cond)
            + " : "
            + self._expr(expr.else_expr)
        )

    def _remove_affix_ternary(self, expr: TTernary, prov: str) -> str:
        rx_cond = expr.cond
        if isinstance(rx_cond, TCall):
            obj_s = self._expr(rx_cond.args[0].value)
            arg_s = self._expr(rx_cond.args[1].value)
            if prov == "removeprefix":
                return (
                    obj_s
                    + ".startsWith("
                    + arg_s
                    + ") ? "
                    + obj_s
                    + ".slice("
                    + arg_s
                    + ".length) : "
                    + obj_s
                )
            else:
                return (
                    obj_s
                    + ".endsWith("
                    + arg_s
                    + ") ? "
                    + obj_s
                    + ".slice(0, "
                    + obj_s
                    + ".length - "
                    + arg_s
                    + ".length) : "
                    + obj_s
                )
        return (
            self._expr(expr.then_expr)
            + " ? "
            + self._expr(expr.cond)
            + " : "
            + self._expr(expr.else_expr)
        )

    def _flatten_isinstance_tuple(self, expr: TExpr, types: list[str]) -> str | None:
        if isinstance(expr, TBinaryOp) and expr.op == "||":
            obj = self._flatten_isinstance_tuple(expr.left, types)
            rhs = expr.right
            if (
                isinstance(rhs, TCall)
                and isinstance(rhs.func, TVar)
                and rhs.func.name == "IsType"
            ):
                type_arg = rhs.args[1].value
                if isinstance(type_arg, TStringLit):
                    types.append(type_arg.value)
                return obj
        if (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "IsType"
        ):
            type_arg = expr.args[1].value
            if isinstance(type_arg, TStringLit):
                types.append(type_arg.value)
            return self._expr(expr.args[0].value)
        return None

    def _maybe_paren(self, expr: TExpr, parent_op: str, is_left: bool) -> str:
        if isinstance(expr, TBinaryOp):
            child_op = expr.op
            if child_op == "==":
                child_op = "==="
            elif child_op == "!=":
                child_op = "!=="
            if _needs_parens(child_op, parent_op, is_left):
                return "(" + self._expr(expr) + ")"
        elif isinstance(expr, TTernary):
            return "(" + self._expr(expr) + ")"
        elif isinstance(expr, TUnaryOp):
            if expr.op == "!" and parent_op in _CMP_OPS:
                return "(" + self._expr(expr) + ")"
            if expr.op in ("-", "+") and parent_op == "**" and is_left:
                return "(" + self._expr(expr) + ")"
        return self._expr(expr)

    def _fn_lit(self, expr: TFnLit) -> str:
        params = ", ".join(
            _restore_name(p.name, p.annotations)
            for p in expr.params
            if p.typ is not None
        )
        first = expr.body[0] if expr.body else None
        if expr.annotations.get("fn_lit.arrow") == "true" and isinstance(
            first, TExprStmt
        ):
            return "(" + params + ") => " + self._expr(first.expr)
        if (
            isinstance(first, TReturnStmt)
            and first.value is not None
            and len(expr.body) == 1
        ):
            return "(" + params + ") => " + self._expr(first.value)
        # Block body: collect lines for inline embedding
        old_lines = self.lines
        old_indent = self.indent
        self.lines = []
        self.indent = 0
        for s in expr.body:
            self._emit_stmt(s)
        body_lines = self.lines
        self.lines = old_lines
        self.indent = old_indent
        result = "(" + params + ") => {\n"
        for bl in body_lines:
            result += "    " * self.indent + "    " + bl + "\n"
        result += "    " * self.indent + "}"
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
            if isinstance(inner, TVar):
                typ = self.var_types.get(inner.name)
                if isinstance(typ, TPrimitive) and typ.kind == "string":
                    return "[..." + self._a(args, 0) + '].reverse().join("")'
            ann_type = inner.annotations.get("type", "")
            if ann_type == "string":
                return "[..." + self._a(args, 0) + '].reverse().join("")'
            return "[..." + self._a(args, 0) + "].reverse()"
        # list(dict) / set(dict)
        if (
            isinstance(func, TVar)
            and func.name in ("ListFrom", "SetFromList")
            and expr.annotations.get("provenance") == "dict_keys"
        ):
            inner = args[0].value
            if isinstance(inner, TCall):
                dict_expr = self._expr(inner.args[0].value)
                if func.name == "ListFrom":
                    return "[..." + dict_expr + ".keys()]"
                else:
                    return "new Set(" + dict_expr + ".keys())"
        # list_repetition
        if (
            isinstance(func, TVar)
            and func.name == "Repeat"
            and expr.annotations.get("provenance") == "list_repetition"
        ):
            count = self._a(args, 1)
            elem = args[0].value
            if isinstance(elem, TListLit) and len(elem.elements) == 1:
                return "Array(" + count + ").fill(" + self._expr(elem.elements[0]) + ")"
            return "Array(" + count + ").fill(" + self._a(args, 0) + ")"
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
        arg_strs = self._join_args(args, ", ")
        return fn_name + "(" + arg_strs + ")"

    def _star_unpack(self, expr: TCall) -> str:
        parts: list[TExpr] = []
        self._flatten_star_unpack(expr, parts)
        items: list[str] = []
        for p in parts:
            if isinstance(p, TListLit):
                for elem in p.elements:
                    items.append(self._expr(elem))
            else:
                items.append("..." + self._expr(p))
        return "[" + ", ".join(items) + "]"

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

    _STRUCT_NAME_MAP: dict[str, str] = {
        "Exception": "Error",
    }

    def _struct_call(self, name: str, args: list[TArg]) -> str:
        name = self._STRUCT_NAME_MAP.get(name, name)
        has_named = False
        for a in args:
            if a.name is not None:
                has_named = True
                break
        if has_named:
            ordered = self.struct_fields.get(name, [])
            if ordered:
                named: dict[str, str] = {}
                for a in args:
                    if a.name is not None:
                        k = a.name
                        if k not in ordered:
                            k = _safe_name(k)
                        named[k] = self._expr(a.value)
                vals: list[str] = []
                for f in ordered:
                    vals.append(named.get(f, "null"))
                return "new " + name + "(" + ", ".join(vals) + ")"
        parts: list[str] = []
        for a in args:
            parts.append(self._expr(a.value))
        return "new " + name + "(" + ", ".join(parts) + ")"

    def _method_call(self, func: TFieldAccess, args: list[TArg]) -> str:
        obj_str = self._expr(func.obj)
        if isinstance(func.obj, (TBinaryOp, TUnaryOp, TTernary)):
            obj_str = "(" + obj_str + ")"
        arg_strs = self._join_args(args, ", ")
        field = _safe_name(_to_lower_camel(func.field))
        return obj_str + "." + field + "(" + arg_strs + ")"

    def _builtin_call(self, name: str, args: list[TArg], ann: Ann | None = None) -> str:
        if ann is None:
            ann = {}
        if name == "FloorDiv":
            return "Math.floor(" + self._a(args, 0) + " / " + self._a(args, 1) + ")"
        if name == "PythonMod":
            a = self._a(args, 0)
            b = self._a(args, 1)
            return "((" + a + " % " + b + ") + " + b + ") % " + b
        if name == "Append":
            return self._a(args, 0) + ".push(" + self._a(args, 1) + ")"
        if name == "Insert":
            return (
                self._a(args, 0)
                + ".splice("
                + self._a(args, 1)
                + ", 0, "
                + self._a(args, 2)
                + ")"
            )
        if name == "Pop":
            return self._a(args, 0) + ".pop()"
        if name == "RemoveAt":
            return self._a(args, 0) + ".splice(" + self._a(args, 1) + ", 1)"
        if name == "IndexOf":
            return self._a(args, 0) + ".indexOf(" + self._a(args, 1) + ")"
        if name == "Upper":
            return self._a(args, 0) + ".toUpperCase()"
        if name == "Lower":
            return self._a(args, 0) + ".toLowerCase()"
        if name == "Trim":
            return self._a(args, 0) + '.replace(/^ +| +$/g, "")'
        if name == "TrimStart":
            return self._a(args, 0) + '.replace(/^ +/, "")'
        if name == "TrimEnd":
            return self._a(args, 0) + '.replace(/ +$/, "")'
        if name == "Split":
            return self._a(args, 0) + ".split(" + self._a(args, 1) + ")"
        if name == "SplitN":
            obj = self._a(args, 0)
            sep = self._a(args, 1)
            n_expr = args[2].value
            if isinstance(n_expr, TIntLit):
                return obj + ".split(" + sep + ", " + str(n_expr.value) + ")"
            return obj + ".split(" + sep + ", " + self._a(args, 2) + ")"
        if name == "SplitWhitespace":
            return self._a(args, 0) + ".trim().split(/\\s+/)"
        if name == "Join":
            return self._a(args, 1) + ".join(" + self._a(args, 0) + ")"
        if name == "Find":
            return self._a(args, 0) + ".indexOf(" + self._a(args, 1) + ")"
        if name == "RFind":
            return self._a(args, 0) + ".lastIndexOf(" + self._a(args, 1) + ")"
        if name == "Count":
            return self._a(args, 0) + ".split(" + self._a(args, 1) + ").length - 1"
        if name == "Replace":
            return (
                self._a(args, 0)
                + ".replaceAll("
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ")"
            )
        if name == "ReplaceCount":
            return (
                self._a(args, 0)
                + ".replace("
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ")"
            )
        if name == "StartsWith":
            return self._a(args, 0) + ".startsWith(" + self._a(args, 1) + ")"
        if name == "EndsWith":
            return self._a(args, 0) + ".endsWith(" + self._a(args, 1) + ")"
        if name == "IsDigit":
            return "/^\\d+$/.test(" + self._a(args, 0) + ")"
        if name == "IsAlpha":
            return "/^[a-zA-Z]+$/.test(" + self._a(args, 0) + ")"
        if name == "IsAlnum":
            return "/^[a-zA-Z0-9]+$/.test(" + self._a(args, 0) + ")"
        if name == "IsSpace":
            return "/^\\s+$/.test(" + self._a(args, 0) + ")"
        if name == "IsUpper":
            return self._a(args, 0) + " === " + self._a(args, 0) + ".toUpperCase()"
        if name == "IsLower":
            return self._a(args, 0) + " === " + self._a(args, 0) + ".toLowerCase()"
        if name == "Encode":
            return "Buffer.from(" + self._a(args, 0) + ', "utf-8")'
        if name == "Decode":
            return self._a(args, 0) + '.toString("utf-8")'
        if name == "Add":
            return self._a(args, 0) + ".add(" + self._a(args, 1) + ")"
        if name == "Remove":
            return self._a(args, 0) + ".delete(" + self._a(args, 1) + ")"
        if name == "Get":
            if len(args) == 3:
                return (
                    self._a(args, 0)
                    + ".get("
                    + self._a(args, 1)
                    + ") ?? "
                    + self._a(args, 2)
                )
            return self._a(args, 0) + ".get(" + self._a(args, 1) + ") ?? null"
        if name == "Delete":
            return self._a(args, 0) + ".delete(" + self._a(args, 1) + ")"
        if name == "Union":
            return "new Set([..." + self._a(args, 0) + ", ..." + self._a(args, 1) + "])"
        if name == "Intersection":
            return (
                "new Set([..."
                + self._a(args, 0)
                + "].filter(x => "
                + self._a(args, 1)
                + ".has(x)))"
            )
        if name == "Difference":
            return (
                "new Set([..."
                + self._a(args, 0)
                + "].filter(x => !"
                + self._a(args, 1)
                + ".has(x)))"
            )
        if name == "Merge":
            return "new Map([..." + self._a(args, 0) + ", ..." + self._a(args, 1) + "])"
        if name == "Keys":
            return "[..." + self._a(args, 0) + ".keys()]"
        if name == "Values":
            return "[..." + self._a(args, 0) + ".values()]"
        if name == "Items":
            return "[..." + self._a(args, 0) + ".entries()]"
        if name == "Len":
            inner = args[0].value
            if self._is_map_type(inner) or self._is_set_type(inner):
                return self._a(args, 0) + ".size"
            return self._a(args, 0) + ".length"
        if name == "Abs":
            return "Math.abs(" + self._a(args, 0) + ")"
        if name == "Min":
            if (
                self.strict_math
                and len(args) == 2
                and self._is_float_expr(args[0].value)
            ):
                return (
                    "strict_min_f64(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            if len(args) == 2:
                key_val = args[1].value
                if isinstance(key_val, TFnLit):
                    return (
                        self._a(args, 0)
                        + ".reduce((a, b) => "
                        + self._min_max_key_cmp(key_val, "<=")
                        + ")"
                    )
            if len(args) == 1:
                return "Math.min(..." + self._a(args, 0) + ")"
            return "Math.min(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Max":
            if (
                self.strict_math
                and len(args) == 2
                and self._is_float_expr(args[0].value)
            ):
                return (
                    "strict_max_f64(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            if len(args) == 2:
                key_val = args[1].value
                if isinstance(key_val, TFnLit):
                    return (
                        self._a(args, 0)
                        + ".reduce((a, b) => "
                        + self._min_max_key_cmp(key_val, ">=")
                        + ")"
                    )
            if len(args) == 1:
                return "Math.max(..." + self._a(args, 0) + ")"
            return "Math.max(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Sum":
            return self._a(args, 0) + ".reduce((a, b) => a + b, 0)"
        if name == "Round":
            return "Math.round(" + self._a(args, 0) + ")"
        if name == "DivMod":
            a = self._a(args, 0)
            b = self._a(args, 1)
            return (
                "Math.trunc("
                + a
                + " / "
                + b
                + "), "
                + a
                + " - Math.trunc("
                + a
                + " / "
                + b
                + ") * "
                + b
            )
        if name == "Sorted":
            if self.strict_math and self._is_float_list(args[0].value):
                return "strict_sorted_f64(" + self._a(args, 0) + ")"
            if len(args) == 2:
                key_val = args[1].value
                if isinstance(key_val, TFnLit):
                    return self._sorted_with_key(args[0].value, key_val)
            if self._is_int_list(args[0].value):
                return "[..." + self._a(args, 0) + "].sort((a, b) => a - b)"
            return "[..." + self._a(args, 0) + "].sort()"
        if name == "RangeList":
            start = args[0].value
            end = self._a(args, 1)
            step = args[2].value
            is_zero_start = isinstance(start, TIntLit) and start.value == 0
            is_one_step = isinstance(step, TIntLit) and step.value == 1
            if is_zero_start and is_one_step:
                return "Array.from({length: " + end + "}, (_, i) => i)"
            if is_one_step:
                start_s = self._a(args, 0)
                return (
                    "Array.from({length: "
                    + end
                    + " - "
                    + start_s
                    + "}, (_, i) => i + "
                    + start_s
                    + ")"
                )
            step_s = self._a(args, 2)
            start_s = self._a(args, 0)
            length = "Math.ceil((" + end + " - " + start_s + ") / " + step_s + ")"
            if is_zero_start:
                length = "Math.ceil(" + end + " / " + step_s + ")"
                end_lit = args[1].value
                step_lit = args[2].value
                if (
                    isinstance(end_lit, TIntLit)
                    and isinstance(step_lit, TIntLit)
                    and step_lit.value != 0
                    and end_lit.value % step_lit.value == 0
                ):
                    length = str(end_lit.value // step_lit.value)
            return "Array.from({length: " + length + "}, (_, i) => i * " + step_s + ")"
        if name == "ListFrom":
            return "[..." + self._a(args, 0) + "]"
        if name == "Reversed":
            return "[..." + self._a(args, 0) + "].reverse()"
        if name == "Reverse":
            inner = args[0].value
            if isinstance(inner, TVar):
                typ = self.var_types.get(inner.name)
                if isinstance(typ, TPrimitive) and typ.kind == "string":
                    return "[..." + self._a(args, 0) + '].reverse().join("")'
            ann_type = inner.annotations.get("type", "")
            if ann_type == "string":
                return "[..." + self._a(args, 0) + '].reverse().join("")'
            if isinstance(inner, TStringLit):
                return "[..." + self._a(args, 0) + '].reverse().join("")'
            return "[..." + self._a(args, 0) + "].reverse()"
        if name == "Map":
            if not args:
                return "new Map()"
            return self._a(args, 1) + ".map(" + self._a(args, 0) + ")"
        if name == "Set":
            if not args:
                return "new Set()"
            return "new Set(" + self._a(args, 0) + ")"
        if name == "SetFromList":
            if isinstance(args[0].value, TSetLit):
                return self._a(args, 0)
            inner = args[0].value
            if isinstance(inner, TCall) and isinstance(inner.func, TVar):
                if inner.func.name in ("Keys", "Values", "Items"):
                    method = {"Keys": "keys", "Values": "values", "Items": "entries"}[
                        inner.func.name
                    ]
                    return (
                        "new Set("
                        + self._expr(inner.args[0].value)
                        + "."
                        + method
                        + "())"
                    )
            return "new Set(" + self._a(args, 0) + ")"
        if name == "ToString":
            return "String(" + self._a(args, 0) + ")"
        if name == "ToRepr":
            return "JSON.stringify(" + self._a(args, 0) + ")"
        if name == "ParseInt":
            return "parseInt(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "ParseFloat":
            return "parseFloat(" + self._a(args, 0) + ")"
        if name == "FormatInt":
            return self._format_int(args)
        if name == "RuneFromInt":
            return "String.fromCodePoint(" + self._a(args, 0) + ")"
        if name == "RuneToInt":
            return self._a(args, 0) + ".codePointAt(0)"
        if name == "IntToFloat":
            return self._a(args, 0)
        if name == "FloatToInt":
            return "Math.trunc(" + self._a(args, 0) + ")"
        if name == "ByteToInt":
            return self._a(args, 0)
        if name == "IntToByte":
            return self._a(args, 0)
        if name == "Unwrap":
            return self._a(args, 0)
        if name == "IsNil":
            return self._a(args, 0) + " === null"
        if name == "Sqrt":
            return "Math.sqrt(" + self._a(args, 0) + ")"
        if name == "Floor":
            return "Math.floor(" + self._a(args, 0) + ")"
        if name == "Ceil":
            return "Math.ceil(" + self._a(args, 0) + ")"
        if name == "IsNaN":
            return "Number.isNaN(" + self._a(args, 0) + ")"
        if name == "IsInf":
            return "!Number.isFinite(" + self._a(args, 0) + ")"
        # I/O
        if name == "WriteOut":
            return "process.stdout.write(" + self._a(args, 0) + ")"
        if name == "WriteErr":
            return "process.stderr.write(" + self._a(args, 0) + ")"
        if name == "WritelnOut":
            return "console.log(" + self._a(args, 0) + ")"
        if name == "WritelnErr":
            return "console.error(" + self._a(args, 0) + ")"
        if name == "ReadLine":
            return "readline()"
        if name == "ReadAll":
            return "input"
        if name == "ReadBytes":
            return 'fs.readFileSync("/dev/stdin")'
        if name == "ReadBytesN":
            return "process.stdin.read(" + self._a(args, 0) + ")"
        if name == "ReadFile":
            return "fs.readFileSync(" + self._a(args, 0) + ', "utf-8")'
        if name == "ReadFileBytes":
            return "fs.readFileSync(" + self._a(args, 0) + ")"
        if name == "WriteFile":
            return (
                "fs.writeFileSync(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            )
        if name == "Args":
            return "process.argv.slice(2)"
        if name == "GetEnv":
            return "process.env[" + self._a(args, 0) + "]"
        if name == "Exit":
            return "process.exit(" + self._a(args, 0) + ")"
        # Operator forms
        if name == "Pow":
            if self.strict_math and self._is_int_expr(args[0].value):
                return (
                    "checked_pow_i64("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ")"
                )
            return self._a(args, 0) + " ** " + self._a(args, 1)
        if name == "Contains":
            inner = args[0].value
            if self._is_map_type(inner):
                return self._a(args, 0) + ".has(" + self._a(args, 1) + ")"
            if self._is_set_type(inner):
                return self._a(args, 0) + ".has(" + self._a(args, 1) + ")"
            return self._a(args, 0) + ".includes(" + self._a(args, 1) + ")"
        if name == "Concat":
            if self._is_bytes_expr(args[0].value):
                return (
                    "Buffer.concat(["
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + "])"
                )
            left = self._maybe_paren(args[0].value, "+", True)
            right = self._maybe_paren(args[1].value, "+", False)
            return left + " + " + right
        if name == "Repeat":
            inner = args[0].value
            count = self._a(args, 1)
            if isinstance(inner, TListLit) and len(inner.elements) == 1:
                return (
                    "Array(" + count + ").fill(" + self._expr(inner.elements[0]) + ")"
                )
            return self._a(args, 0) + ".repeat(" + count + ")"
        if name == "Format":
            return self._format_call(args)
        if name == "Assert":
            cond = self._a(args, 0)
            if len(args) > 1:
                return (
                    "if (!("
                    + cond
                    + ")) { throw new Error("
                    + self._a(args, 1)
                    + "); }"
                )
            return "if (!(" + cond + ')) { throw new Error("assertion failed"); }'
        if name == "IsType":
            type_arg = args[1].value
            if isinstance(type_arg, TStringLit):
                type_name = type_arg.value
            else:
                type_name = self._expr(type_arg)
            prim = self._js_typeof(type_name)
            if prim is not None:
                return "typeof " + self._a(args, 0) + ' === "' + prim + '"'
            return self._a(args, 0) + " instanceof " + type_name
        if name in ("Bytes", "BytesFrom"):
            return "Buffer.from(" + self._a(args, 0) + ")"
        if name == "WrappingAdd":
            return "wrappingAdd(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "WrappingSub":
            return "wrappingSub(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "WrappingMul":
            return "wrappingMul(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        # Fallback
        arg_strs = self._join_args(args, ", ")
        return name + "(" + arg_strs + ")"

    def _is_int_list(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann == "list[int]"
        if isinstance(expr, TListLit) and expr.elements:
            return self._is_int_expr(expr.elements[0])
        if isinstance(expr, TVar):
            typ: TType | None = self.var_types.get(expr.name)
            if isinstance(typ, TListType) and isinstance(typ.element, TPrimitive):
                return typ.element.kind == "int"
        return False

    def _sorted_with_key(self, collection: TExpr, key_fn: TFnLit) -> str:
        params = [p for p in key_fn.params if p.typ is not None]
        if len(params) == 1:
            pname = _restore_name(params[0].name, params[0].annotations)
            first = key_fn.body[0] if key_fn.body else None
            if isinstance(first, (TReturnStmt,)) and first.value is not None:
                key_expr = self._expr(first.value)
                key_a = key_expr.replace(pname, "a", 1)
                key_b = key_expr.replace(pname, "b", 1)
                return (
                    "[..."
                    + self._expr(collection)
                    + "].sort((a, b) => "
                    + key_a
                    + " - "
                    + key_b
                    + ")"
                )
            if isinstance(first, TExprStmt):
                key_expr = self._expr(first.expr)
                key_a = key_expr.replace(pname, "a", 1)
                key_b = key_expr.replace(pname, "b", 1)
                return (
                    "[..."
                    + self._expr(collection)
                    + "].sort((a, b) => "
                    + key_a
                    + " - "
                    + key_b
                    + ")"
                )
        return "[..." + self._expr(collection) + "].sort()"

    def _min_max_key_cmp(self, key_fn: TFnLit, cmp_op: str) -> str:
        params = [p for p in key_fn.params if p.typ is not None]
        if len(params) == 1:
            pname = _restore_name(params[0].name, params[0].annotations)
            first = key_fn.body[0] if key_fn.body else None
            if isinstance(first, (TReturnStmt,)) and first.value is not None:
                key_expr = self._expr(first.value)
            elif isinstance(first, TExprStmt):
                key_expr = self._expr(first.expr)
            else:
                return "a"
            key_a = key_expr.replace(pname, "a", 1)
            key_b = key_expr.replace(pname, "b", 1)
            return key_a + " " + cmp_op + " " + key_b + " ? a : b"
        return "a"

    def _join_args(self, args: list[TArg], sep: str) -> str:
        parts: list[str] = []
        for a in args:
            parts.append(self._expr(a.value))
        return sep.join(parts)

    def _join_exprs(self, exprs: list[TExpr], sep: str) -> str:
        parts: list[str] = []
        for e in exprs:
            parts.append(self._expr(e))
        return sep.join(parts)

    def _format_int(self, args: list[TArg]) -> str:
        n = self._a(args, 0)
        base = self._a(args, 1)
        return "(" + n + ").toString(" + base + ")"

    def _format_call(self, args: list[TArg]) -> str:
        template_expr = args[0].value
        if not isinstance(template_expr, TStringLit):
            arg_strs = self._join_args(args, ", ")
            return "Format(" + arg_strs + ")"
        template = template_expr.value
        fmt_args = args[1:]
        markers: dict[str, int] = {}
        result = template
        for i, _arg in enumerate(fmt_args):
            marker = "\x00PH" + str(i) + "\x00"
            markers[marker] = i
            result = result.replace("{}", marker, 1)
        for mk, idx in markers.items():
            result = result.replace(mk, "${" + self._expr(fmt_args[idx].value) + "}")
        return "`" + result + "`"


# ============================================================
# PUBLIC API
# ============================================================


def emit_javascript(module: TModule) -> str:
    struct_names: set[str] = set(BUILTIN_STRUCTS.keys())
    struct_fields: dict[str, list[str]] = {}
    for decl in module.decls:
        match decl:
            case TStructDecl():
                struct_names.add(decl.name)
                fnames: list[str] = []
                for f in decl.fields:
                    fnames.append(_safe_name(f.name))
                struct_fields[decl.name] = fnames
            case TInterfaceDecl():
                struct_names.add(decl.name)
                if decl.fields:
                    ifnames: list[str] = []
                    for f in decl.fields:
                        ifnames.append(_safe_name(f.name))
                    struct_fields[decl.name] = ifnames
    emitter = _JavaScriptEmitter(
        struct_names, struct_fields, module.strict_math, module.strict_tostring
    )
    emitter.emit_module(module)
    return emitter.output()
