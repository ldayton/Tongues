"""Ruby backend: Taytsh AST → Ruby source code."""

from __future__ import annotations
from typing import assert_never

from .ordering import order_decls
from .util import (
    STRICT_INT_BINARY,
    STRICT_INT_COMPOUND,
    Emitter,
    _check_float_expr,
    _check_float_list,
    _check_int_expr,
    _emit_output,
    collect_builtin_calls,
    escape_string,
    to_snake,
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
    TVar,
    TWhileStmt,
)
from ..taytsh.check import (
    BUILTIN_NAMES,
    BUILTIN_STRUCTS,
)

# ============================================================
# RUBY RESERVED WORDS AND BUILTINS
# ============================================================

_RUBY_RESERVED = frozenset(
    {
        "BEGIN",
        "END",
        "__ENCODING__",
        "__END__",
        "__FILE__",
        "__LINE__",
        "alias",
        "and",
        "begin",
        "break",
        "case",
        "class",
        "def",
        "defined?",
        "do",
        "else",
        "elsif",
        "end",
        "ensure",
        "false",
        "for",
        "if",
        "in",
        "module",
        "next",
        "nil",
        "not",
        "or",
        "redo",
        "rescue",
        "retry",
        "return",
        "self",
        "super",
        "then",
        "true",
        "undef",
        "unless",
        "until",
        "when",
        "while",
        "yield",
    }
)

# Additional Kernel/Object methods that collide with user-defined free function
# definitions but not local variables.  Checked only in _safe_fn_name.
_RUBY_FN_BUILTINS = frozenset(
    {
        # Kernel methods (callable without receiver)
        "abort",
        "at_exit",
        "autoload",
        "binding",
        "block_given?",
        "callcc",
        "caller",
        "caller_locations",
        "catch",
        "chomp",
        "chop",
        "clone",
        "eval",
        "exec",
        "exit",
        "fail",
        "fork",
        "format",
        "freeze",
        "gets",
        "global_variables",
        "gsub",
        "hash",
        "hex",
        "lambda",
        "last",
        "load",
        "local_variables",
        "loop",
        "open",
        "p",
        "pp",
        "print",
        "printf",
        "proc",
        "putc",
        "puts",
        "raise",
        "rand",
        "readline",
        "readlines",
        "require",
        "require_relative",
        "select",
        "set_trace_func",
        "sleep",
        "spawn",
        "sprintf",
        "srand",
        "sub",
        "syscall",
        "system",
        "test",
        "throw",
        "trace_var",
        "trap",
        "untrace_var",
        "warn",
        # Object methods
        "display",
        "dup",
        "enum_for",
        "extend",
        "inspect",
        "method",
        "methods",
        "object_id",
        "send",
        "tap",
        "to_enum",
        # Other
        "__callee__",
        "__dir__",
        "__method__",
    }
)

_RUBY_BUILTINS = frozenset(
    {
        "Array",
        "BasicObject",
        "Binding",
        "Class",
        "Comparable",
        "Complex",
        "Data",
        "Dir",
        "Encoding",
        "Enumerable",
        "Enumerator",
        "Exception",
        "FalseClass",
        "Fiber",
        "File",
        "Float",
        "Hash",
        "Integer",
        "IO",
        "Kernel",
        "Marshal",
        "MatchData",
        "Method",
        "Module",
        "NilClass",
        "Numeric",
        "Object",
        "Proc",
        "Process",
        "Queue",
        "Random",
        "Range",
        "Rational",
        "Regexp",
        "Set",
        "Signal",
        "String",
        "Struct",
        "Symbol",
        "Thread",
        "Time",
        "TracePoint",
        "TrueClass",
        "UnboundMethod",
    }
)

_EXCEPTION_MAP: dict[str, str] = {
    "Exception": "StandardError",
    "ValueError": "ArgumentError",
    "KeyError": "KeyError",
    "IndexError": "IndexError",
    "ZeroDivisionError": "ZeroDivisionError",
    "AssertError": "RuntimeError",
    "NilError": "RuntimeError",
    "IOError": "IOError",
    "BaseException": "StandardError",
    "TypeError": "TypeError",
    "NotImplementedError": "NotImplementedError",
    "RuntimeError": "RuntimeError",
}


_METHOD_NAME_MAP: dict[str, str] = {
    "__repr__": "to_s",
    "__str__": "to_s",
    "__eq__": "==",
    "__ne__": "!=",
    "__lt__": "<",
    "__le__": "<=",
    "__gt__": ">",
    "__ge__": ">=",
    "__hash__": "hash",
    "__len__": "length",
    "__contains__": "include?",
    "__iter__": "each",
}


def _safe_name(name: str) -> str:
    """Convert name to safe Ruby identifier, preserving leading underscores."""
    mapped = _METHOD_NAME_MAP.get(name)
    if mapped is not None:
        return mapped
    prefix = ""
    if name.startswith("_"):
        prefix = "_"
    name = to_snake(name)
    if not name:
        return "_"
    result = prefix + name
    if result in _RUBY_RESERVED:
        return result + "_"
    return result


def _safe_fn_name(name: str) -> str:
    """Like _safe_name but also checks Kernel/Object methods for free functions."""
    safe = _safe_name(name)
    if safe in _RUBY_FN_BUILTINS:
        return safe + "_"
    return safe


def _safe_module_name(name: str) -> str:
    """Like _safe_name but strips leading underscores for module-level vars."""
    name = to_snake(name)
    if not name:
        return "_"
    if name in _RUBY_RESERVED:
        return name + "_"
    return name


def _safe_local_name(name: str) -> str:
    """Like _safe_name but ensures the result starts lowercase (for local vars)."""
    name = _safe_name(name)
    if name and name[0].isupper():
        return name.lower()
    return name


def _restore_local_name(name: str, annotations: Ann) -> str:
    """Restore original name, then make safe for local variable context."""
    key = "name.original." + name
    if key in annotations:
        return _safe_local_name(annotations[key])
    return _safe_local_name(name)


def _restore_name(name: str, annotations: Ann) -> str:
    """Restore original Python name from annotation, then apply target safety."""
    key = "name.original." + name
    if key in annotations:
        return _safe_name(annotations[key])
    return _safe_name(name)


def _restore_module_name(name: str, annotations: Ann) -> str:
    """Restore name for module-level variables (no leading underscore)."""
    key = "name.original." + name
    if key in annotations:
        return _safe_module_name(annotations[key])
    return _safe_module_name(name)


def _restore_fn_name(name: str, annotations: Ann) -> str:
    """Restore name for free function contexts (checks Kernel/Object builtins)."""
    key = "name.original." + name
    if key in annotations:
        return _safe_fn_name(annotations[key])
    return _safe_fn_name(name)


_TYPE_NAME_MAP: dict[str, str] = {
    "dict": "Hash",
    "Dict": "Hash",
    "list": "Array",
    "List": "Array",
    "str": "String",
    "Str": "String",
    "int": "Integer",
    "Int": "Integer",
    "bool": "TrueClass",
    "Bool": "TrueClass",
    "tuple": "Array",
    "Tuple": "Array",
    "set": "Set",
    "Set": "Set",
    "bytes": "Array",
}


def _safe_type_name(name: str) -> str:
    """Ensure name is a valid Ruby constant (starts with uppercase)."""
    mapped = _TYPE_NAME_MAP.get(name)
    if mapped is not None:
        return mapped
    if name in _RUBY_BUILTINS:
        return name + "_"
    if name and name[0] == "_":
        return "X" + name[1:]
    if name and name[0].islower():
        return name[0].upper() + name[1:]
    return name


def _escape_regex_charclass(value: str) -> str:
    """Escape characters for use inside a Ruby regex character class."""
    out: list[str] = []
    i: int = 0
    while i < len(value):
        c: str = value[i : i + 1]
        if c in "\\]^-":
            out.append("\\" + c)
        elif c == "\n":
            out.append("\\n")
        elif c == "\t":
            out.append("\\t")
        elif c == "\r":
            out.append("\\r")
        elif ord(c) < 32 or ord(c) > 126:
            h: str = hex(ord(c))[2:]
            if len(h) == 1:
                h = "0" + h
            out.append("\\x" + h)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _escape_ruby_string(value: str) -> str:
    result = escape_string(value)
    out: list[str] = []
    i = 0
    while i < len(result):
        c = result[i]
        if c == "#" and i + 1 < len(result) and result[i + 1] in "{$@":
            out.append("\\#")
        else:
            out.append(c)
        i += 1
    return "".join(out)


# ============================================================
# OPERATOR MAPS
# ============================================================

_PRECEDENCE: dict[str, int] = {
    "or": 1,
    "and": 2,
    "||": 3,
    "&&": 4,
    "==": 5,
    "!=": 5,
    "<=>": 5,
    "<": 6,
    ">": 6,
    "<=": 6,
    ">=": 6,
    "|": 7,
    "^": 7,
    "&": 8,
    "<<": 9,
    ">>": 9,
    "+": 10,
    "-": 10,
    "*": 11,
    "/": 11,
    "//": 11,
    "%": 11,
    "**": 12,
}

_CMP_OPS = frozenset(["==", "!=", "<", ">", "<=", ">="])


def _needs_parens(child_op: str, parent_op: str, is_left: bool) -> bool:
    child_prec = _PRECEDENCE.get(child_op, 0)
    parent_prec = _PRECEDENCE.get(parent_op, 0)
    if child_prec < parent_prec:
        return True
    if child_prec == parent_prec and not is_left:
        return True
    if parent_op in ("==", "!=") and child_op in _CMP_OPS:
        return True
    return False


# ============================================================
# IMPORT / NEEDS SCANNING
# ============================================================


def _scan_decl_needs(decl: TDecl) -> tuple[bool, bool]:
    needs_set = False
    needs_range = False
    stmts: list[TStmt] = []
    if isinstance(decl, TFnDecl):
        stmts = decl.body
    elif isinstance(decl, TStructDecl):
        for m in decl.methods:
            has_set, has_range = _scan_decl_needs(m)
            if has_set:
                needs_set = True
            if has_range:
                needs_range = True
        for fld in decl.fields:
            if isinstance(fld.typ, TSetType):
                needs_set = True
        return needs_set, needs_range
    for name in collect_builtin_calls(stmts):
        if name in ("Set", "Add", "Remove"):
            needs_set = True
    _scan_stmts_for_needs(stmts, [needs_set, needs_range])
    return needs_set, needs_range


def _scan_stmts_for_needs(stmts: list[TStmt], flags: list[bool]) -> None:
    for stmt in stmts:
        _scan_stmt_for_needs(stmt, flags)


def _scan_stmt_for_needs(stmt: TStmt, flags: list[bool]) -> None:
    match stmt:
        case TLetStmt():
            if stmt.value is not None:
                _scan_expr_for_needs(stmt.value, flags)
        case TAssignStmt():
            _scan_expr_for_needs(stmt.value, flags)
        case TExprStmt():
            _scan_expr_for_needs(stmt.expr, flags)
        case TReturnStmt():
            if stmt.value is not None:
                _scan_expr_for_needs(stmt.value, flags)
        case TIfStmt():
            _scan_stmts_for_needs(stmt.then_body, flags)
            if stmt.else_body is not None and stmt.else_body:
                _scan_stmts_for_needs(stmt.else_body, flags)
        case TWhileStmt():
            _scan_stmts_for_needs(stmt.body, flags)
        case TForStmt():
            _scan_stmts_for_needs(stmt.body, flags)
        case TTryStmt():
            _scan_stmts_for_needs(stmt.body, flags)
            for catch in stmt.catches:
                _scan_stmts_for_needs(catch.body, flags)
        case TMatchStmt():
            for case in stmt.cases:
                _scan_stmts_for_needs(case.body, flags)
            if stmt.default:
                _scan_stmts_for_needs(stmt.default.body, flags)
        case (
            TOpAssignStmt()
            | TTupleAssignStmt()
            | TThrowStmt()
            | TBreakStmt()
            | TContinueStmt()
        ):
            pass
        case _:
            assert_never(stmt)


def _scan_expr_for_needs(expr: TExpr, flags: list[bool]) -> None:
    match expr:
        case TSetLit():
            flags[0] = True
        case TCall():
            if isinstance(expr.func, TVar) and expr.func.name == "Set":
                flags[0] = True
            for a in expr.args:
                _scan_expr_for_needs(a.value, flags)
        case TListLit() | TTupleLit():
            for e in expr.elements:
                _scan_expr_for_needs(e, flags)
        case TMapLit():
            for k, v in expr.entries:
                _scan_expr_for_needs(k, flags)
                _scan_expr_for_needs(v, flags)
        case TBinaryOp():
            _scan_expr_for_needs(expr.left, flags)
            _scan_expr_for_needs(expr.right, flags)
        case TUnaryOp():
            _scan_expr_for_needs(expr.operand, flags)
        case TTernary():
            _scan_expr_for_needs(expr.cond, flags)
            _scan_expr_for_needs(expr.then_expr, flags)
            _scan_expr_for_needs(expr.else_expr, flags)


# ============================================================
# STRICT MATH
# ============================================================


# ============================================================
# EMITTER
# ============================================================


class _RubyEmitter(Emitter):
    def __init__(
        self,
        struct_names: set[str],
        fn_names: set[str],
        struct_fields: dict[str, list[str]],
        field_types: dict[str, dict[str, TType]],
        enum_names: set[str],
        strict_math: bool = False,
        strict_tostring: bool = False,
    ) -> None:
        self.struct_names = struct_names
        self.fn_names = fn_names
        self.struct_fields = struct_fields
        self.field_types = field_types
        self.enum_names = enum_names
        self.strict_math = strict_math
        self.strict_tostring = strict_tostring
        self.indent: int = 0
        self.lines: list[str] = []
        self.self_name: str | None = None
        self.var_types: dict[str, TType] = {}
        self.var_annotations: dict[str, dict[str, str]] = {}
        self._needs_set: bool = False
        self.in_fn: bool = False
        self.local_names: dict[str, str] = {}
        self._needs_range_helper: bool = False
        self._needs_float_repr: bool = False
        self._needs_parse_float: bool = False

    def _line(self, text: str = "") -> None:
        if text:
            self.lines.append("  " * self.indent + text)
        else:
            self.lines.append("")

    def output(self) -> str:
        return _emit_output(self.lines)

    def _is_int_expr(self, expr: TExpr) -> bool:
        return _check_int_expr(expr, self.var_types)

    def _is_float_expr(self, expr: TExpr) -> bool:
        return _check_float_expr(expr, self.var_types)

    def _is_float_list(self, expr: TExpr) -> bool:
        return _check_float_list(expr, self.var_types)

    def _decl_name(self, name: str, annotations: Ann) -> str:
        """Declare a local variable, lowercasing inside function bodies."""
        if self.in_fn:
            safe = _restore_local_name(name, annotations)
            self.local_names[name] = safe
            return safe
        return _restore_module_name(name, annotations)

    def _ref_name(self, name: str, annotations: Ann) -> str:
        """Reference a variable — use local form if declared locally."""
        local = self.local_names.get(name)
        if local is not None:
            return local
        if name in self.fn_names:
            return _restore_fn_name(name, annotations)
        return _restore_module_name(name, annotations)

    # ── Module ────────────────────────────────────────────────

    def emit_module(self, module: TModule) -> None:
        self._line("# frozen_string_literal: true")
        self._line()
        import_insert_pos = len(self.lines)
        for decl in module.decls:
            if isinstance(decl, TLetStmt) and decl.typ is not None:
                self.var_types[decl.name] = decl.typ
        # Collect builtins and emit error classes as needed
        all_builtins: set[str] = set()
        for decl in module.decls:
            if isinstance(decl, TFnDecl):
                all_builtins |= collect_builtin_calls(decl.body)
            elif isinstance(decl, TStructDecl):
                for m in decl.methods:
                    all_builtins |= collect_builtin_calls(m.body)
            elif isinstance(decl, TStmt):
                all_builtins |= collect_builtin_calls([decl])
        if "Decode" in all_builtins or "ReadAll" in all_builtins:
            self._line(
                "class UnicodeDecodeError < ArgumentError;"
                "attr_reader :message; "
                'def initialize(message = ""); @message = message; super(message); end; '
                "end"
            )
            self._line()
        need_blank = False
        for decl in order_decls(module.decls):
            if isinstance(decl, TInterfaceDecl):
                if need_blank:
                    self._line()
                self._line("class " + _safe_type_name(decl.name))
                if decl.fields:
                    self.indent += 1
                    attrs = ", ".join(":" + _safe_name(f.name) for f in decl.fields)
                    self._line("attr_accessor " + attrs)
                    self._line()
                    self._emit_initialize(decl.fields, False)
                    self.indent -= 1
                self._line("end")
                need_blank = True
                continue
            if need_blank:
                self._line()
            match decl:
                case TEnumDecl():
                    self._emit_enum(decl)
                case TStructDecl():
                    self._emit_struct(decl)
                case TLetStmt():
                    self._emit_let(decl)
                case TFnDecl():
                    self._emit_fn(decl)
                case TInterfaceDecl():
                    pass  # handled above via isinstance check
            need_blank = True
        # Insert require 'set' at top if needed
        if self._needs_set:
            self.lines.insert(import_insert_pos, "require 'set'")
            self.lines.insert(import_insert_pos + 1, "")
            import_insert_pos += 2
        if self._needs_range_helper:
            helper = "def _range(start, stop = nil, step = 1); stop.nil? ? (0...start).step(step).to_a : (step > 0 ? (start...stop).step(step).to_a : (stop + 1..start).step(-step).to_a.reverse); end"
            self.lines.insert(import_insert_pos, helper)
            self.lines.insert(import_insert_pos + 1, "")
            import_insert_pos += 2
        if self.strict_tostring:
            helper = (
                "def _py_float_repr(f); return f.to_s if f.nan? || f.infinite?;"
                " b = nil; (1..17).each { |d| s = '%.*g' % [d, f];"
                " if s.to_f == f; b = s; break; end };"
                ' b = "%.17g" % f if b.nil?;'
                " if b.include?('e') || b.include?('E');"
                " a = f.abs; if a != 0.0; e = Math.log10(a).floor;"
                " if e >= 0 && e <= 15;"
                " (1..20).each { |d| s = '%.*f' % [d, f];"
                ' if s.to_f == f; s = s.sub(/0+\\z/, "");'
                ' s = s + "0" if s.end_with?(".");'
                " b = s; break; end }; end; end; end;"
                ' b = b + ".0" if !b.include?(".") && !b.include?("e")'
                ' && !b.include?("E"); b; end'
            )
            self.lines.insert(import_insert_pos, helper)
            self.lines.insert(import_insert_pos + 1, "")
        if self._needs_parse_float:
            helper = (
                "def _do_parse_float(s); "
                "return Float::INFINITY if s == 'inf' || s == 'Infinity'; "
                "return -Float::INFINITY if s == '-inf' || s == '-Infinity'; "
                "return Float::NAN if s == 'nan' || s == 'NaN'; "
                "s.to_f; end"
            )
            self.lines.insert(import_insert_pos, helper)
            self.lines.insert(import_insert_pos + 1, "")

    # ── Enum ──────────────────────────────────────────────────

    def _emit_enum(self, decl: TEnumDecl) -> None:
        self._line("module " + decl.name)
        self.indent += 1
        for i, variant in enumerate(decl.variants):
            self._line(variant + " = " + str(i))
        self.indent -= 1
        self._line("end")

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
        if decl.parent is not None:
            base = _EXCEPTION_MAP.get(decl.parent, _safe_type_name(decl.parent))
        else:
            base = _EXCEPTION_MAP.get(decl.name, "StandardError")
        self._line("class " + _safe_type_name(decl.name) + " < " + base)
        self.indent += 1
        if decl.fields:
            attrs = ", ".join(":" + _safe_name(f.name) for f in decl.fields)
            self._line("attr_accessor " + attrs)
            self._line()
            self._emit_initialize(decl.fields, True)
        for i, method in enumerate(decl.methods):
            if i > 0 or decl.fields:
                self._line()
            self._emit_method(method)
        self.indent -= 1
        self._line("end")

    def _emit_data_struct(self, decl: TStructDecl) -> None:
        if decl.parent is not None:
            self._line(
                "class "
                + _safe_type_name(decl.name)
                + " < "
                + _safe_type_name(decl.parent)
            )
        else:
            self._line("class " + _safe_type_name(decl.name))
        self.indent += 1
        if not decl.fields and not decl.methods:
            pass
        if decl.fields:
            attrs = ", ".join(":" + _safe_name(f.name) for f in decl.fields)
            self._line("attr_accessor " + attrs)
            self._line()
            self._emit_initialize(decl.fields, False)
        for i, method in enumerate(decl.methods):
            if i > 0 or decl.fields:
                self._line()
            self._emit_method(method)
        self.indent -= 1
        self._line("end")

    def _emit_initialize(self, fields: list[TFieldDecl], is_error: bool) -> None:
        param_fields = [f for f in fields if not f.body_computed]
        body_fields = [f for f in fields if f.body_computed]
        params: list[str] = []
        for f in param_fields:
            name = _safe_name(f.name)
            if f.self_ref:
                params.append(name + ": nil")
            else:
                default = self._zero_value(f.typ)
                params.append(name + ": " + default)
        self._line("def initialize(" + ", ".join(params) + ")")
        self.indent += 1
        if is_error:
            msg_field: TFieldDecl | None = None
            for f in param_fields:
                if f.name in ("message", "msg"):
                    msg_field = f
                    break
            if msg_field is not None:
                self._line("super(" + _safe_name(msg_field.name) + ")")
            else:
                self._line("super()")
        for f in param_fields:
            name = _safe_name(f.name)
            if f.self_ref and isinstance(f.typ, TIdentType):
                self._line(
                    "@"
                    + name
                    + " = "
                    + name
                    + " || "
                    + _safe_type_name(f.typ.name)
                    + ".new(self)"
                )
            else:
                self._line("@" + name + " = " + name)
        with self._with_self("this"):
            for f in body_fields:
                name = _safe_name(f.name)
                if f.default_expr is not None:
                    self._line("@" + name + " = " + self._expr(f.default_expr))
        self.indent -= 1
        self._line("end")

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
                return '""'
        if isinstance(typ, TListType):
            return "[]"
        if isinstance(typ, TMapType):
            return "{}"
        if isinstance(typ, TSetType):
            self._needs_set = True
            return "Set.new"
        if isinstance(typ, TTupleType):
            return "[]"
        if isinstance(typ, TIdentType):
            return _safe_type_name(typ.name) + ".new"
        return "nil"

    # ── Function / Method ─────────────────────────────────────

    def _emit_fn(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        old_var_annotations = self.var_annotations.copy()
        old_local_names = self.local_names
        old_in_fn = self.in_fn
        self.local_names = {}
        self.in_fn = True
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
            self._capture_var_annotations(p.name, p.annotations)
        params = self._params(decl.params, with_self=False)
        self._line("def " + _safe_fn_name(decl.name) + "(" + params + ")")
        self.indent += 1
        if not decl.body:
            self._line("nil")
        self._emit_stmts(decl.body)
        self.in_fn = old_in_fn
        self.indent -= 1
        self._line("end")
        self.var_types = old_var_types
        self.var_annotations = old_var_annotations
        self.local_names = old_local_names

    def _emit_method(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        old_var_annotations = self.var_annotations.copy()
        old_local_names = self.local_names
        old_in_fn = self.in_fn
        self.local_names = {}
        self.in_fn = True
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
            self._capture_var_annotations(p.name, p.annotations)
        params = self._params(decl.params, with_self=True)
        self._line("def " + _safe_name(decl.name) + "(" + params + ")")
        self.indent += 1
        old_self = self.self_name
        if decl.params and decl.params[0].typ is None:
            self.self_name = decl.params[0].name
        if not decl.body:
            self._line("nil")
        self._emit_stmts(decl.body)
        self.self_name = old_self
        self.in_fn = old_in_fn
        self.indent -= 1
        self._line("end")
        self.var_types = old_var_types
        self.var_annotations = old_var_annotations
        self.local_names = old_local_names

    def _params(self, params: list[TParam], with_self: bool) -> str:
        parts: list[str] = []
        for p in params:
            if p.typ is None:
                continue
            name = self._decl_name(p.name, p.annotations)
            if p.has_default:
                parts.append(name + " = " + self._zero_value(p.typ))
            else:
                parts.append(name)
        return ", ".join(parts)

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
        acc = self._decl_name(let_stmt.name, let_stmt.annotations)
        binding = for_stmt.binding
        binders = ", ".join(self._decl_name(b, for_stmt.annotations) for b in binding)
        if isinstance(for_stmt.iterable, TRange):
            iterable = self._ruby_range(for_stmt.iterable)
        else:
            iterable = self._expr(for_stmt.iterable)
        iter_is_map = self._is_map_for(for_stmt)
        if iter_is_map and len(binding) == 2:
            pass  # Ruby hash.each gives |k, v|
        elif self._is_enumerate_for(for_stmt):
            iterable += ".each_with_index"
            binders = (
                self._decl_name(binding[1], for_stmt.annotations)
                + ", "
                + self._decl_name(binding[0], for_stmt.annotations)
            )
        body = for_stmt.body
        if prov == "list_comprehension":
            if len(body) == 1 and isinstance(body[0], TExprStmt):
                call = body[0].expr
                if isinstance(call, TCall) and self._is_append_to(call, let_stmt.name):
                    val = self._expr(call.args[1].value)
                    return (
                        acc
                        + " = "
                        + iterable
                        + ".map { |"
                        + binders
                        + "| "
                        + val
                        + " }"
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
                                acc
                                + " = "
                                + iterable
                                + ".select { |"
                                + binders
                                + "| "
                                + guard
                                + " }"
                                + ".map { |"
                                + binders
                                + "| "
                                + val
                                + " }"
                            )
        elif prov == "dict_comprehension":
            if len(body) == 1 and isinstance(body[0], TAssignStmt):
                target = body[0].target
                if isinstance(target, TIndex):
                    key = self._expr(target.index)
                    val = self._expr(body[0].value)
                    return (
                        acc
                        + " = "
                        + iterable
                        + ".map { |"
                        + binders
                        + "| ["
                        + key
                        + ", "
                        + val
                        + "] }.to_h"
                    )
        elif prov == "set_comprehension":
            self._needs_set = True
            if len(body) == 1 and isinstance(body[0], TExprStmt):
                call = body[0].expr
                if isinstance(call, TCall) and self._is_add_to(call, let_stmt.name):
                    val = self._expr(call.args[1].value)
                    if val == binders:
                        return acc + " = Set.new(" + iterable + ")"
                    return (
                        acc
                        + " = Set.new("
                        + iterable
                        + ".map { |"
                        + binders
                        + "| "
                        + val
                        + " })"
                    )
        return None

    def _try_step_slice(self, let_stmt: TLetStmt, for_stmt: TForStmt) -> str | None:
        """Reconstruct xs.each_slice(step).map(&:first) from a step_slice for-loop."""
        if not isinstance(for_stmt.iterable, TRange):
            return None
        range_args = for_stmt.iterable.args
        if len(range_args) != 3:
            return None
        body = for_stmt.body
        if len(body) != 1:
            return None
        is_string, src_obj = self._step_slice_source(body[0], let_stmt.name)
        if src_obj is None:
            return None
        src = self._expr(src_obj)
        acc = self._decl_name(let_stmt.name, let_stmt.annotations)
        start_expr = range_args[0]
        step_expr = range_args[2]
        step_s = self._expr(step_expr)
        if is_string:
            base = src + ".chars"
            suffix = ".join"
        else:
            base = src
            suffix = ""
        if isinstance(start_expr, TIntLit) and start_expr.value == 0:
            return (
                acc + " = " + base + ".each_slice(" + step_s + ").map(&:first)" + suffix
            )
        start_s = self._expr(start_expr)
        return (
            acc
            + " = "
            + base
            + "["
            + start_s
            + "..].each_slice("
            + step_s
            + ").map(&:first)"
            + suffix
        )

    def _step_slice_source(
        self, stmt: TStmt, acc_name: str
    ) -> tuple[bool, TExpr | None]:
        """Extract (is_string, source_obj) from a step_slice loop body."""
        # List: ExprStmt(Append(acc, obj[__i]))
        if isinstance(stmt, TExprStmt):
            call = stmt.expr
            if isinstance(call, TCall) and self._is_append_to(call, acc_name):
                elem = call.args[1].value
                if isinstance(elem, TIndex):
                    return False, elem.obj
        # String: acc = Concat(acc, ToString(obj[__i]))
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

    def _emit_any_all(
        self,
        stmts: list[TStmt],
        i: int,
        let_stmt: TLetStmt,
        for_stmt: TForStmt,
        prov: str,
    ) -> int:
        """Try to emit .any?/.all?. Returns number of statements to skip, or 0."""
        aa = self._try_any_all(let_stmt, for_stmt, prov)
        if aa:
            lhs, rhs = aa
            folded = self._fold_temp_assign(stmts, i, let_stmt.name, rhs)
            if folded is not None:
                self._line(folded)
                return 3
            self._line(lhs + " = " + rhs)
            return 2
        return 0

    def _try_any_all(
        self, let_stmt: TLetStmt, for_stmt: TForStmt, prov: str
    ) -> tuple[str, str] | None:
        """Try to reconstruct .any?/.all? from a let + for pair. Returns (lhs, rhs)."""
        acc = self._decl_name(let_stmt.name, let_stmt.annotations)
        binding = for_stmt.binding
        binders = ", ".join(self._decl_name(b, for_stmt.annotations) for b in binding)
        if isinstance(for_stmt.iterable, TRange):
            iterable = self._ruby_range(for_stmt.iterable)
        else:
            iterable = self._expr(for_stmt.iterable)
        iter_is_map = self._is_map_for(for_stmt)
        if iter_is_map and len(binding) == 2:
            pass
        elif self._is_enumerate_for(for_stmt):
            iterable += ".each_with_index"
            binders = (
                self._decl_name(binding[1], for_stmt.annotations)
                + ", "
                + self._decl_name(binding[0], for_stmt.annotations)
            )
        method = ".any?" if prov == "any_call" else ".all?"
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
                acc,
                iterable + method + " { |" + binders + "| " + cond_s + " }",
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
                    acc,
                    iterable
                    + method
                    + " { |"
                    + binders
                    + "| "
                    + filter_s
                    + " && "
                    + cond_s
                    + " }",
                )
        return None

    def _strip_not(self, expr: TExpr) -> TExpr:
        """Strip a leading ! from a unary-not expression."""
        if isinstance(expr, TUnaryOp) and expr.op == "!":
            return expr.operand
        return expr

    def _fold_temp_assign(
        self, stmts: list[TStmt], i: int, temp_name: str, rhs: str
    ) -> str | None:
        """If stmts[i+2] is `real_name = temp_name`, fold into `real_name = rhs`."""
        if i + 2 >= len(stmts):
            return None
        third = stmts[i + 2]
        if isinstance(third, TLetStmt) and isinstance(third.value, TVar):
            if third.value.name == temp_name:
                real = self._decl_name(third.name, third.annotations)
                return real + " = " + rhs
        if isinstance(third, TAssignStmt) and isinstance(third.value, TVar):
            if third.value.name == temp_name and isinstance(third.target, TVar):
                real = self._decl_name(third.target.name, third.target.annotations)
                return real + " = " + rhs
        return None

    def _emit_stmt(self, stmt: TStmt) -> None:
        match stmt:
            case TLetStmt():
                self._emit_let(stmt)
            case TAssignStmt():
                self._line(self._expr(stmt.target) + " = " + self._expr(stmt.value))
            case TTupleAssignStmt():
                self._emit_tuple_assign(stmt)
            case TOpAssignStmt():
                if (
                    self.strict_math
                    and stmt.op in STRICT_INT_COMPOUND
                    and self._is_int_expr(stmt.target)
                ):
                    fn = STRICT_INT_COMPOUND[stmt.op]
                    tgt = self._expr(stmt.target)
                    self._line(
                        tgt
                        + " = "
                        + fn
                        + "("
                        + tgt
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
                    self._line("return " + self._expr(stmt.value))
                else:
                    self._line("return")
            case TThrowStmt():
                self._line("raise " + self._expr(stmt.expr))
            case TBreakStmt():
                self._line("break")
            case TContinueStmt():
                self._line("next")
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
            case _:
                assert_never(stmt)

    def _emit_let(self, stmt: TLetStmt) -> None:
        safe = self._decl_name(stmt.name, stmt.annotations)
        self.var_types[stmt.name] = stmt.typ
        self._capture_var_annotations(stmt.name, stmt.annotations)
        unused = stmt.annotations.get("liveness.initial_value_unused") == "true"
        if stmt.value is not None and not unused:
            self._line(safe + " = " + self._expr(stmt.value))
        else:
            self._line(safe + " = nil")

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
        self._line(", ".join(parts) + " = " + self._expr(stmt.value))

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
            self._line(q_target + " = (" + a + ".to_f / " + b + ").truncate")
        else:
            r_target = self._expr(stmt.targets[1])
            self._line(q_target + " = (" + a + ".to_f / " + b + ").truncate")
            self._line(r_target + " = " + a + ".remainder(" + b + ")")

    def _emit_expr_stmt(self, stmt: TExprStmt) -> None:
        expr = stmt.expr
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            name = expr.func.name
            if name == "Assert":
                args = expr.args
                cond = self._expr(args[0].value)
                if len(args) > 1:
                    msg = self._expr(args[1].value)
                    self._line("raise " + msg + " unless " + cond)
                else:
                    self._line('raise "assertion failed" unless ' + cond)
                return
            if name == "Delete":
                args = expr.args
                self._line(
                    self._expr(args[0].value)
                    + ".delete("
                    + self._expr(args[1].value)
                    + ")"
                )
                return
        self._line(self._expr(expr))

    def _emit_if(self, stmt: TIfStmt) -> None:
        prov = stmt.annotations.get("provenance", "")
        if prov == "truthiness":
            cond_str = self._truthiness(stmt.cond)
            if cond_str is not None:
                self._line("if " + cond_str)
                self.indent += 1
                if not stmt.then_body:
                    self._line("nil")
                self._emit_stmts(stmt.then_body)
                self.indent -= 1
                self._emit_else_body(stmt.else_body)
                self._line("end")
                return
        if prov == "negated_condition":
            inner = self._negate_inner(stmt.cond)
            if inner is not None:
                self._line("unless " + inner)
                self.indent += 1
                if not stmt.then_body:
                    self._line("nil")
                self._emit_stmts(stmt.then_body)
                self.indent -= 1
                self._emit_else_body(stmt.else_body)
                self._line("end")
                return
        self._line("if " + self._expr(stmt.cond))
        self.indent += 1
        if not stmt.then_body:
            self._line("nil")
        self._emit_stmts(stmt.then_body)
        self.indent -= 1
        self._emit_else_body(stmt.else_body)
        self._line("end")

    def _truthiness(self, cond: TExpr) -> str | None:
        if isinstance(cond, TBinaryOp):
            if (
                cond.op == ">"
                and isinstance(cond.right, TIntLit)
                and cond.right.value == 0
            ):
                if (
                    isinstance(cond.left, TCall)
                    and isinstance(cond.left.func, TVar)
                    and cond.left.func.name == "Len"
                ):
                    obj = self._expr(cond.left.args[0].value)
                    return "!" + obj + ".empty?"
            if (
                cond.op == "!="
                and isinstance(cond.right, TStringLit)
                and not cond.right.value
            ):
                return "!" + self._expr(cond.left) + ".empty?"
        return None

    def _nil_coalesce_lhs(self, expr: TTernary) -> str | None:
        cond = expr.cond
        if (
            isinstance(cond, TBinaryOp)
            and cond.op == "!="
            and isinstance(cond.right, TNilLit)
        ):
            return self._expr(cond.left)
        return None

    def _partition_args(self, expr: TTernary) -> tuple[str | None, str]:
        """Extract (s, sep) from a partition/rpartition ternary."""
        cond = expr.cond
        if (
            isinstance(cond, TBinaryOp)
            and cond.op == ">="
            and isinstance(cond.left, TCall)
        ):
            call = cond.left
            if isinstance(call.func, TVar) and call.func.name in ("Find", "RFind"):
                return self._expr(call.args[0].value), self._expr(call.args[1].value)
        return None, ""

    def _delete_fix_args(
        self, expr: TTernary, func_name: str
    ) -> tuple[str | None, str]:
        """Extract (s, p) from a removeprefix/removesuffix ternary."""
        cond = expr.cond
        if (
            isinstance(cond, TCall)
            and isinstance(cond.func, TVar)
            and cond.func.name == func_name
        ):
            return self._expr(cond.args[0].value), self._expr(cond.args[1].value)
        return None, ""

    def _negate_inner(self, cond: TExpr) -> str | None:
        if isinstance(cond, TUnaryOp) and cond.op == "!":
            return self._expr(cond.operand)
        return None

    def _emit_else_body(self, else_body: list[TStmt] | None) -> None:
        if else_body is None or not else_body:
            return
        elif_stmt: TStmt | None = None
        if len(else_body) == 1:
            elif_stmt = else_body[0]
        if isinstance(elif_stmt, TIfStmt):
            self._line("elsif " + self._expr(elif_stmt.cond))
            self.indent += 1
            if not elif_stmt.then_body:
                self._line("nil")
            self._emit_stmts(elif_stmt.then_body)
            self.indent -= 1
            self._emit_else_body(elif_stmt.else_body)
        else:
            self._line("else")
            self.indent += 1
            self._emit_stmts(else_body)
            self.indent -= 1

    def _emit_while(self, stmt: TWhileStmt) -> None:
        prov = stmt.annotations.get("provenance", "")
        if prov == "negated_while":
            inner = self._negate_inner(stmt.cond)
            if inner is not None:
                self._line("until " + inner)
                self.indent += 1
                if not stmt.body:
                    self._line("nil")
                self._emit_stmts(stmt.body)
                self.indent -= 1
                self._line("end")
                return
        self._line("while " + self._expr(stmt.cond))
        self.indent += 1
        if not stmt.body:
            self._line("nil")
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("end")

    def _emit_for(self, stmt: TForStmt) -> None:
        binding = stmt.binding
        ann = stmt.annotations
        # Reversed(xs) → xs.reverse_each
        if self._is_builtin_call(stmt.iterable, "Reversed") and isinstance(
            stmt.iterable, TCall
        ):
            binders = ", ".join(self._decl_name(b, ann) for b in binding)
            inner = self._expr(stmt.iterable.args[0].value)
            self._line(inner + ".reverse_each do |" + binders + "|")
            self.indent += 1
            if not stmt.body:
                self._line("nil")
            self._emit_stmts(stmt.body)
            self.indent -= 1
            self._line("end")
            return
        # Zip(a, b, ...) → a.zip(b, ...).each
        if self._is_builtin_call(stmt.iterable, "Zip") and isinstance(
            stmt.iterable, TCall
        ):
            zip_args = stmt.iterable.args
            binders = ", ".join(self._decl_name(b, ann) for b in binding)
            zip_parts: list[str] = []
            for a in zip_args:
                s = self._expr(a.value)
                if self._is_bytes_type(a.value):
                    s += ".bytes"
                zip_parts.append(s)
            first = zip_parts[0]
            rest = ", ".join(zip_parts[1:])
            self._line(first + ".zip(" + rest + ").each do |" + binders + "|")
            self.indent += 1
            if not stmt.body:
                self._line("nil")
            self._emit_stmts(stmt.body)
            self.indent -= 1
            self._line("end")
            return
        if isinstance(stmt.iterable, TRange):
            binder_parts: list[str] = []
            for b in binding:
                binder_parts.append(self._decl_name(b, ann))
            binders = ", ".join(binder_parts)
            iterable = self._ruby_range(stmt.iterable)
            if ".downto(" in iterable:
                self._line(iterable + " do |" + binders + "|")
            else:
                self._line(iterable + ".each do |" + binders + "|")
        elif len(binding) == 1:
            iter_str = self._expr(stmt.iterable)
            if self._is_map_type(stmt.iterable):
                method = ".each_key"
            elif self._is_string_type(stmt.iterable):
                method = ".each_char"
            elif self._is_bytes_type(stmt.iterable):
                method = ".each_byte"
            else:
                method = ".each"
            self._line(
                iter_str + method + " do |" + self._decl_name(binding[0], ann) + "|"
            )
        elif len(binding) == 2:
            iter_is_map = self._is_map_for(stmt)
            is_enumerate = self._is_enumerate_for(stmt)
            if iter_is_map:
                self._line(
                    self._expr(stmt.iterable)
                    + ".each do |"
                    + self._decl_name(binding[0], ann)
                    + ", "
                    + self._decl_name(binding[1], ann)
                    + "|"
                )
            elif is_enumerate:
                self._line(
                    self._expr(stmt.iterable)
                    + ".each_with_index do |"
                    + self._decl_name(binding[1], ann)
                    + ", "
                    + self._decl_name(binding[0], ann)
                    + "|"
                )
            else:
                self._line(
                    self._expr(stmt.iterable)
                    + ".each do |"
                    + self._decl_name(binding[0], ann)
                    + ", "
                    + self._decl_name(binding[1], ann)
                    + "|"
                )
        else:
            binder_parts2: list[str] = []
            for b in binding:
                binder_parts2.append(self._decl_name(b, ann))
            binders = ", ".join(binder_parts2)
            self._line(self._expr(stmt.iterable) + ".each do |" + binders + "|")
        self.indent += 1
        if not stmt.body:
            self._line("nil")
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("end")

    def _ruby_range(self, r: TRange) -> str:
        if len(r.args) == 1:
            return "(0..." + self._expr(r.args[0]) + ")"
        if len(r.args) == 2:
            return "(" + self._expr(r.args[0]) + "..." + self._expr(r.args[1]) + ")"
        start_val = self._static_int(r.args[0])
        start = str(start_val) if start_val is not None else self._expr(r.args[0])
        end_expr = r.args[1]
        step = r.args[2]
        is_neg_step = (isinstance(step, TUnaryOp) and step.op == "-") or (
            isinstance(step, TIntLit) and step.value < 0
        )
        if is_neg_step:
            end_val = self._static_int(end_expr)
            if end_val is not None:
                adjusted = str(end_val + 1)
            else:
                adjusted = self._expr(end_expr) + " + 1"
            return start + ".downto(" + adjusted + ")"
        return "(0..." + self._expr(r.args[0]) + ")"

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
        if isinstance(expr, TFieldAccess):
            obj_type = self._resolve_type_name(expr.obj)
            if obj_type is not None:
                ft = self.field_types.get(obj_type, {})
                return isinstance(ft.get(expr.field), TMapType)
        return False

    def _resolve_type_name(self, expr: TExpr) -> str | None:
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            if isinstance(typ, TOptionalType):
                inner = typ.inner
                if isinstance(inner, TIdentType):
                    return inner.name
            if isinstance(typ, TIdentType):
                return typ.name
        return None

    def _is_builtin_call(self, expr: TExpr, name: str) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == name
        )

    def _is_map_for(self, stmt: TForStmt) -> bool:
        if stmt.annotations.get("for.items") == "true":
            return True
        return not isinstance(stmt.iterable, TRange) and self._is_map_type(
            stmt.iterable
        )

    def _is_string_type(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann in ("string", "rune")
        if isinstance(expr, TStringLit):
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "string"
        return False

    def _capture_var_annotations(self, name: str, annotations: Ann) -> None:
        """Capture strings.* annotations for a variable."""
        strings_ann: dict[str, str] = {}
        for key, val in annotations.items():
            if key.startswith("strings."):
                strings_ann[key] = val
        if strings_ann:
            self.var_annotations[name] = strings_ann

    def _is_ascii_content(self, expr: TExpr) -> bool:
        """Check if expression has strings.content=ascii annotation."""
        if isinstance(expr, TVar):
            ann = self.var_annotations.get(expr.name, {})
            return ann.get("strings.content") == "ascii"
        return False

    def _is_bytes_type(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann == "bytes"
        if isinstance(expr, TBytesLit):
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "bytes"
        if isinstance(expr, TFieldAccess):
            obj_type = self._resolve_type_name(expr.obj)
            if obj_type is not None:
                ft = self.field_types.get(obj_type, {})
                ftyp = ft.get(expr.field)
                return isinstance(ftyp, TPrimitive) and ftyp.kind == "bytes"
        return False

    def _emit_try(self, stmt: TTryStmt) -> None:
        self._line("begin")
        self.indent += 1
        if not stmt.body:
            self._line("nil")
        self._emit_stmts(stmt.body)
        self.indent -= 1
        for catch in stmt.catches:
            self._emit_catch(catch)
        if stmt.finally_body is not None:
            self._line("ensure")
            self.indent += 1
            if not stmt.finally_body:
                self._line("nil")
            self._emit_stmts(stmt.finally_body)
            self.indent -= 1
        self._line("end")

    def _emit_catch(self, catch: TCatch) -> None:
        types: list[str] = []
        for t in catch.types:
            if isinstance(t, TIdentType):
                types.append(_EXCEPTION_MAP.get(t.name, _safe_type_name(t.name)))
            else:
                types.append("StandardError")
        type_str = ", ".join(types)
        unused = catch.annotations.get("liveness.catch_var_unused") == "true"
        if unused:
            if type_str:
                self._line("rescue " + type_str)
            else:
                self._line("rescue")
        elif type_str:
            self._line(
                "rescue "
                + type_str
                + " => "
                + self._decl_name(catch.name, catch.annotations)
            )
        else:
            self._line("rescue => " + self._decl_name(catch.name, catch.annotations))
        self.indent += 1
        if not catch.body:
            self._line("nil")
        self._emit_stmts(catch.body)
        self.indent -= 1

    def _emit_match(self, stmt: TMatchStmt) -> None:
        expr_str = self._expr(stmt.expr)
        first = True
        for case in stmt.cases:
            self._emit_match_case(case, expr_str, first)
            first = False
        if stmt.default is not None:
            self._emit_match_default(stmt.default, expr_str, first)
        self._line("end")

    def _emit_match_case(self, case: TMatchCase, expr_str: str, first: bool) -> None:
        pat = case.pattern
        keyword = "if" if first else "elsif"
        if isinstance(pat, TPatternType):
            type_name = self._type_name_for_check(pat.type_name)
            self._line(keyword + " " + expr_str + ".is_a?(" + type_name + ")")
            self.indent += 1
            unused = pat.annotations.get("liveness.match_var_unused") == "true"
            if not unused:
                self._line(_safe_local_name(pat.name) + " = " + expr_str)
            if not case.body and unused:
                self._line("nil")
            self._emit_stmts(case.body)
            self.indent -= 1
        elif isinstance(pat, TPatternEnum):
            self._line(
                keyword + " " + expr_str + " == " + pat.enum_name + "::" + pat.variant
            )
            self.indent += 1
            if not case.body:
                self._line("nil")
            self._emit_stmts(case.body)
            self.indent -= 1
        elif isinstance(pat, TPatternNil):
            self._line(keyword + " " + expr_str + ".nil?")
            self.indent += 1
            if not case.body:
                self._line("nil")
            self._emit_stmts(case.body)
            self.indent -= 1

    def _emit_match_default(
        self, default: TDefault, expr_str: str, first: bool
    ) -> None:
        if first:
            self._line("if true")
        else:
            self._line("else")
        self.indent += 1
        if default.name is not None:
            unused = default.annotations.get("liveness.match_var_unused") == "true"
            if not unused:
                self._line(_safe_local_name(default.name) + " = " + expr_str)
        if not default.body:
            self._line("nil")
        self._emit_stmts(default.body)
        self.indent -= 1

    def _type_name_for_check(self, typ: TType) -> str:
        if isinstance(typ, TIdentType):
            if typ.name in self.struct_names:
                return typ.name
            return _safe_type_name(typ.name)
        if isinstance(typ, TPrimitive):
            if typ.kind == "string":
                return "String"
            if typ.kind == "int":
                return "Integer"
            if typ.kind == "float":
                return "Float"
            if typ.kind == "bool":
                return "TrueClass"
        if isinstance(typ, TListType):
            return "Array"
        if isinstance(typ, TMapType):
            return "Hash"
        if isinstance(typ, TSetType):
            return "Set"
        if isinstance(typ, TTupleType):
            return "Array"
        return "Object"

    # ── Expressions ───────────────────────────────────────────

    def _expr(self, expr: TExpr) -> str:
        if isinstance(expr, TIntLit):
            return self._int_lit(expr)
        if isinstance(expr, TFloatLit):
            return expr.raw
        if isinstance(expr, TStringLit):
            return '"' + _escape_ruby_string(expr.value) + '"'
        if isinstance(expr, TBoolLit):
            return "true" if expr.value else "false"
        if isinstance(expr, TNilLit):
            return "nil"
        if isinstance(expr, TByteLit):
            return expr.raw
        if isinstance(expr, TBytesLit):
            return self._bytes_lit(expr)
        if isinstance(expr, TRuneLit):
            return '"' + _escape_ruby_string(expr.value) + '"'
        if isinstance(expr, TVar):
            if expr.name == self.self_name:
                return "self"
            return self._ref_name(expr.name, expr.annotations)
        if isinstance(expr, TFieldAccess):
            if isinstance(expr.obj, TVar) and expr.obj.name in self.enum_names:
                return expr.obj.name + "::" + expr.field
            return self._expr(expr.obj) + "." + _safe_name(expr.field)
        if isinstance(expr, TTupleAccess):
            return self._expr(expr.obj) + "[" + str(expr.index) + "]"
        if isinstance(expr, TIndex):
            # Use O(1) getbyte for ASCII strings instead of O(n) character indexing
            if self._is_string_type(expr.obj) and self._is_ascii_content(expr.obj):
                idx = self._expr(expr.index)
                if expr.annotations.get("provenance") == "negative_index":
                    neg = self._negative_index(expr)
                    if neg is not None:
                        idx = neg
                return self._expr(expr.obj) + ".getbyte(" + idx + ")&.chr"
            if expr.annotations.get("provenance") == "negative_index":
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
            prov = expr.annotations.get("provenance", "")
            if prov == "none_coalesce":
                lhs = self._nil_coalesce_lhs(expr)
                if lhs is not None:
                    return lhs + " || " + self._expr(expr.else_expr)
            if prov in ("partition", "rpartition"):
                s, sep = self._partition_args(expr)
                if s is not None:
                    method = "partition" if prov == "partition" else "rpartition"
                    return s + "." + method + "(" + sep + ")"
            if prov == "removeprefix":
                s, p = self._delete_fix_args(expr, "StartsWith")
                if s is not None:
                    return s + ".delete_prefix(" + p + ")"
            if prov == "removesuffix":
                s, p = self._delete_fix_args(expr, "EndsWith")
                if s is not None:
                    return s + ".delete_suffix(" + p + ")"
            else_str = self._expr(expr.else_expr)
            if isinstance(expr.else_expr, TTernary):
                else_str = "(" + else_str + ")"
            return (
                self._expr(expr.cond)
                + " ? "
                + self._expr(expr.then_expr)
                + " : "
                + else_str
            )
        if isinstance(expr, TListLit):
            elems = ", ".join(self._expr(e) for e in expr.elements)
            return "[" + elems + "]"
        if isinstance(expr, TMapLit):
            if not expr.entries:
                return "{}"
            pairs = ", ".join(
                self._expr(k) + " => " + self._expr(v) for k, v in expr.entries
            )
            return "{" + pairs + "}"
        if isinstance(expr, TSetLit):
            self._needs_set = True
            if not expr.elements:
                return "Set.new"
            elems = ", ".join(self._expr(e) for e in expr.elements)
            return "Set[" + elems + "]"
        if isinstance(expr, TTupleLit):
            elems = ", ".join(self._expr(e) for e in expr.elements)
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
            if 32 <= b < 127 and b != ord("\\") and b != ord('"'):
                parts.append(chr(b))
            else:
                h = hex(b)[2:]
                if len(h) == 1:
                    h = "0" + h
                parts.append("\\x" + h)
        return '"' + "".join(parts) + '"'

    def _slice(self, expr: TSlice) -> str:
        obj = self._expr(expr.obj)
        prov = expr.annotations.get("provenance", "")
        low = self._expr(expr.low)
        high = self._expr(expr.high)
        if prov == "open_start" and self._is_zero(expr.low):
            low = "0"
        # Use O(1) byteslice for ASCII strings instead of O(n) character slicing
        if self._is_string_type(expr.obj) and self._is_ascii_content(expr.obj):
            if prov == "open_end" and self._is_len_call(expr.high):
                return (
                    obj + ".byteslice(" + low + ", " + obj + ".bytesize - " + low + ")"
                )
            return obj + ".byteslice(" + low + ", (" + high + ") - (" + low + "))"
        if prov == "open_end" and self._is_len_call(expr.high):
            return obj + "[" + low + "..]"
        return obj + "[" + low + "..." + high + "]"

    def _negative_index(self, expr: TIndex) -> str | None:
        idx = expr.index
        if isinstance(idx, TBinaryOp) and idx.op == "-":
            if (
                isinstance(idx.left, TCall)
                and isinstance(idx.left.func, TVar)
                and idx.left.func.name == "Len"
            ):
                return "-" + self._expr(idx.right)
        return None

    def _binary(self, expr: TBinaryOp) -> str:
        op = expr.op
        if self.strict_math and op in STRICT_INT_BINARY:
            if self._is_int_expr(expr.left) and self._is_int_expr(expr.right):
                fn = STRICT_INT_BINARY[op]
                return (
                    fn
                    + "("
                    + self._expr(expr.left)
                    + ", "
                    + self._expr(expr.right)
                    + ")"
                )
        if self.strict_math and op == "%":
            if self._is_float_expr(expr.left) or self._is_float_expr(expr.right):
                return (
                    "strict_fmod("
                    + self._expr(expr.left)
                    + ", "
                    + self._expr(expr.right)
                    + ")"
                )
        # Chained comparison
        if op == "&&" and expr.annotations.get("provenance") == "chained_comparison":
            chained = self._chain_comparison(expr)
            if chained is not None:
                return chained
        # nil comparisons
        if op == "==" and isinstance(expr.right, TNilLit):
            return self._maybe_paren(expr.left, op, is_left=True) + ".nil?"
        if op == "!=" and isinstance(expr.right, TNilLit):
            return "!" + self._maybe_paren(expr.left, op, is_left=True) + ".nil?"
        if op == "==" and isinstance(expr.left, TNilLit):
            return self._maybe_paren(expr.right, op, is_left=False) + ".nil?"
        if op == "!=" and isinstance(expr.left, TNilLit):
            return "!" + self._maybe_paren(expr.right, op, is_left=False) + ".nil?"
        # Floor division
        if op == "//":
            left_str = self._maybe_paren(expr.left, "/", is_left=True)
            right_str = self._maybe_paren(expr.right, "/", is_left=False)
            return "(" + left_str + " / " + right_str + ")"
        # in / not in
        if op == "in":
            return self._expr(expr.right) + ".include?(" + self._expr(expr.left) + ")"
        if op == "not in":
            return (
                "!"
                + self._expr(expr.right)
                + ".include?("
                + self._expr(expr.left)
                + ")"
            )
        # String / list multiplication provenance
        prov = expr.annotations.get("provenance", "")
        if prov in ("string_multiply", "list_multiply"):
            left_str = self._maybe_paren(expr.left, op, is_left=True)
            right_str = self._maybe_paren(expr.right, op, is_left=False)
            return left_str + " * [" + right_str + ", 0].max"
        rb_op = op
        if op == "&&":
            rb_op = "&&"
        elif op == "||":
            rb_op = "||"
        left_str = self._maybe_paren(expr.left, op, is_left=True)
        right_str = self._maybe_paren(expr.right, op, is_left=False)
        return left_str + " " + rb_op + " " + right_str

    def _chain_comparison(self, expr: TBinaryOp) -> str | None:
        left = expr.left
        right = expr.right
        if (
            isinstance(left, TBinaryOp)
            and isinstance(right, TBinaryOp)
            and left.op in _CMP_OPS
            and right.op in _CMP_OPS
        ):
            return (
                self._expr(left.left)
                + " "
                + left.op
                + " "
                + self._expr(left.right)
                + " && "
                + self._expr(right.left)
                + " "
                + right.op
                + " "
                + self._expr(right.right)
            )
        return None

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
                    + ".include?("
                    + self._a(expr.operand.args, 1)
                    + ")"
                )
            if isinstance(expr.operand, (TBinaryOp, TTernary)):
                return "!(" + self._expr(expr.operand) + ")"
            return "!" + self._expr(expr.operand)
        if isinstance(expr.operand, (TBinaryOp, TTernary)):
            return op + "(" + self._expr(expr.operand) + ")"
        operand_str = self._expr(expr.operand)
        if op == "-" and operand_str.startswith("-"):
            return "- " + operand_str
        return op + operand_str

    def _maybe_paren(self, expr: TExpr, parent_op: str, is_left: bool) -> str:
        if isinstance(expr, TBinaryOp):
            if _needs_parens(expr.op, parent_op, is_left):
                return "(" + self._expr(expr) + ")"
        elif isinstance(expr, TTernary):
            return "(" + self._expr(expr) + ")"
        elif isinstance(expr, TUnaryOp):
            if expr.op == "-" and parent_op == "**" and is_left:
                return "(" + self._expr(expr) + ")"
        return self._expr(expr)

    def _fn_lit(self, expr: TFnLit) -> str:
        params = ", ".join(
            self._decl_name(p.name, p.annotations)
            for p in expr.params
            if p.typ is not None
        )
        first = expr.body[0] if expr.body else None
        if expr.annotations.get("fn_lit.arrow") == "true" and isinstance(
            first, TExprStmt
        ):
            return "lambda { |" + params + "| " + self._expr(first.expr) + " }"
        old_lines = self.lines
        self.lines = []
        self.indent += 1
        for s in expr.body:
            self._emit_stmt(s)
        self.indent -= 1
        body_lines = self.lines
        self.lines = old_lines
        result = "lambda { |" + params + "|\n"
        result += "\n".join(body_lines) + "\n"
        result += "  " * self.indent + "}"
        return result

    def _fn_lit_block_form(self, expr: TFnLit) -> str:
        """Render TFnLit as a Ruby block: { |param| expr }."""
        params = ", ".join(
            self._decl_name(p.name, p.annotations)
            for p in expr.params
            if p.typ is not None
        )
        first = expr.body[0] if expr.body else None
        if expr.annotations.get("fn_lit.arrow") == "true" and isinstance(
            first, TExprStmt
        ):
            return "{ |" + params + "| " + self._expr(first.expr) + " }"
        return self._fn_lit(expr)

    # ── Calls ─────────────────────────────────────────────────

    def _call(self, expr: TCall) -> str:
        func = expr.func
        args = expr.args
        if (
            isinstance(func, TVar)
            and func.name == "Concat"
            and expr.annotations.get("provenance") == "star_unpack"
        ):
            return self._star_unpack(expr)
        # list(dict) / set(dict) reconstruction via dict_keys provenance
        if (
            isinstance(func, TVar)
            and func.name in ("ListFrom", "SetFromList")
            and expr.annotations.get("provenance") == "dict_keys"
        ):
            inner = args[0].value
            if isinstance(inner, TCall):
                dict_expr = self._expr(inner.args[0].value)
                if func.name == "ListFrom":
                    return dict_expr + ".keys"
                self._needs_set = True
                return "Set.new(" + dict_expr + ".keys)"
        # Builtin call
        if isinstance(func, TVar) and func.name in BUILTIN_NAMES:
            return self._builtin_call(func.name, args, expr)
        # Struct constructor
        if isinstance(func, TVar) and func.name in self.struct_names:
            return self._struct_call(func.name, args)
        # Method call
        if isinstance(func, TFieldAccess):
            return self._method_call(func, args)
        # Regular call — could be a variable holding a function
        fn_expr = self._expr(func)
        arg_strs = ", ".join(self._expr(a.value) for a in args)
        if isinstance(func, TVar) and func.name in self.fn_names:
            # method(:name) was returned by _expr — just call it
            if arg_strs:
                return _safe_fn_name(func.name) + "(" + arg_strs + ")"
            return _safe_fn_name(func.name) + "()"
        if arg_strs:
            return fn_expr + ".call(" + arg_strs + ")"
        return fn_expr + ".call"

    def _star_unpack(self, expr: TCall) -> str:
        """Reconstruct [*a, x, *b] from a Concat chain with star_unpack provenance."""
        parts: list[TExpr] = []
        self._flatten_star_unpack(expr, parts)
        items: list[str] = []
        for p in parts:
            if isinstance(p, TListLit):
                for elem in p.elements:
                    items.append(self._expr(elem))
            else:
                items.append("*" + self._expr(p))
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

    def _struct_call(self, name: str, args: list[TArg]) -> str:
        fields = self.struct_fields.get(name, [])
        parts: list[str] = []
        for i, a in enumerate(args):
            if a.name is not None:
                parts.append(_safe_name(a.name) + ": " + self._expr(a.value))
            elif i < len(fields):
                parts.append(_safe_name(fields[i]) + ": " + self._expr(a.value))
            else:
                parts.append(self._expr(a.value))
        rname = _EXCEPTION_MAP.get(name, _safe_type_name(name))
        return rname + ".new(" + ", ".join(parts) + ")"

    def _method_call(self, func: TFieldAccess, args: list[TArg]) -> str:
        obj_str = self._expr(func.obj)
        if isinstance(func.obj, (TBinaryOp, TUnaryOp, TTernary)):
            obj_str = "(" + obj_str + ")"
        if func.field == "decode":
            return (
                obj_str
                + ".pack('C*').force_encoding('UTF-8')"
                + ".tap { |s| raise ArgumentError unless s.valid_encoding? }"
            )
        if func.field == "encode":
            return obj_str + '.encode("utf-8").bytes'
        if func.field == "copy" and not args:
            if not isinstance(func.obj, TVar) or self._is_map_type(func.obj):
                return obj_str + ".dup"
        if func.field == "count" and len(args) == 1:
            return obj_str + ".scan(" + self._expr(args[0].value) + ").length"
        if func.field == "replace":
            if len(args) == 3:
                return (
                    obj_str
                    + ".sub("
                    + self._expr(args[0].value)
                    + ", "
                    + self._expr(args[1].value)
                    + ")"
                )
            if len(args) == 2:
                return (
                    obj_str
                    + ".gsub("
                    + self._expr(args[0].value)
                    + ", "
                    + self._expr(args[1].value)
                    + ")"
                )
        if func.field == "get":
            if len(args) == 1:
                return obj_str + "[" + self._expr(args[0].value) + "]"
            if len(args) == 2:
                return (
                    obj_str
                    + ".fetch("
                    + self._expr(args[0].value)
                    + ", "
                    + self._expr(args[1].value)
                    + ")"
                )
        arg_strs = ", ".join(self._expr(a.value) for a in args)
        if arg_strs:
            return obj_str + "." + _safe_name(func.field) + "(" + arg_strs + ")"
        return obj_str + "." + _safe_name(func.field)

    def _builtin_call(self, name: str, args: list[TArg], call: TCall) -> str:
        if name == "FloorDiv":
            left = self._maybe_paren(args[0].value, "/", is_left=True)
            right = self._maybe_paren(args[1].value, "/", is_left=False)
            return left + " / " + right
        if name == "PythonMod":
            left = self._maybe_paren(args[0].value, "%", is_left=True)
            right = self._maybe_paren(args[1].value, "%", is_left=False)
            return left + " % " + right
        # List operations
        if name == "Append":
            return self._a(args, 0) + ".push(" + self._a(args, 1) + ")"
        if name == "Insert":
            return (
                self._a(args, 0)
                + ".insert("
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ")"
            )
        if name == "Pop":
            return self._a(args, 0) + ".pop"
        if name == "RemoveAt":
            return self._a(args, 0) + ".delete_at(" + self._a(args, 1) + ")"
        if name == "ReplaceSlice":
            return (
                self._a(args, 0)
                + "["
                + self._a(args, 1)
                + "..."
                + self._a(args, 2)
                + "] = "
                + self._a(args, 3)
            )
        if name == "IndexOf":
            obj = self._a(args, 0)
            val = self._a(args, 1)
            return obj + ".index(" + val + ") || -1"
        # String operations
        if name == "Upper":
            return self._a(args, 0) + ".upcase"
        if name == "Lower":
            return self._a(args, 0) + ".downcase"
        if name == "Trim":
            a0 = self._a(args, 0)
            return a0 + self._trim_gsub(args[1].value, "both")
        if name == "TrimStart":
            a0 = self._a(args, 0)
            return a0 + self._trim_gsub(args[1].value, "start")
        if name == "TrimEnd":
            a0 = self._a(args, 0)
            return a0 + self._trim_gsub(args[1].value, "end")
        if name == "Split":
            return self._a(args, 0) + ".split(" + self._a(args, 1) + ", -1)"
        if name == "SplitN":
            obj = self._a(args, 0)
            sep = self._a(args, 1)
            n_expr = args[2].value
            if isinstance(n_expr, TIntLit):
                return obj + ".split(" + sep + ", " + str(n_expr.value) + ")"
            return obj + ".split(" + sep + ", " + self._a(args, 2) + ")"
        if name == "SplitWhitespace":
            return self._a(args, 0) + ".split"
        if name == "Join":
            return self._a(args, 1) + ".join(" + self._a(args, 0) + ")"
        if name == "Find":
            if len(args) == 4:
                s = (
                    self._a(args, 0)
                    + "["
                    + self._a(args, 2)
                    + "..."
                    + self._a(args, 3)
                    + "]"
                )
                return (
                    "((i = "
                    + s
                    + ".index("
                    + self._a(args, 1)
                    + ")).nil? ? -1 : i + "
                    + self._a(args, 2)
                    + ")"
                )
            if len(args) == 3:
                return (
                    self._a(args, 0)
                    + ".index("
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ") || -1"
                )
            return self._a(args, 0) + ".index(" + self._a(args, 1) + ") || -1"
        if name == "RFind":
            if len(args) >= 3:
                pos = self._a(args, 2)
                s = self._a(args, 0) + "[" + pos + ".."
                if len(args) == 4:
                    s = self._a(args, 0) + "[" + pos + "..." + self._a(args, 3) + "]"
                else:
                    s += "]"
                return (
                    "((i = "
                    + s
                    + ".rindex("
                    + self._a(args, 1)
                    + ")).nil? ? -1 : i + "
                    + pos
                    + ")"
                )
            return self._a(args, 0) + ".rindex(" + self._a(args, 1) + ") || -1"
        if name == "Count":
            subj = self._a(args, 0)
            if len(args) >= 3:
                subj = subj + "[" + self._a(args, 2) + ".."
                if len(args) == 4:
                    subj = (
                        self._a(args, 0)
                        + "["
                        + self._a(args, 2)
                        + "..."
                        + self._a(args, 3)
                        + "]"
                    )
                else:
                    subj += "]"
            if self._is_string_type(args[0].value) or isinstance(
                args[1].value, TStringLit
            ):
                return subj + ".scan(" + self._a(args, 1) + ").length"
            return subj + ".count(" + self._a(args, 1) + ")"
        if name == "Replace":
            return (
                self._a(args, 0)
                + ".gsub("
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ")"
            )
        if name == "ReplaceCount":
            return (
                self._a(args, 0)
                + ".sub("
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ")"
            )
        if name == "StartsWith":
            if len(args) >= 3:
                s = self._a(args, 0) + "[" + self._a(args, 2) + ".."
                if len(args) == 4:
                    s = (
                        self._a(args, 0)
                        + "["
                        + self._a(args, 2)
                        + "..."
                        + self._a(args, 3)
                        + "]"
                    )
                else:
                    s += "]"
                return s + ".start_with?(" + self._a(args, 1) + ")"
            return self._a(args, 0) + ".start_with?(" + self._a(args, 1) + ")"
        if name == "EndsWith":
            if len(args) >= 3:
                s = self._a(args, 0) + "[" + self._a(args, 2) + ".."
                if len(args) == 4:
                    s = (
                        self._a(args, 0)
                        + "["
                        + self._a(args, 2)
                        + "..."
                        + self._a(args, 3)
                        + "]"
                    )
                else:
                    s += "]"
                return s + ".end_with?(" + self._a(args, 1) + ")"
            return self._a(args, 0) + ".end_with?(" + self._a(args, 1) + ")"
        if name == "IsDigit":
            return self._a(args, 0) + r".match?(/^\d+$/)"
        if name == "IsAlpha":
            return self._a(args, 0) + r".match?(/^[[:alpha:]]+$/)"
        if name == "IsAlnum":
            return self._a(args, 0) + r".match?(/^[[:alnum:]]+$/)"
        if name == "IsSpace":
            return self._a(args, 0) + r".match?(/^\s+$/)"
        if name == "IsUpper":
            a = self._a(args, 0)
            return "(" + a + ".match?(/[[:alpha:]]/) && " + a + " == " + a + ".upcase)"
        if name == "IsLower":
            a = self._a(args, 0)
            return (
                "(" + a + ".match?(/[[:alpha:]]/) && " + a + " == " + a + ".downcase)"
            )
        if name == "Reverse":
            return self._a(args, 0) + ".reverse"
        if name == "Repeat":
            count = self._a(args, 1)
            if isinstance(args[1].value, TBinaryOp):
                count = "(" + count + ")"
            return self._a(args, 0) + " * " + count
        if name == "RemovePrefix":
            return self._a(args, 0) + ".delete_prefix(" + self._a(args, 1) + ")"
        if name == "RemoveSuffix":
            return self._a(args, 0) + ".delete_suffix(" + self._a(args, 1) + ")"
        # Encode / Decode
        if name == "Encode":
            return self._a(args, 0) + '.encode("utf-8").bytes'
        if name == "Decode":
            arg = self._a(args, 0)
            return (
                "(("
                + arg
                + ").is_a?(Array) ? ("
                + arg
                + ").pack('C*') : "
                + "("
                + arg
                + ")).force_encoding('UTF-8')"
                + ".tap { |s| raise UnicodeDecodeError unless s.valid_encoding? }"
            )
        # Set operations
        if name == "Add":
            self._needs_set = True
            return self._a(args, 0) + ".add(" + self._a(args, 1) + ")"
        if name == "Remove":
            self._needs_set = True
            return self._a(args, 0) + ".delete(" + self._a(args, 1) + ")"
        if name == "Union":
            self._needs_set = True
            return self._a(args, 0) + " | " + self._a(args, 1)
        if name == "Intersection":
            self._needs_set = True
            return self._a(args, 0) + " & " + self._a(args, 1)
        if name == "Difference":
            self._needs_set = True
            return self._a(args, 0) + " - " + self._a(args, 1)
        # Map operations
        if name == "Get":
            if len(args) == 3:
                return (
                    self._a(args, 0)
                    + ".fetch("
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ")"
                )
            return self._a(args, 0) + "[" + self._a(args, 1) + "]"
        if name == "Delete":
            return self._a(args, 0) + ".delete(" + self._a(args, 1) + ")"
        if name == "Merge":
            return self._a(args, 0) + ".merge(" + self._a(args, 1) + ")"
        if name == "Keys":
            return self._a(args, 0) + ".keys"
        if name == "Values":
            return self._a(args, 0) + ".values"
        if name == "Items":
            return self._a(args, 0) + ".to_a"
        # Direct functions
        if name == "Len":
            a = self._a(args, 0)
            if isinstance(args[0].value, (TBinaryOp, TUnaryOp, TTernary, TCall)):
                return "(" + a + ").length"
            return a + ".length"
        if name == "Abs":
            a = self._a(args, 0)
            if isinstance(args[0].value, (TBinaryOp, TUnaryOp, TTernary, TCall)):
                return "(" + a + ").abs"
            return a + ".abs"
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
                        self._a(args, 0) + ".min_by " + self._fn_lit_block_form(key_val)
                    )
            if len(args) == 1:
                return self._a(args, 0) + ".min"
            return "[" + self._a(args, 0) + ", " + self._a(args, 1) + "].min"
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
                        self._a(args, 0) + ".max_by " + self._fn_lit_block_form(key_val)
                    )
            if len(args) == 1:
                return self._a(args, 0) + ".max"
            return "[" + self._a(args, 0) + ", " + self._a(args, 1) + "].max"
        if name == "Sum":
            return self._a(args, 0) + ".sum"
        if name == "Round":
            if len(args) == 2:
                return self._a(args, 0) + ".round(" + self._a(args, 1) + ")"
            return self._a(args, 0) + ".round"
        if name == "DivMod":
            a = self._a(args, 0)
            b = self._a(args, 1)
            return (
                "[("
                + a
                + ".to_f / "
                + b
                + ").truncate, "
                + a
                + ".remainder("
                + b
                + ")]"
            )
        if name == "Sorted":
            if self.strict_math and self._is_float_list(args[0].value):
                return "strict_sorted_f64(" + self._a(args, 0) + ")"
            if len(args) == 2:
                key_val = args[1].value
                if isinstance(key_val, TFnLit):
                    return (
                        self._a(args, 0)
                        + ".sort_by "
                        + self._fn_lit_block_form(key_val)
                    )
            return self._a(args, 0) + ".sort"
        if name == "RangeList":
            start = args[0].value
            end = self._expr(args[1].value)
            step = args[2].value
            is_zero_start = isinstance(start, TIntLit) and start.value == 0
            is_one_step = isinstance(step, TIntLit) and step.value == 1
            if is_one_step:
                s = self._expr(start) if not is_zero_start else "0"
                return "(" + s + "..." + end + ").to_a"
            s = self._expr(start)
            return "(" + s + "..." + end + ").step(" + self._expr(step) + ").to_a"
        if name == "ListFrom":
            if self._is_bytes_type(args[0].value):
                return self._a(args, 0) + ".bytes"
            return self._a(args, 0) + ".dup"
        if name == "Reversed":
            return self._a(args, 0) + ".reverse"
        if name == "Map":
            if not args:
                return "{}"
            return self._a(args, 1) + ".map { |_e| " + self._a(args, 0) + ".call(_e) }"
        if name == "Set":
            self._needs_set = True
            if not args:
                return "Set.new"
            return "Set.new(" + self._a(args, 0) + ".to_a)"
        if name == "SetFromList":
            self._needs_set = True
            if isinstance(args[0].value, TSetLit):
                return self._a(args, 0)
            inner = args[0].value
            if isinstance(inner, TCall) and isinstance(inner.func, TVar):
                if inner.func.name in ("Keys", "Values"):
                    method = {"Keys": "keys", "Values": "values"}[inner.func.name]
                    return (
                        "Set.new("
                        + self._expr(inner.args[0].value)
                        + "."
                        + method
                        + ")"
                    )
            return "Set.new(" + self._a(args, 0) + ".to_a)"
        if name in ("ToString", "ToRepr"):
            a = self._a(args, 0)
            if self.strict_tostring and self._is_float_expr(args[0].value):
                self._needs_float_repr = True
                return "_py_float_repr(" + a + ")"
            if isinstance(args[0].value, (TBinaryOp, TTernary)):
                return "(" + a + ").to_s"
            return a + ".to_s"
        if name == "ParseInt":
            base = self._a(args, 1)
            return self._a(args, 0) + ".to_i(" + base + ")"
        if name == "ParseFloat":
            self._needs_parse_float = True
            return "_do_parse_float(" + self._a(args, 0) + ")"
        if name == "FormatInt":
            return self._format_int(args)
        if name == "RuneFromInt":
            return "[" + self._a(args, 0) + "].pack('U')"
        if name == "RuneToInt":
            return self._a(args, 0) + ".ord"
        if name == "IntToFloat":
            return self._a(args, 0) + ".to_f"
        if name == "FloatToInt":
            return self._a(args, 0) + ".to_i"
        if name == "ByteToInt":
            return self._a(args, 0) + ".ord"
        if name == "IntToByte":
            return "[" + self._a(args, 0) + "].pack('U')"
        if name == "Unwrap":
            return self._a(args, 0)
        if name == "IsNil":
            return self._a(args, 0) + ".nil?"
        if name == "Sqrt":
            return "Math.sqrt(" + self._a(args, 0) + ")"
        if name == "Floor":
            return self._a(args, 0) + ".floor"
        if name == "Ceil":
            return self._a(args, 0) + ".ceil"
        if name == "IsNaN":
            return self._a(args, 0) + ".nan?"
        if name == "IsInf":
            return self._a(args, 0) + ".infinite? != nil"
        # I/O
        if name == "WriteOut":
            return "$stdout.write(" + self._a(args, 0) + ")"
        if name == "WriteErr":
            return "$stderr.write(" + self._a(args, 0) + ")"
        if name == "WritelnOut":
            return "puts(" + self._a(args, 0) + ")"
        if name == "WritelnErr":
            return "$stderr.puts(" + self._a(args, 0) + ")"
        if name == "Bytes":
            return '("\0" * ' + self._a(args, 0) + ")"
        if name == "BytesFrom":
            return self._a(args, 0) + '.pack("C*")'
        if name == "ReadLine":
            return "$stdin.gets&.chomp"
        if name == "ReadAll":
            return (
                "$stdin.binmode.read.force_encoding('UTF-8')"
                ".tap { |s| raise UnicodeDecodeError unless s.valid_encoding? }"
            )
        if name == "ReadBytes":
            return "$stdin.binmode.read.bytes"
        if name == "ReadBytesN":
            return "$stdin.binmode.read(" + self._a(args, 0) + ").bytes"
        if name == "Args":
            return "ARGV"
        if name == "GetEnv":
            return "ENV[" + self._a(args, 0) + "]"
        if name == "ReadFile":
            return "File.read(" + self._a(args, 0) + ', encoding: "utf-8")'
        if name == "ReadFileBytes":
            return "File.binread(" + self._a(args, 0) + ").bytes"
        if name == "WriteFile":
            return "File.write(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Exit":
            return "exit(" + self._a(args, 0) + ")"
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
            left = self._maybe_paren(args[0].value, "**", is_left=True)
            right = self._maybe_paren(args[1].value, "**", is_left=False)
            return left + " ** " + right
        if name == "Contains":
            if self._is_map_type(args[0].value):
                return self._a(args, 0) + ".key?(" + self._a(args, 1) + ")"
            return self._a(args, 0) + ".include?(" + self._a(args, 1) + ")"
        if name == "Concat":
            left = self._a(args, 0)
            right = self._a(args, 1)
            if isinstance(args[0].value, TTernary):
                left = "(" + left + ")"
            if isinstance(args[1].value, TTernary):
                right = "(" + right + ")"
            return left + " + " + right
        if name == "Format":
            return self._format_call(args, call)
        if name == "IsType":
            type_arg = args[1].value
            if isinstance(type_arg, TStringLit):
                type_name = type_arg.value
            else:
                type_name = self._expr(type_arg)
            if type_name in self.struct_names:
                return self._a(args, 0) + ".is_a?(" + type_name + ")"
            return self._a(args, 0) + ".is_a?(" + _safe_type_name(type_name) + ")"
        if name == "Assert":
            cond = self._a(args, 0)
            if len(args) > 1:
                return "raise " + self._a(args, 1) + " unless " + cond
            return 'raise "assertion failed" unless ' + cond
        # Fallback
        arg_strs = ", ".join(self._expr(ar.value) for ar in args)
        return _safe_name(name) + "(" + arg_strs + ")"

    def _trim_gsub(self, expr: TExpr, mode: str) -> str:
        """Emit .gsub for Trim/TrimStart/TrimEnd with \\A/\\z anchors."""
        if isinstance(expr, TStringLit):
            c = _escape_regex_charclass(expr.value)
            if mode == "both":
                return ".gsub(/\\A[" + c + "]+|[" + c + ']+\\z/, "")'
            if mode == "start":
                return ".gsub(/\\A[" + c + ']+/, "")'
            return ".gsub(/[" + c + ']+\\z/, "")'
        ce = self._expr(expr)
        if mode == "both":
            return (
                '.gsub(Regexp.new("\\\\A[" + Regexp.escape('
                + ce
                + ') + "]+|[" + Regexp.escape('
                + ce
                + ') + "]+\\\\z"), "")'
            )
        if mode == "start":
            return '.gsub(Regexp.new("\\\\A[" + Regexp.escape(' + ce + ') + "]+"), "")'
        return '.gsub(Regexp.new("[" + Regexp.escape(' + ce + ') + "]+\\\\z"), "")'

    def _format_int(self, args: list[TArg]) -> str:
        n = self._a(args, 0)
        base_expr = args[1].value
        if isinstance(base_expr, TIntLit):
            if base_expr.value == 16:
                return n + ".to_s(16)"
            if base_expr.value == 8:
                return n + ".to_s(8)"
            if base_expr.value == 2:
                return n + ".to_s(2)"
        base = self._a(args, 1)
        return n + ".to_s(" + base + ")"

    def _format_call(self, args: list[TArg], call: TCall) -> str:
        template_expr = args[0].value
        if not isinstance(template_expr, TStringLit):
            _farg_parts: list[str] = []
            for a in args:
                _farg_parts.append(self._expr(a.value))
            joined_args = ", ".join(_farg_parts)
            return "format_(" + joined_args + ")"
        template = template_expr.value
        fmt_args = args[1:]
        prov = call.annotations.get("provenance", "")
        if prov == "f_string":
            parts: list[str] = []
            rest = template
            idx = 0
            while "{}" in rest and idx < len(fmt_args):
                split_pos = rest.index("{}")
                before = rest[:split_pos]
                rest = rest[split_pos + 2 :]
                parts.append(_escape_ruby_string(before))
                parts.append("#{" + self._expr(fmt_args[idx].value) + "}")
                idx += 1
            parts.append(_escape_ruby_string(rest))
            return '"' + "".join(parts) + '"'
        escaped = _escape_ruby_string(template).replace("{}", "%s")
        arg_strs: list[str] = []
        for a in fmt_args:
            arg_strs.append(self._expr(a.value))
        if len(arg_strs) == 1:
            return '"' + escaped + '" % ' + arg_strs[0]
        return '"' + escaped + '" % [' + ", ".join(arg_strs) + "]"


# ============================================================
# PUBLIC API
# ============================================================


def emit_ruby(module: TModule) -> str:
    struct_names: set[str] = set()
    for decl in module.decls:
        match decl:
            case TStructDecl() | TInterfaceDecl():
                struct_names.add(decl.name)
    for _bk in BUILTIN_STRUCTS:
        struct_names.add(_bk)
    struct_fields: dict[str, list[str]] = {}
    field_types: dict[str, dict[str, TType]] = {}
    enum_names: set[str] = set()
    fn_names: set[str] = set()
    for decl in module.decls:
        match decl:
            case TFnDecl():
                fn_names.add(decl.name)
            case TStructDecl():
                fnames: list[str] = []
                ftypes: dict[str, TType] = {}
                for f in decl.fields:
                    fnames.append(f.name)
                    if f.typ is not None:
                        ftypes[f.name] = f.typ
                struct_fields[decl.name] = fnames
                field_types[decl.name] = ftypes
                for m in decl.methods:
                    fn_names.add(m.name)
            case TInterfaceDecl():
                if decl.fields:
                    struct_fields[decl.name] = [f.name for f in decl.fields]
                    iftypes: dict[str, TType] = {}
                    for f in decl.fields:
                        if f.typ is not None:
                            iftypes[f.name] = f.typ
                    field_types[decl.name] = iftypes
            case TEnumDecl():
                enum_names.add(decl.name)
    emitter = _RubyEmitter(
        struct_names,
        fn_names,
        struct_fields,
        field_types,
        enum_names,
        module.strict_math,
        module.strict_tostring,
    )
    emitter.emit_module(module)
    return emitter.output()
