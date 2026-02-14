"""Ruby backend: Taytsh AST → Ruby source code."""

from __future__ import annotations

from .util import escape_string, to_snake
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
        "lambda",
        "proc",
        "loop",
        "raise",
        "fail",
        "catch",
        "throw",
        "format",
        "puts",
        "print",
        "p",
        "gets",
        "require",
        "load",
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
    "ValueError": "ArgumentError",
    "KeyError": "KeyError",
    "IndexError": "IndexError",
    "ZeroDivisionError": "ZeroDivisionError",
    "AssertError": "RuntimeError",
    "NilError": "RuntimeError",
    "IOError": "IOError",
}


def _safe_name(name: str) -> str:
    name = to_snake(name)
    if name in _RUBY_RESERVED:
        return name + "_"
    return name


def _restore_name(name: str, annotations: Ann) -> str:
    """Restore original Python name from annotation, then apply target safety."""
    key = "name.original." + name
    if key in annotations:
        return _safe_name(str(annotations[key]))
    return _safe_name(name)


def _safe_type_name(name: str) -> str:
    if name in _RUBY_BUILTINS:
        return name + "_"
    return name


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

_CMP_OPS = frozenset(("==", "!=", "<", ">", "<=", ">="))


def _needs_parens(child_op: str, parent_op: str, is_left: bool) -> bool:
    child_prec = _PRECEDENCE.get(child_op, 0)
    parent_prec = _PRECEDENCE.get(parent_op, 0)
    if child_prec < parent_prec:
        return True
    if child_prec == parent_prec and not is_left:
        return child_op in _CMP_OPS
    if parent_op in ("==", "!=") and child_op in _CMP_OPS:
        return True
    return False


# ============================================================
# IMPORT / NEEDS SCANNING
# ============================================================


def _scan_needs(module: TModule) -> tuple[bool, bool]:
    """Return (needs_set, needs_range_helper)."""
    needs_set = False
    needs_range = False
    for decl in module.decls:
        if isinstance(decl, (TFnDecl, TStructDecl)):
            s, r = _scan_decl_needs(decl)
            if s:
                needs_set = True
            if r:
                needs_range = True
    return needs_set, needs_range


def _scan_decl_needs(decl: TDecl) -> tuple[bool, bool]:
    needs_set = False
    needs_range = False
    stmts: list[TStmt] = []
    if isinstance(decl, TFnDecl):
        stmts = decl.body
    elif isinstance(decl, TStructDecl):
        for m in decl.methods:
            s, r = _scan_decl_needs(m)
            if s:
                needs_set = True
            if r:
                needs_range = True
        for fld in decl.fields:
            if isinstance(fld.typ, TSetType):
                needs_set = True
        return needs_set, needs_range
    for name in _collect_builtin_calls(stmts):
        if name in ("Set", "Add", "Remove"):
            needs_set = True
    _scan_stmts_for_needs(stmts, [needs_set, needs_range])
    return needs_set, needs_range


def _scan_stmts_for_needs(stmts: list[TStmt], flags: list[bool]) -> None:
    for stmt in stmts:
        _scan_stmt_for_needs(stmt, flags)


def _scan_stmt_for_needs(stmt: TStmt, flags: list[bool]) -> None:
    if isinstance(stmt, TLetStmt):
        if stmt.value is not None:
            _scan_expr_for_needs(stmt.value, flags)
    elif isinstance(stmt, TAssignStmt):
        _scan_expr_for_needs(stmt.value, flags)
    elif isinstance(stmt, TExprStmt):
        _scan_expr_for_needs(stmt.expr, flags)
    elif isinstance(stmt, TReturnStmt):
        if stmt.value is not None:
            _scan_expr_for_needs(stmt.value, flags)
    elif isinstance(stmt, TIfStmt):
        _scan_stmts_for_needs(stmt.then_body, flags)
        if stmt.else_body:
            _scan_stmts_for_needs(stmt.else_body, flags)
    elif isinstance(stmt, TWhileStmt):
        _scan_stmts_for_needs(stmt.body, flags)
    elif isinstance(stmt, TForStmt):
        _scan_stmts_for_needs(stmt.body, flags)
    elif isinstance(stmt, TTryStmt):
        _scan_stmts_for_needs(stmt.body, flags)
        for catch in stmt.catches:
            _scan_stmts_for_needs(catch.body, flags)
    elif isinstance(stmt, TMatchStmt):
        for case in stmt.cases:
            _scan_stmts_for_needs(case.body, flags)
        if stmt.default:
            _scan_stmts_for_needs(stmt.default.body, flags)


def _scan_expr_for_needs(expr: TExpr, flags: list[bool]) -> None:
    if isinstance(expr, TSetLit):
        flags[0] = True
    elif isinstance(expr, TCall):
        if isinstance(expr.func, TVar) and expr.func.name == "Set":
            flags[0] = True
        for a in expr.args:
            _scan_expr_for_needs(a.value, flags)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _scan_expr_for_needs(e, flags)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _scan_expr_for_needs(k, flags)
            _scan_expr_for_needs(v, flags)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _scan_expr_for_needs(e, flags)
    elif isinstance(expr, TBinaryOp):
        _scan_expr_for_needs(expr.left, flags)
        _scan_expr_for_needs(expr.right, flags)
    elif isinstance(expr, TUnaryOp):
        _scan_expr_for_needs(expr.operand, flags)
    elif isinstance(expr, TTernary):
        _scan_expr_for_needs(expr.cond, flags)
        _scan_expr_for_needs(expr.then_expr, flags)
        _scan_expr_for_needs(expr.else_expr, flags)


def _collect_builtin_calls(stmts: list[TStmt]) -> set[str]:
    out: set[str] = set()
    for stmt in stmts:
        _collect_builtin_calls_stmt(stmt, out)
    return out


def _collect_builtin_calls_stmt(stmt: TStmt, out: set[str]) -> None:
    if isinstance(stmt, TExprStmt):
        _collect_builtin_calls_expr(stmt.expr, out)
    elif isinstance(stmt, TLetStmt):
        if stmt.value is not None:
            _collect_builtin_calls_expr(stmt.value, out)
    elif isinstance(stmt, TAssignStmt):
        _collect_builtin_calls_expr(stmt.value, out)
    elif isinstance(stmt, TOpAssignStmt):
        _collect_builtin_calls_expr(stmt.value, out)
    elif isinstance(stmt, TTupleAssignStmt):
        _collect_builtin_calls_expr(stmt.value, out)
    elif isinstance(stmt, TReturnStmt):
        if stmt.value is not None:
            _collect_builtin_calls_expr(stmt.value, out)
    elif isinstance(stmt, TThrowStmt):
        _collect_builtin_calls_expr(stmt.expr, out)
    elif isinstance(stmt, TIfStmt):
        _collect_builtin_calls_expr(stmt.cond, out)
        for s in stmt.then_body:
            _collect_builtin_calls_stmt(s, out)
        if stmt.else_body is not None:
            for s in stmt.else_body:
                _collect_builtin_calls_stmt(s, out)
    elif isinstance(stmt, TWhileStmt):
        _collect_builtin_calls_expr(stmt.cond, out)
        for s in stmt.body:
            _collect_builtin_calls_stmt(s, out)
    elif isinstance(stmt, TForStmt):
        if isinstance(stmt.iterable, TRange):
            for a in stmt.iterable.args:
                _collect_builtin_calls_expr(a, out)
        else:
            _collect_builtin_calls_expr(stmt.iterable, out)
        for s in stmt.body:
            _collect_builtin_calls_stmt(s, out)
    elif isinstance(stmt, TTryStmt):
        for s in stmt.body:
            _collect_builtin_calls_stmt(s, out)
        for catch in stmt.catches:
            for s in catch.body:
                _collect_builtin_calls_stmt(s, out)
        if stmt.finally_body is not None:
            for s in stmt.finally_body:
                _collect_builtin_calls_stmt(s, out)
    elif isinstance(stmt, TMatchStmt):
        _collect_builtin_calls_expr(stmt.expr, out)
        for case in stmt.cases:
            for s in case.body:
                _collect_builtin_calls_stmt(s, out)
        if stmt.default is not None:
            for s in stmt.default.body:
                _collect_builtin_calls_stmt(s, out)


def _collect_builtin_calls_expr(expr: TExpr, out: set[str]) -> None:
    if isinstance(expr, TCall):
        if isinstance(expr.func, TVar) and expr.func.name in BUILTIN_NAMES:
            out.add(expr.func.name)
        _collect_builtin_calls_expr(expr.func, out)
        for a in expr.args:
            _collect_builtin_calls_expr(a.value, out)
    elif isinstance(expr, TBinaryOp):
        _collect_builtin_calls_expr(expr.left, out)
        _collect_builtin_calls_expr(expr.right, out)
    elif isinstance(expr, TUnaryOp):
        _collect_builtin_calls_expr(expr.operand, out)
    elif isinstance(expr, TTernary):
        _collect_builtin_calls_expr(expr.cond, out)
        _collect_builtin_calls_expr(expr.then_expr, out)
        _collect_builtin_calls_expr(expr.else_expr, out)
    elif isinstance(expr, TFieldAccess):
        _collect_builtin_calls_expr(expr.obj, out)
    elif isinstance(expr, TTupleAccess):
        _collect_builtin_calls_expr(expr.obj, out)
    elif isinstance(expr, TIndex):
        _collect_builtin_calls_expr(expr.obj, out)
        _collect_builtin_calls_expr(expr.index, out)
    elif isinstance(expr, TSlice):
        _collect_builtin_calls_expr(expr.obj, out)
        _collect_builtin_calls_expr(expr.low, out)
        _collect_builtin_calls_expr(expr.high, out)
    elif isinstance(expr, TListLit):
        for e in expr.elements:
            _collect_builtin_calls_expr(e, out)
    elif isinstance(expr, TTupleLit):
        for e in expr.elements:
            _collect_builtin_calls_expr(e, out)
    elif isinstance(expr, TSetLit):
        for e in expr.elements:
            _collect_builtin_calls_expr(e, out)
    elif isinstance(expr, TMapLit):
        for k, v in expr.entries:
            _collect_builtin_calls_expr(k, out)
            _collect_builtin_calls_expr(v, out)
    elif isinstance(expr, TFnLit):
        if isinstance(expr.body, list):
            for s in expr.body:
                _collect_builtin_calls_stmt(s, out)
        else:
            _collect_builtin_calls_expr(expr.body, out)


# ============================================================
# EMITTER
# ============================================================


class _RubyEmitter:
    def __init__(self, struct_names: set[str], fn_names: set[str]) -> None:
        self.struct_names = struct_names
        self.fn_names = fn_names
        self.indent: int = 0
        self.lines: list[str] = []
        self.self_name: str | None = None
        self.var_types: dict[str, TType] = {}
        self._needs_set: bool = False
        self._needs_range_helper: bool = False

    def _line(self, text: str = "") -> None:
        if text:
            self.lines.append("  " * self.indent + text)
        else:
            self.lines.append("")

    def output(self) -> str:
        return "\n".join(self.lines)

    # ── Module ────────────────────────────────────────────────

    def emit_module(self, module: TModule) -> None:
        self._line("# frozen_string_literal: true")
        self._line()
        import_insert_pos = len(self.lines)
        need_blank = False
        for decl in module.decls:
            if isinstance(decl, TInterfaceDecl):
                continue
            if need_blank:
                self._line()
            if isinstance(decl, TEnumDecl):
                self._emit_enum(decl)
                need_blank = True
            elif isinstance(decl, TStructDecl):
                self._emit_struct(decl)
                need_blank = True
            elif isinstance(decl, TFnDecl):
                self._emit_fn(decl)
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
            self._emit_initialize(decl.fields, is_error=True)
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
            self._emit_initialize(decl.fields, is_error=False)
        for i, method in enumerate(decl.methods):
            if i > 0 or decl.fields:
                self._line()
            self._emit_method(method)
        self.indent -= 1
        self._line("end")

    def _emit_initialize(self, fields: list[TFieldDecl], is_error: bool) -> None:
        params: list[str] = []
        for f in fields:
            name = _safe_name(f.name)
            default = self._zero_value(f.typ)
            params.append(name + ": " + default)
        self._line("def initialize(" + ", ".join(params) + ")")
        self.indent += 1
        if is_error:
            msg_field = None
            for f in fields:
                if f.name == "message":
                    msg_field = f
                    break
            if msg_field is not None:
                self._line("super(" + _safe_name(msg_field.name) + ")")
            else:
                self._line("super()")
        for f in fields:
            name = _safe_name(f.name)
            self._line("@" + name + " = " + name)
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
        return "nil"

    # ── Function / Method ─────────────────────────────────────

    def _emit_fn(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
        params = self._params(decl.params, with_self=False)
        self._line("def " + _safe_name(decl.name) + "(" + params + ")")
        self.indent += 1
        if not decl.body:
            self._line("nil")
        self._emit_stmts(decl.body)
        self.indent -= 1
        self._line("end")
        self.var_types = old_var_types

    def _emit_method(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
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
        self.indent -= 1
        self._line("end")
        self.var_types = old_var_types

    def _params(self, params: list[TParam], with_self: bool) -> str:
        parts: list[str] = []
        for p in params:
            if p.typ is None:
                continue
            parts.append(_restore_name(p.name, p.annotations))
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
                        comp = self._try_comprehension(stmt, next_stmt, prov)
                        if comp is not None:
                            self._line(comp)
                            i += 2
                            continue
            self._emit_stmt(stmt)
            i += 1

    def _try_comprehension(
        self, let_stmt: TLetStmt, for_stmt: TForStmt, prov: str
    ) -> str | None:
        acc = _restore_name(let_stmt.name, let_stmt.annotations)
        binding = for_stmt.binding
        binders = ", ".join(_restore_name(b, for_stmt.annotations) for b in binding)
        if isinstance(for_stmt.iterable, TRange):
            args = ", ".join(self._expr(a) for a in for_stmt.iterable.args)
            self._needs_range_helper = True
            iterable = "_range(" + args + ")"
        else:
            iterable = self._expr(for_stmt.iterable)
        iter_is_map = not isinstance(for_stmt.iterable, TRange) and self._is_map_type(
            for_stmt.iterable
        )
        if iter_is_map and len(binding) == 2:
            pass  # Ruby hash.each gives |k, v|
        elif len(binding) == 2 and not isinstance(for_stmt.iterable, TRange):
            iterable += ".each_with_index"
            binders = (
                _restore_name(binding[1], for_stmt.annotations)
                + ", "
                + _restore_name(binding[0], for_stmt.annotations)
            )
        body = for_stmt.body
        if prov == "list_comprehension":
            if len(body) == 1 and isinstance(body[0], TExprStmt):
                call = body[0].expr
                if self._is_append_to(call, let_stmt.name):
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
            if len(body) == 1 and isinstance(body[0], TIfStmt):
                if_stmt = body[0]
                if len(if_stmt.then_body) == 1 and isinstance(
                    if_stmt.then_body[0], TExprStmt
                ):
                    call = if_stmt.then_body[0].expr
                    if self._is_append_to(call, let_stmt.name):
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
                if self._is_add_to(call, let_stmt.name):
                    val = self._expr(call.args[1].value)
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

    def _is_append_to(self, expr: TExpr, name: str) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Append"
            and isinstance(expr.args[0].value, TVar)
            and expr.args[0].value.name == name
        )

    def _is_add_to(self, expr: TExpr, name: str) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Add"
            and isinstance(expr.args[0].value, TVar)
            and expr.args[0].value.name == name
        )

    def _emit_stmt(self, stmt: TStmt) -> None:
        if isinstance(stmt, TLetStmt):
            self._emit_let(stmt)
        elif isinstance(stmt, TAssignStmt):
            self._line(self._expr(stmt.target) + " = " + self._expr(stmt.value))
        elif isinstance(stmt, TTupleAssignStmt):
            self._emit_tuple_assign(stmt)
        elif isinstance(stmt, TOpAssignStmt):
            self._line(
                self._expr(stmt.target) + " " + stmt.op + " " + self._expr(stmt.value)
            )
        elif isinstance(stmt, TExprStmt):
            self._emit_expr_stmt(stmt)
        elif isinstance(stmt, TReturnStmt):
            if stmt.value is not None:
                self._line("return " + self._expr(stmt.value))
            else:
                self._line("return")
        elif isinstance(stmt, TThrowStmt):
            self._line("raise " + self._expr(stmt.expr))
        elif isinstance(stmt, TBreakStmt):
            self._line("break")
        elif isinstance(stmt, TContinueStmt):
            self._line("next")
        elif isinstance(stmt, TIfStmt):
            self._emit_if(stmt)
        elif isinstance(stmt, TWhileStmt):
            self._emit_while(stmt)
        elif isinstance(stmt, TForStmt):
            self._emit_for(stmt)
        elif isinstance(stmt, TTryStmt):
            self._emit_try(stmt)
        elif isinstance(stmt, TMatchStmt):
            self._emit_match(stmt)

    def _emit_let(self, stmt: TLetStmt) -> None:
        safe = _restore_name(stmt.name, stmt.annotations)
        self.var_types[stmt.name] = stmt.typ
        unused = stmt.annotations.get("liveness.initial_value_unused", False)
        if stmt.value is not None and not unused:
            self._line(safe + " = " + self._expr(stmt.value))
        else:
            self._line(safe + " = nil")

    def _emit_tuple_assign(self, stmt: TTupleAssignStmt) -> None:
        unused_str = str(stmt.annotations.get("liveness.tuple_unused_indices", ""))
        unused_indices: set[int] = set()
        if unused_str:
            for s in unused_str.split(","):
                if s:
                    unused_indices.add(int(s))
        parts: list[str] = []
        for i, t in enumerate(stmt.targets):
            if i in unused_indices:
                parts.append("_")
            else:
                parts.append(self._expr(t))
        self._line(", ".join(parts) + " = " + self._expr(stmt.value))

    def _emit_expr_stmt(self, stmt: TExprStmt) -> None:
        expr = stmt.expr
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            name = expr.func.name
            if name == "Assert":
                args = expr.args
                cond = self._expr(args[0].value)
                if len(args) > 1:
                    msg = self._expr(args[1].value)
                    self._line('raise "AssertionError: #{' + msg + '}" unless ' + cond)
                else:
                    self._line('raise "AssertionError" unless ' + cond)
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
        self._line("if " + self._expr(stmt.cond))
        self.indent += 1
        if not stmt.then_body:
            self._line("nil")
        self._emit_stmts(stmt.then_body)
        self.indent -= 1
        self._emit_else_body(stmt.else_body)
        self._line("end")

    def _emit_else_body(self, else_body: list[TStmt] | None) -> None:
        if else_body is None or not else_body:
            return
        if len(else_body) == 1 and isinstance(else_body[0], TIfStmt):
            elif_stmt = else_body[0]
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
        if isinstance(stmt.iterable, TRange):
            self._needs_range_helper = True
            args = ", ".join(self._expr(a) for a in stmt.iterable.args)
            binders = ", ".join(_restore_name(b, ann) for b in binding)
            self._line("_range(" + args + ").each do |" + binders + "|")
        elif len(binding) == 1:
            self._line(
                self._expr(stmt.iterable)
                + ".each do |"
                + _restore_name(binding[0], ann)
                + "|"
            )
        elif len(binding) == 2:
            iter_is_map = self._is_map_type(stmt.iterable)
            if iter_is_map:
                self._line(
                    self._expr(stmt.iterable)
                    + ".each do |"
                    + _restore_name(binding[0], ann)
                    + ", "
                    + _restore_name(binding[1], ann)
                    + "|"
                )
            else:
                self._line(
                    self._expr(stmt.iterable)
                    + ".each_with_index do |"
                    + _restore_name(binding[1], ann)
                    + ", "
                    + _restore_name(binding[0], ann)
                    + "|"
                )
        else:
            binders = ", ".join(_restore_name(b, ann) for b in binding)
            self._line(self._expr(stmt.iterable) + ".each do |" + binders + "|")
        self.indent += 1
        if not stmt.body:
            self._line("nil")
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("end")

    def _is_map_type(self, expr: TExpr) -> bool:
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TMapType)
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
        unused = catch.annotations.get("liveness.catch_var_unused", False)
        if unused:
            self._line("rescue " + type_str)
        else:
            self._line(
                "rescue "
                + type_str
                + " => "
                + _restore_name(catch.name, catch.annotations)
            )
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

    def _emit_match_case(self, case: TMatchCase, expr_str: str, first: bool) -> None:
        pat = case.pattern
        keyword = "if" if first else "elsif"
        if isinstance(pat, TPatternType):
            type_name = self._type_name_for_check(pat.type_name)
            self._line(keyword + " " + expr_str + ".is_a?(" + type_name + ")")
            self.indent += 1
            unused = pat.annotations.get("liveness.match_var_unused", False)
            if not unused:
                self._line(_safe_name(pat.name) + " = " + expr_str)
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
            unused = default.annotations.get("liveness.match_var_unused", False)
            if not unused:
                self._line(_safe_name(default.name) + " = " + expr_str)
        if not default.body:
            self._line("nil")
        self._emit_stmts(default.body)
        self.indent -= 1
        if first:
            self._line("end")

    def _type_name_for_check(self, typ: TType) -> str:
        if isinstance(typ, TIdentType):
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
            if expr.name in self.fn_names:
                return "method(:" + _restore_name(expr.name, expr.annotations) + ")"
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
            return self._expr(expr.obj) + "[" + self._expr(expr.index) + "]"
        if isinstance(expr, TSlice):
            return self._slice(expr)
        if isinstance(expr, TBinaryOp):
            return self._binary(expr)
        if isinstance(expr, TUnaryOp):
            return self._unary(expr)
        if isinstance(expr, TTernary):
            return (
                self._expr(expr.cond)
                + " ? "
                + self._expr(expr.then_expr)
                + " : "
                + self._expr(expr.else_expr)
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

    def _is_zero(self, expr: TExpr) -> bool:
        return isinstance(expr, TIntLit) and expr.value == 0

    def _is_len_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Len"
        )

    def _binary(self, expr: TBinaryOp) -> str:
        op = expr.op
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
        if prov == "string_multiply" or prov == "list_multiply":
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
            _restore_name(p.name, p.annotations)
            for p in expr.params
            if p.typ is not None
        )
        if isinstance(expr.body, list):
            self._line("_fn = lambda { |" + params + "|")
            self.indent += 1
            for s in expr.body:
                self._emit_stmt(s)
            self.indent -= 1
            self._line("}")
            return "_fn"
        return "lambda { |" + params + "| " + self._expr(expr.body) + " }"

    # ── Calls ─────────────────────────────────────────────────

    def _call(self, expr: TCall) -> str:
        func = expr.func
        args = expr.args
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
                return _safe_name(func.name) + "(" + arg_strs + ")"
            return _safe_name(func.name) + "()"
        if arg_strs:
            return fn_expr + ".call(" + arg_strs + ")"
        return fn_expr + ".call"

    def _struct_call(self, name: str, args: list[TArg]) -> str:
        parts: list[str] = []
        for a in args:
            if a.name is not None:
                parts.append(_safe_name(a.name) + ": " + self._expr(a.value))
            else:
                parts.append(self._expr(a.value))
        return _safe_type_name(name) + ".new(" + ", ".join(parts) + ")"

    def _method_call(self, func: TFieldAccess, args: list[TArg]) -> str:
        obj_str = self._expr(func.obj)
        if isinstance(func.obj, (TBinaryOp, TUnaryOp, TTernary)):
            obj_str = "(" + obj_str + ")"
        arg_strs = ", ".join(self._expr(a.value) for a in args)
        if arg_strs:
            return obj_str + "." + _safe_name(func.field) + "(" + arg_strs + ")"
        return obj_str + "." + _safe_name(func.field)

    def _builtin_call(self, name: str, args: list[TArg], call: TCall) -> str:
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
        if name == "IndexOf":
            obj = self._a(args, 0)
            val = self._a(args, 1)
            return "(" + obj + ".index(" + val + ") || -1)"
        # String operations
        if name == "Upper":
            return self._a(args, 0) + ".upcase"
        if name == "Lower":
            return self._a(args, 0) + ".downcase"
        if name == "Trim":
            a0 = self._a(args, 0)
            a1 = self._a(args, 1)
            return a0 + ".gsub(/\\A[" + a1 + "]+|[" + a1 + "]+\\z/, '')"
        if name == "TrimStart":
            a0 = self._a(args, 0)
            a1 = self._a(args, 1)
            return a0 + ".gsub(/\\A[" + a1 + "]+/, '')"
        if name == "TrimEnd":
            a0 = self._a(args, 0)
            a1 = self._a(args, 1)
            return a0 + ".gsub(/[" + a1 + "]+\\z/, '')"
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
            return self._a(args, 0) + ".join(" + self._a(args, 1) + ")"
        if name == "Find":
            return "(" + self._a(args, 0) + ".index(" + self._a(args, 1) + ") || -1)"
        if name == "RFind":
            return "(" + self._a(args, 0) + ".rindex(" + self._a(args, 1) + ") || -1)"
        if name == "Count":
            return self._a(args, 0) + ".scan(" + self._a(args, 1) + ").length"
        if name == "Replace":
            return (
                self._a(args, 0)
                + ".gsub("
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + ")"
            )
        if name == "StartsWith":
            return self._a(args, 0) + ".start_with?(" + self._a(args, 1) + ")"
        if name == "EndsWith":
            return self._a(args, 0) + ".end_with?(" + self._a(args, 1) + ")"
        if name == "IsDigit":
            return self._a(args, 0) + r".match?(/\A\d+\z/)"
        if name == "IsAlpha":
            return self._a(args, 0) + r".match?(/\A[[:alpha:]]+\z/)"
        if name == "IsAlnum":
            return self._a(args, 0) + r".match?(/\A[[:alnum:]]+\z/)"
        if name == "IsSpace":
            return self._a(args, 0) + r".match?(/\A\s+\z/)"
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
            return self._a(args, 0) + " * " + self._a(args, 1)
        if name == "RemovePrefix":
            return self._a(args, 0) + ".delete_prefix(" + self._a(args, 1) + ")"
        if name == "RemoveSuffix":
            return self._a(args, 0) + ".delete_suffix(" + self._a(args, 1) + ")"
        # Encode / Decode
        if name == "Encode":
            return self._a(args, 0) + '.encode("utf-8").bytes'
        if name == "Decode":
            return self._a(args, 0) + ".pack('C*').force_encoding('UTF-8')"
        # Set operations
        if name == "Add":
            self._needs_set = True
            return self._a(args, 0) + ".add(" + self._a(args, 1) + ")"
        if name == "Remove":
            self._needs_set = True
            return self._a(args, 0) + ".delete(" + self._a(args, 1) + ")"
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
            return self._a(args, 0) + ".length"
        if name == "Abs":
            a = self._a(args, 0)
            if isinstance(args[0].value, (TBinaryOp, TUnaryOp, TTernary)):
                return "(" + a + ").abs"
            return a + ".abs"
        if name == "Min":
            return "[" + self._a(args, 0) + ", " + self._a(args, 1) + "].min"
        if name == "Max":
            return "[" + self._a(args, 0) + ", " + self._a(args, 1) + "].max"
        if name == "Sum":
            return self._a(args, 0) + ".sum"
        if name == "Round":
            return self._a(args, 0) + ".round"
        if name == "DivMod":
            return self._a(args, 0) + ".divmod(" + self._a(args, 1) + ")"
        if name == "Sorted":
            return self._a(args, 0) + ".sort"
        if name == "Reversed":
            return self._a(args, 0) + ".reverse"
        if name == "Map":
            if len(args) == 0:
                return "{}"
            return self._a(args, 1) + ".map { |_e| " + self._a(args, 0) + ".call(_e) }"
        if name == "Set":
            self._needs_set = True
            if len(args) == 0:
                return "Set.new"
            return "Set.new(" + self._a(args, 0) + ".to_a)"
        if name == "ToString":
            return self._a(args, 0) + ".to_s"
        if name == "ParseInt":
            base = self._a(args, 1)
            if base == "10":
                return self._a(args, 0) + ".to_i"
            return self._a(args, 0) + ".to_i(" + base + ")"
        if name == "ParseFloat":
            return self._a(args, 0) + ".to_f"
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
            return self._a(args, 0)
        if name == "IntToByte":
            return self._a(args, 0)
        if name == "Unwrap":
            return self._a(args, 0)
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
        if name == "ReadLine":
            return "$stdin.gets&.chomp"
        if name == "ReadAll":
            return "$stdin.read"
        if name == "ReadBytes":
            return "$stdin.binmode.read.bytes"
        if name == "ReadBytesN":
            return "$stdin.binmode.read(" + self._a(args, 0) + ").bytes"
        if name == "Args":
            return "ARGV"
        if name == "GetEnv":
            return "ENV.fetch(" + self._a(args, 0) + ', "")'
        if name == "Exit":
            return "exit(" + self._a(args, 0) + ")"
        # Operator forms
        if name == "Pow":
            return self._a(args, 0) + " ** " + self._a(args, 1)
        if name == "Contains":
            prov = call.annotations.get("provenance", "")
            if prov == "in_operator":
                return self._a(args, 0) + ".include?(" + self._a(args, 1) + ")"
            return self._a(args, 0) + ".include?(" + self._a(args, 1) + ")"
        if name == "Concat":
            return self._a(args, 0) + " + " + self._a(args, 1)
        if name == "Format":
            return self._format_call(args)
        if name == "Assert":
            cond = self._a(args, 0)
            if len(args) > 1:
                return (
                    'raise "AssertionError: #{' + self._a(args, 1) + '}" unless ' + cond
                )
            return 'raise "AssertionError" unless ' + cond
        # Fallback
        arg_strs = ", ".join(self._expr(a.value) for a in args)
        return _safe_name(name) + "(" + arg_strs + ")"

    def _a(self, args: list[TArg], i: int) -> str:
        return self._expr(args[i].value)

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

    def _format_call(self, args: list[TArg]) -> str:
        template_expr = args[0].value
        if not isinstance(template_expr, TStringLit):
            arg_strs = ", ".join(self._expr(a.value) for a in args)
            return "format_(" + arg_strs + ")"
        template = template_expr.value
        fmt_args = args[1:]
        markers: dict[str, int] = {}
        result = template
        for i in range(len(fmt_args)):
            marker = "\x00PH" + str(i) + "\x00"
            markers[marker] = i
            result = result.replace("{}", marker, 1)
        result = _escape_ruby_string(result)
        for marker, i in markers.items():
            result = result.replace(marker, "#{" + self._expr(fmt_args[i].value) + "}")
        return '"' + result + '"'


# ============================================================
# PUBLIC API
# ============================================================


def emit_ruby(module: TModule) -> str:
    struct_names = {
        decl.name for decl in module.decls if isinstance(decl, TStructDecl)
    } | set(BUILTIN_STRUCTS.keys())
    fn_names: set[str] = set()
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            fn_names.add(decl.name)
        elif isinstance(decl, TStructDecl):
            for m in decl.methods:
                fn_names.add(m.name)
    emitter = _RubyEmitter(struct_names, fn_names)
    emitter.emit_module(module)
    return emitter.output()
