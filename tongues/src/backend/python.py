"""Python backend: Taytsh AST → Python source code."""

from __future__ import annotations

from .ordering import order_decls
from .util import escape_string
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
# PYTHON BUILTINS
# ============================================================

_PYTHON_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "breakpoint",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "compile",
        "complex",
        "delattr",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "eval",
        "exec",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "globals",
        "hasattr",
        "hash",
        "help",
        "hex",
        "id",
        "input",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "locals",
        "map",
        "max",
        "memoryview",
        "min",
        "next",
        "object",
        "oct",
        "open",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "vars",
        "zip",
        "dataclass",
        "field",
    }
)


def _safe_name(name: str) -> str:
    if name in _PYTHON_BUILTINS:
        return name + "_"
    return name


def _restore_name(name: str, annotations: Ann) -> str:
    """Restore original Python name from annotation, then apply target safety."""
    key = "name.original." + name
    if key in annotations:
        return _safe_name(annotations[key])
    return _safe_name(name)


# ============================================================
# OPERATOR MAPS
# ============================================================

_PRECEDENCE: dict[str, int] = {
    "or": 1,
    "||": 1,
    "and": 2,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 3,
    ">": 3,
    "<=": 3,
    ">=": 3,
    "in": 3,
    "not in": 3,
    "|": 4,
    "^": 5,
    "&": 6,
    "<<": 7,
    ">>": 7,
    "+": 8,
    "-": 8,
    "*": 9,
    "/": 9,
    "//": 9,
    "%": 9,
    "**": 11,
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


def _scan_imports(
    module: TModule,
) -> tuple[bool, bool, bool, bool, bool]:
    """Return (needs_sys, needs_dataclass, needs_field, needs_math, needs_os)."""
    needs_sys = False
    needs_dataclass = False
    needs_field = False
    needs_math = False
    needs_os = False
    for decl in module.decls:
        if isinstance(decl, TStructDecl):
            if decl.name not in BUILTIN_STRUCTS:
                needs_dataclass = True
                for fld in decl.fields:
                    if isinstance(fld.typ, (TListType, TMapType, TSetType)):
                        needs_field = True
                    if fld.has_default and isinstance(fld.typ, TIdentType):
                        needs_field = True
        if isinstance(decl, TInterfaceDecl) and decl.fields:
            needs_dataclass = True
            for fld in decl.fields:
                if isinstance(fld.typ, (TListType, TMapType, TSetType)):
                    needs_field = True
                if fld.has_default and isinstance(fld.typ, TIdentType):
                    needs_field = True
        if isinstance(decl, (TFnDecl, TStructDecl)):
            r_sys, r_math, r_os = _scan_decl_builtins(decl)
            if r_sys:
                needs_sys = True
            if r_math:
                needs_math = True
            if r_os:
                needs_os = True
    return needs_sys, needs_dataclass, needs_field, needs_math, needs_os


_SYS_BUILTINS = frozenset(
    {
        "WriteErr",
        "WritelnErr",
        "ReadAll",
        "ReadBytes",
        "ReadBytesN",
        "Args",
        "Exit",
    }
)

_MATH_BUILTINS = frozenset({"IsNaN", "IsInf", "Sqrt", "Floor", "Ceil"})

_OS_BUILTINS = frozenset({"GetEnv"})


def _scan_decl_builtins(decl: TDecl) -> tuple[bool, bool, bool]:
    """Scan a declaration for sys/math/os builtin usage."""
    needs_sys = False
    needs_math = False
    needs_os = False
    stmts: list[TStmt] = []
    if isinstance(decl, TFnDecl):
        stmts = decl.body
    elif isinstance(decl, TStructDecl):
        for m in decl.methods:
            r = _scan_decl_builtins(m)
            if r[0]:
                needs_sys = True
            if r[1]:
                needs_math = True
            if r[2]:
                needs_os = True
        return needs_sys, needs_math, needs_os
    for name in _collect_builtin_calls(stmts):
        if name in _SYS_BUILTINS:
            needs_sys = True
        if name in _MATH_BUILTINS:
            needs_math = True
        if name in _OS_BUILTINS:
            needs_os = True
    return needs_sys, needs_math, needs_os


def _collect_builtin_calls(stmts: list[TStmt]) -> set[str]:
    """Collect builtin function names called in statements."""
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
        for s in expr.body:
            _collect_builtin_calls_stmt(s, out)


# ============================================================
# EMITTER
# ============================================================


_STRICT_INT_BINARY = {
    "+": "checked_add_i64",
    "-": "checked_sub_i64",
    "*": "checked_mul_i64",
    "/": "checked_div_i64",
    "%": "checked_rem_i64",
    "<<": "checked_shl_i64",
    ">>": "checked_shr_i64",
    ">>>": "logical_shr_i64",
}

_STRICT_INT_COMPOUND = {
    "+=": "checked_add_i64",
    "-=": "checked_sub_i64",
    "*=": "checked_mul_i64",
}


class _PythonEmitter:
    def __init__(
        self,
        struct_names: set[str],
        struct_fields: dict[str, list[str]],
        strict_math: bool = False,
    ) -> None:
        self.struct_names = struct_names
        self.struct_fields = struct_fields
        self.strict_math = strict_math
        self.indent: int = 0
        self.lines: list[str] = []
        self.self_name: str | None = None
        self.var_types: dict[str, TType] = {}

    def _line(self, text: str = "") -> None:
        if text:
            self.lines.append("    " * self.indent + text)
        else:
            self.lines.append("")

    def output(self) -> str:
        return "\n".join(self.lines)

    # ── Module ────────────────────────────────────────────────

    def emit_module(self, module: TModule) -> None:
        needs_sys, needs_dataclass, needs_field, needs_math, needs_os = _scan_imports(
            module
        )
        plain_imports: list[str] = []
        from_imports: list[str] = ["from __future__ import annotations"]
        if needs_sys:
            plain_imports.append("import sys")
        if needs_os:
            plain_imports.append("import os")
        if needs_math:
            plain_imports.append("import math")
        if needs_dataclass and needs_field:
            from_imports.append("from dataclasses import dataclass, field")
        elif needs_dataclass:
            from_imports.append("from dataclasses import dataclass")
        if plain_imports or from_imports:
            for line in from_imports:
                self._line(line)
            if plain_imports and from_imports:
                self._line()
            for line in plain_imports:
                self._line(line)
            self._line()
        self._line()
        need_blank = False
        for decl in order_decls(module.decls):
            if isinstance(decl, TInterfaceDecl):
                if need_blank:
                    self._line()
                    self._line()
                if decl.fields:
                    self._line("@dataclass")
                self._line("class " + decl.name + ":")
                self.indent += 1
                if not decl.fields:
                    self._line("pass")
                for fld in decl.fields:
                    self._emit_field(fld)
                self.indent -= 1
                need_blank = True
                continue
            if need_blank:
                self._line()
                self._line()
            if isinstance(decl, TStructDecl):
                self._emit_struct(decl)
                need_blank = True
            elif isinstance(decl, TEnumDecl):
                self._emit_enum(decl)
                need_blank = True
            elif isinstance(decl, TLetStmt):
                self._emit_let(decl)
                need_blank = True
            elif isinstance(decl, TFnDecl):
                self._emit_fn(decl)
                need_blank = True

    # ── Enum ──────────────────────────────────────────────────

    def _emit_enum(self, decl: TEnumDecl) -> None:
        self._line("class " + decl.name + ":")
        self.indent += 1
        for i, variant in enumerate(decl.variants):
            self._line(variant + " = " + str(i))
        self.indent -= 1

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
        self._line("class " + decl.name + "(Exception):")
        self.indent += 1
        params = ["self"]
        for fld in decl.fields:
            params.append(_safe_name(fld.name) + ": " + self._type(fld.typ))
        self._line("def __init__(" + ", ".join(params) + ") -> None:")
        self.indent += 1
        if len(decl.fields) > 0:
            msg_field: TFieldDecl | None = None
            for fld in decl.fields:
                if fld.name == "message" or fld.name == "msg":
                    msg_field = fld
                    break
            if msg_field is not None:
                self._line("super().__init__(" + _safe_name(msg_field.name) + ")")
            else:
                self._line("super().__init__()")
            for fld in decl.fields:
                safe = _safe_name(fld.name)
                self._line("self." + safe + " = " + safe)
        else:
            self._line("pass")
        self.indent -= 1
        for i, method in enumerate(decl.methods):
            self._line()
            self._emit_method(method)
        self.indent -= 1

    def _emit_data_struct(self, decl: TStructDecl) -> None:
        self._line("@dataclass")
        bases: list[str] = []
        if decl.parent is not None:
            bases.append(decl.parent)
        if bases:
            self._line("class " + decl.name + "(" + ", ".join(bases) + "):")
        else:
            self._line("class " + decl.name + ":")
        self.indent += 1
        if len(decl.fields) == 0 and len(decl.methods) == 0:
            self._line("pass")
        for fld in decl.fields:
            self._emit_field(fld)
        for i, method in enumerate(decl.methods):
            if i > 0 or len(decl.fields) > 0:
                self._line()
            self._emit_method(method)
        self.indent -= 1

    def _emit_field(self, fld: TFieldDecl) -> None:
        typ_str = self._type(fld.typ)
        default = self._field_default(fld.typ, fld.has_default)
        self._line(_safe_name(fld.name) + ": " + typ_str + " = " + default)

    def _field_default(self, typ: TType, has_default: bool = False) -> str:
        if isinstance(typ, TListType):
            return "field(default_factory=list)"
        if isinstance(typ, TMapType):
            return "field(default_factory=dict)"
        if isinstance(typ, TSetType):
            return "field(default_factory=set)"
        if (
            has_default
            and isinstance(typ, TIdentType)
            and typ.name in self.struct_names
        ):
            return "field(default_factory=" + typ.name + ")"
        return self._zero_value(typ)

    def _zero_value(self, typ: TType) -> str:
        if isinstance(typ, TPrimitive):
            if typ.kind == "int" or typ.kind == "byte":
                return "0"
            if typ.kind == "float":
                return "0.0"
            if typ.kind == "bool":
                return "False"
            if typ.kind == "string" or typ.kind == "rune":
                return '""'
            if typ.kind == "bytes":
                return 'b""'
        return "None"

    # ── Function / Method ─────────────────────────────────────

    def _emit_fn(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
        params = self._params(decl.params, with_self=False)
        ret = self._type(decl.ret)
        fname = "main" if decl.name == "Main" else decl.name
        self._line("def " + fname + "(" + params + ") -> " + ret + ":")
        self.indent += 1
        if not decl.body:
            self._line("pass")
        self._emit_stmts(decl.body)
        self.indent -= 1
        self.var_types = old_var_types

    def _emit_method(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
        params = self._params(decl.params, with_self=True)
        ret = self._type(decl.ret)
        self._line("def " + decl.name + "(" + params + ") -> " + ret + ":")
        self.indent += 1
        old_self = self.self_name
        if len(decl.params) > 0 and decl.params[0].typ is None:
            self.self_name = decl.params[0].name
        if len(decl.body) == 0:
            self._line("pass")
        self._emit_stmts(decl.body)
        self.self_name = old_self
        self.indent -= 1
        self.var_types = old_var_types

    def _params(self, params: list[TParam], with_self: bool) -> str:
        parts: list[str] = []
        for p in params:
            if p.typ is None:
                if with_self:
                    parts.append("self")
                continue
            s = _restore_name(p.name, p.annotations) + ": " + self._type(p.typ)
            if p.has_default:
                s = s + " = " + self._zero_value(p.typ)
            parts.append(s)
        return ", ".join(parts)

    # ── Statements ────────────────────────────────────────────

    def _emit_stmts(self, stmts: list[TStmt]) -> None:
        """Emit a statement list with look-ahead for comprehension patterns."""
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
        """Try to emit a comprehension from a let + for pair."""
        acc = _restore_name(let_stmt.name, let_stmt.annotations)
        binding = for_stmt.binding
        if isinstance(for_stmt.iterable, TRange):
            args = self._join_exprs(for_stmt.iterable.args, ", ")
            iterable = "range(" + args + ")"
        else:
            iterable = self._expr(for_stmt.iterable)
        binder_parts2: list[str] = []
        for b in binding:
            binder_parts2.append(_restore_name(b, for_stmt.annotations))
        binders = ", ".join(binder_parts2)
        iter_is_map = self._is_map_for(for_stmt)
        if iter_is_map:
            iterable += ".items()"
        elif self._is_enumerate_for(for_stmt):
            iterable = "enumerate(" + iterable + ")"
        body = for_stmt.body
        if prov == "list_comprehension":
            if len(body) == 1 and isinstance(body[0], TExprStmt):
                call = body[0].expr
                if self._is_append_to(call, let_stmt.name):
                    val = self._expr(call.args[1].value)
                    return (
                        acc + " = [" + val + " for " + binders + " in " + iterable + "]"
                    )
            if len(body) == 1:
                if_stmt = body[0]
                if isinstance(if_stmt, TIfStmt) and len(if_stmt.then_body) == 1:
                    then_first = if_stmt.then_body[0]
                    if isinstance(then_first, TExprStmt):
                        call = then_first.expr
                        if self._is_append_to(call, let_stmt.name):
                            val = self._expr(call.args[1].value)
                            guard = self._expr(if_stmt.cond)
                            return (
                                acc
                                + " = ["
                                + val
                                + " for "
                                + binders
                                + " in "
                                + iterable
                                + " if "
                                + guard
                                + "]"
                            )
        elif prov == "dict_comprehension":
            if len(body) == 1 and isinstance(body[0], TAssignStmt):
                target = body[0].target
                if isinstance(target, TIndex):
                    key = self._expr(target.index)
                    val = self._expr(body[0].value)
                    return (
                        acc
                        + " = {"
                        + key
                        + ": "
                        + val
                        + " for "
                        + binders
                        + " in "
                        + iterable
                        + "}"
                    )
        elif prov == "set_comprehension":
            if len(body) == 1 and isinstance(body[0], TExprStmt):
                call = body[0].expr
                if self._is_add_to(call, let_stmt.name):
                    val = self._expr(call.args[1].value)
                    return (
                        acc + " = {" + val + " for " + binders + " in " + iterable + "}"
                    )
        return None

    def _is_append_to(self, expr: TExpr, name: str) -> bool:
        if not isinstance(expr, TCall):
            return False
        if not isinstance(expr.func, TVar):
            return False
        if expr.func.name != "Append":
            return False
        first = expr.args[0].value
        if not isinstance(first, TVar):
            return False
        return first.name == name

    def _is_add_to(self, expr: TExpr, name: str) -> bool:
        if not isinstance(expr, TCall):
            return False
        if not isinstance(expr.func, TVar):
            return False
        if expr.func.name != "Add":
            return False
        first = expr.args[0].value
        if not isinstance(first, TVar):
            return False
        return first.name == name

    def _emit_stmt(self, stmt: TStmt) -> None:
        if isinstance(stmt, TLetStmt):
            self._emit_let(stmt)
        elif isinstance(stmt, TAssignStmt):
            self._line(self._expr(stmt.target) + " = " + self._expr(stmt.value))
        elif isinstance(stmt, TTupleAssignStmt):
            self._emit_tuple_assign(stmt)
        elif isinstance(stmt, TOpAssignStmt):
            if (
                self.strict_math
                and stmt.op in _STRICT_INT_COMPOUND
                and self._is_int_expr(stmt.target)
            ):
                fn = _STRICT_INT_COMPOUND[stmt.op]
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
            self._line("continue")
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
        typ_str = self._type(stmt.typ)
        self.var_types[stmt.name] = stmt.typ
        unused = stmt.annotations.get("liveness.initial_value_unused") == "true"
        if stmt.value is not None and not unused:
            self._line(safe + ": " + typ_str + " = " + self._expr(stmt.value))
        else:
            self._line(safe + ": " + typ_str)

    def _emit_tuple_assign(self, stmt: TTupleAssignStmt) -> None:
        unused_str = stmt.annotations.get("liveness.tuple_unused_indices", "")
        unused_indices: set[int] = set()
        if unused_str != "":
            for s in unused_str.split(","):
                if s != "":
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
                    self._line("assert " + cond + ", " + msg)
                else:
                    self._line("assert " + cond)
                return
            if name == "Delete":
                args = expr.args
                self._line(
                    self._expr(args[0].value)
                    + ".pop("
                    + self._expr(args[1].value)
                    + ", None)"
                )
                return
        self._line(self._expr(expr))

    def _emit_if(self, stmt: TIfStmt) -> None:
        if stmt.annotations.get("provenance") == "truthiness":
            truth = self._truthiness_expr(stmt.cond)
            if truth is not None:
                self._line("if " + truth + ":")
            else:
                self._line("if " + self._expr(stmt.cond) + ":")
        else:
            self._line("if " + self._expr(stmt.cond) + ":")
        self.indent += 1
        if not stmt.then_body:
            self._line("pass")
        self._emit_stmts(stmt.then_body)
        self.indent -= 1
        self._emit_else_body(stmt.else_body)

    def _truthiness_expr(self, cond: TExpr) -> str | None:
        """Extract truthiness target: Len(xs) > 0 → xs, s != "" → s."""
        if isinstance(cond, TBinaryOp):
            if (
                cond.op == ">"
                and isinstance(cond.right, TIntLit)
                and cond.right.value == 0
                and isinstance(cond.left, TCall)
                and isinstance(cond.left.func, TVar)
                and cond.left.func.name == "Len"
            ):
                return self._expr(cond.left.args[0].value)
            if (
                cond.op == "!="
                and isinstance(cond.right, TStringLit)
                and cond.right.value == ""
            ):
                return self._expr(cond.left)
        return None

    def _emit_else_body(self, else_body: list[TStmt] | None) -> None:
        if else_body is None or len(else_body) == 0:
            return
        elif_stmt: TStmt | None = None
        if len(else_body) == 1:
            elif_stmt = else_body[0]
        if isinstance(elif_stmt, TIfStmt):
            self._line("elif " + self._expr(elif_stmt.cond) + ":")
            self.indent += 1
            if len(elif_stmt.then_body) == 0:
                self._line("pass")
            self._emit_stmts(elif_stmt.then_body)
            self.indent -= 1
            self._emit_else_body(elif_stmt.else_body)
        else:
            self._line("else:")
            self.indent += 1
            self._emit_stmts(else_body)
            self.indent -= 1

    def _emit_while(self, stmt: TWhileStmt) -> None:
        self._line("while " + self._expr(stmt.cond) + ":")
        self.indent += 1
        if not stmt.body:
            self._line("pass")
        self._emit_stmts(stmt.body)
        self.indent -= 1

    def _emit_for(self, stmt: TForStmt) -> None:
        binding = stmt.binding
        ann = stmt.annotations
        if isinstance(stmt.iterable, TRange):
            args = self._join_exprs(stmt.iterable.args, ", ")
            binder_parts: list[str] = []
            for b in binding:
                binder_parts.append(_restore_name(b, ann))
            binders = ", ".join(binder_parts)
            self._line("for " + binders + " in range(" + args + "):")
        elif len(binding) == 1:
            self._line(
                "for "
                + _restore_name(binding[0], ann)
                + " in "
                + self._expr(stmt.iterable)
                + ":"
            )
        elif len(binding) == 2:
            iter_is_map = self._is_map_for(stmt)
            is_enumerate = self._is_enumerate_for(stmt)
            if iter_is_map:
                method = ".items()"
                wrapper = ""
                suffix = ""
            elif is_enumerate:
                method = ""
                wrapper = "enumerate("
                suffix = ")"
            else:
                method = ""
                wrapper = ""
                suffix = ""
            self._line(
                "for "
                + _restore_name(binding[0], ann)
                + ", "
                + _restore_name(binding[1], ann)
                + " in "
                + wrapper
                + self._expr(stmt.iterable)
                + method
                + suffix
                + ":"
            )
        else:
            binder_parts3: list[str] = []
            for b in binding:
                binder_parts3.append(_restore_name(b, ann))
            binders = ", ".join(binder_parts3)
            self._line("for " + binders + " in " + self._expr(stmt.iterable) + ":")
        self.indent += 1
        if not stmt.body:
            self._line("pass")
        self._emit_stmts(stmt.body)
        self.indent -= 1

    def _is_int_expr(self, expr: TExpr) -> bool:
        if isinstance(expr, TIntLit):
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "int"
        if isinstance(expr, TBinaryOp):
            return self._is_int_expr(expr.left)
        if isinstance(expr, TUnaryOp) and expr.op in ("-", "~"):
            return self._is_int_expr(expr.operand)
        return False

    def _is_float_expr(self, expr: TExpr) -> bool:
        if isinstance(expr, TFloatLit):
            return True
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "float"
        if isinstance(expr, TBinaryOp):
            return self._is_float_expr(expr.left)
        if isinstance(expr, TUnaryOp) and expr.op == "-":
            return self._is_float_expr(expr.operand)
        return False

    def _is_float_list(self, expr: TExpr) -> bool:
        if isinstance(expr, TListLit) and expr.elements:
            return self._is_float_expr(expr.elements[0])
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            if isinstance(typ, TListType) and isinstance(typ.element, TPrimitive):
                return typ.element.kind == "float"
        return False

    def _is_map_type(self, expr: TExpr) -> bool:
        """Check if an expression refers to a variable with map type."""
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TMapType)
        return False

    def _is_map_for(self, stmt: TForStmt) -> bool:
        """Check if a for-loop iterates over map items."""
        if stmt.annotations.get("for.items") == "true":
            return True
        return not isinstance(stmt.iterable, TRange) and self._is_map_type(
            stmt.iterable
        )

    def _is_enumerate_for(self, stmt: TForStmt) -> bool:
        """Check if a for-loop is an enumerate iteration."""
        ann = stmt.annotations
        return ann.get("for.enumerate") == "true" or ann.get("iter_kind") == "enumerate"

    def _emit_try(self, stmt: TTryStmt) -> None:
        self._line("try:")
        self.indent += 1
        if not stmt.body:
            self._line("pass")
        self._emit_stmts(stmt.body)
        self.indent -= 1
        for catch in stmt.catches:
            self._emit_catch(catch)
        if stmt.finally_body is not None:
            self._line("finally:")
            self.indent += 1
            if not stmt.finally_body:
                self._line("pass")
            self._emit_stmts(stmt.finally_body)
            self.indent -= 1

    def _emit_catch(self, catch: TCatch) -> None:
        types: list[str] = []
        for t in catch.types:
            if isinstance(t, TIdentType):
                types.append(t.name)
            else:
                types.append(self._type(t))
        if not types:
            type_str = "Exception"
        elif len(types) == 1:
            type_str = types[0]
        else:
            type_str = "(" + ", ".join(types) + ")"
        unused = catch.annotations.get("liveness.catch_var_unused") == "true"
        if unused:
            self._line("except " + type_str + ":")
        else:
            self._line(
                "except "
                + type_str
                + " as "
                + _restore_name(catch.name, catch.annotations)
                + ":"
            )
        self.indent += 1
        if not catch.body:
            self._line("pass")
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
        keyword = "if" if first else "elif"
        if isinstance(pat, TPatternType):
            type_name = self._pattern_type_name(pat.type_name)
            self._line(keyword + " isinstance(" + expr_str + ", " + type_name + "):")
            self.indent += 1
            unused = pat.annotations.get("liveness.match_var_unused") == "true"
            if not unused:
                self._line(_safe_name(pat.name) + " = " + expr_str)
            if not case.body:
                if unused:
                    self._line("pass")
            self._emit_stmts(case.body)
            self.indent -= 1
        elif isinstance(pat, TPatternEnum):
            self._line(
                keyword
                + " "
                + expr_str
                + " == "
                + pat.enum_name
                + "."
                + pat.variant
                + ":"
            )
            self.indent += 1
            if not case.body:
                self._line("pass")
            self._emit_stmts(case.body)
            self.indent -= 1
        elif isinstance(pat, TPatternNil):
            self._line(keyword + " " + expr_str + " is None:")
            self.indent += 1
            if not case.body:
                self._line("pass")
            self._emit_stmts(case.body)
            self.indent -= 1

    def _emit_match_default(
        self, default: TDefault, expr_str: str, first: bool
    ) -> None:
        if first:
            self._line("if True:")
        else:
            self._line("else:")
        self.indent += 1
        if default.name is not None:
            unused = default.annotations.get("liveness.match_var_unused") == "true"
            if not unused:
                self._line(_safe_name(default.name) + " = " + expr_str)
        if not default.body:
            self._line("pass")
        self._emit_stmts(default.body)
        self.indent -= 1

    def _pattern_type_name(self, typ: TType) -> str:
        if isinstance(typ, TIdentType):
            return typ.name
        return self._type(typ)

    # ── Expressions ───────────────────────────────────────────

    def _expr(self, expr: TExpr) -> str:
        if isinstance(expr, TIntLit):
            return self._int_lit(expr)
        if isinstance(expr, TFloatLit):
            return expr.raw
        if isinstance(expr, TStringLit):
            return '"' + escape_string(expr.value) + '"'
        if isinstance(expr, TBoolLit):
            return "True" if expr.value else "False"
        if isinstance(expr, TNilLit):
            return "None"
        if isinstance(expr, TByteLit):
            return expr.raw
        if isinstance(expr, TBytesLit):
            return self._bytes_lit(expr)
        if isinstance(expr, TRuneLit):
            return '"' + escape_string(expr.value) + '"'
        if isinstance(expr, TVar):
            if expr.name == self.self_name:
                return "self"
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
                self._expr(expr.then_expr)
                + " if "
                + self._expr(expr.cond)
                + " else "
                + self._expr(expr.else_expr)
            )
        if isinstance(expr, TListLit):
            elems = self._join_exprs(expr.elements, ", ")
            return "[" + elems + "]"
        if isinstance(expr, TMapLit):
            if not expr.entries:
                return "{}"
            pairs = ", ".join(
                self._expr(k) + ": " + self._expr(v) for k, v in expr.entries
            )
            return "{" + pairs + "}"
        if isinstance(expr, TSetLit):
            if not expr.elements:
                return "set()"
            elems = self._join_exprs(expr.elements, ", ")
            return "{" + elems + "}"
        if isinstance(expr, TTupleLit):
            elems = self._join_exprs(expr.elements, ", ")
            if len(expr.elements) == 1:
                return "(" + elems + ",)"
            return "(" + elems + ")"
        if isinstance(expr, TFnLit):
            return self._fn_lit(expr)
        if isinstance(expr, TCall):
            return self._call(expr)
        raise NotImplementedError("unknown expression")

    def _int_lit(self, expr: TIntLit) -> str:
        raw = expr.raw
        if raw.startswith("0x") or raw.startswith("0X"):
            return raw
        if raw.startswith("0o") or raw.startswith("0O"):
            return raw
        if raw.startswith("0b") or raw.startswith("0B"):
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
        return 'b"' + "".join(parts) + '"'

    def _slice(self, expr: TSlice) -> str:
        obj = self._expr(expr.obj)
        prov = expr.annotations.get("provenance", "")
        low = self._expr(expr.low)
        high = self._expr(expr.high)
        if prov == "open_start" and self._is_zero(expr.low):
            low = ""
        if prov == "open_end" and self._is_len_call(expr.high):
            high = ""
        return obj + "[" + low + ":" + high + "]"

    def _negative_index(self, expr: TIndex) -> str | None:
        """Pattern-match Len(x) - n → -n for negative indexing."""
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
        if isinstance(expr, TCall):
            if isinstance(expr.func, TVar) and expr.func.name == "Len":
                return True
        return False

    def _binary(self, expr: TBinaryOp) -> str:
        op = expr.op
        # 0.0 / 0.0 raises ZeroDivisionError in Python; emit float("nan")
        if (
            op == "/"
            and isinstance(expr.left, TFloatLit)
            and expr.left.value == 0.0
            and isinstance(expr.right, TFloatLit)
            and expr.right.value == 0.0
        ):
            return 'float("nan")'
        if self.strict_math:
            if op in _STRICT_INT_BINARY and self._is_int_expr(expr.left):
                fn = _STRICT_INT_BINARY[op]
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
        # chained comparison: a OP1 b && b OP2 c → a OP1 b OP2 c
        if op == "&&" and expr.annotations.get("provenance") == "chained_comparison":
            chained = self._chain_comparison(expr)
            if chained is not None:
                return chained
        # nil comparisons → is / is not
        if op == "==" and isinstance(expr.right, TNilLit):
            return self._maybe_paren(expr.left, op, is_left=True) + " is None"
        if op == "!=" and isinstance(expr.right, TNilLit):
            return self._maybe_paren(expr.left, op, is_left=True) + " is not None"
        if op == "==" and isinstance(expr.left, TNilLit):
            return self._maybe_paren(expr.right, op, is_left=False) + " is None"
        if op == "!=" and isinstance(expr.left, TNilLit):
            return self._maybe_paren(expr.right, op, is_left=False) + " is not None"
        py_op = op
        if op == "&&":
            py_op = "and"
        elif op == "||":
            py_op = "or"
        left_str = self._maybe_paren(expr.left, op, is_left=True)
        right_str = self._maybe_paren(expr.right, op, is_left=False)
        return left_str + " " + py_op + " " + right_str

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
                    self._a(expr.operand.args, 1)
                    + " not in "
                    + self._a(expr.operand.args, 0)
                )
            py_op = "not "
            if isinstance(expr.operand, (TBinaryOp,)):
                if expr.operand.op in ("&&", "||", "and", "or"):
                    return py_op + "(" + self._expr(expr.operand) + ")"
                return py_op + self._expr(expr.operand)
            if isinstance(expr.operand, (TTernary,)):
                return py_op + "(" + self._expr(expr.operand) + ")"
            return py_op + self._expr(expr.operand)
        if isinstance(expr.operand, (TBinaryOp, TTernary)):
            return op + "(" + self._expr(expr.operand) + ")"
        return op + self._expr(expr.operand)

    def _chain_comparison(self, expr: TBinaryOp) -> str | None:
        """a OP1 b && b OP2 c → a OP1 b OP2 c."""
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
                + " "
                + right.op
                + " "
                + self._expr(right.right)
            )
        return None

    def _maybe_paren(self, expr: TExpr, parent_op: str, is_left: bool) -> str:
        if isinstance(expr, TBinaryOp):
            if _needs_parens(expr.op, parent_op, is_left):
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
            return "lambda " + params + ": " + self._expr(first.expr)
        name = "_fn"
        self._line("def " + name + "(" + params + "):")
        self.indent += 1
        if not expr.body:
            self._line("pass")
        for s in expr.body:
            self._emit_stmt(s)
        self.indent -= 1
        return name

    # ── Calls ─────────────────────────────────────────────────

    def _call(self, expr: TCall) -> str:
        func = expr.func
        args = expr.args
        # Builtin call
        if isinstance(func, TVar) and func.name in BUILTIN_NAMES:
            return self._builtin_call(func.name, args)
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

    def _struct_call(self, name: str, args: list[TArg]) -> str:
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
                if len(named) < len(ordered):
                    parts2: list[str] = []
                    for f in ordered:
                        if f in named:
                            parts2.append(f + "=" + named[f])
                    return name + "(" + ", ".join(parts2) + ")"
                vals: list[str] = []
                for f in ordered:
                    vals.append(named.get(f, "None"))
                return name + "(" + ", ".join(vals) + ")"
        parts: list[str] = []
        for a in args:
            parts.append(self._expr(a.value))
        return name + "(" + ", ".join(parts) + ")"

    def _method_call(self, func: TFieldAccess, args: list[TArg]) -> str:
        obj_str = self._expr(func.obj)
        if isinstance(func.obj, (TBinaryOp, TUnaryOp, TTernary)):
            obj_str = "(" + obj_str + ")"
        arg_strs = self._join_args(args, ", ")
        return obj_str + "." + func.field + "(" + arg_strs + ")"

    def _builtin_call(self, name: str, args: list[TArg]) -> str:
        if name == "FloorDiv":
            return self._a(args, 0) + " // " + self._a(args, 1)
        if name == "PythonMod":
            return self._a(args, 0) + " % " + self._a(args, 1)
        # Method-on-first-arg
        if name == "Append":
            return self._a(args, 0) + ".append(" + self._a(args, 1) + ")"
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
            return self._a(args, 0) + ".pop()"
        if name == "RemoveAt":
            return self._a(args, 0) + ".pop(" + self._a(args, 1) + ")"
        if name == "ReplaceSlice":
            return (
                self._a(args, 0)
                + "["
                + self._a(args, 1)
                + ":"
                + self._a(args, 2)
                + "] = "
                + self._a(args, 3)
            )
        if name == "IndexOf":
            obj = self._a(args, 0)
            val = self._a(args, 1)
            return obj + ".index(" + val + ") if " + val + " in " + obj + " else -1"
        if name == "Upper":
            return self._a(args, 0) + ".upper()"
        if name == "Lower":
            return self._a(args, 0) + ".lower()"
        if name == "Trim":
            return self._a(args, 0) + ".strip(" + self._a(args, 1) + ")"
        if name == "TrimStart":
            return self._a(args, 0) + ".lstrip(" + self._a(args, 1) + ")"
        if name == "TrimEnd":
            return self._a(args, 0) + ".rstrip(" + self._a(args, 1) + ")"
        if name == "Split":
            return self._a(args, 0) + ".split(" + self._a(args, 1) + ")"
        if name == "SplitN":
            obj = self._a(args, 0)
            sep = self._a(args, 1)
            n_expr = args[2].value
            if isinstance(n_expr, TIntLit):
                return obj + ".split(" + sep + ", " + str(n_expr.value - 1) + ")"
            return obj + ".split(" + sep + ", " + self._a(args, 2) + " - 1)"
        if name == "SplitWhitespace":
            return self._a(args, 0) + ".split()"
        if name == "Join":
            return self._a(args, 0) + ".join(" + self._a(args, 1) + ")"
        if name == "Find":
            return self._a(args, 0) + ".find(" + self._a(args, 1) + ")"
        if name == "RFind":
            return self._a(args, 0) + ".rfind(" + self._a(args, 1) + ")"
        if name == "Count":
            return self._a(args, 0) + ".count(" + self._a(args, 1) + ")"
        if name == "Replace":
            return (
                self._a(args, 0)
                + ".replace("
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
                + ", "
                + self._a(args, 3)
                + ")"
            )
        if name == "StartsWith":
            return self._a(args, 0) + ".startswith(" + self._a(args, 1) + ")"
        if name == "EndsWith":
            return self._a(args, 0) + ".endswith(" + self._a(args, 1) + ")"
        if name == "IsDigit":
            return self._a(args, 0) + ".isdigit()"
        if name == "IsAlpha":
            return self._a(args, 0) + ".isalpha()"
        if name == "IsAlnum":
            return self._a(args, 0) + ".isalnum()"
        if name == "IsSpace":
            return self._a(args, 0) + ".isspace()"
        if name == "IsUpper":
            return self._a(args, 0) + ".isupper()"
        if name == "IsLower":
            return self._a(args, 0) + ".islower()"
        if name == "Encode":
            return self._a(args, 0) + '.encode("utf-8")'
        if name == "Decode":
            return self._a(args, 0) + '.decode("utf-8")'
        if name == "Add":
            return self._a(args, 0) + ".add(" + self._a(args, 1) + ")"
        if name == "Remove":
            return self._a(args, 0) + ".discard(" + self._a(args, 1) + ")"
        if name == "Get":
            if len(args) == 3:
                return (
                    self._a(args, 0)
                    + ".get("
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ")"
                )
            return self._a(args, 0) + ".get(" + self._a(args, 1) + ")"
        if name == "Delete":
            return self._a(args, 0) + ".pop(" + self._a(args, 1) + ", None)"
        if name == "Union":
            return self._a(args, 0) + " | " + self._a(args, 1)
        if name == "Intersection":
            return self._a(args, 0) + " & " + self._a(args, 1)
        if name == "Difference":
            return self._a(args, 0) + " - " + self._a(args, 1)
        if name == "Merge":
            return "{**" + self._a(args, 0) + ", **" + self._a(args, 1) + "}"
        if name == "Keys":
            return "list(" + self._a(args, 0) + ".keys())"
        if name == "Values":
            return "list(" + self._a(args, 0) + ".values())"
        if name == "Items":
            return "list(" + self._a(args, 0) + ".items())"
        # Direct functions
        if name == "Len":
            return "len(" + self._a(args, 0) + ")"
        if name == "Abs":
            return "abs(" + self._a(args, 0) + ")"
        if name == "Min":
            if (
                self.strict_math
                and len(args) == 2
                and self._is_float_expr(args[0].value)
            ):
                return (
                    "strict_min_f64(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            if len(args) == 1:
                return "min(" + self._a(args, 0) + ")"
            return "min(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Max":
            if (
                self.strict_math
                and len(args) == 2
                and self._is_float_expr(args[0].value)
            ):
                return (
                    "strict_max_f64(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
                )
            if len(args) == 1:
                return "max(" + self._a(args, 0) + ")"
            return "max(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Sum":
            return "sum(" + self._a(args, 0) + ")"
        if name == "Round":
            if len(args) == 2:
                return "round(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
            return "round(" + self._a(args, 0) + ")"
        if name == "DivMod":
            return "divmod(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Sorted":
            if self.strict_math and self._is_float_list(args[0].value):
                return "strict_sorted_f64(" + self._a(args, 0) + ")"
            return "sorted(" + self._a(args, 0) + ")"
        if name == "ListFrom":
            return "list(" + self._a(args, 0) + ")"
        if name == "Reversed":
            return "list(reversed(" + self._a(args, 0) + "))"
        if name == "Reverse":
            return self._a(args, 0) + "[::-1]"
        if name == "Map":
            if len(args) == 0:
                return "{}"
            return "list(map(" + self._a(args, 0) + ", " + self._a(args, 1) + "))"
        if name == "Set":
            if len(args) == 0:
                return "set()"
            return "set(" + self._a(args, 0) + ")"
        if name == "SetFromList":
            if isinstance(args[0].value, TSetLit):
                return self._a(args, 0)
            return "set(" + self._a(args, 0) + ")"
        if name == "ToString":
            return "str(" + self._a(args, 0) + ")"
        if name == "ParseInt":
            return "int(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "ParseFloat":
            return "float(" + self._a(args, 0) + ")"
        if name == "FormatInt":
            return self._format_int(args)
        if name == "RuneFromInt":
            return "chr(" + self._a(args, 0) + ")"
        if name == "RuneToInt":
            return "ord(" + self._a(args, 0) + ")"
        if name == "IntToFloat":
            return "float(" + self._a(args, 0) + ")"
        if name == "FloatToInt":
            return "int(" + self._a(args, 0) + ")"
        if name == "ByteToInt":
            return self._a(args, 0)
        if name == "IntToByte":
            return self._a(args, 0)
        if name == "Unwrap":
            return self._a(args, 0)
        if name == "IsNil":
            return self._a(args, 0) + " is None"
        if name == "Sqrt":
            return "math.sqrt(" + self._a(args, 0) + ")"
        if name == "Floor":
            return "math.floor(" + self._a(args, 0) + ")"
        if name == "Ceil":
            return "math.ceil(" + self._a(args, 0) + ")"
        if name == "IsNaN":
            return "math.isnan(" + self._a(args, 0) + ")"
        if name == "IsInf":
            return "math.isinf(" + self._a(args, 0) + ")"
        # I/O
        if name == "WriteOut":
            return "print(" + self._a(args, 0) + ', end="")'
        if name == "WriteErr":
            return "print(" + self._a(args, 0) + ', end="", file=sys.stderr)'
        if name == "WritelnOut":
            return "print(" + self._a(args, 0) + ")"
        if name == "WritelnErr":
            return "print(" + self._a(args, 0) + ", file=sys.stderr)"
        if name == "ReadLine":
            return "input()"
        if name == "ReadAll":
            return "sys.stdin.read()"
        if name == "ReadBytes":
            return "sys.stdin.buffer.read()"
        if name == "ReadBytesN":
            return "sys.stdin.buffer.read(" + self._a(args, 0) + ")"
        if name == "ReadFile":
            p = self._a(args, 0)
            return "open(" + p + ', "rb").read()'
        if name == "WriteFile":
            p = self._a(args, 0)
            d = self._a(args, 1)
            return "open(" + p + ', "w").write(' + d + ")"
        if name == "Args":
            return "sys.argv[1:]"
        if name == "GetEnv":
            return "os.environ.get(" + self._a(args, 0) + ', "")'
        if name == "Exit":
            return "sys.exit(" + self._a(args, 0) + ")"
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
            return self._a(args, 1) + " in " + self._a(args, 0)
        if name == "Concat":
            left = self._maybe_paren(args[0].value, "+", True)
            right = self._maybe_paren(args[1].value, "+", False)
            return left + " + " + right
        if name == "Repeat":
            count = self._maybe_paren(args[1].value, "*", False)
            return self._a(args, 0) + " * " + count
        if name == "Format":
            return self._format_call(args)
        if name == "Assert":
            cond = self._a(args, 0)
            if len(args) > 1:
                return "assert " + cond + ", " + self._a(args, 1)
            return "assert " + cond
        if name == "IsType":
            type_arg = args[1].value
            if isinstance(type_arg, TStringLit):
                type_name = type_arg.value
            else:
                type_name = self._expr(type_arg)
            return "isinstance(" + self._a(args, 0) + ", " + type_name + ")"
        if name == "Bytes" or name == "BytesFrom":
            return "bytes(" + self._a(args, 0) + ")"
        # Fallback
        arg_strs = self._join_args(args, ", ")
        return name + "(" + arg_strs + ")"

    def _a(self, args: list[TArg], i: int) -> str:
        return self._expr(args[i].value)

    def _join_args(self, args: list[TArg], sep: str = ", ") -> str:
        parts: list[str] = []
        for a in args:
            parts.append(self._expr(a.value))
        return sep.join(parts)

    def _join_exprs(self, exprs: list[TExpr], sep: str = ", ") -> str:
        parts: list[str] = []
        for e in exprs:
            parts.append(self._expr(e))
        return sep.join(parts)

    def _join_types(self, types: list[TType], sep: str = ", ") -> str:
        parts: list[str] = []
        for t in types:
            parts.append(self._type(t))
        return sep.join(parts)

    def _format_int(self, args: list[TArg]) -> str:
        n = self._a(args, 0)
        base_expr = args[1].value
        if isinstance(base_expr, TIntLit):
            if base_expr.value == 16:
                return "format(" + n + ', "x")'
            if base_expr.value == 8:
                return "format(" + n + ', "o")'
            if base_expr.value == 2:
                return "format(" + n + ', "b")'
        base = self._a(args, 1)
        return "_format_int(" + n + ", " + base + ")"

    def _format_call(self, args: list[TArg]) -> str:
        template_expr = args[0].value
        if not isinstance(template_expr, TStringLit):
            arg_strs = self._join_args(args, ", ")
            return "Format(" + arg_strs + ")"
        template = template_expr.value
        fmt_args = args[1:]
        # Replace sequential {} placeholders with markers
        markers: dict[str, int] = {}
        result = template
        for i in range(len(fmt_args)):
            marker = "\x00PH" + str(i) + "\x00"
            markers[marker] = i
            result = result.replace("{}", marker, 1)
        # Escape remaining literal braces and quotes in the template
        result = result.replace("{", "{{").replace("}", "}}")
        result = result.replace('"', '\\"')
        # Restore placeholders as f-string interpolations
        for mk, idx in markers.items():
            result = result.replace(mk, "{" + self._expr(fmt_args[idx].value) + "}")
        return 'f"' + result + '"'

    # ── Types ─────────────────────────────────────────────────

    def _type(self, typ: TType) -> str:
        if isinstance(typ, TPrimitive):
            return self._primitive_type(typ.kind)
        if isinstance(typ, TListType):
            return "list[" + self._type(typ.element) + "]"
        if isinstance(typ, TMapType):
            return "dict[" + self._type(typ.key) + ", " + self._type(typ.value) + "]"
        if isinstance(typ, TSetType):
            return "set[" + self._type(typ.element) + "]"
        if isinstance(typ, TTupleType):
            return "tuple[" + self._join_types(typ.elements, ", ") + "]"
        if isinstance(typ, TIdentType):
            return typ.name
        if isinstance(typ, TOptionalType):
            return self._type(typ.inner) + " | None"
        if isinstance(typ, TUnionType):
            return self._join_types(typ.members, " | ")
        if isinstance(typ, TFuncType):
            return "object"
        return "object"

    def _primitive_type(self, kind: str) -> str:
        if kind == "int":
            return "int"
        if kind == "float":
            return "float"
        if kind == "bool":
            return "bool"
        if kind == "string":
            return "str"
        if kind == "byte":
            return "int"
        if kind == "rune":
            return "str"
        if kind == "bytes":
            return "bytes"
        if kind == "void":
            return "None"
        if kind == "nil":
            return "None"
        return "object"


# ============================================================
# PUBLIC API
# ============================================================


def emit_python(module: TModule) -> str:
    struct_names: set[str] = set(BUILTIN_STRUCTS.keys())
    struct_fields: dict[str, list[str]] = {}
    for decl in module.decls:
        if isinstance(decl, TStructDecl):
            struct_names.add(decl.name)
            fnames: list[str] = []
            for f in decl.fields:
                fnames.append(_safe_name(f.name))
            struct_fields[decl.name] = fnames
        elif isinstance(decl, TInterfaceDecl):
            struct_names.add(decl.name)
            if decl.fields:
                ifnames: list[str] = []
                for f in decl.fields:
                    ifnames.append(_safe_name(f.name))
                struct_fields[decl.name] = ifnames
    emitter = _PythonEmitter(struct_names, struct_fields, module.strict_math)
    emitter.emit_module(module)
    return emitter.output()
