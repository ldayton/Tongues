"""Python backend: Taytsh AST → Python source code."""

from __future__ import annotations

from .util import escape_string
from ..taytsh.ast import (
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
    }
)


def _safe_name(name: str) -> str:
    if name in _PYTHON_BUILTINS:
        return name + "_"
    return name


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

_CMP_OPS = frozenset(("==", "!=", "<", ">", "<=", ">="))


def _needs_parens(child_op: str, parent_op: str, is_left: bool) -> bool:
    child_prec = _PRECEDENCE.get(child_op, 0)
    parent_prec = _PRECEDENCE.get(parent_op, 0)
    if child_prec < parent_prec:
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
        if isinstance(decl, (TFnDecl, TStructDecl)):
            result = _scan_decl_builtins(decl)
            if result[0]:
                needs_sys = True
            if result[1]:
                needs_math = True
            if result[2]:
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

_MATH_BUILTINS = frozenset({"IsNaN", "IsInf"})

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
        if isinstance(expr.body, list):
            for s in expr.body:
                _collect_builtin_calls_stmt(s, out)
        else:
            _collect_builtin_calls_expr(expr.body, out)


# ============================================================
# EMITTER
# ============================================================


class _PythonEmitter:
    def __init__(self, struct_names: set[str]) -> None:
        self.struct_names = struct_names
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
        from_imports: list[str] = []
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
            for line in plain_imports:
                self._line(line)
            if plain_imports and from_imports:
                self._line()
            for line in from_imports:
                self._line(line)
            self._line()
        self._line()
        need_blank = False
        for decl in module.decls:
            if isinstance(decl, TInterfaceDecl):
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
        if decl.fields:
            msg_field = None
            for fld in decl.fields:
                if fld.name == "message":
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
        if not decl.fields and not decl.methods:
            self._line("pass")
        for fld in decl.fields:
            self._emit_field(fld)
        for i, method in enumerate(decl.methods):
            if i > 0 or decl.fields:
                self._line()
            self._emit_method(method)
        self.indent -= 1

    def _emit_field(self, fld: TFieldDecl) -> None:
        typ_str = self._type(fld.typ)
        default = self._field_default(fld.typ)
        self._line(fld.name + ": " + typ_str + " = " + default)

    def _field_default(self, typ: TType) -> str:
        if isinstance(typ, TListType):
            return "field(default_factory=list)"
        if isinstance(typ, TMapType):
            return "field(default_factory=dict)"
        if isinstance(typ, TSetType):
            return "field(default_factory=set)"
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
        self._line("def " + decl.name + "(" + params + ") -> " + ret + ":")
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
        if decl.params and decl.params[0].typ is None:
            self.self_name = decl.params[0].name
        if not decl.body:
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
            parts.append(_safe_name(p.name) + ": " + self._type(p.typ))
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
        acc = _safe_name(let_stmt.name)
        binding = for_stmt.binding
        if isinstance(for_stmt.iterable, TRange):
            args = ", ".join(self._expr(a) for a in for_stmt.iterable.args)
            iterable = "range(" + args + ")"
        else:
            iterable = self._expr(for_stmt.iterable)
        binders = ", ".join(_safe_name(b) for b in binding)
        iter_is_map = not isinstance(for_stmt.iterable, TRange) and self._is_map_type(
            for_stmt.iterable
        )
        if iter_is_map:
            iterable += ".items()"
        elif len(binding) == 2 and not isinstance(for_stmt.iterable, TRange):
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
        safe = _safe_name(stmt.name)
        typ_str = self._type(stmt.typ)
        self.var_types[stmt.name] = stmt.typ
        unused = stmt.annotations.get("liveness.initial_value_unused", False)
        if stmt.value is not None and not unused:
            self._line(safe + ": " + typ_str + " = " + self._expr(stmt.value))
        else:
            self._line(safe + ": " + typ_str)

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
        if else_body is None or not else_body:
            return
        if len(else_body) == 1 and isinstance(else_body[0], TIfStmt):
            elif_stmt = else_body[0]
            self._line("elif " + self._expr(elif_stmt.cond) + ":")
            self.indent += 1
            if not elif_stmt.then_body:
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
        if isinstance(stmt.iterable, TRange):
            args = ", ".join(self._expr(a) for a in stmt.iterable.args)
            binders = ", ".join(_safe_name(b) for b in binding)
            self._line("for " + binders + " in range(" + args + "):")
        elif len(binding) == 1:
            self._line(
                "for "
                + _safe_name(binding[0])
                + " in "
                + self._expr(stmt.iterable)
                + ":"
            )
        elif len(binding) == 2:
            iter_is_map = self._is_map_type(stmt.iterable)
            method = ".items()" if iter_is_map else ""
            wrapper = "" if iter_is_map else "enumerate("
            suffix = "" if iter_is_map else ")"
            self._line(
                "for "
                + _safe_name(binding[0])
                + ", "
                + _safe_name(binding[1])
                + " in "
                + wrapper
                + self._expr(stmt.iterable)
                + method
                + suffix
                + ":"
            )
        else:
            binders = ", ".join(_safe_name(b) for b in binding)
            self._line("for " + binders + " in " + self._expr(stmt.iterable) + ":")
        self.indent += 1
        if not stmt.body:
            self._line("pass")
        self._emit_stmts(stmt.body)
        self.indent -= 1

    def _is_map_type(self, expr: TExpr) -> bool:
        """Check if an expression refers to a variable with map type."""
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            return isinstance(typ, TMapType)
        return False

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
        if len(types) == 1:
            type_str = types[0]
        else:
            type_str = "(" + ", ".join(types) + ")"
        unused = catch.annotations.get("liveness.catch_var_unused", False)
        if unused:
            self._line("except " + type_str + ":")
        else:
            self._line("except " + type_str + " as " + _safe_name(catch.name) + ":")
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
            unused = pat.annotations.get("liveness.match_var_unused", False)
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
            unused = default.annotations.get("liveness.match_var_unused", False)
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
            return _safe_name(expr.name)
        if isinstance(expr, TFieldAccess):
            return self._expr(expr.obj) + "." + expr.field
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
            elems = ", ".join(self._expr(e) for e in expr.elements)
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
            elems = ", ".join(self._expr(e) for e in expr.elements)
            return "{" + elems + "}"
        if isinstance(expr, TTupleLit):
            elems = ", ".join(self._expr(e) for e in expr.elements)
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
        params = ", ".join(_safe_name(p.name) for p in expr.params if p.typ is not None)
        if isinstance(expr.body, list):
            # Block body — emit as nested def
            # This is rare, but handle it
            name = "_fn"
            self._line("def " + name + "(" + params + "):")
            self.indent += 1
            if not expr.body:
                self._line("pass")
            for s in expr.body:
                self._emit_stmt(s)
            self.indent -= 1
            return name
        return "lambda " + params + ": " + self._expr(expr.body)

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
        arg_strs = ", ".join(self._expr(a.value) for a in args)
        return fn_name + "(" + arg_strs + ")"

    def _struct_call(self, name: str, args: list[TArg]) -> str:
        parts: list[str] = []
        for a in args:
            if a.name is not None:
                parts.append(a.name + "=" + self._expr(a.value))
            else:
                parts.append(self._expr(a.value))
        return name + "(" + ", ".join(parts) + ")"

    def _method_call(self, func: TFieldAccess, args: list[TArg]) -> str:
        obj_str = self._expr(func.obj)
        if isinstance(func.obj, (TBinaryOp, TUnaryOp, TTernary)):
            obj_str = "(" + obj_str + ")"
        arg_strs = ", ".join(self._expr(a.value) for a in args)
        return obj_str + "." + func.field + "(" + arg_strs + ")"

    def _builtin_call(self, name: str, args: list[TArg]) -> str:
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
            return "min(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Max":
            return "max(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Sum":
            return "sum(" + self._a(args, 0) + ")"
        if name == "Round":
            return "round(" + self._a(args, 0) + ")"
        if name == "DivMod":
            return "divmod(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Sorted":
            return "sorted(" + self._a(args, 0) + ")"
        if name == "Reversed":
            return "list(reversed(" + self._a(args, 0) + "))"
        if name == "Map":
            if len(args) == 0:
                return "{}"
            return "list(map(" + self._a(args, 0) + ", " + self._a(args, 1) + "))"
        if name == "Set":
            if len(args) == 0:
                return "set()"
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
        if name == "Args":
            return "sys.argv[1:]"
        if name == "GetEnv":
            return "os.environ.get(" + self._a(args, 0) + ', "")'
        if name == "Exit":
            return "sys.exit(" + self._a(args, 0) + ")"
        # Operator forms
        if name == "Pow":
            return self._a(args, 0) + " ** " + self._a(args, 1)
        if name == "Contains":
            return self._a(args, 1) + " in " + self._a(args, 0)
        if name == "Concat":
            return self._a(args, 0) + " + " + self._a(args, 1)
        if name == "Repeat":
            return self._a(args, 0) + " * " + self._a(args, 1)
        if name == "Format":
            return self._format_call(args)
        if name == "Assert":
            cond = self._a(args, 0)
            if len(args) > 1:
                return "assert " + cond + ", " + self._a(args, 1)
            return "assert " + cond
        # Fallback
        arg_strs = ", ".join(self._expr(a.value) for a in args)
        return name + "(" + arg_strs + ")"

    def _a(self, args: list[TArg], i: int) -> str:
        return self._expr(args[i].value)

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
            arg_strs = ", ".join(self._expr(a.value) for a in args)
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
        # Escape remaining literal braces
        result = result.replace("{", "{{").replace("}", "}}")
        # Restore placeholders as f-string interpolations
        for marker, i in markers.items():
            result = result.replace(marker, "{" + self._expr(fmt_args[i].value) + "}")
        result = result.replace('"', '\\"')
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
            parts = ", ".join(self._type(e) for e in typ.elements)
            return "tuple[" + parts + "]"
        if isinstance(typ, TIdentType):
            return typ.name
        if isinstance(typ, TOptionalType):
            return self._type(typ.inner) + " | None"
        if isinstance(typ, TUnionType):
            return " | ".join(self._type(m) for m in typ.members)
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
        if kind == "obj":
            return "object"
        if kind == "nil":
            return "None"
        return "object"


# ============================================================
# PUBLIC API
# ============================================================


def emit_python(module: TModule) -> str:
    struct_names = {
        decl.name for decl in module.decls if isinstance(decl, TStructDecl)
    } | set(BUILTIN_STRUCTS.keys())
    emitter = _PythonEmitter(struct_names)
    emitter.emit_module(module)
    return emitter.output()
