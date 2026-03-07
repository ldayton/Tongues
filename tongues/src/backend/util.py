"""Shared utilities for backend code emitters."""

from __future__ import annotations

from ..taytsh.ast import (
    TArg,
    TAssignStmt,
    TBinaryOp,
    TCall,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFloatLit,
    TFnLit,
    TForStmt,
    TIfStmt,
    TIndex,
    TIntLit,
    TLetStmt,
    TListLit,
    TListType,
    TMapLit,
    TMatchStmt,
    TOpAssignStmt,
    TPrimitive,
    TRange,
    TReturnStmt,
    TSetLit,
    TSlice,
    TStmt,
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


def to_snake(name: str) -> str:
    """Convert camelCase/PascalCase to snake_case."""
    if name.startswith("_"):
        name = name[1:]
    if name.isupper():
        return name
    if "_" in name or name.islower():
        return name.lower()
    result: list[str] = []
    i = 0
    while i < len(name):
        ch = name[i]
        if ch.isupper() and i > 0:
            prev = name[i - 1]
            if prev.islower() or prev.isdigit():
                result.append("_")
            elif prev.isupper() and i + 1 < len(name) and name[i + 1].islower():
                result.append("_")
        result.append(ch)
        i += 1
    return "".join(result).lower()


_STRING_ESCAPE_MAP: dict[str, str] = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
    "\f": "\\f",
    "\v": "\\v",
    "\x00": "\\x00",
    "\x01": "\\u0001",
    "\x7f": "\\u007f",
}


def escape_string(value: str) -> str:
    """Escape a string for use in a string literal (without quotes).

    Non-ASCII characters are emitted as \\uXXXX (BMP) or \\UXXXXXXXX (supplementary).
    On JS targets, strings are UTF-16 so characters above U+FFFF appear as surrogate
    pairs; this function detects and recombines them into full code points.
    """
    out: list[str] = []
    i: int = 0
    while i < len(value):
        c: str = value[i : i + 1]
        esc = _STRING_ESCAPE_MAP.get(c)
        if esc is not None:
            out.append(esc)
        elif ord(c) < 32 or ord(c) > 126:
            cp = ord(c)
            if 0xD800 <= cp <= 0xDBFF and i + 1 < len(value):
                lo = ord(value[i + 1 : i + 2])
                if 0xDC00 <= lo <= 0xDFFF:
                    cp = 0x10000 + (cp - 0xD800) * 0x400 + (lo - 0xDC00)
                    i += 1
            if cp <= 0xFFFF:
                h = hex(cp)[2:]
                out.append("\\u" + "0" * (4 - len(h)) + h)
            else:
                h = hex(cp)[2:]
                out.append("\\U" + "0" * (8 - len(h)) + h)
        else:
            out.append(c)
        i += 1
    return "".join(out)


STRICT_INT_BINARY: dict[str, str] = {
    "+": "checked_add_i64",
    "-": "checked_sub_i64",
    "*": "checked_mul_i64",
    "/": "checked_div_i64",
    "%": "checked_rem_i64",
    "<<": "checked_shl_i64",
    ">>": "checked_shr_i64",
    ">>>": "logical_shr_i64",
}

STRICT_INT_COMPOUND: dict[str, str] = {
    "+=": "checked_add_i64",
    "-=": "checked_sub_i64",
    "*=": "checked_mul_i64",
}


class Emitter:
    """Base class for code emitters with indentation tracking."""

    def _line(self, text: str = "") -> None:
        if text:
            self.lines.append("    " * self.indent + text)
        else:
            self.lines.append("")

    def output(self) -> str:
        return "\n".join(self.lines) + "\n"

    def _expr(self, expr: TExpr) -> str:
        raise NotImplementedError

    def _a(self, args: list[TArg], i: int) -> str:
        return self._expr(args[i].value)

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

    def _is_int_expr(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann == "int"
        if isinstance(expr, TIntLit):
            return True
        if isinstance(expr, TVar):
            typ: TType | None = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "int"
        if isinstance(expr, TBinaryOp):
            return self._is_int_expr(expr.left)
        if isinstance(expr, TUnaryOp) and (expr.op in ("-", "~")):
            return self._is_int_expr(expr.operand)
        return False

    def _is_float_expr(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann == "float"
        if isinstance(expr, TFloatLit):
            return True
        if isinstance(expr, TVar):
            typ: TType | None = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "float"
        if isinstance(expr, TBinaryOp):
            return self._is_float_expr(expr.left)
        if isinstance(expr, TUnaryOp) and expr.op == "-":
            return self._is_float_expr(expr.operand)
        return False

    def _is_float_list(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann == "list[float]"
        if isinstance(expr, TListLit) and expr.elements:
            return self._is_float_expr(expr.elements[0])
        if isinstance(expr, TVar):
            typ: TType | None = self.var_types.get(expr.name)
            if isinstance(typ, TListType) and isinstance(typ.element, TPrimitive):
                return typ.element.kind == "float"
        return False

    def _is_zero(self, expr: TExpr) -> bool:
        return isinstance(expr, TIntLit) and expr.value == 0

    def _is_len_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Len"
        )

    def _is_enumerate_for(self, stmt: TForStmt) -> bool:
        ann = stmt.annotations
        return ann.get("for.enumerate") == "true" or ann.get("iter_kind") == "enumerate"


# ── Builtin call collection ──────────────────────────────────


def collect_builtin_calls(stmts: list[TStmt]) -> set[str]:
    """Collect builtin function names called in statements."""
    from ..taytsh.check import BUILTIN_NAMES

    out: set[str] = set()
    for stmt in stmts:
        _collect_builtin_calls_stmt(stmt, out, BUILTIN_NAMES)
    return out


def _collect_builtin_calls_stmt(
    stmt: TStmt, out: set[str], builtin_names: frozenset[str]
) -> None:
    match stmt:
        case TExprStmt():
            _collect_builtin_calls_expr(stmt.expr, out, builtin_names)
        case TLetStmt():
            if stmt.value is not None:
                _collect_builtin_calls_expr(stmt.value, out, builtin_names)
        case TAssignStmt() | TOpAssignStmt() | TTupleAssignStmt():
            _collect_builtin_calls_expr(stmt.value, out, builtin_names)
        case TReturnStmt():
            if stmt.value is not None:
                _collect_builtin_calls_expr(stmt.value, out, builtin_names)
        case TThrowStmt():
            _collect_builtin_calls_expr(stmt.expr, out, builtin_names)
        case TIfStmt():
            _collect_builtin_calls_expr(stmt.cond, out, builtin_names)
            for s in stmt.then_body:
                _collect_builtin_calls_stmt(s, out, builtin_names)
            if stmt.else_body is not None:
                for s in stmt.else_body:
                    _collect_builtin_calls_stmt(s, out, builtin_names)
        case TWhileStmt():
            _collect_builtin_calls_expr(stmt.cond, out, builtin_names)
            for s in stmt.body:
                _collect_builtin_calls_stmt(s, out, builtin_names)
        case TForStmt():
            if isinstance(stmt.iterable, TRange):
                for a in stmt.iterable.args:
                    _collect_builtin_calls_expr(a, out, builtin_names)
            else:
                _collect_builtin_calls_expr(stmt.iterable, out, builtin_names)
            for s in stmt.body:
                _collect_builtin_calls_stmt(s, out, builtin_names)
        case TTryStmt():
            for s in stmt.body:
                _collect_builtin_calls_stmt(s, out, builtin_names)
            for catch in stmt.catches:
                for s in catch.body:
                    _collect_builtin_calls_stmt(s, out, builtin_names)
            if stmt.finally_body is not None:
                for s in stmt.finally_body:
                    _collect_builtin_calls_stmt(s, out, builtin_names)
        case TMatchStmt():
            _collect_builtin_calls_expr(stmt.expr, out, builtin_names)
            for case in stmt.cases:
                for s in case.body:
                    _collect_builtin_calls_stmt(s, out, builtin_names)
            if stmt.default is not None:
                for s in stmt.default.body:
                    _collect_builtin_calls_stmt(s, out, builtin_names)


def _collect_builtin_calls_expr(
    expr: TExpr, out: set[str], builtin_names: frozenset[str]
) -> None:
    match expr:
        case TCall():
            if isinstance(expr.func, TVar) and expr.func.name in builtin_names:
                out.add(expr.func.name)
            _collect_builtin_calls_expr(expr.func, out, builtin_names)
            for a in expr.args:
                _collect_builtin_calls_expr(a.value, out, builtin_names)
        case TBinaryOp():
            _collect_builtin_calls_expr(expr.left, out, builtin_names)
            _collect_builtin_calls_expr(expr.right, out, builtin_names)
        case TUnaryOp():
            _collect_builtin_calls_expr(expr.operand, out, builtin_names)
        case TTernary():
            _collect_builtin_calls_expr(expr.cond, out, builtin_names)
            _collect_builtin_calls_expr(expr.then_expr, out, builtin_names)
            _collect_builtin_calls_expr(expr.else_expr, out, builtin_names)
        case TFieldAccess() | TTupleAccess():
            _collect_builtin_calls_expr(expr.obj, out, builtin_names)
        case TIndex():
            _collect_builtin_calls_expr(expr.obj, out, builtin_names)
            _collect_builtin_calls_expr(expr.index, out, builtin_names)
        case TSlice():
            _collect_builtin_calls_expr(expr.obj, out, builtin_names)
            _collect_builtin_calls_expr(expr.low, out, builtin_names)
            _collect_builtin_calls_expr(expr.high, out, builtin_names)
        case TListLit() | TTupleLit() | TSetLit():
            for e in expr.elements:
                _collect_builtin_calls_expr(e, out, builtin_names)
        case TMapLit():
            for k, v in expr.entries:
                _collect_builtin_calls_expr(k, out, builtin_names)
                _collect_builtin_calls_expr(v, out, builtin_names)
        case TFnLit():
            for s in expr.body:
                _collect_builtin_calls_stmt(s, out, builtin_names)
