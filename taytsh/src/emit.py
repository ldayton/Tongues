"""Taytsh emitter — converts AST back into Taytsh textual syntax.

This is intentionally "total" over the parse-time AST in `taytsh/ast.py`: if a new
node type is added, this emitter should be updated alongside it.
"""

from __future__ import annotations

from .ast import (
    TAssignStmt,
    TBinaryOp,
    TBoolLit,
    TBreakStmt,
    TByteLit,
    TBytesLit,
    TCall,
    TContinueStmt,
    TDecl,
    TDefault,
    TEnumDecl,
    TExpr,
    TExprStmt,
    TFieldAccess,
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
    TThrowStmt,
    TTernary,
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


def to_source(module: TModule) -> str:
    """Render a `TModule` back into Taytsh source text."""
    return _Emitter().emit_module(module)


class _Emitter:
    _INDENT: str = "    "

    # Expression precedence (higher binds tighter)
    _PREC_TERNARY: int = 1
    _PREC_OR: int = 2
    _PREC_AND: int = 3
    _PREC_COMPARE: int = 4
    _PREC_BITOR: int = 5
    _PREC_BITXOR: int = 6
    _PREC_BITAND: int = 7
    _PREC_SHIFT: int = 8
    _PREC_SUM: int = 9
    _PREC_PRODUCT: int = 10
    _PREC_UNARY: int = 11
    _PREC_POSTFIX: int = 12
    _PREC_PRIMARY: int = 13

    _BIN_PREC: dict[str, int] = {
        "||": _PREC_OR,
        "&&": _PREC_AND,
        "==": _PREC_COMPARE,
        "!=": _PREC_COMPARE,
        "<": _PREC_COMPARE,
        "<=": _PREC_COMPARE,
        ">": _PREC_COMPARE,
        ">=": _PREC_COMPARE,
        "|": _PREC_BITOR,
        "^": _PREC_BITXOR,
        "&": _PREC_BITAND,
        "<<": _PREC_SHIFT,
        ">>": _PREC_SHIFT,
        "+": _PREC_SUM,
        "-": _PREC_SUM,
        "*": _PREC_PRODUCT,
        "/": _PREC_PRODUCT,
        "%": _PREC_PRODUCT,
    }

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent_level: int = 0

    # ── Public ──────────────────────────────────────────────

    def emit_module(self, module: TModule) -> str:
        self._lines = []
        self._indent_level = 0
        first = True
        for decl in module.decls:
            if not first:
                self._lines.append("")
            first = False
            self._emit_decl(decl)
        text = "\n".join(self._lines)
        if text == "":
            return ""
        if not text.endswith("\n"):
            text += "\n"
        return text

    # ── Lines / Blocks ──────────────────────────────────────

    def _emit_line(self, line: str) -> None:
        self._lines.append(self._INDENT * self._indent_level + line)

    def _emit_stmt_block(self, stmts: list[TStmt]) -> None:
        self._indent_level += 1
        for stmt in stmts:
            self._emit_stmt(stmt)
        self._indent_level -= 1

    # ── Decls ───────────────────────────────────────────────

    def _emit_decl(self, decl: TDecl) -> None:
        if isinstance(decl, TFnDecl):
            self._emit_fn_decl(decl)
            return
        if isinstance(decl, TStructDecl):
            self._emit_struct_decl(decl)
            return
        if isinstance(decl, TInterfaceDecl):
            self._emit_line("interface " + decl.name + " { }")
            return
        if isinstance(decl, TEnumDecl):
            self._emit_enum_decl(decl)
            return
        raise TypeError("unhandled decl type")

    def _emit_fn_decl(self, decl: TFnDecl) -> None:
        params = self._render_param_list(decl.params)
        ret = self._render_type(decl.ret)
        self._emit_line("fn " + decl.name + "(" + params + ") -> " + ret + " {")
        self._emit_stmt_block(decl.body)
        self._emit_line("}")

    def _emit_struct_decl(self, decl: TStructDecl) -> None:
        if decl.parent is None:
            header = "struct " + decl.name + " {"
        else:
            header = "struct " + decl.name + " : " + decl.parent + " {"
        self._emit_line(header)
        self._indent_level += 1

        for field in decl.fields:
            self._emit_line(f"{field.name}: {self._render_type(field.typ)}")

        for method in decl.methods:
            self._lines.append("")
            self._emit_fn_decl(method)

        self._indent_level -= 1
        self._emit_line("}")

    def _emit_enum_decl(self, decl: TEnumDecl) -> None:
        self._emit_line("enum " + decl.name + " {")
        self._indent_level += 1
        for v in decl.variants:
            self._emit_line(v)
        self._indent_level -= 1
        self._emit_line("}")

    # ── Stmts ───────────────────────────────────────────────

    def _emit_stmt(self, stmt: TStmt) -> None:
        if isinstance(stmt, TLetStmt):
            line = f"let {stmt.name}: {self._render_type(stmt.typ)}"
            if stmt.value is not None:
                line += f" = {self._render_expr(stmt.value, self._PREC_TERNARY)}"
            self._emit_line(line)
            return
        if isinstance(stmt, TAssignStmt):
            self._emit_line(
                f"{self._render_expr(stmt.target, self._PREC_TERNARY)} = {self._render_expr(stmt.value, self._PREC_TERNARY)}"
            )
            return
        if isinstance(stmt, TOpAssignStmt):
            self._emit_line(
                f"{self._render_expr(stmt.target, self._PREC_TERNARY)} {stmt.op} {self._render_expr(stmt.value, self._PREC_TERNARY)}"
            )
            return
        if isinstance(stmt, TTupleAssignStmt):
            targets: list[str] = []
            for t in stmt.targets:
                targets.append(self._render_expr(t, self._PREC_TERNARY))
            left = ", ".join(targets)
            self._emit_line(
                f"{left} = {self._render_expr(stmt.value, self._PREC_TERNARY)}"
            )
            return
        if isinstance(stmt, TReturnStmt):
            if stmt.value is None:
                self._emit_line("return")
            else:
                self._emit_line(
                    f"return {self._render_expr(stmt.value, self._PREC_TERNARY)}"
                )
            return
        if isinstance(stmt, TBreakStmt):
            self._emit_line("break")
            return
        if isinstance(stmt, TContinueStmt):
            self._emit_line("continue")
            return
        if isinstance(stmt, TThrowStmt):
            self._emit_line(f"throw {self._render_expr(stmt.expr, self._PREC_TERNARY)}")
            return
        if isinstance(stmt, TExprStmt):
            self._emit_line(self._render_expr(stmt.expr, self._PREC_TERNARY))
            return
        if isinstance(stmt, TIfStmt):
            self._emit_if_chain(stmt)
            return
        if isinstance(stmt, TWhileStmt):
            self._emit_line(
                "while " + self._render_expr(stmt.cond, self._PREC_TERNARY) + " {"
            )
            self._emit_stmt_block(stmt.body)
            self._emit_line("}")
            return
        if isinstance(stmt, TForStmt):
            self._emit_for_stmt(stmt)
            return
        if isinstance(stmt, TMatchStmt):
            self._emit_match_stmt(stmt)
            return
        if isinstance(stmt, TTryStmt):
            self._emit_try_stmt(stmt)
            return
        raise TypeError("unhandled stmt type")

    def _emit_if_chain(self, stmt: TIfStmt) -> None:
        branches: list[tuple[TExpr, list[TStmt]]] = []
        final_else: list[TStmt] | None = None

        current: TIfStmt | None = stmt
        while current is not None:
            branches.append((current.cond, current.then_body))
            else_body = current.else_body
            if (
                else_body is not None
                and len(else_body) == 1
                and isinstance(else_body[0], TIfStmt)
            ):
                current = else_body[0]
                continue
            final_else = else_body
            current = None

        # Emit first branch
        first_cond, first_body = branches[0]
        self._emit_line(
            "if " + self._render_expr(first_cond, self._PREC_TERNARY) + " {"
        )
        self._emit_stmt_block(first_body)

        # else-if branches
        i = 1
        while i < len(branches):
            cond, body = branches[i]
            self._emit_line(
                "} else if " + self._render_expr(cond, self._PREC_TERNARY) + " {"
            )
            self._emit_stmt_block(body)
            i += 1

        # final else
        if final_else is not None:
            self._emit_line("} else {")
            self._emit_stmt_block(final_else)
            self._emit_line("}")
        else:
            self._emit_line("}")

    def _emit_for_stmt(self, stmt: TForStmt) -> None:
        if len(stmt.binding) == 1:
            binding = stmt.binding[0]
        elif len(stmt.binding) == 2:
            binding = f"{stmt.binding[0]}, {stmt.binding[1]}"
        else:
            raise ValueError("for binding must have 1 or 2 names")

        if isinstance(stmt.iterable, TRange):
            args: list[str] = []
            for a in stmt.iterable.args:
                args.append(self._render_expr(a, self._PREC_TERNARY))
            iterable = f"range({', '.join(args)})"
        else:
            iterable = self._render_expr(stmt.iterable, self._PREC_TERNARY)

        self._emit_line("for " + binding + " in " + iterable + " {")
        self._emit_stmt_block(stmt.body)
        self._emit_line("}")

    def _emit_match_stmt(self, stmt: TMatchStmt) -> None:
        self._emit_line(
            "match " + self._render_expr(stmt.expr, self._PREC_TERNARY) + " {"
        )
        self._indent_level += 1

        for case in stmt.cases:
            self._emit_case(case)

        if stmt.default is not None:
            self._emit_default(stmt.default)

        self._indent_level -= 1
        self._emit_line("}")

    def _emit_case(self, case: TMatchCase) -> None:
        pat = self._render_pattern(case.pattern)
        self._emit_line("case " + pat + " {")
        self._emit_stmt_block(case.body)
        self._emit_line("}")

    def _emit_default(self, default: TDefault) -> None:
        if default.name is None:
            self._emit_line("default {")
        else:
            self._emit_line("default " + default.name + " {")
        self._emit_stmt_block(default.body)
        self._emit_line("}")

    def _emit_try_stmt(self, stmt: TTryStmt) -> None:
        self._emit_line("try {")
        self._emit_stmt_block(stmt.body)

        idx = 0
        while idx < len(stmt.catches):
            c = stmt.catches[idx]
            self._emit_line(
                "} catch " + c.name + ": " + self._render_catch_types(c.types) + " {"
            )
            self._emit_stmt_block(c.body)
            idx += 1

        if stmt.finally_body is not None:
            self._emit_line("} finally {")
            self._emit_stmt_block(stmt.finally_body)
            self._emit_line("}")
        else:
            self._emit_line("}")

    # ── Patterns / TypeNames ────────────────────────────────

    def _render_pattern(self, pat: TPatternType | TPatternEnum | TPatternNil) -> str:
        if isinstance(pat, TPatternNil):
            return "nil"
        if isinstance(pat, TPatternEnum):
            return f"{pat.enum_name}.{pat.variant}"
        if isinstance(pat, TPatternType):
            return f"{pat.name}: {self._render_type_name(pat.type_name)}"
        raise TypeError("unhandled pattern type")

    def _render_catch_types(self, types: list[TType]) -> str:
        parts: list[str] = []
        for t in types:
            parts.append(self._render_type_name(t))
        return " | ".join(parts)

    def _render_type_name(self, typ: TType) -> str:
        # TypeName in the grammar intentionally excludes union and optional.
        if isinstance(typ, TOptionalType) or isinstance(typ, TUnionType):
            raise ValueError("TypeName cannot be optional/union")
        return self._render_type(typ)

    # ── Types ───────────────────────────────────────────────

    def _render_type(self, typ: TType) -> str:
        if isinstance(typ, TPrimitive):
            return typ.kind
        if isinstance(typ, TListType):
            return f"list[{self._render_type(typ.element)}]"
        if isinstance(typ, TMapType):
            return f"map[{self._render_type(typ.key)}, {self._render_type(typ.value)}]"
        if isinstance(typ, TSetType):
            return f"set[{self._render_type(typ.element)}]"
        if isinstance(typ, TTupleType):
            if len(typ.elements) < 2:
                raise ValueError("tuple type must have 2+ elements")
            parts: list[str] = []
            for e in typ.elements:
                parts.append(self._render_type(e))
            return f"({', '.join(parts)})"
        if isinstance(typ, TFuncType):
            parts2: list[str] = []
            for p in typ.params:
                parts2.append(self._render_type(p))
            return f"fn[{', '.join(parts2)}]"
        if isinstance(typ, TIdentType):
            return typ.name
        if isinstance(typ, TUnionType):
            members: list[str] = []
            for m in typ.members:
                members.append(self._render_type(m))
            return " | ".join(members)
        if isinstance(typ, TOptionalType):
            if isinstance(typ.inner, TOptionalType):
                raise ValueError("nested optional types are not valid (T??)")
            return f"{self._render_type(typ.inner)}?"
        raise TypeError("unhandled type node")

    # ── Params ──────────────────────────────────────────────

    def _render_param_list(self, params: list[TParam]) -> str:
        parts: list[str] = []
        for p in params:
            if p.typ is None:
                parts.append("self")
            else:
                parts.append(f"{p.name}: {self._render_type(p.typ)}")
        return ", ".join(parts)

    # ── Exprs ───────────────────────────────────────────────

    def _expr_prec(self, expr: TExpr) -> int:
        if isinstance(expr, TTernary):
            return self._PREC_TERNARY
        if isinstance(expr, TBinaryOp):
            if expr.op not in self._BIN_PREC:
                raise ValueError(f"unknown binary operator: {expr.op}")
            return self._BIN_PREC[expr.op]
        if isinstance(expr, TUnaryOp):
            return self._PREC_UNARY
        if isinstance(expr, (TFieldAccess, TTupleAccess, TIndex, TSlice, TCall)):
            return self._PREC_POSTFIX
        return self._PREC_PRIMARY

    def _render_expr(self, expr: TExpr, parent_prec: int, side: str = "") -> str:
        prec = self._expr_prec(expr)
        text = self._render_expr_inner(expr, prec)

        need_parens = False
        if prec < parent_prec:
            need_parens = True
        elif (
            prec == parent_prec
            and side == "right"
            and prec
            in (
                self._PREC_PRODUCT,
                self._PREC_SUM,
                self._PREC_SHIFT,
                self._PREC_BITAND,
                self._PREC_BITXOR,
                self._PREC_BITOR,
                self._PREC_AND,
                self._PREC_OR,
            )
        ):
            need_parens = True
        elif prec == parent_prec and side != "" and prec == self._PREC_COMPARE:
            need_parens = True

        if need_parens:
            return f"({text})"
        return text

    def _render_expr_inner(self, expr: TExpr, prec: int) -> str:
        if isinstance(expr, TIntLit):
            return expr.raw
        if isinstance(expr, TFloatLit):
            return expr.raw
        if isinstance(expr, TByteLit):
            return expr.raw
        if isinstance(expr, TStringLit):
            return self._quote_string(expr.value)
        if isinstance(expr, TRuneLit):
            return self._quote_rune(expr.value)
        if isinstance(expr, TBytesLit):
            return self._quote_bytes(expr.value)
        if isinstance(expr, TBoolLit):
            return "true" if expr.value else "false"
        if isinstance(expr, TNilLit):
            return "nil"
        if isinstance(expr, TVar):
            return expr.name
        if isinstance(expr, TTernary):
            c = self._render_expr(expr.cond, self._PREC_TERNARY, "left")
            t = self._render_expr(expr.then_expr, self._PREC_TERNARY, "right")
            e = self._render_expr(expr.else_expr, self._PREC_TERNARY, "right")
            return f"{c} ? {t} : {e}"
        if isinstance(expr, TUnaryOp):
            operand = self._render_expr(expr.operand, self._PREC_UNARY, "right")
            return f"{expr.op}{operand}"
        if isinstance(expr, TBinaryOp):
            op_prec = self._BIN_PREC[expr.op]
            left = self._render_expr(expr.left, op_prec, "left")
            right = self._render_expr(expr.right, op_prec, "right")
            return f"{left} {expr.op} {right}"
        if isinstance(expr, TFieldAccess):
            obj = self._render_expr(expr.obj, self._PREC_POSTFIX, "left")
            return f"{obj}.{expr.field}"
        if isinstance(expr, TTupleAccess):
            obj2 = self._render_expr(expr.obj, self._PREC_POSTFIX, "left")
            return f"{obj2}.{expr.index}"
        if isinstance(expr, TIndex):
            obj3 = self._render_expr(expr.obj, self._PREC_POSTFIX, "left")
            idx = self._render_expr(expr.index, self._PREC_TERNARY)
            return f"{obj3}[{idx}]"
        if isinstance(expr, TSlice):
            obj4 = self._render_expr(expr.obj, self._PREC_POSTFIX, "left")
            low = self._render_expr(expr.low, self._PREC_TERNARY)
            high = self._render_expr(expr.high, self._PREC_TERNARY)
            return f"{obj4}[{low}:{high}]"
        if isinstance(expr, TCall):
            func = self._render_expr(expr.func, self._PREC_POSTFIX, "left")
            args: list[str] = []
            for a in expr.args:
                if a.name is None:
                    args.append(self._render_expr(a.value, self._PREC_TERNARY))
                else:
                    args.append(
                        f"{a.name}: {self._render_expr(a.value, self._PREC_TERNARY)}"
                    )
            return f"{func}({', '.join(args)})"
        if isinstance(expr, TListLit):
            parts3: list[str] = []
            for e2 in expr.elements:
                parts3.append(self._render_expr(e2, self._PREC_TERNARY))
            return f"[{', '.join(parts3)}]"
        if isinstance(expr, TMapLit):
            entries: list[str] = []
            for k, v in expr.entries:
                ks = self._render_expr(k, self._PREC_TERNARY)
                vs = self._render_expr(v, self._PREC_TERNARY)
                entries.append(f"{ks}: {vs}")
            return "{ " + ", ".join(entries) + " }"
        if isinstance(expr, TSetLit):
            elems: list[str] = []
            for e3 in expr.elements:
                elems.append(self._render_expr(e3, self._PREC_TERNARY))
            return "{ " + ", ".join(elems) + " }"
        if isinstance(expr, TTupleLit):
            if len(expr.elements) < 2:
                raise ValueError("tuple literal must have 2+ elements")
            elems2: list[str] = []
            for e4 in expr.elements:
                elems2.append(self._render_expr(e4, self._PREC_TERNARY))
            return "(" + ", ".join(elems2) + ")"
        if isinstance(expr, TFnLit):
            params = self._render_param_list(expr.params)
            ret = self._render_type(expr.ret)
            if isinstance(expr.body, list):
                body = self._render_inline_block(expr.body)
                return f"({params}) -> {ret} {body}"
            return f"({params}) -> {ret} => {self._render_expr(expr.body, self._PREC_TERNARY)}"

        raise TypeError("unhandled expr type")

    def _render_inline_block(self, stmts: list[TStmt]) -> str:
        if len(stmts) == 0:
            return "{ }"
        parts: list[str] = []
        for s in stmts:
            parts.append(self._render_stmt_inline(s))
        return "{ " + " ".join(parts) + " }"

    def _render_stmt_inline(self, stmt: TStmt) -> str:
        if isinstance(stmt, TLetStmt):
            line = f"let {stmt.name}: {self._render_type(stmt.typ)}"
            if stmt.value is not None:
                line += f" = {self._render_expr(stmt.value, self._PREC_TERNARY)}"
            return line
        if isinstance(stmt, TAssignStmt):
            return f"{self._render_expr(stmt.target, self._PREC_TERNARY)} = {self._render_expr(stmt.value, self._PREC_TERNARY)}"
        if isinstance(stmt, TOpAssignStmt):
            return f"{self._render_expr(stmt.target, self._PREC_TERNARY)} {stmt.op} {self._render_expr(stmt.value, self._PREC_TERNARY)}"
        if isinstance(stmt, TTupleAssignStmt):
            targets: list[str] = []
            for t in stmt.targets:
                targets.append(self._render_expr(t, self._PREC_TERNARY))
            return f"{', '.join(targets)} = {self._render_expr(stmt.value, self._PREC_TERNARY)}"
        if isinstance(stmt, TReturnStmt):
            if stmt.value is None:
                return "return"
            return f"return {self._render_expr(stmt.value, self._PREC_TERNARY)}"
        if isinstance(stmt, TBreakStmt):
            return "break"
        if isinstance(stmt, TContinueStmt):
            return "continue"
        if isinstance(stmt, TThrowStmt):
            return f"throw {self._render_expr(stmt.expr, self._PREC_TERNARY)}"
        if isinstance(stmt, TExprStmt):
            return self._render_expr(stmt.expr, self._PREC_TERNARY)
        if isinstance(stmt, TIfStmt):
            branches: list[tuple[TExpr, list[TStmt]]] = []
            final_else: list[TStmt] | None = None

            current: TIfStmt | None = stmt
            while current is not None:
                branches.append((current.cond, current.then_body))
                else_body = current.else_body
                if (
                    else_body is not None
                    and len(else_body) == 1
                    and isinstance(else_body[0], TIfStmt)
                ):
                    current = else_body[0]
                    continue
                final_else = else_body
                current = None

            out = f"if {self._render_expr(branches[0][0], self._PREC_TERNARY)} {self._render_inline_block(branches[0][1])}"
            i = 1
            while i < len(branches):
                cond, body = branches[i]
                out += f" else if {self._render_expr(cond, self._PREC_TERNARY)} {self._render_inline_block(body)}"
                i += 1
            if final_else is not None:
                out += f" else {self._render_inline_block(final_else)}"
            return out
        if isinstance(stmt, TWhileStmt):
            return f"while {self._render_expr(stmt.cond, self._PREC_TERNARY)} {self._render_inline_block(stmt.body)}"
        if isinstance(stmt, TForStmt):
            if len(stmt.binding) == 1:
                binding = stmt.binding[0]
            elif len(stmt.binding) == 2:
                binding = f"{stmt.binding[0]}, {stmt.binding[1]}"
            else:
                raise ValueError("for binding must have 1 or 2 names")

            if isinstance(stmt.iterable, TRange):
                args: list[str] = []
                for a in stmt.iterable.args:
                    args.append(self._render_expr(a, self._PREC_TERNARY))
                iterable = f"range({', '.join(args)})"
            else:
                iterable = self._render_expr(stmt.iterable, self._PREC_TERNARY)
            return f"for {binding} in {iterable} {self._render_inline_block(stmt.body)}"
        if isinstance(stmt, TMatchStmt):
            cases: list[str] = []
            for c in stmt.cases:
                cases.append(
                    f"case {self._render_pattern(c.pattern)} {self._render_inline_block(c.body)}"
                )
            if stmt.default is not None:
                if stmt.default.name is None:
                    cases.append(
                        f"default {self._render_inline_block(stmt.default.body)}"
                    )
                else:
                    cases.append(
                        f"default {stmt.default.name} {self._render_inline_block(stmt.default.body)}"
                    )
            return (
                "match "
                + self._render_expr(stmt.expr, self._PREC_TERNARY)
                + " { "
                + " ".join(cases)
                + " }"
            )
        if isinstance(stmt, TTryStmt):
            out2 = f"try {self._render_inline_block(stmt.body)}"
            for c2 in stmt.catches:
                out2 += f" catch {c2.name}: {self._render_catch_types(c2.types)} {self._render_inline_block(c2.body)}"
            if stmt.finally_body is not None:
                out2 += f" finally {self._render_inline_block(stmt.finally_body)}"
            return out2

        raise TypeError("unhandled stmt type")

    # ── Literals / Escapes ──────────────────────────────────

    def _quote_string(self, s: str) -> str:
        return '"' + self._escape_text(s, quote='"') + '"'

    def _quote_rune(self, s: str) -> str:
        if len(s) != 1:
            raise ValueError("rune literal must be exactly one character")
        return "'" + self._escape_text(s, quote="'") + "'"

    def _quote_bytes(self, b: bytes) -> str:
        out = 'b"'
        i = 0
        while i < len(b):
            v = b[i]
            if v == 10:
                out += "\\n"
            elif v == 13:
                out += "\\r"
            elif v == 9:
                out += "\\t"
            elif v == 0:
                out += "\\0"
            elif v == 34:
                out += '\\"'
            elif v == 92:
                out += "\\\\"
            elif 32 <= v <= 126:
                out += chr(v)
            else:
                hx = hex(v)[2:]
                if len(hx) == 1:
                    hx = "0" + hx
                out += "\\x" + hx
            i += 1
        out += '"'
        return out

    def _escape_text(self, s: str, quote: str) -> str:
        out = ""
        for ch in s:
            if ch == "\n":
                out += "\\n"
            elif ch == "\r":
                out += "\\r"
            elif ch == "\t":
                out += "\\t"
            elif ch == "\0":
                out += "\\0"
            elif ch == "\\":
                out += "\\\\"
            elif ch == quote:
                out += "\\" + quote
            else:
                code = ord(ch)
                if 32 <= code <= 126:
                    out += ch
                else:
                    hx2 = hex(code)[2:]
                    if len(hx2) == 1:
                        hx2 = "0" + hx2
                    if len(hx2) == 2:
                        out += "\\x" + hx2
                    else:
                        # Taytsh only specifies \xHH escapes; emit a literal unicode char.
                        out += ch
        return out
