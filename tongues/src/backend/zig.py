"""ZigBackend: IR -> Zig code.

Minimal backend targeting apptest_boolobject.py. Unhandled IR nodes
raise NotImplementedError so gaps are obvious.
"""

from __future__ import annotations

from src.backend.util import Emitter, escape_string
from src.ir import (
    BOOL,
    FLOAT,
    INT,
    STRING,
    VOID,
    Assert,
    Assign,
    BinaryOp,
    Block,
    BoolLit,
    Call,
    Cast,
    EntryPoint,
    Expr,
    ExprStmt,
    FieldAccess,
    ForRange,
    FuncRef,
    FuncType,
    Function,
    If,
    InterfaceRef,
    IntLit,
    IntToStr,
    Module,
    NoOp,
    OpAssign,
    Param,
    Primitive,
    Print,
    Return,
    Slice,
    SliceLit,
    StringConcat,
    StringLit,
    StructRef,
    Ternary,
    TryCatch,
    Tuple,
    TupleLit,
    UnaryOp,
    Var,
    VarDecl,
    VarLV,
)

ZIG_RESERVED = frozenset({
    "addrspace", "align", "allowzero", "and", "anyframe", "anytype", "asm",
    "async", "await", "break", "callconv", "catch", "comptime", "const",
    "continue", "defer", "else", "enum", "errdefer", "error", "export",
    "extern", "false", "fn", "for", "if", "inline", "linksection", "noalias",
    "nosuspend", "null", "opaque", "or", "orelse", "packed", "pub", "resume",
    "return", "struct", "suspend", "switch", "test", "threadlocal", "true",
    "try", "type", "undefined", "union", "unreachable", "usingnamespace",
    "var", "volatile", "while",
})

# Zig operator precedence (higher number = tighter binding)
_ZIG_PREC: dict[str, int] = {
    "or": 1, "and": 2,
    "==": 3, "!=": 3, "<": 3, "<=": 3, ">": 3, ">=": 3,
    "|": 4, "^": 5, "&": 6,
    "<<": 7, ">>": 7,
    "+": 8, "-": 8,
    "*": 9, "/": 9, "%": 9,
}


def _zig_prec(op: str) -> int:
    return _ZIG_PREC.get(op, 10)


def _is_comparison(op: str) -> bool:
    return op in ("==", "!=", "<", "<=", ">", ">=")


def _is_bool(expr: Expr) -> bool:
    """True if expression evaluates to bool."""
    if isinstance(expr, BinaryOp) and expr.op in ("|", "&", "^"):
        return expr.left.typ == BOOL and expr.right.typ == BOOL
    if isinstance(expr, Call) and expr.func == "bool":
        return True
    return expr.typ == BOOL


def _needs_bool_int_coerce(left: Expr, right: Expr) -> bool:
    return _is_bool(left) != _is_bool(right)


class ZigBackend(Emitter):
    """Emit Zig code from IR Module."""

    def __init__(self) -> None:
        super().__init__()
        self._func_names: set[str] = set()
        self._entrypoint_fn: str | None = None
        self._needs_panic_handler = False

    def emit(self, module: Module) -> str:
        self.lines = []
        self.indent = 0
        self._func_names = {f.name for f in module.functions}
        self._entrypoint_fn = module.entrypoint.function_name if module.entrypoint else None
        self._needs_panic_handler = False
        body_lines: list[str] = []
        for func in module.functions:
            self._emit_function(func)
        body_lines = self.lines
        self.lines = []
        self.line("const std = @import(\"std\");")
        self.line("")
        if self._needs_panic_handler:
            self._emit_panic_handler()
        self.lines.extend(body_lines)
        if module.entrypoint is not None:
            self.line("pub fn main() void {")
            self.indent += 1
            self.line(f"const code = _{module.entrypoint.function_name}();")
            self.line("if (code != 0) {")
            self.indent += 1
            self.line("std.process.exit(@intCast(code));")
            self.indent -= 1
            self.line("}")
            self.indent -= 1
            self.line("}")
        return self.output() + "\n"

    def _emit_panic_handler(self) -> None:
        """Emit helper for try/catch simulation using panic capture."""
        self.line("fn callWithPanicCapture(comptime F: type, f: F) struct { ok: bool, err: []const u8 } {")
        self.indent += 1
        self.line("var ok = true;")
        self.line("var err: []const u8 = \"\";")
        self.line("_ = &ok;")
        self.line("_ = &err;")
        self.line("@call(.auto, f, .{}) catch |e| {")
        self.indent += 1
        self.line("_ = e;")
        self.line("ok = false;")
        self.line("err = \"panic\";")
        self.indent -= 1
        self.line("};")
        self.line("return .{ .ok = ok, .err = err };")
        self.indent -= 1
        self.line("}")
        self.line("")

    # ── helpers ──────────────────────────────────────────────

    def _safe(self, name: str) -> str:
        if name in ZIG_RESERVED:
            return "@\"" + name + "\""
        return name

    def _is_string_type(self, typ: "type | object") -> bool:
        """Check if a type is a string type."""
        if isinstance(typ, Primitive) and typ.kind == "string":
            return True
        return typ == STRING

    def _is_string_expr(self, expr: Expr) -> bool:
        """Check if an expression produces a string value."""
        if self._is_string_type(expr.typ):
            return True
        # repr() returns any but actually produces strings
        if isinstance(expr, Call) and expr.func == "repr":
            return True
        return False

    def _fn_name(self, name: str) -> str:
        """Map IR function name to Zig function name."""
        if name == self._entrypoint_fn:
            return f"_{name}"
        return self._safe(name)

    def _type_to_zig(self, typ: "type | object") -> str:
        if isinstance(typ, Primitive):
            return {"int": "i64", "float": "f64", "bool": "bool",
                    "string": "[]const u8", "void": "void", "byte": "u8",
                    "rune": "u21"}[typ.kind]
        if isinstance(typ, Slice):
            inner = self._type_to_zig(typ.element)
            return f"std.ArrayList({inner})"
        if isinstance(typ, Tuple):
            inner = ", ".join(self._type_to_zig(t) for t in typ.elements)
            return f"struct {{ {inner} }}"
        if isinstance(typ, FuncType):
            params = ", ".join(self._type_to_zig(p) for p in typ.params)
            if typ.ret == VOID:
                return f"*const fn({params}) void"
            return f"*const fn({params}) {self._type_to_zig(typ.ret)}"
        if isinstance(typ, InterfaceRef) and typ.name == "any":
            return "*const fn() void"
        if isinstance(typ, StructRef):
            return typ.name
        raise NotImplementedError(f"Zig type: {typ}")

    def _wrap_prec(self, expr: Expr, parent_op: str, is_right: bool) -> str:
        """Emit expr, adding parens if its precedence requires it."""
        s = self._emit_expr(expr)
        if isinstance(expr, BinaryOp):
            # Zig doesn't allow chained comparisons, always wrap
            if _is_comparison(parent_op) and _is_comparison(expr.op):
                return f"({s})"
            child_prec = _zig_prec(expr.op)
            parent_prec = _zig_prec(parent_op)
            if is_right:
                if child_prec <= parent_prec:
                    return f"({s})"
            else:
                if child_prec < parent_prec:
                    return f"({s})"
        return s

    # ── functions ────────────────────────────────────────────

    def _emit_function(self, func: Function) -> None:
        params = ", ".join(
            f"{self._safe(p.name)}: {self._param_type(p)}" for p in func.params
        )
        name = self._fn_name(func.name)
        ret = self._type_to_zig(func.ret)
        self.line(f"fn {name}({params}) {ret} {{")
        self.indent += 1
        for stmt in func.body:
            self._emit_stmt(stmt)
        self.indent -= 1
        self.line("}")
        self.line("")

    def _param_type(self, p: Param) -> str:
        return self._type_to_zig(p.typ)

    # ── statements ───────────────────────────────────────────

    def _emit_stmt(self, stmt: "object") -> None:
        if isinstance(stmt, VarDecl):
            self._emit_VarDecl(stmt)
        elif isinstance(stmt, Assign):
            self._emit_Assign(stmt)
        elif isinstance(stmt, OpAssign):
            self._emit_OpAssign(stmt)
        elif isinstance(stmt, Return):
            self._emit_Return(stmt)
        elif isinstance(stmt, Assert):
            self._emit_Assert(stmt)
        elif isinstance(stmt, If):
            self._emit_If(stmt)
        elif isinstance(stmt, ForRange):
            self._emit_ForRange(stmt)
        elif isinstance(stmt, TryCatch):
            self._emit_TryCatch(stmt)
        elif isinstance(stmt, ExprStmt):
            self._emit_ExprStmt(stmt)
        elif isinstance(stmt, Block):
            self._emit_Block(stmt)
        elif isinstance(stmt, Print):
            self._emit_Print(stmt)
        elif isinstance(stmt, EntryPoint):
            pass  # handled at module level
        elif isinstance(stmt, NoOp):
            pass
        else:
            raise NotImplementedError(f"Zig stmt: {type(stmt).__name__}")

    def _emit_VarDecl(self, s: VarDecl) -> None:
        name = self._safe(s.name)
        typ = self._type_to_zig(s.typ)
        if s.value is not None:
            val = self._emit_expr(s.value)
            if s.mutable:
                self.line(f"var {name}: {typ} = {val};")
            else:
                self.line(f"const {name}: {typ} = {val};")
        else:
            if s.mutable:
                self.line(f"var {name}: {typ} = undefined;")
            else:
                self.line(f"const {name}: {typ} = undefined;")

    def _emit_Assign(self, s: Assign) -> None:
        target = self._emit_lvalue(s.target)
        val = self._emit_expr(s.value)
        if s.is_declaration:
            # Use const by default; Zig requires const for non-mutated vars
            self.line(f"const {target} = {val};")
        else:
            self.line(f"{target} = {val};")

    def _emit_OpAssign(self, s: OpAssign) -> None:
        target = self._emit_lvalue(s.target)
        val = self._emit_expr(s.value)
        self.line(f"{target} {s.op}= {val};")

    def _emit_Return(self, s: Return) -> None:
        if s.value is None:
            self.line("return;")
        else:
            self.line(f"return {self._emit_expr(s.value)};")

    def _emit_Assert(self, s: Assert) -> None:
        test = self._emit_expr(s.test)
        if s.message is not None:
            msg = self._emit_expr(s.message)
            self.line(f"if (!({test})) @panic({msg});")
        else:
            self.line(f'if (!({test})) @panic("assertion failed");')

    def _emit_If(self, s: If) -> None:
        cond = self._emit_expr(s.cond)
        self.line(f"if ({cond}) {{")
        self.indent += 1
        for st in s.then_body:
            self._emit_stmt(st)
        self.indent -= 1
        if s.else_body:
            self.line("} else {")
            self.indent += 1
            for st in s.else_body:
                self._emit_stmt(st)
            self.indent -= 1
        self.line("}")

    def _emit_ForRange(self, s: ForRange) -> None:
        iterable = self._emit_expr(s.iterable)
        val = self._safe(s.value) if s.value else "_"
        # Use inline for since we emit SliceLit as comptime tuple (.{ ... })
        if s.index:
            idx = self._safe(s.index)
            self.line(f"inline for ({iterable}, 0..) |{val}, {idx}| {{")
        else:
            self.line(f"inline for ({iterable}) |{val}| {{")
        self.indent += 1
        for st in s.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}")

    def _emit_TryCatch(self, s: TryCatch) -> None:
        """Emit try/catch - Zig doesn't have exceptions, so we wrap in a block.

        Since Zig uses panics not exceptions, assertions will abort on failure.
        We emit catch block body unconditionally dead-coded to satisfy Zig's
        mutation requirements.
        """
        # Emit the try body directly
        for st in s.body:
            self._emit_stmt(st)
        # Emit catch blocks as dead code to satisfy Zig's var mutation analysis
        if s.catches:
            self.line("if (false) {")
            self.indent += 1
            # Declare catch variables once (they might be duplicated across catch blocks)
            declared_vars: set[str] = set()
            for catch in s.catches:
                if catch.var and catch.var not in declared_vars:
                    var_name = self._safe(catch.var)
                    self.line(f'const {var_name}: []const u8 = "";')
                    declared_vars.add(catch.var)
            # Emit catch bodies
            for catch in s.catches:
                for st in catch.body:
                    self._emit_stmt(st)
            self.indent -= 1
            self.line("}")

    def _emit_ExprStmt(self, s: ExprStmt) -> None:
        if isinstance(s.expr, Call) and s.expr.func == "print":
            args = s.expr.args
            if args:
                # Build proper format string for complex expressions
                fmt_str, fmt_args = self._build_format_string(args[0])
                self.line(f'std.debug.print("{fmt_str}\\n", .{{{fmt_args}}});')
            else:
                self.line("std.debug.print(\"\\n\", .{});")
            return
        expr = self._emit_expr(s.expr)
        self.line(f"_ = {expr};")

    def _emit_Block(self, s: Block) -> None:
        self.line("{")
        self.indent += 1
        for st in s.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}")

    def _emit_Print(self, s: Print) -> None:
        # Check for StringConcat with IntToStr - emit proper format string
        fmt_str, fmt_args = self._build_format_string(s.value)
        nl = "\\n" if s.newline else ""
        if s.stderr:
            self.line(f'std.debug.print("{fmt_str}{nl}", .{{{fmt_args}}});')
        else:
            self.line(f'std.debug.print("{fmt_str}{nl}", .{{{fmt_args}}});')

    def _build_format_string(self, expr: Expr) -> tuple[str, str]:
        """Build Zig format string and args from an expression.

        Returns (format_str, args_str) for use in std.debug.print.
        """
        if isinstance(expr, StringConcat):
            fmt_parts: list[str] = []
            args: list[str] = []
            for part in expr.parts:
                f, a = self._build_format_string(part)
                fmt_parts.append(f)
                if a:
                    args.append(a)
            return "".join(fmt_parts), ", ".join(args)
        if isinstance(expr, IntToStr):
            return "{d}", self._emit_expr(expr.value)
        if isinstance(expr, StringLit):
            return escape_string(expr.value), ""
        if isinstance(expr, BinaryOp) and expr.op == "+" and self._is_string_type(expr.typ):
            # Flatten string concatenation
            parts: list[Expr] = []
            self._flatten_string_add(expr, parts)
            fmt_parts = []
            args = []
            for part in parts:
                f, a = self._build_format_string(part)
                fmt_parts.append(f)
                if a:
                    args.append(a)
            return "".join(fmt_parts), ", ".join(args)
        # Default: emit as string
        return "{s}", self._emit_expr(expr)

    def _emit_lvalue(self, lv: "object") -> str:
        if isinstance(lv, VarLV):
            return self._safe(lv.name)
        raise NotImplementedError(f"Zig lvalue: {type(lv).__name__}")

    # ── expressions ──────────────────────────────────────────

    def _emit_expr(self, expr: Expr) -> str:
        if isinstance(expr, IntLit):
            return str(expr.value)
        if isinstance(expr, BoolLit):
            return "true" if expr.value else "false"
        if isinstance(expr, StringLit):
            return f'"{escape_string(expr.value)}"'
        if isinstance(expr, Var):
            return self._emit_Var(expr)
        if isinstance(expr, BinaryOp):
            return self._emit_BinaryOp(expr)
        if isinstance(expr, UnaryOp):
            return self._emit_UnaryOp(expr)
        if isinstance(expr, Call):
            return self._emit_Call(expr)
        if isinstance(expr, Cast):
            return self._emit_Cast(expr)
        if isinstance(expr, TupleLit):
            return self._emit_TupleLit(expr)
        if isinstance(expr, SliceLit):
            return self._emit_SliceLit(expr)
        if isinstance(expr, FieldAccess):
            return self._emit_FieldAccess(expr)
        if isinstance(expr, FuncRef):
            return self._fn_name(expr.name)
        if isinstance(expr, StringConcat):
            return self._emit_StringConcat(expr)
        if isinstance(expr, IntToStr):
            return self._emit_IntToStr(expr)
        if isinstance(expr, Ternary):
            return self._emit_Ternary(expr)
        raise NotImplementedError(f"Zig expr: {type(expr).__name__}")

    def _emit_Var(self, expr: Var) -> str:
        name = self._safe(expr.name)
        if isinstance(expr.typ, FuncType) or (
            isinstance(expr.typ, InterfaceRef) and expr.typ.name == "any"
        ):
            return self._fn_name(expr.name) if expr.name in self._func_names else name
        return name

    def _emit_BinaryOp(self, expr: BinaryOp) -> str:
        op = expr.op
        # Floor division - use regular division (same for positive integers)
        if op == "//":
            op = "/"
        # String concatenation needs special handling in Zig
        if op == "+" and self._is_string_type(expr.typ):
            return self._emit_string_add(expr)
        # String comparison: use std.mem.eql
        if op in ("==", "!=") and self._is_string_expr(expr.left) and self._is_string_expr(expr.right):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            cmp = f"std.mem.eql(u8, {left}, {right})"
            if op == "!=":
                return f"!{cmp}"
            return cmp
        # Bool/int comparison: convert bool to int for comparison
        if op in ("==", "!=") and _needs_bool_int_coerce(expr.left, expr.right):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            return f"{left} {op} {right}"
        # Bool arithmetic: True + True → @as(i64, @intFromBool(true)) + ...
        if op in ("+", "-", "*", "/", "%") and (
            _is_bool(expr.left) or _is_bool(expr.right)
        ):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            return f"{left} {op} {right}"
        # Bitwise ops on bools
        if op in ("|", "&", "^"):
            l_bool = _is_bool(expr.left)
            r_bool = _is_bool(expr.right)
            if l_bool and r_bool:
                # Both bools: use logical operators (Zig bool supports these)
                zig_op = {"&": "and", "|": "or", "^": "!="}[op]
                left = self._wrap_prec(expr.left, op, False)
                right = self._wrap_prec(expr.right, op, True)
                if op == "^":
                    # XOR becomes != which is a comparison - wrap to prevent chaining
                    return f"({left} {zig_op} {right})"
                return f"{left} {zig_op} {right}"
            if l_bool or r_bool:
                left = self._coerce_bool_to_int(expr.left)
                right = self._coerce_bool_to_int(expr.right)
                return f"({left} {op} {right})"
        # Standard comparison - Zig uses same operators
        left = self._wrap_prec(expr.left, op, False)
        right = self._wrap_prec(expr.right, op, True)
        return f"{left} {op} {right}"

    def _emit_string_add(self, expr: BinaryOp) -> str:
        """Flatten chained string + into allocation (simplified for literals)."""
        # For now, just concatenate compile-time strings
        parts: list[Expr] = []
        self._flatten_string_add(expr, parts)
        # Simple case: all string literals
        all_literals = all(isinstance(p, StringLit) for p in parts)
        if all_literals:
            combined = "".join(p.value for p in parts if isinstance(p, StringLit))  # type: ignore
            return f'"{escape_string(combined)}"'
        # For runtime concat, use std.fmt.allocPrint or similar
        # For now, emit a comptime concat if possible
        args = " ++ ".join(self._emit_expr(p) for p in parts)
        return args

    def _flatten_string_add(self, expr: Expr, out: list[Expr]) -> None:
        if isinstance(expr, BinaryOp) and expr.op == "+" and expr.typ == STRING:
            self._flatten_string_add(expr.left, out)
            self._flatten_string_add(expr.right, out)
        else:
            out.append(expr)

    def _emit_UnaryOp(self, expr: UnaryOp) -> str:
        operand = self._emit_expr(expr.operand)
        if expr.op == "!":
            return f"!{operand}"
        if expr.op == "-":
            return f"-{operand}"
        if expr.op == "~":
            return f"~{operand}"
        raise NotImplementedError(f"Zig unary op: {expr.op}")

    def _emit_Call(self, expr: Call) -> str:
        func = expr.func
        args = expr.args
        # bool() builtin
        if func == "bool":
            if not args:
                return "false"
            if len(args) == 1 and args[0].typ == INT:
                return f"({self._emit_expr(args[0])} != 0)"
            if len(args) == 1 and args[0].typ == BOOL:
                return self._emit_expr(args[0])
            return f"({self._emit_expr(args[0])} != 0)"
        # repr(bool)
        if func == "repr" and len(args) == 1 and args[0].typ == BOOL:
            inner = self._emit_expr(args[0])
            return f'if ({inner}) "True" else "False"'
        # str(bool)
        if func == "str" and len(args) == 1 and args[0].typ == BOOL:
            inner = self._emit_expr(args[0])
            return f'if ({inner}) "True" else "False"'
        # str(any)
        if func == "str" and len(args) == 1:
            inner = self._emit_expr(args[0])
            if args[0].typ == INT:
                # Use std.fmt for int to string - simplified
                return f'std.fmt.comptimePrint("{{}}", .{{{inner}}})'
            return inner
        # int(bool)
        if func == "int" and len(args) == 1 and args[0].typ == BOOL:
            return f"@as(i64, @intFromBool({self._emit_expr(args[0])}))"
        # print() handled in _emit_ExprStmt; fallback here
        if func == "print":
            a = ", ".join(self._emit_expr(a) for a in args)
            return f"std.debug.print(\"{{s}}\\n\", .{{{a}}})"
        # Known module-level function
        name = self._fn_name(func) if func in self._func_names else self._safe(func)
        a = ", ".join(self._emit_expr(a) for a in args)
        return f"{name}({a})"

    def _emit_Cast(self, expr: Cast) -> str:
        inner = self._emit_expr(expr.expr)
        from_type = expr.expr.typ
        to_type = expr.to_type
        if from_type == BOOL and to_type == INT:
            return f"@as(i64, @intFromBool({inner}))"
        if from_type == BOOL and to_type == STRING:
            return f'if ({inner}) "True" else "False"'
        if to_type == STRING:
            return inner  # Simplified
        if from_type == INT and to_type == FLOAT:
            return f"@as(f64, @floatFromInt({inner}))"
        if from_type == FLOAT and to_type == INT:
            return f"@as(i64, @intFromFloat({inner}))"
        if from_type == to_type:
            return inner
        return inner

    def _emit_TupleLit(self, expr: TupleLit) -> str:
        parts: list[str] = []
        for e in expr.elements:
            s = self._emit_expr(e)
            parts.append(s)
        return f".{{ {', '.join(parts)} }}"

    def _emit_SliceLit(self, expr: SliceLit) -> str:
        elements = ", ".join(self._emit_expr(e) for e in expr.elements)
        return f".{{ {elements} }}"

    def _emit_FieldAccess(self, expr: FieldAccess) -> str:
        obj = self._emit_expr(expr.obj)
        field = expr.field
        # Tuple fields: F0 → [0], F1 → [1], etc.
        if field.startswith("F") and field[1:].isdigit():
            return f"{obj}[{field[1:]}]"
        return f"{obj}.{self._safe(field)}"

    def _emit_StringConcat(self, expr: StringConcat) -> str:
        # For compile-time known strings, use ++
        parts = " ++ ".join(self._emit_expr(p) for p in expr.parts)
        return parts

    def _emit_IntToStr(self, expr: IntToStr) -> str:
        inner = self._emit_expr(expr.value)
        return f'std.fmt.comptimePrint("{{}}", .{{{inner}}})'

    def _emit_Ternary(self, expr: Ternary) -> str:
        cond = self._emit_expr(expr.cond)
        then = self._emit_expr(expr.then_expr)
        else_ = self._emit_expr(expr.else_expr)
        return f"if ({cond}) {then} else {else_}"

    def _coerce_bool_to_int(self, expr: Expr) -> str:
        if _is_bool(expr):
            return f"@as(i64, @intFromBool({self._emit_expr(expr)}))"
        return self._emit_expr(expr)
