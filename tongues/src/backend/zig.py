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
    Index,
    InterfaceRef,
    IntLit,
    IntToStr,
    Len,
    MapLit,
    MaxExpr,
    MethodCall,
    MinExpr,
    Module,
    NoOp,
    OpAssign,
    Param,
    Primitive,
    Print,
    Return,
    Slice,
    SliceExpr,
    SliceLit,
    StringConcat,
    StringLit,
    StructRef,
    Ternary,
    TrimChars,
    TryCatch,
    Truthy,
    Tuple,
    TupleLit,
    TupleAssign,
    UnaryOp,
    Var,
    VarDecl,
    VarLV,
    While,
)

ZIG_RESERVED = frozenset(
    {
        "addrspace",
        "align",
        "allowzero",
        "and",
        "anyframe",
        "anytype",
        "asm",
        "async",
        "await",
        "break",
        "callconv",
        "catch",
        "comptime",
        "const",
        "continue",
        "defer",
        "else",
        "enum",
        "errdefer",
        "error",
        "export",
        "extern",
        "false",
        "fn",
        "for",
        "if",
        "inline",
        "linksection",
        "noalias",
        "nosuspend",
        "null",
        "opaque",
        "or",
        "orelse",
        "packed",
        "pub",
        "resume",
        "return",
        "struct",
        "suspend",
        "switch",
        "test",
        "threadlocal",
        "true",
        "try",
        "type",
        "undefined",
        "union",
        "unreachable",
        "usingnamespace",
        "var",
        "volatile",
        "while",
    }
)

# Zig operator precedence (higher number = tighter binding)
# NOTE: Zig has & ^ | at SAME precedence (left-to-right), unlike Python where & > ^ > |
_PRECEDENCE: dict[str, int] = {
    "or": 1,
    "and": 2,
    "==": 3,
    "!=": 3,
    "<": 3,
    "<=": 3,
    ">": 3,
    ">=": 3,
    "&": 4,
    "^": 4,
    "|": 4,
    "<<": 5,
    ">>": 5,
    "+": 6,
    "-": 6,
    "*": 7,
    "/": 7,
    "%": 7,
}


def _prec(op: str) -> int:
    return _PRECEDENCE.get(op, 10)


def _is_comparison(op: str) -> bool:
    return op in ("==", "!=", "<", "<=", ">", ">=")


def _is_bool(expr: Expr) -> bool:
    """True if expression evaluates to bool in Zig."""
    if isinstance(expr, BinaryOp) and expr.op in ("|", "&", "^"):
        return expr.left.typ == BOOL and expr.right.typ == BOOL
    if isinstance(expr, Call) and expr.func == "bool":
        return True
    # MinExpr/MaxExpr with non-bool operands produce int in Zig
    if isinstance(expr, (MinExpr, MaxExpr)):
        return expr.left.typ == BOOL and expr.right.typ == BOOL
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
        self._needs_step_slice_helper = False
        self._needs_upper_lower_helper = False
        self._needs_join_helper = False
        self._needs_replace_helper = False
        self._needs_count_helper = False
        self._needs_repeat_helper = False
        self._needs_split_helper = False
        self._tmp_counter = 0

    def emit(self, module: Module) -> str:
        self.lines: list[str] = []
        self.indent = 0
        self._func_names = {f.name for f in module.functions}
        self._entrypoint_fn = module.entrypoint.function_name if module.entrypoint else None
        self._needs_panic_handler = False
        body_lines: list[str] = []
        for func in module.functions:
            self._emit_function(func)
        body_lines = self.lines
        self.lines = []
        self.line('const std = @import("std");')
        self.line("")
        if self._needs_panic_handler:
            self._emit_panic_handler()
        if self._needs_step_slice_helper:
            self._emit_step_slice_helper()
        if self._needs_upper_lower_helper:
            self._emit_upper_lower_helper()
        if self._needs_join_helper:
            self._emit_join_helper()
        if self._needs_replace_helper:
            self._emit_replace_helper()
        if self._needs_count_helper:
            self._emit_count_helper()
        if self._needs_repeat_helper:
            self._emit_repeat_helper()
        if self._needs_split_helper:
            self._emit_split_helper()
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
        self.line(
            "fn callWithPanicCapture(comptime F: type, f: F) struct { ok: bool, err: []const u8 } {"
        )
        self.indent += 1
        self.line("var ok = true;")
        self.line('var err: []const u8 = "";')
        self.line("_ = &ok;")
        self.line("_ = &err;")
        self.line("@call(.auto, f, .{}) catch |e| {")
        self.indent += 1
        self.line("_ = e;")
        self.line("ok = false;")
        self.line('err = "panic";')
        self.indent -= 1
        self.line("};")
        self.line("return .{ .ok = ok, .err = err };")
        self.indent -= 1
        self.line("}")
        self.line("")

    def _emit_step_slice_helper(self) -> None:
        """Emit helper for step slicing (e.g., s[::2], s[::-1])."""
        # Global buffer for step slice results (not thread-safe but simple)
        self.line("var _step_slice_buf: [4096]u8 = undefined;")
        self.line("")
        self.line("fn _stepSlice(src: []const u8, step: i64) []const u8 {")
        self.indent += 1
        self.line("var result_len: usize = 0;")
        self.line("if (step > 0) {")
        self.indent += 1
        self.line("const s: usize = @intCast(step);")
        self.line("var j: usize = 0;")
        self.line("while (j < src.len) : (j += s) {")
        self.indent += 1
        self.line("_step_slice_buf[result_len] = src[j];")
        self.line("result_len += 1;")
        self.indent -= 1
        self.line("}")
        self.indent -= 1
        self.line("} else {")
        self.indent += 1
        self.line("const s: usize = @intCast(-step);")
        self.line("var j: isize = @as(isize, @intCast(src.len)) - 1;")
        self.line("while (j >= 0) : (j -= @as(isize, @intCast(s))) {")
        self.indent += 1
        self.line("_step_slice_buf[result_len] = src[@intCast(j)];")
        self.line("result_len += 1;")
        self.indent -= 1
        self.line("}")
        self.indent -= 1
        self.line("}")
        self.line("return _step_slice_buf[0..result_len];")
        self.indent -= 1
        self.line("}")
        self.line("")

    def _emit_upper_lower_helper(self) -> None:
        """Emit helpers for upper() and lower() string methods."""
        self.line("var _case_buf: [4096]u8 = undefined;")
        self.line("")
        self.line("fn _toUpper(src: []const u8) []const u8 {")
        self.indent += 1
        self.line("for (src, 0..) |c, i| {")
        self.indent += 1
        self.line("_case_buf[i] = std.ascii.toUpper(c);")
        self.indent -= 1
        self.line("}")
        self.line("return _case_buf[0..src.len];")
        self.indent -= 1
        self.line("}")
        self.line("")
        self.line("fn _toLower(src: []const u8) []const u8 {")
        self.indent += 1
        self.line("for (src, 0..) |c, i| {")
        self.indent += 1
        self.line("_case_buf[i] = std.ascii.toLower(c);")
        self.indent -= 1
        self.line("}")
        self.line("return _case_buf[0..src.len];")
        self.indent -= 1
        self.line("}")
        self.line("")

    def _emit_join_helper(self) -> None:
        """Emit helper for join() method."""
        self.line("var _join_buf: [4096]u8 = undefined;")
        self.line("")
        self.line("fn _join(sep: []const u8, items: anytype) []const u8 {")
        self.indent += 1
        self.line("var pos: usize = 0;")
        self.line("inline for (items, 0..) |item, i| {")
        self.indent += 1
        self.line("if (i > 0) {")
        self.indent += 1
        self.line("@memcpy(_join_buf[pos..][0..sep.len], sep);")
        self.line("pos += sep.len;")
        self.indent -= 1
        self.line("}")
        self.line("@memcpy(_join_buf[pos..][0..item.len], item);")
        self.line("pos += item.len;")
        self.indent -= 1
        self.line("}")
        self.line("return _join_buf[0..pos];")
        self.indent -= 1
        self.line("}")
        self.line("")

    def _emit_replace_helper(self) -> None:
        """Emit helper for replace() method."""
        self.line("var _replace_buf: [4096]u8 = undefined;")
        self.line("")
        self.line("fn _replace(src: []const u8, old: []const u8, new: []const u8) []const u8 {")
        self.indent += 1
        self.line("var pos: usize = 0;")
        self.line("var i: usize = 0;")
        self.line("while (i < src.len) {")
        self.indent += 1
        self.line("if (i + old.len <= src.len and std.mem.eql(u8, src[i..][0..old.len], old)) {")
        self.indent += 1
        self.line("@memcpy(_replace_buf[pos..][0..new.len], new);")
        self.line("pos += new.len;")
        self.line("i += old.len;")
        self.indent -= 1
        self.line("} else {")
        self.indent += 1
        self.line("_replace_buf[pos] = src[i];")
        self.line("pos += 1;")
        self.line("i += 1;")
        self.indent -= 1
        self.line("}")
        self.indent -= 1
        self.line("}")
        self.line("return _replace_buf[0..pos];")
        self.indent -= 1
        self.line("}")
        self.line("")

    def _emit_count_helper(self) -> None:
        """Emit helper for count() method that counts non-overlapping occurrences."""
        self.line("fn _count(haystack: []const u8, needle: []const u8) i64 {")
        self.indent += 1
        self.line("if (needle.len == 0) return @intCast(haystack.len + 1);")
        self.line("var count: i64 = 0;")
        self.line("var i: usize = 0;")
        self.line("while (i + needle.len <= haystack.len) {")
        self.indent += 1
        self.line("if (std.mem.eql(u8, haystack[i..][0..needle.len], needle)) {")
        self.indent += 1
        self.line("count += 1;")
        self.line("i += needle.len;")
        self.indent -= 1
        self.line("} else {")
        self.indent += 1
        self.line("i += 1;")
        self.indent -= 1
        self.line("}")
        self.indent -= 1
        self.line("}")
        self.line("return count;")
        self.indent -= 1
        self.line("}")
        self.line("")

    def _emit_repeat_helper(self) -> None:
        """Emit helper for bytes/string repetition."""
        self.line("var _repeat_buf: [4096]u8 = undefined;")
        self.line("")
        self.line("fn _repeat(src: []const u8, count: usize) []const u8 {")
        self.indent += 1
        self.line("var pos: usize = 0;")
        self.line("for (0..count) |_| {")
        self.indent += 1
        self.line("@memcpy(_repeat_buf[pos..][0..src.len], src);")
        self.line("pos += src.len;")
        self.indent -= 1
        self.line("}")
        self.line("return _repeat_buf[0..pos];")
        self.indent -= 1
        self.line("}")
        self.line("")

    def _emit_split_helper(self) -> None:
        """Emit helper for bytes/string split."""
        self.line("const SplitResult = struct {")
        self.indent += 1
        self.line("items: [64][]const u8,")
        self.line("len: usize,")
        self.indent -= 1
        self.line("};")
        self.line("")
        self.line("fn _split(src: []const u8, delim: []const u8) SplitResult {")
        self.indent += 1
        self.line("var result: SplitResult = .{ .items = undefined, .len = 0 };")
        self.line("var it = std.mem.splitSequence(u8, src, delim);")
        self.line("while (it.next()) |part| {")
        self.indent += 1
        self.line("result.items[result.len] = part;")
        self.line("result.len += 1;")
        self.indent -= 1
        self.line("}")
        self.line("return result;")
        self.indent -= 1
        self.line("}")
        self.line("")
        self.line("fn _splitEql(a: SplitResult, b: anytype) bool {")
        self.indent += 1
        self.line("if (a.len != b.len) return false;")
        self.line("inline for (0..b.len) |i| {")
        self.indent += 1
        self.line("if (!std.mem.eql(u8, a.items[i], b[i])) return false;")
        self.indent -= 1
        self.line("}")
        self.line("return true;")
        self.indent -= 1
        self.line("}")
        self.line("")

    # ── helpers ──────────────────────────────────────────────

    def _safe(self, name: str) -> str:
        if name in ZIG_RESERVED:
            return '@"' + name + '"'
        return name

    def _is_string_type(self, typ: "type | object") -> bool:
        """Check if a type is a string type."""
        if isinstance(typ, Primitive) and typ.kind == "string":
            return True
        return typ == STRING

    def _is_bytes_type(self, typ: "type | object") -> bool:
        """Check if a type is a bytes type (slice of byte)."""
        if (
            isinstance(typ, Slice)
            and isinstance(typ.element, Primitive)
            and typ.element.kind == "byte"
        ):
            return True
        if isinstance(typ, Primitive) and typ.kind == "bytes":
            return True
        return False

    def _is_string_or_bytes_type(self, typ: "type | object") -> bool:
        """Check if a type is string or bytes."""
        return self._is_string_type(typ) or self._is_bytes_type(typ)

    def _is_string_expr(self, expr: Expr) -> bool:
        """Check if an expression produces a string value."""
        if self._is_string_type(expr.typ):
            return True
        # repr() returns any but actually produces strings
        if isinstance(expr, Call) and expr.func == "repr":
            return True
        return False

    def _is_string_or_bytes_expr(self, expr: Expr) -> bool:
        """Check if an expression produces a string or bytes value."""
        return self._is_string_expr(expr) or self._is_bytes_type(expr.typ)

    def _might_be_bytes(self, expr: Expr) -> bool:
        """Check if an expression might produce bytes (for loose type checking)."""
        if self._is_bytes_type(expr.typ):
            return True
        # SliceLit with byte elements
        if (
            isinstance(expr, SliceLit)
            and isinstance(expr.element_type, Primitive)
            and expr.element_type.kind == "byte"
        ):
            return True
        # Check for expressions that return []const u8
        if isinstance(expr, (BinaryOp,)) and expr.op == "+":
            return self._might_be_bytes(expr.left) or self._might_be_bytes(expr.right)
        return False

    def _fn_name(self, name: str) -> str:
        """Map IR function name to Zig function name."""
        if name == self._entrypoint_fn:
            return f"_{name}"
        return self._safe(name)

    def _type_to_zig(self, typ: "type | object") -> str:
        if isinstance(typ, Primitive):
            return {
                "int": "i64",
                "float": "f64",
                "bool": "bool",
                "string": "[]const u8",
                "void": "void",
                "byte": "u8",
                "rune": "u21",
            }[typ.kind]
        if isinstance(typ, Slice):
            # Bytes (slice of byte) are []const u8, not ArrayList
            if isinstance(typ.element, Primitive) and typ.element.kind == "byte":
                return "[]const u8"
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

    def _maybe_paren(self, expr: Expr, parent_op: str, is_left: bool) -> str:
        """Emit expr, adding parens if its precedence requires it."""
        s = self._emit_expr(expr)
        if isinstance(expr, BinaryOp):
            # Zig doesn't allow chained comparisons, always wrap
            if _is_comparison(parent_op) and _is_comparison(expr.op):
                return f"({s})"
            # Zig doesn't allow mixing comparison with bitwise ops without parens
            if _is_comparison(parent_op) and expr.op in ("|", "&", "^", "<<", ">>"):
                return f"({s})"
            child_prec = _prec(expr.op)
            parent_prec = _prec(parent_op)
            if not is_left:
                if child_prec <= parent_prec:
                    return f"({s})"
            else:
                if child_prec < parent_prec:
                    return f"({s})"
        return s

    # ── functions ────────────────────────────────────────────

    def _emit_function(self, func: Function) -> None:
        params = ", ".join(f"{self._safe(p.name)}: {self._param_type(p)}" for p in func.params)
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
        elif isinstance(stmt, While):
            self._emit_While(stmt)
        elif isinstance(stmt, TupleAssign):
            self._emit_TupleAssign(stmt)
        elif isinstance(stmt, NoOp):
            pass
        else:
            raise NotImplementedError(f"Zig stmt: {stmt}")

    def _emit_VarDecl(self, s: VarDecl) -> None:
        name = self._safe(s.name)
        typ = self._type_to_zig(s.typ)
        # Use is_reassigned from middleend; Zig requires const for non-mutated vars
        use_var = s.mutable and s.is_reassigned
        if s.value is not None:
            val = self._emit_expr(s.value)
            if use_var:
                self.line(f"var {name}: {typ} = {val};")
            else:
                self.line(f"const {name}: {typ} = {val};")
        else:
            if use_var:
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

    def _emit_While(self, s: While) -> None:
        cond = self._emit_expr(s.cond)
        self.line(f"while ({cond}) {{")
        self.indent += 1
        for st in s.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}")

    def _emit_TupleAssign(self, s: TupleAssign) -> None:
        val = self._emit_expr(s.value)
        tmp_name = f"_tmp{self._tmp_counter}"
        self._tmp_counter += 1
        self.line(f"const {tmp_name} = {val};")
        for i, t in enumerate(s.targets):
            if i in s.unused_indices:
                continue
            name = self._emit_lvalue(t)
            if s.is_declaration and isinstance(t, VarLV) and t.name in s.new_targets:
                # Use var since the variable may be reassigned later
                self.line(f"var {name} = {tmp_name}[{i}];")
            else:
                self.line(f"{name} = {tmp_name}[{i}];")

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
                self.line('std.debug.print("\\n", .{});')
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
        raise NotImplementedError(f"Zig lvalue: {lv}")

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
        if isinstance(expr, MinExpr):
            return self._emit_MinExpr(expr)
        if isinstance(expr, MaxExpr):
            return self._emit_MaxExpr(expr)
        if isinstance(expr, Len):
            return self._emit_Len(expr)
        if isinstance(expr, MapLit):
            return self._emit_MapLit(expr)
        if isinstance(expr, Index):
            return self._emit_Index(expr)
        if isinstance(expr, SliceExpr):
            return self._emit_SliceExpr(expr)
        if isinstance(expr, Truthy):
            return self._emit_Truthy(expr)
        if isinstance(expr, MethodCall):
            return self._emit_MethodCall(expr)
        if isinstance(expr, TrimChars):
            return self._emit_TrimChars(expr)
        raise NotImplementedError(f"Zig expr: {expr}")

    def _emit_Var(self, expr: Var) -> str:
        name = self._safe(expr.name)
        if isinstance(expr.typ, FuncType) or (
            isinstance(expr.typ, InterfaceRef) and expr.typ.name == "any"
        ):
            return self._fn_name(expr.name) if expr.name in self._func_names else name
        return name

    def _emit_BinaryOp(self, expr: BinaryOp) -> str:
        op = expr.op
        # 'in' operator for containment check (bytes/strings)
        if op == "in":
            needle = self._emit_expr(expr.left)
            haystack = self._emit_expr(expr.right)
            return f"(std.mem.indexOf(u8, {haystack}, {needle}) != null)"
        # Floor division requires @divFloor for signed integers in Zig
        if op == "//":
            # Convert bools to int for division
            if _is_bool(expr.left) or _is_bool(expr.right):
                left = self._coerce_bool_to_int(expr.left)
                right = self._coerce_bool_to_int(expr.right)
            else:
                left = self._maybe_paren(expr.left, op, is_left=True)
                right = self._maybe_paren(expr.right, op, is_left=False)
            return f"@divFloor({left}, {right})"
        # Regular division also requires @divFloor for signed integers
        if op == "/" and expr.typ == INT:
            if _is_bool(expr.left) or _is_bool(expr.right):
                left = self._coerce_bool_to_int(expr.left)
                right = self._coerce_bool_to_int(expr.right)
            else:
                left = self._maybe_paren(expr.left, op, is_left=True)
                right = self._maybe_paren(expr.right, op, is_left=False)
            return f"@divFloor({left}, {right})"
        # String/bytes concatenation needs special handling in Zig
        if op == "+" and self._is_string_or_bytes_type(expr.typ):
            return self._emit_string_add(expr)
        # Also check if operands are bytes (for concatenation where type might not be set correctly)
        if op == "+" and (
            self._is_bytes_type(expr.left.typ) or self._is_bytes_type(expr.right.typ)
        ):
            return self._emit_string_add(expr)
        # Check for split result comparison (needs special handling)
        if op in ("==", "!="):
            left_is_split = isinstance(expr.left, MethodCall) and expr.left.method == "split"
            right_is_split = isinstance(expr.right, MethodCall) and expr.right.method == "split"
            if left_is_split or right_is_split:
                left = self._emit_expr(expr.left)
                right = self._emit_expr(expr.right)
                cmp = f"_splitEql({left}, {right})"
                if op == "!=":
                    return f"!{cmp}"
                return cmp
        # String/bytes comparison: use std.mem.eql
        # Check both expression type and operand types (some expressions like function calls may have different return types)
        left_is_sb = (
            self._is_string_or_bytes_expr(expr.left)
            or isinstance(expr.left, (StringConcat, SliceLit, MethodCall, Call))
            and self._might_be_bytes(expr.left)
        )
        right_is_sb = (
            self._is_string_or_bytes_expr(expr.right)
            or isinstance(expr.right, (StringConcat, SliceLit, MethodCall, Call))
            and self._might_be_bytes(expr.right)
        )
        if op in ("==", "!=") and (left_is_sb or right_is_sb):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            cmp = f"std.mem.eql(u8, {left}, {right})"
            if op == "!=":
                return f"!{cmp}"
            return cmp
        # String/bytes ordering: use std.mem.lessThan / std.mem.order
        if (
            op in ("<", "<=", ">", ">=")
            and self._is_string_or_bytes_expr(expr.left)
            and self._is_string_or_bytes_expr(expr.right)
        ):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            order = f"std.mem.order(u8, {left}, {right})"
            if op == "<":
                return f"({order} == .lt)"
            if op == "<=":
                return f"({order} != .gt)"
            if op == ">":
                return f"({order} == .gt)"
            if op == ">=":
                return f"({order} != .lt)"
        # Bool/int comparison: convert bool to int for comparison
        if op in ("==", "!=") and _needs_bool_int_coerce(expr.left, expr.right):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            return f"{left} {op} {right}"
        # Bytes/string repetition: b"a" * 3 or 3 * b"a"
        if op == "*":
            left_is_bytes = self._is_string_or_bytes_type(expr.left.typ)
            right_is_bytes = self._is_string_or_bytes_type(expr.right.typ)
            if left_is_bytes or right_is_bytes:
                self._needs_repeat_helper = True
                if left_is_bytes:
                    string = self._emit_expr(expr.left)
                    count = self._emit_expr(expr.right)
                else:
                    count = self._emit_expr(expr.left)
                    string = self._emit_expr(expr.right)
                return f"_repeat({string}, @intCast({count}))"
        # Bool arithmetic: True + True → @as(i64, @intFromBool(true)) + ...
        if op in ("+", "-", "*", "/", "%") and (_is_bool(expr.left) or _is_bool(expr.right)):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            return f"{left} {op} {right}"
        # Bool comparison (>, <, >=, <=): convert to int
        if op in (">", "<", ">=", "<=") and (_is_bool(expr.left) or _is_bool(expr.right)):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            return f"{left} {op} {right}"
        # Bool shift: convert to int
        if op in ("<<", ">>") and (_is_bool(expr.left) or _is_bool(expr.right)):
            left = self._coerce_bool_to_int(expr.left)
            # Shift amount needs different type
            if _is_bool(expr.right):
                right = f"@as(u6, @intFromBool({self._emit_expr(expr.right)}))"
            else:
                right = self._emit_expr(expr.right)
            return f"({left} {op} {right})"
        # Bitwise ops on bools
        if op in ("|", "&", "^"):
            l_bool = _is_bool(expr.left)
            r_bool = _is_bool(expr.right)
            if l_bool and r_bool:
                # Both bools: use logical operators (Zig bool supports these)
                zig_op = {"&": "and", "|": "or", "^": "!="}[op]
                left = self._maybe_paren(expr.left, op, is_left=True)
                right = self._maybe_paren(expr.right, op, is_left=False)
                if op == "^":
                    # XOR becomes != which is a comparison - wrap to prevent chaining
                    return f"({left} {zig_op} {right})"
                return f"{left} {zig_op} {right}"
            if l_bool or r_bool:
                left = self._coerce_bool_to_int(expr.left)
                right = self._coerce_bool_to_int(expr.right)
                return f"({left} {op} {right})"
        # Standard comparison - Zig uses same operators
        left = self._maybe_paren(expr.left, op, is_left=True)
        right = self._maybe_paren(expr.right, op, is_left=False)
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
        # Check if this is a string/bytes concatenation
        is_concat = (
            isinstance(expr, BinaryOp)
            and expr.op == "+"
            and (
                self._is_string_or_bytes_type(expr.typ)
                or self._is_bytes_type(expr.left.typ)
                or self._is_bytes_type(expr.right.typ)
            )
        )
        if is_concat:
            self._flatten_string_add(expr.left, out)
            self._flatten_string_add(expr.right, out)
        else:
            out.append(expr)

    def _emit_UnaryOp(self, expr: UnaryOp) -> str:
        if expr.op == "-":
            # Zig doesn't allow -0 for integers
            if isinstance(expr.operand, IntLit) and expr.operand.value == 0:
                return "0"
            # Zig doesn't allow negation of bool
            if _is_bool(expr.operand):
                return f"-@as(i64, @intFromBool({self._emit_expr(expr.operand)}))"
            return f"-{self._emit_expr(expr.operand)}"
        if expr.op == "~":
            # Zig doesn't allow ~ on bool
            if _is_bool(expr.operand):
                return f"~@as(i64, @intFromBool({self._emit_expr(expr.operand)}))"
            # Zig doesn't allow ~ on comptime_int; cast literals to i64
            if self._is_comptime_int(expr.operand):
                return f"~@as(i64, {self._emit_expr(expr.operand)})"
            # Wrap BinaryOp in parens since ~ binds tighter than binary ops in Zig
            operand = self._emit_expr(expr.operand)
            if isinstance(expr.operand, BinaryOp):
                return f"~({operand})"
            return f"~{operand}"
        operand = self._emit_expr(expr.operand)
        if expr.op == "!":
            return f"!{operand}"
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
        # divmod(a, b) -> .{ @divFloor(a, b), @mod(a, b) }
        if func == "divmod" and len(args) == 2:
            a = self._emit_expr(args[0])
            b = self._emit_expr(args[1])
            # Convert bools to int for divmod
            if args[0].typ == BOOL:
                a = f"@as(i64, @intFromBool({a}))"
            if args[1].typ == BOOL:
                b = f"@as(i64, @intFromBool({b}))"
            return f".{{ @divFloor({a}, {b}), @mod({a}, {b}) }}"
        # pow(base, exp)
        if func == "pow" and len(args) == 2:
            base = self._emit_expr(args[0])
            exp = self._emit_expr(args[1])
            # Convert bools to int
            if args[0].typ == BOOL:
                base = f"@as(i64, @intFromBool({base}))"
            if args[1].typ == BOOL:
                exp = f"@as(u6, @intFromBool({exp}))"
            return f"std.math.pow(i64, {base}, {exp})"
        # abs(x)
        if func == "abs" and len(args) == 1:
            x = self._emit_expr(args[0])
            if args[0].typ == BOOL:
                x = f"@as(i64, @intFromBool({x}))"
            return f"@abs({x})"
        # bytes() constructor
        if func == "bytes":
            if len(args) == 0:
                return '""'
            if len(args) == 1:
                arg = args[0]
                # bytes(int) - creates n zero bytes
                if arg.typ == INT:
                    n = self._emit_expr(arg)
                    return f"([1]u8{{0}} ** @intCast({n}))[0..]"
                # bytes(list[int]) - convert list to bytes
                # Emit as &[_]u8{ ... } slice
                if isinstance(arg, SliceLit):
                    elements = ", ".join(self._emit_expr(e) for e in arg.elements)
                    if not arg.elements:
                        return '""'
                    return f"&[_]u8{{ {elements} }}"
                return self._emit_expr(arg)
            return f"&[_]u8{{ {', '.join(self._emit_expr(a) for a in args)} }}"
        # print() handled in _emit_ExprStmt; fallback here
        if func == "print":
            a = ", ".join(self._emit_expr(a) for a in args)
            return f'std.debug.print("{{s}}\\n", .{{{a}}})'
        # Known module-level function
        name = self._fn_name(func) if func in self._func_names else self._safe(func)
        a = ", ".join(self._emit_expr(a) for a in args)
        return f"{name}({a})"

    def _emit_Cast(self, expr: Cast) -> str:
        inner = self._emit_expr(expr.expr)
        from_type = expr.expr.typ
        to_type = expr.to_type
        if from_type == BOOL and to_type == INT:
            # Check if inner expression already produces int (e.g., -True, ~True)
            if isinstance(expr.expr, UnaryOp) and expr.expr.op in ("-", "~"):
                return inner  # Already converted to int by _emit_UnaryOp
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
        # Check if this is a bytes slice (slice of byte)
        if isinstance(expr.element_type, Primitive) and expr.element_type.kind == "byte":
            # Emit as string literal with escape sequences
            chars = []
            for e in expr.elements:
                if isinstance(e, IntLit):
                    val = e.value
                    # Use printable ASCII if possible, otherwise hex escape
                    if 32 <= val < 127 and chr(val) not in ('"', "\\"):
                        chars.append(chr(val))
                    else:
                        chars.append(f"\\x{val:02x}")
                else:
                    # Non-literal, fall back to array
                    elements = ", ".join(self._emit_expr(e) for e in expr.elements)
                    return f"&[_]u8{{ {elements} }}"
            return '"' + "".join(chars) + '"'
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

    def _emit_MinExpr(self, expr: MinExpr) -> str:
        # For bool result with both bool operands: min(a, b) = a and b
        if expr.typ == BOOL and _is_bool(expr.left) and _is_bool(expr.right):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            return f"({left} and {right})"
        # For int result, use @min with coerced values
        left = (
            self._coerce_bool_to_int(expr.left)
            if _is_bool(expr.left)
            else self._emit_expr(expr.left)
        )
        right = (
            self._coerce_bool_to_int(expr.right)
            if _is_bool(expr.right)
            else self._emit_expr(expr.right)
        )
        return f"@min({left}, {right})"

    def _emit_MaxExpr(self, expr: MaxExpr) -> str:
        # For bool result with both bool operands: max(a, b) = a or b
        if expr.typ == BOOL and _is_bool(expr.left) and _is_bool(expr.right):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            return f"({left} or {right})"
        # For int result, use @max with coerced values
        left = (
            self._coerce_bool_to_int(expr.left)
            if _is_bool(expr.left)
            else self._emit_expr(expr.left)
        )
        right = (
            self._coerce_bool_to_int(expr.right)
            if _is_bool(expr.right)
            else self._emit_expr(expr.right)
        )
        return f"@max({left}, {right})"

    def _emit_Len(self, expr: Len) -> str:
        inner = self._emit_expr(expr.expr)
        # ArrayList needs .items.len, but only for actual ArrayList variables
        # Literals (SliceLit, TupleLit) emit as tuples which have .len directly
        inner_typ = expr.expr.typ
        is_arraylist = (
            isinstance(inner_typ, Slice)
            and not (isinstance(inner_typ.element, Primitive) and inner_typ.element.kind == "byte")
            and not isinstance(expr.expr, (SliceLit, TupleLit))
        )
        if is_arraylist:
            return f"{inner}.items.len"
        return f"{inner}.len"

    def _emit_MapLit(self, expr: MapLit) -> str:
        # Use anonymous struct with len field for bool()/len() checks
        return f".{{ .len = {len(expr.entries)} }}"

    def _emit_Index(self, expr: Index) -> str:
        obj = self._emit_expr(expr.obj)
        # Zig requires usize for array indexing
        idx = self._emit_expr(expr.index)
        # ArrayList needs .items[i] for indexing
        obj_typ = expr.obj.typ
        is_arraylist = isinstance(obj_typ, Slice) and not (
            isinstance(obj_typ.element, Primitive) and obj_typ.element.kind == "byte"
        )
        if is_arraylist:
            if isinstance(expr.index, IntLit):
                return f"{obj}.items[{idx}]"
            return f"{obj}.items[@intCast({idx})]"
        if isinstance(expr.index, IntLit):
            return f"{obj}[{idx}]"
        return f"{obj}[@intCast({idx})]"

    def _emit_SliceExpr(self, expr: SliceExpr) -> str:
        obj = self._emit_expr(expr.obj)
        # Zig slice syntax: obj[low..high]
        if expr.low is None:
            low = "0"
        elif isinstance(expr.low, IntLit):
            low = str(expr.low.value)
        else:
            low = f"@intCast({self._emit_expr(expr.low)})"
        if expr.high is None:
            high = f"{obj}.len"
        elif isinstance(expr.high, IntLit):
            high = str(expr.high.value)
        else:
            high = f"@intCast({self._emit_expr(expr.high)})"
        # Step slicing requires helper function
        if expr.step is not None:
            self._needs_step_slice_helper = True
            step = self._emit_expr(expr.step)
            base_slice = f"{obj}[{low}..{high}]"
            return f"_stepSlice({base_slice}, {step})"
        return f"{obj}[{low}..{high}]"

    def _emit_Truthy(self, expr: Truthy) -> str:
        """Emit truthiness test: non-empty for slices/strings, non-zero for ints."""
        inner = expr.expr
        inner_typ = inner.typ
        val = self._emit_expr(inner)
        # Strings and bytes: check length
        if isinstance(inner_typ, Primitive) and inner_typ.kind in ("string", "bytes"):
            return f"({val}.len != 0)"
        # Slices/lists: check length
        if isinstance(inner_typ, Slice):
            return f"({val}.len != 0)"
        # SliceLit: check length
        if isinstance(inner, SliceLit):
            return f"({val}.len != 0)"
        # Ints: compare to 0
        if isinstance(inner_typ, Primitive) and inner_typ.kind == "int":
            return f"({val} != 0)"
        # Bools: just return value
        if isinstance(inner_typ, Primitive) and inner_typ.kind == "bool":
            return val
        # Default: assume it has .len
        return f"({val}.len != 0)"

    def _emit_MethodCall(self, expr: MethodCall) -> str:
        """Emit method call on an object."""
        obj = self._emit_expr(expr.obj)
        method = expr.method
        args = [self._emit_expr(a) for a in expr.args]
        # Check if this is a bytes type (Slice of byte)
        is_bytes_slice = (
            isinstance(expr.receiver_type, Slice)
            and isinstance(expr.receiver_type.element, Primitive)
            and expr.receiver_type.element.kind == "byte"
        )
        # Handle ArrayList methods (slice type in IR maps to ArrayList in Zig)
        # But not for bytes slices which are []const u8, not ArrayList
        if isinstance(expr.receiver_type, Slice) and not is_bytes_slice:
            if method == "append":
                # ArrayList.append requires allocator in Zig 0.15+
                arg = args[0]
                return f"{obj}.append(std.heap.page_allocator, {arg}) catch unreachable"
            if method == "pop":
                return f"{obj}.pop()"
            if method == "len":
                return f"{obj}.items.len"
        # Handle string/bytes methods
        is_string_or_bytes = (
            isinstance(expr.receiver_type, Primitive)
            and expr.receiver_type.kind in ("string", "bytes")
        ) or is_bytes_slice
        if is_string_or_bytes:
            if method == "count":
                # std.mem.count(u8, haystack, needle)
                return f"std.mem.count(u8, {obj}, {args[0]})"
            if method == "find":
                # Use std.mem.indexOf - returns ?usize, need to convert to i64
                # Wrap in parens so comparison works correctly
                return f"(if (std.mem.indexOf(u8, {obj}, {args[0]})) |i| @as(i64, @intCast(i)) else -1)"
            if method == "startswith":
                return f"std.mem.startsWith(u8, {obj}, {args[0]})"
            if method == "endswith":
                return f"std.mem.endsWith(u8, {obj}, {args[0]})"
            if method == "count":
                self._needs_count_helper = True
                return f"_count({obj}, {args[0]})"
            if method == "upper":
                self._needs_upper_lower_helper = True
                return f"_toUpper({obj})"
            if method == "lower":
                self._needs_upper_lower_helper = True
                return f"_toLower({obj})"
            if method == "strip":
                chars = (
                    args[0]
                    if args
                    else '" "'
                    if isinstance(expr.receiver_type, Primitive)
                    and expr.receiver_type.kind == "string"
                    else '" "'
                )
                return f"std.mem.trim(u8, {obj}, {chars})"
            if method == "lstrip":
                chars = args[0] if args else '" "'
                return f"std.mem.trimLeft(u8, {obj}, {chars})"
            if method == "rstrip":
                chars = args[0] if args else '" "'
                return f"std.mem.trimRight(u8, {obj}, {chars})"
            if method == "split":
                self._needs_split_helper = True
                return f"_split({obj}, {args[0]})"
            if method == "join":
                self._needs_join_helper = True
                return f"_join({obj}, {args[0]})"
            if method == "replace":
                self._needs_replace_helper = True
                return f"_replace({obj}, {args[0]}, {args[1]})"
        raise NotImplementedError(f"Zig MethodCall: {expr.method} on {expr.receiver_type}")

    def _emit_TrimChars(self, expr: TrimChars) -> str:
        """Emit trim operations (strip/lstrip/rstrip)."""
        string = self._emit_expr(expr.string)
        chars = self._emit_expr(expr.chars)
        if expr.mode == "both":
            return f"std.mem.trim(u8, {string}, {chars})"
        elif expr.mode == "left":
            return f"std.mem.trimLeft(u8, {string}, {chars})"
        elif expr.mode == "right":
            return f"std.mem.trimRight(u8, {string}, {chars})"
        raise NotImplementedError(f"Zig TrimChars mode: {expr.mode}")

    def _coerce_bool_to_int(self, expr: Expr) -> str:
        if _is_bool(expr):
            # Check if expression already produces int (e.g., -True, ~True)
            if isinstance(expr, UnaryOp) and expr.op in ("-", "~"):
                return self._emit_expr(expr)  # Already converted by _emit_UnaryOp
            return f"@as(i64, @intFromBool({self._emit_expr(expr)}))"
        return self._emit_expr(expr)

    def _is_comptime_int(self, expr: Expr) -> bool:
        """Check if expression is a compile-time integer (needs cast for bitwise ops)."""
        if isinstance(expr, IntLit):
            return True
        if isinstance(expr, UnaryOp) and expr.op in ("-", "~"):
            return self._is_comptime_int(expr.operand)
        if isinstance(expr, BinaryOp) and expr.op in (
            "+",
            "-",
            "*",
            "/",
            "%",
            "&",
            "|",
            "^",
            "<<",
            ">>",
        ):
            return self._is_comptime_int(expr.left) and self._is_comptime_int(expr.right)
        return False
