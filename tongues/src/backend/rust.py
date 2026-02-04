"""RustBackend: IR -> Rust code.

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

RUST_RESERVED = frozenset({
    "as", "async", "await", "break", "const", "continue", "crate", "dyn",
    "else", "enum", "extern", "false", "fn", "for", "if", "impl", "in",
    "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return",
    "self", "Self", "static", "struct", "super", "trait", "true", "type",
    "union", "unsafe", "use", "where", "while", "abstract", "become", "box",
    "do", "final", "macro", "override", "priv", "try", "typeof", "unsized",
    "virtual", "yield",
})

# Rust operator precedence (higher number = tighter binding).
# In Rust, bitwise ops bind tighter than comparisons (unlike C).
_RUST_PREC: dict[str, int] = {
    "||": 1, "&&": 2,
    "==": 3, "!=": 3, "<": 3, "<=": 3, ">": 3, ">=": 3,
    "|": 4, "^": 5, "&": 6,
    "<<": 7, ">>": 7,
    "+": 8, "-": 8,
    "*": 9, "/": 9, "%": 9,
}


def _rust_prec(op: str) -> int:
    return _RUST_PREC.get(op, 10)


def _is_bool(expr: Expr) -> bool:
    """True if expression evaluates to bool in Rust output.

    Unlike Go, Rust supports bitwise ops on bools natively, so
    bool & bool → bool (not int).  Mixed bool & int → int.
    """
    if isinstance(expr, BinaryOp) and expr.op in ("|", "&", "^"):
        return expr.left.typ == BOOL and expr.right.typ == BOOL
    if isinstance(expr, Call) and expr.func == "bool":
        return True
    return expr.typ == BOOL


def _needs_bool_int_coerce(left: Expr, right: Expr) -> bool:
    return _is_bool(left) != _is_bool(right)


class RustBackend(Emitter):
    """Emit Rust code from IR Module."""

    def __init__(self) -> None:
        super().__init__()
        self._func_names: set[str] = set()
        self._needs_catch_unwind = False
        self._entrypoint_fn: str | None = None

    def emit(self, module: Module) -> str:
        self.lines = []
        self.indent = 0
        self._func_names = {f.name for f in module.functions}
        self._needs_catch_unwind = False
        self._entrypoint_fn = module.entrypoint.function_name if module.entrypoint else None
        body_lines: list[str] = []
        for func in module.functions:
            self._emit_function(func)
        body_lines = self.lines
        self.lines = []
        if self._needs_catch_unwind:
            self.line("use std::panic::{catch_unwind, AssertUnwindSafe};")
            self.line("")
        self.lines.extend(body_lines)
        if module.entrypoint is not None:
            self.line("fn main() {")
            self.indent += 1
            self.line(f"std::process::exit(_{module.entrypoint.function_name}() as i32);")
            self.indent -= 1
            self.line("}")
        return self.output() + "\n"

    # ── helpers ──────────────────────────────────────────────

    def _safe(self, name: str) -> str:
        if name in RUST_RESERVED:
            return name + "_"
        return name

    def _fn_name(self, name: str) -> str:
        """Map IR function name to Rust function name."""
        if name == self._entrypoint_fn:
            return f"_{name}"
        return self._safe(name)

    def _type_to_rust(self, typ: "type | object") -> str:
        if isinstance(typ, Primitive):
            return {"int": "i64", "float": "f64", "bool": "bool",
                    "string": "&str", "void": "()", "byte": "u8",
                    "rune": "char"}[typ.kind]
        if isinstance(typ, Slice):
            return f"Vec<{self._type_to_rust(typ.element)}>"
        if isinstance(typ, Tuple):
            inner = ", ".join(self._type_to_rust(t) for t in typ.elements)
            return f"({inner})"
        if isinstance(typ, FuncType):
            params = ", ".join(self._type_to_rust(p) for p in typ.params)
            if typ.ret == VOID:
                return f"fn({params})"
            return f"fn({params}) -> {self._type_to_rust(typ.ret)}"
        if isinstance(typ, InterfaceRef) and typ.name == "any":
            return "fn()"
        if isinstance(typ, StructRef):
            return typ.name
        raise NotImplementedError(f"Rust type: {typ}")

    def _wrap_prec(self, expr: Expr, parent_op: str, is_right: bool) -> str:
        """Emit expr, adding parens if its precedence requires it."""
        s = self._emit_expr(expr)
        if isinstance(expr, BinaryOp):
            child_prec = _rust_prec(expr.op)
            parent_prec = _rust_prec(parent_op)
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
        if func.ret == VOID:
            self.line(f"fn {name}({params}) {{")
        else:
            ret = self._type_to_rust(func.ret)
            self.line(f"fn {name}({params}) -> {ret} {{")
        self.indent += 1
        for stmt in func.body:
            self._emit_stmt(stmt)
        self.indent -= 1
        self.line("}")
        self.line("")

    def _param_type(self, p: Param) -> str:
        return self._type_to_rust(p.typ)

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
            raise NotImplementedError(f"Rust stmt: {type(stmt).__name__}")

    def _emit_VarDecl(self, s: VarDecl) -> None:
        name = self._safe(s.name)
        mut = "mut " if s.mutable else ""
        if s.value is not None:
            val = self._emit_expr(s.value)
            # Omit type annotation for slices — Rust infers from vec! and the
            # type annotation helps coerce fn-item types to fn-pointer types.
            if isinstance(s.typ, Slice):
                typ = self._type_to_rust(s.typ)
                self.line(f"let {mut}{name}: {typ} = {val};")
            else:
                self.line(f"let {mut}{name} = {val};")
        else:
            typ = self._type_to_rust(s.typ)
            self.line(f"let {mut}{name}: {typ} = Default::default();")

    def _emit_Assign(self, s: Assign) -> None:
        target = self._emit_lvalue(s.target)
        val = self._emit_expr(s.value)
        if s.is_declaration:
            self.line(f"let {target} = {val};")
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
            self.line(f"if !({test}) {{ panic!(\"{{}}\", {msg}); }}")
        else:
            self.line(f'if !({test}) {{ panic!("assertion failed"); }}')

    def _emit_If(self, s: If) -> None:
        cond = self._emit_expr(s.cond)
        self.line(f"if {cond} {{")
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
        if s.index:
            idx = self._safe(s.index)
            self.line(f"for ({idx}, {val}) in {iterable}.iter().enumerate() {{")
        else:
            self.line(f"for {val} in {iterable}.iter() {{")
        self.indent += 1
        for st in s.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}")

    def _emit_TryCatch(self, s: TryCatch) -> None:
        self._needs_catch_unwind = True
        self.line("let _result = catch_unwind(AssertUnwindSafe(|| {")
        self.indent += 1
        for st in s.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}));")
        if not s.catches:
            return
        catch = s.catches[0]
        var = self._safe(catch.var) if catch.var else "_e"
        self.line("if let Err(_panic_val) = _result {")
        self.indent += 1
        self.line(f"let {var}: String = if let Some(s) = _panic_val.downcast_ref::<String>() {{")
        self.indent += 1
        self.line("s.clone()")
        self.indent -= 1
        self.line("} else if let Some(s) = _panic_val.downcast_ref::<&str>() {")
        self.indent += 1
        self.line("s.to_string()")
        self.indent -= 1
        self.line("} else {")
        self.indent += 1
        self.line('"unknown error".to_string()')
        self.indent -= 1
        self.line("};")
        for st in catch.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}")

    def _emit_ExprStmt(self, s: ExprStmt) -> None:
        if isinstance(s.expr, Call) and s.expr.func == "print":
            args = s.expr.args
            if args:
                val = self._emit_expr(args[0])
                self.line(f"println!(\"{{}}\", {val});")
            else:
                self.line("println!();")
            return
        expr = self._emit_expr(s.expr)
        self.line(f"{expr};")

    def _emit_Block(self, s: Block) -> None:
        self.line("{")
        self.indent += 1
        for st in s.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}")

    def _emit_Print(self, s: Print) -> None:
        val = self._emit_expr(s.value)
        macro = "eprint" if s.stderr else "print"
        if s.newline:
            macro += "ln"
        self.line(f"{macro}!(\"{{}}\", {val});")

    def _emit_lvalue(self, lv: "object") -> str:
        if isinstance(lv, VarLV):
            return self._safe(lv.name)
        raise NotImplementedError(f"Rust lvalue: {type(lv).__name__}")

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
            return f"{self._emit_expr(expr.value)}.to_string()"
        if isinstance(expr, Ternary):
            return self._emit_Ternary(expr)
        raise NotImplementedError(f"Rust expr: {type(expr).__name__}")

    def _emit_Var(self, expr: Var) -> str:
        name = self._safe(expr.name)
        # Function references used as values: use the fn_name mapping
        if isinstance(expr.typ, FuncType) or (
            isinstance(expr.typ, InterfaceRef) and expr.typ.name == "any"
        ):
            return self._fn_name(expr.name) if expr.name in self._func_names else name
        return name

    def _emit_BinaryOp(self, expr: BinaryOp) -> str:
        op = expr.op
        # String concatenation: &str + &str doesn't work in Rust, use format!
        if op == "+" and expr.typ == STRING:
            return self._emit_string_add(expr)
        # Bool/int comparison coercion: True == 1 → (true as i64) == 1
        if op in ("==", "!=") and _needs_bool_int_coerce(expr.left, expr.right):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            return f"{left} {op} {right}"
        # Bool arithmetic: True + True → (true as i64) + (true as i64)
        if op in ("+", "-", "*", "/", "%") and (
            _is_bool(expr.left) or _is_bool(expr.right)
        ):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            return f"{left} {op} {right}"
        # Bitwise ops
        if op in ("|", "&", "^"):
            l_bool = _is_bool(expr.left)
            r_bool = _is_bool(expr.right)
            if l_bool and r_bool:
                # Both bools: Rust bool natively supports |, &, ^
                left = self._wrap_prec(expr.left, op, False)
                right = self._wrap_prec(expr.right, op, True)
                return f"{left} {op} {right}"
            if l_bool or r_bool:
                left = self._coerce_bool_to_int(expr.left)
                right = self._coerce_bool_to_int(expr.right)
                return f"({left} {op} {right})"
        left = self._wrap_prec(expr.left, op, False)
        right = self._wrap_prec(expr.right, op, True)
        return f"{left} {op} {right}"

    def _emit_string_add(self, expr: BinaryOp) -> str:
        """Flatten chained string + into a single format! call."""
        parts: list[Expr] = []
        self._flatten_string_add(expr, parts)
        placeholders = "{}" * len(parts)
        args = ", ".join(self._emit_expr(p) for p in parts)
        return f'format!("{placeholders}", {args})'

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
            return f"!{operand}"  # Rust bitwise NOT is !
        raise NotImplementedError(f"Rust unary op: {expr.op}")

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
            return f'if {inner} {{ "True" }} else {{ "False" }}'
        # str(bool)
        if func == "str" and len(args) == 1 and args[0].typ == BOOL:
            inner = self._emit_expr(args[0])
            return f'if {inner} {{ "True" }} else {{ "False" }}'
        # str(any) → .to_string()
        if func == "str" and len(args) == 1:
            return f"{self._emit_expr(args[0])}.to_string()"
        # int(bool)
        if func == "int" and len(args) == 1 and args[0].typ == BOOL:
            return f"({self._emit_expr(args[0])} as i64)"
        # print() handled in _emit_ExprStmt; fallback here
        if func == "print":
            a = ", ".join(self._emit_expr(a) for a in args)
            return f'println!("{{}}", {a})'
        # Known module-level function
        name = self._fn_name(func) if func in self._func_names else self._safe(func)
        a = ", ".join(self._emit_expr(a) for a in args)
        return f"{name}({a})"

    def _emit_Cast(self, expr: Cast) -> str:
        inner = self._emit_expr(expr.expr)
        from_type = expr.expr.typ
        to_type = expr.to_type
        if from_type == BOOL and to_type == INT:
            return f"({inner} as i64)"
        if from_type == BOOL and to_type == STRING:
            return f'if {inner} {{ "True".to_string() }} else {{ "False".to_string() }}'
        if to_type == STRING:
            return f"{inner}.to_string()"
        if from_type == INT and to_type == FLOAT:
            return f"({inner} as f64)"
        if from_type == FLOAT and to_type == INT:
            return f"({inner} as i64)"
        if from_type == to_type:
            return inner
        return f"({inner} as {self._type_to_rust(to_type)})"

    def _emit_TupleLit(self, expr: TupleLit) -> str:
        parts: list[str] = []
        for e in expr.elements:
            s = self._emit_expr(e)
            if isinstance(e, FuncRef) or (
                isinstance(e.typ, (FuncType, InterfaceRef))
                and e.typ != VOID
            ):
                s += " as fn()"
            parts.append(s)
        return f"({', '.join(parts)})"

    def _emit_SliceLit(self, expr: SliceLit) -> str:
        elements = ", ".join(self._emit_expr(e) for e in expr.elements)
        return f"vec![{elements}]"

    def _emit_FieldAccess(self, expr: FieldAccess) -> str:
        obj = self._emit_expr(expr.obj)
        field = expr.field
        # Tuple fields: F0 → .0, F1 → .1, etc.
        if field.startswith("F") and field[1:].isdigit():
            return f"{obj}.{field[1:]}"
        return f"{obj}.{self._safe(field)}"

    def _emit_StringConcat(self, expr: StringConcat) -> str:
        placeholders = "{}" * len(expr.parts)
        args = ", ".join(self._emit_expr(p) for p in expr.parts)
        return f'format!("{placeholders}", {args})'

    def _emit_Ternary(self, expr: Ternary) -> str:
        cond = self._emit_expr(expr.cond)
        then = self._emit_expr(expr.then_expr)
        else_ = self._emit_expr(expr.else_expr)
        return f"if {cond} {{ {then} }} else {{ {else_} }}"

    def _coerce_bool_to_int(self, expr: Expr) -> str:
        if _is_bool(expr):
            return f"({self._emit_expr(expr)} as i64)"
        return self._emit_expr(expr)
