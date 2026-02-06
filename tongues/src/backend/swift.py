"""SwiftBackend: IR -> Swift code.

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
    ChainedCompare,
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
    IsNil,
    Len,
    Map,
    MapLit,
    MaxExpr,
    MinExpr,
    Module,
    NilLit,
    NoOp,
    OpAssign,
    Optional,
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
    Truthy,
    Tuple,
    TupleAssign,
    TupleLit,
    UnaryOp,
    Var,
    VarDecl,
    VarLV,
    While,
)

SWIFT_RESERVED = frozenset(
    {
        "as",
        "associatedtype",
        "break",
        "case",
        "catch",
        "class",
        "continue",
        "default",
        "defer",
        "deinit",
        "do",
        "else",
        "enum",
        "extension",
        "fallthrough",
        "false",
        "fileprivate",
        "final",
        "for",
        "func",
        "guard",
        "if",
        "import",
        "in",
        "init",
        "inout",
        "internal",
        "is",
        "let",
        "nil",
        "open",
        "operator",
        "override",
        "precedencegroup",
        "private",
        "protocol",
        "public",
        "repeat",
        "required",
        "rethrows",
        "return",
        "self",
        "Self",
        "static",
        "struct",
        "subscript",
        "super",
        "switch",
        "throw",
        "throws",
        "true",
        "try",
        "typealias",
        "var",
        "where",
        "while",
    }
)

# Swift operator precedence (higher number = tighter binding).
# Note: Swift's precedence differs from Python's - notably & is higher than +/-
_PRECEDENCE: dict[str, int] = {
    "||": 1,  # LogicalDisjunctionPrecedence
    "&&": 2,  # LogicalConjunctionPrecedence
    "==": 3,  # ComparisonPrecedence
    "!=": 3,
    "<": 3,
    "<=": 3,
    ">": 3,
    ">=": 3,
    "+": 4,  # AdditionPrecedence
    "-": 4,
    "|": 4,
    "^": 4,
    "*": 5,  # MultiplicationPrecedence
    "/": 5,
    "%": 5,
    "&": 5,
    "<<": 6,  # BitwiseShiftPrecedence
    ">>": 6,
}


def _prec(op: str) -> int:
    return _PRECEDENCE.get(op, 10)


def _is_comparison(op: str) -> bool:
    return op in ("==", "!=", "<", "<=", ">", ">=")


def _is_bool(expr: Expr) -> bool:
    """True if expression evaluates to Bool in Swift output.

    Swift doesn't support bitwise ops on Bool, so we convert to Int and back.
    But the result (via != 0) is still Bool, so bool & bool → bool.
    """
    if isinstance(expr, BinaryOp) and expr.op in ("|", "&", "^"):
        return expr.left.typ == BOOL and expr.right.typ == BOOL
    if isinstance(expr, Call) and expr.func == "bool":
        return True
    if expr.typ != BOOL:
        return False
    # These expressions produce Int in Swift even though IR type is BOOL
    if isinstance(expr, (MinExpr, MaxExpr)):
        # MinExpr/MaxExpr with any bool operand produces Int
        if expr.left.typ == BOOL or expr.right.typ == BOOL:
            return False
    if isinstance(expr, UnaryOp) and expr.op in ("-", "~"):
        return False
    if isinstance(expr, BinaryOp) and expr.op in ("+", "-", "*", "/", "%", "//", "<<", ">>"):
        return False
    return True


def _needs_bool_int_coerce(left: Expr, right: Expr) -> bool:
    return _is_bool(left) != _is_bool(right)


class SwiftBackend(Emitter):
    """Emit Swift code from IR Module."""

    def __init__(self) -> None:
        super().__init__()
        self._func_names: set[str] = set()
        self._entrypoint_fn: str | None = None

    def emit(self, module: Module) -> str:
        self.lines: list[str] = []
        self.indent = 0
        self._func_names = {f.name for f in module.functions}
        self._entrypoint_fn = module.entrypoint.function_name if module.entrypoint else None
        self.line("import Foundation")
        self.line("")
        for func in module.functions:
            self._emit_function(func)
        if module.entrypoint is not None:
            self.line(f"exit(Int32(_{module.entrypoint.function_name}()))")
        return self.output() + "\n"

    # ── helpers ──────────────────────────────────────────────

    def _safe(self, name: str) -> str:
        if name in SWIFT_RESERVED:
            return "`" + name + "`"
        return name

    def _fn_name(self, name: str) -> str:
        """Map IR function name to Swift function name."""
        if name == self._entrypoint_fn:
            return f"_{name}"
        return self._safe(name)

    def _type_to_swift(self, typ: "type | object") -> str:
        if isinstance(typ, Primitive):
            return {
                "int": "Int",
                "float": "Double",
                "bool": "Bool",
                "string": "String",
                "void": "Void",
                "byte": "UInt8",
                "rune": "Character",
            }[typ.kind]
        if isinstance(typ, Slice):
            return f"[{self._type_to_swift(typ.element)}]"
        if isinstance(typ, Tuple):
            inner = ", ".join(self._type_to_swift(t) for t in typ.elements)
            return f"({inner})"
        if isinstance(typ, FuncType):
            params = ", ".join(self._type_to_swift(p) for p in typ.params)
            if typ.ret == VOID:
                return f"({params}) -> Void"
            return f"({params}) -> {self._type_to_swift(typ.ret)}"
        if isinstance(typ, InterfaceRef) and typ.name == "any":
            return "() -> Void"
        if isinstance(typ, StructRef):
            return typ.name
        raise NotImplementedError(f"Swift type: {typ}")

    def _maybe_paren(self, expr: Expr, parent_op: str, is_left: bool) -> str:
        """Emit expr, adding parens if its precedence requires it."""
        s = self._emit_expr(expr)
        if isinstance(expr, BinaryOp):
            # Swift doesn't allow chained comparisons
            if _is_comparison(parent_op) and _is_comparison(expr.op):
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
        params = ", ".join(f"_ {self._safe(p.name)}: {self._param_type(p)}" for p in func.params)
        name = self._fn_name(func.name)
        if func.ret == VOID:
            self.line(f"func {name}({params}) {{")
        else:
            ret = self._type_to_swift(func.ret)
            self.line(f"func {name}({params}) -> {ret} {{")
        self.indent += 1
        for stmt in func.body:
            self._emit_stmt(stmt)
        self.indent -= 1
        self.line("}")
        self.line("")

    def _param_type(self, p: Param) -> str:
        return self._type_to_swift(p.typ)

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
        elif isinstance(stmt, While):
            self._emit_While(stmt)
        elif isinstance(stmt, TupleAssign):
            self._emit_TupleAssign(stmt)
        elif isinstance(stmt, EntryPoint):
            pass  # handled at module level
        elif isinstance(stmt, NoOp):
            pass
        else:
            raise NotImplementedError(f"Swift stmt: {stmt}")

    def _emit_VarDecl(self, s: VarDecl) -> None:
        name = self._safe(s.name)
        kw = "var" if s.mutable else "let"
        if s.value is not None:
            val = self._emit_expr(s.value)
            self.line(f"{kw} {name} = {val}")
        else:
            typ = self._type_to_swift(s.typ)
            self.line(f"{kw} {name}: {typ}")

    def _emit_Assign(self, s: Assign) -> None:
        target = self._emit_lvalue(s.target)
        val = self._emit_expr(s.value)
        if s.is_declaration:
            self.line(f"var {target} = {val}")
        else:
            self.line(f"{target} = {val}")

    def _emit_OpAssign(self, s: OpAssign) -> None:
        target = self._emit_lvalue(s.target)
        val = self._emit_expr(s.value)
        self.line(f"{target} {s.op}= {val}")

    def _emit_Return(self, s: Return) -> None:
        if s.value is None:
            self.line("return")
        else:
            self.line(f"return {self._emit_expr(s.value)}")

    def _emit_Assert(self, s: Assert) -> None:
        test = self._emit_expr(s.test)
        if s.message is not None:
            msg = self._emit_expr(s.message)
            self.line(f'if !({test}) {{ fatalError("\\({msg})") }}')
        else:
            self.line(f'if !({test}) {{ fatalError("assertion failed") }}')

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
            self.line(f"for ({idx}, {val}) in {iterable}.enumerated() {{")
        else:
            self.line(f"for {val} in {iterable} {{")
        self.indent += 1
        for st in s.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}")

    def _emit_TryCatch(self, s: TryCatch) -> None:
        self.line("do {")
        self.indent += 1
        for st in s.body:
            self._emit_stmt(st)
        self.indent -= 1
        if not s.catches:
            self.line("}")
            return
        catch = s.catches[0]
        var = self._safe(catch.var) if catch.var else "_e"
        self.line(f"}} catch let {var} {{")
        self.indent += 1
        self.line(f"let {var} = String(describing: {var})")
        for st in catch.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}")

    def _emit_ExprStmt(self, s: ExprStmt) -> None:
        if isinstance(s.expr, Call) and s.expr.func == "print":
            args = s.expr.args
            if args:
                val = self._emit_expr(args[0])
                self.line(f"print({val})")
            else:
                self.line("print()")
            return
        expr = self._emit_expr(s.expr)
        self.line(f"_ = {expr}")

    def _emit_Block(self, s: Block) -> None:
        self.line("do {")
        self.indent += 1
        for st in s.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}")

    def _emit_Print(self, s: Print) -> None:
        val = self._emit_expr(s.value)
        if s.stderr:
            if s.newline:
                self.line(f'fputs("\\({val})\\n", stderr)')
            else:
                self.line(f'fputs("\\({val})", stderr)')
        else:
            if s.newline:
                self.line(f"print({val})")
            else:
                self.line(f'print({val}, terminator: "")')

    def _emit_While(self, s: While) -> None:
        cond = self._emit_expr(s.cond)
        self.line(f"while {cond} {{")
        self.indent += 1
        for st in s.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}")

    def _emit_TupleAssign(self, s: TupleAssign) -> None:
        targets = [self._emit_lvalue(t) for t in s.targets]
        val = self._emit_expr(s.value)
        if s.is_declaration:
            self.line(f"var ({', '.join(targets)}) = {val}")
        else:
            self.line(f"({', '.join(targets)}) = {val}")

    def _emit_lvalue(self, lv: "object") -> str:
        if isinstance(lv, VarLV):
            return self._safe(lv.name)
        raise NotImplementedError(f"Swift lvalue: {lv}")

    # ── expressions ──────────────────────────────────────────

    def _emit_expr(self, expr: Expr) -> str:
        if isinstance(expr, IntLit):
            return str(expr.value)
        if isinstance(expr, BoolLit):
            return "true" if expr.value else "false"
        if isinstance(expr, NilLit):
            return "nil"
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
            return f"String({self._emit_expr(expr.value)})"
        if isinstance(expr, Ternary):
            return self._emit_Ternary(expr)
        if isinstance(expr, Truthy):
            return self._emit_Truthy(expr)
        if isinstance(expr, IsNil):
            return self._emit_IsNil(expr)
        if isinstance(expr, MinExpr):
            return self._emit_MinExpr(expr)
        if isinstance(expr, MaxExpr):
            return self._emit_MaxExpr(expr)
        if isinstance(expr, ChainedCompare):
            return self._emit_ChainedCompare(expr)
        if isinstance(expr, Len):
            return self._emit_Len(expr)
        if isinstance(expr, MapLit):
            return self._emit_MapLit(expr)
        raise NotImplementedError(f"Swift expr: {expr}")

    def _emit_Var(self, expr: Var) -> str:
        name = self._safe(expr.name)
        if isinstance(expr.typ, FuncType) or (
            isinstance(expr.typ, InterfaceRef) and expr.typ.name == "any"
        ):
            return self._fn_name(expr.name) if expr.name in self._func_names else name
        return name

    def _emit_BinaryOp(self, expr: BinaryOp) -> str:
        op = expr.op
        # String concatenation
        if op == "+" and expr.typ == STRING:
            return self._emit_string_add(expr)
        # Bool/int comparison coercion: True == 1 → (true ? 1 : 0) == 1
        if op in ("==", "!=") and _needs_bool_int_coerce(expr.left, expr.right):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            return f"{left} {op} {right}"
        # Floor division - Swift integer division already floors
        if op == "//":
            op = "/"
        # Bool arithmetic: True + True → (true ? 1 : 0) + (true ? 1 : 0)
        if op in ("+", "-", "*", "/", "%") and (_is_bool(expr.left) or _is_bool(expr.right)):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            # Need proper precedence for nested operations
            if not _is_bool(expr.left):
                left = self._maybe_paren(expr.left, op, is_left=True)
            if not _is_bool(expr.right):
                right = self._maybe_paren(expr.right, op, is_left=False)
            return f"{left} {op} {right}"
        # Shift operations with bools
        if op in ("<<", ">>") and (_is_bool(expr.left) or _is_bool(expr.right)):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            if not _is_bool(expr.left):
                left = self._maybe_paren(expr.left, op, is_left=True)
            if not _is_bool(expr.right):
                right = self._maybe_paren(expr.right, op, is_left=False)
            return f"{left} {op} {right}"
        # Ordered comparisons with bools
        if op in (">", "<", ">=", "<=") and (_is_bool(expr.left) or _is_bool(expr.right)):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            return f"{left} {op} {right}"
        # Bitwise ops - Swift doesn't support bitwise ops on Bool
        if op in ("|", "&", "^"):
            l_bool = _is_bool(expr.left)
            r_bool = _is_bool(expr.right)
            if l_bool and r_bool:
                # Both bools: convert to int, do bitwise, convert result back
                left = self._coerce_bool_to_int(expr.left)
                right = self._coerce_bool_to_int(expr.right)
                return f"(({left} {op} {right}) != 0)"
            if l_bool or r_bool:
                left = self._coerce_bool_to_int(expr.left)
                right = self._coerce_bool_to_int(expr.right)
                return f"({left} {op} {right})"
        left = self._maybe_paren(expr.left, op, is_left=True)
        right = self._maybe_paren(expr.right, op, is_left=False)
        return f"{left} {op} {right}"

    def _emit_string_add(self, expr: BinaryOp) -> str:
        """Flatten chained string + into a single interpolation."""
        parts: list[Expr] = []
        self._flatten_string_add(expr, parts)
        segments = []
        for p in parts:
            if isinstance(p, StringLit):
                segments.append(escape_string(p.value))
            else:
                segments.append(f"\\({self._emit_expr(p)})")
        return '"' + "".join(segments) + '"'

    def _flatten_string_add(self, expr: Expr, out: list[Expr]) -> None:
        if isinstance(expr, BinaryOp) and expr.op == "+" and expr.typ == STRING:
            self._flatten_string_add(expr.left, out)
            self._flatten_string_add(expr.right, out)
        else:
            out.append(expr)

    def _emit_UnaryOp(self, expr: UnaryOp) -> str:
        operand_is_bool = _is_bool(expr.operand)
        # Swift bools don't support unary minus or bitwise NOT
        if expr.op in ("-", "~") and operand_is_bool:
            return f"{expr.op}({self._emit_expr(expr.operand)} ? 1 : 0)"
        operand = self._emit_expr(expr.operand)
        # Swift doesn't allow juxtaposed unary operators, and needs parens for binops
        needs_paren = (
            isinstance(expr.operand, UnaryOp)
            or isinstance(expr.operand, BinaryOp)
            or (isinstance(expr.operand, IntLit) and expr.operand.value < 0)
        )
        if needs_paren:
            operand = f"({operand})"
        if expr.op == "!":
            return f"!{operand}"
        if expr.op == "-":
            return f"-{operand}"
        if expr.op == "~":
            return f"~{operand}"
        raise NotImplementedError(f"Swift unary op: {expr.op}")

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
            return f'({inner} ? "True" : "False")'
        # str(bool)
        if func == "str" and len(args) == 1 and args[0].typ == BOOL:
            inner = self._emit_expr(args[0])
            return f'({inner} ? "True" : "False")'
        # str(any) → String()
        if func == "str" and len(args) == 1:
            return f"String({self._emit_expr(args[0])})"
        # int(bool)
        if func == "int" and len(args) == 1 and args[0].typ == BOOL:
            return f"({self._emit_expr(args[0])} ? 1 : 0)"
        # print() handled in _emit_ExprStmt; fallback here
        if func == "print":
            a = ", ".join(self._emit_expr(a) for a in args)
            return f"print({a})"
        # abs() with bool
        if func == "abs" and len(args) == 1 and args[0].typ == BOOL:
            return f"({self._emit_expr(args[0])} ? 1 : 0)"
        # abs() with int
        if func == "abs" and len(args) == 1:
            return f"abs({self._emit_expr(args[0])})"
        # pow() - Swift uses Foundation's pow() which returns Double
        if func == "pow" and len(args) == 2:
            arg0, arg1 = args[0], args[1]
            a0 = self._coerce_bool_to_int(arg0) if _is_bool(arg0) else self._emit_expr(arg0)
            a1 = self._coerce_bool_to_int(arg1) if _is_bool(arg1) else self._emit_expr(arg1)
            return f"Int(pow(Double({a0}), Double({a1})))"
        # divmod()
        if func == "divmod" and len(args) == 2:
            arg0, arg1 = args[0], args[1]
            a0 = self._coerce_bool_to_int(arg0) if _is_bool(arg0) else self._emit_expr(arg0)
            a1 = self._coerce_bool_to_int(arg1) if _is_bool(arg1) else self._emit_expr(arg1)
            return f"({a0} / {a1}, {a0} % {a1})"
        # Known module-level function
        name = self._fn_name(func) if func in self._func_names else self._safe(func)
        a = ", ".join(self._emit_expr(a) for a in args)
        return f"{name}({a})"

    def _emit_Cast(self, expr: Cast) -> str:
        inner_expr = expr.expr
        inner = self._emit_expr(inner_expr)
        from_type = inner_expr.typ
        to_type = expr.to_type
        if from_type == BOOL and to_type == INT:
            # MinExpr/MaxExpr with bool operands already produce int
            if isinstance(inner_expr, (MinExpr, MaxExpr)):
                return inner
            # UnaryOp('-' or '~') on bool already produces int
            if isinstance(inner_expr, UnaryOp) and inner_expr.op in ("-", "~"):
                return inner
            return f"({inner} ? 1 : 0)"
        if from_type == BOOL and to_type == STRING:
            return f'({inner} ? "True" : "False")'
        if to_type == STRING:
            return f"String({inner})"
        if from_type == INT and to_type == FLOAT:
            return f"Double({inner})"
        if from_type == FLOAT and to_type == INT:
            return f"Int({inner})"
        if from_type == to_type:
            return inner
        return f"{self._type_to_swift(to_type)}({inner})"

    def _emit_TupleLit(self, expr: TupleLit) -> str:
        parts = [self._emit_expr(e) for e in expr.elements]
        return f"({', '.join(parts)})"

    def _emit_SliceLit(self, expr: SliceLit) -> str:
        elements = ", ".join(self._emit_expr(e) for e in expr.elements)
        return f"[{elements}]"

    def _emit_FieldAccess(self, expr: FieldAccess) -> str:
        obj = self._emit_expr(expr.obj)
        field = expr.field
        # Tuple fields: F0 → .0, F1 → .1, etc.
        if field.startswith("F") and field[1:].isdigit():
            return f"{obj}.{field[1:]}"
        return f"{obj}.{self._safe(field)}"

    def _emit_StringConcat(self, expr: StringConcat) -> str:
        segments = []
        for p in expr.parts:
            if isinstance(p, StringLit):
                segments.append(escape_string(p.value))
            else:
                segments.append(f"\\({self._emit_expr(p)})")
        return '"' + "".join(segments) + '"'

    def _emit_Ternary(self, expr: Ternary) -> str:
        cond = self._emit_expr(expr.cond)
        then = self._emit_expr(expr.then_expr)
        else_ = self._emit_expr(expr.else_expr)
        return f"({cond} ? {then} : {else_})"

    def _emit_Truthy(self, expr: Truthy) -> str:
        inner = self._emit_expr(expr.expr)
        inner_type = expr.expr.typ
        if inner_type == STRING:
            return f"!{inner}.isEmpty"
        if inner_type == INT:
            if isinstance(expr.expr, BinaryOp):
                return f"(({inner}) != 0)"
            return f"({inner} != 0)"
        if isinstance(inner_type, Slice):
            return f"!{inner}.isEmpty"
        if isinstance(inner_type, Optional):
            return f"({inner} != nil)"
        return f"({inner} != nil)"

    def _emit_IsNil(self, expr: IsNil) -> str:
        inner = self._emit_expr(expr.expr)
        if expr.negated:
            return f"({inner} != nil)"
        return f"({inner} == nil)"

    def _coerce_bool_to_int(self, expr: Expr) -> str:
        if _is_bool(expr):
            return f"({self._emit_expr(expr)} ? 1 : 0)"
        return self._emit_expr(expr)

    def _emit_MinExpr(self, expr: MinExpr) -> str:
        l_bool = _is_bool(expr.left)
        r_bool = _is_bool(expr.right)
        if l_bool or r_bool:
            l = self._coerce_bool_to_int(expr.left)
            r = self._coerce_bool_to_int(expr.right)
        else:
            l, r = self._emit_expr(expr.left), self._emit_expr(expr.right)
        return f"min({l}, {r})"

    def _emit_MaxExpr(self, expr: MaxExpr) -> str:
        l_bool = _is_bool(expr.left)
        r_bool = _is_bool(expr.right)
        if l_bool or r_bool:
            l = self._coerce_bool_to_int(expr.left)
            r = self._coerce_bool_to_int(expr.right)
        else:
            l, r = self._emit_expr(expr.left), self._emit_expr(expr.right)
        return f"max({l}, {r})"

    def _emit_ChainedCompare(self, expr: ChainedCompare) -> str:
        parts = []
        for i, op in enumerate(expr.ops):
            left_str = self._emit_expr(expr.operands[i])
            right_str = self._emit_expr(expr.operands[i + 1])
            parts.append(f"{left_str} {op} {right_str}")
        return "(" + " && ".join(parts) + ")"

    def _emit_Len(self, expr: Len) -> str:
        inner = self._emit_expr(expr.expr)
        return f"{inner}.count"

    def _emit_MapLit(self, expr: MapLit) -> str:
        if not expr.entries:
            # Empty dict with type annotation
            key_type = self._type_to_swift(expr.key_type)
            val_type = self._type_to_swift(expr.value_type)
            return f"[{key_type}: {val_type}]()"
        pairs = ", ".join(f"{self._emit_expr(k)}: {self._emit_expr(v)}" for k, v in expr.entries)
        return f"[{pairs}]"
