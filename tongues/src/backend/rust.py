"""RustBackend: IR -> Rust code.

Minimal backend targeting apptest_boolobject.py. Unhandled IR nodes
raise NotImplementedError so gaps are obvious.
"""

from __future__ import annotations

from src.backend.util import Emitter


def escape_string_rust(value: str) -> str:
    """Escape a string for Rust string literals."""
    result = []
    for ch in value:
        code = ord(ch)
        if ch == "\\":
            result.append("\\\\")
        elif ch == '"':
            result.append('\\"')
        elif ch == "\n":
            result.append("\\n")
        elif ch == "\t":
            result.append("\\t")
        elif ch == "\r":
            result.append("\\r")
        elif ch == "\x0c":  # form feed
            result.append("\\x0c")
        elif ch == "\x0b":  # vertical tab
            result.append("\\x0b")
        elif code < 32 or code == 127:
            # Control characters: use \x or \u{} format
            if code < 256:
                result.append(f"\\x{code:02x}")
            else:
                result.append(f"\\u{{{code:04x}}}")
        elif code > 127:
            # Non-ASCII: use \u{} for safety
            result.append(f"\\u{{{code:04x}}}")
        else:
            result.append(ch)
    return "".join(result)


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
    Break,
    Call,
    Cast,
    CharClassify,
    Continue,
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
    Map,
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
    SliceLit,
    StringConcat,
    StringLit,
    StructRef,
    Ternary,
    TryCatch,
    Tuple,
    TupleAssign,
    TupleLit,
    UnaryOp,
    Var,
    VarDecl,
    VarLV,
    While,
    IndexLV,
    Truthy,
    IsNil,
    NilLit,
    DictComp,
    ListComp,
    CompGenerator,
    FloatLit,
    SetLit,
    Set,
    Optional,
    StaticCall,
)

RUST_RESERVED = frozenset(
    {
        "as",
        "async",
        "await",
        "break",
        "const",
        "continue",
        "crate",
        "dyn",
        "else",
        "enum",
        "extern",
        "false",
        "fn",
        "for",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "match",
        "mod",
        "move",
        "mut",
        "pub",
        "ref",
        "return",
        "self",
        "Self",
        "static",
        "struct",
        "super",
        "trait",
        "true",
        "type",
        "union",
        "unsafe",
        "use",
        "where",
        "while",
        "abstract",
        "become",
        "box",
        "do",
        "final",
        "macro",
        "override",
        "priv",
        "try",
        "typeof",
        "unsized",
        "virtual",
        "yield",
    }
)

# Rust operator precedence (higher number = tighter binding).
# In Rust, bitwise ops bind tighter than comparisons (unlike C).
_PRECEDENCE: dict[str, int] = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 3,
    "<=": 3,
    ">": 3,
    ">=": 3,
    "|": 4,
    "^": 5,
    "&": 6,
    "<<": 7,
    ">>": 7,
    "+": 8,
    "-": 8,
    "*": 9,
    "/": 9,
    "%": 9,
}


def _prec(op: str) -> int:
    return _PRECEDENCE.get(op, 10)


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
        self.lines: list[str] = []
        self.indent = 0
        self._func_names = {f.name for f in module.functions}
        self._needs_catch_unwind = False
        self._entrypoint_fn = (
            module.entrypoint.function_name if module.entrypoint else None
        )
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
            self.line(
                f"std::process::exit(_{module.entrypoint.function_name}() as i32);"
            )
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
            return {
                "int": "i64",
                "float": "f64",
                "bool": "bool",
                "string": "&str",
                "void": "()",
                "byte": "u8",
                "rune": "char",
            }[typ.kind]
        if isinstance(typ, Slice):
            return f"Vec<{self._type_to_rust(typ.element)}>"
        if isinstance(typ, Set):
            return f"std::collections::HashSet<{self._type_to_rust(typ.element)}>"
        if isinstance(typ, Map):
            # Handle any/unknown types in map - use i64 as default
            if isinstance(typ.key, InterfaceRef) and typ.key.name == "any":
                key_type = "i64"
            else:
                key_type = self._type_to_rust(typ.key)
            if isinstance(typ.value, InterfaceRef) and typ.value.name == "any":
                val_type = "i64"
            else:
                val_type = self._type_to_rust(typ.value)
            return f"std::collections::HashMap<{key_type}, {val_type}>"
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
        if isinstance(typ, InterfaceRef) and typ.name == "None":
            return "()"  # Unit type for None-only type
        if isinstance(typ, Optional):
            inner = self._type_to_rust(typ.inner)
            return f"Option<{inner}>"
        if isinstance(typ, StructRef):
            return typ.name
        raise NotImplementedError(f"Rust type: {typ}")

    def _maybe_paren(self, expr: Expr, parent_op: str, is_left: bool) -> str:
        """Emit expr, adding parens if its precedence requires it."""
        s = self._emit_expr(expr)
        if isinstance(expr, BinaryOp):
            child_prec = _prec(expr.op)
            parent_prec = _prec(parent_op)
            # Comparison operators are non-associative in Rust
            cmp_ops = ("==", "!=", "<", "<=", ">", ">=")
            if parent_op in cmp_ops and expr.op in cmp_ops:
                return f"({s})"
            if not is_left:
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
        elif isinstance(stmt, TupleAssign):
            self._emit_TupleAssign(stmt)
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
        elif isinstance(stmt, While):
            self._emit_While(stmt)
        elif isinstance(stmt, Break):
            self.line("break;")
        elif isinstance(stmt, Continue):
            self.line("continue;")
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
            raise NotImplementedError(f"Rust stmt: {stmt}")

    def _emit_VarDecl(self, s: VarDecl) -> None:
        name = self._safe(s.name)
        mut = "mut " if s.mutable else ""
        if s.value is not None:
            # Special handling for empty MapLit - use VarDecl type
            if (
                isinstance(s.value, MapLit)
                and isinstance(s.typ, Map)
                and not s.value.entries
            ):
                typ = self._type_to_rust(s.typ)
                self.line(f"let {mut}{name}: {typ} = std::collections::HashMap::new();")
                return
            # Special handling for MapLit when declared type has Optional values
            if (
                isinstance(s.value, MapLit)
                and isinstance(s.typ, Map)
                and isinstance(s.typ.value, Optional)
            ):
                val = self._emit_MapLit_with_optional(s.value, s.typ.value)
            else:
                val = self._emit_expr(s.value)
            # Add type annotation for slices, maps, and sets to help Rust infer types
            if isinstance(s.typ, (Slice, Map, Set)):
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

    def _emit_TupleAssign(self, s: TupleAssign) -> None:
        targets = ", ".join(
            "_"
            if i in s.unused_indices
            else f"mut {self._emit_lvalue(t)}"
            if s.is_declaration
            else self._emit_lvalue(t)
            for i, t in enumerate(s.targets)
        )
        val = self._emit_expr(s.value)
        if s.is_declaration:
            self.line(f"let ({targets}) = {val};")
        else:
            self.line(f"({targets}) = {val};")

    def _emit_OpAssign(self, s: OpAssign) -> None:
        target = self._emit_lvalue(s.target)
        val = self._emit_expr(s.value)
        # Dict merge operator |= uses extend
        if s.op == "|" and isinstance(s.value.typ, Map):
            self.line(f"{target}.extend({val}.iter().map(|(k, v)| (k.clone(), *v)));")
            return
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
            self.line(f'if !({test}) {{ panic!("{{}}", {msg}); }}')
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
        # Iterating over a dict yields keys
        if isinstance(s.iterable.typ, Map):
            self.line(f"for {val} in {iterable}.keys() {{")
        elif s.index:
            idx = self._safe(s.index)
            self.line(f"for ({idx}, {val}) in {iterable}.iter().enumerate() {{")
        else:
            self.line(f"for {val} in {iterable}.iter() {{")
        self.indent += 1
        for st in s.body:
            self._emit_stmt(st)
        self.indent -= 1
        self.line("}")

    def _emit_While(self, s: While) -> None:
        cond = self._emit_expr(s.cond)
        self.line(f"while {cond} {{")
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
        self.line(
            f"let {var}: String = if let Some(s) = _panic_val.downcast_ref::<String>() {{"
        )
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
                self.line(f'println!("{{}}", {val});')
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
        self.line(f'{macro}!("{{}}", {val});')

    def _emit_lvalue(self, lv: "object") -> str:
        if isinstance(lv, VarLV):
            return self._safe(lv.name)
        if isinstance(lv, IndexLV):
            idx = self._emit_expr(lv.index)
            # Map indexing
            if isinstance(lv.obj.typ, Map):
                obj = self._emit_expr(lv.obj)
                return f"*{obj}.entry({idx}).or_insert(Default::default())"
            # Nested map indexing - need get_mut for mutable access to inner map
            if isinstance(lv.obj, Index) and isinstance(lv.obj.obj.typ, Map):
                inner_map = lv.obj.obj.typ
                if isinstance(inner_map.value, Map):
                    # Use get_mut for mutable access to inner map
                    outer_obj = self._emit_expr(lv.obj.obj)
                    outer_key = self._emit_expr(lv.obj.index)
                    return f"*{outer_obj}.get_mut(&{outer_key}).unwrap().entry({idx}).or_insert(Default::default())"
            # Slice/vec indexing
            obj = self._emit_expr(lv.obj)
            return f"{obj}[{idx} as usize]"
        raise NotImplementedError(f"Rust lvalue: {lv}")

    # ── expressions ──────────────────────────────────────────

    def _emit_expr(self, expr: Expr) -> str:
        if isinstance(expr, IntLit):
            return str(expr.value)
        if isinstance(expr, FloatLit):
            # Ensure float representation has decimal point
            s = str(expr.value)
            if "." not in s and "e" not in s.lower():
                s += ".0"
            return s
        if isinstance(expr, BoolLit):
            return "true" if expr.value else "false"
        if isinstance(expr, NilLit):
            return "None"
        if isinstance(expr, StringLit):
            return f'"{escape_string_rust(expr.value)}"'
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
        if isinstance(expr, MinExpr):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            # Coerce bool to int if mixing bool and int
            if _needs_bool_int_coerce(expr.left, expr.right):
                if _is_bool(expr.left):
                    left = f"({left} as i64)"
                if _is_bool(expr.right):
                    right = f"({right} as i64)"
            return f"std::cmp::min({left}, {right})"
        if isinstance(expr, MaxExpr):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            # Coerce bool to int if mixing bool and int
            if _needs_bool_int_coerce(expr.left, expr.right):
                if _is_bool(expr.left):
                    left = f"({left} as i64)"
                if _is_bool(expr.right):
                    right = f"({right} as i64)"
            return f"std::cmp::max({left}, {right})"
        if isinstance(expr, Len):
            inner = self._emit_expr(expr.expr)
            return f"{inner}.len() as i64"
        if isinstance(expr, MapLit):
            return self._emit_MapLit(expr)
        if isinstance(expr, Index):
            return self._emit_Index(expr)
        if isinstance(expr, CharClassify):
            return self._emit_CharClassify(expr)
        if isinstance(expr, MethodCall):
            return self._emit_MethodCall(expr)
        if isinstance(expr, Truthy):
            return self._emit_Truthy(expr)
        if isinstance(expr, IsNil):
            return self._emit_IsNil(expr)
        if isinstance(expr, DictComp):
            return self._emit_DictComp(expr)
        if isinstance(expr, ListComp):
            return self._emit_ListComp(expr)
        if isinstance(expr, SetLit):
            return self._emit_SetLit(expr)
        if isinstance(expr, StaticCall):
            return self._emit_StaticCall(expr)
        raise NotImplementedError(f"Rust expr: {expr}")

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
        # Dict merge operator |
        if op == "|" and isinstance(expr.left.typ, Map):
            # Handle empty dict cases - d | {} = d, {} | d = d
            if isinstance(expr.right, MapLit) and not expr.right.entries:
                return f"{self._emit_expr(expr.left)}.clone()"
            if isinstance(expr.left, MapLit) and not expr.left.entries:
                return f"{self._emit_expr(expr.right)}.clone()"
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            return f"{{ let mut m = {left}.clone(); m.extend({right}.iter().map(|(k, v)| (k.clone(), *v))); m }}"
        # Set operations on dict keys/items views
        if op in ("&", "|", "^", "-"):
            left_is_keys = isinstance(expr.left, MethodCall) and expr.left.method in (
                "keys",
                "items",
            )
            right_is_keys = isinstance(
                expr.right, MethodCall
            ) and expr.right.method in (
                "keys",
                "items",
            )
            if left_is_keys and right_is_keys:
                left = self._emit_expr(expr.left)
                right = self._emit_expr(expr.right)
                # Convert to HashSet and apply set operation
                left_set = (
                    f"{left}.into_iter().collect::<std::collections::HashSet<_>>()"
                )
                right_set = (
                    f"{right}.into_iter().collect::<std::collections::HashSet<_>>()"
                )
                if op == "&":
                    return f"{left_set}.intersection(&{right_set}).cloned().collect::<std::collections::HashSet<_>>()"
                if op == "|":
                    return f"{left_set}.union(&{right_set}).cloned().collect::<std::collections::HashSet<_>>()"
                if op == "^":
                    return f"{left_set}.symmetric_difference(&{right_set}).cloned().collect::<std::collections::HashSet<_>>()"
                if op == "-":
                    return f"{left_set}.difference(&{right_set}).cloned().collect::<std::collections::HashSet<_>>()"
        # "in" operator for membership testing
        if op == "in":
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            right_type = expr.right.typ
            if isinstance(right_type, Map):
                return f"{right}.contains_key(&{left})"
            if isinstance(right_type, Set):
                return f"{right}.contains(&{left})"
            if isinstance(right_type, Slice):
                return f"{right}.contains(&{left})"
            if right_type == STRING:
                return f"{right}.contains({left})"
            return f"{right}.contains(&{left})"
        # "not in" operator
        if op == "not in":
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            right_type = expr.right.typ
            if isinstance(right_type, Map):
                return f"!{right}.contains_key(&{left})"
            if isinstance(right_type, Set):
                return f"!{right}.contains(&{left})"
            if isinstance(right_type, Slice):
                return f"!{right}.contains(&{left})"
            if right_type == STRING:
                return f"!{right}.contains({left})"
            return f"!{right}.contains(&{left})"
        # Floor division - Rust integer division already floors
        if op == "//":
            op = "/"
        # Comparison with empty MapLit - use the other operand's type
        if op in ("==", "!="):
            # {} == {} - both empty maps, compare as true/false directly
            left_empty = isinstance(expr.left, MapLit) and not expr.left.entries
            right_empty = isinstance(expr.right, MapLit) and not expr.right.entries
            if left_empty and right_empty:
                return "true" if op == "==" else "false"
            if right_empty and isinstance(expr.left.typ, Map):
                left = self._emit_expr(expr.left)
                # Create empty map with same type as left operand (turbofish syntax)
                map_type = expr.left.typ
                key_type = self._type_to_rust(map_type.key)
                val_type = self._type_to_rust(map_type.value)
                return f"{left} {op} std::collections::HashMap::<{key_type}, {val_type}>::new()"
            if left_empty and isinstance(expr.right.typ, Map):
                right = self._emit_expr(expr.right)
                map_type = expr.right.typ
                key_type = self._type_to_rust(map_type.key)
                val_type = self._type_to_rust(map_type.value)
                return f"std::collections::HashMap::<{key_type}, {val_type}>::new() {op} {right}"
        # Comparison with map.get() result (Option) - wrap other side in Some()
        if op in ("==", "!="):
            # Check if either side is a MethodCall returning Option (get without default on Optional value dict)
            left_returns_option = False
            right_returns_option = False
            # Check for .get(key) without default
            if (
                isinstance(expr.left, MethodCall)
                and expr.left.method == "get"
                and len(expr.left.args) == 1
            ):
                left_returns_option = True
            # Check for .get(key, default) on Optional value dict - still returns Option
            if (
                isinstance(expr.left, MethodCall)
                and expr.left.method == "get"
                and len(expr.left.args) == 2
            ):
                if isinstance(expr.left.receiver_type, Map) and isinstance(
                    expr.left.receiver_type.value, Optional
                ):
                    left_returns_option = True
            if (
                isinstance(expr.right, MethodCall)
                and expr.right.method == "get"
                and len(expr.right.args) == 1
            ):
                right_returns_option = True
            if (
                isinstance(expr.right, MethodCall)
                and expr.right.method == "get"
                and len(expr.right.args) == 2
            ):
                if isinstance(expr.right.receiver_type, Map) and isinstance(
                    expr.right.receiver_type.value, Optional
                ):
                    right_returns_option = True
            if left_returns_option and not right_returns_option:
                left = self._emit_expr(expr.left)
                right = self._emit_expr(expr.right)
                return f"{left} {op} Some({right})"
            if right_returns_option and not left_returns_option:
                left = self._emit_expr(expr.left)
                right = self._emit_expr(expr.right)
                return f"Some({left}) {op} {right}"
        # String concatenation: &str + &str doesn't work in Rust, use format!
        if op == "+" and expr.typ == STRING:
            return self._emit_string_add(expr)
        # Bool/int comparison coercion: True == 1 → (true as i64) == 1
        # Also handles <, <=, >, >= with bool operands
        if op in ("==", "!=", "<", "<=", ">", ">=") and _needs_bool_int_coerce(
            expr.left, expr.right
        ):
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
        # Shift ops with bool operands need coercion
        if op in ("<<", ">>") and (_is_bool(expr.left) or _is_bool(expr.right)):
            left = self._coerce_bool_to_int(expr.left)
            right = self._coerce_bool_to_int(expr.right)
            return f"{left} {op} {right}"
        # Bitwise ops
        if op in ("|", "&", "^"):
            l_bool = _is_bool(expr.left)
            r_bool = _is_bool(expr.right)
            if l_bool and r_bool:
                # Both bools: Rust bool natively supports |, &, ^
                left = self._maybe_paren(expr.left, op, is_left=True)
                right = self._maybe_paren(expr.right, op, is_left=False)
                return f"{left} {op} {right}"
            if l_bool or r_bool:
                left = self._coerce_bool_to_int(expr.left)
                right = self._coerce_bool_to_int(expr.right)
                return f"({left} {op} {right})"
        left = self._maybe_paren(expr.left, op, is_left=True)
        right = self._maybe_paren(expr.right, op, is_left=False)
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
        # Wrap in parens if operand is a binary op (precedence issue)
        needs_parens = isinstance(expr.operand, BinaryOp)
        if expr.op == "!":
            if needs_parens:
                return f"!({operand})"
            return f"!{operand}"
        if expr.op == "-":
            # Unary minus on bool needs coercion: -True → -(true as i64)
            if _is_bool(expr.operand):
                return f"-({operand} as i64)"
            if needs_parens:
                return f"-({operand})"
            return f"-{operand}"
        if expr.op == "~":
            # Bitwise NOT: In Rust, ! on bool is logical NOT, not bitwise
            # ~True in Python is -2, so we need: !((true as i64))
            if _is_bool(expr.operand):
                return f"!({operand} as i64)"
            if needs_parens:
                return f"!({operand})"
            return f"!{operand}"
        if expr.op == "*":
            # Dereference - skip for map.get() with default, as unwrap_or already unwraps
            if isinstance(expr.operand, MethodCall):
                if (
                    isinstance(expr.operand.receiver_type, Map)
                    and expr.operand.method == "get"
                ):
                    if len(expr.operand.args) >= 2:
                        # .get(key, default) already returns unwrapped value
                        return operand
            if needs_parens:
                return f"*({operand})"
            return f"*{operand}"
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
        # repr(string) - single char string gives 'x' format
        if func == "repr" and len(args) == 1 and args[0].typ == STRING:
            inner = self._emit_expr(args[0])
            return f"format!(\"'{{}}'\", {inner})"
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
        # abs() builtin
        if func == "abs" and len(args) == 1:
            inner = self._emit_expr(args[0])
            if _is_bool(args[0]):
                return f"({inner} as i64).abs()"
            return f"{inner}.abs()"
        # pow() builtin
        if func == "pow" and len(args) >= 2:
            base = self._emit_expr(args[0])
            exp = self._emit_expr(args[1])
            if _is_bool(args[0]):
                base = f"({base} as i64)"
            elif isinstance(args[0], IntLit):
                base = f"{base}_i64"
            if _is_bool(args[1]):
                exp = f"{exp} as u32"
            elif args[1].typ == INT:
                exp = f"{exp} as u32"
            if len(args) == 3:
                mod = self._emit_expr(args[2])
                return f"{base}.pow({exp}) % {mod}"
            return f"{base}.pow({exp})"
        # divmod() builtin
        if func == "divmod" and len(args) == 2:
            a = self._emit_expr(args[0])
            b = self._emit_expr(args[1])
            if _is_bool(args[0]):
                a = f"({a} as i64)"
            if _is_bool(args[1]):
                b = f"({b} as i64)"
            return f"({a} / {b}, {a} % {b})"
        # list() builtin - convert iterable to Vec
        if func == "list" and len(args) == 1:
            inner = self._emit_expr(args[0])
            return f"{inner}.into_iter().collect::<Vec<_>>()"
        # dict() builtin - convert iterable of pairs to HashMap
        if func == "dict" and len(args) == 1:
            inner = self._emit_expr(args[0])
            return f"{inner}.into_iter().collect::<std::collections::HashMap<_, _>>()"
        # zip() builtin
        if func == "zip" and len(args) == 2:
            a = self._emit_expr(args[0])
            b = self._emit_expr(args[1])
            return f"{a}.iter().zip({b}.iter()).map(|(a, b)| (a.clone(), b.clone())).collect::<Vec<_>>()"
        # sum() builtin
        if func == "sum" and len(args) >= 1:
            inner = self._emit_expr(args[0])
            return f"{inner}.iter().sum::<i64>()"
        # sorted() builtin
        if func == "sorted" and len(args) >= 1:
            inner = self._emit_expr(args[0])
            return f"{{ let mut v: Vec<_> = {inner}.iter().cloned().collect(); v.sort(); v }}"
        # min() on iterable
        if func == "min" and len(args) == 1:
            inner = self._emit_expr(args[0])
            inner_type = args[0].typ
            if isinstance(inner_type, Map):
                # min(dict) returns min key
                return f"*{inner}.keys().min().unwrap()"
            if isinstance(inner_type, (Slice, Set)):
                return f"*{inner}.iter().min().unwrap()"
            return f"*{inner}.iter().min().unwrap()"
        # max() on iterable
        if func == "max" and len(args) == 1:
            inner = self._emit_expr(args[0])
            inner_type = args[0].typ
            if isinstance(inner_type, Map):
                # max(dict) returns max key
                return f"*{inner}.keys().max().unwrap()"
            if isinstance(inner_type, (Slice, Set)):
                return f"*{inner}.iter().max().unwrap()"
            return f"*{inner}.iter().max().unwrap()"
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
        # int to char: must go through u32 in Rust
        if isinstance(to_type, Primitive) and to_type.kind == "rune":
            if from_type == INT:
                return f"char::from_u32({inner} as u32).unwrap()"
            # byte to char
            return f"({inner} as char)"
        if from_type == to_type:
            return inner
        return f"({inner} as {self._type_to_rust(to_type)})"

    def _emit_TupleLit(self, expr: TupleLit) -> str:
        parts: list[str] = []
        for e in expr.elements:
            s = self._emit_expr(e)
            if isinstance(e, FuncRef) or (
                isinstance(e.typ, (FuncType, InterfaceRef)) and e.typ != VOID
            ):
                s += " as fn()"
            parts.append(s)
        return f"({', '.join(parts)})"

    def _emit_SliceLit(self, expr: SliceLit) -> str:
        if not expr.elements:
            # Empty vec needs type annotation
            elem_type = self._type_to_rust(expr.typ.element)
            return f"Vec::<{elem_type}>::new()"
        elements = ", ".join(self._emit_expr(e) for e in expr.elements)
        return f"vec![{elements}]"

    def _emit_MapLit(self, expr: MapLit) -> str:
        if not expr.entries:
            # For empty maps, if type is 'any', use placeholder types
            is_any_key = (
                isinstance(expr.key_type, InterfaceRef) and expr.key_type.name == "any"
            )
            is_any_val = (
                isinstance(expr.value_type, InterfaceRef)
                and expr.value_type.name == "any"
            )
            if is_any_key and is_any_val:
                # Use () as placeholder - works for comparison with other empty maps
                return "std::collections::HashMap::<(), ()>::new()"
            if is_any_key:
                val_type = self._type_to_rust(expr.value_type)
                return f"std::collections::HashMap::<(), {val_type}>::new()"
            if is_any_val:
                key_type = self._type_to_rust(expr.key_type)
                return f"std::collections::HashMap::<{key_type}, ()>::new()"
            key_type = self._type_to_rust(expr.key_type)
            val_type = self._type_to_rust(expr.value_type)
            return f"std::collections::HashMap::<{key_type}, {val_type}>::new()"
        pairs = ", ".join(
            f"({self._emit_expr(k)}, {self._emit_expr(v)})" for k, v in expr.entries
        )
        return f"std::collections::HashMap::from([{pairs}])"

    def _emit_MapLit_with_optional(self, expr: MapLit, opt_type: Optional) -> str:
        """Emit MapLit where values should be Option<T>, wrapping non-None values in Some()."""
        if not expr.entries:
            return "std::collections::HashMap::new()"
        pairs = []
        for k, v in expr.entries:
            key = self._emit_expr(k)
            if isinstance(v, NilLit):
                val = "None"
            else:
                val = f"Some({self._emit_expr(v)})"
            pairs.append(f"({key}, {val})")
        return f"std::collections::HashMap::from([{', '.join(pairs)}])"

    def _emit_Index(self, expr: Index) -> str:
        obj = self._emit_expr(expr.obj)
        idx = self._emit_expr(expr.index)
        # String indexing - get Unicode code point (for ord() semantics)
        if expr.obj.typ == STRING:
            return f"({obj}.chars().nth({idx} as usize).unwrap() as i64)"
        # Map/dict indexing
        if isinstance(expr.obj.typ, Map):
            return f"*{obj}.get(&{idx}).unwrap()"
        # Nested map indexing: when obj is Index on a Map, result is the value type
        # Check if obj is an Index on a Map (for nested dict access)
        if isinstance(expr.obj, Index) and isinstance(expr.obj.obj.typ, Map):
            inner_map = expr.obj.obj.typ
            if isinstance(inner_map.value, Map):
                # Nested dict: outer[key1][key2] - the inner Index already derefs to HashMap
                # so we just need .get().unwrap() without extra deref
                return f"{obj}.get(&{idx}).unwrap()"
        # Slice/vec indexing
        return f"{obj}[{idx} as usize]"

    def _emit_CharClassify(self, expr: CharClassify) -> str:
        char = self._emit_expr(expr.char)
        # Convert string to char: "a".chars().next().unwrap()
        char_expr = f"{char}.chars().next().unwrap()"
        kind_map = {
            "alpha": "is_alphabetic",
            "digit": "is_ascii_digit",
            "alnum": "is_alphanumeric",
            "upper": "is_uppercase",
            "lower": "is_lowercase",
            "space": "is_whitespace",
            "ascii": "is_ascii",
            "print": "is_ascii_graphic",
        }
        method = kind_map.get(expr.kind)
        if method:
            return f"{char_expr}.{method}()"
        raise NotImplementedError(f"CharClassify kind: {expr.kind}")

    def _emit_MethodCall(self, expr: MethodCall) -> str:
        # Handle dict.fromkeys() class method
        if (
            isinstance(expr.obj, Var)
            and expr.obj.name == "dict"
            and expr.method == "fromkeys"
        ):
            keys = self._emit_expr(expr.args[0])
            if len(expr.args) >= 2:
                value = self._emit_expr(expr.args[1])
                return f"{keys}.iter().map(|k| (k.clone(), {value}.clone())).collect::<std::collections::HashMap<_, _>>()"
            else:
                return f"{keys}.iter().map(|k| (k.clone(), ())).collect::<std::collections::HashMap<_, _>>()"
        # Handle mutating methods on dict values (d[key].append(x) etc.)
        # Need to use get_mut instead of get for mutable access
        if isinstance(expr.obj, Index) and isinstance(expr.obj.obj.typ, Map):
            if isinstance(expr.receiver_type, Slice) and expr.method == "append":
                dict_var = self._emit_expr(expr.obj.obj)
                key = self._emit_expr(expr.obj.index)
                val = self._emit_expr(expr.args[0])
                return f"{dict_var}.get_mut(&{key}).unwrap().push({val})"
        obj = self._emit_expr(expr.obj)
        args = ", ".join(self._emit_expr(a) for a in expr.args)
        # String methods
        if expr.receiver_type == STRING:
            method = expr.method
            if method == "upper":
                return f"{obj}.to_uppercase()"
            if method == "lower":
                return f"{obj}.to_lowercase()"
            if method == "strip":
                return f"{obj}.trim()"
            if method == "lstrip":
                return f"{obj}.trim_start()"
            if method == "rstrip":
                return f"{obj}.trim_end()"
            if method == "startswith":
                return f"{obj}.starts_with({args})"
            if method == "endswith":
                return f"{obj}.ends_with({args})"
            if method == "find":
                return f"{obj}.find({args}).map_or(-1, |i| i as i64)"
            if method == "replace":
                return f"{obj}.replace({args})"
            if method == "split":
                return f"{obj}.split({args}).collect::<Vec<_>>()"
            if method == "join":
                return f"{args}.join({obj})"
        # Map/dict methods
        if isinstance(expr.receiver_type, Map):
            method = expr.method
            if method == "get":
                key = self._emit_expr(expr.args[0])
                map_type = expr.receiver_type
                if len(expr.args) == 1:
                    # .get(key) -> .get(&key).copied()
                    return f"{obj}.get(&{key}).copied()"
                else:
                    # .get(key, default) -> .get(&key).copied().unwrap_or(default)
                    default = self._emit_expr(expr.args[1])
                    # If map value is Optional, wrap default in Some()
                    if isinstance(map_type.value, Optional):
                        return f"{obj}.get(&{key}).copied().unwrap_or(Some({default}))"
                    return f"{obj}.get(&{key}).copied().unwrap_or({default})"
            if method == "keys":
                return f"{obj}.keys().cloned().collect::<Vec<_>>()"
            if method == "values":
                return f"{obj}.values().cloned().collect::<Vec<_>>()"
            if method == "items":
                return f"{obj}.iter().map(|(k, v)| (k.clone(), *v)).collect::<Vec<_>>()"
            if method == "pop":
                key = self._emit_expr(expr.args[0])
                if len(expr.args) == 1:
                    return f"{obj}.remove(&{key}).unwrap()"
                else:
                    default = self._emit_expr(expr.args[1])
                    return f"{obj}.remove(&{key}).unwrap_or({default})"
            if method == "setdefault":
                key = self._emit_expr(expr.args[0])
                if len(expr.args) == 1:
                    return f"*{obj}.entry({key}).or_insert(Default::default())"
                else:
                    default = self._emit_expr(expr.args[1])
                    return f"*{obj}.entry({key}).or_insert({default})"
            if method == "update":
                # Check if updating with empty dict - no-op
                if isinstance(expr.args[0], MapLit) and not expr.args[0].entries:
                    return f"()"  # No-op in Rust
                other = self._emit_expr(expr.args[0])
                return f"{obj}.extend({other})"
            if method == "clear":
                return f"{obj}.clear()"
            if method == "copy":
                return f"{obj}.clone()"
            if method == "popitem":
                # Remove and return last item - Rust HashMap doesn't preserve order
                # Use IndexMap or handle specially
                return f"{{ let k = {obj}.keys().last().unwrap().clone(); let v = {obj}.remove(&k).unwrap(); (k, v) }}"
        # Slice/vec methods
        if isinstance(expr.receiver_type, Slice):
            method = expr.method
            if method == "append":
                val = self._emit_expr(expr.args[0])
                return f"{obj}.push({val})"
        # Default: direct method call
        return f"{obj}.{expr.method}({args})"

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

    def _emit_Truthy(self, expr: Truthy) -> str:
        inner = self._emit_expr(expr.expr)
        inner_type = expr.expr.typ
        # Map/dict: non-empty
        if isinstance(inner_type, Map):
            return f"!{inner}.is_empty()"
        # Slice/vec: non-empty
        if isinstance(inner_type, Slice):
            return f"!{inner}.is_empty()"
        # String: non-empty
        if inner_type == STRING:
            return f"!{inner}.is_empty()"
        # Int: non-zero
        if inner_type == INT:
            return f"({inner} != 0)"
        # Bool: the value itself
        if inner_type == BOOL:
            return inner
        # Default: try is_empty or != 0
        return f"!{inner}.is_empty()"

    def _emit_IsNil(self, expr: IsNil) -> str:
        inner = self._emit_expr(expr.expr)
        # For map.get() which returns Option, check is_none()/is_some()
        if isinstance(expr.expr, MethodCall) and expr.expr.method == "get":
            # The MethodCall emitted .get(&key).copied() which returns Option<V>
            # We need to wrap it to check is_none/is_some
            obj = self._emit_expr(expr.expr.obj)
            key = self._emit_expr(expr.expr.args[0])
            if expr.negated:
                return f"{obj}.get(&{key}).is_some()"
            return f"{obj}.get(&{key}).is_none()"
        # General case
        if expr.negated:
            return f"{inner}.is_some()"
        return f"{inner}.is_none()"

    def _emit_DictComp(self, expr: DictComp) -> str:
        # {key: value for x in iterable if cond}
        # -> iterable.into_iter().filter(|x| cond).map(|x| (key, value)).collect::<HashMap<_, _>>()
        gen = expr.generators[0]
        target_names = [self._safe(t) for t in gen.targets]
        iterable = self._emit_expr(gen.iterable)
        key = self._emit_expr(expr.key)
        value = self._emit_expr(expr.value)
        # When iterating over tuples with multiple targets, wrap in parens: |(k, v)|
        if len(target_names) > 1:
            targets = f"({', '.join(target_names)})"
        else:
            targets = target_names[0]
        # Use into_iter() to get owned values, avoiding reference issues in closures
        result = f"{iterable}.into_iter()"
        if gen.conditions:
            conds = " && ".join(self._emit_expr(c) for c in gen.conditions)
            result += f".filter(|{targets}| {conds})"
        result += f".map(|{targets}| ({key}, {value})).collect::<std::collections::HashMap<_, _>>()"
        return result

    def _emit_ListComp(self, expr: ListComp) -> str:
        # [element for x in iterable if cond]
        # -> iterable.into_iter().filter(|x| cond).map(|x| element).collect::<Vec<_>>()
        gen = expr.generators[0]
        target_names = [self._safe(t) for t in gen.targets]
        iterable = self._emit_expr(gen.iterable)
        element = self._emit_expr(expr.element)
        # When iterating over tuples with multiple targets, wrap in parens: |(k, v)|
        if len(target_names) > 1:
            targets = f"({', '.join(target_names)})"
        else:
            targets = target_names[0]
        # Use into_iter() to get owned values, avoiding reference issues in closures
        result = f"{iterable}.into_iter()"
        if gen.conditions:
            conds = " && ".join(self._emit_expr(c) for c in gen.conditions)
            result += f".filter(|{targets}| {conds})"
        result += f".map(|{targets}| {element}).collect::<Vec<_>>()"
        return result

    def _emit_SetLit(self, expr: SetLit) -> str:
        if not expr.elements:
            elem_type = self._type_to_rust(expr.element_type)
            return f"std::collections::HashSet::<{elem_type}>::new()"
        elements = ", ".join(self._emit_expr(e) for e in expr.elements)
        return f"std::collections::HashSet::from([{elements}])"

    def _emit_StaticCall(self, expr: StaticCall) -> str:
        method = expr.method
        args = expr.args
        # dict.fromkeys(keys, value)
        if isinstance(expr.on_type, Map) and method == "fromkeys":
            keys = self._emit_expr(args[0])
            if len(args) >= 2:
                value = self._emit_expr(args[1])
                return f"{keys}.iter().map(|k| (k.clone(), {value})).collect::<std::collections::HashMap<_, _>>()"
            else:
                return f"{keys}.iter().map(|k| (k.clone(), None)).collect::<std::collections::HashMap<_, _>>()"
        # Default: Type::method(args)
        type_name = self._type_to_rust(expr.on_type)
        args_str = ", ".join(self._emit_expr(a) for a in args)
        return f"{type_name}::{method}({args_str})"
