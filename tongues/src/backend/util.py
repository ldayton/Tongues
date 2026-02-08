"""Shared utilities for backend code emitters."""

from __future__ import annotations

import re

from src.ir import (
    Assign,
    Assert,
    BinaryOp,
    Block,
    Call,
    Cast,
    CharAt,
    CharClassify,
    CharLen,
    CompGenerator,
    DictComp,
    EntryPoint,
    Expr,
    ExprStmt,
    FieldAccess,
    ForClassic,
    ForRange,
    If,
    Index,
    IntToStr,
    IsNil,
    IsType,
    LastElement,
    Len,
    ListComp,
    MakeMap,
    MakeSlice,
    Map,
    MapLit,
    Match,
    MethodCall,
    Module,
    OpAssign,
    ParseInt,
    Primitive,
    Print,
    Raise,
    Return,
    SentinelToOptional,
    Set,
    SetComp,
    SetLit,
    Slice,
    SliceConvert,
    SliceExpr,
    SliceLit,
    SoftFail,
    StaticCall,
    Stmt,
    StringConcat,
    StringFormat,
    StructLit,
    Substring,
    Ternary,
    TrimChars,
    TryCatch,
    Tuple,
    TupleAssign,
    TupleLit,
    Type,
    TypeAssert,
    TypeSwitch,
    UnaryOp,
    VarDecl,
    While,
)

# Go reserved words that need renaming
GO_RESERVED = frozenset(
    {
        "break",
        "case",
        "chan",
        "const",
        "continue",
        "default",
        "defer",
        "else",
        "fallthrough",
        "for",
        "func",
        "go",
        "goto",
        "if",
        "import",
        "interface",
        "map",
        "package",
        "range",
        "return",
        "select",
        "struct",
        "switch",
        "type",
        "var",
    }
)


def _upper_first(s: str) -> str:
    """Uppercase the first character of a string."""
    return (s[0].upper() + s[1:]) if s else ""


def go_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase for Go. Private methods (underscore prefix) become unexported."""
    is_private = name.startswith("_")
    if is_private:
        name = name[1:]
    parts = name.split("_")
    # Use upper on first char only (not capitalize which lowercases rest)
    result = "".join(_upper_first(p) for p in parts)
    # All-caps names (constants) stay all-caps even if originally private
    if name.isupper():
        return result
    if is_private:
        # Make first letter lowercase for unexported (private) names
        return result[0].lower() + result[1:] if result else result
    return result


def go_to_camel(name: str) -> str:
    """Convert snake_case to camelCase for Go."""
    if name == "self":
        return name
    if name.startswith("_"):
        name = name[1:]
    parts = name.split("_")
    if not parts:
        return name
    # All-caps names (constants) should use PascalCase in Go
    if name.isupper():
        return "".join(_upper_first(p) for p in parts)
    result = parts[0] + "".join(_upper_first(p) for p in parts[1:])
    # Handle Go reserved words
    if result in GO_RESERVED:
        return result + "_"
    return result


def to_snake(name: str) -> str:
    """Convert camelCase/PascalCase to snake_case."""
    if name.startswith("_"):
        name = name[1:]
    if "_" in name or name.islower():
        return name.lower()
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def to_camel(name: str) -> str:
    """Convert snake_case to camelCase, preserving leading underscores."""
    prefix = ""
    if name.startswith("_"):
        prefix = "_"
        name = name[1:]
    if "_" not in name:
        return prefix + (name[0].lower() + name[1:] if name else name)
    parts = name.split("_")
    return prefix + parts[0].lower() + "".join(p.capitalize() for p in parts[1:])


def to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase."""
    if name.startswith("_"):
        name = name[1:]
    parts = name.split("_")
    return "".join(p.capitalize() for p in parts)


def to_screaming_snake(name: str) -> str:
    """Convert to SCREAMING_SNAKE_CASE."""
    return to_snake(name).upper()


def escape_string(value: str) -> str:
    """Escape a string for use in a string literal (without quotes)."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("\f", "\\f")
        .replace("\v", "\\v")
        .replace("\x00", "\\x00")
        .replace("\x01", "\\u0001")
        .replace("\x7f", "\\u007f")
    )


class Emitter:
    """Base class for code emitters with indentation tracking."""

    def __init__(self, indent_str: str = "    ") -> None:
        self.indent: int = 0
        self.lines: list[str] = []
        self._indent_str = indent_str

    def line(self, text: str = "") -> None:
        """Emit a line with current indentation."""
        if text:
            self.lines.append(self._indent_str * self.indent + text)
        else:
            self.lines.append("")

    def output(self) -> str:
        """Return the accumulated output as a string."""
        return "\n".join(self.lines)


def _visit_expr_for_call(expr: Expr | None, func: str) -> bool:
    """Check if an expression or its children contain a Call to func."""
    if expr is None:
        return False
    if isinstance(expr, Call) and expr.func == func:
        return True
    # Recurse into child expressions based on type
    if isinstance(expr, FieldAccess):
        return _visit_expr_for_call(expr.obj, func)
    if isinstance(expr, Index):
        return _visit_expr_for_call(expr.obj, func) or _visit_expr_for_call(
            expr.index, func
        )
    if isinstance(expr, SliceExpr):
        return (
            _visit_expr_for_call(expr.obj, func)
            or _visit_expr_for_call(expr.low, func)
            or _visit_expr_for_call(expr.high, func)
            or _visit_expr_for_call(expr.step, func)
        )
    if isinstance(expr, Call):
        for arg in expr.args:
            if _visit_expr_for_call(arg, func):
                return True
        return False
    if isinstance(expr, MethodCall):
        if _visit_expr_for_call(expr.obj, func):
            return True
        for arg in expr.args:
            if _visit_expr_for_call(arg, func):
                return True
        return False
    if isinstance(expr, StaticCall):
        for arg in expr.args:
            if _visit_expr_for_call(arg, func):
                return True
        return False
    if isinstance(expr, BinaryOp):
        return _visit_expr_for_call(expr.left, func) or _visit_expr_for_call(
            expr.right, func
        )
    if isinstance(expr, UnaryOp):
        return _visit_expr_for_call(expr.operand, func)
    if isinstance(expr, Ternary):
        return (
            _visit_expr_for_call(expr.cond, func)
            or _visit_expr_for_call(expr.then_expr, func)
            or _visit_expr_for_call(expr.else_expr, func)
        )
    if isinstance(expr, Cast):
        return _visit_expr_for_call(expr.expr, func)
    if isinstance(expr, TypeAssert):
        return _visit_expr_for_call(expr.expr, func)
    if isinstance(expr, IsType):
        return _visit_expr_for_call(expr.expr, func)
    if isinstance(expr, IsNil):
        return _visit_expr_for_call(expr.expr, func)
    if isinstance(expr, Len):
        return _visit_expr_for_call(expr.expr, func)
    if isinstance(expr, MakeSlice):
        return _visit_expr_for_call(expr.length, func) or _visit_expr_for_call(
            expr.capacity, func
        )
    if isinstance(expr, MakeMap):
        return _visit_expr_for_call(expr.capacity, func)
    if isinstance(expr, SliceLit):
        for elem in expr.elements:
            if _visit_expr_for_call(elem, func):
                return True
        return False
    if isinstance(expr, MapLit):
        for k, v in expr.entries:
            if _visit_expr_for_call(k, func) or _visit_expr_for_call(v, func):
                return True
        return False
    if isinstance(expr, SetLit):
        for elem in expr.elements:
            if _visit_expr_for_call(elem, func):
                return True
        return False
    if isinstance(expr, TupleLit):
        for elem in expr.elements:
            if _visit_expr_for_call(elem, func):
                return True
        return False
    if isinstance(expr, StructLit):
        for v in expr.fields.values():
            if _visit_expr_for_call(v, func):
                return True
        return False
    if isinstance(expr, StringConcat):
        for part in expr.parts:
            if _visit_expr_for_call(part, func):
                return True
        return False
    if isinstance(expr, StringFormat):
        for arg in expr.args:
            if _visit_expr_for_call(arg, func):
                return True
        return False
    if isinstance(expr, Substring):
        return (
            _visit_expr_for_call(expr.string, func)
            or _visit_expr_for_call(expr.start, func)
            or _visit_expr_for_call(expr.end, func)
        )
    if isinstance(expr, CharAt):
        return _visit_expr_for_call(expr.string, func) or _visit_expr_for_call(
            expr.index, func
        )
    if isinstance(expr, CharLen):
        return _visit_expr_for_call(expr.string, func)
    if isinstance(expr, TrimChars):
        return _visit_expr_for_call(expr.string, func) or _visit_expr_for_call(
            expr.chars, func
        )
    if isinstance(expr, CharClassify):
        return _visit_expr_for_call(expr.char, func)
    if isinstance(expr, ParseInt):
        return _visit_expr_for_call(expr.string, func)
    if isinstance(expr, IntToStr):
        return _visit_expr_for_call(expr.value, func)
    if isinstance(expr, SliceConvert):
        return _visit_expr_for_call(expr.expr, func)
    if isinstance(expr, LastElement):
        return _visit_expr_for_call(expr.slice, func)
    if isinstance(expr, SentinelToOptional):
        return _visit_expr_for_call(expr.value, func)
    if isinstance(expr, ListComp):
        if _visit_expr_for_call(expr.element, func):
            return True
        for gen in expr.generators:
            if _visit_generator_for_call(gen, func):
                return True
        return False
    if isinstance(expr, SetComp):
        if _visit_expr_for_call(expr.element, func):
            return True
        for gen in expr.generators:
            if _visit_generator_for_call(gen, func):
                return True
        return False
    if isinstance(expr, DictComp):
        if _visit_expr_for_call(expr.key, func) or _visit_expr_for_call(
            expr.value, func
        ):
            return True
        for gen in expr.generators:
            if _visit_generator_for_call(gen, func):
                return True
        return False
    return False


def _visit_generator_for_call(gen: CompGenerator, func: str) -> bool:
    """Check if a comprehension generator contains a Call to func."""
    if _visit_expr_for_call(gen.iterable, func):
        return True
    for cond in gen.conditions:
        if _visit_expr_for_call(cond, func):
            return True
    return False


def _visit_stmt_for_call(stmt: Stmt, func: str) -> bool:
    """Check if a statement or its children contain a Call to func."""
    if isinstance(stmt, VarDecl):
        return _visit_expr_for_call(stmt.value, func)
    if isinstance(stmt, Assign):
        return _visit_expr_for_call(stmt.value, func)
    if isinstance(stmt, TupleAssign):
        return _visit_expr_for_call(stmt.value, func)
    if isinstance(stmt, OpAssign):
        return _visit_expr_for_call(stmt.value, func)
    if isinstance(stmt, ExprStmt):
        return _visit_expr_for_call(stmt.expr, func)
    if isinstance(stmt, Return):
        return _visit_expr_for_call(stmt.value, func)
    if isinstance(stmt, If):
        if _visit_expr_for_call(stmt.cond, func):
            return True
        for s in stmt.then_body:
            if _visit_stmt_for_call(s, func):
                return True
        for s in stmt.else_body:
            if _visit_stmt_for_call(s, func):
                return True
        return False
    if isinstance(stmt, TypeSwitch):
        if _visit_expr_for_call(stmt.expr, func):
            return True
        for case in stmt.cases:
            for s in case.body:
                if _visit_stmt_for_call(s, func):
                    return True
        for s in stmt.default_body:
            if _visit_stmt_for_call(s, func):
                return True
        return False
    if isinstance(stmt, Match):
        if _visit_expr_for_call(stmt.subject, func):
            return True
        for case in stmt.cases:
            if _visit_expr_for_call(case.guard, func):
                return True
            for s in case.body:
                if _visit_stmt_for_call(s, func):
                    return True
        return False
    if isinstance(stmt, ForRange):
        if _visit_expr_for_call(stmt.iterable, func):
            return True
        for s in stmt.body:
            if _visit_stmt_for_call(s, func):
                return True
        return False
    if isinstance(stmt, ForClassic):
        if stmt.init is not None and _visit_stmt_for_call(stmt.init, func):
            return True
        if _visit_expr_for_call(stmt.cond, func):
            return True
        if stmt.post is not None and _visit_stmt_for_call(stmt.post, func):
            return True
        for s in stmt.body:
            if _visit_stmt_for_call(s, func):
                return True
        return False
    if isinstance(stmt, While):
        if _visit_expr_for_call(stmt.cond, func):
            return True
        for s in stmt.body:
            if _visit_stmt_for_call(s, func):
                return True
        return False
    if isinstance(stmt, Block):
        for s in stmt.body:
            if _visit_stmt_for_call(s, func):
                return True
        return False
    if isinstance(stmt, TryCatch):
        for s in stmt.body:
            if _visit_stmt_for_call(s, func):
                return True
        for clause in stmt.catches:
            for s in clause.body:
                if _visit_stmt_for_call(s, func):
                    return True
        return False
    if isinstance(stmt, Raise):
        return _visit_expr_for_call(stmt.exception, func)
    if isinstance(stmt, Assert):
        return _visit_expr_for_call(stmt.test, func) or _visit_expr_for_call(
            stmt.message, func
        )
    if isinstance(stmt, SoftFail):
        return _visit_expr_for_call(stmt.msg, func)
    if isinstance(stmt, Print):
        for arg in stmt.args:
            if _visit_expr_for_call(arg, func):
                return True
        return False
    if isinstance(stmt, EntryPoint):
        for s in stmt.body:
            if _visit_stmt_for_call(s, func):
                return True
        return False
    return False


def ir_contains_call(module: Module, func: str) -> bool:
    """Check if the module's IR contains a Call to the given function name."""
    # Check functions
    for fn in module.functions:
        for stmt in fn.body:
            if _visit_stmt_for_call(stmt, func):
                return True
    # Check struct methods
    for struct in module.structs:
        for method in struct.methods:
            for stmt in method.body:
                if _visit_stmt_for_call(stmt, func):
                    return True
    return False


def is_bytes_type(typ: Type | None) -> bool:
    """Check if type is a byte array (Slice with byte element)."""
    if typ is None:
        return False
    if isinstance(typ, Slice):
        elem = typ.element
        return isinstance(elem, Primitive) and elem.kind == "byte"
    return False


def _visit_expr_for_bytes_ops(expr: Expr | None) -> bool:
    """Check if an expression contains operations on byte arrays."""
    if expr is None:
        return False
    # Check for byte array operations in BinaryOp
    if isinstance(expr, BinaryOp):
        if expr.op in ("==", "!=", "<", "<=", ">", ">=", "+", "*", "in", "not in"):
            if is_bytes_type(expr.left.typ) or is_bytes_type(expr.right.typ):
                return True
        return _visit_expr_for_bytes_ops(expr.left) or _visit_expr_for_bytes_ops(
            expr.right
        )
    # Check for TrimChars with bytes
    if isinstance(expr, TrimChars) and is_bytes_type(expr.string.typ):
        return True
    # Check for step slices (need arrStep helper)
    if isinstance(expr, SliceExpr) and expr.step is not None:
        return True
    # Check for MethodCall on bytes
    if isinstance(expr, MethodCall):
        if is_bytes_type(expr.receiver_type):
            if expr.method in (
                "count",
                "find",
                "rfind",
                "startswith",
                "endswith",
                "upper",
                "lower",
                "strip",
                "lstrip",
                "rstrip",
                "split",
                "join",
                "replace",
            ):
                return True
        if _visit_expr_for_bytes_ops(expr.obj):
            return True
        for arg in expr.args:
            if _visit_expr_for_bytes_ops(arg):
                return True
        return False
    # Recurse into child expressions
    if isinstance(expr, FieldAccess):
        return _visit_expr_for_bytes_ops(expr.obj)
    if isinstance(expr, Index):
        return _visit_expr_for_bytes_ops(expr.obj) or _visit_expr_for_bytes_ops(
            expr.index
        )
    if isinstance(expr, SliceExpr):
        return (
            _visit_expr_for_bytes_ops(expr.obj)
            or _visit_expr_for_bytes_ops(expr.low)
            or _visit_expr_for_bytes_ops(expr.high)
            or _visit_expr_for_bytes_ops(expr.step)
        )
    if isinstance(expr, Call):
        for arg in expr.args:
            if _visit_expr_for_bytes_ops(arg):
                return True
        return False
    if isinstance(expr, StaticCall):
        for arg in expr.args:
            if _visit_expr_for_bytes_ops(arg):
                return True
        return False
    if isinstance(expr, UnaryOp):
        return _visit_expr_for_bytes_ops(expr.operand)
    if isinstance(expr, Ternary):
        return (
            _visit_expr_for_bytes_ops(expr.cond)
            or _visit_expr_for_bytes_ops(expr.then_expr)
            or _visit_expr_for_bytes_ops(expr.else_expr)
        )
    if isinstance(expr, Cast):
        return _visit_expr_for_bytes_ops(expr.expr)
    if isinstance(expr, TypeAssert):
        return _visit_expr_for_bytes_ops(expr.expr)
    if isinstance(expr, IsType):
        return _visit_expr_for_bytes_ops(expr.expr)
    if isinstance(expr, IsNil):
        return _visit_expr_for_bytes_ops(expr.expr)
    if isinstance(expr, Len):
        return _visit_expr_for_bytes_ops(expr.expr)
    if isinstance(expr, MakeSlice):
        return _visit_expr_for_bytes_ops(expr.length) or _visit_expr_for_bytes_ops(
            expr.capacity
        )
    if isinstance(expr, MakeMap):
        return _visit_expr_for_bytes_ops(expr.capacity)
    if isinstance(expr, SliceLit):
        for elem in expr.elements:
            if _visit_expr_for_bytes_ops(elem):
                return True
        return False
    if isinstance(expr, MapLit):
        for k, v in expr.entries:
            if _visit_expr_for_bytes_ops(k) or _visit_expr_for_bytes_ops(v):
                return True
        return False
    if isinstance(expr, SetLit):
        for elem in expr.elements:
            if _visit_expr_for_bytes_ops(elem):
                return True
        return False
    if isinstance(expr, TupleLit):
        for elem in expr.elements:
            if _visit_expr_for_bytes_ops(elem):
                return True
        return False
    if isinstance(expr, StructLit):
        for v in expr.fields.values():
            if _visit_expr_for_bytes_ops(v):
                return True
        return False
    if isinstance(expr, StringConcat):
        for part in expr.parts:
            if _visit_expr_for_bytes_ops(part):
                return True
        return False
    if isinstance(expr, StringFormat):
        for arg in expr.args:
            if _visit_expr_for_bytes_ops(arg):
                return True
        return False
    if isinstance(expr, TrimChars):
        return _visit_expr_for_bytes_ops(expr.string) or _visit_expr_for_bytes_ops(
            expr.chars
        )
    if isinstance(expr, Substring):
        return (
            _visit_expr_for_bytes_ops(expr.string)
            or _visit_expr_for_bytes_ops(expr.start)
            or _visit_expr_for_bytes_ops(expr.end)
        )
    if isinstance(expr, CharAt):
        return _visit_expr_for_bytes_ops(expr.string) or _visit_expr_for_bytes_ops(
            expr.index
        )
    if isinstance(expr, CharLen):
        return _visit_expr_for_bytes_ops(expr.string)
    if isinstance(expr, CharClassify):
        return _visit_expr_for_bytes_ops(expr.char)
    if isinstance(expr, ParseInt):
        return _visit_expr_for_bytes_ops(expr.string)
    if isinstance(expr, IntToStr):
        return _visit_expr_for_bytes_ops(expr.value)
    if isinstance(expr, SliceConvert):
        return _visit_expr_for_bytes_ops(expr.expr)
    if isinstance(expr, LastElement):
        return _visit_expr_for_bytes_ops(expr.slice)
    if isinstance(expr, SentinelToOptional):
        return _visit_expr_for_bytes_ops(expr.value)
    if isinstance(expr, ListComp):
        if _visit_expr_for_bytes_ops(expr.element):
            return True
        for gen in expr.generators:
            if _visit_expr_for_bytes_ops(gen.iterable):
                return True
            for cond in gen.conditions:
                if _visit_expr_for_bytes_ops(cond):
                    return True
        return False
    if isinstance(expr, SetComp):
        if _visit_expr_for_bytes_ops(expr.element):
            return True
        for gen in expr.generators:
            if _visit_expr_for_bytes_ops(gen.iterable):
                return True
            for cond in gen.conditions:
                if _visit_expr_for_bytes_ops(cond):
                    return True
        return False
    if isinstance(expr, DictComp):
        if _visit_expr_for_bytes_ops(expr.key) or _visit_expr_for_bytes_ops(expr.value):
            return True
        for gen in expr.generators:
            if _visit_expr_for_bytes_ops(gen.iterable):
                return True
            for cond in gen.conditions:
                if _visit_expr_for_bytes_ops(cond):
                    return True
        return False
    return False


def _visit_stmt_for_bytes_ops(stmt: Stmt) -> bool:
    """Check if a statement contains operations on byte arrays."""
    if isinstance(stmt, VarDecl):
        return _visit_expr_for_bytes_ops(stmt.value)
    if isinstance(stmt, Assign):
        return _visit_expr_for_bytes_ops(stmt.value)
    if isinstance(stmt, TupleAssign):
        return _visit_expr_for_bytes_ops(stmt.value)
    if isinstance(stmt, OpAssign):
        return _visit_expr_for_bytes_ops(stmt.value)
    if isinstance(stmt, ExprStmt):
        return _visit_expr_for_bytes_ops(stmt.expr)
    if isinstance(stmt, Return):
        return _visit_expr_for_bytes_ops(stmt.value)
    if isinstance(stmt, If):
        if _visit_expr_for_bytes_ops(stmt.cond):
            return True
        for s in stmt.then_body:
            if _visit_stmt_for_bytes_ops(s):
                return True
        for s in stmt.else_body:
            if _visit_stmt_for_bytes_ops(s):
                return True
        return False
    if isinstance(stmt, TypeSwitch):
        if _visit_expr_for_bytes_ops(stmt.expr):
            return True
        for case in stmt.cases:
            for s in case.body:
                if _visit_stmt_for_bytes_ops(s):
                    return True
        for s in stmt.default_body:
            if _visit_stmt_for_bytes_ops(s):
                return True
        return False
    if isinstance(stmt, Match):
        if _visit_expr_for_bytes_ops(stmt.subject):
            return True
        for case in stmt.cases:
            if _visit_expr_for_bytes_ops(case.guard):
                return True
            for s in case.body:
                if _visit_stmt_for_bytes_ops(s):
                    return True
        return False
    if isinstance(stmt, ForRange):
        if _visit_expr_for_bytes_ops(stmt.iterable):
            return True
        for s in stmt.body:
            if _visit_stmt_for_bytes_ops(s):
                return True
        return False
    if isinstance(stmt, ForClassic):
        if stmt.init is not None and _visit_stmt_for_bytes_ops(stmt.init):
            return True
        if _visit_expr_for_bytes_ops(stmt.cond):
            return True
        if stmt.post is not None and _visit_stmt_for_bytes_ops(stmt.post):
            return True
        for s in stmt.body:
            if _visit_stmt_for_bytes_ops(s):
                return True
        return False
    if isinstance(stmt, While):
        if _visit_expr_for_bytes_ops(stmt.cond):
            return True
        for s in stmt.body:
            if _visit_stmt_for_bytes_ops(s):
                return True
        return False
    if isinstance(stmt, Block):
        for s in stmt.body:
            if _visit_stmt_for_bytes_ops(s):
                return True
        return False
    if isinstance(stmt, TryCatch):
        for s in stmt.body:
            if _visit_stmt_for_bytes_ops(s):
                return True
        for clause in stmt.catches:
            for s in clause.body:
                if _visit_stmt_for_bytes_ops(s):
                    return True
        return False
    if isinstance(stmt, Raise):
        return _visit_expr_for_bytes_ops(stmt.exception)
    if isinstance(stmt, Assert):
        return _visit_expr_for_bytes_ops(stmt.test) or _visit_expr_for_bytes_ops(
            stmt.message
        )
    if isinstance(stmt, SoftFail):
        return _visit_expr_for_bytes_ops(stmt.msg)
    if isinstance(stmt, Print):
        for arg in stmt.args:
            if _visit_expr_for_bytes_ops(arg):
                return True
        return False
    if isinstance(stmt, EntryPoint):
        for s in stmt.body:
            if _visit_stmt_for_bytes_ops(s):
                return True
        return False
    return False


def ir_has_bytes_ops(module: Module) -> bool:
    """Check if the module's IR contains operations on byte arrays."""
    # Check functions
    for fn in module.functions:
        for stmt in fn.body:
            if _visit_stmt_for_bytes_ops(stmt):
                return True
    # Check struct methods
    for struct in module.structs:
        for method in struct.methods:
            for stmt in method.body:
                if _visit_stmt_for_bytes_ops(stmt):
                    return True
    return False


def _visit_expr_for_cast(expr: Expr | None, from_kind: str, to_kind: str) -> bool:
    """Check if an expression contains a Cast between the given primitive kinds."""
    if expr is None:
        return False
    if isinstance(expr, Cast):
        inner_type = expr.expr.typ
        if (
            isinstance(expr.to_type, Primitive)
            and expr.to_type.kind == to_kind
            and isinstance(inner_type, Primitive)
            and inner_type.kind == from_kind
        ):
            return True
        if (
            isinstance(expr.to_type, Slice)
            and isinstance(expr.to_type.element, Primitive)
            and expr.to_type.element.kind == to_kind
            and isinstance(inner_type, Primitive)
            and inner_type.kind == from_kind
        ):
            return True
        return _visit_expr_for_cast(expr.expr, from_kind, to_kind)
    # Recurse into child expressions
    if isinstance(expr, FieldAccess):
        return _visit_expr_for_cast(expr.obj, from_kind, to_kind)
    if isinstance(expr, Index):
        return _visit_expr_for_cast(
            expr.obj, from_kind, to_kind
        ) or _visit_expr_for_cast(expr.index, from_kind, to_kind)
    if isinstance(expr, SliceExpr):
        return (
            _visit_expr_for_cast(expr.obj, from_kind, to_kind)
            or _visit_expr_for_cast(expr.low, from_kind, to_kind)
            or _visit_expr_for_cast(expr.high, from_kind, to_kind)
            or _visit_expr_for_cast(expr.step, from_kind, to_kind)
        )
    if isinstance(expr, Call):
        for arg in expr.args:
            if _visit_expr_for_cast(arg, from_kind, to_kind):
                return True
        return False
    if isinstance(expr, MethodCall):
        if _visit_expr_for_cast(expr.obj, from_kind, to_kind):
            return True
        for arg in expr.args:
            if _visit_expr_for_cast(arg, from_kind, to_kind):
                return True
        return False
    if isinstance(expr, StaticCall):
        for arg in expr.args:
            if _visit_expr_for_cast(arg, from_kind, to_kind):
                return True
        return False
    if isinstance(expr, BinaryOp):
        return _visit_expr_for_cast(
            expr.left, from_kind, to_kind
        ) or _visit_expr_for_cast(expr.right, from_kind, to_kind)
    if isinstance(expr, UnaryOp):
        return _visit_expr_for_cast(expr.operand, from_kind, to_kind)
    if isinstance(expr, Ternary):
        return (
            _visit_expr_for_cast(expr.cond, from_kind, to_kind)
            or _visit_expr_for_cast(expr.then_expr, from_kind, to_kind)
            or _visit_expr_for_cast(expr.else_expr, from_kind, to_kind)
        )
    if isinstance(expr, TypeAssert):
        return _visit_expr_for_cast(expr.expr, from_kind, to_kind)
    if isinstance(expr, IsType):
        return _visit_expr_for_cast(expr.expr, from_kind, to_kind)
    if isinstance(expr, IsNil):
        return _visit_expr_for_cast(expr.expr, from_kind, to_kind)
    if isinstance(expr, Len):
        return _visit_expr_for_cast(expr.expr, from_kind, to_kind)
    if isinstance(expr, MakeSlice):
        return _visit_expr_for_cast(
            expr.length, from_kind, to_kind
        ) or _visit_expr_for_cast(expr.capacity, from_kind, to_kind)
    if isinstance(expr, MakeMap):
        return _visit_expr_for_cast(expr.capacity, from_kind, to_kind)
    if isinstance(expr, SliceLit):
        for elem in expr.elements:
            if _visit_expr_for_cast(elem, from_kind, to_kind):
                return True
        return False
    if isinstance(expr, MapLit):
        for k, v in expr.entries:
            if _visit_expr_for_cast(k, from_kind, to_kind) or _visit_expr_for_cast(
                v, from_kind, to_kind
            ):
                return True
        return False
    if isinstance(expr, SetLit):
        for elem in expr.elements:
            if _visit_expr_for_cast(elem, from_kind, to_kind):
                return True
        return False
    if isinstance(expr, TupleLit):
        for elem in expr.elements:
            if _visit_expr_for_cast(elem, from_kind, to_kind):
                return True
        return False
    if isinstance(expr, StructLit):
        for v in expr.fields.values():
            if _visit_expr_for_cast(v, from_kind, to_kind):
                return True
        return False
    if isinstance(expr, StringConcat):
        for part in expr.parts:
            if _visit_expr_for_cast(part, from_kind, to_kind):
                return True
        return False
    if isinstance(expr, StringFormat):
        for arg in expr.args:
            if _visit_expr_for_cast(arg, from_kind, to_kind):
                return True
        return False
    if isinstance(expr, Substring):
        return (
            _visit_expr_for_cast(expr.string, from_kind, to_kind)
            or _visit_expr_for_cast(expr.start, from_kind, to_kind)
            or _visit_expr_for_cast(expr.end, from_kind, to_kind)
        )
    if isinstance(expr, CharAt):
        return _visit_expr_for_cast(
            expr.string, from_kind, to_kind
        ) or _visit_expr_for_cast(expr.index, from_kind, to_kind)
    if isinstance(expr, CharLen):
        return _visit_expr_for_cast(expr.string, from_kind, to_kind)
    if isinstance(expr, TrimChars):
        return _visit_expr_for_cast(
            expr.string, from_kind, to_kind
        ) or _visit_expr_for_cast(expr.chars, from_kind, to_kind)
    if isinstance(expr, CharClassify):
        return _visit_expr_for_cast(expr.char, from_kind, to_kind)
    if isinstance(expr, ParseInt):
        return _visit_expr_for_cast(expr.string, from_kind, to_kind)
    if isinstance(expr, IntToStr):
        return _visit_expr_for_cast(expr.value, from_kind, to_kind)
    if isinstance(expr, SliceConvert):
        return _visit_expr_for_cast(expr.expr, from_kind, to_kind)
    if isinstance(expr, LastElement):
        return _visit_expr_for_cast(expr.slice, from_kind, to_kind)
    if isinstance(expr, SentinelToOptional):
        return _visit_expr_for_cast(expr.value, from_kind, to_kind)
    if isinstance(expr, ListComp):
        if _visit_expr_for_cast(expr.element, from_kind, to_kind):
            return True
        for gen in expr.generators:
            if _visit_expr_for_cast(gen.iterable, from_kind, to_kind):
                return True
            for cond in gen.conditions:
                if _visit_expr_for_cast(cond, from_kind, to_kind):
                    return True
        return False
    if isinstance(expr, SetComp):
        if _visit_expr_for_cast(expr.element, from_kind, to_kind):
            return True
        for gen in expr.generators:
            if _visit_expr_for_cast(gen.iterable, from_kind, to_kind):
                return True
            for cond in gen.conditions:
                if _visit_expr_for_cast(cond, from_kind, to_kind):
                    return True
        return False
    if isinstance(expr, DictComp):
        if _visit_expr_for_cast(expr.key, from_kind, to_kind) or _visit_expr_for_cast(
            expr.value, from_kind, to_kind
        ):
            return True
        for gen in expr.generators:
            if _visit_expr_for_cast(gen.iterable, from_kind, to_kind):
                return True
            for cond in gen.conditions:
                if _visit_expr_for_cast(cond, from_kind, to_kind):
                    return True
        return False
    return False


def _visit_stmt_for_cast(stmt: Stmt, from_kind: str, to_kind: str) -> bool:
    """Check if a statement contains a Cast between the given primitive kinds."""
    if isinstance(stmt, VarDecl):
        return _visit_expr_for_cast(stmt.value, from_kind, to_kind)
    if isinstance(stmt, Assign):
        return _visit_expr_for_cast(stmt.value, from_kind, to_kind)
    if isinstance(stmt, TupleAssign):
        return _visit_expr_for_cast(stmt.value, from_kind, to_kind)
    if isinstance(stmt, OpAssign):
        return _visit_expr_for_cast(stmt.value, from_kind, to_kind)
    if isinstance(stmt, ExprStmt):
        return _visit_expr_for_cast(stmt.expr, from_kind, to_kind)
    if isinstance(stmt, Return):
        return _visit_expr_for_cast(stmt.value, from_kind, to_kind)
    if isinstance(stmt, If):
        if _visit_expr_for_cast(stmt.cond, from_kind, to_kind):
            return True
        for s in stmt.then_body:
            if _visit_stmt_for_cast(s, from_kind, to_kind):
                return True
        for s in stmt.else_body:
            if _visit_stmt_for_cast(s, from_kind, to_kind):
                return True
        return False
    if isinstance(stmt, TypeSwitch):
        if _visit_expr_for_cast(stmt.expr, from_kind, to_kind):
            return True
        for case in stmt.cases:
            for s in case.body:
                if _visit_stmt_for_cast(s, from_kind, to_kind):
                    return True
        for s in stmt.default_body:
            if _visit_stmt_for_cast(s, from_kind, to_kind):
                return True
        return False
    if isinstance(stmt, Match):
        if _visit_expr_for_cast(stmt.subject, from_kind, to_kind):
            return True
        for case in stmt.cases:
            if _visit_expr_for_cast(case.guard, from_kind, to_kind):
                return True
            for s in case.body:
                if _visit_stmt_for_cast(s, from_kind, to_kind):
                    return True
        return False
    if isinstance(stmt, ForRange):
        if _visit_expr_for_cast(stmt.iterable, from_kind, to_kind):
            return True
        for s in stmt.body:
            if _visit_stmt_for_cast(s, from_kind, to_kind):
                return True
        return False
    if isinstance(stmt, ForClassic):
        if stmt.init is not None and _visit_stmt_for_cast(
            stmt.init, from_kind, to_kind
        ):
            return True
        if _visit_expr_for_cast(stmt.cond, from_kind, to_kind):
            return True
        if stmt.post is not None and _visit_stmt_for_cast(
            stmt.post, from_kind, to_kind
        ):
            return True
        for s in stmt.body:
            if _visit_stmt_for_cast(s, from_kind, to_kind):
                return True
        return False
    if isinstance(stmt, While):
        if _visit_expr_for_cast(stmt.cond, from_kind, to_kind):
            return True
        for s in stmt.body:
            if _visit_stmt_for_cast(s, from_kind, to_kind):
                return True
        return False
    if isinstance(stmt, Block):
        for s in stmt.body:
            if _visit_stmt_for_cast(s, from_kind, to_kind):
                return True
        return False
    if isinstance(stmt, TryCatch):
        for s in stmt.body:
            if _visit_stmt_for_cast(s, from_kind, to_kind):
                return True
        for clause in stmt.catches:
            for s in clause.body:
                if _visit_stmt_for_cast(s, from_kind, to_kind):
                    return True
        return False
    if isinstance(stmt, Raise):
        return _visit_expr_for_cast(stmt.exception, from_kind, to_kind)
    if isinstance(stmt, Assert):
        return _visit_expr_for_cast(
            stmt.test, from_kind, to_kind
        ) or _visit_expr_for_cast(stmt.message, from_kind, to_kind)
    if isinstance(stmt, SoftFail):
        return _visit_expr_for_cast(stmt.msg, from_kind, to_kind)
    if isinstance(stmt, Print):
        for arg in stmt.args:
            if _visit_expr_for_cast(arg, from_kind, to_kind):
                return True
        return False
    if isinstance(stmt, EntryPoint):
        for s in stmt.body:
            if _visit_stmt_for_cast(s, from_kind, to_kind):
                return True
        return False
    return False


def ir_contains_cast(module: Module, from_kind: str, to_kind: str) -> bool:
    """Check if the module's IR contains a Cast between the given primitive kinds."""
    # Check functions
    for fn in module.functions:
        for stmt in fn.body:
            if _visit_stmt_for_cast(stmt, from_kind, to_kind):
                return True
    # Check struct methods
    for struct in module.structs:
        for method in struct.methods:
            for stmt in method.body:
                if _visit_stmt_for_cast(stmt, from_kind, to_kind):
                    return True
    return False


def _is_tuple_set_type(typ: Type | None) -> bool:
    """Check if type is a set with tuple elements."""
    if typ is None:
        return False
    if isinstance(typ, Set) and isinstance(typ.element, Tuple):
        return True
    return False


def _visit_expr_for_tuple_sets(expr: Expr | None) -> bool:
    """Check if an expression involves sets with tuple elements."""
    if expr is None:
        return False
    if isinstance(expr, SetLit) and isinstance(expr.element_type, Tuple):
        return True
    if isinstance(expr, SetComp) and isinstance(expr.element.typ, Tuple):
        return True
    if isinstance(expr, BinaryOp):
        if expr.op in ("in", "not in"):
            if _is_tuple_set_type(expr.right.typ):
                return True
        return _visit_expr_for_tuple_sets(expr.left) or _visit_expr_for_tuple_sets(
            expr.right
        )
    if isinstance(expr, FieldAccess):
        return _visit_expr_for_tuple_sets(expr.obj)
    if isinstance(expr, Index):
        return _visit_expr_for_tuple_sets(expr.obj) or _visit_expr_for_tuple_sets(
            expr.index
        )
    if isinstance(expr, SliceExpr):
        return (
            _visit_expr_for_tuple_sets(expr.obj)
            or _visit_expr_for_tuple_sets(expr.low)
            or _visit_expr_for_tuple_sets(expr.high)
            or _visit_expr_for_tuple_sets(expr.step)
        )
    if isinstance(expr, Call):
        for arg in expr.args:
            if _visit_expr_for_tuple_sets(arg):
                return True
        return False
    if isinstance(expr, MethodCall):
        if _visit_expr_for_tuple_sets(expr.obj):
            return True
        for arg in expr.args:
            if _visit_expr_for_tuple_sets(arg):
                return True
        return False
    if isinstance(expr, StaticCall):
        for arg in expr.args:
            if _visit_expr_for_tuple_sets(arg):
                return True
        return False
    if isinstance(expr, UnaryOp):
        return _visit_expr_for_tuple_sets(expr.operand)
    if isinstance(expr, Ternary):
        return (
            _visit_expr_for_tuple_sets(expr.cond)
            or _visit_expr_for_tuple_sets(expr.then_expr)
            or _visit_expr_for_tuple_sets(expr.else_expr)
        )
    if isinstance(expr, Cast):
        return _visit_expr_for_tuple_sets(expr.expr)
    if isinstance(expr, TypeAssert):
        return _visit_expr_for_tuple_sets(expr.expr)
    if isinstance(expr, IsType):
        return _visit_expr_for_tuple_sets(expr.expr)
    if isinstance(expr, IsNil):
        return _visit_expr_for_tuple_sets(expr.expr)
    if isinstance(expr, Len):
        return _visit_expr_for_tuple_sets(expr.expr)
    if isinstance(expr, MakeSlice):
        return _visit_expr_for_tuple_sets(expr.length) or _visit_expr_for_tuple_sets(
            expr.capacity
        )
    if isinstance(expr, MakeMap):
        return _visit_expr_for_tuple_sets(expr.capacity)
    if isinstance(expr, SliceLit):
        for elem in expr.elements:
            if _visit_expr_for_tuple_sets(elem):
                return True
        return False
    if isinstance(expr, MapLit):
        for k, v in expr.entries:
            if _visit_expr_for_tuple_sets(k) or _visit_expr_for_tuple_sets(v):
                return True
        return False
    if isinstance(expr, TupleLit):
        for elem in expr.elements:
            if _visit_expr_for_tuple_sets(elem):
                return True
        return False
    if isinstance(expr, StructLit):
        for v in expr.fields.values():
            if _visit_expr_for_tuple_sets(v):
                return True
        return False
    if isinstance(expr, StringConcat):
        for part in expr.parts:
            if _visit_expr_for_tuple_sets(part):
                return True
        return False
    if isinstance(expr, StringFormat):
        for arg in expr.args:
            if _visit_expr_for_tuple_sets(arg):
                return True
        return False
    if isinstance(expr, Substring):
        return (
            _visit_expr_for_tuple_sets(expr.string)
            or _visit_expr_for_tuple_sets(expr.start)
            or _visit_expr_for_tuple_sets(expr.end)
        )
    if isinstance(expr, CharAt):
        return _visit_expr_for_tuple_sets(expr.string) or _visit_expr_for_tuple_sets(
            expr.index
        )
    if isinstance(expr, CharLen):
        return _visit_expr_for_tuple_sets(expr.string)
    if isinstance(expr, TrimChars):
        return _visit_expr_for_tuple_sets(expr.string) or _visit_expr_for_tuple_sets(
            expr.chars
        )
    if isinstance(expr, CharClassify):
        return _visit_expr_for_tuple_sets(expr.char)
    if isinstance(expr, ParseInt):
        return _visit_expr_for_tuple_sets(expr.string)
    if isinstance(expr, IntToStr):
        return _visit_expr_for_tuple_sets(expr.value)
    if isinstance(expr, SliceConvert):
        return _visit_expr_for_tuple_sets(expr.expr)
    if isinstance(expr, LastElement):
        return _visit_expr_for_tuple_sets(expr.slice)
    if isinstance(expr, SentinelToOptional):
        return _visit_expr_for_tuple_sets(expr.value)
    if isinstance(expr, ListComp):
        if _visit_expr_for_tuple_sets(expr.element):
            return True
        for gen in expr.generators:
            if _visit_expr_for_tuple_sets(gen.iterable):
                return True
            for cond in gen.conditions:
                if _visit_expr_for_tuple_sets(cond):
                    return True
        return False
    if isinstance(expr, DictComp):
        if _visit_expr_for_tuple_sets(expr.key) or _visit_expr_for_tuple_sets(
            expr.value
        ):
            return True
        for gen in expr.generators:
            if _visit_expr_for_tuple_sets(gen.iterable):
                return True
            for cond in gen.conditions:
                if _visit_expr_for_tuple_sets(cond):
                    return True
        return False
    return False


def _visit_stmt_for_tuple_sets(stmt: Stmt) -> bool:
    """Check if a statement involves sets with tuple elements."""
    if isinstance(stmt, VarDecl):
        return _visit_expr_for_tuple_sets(stmt.value)
    if isinstance(stmt, Assign):
        return _visit_expr_for_tuple_sets(stmt.value)
    if isinstance(stmt, TupleAssign):
        return _visit_expr_for_tuple_sets(stmt.value)
    if isinstance(stmt, OpAssign):
        return _visit_expr_for_tuple_sets(stmt.value)
    if isinstance(stmt, ExprStmt):
        return _visit_expr_for_tuple_sets(stmt.expr)
    if isinstance(stmt, Return):
        return _visit_expr_for_tuple_sets(stmt.value)
    if isinstance(stmt, If):
        if _visit_expr_for_tuple_sets(stmt.cond):
            return True
        for s in stmt.then_body:
            if _visit_stmt_for_tuple_sets(s):
                return True
        for s in stmt.else_body:
            if _visit_stmt_for_tuple_sets(s):
                return True
        return False
    if isinstance(stmt, TypeSwitch):
        if _visit_expr_for_tuple_sets(stmt.expr):
            return True
        for case in stmt.cases:
            for s in case.body:
                if _visit_stmt_for_tuple_sets(s):
                    return True
        for s in stmt.default_body:
            if _visit_stmt_for_tuple_sets(s):
                return True
        return False
    if isinstance(stmt, Match):
        if _visit_expr_for_tuple_sets(stmt.subject):
            return True
        for case in stmt.cases:
            if _visit_expr_for_tuple_sets(case.guard):
                return True
            for s in case.body:
                if _visit_stmt_for_tuple_sets(s):
                    return True
        return False
    if isinstance(stmt, ForRange):
        if _visit_expr_for_tuple_sets(stmt.iterable):
            return True
        for s in stmt.body:
            if _visit_stmt_for_tuple_sets(s):
                return True
        return False
    if isinstance(stmt, ForClassic):
        if stmt.init is not None and _visit_stmt_for_tuple_sets(stmt.init):
            return True
        if _visit_expr_for_tuple_sets(stmt.cond):
            return True
        if stmt.post is not None and _visit_stmt_for_tuple_sets(stmt.post):
            return True
        for s in stmt.body:
            if _visit_stmt_for_tuple_sets(s):
                return True
        return False
    if isinstance(stmt, While):
        if _visit_expr_for_tuple_sets(stmt.cond):
            return True
        for s in stmt.body:
            if _visit_stmt_for_tuple_sets(s):
                return True
        return False
    if isinstance(stmt, Block):
        for s in stmt.body:
            if _visit_stmt_for_tuple_sets(s):
                return True
        return False
    if isinstance(stmt, TryCatch):
        for s in stmt.body:
            if _visit_stmt_for_tuple_sets(s):
                return True
        for clause in stmt.catches:
            for s in clause.body:
                if _visit_stmt_for_tuple_sets(s):
                    return True
        return False
    if isinstance(stmt, Raise):
        return _visit_expr_for_tuple_sets(stmt.exception)
    if isinstance(stmt, Assert):
        return _visit_expr_for_tuple_sets(stmt.test) or _visit_expr_for_tuple_sets(
            stmt.message
        )
    if isinstance(stmt, SoftFail):
        return _visit_expr_for_tuple_sets(stmt.msg)
    if isinstance(stmt, Print):
        for arg in stmt.args:
            if _visit_expr_for_tuple_sets(arg):
                return True
        return False
    if isinstance(stmt, EntryPoint):
        for s in stmt.body:
            if _visit_stmt_for_tuple_sets(s):
                return True
        return False
    return False


def ir_has_tuple_sets(module: Module) -> bool:
    """Check if the module uses sets with tuple elements."""
    for fn in module.functions:
        for stmt in fn.body:
            if _visit_stmt_for_tuple_sets(stmt):
                return True
    for struct in module.structs:
        for method in struct.methods:
            for stmt in method.body:
                if _visit_stmt_for_tuple_sets(stmt):
                    return True
    return False


def _is_tuple_map_type(typ: Type | None) -> bool:
    """Check if type is a map with tuple keys."""
    if typ is None:
        return False
    if isinstance(typ, Map) and isinstance(typ.key, Tuple):
        return True
    return False


def _visit_expr_for_tuple_maps(expr: Expr | None) -> bool:
    """Check if an expression involves maps with tuple keys."""
    if expr is None:
        return False
    if isinstance(expr, MapLit) and expr.entries:
        first_key = expr.entries[0][0]
        if isinstance(first_key.typ, Tuple):
            return True
    if isinstance(expr, Index) and _is_tuple_map_type(expr.obj.typ):
        return True
    if isinstance(expr, BinaryOp):
        if expr.op in ("in", "not in"):
            if _is_tuple_map_type(expr.right.typ):
                return True
        return _visit_expr_for_tuple_maps(expr.left) or _visit_expr_for_tuple_maps(
            expr.right
        )
    if isinstance(expr, FieldAccess):
        return _visit_expr_for_tuple_maps(expr.obj)
    if isinstance(expr, SliceExpr):
        return (
            _visit_expr_for_tuple_maps(expr.obj)
            or _visit_expr_for_tuple_maps(expr.low)
            or _visit_expr_for_tuple_maps(expr.high)
            or _visit_expr_for_tuple_maps(expr.step)
        )
    if isinstance(expr, Call):
        for arg in expr.args:
            if _visit_expr_for_tuple_maps(arg):
                return True
        return False
    if isinstance(expr, MethodCall):
        if _visit_expr_for_tuple_maps(expr.obj):
            return True
        for arg in expr.args:
            if _visit_expr_for_tuple_maps(arg):
                return True
        return False
    if isinstance(expr, StaticCall):
        for arg in expr.args:
            if _visit_expr_for_tuple_maps(arg):
                return True
        return False
    if isinstance(expr, UnaryOp):
        return _visit_expr_for_tuple_maps(expr.operand)
    if isinstance(expr, Ternary):
        return (
            _visit_expr_for_tuple_maps(expr.cond)
            or _visit_expr_for_tuple_maps(expr.then_expr)
            or _visit_expr_for_tuple_maps(expr.else_expr)
        )
    if isinstance(expr, Cast):
        return _visit_expr_for_tuple_maps(expr.expr)
    if isinstance(expr, TypeAssert):
        return _visit_expr_for_tuple_maps(expr.expr)
    if isinstance(expr, IsType):
        return _visit_expr_for_tuple_maps(expr.expr)
    if isinstance(expr, IsNil):
        return _visit_expr_for_tuple_maps(expr.expr)
    if isinstance(expr, Len):
        return _visit_expr_for_tuple_maps(expr.expr)
    if isinstance(expr, MakeSlice):
        return _visit_expr_for_tuple_maps(expr.length) or _visit_expr_for_tuple_maps(
            expr.capacity
        )
    if isinstance(expr, MakeMap):
        return _visit_expr_for_tuple_maps(expr.capacity)
    if isinstance(expr, SliceLit):
        for elem in expr.elements:
            if _visit_expr_for_tuple_maps(elem):
                return True
        return False
    if isinstance(expr, TupleLit):
        for elem in expr.elements:
            if _visit_expr_for_tuple_maps(elem):
                return True
        return False
    if isinstance(expr, StructLit):
        for v in expr.fields.values():
            if _visit_expr_for_tuple_maps(v):
                return True
        return False
    if isinstance(expr, StringConcat):
        for part in expr.parts:
            if _visit_expr_for_tuple_maps(part):
                return True
        return False
    if isinstance(expr, StringFormat):
        for arg in expr.args:
            if _visit_expr_for_tuple_maps(arg):
                return True
        return False
    if isinstance(expr, Substring):
        return (
            _visit_expr_for_tuple_maps(expr.string)
            or _visit_expr_for_tuple_maps(expr.start)
            or _visit_expr_for_tuple_maps(expr.end)
        )
    if isinstance(expr, CharAt):
        return _visit_expr_for_tuple_maps(expr.string) or _visit_expr_for_tuple_maps(
            expr.index
        )
    if isinstance(expr, CharLen):
        return _visit_expr_for_tuple_maps(expr.string)
    if isinstance(expr, TrimChars):
        return _visit_expr_for_tuple_maps(expr.string) or _visit_expr_for_tuple_maps(
            expr.chars
        )
    if isinstance(expr, CharClassify):
        return _visit_expr_for_tuple_maps(expr.char)
    if isinstance(expr, ParseInt):
        return _visit_expr_for_tuple_maps(expr.string)
    if isinstance(expr, IntToStr):
        return _visit_expr_for_tuple_maps(expr.value)
    if isinstance(expr, SliceConvert):
        return _visit_expr_for_tuple_maps(expr.expr)
    if isinstance(expr, LastElement):
        return _visit_expr_for_tuple_maps(expr.slice)
    if isinstance(expr, SentinelToOptional):
        return _visit_expr_for_tuple_maps(expr.value)
    if isinstance(expr, ListComp):
        if _visit_expr_for_tuple_maps(expr.element):
            return True
        for gen in expr.generators:
            if _visit_expr_for_tuple_maps(gen.iterable):
                return True
            for cond in gen.conditions:
                if _visit_expr_for_tuple_maps(cond):
                    return True
        return False
    if isinstance(expr, DictComp):
        if _visit_expr_for_tuple_maps(expr.key) or _visit_expr_for_tuple_maps(
            expr.value
        ):
            return True
        for gen in expr.generators:
            if _visit_expr_for_tuple_maps(gen.iterable):
                return True
            for cond in gen.conditions:
                if _visit_expr_for_tuple_maps(cond):
                    return True
        return False
    return False


def _visit_stmt_for_tuple_maps(stmt: Stmt) -> bool:
    """Check if a statement involves maps with tuple keys."""
    if isinstance(stmt, VarDecl):
        return _visit_expr_for_tuple_maps(stmt.value)
    if isinstance(stmt, Assign):
        return _visit_expr_for_tuple_maps(stmt.value)
    if isinstance(stmt, TupleAssign):
        return _visit_expr_for_tuple_maps(stmt.value)
    if isinstance(stmt, OpAssign):
        return _visit_expr_for_tuple_maps(stmt.value)
    if isinstance(stmt, ExprStmt):
        return _visit_expr_for_tuple_maps(stmt.expr)
    if isinstance(stmt, Return):
        return _visit_expr_for_tuple_maps(stmt.value)
    if isinstance(stmt, If):
        if _visit_expr_for_tuple_maps(stmt.cond):
            return True
        for s in stmt.then_body:
            if _visit_stmt_for_tuple_maps(s):
                return True
        for s in stmt.else_body:
            if _visit_stmt_for_tuple_maps(s):
                return True
        return False
    if isinstance(stmt, TypeSwitch):
        if _visit_expr_for_tuple_maps(stmt.expr):
            return True
        for case in stmt.cases:
            for s in case.body:
                if _visit_stmt_for_tuple_maps(s):
                    return True
        for s in stmt.default_body:
            if _visit_stmt_for_tuple_maps(s):
                return True
        return False
    if isinstance(stmt, Match):
        if _visit_expr_for_tuple_maps(stmt.subject):
            return True
        for case in stmt.cases:
            if _visit_expr_for_tuple_maps(case.guard):
                return True
            for s in case.body:
                if _visit_stmt_for_tuple_maps(s):
                    return True
        return False
    if isinstance(stmt, ForRange):
        if _visit_expr_for_tuple_maps(stmt.iterable):
            return True
        for s in stmt.body:
            if _visit_stmt_for_tuple_maps(s):
                return True
        return False
    if isinstance(stmt, ForClassic):
        if stmt.init is not None and _visit_stmt_for_tuple_maps(stmt.init):
            return True
        if _visit_expr_for_tuple_maps(stmt.cond):
            return True
        if stmt.post is not None and _visit_stmt_for_tuple_maps(stmt.post):
            return True
        for s in stmt.body:
            if _visit_stmt_for_tuple_maps(s):
                return True
        return False
    if isinstance(stmt, While):
        if _visit_expr_for_tuple_maps(stmt.cond):
            return True
        for s in stmt.body:
            if _visit_stmt_for_tuple_maps(s):
                return True
        return False
    if isinstance(stmt, Block):
        for s in stmt.body:
            if _visit_stmt_for_tuple_maps(s):
                return True
        return False
    if isinstance(stmt, TryCatch):
        for s in stmt.body:
            if _visit_stmt_for_tuple_maps(s):
                return True
        for clause in stmt.catches:
            for s in clause.body:
                if _visit_stmt_for_tuple_maps(s):
                    return True
        return False
    if isinstance(stmt, Raise):
        return _visit_expr_for_tuple_maps(stmt.exception)
    if isinstance(stmt, Assert):
        return _visit_expr_for_tuple_maps(stmt.test) or _visit_expr_for_tuple_maps(
            stmt.message
        )
    if isinstance(stmt, SoftFail):
        return _visit_expr_for_tuple_maps(stmt.msg)
    if isinstance(stmt, Print):
        for arg in stmt.args:
            if _visit_expr_for_tuple_maps(arg):
                return True
        return False
    if isinstance(stmt, EntryPoint):
        for s in stmt.body:
            if _visit_stmt_for_tuple_maps(s):
                return True
        return False
    return False


def ir_has_tuple_maps(module: Module) -> bool:
    """Check if the module uses maps with tuple keys."""
    for fn in module.functions:
        for stmt in fn.body:
            if _visit_stmt_for_tuple_maps(stmt):
                return True
    for struct in module.structs:
        for method in struct.methods:
            for stmt in method.body:
                if _visit_stmt_for_tuple_maps(stmt):
                    return True
    return False
