"""Perl backend: Taytsh AST -> Perl 5.36+ source code."""

from __future__ import annotations

from .ordering import order_decls
from .util import STRICT_INT_BINARY, STRICT_INT_COMPOUND, Emitter, to_snake
from ..taytsh.ast import (
    Ann,
    Pos,
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
    TTryStmt,
    TTupleAccess,
    TTupleAssignStmt,
    TTupleLit,
    TTupleType,
    TType,
    TUnaryOp,
    TUnionType,
    TVar,
    TWhileStmt,
)
from ..taytsh.check import BUILTIN_NAMES, BUILTIN_STRUCTS

_PERL_RESERVED = frozenset(
    {
        "and",
        "cmp",
        "continue",
        "do",
        "else",
        "elsif",
        "eq",
        "for",
        "foreach",
        "ge",
        "gt",
        "if",
        "last",
        "le",
        "lt",
        "my",
        "ne",
        "next",
        "no",
        "not",
        "or",
        "package",
        "return",
        "sub",
        "undef",
        "unless",
        "until",
        "use",
        "while",
        "xor",
    }
)


def _safe_name(name: str) -> str:
    if name == "_":
        return "_unused"
    prefix = "_" if name.startswith("_") and len(name) > 1 else ""
    safe = to_snake(name)
    if not safe:
        return "_unused"
    safe = prefix + safe
    if safe in _PERL_RESERVED:
        return safe + "_"
    return safe


def _restore_name(name: str, annotations: Ann) -> str:
    """Restore original Python name from annotation, then apply target safety."""
    key = "name.original." + name
    if key in annotations:
        return _safe_name(annotations[key])
    return _safe_name(name)


_PERL_ESCAPE_MAP: dict[str, str] = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
    "$": "\\$",
    "@": "\\@",
}


def _escape_perl_string(value: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(value):
        c = value[i]
        esc = _PERL_ESCAPE_MAP.get(c)
        if esc is not None:
            out.append(esc)
        elif ord(c) < 32 or ord(c) > 126:
            out.append("\\x{" + hex(ord(c))[2:] + "}")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _escape_perl_regex(s: str) -> str:
    result: list[str] = []
    for ch in s:
        if ch == "$" or ch == "@":
            h = hex(ord(ch))[2:]
            if len(h) == 1:
                h = "0" + h
            result.append("\\x{" + h + "}")
        elif ch in r".^*+?{}[]\|()/":
            result.append("\\" + ch)
        elif ch == "\n":
            result.append("\\n")
        elif ch == "\t":
            result.append("\\t")
        elif ch == "\r":
            result.append("\\r")
        elif ord(ch) < 32 or ord(ch) > 126:
            h = hex(ord(ch))[2:]
            if len(h) == 1:
                h = "0" + h
            result.append("\\x{" + h + "}")
        else:
            result.append(ch)
    return "".join(result)


def _escape_perl_replacement(s: str) -> str:
    result: list[str] = []
    for ch in s:
        if ch == "\\":
            result.append("\\\\")
        elif ch == "$":
            result.append("\\$")
        elif ch == "/":
            result.append("\\/")
        elif ch == "\n":
            result.append("\\n")
        elif ch == "\t":
            result.append("\\t")
        elif ch == "\r":
            result.append("\\r")
        elif ord(ch) < 32 or ord(ch) > 126:
            h = hex(ord(ch))[2:]
            if len(h) == 1:
                h = "0" + h
            result.append("\\x{" + h + "}")
        else:
            result.append(ch)
    return "".join(result)


def _escape_regex_charclass(s: str) -> str:
    result: list[str] = []
    for ch in s:
        if ch in r"]\^-":
            result.append("\\" + ch)
        elif ch == "\n":
            result.append("\\n")
        elif ch == "\t":
            result.append("\\t")
        elif ch == "\r":
            result.append("\\r")
        elif ord(ch) < 32 or ord(ch) > 126:
            h = hex(ord(ch))[2:]
            if len(h) == 1:
                h = "0" + h
            result.append("\\x{" + h + "}")
        else:
            result.append(ch)
    return "".join(result)


def _string_literal(value: str) -> str:
    return '"' + _escape_perl_string(value) + '"'


_PRECEDENCE: dict[str, int] = {
    "or": 1,
    "xor": 1,
    "and": 2,
    "||": 3,
    "&&": 4,
    "|": 5,
    "^": 5,
    "&": 6,
    "eq": 7,
    "ne": 7,
    "lt": 7,
    "gt": 7,
    "le": 7,
    "ge": 7,
    "==": 7,
    "!=": 7,
    "<": 7,
    ">": 7,
    "<=": 7,
    ">=": 7,
    "<<": 8,
    ">>": 8,
    ".": 9,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
    "//": 10,
    "**": 11,
}

_CMP_OPS = frozenset(
    ["==", "!=", "<", ">", "<=", ">=", "eq", "ne", "lt", "gt", "le", "ge"]
)


def _needs_parens(child_op: str, parent_op: str, is_left: bool) -> bool:
    if child_op in _CMP_OPS and parent_op in _CMP_OPS:
        return True
    child_prec = _PRECEDENCE.get(child_op, 0)
    parent_prec = _PRECEDENCE.get(parent_op, 0)
    if child_prec < parent_prec:
        return True
    if child_prec == parent_prec and not is_left:
        if child_op in _CMP_OPS:
            return True
        if parent_op in ("-", "/", "%", "//"):
            return True
    return False


def _is_primitive(typ: TType | None, kind: str) -> bool:
    if isinstance(typ, TOptionalType):
        typ = typ.inner
    return isinstance(typ, TPrimitive) and typ.kind == kind


def _is_list_type(typ: TType | None) -> bool:
    return isinstance(typ, (TListType, TTupleType))


def _is_map_type(typ: TType | None) -> bool:
    return isinstance(typ, TMapType)


def _is_set_type(typ: TType | None) -> bool:
    return isinstance(typ, TSetType)


def _is_string_type(typ: TType | None) -> bool:
    return _is_primitive(typ, "string") or _is_primitive(typ, "rune")


def _is_bytes_type(typ: TType | None) -> bool:
    return _is_primitive(typ, "bytes")


def _same_type_kind(a: TType, b: TType) -> bool:
    if isinstance(a, TPrimitive) and isinstance(b, TPrimitive):
        return a.kind == b.kind
    if isinstance(a, TListType) and isinstance(b, TListType):
        return True
    if isinstance(a, TMapType) and isinstance(b, TMapType):
        return True
    if isinstance(a, TSetType) and isinstance(b, TSetType):
        return True
    if isinstance(a, TIdentType) and isinstance(b, TIdentType):
        return a.name == b.name
    return False


class _PerlEmitter(Emitter):
    def __init__(
        self,
        struct_names: set[str],
        enum_names: set[str],
        function_names: set[str],
        struct_fields: dict[str, list[str]],
        strict_math: bool = False,
    ) -> None:
        self.struct_names = struct_names
        self.enum_names = enum_names
        self.function_names = function_names
        self.struct_fields = struct_fields
        self.strict_math = strict_math
        self.sft: dict[str, dict[str, TType]] = {}
        self.indent: int = 0
        self.lines: list[str] = []
        self.self_name: str | None = None
        self.var_types: dict[str, TType] = {}
        self.tmp_counter: int = 0
        self.var_alias: dict[str, str] = {}
        self.fwd_declared: set[str] = set()
        self.in_package: bool = False
        self.module_var_names: set[str] = set()
        self.local_names: set[str] = set()
        self.fn_ret: dict[str, TType] = {}

    def _tmp(self, prefix: str = "__t") -> str:
        name = "$" + prefix + str(self.tmp_counter)
        self.tmp_counter += 1
        return name

    def emit_module(self, module: TModule) -> None:
        self._line("use v5.36;")
        self._line("use utf8;")
        self._line("no warnings 'uninitialized', 'numeric';")
        self._line("use POSIX qw(floor ceil);")
        self._line("use List::Util qw(min max sum);")
        self._line("use Scalar::Util qw(looks_like_number);")
        self._line("use Encode qw(encode decode);")
        self._line("binmode(STDOUT, ':utf8');")
        self._line("binmode(STDERR, ':utf8');")
        self._line()
        self._line("package main;")
        ordered = order_decls(module.decls)
        has_types = any(
            isinstance(d, (TStructDecl, TEnumDecl, TInterfaceDecl)) for d in ordered
        )
        for decl in ordered:
            if isinstance(decl, TLetStmt):
                self.var_types[decl.name] = decl.typ
                self.module_var_names.add(decl.name)
                if has_types:
                    safe = _restore_name(decl.name, decl.annotations)
                    self.fwd_declared.add(decl.name)
                    self._line("our $" + safe + ";")
        for decl in ordered:
            if isinstance(decl, TFnDecl) and decl.ret is not None:
                self.fn_ret[decl.name] = decl.ret
        need_blank = False
        current_package = "main"
        for decl in ordered:
            if isinstance(decl, TInterfaceDecl):
                if need_blank:
                    self._line()
                self._line("package " + decl.name + ";")
                current_package = decl.name
                if decl.fields:
                    self._line()
                    self._emit_interface_constructor(decl)
                need_blank = True
                continue
            if need_blank:
                self._line()
            if isinstance(decl, TStructDecl):
                if current_package != decl.name:
                    self._line("package " + decl.name + ";")
                    current_package = decl.name
                    self._line()
                self._emit_struct(decl)
                need_blank = True
            elif isinstance(decl, TEnumDecl):
                if current_package != decl.name:
                    self._line("package " + decl.name + ";")
                    current_package = decl.name
                    self._line()
                self._emit_enum(decl)
                need_blank = True
            elif isinstance(decl, TLetStmt):
                if current_package != "main":
                    self._line("package main;")
                    self._line()
                    current_package = "main"
                self._emit_stmt(decl)
                need_blank = True
            elif isinstance(decl, TFnDecl):
                if current_package != "main":
                    self._line("package main;")
                    self._line()
                    current_package = "main"
                self._emit_fn(decl)
                need_blank = True

    def _emit_enum(self, decl: TEnumDecl) -> None:
        for i, variant in enumerate(decl.variants):
            self._line("use constant " + variant + " => " + str(i) + ";")

    def _emit_struct(self, decl: TStructDecl) -> None:
        if decl.parent is not None:
            self._line("use parent -norequire, '" + decl.parent + "';")
            self._line()
        str_method = ""
        for method in decl.methods:
            if method.name == "to_string":
                str_method = "to_string"
                break
            if method.name == "__repr__":
                str_method = "__repr__"
        if str_method != "":
            self._line("use overload '\"\"' => \\&" + str_method + ", fallback => 1;")
            self._line()
        self._emit_constructor(decl)
        for method in decl.methods:
            self._line()
            self._emit_method(method, decl.name)

    def _emit_constructor(self, decl: TStructDecl) -> None:
        self._emit_constructor_fields(decl.fields)

    def _emit_interface_constructor(self, decl: TInterfaceDecl) -> None:
        self._emit_constructor_fields(decl.fields)

    def _emit_constructor_fields(self, fields: list[TFieldDecl]) -> None:
        self._line("sub new {")
        self.indent += 1
        args: list[str] = ["$class"]
        for f in fields:
            args.append("$" + _safe_name(f.name))
        self._line("my (" + ", ".join(args) + ") = @_;")
        self._line("my $self = bless {}, $class;")
        for fld in fields:
            safe = _safe_name(fld.name)
            default = self._zero_value(fld.typ)
            self._line(
                "$self->{"
                + safe
                + "} = defined $"
                + safe
                + " ? $"
                + safe
                + " : "
                + default
                + ";"
            )
        self._line("return $self;")
        self.indent -= 1
        self._line("}")

    def _emit_fn(self, decl: TFnDecl) -> None:
        old_var_types = self.var_types.copy()
        fn_types: dict[str, TType] = {}
        for name in self.module_var_names:
            if name in old_var_types:
                fn_types[name] = old_var_types[name]
        self.var_types = fn_types
        args: list[str] = []
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
                args.append("$" + _restore_name(p.name, p.annotations))
        self._line("sub " + _safe_name(decl.name) + " {")
        self.indent += 1
        if len(args) > 0:
            self._line("my (" + ", ".join(args) + ") = @_;")
        if len(decl.body) == 0:
            self._line("return;")
        self._emit_stmts(decl.body)
        self.indent -= 1
        self._line("}")
        self.var_types = old_var_types

    def _emit_method(self, decl: TFnDecl, struct_name: str = "") -> None:
        old_var_types = self.var_types.copy()
        method_types: dict[str, TType] = {}
        for name in self.module_var_names:
            if name in old_var_types:
                method_types[name] = old_var_types[name]
        self.var_types = method_types
        old_in_package = self.in_package
        self.in_package = True
        old_local_names = self.local_names
        self.local_names = set()
        args: list[str] = ["$self"]
        for p in decl.params:
            if p.typ is not None:
                self.var_types[p.name] = p.typ
                self.local_names.add(p.name)
                args.append("$" + _restore_name(p.name, p.annotations))
            elif struct_name:
                self.var_types[p.name] = TIdentType(decl.pos, struct_name)
        self._line("sub " + _safe_name(decl.name) + " {")
        self.indent += 1
        self._line("my (" + ", ".join(args) + ") = @_;")
        old_self = self.self_name
        if len(decl.params) > 0 and decl.params[0].typ is None:
            self.self_name = decl.params[0].name
        if len(decl.body) == 0:
            self._line("return;")
        self._emit_stmts(decl.body)
        self.self_name = old_self
        self.in_package = old_in_package
        self.local_names = old_local_names
        self.indent -= 1
        self._line("}")
        self.var_types = old_var_types

    def _emit_stmts(self, stmts: list[TStmt]) -> None:
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
                            self.var_types[stmt.name] = stmt.typ
                            self._line(comp)
                            i += 2
                            continue
            self._emit_stmt(stmt)
            i += 1

    def _try_comprehension(
        self, let_stmt: TLetStmt, for_stmt: TForStmt, prov: str
    ) -> str | None:
        acc = "$" + _restore_name(let_stmt.name, let_stmt.annotations)
        iterable = self._expr(for_stmt.iterable)
        body = for_stmt.body
        if prov == "list_comprehension":
            binding_name = for_stmt.binding[0] if for_stmt.binding else None
            if len(body) == 1 and isinstance(body[0], TExprStmt):
                call = body[0].expr
                if self._is_append_to(call, let_stmt.name):
                    if binding_name is not None:
                        self.var_alias[binding_name] = "$_"
                    val = self._expr(call.args[1].value)
                    if binding_name is not None:
                        self.var_alias.pop(binding_name)
                    return "my " + acc + " = [map { " + val + " } @{" + iterable + "}];"
            if len(body) == 1:
                if_stmt = body[0]
                if isinstance(if_stmt, TIfStmt) and len(if_stmt.then_body) == 1:
                    then_first = if_stmt.then_body[0]
                    if isinstance(then_first, TExprStmt) and isinstance(
                        then_first.expr, TCall
                    ):
                        call = then_first.expr
                        if self._is_append_to(call, let_stmt.name):
                            if binding_name is not None:
                                self.var_alias[binding_name] = "$_"
                            guard = self._expr(if_stmt.cond)
                            if binding_name is not None:
                                self.var_alias.pop(binding_name)
                            return (
                                "my "
                                + acc
                                + " = [grep { "
                                + guard
                                + " } @{"
                                + iterable
                                + "}];"
                            )
        elif prov == "dict_comprehension":
            binding = for_stmt.binding
            if (
                len(binding) == 2
                and len(body) == 1
                and isinstance(body[0], TAssignStmt)
            ):
                target = body[0].target
                if isinstance(target, TIndex):
                    key_var = "$" + _safe_name(binding[0])
                    val_var = "$" + _safe_name(binding[1])
                    key = self._expr(target.index)
                    val = self._expr(body[0].value)
                    pad = "    " * (self.indent + 1)
                    return (
                        "my "
                        + acc
                        + " = do {\n"
                        + pad
                        + "my $__m = {};\n"
                        + pad
                        + "for my "
                        + key_var
                        + " (0 .. $#{"
                        + iterable
                        + "}) { my "
                        + val_var
                        + " = "
                        + iterable
                        + "->["
                        + key_var
                        + "]; $__m->{"
                        + key
                        + "} = "
                        + val
                        + "; }\n"
                        + pad
                        + "$__m;\n"
                        + "    " * self.indent
                        + "};"
                    )
        elif prov == "set_comprehension":
            if len(body) == 1 and isinstance(body[0], TExprStmt):
                call = body[0].expr
                if self._is_add_to(call, let_stmt.name):
                    return (
                        "my "
                        + acc
                        + " = do { my $__s = {}; $__s->{$_} = 1 for @{"
                        + self._deref_safe(iterable)
                        + "}; $__s };"
                    )
        return None

    def _emit_stmt(self, stmt: TStmt) -> None:
        if isinstance(stmt, TLetStmt):
            self.var_types[stmt.name] = stmt.typ
            self.local_names.add(stmt.name)
            safe = "$" + _restore_name(stmt.name, stmt.annotations)
            prefix = "" if stmt.name in self.fwd_declared else "my "
            unused = stmt.annotations.get("liveness.initial_value_unused") == "true"
            if stmt.value is not None and not unused:
                self._line(prefix + safe + " = " + self._expr(stmt.value) + ";")
            else:
                self._line(prefix + safe + " = " + self._zero_value(stmt.typ) + ";")
            return
        if isinstance(stmt, TAssignStmt):
            self._line(self._target(stmt.target) + " = " + self._expr(stmt.value) + ";")
            return
        if isinstance(stmt, TOpAssignStmt):
            op = stmt.op
            if (
                self.strict_math
                and op in STRICT_INT_COMPOUND
                and self._is_int_expr(stmt.target)
            ):
                fn = STRICT_INT_COMPOUND[op]
                tgt = self._target(stmt.target)
                self._line(
                    tgt + " = " + fn + "(" + tgt + ", " + self._expr(stmt.value) + ");"
                )
                return
            if op == "+=" and self._is_string_expr(stmt.target):
                op = ".="
            self._line(
                self._target(stmt.target)
                + " "
                + op
                + " "
                + self._expr(stmt.value)
                + ";"
            )
            return
        if isinstance(stmt, TTupleAssignStmt):
            self._emit_tuple_assign(stmt)
            return
        if isinstance(stmt, TExprStmt):
            self._emit_expr_stmt(stmt)
            return
        if isinstance(stmt, TReturnStmt):
            if stmt.value is None:
                self._line("return;")
            else:
                self._line("return " + self._expr(stmt.value) + ";")
            return
        if isinstance(stmt, TThrowStmt):
            self._line("die " + self._expr(stmt.expr) + ";")
            return
        if isinstance(stmt, TBreakStmt):
            self._line("last;")
            return
        if isinstance(stmt, TContinueStmt):
            self._line("next;")
            return
        if isinstance(stmt, TIfStmt):
            self._emit_if(stmt)
            return
        if isinstance(stmt, TWhileStmt):
            if stmt.annotations.get("provenance") == "negated_while":
                inner = self._negated_inner(stmt.cond)
                if inner is not None:
                    self._line("until (" + inner + ") {")
                    self.indent += 1
                    self._emit_stmts(stmt.body)
                    self.indent -= 1
                    self._line("}")
                    return
            self._line("while (" + self._expr(stmt.cond) + ") {")
            self.indent += 1
            self._emit_stmts(stmt.body)
            self.indent -= 1
            self._line("}")
            return
        if isinstance(stmt, TForStmt):
            self._emit_for(stmt)
            return
        if isinstance(stmt, TTryStmt):
            self._emit_try(stmt)
            return
        if isinstance(stmt, TMatchStmt):
            self._emit_match(stmt)
            return
        raise NotImplementedError("unknown statement")

    def _emit_expr_stmt(self, stmt: TExprStmt) -> None:
        expr = stmt.expr
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            name = expr.func.name
            if name == "Assert":
                cond = self._a(expr.args, 0)
                if len(expr.args) > 1:
                    msg = self._a(expr.args, 1)
                else:
                    msg = '"assertion failed"'
                self._line("die(" + msg + ") unless (" + cond + ");")
                return
            if name == "Delete":
                m = self._a(expr.args, 0)
                k = self._hash_key(expr.args[1].value)
                self._line("delete " + m + "->{" + k + "};")
                return
        if isinstance(expr, TStringLit):
            return
        self._line(self._expr(expr) + ";")

    def _emit_tuple_assign(self, stmt: TTupleAssignStmt) -> None:
        unused_str = stmt.annotations.get("liveness.tuple_unused_indices", "")
        unused_indices: set[int] = set()
        if unused_str != "":
            for s in unused_str.split(","):
                if s != "":
                    unused_indices.add(int(s))
        parts: list[str] = []
        for i, t in enumerate(stmt.targets):
            is_discard = isinstance(t, TVar) and t.name == "_"
            if i in unused_indices or is_discard:
                parts.append("undef")
            else:
                parts.append(self._target(t))
        rhs = self._expr(stmt.value)
        if rhs.startswith("[") and rhs.endswith("]"):
            self._line("(" + ", ".join(parts) + ") = (" + rhs[1:-1] + ");")
        else:
            self._line("(" + ", ".join(parts) + ") = @{" + rhs + "};")

    def _emit_if(self, stmt: TIfStmt) -> None:
        prov = stmt.annotations.get("provenance")
        if prov == "truthiness":
            truth = self._truthiness_expr(stmt.cond)
            if truth is not None:
                self._line("if (" + truth + ") {")
                self.indent += 1
                self._emit_stmts(stmt.then_body)
                self.indent -= 1
                self._emit_else_body(stmt.else_body)
                self._line("}")
                return
        if prov == "negated_condition" and stmt.else_body is None:
            inner = self._negated_inner(stmt.cond)
            if inner is not None:
                self._line("unless (" + inner + ") {")
                self.indent += 1
                self._emit_stmts(stmt.then_body)
                self.indent -= 1
                self._line("}")
                return
        self._line("if (" + self._expr(stmt.cond) + ") {")
        self.indent += 1
        self._emit_stmts(stmt.then_body)
        self.indent -= 1
        self._emit_else_body(stmt.else_body)
        self._line("}")

    def _negated_inner(self, cond: TExpr) -> str | None:
        if isinstance(cond, TUnaryOp) and cond.op == "!":
            return self._expr(cond.operand)
        return None

    def _truthiness_expr(self, cond: TExpr) -> str | None:
        if isinstance(cond, TBinaryOp):
            if (
                cond.op == ">"
                and isinstance(cond.right, TIntLit)
                and cond.right.value == 0
                and isinstance(cond.left, TCall)
                and isinstance(cond.left.func, TVar)
                and cond.left.func.name == "Len"
            ):
                inner = cond.left.args[0].value
                s = self._expr(inner)
                if self._is_list_expr(inner):
                    return "@{" + s + "}"
                if self._is_map_expr(inner) or self._is_set_expr(inner):
                    return "%{" + s + "}"
                return s
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
            self._line("} elsif (" + self._expr(elif_stmt.cond) + ") {")
            self.indent += 1
            self._emit_stmts(elif_stmt.then_body)
            self.indent -= 1
            self._emit_else_body(elif_stmt.else_body)
            return
        self._line("} else {")
        self.indent += 1
        self._emit_stmts(else_body)
        self.indent -= 1

    def _emit_for(self, stmt: TForStmt) -> None:
        if isinstance(stmt.iterable, TRange):
            self._emit_for_range(
                stmt.binding, stmt.iterable.args, stmt.body, stmt.annotations
            )
        else:
            self._emit_for_iter(
                stmt.binding, stmt.iterable, stmt.body, stmt.annotations
            )

    def _emit_for_range(
        self, binding: list[str], args: list[TExpr], body: list[TStmt], ann: Ann
    ) -> None:
        var_name = _restore_name(binding[0], ann) if binding else "_i"
        i = "$" + var_name
        if len(args) == 1:
            end_val = self._static_int(args[0])
            if end_val is not None:
                range_expr = "0 .. " + str(end_val - 1)
            else:
                range_expr = "0 .. " + self._expr(args[0]) + " - 1"
        elif len(args) == 2:
            start = self._expr(args[0])
            end_val = self._static_int(args[1])
            if end_val is not None:
                range_expr = start + " .. " + str(end_val - 1)
            else:
                range_expr = start + " .. " + self._expr(args[1]) + " - 1"
        else:
            step_val = self._static_int(args[2])
            if step_val is not None and step_val < 0:
                end_val = self._static_int(args[1])
                if end_val is not None:
                    range_expr = (
                        "reverse " + str(end_val + 1) + " .. " + self._expr(args[0])
                    )
                else:
                    range_expr = (
                        "reverse "
                        + self._expr(args[1])
                        + " + 1 .. "
                        + self._expr(args[0])
                    )
            else:
                start = self._expr(args[0])
                end = self._expr(args[1])
                step = self._expr(args[2])
                idx = self.tmp_counter
                self.tmp_counter += 1
                st = "$__start" + str(idx)
                en = "$__end" + str(idx)
                sp = "$__step" + str(idx)
                self._line("my " + st + " = " + start + ";")
                self._line("my " + en + " = " + end + ";")
                self._line("my " + sp + " = " + step + ";")
                cond = (
                    "("
                    + sp
                    + " >= 0 ? "
                    + i
                    + " < "
                    + en
                    + " : "
                    + i
                    + " > "
                    + en
                    + ")"
                )
                inc = i + " += " + sp
                self._line(
                    "for (my " + i + " = " + st + "; " + cond + "; " + inc + ") {"
                )
                self.indent += 1
                self._emit_stmts(body)
                self.indent -= 1
                self._line("}")
                return
        self._line("for my " + i + " (" + range_expr + ") {")
        self.indent += 1
        if binding:
            self.var_types[binding[0]] = TPrimitive(Pos(0, 0), "int")
        if len(binding) >= 2:
            self._line("my $" + _restore_name(binding[1], ann) + " = " + i + ";")
        self._emit_stmts(body)
        self.indent -= 1
        self._line("}")

    def _emit_for_iter(
        self, binding: list[str], iterable: TExpr, body: list[TStmt], ann: Ann
    ) -> None:
        it = self._expr(iterable)
        safe = self._deref_safe(it)
        if len(binding) == 1:
            name = "$" + _restore_name(binding[0], ann)
            iter_type = self._expr_type(iterable)
            if self._is_map_expr(iterable):
                if isinstance(iter_type, TMapType) and isinstance(
                    iter_type.key, TTupleType
                ):
                    tmp = self._tmp("__k")
                    self._line("for my " + tmp + " (sort keys %{" + safe + "}) {")
                    self.indent += 1
                    self._line("my " + name + ' = [split("\\0", ' + tmp + ")];")
                    self.var_types[binding[0]] = iter_type.key
                    self._emit_stmts(body)
                    self.indent -= 1
                    self._line("}")
                    return
                self._line("for my " + name + " (sort keys %{" + safe + "}) {")
                if isinstance(iter_type, TMapType):
                    self.var_types[binding[0]] = iter_type.key
            elif self._is_set_expr(iterable):
                self._line("for my " + name + " (sort keys %{" + safe + "}) {")
                if isinstance(iter_type, TSetType):
                    self.var_types[binding[0]] = iter_type.element
            elif self._is_string_expr(iterable):
                self._line("for my " + name + " (split(//, " + it + ")) {")
            elif self._is_bytes_expr(iterable):
                self._line("for my " + name + " (unpack('C*', " + it + ")) {")
            else:
                if iter_type is None and not self._is_list_expr(iterable):
                    self._line(
                        "for my "
                        + name
                        + " (ref("
                        + it
                        + ") eq 'ARRAY' ? @{"
                        + safe
                        + "} : split(//, "
                        + it
                        + ")) {"
                    )
                else:
                    self._line("for my " + name + " (@{" + safe + "}) {")
                if isinstance(iter_type, TListType):
                    self.var_types[binding[0]] = iter_type.element
            self.indent += 1
            self._emit_stmts(body)
            self.indent -= 1
            self._line("}")
            return
        if len(binding) == 2:
            key_var = "$" + _restore_name(binding[0], ann)
            val_var = "$" + _restore_name(binding[1], ann)
            if self._is_map_expr(iterable) or ann.get("for.items") == "true":
                self._line("for my " + key_var + " (sort keys %{" + safe + "}) {")
                self.indent += 1
                iter_type = self._expr_type(iterable)
                if isinstance(iter_type, TMapType):
                    self.var_types[binding[0]] = iter_type.key
                    self.var_types[binding[1]] = iter_type.value
                self._line("my " + val_var + " = " + it + "->{" + key_var + "};")
                self._emit_stmts(body)
                self.indent -= 1
                self._line("}")
                return
            if (
                ann.get("for.enumerate") == "true"
                or ann.get("iter_kind") == "enumerate"
            ):
                if isinstance(iterable, TVar):
                    src = it
                else:
                    src = self._tmp("__src")
                    self._line("my " + src + " = " + it + ";")
                if self._is_string_expr(iterable):
                    chars = self._tmp("__chars")
                    self._line("my " + chars + " = [split(//, " + src + ")];")
                    self._line("for my " + key_var + " (0 .. $#{" + chars + "}) {")
                    self.indent += 1
                    self._line("my " + val_var + " = " + chars + "->[" + key_var + "];")
                else:
                    self._line("for my " + key_var + " (0 .. $#{" + src + "}) {")
                    self.indent += 1
                    self._line("my " + val_var + " = " + src + "->[" + key_var + "];")
                self.var_types[binding[0]] = TPrimitive(Pos(0, 0), "int")
                self._emit_stmts(body)
                self.indent -= 1
                self._line("}")
                return
        if len(binding) == 2:
            item = self._tmp("__item")
            self._line("for my " + item + " (@{" + safe + "}) {")
            self.indent += 1
            self._line("my " + key_var + " = " + item + "->[0];")
            self._line("my " + val_var + " = " + item + "->[1];")
            self._emit_stmts(body)
            self.indent -= 1
            self._line("}")
            return
        item = self._tmp("__item")
        self._line("for my " + item + " (@{" + safe + "}) {")
        self.indent += 1
        for idx, b in enumerate(binding):
            self._line(
                "my $" + _restore_name(b, ann) + " = " + item + "->[" + str(idx) + "];"
            )
        self._emit_stmts(body)
        self.indent -= 1
        self._line("}")

    def _emit_try(self, stmt: TTryStmt) -> None:
        ok = self._tmp("__ok")
        err = self._tmp("__err")
        self._line("my " + ok + " = eval {")
        self.indent += 1
        self._emit_stmts(stmt.body)
        self._line("1;")
        self.indent -= 1
        self._line("};")
        self._line("if (!" + ok + ") {")
        self.indent += 1
        self._line("my " + err + " = $@;")
        if stmt.catches:
            self._emit_catches(stmt.catches, err)
        else:
            self._line("die " + err + ";")
        self.indent -= 1
        self._line("}")
        if stmt.finally_body is not None:
            self._emit_stmts(stmt.finally_body)

    def _emit_catches(self, catches: list[TCatch], err: str) -> None:
        has_chain = False
        has_default = False
        for catch in catches:
            cond = self._catch_condition(catch, err)
            if cond is None:
                if not has_chain:
                    unused = (
                        catch.annotations.get("liveness.catch_var_unused") == "true"
                    )
                    if not unused:
                        self._line(
                            "my $"
                            + _restore_name(catch.name, catch.annotations)
                            + " = "
                            + err
                            + ";"
                        )
                    self._emit_stmts(catch.body)
                    return
                self._line("} else {")
                self.indent += 1
                unused = catch.annotations.get("liveness.catch_var_unused") == "true"
                if not unused:
                    self._line(
                        "my $"
                        + _restore_name(catch.name, catch.annotations)
                        + " = "
                        + err
                        + ";"
                    )
                self._emit_stmts(catch.body)
                self.indent -= 1
                has_default = True
                break
            if not has_chain:
                self._line("if (" + cond + ") {")
                has_chain = True
            else:
                self._line("} elsif (" + cond + ") {")
            self.indent += 1
            unused = catch.annotations.get("liveness.catch_var_unused") == "true"
            if not unused:
                self._line(
                    "my $"
                    + _restore_name(catch.name, catch.annotations)
                    + " = "
                    + err
                    + ";"
                )
            self._emit_stmts(catch.body)
            self.indent -= 1
        if has_chain:
            if not has_default:
                self._line("} else {")
                self.indent += 1
                self._line("die " + err + ";")
                self.indent -= 1
            self._line("}")

    def _catch_condition(self, catch: TCatch, err: str) -> str | None:
        parts: list[str] = []
        for t in catch.types:
            if isinstance(t, TIdentType):
                parts.append("ref(" + err + ") eq '" + t.name + "'")
            else:
                return None
        if not parts:
            return None
        return " || ".join(parts)

    def _emit_match(self, stmt: TMatchStmt) -> None:
        expr = self._expr(stmt.expr)
        has_nil_case = any(isinstance(c.pattern, TPatternNil) for c in stmt.cases)
        first = True
        num_cases = len(stmt.cases)
        has_default = stmt.default is not None
        for idx, case in enumerate(stmt.cases):
            is_last = idx == num_cases - 1 and not has_default
            self._emit_match_case(case, expr, first, is_last, has_nil_case)
            first = False
        default = stmt.default
        if default is not None:
            self._emit_match_default(default, expr, first)
        elif not first:
            self._line("}")

    def _emit_match_case(
        self,
        case: TMatchCase,
        expr: str,
        first: bool,
        is_last: bool,
        has_nil_case: bool = False,
    ) -> None:
        pat = case.pattern
        if is_last and not first:
            self._line("} else {")
            self.indent += 1
            if isinstance(pat, TPatternType):
                unused = pat.annotations.get("liveness.match_var_unused") == "true"
                if not unused:
                    self._line("my $" + _safe_name(pat.name) + " = " + expr + ";")
            self._emit_stmts(case.body)
            self.indent -= 1
            return
        keyword = "if" if first else "} elsif"
        if isinstance(pat, TPatternType):
            cond = self._type_match_cond(pat.type_name, expr, has_nil_case)
            self._line(keyword + " (" + cond + ") {")
            self.indent += 1
            unused = pat.annotations.get("liveness.match_var_unused") == "true"
            if not unused:
                self._line("my $" + _safe_name(pat.name) + " = " + expr + ";")
            self._emit_stmts(case.body)
            self.indent -= 1
            return
        if isinstance(pat, TPatternEnum):
            self._line(
                keyword
                + " ("
                + expr
                + " == "
                + pat.enum_name
                + "::"
                + pat.variant
                + ") {"
            )
            self.indent += 1
            self._emit_stmts(case.body)
            self.indent -= 1
            return
        if isinstance(pat, TPatternNil):
            self._line(keyword + " (!defined(" + expr + ")) {")
            self.indent += 1
            self._emit_stmts(case.body)
            self.indent -= 1

    def _type_match_cond(self, typ: TType, expr: str, is_optional: bool = False) -> str:
        if isinstance(typ, TIdentType):
            if typ.name in ("dict", "Dict"):
                return "(ref(" + expr + ") eq 'HASH')"
            if typ.name in ("list", "List"):
                return "(ref(" + expr + ") eq 'ARRAY')"
            if typ.name in self.struct_names:
                return "eval { " + expr + "->isa('" + typ.name + "') }"
            return (
                "defined("
                + expr
                + ") && eval { "
                + expr
                + "->isa('"
                + typ.name
                + "') }"
            )
        if isinstance(typ, TPrimitive):
            if is_optional:
                return "defined(" + expr + ")"
            if typ.kind in ("int", "float"):
                return "looks_like_number(" + expr + ")"
            if typ.kind == "string":
                return "!ref(" + expr + ") && !looks_like_number(" + expr + ")"
            if typ.kind == "bool":
                return "!ref(" + expr + ")"
        return "defined(" + expr + ") && eval { " + expr + "->isa('UNSUPPORTED') }"

    def _emit_match_default(self, default: TDefault, expr: str, first: bool) -> None:
        if first:
            self._line("if (1) {")
        else:
            self._line("} else {")
        self.indent += 1
        if default.name is not None:
            unused = default.annotations.get("liveness.match_var_unused") == "true"
            if not unused:
                self._line("my $" + _safe_name(default.name) + " = " + expr + ";")
        self._emit_stmts(default.body)
        self.indent -= 1
        self._line("}")

    def _pattern_type_name(self, typ: TType) -> str:
        if isinstance(typ, TIdentType):
            return typ.name
        return "UNSUPPORTED"

    def _target(self, expr: TExpr) -> str:
        return self._expr(expr)

    def _expr(self, expr: TExpr) -> str:
        if isinstance(expr, TIntLit):
            return self._int_lit(expr)
        if isinstance(expr, TFloatLit):
            return expr.raw
        if isinstance(expr, TStringLit):
            return _string_literal(expr.value)
        if isinstance(expr, TRuneLit):
            return _string_literal(expr.value)
        if isinstance(expr, TBoolLit):
            return "1" if expr.value else "0"
        if isinstance(expr, TNilLit):
            return "undef"
        if isinstance(expr, TByteLit):
            return expr.raw
        if isinstance(expr, TBytesLit):
            return self._bytes_lit(expr)
        if isinstance(expr, TVar):
            if expr.name == self.self_name:
                return "$self"
            if expr.name in self.var_alias:
                return self.var_alias[expr.name]
            if expr.name in self.function_names and expr.name not in self.var_types:
                prefix = "main::" if self.in_package else ""
                return "\\&" + prefix + _restore_name(expr.name, expr.annotations)
            if expr.name in self.struct_names and expr.name not in self.var_types:
                return expr.name + "->new()"
            if expr.name in self.enum_names and expr.name not in self.var_types:
                return expr.name
            if expr.name in self._PYTHON_BUILTINS and expr.name not in self.var_types:
                return "(" + self._PYTHON_BUILTINS[expr.name] + ")"
            return "$" + _restore_name(expr.name, expr.annotations)
        if isinstance(expr, TFieldAccess):
            if isinstance(expr.obj, TVar) and expr.obj.name in self.enum_names:
                return expr.obj.name + "::" + expr.field
            return self._expr(expr.obj) + "->{" + _safe_name(expr.field) + "}"
        if isinstance(expr, TTupleAccess):
            obj = self._expr(expr.obj)
            idx = str(expr.index)
            obj_t = self._expr_type(expr.obj)
            if isinstance(obj_t, TTupleType):
                return (
                    "(ref("
                    + obj
                    + ") ? "
                    + obj
                    + "->["
                    + idx
                    + '] : (split("\\0", '
                    + obj
                    + "))["
                    + idx
                    + "])"
                )
            return obj + "->[" + idx + "]"
        if isinstance(expr, TIndex):
            if self._is_map_expr(expr.obj) or self._is_set_expr(expr.obj):
                return self._expr(expr.obj) + "->{" + self._hash_key(expr.index) + "}"
            if self._is_string_expr(expr.obj) or self._is_bytes_expr(expr.obj):
                idx = self._expr(expr.index)
                if expr.annotations.get("provenance") == "negative_index":
                    neg = self._negative_index(expr)
                    if neg is not None:
                        idx = neg
                return "substr(" + self._expr(expr.obj) + ", " + idx + ", 1)"
            if self._is_list_expr(expr.obj):
                idx2 = self._expr(expr.index)
                if expr.annotations.get("provenance") == "negative_index":
                    neg2 = self._negative_index(expr)
                    if neg2 is not None:
                        idx2 = neg2
                return self._expr(expr.obj) + "->[" + idx2 + "]"
            if self._is_int_expr(expr.index):
                obj_s = self._expr(expr.obj)
                idx2 = self._expr(expr.index)
                if expr.annotations.get("provenance") == "negative_index":
                    neg2 = self._negative_index(expr)
                    if neg2 is not None:
                        idx2 = neg2
                return (
                    "(ref("
                    + obj_s
                    + ") ? "
                    + obj_s
                    + "->["
                    + idx2
                    + "] : substr("
                    + obj_s
                    + ", "
                    + idx2
                    + ", 1))"
                )
            return self._expr(expr.obj) + "->{" + self._hash_key(expr.index) + "}"
        if isinstance(expr, TSlice):
            return self._slice(expr)
        if isinstance(expr, TBinaryOp):
            return self._binary(expr)
        if isinstance(expr, TUnaryOp):
            return self._unary(expr)
        if isinstance(expr, TTernary):
            if expr.annotations.get("provenance") == "none_coalesce":
                val = self._nil_coalesce_value(expr)
                if val is not None:
                    return val
            return (
                "("
                + self._expr(expr.cond)
                + " ? "
                + self._expr(expr.then_expr)
                + " : "
                + self._expr(expr.else_expr)
                + ")"
            )
        if isinstance(expr, TListLit):
            elems = ", ".join(self._expr(e) for e in expr.elements)
            return "[" + elems + "]"
        if isinstance(expr, TTupleLit):
            elems = ", ".join(self._expr(e) for e in expr.elements)
            return "[" + elems + "]"
        if isinstance(expr, TMapLit):
            if not expr.entries:
                return "{}"
            pairs = ", ".join(
                self._hash_key(k) + " => " + self._expr(v) for k, v in expr.entries
            )
            return "{ " + pairs + " }"
        if isinstance(expr, TSetLit):
            if not expr.elements:
                return "{}"
            elems = ", ".join(self._hash_key(e) for e in expr.elements)
            return "do { my $__s = {}; $__s->{$_} = 1 for (" + elems + "); $__s }"
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
        if len(expr.value) == 0:
            return '""'
        nums = ", ".join(str(b) for b in expr.value)
        return "pack('C*', " + nums + ")"

    def _slice(self, expr: TSlice) -> str:
        obj = self._expr(expr.obj)
        prov = expr.annotations.get("provenance", "")
        low = self._expr(expr.low)
        high = self._expr(expr.high)
        if prov == "open_start" and self._is_zero(expr.low):
            low = "0"
        if self._is_string_expr(expr.obj) or self._is_bytes_expr(expr.obj):
            if prov == "open_end" and self._is_len_call(expr.high):
                return "substr(" + obj + ", " + low + ")"
            if self._is_negative_literal(expr.high):
                return (
                    "substr("
                    + obj
                    + ", "
                    + low
                    + ", length("
                    + obj
                    + ") + "
                    + high
                    + " - ("
                    + low
                    + "))"
                )
            return "substr(" + obj + ", " + low + ", (" + high + ") - (" + low + "))"
        obj_type = self._expr_type(expr.obj)
        is_list = self._is_list_expr(expr.obj) or _is_list_type(obj_type)
        if not is_list:
            if prov == "open_end" and self._is_len_call(expr.high):
                return (
                    "(!ref("
                    + obj
                    + ") ? substr("
                    + obj
                    + ", "
                    + low
                    + ") : [ @{"
                    + obj
                    + "}["
                    + low
                    + " .. $#{"
                    + obj
                    + "}] ])"
                )
            return (
                "(!ref("
                + obj
                + ") ? substr("
                + obj
                + ", "
                + low
                + ", ("
                + high
                + ") - ("
                + low
                + ")) : [ @{"
                + obj
                + "}["
                + low
                + " .. ("
                + high
                + ") - 1] ])"
            )
        if prov == "open_end" and self._is_len_call(expr.high):
            safe = self._deref_safe(obj)
            return "[ @{" + safe + "}[" + low + " .. $#{" + safe + "}] ]"
        safe = self._deref_safe(obj)
        return "[ @{" + safe + "}[" + low + " .. (" + high + ") - 1] ]"

    def _negative_index(self, expr: TIndex) -> str | None:
        idx = expr.index
        if isinstance(idx, TBinaryOp) and idx.op == "-":
            if (
                isinstance(idx.left, TCall)
                and isinstance(idx.left.func, TVar)
                and idx.left.func.name == "Len"
            ):
                return "-" + self._expr(idx.right)
        return None

    def _binary(self, expr: TBinaryOp) -> str:
        op = expr.op
        if (
            op == "*"
            and isinstance(expr.left, TCall)
            and isinstance(expr.left.func, TVar)
            and expr.left.func.name == "Repeat"
            and self._is_string_expr(expr.left.args[0].value)
        ):
            s = self._a(expr.left.args, 0)
            repeat_n = self._a(expr.left.args, 1)
            right = self._expr(expr.right)
            return "(" + s + " x (" + repeat_n + " * " + right + "))"
        if self.strict_math and op in STRICT_INT_BINARY:
            if self._is_int_expr(expr.left) and self._is_int_expr(expr.right):
                fn = STRICT_INT_BINARY[op]
                return (
                    fn
                    + "("
                    + self._expr(expr.left)
                    + ", "
                    + self._expr(expr.right)
                    + ")"
                )
        if self.strict_math and op == "%":
            if self._is_float_expr(expr.left) or self._is_float_expr(expr.right):
                return (
                    "strict_fmod("
                    + self._expr(expr.left)
                    + ", "
                    + self._expr(expr.right)
                    + ")"
                )
        if op == "&&" and expr.annotations.get("provenance") == "chained_comparison":
            chained = self._chain_comparison(expr)
            if chained is not None:
                return chained
        if op == "==" and isinstance(expr.right, TNilLit):
            return "!defined(" + self._expr(expr.left) + ")"
        if op == "!=" and isinstance(expr.right, TNilLit):
            return "defined(" + self._expr(expr.left) + ")"
        if op == "==" and isinstance(expr.left, TNilLit):
            return "!defined(" + self._expr(expr.right) + ")"
        if op == "!=" and isinstance(expr.left, TNilLit):
            return "defined(" + self._expr(expr.right) + ")"
        perl_op = self._binary_op(op, expr.left, expr.right)
        left = self._maybe_paren(expr.left, perl_op, True)
        right = self._maybe_paren(expr.right, perl_op, False)
        return left + " " + perl_op + " " + right

    def _unary(self, expr: TUnaryOp) -> str:
        if self.strict_math and expr.op == "-" and self._is_int_expr(expr.operand):
            return "checked_neg_i64(" + self._expr(expr.operand) + ")"
        if expr.op == "!":
            if (
                isinstance(expr.operand, TCall)
                and isinstance(expr.operand.func, TVar)
                and expr.operand.func.name == "Contains"
            ):
                return "!(" + self._builtin_call("Contains", expr.operand.args) + ")"
            inner = self._expr(expr.operand)
            if isinstance(expr.operand, (TBinaryOp, TTernary)):
                return "!(" + inner + ")"
            return "!" + inner
        if isinstance(expr.operand, (TBinaryOp, TTernary, TUnaryOp)):
            return expr.op + "(" + self._expr(expr.operand) + ")"
        return expr.op + self._expr(expr.operand)

    def _chain_comparison(self, expr: TBinaryOp) -> str | None:
        left = expr.left
        right = expr.right
        if (
            isinstance(left, TBinaryOp)
            and isinstance(right, TBinaryOp)
            and left.op in _CMP_OPS
            and right.op in _CMP_OPS
        ):
            left_op = self._binary_op(left.op, left.left, left.right)
            right_op = self._binary_op(right.op, right.left, right.right)
            return (
                self._expr(left.left)
                + " "
                + left_op
                + " "
                + self._expr(left.right)
                + " && "
                + self._expr(right.left)
                + " "
                + right_op
                + " "
                + self._expr(right.right)
            )
        return None

    def _maybe_paren(self, expr: TExpr, parent_op: str, is_left: bool) -> str:
        if isinstance(expr, TBinaryOp):
            child_op = self._binary_op(expr.op, expr.left, expr.right)
            if _needs_parens(child_op, parent_op, is_left):
                return "(" + self._expr(expr) + ")"
        elif isinstance(expr, TTernary):
            return "(" + self._expr(expr) + ")"
        return self._expr(expr)

    def _is_numeric_expr(self, expr: TExpr) -> bool:
        if isinstance(expr, (TIntLit, TFloatLit, TBoolLit)):
            return True
        if isinstance(expr, TBinaryOp) and expr.op in (
            "+",
            "-",
            "*",
            "/",
            "//",
            "%",
            "**",
            "&",
            "|",
            "^",
            "<<",
            ">>",
        ):
            return True
        if isinstance(expr, TUnaryOp) and expr.op in ("-", "~"):
            return True
        if isinstance(expr, TFieldAccess):
            if isinstance(expr.obj, TVar) and expr.obj.name in self.enum_names:
                return True
        if isinstance(expr, TVar) and expr.name in self.enum_names:
            return True
        typ = self._expr_type(expr)
        return isinstance(typ, TPrimitive) and typ.kind in ("int", "float", "bool")

    def _binary_op(self, op: str, left: TExpr, right: TExpr | None = None) -> str:
        is_str = self._is_string_expr(left) or (
            right is not None and self._is_string_expr(right)
        )
        is_num = self._is_numeric_expr(left) or (
            right is not None and self._is_numeric_expr(right)
        )
        if op in ("and", "&&"):
            return "&&"
        if op in ("or", "||"):
            return "||"
        if op in ("==", "!=", "<", ">", "<=", ">=") and is_str and not is_num:
            return {
                "==": "eq",
                "!=": "ne",
                "<": "lt",
                ">": "gt",
                "<=": "le",
                ">=": "ge",
            }[op]
        if op in ("==", "!=") and not is_num:
            return "eq" if op == "==" else "ne"
        if op == "+" and is_str:
            return "."
        return op

    def _fn_lit(self, expr: TFnLit) -> str:
        params = ", ".join(
            "$" + _restore_name(p.name, p.annotations)
            for p in expr.params
            if p.typ is not None
        )
        first = expr.body[0] if expr.body else None
        if expr.annotations.get("fn_lit.arrow") == "true" and isinstance(
            first, TExprStmt
        ):
            if params:
                return (
                    "sub { my ("
                    + params
                    + ") = @_; return "
                    + self._expr(first.expr)
                    + "; }"
                )
            return "sub { return " + self._expr(first.expr) + "; }"
        return self._fn_lit_block(expr.body, params)

    def _fn_lit_block(self, stmts: list[TStmt], params: str) -> str:
        pad = "    " * (self.indent + 1)
        lines: list[str] = []
        lines.append("sub {")
        if params:
            lines.append(pad + "my (" + params + ") = @_;")
        old_lines = self.lines
        old_indent = self.indent
        self.lines = []
        self.indent += 1
        self._emit_stmts(stmts)
        for ln in self.lines:
            lines.append(ln)
        self.lines = old_lines
        self.indent = old_indent
        lines.append("    " * self.indent + "}")
        return "\n".join(lines)

    def _stmt_inline(self, stmt: TStmt) -> str | None:
        if isinstance(stmt, TExprStmt):
            return self._expr(stmt.expr) + ";"
        if isinstance(stmt, TReturnStmt):
            if stmt.value is None:
                return "return;"
            return "return " + self._expr(stmt.value) + ";"
        if isinstance(stmt, TLetStmt):
            n = "$" + _restore_name(stmt.name, stmt.annotations)
            if stmt.value is not None:
                return "my " + n + " = " + self._expr(stmt.value) + ";"
            return "my " + n + " = " + self._zero_value(stmt.typ) + ";"
        if isinstance(stmt, TAssignStmt):
            return self._target(stmt.target) + " = " + self._expr(stmt.value) + ";"
        return None

    _PYTHON_BUILTINS: dict[str, str] = {
        "oct": "sub { sprintf('0o%o', $_[0]) }",
        "bin": "sub { sprintf('0b%b', $_[0]) }",
        "hex": "sub { sprintf('0x%x', $_[0]) }",
        "reversed": "sub { [reverse @{$_[0]}] }",
        "bytes_": 'sub { "\\0" x $_[0] }',
    }
    _PYTHON_CALL_MAP: dict[str, str] = {
        "bytes_": "bytes",
        "reversed": "reversed",
    }

    def _python_builtin_call(self, name: str, args: list[TArg]) -> str:
        orig = self._PYTHON_CALL_MAP[name]
        a = self._expr(args[0].value)
        if orig == "bytes":
            return '("\\0" x ' + a + ")"
        if orig == "reversed":
            return "[reverse @{" + a + "}]"
        raise ValueError(name)

    def _call(self, expr: TCall) -> str:
        func = expr.func
        args = expr.args
        if isinstance(func, TVar) and func.name in BUILTIN_NAMES:
            return self._builtin_call(func.name, args, expr.annotations)
        if isinstance(func, TVar) and func.name in self._PYTHON_CALL_MAP:
            return self._python_builtin_call(func.name, args)
        if isinstance(func, TVar) and func.name in self.struct_names:
            return self._struct_call(func.name, args)
        if isinstance(func, TFieldAccess):
            return self._method_call(func, args)
        if isinstance(func, TVar):
            vtyp = self.var_types.get(func.name)
            arg_strs = ", ".join(self._expr(a.value) for a in args)
            if isinstance(vtyp, TFuncType):
                return self._expr(func) + "->(" + arg_strs + ")"
            if func.name in self.function_names:
                prefix = "main::" if self.in_package else ""
                return prefix + _safe_name(func.name) + "(" + arg_strs + ")"
        fn_expr = self._expr(func)
        arg_strs2 = ", ".join(self._expr(a.value) for a in args)
        return fn_expr + "->(" + arg_strs2 + ")"

    def _struct_call(self, name: str, args: list[TArg]) -> str:
        ordered = self.struct_fields.get(name, [])
        if not args:
            return name + "->new()"
        has_named = any(a.name is not None for a in args)
        if not has_named or not ordered:
            vals = ", ".join(self._expr(a.value) for a in args)
            return name + "->new(" + vals + ")"
        named: dict[str, str] = {}
        for a in args:
            if a.name is not None:
                named[a.name] = self._expr(a.value)
        vals2: list[str] = []
        for fld in ordered:
            vals2.append(named.get(fld, "undef"))
        return name + "->new(" + ", ".join(vals2) + ")"

    def _method_call(self, func: TFieldAccess, args: list[TArg]) -> str:
        method = func.field
        if method == "decode" and not self._is_known_struct_method(func.obj, method):
            return self._expr(func.obj)
        if method == "clear" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            safe = self._deref_safe(obj)
            if self._is_map_expr(func.obj) or self._is_set_expr(func.obj):
                return "do { %{" + safe + "} = () }"
            return "do { @{" + safe + "} = () }"
        if method == "get" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            key = self._hash_key(args[0].value)
            if len(args) >= 2:
                default = self._expr(args[1].value)
                return "(" + obj + "->{" + key + "} // " + default + ")"
            return obj + "->{" + key + "}"
        if method == "append" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            val = self._expr(args[0].value)
            return "push(@{" + obj + "}, " + val + ")"
        if method == "keys" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            return "[sort keys %{" + obj + "}]"
        if method == "values" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            return "[values %{" + obj + "}]"
        if method == "items" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            return "[map { [$_, " + obj + "->{$_}] } sort keys %{" + obj + "}]"
        if method == "update" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            src = self._expr(args[0].value)
            return (
                "do { my $__src = "
                + src
                + "; "
                + obj
                + "->{$_} = $__src->{$_} for sort keys %{$__src} }"
            )
        if method == "setdefault" and not self._is_known_struct_method(
            func.obj, method
        ):
            obj = self._expr(func.obj)
            key = self._hash_key(args[0].value)
            default = self._expr(args[1].value) if len(args) >= 2 else "undef"
            return "(" + obj + "->{" + key + "} //= " + default + ")"
        if method == "pop" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            if self._is_map_expr(func.obj):
                key = self._hash_key(args[0].value)
                return "delete " + obj + "->{" + key + "}"
            if len(args) > 0:
                return "splice(@{" + obj + "}, " + self._expr(args[0].value) + ", 1)"
            return "pop(@{" + obj + "})"
        if method == "copy" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            if self._is_list_expr(func.obj):
                return "[@{" + obj + "}]"
            return "{%{" + obj + "}}"
        if method == "replace" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            old = self._expr(args[0].value)
            new = self._expr(args[1].value)
            if len(args) == 3:
                cnt = self._expr(args[2].value)
                return (
                    "do { my $__s = "
                    + obj
                    + "; my $__o = "
                    + old
                    + "; my $__n = "
                    + new
                    + "; my $__c = "
                    + cnt
                    + "; while ($__c > 0 && $__s =~ s/\\Q$__o\\E/$__n/) { $__c-- } $__s }"
                )
            return (
                "do { my $__s = "
                + obj
                + "; my $__o = "
                + old
                + "; my $__n = "
                + new
                + "; $__s =~ s/\\Q$__o\\E/$__n/g; $__s }"
            )
        if method == "index" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            sub_str = self._expr(args[0].value)
            return "index(" + obj + ", " + sub_str + ")"
        if method == "find" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            sub_str = self._expr(args[0].value)
            return "index(" + obj + ", " + sub_str + ")"
        obj = self._expr(func.obj)
        safe_method = _safe_name(method)
        arg_strs = ", ".join(self._expr(a.value) for a in args)
        return obj + "->" + safe_method + "(" + arg_strs + ")"

    def _is_known_struct_method(self, obj: TExpr, method: str) -> bool:
        if isinstance(obj, TVar):
            typ = self.var_types.get(obj.name)
            if isinstance(typ, TIdentType):
                return True
            if isinstance(typ, TOptionalType) and isinstance(typ.inner, TIdentType):
                return True
        return False

    def _builtin_call(self, name: str, args: list[TArg], ann: Ann | None = None) -> str:
        if name == "FloorDiv":
            return "POSIX::floor(" + self._a(args, 0) + " / " + self._a(args, 1) + ")"
        if name == "PythonMod":
            a = self._a(args, 0)
            b = self._a(args, 1)
            return "((" + a + " % " + b + ") + " + b + ") % " + b
        if name == "Append":
            return "push(@{" + self._a(args, 0) + "}, " + self._a(args, 1) + ")"
        if name == "Insert":
            return (
                "splice(@{"
                + self._a(args, 0)
                + "}, "
                + self._a(args, 1)
                + ", 0, "
                + self._a(args, 2)
                + ")"
            )
        if name == "Pop":
            return "pop(@{" + self._a(args, 0) + "})"
        if name == "RemoveAt":
            return "splice(@{" + self._a(args, 0) + "}, " + self._a(args, 1) + ", 1)"
        if name == "ReplaceSlice":
            return (
                "splice(@{"
                + self._a(args, 0)
                + "}, "
                + self._a(args, 1)
                + ", "
                + self._a(args, 2)
                + " - "
                + self._a(args, 1)
                + ", @{"
                + self._a(args, 3)
                + "})"
            )
        if name == "IndexOf":
            arr = self._a(args, 0)
            val = self._a(args, 1)
            pad = "    " * (self.indent + 1)
            return (
                "do {\n"
                + pad
                + "my $__i = 0;\n"
                + pad
                + "my $__r = -1;\n"
                + pad
                + "for my $v (@{"
                + arr
                + "}) { if ($v "
                + ("eq" if self._is_string_expr(args[1].value) else "==")
                + " "
                + val
                + ") { $__r = $__i; last; } $__i += 1; }\n"
                + pad
                + "$__r;\n"
                + "    " * self.indent
                + "}"
            )
        if name == "Upper":
            return "uc(" + self._a(args, 0) + ")"
        if name == "Lower":
            return "lc(" + self._a(args, 0) + ")"
        if name == "Trim":
            s = self._a(args, 0)
            chars = args[1].value
            if isinstance(chars, TStringLit):
                if chars.value == " \t\n\r":
                    return "do { my $__t = " + s + "; $__t =~ s/^\\s+|\\s+$//g; $__t }"
                raw = _escape_regex_charclass(chars.value)
                return (
                    "do { my $__t = "
                    + s
                    + "; $__t =~ s/^["
                    + raw
                    + "]+|["
                    + raw
                    + "]+$//g; $__t }"
                )
            c = self._a(args, 1)
            return (
                "do { my $__t = "
                + s
                + "; $__t =~ s/^["
                + c
                + "]+|["
                + c
                + "]+$//g; $__t }"
            )
        if name == "TrimStart":
            s2 = self._a(args, 0)
            chars2_expr: TExpr = args[1].value
            if isinstance(chars2_expr, TStringLit):
                if chars2_expr.value == " \t\n\r":
                    return "do { my $__t = " + s2 + "; $__t =~ s/^\\s+//; $__t }"
                raw2 = _escape_regex_charclass(chars2_expr.value)
                return "do { my $__t = " + s2 + "; $__t =~ s/^[" + raw2 + "]+//; $__t }"
            c2 = self._a(args, 1)
            return "do { my $__t = " + s2 + "; $__t =~ s/^[" + c2 + "]+//; $__t }"
        if name == "TrimEnd":
            s3 = self._a(args, 0)
            chars3_expr: TExpr = args[1].value
            if isinstance(chars3_expr, TStringLit):
                if chars3_expr.value == " \t\n\r":
                    return "do { my $__t = " + s3 + "; $__t =~ s/\\s+$//; $__t }"
                raw3 = _escape_regex_charclass(chars3_expr.value)
                return "do { my $__t = " + s3 + "; $__t =~ s/[" + raw3 + "]+$//; $__t }"
            c3 = self._a(args, 1)
            return "do { my $__t = " + s3 + "; $__t =~ s/[" + c3 + "]+$//; $__t }"
        if name == "Split":
            return (
                "do { my $__s = "
                + self._a(args, 0)
                + "; my $__sep = "
                + self._a(args, 1)
                + "; [split(/\\Q$__sep\\E/, $__s)] }"
            )
        if name == "SplitN":
            splitn_val: TExpr = args[1].value
            if isinstance(splitn_val, TStringLit):
                pat = _escape_perl_regex(splitn_val.value)
                return (
                    "[split(/"
                    + pat
                    + "/, "
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 2)
                    + ")]"
                )
            return (
                "do { my $__s = "
                + self._a(args, 0)
                + "; my $__sep = "
                + self._a(args, 1)
                + "; [split(/\\Q$__sep\\E/, $__s, "
                + self._a(args, 2)
                + ")] }"
            )
        if name == "SplitWhitespace":
            return "[grep { length($_) > 0 } split(/\\s+/, " + self._a(args, 0) + ")]"
        if name == "Join":
            return "join(" + self._a(args, 0) + ", @{" + self._a(args, 1) + "})"
        if name == "Find":
            return "index(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "RFind":
            return "rindex(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Count":
            count_arg0: TExpr = args[0].value
            count_arg1: TExpr = args[1].value
            if self._is_string_expr(count_arg0) or self._is_bytes_expr(count_arg0):
                if isinstance(count_arg1, TStringLit):
                    pat = _escape_perl_regex(count_arg1.value)
                    return "() = " + self._a(args, 0) + " =~ /" + pat + "/g"
                return (
                    "do { my $__s = "
                    + self._a(args, 0)
                    + "; my $__n = "
                    + self._a(args, 1)
                    + "; my $__c = () = $__s =~ /\\Q$__n\\E/g; $__c }"
                )
            a0 = self._a(args, 0)
            a1 = self._a(args, 1)
            cmp_op = "eq" if self._is_string_expr(count_arg1) else "=="
            if isinstance(count_arg1, TStringLit):
                pat = _escape_perl_regex(count_arg1.value)
                str_count = "do { my $__c = () = " + a0 + " =~ /" + pat + "/g; $__c }"
            else:
                str_count = (
                    "do { my $__n = "
                    + a1
                    + "; my $__c = () = "
                    + a0
                    + " =~ /\\Q$__n\\E/g; $__c }"
                )
            return (
                "(!ref("
                + a0
                + ") ? "
                + str_count
                + " : "
                + "scalar(grep { $_ "
                + cmp_op
                + " "
                + a1
                + " } @{"
                + a0
                + "}))"
            )
        if name == "Replace":
            repl_arg1: TExpr = args[1].value
            repl_arg2: TExpr = args[2].value
            if isinstance(repl_arg1, TStringLit) and isinstance(repl_arg2, TStringLit):
                old_lit = _escape_perl_regex(repl_arg1.value)
                new_lit = _escape_perl_replacement(repl_arg2.value)
                return (
                    "do { my $__s = "
                    + self._a(args, 0)
                    + "; $__s =~ s/"
                    + old_lit
                    + "/"
                    + new_lit
                    + "/g; $__s }"
                )
            return (
                "do { my $__s = "
                + self._a(args, 0)
                + "; my $__o = "
                + self._a(args, 1)
                + "; my $__n = "
                + self._a(args, 2)
                + "; $__s =~ s/\\Q$__o\\E/$__n/g; $__s }"
            )
        if name == "ReplaceCount":
            rcount_arg1: TExpr = args[1].value
            rcount_arg2: TExpr = args[2].value
            if isinstance(rcount_arg1, TStringLit) and isinstance(
                rcount_arg2, TStringLit
            ):
                old_lit = _escape_perl_regex(rcount_arg1.value)
                new_lit = _escape_perl_replacement(rcount_arg2.value)
                return (
                    "do { my $__s = "
                    + self._a(args, 0)
                    + "; my $__c = "
                    + self._a(args, 3)
                    + "; while ($__c > 0 && $__s =~ s/"
                    + old_lit
                    + "/"
                    + new_lit
                    + "/) { $__c-- } $__s }"
                )
            return (
                "do { my $__s = "
                + self._a(args, 0)
                + "; my $__o = "
                + self._a(args, 1)
                + "; my $__n = "
                + self._a(args, 2)
                + "; my $__c = "
                + self._a(args, 3)
                + "; while ($__c > 0 && $__s =~ s/\\Q$__o\\E/$__n/) { $__c-- } $__s }"
            )
        if name == "Reverse":
            return "scalar(reverse(" + self._a(args, 0) + "))"
        if name == "StartsWith":
            s = self._a(args, 0)
            pfx: TExpr = args[1].value
            if isinstance(pfx, TStringLit):
                pat = _escape_perl_regex(pfx.value)
                return "((" + s + " =~ /^" + pat + "/) ? 1 : 0)"
            return "((" + s + " =~ /^\\Q${\\ " + self._a(args, 1) + "}\\E/) ? 1 : 0)"
        if name == "EndsWith":
            s = self._a(args, 0)
            sfx: TExpr = args[1].value
            if isinstance(sfx, TStringLit):
                pat = _escape_perl_regex(sfx.value)
                return "((" + s + " =~ /" + pat + "$/) ? 1 : 0)"
            return "((" + s + " =~ /\\Q${\\ " + self._a(args, 1) + "}\\E$/) ? 1 : 0)"
        if name == "IsDigit":
            return "(" + self._a(args, 0) + " =~ /^\\d+$/ ? 1 : 0)"
        if name == "IsAlpha":
            return "(" + self._a(args, 0) + " =~ /^[A-Za-z]+$/ ? 1 : 0)"
        if name == "IsAlnum":
            return "(" + self._a(args, 0) + " =~ /^[A-Za-z0-9]+$/ ? 1 : 0)"
        if name == "IsSpace":
            return "(" + self._a(args, 0) + " =~ /^\\s+$/ ? 1 : 0)"
        if name == "IsUpper":
            return (
                "("
                + self._a(args, 0)
                + " =~ /[A-Z]/ && "
                + self._a(args, 0)
                + " !~ /[a-z]/ ? 1 : 0)"
            )
        if name == "IsLower":
            return (
                "("
                + self._a(args, 0)
                + " =~ /[a-z]/ && "
                + self._a(args, 0)
                + " !~ /[A-Z]/ ? 1 : 0)"
            )
        if name == "Encode":
            return "encode('UTF-8', " + self._a(args, 0) + ")"
        if name == "Decode":
            a = self._a(args, 0)
            return (
                "do { my $__d = eval { decode('UTF-8', "
                + a
                + ", Encode::FB_CROAK) }; if ($@) { die bless({message => \"$@\"}, 'ValueError') } $__d }"
            )
        if name == "Bytes":
            return '("\\0" x ' + self._a(args, 0) + ")"
        if name == "BytesFrom":
            return "pack('C*', @{" + self._a(args, 0) + "})"
        if name == "Add":
            return self._a(args, 0) + "->{" + self._hash_key(args[1].value) + "} = 1"
        if name == "Remove":
            return (
                "delete "
                + self._a(args, 0)
                + "->{"
                + self._hash_key(args[1].value)
                + "}"
            )
        if name == "Union":
            a = self._deref_safe(self._a(args, 0))
            b = self._deref_safe(self._a(args, 1))
            return "do { my $__s = {%{" + a + "}, %{" + b + "}}; $__s }"
        if name == "Intersection":
            a = self._a(args, 0)
            b = self._a(args, 1)
            return (
                "do { my $__a = "
                + a
                + "; my $__b = "
                + b
                + "; my $__s = {}; for (sort keys %{$__a}) { $__s->{$_} = 1 if exists $__b->{$_} } $__s }"
            )
        if name == "Difference":
            a = self._a(args, 0)
            b = self._a(args, 1)
            return (
                "do { my $__a = "
                + a
                + "; my $__b = "
                + b
                + "; my $__s = {}; for (sort keys %{$__a}) { $__s->{$_} = 1 unless exists $__b->{$_} } $__s }"
            )
        if name == "Get":
            k = self._hash_key(args[1].value)
            if len(args) == 3:
                if ann is not None and ann.get("provenance") == "dict_get_default":
                    return (
                        "("
                        + self._a(args, 0)
                        + "->{"
                        + k
                        + "} // "
                        + self._a(args, 2)
                        + ")"
                    )
                return (
                    "(exists "
                    + self._a(args, 0)
                    + "->{"
                    + k
                    + "} ? "
                    + self._a(args, 0)
                    + "->{"
                    + k
                    + "} : "
                    + self._a(args, 2)
                    + ")"
                )
            return self._a(args, 0) + "->{" + k + "}"
        if name == "Delete":
            return (
                "delete "
                + self._a(args, 0)
                + "->{"
                + self._hash_key(args[1].value)
                + "}"
            )
        if name == "Merge":
            a0 = self._deref_safe(self._a(args, 0))
            a1 = self._deref_safe(self._a(args, 1))
            return "{ %{" + a0 + "}, %{" + a1 + "} }"
        if name == "Keys":
            return "[sort keys %{" + self._a(args, 0) + "}]"
        if name == "Values":
            return "[values %{" + self._a(args, 0) + "}]"
        if name == "Items":
            return (
                "do { my $__m = "
                + self._a(args, 0)
                + "; [map { [$_, $__m->{$_}] } sort keys %{$__m}] }"
            )
        if name == "Len":
            return self._len_call(args[0].value)
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
                return "min(@{" + self._a(args, 0) + "})"
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
                return "max(@{" + self._a(args, 0) + "})"
            return "max(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Sum":
            if self._is_set_expr(args[0].value):
                return "(sum(keys %{" + self._deref_safe(self._a(args, 0)) + "}) // 0)"
            return "(sum(@{" + self._a(args, 0) + "}) // 0)"
        if name == "Round":
            if len(args) == 2:
                return (
                    'sprintf("%." . '
                    + self._a(args, 1)
                    + ' . "f", '
                    + self._a(args, 0)
                    + ") + 0"
                )
            return (
                "int("
                + self._a(args, 0)
                + " + ("
                + self._a(args, 0)
                + " >= 0 ? 0.5 : -0.5))"
            )
        if name == "Floor":
            return "floor(" + self._a(args, 0) + ")"
        if name == "Ceil":
            return "ceil(" + self._a(args, 0) + ")"
        if name == "DivMod":
            return (
                "[int("
                + self._a(args, 0)
                + " / "
                + self._a(args, 1)
                + "), "
                + self._a(args, 0)
                + " % "
                + self._a(args, 1)
                + "]"
            )
        if name == "Sorted":
            sorted_arg = args[0].value
            if self.strict_math and self._is_float_list(sorted_arg):
                return "strict_sorted_f64(" + self._a(args, 0) + ")"
            a = self._a(args, 0)
            if self._is_set_expr(sorted_arg):
                return "[sort keys %{" + a + "}]"
            typ = self._expr_type(sorted_arg)
            if isinstance(typ, TListType) and _is_string_type(typ.element):
                return "[sort @{" + a + "}]"
            return "[sort { $a <=> $b } @{" + a + "}]"
        if name == "ListFrom":
            a = self._a(args, 0)
            if self._is_set_expr(args[0].value):
                return "[sort keys %{" + self._deref_safe(a) + "}]"
            return "[@{" + self._deref_safe(a) + "}]"
        if name == "Reversed":
            a = self._a(args, 0)
            if self._is_set_expr(args[0].value):
                return "[reverse keys %{" + a + "}]"
            return "[reverse @{" + a + "}]"
        if name == "Map":
            if len(args) == 0:
                return "{}"
            return (
                "[map { " + self._a(args, 0) + "->($_) } @{" + self._a(args, 1) + "}]"
            )
        if name == "Set":
            if len(args) == 0:
                return "{}"
            return (
                "do { my $__s = {}; $__s->{$_} = 1 for @{"
                + self._deref_safe(self._a(args, 0))
                + "}; $__s }"
            )
        if name == "SetFromList":
            if isinstance(args[0].value, TSetLit):
                return self._a(args, 0)
            a = self._deref_safe(self._a(args, 0))
            if self._is_set_expr(args[0].value):
                return (
                    "do { my $__s = {}; $__s->{$_} = 1 for sort keys %{"
                    + a
                    + "}; $__s }"
                )
            return "do { my $__s = {}; $__s->{$_} = 1 for @{" + a + "}; $__s }"
        if name == "ToString":
            inner_expr = args[0].value
            inner = self._expr(inner_expr)
            if self._needs_concat_parens(inner_expr):
                inner = "(" + inner + ")"
            return '("" . ' + inner + ")"
        if name == "ParseInt":
            s = self._a(args, 0)
            base = self._a(args, 1)
            return (
                "do { my $__s = "
                + s
                + "; my $__b = "
                + base
                + "; $__b == 0 ? ($__s =~ /^0[xX]/ ? hex($__s) : $__s =~ /^0[oO]/ ? oct($__s) : $__s =~ /^0[bB]/ ? oct($__s) : int($__s)) : $__b == 10 ? int($__s) : $__b == 16 ? hex($__s) : $__b == 8 ? oct($__s) : $__b == 2 ? oct('0b' . $__s) : int($__s) }"
            )
        if name == "ParseFloat":
            return "(" + self._a(args, 0) + " + 0.0)"
        if name == "FormatInt":
            return self._format_int(args)
        if name == "RuneFromInt":
            return "chr(" + self._a(args, 0) + ")"
        if name == "RuneToInt":
            return "ord(" + self._a(args, 0) + ")"
        if name == "IntToFloat":
            itf_v: TExpr = args[0].value
            if isinstance(itf_v, TIntLit):
                return str(itf_v.value) + ".0"
            return "(" + self._a(args, 0) + " + 0.0)"
        if name == "FloatToInt":
            return "int(" + self._a(args, 0) + ")"
        if name == "ByteToInt":
            return "ord(" + self._a(args, 0) + ")"
        if name == "IntToByte":
            return "chr(" + self._a(args, 0) + ")"
        if name == "Unwrap":
            return self._a(args, 0)
        if name == "IsNil":
            return "!defined(" + self._a(args, 0) + ")"
        if name == "Sqrt":
            return "sqrt(" + self._a(args, 0) + ")"
        if name == "IsNaN":
            v = self._a(args, 0)
            return "((" + v + " != " + v + ") ? 1 : 0)"
        if name == "IsInf":
            v2 = self._a(args, 0)
            return "(POSIX::isinf(" + v2 + ") ? 1 : 0)"
        if name == "WriteOut":
            return "print(" + self._a(args, 0) + ")"
        if name == "WriteErr":
            return "print STDERR " + self._a(args, 0)
        if name == "WritelnOut":
            return "say(" + self._a(args, 0) + ")"
        if name == "WritelnErr":
            return "say STDERR " + self._a(args, 0)
        if name == "ReadLine":
            return "do { my $__l = scalar(<STDIN>); defined($__l) ? decode('UTF-8', $__l, Encode::FB_CROAK) : $__l }"
        if name == "ReadAll":
            return "do { local $/; decode('UTF-8', scalar(<STDIN>), Encode::FB_CROAK) }"
        if name == "ReadBytes":
            return "do { local $/; scalar(<STDIN>) }"
        if name == "ReadBytesN":
            return "do { read(STDIN, my $__b, " + self._a(args, 0) + "); $__b }"
        if name == "ReadFile":
            return (
                "do { my $__p = "
                + self._a(args, 0)
                + "; open(my $__fh, '<:encoding(UTF-8)', $__p) or die $__p; local $/; my $__d = <$__fh>; close($__fh); $__d }"
            )
        if name == "ReadFileBytes":
            return (
                "do { my $__p = "
                + self._a(args, 0)
                + "; open(my $__fh, '<:raw', $__p) or die $__p; local $/; my $__d = <$__fh>; close($__fh); $__d }"
            )
        if name == "WriteFile":
            return (
                "do { my $__p = "
                + self._a(args, 0)
                + "; my $__d = "
                + self._a(args, 1)
                + "; open(my $__fh, '>', $__p) or die $__p; print $__fh $__d; close($__fh) }"
            )
        if name == "Args":
            return "[@ARGV]"
        if name == "GetEnv":
            return "$ENV{" + self._a(args, 0) + "}"
        if name == "Exit":
            return "exit(" + self._a(args, 0) + ")"
        if name == "Pow":
            if self.strict_math and self._is_int_expr(args[0].value):
                return (
                    "checked_pow_i64("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ")"
                )
            return "(" + self._a(args, 0) + " ** " + self._a(args, 1) + ")"
        if name == "Contains":
            return self._contains_expr(args[0].value, args[1].value)
        if name == "Concat":
            left = args[0].value
            right = args[1].value
            if self._is_list_expr(left) and self._is_list_expr(right):
                return "[ @{" + self._expr(left) + "}, @{" + self._expr(right) + "} ]"
            return "(" + self._expr(left) + " . " + self._expr(right) + ")"
        if name == "Repeat":
            count = self._a(args, 1)
            if isinstance(args[1].value, (TBinaryOp, TTernary)):
                count = "(" + count + ")"
            if self._is_list_expr(args[0].value):
                elem = args[0].value
                if isinstance(elem, TListLit) and len(elem.elements) == 1:
                    inner = self._expr(elem.elements[0])
                else:
                    inner = "@{" + self._a(args, 0) + "}"
                return "[(" + inner + ") x " + count + "]"
            return "(" + self._a(args, 0) + " x " + count + ")"
        if name == "Format":
            if ann is not None and ann.get("provenance") == "f_string":
                return self._format_interpolated(args)
            return self._format_call(args)
        if name == "IsType":
            type_arg = args[1].value
            if isinstance(type_arg, TStringLit):
                type_name = type_arg.value
            else:
                type_name = self._expr(type_arg)
            a0 = self._a(args, 0)
            if type_name in ("dict", "Dict"):
                return "(ref(" + a0 + ") eq 'HASH')"
            if type_name in ("list", "List"):
                return "(ref(" + a0 + ") eq 'ARRAY')"
            if type_name in ("str", "Str"):
                return "(!ref(" + a0 + ") && !looks_like_number(" + a0 + "))"
            if type_name in ("int", "Int", "float", "Float"):
                return "looks_like_number(" + a0 + ")"
            if type_name in ("bool", "Bool"):
                return "!ref(" + a0 + ")"
            return "eval { " + a0 + "->isa('" + type_name + "') }"
        if name == "Assert":
            cond = self._a(args, 0)
            if len(args) > 1:
                return "do { die " + self._a(args, 1) + " unless (" + cond + "); 1 }"
            return "do { die 'AssertError' unless (" + cond + "); 1 }"
        arg_strs = ", ".join(self._expr(ar.value) for ar in args)
        return _safe_name(name) + "(" + arg_strs + ")"

    def _contains_expr(self, container: TExpr, needle: TExpr) -> str:
        c = self._expr(container)
        n = self._expr(needle)
        if self._is_string_expr(container) or self._is_bytes_expr(container):
            return "index(" + c + ", " + n + ") >= 0"
        if self._is_map_expr(container) or self._is_set_expr(container):
            return "exists(" + c + "->{" + self._hash_key(needle) + "})"
        if self._is_list_expr(container):
            cmp_op = self._list_elem_cmp(container, needle)
            return "grep { $_ " + cmp_op + " " + n + " } @{" + c + "}"
        cmp_op = self._list_elem_cmp(container, needle)
        nk = self._hash_key(needle)
        return (
            "(!ref("
            + c
            + ") ? index("
            + c
            + ", "
            + n
            + ") >= 0 : ref("
            + c
            + ") eq 'HASH' ? exists("
            + c
            + "->{"
            + nk
            + "}) : grep { $_ "
            + cmp_op
            + " "
            + n
            + " } @{"
            + c
            + "})"
        )

    def _list_elem_cmp(self, container: TExpr, needle: TExpr) -> str:
        if self._is_string_expr(needle):
            return "eq"
        typ = self._expr_type(container)
        if isinstance(typ, TListType) and _is_string_type(typ.element):
            return "eq"
        nt = self._expr_type(needle)
        if isinstance(nt, TOptionalType) and _is_string_type(nt.inner):
            return "eq"
        if self._is_numeric_expr(needle):
            return "=="
        return "eq"

    def _len_call(self, expr: TExpr) -> str:
        s = self._expr(expr)
        if self._is_string_expr(expr) or self._is_bytes_expr(expr):
            return "length(" + s + ")"
        if self._is_map_expr(expr) or self._is_set_expr(expr):
            return "scalar(keys %{ +" + s + " })"
        if self._is_list_expr(expr):
            return "scalar(@{" + self._deref_safe(s) + "})"
        safe = self._deref_safe(s)
        return (
            "(ref("
            + s
            + ") eq 'ARRAY' ? scalar(@{"
            + safe
            + "}) : ref("
            + s
            + ") eq 'HASH' ? scalar(keys %{"
            + safe
            + "}) : length("
            + s
            + "))"
        )

    def _deref_safe(self, s: str) -> str:
        """Wrap expressions that are ambiguous inside @{} / %{} deref."""
        if s.startswith("do ") or s.startswith("do{"):
            return "(" + s + ")"
        if s == "{}":
            return "({})"
        return s

    def _hash_key(self, expr: TExpr) -> str:
        if isinstance(expr, TTupleLit):
            parts = [self._expr(e) for e in expr.elements]
            return 'join("\\0", ' + ", ".join(parts) + ")"
        if isinstance(expr, TVar):
            typ = self.var_types.get(expr.name)
            if isinstance(typ, TTupleType):
                v = self._expr(expr)
                return "(ref(" + v + ') ? join("\\0", @{' + v + "}) : " + v + ")"
        return self._expr(expr)

    def _format_int(self, args: list[TArg]) -> str:
        n = self._a(args, 0)
        base_expr = args[1].value
        if isinstance(base_expr, TIntLit):
            if base_expr.value == 16:
                return "sprintf('%x', " + n + ")"
            if base_expr.value == 8:
                return "sprintf('%o', " + n + ")"
            if base_expr.value == 2:
                return "sprintf('%b', " + n + ")"
        return (
            "do { my $__n = "
            + n
            + "; my $__b = "
            + self._a(args, 1)
            + "; $__b == 16 ? sprintf('%x', $__n) : $__b == 8 ? sprintf('%o', $__n) : $__b == 2 ? sprintf('%b', $__n) : ('' . $__n) }"
        )

    def _format_call(self, args: list[TArg]) -> str:
        template_expr = args[0].value
        if not isinstance(template_expr, TStringLit):
            arg_strs = ", ".join(self._expr(a.value) for a in args[1:])
            if arg_strs:
                return "sprintf(" + self._expr(template_expr) + ", " + arg_strs + ")"
            return "sprintf(" + self._expr(template_expr) + ")"
        template = template_expr.value
        fmt_parts: list[str] = []
        i = 0
        placeholders = 0
        while i < len(template):
            if i + 1 < len(template) and template[i] == "{" and template[i + 1] == "}":
                fmt_parts.append("%s")
                placeholders += 1
                i += 2
                continue
            ch = template[i]
            if ch == "%":
                fmt_parts.append("%%")
            else:
                fmt_parts.append(ch)
            i += 1
        fmt = _string_literal("".join(fmt_parts))
        used_args = ", ".join(self._expr(a.value) for a in args[1 : 1 + placeholders])
        if used_args:
            return "sprintf(" + fmt + ", " + used_args + ")"
        return fmt

    def _zero_value(self, typ: TType) -> str:
        if isinstance(typ, TPrimitive):
            if typ.kind in ("int", "byte"):
                return "0"
            if typ.kind == "float":
                return "0.0"
            if typ.kind == "bool":
                return "0"
            if typ.kind in ("string", "rune", "bytes"):
                return '""'
            if typ.kind in ("void", "nil"):
                return "undef"
        if isinstance(typ, TListType):
            return "[]"
        if isinstance(typ, TMapType):
            return "{}"
        if isinstance(typ, TSetType):
            return "{}"
        if isinstance(typ, TTupleType):
            return "[]"
        if isinstance(typ, TOptionalType):
            return "undef"
        if isinstance(typ, TUnionType):
            return "undef"
        if isinstance(typ, TIdentType):
            if typ.name in self.struct_fields:
                return typ.name + "->new()"
            return "undef"
        if isinstance(typ, TFuncType):
            return "undef"
        return "undef"

    def _expr_type(self, expr: TExpr) -> TType | None:
        if isinstance(expr, TVar):
            return self.var_types.get(expr.name)
        if isinstance(expr, TStringLit):
            return TPrimitive(expr.pos, "string")
        if isinstance(expr, TRuneLit):
            return TPrimitive(expr.pos, "rune")
        if isinstance(expr, TBytesLit):
            return TPrimitive(expr.pos, "bytes")
        if isinstance(expr, TBoolLit):
            return TPrimitive(expr.pos, "bool")
        if isinstance(expr, TIntLit):
            return TPrimitive(expr.pos, "int")
        if isinstance(expr, TFloatLit):
            return TPrimitive(expr.pos, "float")
        if isinstance(expr, TListLit):
            return TListType(expr.pos, TPrimitive(expr.pos, "int"))
        if isinstance(expr, TTupleLit):
            return TTupleType(expr.pos, [TPrimitive(expr.pos, "int")])
        if isinstance(expr, TMapLit):
            return TMapType(
                expr.pos, TPrimitive(expr.pos, "int"), TPrimitive(expr.pos, "int")
            )
        if isinstance(expr, TSetLit):
            return TSetType(expr.pos, TPrimitive(expr.pos, "int"))
        if isinstance(expr, TSlice):
            obj_t = self._expr_type(expr.obj)
            if _is_string_type(obj_t) or _is_bytes_type(obj_t):
                return obj_t
            if _is_list_type(obj_t):
                return obj_t
        if isinstance(expr, TIndex):
            obj_t = self._expr_type(expr.obj)
            if _is_string_type(obj_t) or _is_bytes_type(obj_t):
                return TPrimitive(expr.pos, "string")
            if isinstance(obj_t, TListType):
                return obj_t.element
            if isinstance(obj_t, TMapType):
                return obj_t.value
        if isinstance(expr, TFieldAccess):
            obj_t = self._expr_type(expr.obj)
            if isinstance(obj_t, TIdentType):
                ftypes = self.sft.get(obj_t.name, {})
                result = ftypes.get(expr.field)
                if result is not None:
                    return result
            if isinstance(obj_t, TOptionalType) and isinstance(obj_t.inner, TIdentType):
                ftypes = self.sft.get(obj_t.inner.name, {})
                result = ftypes.get(expr.field)
                if result is not None:
                    return result
            found: TType | None = None
            for sname, sft_entry in self.sft.items():
                if expr.field in sft_entry:
                    ft = sft_entry[expr.field]
                    if found is None:
                        found = ft
                    elif not _same_type_kind(found, ft):
                        found = None
                        break
            return found
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            name = expr.func.name
            if name in ("Len", "Ord", "Hash", "IndexOf", "Count"):
                return TPrimitive(expr.pos, "int")
            if name in (
                "ToString",
                "Join",
                "Lower",
                "Upper",
                "Strip",
                "LStrip",
                "RStrip",
                "Replace",
                "Trim",
            ):
                return TPrimitive(expr.pos, "string")
            if name == "ToFloat":
                return TPrimitive(expr.pos, "float")
            if name == "Sorted" or name == "ListFrom" or name == "Reversed":
                if expr.args:
                    inner = self._expr_type(expr.args[0].value)
                    if isinstance(inner, TListType):
                        return inner
                    if isinstance(inner, TSetType):
                        return TListType(expr.pos, inner.element)
                return TListType(expr.pos, TPrimitive(expr.pos, "int"))
            if name == "Get":
                if len(expr.args) >= 3:
                    return self._expr_type(expr.args[2].value)
                if expr.args:
                    obj_t = self._expr_type(expr.args[0].value)
                    if isinstance(obj_t, TMapType):
                        return obj_t.value
                return None
            if name == "Map" or name == "Merge":
                return TMapType(
                    expr.pos, TPrimitive(expr.pos, "int"), TPrimitive(expr.pos, "int")
                )
            if name == "Set":
                return TSetType(expr.pos, TPrimitive(expr.pos, "int"))
            ret = self.fn_ret.get(name)
            if ret is not None:
                return ret
        if isinstance(expr, TBinaryOp):
            if expr.op in ("+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"):
                lt = self._expr_type(expr.left)
                if lt is not None:
                    return lt
                return self._expr_type(expr.right)
            if expr.op in ("==", "!=", "<", ">", "<=", ">=", "and", "or", "not"):
                return TPrimitive(expr.pos, "bool")
        if isinstance(expr, TUnaryOp):
            if expr.op in ("-", "~"):
                return self._expr_type(expr.operand)
            if expr.op == "!":
                return TPrimitive(expr.pos, "bool")
        if isinstance(expr, TTernary):
            return self._expr_type(expr.then_expr)
        return None

    def _is_string_expr(self, expr: TExpr) -> bool:
        if isinstance(expr, (TStringLit, TRuneLit)):
            return True
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            if expr.func.name == "ToString":
                return True
        typ = self._expr_type(expr)
        return _is_string_type(typ)

    def _is_bytes_expr(self, expr: TExpr) -> bool:
        if isinstance(expr, TBytesLit):
            return True
        typ = self._expr_type(expr)
        return _is_bytes_type(typ)

    def _needs_concat_parens(self, expr: TExpr) -> bool:
        if isinstance(expr, TBinaryOp):
            op = self._binary_op(expr.op, expr.left, expr.right)
            return _PRECEDENCE.get(op, 0) < _PRECEDENCE.get(".", 0)
        if isinstance(expr, TUnaryOp) and expr.op == "!":
            return True
        return False

    def _is_list_expr(self, expr: TExpr) -> bool:
        if isinstance(expr, (TListLit, TTupleLit)):
            return True
        typ = self._expr_type(expr)
        return _is_list_type(typ)

    def _is_map_expr(self, expr: TExpr) -> bool:
        if isinstance(expr, TMapLit):
            return True
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            if expr.func.name == "Map" or expr.func.name == "Merge":
                return True
        typ = self._expr_type(expr)
        return _is_map_type(typ)

    def _is_set_expr(self, expr: TExpr) -> bool:
        if isinstance(expr, TSetLit):
            return True
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            if expr.func.name == "Set":
                return True
        typ = self._expr_type(expr)
        return _is_set_type(typ)

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
        typ = self._expr_type(expr)
        return isinstance(typ, TPrimitive) and typ.kind == "int"

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
        typ = self._expr_type(expr)
        return isinstance(typ, TListType) and _is_primitive(typ.element, "float")

    def _static_int(self, expr: TExpr) -> int | None:
        if isinstance(expr, TIntLit):
            return expr.value
        if (
            isinstance(expr, TUnaryOp)
            and expr.op == "-"
            and isinstance(expr.operand, TIntLit)
        ):
            return -expr.operand.value
        return None

    def _is_negative_literal(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TUnaryOp)
            and expr.op == "-"
            and isinstance(expr.operand, TIntLit)
        )

    def _nil_coalesce_value(self, expr: TTernary) -> str | None:
        """Emit ($x // default) for none_coalesce provenance."""
        cond = expr.cond
        if not isinstance(cond, TBinaryOp) or cond.op != "!=":
            return None
        if not isinstance(cond.right, TNilLit):
            return None
        if not isinstance(cond.left, TVar):
            return None
        var = "$" + _safe_name(cond.left.name)
        default = self._expr(expr.else_expr)
        return "(" + var + " // " + default + ")"

    def _format_interpolated(self, args: list[TArg]) -> str:
        """Emit Perl double-quoted string with variable interpolation."""
        template_expr = args[0].value
        if not isinstance(template_expr, TStringLit):
            return self._format_call(args)
        template = template_expr.value
        parts: list[str] = []
        arg_idx = 0
        i = 0
        while i < len(template):
            if i + 1 < len(template) and template[i] == "{" and template[i + 1] == "}":
                if arg_idx < len(args) - 1:
                    parts.append(self._expr(args[arg_idx + 1].value))
                arg_idx += 1
                i += 2
                continue
            parts.append(template[i])
            i += 1
        result: list[str] = []
        buf: list[str] = []
        for p in parts:
            if p.startswith("$"):
                if buf:
                    result.append("".join(buf))
                    buf: list[str] = []
                result.append(p)
            else:
                buf.append(p)
        if buf:
            result.append("".join(buf))
        out: list[str] = []
        for r in result:
            if r.startswith("$"):
                out.append(r)
            else:
                out.append(_escape_perl_string(r))
        return '"' + "".join(out) + '"'


def emit_perl(module: TModule) -> str:
    struct_names: set[str] = set()
    for decl in module.decls:
        if isinstance(decl, TStructDecl):
            struct_names.add(decl.name)
        elif isinstance(decl, TInterfaceDecl):
            struct_names.add(decl.name)
    for _bk in BUILTIN_STRUCTS:
        struct_names.add(_bk)
    enum_names: set[str] = set()
    for decl in module.decls:
        if isinstance(decl, TEnumDecl):
            enum_names.add(decl.name)
    function_names: set[str] = set()
    struct_fields: dict[str, list[str]] = {}
    struct_field_types: dict[str, dict[str, TType]] = {}
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            function_names.add(decl.name)
        elif isinstance(decl, TStructDecl):
            fnames: list[str] = []
            ftypes: dict[str, TType] = {}
            for f in decl.fields:
                fnames.append(f.name)
                ftypes[f.name] = f.typ
            struct_fields[decl.name] = fnames
            struct_field_types[decl.name] = ftypes
            for method in decl.methods:
                function_names.add(method.name)
        elif isinstance(decl, TInterfaceDecl) and decl.fields:
            struct_fields[decl.name] = [f.name for f in decl.fields]
            struct_field_types[decl.name] = {f.name: f.typ for f in decl.fields}
    emitter = _PerlEmitter(
        struct_names,
        enum_names,
        function_names,
        struct_fields,
        module.strict_math,
    )
    emitter.sft = struct_field_types
    emitter.emit_module(module)
    return emitter.output()
