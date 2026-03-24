"""Perl backend: Taytsh AST -> Perl 5.36+ source code."""

from __future__ import annotations

from .ordering import order_decls
from .util import (
    STRICT_INT_BINARY,
    STRICT_INT_COMPOUND,
    Emitter,
    _check_float_expr,
    _check_float_list,
    _check_int_expr,
    _emit_line,
    _emit_output,
    collect_builtin_calls,
    to_snake,
)
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

_PERL_KEYWORDS = frozenset(
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
        "our",
        "package",
        "return",
        "sub",
        "undef",
        "unless",
        "until",
        "use",
        "while",
        "xor",
        "local",
        "state",
    }
)

# Built-in functions and preamble-injected names that collide with
# user-defined *subroutine* names.  Perl's sigil system means $foo
# never shadows foo(), so these only matter for sub definitions and
# bare-word function calls.
_PERL_BUILTIN_FUNCS = frozenset(
    {
        "abs",
        "accept",
        "alarm",
        "atan2",
        "bind",
        "binmode",
        "bless",
        "break",
        "caller",
        "chdir",
        "chmod",
        "chomp",
        "chop",
        "chown",
        "chr",
        "chroot",
        "close",
        "closedir",
        "connect",
        "cos",
        "crypt",
        "dbmclose",
        "dbmopen",
        "defined",
        "delete",
        "die",
        "dump",
        "each",
        "endgrent",
        "endhostent",
        "endnetent",
        "endprotoent",
        "endpwent",
        "endservent",
        "eof",
        "eval",
        "exec",
        "exists",
        "exit",
        "exp",
        "fcntl",
        "fileno",
        "flock",
        "fork",
        "format",
        "formline",
        "getc",
        "getgrent",
        "getgrgid",
        "getgrnam",
        "gethostbyaddr",
        "gethostbyname",
        "gethostent",
        "getlogin",
        "getnetbyaddr",
        "getnetbyname",
        "getnetent",
        "getpeername",
        "getpgrp",
        "getppid",
        "getpriority",
        "getprotobyname",
        "getprotobynumber",
        "getprotoent",
        "getpwent",
        "getpwnam",
        "getpwuid",
        "getservbyname",
        "getservbyport",
        "getservent",
        "getsockname",
        "getsockopt",
        "glob",
        "gmtime",
        "goto",
        "grep",
        "hex",
        "import",
        "index",
        "int",
        "ioctl",
        "join",
        "keys",
        "kill",
        "lc",
        "lcfirst",
        "length",
        "link",
        "listen",
        "localtime",
        "log",
        "lstat",
        "map",
        "mkdir",
        "msgctl",
        "msgget",
        "msgrcv",
        "msgsnd",
        "oct",
        "open",
        "opendir",
        "ord",
        "pack",
        "pipe",
        "pop",
        "pos",
        "print",
        "printf",
        "prototype",
        "push",
        "quotemeta",
        "rand",
        "read",
        "readdir",
        "readline",
        "readlink",
        "readpipe",
        "recv",
        "redo",
        "ref",
        "rename",
        "require",
        "reset",
        "reverse",
        "rewinddir",
        "rindex",
        "rmdir",
        "say",
        "scalar",
        "seek",
        "seekdir",
        "select",
        "semctl",
        "semget",
        "semop",
        "send",
        "setgrent",
        "sethostent",
        "setnetent",
        "setpgrp",
        "setpriority",
        "setprotoent",
        "setpwent",
        "setservent",
        "setsockopt",
        "shift",
        "shmctl",
        "shmget",
        "shmread",
        "shmwrite",
        "shutdown",
        "sin",
        "sleep",
        "socket",
        "socketpair",
        "sort",
        "splice",
        "split",
        "sprintf",
        "sqrt",
        "srand",
        "stat",
        "study",
        "substr",
        "symlink",
        "syscall",
        "sysopen",
        "sysread",
        "sysseek",
        "system",
        "syswrite",
        "tell",
        "telldir",
        "tie",
        "tied",
        "time",
        "times",
        "truncate",
        "uc",
        "ucfirst",
        "umask",
        "unlink",
        "unpack",
        "unshift",
        "untie",
        "values",
        "vec",
        "wait",
        "waitpid",
        "wantarray",
        "warn",
        "write",
        # Preamble imports (injected by emit_module)
        "floor",
        "ceil",
        "min",
        "max",
        "sum",
        "looks_like_number",
        "encode",
        "decode",
    }
)

# Perl pragmas that poison an entire variable-name prefix.
_PERL_POISONED_PREFIXES = ("utf8_", "bytes_")


def _safe_name(name: str) -> str:
    """Mangle a name for use as a Perl variable ($name)."""
    if name == "_":
        return "_unused"
    prefix = "_" if name.startswith("_") and len(name) > 1 else ""
    safe = to_snake(name)
    if not safe:
        return "_unused"
    safe = prefix + safe
    if safe in _PERL_KEYWORDS:
        return safe + "_"
    for pp in _PERL_POISONED_PREFIXES:
        if safe.startswith(pp) or safe == pp[:-1]:
            return "t_" + safe
    return safe


def _safe_fn_name(name: str) -> str:
    """Mangle a name for use as a Perl subroutine (sub name / name())."""
    safe = _safe_name(name)
    if safe in _PERL_BUILTIN_FUNCS:
        return safe + "_"
    return safe


def _safe_module_name(name: str) -> str:
    """Like _safe_name but for module-level vars (no underscore prefix added)."""
    if name == "_":
        return "_unused"
    safe = to_snake(name)
    if not safe:
        return "_unused"
    if safe in _PERL_KEYWORDS:
        return safe + "_"
    for pp in _PERL_POISONED_PREFIXES:
        if safe.startswith(pp) or safe == pp[:-1]:
            return "t_" + safe
    return safe


def _restore_module_name(name: str, annotations: Ann) -> str:
    """Restore original name for module-level vars."""
    key = "name.original." + name
    if key in annotations:
        return _safe_module_name(annotations[key])
    return _safe_module_name(name)


def _restore_name(name: str, annotations: Ann) -> str:
    """Restore original Python name from annotation, then apply target safety."""
    key = "name.original." + name
    if key in annotations:
        return _safe_name(annotations[key])
    return _safe_name(name)


def _restore_fn_name(name: str, annotations: Ann) -> str:
    """Like _restore_name but for subroutine contexts."""
    key = "name.original." + name
    if key in annotations:
        return _safe_fn_name(annotations[key])
    return _safe_fn_name(name)


_LIST_UTIL_BUILTINS = frozenset({"Min", "Max", "Sum"})


def _has_any_all_provenance(stmts: list[TStmt]) -> bool:
    """Check if any for-loop in stmts has any_call/all_call provenance."""
    for stmt in stmts:
        if isinstance(stmt, TForStmt):
            if stmt.annotations.get("provenance") in ("any_call", "all_call"):
                return True
            if _has_any_all_provenance(stmt.body):
                return True
        elif isinstance(stmt, TIfStmt):
            if _has_any_all_provenance(stmt.then_body):
                return True
            if stmt.else_body is not None and _has_any_all_provenance(stmt.else_body):
                return True
        elif isinstance(stmt, TWhileStmt):
            if _has_any_all_provenance(stmt.body):
                return True
    return False


def _struct_needs_list_util(decl: TStructDecl) -> bool:
    """Check if any method in a struct uses Min/Max/Sum builtins."""
    for method in decl.methods:
        names = collect_builtin_calls(method.body)
        if not names.isdisjoint(_LIST_UTIL_BUILTINS):
            return True
    return False


def _struct_needs_any_all(decl: TStructDecl) -> bool:
    """Check if any method in a struct uses any_call/all_call provenance."""
    for method in decl.methods:
        if _has_any_all_provenance(method.body):
            return True
    return False


def _module_needs_any_all(module: TModule) -> bool:
    """Check if any top-level function uses any_call/all_call provenance."""
    for decl in module.decls:
        if isinstance(decl, TFnDecl) and _has_any_all_provenance(decl.body):
            return True
    return False


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
    i: int = 0
    while i < len(value):
        c: str = value[i : i + 1]
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
    i: int = 0
    while i < len(s):
        ch: str = s[i : i + 1]
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
        i += 1
    return "".join(result)


def _escape_perl_replacement(s: str) -> str:
    result: list[str] = []
    i: int = 0
    while i < len(s):
        ch: str = s[i : i + 1]
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
        i += 1
    return "".join(result)


def _escape_regex_charclass(s: str) -> str:
    result: list[str] = []
    i: int = 0
    while i < len(s):
        ch: str = s[i : i + 1]
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
        i += 1
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


class _PerlEmitter(Emitter):
    def __init__(
        self,
        struct_names: set[str],
        enum_names: set[str],
        function_names: set[str],
        struct_fields: dict[str, list[str]],
        strict_math: bool = False,
        strict_tostring: bool = False,
    ) -> None:
        self.struct_names = struct_names
        self.enum_names = enum_names
        self.function_names = function_names
        self.struct_fields = struct_fields
        self.strict_math = strict_math
        self.strict_tostring = strict_tostring
        self._needs_float_repr: bool = False
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

    def _line(self, text: str = "") -> None:
        _emit_line(self.lines, self.indent, text)

    def output(self) -> str:
        return _emit_output(self.lines)

    def _is_int_expr(self, expr: TExpr) -> bool:
        return _check_int_expr(expr, self.var_types)

    def _is_float_expr(self, expr: TExpr) -> bool:
        return _check_float_expr(expr, self.var_types)

    def _is_float_list(self, expr: TExpr) -> bool:
        return _check_float_list(expr, self.var_types)

    def _tmp(self, prefix: str) -> str:
        name = "$" + prefix + str(self.tmp_counter)
        self.tmp_counter += 1
        return name

    def emit_module(self, module: TModule) -> None:
        self._line("use v5.36;")
        self._line("use utf8;")
        self._line("no warnings 'uninitialized', 'numeric';")
        self._line("use POSIX qw(floor ceil);")
        lu = "min max sum"
        if _module_needs_any_all(module):
            lu += " any all"
        self._line("use List::Util qw(" + lu + ");")
        self._line("use Scalar::Util qw(looks_like_number);")
        self._line("use Encode qw(encode decode);")
        self._line("binmode(STDOUT, ':utf8');")
        self._line("binmode(STDERR, ':utf8');")
        self._line(
            "package ValueError; package UnicodeDecodeError; our @ISA = ('ValueError'); package main;"
        )
        self._line()
        if self.strict_tostring:
            self._line(
                "sub _py_float_repr {"
                " my ($f) = @_;"
                ' return "$f" if $f != $f;'
                ' return "inf" if $f == 9**9**9;'
                ' return "-inf" if $f == -(9**9**9);'
                " my $b;"
                ' for my $d (1..17) { $b = sprintf("%.*g", $d, $f);'
                " last if $b + 0 == $f }"
                ' $b = sprintf("%.17g", $f) if !defined $b;'
                " if ($b =~ /[eE]/) {"
                " my $a = abs($f); if ($a != 0) {"
                " my $e = int(log($a) / log(10));"
                " $e-- if 10**($e+1) <= $a;"
                " if ($e >= 0 && $e <= 15) {"
                ' for my $d (1..20) { my $s = sprintf("%.*f", $d, $f);'
                " if ($s + 0 == $f) {"
                ' $s =~ s/0+$//; $s .= "0" if $s =~ /\\.$/;'
                " $b = $s; last } } } } }"
                ' $b .= ".0" if $b !~ /\\./ && $b !~ /[eE]/;'
                " return $b }"
            )
            self._line()
        self._line("package main;")
        ordered = order_decls(module.decls, lets_first=True)
        has_types = any(
            isinstance(d, (TStructDecl, TEnumDecl, TInterfaceDecl)) for d in ordered
        )
        for decl in ordered:
            if isinstance(decl, TLetStmt):
                self.var_types[decl.name] = decl.typ
                self.module_var_names.add(decl.name)
                if has_types:
                    safe = _restore_module_name(decl.name, decl.annotations)
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
            match decl:
                case TStructDecl():
                    if current_package != decl.name:
                        self._line("package " + decl.name + ";")
                        needs_bu = _struct_needs_list_util(decl)
                        needs_aa = _struct_needs_any_all(decl)
                        if needs_bu or needs_aa:
                            slu = "min max sum"
                            if needs_aa:
                                slu += " any all"
                            self._line("use List::Util qw(" + slu + ");")
                        current_package = decl.name
                        self._line()
                    self._emit_struct(decl)
                case TEnumDecl():
                    if current_package != decl.name:
                        self._line("package " + decl.name + ";")
                        current_package = decl.name
                        self._line()
                    self._emit_enum(decl)
                case TLetStmt():
                    if current_package != "main":
                        self._line("package main;")
                        self._line()
                        current_package = "main"
                    self._emit_stmt(decl)
                case TFnDecl():
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
        has_eq = False
        for method in decl.methods:
            if method.name == "to_string":
                str_method = "to_string"
            elif method.name == "__repr__" and not str_method:
                str_method = "__repr__"
            elif method.name == "__eq__":
                has_eq = True
        if str_method or has_eq:
            parts: list[str] = []
            if str_method:
                parts.append("'\"\"' => \\&" + str_method)
            if has_eq:
                parts.append("'==' => \\&__eq__")
                parts.append("'eq' => \\&__eq__")
                parts.append("'!=' => sub { !__eq__(@_) }")
                parts.append("'ne' => sub { !__eq__(@_) }")
            parts.append("fallback => 1")
            self._line("use overload " + ", ".join(parts) + ";")
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
        param_fields = [f for f in fields if not f.body_computed]
        body_fields = [f for f in fields if f.body_computed]
        self._line("sub new {")
        self.indent += 1
        args: list[str] = ["$class"]
        for f in param_fields:
            args.append("$" + _safe_name(f.name))
        self._line("my (" + ", ".join(args) + ") = @_;")
        self._line("my $self = bless {}, $class;")
        for fld in param_fields:
            safe = _safe_name(fld.name)
            if fld.self_ref and isinstance(fld.typ, TIdentType):
                default = fld.typ.name + "->new($self)"
            else:
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
        for fld in body_fields:
            safe = _safe_name(fld.name)
            if fld.default_expr is not None:
                self._line(
                    "$self->{" + safe + "} = " + self._expr(fld.default_expr) + ";"
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
        self._line("sub " + _safe_fn_name(decl.name) + " {")
        self.indent += 1
        if args:
            self._line("my (" + ", ".join(args) + ") = @_;")
        if not decl.body:
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
        if decl.params and decl.params[0].typ is None:
            self.self_name = decl.params[0].name
        if not decl.body:
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
                        lc = self._try_comprehension(stmt, next_stmt, prov)
                        if lc is not None:
                            self.var_types[stmt.name] = stmt.typ
                            self._line(lc)
                            i += 2
                            continue
                    if prov == "step_slice":
                        ss = self._try_step_slice(stmt, next_stmt)
                        if ss is not None:
                            self.var_types[stmt.name] = stmt.typ
                            self._line(ss)
                            i += 2
                            continue
                    if prov in ("any_call", "all_call"):
                        result = self._emit_any_all(stmts, i, stmt, next_stmt, prov)
                        if result > 0:
                            i += result
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
                if isinstance(call, TCall) and self._is_append_to(call, let_stmt.name):
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

    def _try_step_slice(self, let_stmt: TLetStmt, for_stmt: TForStmt) -> str | None:
        """Reconstruct [@{obj}[grep { $_ % step == offset } 0 .. $#{obj}]]."""
        if not isinstance(for_stmt.iterable, TRange):
            return None
        range_args = for_stmt.iterable.args
        if len(range_args) != 3:
            return None
        body = for_stmt.body
        if len(body) != 1:
            return None
        is_string, src_obj = self._step_slice_source(body[0], let_stmt.name)
        if src_obj is None:
            return None
        src = self._expr(src_obj)
        acc = "$" + _restore_name(let_stmt.name, let_stmt.annotations)
        start_expr = range_args[0]
        step_expr = range_args[2]
        step_s = self._expr(step_expr)
        start_val = start_expr.value if isinstance(start_expr, TIntLit) else 0
        step_val = step_expr.value if isinstance(step_expr, TIntLit) else None
        offset = start_val % step_val if step_val is not None else start_val
        if is_string:
            return (
                "my "
                + acc
                + ' = join("", @{[split("", '
                + src
                + ")]}[grep { $_ % "
                + step_s
                + " == "
                + str(offset)
                + " } 0 .. length("
                + src
                + ") - 1]);"
            )
        return (
            "my "
            + acc
            + " = [@{"
            + src
            + "}[grep { $_ % "
            + step_s
            + " == "
            + str(offset)
            + " } 0 .. $#{"
            + src
            + "}]];"
        )

    def _step_slice_source(
        self, stmt: TStmt, acc_name: str
    ) -> tuple[bool, TExpr | None]:
        """Extract (is_string, source_obj) from a step_slice loop body."""
        # List: ExprStmt(Append(acc, obj[__i]))
        if isinstance(stmt, TExprStmt):
            call = stmt.expr
            if isinstance(call, TCall) and self._is_append_to(call, acc_name):
                elem = call.args[1].value
                if isinstance(elem, TIndex):
                    return False, elem.obj
        # String: acc = Concat(acc, ToString(obj[__i]))
        if isinstance(stmt, TAssignStmt) and isinstance(stmt.target, TVar):
            if stmt.target.name == acc_name and isinstance(stmt.value, TCall):
                if (
                    isinstance(stmt.value.func, TVar)
                    and stmt.value.func.name == "Concat"
                ):
                    second = stmt.value.args[1].value
                    if (
                        isinstance(second, TCall)
                        and isinstance(second.func, TVar)
                        and second.func.name == "ToString"
                    ):
                        inner = second.args[0].value
                        if isinstance(inner, TIndex):
                            return True, inner.obj
        return False, None

    def _emit_any_all(
        self,
        stmts: list[TStmt],
        i: int,
        let_stmt: TLetStmt,
        for_stmt: TForStmt,
        prov: str,
    ) -> int:
        """Try to emit any/all. Returns number of statements to skip, or 0."""
        aa = self._try_any_all(let_stmt, for_stmt, prov)
        if aa:
            lhs, rhs = aa
            self.var_types[let_stmt.name] = let_stmt.typ
            folded = self._fold_temp_assign(stmts, i, let_stmt.name, rhs)
            if folded is not None:
                self._line(folded)
                return 3
            self._line("my " + lhs + " = " + rhs + ";")
            return 2
        return 0

    def _try_any_all(
        self, let_stmt: TLetStmt, for_stmt: TForStmt, prov: str
    ) -> tuple[str, str] | None:
        """Try to reconstruct any/all from a let + for pair. Returns (lhs, rhs)."""
        ann = for_stmt.annotations
        # Perl's $_ can't destructure tuples, so bail on multi-binding
        # unless it's a dict .items() iteration
        if len(for_stmt.binding) > 1:
            if ann.get("for.items") != "true" or len(for_stmt.binding) != 2:
                return None
        acc = "$" + _restore_name(let_stmt.name, let_stmt.annotations)
        iterable = self._expr(for_stmt.iterable)
        binding_name = for_stmt.binding[0] if for_stmt.binding else None
        iter_is_map = ann.get("for.items") == "true"
        if iter_is_map and len(for_stmt.binding) == 2:
            safe = self._deref_safe(iterable)
            iter_spread = (
                "@{(do { my $__m = "
                + safe
                + "; [map { [$_, $__m->{$_}] } sort keys %{$__m}] })}"
            )
        elif self._is_set_expr(for_stmt.iterable):
            iter_spread = "keys %{" + self._deref_safe(iterable) + "}"
        else:
            iter_spread = "@{" + self._deref_safe(iterable) + "}"
        func = "any" if prov == "any_call" else "all"
        body = for_stmt.body
        if len(body) != 1:
            return None
        outer_if = body[0]
        if not isinstance(outer_if, TIfStmt):
            return None
        if (
            len(outer_if.then_body) == 2
            and isinstance(outer_if.then_body[0], TAssignStmt)
            and isinstance(outer_if.then_body[1], TBreakStmt)
        ):
            cond = (
                self._strip_not(outer_if.cond) if prov == "all_call" else outer_if.cond
            )
            if iter_is_map and len(for_stmt.binding) == 2:
                self.var_alias[for_stmt.binding[0]] = "$_->[0]"
                self.var_alias[for_stmt.binding[1]] = "$_->[1]"
            elif binding_name is not None:
                self.var_alias[binding_name] = "$_"
            cond_s = self._expr(cond)
            if iter_is_map and len(for_stmt.binding) == 2:
                self.var_alias.pop(for_stmt.binding[0])
                self.var_alias.pop(for_stmt.binding[1])
            elif binding_name is not None:
                self.var_alias.pop(binding_name)
            return (acc, func + " { " + cond_s + " } " + iter_spread)
        if len(outer_if.then_body) == 1:
            inner_if = outer_if.then_body[0]
            if (
                isinstance(inner_if, TIfStmt)
                and len(inner_if.then_body) == 2
                and isinstance(inner_if.then_body[0], TAssignStmt)
                and isinstance(inner_if.then_body[1], TBreakStmt)
            ):
                cond = (
                    self._strip_not(inner_if.cond)
                    if prov == "all_call"
                    else inner_if.cond
                )
                if iter_is_map and len(for_stmt.binding) == 2:
                    self.var_alias[for_stmt.binding[0]] = "$_->[0]"
                    self.var_alias[for_stmt.binding[1]] = "$_->[1]"
                elif binding_name is not None:
                    self.var_alias[binding_name] = "$_"
                filter_s = self._expr(outer_if.cond)
                cond_s = self._expr(cond)
                if iter_is_map and len(for_stmt.binding) == 2:
                    self.var_alias.pop(for_stmt.binding[0])
                    self.var_alias.pop(for_stmt.binding[1])
                elif binding_name is not None:
                    self.var_alias.pop(binding_name)
                return (
                    acc,
                    func + " { " + filter_s + " && " + cond_s + " } " + iter_spread,
                )
        return None

    def _strip_not(self, expr: TExpr) -> TExpr:
        """Strip a leading ! from a unary-not expression."""
        if isinstance(expr, TUnaryOp) and expr.op == "!":
            return expr.operand
        return expr

    def _fold_temp_assign(
        self, stmts: list[TStmt], i: int, temp_name: str, rhs: str
    ) -> str | None:
        """If stmts[i+2] is `real_name = temp_name`, fold into `real_name = rhs`."""
        if i + 2 >= len(stmts):
            return None
        third = stmts[i + 2]
        if isinstance(third, TLetStmt) and isinstance(third.value, TVar):
            if third.value.name == temp_name:
                real = "$" + _restore_name(third.name, third.annotations)
                return "my " + real + " = " + rhs + ";"
        if isinstance(third, TAssignStmt) and isinstance(third.value, TVar):
            if third.value.name == temp_name and isinstance(third.target, TVar):
                real = "$" + _restore_name(third.target.name, third.target.annotations)
                return real + " = " + rhs + ";"
        return None

    def _emit_stmt(self, stmt: TStmt) -> None:
        if isinstance(stmt, TLetStmt):
            self.var_types[stmt.name] = stmt.typ
            self.local_names.add(stmt.name)
            # Use module name for module-level vars, regular name for locals
            if stmt.name in self.module_var_names:
                safe = "$" + _restore_module_name(stmt.name, stmt.annotations)
            else:
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
        if unused_str:
            for s in unused_str.split(","):
                if s:
                    unused_indices.add(int(s))
        if self._is_divmod_call(stmt.value) and 1 in unused_indices:
            call = stmt.value
            assert isinstance(call, TCall)
            a = self._expr(call.args[0].value)
            b = self._expr(call.args[1].value)
            q_target = self._target(stmt.targets[0])
            self._line(q_target + " = int(" + a + " / " + b + ");")
            return
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

    def _is_divmod_call(self, expr: TExpr) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "DivMod"
        )

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
                and not cond.right.value
            ):
                return self._expr(cond.left)
        return None

    def _emit_else_body(self, else_body: list[TStmt] | None) -> None:
        if else_body is None or not else_body:
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
        elif self._is_builtin_call(stmt.iterable, "Reversed"):
            self._emit_for_reversed(stmt)
        elif self._is_builtin_call(stmt.iterable, "Zip"):
            self._emit_for_zip(stmt)
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
                start_val = self._static_int(args[0])
                start_str = (
                    str(start_val) if start_val is not None else self._expr(args[0])
                )
                if end_val is not None:
                    range_expr = "reverse " + str(end_val + 1) + " .. " + start_str
                else:
                    range_expr = (
                        "reverse " + self._expr(args[1]) + " + 1 .. " + start_str
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

    def _is_builtin_call(self, expr: TExpr, name: str) -> bool:
        return (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == name
        )

    def _emit_for_reversed(self, stmt: TForStmt) -> None:
        ann = stmt.annotations
        assert isinstance(stmt.iterable, TCall)
        inner = stmt.iterable.args[0].value
        name = "$" + _restore_name(stmt.binding[0], ann)
        inner_str = self._expr(inner)
        safe = self._deref_safe(inner_str)
        self._line("for my " + name + " (reverse @{" + safe + "}) {")
        self.indent += 1
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _emit_for_zip(self, stmt: TForStmt) -> None:
        ann = stmt.annotations
        assert isinstance(stmt.iterable, TCall)
        zip_args = stmt.iterable.args
        sizes: list[str] = []
        for a in zip_args:
            arg_expr = self._expr(a.value)
            if self._is_bytes_expr(a.value):
                sizes.append("length(" + arg_expr + ")")
            else:
                sizes.append("scalar(@{" + arg_expr + "})")
        idx = self._tmp("__i")
        min_expr = "min(" + ", ".join(sizes) + ") - 1"
        self._line("for my " + idx + " (0 .. " + min_expr + ") {")
        self.indent += 1
        for i, b in enumerate(stmt.binding):
            var = "$" + _restore_name(b, ann)
            arg_expr = self._expr(zip_args[i].value)
            if self._is_bytes_expr(zip_args[i].value):
                self._line(
                    "my " + var + " = ord(substr(" + arg_expr + ", " + idx + ", 1));"
                )
            else:
                self._line("my " + var + " = " + arg_expr + "->[" + idx + "];")
        self._emit_stmts(stmt.body)
        self.indent -= 1
        self._line("}")

    def _emit_for_iter(
        self, binding: list[str], iterable: TExpr, body: list[TStmt], ann: Ann
    ) -> None:
        it = self._expr(iterable)
        safe = self._deref_safe(it)
        if len(binding) == 1:
            name = "$" + _restore_name(binding[0], ann)
            iter_type: TType | None = (
                self.var_types.get(iterable.name)
                if isinstance(iterable, TVar)
                else None
            )
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
                self._line("for my " + name + " (split(//, " + it + ")) {")
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
                if isinstance(iterable, TVar):
                    iter_type2: TType | None = self.var_types.get(iterable.name)
                    if isinstance(iter_type2, TMapType):
                        self.var_types[binding[0]] = iter_type2.key
                        self.var_types[binding[1]] = iter_type2.value
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
        # Use a unique sentinel to distinguish normal completion from early return
        # Early return: $ok is return value, $@ is empty
        # Exception: $ok is undef, $@ has exception
        # Normal: $ok is 1, $@ is empty
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
        # Handle early return: if $ok is a reference (not the sentinel 1), return it
        self._line("} elsif (ref(" + ok + ")) {")
        self.indent += 1
        self._line("return " + ok + ";")
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
                parts.append("eval { " + err + "->isa('" + t.name + "') }")
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
            if typ.name in ("str", "string"):
                return "(!ref(" + expr + ") && !looks_like_number(" + expr + "))"
            if typ.name in ("int", "float"):
                return "looks_like_number(" + expr + ")"
            if typ.name == "bool":
                return "!ref(" + expr + ")"
            if typ.name in self.struct_names:
                return "UNIVERSAL::isa(" + expr + ", '" + typ.name + "')"
            if typ.name in ("dict", "Dict"):
                return "(ref(" + expr + ") eq 'HASH')"
            if typ.name in ("list", "List"):
                return "(ref(" + expr + ") eq 'ARRAY')"
            return (
                "defined("
                + expr
                + ") && UNIVERSAL::isa("
                + expr
                + ", '"
                + typ.name
                + "')"
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
        return "defined(" + expr + ") && UNIVERSAL::isa(" + expr + ", 'UNSUPPORTED')"

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
                return "\\&" + prefix + _restore_fn_name(expr.name, expr.annotations)
            if expr.name in self.struct_names and expr.name not in self.var_types:
                return expr.name + "->new()"
            if expr.name in self.enum_names and expr.name not in self.var_types:
                return expr.name
            if expr.name in self._PYTHON_BUILTINS and expr.name not in self.var_types:
                return "(" + self._PYTHON_BUILTINS[expr.name] + ")"
            # Module-level globals need main:: prefix when inside a package
            if self.in_package and expr.name in self.module_var_names:
                return "$main::" + _restore_module_name(expr.name, expr.annotations)
            # Use module name for module-level vars, regular name for locals
            if expr.name in self.module_var_names:
                return "$" + _restore_module_name(expr.name, expr.annotations)
            return "$" + _restore_name(expr.name, expr.annotations)
        if isinstance(expr, TFieldAccess):
            if isinstance(expr.obj, TVar) and expr.obj.name in self.enum_names:
                return expr.obj.name + "::" + expr.field
            return self._expr(expr.obj) + "->{" + _safe_name(expr.field) + "}"
        if isinstance(expr, TTupleAccess):
            obj = self._expr(expr.obj)
            idx = str(expr.index)
            obj_ann: str = expr.obj.annotations.get("type", "")
            is_tuple = obj_ann.startswith("(")
            if is_tuple:
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
            prov = expr.annotations.get("provenance")
            if prov == "none_coalesce":
                val = self._nil_coalesce_value(expr)
                if val is not None:
                    return val
            if prov == "removeprefix":
                s, p = self._removefix_args(expr, "StartsWith")
                if s is not None:
                    return "(" + s + " =~ s/^\\Q" + p + "\\E//r)"
            if prov == "removesuffix":
                s, p = self._removefix_args(expr, "EndsWith")
                if s is not None:
                    return "(" + s + " =~ s/\\Q" + p + "\\E$//r)"
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
        if raw.startswith(("0x", "0X", "0o", "0O", "0b", "0B")):
            return raw
        return str(expr.value)

    def _bytes_lit(self, expr: TBytesLit) -> str:
        if not expr.value:
            return '""'
        parts: list[str] = []
        i: int = 0
        while i < len(expr.value):
            parts.append(str(expr.value[i]))
            i += 1
        nums = ", ".join(parts)
        return "pack('C*', " + nums + ")"

    def _slice(self, expr: TSlice) -> str:
        obj = self._expr(expr.obj)
        prov = expr.annotations.get("provenance", "")
        low = self._expr(expr.low)
        high = self._expr(expr.high)
        if prov == "open_start" and self._is_zero(expr.low):
            low = "0"
        if self._is_string_expr(expr.obj) or self._is_bytes_expr(expr.obj):
            if self._is_len_call(expr.high) and self._len_matches_obj(
                expr.high, expr.obj
            ):
                return "substr(" + obj + ", " + low + ")"
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
            if self._is_zero(expr.low):
                return "substr(" + obj + ", 0, " + high + ")"
            return "substr(" + obj + ", " + low + ", (" + high + ") - (" + low + "))"
        is_list = self._is_list_expr(expr.obj)
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
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann in ("int", "float", "bool") or ann in self.enum_names
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
        if isinstance(expr, TVar):
            if expr.name in self.enum_names:
                return True
            typ: TType | None = self.var_types.get(expr.name)
            if isinstance(typ, TPrimitive):
                return typ.kind in ("int", "float", "bool")
        return False

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

    def _perl_key_sort_body(self, fn_lit: TFnLit, reverse: bool = False) -> str:
        """Build Perl sort comparator from a key function TFnLit."""
        first = fn_lit.body[0] if fn_lit.body else None
        if not isinstance(first, TExprStmt):
            return "$a <=> $b"
        param_name = fn_lit.params[0].name if fn_lit.params else "x"
        expr_a = self._perl_key_subst(first.expr, param_name, "$a")
        expr_b = self._perl_key_subst(first.expr, param_name, "$b")
        if reverse:
            return expr_b + " <=> " + expr_a
        return expr_a + " <=> " + expr_b

    def _perl_key_subst(self, expr: TExpr, param_name: str, replacement: str) -> str:
        """Render a TExpr with a variable name substituted."""
        self.var_alias[param_name] = replacement
        result = self._expr(expr)
        del self.var_alias[param_name]
        return result

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
        if (
            isinstance(func, TVar)
            and func.name == "Concat"
            and expr.annotations.get("provenance") == "star_unpack"
        ):
            return self._star_unpack(expr)
        # list(dict) / set(dict) reconstruction via dict_keys provenance
        if (
            isinstance(func, TVar)
            and func.name in ("ListFrom", "SetFromList")
            and expr.annotations.get("provenance") == "dict_keys"
        ):
            inner = args[0].value
            if isinstance(inner, TCall):
                dict_expr = self._deref_safe(self._a(inner.args, 0))
                if func.name == "ListFrom":
                    return "[sort keys %{" + dict_expr + "}]"
                return (
                    "do { my $__s = {}; $__s->{$_} = 1 for sort keys %{"
                    + dict_expr
                    + "}; $__s }"
                )
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
                return prefix + _safe_fn_name(func.name) + "(" + arg_strs + ")"
        fn_expr = self._expr(func)
        arg_strs = ", ".join(self._expr(a.value) for a in args)
        return fn_expr + "->(" + arg_strs + ")"

    def _star_unpack(self, expr: TCall) -> str:
        """Reconstruct [ @{$a}, $x, @{$b} ] from a Concat chain."""
        parts: list[TExpr] = []
        self._flatten_star_unpack(expr, parts)
        items: list[str] = []
        for p in parts:
            if isinstance(p, TListLit):
                for elem in p.elements:
                    items.append(self._expr(elem))
            else:
                items.append("@{" + self._expr(p) + "}")
        return "[ " + ", ".join(items) + " ]"

    def _flatten_star_unpack(self, expr: TExpr, parts: list[TExpr]) -> None:
        if (
            isinstance(expr, TCall)
            and isinstance(expr.func, TVar)
            and expr.func.name == "Concat"
            and expr.annotations.get("provenance") == "star_unpack"
        ):
            self._flatten_star_unpack(expr.args[0].value, parts)
            parts.append(expr.args[1].value)
        else:
            parts.append(expr)

    def _struct_call(self, name: str, args: list[TArg]) -> str:
        if name in BUILTIN_STRUCTS and name not in self.struct_fields:
            return self._builtin_error_call(name, args)
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

    def _builtin_error_call(self, name: str, args: list[TArg]) -> str:
        fields = list(BUILTIN_STRUCTS[name].keys())
        pairs: list[str] = []
        for i, a in enumerate(args):
            fname = (
                a.name
                if a.name is not None
                else (fields[i] if i < len(fields) else str(i))
            )
            pairs.append(fname + " => " + self._expr(a.value))
        return "bless({" + ", ".join(pairs) + "}, '" + name + "')"

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
                return (
                    "(exists "
                    + obj
                    + "->{"
                    + key
                    + "} ? "
                    + obj
                    + "->{"
                    + key
                    + "} : "
                    + default
                    + ")"
                )
            return obj + "->{" + key + "}"
        if method == "append" and not self._is_known_struct_method(func.obj, method):
            obj = self._expr(func.obj)
            val = self._expr(args[0].value)
            return "push(@{" + obj + "}, " + val + ")"
        if method == "keys" and not self._is_known_struct_method(func.obj, method):
            obj = self._deref_safe(self._expr(func.obj))
            return "[sort keys %{" + obj + "}]"
        if method == "values" and not self._is_known_struct_method(func.obj, method):
            obj = self._deref_safe(self._expr(func.obj))
            return "[values %{" + obj + "}]"
        if method == "items" and not self._is_known_struct_method(func.obj, method):
            obj = self._deref_safe(self._expr(func.obj))
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
            if args:
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
            a_str = self._maybe_paren(args[0].value, "/", is_left=True)
            b_str = self._maybe_paren(args[1].value, "/", is_left=False)
            return "POSIX::floor(" + a_str + " / " + b_str + ")"
        if name == "PythonMod":
            # a - floor(a / b) * b, need parens around a when used in division
            # Outer parens needed for correct precedence when used in comparisons
            a_str = self._maybe_paren(args[0].value, "/", is_left=True)
            b_str = self._maybe_paren(args[1].value, "/", is_left=False)
            return (
                "("
                + a_str
                + " - POSIX::floor("
                + a_str
                + " / "
                + b_str
                + ") * "
                + b_str
                + ")"
            )
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
            s = self._a(args, 0)
            chars = args[1].value
            if isinstance(chars, TStringLit):
                if chars.value == " \t\n\r":
                    return "do { my $__t = " + s + "; $__t =~ s/^\\s+//; $__t }"
                raw = _escape_regex_charclass(chars.value)
                return "do { my $__t = " + s + "; $__t =~ s/^[" + raw + "]+//; $__t }"
            c = self._a(args, 1)
            return "do { my $__t = " + s + "; $__t =~ s/^[" + c + "]+//; $__t }"
        if name == "TrimEnd":
            s = self._a(args, 0)
            chars = args[1].value
            if isinstance(chars, TStringLit):
                if chars.value == " \t\n\r":
                    return "do { my $__t = " + s + "; $__t =~ s/\\s+$//; $__t }"
                raw = _escape_regex_charclass(chars.value)
                return "do { my $__t = " + s + "; $__t =~ s/[" + raw + "]+$//; $__t }"
            c = self._a(args, 1)
            return "do { my $__t = " + s + "; $__t =~ s/[" + c + "]+$//; $__t }"
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
            if len(args) == 4:
                return (
                    "do { my $__i = index(substr("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 2)
                    + ", "
                    + self._a(args, 3)
                    + " - "
                    + self._a(args, 2)
                    + "), "
                    + self._a(args, 1)
                    + "); $__i == -1 ? -1 : $__i + "
                    + self._a(args, 2)
                    + " }"
                )
            if len(args) == 3:
                return (
                    "index("
                    + self._a(args, 0)
                    + ", "
                    + self._a(args, 1)
                    + ", "
                    + self._a(args, 2)
                    + ")"
                )
            return "index(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "RFind":
            if len(args) >= 3:
                start = self._a(args, 2)
                sliced = "substr(" + self._a(args, 0) + ", " + start
                if len(args) == 4:
                    sliced += ", " + self._a(args, 3) + " - " + start
                sliced += ")"
                return (
                    "do { my $__i = rindex("
                    + sliced
                    + ", "
                    + self._a(args, 1)
                    + "); $__i == -1 ? -1 : $__i + "
                    + start
                    + " }"
                )
            return "rindex(" + self._a(args, 0) + ", " + self._a(args, 1) + ")"
        if name == "Count":
            count_arg0: TExpr = args[0].value
            count_arg1: TExpr = args[1].value
            a0 = self._a(args, 0)
            if len(args) >= 3:
                a0 = "substr(" + a0 + ", " + self._a(args, 2)
                if len(args) == 4:
                    a0 += ", " + self._a(args, 3) + " - " + self._a(args, 2)
                a0 += ")"
            if self._is_string_expr(count_arg0) or self._is_bytes_expr(count_arg0):
                if isinstance(count_arg1, TStringLit):
                    pat = _escape_perl_regex(count_arg1.value)
                    return "() = " + a0 + " =~ /" + pat + "/g"
                return (
                    "do { my $__s = "
                    + a0
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
            if len(args) >= 3:
                s = "substr(" + s + ", " + self._a(args, 2)
                if len(args) == 4:
                    s += ", " + self._a(args, 3) + " - " + self._a(args, 2)
                s += ")"
            pfx: TExpr = args[1].value
            if isinstance(pfx, TStringLit):
                pat = _escape_perl_regex(pfx.value)
                return "((" + s + " =~ /^" + pat + "/) ? 1 : 0)"
            return "((" + s + " =~ /^\\Q${\\ " + self._a(args, 1) + "}\\E/) ? 1 : 0)"
        if name == "EndsWith":
            s = self._a(args, 0)
            if len(args) >= 3:
                s = "substr(" + s + ", " + self._a(args, 2)
                if len(args) == 4:
                    s += ", " + self._a(args, 3) + " - " + self._a(args, 2)
                s += ")"
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
            return "Encode::encode('UTF-8', " + self._a(args, 0) + ")"
        if name == "Decode":
            a = self._a(args, 0)
            return (
                "do { my $__d = eval { Encode::decode('UTF-8', "
                + a
                + ", Encode::FB_CROAK) }; if ($@) { die bless({message => \"$@\"}, 'UnicodeDecodeError') } $__d }"
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
            return "+{%{" + a + "}, %{" + b + "}}"
        if name == "Intersection":
            a = self._deref_safe(self._a(args, 0))
            b = self._a(args, 1)
            return (
                "do { my $s = {}; $s->{$_} = 1"
                + " for grep { exists "
                + b
                + "->{$_} } keys %{"
                + a
                + "}; $s }"
            )
        if name == "Difference":
            a = self._deref_safe(self._a(args, 0))
            b = self._a(args, 1)
            return (
                "do { my $s = {}; $s->{$_} = 1"
                + " for grep { !exists "
                + b
                + "->{$_} } keys %{"
                + a
                + "}; $s }"
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
            return "[sort keys %{" + self._deref_safe(self._a(args, 0)) + "}]"
        if name == "Values":
            return "[values %{" + self._deref_safe(self._a(args, 0)) + "}]"
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
            if len(args) == 2:
                key_val = args[1].value
                if isinstance(key_val, TFnLit):
                    key_body = self._perl_key_sort_body(key_val)
                    return "(sort { " + key_body + " } @{" + self._a(args, 0) + "})[0]"
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
            if len(args) == 2:
                key_val = args[1].value
                if isinstance(key_val, TFnLit):
                    key_body = self._perl_key_sort_body(key_val, reverse=True)
                    return "(sort { " + key_body + " } @{" + self._a(args, 0) + "})[0]"
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
            a = self._a(args, 0)
            b = self._a(args, 1)
            return (
                "[int("
                + a
                + " / "
                + b
                + "), "
                + a
                + " - int("
                + a
                + " / "
                + b
                + ") * "
                + b
                + "]"
            )
        if name == "Sorted":
            sorted_arg = args[0].value
            if self.strict_math and self._is_float_list(sorted_arg):
                return "strict_sorted_f64(" + self._a(args, 0) + ")"
            a = self._a(args, 0)
            if len(args) == 2:
                key_val = args[1].value
                if isinstance(key_val, TFnLit):
                    key_body = self._perl_key_sort_body(key_val)
                    return "[sort { " + key_body + " } @{" + a + "}]"
            if self._is_set_expr(sorted_arg):
                return "[sort keys %{" + self._deref_safe(a) + "}]"
            sorted_ann: str = sorted_arg.annotations.get("type", "")
            is_str_list = sorted_ann in ("list[string]", "list[rune]")
            if is_str_list:
                return "[sort @{" + a + "}]"
            return "[sort { $a <=> $b } @{" + a + "}]"
        if name == "RangeList":
            start_val = self._static_int(args[0].value)
            end_val = self._static_int(args[1].value)
            step_val = self._static_int(args[2].value)
            if step_val == 1:
                if start_val == 0 and end_val is not None:
                    return "[0.." + str(end_val - 1) + "]"
                s = self._a(args, 0)
                if end_val is not None:
                    return "[" + s + " .. " + str(end_val - 1) + "]"
                return "[" + s + " .. " + self._a(args, 1) + " - 1]"
            s = self._a(args, 0)
            e = self._a(args, 1)
            st = self._a(args, 2)
            return (
                "[map { "
                + s
                + " + $_ * "
                + st
                + " } 0 .. int(("
                + e
                + " - "
                + s
                + " - 1) / "
                + st
                + ")]"
            )
        if name == "ListFrom":
            a = self._a(args, 0)
            if self._is_bytes_expr(args[0].value):
                return "[unpack('C*', " + a + ")]"
            if self._is_set_expr(args[0].value):
                return "[sort keys %{" + self._deref_safe(a) + "}]"
            return "[@{" + self._deref_safe(a) + "}]"
        if name == "Reversed":
            a = self._a(args, 0)
            if self._is_set_expr(args[0].value):
                return "[reverse keys %{" + a + "}]"
            return "[reverse @{" + a + "}]"
        if name == "Map":
            if not args:
                return "{}"
            return (
                "[map { " + self._a(args, 0) + "->($_) } @{" + self._a(args, 1) + "}]"
            )
        if name == "Set":
            if not args:
                return "{}"
            return (
                "do { my $__s = {}; $__s->{$_} = 1 for @{"
                + self._deref_safe(self._a(args, 0))
                + "}; $__s }"
            )
        if name == "SetFromList":
            if isinstance(args[0].value, TSetLit):
                return self._a(args, 0)
            sfl_inner = args[0].value
            if isinstance(sfl_inner, TCall) and isinstance(sfl_inner.func, TVar):
                if sfl_inner.func.name == "Keys":
                    d = self._deref_safe(self._a(sfl_inner.args, 0))
                    return (
                        "do { my $__s = {}; $__s->{$_} = 1 for sort keys %{"
                        + d
                        + "}; $__s }"
                    )
            a = self._deref_safe(self._a(args, 0))
            if self._is_set_expr(args[0].value):
                return (
                    "do { my $__s = {}; $__s->{$_} = 1 for sort keys %{"
                    + a
                    + "}; $__s }"
                )
            return "do { my $__s = {}; $__s->{$_} = 1 for @{" + a + "}; $__s }"
        if name in ("ToString", "ToRepr"):
            inner_expr = args[0].value
            inner = self._expr(inner_expr)
            if self.strict_tostring and self._is_float_expr(inner_expr):
                self._needs_float_repr = True
                return "_py_float_repr(" + inner + ")"
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
            return "do { my $__l = scalar(<STDIN>); defined($__l) ? Encode::decode('UTF-8', $__l, Encode::FB_CROAK) : $__l }"
        if name == "ReadAll":
            return "do { local $/; Encode::decode('UTF-8', scalar(<STDIN>), Encode::FB_CROAK) }"
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
            a_str = self._maybe_paren(args[0].value, "**", is_left=True)
            b_str = self._maybe_paren(args[1].value, "**", is_left=False)
            return a_str + " ** " + b_str
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
            if type_name in ("str", "string"):
                return "(!ref(" + a0 + ") && !looks_like_number(" + a0 + "))"
            if type_name in ("int", "float"):
                return "looks_like_number(" + a0 + ")"
            if type_name == "bool":
                return "!ref(" + a0 + ")"
            if type_name in self.struct_names:
                return "(UNIVERSAL::isa(" + a0 + ", '" + type_name + "'))"
            if type_name == "dict":
                return "(ref(" + a0 + ") eq 'HASH')"
            if type_name == "list":
                return "(ref(" + a0 + ") eq 'ARRAY')"
            return "(UNIVERSAL::isa(" + a0 + ", '" + type_name + "'))"
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
        c_ann: str = container.annotations.get("type", "")
        if c_ann in ("list[string]", "list[rune]"):
            return "eq"
        n_ann: str = needle.annotations.get("type", "")
        if n_ann in ("nil | string", "nil | rune"):
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
        if s.startswith(("do ", "do{")):
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

    def _is_string_expr(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann in ("string", "rune")
        if isinstance(expr, (TStringLit, TRuneLit)):
            return True
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            if expr.func.name in ("ToString", "ToRepr"):
                return True
        if isinstance(expr, TVar):
            typ: TType | None = self.var_types.get(expr.name)
            if isinstance(typ, TPrimitive):
                return typ.kind in ("string", "rune")
        return False

    def _is_bytes_expr(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann == "bytes"
        if isinstance(expr, TBytesLit):
            return True
        if isinstance(expr, TVar):
            typ: TType | None = self.var_types.get(expr.name)
            return isinstance(typ, TPrimitive) and typ.kind == "bytes"
        return False

    def _needs_concat_parens(self, expr: TExpr) -> bool:
        if isinstance(expr, TBinaryOp):
            op = self._binary_op(expr.op, expr.left, expr.right)
            return _PRECEDENCE.get(op, 0) < _PRECEDENCE.get(".", 0)
        if isinstance(expr, TUnaryOp) and expr.op == "!":
            return True
        return False

    def _is_list_expr(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann.startswith(("list[", "("))
        if isinstance(expr, (TListLit, TTupleLit)):
            return True
        if isinstance(expr, TVar):
            typ: TType | None = self.var_types.get(expr.name)
            return isinstance(typ, (TListType, TTupleType))
        return False

    def _is_map_expr(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann.startswith("map[")
        if isinstance(expr, TMapLit):
            return True
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            return expr.func.name in ("Map", "Merge")
        if isinstance(expr, TVar):
            return isinstance(self.var_types.get(expr.name), TMapType)
        return False

    def _is_set_expr(self, expr: TExpr) -> bool:
        ann: str = expr.annotations.get("type", "")
        if ann:
            return ann.startswith("set[")
        if isinstance(expr, TSetLit):
            return True
        if isinstance(expr, TCall) and isinstance(expr.func, TVar):
            return expr.func.name == "Set"
        if isinstance(expr, TVar):
            return isinstance(self.var_types.get(expr.name), TSetType)
        return False

    def _static_int(self, expr: TExpr) -> int | None:
        if isinstance(expr, TIntLit):
            return expr.value
        if (
            isinstance(expr, TUnaryOp)
            and expr.op == "-"
            and isinstance(expr.operand, TIntLit)
        ):
            return -expr.operand.value
        if (
            isinstance(expr, TBinaryOp)
            and expr.op in ("+", "-")
            and isinstance(expr.left, TIntLit)
            and isinstance(expr.right, TIntLit)
        ):
            if expr.op == "+":
                return expr.left.value + expr.right.value
            return expr.left.value - expr.right.value
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

    def _len_matches_obj(self, len_expr: TExpr, obj: TExpr) -> bool:
        """Check if Len(x) refers to the same variable as obj."""
        if not isinstance(len_expr, TCall) or not len_expr.args:
            return False
        inner = len_expr.args[0].value
        return (
            isinstance(inner, TVar) and isinstance(obj, TVar) and inner.name == obj.name
        )

    def _removefix_args(self, expr: TTernary, func_name: str) -> tuple[str | None, str]:
        """Extract (s, p) from a removeprefix/removesuffix ternary."""
        cond = expr.cond
        if (
            isinstance(cond, TCall)
            and isinstance(cond.func, TVar)
            and cond.func.name == func_name
        ):
            return self._expr(cond.args[0].value), self._expr(cond.args[1].value)
        return None, ""

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
        match decl:
            case TStructDecl() | TInterfaceDecl():
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
        match decl:
            case TFnDecl():
                function_names.add(decl.name)
            case TStructDecl():
                fnames: list[str] = []
                ftypes: dict[str, TType] = {}
                for f in decl.fields:
                    fnames.append(f.name)
                    ftypes[f.name] = f.typ
                struct_fields[decl.name] = fnames
                struct_field_types[decl.name] = ftypes
                for method in decl.methods:
                    function_names.add(method.name)
            case TInterfaceDecl():
                if decl.fields:
                    struct_fields[decl.name] = [f.name for f in decl.fields]
                    struct_field_types[decl.name] = {f.name: f.typ for f in decl.fields}
    emitter = _PerlEmitter(
        struct_names,
        enum_names,
        function_names,
        struct_fields,
        module.strict_math,
        module.strict_tostring,
    )
    emitter.emit_module(module)
    return emitter.output()
