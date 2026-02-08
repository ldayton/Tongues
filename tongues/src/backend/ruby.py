"""Ruby backend: IR → Ruby code.

Known limitation: Dict key coercion (int/float/bool equivalence) only works for
VarDecl initializers and direct Index access. Not covered: Assign, function args,
return statements, dict methods (get/pop/setdefault), `in` operator, dict
comprehensions, nested contexts. Proper fix requires frontend type propagation.
"""

from __future__ import annotations

from src.backend.util import escape_string, to_snake, to_screaming_snake
from src.ir import (
    BOOL,
    FLOAT,
    INT,
    STRING,
    Array,
    Assert,
    Assign,
    BinaryOp,
    Block,
    BoolLit,
    Break,
    Call,
    Cast,
    ChainedCompare,
    CharClassify,
    CatchClause,
    CompGenerator,
    Constant,
    Continue,
    DictComp,
    DerefLV,
    EntryPoint,
    Expr,
    ExprStmt,
    Field,
    FieldAccess,
    FieldLV,
    FloatLit,
    ForClassic,
    ForRange,
    FuncRef,
    FuncType,
    Function,
    If,
    Index,
    IndexLV,
    IntLit,
    IntToStr,
    InterfaceDef,
    InterfaceRef,
    IsNil,
    IsType,
    Len,
    ListComp,
    LValue,
    MakeMap,
    MakeSlice,
    Map,
    MapLit,
    Match,
    MatchCase,
    MaxExpr,
    MethodCall,
    MinExpr,
    Module,
    NilLit,
    NoOp,
    OpAssign,
    Optional,
    Param,
    ParseInt,
    Pointer,
    Primitive,
    Raise,
    Return,
    Set,
    SetComp,
    SetLit,
    Slice,
    SliceConvert,
    SliceExpr,
    SliceLit,
    SliceLV,
    SoftFail,
    StaticCall,
    Stmt,
    StringConcat,
    StringFormat,
    StringLit,
    Struct,
    StructLit,
    StructRef,
    Ternary,
    TrimChars,
    TryCatch,
    Tuple,
    TupleAssign,
    Truthy,
    TupleLit,
    Type,
    TypeAssert,
    TypeCase,
    TypeSwitch,
    UnaryOp,
    Union,
    Var,
    VarDecl,
    VarLV,
    While,
)


_PYTHON_EXCEPTION_MAP = {
    "Exception": "StandardError",
    "AssertionError": "RuntimeError",
    "ValueError": "ArgumentError",
    "RuntimeError": "RuntimeError",
    "KeyError": "KeyError",
    "IndexError": "IndexError",
    "TypeError": "TypeError",
}


def _is_bool_int_compare(left: Expr, right: Expr) -> bool:
    """True when one operand is bool and the other is int."""
    l, r = left.typ, right.typ
    return (l == BOOL and r == INT) or (l == INT and r == BOOL)


def _is_empty_body(body: list[Stmt]) -> bool:
    """Check if body is empty or contains only NoOp statements."""
    return not body or all(isinstance(s, NoOp) for s in body)


_RUBY_RESERVED = frozenset(
    {
        "BEGIN",
        "END",
        "__ENCODING__",
        "__END__",
        "__FILE__",
        "__LINE__",
        "alias",
        "and",
        "begin",
        "break",
        "case",
        "class",
        "def",
        "defined?",
        "do",
        "else",
        "elsif",
        "end",
        "ensure",
        "false",
        "for",
        "if",
        "in",
        "module",
        "next",
        "nil",
        "not",
        "or",
        "redo",
        "rescue",
        "retry",
        "return",
        "self",
        "super",
        "then",
        "true",
        "undef",
        "unless",
        "until",
        "when",
        "while",
        "yield",
        # Common built-in methods that may conflict
        "lambda",
        "proc",
        "loop",
        "raise",
        "fail",
        "catch",
        "throw",
        "format",
        "puts",
        "print",
        "p",
        "gets",
        "require",
        "load",
    }
)


_RUBY_BUILTINS = frozenset(
    {
        "Array",
        "BasicObject",
        "Binding",
        "Class",
        "Comparable",
        "Complex",
        "Data",
        "Dir",
        "Encoding",
        "Enumerable",
        "Enumerator",
        "Exception",
        "FalseClass",
        "Fiber",
        "File",
        "Float",
        "Hash",
        "Integer",
        "IO",
        "Kernel",
        "Marshal",
        "MatchData",
        "Method",
        "Module",
        "NilClass",
        "Numeric",
        "Object",
        "Proc",
        "Process",
        "Queue",
        "Random",
        "Range",
        "Rational",
        "Regexp",
        "Set",
        "Signal",
        "String",
        "Struct",
        "Symbol",
        "Thread",
        "Time",
        "TracePoint",
        "TrueClass",
        "UnboundMethod",
    }
)


def _safe_name(name: str) -> str:
    """Rename variables that conflict with Ruby reserved words."""
    name = to_snake(name)
    if name in _RUBY_RESERVED:
        return name + "_"
    return name


def _is_constant(name: str) -> bool:
    """Check if name looks like a constant (SCREAMING_SNAKE_CASE)."""
    return name.isupper() or (name.replace("_", "").isupper() and "_" in name)


def _safe_type_name(name: str) -> str:
    """Rename types that conflict with Ruby built-in classes."""
    if name in _RUBY_BUILTINS:
        return name + "_"
    return name


_BYTES_CLASS = """\
class TonguesBytes
  include Comparable
  attr_reader :data
  def initialize(data)
    @data = data.is_a?(String) ? data.bytes : data
  end
  def self.from_str(s); new(s.bytes); end
  def self.zeros(n); new(Array.new(n, 0)); end
  def <=>(other); @data <=> other.data; end
  def ==(other); other.is_a?(TonguesBytes) && @data == other.data; end
  def eql?(other); self == other; end
  def hash; @data.hash; end
  def length; @data.length; end
  def empty?; @data.empty?; end
  def [](idx)
    return @data[idx] if idx.is_a?(Integer)
    TonguesBytes.new(@data[idx])
  end
  def +(other); TonguesBytes.new(@data + other.data); end
  def *(n); TonguesBytes.new(@data * n); end
  def coerce(other); [self, other]; end
  def include?(sub); to_s.include?(sub.to_s); end
  def to_s; @data.pack('C*').force_encoding('UTF-8'); end
  def to_str; to_s; end
  def inspect; "b" + to_s.inspect; end
  def each(&block); @data.each(&block); end
  def count(sub); to_s.scan(sub.to_s).length; end
  def index(sub); to_s.index(sub.to_s); end
  def find(sub); (i = index(sub)) ? i : -1; end
  def start_with?(prefix); to_s.start_with?(prefix.to_s); end
  def end_with?(suffix); to_s.end_with?(suffix.to_s); end
  def upper; TonguesBytes.new(to_s.upcase.bytes); end
  def lower; TonguesBytes.new(to_s.downcase.bytes); end
  def strip(chars = nil)
    chars.nil? ? TonguesBytes.new(to_s.strip.bytes) : TonguesBytes.new(to_s.gsub(/\\A[#{Regexp.escape(chars.to_s)}]+|[#{Regexp.escape(chars.to_s)}]+\\z/, '').bytes)
  end
  def lstrip; TonguesBytes.new(to_s.lstrip.bytes); end
  def rstrip; TonguesBytes.new(to_s.rstrip.bytes); end
  def split(sep); to_s.split(sep.to_s, -1).map { |s| TonguesBytes.new(s.bytes) }; end
  def join(arr); TonguesBytes.new(arr.map(&:to_s).join(to_s).bytes); end
  def gsub(old, new_s); TonguesBytes.new(to_s.gsub(old.to_s, new_s.to_s).bytes); end
end
"""


class RubyBackend:
    """Emit Ruby code from IR."""

    def __init__(self) -> None:
        self.indent = 0
        self.lines: list[str] = []
        self.receiver_name: str | None = None
        self._known_functions: set[str] = set()
        self._needs_set = False
        self._needs_bytes = False
        self._needs_truthy_helper = False
        self._needs_range_helper = False

    def emit(self, module: Module) -> str:
        """Emit Ruby code from IR Module."""
        self.indent = 0
        self.lines = []
        self._needs_set = False
        self._needs_bytes = False
        self._needs_truthy_helper = False
        self._needs_range_helper = False
        self._emit_module(module)
        insert_pos = self._import_insert_pos
        if self._needs_set:
            self.lines.insert(insert_pos, "require 'set'")
            self.lines.insert(insert_pos + 1, "")
            insert_pos += 2
        if self._needs_bytes:
            for i, line in enumerate(_BYTES_CLASS.strip().split("\n")):
                self.lines.insert(insert_pos + i, line)
            self.lines.insert(insert_pos + len(_BYTES_CLASS.strip().split("\n")), "")
            insert_pos += len(_BYTES_CLASS.strip().split("\n")) + 1
        if self._needs_truthy_helper:
            helper = "def _truthy(v); v.is_a?(Numeric) ? v != 0 : (v.respond_to?(:empty?) ? !v.empty? : !!v); end"
            self.lines.insert(insert_pos, helper)
            self.lines.insert(insert_pos + 1, "")
            insert_pos += 2
        if self._needs_range_helper:
            helper = "def _range(start, stop = nil, step = 1); stop.nil? ? (0...start).step(step).to_a : (step > 0 ? (start...stop).step(step).to_a : (stop + 1..start).step(-step).to_a.reverse); end"
            self.lines.insert(insert_pos, helper)
            self.lines.insert(insert_pos + 1, "")
        return "\n".join(self.lines)

    def _line(self, text: str = "") -> None:
        if text:
            self.lines.append("  " * self.indent + text)
        else:
            self.lines.append("")

    def _emit_module(self, module: Module) -> None:
        self._known_functions = {f.name for f in module.functions}
        for s in module.structs:
            for m in s.methods:
                self._known_functions.add(m.name)
        self._line("# frozen_string_literal: true")
        self._line()
        self._import_insert_pos: int = len(self.lines)
        need_blank = False
        if module.constants:
            for const in module.constants:
                self._emit_constant(const)
            need_blank = True
        for iface in module.interfaces:
            if need_blank:
                self._line()
            self._emit_interface(iface)
            need_blank = True
        for struct in module.structs:
            if need_blank:
                self._line()
            self._emit_struct(struct)
            need_blank = True
        for func in module.functions:
            if need_blank:
                self._line()
            self._emit_function(func)
            need_blank = True
        for stmt in module.statements:
            if need_blank:
                self._line()
            self._emit_stmt(stmt)
            need_blank = True
        if module.entrypoint is not None:
            self._line()
            self._emit_stmt(module.entrypoint)

    def _emit_constant(self, const: Constant) -> None:
        name = to_screaming_snake(const.name)
        val = self._expr(const.value)
        self._line(f"{name} = {val}")

    def _emit_interface(self, iface: InterfaceDef) -> None:
        self._line(f"module {iface.name}")
        self.indent += 1
        if not iface.methods:
            self._line("# Empty interface")
        for method in iface.methods:
            params = ", ".join(_safe_name(p.name) for p in method.params)
            self._line(f"def {_safe_name(method.name)}({params})")
            self.indent += 1
            self._line("raise NotImplementedError")
            self.indent -= 1
            self._line("end")
        self.indent -= 1
        self._line("end")

    def _emit_struct(self, struct: Struct) -> None:
        is_empty = not struct.fields and not struct.methods and not struct.doc
        if is_empty and not struct.is_exception and not struct.implements:
            return
        if struct.is_exception:
            base = (
                _safe_type_name(struct.embedded_type)
                if struct.embedded_type
                else "StandardError"
            )
            self._line(f"class {_safe_type_name(struct.name)} < {base}")
        else:
            self._line(f"class {_safe_type_name(struct.name)}")
        self.indent += 1
        if struct.doc:
            self._line(f"# {struct.doc}")
        if struct.fields:
            attrs = ", ".join(f":{_safe_name(f.name)}" for f in struct.fields)
            self._line(f"attr_accessor {attrs}")
            self._line()
            self._emit_initialize(struct.fields)
        elif not struct.methods:
            self._line("# Empty class")
        for i, method in enumerate(struct.methods):
            if i > 0 or struct.fields:
                self._line()
            self._emit_method(method)
        self.indent -= 1
        self._line("end")

    def _emit_initialize(self, fields: list[Field]) -> None:
        params = []
        for f in fields:
            name = _safe_name(f.name)
            if f.default is not None:
                default = self._expr(f.default)
                params.append(f"{name}: {default}")
            else:
                default = self._zero_value(f.typ)
                params.append(f"{name}: {default}")
        self._line(f"def initialize({', '.join(params)})")
        self.indent += 1
        for f in fields:
            name = _safe_name(f.name)
            self._line(f"@{name} = {name}")
        self.indent -= 1
        self._line("end")

    def _emit_function(self, func: Function) -> None:
        params = self._params(func.params)
        self._line(f"def {_safe_name(func.name)}({params})")
        self.indent += 1
        if func.doc:
            self._line(f"# {func.doc}")
        self._emit_body_with_implicit_return(func.body)
        self.indent -= 1
        self._line("end")

    def _emit_method(self, func: Function) -> None:
        params = self._params(func.params)
        self._line(f"def {_safe_name(func.name)}({params})")
        self.indent += 1
        if func.doc:
            self._line(f"# {func.doc}")
        if func.receiver:
            self.receiver_name = func.receiver.name
        self._emit_body_with_implicit_return(func.body)
        self.receiver_name = None
        self.indent -= 1
        self._line("end")

    def _params(self, params: list[Param]) -> str:
        parts = []
        for p in params:
            name = _safe_name(p.name)
            if p.default is not None:
                default_val = self._expr(p.default)
                parts.append(f"{name} = {default_val}")
            else:
                parts.append(name)
        return ", ".join(parts)

    def _emit_body_with_implicit_return(self, body: list[Stmt]) -> None:
        """Emit function body with Ruby implicit return for last statement."""
        if _is_empty_body(body):
            self._line("nil")
            return
        # Emit all but the last statement normally
        for stmt in body[:-1]:
            self._emit_stmt(stmt)
        # Handle last statement - if it's a Return, emit just the value
        last = body[-1]
        if isinstance(last, Return) and last.value is not None:
            self._line(self._expr(last.value))
        else:
            self._emit_stmt(last)

    def _emit_stmt(self, stmt: Stmt) -> None:
        match stmt:
            case VarDecl(name=name, typ=decl_typ, value=value):
                safe = _safe_name(name)
                if value is not None:
                    # Python sort()/reverse() return None, Ruby sort!/reverse! return self
                    if (
                        isinstance(value, MethodCall)
                        and value.method in ("sort", "reverse")
                        and isinstance(value.receiver_type, Slice)
                    ):
                        # Execute the in-place operation, then assign nil
                        self._line(self._expr(value))
                        self._line(f"{safe} = nil")
                    else:
                        # Emit MapLit with coerced keys if target type differs from literal type
                        val = self._emit_map_lit_coerced(value, decl_typ)
                        self._line(f"{safe} = {val}")
                else:
                    self._line(f"{safe} = nil")
            case Assign(target=target, value=value):
                if isinstance(target, SliceLV):
                    self._emit_slice_assign(target, value)
                # Python sort()/reverse() return None, Ruby sort!/reverse! return self
                elif (
                    isinstance(value, MethodCall)
                    and value.method in ("sort", "reverse")
                    and isinstance(value.receiver_type, Slice)
                ):
                    # Execute the in-place operation, then assign nil
                    self._line(self._expr(value))
                    lv = self._lvalue(target)
                    self._line(f"{lv} = nil")
                else:
                    lv = self._lvalue(target)
                    val = self._expr(value)
                    self._line(f"{lv} = {val}")
            case TupleAssign(targets=targets, value=value):
                lvalues = ", ".join(self._lvalue(t) for t in targets)
                val = self._expr(value)
                self._line(f"{lvalues} = {val}")
            case OpAssign(target=target, op=op, value=value):
                lv = self._lvalue(target)
                val = self._expr(value)
                # list += string needs to convert string to chars
                if op == "+" and value.typ == STRING:
                    self._line(f"{lv}.concat({val}.chars)")
                # dict |= other_dict -> dict.merge!(other_dict)
                elif op == "|" and isinstance(value.typ, Map):
                    self._line(f"{lv}.merge!({val})")
                else:
                    self._line(f"{lv} {op}= {val}")
            case NoOp():
                pass
            case ExprStmt(expr=expr):
                self._line(self._expr(expr))
            case Return(value=value):
                if value is not None:
                    self._line(f"return {self._expr(value)}")
                else:
                    self._line("return")
            case Assert(test=test, message=message):
                cond_str = self._expr(test)
                if message is not None:
                    msg = self._expr(message)
                    self._line(f'raise "AssertionError: #{{{msg}}}" unless {cond_str}')
                else:
                    self._line(f'raise "AssertionError" unless {cond_str}')
            case If(cond=cond, then_body=then_body, else_body=else_body, init=init):
                if init is not None:
                    self._emit_stmt(init)
                self._line(f"if {self._expr(cond)}")
                self.indent += 1
                if _is_empty_body(then_body):
                    self._line("nil")
                for s in then_body:
                    self._emit_stmt(s)
                self.indent -= 1
                self._emit_else_body(else_body)
                self._line("end")
            case TypeSwitch(expr=expr, binding=binding, cases=cases, default=default):
                self._emit_type_switch(expr, binding, cases, default)
            case Match(expr=expr, cases=cases, default=default):
                self._emit_match(expr, cases, default)
            case ForRange(index=index, value=value, iterable=iterable, body=body):
                self._emit_for_range(index, value, iterable, body)
            case ForClassic(init=init, cond=cond, post=post, body=body):
                self._emit_for_classic(init, cond, post, body)
            case While(cond=cond, body=body):
                self._line(f"while {self._expr(cond)}")
                self.indent += 1
                if _is_empty_body(body):
                    self._line("nil")
                for s in body:
                    self._emit_stmt(s)
                self.indent -= 1
                self._line("end")
            case Break(label=_):
                self._line("break")
            case Continue(label=_):
                self._line("next")
            case Block(body=body):
                for s in body:
                    self._emit_stmt(s)
            case TryCatch(
                body=body,
                catches=catches,
                reraise=reraise,
            ):
                self._emit_try_catch(body, catches, reraise)
            case Raise(
                error_type=error_type, message=message, pos=pos, reraise_var=reraise_var
            ):
                if reraise_var:
                    self._line(f"raise {reraise_var}")
                else:
                    msg = self._expr(message)
                    if pos is not None:
                        pos_expr = self._expr(pos)
                        self._line(
                            f"raise {error_type}.new(message: {msg}, pos: {pos_expr})"
                        )
                    else:
                        self._line(f"raise {error_type}.new(message: {msg})")
            case SoftFail():
                self._line("return nil")
            case EntryPoint(function_name=function_name):
                self._line(f"exit({_safe_name(function_name)})")
            case _:
                raise NotImplementedError("Unknown statement")

    def _emit_type_switch(
        self, expr: Expr, binding: str, cases: list[TypeCase], default: list[Stmt]
    ) -> None:
        var = self._expr(expr)
        safe_binding = _safe_name(binding)
        self._line(f"case {var}")
        for case in cases:
            type_name = self._type_name_for_check(case.typ)
            self._line(f"when {type_name}")
            self.indent += 1
            self._line(f"{safe_binding} = {var}")
            if _is_empty_body(case.body):
                self._line("nil")
            for s in case.body:
                self._emit_stmt(s)
            self.indent -= 1
        if default:
            self._line("else")
            self.indent += 1
            for s in default:
                self._emit_stmt(s)
            self.indent -= 1
        self._line("end")

    def _emit_match(
        self, expr: Expr, cases: list[MatchCase], default: list[Stmt]
    ) -> None:
        self._line(f"case {self._expr(expr)}")
        for case in cases:
            patterns = ", ".join(self._expr(p) for p in case.patterns)
            self._line(f"when {patterns}")
            self.indent += 1
            if _is_empty_body(case.body):
                self._line("nil")
            for s in case.body:
                self._emit_stmt(s)
            self.indent -= 1
        if default:
            self._line("else")
            self.indent += 1
            for s in default:
                self._emit_stmt(s)
            self.indent -= 1
        self._line("end")

    def _emit_for_range(
        self,
        index: str | None,
        value: str | None,
        iterable: Expr,
        body: list[Stmt],
    ) -> None:
        iter_expr = self._expr(iterable)
        if isinstance(iterable.typ, Optional) or isinstance(iterable, FieldAccess):
            iter_expr = f"({iter_expr} || [])"
        idx = _safe_name(index) if index else None
        val = _safe_name(value) if value else None
        # Check if iterable is enumerate() call - unpack [i, v] pairs directly
        is_enumerate = isinstance(iterable, Call) and iterable.func == "enumerate"
        # Use each_char for strings
        is_string = iterable.typ == Primitive(kind="string")
        # For dict iteration with single variable, iterate over keys only
        is_map = isinstance(iterable.typ, Map)
        each_method = "each_char" if is_string else ("each_key" if is_map else "each")
        if idx is not None and val is not None:
            if is_enumerate:
                # enumerate already produces [idx, val] pairs
                self._line(f"{iter_expr}.each do |({idx}, {val})|")
            elif is_string:
                self._line(f"{iter_expr}.{each_method}.with_index do |{val}, {idx}|")
            else:
                self._line(f"{iter_expr}.each_with_index do |{val}, {idx}|")
        elif val is not None:
            self._line(f"{iter_expr}.{each_method} do |{val}|")
        elif idx is not None:
            if is_string:
                self._line(f"{iter_expr}.{each_method}.with_index do |_, {idx}|")
            else:
                self._line(f"{iter_expr}.each_index do |{idx}|")
        else:
            self._line(f"{iter_expr}.{each_method} do")
        self.indent += 1
        if _is_empty_body(body):
            self._line("nil")
        for s in body:
            self._emit_stmt(s)
        self.indent -= 1
        self._line("end")

    def _emit_for_classic(
        self,
        init: Stmt | None,
        cond: Expr | None,
        post: Stmt | None,
        body: list[Stmt],
    ) -> None:
        if (range_info := _extract_range_pattern(init, cond, post)) is not None:
            var_name, iterable_expr = range_info
            self._line(
                f"(0...{self._expr(iterable_expr)}.length).each do |{_safe_name(var_name)}|"
            )
            self.indent += 1
            if _is_empty_body(body):
                self._line("nil")
            for s in body:
                self._emit_stmt(s)
            self.indent -= 1
            self._line("end")
            return
        if init is not None:
            self._emit_stmt(init)
        cond_str = self._expr(cond) if cond else "true"
        self._line(f"while {cond_str}")
        self.indent += 1
        if _is_empty_body(body) and post is None:
            self._line("nil")
        for s in body:
            self._emit_stmt(s)
        if post is not None:
            self._emit_stmt(post)
        self.indent -= 1
        self._line("end")

    def _emit_try_catch(
        self,
        body: list[Stmt],
        catches: list[CatchClause],
        reraise: bool,
    ) -> None:
        self._line("begin")
        self.indent += 1
        if _is_empty_body(body):
            self._line("nil")
        for s in body:
            self._emit_stmt(s)
        self.indent -= 1
        for clause in catches:
            var = _safe_name(clause.var) if clause.var else "_e"
            if isinstance(clause.typ, StructRef):
                exc_type = _PYTHON_EXCEPTION_MAP.get(
                    clause.typ.name, _safe_type_name(clause.typ.name)
                )
            else:
                exc_type = "StandardError"
            self._line(f"rescue {exc_type} => {var}")
            self.indent += 1
            if _is_empty_body(clause.body) and not reraise:
                self._line("nil")
            for s in clause.body:
                self._emit_stmt(s)
            if reraise:
                self._line("raise")
            self.indent -= 1
        self._line("end")

    def _emit_else_body(self, else_body: list[Stmt]) -> None:
        """Emit else body, converting single-If else to elsif chains."""
        if _is_empty_body(else_body):
            return
        if len(else_body) == 1 and isinstance(else_body[0], If):
            elif_stmt = else_body[0]
            if elif_stmt.init is not None:
                self._line("else")
                self.indent += 1
                self._emit_stmt(elif_stmt.init)
                self._line(f"if {self._expr(elif_stmt.cond)}")
                self.indent += 1
                if _is_empty_body(elif_stmt.then_body):
                    self._line("nil")
                for s in elif_stmt.then_body:
                    self._emit_stmt(s)
                self.indent -= 1
                self._emit_else_body(elif_stmt.else_body)
                self._line("end")
                self.indent -= 1
            else:
                self._line(f"elsif {self._expr(elif_stmt.cond)}")
                self.indent += 1
                if _is_empty_body(elif_stmt.then_body):
                    self._line("nil")
                for s in elif_stmt.then_body:
                    self._emit_stmt(s)
                self.indent -= 1
                self._emit_else_body(elif_stmt.else_body)
        else:
            self._line("else")
            self.indent += 1
            for s in else_body:
                self._emit_stmt(s)
            self.indent -= 1

    def _expr(self, expr: Expr) -> str:
        match expr:
            case IntLit(value=value, format=fmt):
                return self._int_lit(value, fmt)
            case FloatLit(value=value, format=fmt):
                return self._float_lit(value, fmt)
            case StringLit(value=value):
                return _string_literal(value)
            case BoolLit(value=value):
                return "true" if value else "false"
            case NilLit():
                return "nil"
            case Var(name=name):
                if name == self.receiver_name:
                    return "self"
                if _is_constant(name):
                    return to_screaming_snake(name)
                if "_" in name and name[0].isupper():
                    return to_screaming_snake(name)
                if isinstance(expr.typ, FuncType) or name in self._known_functions:
                    return f"method(:{_safe_name(name)})"
                return _safe_name(name)
            case FieldAccess(obj=obj, field=field):
                if field.startswith("F") and field[1:].isdigit():
                    return f"{self._expr(obj)}[{field[1:]}]"
                return f"{self._expr(obj)}.{_safe_name(field)}"
            case FuncRef(name=name, obj=obj):
                if obj is not None:
                    return f"{self._expr(obj)}.method(:{_safe_name(name)})"
                return f"method(:{_safe_name(name)})"
            case Index(obj=obj, index=index):
                if neg_idx := self._negative_index(obj, index):
                    return f"{self._expr(obj)}[{neg_idx}]"
                if isinstance(obj.typ, Map):
                    key_code = self._coerce_map_key(obj.typ.key, index)
                    return f"{self._expr(obj)}[{key_code}]"
                return f"{self._expr(obj)}[{self._expr(index)}]"
            case SliceExpr(obj=obj, low=low, high=high, step=step):
                return self._slice_expr(obj, low, high, step)
            case ParseInt(string=s, base=b):
                base_val = self._expr(b)
                if base_val == "10":
                    return f"{self._expr(s)}.to_i"
                return f"{self._expr(s)}.to_i({base_val})"
            case IntToStr(value=v):
                return f"{self._expr(v)}.to_s"
            case CharClassify(kind=kind, char=char):
                char_expr = self._expr(char)
                # Python semantics: isupper/islower require at least one cased char,
                # and all cased chars match the case. Non-cased chars (digits, etc) are ignored.
                method_map = {
                    "digit": f"({char_expr}).match?(/\\A\\d+\\z/)",
                    "alpha": f"({char_expr}).match?(/\\A[[:alpha:]]+\\z/)",
                    "alnum": f"({char_expr}).match?(/\\A[[:alnum:]]+\\z/)",
                    "space": f"({char_expr}).match?(/\\A\\s+\\z/)",
                    "upper": f"(({char_expr}).match?(/[[:alpha:]]/) && {char_expr} == {char_expr}.upcase)",
                    "lower": f"(({char_expr}).match?(/[[:alpha:]]/) && {char_expr} == {char_expr}.downcase)",
                }
                return method_map[kind]
            case TrimChars(string=s, chars=chars, mode=mode):
                method_map = {"left": "lstrip", "right": "rstrip", "both": "strip"}
                s_expr = self._expr(s)
                # Check if operating on bytes
                is_bytes = isinstance(s.typ, Slice) and s.typ.element == Primitive(
                    kind="byte"
                )
                if is_bytes:
                    # Use TonguesBytes strip methods
                    if isinstance(chars, SliceLit) and all(
                        isinstance(e, IntLit) and e.value in (32, 9, 10, 13)
                        for e in chars.elements
                    ):
                        # Default whitespace - use simple strip/lstrip/rstrip
                        return f"{s_expr}.{method_map[mode]}"
                    # Custom chars - pass to strip method
                    chars_expr = self._expr(chars)
                    return f"{s_expr}.{method_map[mode]}({chars_expr})"
                chars_expr = self._expr(chars)
                if chars_expr == '" "':
                    return f"{s_expr}.{method_map[mode]}"
                return (
                    f"{s_expr}.gsub(/\\A[{chars_expr[1:-1]}]+/, '')"
                    if mode == "left"
                    else (
                        f"{s_expr}.gsub(/[{chars_expr[1:-1]}]+\\z/, '')"
                        if mode == "right"
                        else f"{s_expr}.gsub(/\\A[{chars_expr[1:-1]}]+|[{chars_expr[1:-1]}]+\\z/, '')"
                    )
                )
            case Call(func="_intPtr", args=[arg]):
                val = self._expr(arg)
                return f"({val} == -1 ? nil : {val})"
            case Call(func="print", args=args):
                args_str = ", ".join(self._expr(a) for a in args)
                return f"puts({args_str})"
            case Call(func="repr", args=[arg]) if arg.typ == BOOL:
                return f'({self._expr(arg)} ? "True" : "False")'
            case Call(func="repr", args=[arg]) if arg.typ == Primitive(kind="string"):
                # Python repr uses single quotes, or double if string contains single quote
                arg_str = self._expr(arg)
                return f'({arg_str}.include?("\'") ? \'"\' + {arg_str} + \'"\' : "\'" + {arg_str} + "\'")'
            case Call(func="repr", args=[arg]) if isinstance(arg, NilLit):
                return '"None"'
            case Call(func="repr", args=[arg]):
                return f"{self._expr(arg)}.inspect"
            case Call(func="bool", args=[]):
                return "false"
            case Call(func="bool", args=[arg]):
                # Handle collection types - Ruby's !! doesn't check emptiness
                arg_type = arg.typ
                if isinstance(arg_type, (Slice, Tuple, Map, Set)):
                    return f"({self._expr(arg)}.length != 0)"
                if arg_type == STRING:
                    return f"!{self._expr(arg)}.empty?"
                if arg_type == INT:
                    return f"({self._expr(arg)} != 0)"
                if arg_type == FLOAT:
                    return f"({self._expr(arg)} != 0.0)"
                return f"!!{self._expr(arg)}"
            case Call(func="abs", args=[arg]):
                # Coerce bool to int for abs()
                if arg.typ == BOOL:
                    return f"({self._coerce_bool_to_int(arg, raw=True)}).abs"
                inner = self._expr(arg)
                if isinstance(arg, (BinaryOp, UnaryOp, Ternary)):
                    return f"({inner}).abs"
                return f"{inner}.abs"
            case Call(func="min", args=args):
                # Single iterable arg: call .min directly
                if len(args) == 1 and isinstance(args[0].typ, (Slice, Tuple, Set)):
                    return f"{self._expr(args[0])}.min"
                # min(dict) returns min key
                if len(args) == 1 and isinstance(args[0].typ, Map):
                    return f"{self._expr(args[0])}.keys.min"
                # Ruby can't compare bools, always coerce to int
                parts = []
                for a in args:
                    if a.typ == BOOL:
                        parts.append(self._coerce_bool_to_int(a, raw=True))
                    else:
                        parts.append(self._expr(a))
                return f"[{', '.join(parts)}].min"
            case Call(func="max", args=args):
                # Single iterable arg: call .max directly
                if len(args) == 1 and isinstance(args[0].typ, (Slice, Tuple, Set)):
                    return f"{self._expr(args[0])}.max"
                # max(dict) returns max key
                if len(args) == 1 and isinstance(args[0].typ, Map):
                    return f"{self._expr(args[0])}.keys.max"
                # Ruby can't compare bools, always coerce to int
                parts = []
                for a in args:
                    if a.typ == BOOL:
                        parts.append(self._coerce_bool_to_int(a, raw=True))
                    else:
                        parts.append(self._expr(a))
                return f"[{', '.join(parts)}].max"
            case Call(func="round", args=[arg]):
                inner = self._expr(arg)
                if isinstance(arg, (BinaryOp, UnaryOp, Ternary)):
                    return f"({inner}).round(half: :even)"
                return f"{inner}.round(half: :even)"
            case Call(func="round", args=[arg, precision]):
                inner = self._expr(arg)
                prec = self._expr(precision)
                if isinstance(arg, (BinaryOp, UnaryOp, Ternary)):
                    return f"({inner}).round({prec}, half: :even)"
                return f"{inner}.round({prec}, half: :even)"
            case Call(func="int", args=[arg]):
                inner = self._expr(arg)
                if isinstance(arg, (BinaryOp, UnaryOp, Ternary)):
                    return f"({inner}).to_i"
                return f"{inner}.to_i"
            case Call(func="float", args=[arg]):
                # Handle special float values
                if isinstance(arg, StringLit):
                    val = arg.value.strip().lower()
                    if val in ("inf", "infinity"):
                        return "Float::INFINITY"
                    if val == "-inf":
                        return "-Float::INFINITY"
                    if val == "nan":
                        return "Float::NAN"
                inner = self._expr(arg)
                if isinstance(arg, (BinaryOp, UnaryOp, Ternary)):
                    return f"({inner}).to_f"
                return f"{inner}.to_f"
            case Call(func="divmod", args=[a, b]):
                # Coerce bool operands to int
                if a.typ == BOOL:
                    inner = self._coerce_bool_to_int(a, raw=True)
                else:
                    inner = self._expr(a)
                if b.typ == BOOL:
                    divisor = self._coerce_bool_to_int(b, raw=True)
                else:
                    divisor = self._expr(b)
                if isinstance(a, (BinaryOp, UnaryOp, Ternary)) and a.typ != BOOL:
                    return f"({inner}).divmod({divisor})"
                return f"{inner}.divmod({divisor})"
            case Call(func="pow", args=[base, exp]):
                # Coerce bool operands to int
                if base.typ == BOOL:
                    base_str = self._coerce_bool_to_int(base, raw=True)
                else:
                    base_str = self._expr(base)
                    if isinstance(base, (BinaryOp, UnaryOp, Ternary)):
                        base_str = f"({base_str})"
                if exp.typ == BOOL:
                    exp_str = self._coerce_bool_to_int(exp, raw=True)
                else:
                    exp_str = self._expr(exp)
                    # Wrap exp in parens if it's a binary op (** has high precedence)
                    if isinstance(exp, (BinaryOp, UnaryOp, Ternary)):
                        exp_str = f"({exp_str})"
                return f"{base_str} ** {exp_str}"
            case Call(func="pow", args=[base, exp, mod]):
                # Coerce bool operands to int
                if base.typ == BOOL:
                    base_str = self._coerce_bool_to_int(base, raw=True)
                else:
                    base_str = self._expr(base)
                    if isinstance(base, (BinaryOp, UnaryOp, Ternary)):
                        base_str = f"({base_str})"
                if exp.typ == BOOL:
                    exp_str = self._coerce_bool_to_int(exp, raw=True)
                else:
                    exp_str = self._expr(exp)
                if mod.typ == BOOL:
                    mod_str = self._coerce_bool_to_int(mod, raw=True)
                else:
                    mod_str = self._expr(mod)
                return f"{base_str}.pow({exp_str}, {mod_str})"
            case Call(func="bytes", args=[arg]) if isinstance(arg.typ, Slice):
                # bytes([int_list]) -> TonguesBytes.new([...])
                self._needs_bytes = True
                return f"TonguesBytes.new({self._expr(arg)})"
            case Call(func="bytes", args=[arg]) if arg.typ == INT:
                # bytes(n) -> TonguesBytes.zeros(n)
                self._needs_bytes = True
                return f"TonguesBytes.zeros({self._expr(arg)})"
            case Call(func="enumerate", args=[iterable]):
                # enumerate(x) -> x.each_with_index.to_a (gives [[v0,0], [v1,1]...])
                # But Python gives [(0,v0), (1,v1)...] so we need to swap
                return f"{self._expr(iterable)}.each_with_index.map {{ |v, i| [i, v] }}"
            case Call(func="sum", args=[iterable]):
                # sum(x) -> x.sum
                return f"{self._expr(iterable)}.sum"
            case Call(func="sorted", args=[iterable], reverse=True):
                # sorted(x, reverse=True) -> x.sort.reverse
                return f"{self._expr(iterable)}.sort.reverse"
            case Call(func="sorted", args=[iterable]):
                # sorted(x) -> x.sort
                return f"{self._expr(iterable)}.sort"
            case Call(func="all", args=[iterable]):
                # all(x) -> x.all? { |v| python_truthy(v) }
                self._needs_truthy_helper = True
                return f"{self._expr(iterable)}.all? {{ |v| _truthy(v) }}"
            case Call(func="any", args=[iterable]):
                # any(x) -> x.any? { |v| python_truthy(v) }
                self._needs_truthy_helper = True
                return f"{self._expr(iterable)}.any? {{ |v| _truthy(v) }}"
            case Call(func="list", args=[iterable]):
                # list(x) -> x.to_a, but list(string) -> string.chars
                if iterable.typ == STRING:
                    return f"{self._expr(iterable)}.chars"
                return f"{self._expr(iterable)}.to_a"
            case Call(func="tuple", args=[iterable]):
                # tuple(x) -> x.to_a (Ruby uses arrays for tuples)
                if iterable.typ == STRING:
                    return f"{self._expr(iterable)}.chars"
                return f"{self._expr(iterable)}.to_a"
            case Call(func="tuple", args=[]):
                # tuple() -> []
                return "[]"
            case Call(func="dict", args=[iterable]):
                # dict(pairs) -> pairs.to_h
                return f"{self._expr(iterable)}.to_h"
            case Call(func="dict", args=[]):
                # dict() -> {}
                return "{}"
            case Call(func="set", args=[iterable]):
                # set(x) -> Set.new(x.to_a)
                self._needs_set = True
                if iterable.typ == STRING:
                    return f"Set.new({self._expr(iterable)}.chars)"
                return f"Set.new({self._expr(iterable)}.to_a)"
            case Call(func="set", args=[]):
                # set() -> Set.new
                self._needs_set = True
                return "Set.new"
            case Call(func="range", args=args):
                # range(stop), range(start, stop), range(start, stop, step)
                self._needs_range_helper = True
                args_str = ", ".join(self._expr(a) for a in args)
                return f"_range({args_str})"
            case Call(func="zip", args=args):
                # Python zip stops at shortest, Ruby pads with nil
                # Use take_while to stop at first nil-containing tuple
                if len(args) >= 1:
                    first = self._expr(args[0])
                    rest = ", ".join(self._expr(a) for a in args[1:])
                    if rest:
                        return (
                            f"{first}.zip({rest}).take_while {{ |x| !x.include?(nil) }}"
                        )
                    return f"{first}.zip"
                return "[]"
            case Call(func=func, args=args):
                args_str = ", ".join(self._expr(a) for a in args)
                if func not in self._known_functions:
                    if args_str:
                        return f"{_safe_name(func)}.call({args_str})"
                    return f"{_safe_name(func)}.call"
                return f"{_safe_name(func)}({args_str})"
            case MethodCall(
                obj=obj,
                method=method,
                args=args,
                receiver_type=receiver_type,
                reverse=reverse,
            ):
                # Python: dict.fromkeys(keys) or dict.fromkeys(keys, default) (as MethodCall)
                # Note: Python shares the same default value across all keys (mutable gotcha)
                if isinstance(obj, Var) and obj.name == "dict" and method == "fromkeys":
                    keys_str = self._expr(args[0])
                    if len(args) == 2:
                        default_str = self._expr(args[1])
                        return f"(-> {{ _d = {default_str}; {keys_str}.map {{ |k| [k, _d] }}.to_h }}).call"
                    return f"{keys_str}.map {{ |k| [k, nil] }}.to_h"
                # Python: list.sort(reverse=True) -> Ruby: list.sort!.reverse!
                if method == "sort" and reverse and isinstance(receiver_type, Slice):
                    obj_str = self._expr(obj)
                    return f"{obj_str}.sort!.reverse!"
                # divmod on bool: coerce receiver to int
                if method == "divmod" and obj.typ == BOOL:
                    args_str = ", ".join(self._expr(a) for a in args)
                    return f"({self._coerce_bool_to_int(obj, raw=True)}).divmod({args_str})"
                # Python: "sep".join(iterable) -> Ruby: iterable.join("sep")
                # But for bytes, TonguesBytes has its own join method
                if method == "join" and len(args) == 1:
                    is_bytes = isinstance(
                        receiver_type, Slice
                    ) and receiver_type.element == Primitive(kind="byte")
                    if is_bytes:
                        # TonguesBytes#join(arr) - separator.join(array)
                        sep_str = self._expr(obj)
                        arr_str = self._expr(args[0])
                        return f"{sep_str}.join({arr_str})"
                    sep_str = self._expr(obj)
                    arr_str = self._expr(args[0])
                    if sep_str == '""':
                        return f"{arr_str}.join"
                    return f"{arr_str}.join({sep_str})"
                # Python: s.startswith(prefix, pos) -> Ruby: s[pos..].start_with?(prefix)
                if method == "startswith" and len(args) == 2:
                    obj_str = self._expr(obj)
                    prefix = self._expr(args[0])
                    pos = self._expr(args[1])
                    return f"{obj_str}[{pos}..].start_with?({prefix})"
                # Python: s.endswith((a, b)) -> Ruby: s.end_with?(a, b)
                if method in ("startswith", "endswith") and len(args) == 1:
                    if isinstance(args[0], TupleLit):
                        obj_str = self._expr(obj)
                        unpacked = ", ".join(self._expr(e) for e in args[0].elements)
                        rb_method = _method_name(method, receiver_type)
                        return f"{obj_str}.{rb_method}({unpacked})"
                # Python: dict.get(key) or dict.get(key, default)
                if method == "get" and isinstance(receiver_type, Map):
                    obj_str = self._expr(obj)
                    key = self._expr(args[0])
                    if len(args) == 2:
                        default = self._expr(args[1])
                        return f"{obj_str}.fetch({key}, {default})"
                    return f"{obj_str}[{key}]"
                # Python: dict.items() -> Ruby: hash.to_a (returns [[k,v], ...])
                if (
                    method == "items"
                    and isinstance(receiver_type, Map)
                    and len(args) == 0
                ):
                    obj_str = self._expr(obj)
                    return f"{obj_str}.to_a"
                # Python: dict.keys() -> Ruby: hash.keys
                if (
                    method == "keys"
                    and isinstance(receiver_type, Map)
                    and len(args) == 0
                ):
                    obj_str = self._expr(obj)
                    return f"{obj_str}.keys"
                # Python: dict.values() -> Ruby: hash.values
                if (
                    method == "values"
                    and isinstance(receiver_type, Map)
                    and len(args) == 0
                ):
                    obj_str = self._expr(obj)
                    return f"{obj_str}.values"
                # Python: dict.copy() -> Ruby: hash.dup
                if (
                    method == "copy"
                    and isinstance(receiver_type, Map)
                    and len(args) == 0
                ):
                    obj_str = self._expr(obj)
                    return f"{obj_str}.dup"
                # Python: dict.setdefault(key) or dict.setdefault(key, default)
                if method == "setdefault" and isinstance(receiver_type, Map):
                    obj_str = self._expr(obj)
                    key_str = self._expr(args[0])
                    if len(args) == 2:
                        default_str = self._expr(args[1])
                        return f"({obj_str}.key?({key_str}) ? {obj_str}[{key_str}] : {obj_str}[{key_str}] = {default_str})"
                    return f"({obj_str}.key?({key_str}) ? {obj_str}[{key_str}] : {obj_str}[{key_str}] = nil)"
                # Python: dict.popitem() -> Ruby: [k, v] = hash.shift (but returns last, not first)
                # Ruby's shift returns first, Python's popitem returns last (LIFO)
                if (
                    method == "popitem"
                    and isinstance(receiver_type, Map)
                    and len(args) == 0
                ):
                    obj_str = self._expr(obj)
                    return f"(-> {{ _k = {obj_str}.keys.last; _v = {obj_str}.delete(_k); [_k, _v] }}).call"
                # Python: dict.pop(key) or dict.pop(key, default)
                if method == "pop" and isinstance(receiver_type, Map):
                    obj_str = self._expr(obj)
                    key_str = self._expr(args[0])
                    if len(args) == 2:
                        default_str = self._expr(args[1])
                        return f"{obj_str}.delete({key_str}) {{ {default_str} }}"
                    return f"{obj_str}.delete({key_str})"
                # Set methods
                if isinstance(receiver_type, Set):
                    obj_str = self._expr(obj)
                    # Python: set.remove(x) raises KeyError if not found
                    # Ruby: delete doesn't raise, so we check first
                    if method == "remove" and len(args) == 1:
                        arg_str = self._expr(args[0])
                        return f"(raise 'KeyError' unless {obj_str}.include?({arg_str}); {obj_str}.delete({arg_str}))"
                    # Python: set.discard(x) doesn't raise
                    if method == "discard" and len(args) == 1:
                        return f"{obj_str}.delete({self._expr(args[0])})"
                    # Python: set.pop() removes and returns arbitrary element
                    if method == "pop" and len(args) == 0:
                        return f"(-> {{ _e = {obj_str}.first; {obj_str}.delete(_e); _e }}).call"
                    # Python: set.copy() -> Ruby: dup
                    if method == "copy" and len(args) == 0:
                        return f"{obj_str}.dup"
                    # Python: set.issubset(other) -> Ruby: subset?
                    if method == "issubset" and len(args) == 1:
                        return f"{obj_str}.subset?({self._expr(args[0])})"
                    # Python: set.issuperset(other) -> Ruby: superset?
                    if method == "issuperset" and len(args) == 1:
                        return f"{obj_str}.superset?({self._expr(args[0])})"
                    # Python: set.isdisjoint(other) -> Ruby: disjoint?
                    if method == "isdisjoint" and len(args) == 1:
                        return f"{obj_str}.disjoint?({self._expr(args[0])})"
                    # Python: set.update(*others) -> Ruby: merge (but Ruby only takes one arg)
                    # Handle special cases: string iterates chars, dict iterates keys
                    if method == "update":
                        if len(args) == 1:
                            a = args[0]
                            arg_str = self._expr(a)
                            if a.typ == STRING:
                                arg_str = f"{arg_str}.chars"
                            elif isinstance(a.typ, Map):
                                arg_str = f"{arg_str}.keys"
                            return f"{obj_str}.merge({arg_str})"
                        # Multiple args: chain merge calls
                        parts: list[str] = []
                        for a in args:
                            arg_str = self._expr(a)
                            if a.typ == STRING:
                                arg_str = f"{arg_str}.chars"
                            elif isinstance(a.typ, Map):
                                arg_str = f"{arg_str}.keys"
                            parts.append(arg_str)
                        merges = ".merge(" + ").merge(".join(parts) + ")"
                        return f"{obj_str}{merges}"
                    # Python: set.symmetric_difference(other) -> Ruby: ^
                    if method == "symmetric_difference" and len(args) == 1:
                        return f"{obj_str} ^ {self._expr(args[0])}"
                    # Python: set.union(*others) - Ruby only takes one arg
                    if method == "union":
                        if len(args) == 1:
                            return f"{obj_str}.union({self._expr(args[0])})"
                        parts = [obj_str] + [self._expr(a) for a in args]
                        return " | ".join(parts)
                    # Python: set.intersection(*others) - Ruby only takes one arg
                    if method == "intersection":
                        if len(args) == 1:
                            return f"{obj_str}.intersection({self._expr(args[0])})"
                        parts = [obj_str] + [self._expr(a) for a in args]
                        return " & ".join(parts)
                    # Python: set.difference(*others) - Ruby only takes one arg
                    if method == "difference":
                        if len(args) == 1:
                            return f"{obj_str}.difference({self._expr(args[0])})"
                        parts = [obj_str] + [self._expr(a) for a in args]
                        return " - ".join(parts)
                # Python: list.pop(i) -> Ruby: list.delete_at(i)
                # Special case: pop(0) -> shift for efficiency
                if (
                    method == "pop"
                    and isinstance(receiver_type, Slice)
                    and len(args) == 1
                ):
                    obj_str = self._expr(obj)
                    if isinstance(args[0], IntLit) and args[0].value == 0:
                        return f"{obj_str}.shift"
                    arg_str = self._expr(args[0])
                    return f"{obj_str}.delete_at({arg_str})"
                # Python: list.extend(string) -> Ruby: list.concat(string.chars)
                if (
                    method == "extend"
                    and isinstance(receiver_type, Slice)
                    and len(args) == 1
                ):
                    if args[0].typ == STRING:
                        obj_str = self._expr(obj)
                        arg_str = self._expr(args[0])
                        return f"{obj_str}.concat({arg_str}.chars)"
                # Python: list.index(x, start) or list.index(x, start, end)
                # Ruby's index doesn't support start/end, use slice + offset
                if (
                    method == "index"
                    and isinstance(receiver_type, (Slice, Tuple))
                    and len(args) >= 2
                ):
                    obj_str = self._expr(obj)
                    val_str = self._expr(args[0])
                    start_str = self._expr(args[1])
                    if len(args) == 3:
                        end_str = self._expr(args[2])
                        # slice[start...end].index(val) + start if found
                        return f"(-> {{ _i = {obj_str}[{start_str}...{end_str}].index({val_str}); _i.nil? ? nil : _i + {start_str} }}).call"
                    else:
                        # slice[start..].index(val) + start if found
                        return f"(-> {{ _i = {obj_str}[{start_str}..].index({val_str}); _i.nil? ? nil : _i + {start_str} }}).call"
                # Python: list.remove(x) removes first occurrence
                # Ruby: delete(x) removes ALL occurrences, so use delete_at(index(x))
                if (
                    method == "remove"
                    and isinstance(receiver_type, Slice)
                    and len(args) == 1
                ):
                    obj_str = self._expr(obj)
                    arg_str = self._expr(args[0])
                    return f"{obj_str}.delete_at({obj_str}.index({arg_str}))"
                # Python: list.insert(i, x) - handle both positive and negative indices
                # Ruby's insert(100, x) on a 5-element array pads with nil, Python clips
                # Ruby's insert(-1, x) appends, Python inserts before last
                if (
                    method == "insert"
                    and isinstance(receiver_type, Slice)
                    and len(args) == 2
                ):
                    obj_str = self._expr(obj)
                    idx_str = self._expr(args[0])
                    val_str = self._expr(args[1])
                    # For all indices: clip to valid range [0, length]
                    # Python insert(-n, x) = before index len-n (clamped to 0)
                    # Python insert(n, x) where n > len = append (clamped to len)
                    return f"(-> {{ _idx = {idx_str}; {obj_str}.insert([[_idx < 0 ? {obj_str}.length + _idx : _idx, 0].max, {obj_str}.length].min, {val_str}) }}).call"
                # Python: str.replace("\\", "\\\\") needs special handling in Ruby
                # because gsub's replacement string interprets \\ specially
                if method == "replace" and len(args) == 2:
                    if isinstance(args[0], StringLit) and isinstance(
                        args[1], StringLit
                    ):
                        pattern = args[0].value
                        replacement = args[1].value
                        # When escaping backslashes, use block form to avoid gsub quirks
                        if "\\" in pattern or "\\" in replacement:
                            obj_str = self._expr(obj)
                            # Escape for Ruby string literal
                            pat_escaped = pattern.replace("\\", "\\\\").replace(
                                '"', '\\"'
                            )
                            repl_escaped = replacement.replace("\\", "\\\\").replace(
                                '"', '\\"'
                            )
                            return f'{obj_str}.gsub("{pat_escaped}") {{ "{repl_escaped}" }}'
                # Python: s.rfind(x) returns -1 if not found, Ruby rindex returns nil
                if method == "rfind" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    arg_str = self._expr(args[0])
                    return f"({obj_str}.rindex({arg_str}) || -1)"
                # Python: s.count(sub) counts non-overlapping substrings
                # Ruby's count counts characters, so use scan
                if method == "count" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    arg_str = self._expr(args[0])
                    return f"{obj_str}.scan(Regexp.escape({arg_str})).size"
                # Python: s.split(sep, maxsplit) vs Ruby: s.split(sep, limit)
                # Python maxsplit=n gives n+1 parts, Ruby limit=n gives n parts
                if method == "split" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    if len(args) == 0:
                        # Python split() splits on whitespace and removes empty strings
                        return f"{obj_str}.split"
                    elif len(args) == 1:
                        arg_str = self._expr(args[0])
                        # Ruby's split(" ") collapses whitespace like Python's split()
                        # Use Regexp.new to convert string to regex for consistent behavior
                        # Ruby returns [] for "".split(x), Python returns [""]
                        return f'({obj_str}.empty? ? [""] : {obj_str}.split(Regexp.new(Regexp.escape({arg_str})), -1))'
                    else:
                        sep_str = self._expr(args[0])
                        maxsplit_str = self._expr(args[1])
                        # Python maxsplit=0 means no splits, Ruby limit=1 means no splits
                        # Python maxsplit=n means n splits (n+1 parts), Ruby limit=n+1
                        return f'({obj_str}.empty? ? [""] : {obj_str}.split(Regexp.new(Regexp.escape({sep_str})), {maxsplit_str} == 0 ? 1 : {maxsplit_str} + 1))'
                # Python: s.rsplit(sep, maxsplit) - split from right
                if method == "rsplit" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    if len(args) == 1:
                        arg_str = self._expr(args[0])
                        return f"{obj_str}.split({arg_str}, -1)"
                    else:
                        sep_str = self._expr(args[0])
                        maxsplit_str = self._expr(args[1])
                        return f"{obj_str}.reverse.split({sep_str}, {maxsplit_str} == 0 ? 1 : {maxsplit_str} + 1).map(&:reverse).reverse"
                # Python: s.isupper() - has at least one cased char, all cased are upper
                if method == "isupper" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    return f"({obj_str}.match?(/[[:alpha:]]/) && {obj_str} == {obj_str}.upcase)"
                # Python: s.islower() - has at least one cased char, all cased are lower
                if method == "islower" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    return f"({obj_str}.match?(/[[:alpha:]]/) && {obj_str} == {obj_str}.downcase)"
                # Python: s.title() - titlecase each word (first char upper, rest lower)
                if method == "title" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    return f"{obj_str}.gsub(/\\b\\w+/) {{ |w| w.capitalize }}"
                # Python: s.zfill(width) - zero-pad, preserving sign
                if method == "zfill" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    width_str = self._expr(args[0])
                    return f'(({obj_str}[0] == "-" || {obj_str}[0] == "+") ? {obj_str}[0] + {obj_str}[1..].rjust({width_str} - 1, "0") : {obj_str}.rjust({width_str}, "0"))'
                # Python: s.splitlines() - split on line boundaries
                if method == "splitlines" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    return f'{obj_str}.split(/\\r?\\n/, -1).tap {{ |a| a.pop if a.last == "" }}'
                # Python: s.expandtabs(tabsize) - column-aware tab expansion
                if method == "expandtabs" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    tabsize = "8" if len(args) == 0 else self._expr(args[0])
                    # Column-aware: tab expands to next multiple of tabsize
                    return f'(-> {{ _ts = {tabsize}; _col = 0; {obj_str}.chars.map {{ |c| c == "\\t" ? (" " * (_ts - _col % _ts)).tap {{ _col = 0 }} : c.tap {{ _col = c == "\\n" ? 0 : _col + 1 }} }}.join }}).call'
                # Python: s.casefold() - aggressive lowercase for caseless matching
                if method == "casefold" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    return f"{obj_str}.downcase"
                # Python: s.format(*args) - string formatting (renamed to format_ in IR)
                if method in ("format", "format_") and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    format_args = ", ".join(self._expr(a) for a in args)
                    # Use a lambda to create the args array once
                    return f"(-> {{ _args = [{format_args}]; {obj_str}.gsub(/\\{{(\\d*)\\}}/) {{ |m| m =~ /\\{{(\\d+)\\}}/ ? _args[$1.to_i].to_s : _args.shift.to_s }} }}).call"
                # Python: s.removeprefix(prefix)
                if method == "removeprefix" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    prefix_str = self._expr(args[0])
                    return f"{obj_str}.delete_prefix({prefix_str})"
                # Python: s.removesuffix(suffix)
                if method == "removesuffix" and receiver_type == STRING:
                    obj_str = self._expr(obj)
                    suffix_str = self._expr(args[0])
                    return f"{obj_str}.delete_suffix({suffix_str})"
                args_str = ", ".join(self._expr(a) for a in args)
                rb_method = _method_name(method, receiver_type)
                obj_str = self._expr(obj)
                if isinstance(obj, (BinaryOp, UnaryOp, Ternary)):
                    obj_str = f"({obj_str})"
                if args_str:
                    return f"{obj_str}.{rb_method}({args_str})"
                return f"{obj_str}.{rb_method}"
            case StaticCall(on_type=on_type, method=method, args=args):
                # Python: dict.fromkeys(keys) or dict.fromkeys(keys, default)
                # Note: Python shares the same default value across all keys (mutable gotcha)
                if (
                    method == "fromkeys"
                    and isinstance(on_type, InterfaceRef)
                    and on_type.name == "dict"
                ):
                    keys_str = self._expr(args[0])
                    if len(args) == 2:
                        default_str = self._expr(args[1])
                        return f"(-> {{ _d = {default_str}; {keys_str}.map {{ |k| [k, _d] }}.to_h }}).call"
                    return f"{keys_str}.map {{ |k| [k, nil] }}.to_h"
                args_str = ", ".join(self._expr(a) for a in args)
                type_name = self._type_name_for_check(on_type)
                return f"{type_name}.{_safe_name(method)}({args_str})"
            case Truthy(expr=e):
                inner_type = e.typ
                expr_str = self._expr(e)
                if (
                    isinstance(inner_type, Slice)
                    or isinstance(inner_type, Tuple)
                    or isinstance(inner_type, Map)
                    or isinstance(inner_type, Set)
                ):
                    return f"({expr_str} && !{expr_str}.empty?)"
                # InterfaceRef for collection types (e.g., set(), list(), dict())
                if isinstance(inner_type, InterfaceRef) and inner_type.name in (
                    "set",
                    "list",
                    "dict",
                    "tuple",
                ):
                    return f"({expr_str} && !{expr_str}.empty?)"
                # Check for builtin calls like set(), list(), dict()
                if isinstance(e, Call) and e.func in ("set", "list", "dict", "tuple"):
                    return f"({expr_str} && !{expr_str}.empty?)"
                if isinstance(inner_type, Optional):
                    inner = inner_type.inner
                    if isinstance(inner, (Slice, Map, Set)):
                        return f"({expr_str} && !{expr_str}.empty?)"
                    return f"!{expr_str}.nil?"
                if isinstance(inner_type, Pointer):
                    return f"!{expr_str}.nil?"
                if inner_type == Primitive(kind="string"):
                    return f"({expr_str} && !{expr_str}.empty?)"
                # In Python, len(x) is falsy when 0; in Ruby, 0 is truthy
                # So Truthy(Len(x)) needs to become x.length > 0
                if isinstance(e, Len):
                    return f"{expr_str} > 0"
                # In Python, 0 is falsy; in Ruby, only nil/false are falsy
                # So Truthy(int) needs to become expr != 0
                if inner_type == INT:
                    if isinstance(e, (BinaryOp, UnaryOp, Ternary)):
                        return f"({expr_str}) != 0"
                    return f"{expr_str} != 0"
                # In Python, 0.0 is falsy; in Ruby, only nil/false are falsy
                # So Truthy(float) needs to become expr != 0.0
                if inner_type == FLOAT:
                    if isinstance(e, (BinaryOp, UnaryOp, Ternary)):
                        return f"({expr_str}) != 0.0"
                    return f"{expr_str} != 0.0"
                # Non-int BinaryOp (e.g., comparisons) - wrap in parens for precedence
                if isinstance(e, BinaryOp):
                    return f"!!({expr_str})"
                return f"!!{expr_str}"
            case BinaryOp(op="&&", left=left_expr, right=right_expr) if (
                self._is_value_and_or(left_expr)
            ):
                # Python's `and` returns first falsy value or last value
                # Ruby: python_truthy(x) ? y : x
                left_val, left_str, cond = self._extract_and_or_value(left_expr)
                right_str = self._and_or_operand(right_expr)
                return f"({cond} ? {right_str} : {left_str})"
            case BinaryOp(op="||", left=left_expr, right=right_expr) if (
                self._is_value_and_or(left_expr)
            ):
                # Python's `or` returns first truthy value or last value
                # Ruby: python_truthy(x) ? x : y
                left_val, left_str, cond = self._extract_and_or_value(left_expr)
                right_str = self._and_or_operand(right_expr)
                return f"({cond} ? {left_str} : {right_str})"
            case BinaryOp(op="//", left=left, right=right):
                # Floor division - Ruby's / on integers floors toward -inf
                # For floats, we need (a / b).floor.to_f to match Python semantics
                if left.typ == BOOL:
                    left_str = self._coerce_bool_to_int(left, raw=True)
                else:
                    left_str = self._expr(left)
                if right.typ == BOOL:
                    right_str = self._coerce_bool_to_int(right, raw=True)
                else:
                    right_str = self._expr(right)
                left_str = self._maybe_paren(left_str, left, "/", is_left=True)
                right_str = self._maybe_paren(right_str, right, "/", is_left=False)
                if left.typ == FLOAT or right.typ == FLOAT:
                    return f"({left_str} / {right_str}).floor.to_f"
                return f"{left_str} / {right_str}"
            case BinaryOp(op=op, left=left, right=right) if op in (
                "==",
                "!=",
            ) and _is_bool_int_compare(left, right):
                # For bool coercion, the ternary already provides parens, no need for _maybe_paren
                if left.typ == BOOL:
                    left_str = self._coerce_bool_to_int(left, raw=True)
                else:
                    left_str = self._maybe_paren(
                        self._expr(left), left, op, is_left=True
                    )
                if right.typ == BOOL:
                    right_str = self._coerce_bool_to_int(right, raw=True)
                else:
                    right_str = self._maybe_paren(
                        self._expr(right), right, op, is_left=False
                    )
                rb_op = _binary_op(op)
                return f"{left_str} {rb_op} {right_str}"
            case BinaryOp(op=op, left=left, right=right) if (
                op
                in (
                    "==",
                    "!=",
                )
                and left.typ == BOOL
                and right.typ == BOOL
                and (
                    isinstance(left, (Call, MinExpr, MaxExpr))
                    or isinstance(right, (Call, MinExpr, MaxExpr))
                )
            ):
                # Ruby's all?/any? return actual booleans - compare directly, no coercion
                left_is_all_any = isinstance(left, Call) and left.func in ("all", "any")
                right_is_all_any = isinstance(right, Call) and right.func in (
                    "all",
                    "any",
                )
                if left_is_all_any or right_is_all_any:
                    left_str = self._expr(left)
                    right_str = self._expr(right)
                    rb_op = _binary_op(op)
                    return f"{left_str} {rb_op} {right_str}"
                # MinExpr/MaxExpr with bool args already produce ints, compare directly
                # Just coerce the BoolLit side to int, don't re-coerce the min/max result
                if isinstance(left, (MinExpr, MaxExpr)):
                    left_str = self._expr(left)
                else:
                    left_str = self._coerce_bool_to_int(left, raw=True)
                if isinstance(right, (MinExpr, MaxExpr)):
                    right_str = self._expr(right)
                else:
                    right_str = self._coerce_bool_to_int(right, raw=True)
                rb_op = _binary_op(op)
                return f"{left_str} {rb_op} {right_str}"
            case BinaryOp(op=op, left=left, right=right) if op in (
                "==",
                "!=",
            ) and (
                (isinstance(left.typ, InterfaceRef) and right.typ == BOOL)
                or (left.typ == BOOL and isinstance(right.typ, InterfaceRef))
            ):
                # Ruby's all?/any? return actual booleans - compare directly
                left_is_all_any = isinstance(left, Call) and left.func in ("all", "any")
                right_is_all_any = isinstance(right, Call) and right.func in (
                    "all",
                    "any",
                )
                if left_is_all_any or right_is_all_any:
                    left_str = self._expr(left)
                    right_str = self._expr(right)
                    rb_op = _binary_op(op)
                    return f"{left_str} {rb_op} {right_str}"
                # Comparing any-typed expr (e.g., bool arithmetic) with bool: coerce bool side
                left_str = self._expr(left)
                right_str = (
                    self._coerce_bool_to_int(right, raw=True)
                    if right.typ == BOOL
                    else self._expr(right)
                )
                rb_op = _binary_op(op)
                return f"{left_str} {rb_op} {right_str}"
            case BinaryOp(op="**", left=left, right=right) if (
                left.typ == BOOL or right.typ == BOOL
            ):
                # Exponentiation with bool: coerce to int
                if left.typ == BOOL:
                    left_str = self._coerce_bool_to_int(left, raw=True)
                else:
                    left_str = self._maybe_paren(
                        self._expr(left), left, "**", is_left=True
                    )
                if right.typ == BOOL:
                    right_str = self._coerce_bool_to_int(right, raw=True)
                else:
                    right_str = self._maybe_paren(
                        self._expr(right), right, "**", is_left=False
                    )
                return f"{left_str} ** {right_str}"
            case BinaryOp(op=op, left=left, right=right) if (
                op
                in (
                    "<",
                    ">",
                    "<=",
                    ">=",
                )
                and isinstance(left.typ, (Slice, Tuple))
                and isinstance(right.typ, (Slice, Tuple))
            ):
                # Ruby arrays don't have <, >, <=, >= - use <=> spaceship
                left_str = self._expr(left)
                right_str = self._expr(right)
                return f"(({left_str} <=> {right_str}) {op} 0)"
            case BinaryOp(op=op, left=left, right=right) if op in (
                "<",
                ">",
                "<=",
                ">=",
            ) and (left.typ == BOOL or right.typ == BOOL):
                # Comparisons with bool: coerce to int
                if left.typ == BOOL:
                    left_str = self._coerce_bool_to_int(left, raw=True)
                else:
                    left_str = self._maybe_paren(
                        self._expr(left), left, op, is_left=True
                    )
                if right.typ == BOOL:
                    right_str = self._coerce_bool_to_int(right, raw=True)
                else:
                    right_str = self._maybe_paren(
                        self._expr(right), right, op, is_left=False
                    )
                return f"{left_str} {op} {right_str}"
            case BinaryOp(op=op, left=left, right=right) if op in (
                "<<",
                ">>",
            ) and (left.typ == BOOL or right.typ == BOOL):
                # Shifts with bool: coerce to int
                if left.typ == BOOL:
                    left_str = self._coerce_bool_to_int(left, raw=True)
                else:
                    left_str = self._maybe_paren(
                        self._expr(left), left, op, is_left=True
                    )
                if right.typ == BOOL:
                    right_str = self._coerce_bool_to_int(right, raw=True)
                else:
                    right_str = self._maybe_paren(
                        self._expr(right), right, op, is_left=False
                    )
                return f"{left_str} {op} {right_str}"
            case BinaryOp(op="*", left=left, right=right) if (
                left.typ == INT and right.typ == Primitive(kind="string")
            ):
                # Ruby can't do int * string, swap to string * int
                # Also handle negative: [s, 0].max for n < 0
                left_str = self._expr(left)
                right_str = self._expr(right)
                return f"{right_str} * [{left_str}, 0].max"
            case BinaryOp(op="*", left=left, right=right) if (
                left.typ == Primitive(kind="string") and right.typ == INT
            ):
                # Handle negative multiplier: [n, 0].max
                left_str = self._expr(left)
                right_str = self._expr(right)
                return f"{left_str} * [{right_str}, 0].max"
            case BinaryOp(op="*", left=left, right=right) if (
                left.typ == INT and isinstance(right.typ, (Slice, Tuple))
            ):
                # Ruby can't do int * array, swap to array * int
                # Also handle negative: [n, 0].max for n < 0
                left_str = self._expr(left)
                right_str = self._expr(right)
                return f"{right_str} * [{left_str}, 0].max"
            case BinaryOp(op="*", left=left, right=right) if (
                isinstance(left.typ, (Slice, Tuple)) and right.typ == INT
            ):
                # Handle negative multiplier: [n, 0].max
                left_str = self._expr(left)
                right_str = self._expr(right)
                return f"{left_str} * [{right_str}, 0].max"
            case BinaryOp(op=op, left=left, right=right) if op in (
                "+",
                "-",
                "*",
                "/",
                "%",
                "|",
                "&",
                "^",
            ) and (left.typ == BOOL or right.typ == BOOL):
                # For bool coercion, the ternary already provides parens, no need for _maybe_paren
                if left.typ == BOOL:
                    left_str = self._coerce_bool_to_int(left, raw=True)
                else:
                    left_str = self._maybe_paren(
                        self._expr(left), left, op, is_left=True
                    )
                if right.typ == BOOL:
                    right_str = self._coerce_bool_to_int(right, raw=True)
                else:
                    right_str = self._maybe_paren(
                        self._expr(right), right, op, is_left=False
                    )
                rb_op = _binary_op(op)
                return f"{left_str} {rb_op} {right_str}"
            # Python: dict | dict -> Ruby: dict.merge(dict)
            case BinaryOp(op="|", left=left, right=right) if isinstance(left.typ, Map):
                left_str = self._expr(left)
                right_str = self._expr(right)
                return f"{left_str}.merge({right_str})"
            case BinaryOp(op=op, left=left, right=right):
                if op == "in":
                    return f"{self._expr(right)}.include?({self._expr(left)})"
                if op == "not in":
                    return f"!{self._expr(right)}.include?({self._expr(left)})"
                # Python: s.find(x) == -1 -> Ruby: s.index(x).nil?
                # Python: s.find(x) != -1 -> Ruby: !s.index(x).nil?
                if op in ("==", "!="):
                    find_expr, neg_one = None, None
                    if isinstance(left, MethodCall) and left.method == "find":
                        if (
                            isinstance(right, UnaryOp)
                            and right.op == "-"
                            and isinstance(right.operand, IntLit)
                            and right.operand.value == 1
                        ):
                            find_expr, neg_one = left, right
                    elif isinstance(right, MethodCall) and right.method == "find":
                        if (
                            isinstance(left, UnaryOp)
                            and left.op == "-"
                            and isinstance(left.operand, IntLit)
                            and left.operand.value == 1
                        ):
                            find_expr, neg_one = right, left
                    if find_expr is not None:
                        obj_str = self._expr(find_expr.obj)
                        args_str = ", ".join(self._expr(a) for a in find_expr.args)
                        if op == "==":
                            return f"{obj_str}.index({args_str}).nil?"
                        else:
                            return f"!{obj_str}.index({args_str}).nil?"
                rb_op = _binary_op(op)
                left_str = self._maybe_paren_expr(left, op, is_left=True)
                right_str = self._maybe_paren_expr(right, op, is_left=False)
                result = f"{left_str} {rb_op} {right_str}"
                # If result type is Set (e.g., dict.keys() & dict.keys()), convert to Set
                if isinstance(expr.typ, Set):
                    self._needs_set = True
                    return f"Set[*({result})]"
                # Detect set operations on dict views (keys/items/values)
                # These return arrays in Ruby but sets in Python
                # Convert to Sets first since Ruby arrays don't have ^ operator
                if op in ("&", "|", "-", "^") and (
                    _is_dict_view(left) or _is_dict_view(right)
                ):
                    self._needs_set = True
                    left_set = f"Set[*{left_str}]" if _is_dict_view(left) else left_str
                    right_set = (
                        f"Set[*{right_str}]" if _is_dict_view(right) else right_str
                    )
                    return f"{left_set} {rb_op} {right_set}"
                return result
            case ChainedCompare(operands=operands, ops=ops):
                parts: list[str] = []
                for i, op in enumerate(ops):
                    left_str = self._expr(operands[i])
                    right_str = self._expr(operands[i + 1])
                    rb_op = _binary_op(op)
                    parts.append(f"{left_str} {rb_op} {right_str}")
                return " && ".join(parts)
            case MinExpr(left=left, right=right):
                # Ruby can't compare bools, coerce to int
                left_str = (
                    self._coerce_bool_to_int(left, raw=True)
                    if left.typ == BOOL
                    else self._expr(left)
                )
                right_str = (
                    self._coerce_bool_to_int(right, raw=True)
                    if right.typ == BOOL
                    else self._expr(right)
                )
                return f"[{left_str}, {right_str}].min"
            case MaxExpr(left=left, right=right):
                # Ruby can't compare bools, coerce to int
                left_str = (
                    self._coerce_bool_to_int(left, raw=True)
                    if left.typ == BOOL
                    else self._expr(left)
                )
                right_str = (
                    self._coerce_bool_to_int(right, raw=True)
                    if right.typ == BOOL
                    else self._expr(right)
                )
                return f"[{left_str}, {right_str}].max"
            case UnaryOp(op=op, operand=operand):
                rb_op = _unary_op(op)
                # Unary minus on bool needs parens around the coercion
                if op == "-" and operand.typ == BOOL:
                    return f"-({self._coerce_bool_to_int(operand, raw=True)})"
                # Bitwise NOT on bool needs coercion to int
                if op == "~" and operand.typ == BOOL:
                    return f"~({self._coerce_bool_to_int(operand, raw=True)})"
                # Handle not(truthy(expr)) for various types
                if op == "!" and isinstance(operand, Truthy):
                    inner = operand.expr
                    inner_type = inner.typ
                    inner_str = self._expr(inner)
                    # not(truthy(collection)) -> collection.empty?
                    if isinstance(inner_type, (Slice, Tuple, Map, Set)):
                        return f"{inner_str}.empty?"
                    # InterfaceRef for collection types
                    if isinstance(inner_type, InterfaceRef) and inner_type.name in (
                        "set",
                        "list",
                        "dict",
                        "tuple",
                    ):
                        return f"{inner_str}.empty?"
                    # Check for builtin calls like set(), list(), dict()
                    if isinstance(inner, Call) and inner.func in (
                        "set",
                        "list",
                        "dict",
                        "tuple",
                    ):
                        return f"{inner_str}.empty?"
                    if inner_type == STRING:
                        return f"{inner_str}.empty?"
                    # not(truthy(int)) -> expr == 0
                    if inner_type == Primitive(kind="int"):
                        if isinstance(inner, BinaryOp):
                            return f"(({inner_str}) == 0)"
                        return f"({inner_str} == 0)"
                    # not(truthy(float)) -> expr == 0.0
                    if inner_type == FLOAT:
                        return f"({inner_str} == 0.0)"
                # Wrap BinaryOp in parens for all unary operators
                if isinstance(operand, BinaryOp):
                    return f"{rb_op}({self._expr(operand)})"
                operand_str = self._expr(operand)
                # Add space for double negation to avoid Ruby's -- decrement
                if op == "-" and operand_str.startswith("-"):
                    return f"- {operand_str}"
                return f"{rb_op}{operand_str}"
            case Ternary(cond=cond, then_expr=then_expr, else_expr=else_expr):
                return f"{self._cond_expr(cond)} ? {self._expr(then_expr)} : {self._expr(else_expr)}"
            case Cast(expr=inner, to_type=to_type):
                if (
                    isinstance(to_type, Primitive)
                    and to_type.kind in ("int", "byte", "rune")
                    and inner.typ == BOOL
                ):
                    # Skip coercion if inner is unary minus - already produces int
                    if isinstance(inner, UnaryOp) and inner.op == "-":
                        return self._expr(inner)
                    # Skip coercion if inner is MinExpr/MaxExpr - already produces int
                    if isinstance(inner, (MinExpr, MaxExpr)):
                        return self._expr(inner)
                    return f"({self._expr(inner)} ? 1 : 0)"
                if (
                    isinstance(to_type, Primitive)
                    and to_type.kind == "string"
                    and inner.typ == BOOL
                ):
                    return f'({self._expr(inner)} ? "True" : "False")'
                if to_type == Primitive(kind="string") and isinstance(inner.typ, Slice):
                    if inner.typ.element == Primitive(kind="byte"):
                        # TonguesBytes has .to_s method
                        return f"{self._expr(inner)}.to_s"
                    return f"{self._expr(inner)}.pack('C*').force_encoding('UTF-8').scrub(\"\\uFFFD\")"
                if to_type == Primitive(kind="string") and inner.typ == Primitive(
                    kind="rune"
                ):
                    return f"[{self._expr(inner)}].pack('U')"
                if isinstance(to_type, Slice) and to_type.element == Primitive(
                    kind="byte"
                ):
                    self._needs_bytes = True
                    return f"TonguesBytes.from_str({self._expr(inner)})"
                if to_type == Primitive(kind="int") and inner.typ in (
                    Primitive(kind="string"),
                    Primitive(kind="byte"),
                    Primitive(kind="rune"),
                ):
                    return f"{self._expr(inner)}.ord"
                # float -> int conversion
                if to_type == INT and inner.typ == FLOAT:
                    expr_str = self._expr(inner)
                    if isinstance(inner, (BinaryOp, UnaryOp, Ternary)):
                        return f"({expr_str}).to_i"
                    return f"{expr_str}.to_i"
                # int -> float conversion
                if to_type == FLOAT and inner.typ == INT:
                    expr_str = self._expr(inner)
                    if isinstance(inner, (BinaryOp, UnaryOp, Ternary)):
                        return f"({expr_str}).to_f"
                    return f"{expr_str}.to_f"
                # string -> float conversion (handles inf/nan)
                if to_type == FLOAT and inner.typ == Primitive(kind="string"):
                    if isinstance(inner, StringLit):
                        val = inner.value.strip().lower()
                        if val in ("inf", "infinity"):
                            return "Float::INFINITY"
                        if val == "-inf":
                            return "-Float::INFINITY"
                        if val == "nan":
                            return "Float::NAN"
                    expr_str = self._expr(inner)
                    return f"{expr_str}.to_f"
                if isinstance(to_type, Primitive) and to_type.kind == "string":
                    # str(None) should return "None", not "" (Ruby's nil.to_s)
                    if isinstance(inner, NilLit):
                        return '"None"'
                    return f"{self._expr(inner)}.to_s"
                return self._expr(inner)
            case TypeAssert(expr=inner):
                return self._expr(inner)
            case IsType(expr=inner, tested_type=tested_type):
                type_name = self._type_name_for_check(tested_type)
                return f"{self._expr(inner)}.is_a?({type_name})"
            case IsNil(expr=inner, negated=negated):
                if negated:
                    return f"!{self._expr(inner)}.nil?"
                return f"{self._expr(inner)}.nil?"
            case Len(expr=inner):
                return f"{self._expr(inner)}.length"
            case MakeSlice(element_type=element_type, length=length):
                if element_type == Primitive(kind="byte"):
                    self._needs_bytes = True
                    if length is not None:
                        return f"TonguesBytes.zeros({self._expr(length)})"
                    return "TonguesBytes.new([])"
                if length is not None:
                    zero = self._zero_value(element_type)
                    return f"Array.new({self._expr(length)}, {zero})"
                return "[]"
            case MakeMap():
                return "{}"
            case SliceLit(element_type=element_type, elements=elements):
                elems = ", ".join(self._expr(e) for e in elements)
                if element_type == Primitive(kind="byte"):
                    self._needs_bytes = True
                    return f"TonguesBytes.new([{elems}])"
                return f"[{elems}]"
            case MapLit(entries=entries) as ml:
                if not entries:
                    return "{}"
                # Use expr.typ.key for coercion (may be same as key_type if no context)
                map_key_type = ml.typ.key if isinstance(ml.typ, Map) else ml.key_type
                pairs = ", ".join(
                    f"{self._coerce_map_key(map_key_type, k)} => {self._expr(v)}"
                    for k, v in entries
                )
                return f"{{{pairs}}}"
            case SetLit(elements=elements):
                self._needs_set = True
                if not elements:
                    return "Set.new"
                elems = ", ".join(self._expr(e) for e in elements)
                return f"Set[{elems}]"
            case StructLit(struct_name=struct_name, fields=fields):
                non_none = [
                    (k, v) for k, v in fields.items() if not isinstance(v, NilLit)
                ]
                args = ", ".join(
                    f"{_safe_name(k)}: {self._expr(v)}" for k, v in non_none
                )
                return f"{_safe_type_name(struct_name)}.new({args})"
            case TupleLit(elements=elements):
                elems = ", ".join(self._expr(e) for e in elements)
                return f"[{elems}]"
            case StringConcat(parts=parts):
                return " + ".join(self._expr(p) for p in parts)
            case StringFormat(template=template, args=args):
                return self._format_string(template, args)
            case SliceConvert(source=source):
                return self._expr(source)
            case ListComp(element=element, generators=generators):
                return self._list_comp(element, generators)
            case SetComp(element=element, generators=generators):
                return self._set_comp(element, generators)
            case DictComp(key=key, value=value, generators=generators):
                return self._dict_comp(key, value, generators)
            case _:
                raise NotImplementedError("Unknown expression")

    def _list_comp(self, element: Expr, generators: list[CompGenerator]) -> str:
        """Emit list comprehension as Ruby lambda IIFE."""
        body_lines: list[str] = ["_result = []"]
        body_stmt = f"_result << {self._expr(element)}"
        self._emit_comp_loops(generators, 0, body_lines, body_stmt)
        body_lines.append("_result")
        body = "; ".join(body_lines)
        return f"(-> {{ {body} }}).call"

    def _set_comp(self, element: Expr, generators: list[CompGenerator]) -> str:
        """Emit set comprehension as Ruby lambda IIFE."""
        body_lines: list[str] = ["_result = Set.new"]
        body_stmt = f"_result << {self._expr(element)}"
        self._emit_comp_loops(generators, 0, body_lines, body_stmt)
        body_lines.append("_result")
        body = "; ".join(body_lines)
        return f"(-> {{ {body} }}).call"

    def _dict_comp(
        self, key: Expr, value: Expr, generators: list[CompGenerator]
    ) -> str:
        """Emit dict comprehension as Ruby lambda IIFE."""
        body_lines: list[str] = ["_result = {}"]
        body_stmt = f"_result[{self._expr(key)}] = {self._expr(value)}"
        self._emit_comp_loops(generators, 0, body_lines, body_stmt)
        body_lines.append("_result")
        body = "; ".join(body_lines)
        return f"(-> {{ {body} }}).call"

    def _emit_comp_loops(
        self,
        generators: list[CompGenerator],
        idx: int,
        lines: list[str],
        body_stmt: str,
    ) -> None:
        """Recursively emit nested loops for comprehension generators."""
        if idx >= len(generators):
            lines.append(body_stmt)
            return
        gen = generators[idx]
        iter_expr = self._expr(gen.iterable)
        if len(gen.targets) == 1:
            target = _safe_name(gen.targets[0])
            lines.append(f"{iter_expr}.each {{ |{target}|")
        else:
            targets = ", ".join(_safe_name(t) for t in gen.targets)
            lines.append(f"{iter_expr}.each {{ |({targets})|")
        for cond in gen.conditions:
            lines.append(f"next unless {self._expr(cond)}")
        self._emit_comp_loops(generators, idx + 1, lines, body_stmt)
        lines.append("}")

    def _emit_slice_assign(self, target: SliceLV, value: Expr) -> None:
        """Emit slice assignment: arr[lo:hi] = value."""
        obj_str = self._expr(target.obj)
        val_str = self._expr(value)
        low = self._expr(target.low) if target.low else "0"
        if target.high is None:
            # arr[lo:] = value -> arr[lo..] = value
            self._line(f"{obj_str}[{low}..] = {val_str}")
        else:
            high = self._expr(target.high)
            # arr[lo:hi] = value -> arr[lo...hi] = value (exclusive end)
            self._line(f"{obj_str}[{low}...{high}] = {val_str}")

    def _slice_expr(
        self, obj: Expr, low: Expr | None, high: Expr | None, step: Expr | None = None
    ) -> str:
        obj_str = self._expr(obj)
        is_bytes = isinstance(obj.typ, Slice) and obj.typ.element == Primitive(
            kind="byte"
        )
        # Handle step (e.g., arr[::2] or arr[::-1])
        if step is not None:
            step_val = self._expr(step)
            if low and (neg_idx := self._negative_index(obj, low)):
                low_str = neg_idx
            else:
                low_str = self._expr(low) if low else "0"
            if high and (neg_idx := self._negative_index(obj, high)):
                high_str = neg_idx
            else:
                high_str = self._expr(high) if high else ""
            # Negative step: reverse direction
            # step_val is the Ruby expression like "-1" or "-2"
            if step_val.startswith("-"):
                abs_step = step_val[1:]  # Remove the minus sign
                if is_bytes:
                    if not low and not high:
                        if abs_step == "1":
                            return f"TonguesBytes.new({obj_str}.data.reverse)"
                        return f"TonguesBytes.new({obj_str}.data.reverse.each_slice({abs_step}).map(&:first))"
                    # arr[high:low:-step] - from high down to low+1
                    if low and high:
                        # Ruby: arr[(low+1)..high].reverse
                        base = f"{obj_str}[({high_str} + 1)..{low_str}].data.reverse"
                    elif low:
                        base = f"{obj_str}[0..{low_str}].data.reverse"
                    else:
                        base = f"{obj_str}[({high_str} + 1)..].data.reverse"
                    if abs_step == "1":
                        return f"TonguesBytes.new({base})"
                    return (
                        f"TonguesBytes.new({base}.each_slice({abs_step}).map(&:first))"
                    )
                if not low and not high:
                    if abs_step == "1":
                        return f"{obj_str}.reverse"
                    return f"{obj_str}.reverse.each_slice({abs_step}).map(&:first)"
                # arr[high:low:-step] - from high down to low+1
                if low and high:
                    # Ruby: arr[(low+1)..high].reverse
                    base = f"{obj_str}[({high_str} + 1)..{low_str}].reverse"
                elif low:
                    base = f"{obj_str}[0..{low_str}].reverse"
                else:
                    base = f"{obj_str}[({high_str} + 1)..].reverse"
                if abs_step == "1":
                    return base
                return f"{base}.each_slice({abs_step}).map(&:first)"
            # Positive step: select every nth element
            is_string = obj.typ == Primitive(kind="string")
            if is_bytes:
                if not low and not high:
                    return f"TonguesBytes.new({obj_str}.data.each_slice({step_val}).map(&:first))"
                if high_str:
                    return f"TonguesBytes.new({obj_str}[{low_str}...{high_str}].data.each_slice({step_val}).map(&:first))"
                return f"TonguesBytes.new({obj_str}[{low_str}..].data.each_slice({step_val}).map(&:first))"
            if is_string:
                # For strings, convert to chars, slice, then join back
                if not low and not high:
                    return f"{obj_str}.chars.each_slice({step_val}).map(&:first).join"
                if high_str:
                    return f"{obj_str}[{low_str}...{high_str}].chars.each_slice({step_val}).map(&:first).join"
                return f"{obj_str}[{low_str}..].chars.each_slice({step_val}).map(&:first).join"
            if not low and not high:
                return f"{obj_str}.each_slice({step_val}).map(&:first)"
            if high_str:
                return f"{obj_str}[{low_str}...{high_str}].each_slice({step_val}).map(&:first)"
            return f"{obj_str}[{low_str}..].each_slice({step_val}).map(&:first)"
        clamp_low = False
        if low and (neg_idx := self._negative_index(obj, low)):
            low_str = neg_idx
            clamp_low = True  # len(x) - N pattern needs clamping
        else:
            low_str = self._expr(low) if low else "0"
            if low and self._is_negative_literal(low):
                clamp_low = True
        if high and (neg_idx := self._negative_index(obj, high)):
            high_str = neg_idx
        else:
            high_str = self._expr(high) if high else ""
        # Clamp negative start indices: Ruby returns nil if negative index goes past start
        # Python clamps to 0, so items[-100:2] gives items[0:2]
        if clamp_low:
            low_str = f"[{low_str}, -{obj_str}.length].max"
        # Ruby returns nil for out-of-bounds slices, Python returns []
        # Use || [] to match Python semantics
        if high_str:
            return f"({obj_str}[{low_str}...{high_str}] || [])"
        return f"({obj_str}[{low_str}..] || [])"

    def _negative_index(self, obj: Expr, index: Expr) -> str | None:
        """Detect len(obj) - N and return -N as string, or None if no match."""
        if not isinstance(index, BinaryOp) or index.op != "-":
            return None
        if not isinstance(index.left, Len) or not isinstance(index.right, IntLit):
            return None
        if self._expr(index.left.expr) != self._expr(obj):
            return None
        return f"-{index.right.value}"

    def _is_negative_literal(self, expr: Expr) -> bool:
        """Check if expression is a negative integer literal."""
        return (
            isinstance(expr, UnaryOp)
            and expr.op == "-"
            and isinstance(expr.operand, IntLit)
        )

    def _emit_map_lit_coerced(self, value: Expr, target_type: Type) -> str:
        """Emit expression, coercing MapLit keys if target type differs from literal type."""
        if not isinstance(value, MapLit) or not isinstance(target_type, Map):
            return self._expr(value)
        # Use target type's key type for coercion (from annotation, not literal inference)
        if not value.entries:
            return "{}"
        pairs = ", ".join(
            f"{self._coerce_map_key(target_type.key, k)} => {self._expr(v)}"
            for k, v in value.entries
        )
        return f"{{{pairs}}}"

    def _coerce_map_key(self, map_key_type: Type, key: Expr) -> str:
        """Emit key expression, coercing to match map key type if needed.

        Python treats 1, 1.0, and True as equivalent dict keys (same hash).
        Ruby doesn't, so we coerce keys to match the map's declared key type.

        For literals, we detect the actual value type from the node class since
        the frontend may have already set key.typ to match the map's key type.
        """
        if not isinstance(map_key_type, Primitive):
            return self._expr(key)
        map_key = map_key_type.kind
        # Handle literals by checking node type (frontend sets typ to match map)
        if isinstance(key, BoolLit):
            if map_key == "int":
                return "1" if key.value else "0"
            if map_key == "float":
                return "1.0" if key.value else "0.0"
        elif isinstance(key, IntLit):
            if map_key == "float":
                return f"{key.value}.0"
        elif isinstance(key, FloatLit):
            if map_key == "int" and key.value == int(key.value):
                return str(int(key.value))
        # For non-literals, check key.typ for coercion
        key_code = self._expr(key)
        if not isinstance(key.typ, Primitive):
            return key_code
        key_typ = key.typ.kind
        if map_key == key_typ:
            return key_code
        # BOOL variable → INT
        if map_key == "int" and key_typ == "bool":
            return f"({key_code} ? 1 : 0)"
        # FLOAT variable → INT
        if map_key == "int" and key_typ == "float":
            return f"({key_code}).to_i"
        # INT variable → FLOAT
        if map_key == "float" and key_typ == "int":
            return f"({key_code}).to_f"
        # BOOL variable → FLOAT
        if map_key == "float" and key_typ == "bool":
            return f"({key_code} ? 1.0 : 0.0)"
        return key_code

    def _format_string(self, template: str, args: list[Expr]) -> str:
        markers = {}
        result = template
        for i in range(len(args)):
            marker = f"\x00PLACEHOLDER{i}\x00"
            markers[marker] = i
            result = result.replace(f"{{{i}}}", marker, 1)
        pv_markers = []
        while "%v" in result:
            marker = f"\x00PV{len(pv_markers)}\x00"
            pv_markers.append(marker)
            result = result.replace("%v", marker, 1)
        result = result.replace("#", "\\#")
        for marker, i in markers.items():
            if i < len(args):
                result = result.replace(marker, f"#{{{self._expr(args[i])}}}")
        for j, marker in enumerate(pv_markers):
            if j < len(args):
                result = result.replace(marker, f"#{{{self._expr(args[j])}}}")
        if "\n" in result:
            return f'"{result}"'
        result = result.replace('"', '\\"')
        return f'"{result}"'

    def _lvalue(self, lv: LValue) -> str:
        match lv:
            case VarLV(name=name):
                if name == self.receiver_name:
                    return "self"
                return _safe_name(name)
            case FieldLV(obj=obj, field=field):
                return f"{self._expr(obj)}.{_safe_name(field)}"
            case IndexLV(obj=obj, index=index):
                if neg_idx := self._negative_index(obj, index):
                    return f"{self._expr(obj)}[{neg_idx}]"
                if isinstance(obj.typ, Map):
                    key_code = self._coerce_map_key(obj.typ.key, index)
                    return f"{self._expr(obj)}[{key_code}]"
                return f"{self._expr(obj)}[{self._expr(index)}]"
            case DerefLV(ptr=ptr):
                return self._expr(ptr)
            case _:
                raise NotImplementedError("Unknown lvalue")

    def _type(self, typ: Type) -> str:
        match typ:
            case Primitive(kind=kind):
                return _primitive_type(kind)
            case Slice(element=element):
                return "Array"
            case Array(element=element, size=size):
                return "Array"
            case Map(key=key, value=value):
                return "Hash"
            case Set(element=element):
                return "Set"
            case Tuple(elements=elements):
                return "Array"
            case Pointer(target=target):
                return self._type(target)
            case Optional(inner=inner):
                return self._type(inner)
            case StructRef(name=name):
                return _safe_type_name(name)
            case InterfaceRef(name=name):
                return _safe_type_name(name)
            case Union(name=name, variants=variants):
                if name:
                    return _safe_type_name(name)
                return "Object"
            case FuncType(params=params, ret=ret):
                return "Proc"
            case _:
                return "Object"

    def _type_name_for_check(self, typ: Type) -> str:
        match typ:
            case StructRef(name=name):
                return _safe_type_name(name)
            case InterfaceRef(name=name):
                # tuple and list both map to Array in Ruby
                if name in ("tuple", "list"):
                    return "Array"
                if name == "dict":
                    return "Hash"
                if name == "set":
                    return "Set"
                return _safe_type_name(name)
            case Primitive(kind="string"):
                return "String"
            case Primitive(kind="int"):
                return "Integer"
            case Primitive(kind="float"):
                return "Float"
            case Primitive(kind="bool"):
                return "TrueClass"
            case Slice() | Tuple():
                return "Array"
            case _:
                return self._type(typ)

    def _zero_value(self, typ: Type) -> str:
        match typ:
            case Primitive(kind="int") | Primitive(kind="byte"):
                return "0"
            case Primitive(kind="float"):
                return "0.0"
            case Primitive(kind="bool"):
                return "false"
            case Primitive(kind="string"):
                return '""'
            case Slice():
                return "[]"
            case Map():
                return "{}"
            case Set():
                self._needs_set = True
                return "Set.new"
            case _:
                return "nil"

    def _cond_expr(self, expr: Expr) -> str:
        """Emit a condition expression for ternary. In Ruby && and || have higher precedence than ?:."""
        return self._expr(expr)

    def _python_truthy_check(self, expr: Expr) -> str:
        """Emit a Python-style truthy check for an expression (for and/or semantics)."""
        expr_str = self._expr(expr)
        typ = expr.typ
        if isinstance(typ, (Slice, Map, Set)):
            return f"({expr_str} && !{expr_str}.empty?)"
        if isinstance(typ, Optional):
            inner = typ.inner
            if isinstance(inner, (Slice, Map, Set)):
                return f"({expr_str} && !{expr_str}.empty?)"
            return f"!{expr_str}.nil?"
        if isinstance(typ, Pointer):
            return f"!{expr_str}.nil?"
        if typ == Primitive(kind="string"):
            return f"({expr_str} && !{expr_str}.empty?)"
        if typ == INT:
            return f"{expr_str} != 0"
        if typ == FLOAT:
            return f"{expr_str} != 0.0"
        if typ == BOOL:
            return expr_str
        return f"!!{expr_str}"

    def _is_value_and_or(self, expr: Expr) -> bool:
        """Check if expression is a Truthy or value-returning and/or."""
        if isinstance(expr, Truthy):
            return True
        if isinstance(expr, BinaryOp) and expr.op in ("&&", "||"):
            return self._is_value_and_or(expr.left)
        return False

    def _extract_and_or_value(self, expr: Expr) -> tuple[Expr, str, str]:
        """Extract value, value string, and truthy check from and/or operand."""
        if isinstance(expr, Truthy):
            val = expr.expr
            val_str = self._expr(val)
            cond = self._python_truthy_check(val)
            return val, val_str, cond
        if isinstance(expr, BinaryOp) and expr.op in ("&&", "||"):
            # For chained and/or, the "value" is the whole expression
            # and the truthy check is on that value
            val_str = self._expr(expr)
            # The truthy check needs to be on the result type
            cond = self._python_truthy_check_str(val_str, expr.left)
            return expr, val_str, cond
        val_str = self._expr(expr)
        cond = self._python_truthy_check(expr)
        return expr, val_str, cond

    def _python_truthy_check_str(self, expr_str: str, sample_expr: Expr) -> str:
        """Generate truthy check when we have the string but need type info from a sample."""
        # For chained and/or, infer type from the innermost Truthy
        typ = self._get_innermost_type(sample_expr)
        if isinstance(typ, (Slice, Map, Set)):
            return f"({expr_str} && !{expr_str}.empty?)"
        if isinstance(typ, Optional):
            inner = typ.inner
            if isinstance(inner, (Slice, Map, Set)):
                return f"({expr_str} && !{expr_str}.empty?)"
            return f"!{expr_str}.nil?"
        if isinstance(typ, Pointer):
            return f"!{expr_str}.nil?"
        if typ == Primitive(kind="string"):
            return f"({expr_str} && !{expr_str}.empty?)"
        if typ == INT:
            return f"{expr_str} != 0"
        if typ == FLOAT:
            return f"{expr_str} != 0.0"
        if typ == BOOL:
            return expr_str
        return f"!!{expr_str}"

    def _get_innermost_type(self, expr: Expr) -> "Type":
        """Get the type of the innermost value in a chained and/or."""
        if isinstance(expr, Truthy):
            return expr.expr.typ
        if isinstance(expr, BinaryOp) and expr.op in ("&&", "||"):
            return self._get_innermost_type(expr.left)
        return expr.typ

    def _and_or_operand(self, expr: Expr) -> str:
        """Extract value from and/or operand, handling nested Truthy and BinaryOp."""
        if isinstance(expr, Truthy):
            return self._expr(expr.expr)
        if isinstance(expr, BinaryOp) and expr.op in ("&&", "||"):
            # Nested and/or - recursively emit value-preserving code
            return self._expr(expr)
        return self._expr(expr)

    def _coerce_bool_to_int(self, expr: Expr, raw: bool = False) -> str:
        """Coerce a bool expression to int for comparison with int."""
        if expr.typ == BOOL:
            # Unary minus on bool already produces int via _expr, skip double coercion
            if isinstance(expr, UnaryOp) and expr.op == "-":
                return self._expr(expr)
            # MinExpr/MaxExpr already produce int via _expr, skip double coercion
            if isinstance(expr, (MinExpr, MaxExpr)):
                return self._expr(expr)
            # BinaryOp with &&/|| and Truthy operands already preserves values
            if isinstance(expr, BinaryOp) and expr.op in ("&&", "||"):
                if self._is_value_and_or(expr.left):
                    return self._expr(expr)
            # Ruby's all?/any? return actual booleans, not 0/1 - don't coerce
            if isinstance(expr, Call) and expr.func in ("all", "any"):
                return self._expr(expr)
            inner = self._expr(expr)
            # Ternary has lower precedence than arithmetic, so wrap in parens
            return f"({inner} ? 1 : 0)"
        if raw:
            return self._expr(expr)
        return self._expr(expr)

    def _int_lit(self, value: int, fmt: str | None) -> str:
        """Format integer literal, preserving hex/oct/bin format."""
        if fmt == "hex":
            return hex(value)
        if fmt == "oct":
            return f"0o{oct(value)[2:]}"
        if fmt == "bin":
            return f"0b{bin(value)[2:]}"
        return str(value)

    def _float_lit(self, value: float, fmt: str | None) -> str:
        """Format float literal, preserving scientific notation."""
        if fmt == "exp":
            return ("%g" % value).replace("e+", "e")
        return str(value)

    def _maybe_paren(self, text: str, expr: Expr, parent_op: str, is_left: bool) -> str:
        """Wrap pre-rendered text in parens if the original expr needs it."""
        match expr:
            case BinaryOp(op=child_op):
                if _needs_parens(child_op, parent_op, is_left):
                    return f"({text})"
            case Ternary():
                return f"({text})"
            case UnaryOp(op="-") if parent_op == "**" and is_left:
                # Ruby ** binds tighter than unary -, so (-2) ** 3 needs parens
                return f"({text})"
        return text

    def _maybe_paren_expr(self, expr: Expr, parent_op: str, is_left: bool) -> str:
        """Wrap expression in parens if needed for operator precedence."""
        return self._maybe_paren(self._expr(expr), expr, parent_op, is_left)


def _primitive_type(kind: str) -> str:
    match kind:
        case "string":
            return "String"
        case "int":
            return "Integer"
        case "float":
            return "Float"
        case "bool":
            return "TrueClass"
        case "byte":
            return "Integer"
        case "rune":
            return "String"
        case "void":
            return "NilClass"
        case _:
            return "Object"


def _is_dict_view(expr: Expr) -> bool:
    """Check if expression is a dict view (keys/items/values on a Map)."""
    if isinstance(expr, MethodCall):
        if expr.method in ("keys", "values", "items"):
            if isinstance(expr.receiver_type, Map):
                return True
    return False


def _method_name(method: str, receiver_type: Type) -> str:
    """Convert method name for Ruby idioms."""
    if isinstance(receiver_type, Slice):
        if method == "append":
            return "push"
        if method == "copy":
            return "dup"
        if method == "extend":
            return "concat"
        # remove is handled specially in _expr (delete_at + index)
        # Python's in-place sort/reverse -> Ruby's bang methods
        if method == "sort":
            return "sort!"
        if method == "reverse":
            return "reverse!"
    # Python string methods to Ruby
    if method == "startswith":
        return "start_with?"
    if method == "endswith":
        return "end_with?"
    if method == "find":
        return "index"
    if method == "rfind":
        return "rindex"
    if method == "replace":
        return "gsub"
    # upper/lower for strings only (bytes have their own upper/lower methods)
    if receiver_type == STRING:
        if method == "upper":
            return "upcase"
        if method == "lower":
            return "downcase"
    if method == "isdigit":
        return "match?(/\\A\\d+\\z/)"
    if method == "isalpha":
        return "match?(/\\A[a-zA-Z]+\\z/)"
    return _safe_name(method)


def _binary_op(op: str) -> str:
    match op:
        case "and" | "&&":
            return "&&"
        case "or" | "||":
            return "||"
        case _:
            return op


def _unary_op(op: str) -> str:
    match op:
        case "!":
            return "!"
        case "&" | "*":
            return ""
        case _:
            return op


# Ruby operator precedence (higher number = tighter binding).
# From ruby-doc.org/docs/ruby-doc-bundle/Manual/man-1.4/syntax.html#op
_PRECEDENCE = {
    "or": 1,
    "and": 2,
    "||": 3,
    "&&": 4,
    "==": 5,
    "!=": 5,
    "<=>": 5,
    "<": 6,
    ">": 6,
    "<=": 6,
    ">=": 6,
    "|": 7,
    "^": 7,
    "&": 8,
    "<<": 9,
    ">>": 9,
    "+": 10,
    "-": 10,
    "*": 11,
    "/": 11,
    "//": 11,
    "%": 11,
    "**": 12,
}


def _needs_parens(child_op: str, parent_op: str, is_left: bool) -> bool:
    """Determine if a child binary op needs parens inside a parent binary op."""
    child_prec = _PRECEDENCE.get(child_op, 0)
    parent_prec = _PRECEDENCE.get(parent_op, 0)
    if child_prec < parent_prec:
        return True
    if child_prec == parent_prec and not is_left:
        return child_op in ("==", "!=", "<", ">", "<=", ">=")
    # Chained comparisons: (a != 0) == true, not a != (0 == true)
    if parent_op in ("==", "!=") and child_op in ("==", "!=", "<", ">", "<=", ">="):
        return True
    return False


def _escape_ruby_string(value: str) -> str:
    """Escape a string for Ruby double-quoted literals.

    Ruby interprets #{...}, #$..., and #@... as interpolation in double-quoted
    strings. We need to escape # when followed by { $ or @.
    """
    result = escape_string(value)
    # Escape # before interpolation triggers: { $ @
    out = []
    i = 0
    while i < len(result):
        c = result[i]
        if c == "#" and i + 1 < len(result) and result[i + 1] in "{$@":
            out.append("\\#")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _string_literal(value: str) -> str:
    return f'"{_escape_ruby_string(value)}"'


def _extract_range_pattern(
    init: Stmt | None, cond: Expr | None, post: Stmt | None
) -> tuple[str, Expr] | None:
    """Extract simple range iteration pattern: for i := 0; i < len(x); i++."""
    if init is None or cond is None or post is None:
        return None
    if not isinstance(init, VarDecl):
        return None
    if not isinstance(init.value, IntLit) or init.value.value != 0:
        return None
    var_name = init.name
    if not isinstance(cond, BinaryOp) or cond.op != "<":
        return None
    if not isinstance(cond.left, Var) or cond.left.name != var_name:
        return None
    if not isinstance(cond.right, Len):
        return None
    iterable_expr = cond.right.expr
    if isinstance(post, OpAssign):
        if post.op != "+" or not isinstance(post.target, VarLV):
            return None
        if post.target.name != var_name:
            return None
        if not isinstance(post.value, IntLit) or post.value.value != 1:
            return None
    elif isinstance(post, Assign):
        if not isinstance(post.target, VarLV) or post.target.name != var_name:
            return None
        if not isinstance(post.value, BinaryOp) or post.value.op != "+":
            return None
        if not isinstance(post.value.left, Var) or post.value.left.name != var_name:
            return None
        if not isinstance(post.value.right, IntLit) or post.value.right.value != 1:
            return None
    else:
        return None
    return (var_name, iterable_expr)
