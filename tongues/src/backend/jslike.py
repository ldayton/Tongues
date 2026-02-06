"""Base backend for JavaScript-like languages (JS, TS).

Shared logic for JS and TS code generation. Subclasses override hooks
for type annotations and language-specific features.
"""

from __future__ import annotations

from src.backend.util import escape_string
from src.ir import (
    BOOL,
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
    DerefLV,
    DictComp,
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
    Function,
    If,
    Index,
    IndexLV,
    IntLit,
    IntToStr,
    InterfaceDef,
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
    SliceLV,
    SliceExpr,
    SliceLit,
    SoftFail,
    StaticCall,
    Stmt,
    StringConcat,
    StringFormat,
    StringLit,
    Struct,
    StructLit,
    StructRef,
    InterfaceRef,
    TupleAssign,
    TupleLit,
    Ternary,
    TryCatch,
    TrimChars,
    Truthy,
    Tuple,
    Type,
    TypeAssert,
    TypeCase,
    TypeSwitch,
    UnaryOp,
    Var,
    VarDecl,
    VarLV,
    While,
)


class JsLikeBackend:
    """Base class for JavaScript-like code generators."""

    def __init__(self) -> None:
        self.indent = 0
        self.lines: list[str] = []
        self.receiver_name: str | None = None
        self.current_struct: str | None = None
        self.struct_fields: dict[str, list[str]] = {}
        self._struct_field_count: dict[str, int] = {}

    def emit(self, module: Module) -> str:
        """Emit code from IR Module."""
        self.indent = 0
        self.lines = []
        self.struct_fields = {}
        self._struct_field_count = {}
        for struct in module.structs:
            self.struct_fields[struct.name] = [f.name for f in struct.fields]
            self._struct_field_count[struct.name] = len(struct.fields)
        self._emit_module(module)
        return "\n".join(self.lines)

    def _line(self, text: str = "") -> None:
        if text:
            self.lines.append("  " * self.indent + text)
        else:
            self.lines.append("")

    def _get_public_symbols(self, module: Module) -> list[str]:
        """Collect public (non-underscore) symbols for export."""
        symbols = []
        for func in module.functions:
            if not func.name.startswith("_"):
                symbols.append(_camel(func.name))
        for struct in module.structs:
            if not struct.name.startswith("_"):
                symbols.append(_safe_name(struct.name))
        return symbols

    # --- Hooks for subclasses ---

    def _emit_preamble(self, module: Module) -> bool:
        """Emit language-specific preamble. Return True if anything was emitted."""
        raise NotImplementedError

    def _emit_interface(self, iface: InterfaceDef) -> None:
        """Emit interface definition (TS only, JS is a no-op)."""
        raise NotImplementedError

    def _emit_field(self, fld: Field) -> None:
        """Emit field declaration (TS only, JS is a no-op)."""
        raise NotImplementedError

    def _func_signature(self, name: str, params: list[Param], ret: Type) -> str:
        """Return function signature string."""
        raise NotImplementedError

    def _method_signature(self, name: str, params: list[Param], ret: Type) -> str:
        """Return method signature string."""
        raise NotImplementedError

    def _param_list(self, params: list[Param]) -> str:
        """Return parameter list string."""
        raise NotImplementedError

    def _var_decl(self, name: str, typ: Type | None, value: Expr | None) -> None:
        """Emit variable declaration."""
        raise NotImplementedError

    def _assign_decl(self, lv: str, value: Expr) -> None:
        """Emit assignment that is a declaration."""
        raise NotImplementedError

    def _tuple_assign_decl(self, lvalues: str, value: Expr, value_type: Type | None) -> None:
        """Emit tuple assignment that is a declaration."""
        raise NotImplementedError

    def _for_value_decl(
        self, name: str, iter_expr: str, index_name: str | None, elem_type: str
    ) -> None:
        """Emit loop value variable declaration."""
        raise NotImplementedError

    def _emit_exports(self, symbols: list[str]) -> None:
        """Emit module exports."""
        raise NotImplementedError

    def _hoisted_vars_hook(self, stmt: Stmt) -> None:
        """Hook for hoisted variable handling (JS only)."""
        pass

    # --- Module structure ---

    def _emit_module(self, module: Module) -> None:
        if module.doc:
            self._line(f"/** {module.doc} */")
        need_blank = self._emit_preamble(module)
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
        else:
            symbols = self._get_public_symbols(module)
            if symbols:
                self._line()
                self._emit_exports(symbols)

    def _emit_constant(self, const: Constant) -> None:
        val = self._expr(const.value)
        self._line(f"const {const.name} = {val};")

    def _emit_struct(self, struct: Struct) -> None:
        if struct.doc:
            self._line(f"/** {struct.doc} */")
        extends = ""
        if struct.embedded_type:
            extends = f" extends {struct.embedded_type}"
        implements = self._struct_implements(struct)
        self._line(f"class {_safe_name(struct.name)}{extends}{implements} {{")
        self.indent += 1
        for fld in struct.fields:
            self._emit_field(fld)
        if struct.fields:
            self._emit_constructor(struct)
        self.current_struct = struct.name
        for i, method in enumerate(struct.methods):
            if i > 0 or struct.fields:
                self._line()
            self._emit_method(method)
        self.current_struct = None
        self.indent -= 1
        self._line("}")

    def _struct_implements(self, struct: Struct) -> str:
        """Return implements clause (TS only, JS returns empty)."""
        return ""

    def _emit_constructor(self, struct: Struct) -> None:
        params = ", ".join(
            f"{_camel(f.name)} = {self._default_value(f.typ)}" for f in struct.fields
        )
        self._line(f"constructor({params}) {{")
        self.indent += 1
        for fld in struct.fields:
            name = _camel(fld.name)
            is_nullable_slice = isinstance(fld.typ, Optional) and isinstance(
                fld.typ.inner, (Slice, Array)
            )
            if isinstance(fld.typ, (Slice, Array)) and not is_nullable_slice:
                self._line(f"this.{name} = {name} ?? [];")
            else:
                self._line(f"this.{name} = {name};")
        self.indent -= 1
        self._line("}")

    def _default_value(self, typ: Type) -> str:
        """Return default value for a type."""
        if isinstance(typ, Primitive):
            if typ.kind == "string":
                return '""'
            if typ.kind in ("int", "float", "byte", "rune"):
                return "0"
            if typ.kind == "bool":
                return "false"
        if isinstance(typ, Optional):
            return "null"
        if isinstance(typ, (Slice, Array)):
            return "[]"
        if isinstance(typ, Map):
            return "new Map()"
        if isinstance(typ, Set):
            return "new Set()"
        return "null"

    def _emit_function(self, func: Function) -> None:
        self._pre_function_hook()
        if func.doc:
            self._line(f"/** {func.doc} */")
        sig = self._func_signature(func.name, func.params, func.ret)
        self._line(f"{sig} {{")
        self.indent += 1
        for stmt in func.body:
            self._emit_stmt(stmt)
        self._post_function_body(func)
        self.indent -= 1
        self._line("}")

    def _pre_function_hook(self) -> None:
        """Hook called before emitting a function (JS clears hoisted vars)."""

    def _post_function_body(self, func: Function) -> None:
        """Hook called after function body (JS adds implicit return null)."""

    def _emit_method(self, func: Function) -> None:
        self._pre_function_hook()
        if func.doc:
            self._line(f"/** {func.doc} */")
        sig = self._method_signature(func.name, func.params, func.ret)
        self._line(f"{sig} {{")
        self.indent += 1
        if func.receiver:
            self.receiver_name = func.receiver.name
        for stmt in func.body:
            self._emit_stmt(stmt)
        self.receiver_name = None
        self.indent -= 1
        self._line("}")

    # --- Statements ---

    def _emit_stmt(self, stmt: Stmt) -> None:
        self._hoisted_vars_hook(stmt)
        match stmt:
            case VarDecl(name=name, typ=typ, value=value):
                self._var_decl(name, typ, value)
            case Assign(target=target, value=value):
                if isinstance(target, SliceLV):
                    self._emit_slice_assign(target, value)
                else:
                    lv = self._lvalue(target)
                    if stmt.is_declaration:
                        self._assign_decl(lv, value)
                    else:
                        val = self._expr(value)
                        self._line(f"{lv} = {val};")
            case TupleAssign(targets=targets, value=value):
                lvalues = ", ".join(self._lvalue(t) for t in targets)
                if stmt.is_declaration:
                    self._tuple_assign_decl(lvalues, value, value.typ)
                else:
                    self._emit_tuple_reassign(stmt, targets, lvalues, value)
            case OpAssign(target=target, op=op, value=value):
                lv = self._lvalue(target)
                val = self._expr(value)
                self._line(f"{lv} {op}= {val};")
            case NoOp():
                pass
            case ExprStmt(expr=expr):
                self._line(f"{self._expr(expr)};")
            case Return(value=value):
                if value is not None:
                    self._line(f"return {self._expr(value)};")
                else:
                    self._line("return;")
            case Assert(test=test, message=message):
                cond_str = self._expr(test)
                if message is not None:
                    self._line(f"if (!({cond_str})) {{ throw new Error({self._expr(message)}); }}")
                else:
                    self._line(f'if (!({cond_str})) {{ throw new Error("Assertion failed"); }}')
            case EntryPoint(function_name=function_name):
                self._line(f"process.exitCode = {_camel(function_name)}();")
            case If(cond=cond, then_body=then_body, else_body=else_body, init=init):
                if init is not None:
                    self._emit_stmt(init)
                self._line(f"if ({self._expr(cond)}) {{")
                self.indent += 1
                for s in then_body:
                    self._emit_stmt(s)
                self.indent -= 1
                self._emit_else_body(else_body)
            case TypeSwitch(expr=expr, binding=binding, cases=cases, default=default):
                self._emit_type_switch(expr, binding, cases, default)
            case Match(expr=expr, cases=cases, default=default):
                self._emit_match(expr, cases, default)
            case ForRange(index=index, value=value, iterable=iterable, body=body):
                self._emit_for_range(index, value, iterable, body)
            case ForClassic(init=init, cond=cond, post=post, body=body):
                self._emit_for_classic(init, cond, post, body)
            case While(cond=cond, body=body):
                self._line(f"while ({self._expr(cond)}) {{")
                self.indent += 1
                for s in body:
                    self._emit_stmt(s)
                self.indent -= 1
                self._line("}")
            case Break(label=label):
                if label:
                    self._line(f"break {label};")
                else:
                    self._line("break;")
            case Continue(label=label):
                if label:
                    self._line(f"continue {label};")
                else:
                    self._line("continue;")
            case Block(body=body):
                for s in body:
                    self._emit_stmt(s)
            case TryCatch(body=body, catches=catches, reraise=reraise):
                self._emit_try_catch(body, catches, reraise)
            case Raise(error_type=error_type, message=message, pos=pos, reraise_var=reraise_var):
                self._emit_raise(error_type, message, pos, reraise_var)
            case SoftFail():
                self._line("return null;")
            case _:
                raise NotImplementedError("Unknown statement")

    def _emit_tuple_reassign(
        self, stmt: TupleAssign, targets: list[LValue], lvalues: str, value: Expr
    ) -> None:
        """Emit tuple reassignment (non-declaration). Override for JS hoisting."""
        val = self._expr(value)
        self._line(f"[{lvalues}] = {val};")

    def _emit_raise(
        self,
        error_type: str | None,
        message: Expr | None,
        pos: Expr | None,
        reraise_var: str | None,
    ) -> None:
        if reraise_var:
            self._line(f"throw {_camel(reraise_var)}")
        elif error_type and self._struct_field_count.get(error_type, 0) == 0:
            self._line(f"throw new {error_type}()")
        else:
            p = self._expr(pos)
            if isinstance(message, StringLit):
                msg_val = (
                    message.value.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
                )
                self._line(f"throw new {error_type}(`{msg_val} at position ${{{p}}}`, {p})")
            else:
                msg = self._expr(message)
                self._line(f"throw new {error_type}(`${{{msg}}} at position ${{{p}}}`, {p})")

    def _emit_slice_assign(self, target: SliceLV, value: Expr) -> None:
        """Emit slice assignment: arr[lo:hi] = value -> splice."""
        obj_str = self._expr(target.obj)
        val_str = self._expr(value)
        low = self._expr(target.low) if target.low else "0"
        if target.high is None:
            # arr[lo:] = value -> splice from lo to end
            self._line(f"{obj_str}.splice({low}, {obj_str}.length - {low}, ...{val_str});")
        else:
            high = self._expr(target.high)
            # arr[lo:hi] = value -> splice(lo, hi - lo, ...value)
            self._line(f"{obj_str}.splice({low}, {high} - {low}, ...{val_str});")

    def _emit_else_body(self, else_body: list[Stmt]) -> None:
        """Emit else body, converting single-If else to else-if chains."""
        if not else_body:
            self._line("}")
            return
        if len(else_body) == 1 and isinstance(else_body[0], If):
            elif_stmt = else_body[0]
            if elif_stmt.init is not None:
                self._emit_stmt(elif_stmt.init)
            self._line(f"}} else if ({self._expr(elif_stmt.cond)}) {{")
            self.indent += 1
            for s in elif_stmt.then_body:
                self._emit_stmt(s)
            self.indent -= 1
            self._emit_else_body(elif_stmt.else_body)
        else:
            self._line("} else {")
            self.indent += 1
            for s in else_body:
                self._emit_stmt(s)
            self.indent -= 1
            self._line("}")

    def _emit_type_switch(
        self, expr: Expr, binding: str, cases: list[TypeCase], default: list[Stmt]
    ) -> None:
        var = self._expr(expr)
        binding_name = _camel(binding)
        shadows = isinstance(expr, Var) and _camel(expr.name) == binding_name
        for i, case in enumerate(cases):
            keyword = "if" if i == 0 else "} else if"
            if isinstance(case.typ, Primitive):
                js_typeof = _typeof_check(case.typ.kind)
                self._line(f"{keyword} (typeof {var} === '{js_typeof}') {{")
            else:
                type_name = self._type_name_for_check(case.typ)
                self._line(f"{keyword} ({var} instanceof {type_name}) {{")
            self.indent += 1
            if not shadows:
                self._emit_type_switch_binding(binding_name, var, case.typ)
            for s in case.body:
                self._emit_stmt(s)
            self.indent -= 1
        if default:
            self._line("} else {")
            self.indent += 1
            for s in default:
                self._emit_stmt(s)
            self.indent -= 1
        self._line("}")

    def _emit_type_switch_binding(self, binding_name: str, var: str, typ: Type) -> None:
        """Emit binding in type switch case. Override for TS type annotation."""
        self._line(f"const {binding_name} = {var};")

    def _emit_match(self, expr: Expr, cases: list[MatchCase], default: list[Stmt]) -> None:
        self._line(f"switch ({self._expr(expr)}) {{")
        self.indent += 1
        for case in cases:
            for pattern in case.patterns:
                self._line(f"case {self._expr(pattern)}:")
            self.indent += 1
            for s in case.body:
                self._emit_stmt(s)
            if not _ends_with_return(case.body):
                self._line("break;")
            self.indent -= 1
        if default:
            self._line("default:")
            self.indent += 1
            for s in default:
                self._emit_stmt(s)
            self.indent -= 1
        self.indent -= 1
        self._line("}")

    def _emit_for_range(
        self,
        index: str | None,
        value: str | None,
        iterable: Expr,
        body: list[Stmt],
    ) -> None:
        if self._try_emit_map(index, value, iterable, body):
            return
        iter_expr = self._expr(iterable)
        iter_type = iterable.typ
        if index is not None and value is not None:
            self._line(
                f"for (var {_camel(index)} = 0; {_camel(index)} < {iter_expr}.length; {_camel(index)}++) {{"
            )
            self.indent += 1
            self._for_value_decl(
                _camel(value), iter_expr, _camel(index), self._element_type_str(iter_type)
            )
        elif value is not None:
            self._emit_for_of(value, iter_expr, iter_type)
            self.indent += 1
        elif index is not None:
            self._line(
                f"for (var {_camel(index)} = 0; {_camel(index)} < {iter_expr}.length; {_camel(index)}++) {{"
            )
            self.indent += 1
        else:
            self._line(f"for (const _ of {iter_expr}) {{")
            self.indent += 1
        for s in body:
            self._emit_stmt(s)
        self.indent -= 1
        self._line("}")

    def _emit_for_of(self, value: str, iter_expr: str, iter_type: Type | None) -> None:
        """Emit for-of loop header. Override for TS type casting."""
        self._line(f"for (const {_camel(value)} of {iter_expr}) {{")

    def _element_type_str(self, typ: Type | None) -> str:
        """Get element type string for loop variable. Override for TS."""
        return "any"

    def _try_emit_map(
        self,
        index: str | None,
        value: str | None,
        iterable: Expr,
        body: list[Stmt],
    ) -> bool:
        """Try to emit ForRange as .map(). Returns True if successful."""
        if index is not None or value is None:
            return False
        if len(body) != 1:
            return False
        stmt = body[0]
        if not isinstance(stmt, ExprStmt):
            return False
        expr = stmt.expr
        if not isinstance(expr, MethodCall):
            return False
        if expr.method != "append":
            return False
        if len(expr.args) != 1:
            return False
        if not isinstance(expr.obj, Var):
            return False
        transform_expr = expr.args[0]
        if not self._expr_uses_var(transform_expr, value):
            return False
        accumulator = self._expr(expr.obj)
        iter_expr = self._expr(iterable)
        transform = self._expr(transform_expr)
        self._line(f"{accumulator}.push(...{iter_expr}.map({_camel(value)} => {transform}));")
        return True

    def _expr_uses_var(self, expr: Expr, var_name: str) -> bool:
        """Check if an expression uses a variable (shallow check)."""
        if isinstance(expr, Var):
            return expr.name == var_name
        if isinstance(expr, MethodCall):
            if self._expr_uses_var(expr.obj, var_name):
                return True
            return any(self._expr_uses_var(a, var_name) for a in expr.args)
        if isinstance(expr, Call):
            return any(self._expr_uses_var(a, var_name) for a in expr.args)
        if isinstance(expr, FieldAccess):
            return self._expr_uses_var(expr.obj, var_name)
        if isinstance(expr, Index):
            return self._expr_uses_var(expr.obj, var_name) or self._expr_uses_var(
                expr.index, var_name
            )
        if isinstance(expr, BinaryOp):
            return self._expr_uses_var(expr.left, var_name) or self._expr_uses_var(
                expr.right, var_name
            )
        if isinstance(expr, UnaryOp):
            return self._expr_uses_var(expr.operand, var_name)
        if isinstance(expr, Ternary):
            return (
                self._expr_uses_var(expr.cond, var_name)
                or self._expr_uses_var(expr.then_expr, var_name)
                or self._expr_uses_var(expr.else_expr, var_name)
            )
        if isinstance(expr, Cast):
            return self._expr_uses_var(expr.expr, var_name)
        if isinstance(expr, SliceLit):
            return any(self._expr_uses_var(e, var_name) for e in expr.elements)
        if isinstance(expr, TupleLit):
            return any(self._expr_uses_var(e, var_name) for e in expr.elements)
        if isinstance(expr, StructLit):
            return any(self._expr_uses_var(v, var_name) for v in expr.fields.values())
        return False

    def _emit_for_classic(
        self,
        init: Stmt | None,
        cond: Expr | None,
        post: Stmt | None,
        body: list[Stmt],
    ) -> None:
        init_str = self._stmt_inline(init) if init else ""
        cond_str = self._expr(cond) if cond else ""
        post_str = self._stmt_inline(post) if post else ""
        self._line(f"for ({init_str}; {cond_str}; {post_str}) {{")
        self.indent += 1
        for s in body:
            self._emit_stmt(s)
        self.indent -= 1
        self._line("}")

    def _stmt_inline(self, stmt: Stmt) -> str:
        match stmt:
            case VarDecl(name=name, typ=typ, value=value):
                return self._var_decl_inline(name, typ, value)
            case Assign(
                target=VarLV(name=name),
                value=BinaryOp(op=op, left=Var(name=left_name), right=IntLit(value=1)),
            ) if name == left_name and op in ("+", "-"):
                return f"{_camel(name)}++" if op == "+" else f"{_camel(name)}--"
            case Assign(target=target, value=value):
                lv = self._lvalue(target)
                val = self._expr(value)
                if stmt.is_declaration:
                    return self._assign_decl_inline(lv, value)
                return f"{lv} = {val}"
            case OpAssign(target=target, op=op, value=value):
                return f"{self._lvalue(target)} {op}= {self._expr(value)}"
            case ExprStmt(expr=expr):
                return self._expr(expr)
            case _:
                raise NotImplementedError("Cannot inline")

    def _var_decl_inline(self, name: str, typ: Type | None, value: Expr | None) -> str:
        """Return inline variable declaration string."""
        raise NotImplementedError

    def _assign_decl_inline(self, lv: str, value: Expr) -> str:
        """Return inline assignment declaration string."""
        raise NotImplementedError

    def _emit_try_catch(
        self,
        body: list[Stmt],
        catches: list[CatchClause],
        reraise: bool,
    ) -> None:
        self._line("try {")
        self.indent += 1
        for s in body:
            self._emit_stmt(s)
        self.indent -= 1
        self._line("} catch (_e) {")
        self.indent += 1
        if not catches:
            if reraise:
                self._line("throw _e;")
            self.indent -= 1
            self._line("}")
            return

        if len(catches) == 1:
            clause = catches[0]
            if clause.var:
                self._line(f"let {_camel(clause.var)} = _e;")
            for s in clause.body:
                self._emit_stmt(s)
            if reraise:
                self._line("throw _e;")
            self.indent -= 1
            self._line("}")
            return

        emitted_chain = False
        has_default = False
        seen_conds: set[str] = set()
        for clause in catches:
            cond: str | None = None
            if isinstance(clause.typ, StructRef):
                exc_name = _PYTHON_EXCEPTION_MAP.get(clause.typ.name, clause.typ.name)
                cond = f"_e instanceof {exc_name}"
            if cond is not None and cond in seen_conds:
                continue
            if cond is not None:
                seen_conds.add(cond)
            if cond is None:
                if not emitted_chain:
                    if clause.var:
                        self._line(f"let {_camel(clause.var)} = _e;")
                    for s in clause.body:
                        self._emit_stmt(s)
                    if reraise:
                        self._line("throw _e;")
                    self.indent -= 1
                    self._line("}")
                    return
                self._line("else {")
                self.indent += 1
                if clause.var:
                    self._line(f"let {_camel(clause.var)} = _e;")
                for s in clause.body:
                    self._emit_stmt(s)
                if reraise:
                    self._line("throw _e;")
                self.indent -= 1
                self._line("}")
                emitted_chain = True
                has_default = True
                break

            keyword = "if" if not emitted_chain else "else if"
            self._line(f"{keyword} ({cond}) {{")
            self.indent += 1
            if clause.var:
                self._line(f"let {_camel(clause.var)} = _e;")
            for s in clause.body:
                self._emit_stmt(s)
            if reraise:
                self._line("throw _e;")
            self.indent -= 1
            self._line("}")
            emitted_chain = True

        if emitted_chain and not has_default:
            self._line("else {")
            self.indent += 1
            self._line("throw _e;")
            self.indent -= 1
            self._line("}")
        self.indent -= 1
        self._line("}")

    # --- Expressions ---

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
                return "null"
            case Var(name=name):
                if name == self.receiver_name:
                    return "this"
                return _camel(name)
            case FieldAccess(obj=obj, field=field):
                if field.startswith("F") and field[1:].isdigit():
                    return f"{self._expr(obj)}[{field[1:]}]"
                obj_str = self._expr(obj)
                field_str = _camel(field)
                return f"{obj_str}.{field_str}"
            case FuncRef(name=name, obj=obj):
                if obj is not None:
                    obj_str = self._expr(obj)
                    return f"{obj_str}.{_camel(name)}.bind({obj_str})"
                return _camel(name)
            case Index(obj=obj, index=index, typ=typ):
                return self._index_expr(obj, index, typ)
            case SliceExpr(obj=obj, low=low, high=high, step=step):
                return self._slice_expr(obj, low, high, step)
            case ParseInt(string=s, base=b):
                return f"parseInt({self._expr(s)}, {self._expr(b)})"
            case IntToStr(value=v):
                return f"String({self._expr(v)})"
            case CharClassify(kind=kind, char=char):
                regex_map = {
                    "digit": r"/^\d+$/",
                    "alpha": r"/^[a-zA-Z]+$/",
                    "alnum": r"/^[a-zA-Z0-9]+$/",
                    "space": r"/^\s+$/",
                    "upper": r"/^[A-Z]+$/",
                    "lower": r"/^[a-z]+$/",
                }
                return f"{regex_map[kind]}.test({self._expr(char)})"
            case TrimChars(string=s, chars=chars, mode=mode):
                return self._trim_chars(s, chars, mode)
            case Call(func="_intPtr", args=[arg]):
                return self._expr(arg)
            case Call(func="print", args=args):
                args_str = ", ".join(self._expr(a) for a in args)
                return f"console.log({args_str})"
            case Call(func="repr", args=[NilLit()]):
                return '"None"'
            case Call(func="repr", args=[arg]) if arg.typ == BOOL:
                return f'({self._expr(arg)} ? "True" : "False")'
            case Call(func="repr", args=[arg]):
                return f"String({self._expr(arg)})"
            case Call(func="bool", args=args):
                args_str = ", ".join(self._expr(a) for a in args)
                return f"Boolean({args_str})"
            case Call(func="abs", args=[arg]):
                return f"Math.abs({self._expr(arg)})"
            case Call(func="min", args=args):
                args_str = ", ".join(self._expr(a) for a in args)
                return f"Math.min({args_str})"
            case Call(func="max", args=args):
                args_str = ", ".join(self._expr(a) for a in args)
                return f"Math.max({args_str})"
            case Call(func="round", args=[arg]):
                return f"Math.round({self._expr(arg)})"
            case Call(func="round", args=[arg, IntLit(value=n)]):
                mult = 10**n
                return f"Math.round({self._expr(arg)} * {mult}) / {mult}"
            case Call(func="round", args=[arg, precision]):
                prec = self._expr(precision)
                return f"Math.round({self._expr(arg)} * 10 ** {prec}) / 10 ** {prec}"
            case Call(func="int", args=[arg]):
                return f"Math.trunc({self._expr(arg)})"
            case Call(func="divmod", args=[a, b]):
                return f"[Math.floor({self._expr(a)} / {self._expr(b)}), {self._expr(a)} % {self._expr(b)}]"
            case Call(func="pow", args=[base, exp]):
                return f"{self._expr(base)} ** {self._expr(exp)}"
            case Call(func="pow", args=[base, exp, mod]):
                return f"{self._expr(base)} ** {self._expr(exp)} % {self._expr(mod)}"
            case Call(func=func, args=args):
                return self._call_expr(func, args)
            case MethodCall(obj=obj, method="join", args=[arr], receiver_type=_):
                return self._join_expr(obj, arr)
            case MethodCall(
                obj=obj, method="extend", args=[other], receiver_type=receiver_type
            ) if _is_array_type(receiver_type):
                return f"{self._expr(obj)}.push(...{self._expr(other)})"
            case MethodCall(obj=obj, method="copy", args=[], receiver_type=receiver_type) if (
                _is_array_type(receiver_type)
            ):
                return f"{self._expr(obj)}.slice()"
            case MethodCall(obj=obj, method="get", args=[key], receiver_type=receiver_type) if (
                isinstance(receiver_type, Map)
            ):
                return self._map_get(obj, key, None)
            case MethodCall(
                obj=obj, method="get", args=[key, default], receiver_type=receiver_type
            ) if isinstance(receiver_type, Map):
                return self._map_get(obj, key, default)
            case MethodCall(
                obj=obj, method="replace", args=[StringLit(value=old_str), new], receiver_type=_
            ):
                escaped = _escape_regex_literal(old_str)
                return f"{self._expr(obj)}.replace(/{escaped}/g, {self._expr(new)})"
            case MethodCall(
                obj=obj, method=method, args=[TupleLit(elements=elements)], receiver_type=_
            ) if method in ("startswith", "endswith"):
                js_method = "startsWith" if method == "startswith" else "endsWith"
                obj_str = self._expr(obj)
                checks = [f"{obj_str}.{js_method}({self._expr(e)})" for e in elements]
                return f"({' || '.join(checks)})"
            case MethodCall(
                obj=obj, method="pop", args=[IntLit(value=0)], receiver_type=receiver_type
            ) if _is_array_type(receiver_type):
                return f"{self._expr(obj)}.shift()"
            case MethodCall(obj=obj, method=method, args=args, receiver_type=receiver_type):
                return self._method_call(obj, method, args, receiver_type)
            case StaticCall(on_type=on_type, method=method, args=args):
                args_str = ", ".join(self._expr(a) for a in args)
                type_name = self._type_name_for_check(on_type)
                return f"{type_name}.{_camel(method)}({args_str})"
            case Truthy(expr=e):
                return self._truthy_expr(e)
            case BinaryOp(op="in", left=left, right=right):
                return self._containment_check(left, right, negated=False)
            case BinaryOp(op="not in", left=left, right=right):
                return self._containment_check(left, right, negated=True)
            case BinaryOp(op="//", left=left, right=right):
                left_str = self._maybe_paren(left, "/", is_left=True)
                right_str = self._maybe_paren(right, "/", is_left=False)
                return f"Math.floor({left_str} / {right_str})"
            case BinaryOp(op=op, left=left, right=right):
                return self._binary_expr(op, left, right)
            case ChainedCompare(operands=operands, ops=ops):
                parts = []
                for i, op in enumerate(ops):
                    left_str = self._expr(operands[i])
                    right_str = self._expr(operands[i + 1])
                    js_op = _binary_op(op)
                    parts.append(f"{left_str} {js_op} {right_str}")
                return " && ".join(parts)
            case MinExpr(left=left, right=right):
                # Use ternary to preserve original value types (bools stay bools)
                l, r = self._expr(left), self._expr(right)
                return f"({l} <= {r} ? {l} : {r})"
            case MaxExpr(left=left, right=right):
                l, r = self._expr(left), self._expr(right)
                return f"({l} >= {r} ? {l} : {r})"
            case UnaryOp(op="&", operand=operand):
                return self._expr(operand)
            case UnaryOp(op="*", operand=operand):
                return self._expr(operand)
            case UnaryOp(op="!", operand=BinaryOp(op="!=", left=left, right=right)):
                left_str = self._maybe_paren(left, "===", is_left=True)
                right_str = self._maybe_paren(right, "===", is_left=False)
                return f"{left_str} === {right_str}"
            case UnaryOp(op="!", operand=BinaryOp(op="==", left=left, right=right)):
                left_str = self._maybe_paren(left, "!==", is_left=True)
                right_str = self._maybe_paren(right, "!==", is_left=False)
                return f"{left_str} !== {right_str}"
            case UnaryOp(op=op, operand=operand):
                inner = self._expr(operand)
                if isinstance(operand, UnaryOp):
                    if op in ("-", "+"):
                        return f"{op} {inner}"
                    return f"{op}{inner}"
                if isinstance(operand, BinaryOp):
                    inner = f"({inner})"
                return f"{op}{inner}"
            case Ternary(cond=cond, then_expr=then_expr, else_expr=else_expr):
                return f"{self._expr(cond)} ? {self._expr(then_expr)} : {self._expr(else_expr)}"
            case Cast(expr=inner, to_type=to_type):
                return self._cast_expr(inner, to_type)
            case TypeAssert(expr=inner, asserted=asserted):
                return self._type_assert(inner, asserted)
            case IsType(expr=inner, tested_type=tested_type):
                if isinstance(tested_type, Primitive):
                    js_typeof = _typeof_check(tested_type.kind)
                    return f"typeof {self._expr(inner)} === '{js_typeof}'"
                type_name = self._type_name_for_check(tested_type)
                return f"{self._expr(inner)} instanceof {type_name}"
            case IsNil(expr=inner, negated=negated):
                op = "!==" if negated else "==="
                return f"{self._expr(inner)} {op} null"
            case Len(expr=inner):
                return self._len_expr(inner)
            case MakeSlice(element_type=_, length=length, capacity=_):
                if length is not None:
                    return f"new Array({self._expr(length)})"
                return "[]"
            case MakeMap():
                return "new Map()"
            case SliceLit(elements=elements):
                elems = ", ".join(self._expr(e) for e in elements)
                return f"[{elems}]"
            case MapLit(entries=entries):
                if not entries:
                    return "new Map()"
                pairs = ", ".join(f"[{self._expr(k)}, {self._expr(v)}]" for k, v in entries)
                return f"new Map([{pairs}])"
            case SetLit(elements=elements):
                elems = ", ".join(self._expr(e) for e in elements)
                return f"new Set([{elems}])"
            case StructLit(struct_name=struct_name, fields=fields):
                return self._struct_lit(struct_name, fields)
            case TupleLit(elements=elements):
                elems = ", ".join(self._expr(e) for e in elements)
                return f"[{elems}]"
            case StringConcat(parts=parts):
                wrapped_parts: list[str] = []
                for p in parts:
                    expr_str = self._expr(p)
                    if " ?? " in expr_str:
                        wrapped_parts.append(f"({expr_str})")
                    else:
                        wrapped_parts.append(expr_str)
                return " + ".join(wrapped_parts)
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

    def _int_lit(self, value: int, fmt: str | None) -> str:
        if fmt == "hex":
            return hex(value)
        if fmt == "oct":
            return f"0o{oct(value)[2:]}"
        if fmt == "bin":
            return bin(value)
        if abs(value) > 9007199254740991:
            return f"{value}n"
        return str(value)

    def _float_lit(self, value: float, fmt: str | None) -> str:
        if fmt == "exp":
            return f"{value:g}".replace("e+", "e")
        return str(value)

    def _index_expr(self, obj: Expr, index: Expr, typ: Type | None) -> str:
        """Emit index expression. Override for Map handling."""
        obj_str = self._expr(obj)
        idx_str = self._expr(index)
        obj_type = obj.typ
        if (
            obj_type == STRING
            and isinstance(typ, Primitive)
            and typ.kind in ("int", "byte", "rune")
        ):
            return f"{obj_str}.charCodeAt({idx_str})"
        return f"{obj_str}[{idx_str}]"

    def _slice_expr(
        self, obj: Expr, low: Expr | None, high: Expr | None, step: Expr | None = None
    ) -> str:
        """Emit slice expression. Override for step handling."""
        obj_str = self._expr(obj)
        if low is None and high is None:
            return f"{obj_str}.slice()"
        elif low is None:
            return f"{obj_str}.slice(0, {self._expr(high)})"
        elif high is None:
            return f"{obj_str}.slice({self._expr(low)})"
        else:
            return f"{obj_str}.slice({self._expr(low)}, {self._expr(high)})"

    def _trim_chars(self, s: Expr, chars: Expr, mode: str) -> str:
        s_str = self._expr(s)
        if isinstance(chars, StringLit) and chars.value == " \t\n\r":
            method_map = {"left": "trimStart", "right": "trimEnd", "both": "trim"}
            return f"{s_str}.{method_map[mode]}()"
        escaped = _escape_regex_class(chars.value) if isinstance(chars, StringLit) else "..."
        if mode == "left":
            return f"{s_str}.replace(/^[{escaped}]+/, '')"
        elif mode == "right":
            return f"{s_str}.replace(/[{escaped}]+$/, '')"
        else:
            return f"{s_str}.replace(/^[{escaped}]+/, '').replace(/[{escaped}]+$/, '')"

    def _call_expr(self, func: str, args: list[Expr]) -> str:
        """Emit function call. Override for language-specific calls."""
        args_str = ", ".join(self._expr(a) for a in args)
        return f"{_camel(func)}({args_str})"

    def _join_expr(self, sep: Expr, arr: Expr) -> str:
        """Emit join expression. Override for bytes handling."""
        return f"{self._expr(arr)}.join({self._expr(sep)})"

    def _map_get(self, obj: Expr, key: Expr, default: Expr | None) -> str:
        """Emit Map.get expression. Override for JS/TS differences."""
        obj_str = self._expr(obj)
        key_str = self._expr(key)
        if default is not None:
            return f"{obj_str}.get({key_str}) ?? {self._expr(default)}"
        return f"({obj_str}.get({key_str}) ?? null)"

    def _method_call(self, obj: Expr, method: str, args: list[Expr], receiver_type: Type) -> str:
        """Emit method call. Override for bytes handling."""
        args_str = ", ".join(self._expr(a) for a in args)
        js_method = _method_name(method, receiver_type)
        return f"{self._expr(obj)}.{js_method}({args_str})"

    def _truthy_expr(self, e: Expr) -> str:
        inner_str = self._expr(e)
        inner_type = e.typ
        if isinstance(inner_type, (Map, Set)):
            return f"({inner_str}.size > 0)"
        if isinstance(inner_type, Slice) or inner_type == STRING:
            return f"({inner_str}.length > 0)"
        if isinstance(inner_type, Optional) and isinstance(inner_type.inner, (Map, Set)):
            return f"({inner_str} != null && {inner_str}.size > 0)"
        if isinstance(inner_type, Optional) and isinstance(inner_type.inner, Slice):
            return f"({inner_str} != null && {inner_str}.length > 0)"
        if inner_type == INT or (isinstance(inner_type, Primitive) and inner_type.kind == "float"):
            if isinstance(e, BinaryOp):
                return f"(({inner_str}) !== 0)"
            return f"({inner_str} !== 0)"
        return f"({inner_str} != null)"

    def _binary_expr(self, op: str, left: Expr, right: Expr) -> str:
        """Emit binary expression. Override for bytes handling."""
        js_op = _binary_op(op)
        if op in ("==", "!=") and _is_bool_int_compare(left, right):
            js_op = op
        # ** with unary on left needs parens
        if op == "**" and isinstance(left, UnaryOp):
            left_str = f"({self._expr(left)})"
        else:
            left_str = self._maybe_paren(left, op, is_left=True)
        right_str = self._maybe_paren(right, op, is_left=False)
        return f"{left_str} {js_op} {right_str}"

    def _cast_expr(self, inner: Expr, to_type: Type) -> str:
        """Emit cast expression. Override for language-specific handling."""
        if (
            isinstance(to_type, Primitive)
            and to_type.kind in ("int", "byte", "rune")
            and inner.typ == BOOL
        ):
            return f"Number({self._expr(inner)})"
        if isinstance(to_type, Primitive) and to_type.kind == "string" and inner.typ == BOOL:
            return f'({self._expr(inner)} ? "True" : "False")'
        if (
            isinstance(to_type, Slice)
            and isinstance(to_type.element, Primitive)
            and to_type.element.kind == "byte"
            and isinstance(inner.typ, Primitive)
            and inner.typ.kind == "string"
        ):
            return f"Array.from(new TextEncoder().encode({self._expr(inner)}))"
        if (
            isinstance(to_type, Primitive)
            and to_type.kind == "string"
            and isinstance(inner.typ, Slice)
            and isinstance(inner.typ.element, Primitive)
            and inner.typ.element.kind == "byte"
        ):
            return f"new TextDecoder().decode(new Uint8Array({self._expr(inner)}))"
        if (
            isinstance(to_type, Primitive)
            and to_type.kind == "string"
            and isinstance(inner.typ, Primitive)
            and inner.typ.kind in ("rune", "int")
        ):
            return f"String.fromCodePoint({self._expr(inner)})"
        if isinstance(to_type, Primitive) and to_type.kind == "string":
            return f"String({self._expr(inner)})"
        if (
            isinstance(to_type, Primitive)
            and to_type.kind in ("int", "byte", "rune")
            and isinstance(inner.typ, Primitive)
            and inner.typ.kind == "float"
        ):
            return f"Math.trunc({self._expr(inner)})"
        return self._expr(inner)

    def _type_assert(self, inner: Expr, asserted: Type) -> str:
        """Emit type assertion. JS ignores, TS uses 'as'."""
        return self._expr(inner)

    def _len_expr(self, inner: Expr) -> str:
        inner_type = inner.typ
        if isinstance(inner_type, (Map, Set)):
            return f"{self._expr(inner)}.size"
        return f"{self._expr(inner)}.length"

    def _struct_lit(self, struct_name: str, fields: dict[str, Expr]) -> str:
        field_names = self.struct_fields.get(struct_name, [])
        ordered_args = []
        for field_name in field_names:
            if field_name in fields:
                ordered_args.append(self._expr(fields[field_name]))
            else:
                ordered_args.append("null")
        args = ", ".join(ordered_args)
        return f"new {_safe_name(struct_name)}({args})"

    def _list_comp(self, element: Expr, generators: list[CompGenerator]) -> str:
        """Emit list comprehension as IIFE with nested loops."""
        result_var = "_result"
        body_lines: list[str] = [f"const {result_var} = [];"]
        body_stmt = f"{result_var}.push({self._expr(element)});"
        self._emit_comp_loops(generators, 0, body_lines, body_stmt)
        body_lines.append(f"return {result_var};")
        body = " ".join(body_lines)
        return f"(() => {{ {body} }})()"

    def _set_comp(self, element: Expr, generators: list[CompGenerator]) -> str:
        """Emit set comprehension as IIFE with nested loops."""
        result_var = "_result"
        body_lines: list[str] = [f"const {result_var} = new Set();"]
        body_stmt = f"{result_var}.add({self._expr(element)});"
        self._emit_comp_loops(generators, 0, body_lines, body_stmt)
        body_lines.append(f"return {result_var};")
        body = " ".join(body_lines)
        return f"(() => {{ {body} }})()"

    def _dict_comp(self, key: Expr, value: Expr, generators: list[CompGenerator]) -> str:
        """Emit dict comprehension as IIFE with nested loops."""
        result_var = "_result"
        body_lines: list[str] = [f"const {result_var} = new Map();"]
        body_stmt = f"{result_var}.set({self._expr(key)}, {self._expr(value)});"
        self._emit_comp_loops(generators, 0, body_lines, body_stmt)
        body_lines.append(f"return {result_var};")
        body = " ".join(body_lines)
        return f"(() => {{ {body} }})()"

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
            target_name = gen.targets[0]
            if target_name == "_":
                target = "_unused"
            else:
                target = _camel(target_name)
            lines.append(f"for (const {target} of {iter_expr}) {{")
        else:
            targets = ", ".join("_unused" if t == "_" else _camel(t) for t in gen.targets)
            lines.append(f"for (const [{targets}] of {iter_expr}) {{")
        for cond in gen.conditions:
            lines.append(f"if (!({self._expr(cond)})) continue;")
        self._emit_comp_loops(generators, idx + 1, lines, body_stmt)
        lines.append("}")

    def _containment_check(self, item: Expr, container: Expr, negated: bool) -> str:
        """Generate containment check: `x in y` or `x not in y`."""
        item_str = self._expr(item)
        container_str = self._expr(container)
        container_type = container.typ
        neg = "!" if negated else ""
        if isinstance(container_type, (Set, Map)):
            return f"{neg}{container_str}.has({item_str})"
        return f"{neg}{container_str}.includes({item_str})"

    def _maybe_paren(self, expr: Expr, parent_op: str, is_left: bool) -> str:
        """Wrap expr in parens if needed based on operator precedence."""
        inner = self._expr(expr)
        if " ?? " in inner and parent_op in ("+", "-", "*", "/", "%"):
            return f"({inner})"
        if isinstance(expr, Ternary):
            return f"({inner})"
        if not isinstance(expr, BinaryOp):
            return inner
        child_prec = _prec(expr.op)
        parent_prec = _prec(parent_op)
        if parent_op == "**" and expr.op == "**" and not is_left:
            return inner
        needs_parens = child_prec < parent_prec or (child_prec == parent_prec and not is_left)
        return f"({inner})" if needs_parens else inner

    def _format_string(self, template: str, args: list[Expr]) -> str:
        result = template
        for i, arg in enumerate(args):
            if isinstance(arg, StringLit):
                val = arg.value.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
                result = result.replace(f"{{{i}}}", val, 1)
            else:
                result = result.replace(f"{{{i}}}", f"${{{self._expr(arg)}}}", 1)
        arg_idx = 0
        while "%v" in result:
            if arg_idx >= len(args):
                break
            arg = args[arg_idx]
            arg_idx += 1
            if isinstance(arg, StringLit):
                val = arg.value.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
                result = result.replace("%v", val, 1)
            else:
                result = result.replace("%v", f"${{{self._expr(arg)}}}", 1)
        result = result.replace("`", "\\`")
        return f"`{result}`"

    def _lvalue(self, lv: LValue) -> str:
        match lv:
            case VarLV(name=name):
                if name == self.receiver_name:
                    return "this"
                return _camel(name)
            case FieldLV(obj=obj, field=field):
                return f"{self._expr(obj)}.{_camel(field)}"
            case IndexLV(obj=obj, index=index):
                return f"{self._expr(obj)}[{self._expr(index)}]"
            case DerefLV(ptr=ptr):
                return self._expr(ptr)
            case SliceLV():
                # SliceLV is handled specially in _emit_stmt for Assign
                raise NotImplementedError("SliceLV cannot be used as a simple lvalue")
            case _:
                raise NotImplementedError("Unknown lvalue")

    def _type_name_for_check(self, typ: Type) -> str:
        match typ:
            case StructRef(name=name):
                return _safe_name(name)
            case InterfaceRef(name=name):
                return _safe_name(name)
            case Pointer(target=target):
                return self._type_name_for_check(target)
            case _:
                return "Object"


# --- Shared utilities ---


def _typeof_check(kind: str) -> str:
    """Return the typeof result for a primitive type check."""
    match kind:
        case "string":
            return "string"
        case "int" | "float" | "byte" | "rune":
            return "number"
        case "bool":
            return "boolean"
        case _:
            return "object"


def _camel(name: str, is_receiver_ref: bool = False) -> str:
    """Convert snake_case or PascalCase to camelCase."""
    if name in ("this", "self") and is_receiver_ref:
        return "this"
    if "_" not in name:
        if name and name[0].isupper():
            name = name[0].lower() + name[1:]
        return _safe_name(name)
    parts = name.split("_")
    if len(parts) > 1 and any(p.isupper() for p in parts[1:]):
        return _safe_name(name)
    result = parts[0] + "".join(p.capitalize() for p in parts[1:])
    return _safe_name(result)


_JS_RESERVED = {
    "var",
    "let",
    "const",
    "function",
    "class",
    "interface",
    "type",
    "enum",
    "Array",
    "Function",
    "Object",
    "String",
    "Number",
    "Boolean",
    "Symbol",
    "Map",
    "Set",
    "Promise",
    "Error",
}


def _safe_name(name: str) -> str:
    """Rename JavaScript reserved words to safe alternatives."""
    if name in _JS_RESERVED:
        return f"{name}Name"
    return name


_PYTHON_EXCEPTION_MAP = {
    "Exception": "Error",
    "AssertionError": "Error",
    "ValueError": "Error",
    "RuntimeError": "Error",
    "KeyError": "Error",
    "IndexError": "RangeError",
    "TypeError": "TypeError",
}

_STRING_METHOD_MAP = {
    "startswith": "startsWith",
    "endswith": "endsWith",
    "lower": "toLowerCase",
    "upper": "toUpperCase",
    "find": "indexOf",
    "rfind": "lastIndexOf",
}


def _is_array_type(typ: Type) -> bool:
    """Check if type is an array/slice type, possibly wrapped in Pointer."""
    if isinstance(typ, (Slice, Array)):
        return True
    if isinstance(typ, Pointer) and isinstance(typ.target, (Slice, Array)):
        return True
    return False


def _method_name(method: str, receiver_type: Type) -> str:
    """Convert method name based on receiver type."""
    if _is_array_type(receiver_type) and method == "append":
        return "push"
    if method in _STRING_METHOD_MAP:
        return _STRING_METHOD_MAP[method]
    return _camel(method)


def _binary_op(op: str) -> str:
    match op:
        case "&&":
            return "&&"
        case "||":
            return "||"
        case "==":
            return "==="
        case "!=":
            return "!=="
        case "//":
            return "/"
        case _:
            return op


def _string_literal(value: str) -> str:
    return f'"{escape_string(value)}"'


def _escape_regex_class(chars: str) -> str:
    """Escape characters for use in a regex character class []."""
    result = []
    for c in chars:
        if c in r"]\^-":
            result.append(f"\\{c}")
        elif c == "\t":
            result.append("\\t")
        elif c == "\n":
            result.append("\\n")
        elif c == "\r":
            result.append("\\r")
        else:
            result.append(c)
    return "".join(result)


def _escape_regex_literal(s: str) -> str:
    """Escape a string for use as a literal in a regex pattern."""
    result = []
    for c in s:
        if c in r"\^$.*+?()[]{}|":
            result.append(f"\\{c}")
        elif c == "\t":
            result.append("\\t")
        elif c == "\n":
            result.append("\\n")
        elif c == "\r":
            result.append("\\r")
        else:
            result.append(c)
    return "".join(result)


def _is_bool_int_compare(left: Expr, right: Expr) -> bool:
    """True when one operand is bool and the other is int or any."""
    l, r = left.typ, right.typ
    l_is_int_like = l == INT or (isinstance(l, InterfaceRef) and l.name == "any")
    r_is_int_like = r == INT or (isinstance(r, InterfaceRef) and r.name == "any")
    return (l == BOOL and r_is_int_like) or (l_is_int_like and r == BOOL)


def _ends_with_return(body: list[Stmt]) -> bool:
    """Check if a statement list ends with a return (no break needed)."""
    return bool(body) and isinstance(body[-1], Return)


def _prec(op: str) -> int:
    """Return precedence level for binary operator (higher = binds tighter)."""
    match op:
        case "||":
            return 1
        case "&&":
            return 2
        case "|":
            return 3
        case "^":
            return 4
        case "&":
            return 5
        case "==" | "!=" | "===" | "!==":
            return 6
        case "<" | ">" | "<=" | ">=":
            return 7
        case "<<" | ">>":
            return 8
        case "+" | "-":
            return 9
        case "*" | "/" | "%":
            return 10
        case "**":
            return 11
        case _:
            return 12
