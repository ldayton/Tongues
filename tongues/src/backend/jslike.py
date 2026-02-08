"""Base backend for JavaScript-like languages (JS, TS).

Shared logic for JS and TS code generation. Subclasses override hooks
for type annotations and language-specific features.

Known limitation: Dict key coercion (int/float/bool equivalence) only works for
VarDecl initializers and direct Index/method access. Not covered: Assign, function
args, return statements, dict comprehensions, nested contexts. Proper fix requires
frontend type propagation.
"""

from __future__ import annotations

from src.backend.util import escape_string, is_bytes_type
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
        self._var_types: dict[str, Type] = {}

    def emit(self, module: Module) -> str:
        """Emit code from IR Module."""
        self.indent = 0
        self.lines = []
        self.struct_fields = {}
        self._struct_field_count = {}
        self._var_types = {}
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

    def _tuple_assign_decl(
        self, lvalues: str, value: Expr, value_type: Type | None
    ) -> None:
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
                if typ is not None:
                    self._var_types[name] = typ
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
                # Get target type for tuple vs list distinction
                target_type = None
                if isinstance(target, VarLV):
                    target_type = self._var_types.get(target.name)
                # For list/array += iterable, use push (mutates)
                # For tuple += iterable, use concatenation (immutable, reassigns)
                if op == "+" and (
                    _is_array_type(value.typ)
                    or value.typ == STRING
                    or (isinstance(value, Call) and value.func == "range")
                ):
                    # Tuples are immutable - use concatenation, not push
                    if isinstance(value.typ, Tuple) or isinstance(target_type, Tuple):
                        self._line(f"{lv} = [...{lv}, ...{val}];")
                    else:
                        self._line(f"{lv}.push(...{val});")
                elif op == "*" and isinstance(target_type, Tuple):
                    # Tuple *= int creates new tuple (immutable)
                    self._line(
                        f"{lv} = Array({val} > 0 ? {val} : 0).fill({lv}).flat();"
                    )
                elif isinstance(target_type, Set) or isinstance(value.typ, Set):
                    # Set augmented assignment operators
                    if op == "|":
                        self._line(f"for (const x of {val}) {lv}.add(x);")
                    elif op == "&":
                        self._line(
                            f"for (const x of [...{lv}]) if (!{val}.has(x)) {lv}.delete(x);"
                        )
                    elif op == "-":
                        self._line(f"for (const x of {val}) {lv}.delete(x);")
                    elif op == "^":
                        self._line(
                            f"for (const x of {val}) if ({lv}.has(x)) {lv}.delete(x); else {lv}.add(x);"
                        )
                    else:
                        self._line(f"{lv} {op}= {val};")
                elif isinstance(target_type, Map) or isinstance(value.typ, Map):
                    # Map augmented assignment operators
                    if op == "|":
                        self._line(f"{val}.forEach((v, k) => {lv}.set(k, v));")
                else:
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
                    self._line(
                        f"if (!({cond_str})) {{ throw new Error({self._expr(message)}); }}"
                    )
                else:
                    self._line(
                        f'if (!({cond_str})) {{ throw new Error("Assertion failed"); }}'
                    )
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
            case Raise(
                error_type=error_type, message=message, pos=pos, reraise_var=reraise_var
            ):
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
                    message.value.replace("\\", "\\\\")
                    .replace("`", "\\`")
                    .replace("$", "\\$")
                )
                self._line(
                    f"throw new {error_type}(`{msg_val} at position ${{{p}}}`, {p})"
                )
            else:
                msg = self._expr(message)
                self._line(
                    f"throw new {error_type}(`${{{msg}}} at position ${{{p}}}`, {p})"
                )

    def _emit_slice_assign(self, target: SliceLV, value: Expr) -> None:
        """Emit slice assignment: arr[lo:hi] = value -> splice."""
        obj_str = self._expr(target.obj)
        val_str = self._expr(value)
        low = self._expr(target.low) if target.low else "0"
        if target.high is None:
            # arr[lo:] = value -> splice from lo to end
            self._line(
                f"{obj_str}.splice({low}, {obj_str}.length - {low}, ...{val_str});"
            )
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

    def _emit_match(
        self, expr: Expr, cases: list[MatchCase], default: list[Stmt]
    ) -> None:
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
            # Check if iterating over tuples (like enumerate returns)
            elem_type = None
            if isinstance(iter_type, (Slice, Array)):
                elem_type = iter_type.element
            # Also check if it's an enumerate call (returns tuples)
            is_enumerate = isinstance(iterable, Call) and iterable.func == "enumerate"
            if isinstance(elem_type, Tuple) or is_enumerate:
                # Destructure: for (const [i, v] of iterable)
                self._emit_for_tuple_destructure(index, value, iter_expr, iter_type)
                self.indent += 1
            else:
                # Classic: for (var i = 0; i < length; i++) { v = items[i] }
                self._line(
                    f"for (var {_camel(index)} = 0; {_camel(index)} < {iter_expr}.length; {_camel(index)}++) {{"
                )
                self.indent += 1
                self._for_value_decl(
                    _camel(value),
                    iter_expr,
                    _camel(index),
                    self._element_type_str(iter_type),
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
        # Iterating over Map yields keys (like Python dict)
        if isinstance(iter_type, Map):
            self._line(f"for (const {_camel(value)} of {iter_expr}.keys()) {{")
        else:
            self._line(f"for (const {_camel(value)} of {iter_expr}) {{")

    def _emit_for_tuple_destructure(
        self, index: str, value: str, iter_expr: str, iter_type: Type | None
    ) -> None:
        """Emit for-of with tuple destructuring. Override for TS types."""
        self._line(f"for (const [{_camel(index)}, {_camel(value)}] of {iter_expr}) {{")

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
        # Spread sets to arrays since Set doesn't have .map()
        if _is_set_expr(iterable):
            iter_expr = f"[...{iter_expr}]"
        # Map iteration yields keys
        elif isinstance(iterable.typ, Map):
            iter_expr = f"[...{iter_expr}.keys()]"
        self._line(
            f"{accumulator}.push(...{iter_expr}.map({_camel(value)} => {transform}));"
        )
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
                char_str = self._expr(char)
                # Python's isupper/islower: has at least one cased char, all cased are upper/lower
                if kind == "upper":
                    return f"(/[A-Z]/.test({char_str}) && !/[a-z]/.test({char_str}))"
                if kind == "lower":
                    return f"(/[a-z]/.test({char_str}) && !/[A-Z]/.test({char_str}))"
                regex_map = {
                    "digit": r"/^\d+$/",
                    "alpha": r"/^[a-zA-Z]+$/",
                    "alnum": r"/^[a-zA-Z0-9]+$/",
                    "space": r"/^\s+$/",
                }
                return f"{regex_map[kind]}.test({char_str})"
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
            case Call(func="repr", args=[arg]) if arg.typ == STRING:
                inner = self._expr(arg)
                # Python uses double quotes if string contains single but not double
                return f"({inner}.includes(\"'\") && !{inner}.includes('\"') ? '\"' + {inner} + '\"' : \"'\" + {inner} + \"'\")"
            case Call(func="repr", args=[arg]):
                return f"String({self._expr(arg)})"
            case Call(func="bool", args=args):
                args_str = ", ".join(self._expr(a) for a in args)
                return f"Boolean({args_str})"
            case Call(func="abs", args=[arg]):
                return f"Math.abs({self._expr(arg)})"
            case Call(func="min", args=args):
                if len(args) == 1:
                    arg = args[0]
                    if isinstance(arg.typ, Map):
                        return f"[...{self._expr(arg)}.keys()].reduce((a, b) => a < b ? a : b)"
                    if _is_array_type(arg.typ) or _is_set_expr(arg):
                        arg_str = self._expr(arg)
                        # Use reduce for string arrays (d.keys() etc)
                        if isinstance(arg.typ, Slice) and arg.typ.element == STRING:
                            return f"[...{arg_str}].reduce((a, b) => a < b ? a : b)"
                        return f"Math.min(...{arg_str})"
                args_str = ", ".join(self._expr(a) for a in args)
                return f"Math.min({args_str})"
            case Call(func="max", args=args):
                if len(args) == 1:
                    arg = args[0]
                    if isinstance(arg.typ, Map):
                        return f"[...{self._expr(arg)}.keys()].reduce((a, b) => a > b ? a : b)"
                    if _is_array_type(arg.typ) or _is_set_expr(arg):
                        arg_str = self._expr(arg)
                        # Use reduce for string arrays (d.keys() etc)
                        if isinstance(arg.typ, Slice) and arg.typ.element == STRING:
                            return f"[...{arg_str}].reduce((a, b) => a > b ? a : b)"
                        return f"Math.max(...{arg_str})"
                args_str = ", ".join(self._expr(a) for a in args)
                return f"Math.max({args_str})"
            case Call(func="round", args=[arg]):
                return f"bankersRound({self._expr(arg)})"
            case Call(func="round", args=[arg, ndigits]):
                return f"bankersRound({self._expr(arg)}, {self._expr(ndigits)})"
            case Call(func="int", args=[arg]):
                return f"Math.trunc({self._expr(arg)})"
            case Call(func="divmod", args=[a, b]):
                a_str, b_str = self._expr(a), self._expr(b)
                if self._is_known_non_negative(a) and self._is_known_non_negative(b):
                    return f"[Math.floor({a_str} / {b_str}), {a_str} % {b_str}]"
                return f"[Math.floor({a_str} / {b_str}), (({a_str} % {b_str}) + {b_str}) % {b_str}]"
            case Call(func="pow", args=[base, exp]):
                base_str = self._pow_base(base)
                exp_str = self._pow_exp(exp)
                return f"{base_str} ** {exp_str}"
            case Call(func="pow", args=[base, exp, mod]):
                base_str = self._pow_base(base)
                exp_str = self._pow_exp(exp)
                return f"{base_str} ** {exp_str} % {self._expr(mod)}"
            case Call(func="sorted", args=[arr], reverse=reverse):
                if reverse:
                    return f"[...{self._expr(arr)}].sort((a, b) => a < b ? 1 : a > b ? -1 : 0)"
                return (
                    f"[...{self._expr(arr)}].sort((a, b) => a < b ? -1 : a > b ? 1 : 0)"
                )
            case Call(func=func, args=args):
                return self._call_expr(func, args)
            case MethodCall(obj=obj, method="join", args=[arr], receiver_type=_):
                return self._join_expr(obj, arr)
            case MethodCall(
                obj=obj, method="extend", args=[other], receiver_type=receiver_type
            ) if _is_array_type(receiver_type):
                return f"{self._expr(obj)}.push(...{self._expr(other)})"
            case MethodCall(
                obj=obj, method="copy", args=[], receiver_type=receiver_type
            ) if _is_array_type(receiver_type):
                return f"{self._expr(obj)}.slice()"
            case MethodCall(
                obj=obj, method="get", args=[key], receiver_type=receiver_type
            ) if isinstance(receiver_type, Map):
                return self._map_get(obj, key, None, receiver_type.key)
            case MethodCall(
                obj=obj, method="get", args=[key, default], receiver_type=receiver_type
            ) if isinstance(receiver_type, Map):
                return self._map_get(obj, key, default, receiver_type.key)
            case MethodCall(
                obj=obj, method="items", args=[], receiver_type=receiver_type
            ) if isinstance(receiver_type, Map):
                return f"[...{self._expr(obj)}.entries()]"
            case MethodCall(
                obj=obj, method="keys", args=[], receiver_type=receiver_type
            ) if isinstance(receiver_type, Map):
                return f"[...{self._expr(obj)}.keys()]"
            case MethodCall(
                obj=obj, method="values", args=[], receiver_type=receiver_type
            ) if isinstance(receiver_type, Map):
                return f"[...{self._expr(obj)}.values()]"
            case MethodCall(
                obj=obj, method="copy", args=[], receiver_type=receiver_type
            ) if isinstance(receiver_type, Map):
                return f"new Map({self._expr(obj)})"
            case MethodCall(
                obj=obj, method="pop", args=[key], receiver_type=receiver_type
            ) if isinstance(receiver_type, Map):
                obj_str = self._expr(obj)
                key_str = self._coerce_map_key(receiver_type.key, key)
                return (
                    f"((v = {obj_str}.get({key_str})), {obj_str}.delete({key_str}), v)"
                )
            case MethodCall(
                obj=obj, method="pop", args=[key, default], receiver_type=receiver_type
            ) if isinstance(receiver_type, Map):
                obj_str = self._expr(obj)
                key_str = self._coerce_map_key(receiver_type.key, key)
                default_str = self._expr(default)
                return f"({obj_str}.has({key_str}) ? ((v = {obj_str}.get({key_str})), {obj_str}.delete({key_str}), v) : {default_str})"
            case MethodCall(
                obj=obj, method="setdefault", args=[key], receiver_type=receiver_type
            ) if isinstance(receiver_type, Map):
                obj_str = self._expr(obj)
                key_str = self._coerce_map_key(receiver_type.key, key)
                return f"({obj_str}.has({key_str}) ? {obj_str}.get({key_str}) : ({obj_str}.set({key_str}, null), null))"
            case MethodCall(
                obj=obj,
                method="setdefault",
                args=[key, default],
                receiver_type=receiver_type,
            ) if isinstance(receiver_type, Map):
                obj_str = self._expr(obj)
                key_str = self._coerce_map_key(receiver_type.key, key)
                default_str = self._expr(default)
                return f"({obj_str}.has({key_str}) ? {obj_str}.get({key_str}) : ({obj_str}.set({key_str}, {default_str}), {default_str}))"
            case MethodCall(
                obj=obj, method="update", args=args, receiver_type=receiver_type
            ) if isinstance(receiver_type, Map) and len(args) >= 1:
                obj_str = self._expr(obj)
                updates = []
                for arg in args:
                    arg_str = self._expr(arg)
                    updates.append(f"{arg_str}.forEach((v, k) => {obj_str}.set(k, v))")
                return f"(({', '.join(updates)}), null)"
            case MethodCall(
                obj=obj, method="popitem", args=[], receiver_type=receiver_type
            ) if isinstance(receiver_type, Map):
                obj_str = self._expr(obj)
                return (
                    f"((e = [...{obj_str}.entries()].pop()), {obj_str}.delete(e[0]), e)"
                )
            case MethodCall(
                obj=obj,
                method="replace",
                args=[StringLit(value=old_str), new],
                receiver_type=_,
            ):
                if old_str == "":
                    # Empty string replacement: "ab".replace("", "-") -> "-a-b-"
                    new_str = self._expr(new)
                    return f"({new_str} + {self._expr(obj)}.split('').join({new_str}) + {new_str})"
                escaped = _escape_regex_literal(old_str)
                return f"{self._expr(obj)}.replace(/{escaped}/g, {self._expr(new)})"
            case MethodCall(
                obj=obj,
                method=method,
                args=[TupleLit(elements=elements)],
                receiver_type=_,
            ) if method in ("startswith", "endswith"):
                js_method = "startsWith" if method == "startswith" else "endsWith"
                obj_str = self._expr(obj)
                checks = [f"{obj_str}.{js_method}({self._expr(e)})" for e in elements]
                return f"({' || '.join(checks)})"
            case MethodCall(
                obj=obj,
                method="pop",
                args=[IntLit(value=0)],
                receiver_type=receiver_type,
            ) if _is_array_type(receiver_type):
                return f"{self._expr(obj)}.shift()"
            case MethodCall(
                obj=obj, method="pop", args=[idx], receiver_type=receiver_type
            ) if _is_array_type(receiver_type):
                return f"{self._expr(obj)}.splice({self._expr(idx)}, 1)[0]"
            case MethodCall(obj=obj, method="fromkeys", args=args) if (
                isinstance(obj, Var) and obj.name == "dict"
            ):
                keys_str = self._expr(args[0])
                if len(args) >= 2:
                    val_str = self._expr(args[1])
                    # Python shares the same value object across all keys (mutable gotcha)
                    return f"((_v) => new Map([...{keys_str}].map(k => [k, _v])))({val_str})"
                return f"new Map([...{keys_str}].map(k => [k, null]))"
            case MethodCall(
                obj=obj,
                method=method,
                args=args,
                receiver_type=receiver_type,
                reverse=reverse,
            ):
                return self._method_call(
                    obj, method, args, receiver_type, reverse=reverse
                )
            case StaticCall(on_type=on_type, method="fromkeys", args=args) if (
                isinstance(on_type, Map)
            ):
                keys_str = self._expr(args[0])
                if len(args) >= 2:
                    val_str = self._expr(args[1])
                    return f"((_v) => new Map([...{keys_str}].map(k => [k, _v])))({val_str})"
                return f"new Map([...{keys_str}].map(k => [k, null]))"
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
            case BinaryOp(op="&&", left=left_expr, right=right_expr) if (
                self._is_value_and_or(left_expr)
            ):
                # Python's `and` returns first falsy value or last value
                _, left_str, cond = self._extract_and_or_value(left_expr)
                right_str = self._and_or_operand(right_expr)
                return f"({cond} ? {right_str} : {left_str})"
            case BinaryOp(op="||", left=left_expr, right=right_expr) if (
                self._is_value_and_or(left_expr)
            ):
                # Python's `or` returns first truthy value or last value
                _, left_str, cond = self._extract_and_or_value(left_expr)
                right_str = self._and_or_operand(right_expr)
                return f"({cond} ? {left_str} : {right_str})"
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
                # Coerce bools to ints so Math.min returns number, not bool
                l = (
                    f"({self._expr(left)} ? 1 : 0)"
                    if left.typ == BOOL
                    else self._expr(left)
                )
                r = (
                    f"({self._expr(right)} ? 1 : 0)"
                    if right.typ == BOOL
                    else self._expr(right)
                )
                return f"Math.min({l}, {r})"
            case MaxExpr(left=left, right=right):
                # Coerce bools to ints so Math.max returns number, not bool
                l = (
                    f"({self._expr(left)} ? 1 : 0)"
                    if left.typ == BOOL
                    else self._expr(left)
                )
                r = (
                    f"({self._expr(right)} ? 1 : 0)"
                    if right.typ == BOOL
                    else self._expr(right)
                )
                return f"Math.max({l}, {r})"
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
                # Tuples and lists are both arrays in JS
                if isinstance(tested_type, (Tuple, Slice, Array)):
                    return f"Array.isArray({self._expr(inner)})"
                # isinstance(x, set) -> x instanceof Set
                if isinstance(tested_type, Set):
                    return f"{self._expr(inner)} instanceof Set"
                # isinstance(x, dict) -> x instanceof Map
                if isinstance(tested_type, Map):
                    return f"{self._expr(inner)} instanceof Map"
                # isinstance(x, tuple) or isinstance(x, list) -> Array.isArray
                if isinstance(tested_type, (StructRef, InterfaceRef)):
                    if tested_type.name in ("tuple", "list"):
                        return f"Array.isArray({self._expr(inner)})"
                    if tested_type.name == "set":
                        return f"{self._expr(inner)} instanceof Set"
                    if tested_type.name == "dict":
                        return f"{self._expr(inner)} instanceof Map"
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
            case MapLit(entries=entries) as ml:
                if not entries:
                    return "new Map()"
                # Use ml.typ.key for coercion (from annotation, not inference)
                map_key_type = ml.typ.key if isinstance(ml.typ, Map) else ml.key_type
                pairs = ", ".join(
                    f"[{self._coerce_map_key(map_key_type, k)}, {self._expr(v)}]"
                    for k, v in entries
                )
                return f"new Map([{pairs}])"
            case SetLit(element_type=element_type, elements=elements):
                elems = ", ".join(self._expr(e) for e in elements)
                if isinstance(element_type, Tuple):
                    return f"(function() {{ const s = new Set(); for (const t of [{elems}]) tupleSetAdd(s, t); return s; }})()"
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
            return ("%g" % value).replace("e+", "e")
        return str(value)

    def _coerce_map_key(self, map_key_type: Type, key: Expr) -> str:
        """Coerce key to match map's declared key type (Python key equivalence).

        In Python, True==1 and False==0 as dict keys. JS Map treats them as different.
        This coerces bool keys to int when the map's key type is int.
        """
        if not isinstance(map_key_type, Primitive):
            return self._expr(key)
        map_key = map_key_type.kind
        # Handle literals by checking node type
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
            return f"Math.trunc({key_code})"
        # INT variable → FLOAT (JS already treats these the same)
        # BOOL variable → FLOAT
        if map_key == "float" and key_typ == "bool":
            return f"({key_code} ? 1.0 : 0.0)"
        return key_code

    def _index_expr(self, obj: Expr, index: Expr, typ: Type | None) -> str:
        """Emit index expression with Map handling."""
        obj_str = self._expr(obj)
        obj_type = obj.typ
        if (
            obj_type == STRING
            and isinstance(typ, Primitive)
            and typ.kind in ("int", "byte", "rune")
        ):
            return f"{obj_str}.codePointAt({self._expr(index)})"
        if isinstance(obj_type, Map):
            if isinstance(obj_type.key, Tuple):
                return f"tupleMapGet({obj_str}, {self._expr(index)})"
            idx_str = self._coerce_map_key(obj_type.key, index)
            return f"{obj_str}.get({idx_str})"
        # Nested map access: if indexing result of another Map index, use .get()
        if isinstance(obj, Index) and isinstance(obj.obj.typ, Map):
            value_type = obj.obj.typ.value
            if isinstance(value_type, Map):
                idx_str = self._coerce_map_key(value_type.key, index)
                return f"{obj_str}.get({idx_str})"
        return f"{obj_str}[{self._expr(index)}]"

    def _slice_expr(
        self, obj: Expr, low: Expr | None, high: Expr | None, step: Expr | None = None
    ) -> str:
        """Emit slice expression with step handling."""
        obj_str = self._expr(obj)
        if step is not None:
            low_str = self._expr(low) if low else "null"
            high_str = self._expr(high) if high else "null"
            step_str = self._expr(step)
            return f"arrStep({obj_str}, {low_str}, {high_str}, {step_str})"
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
        if is_bytes_type(s.typ):
            chars_str = self._expr(chars)
            if mode == "left":
                return f"arrLstrip({s_str}, {chars_str})"
            elif mode == "right":
                return f"arrRstrip({s_str}, {chars_str})"
            else:
                return f"arrStrip({s_str}, {chars_str})"
        if isinstance(chars, StringLit) and chars.value == " \t\n\r":
            method_map = {"left": "trimStart", "right": "trimEnd", "both": "trim"}
            return f"{s_str}.{method_map[mode]}()"
        escaped = (
            _escape_regex_class(chars.value) if isinstance(chars, StringLit) else "..."
        )
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
        """Emit join expression with bytes handling."""
        if is_bytes_type(sep.typ) or _is_bytes_list_type(arr.typ):
            return f"arrJoin({self._expr(arr)}, {self._expr(sep)})"
        return f"{self._expr(arr)}.join({self._expr(sep)})"

    def _emit_map_lit_coerced(self, value: Expr, target_type: Type) -> str:
        """Emit expression, coercing MapLit keys if target type differs from literal type."""
        if not isinstance(value, MapLit) or not isinstance(target_type, Map):
            return self._expr(value)
        # Use target type's key type for coercion (from annotation, not literal inference)
        if not value.entries:
            return "new Map()"
        pairs = ", ".join(
            f"[{self._coerce_map_key(target_type.key, k)}, {self._expr(v)}]"
            for k, v in value.entries
        )
        return f"new Map([{pairs}])"

    def _map_get(
        self, obj: Expr, key: Expr, default: Expr | None, key_type: Type
    ) -> str:
        """Emit Map.get expression with proper null handling."""
        obj_str = self._expr(obj)
        key_str = self._coerce_map_key(key_type, key)
        if default is not None:
            # Use ternary to distinguish null value from missing key
            return f"({obj_str}.has({key_str}) ? {obj_str}.get({key_str}) : {self._expr(default)})"
        return f"({obj_str}.get({key_str}) ?? null)"

    def _method_call(
        self,
        obj: Expr,
        method: str,
        args: list[Expr],
        receiver_type: Type,
        reverse: bool = False,
    ) -> str:
        """Emit method call with bytes handling."""
        # Handle bytes join (separator is obj, list is first arg)
        if method == "join" and len(args) == 1 and is_bytes_type(receiver_type):
            return f"arrJoin({self._expr(args[0])}, {self._expr(obj)})"
        # Handle bytes list join
        if method == "join" and len(args) == 1 and _is_bytes_list_type(obj.typ):
            sep_str = self._expr(args[0]) if args else "[]"
            return f"arrJoin({self._expr(obj)}, {sep_str})"
        # Byte array methods
        if is_bytes_type(receiver_type):
            obj_str = self._expr(obj)
            if method == "count" and len(args) == 1:
                return f"arrCount({obj_str}, {self._expr(args[0])})"
            if method == "find" and len(args) == 1:
                return f"arrFind({obj_str}, {self._expr(args[0])})"
            if method == "startswith" and len(args) == 1:
                return f"arrStartsWith({obj_str}, {self._expr(args[0])})"
            if method == "endswith" and len(args) == 1:
                return f"arrEndsWith({obj_str}, {self._expr(args[0])})"
            if method == "upper":
                return f"arrUpper({obj_str})"
            if method == "lower":
                return f"arrLower({obj_str})"
            if method == "strip" and len(args) == 1:
                return f"arrStrip({obj_str}, {self._expr(args[0])})"
            if method == "lstrip" and len(args) == 1:
                return f"arrLstrip({obj_str}, {self._expr(args[0])})"
            if method == "rstrip" and len(args) == 1:
                return f"arrRstrip({obj_str}, {self._expr(args[0])})"
            if method == "split" and len(args) == 1:
                return f"arrSplit({obj_str}, {self._expr(args[0])})"
            if method == "replace" and len(args) == 2:
                return f"arrReplace({obj_str}, {self._expr(args[0])}, {self._expr(args[1])})"
        # List/array methods not in JS
        if _is_array_type(receiver_type):
            obj_str = self._expr(obj)
            if method == "insert" and len(args) == 2:
                idx = self._expr(args[0])
                val = self._expr(args[1])
                return f"{obj_str}.splice({idx} < 0 ? {obj_str}.length + {idx} : {idx}, 0, {val})"
            if method == "remove" and len(args) == 1:
                val = self._expr(args[0])
                return f"{obj_str}.splice({obj_str}.indexOf({val}), 1)"
            if method == "clear" and len(args) == 0:
                return f"{obj_str}.length = 0"
            if method == "index" and len(args) >= 1:
                val = self._expr(args[0])
                if len(args) == 1:
                    return f"{obj_str}.indexOf({val})"
                start = self._expr(args[1])
                if len(args) == 2:
                    return f"{obj_str}.indexOf({val}, {start})"
                end = self._expr(args[2])
                return f"{obj_str}.slice(0, {end}).indexOf({val}, {start})"
            if method == "count" and len(args) == 1:
                val = self._expr(args[0])
                return f"{obj_str}.filter(x => x === {val}).length"
            if method == "sort" and len(args) == 0:
                # JS sort with generic comparator, handle reverse, return null
                if reverse:
                    return (
                        f"({obj_str}.sort((a, b) => a < b ? 1 : a > b ? -1 : 0), null)"
                    )
                return f"({obj_str}.sort((a, b) => a < b ? -1 : a > b ? 1 : 0), null)"
            if method == "reverse" and len(args) == 0:
                return f"({obj_str}.reverse(), null)"
        # Set methods
        if isinstance(receiver_type, Set):
            obj_str = self._expr(obj)
            if method == "remove" and len(args) == 1:
                return f"({obj_str}.delete({self._expr(args[0])}), null)"
            if method == "discard" and len(args) == 1:
                return f"({obj_str}.delete({self._expr(args[0])}), null)"
            if method == "pop" and len(args) == 0:
                return (
                    f"((v = {obj_str}.values().next().value), {obj_str}.delete(v), v)"
                )
            if method == "copy" and len(args) == 0:
                return f"new Set({obj_str})"
            if method == "union" and len(args) >= 1:
                union_parts = [f"...{obj_str}"]
                for arg in args:
                    union_parts.append(f"...{self._expr(arg)}")
                return f"new Set([{', '.join(union_parts)}])"
            if method == "intersection" and len(args) >= 1:
                result = f"[...{obj_str}]"
                for arg in args:
                    arg_str = self._expr(arg)
                    result = f"{result}.filter(x => {arg_str}.has(x))"
                return f"new Set({result})"
            if method == "difference" and len(args) >= 1:
                result = f"[...{obj_str}]"
                for arg in args:
                    arg_str = self._expr(arg)
                    result = f"{result}.filter(x => !{arg_str}.has(x))"
                return f"new Set({result})"
            if method == "symmetric_difference" and len(args) == 1:
                other = self._expr(args[0])
                return f"new Set([...{obj_str}].filter(x => !{other}.has(x)).concat([...{other}].filter(x => !{obj_str}.has(x))))"
            if method == "issubset" and len(args) == 1:
                other = self._expr(args[0])
                return f"[...{obj_str}].every(x => {other}.has(x))"
            if method == "issuperset" and len(args) == 1:
                other = self._expr(args[0])
                return f"[...{other}].every(x => {obj_str}.has(x))"
            if method == "isdisjoint" and len(args) == 1:
                other = self._expr(args[0])
                return f"![...{obj_str}].some(x => {other}.has(x))"
            if method == "update" and len(args) >= 1:
                updates = []
                for arg in args:
                    arg_str = self._expr(arg)
                    # For dicts/Maps, update adds keys not key-value pairs
                    if isinstance(arg.typ, Map):
                        updates.append(
                            f"[...{arg_str}.keys()].forEach(x => {obj_str}.add(x))"
                        )
                    else:
                        updates.append(f"[...{arg_str}].forEach(x => {obj_str}.add(x))")
                return f"(({', '.join(updates)}), null)"
        # Handle string.split() with no args - splits on whitespace, removes empties
        if receiver_type == STRING and method == "split" and len(args) == 0:
            return f"{self._expr(obj)}.trim().split(/\\s+/).filter(Boolean)"
        # Handle string.split(sep, maxsplit) - Python splits at most maxsplit times
        if receiver_type == STRING and method == "split" and len(args) == 2:
            obj_str = self._expr(obj)
            sep = self._expr(args[0])
            maxsplit = self._expr(args[1])
            return f"((m = {maxsplit}) === 0 ? [{obj_str}] : (p = {obj_str}.split({sep}), m >= p.length - 1 ? p : [...p.slice(0, m), p.slice(m).join({sep})]))"
        # Handle string.rsplit(sep, maxsplit) - splits from right
        if receiver_type == STRING and method == "rsplit" and len(args) == 2:
            obj_str = self._expr(obj)
            sep = self._expr(args[0])
            maxsplit = self._expr(args[1])
            return f"((m = {maxsplit}) === 0 ? [{obj_str}] : (p = {obj_str}.split({sep}), m >= p.length - 1 ? p : [p.slice(0, -m).join({sep}), ...p.slice(-m)]))"
        # String methods not in JS
        if receiver_type == STRING:
            obj_str = self._expr(obj)
            if method == "count" and len(args) == 1:
                arg = args[0]
                if isinstance(arg, StringLit) and arg.value == "":
                    # count("") returns len + 1
                    return f"({obj_str}.length + 1)"
                sub = self._expr(arg)
                return f"({obj_str}.split({sub}).length - 1)"
            if method == "capitalize" and len(args) == 0:
                return f"({obj_str}.charAt(0).toUpperCase() + {obj_str}.slice(1).toLowerCase())"
            if method == "title" and len(args) == 0:
                return (
                    f"{obj_str}.toLowerCase().replace(/\\b\\w/g, c => c.toUpperCase())"
                )
            if method == "swapcase" and len(args) == 0:
                return f"{obj_str}.split('').map(c => c === c.toUpperCase() ? c.toLowerCase() : c.toUpperCase()).join('')"
            if method == "casefold" and len(args) == 0:
                return f"{obj_str}.toLowerCase()"
            if method == "removeprefix" and len(args) == 1:
                prefix = self._expr(args[0])
                return f"({obj_str}.startsWith({prefix}) ? {obj_str}.slice({prefix}.length) : {obj_str})"
            if method == "removesuffix" and len(args) == 1:
                suffix = self._expr(args[0])
                return f"({obj_str}.endsWith({suffix}) ? {obj_str}.slice(0, -{suffix}.length) : {obj_str})"
            if method == "zfill" and len(args) == 1:
                width = self._expr(args[0])
                return f"(((s = {obj_str})[0] === '-' || s[0] === '+') ? s[0] + s.slice(1).padStart({width} - 1, '0') : s.padStart({width}, '0'))"
            if method == "center" and len(args) >= 1:
                width = self._expr(args[0])
                fill = self._expr(args[1]) if len(args) > 1 else "' '"
                return f"{obj_str}.padStart(Math.floor(({width} + {obj_str}.length) / 2), {fill}).padEnd({width}, {fill})"
            if method == "ljust" and len(args) >= 1:
                width = self._expr(args[0])
                fill = self._expr(args[1]) if len(args) > 1 else "' '"
                return f"{obj_str}.padEnd({width}, {fill})"
            if method == "rjust" and len(args) >= 1:
                width = self._expr(args[0])
                fill = self._expr(args[1]) if len(args) > 1 else "' '"
                return f"{obj_str}.padStart({width}, {fill})"
            if method == "splitlines" and len(args) == 0:
                return f"((a = {obj_str}.split(/\\r\\n|\\r|\\n/)), a.length && a[a.length - 1] === '' ? a.slice(0, -1) : a)"
            if method == "expandtabs" and len(args) <= 1:
                tabsize = self._expr(args[0]) if len(args) == 1 else "8"
                # Column-aware tab expansion: each tab goes to next multiple of tabsize
                return f"((t = {tabsize}, r = '', c = 0) => {{ for (let x of {obj_str}) {{ if (x === '\\t') {{ const sp = t - (c % t); r += ' '.repeat(sp); c += sp; }} else {{ r += x; c++; }} }} return r; }})()"
            if method == "partition" and len(args) == 1:
                sep = self._expr(args[0])
                return f"((i = {obj_str}.indexOf({sep})) === -1 ? [{obj_str}, '', ''] : [{obj_str}.slice(0, i), {sep}, {obj_str}.slice(i + {sep}.length)])"
            if method == "rpartition" and len(args) == 1:
                sep = self._expr(args[0])
                return f"((i = {obj_str}.lastIndexOf({sep})) === -1 ? ['', '', {obj_str}] : [{obj_str}.slice(0, i), {sep}, {obj_str}.slice(i + {sep}.length)])"
            if method == "format" and len(args) > 0:
                args_list = ", ".join(self._expr(a) for a in args)
                return f"((a = [{args_list}], i = 0) => {obj_str}.replace(/\\{{(\\d*)\\}}/g, (_, n) => a[n === '' ? i++ : +n]))()"
        args_str = ", ".join(self._expr(a) for a in args)
        js_method = _method_name(method, receiver_type)
        return f"{self._expr(obj)}.{js_method}({args_str})"

    def _truthy_expr(self, e: Expr) -> str:
        inner_str = self._expr(e)
        inner_type = e.typ
        if isinstance(inner_type, Map) or _is_set_expr(e):
            return f"({inner_str}.size > 0)"
        if isinstance(inner_type, (Slice, Tuple, Array)) or inner_type == STRING:
            return f"({inner_str}.length > 0)"
        if isinstance(inner_type, Optional) and isinstance(inner_type.inner, Map):
            return f"({inner_str} != null && {inner_str}.size > 0)"
        if isinstance(inner_type, Optional) and isinstance(inner_type.inner, Set):
            return f"({inner_str} != null && {inner_str}.size > 0)"
        if isinstance(inner_type, Optional) and isinstance(
            inner_type.inner, (Slice, Tuple, Array)
        ):
            return f"({inner_str} != null && {inner_str}.length > 0)"
        if inner_type == INT or (
            isinstance(inner_type, Primitive) and inner_type.kind == "float"
        ):
            if isinstance(e, BinaryOp):
                return f"(({inner_str}) !== 0)"
            return f"({inner_str} !== 0)"
        return f"({inner_str} != null)"

    def _is_known_non_negative(self, expr: Expr) -> bool:
        """Check if expression is known to be non-negative at compile time."""
        if isinstance(expr, IntLit):
            return expr.value >= 0
        if isinstance(expr, FloatLit):
            return expr.value >= 0
        # Len always returns non-negative
        if isinstance(expr, Len):
            return True
        # Absolute value is always non-negative
        if isinstance(expr, Call) and expr.func == "abs":
            return True
        return False

    def _is_value_and_or(self, expr: Expr) -> bool:
        """Check if expression is a Truthy or value-returning and/or."""
        if isinstance(expr, Truthy):
            return True
        if isinstance(expr, BinaryOp) and expr.op in ("&&", "||"):
            return self._is_value_and_or(expr.left)
        return False

    def _js_truthy_check(self, expr_str: str, typ: Type) -> str:
        """Generate JS truthy check for a given type."""
        if isinstance(typ, (Slice, Tuple, Array)):
            return f"{expr_str}.length > 0"
        if isinstance(typ, Map) or isinstance(typ, Set):
            return f"{expr_str}.size > 0"
        if isinstance(typ, Optional):
            return f"{expr_str} != null"
        if typ == STRING:
            return f"{expr_str}.length > 0"
        if typ == INT or (isinstance(typ, Primitive) and typ.kind == "float"):
            return f"{expr_str} !== 0"
        return f"!!{expr_str}"

    def _extract_and_or_value(self, expr: Expr) -> tuple[Expr, str, str]:
        """Extract value, value string, and truthy check from and/or operand."""
        if isinstance(expr, Truthy):
            val = expr.expr
            val_str = self._expr(val)
            cond = self._js_truthy_check(val_str, val.typ)
            return val, val_str, cond
        val_str = self._expr(expr)
        cond = self._js_truthy_check(val_str, expr.typ)
        return expr, val_str, cond

    def _and_or_operand(self, expr: Expr) -> str:
        """Extract value from and/or operand, handling Truthy wrapper."""
        if isinstance(expr, Truthy):
            return self._expr(expr.expr)
        return self._expr(expr)

    def _set_operand(self, expr: Expr) -> str:
        """Emit set operand, wrapping dict views in Sets and handling tuple elements."""
        expr_str = self._expr(expr)
        if _is_tuple_set_expr(expr):
            # Sets with tuple elements already use tupleSetAdd in their construction
            # Dict items views need wrapping
            if _is_dict_items_view(expr):
                return f"(function() {{ const s = new Set(); for (const t of {expr_str}) tupleSetAdd(s, t); return s; }})()"
            return expr_str
        if _is_dict_view_expr(expr):
            return f"new Set({expr_str})"
        return expr_str

    def _binary_expr(self, op: str, left: Expr, right: Expr) -> str:
        """Emit binary expression with bytes handling."""
        # Handle bytes list comparison
        if _is_bytes_list_type(left.typ) or _is_bytes_list_type(right.typ):
            left_str = self._expr(left)
            right_str = self._expr(right)
            if op == "==":
                return f"deepArrEq({left_str}, {right_str})"
            if op == "!=":
                return f"!deepArrEq({left_str}, {right_str})"
        # Handle bytes comparison
        if is_bytes_type(left.typ) or is_bytes_type(right.typ):
            left_str = self._expr(left)
            right_str = self._expr(right)
            if op == "==":
                return f"arrEq({left_str}, {right_str})"
            if op == "!=":
                return f"!arrEq({left_str}, {right_str})"
            if op == "<":
                return f"arrLt({left_str}, {right_str})"
            if op == "<=":
                return f"(arrLt({left_str}, {right_str}) || arrEq({left_str}, {right_str}))"
            if op == ">":
                return f"arrLt({right_str}, {left_str})"
            if op == ">=":
                return f"(arrLt({right_str}, {left_str}) || arrEq({left_str}, {right_str}))"
            if op == "+":
                return f"arrConcat({left_str}, {right_str})"
            if op == "*":
                if is_bytes_type(left.typ):
                    return f"arrRepeat({left_str}, {right_str})"
                return f"arrRepeat({right_str}, {left_str})"
        # Handle list/array comparison and ops
        if _is_array_type(left.typ) or _is_array_type(right.typ):
            left_str = self._expr(left)
            right_str = self._expr(right)
            if op == "==":
                return f"arrEq({left_str}, {right_str})"
            if op == "!=":
                return f"!arrEq({left_str}, {right_str})"
            if op == "+":
                return f"[...{left_str}, ...{right_str}]"
            if op == "*":
                if _is_array_type(left.typ):
                    return f"Array({right_str} > 0 ? {right_str} : 0).fill({left_str}).flat()"
                return (
                    f"Array({left_str} > 0 ? {left_str} : 0).fill({right_str}).flat()"
                )
        # Handle Set comparison and operators
        # For equality, both sides must be sets; otherwise use default comparison
        if _is_set_expr(left) and _is_set_expr(right):
            is_tuple_set = _is_tuple_set_expr(left) or _is_tuple_set_expr(right)
            if is_tuple_set:
                left_str = self._set_operand(left)  # Use _set_operand for consistency
                right_str = self._set_operand(right)
                if op == "==":
                    return f"(({left_str}.size === {right_str}.size) && [...{left_str}].every(x => tupleSetHas({right_str}, x)))"
                if op == "!=":
                    return f"!(({left_str}.size === {right_str}.size) && [...{left_str}].every(x => tupleSetHas({right_str}, x)))"
            else:
                left_str = self._set_operand(left)
                right_str = self._set_operand(right)
                if op == "==":
                    return f"(({left_str}.size === {right_str}.size) && [...{left_str}].every(x => {right_str}.has(x)))"
                if op == "!=":
                    return f"!(({left_str}.size === {right_str}.size) && [...{left_str}].every(x => {right_str}.has(x)))"
        if _is_set_expr(left) or _is_set_expr(right):
            # Use tuple-aware operations for dict items views
            is_tuple_set = _is_tuple_set_expr(left) or _is_tuple_set_expr(right)
            left_str = self._set_operand(left)
            right_str = self._set_operand(right)
            if is_tuple_set:
                if op == "|":
                    return f"(function() {{ const s = new Set(); for (const t of [...{left_str}, ...{right_str}]) tupleSetAdd(s, t); return s; }})()"
                if op == "&":
                    return f"(function() {{ const s = new Set(); for (const t of [...{left_str}]) if (tupleSetHas({right_str}, t)) tupleSetAdd(s, t); return s; }})()"
                if op == "-":
                    return f"(function() {{ const s = new Set(); for (const t of [...{left_str}]) if (!tupleSetHas({right_str}, t)) tupleSetAdd(s, t); return s; }})()"
                if op == "^":
                    return f"(function() {{ const s = new Set(); for (const t of [...{left_str}]) if (!tupleSetHas({right_str}, t)) tupleSetAdd(s, t); for (const t of [...{right_str}]) if (!tupleSetHas({left_str}, t)) tupleSetAdd(s, t); return s; }})()"
                if op == "<=":
                    return f"[...{left_str}].every(x => tupleSetHas({right_str}, x))"
                if op == "<":
                    return f"({left_str}.size < {right_str}.size && [...{left_str}].every(x => tupleSetHas({right_str}, x)))"
                if op == ">=":
                    return f"[...{right_str}].every(x => tupleSetHas({left_str}, x))"
                if op == ">":
                    return f"({left_str}.size > {right_str}.size && [...{right_str}].every(x => tupleSetHas({left_str}, x)))"
            if op == "|":
                return f"new Set([...{left_str}, ...{right_str}])"
            if op == "&":
                return f"new Set([...{left_str}].filter(x => {right_str}.has(x)))"
            if op == "-":
                return f"new Set([...{left_str}].filter(x => !{right_str}.has(x)))"
            if op == "^":
                return f"new Set([...{left_str}].filter(x => !{right_str}.has(x)).concat([...{right_str}].filter(x => !{left_str}.has(x))))"
            if op == "<=":
                return f"[...{left_str}].every(x => {right_str}.has(x))"
            if op == "<":
                return f"({left_str}.size < {right_str}.size && [...{left_str}].every(x => {right_str}.has(x)))"
            if op == ">=":
                return f"[...{right_str}].every(x => {left_str}.has(x))"
            if op == ">":
                return f"({left_str}.size > {right_str}.size && [...{right_str}].every(x => {left_str}.has(x)))"
        # Handle Map comparison and merge operators
        if _is_map_expr(left) and _is_map_expr(right):
            left_str = self._expr(left)
            right_str = self._expr(right)
            if op == "==":
                return f"mapEq({left_str}, {right_str})"
            if op == "!=":
                return f"!mapEq({left_str}, {right_str})"
            if op == "|":
                return f"new Map([...{left_str}, ...{right_str}])"
        if _is_map_expr(left) or _is_map_expr(right):
            left_str = self._expr(left)
            right_str = self._expr(right)
            if op == "|":
                return f"new Map([...{left_str}, ...{right_str}])"
        # Handle string repetition: "a" * 3 -> "a".repeat(3), negative -> ""
        if op == "*":
            if left.typ == STRING:
                n = self._expr(right)
                return f"{self._expr(left)}.repeat(Math.max(0, {n}))"
            if right.typ == STRING:
                n = self._expr(left)
                return f"{self._expr(right)}.repeat(Math.max(0, {n}))"
        # Python modulo semantics: result has sign of divisor (differs from JS)
        if op == "%":
            left_str = self._maybe_paren(left, op, is_left=True)
            right_str = self._maybe_paren(right, op, is_left=False)
            # Only need Python semantics if either operand might be negative
            if self._is_known_non_negative(left) and self._is_known_non_negative(right):
                return f"{left_str} % {right_str}"
            return f"(({left_str} % {right_str}) + {right_str}) % {right_str}"
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
        # float(string) with special values
        if (
            isinstance(to_type, Primitive)
            and to_type.kind == "float"
            and isinstance(inner, StringLit)
        ):
            if inner.value == "inf" or inner.value == "Infinity":
                return "Infinity"
            if inner.value == "-inf" or inner.value == "-Infinity":
                return "-Infinity"
            if inner.value.lower() == "nan":
                return "NaN"
            return f"parseFloat({self._expr(inner)})"
        # str(None) -> "None"
        if (
            isinstance(inner, NilLit)
            and isinstance(to_type, Primitive)
            and to_type.kind == "string"
        ):
            return '"None"'
        if (
            isinstance(to_type, Primitive)
            and to_type.kind in ("int", "byte", "rune")
            and inner.typ == BOOL
        ):
            return f"Number({self._expr(inner)})"
        if (
            isinstance(to_type, Primitive)
            and to_type.kind == "string"
            and inner.typ == BOOL
        ):
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
        if isinstance(inner_type, Map) or _is_set_expr(inner):
            return f"{self._expr(inner)}.size"
        # Use spread for strings to properly count code points (emoji support)
        if inner_type == STRING:
            return f"[...{self._expr(inner)}].length"
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

    def _dict_comp(
        self, key: Expr, value: Expr, generators: list[CompGenerator]
    ) -> str:
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
            targets = ", ".join(
                "_unused" if t == "_" else _camel(t) for t in gen.targets
            )
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
        if isinstance(container_type, Set):
            if isinstance(container_type.element, Tuple):
                return f"{neg}tupleSetHas({container_str}, {item_str})"
            return f"{neg}{container_str}.has({item_str})"
        if isinstance(container_type, Map):
            if isinstance(container_type.key, Tuple):
                return f"{neg}tupleMapHas({container_str}, {item_str})"
            coerced_key = self._coerce_map_key(container_type.key, item)
            return f"{neg}{container_str}.has({coerced_key})"
        if is_bytes_type(container_type):
            return f"{neg}arrContains({container_str}, {item_str})"
        # Tuple in list of tuples needs value-based comparison
        if isinstance(item.typ, Tuple) and isinstance(container_type, (Slice, Array)):
            if isinstance(container_type.element, Tuple):
                return f"{neg}{container_str}.some(x => arrEq(x, {item_str}))"
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
        needs_parens = child_prec < parent_prec or (
            child_prec == parent_prec and not is_left
        )
        return f"({inner})" if needs_parens else inner

    def _pow_base(self, base: Expr) -> str:
        """Wrap pow() base in parens if needed (JS requires parens for unary before **)."""
        base_str = self._expr(base)
        if isinstance(base, UnaryOp):
            return f"({base_str})"
        if isinstance(base, IntLit) and base.value < 0:
            return f"({base_str})"
        if isinstance(base, FloatLit) and base.value < 0:
            return f"({base_str})"
        return base_str

    def _pow_exp(self, exp: Expr) -> str:
        """Wrap pow() exponent in parens if it's a binary op (precedence issues)."""
        exp_str = self._expr(exp)
        if isinstance(exp, BinaryOp):
            return f"({exp_str})"
        return exp_str

    def _format_string(self, template: str, args: list[Expr]) -> str:
        result = template
        for i, arg in enumerate(args):
            if isinstance(arg, StringLit):
                val = (
                    arg.value.replace("\\", "\\\\")
                    .replace("`", "\\`")
                    .replace("$", "\\$")
                )
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
                val = (
                    arg.value.replace("\\", "\\\\")
                    .replace("`", "\\`")
                    .replace("$", "\\$")
                )
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
            case IndexLV(obj=obj, index=IntLit(value=n)) if n < 0:
                # Negative literal index - convert to length-based
                obj_str = self._expr(obj)
                return f"{obj_str}[{obj_str}.length - {-n}]"
            case IndexLV(obj=obj, index=UnaryOp(op="-", operand=operand)):
                # Negative dynamic index: -x -> arr[arr.length - x]
                obj_str = self._expr(obj)
                return f"{obj_str}[{obj_str}.length - {self._expr(operand)}]"
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
    """Check if type is an array/slice/tuple type, possibly wrapped in Pointer."""
    if isinstance(typ, (Slice, Array, Tuple)):
        return True
    if isinstance(typ, Pointer) and isinstance(typ.target, (Slice, Array, Tuple)):
        return True
    return False


def _is_dict_view_expr(expr: Expr) -> bool:
    """Check if expression is a dict view (keys or items method on a Map)."""
    if isinstance(expr, MethodCall) and isinstance(expr.receiver_type, Map):
        if expr.method in ("keys", "items"):
            return True
    return False


def _is_dict_items_view(expr: Expr) -> bool:
    """Check if expression is specifically a dict items view (has tuple elements)."""
    if isinstance(expr, MethodCall) and isinstance(expr.receiver_type, Map):
        if expr.method == "items":
            return True
    return False


def _is_tuple_set_expr(expr: Expr) -> bool:
    """Check if expression is a set with tuple elements (needs tuple-aware comparison)."""
    if _is_dict_items_view(expr):
        return True
    # SetLit with tuple elements
    if isinstance(expr, SetLit) and isinstance(expr.element_type, Tuple):
        return True
    # Set with tuple element type
    if isinstance(expr.typ, Set) and isinstance(expr.typ.element, Tuple):
        return True
    return False


def _is_set_expr(expr: Expr) -> bool:
    """Check if expression is a set (by type or by being a set() call or set operator)."""
    if isinstance(expr.typ, Set):
        return True
    if isinstance(expr, Call) and expr.func == "set":
        return True
    # Dict view methods (keys, items) behave like sets for set operations
    if _is_dict_view_expr(expr):
        return True
    # Set binary operators produce sets
    if isinstance(expr, BinaryOp) and expr.op in ("|", "&", "-", "^"):
        if _is_set_expr(expr.left) or _is_set_expr(expr.right):
            return True
    # Set methods that return sets
    if isinstance(expr, MethodCall) and isinstance(expr.receiver_type, Set):
        if expr.method in (
            "copy",
            "union",
            "intersection",
            "difference",
            "symmetric_difference",
        ):
            return True
    return False


def _is_map_expr(expr: Expr) -> bool:
    """Check if expression is a map/dict (by type or by being a dict() call or merge operator)."""
    if isinstance(expr.typ, Map):
        return True
    if isinstance(expr, Call) and expr.func == "dict":
        return True
    # Nested map access: d["key"] where d is Map with Map value type
    if isinstance(expr, Index) and isinstance(expr.obj.typ, Map):
        if isinstance(expr.obj.typ.value, Map):
            return True
    # Map merge operator produces maps
    if isinstance(expr, BinaryOp) and expr.op == "|":
        if _is_map_expr(expr.left) or _is_map_expr(expr.right):
            return True
    # Map methods that return maps
    if isinstance(expr, MethodCall) and isinstance(expr.receiver_type, Map):
        if expr.method == "copy":
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
    # MinExpr/MaxExpr with bool args produce ints after coercion
    l_is_minmax_bool = isinstance(left, (MinExpr, MaxExpr)) and left.left.typ == BOOL
    r_is_minmax_bool = isinstance(right, (MinExpr, MaxExpr)) and right.left.typ == BOOL
    if l_is_minmax_bool and r == BOOL:
        return True
    if r_is_minmax_bool and l == BOOL:
        return True
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


def _is_bytes_list_type(typ: Type | None) -> bool:
    """Check if type is a list of byte arrays."""
    if isinstance(typ, Slice):
        return is_bytes_type(typ.element)
    return False
