"""TypeScript backend: IR → TypeScript code.

Inherits from JsLikeBackend. Adds TypeScript-specific features:
- Type annotations on functions, parameters, variables
- Interface emission
- Field declarations in classes
- `as unknown as X` cast pattern for type assertions
"""

from __future__ import annotations

from src.backend.jslike import (
    JsLikeBackend,
    _camel,
    _is_array_type,
    _is_bool_int_compare,
    _safe_name,
)
from src.backend.util import escape_string, ir_contains_call, ir_contains_cast
from src.ir import (
    BOOL,
    INT,
    STRING,
    Array,
    BinaryOp,
    BoolLit,
    Call,
    Cast,
    Expr,
    Field,
    FloatLit,
    FuncType,
    Index,
    IntLit,
    InterfaceDef,
    Len,
    MakeMap,
    MakeSlice,
    Map,
    MapLit,
    MethodCall,
    Module,
    NilLit,
    Optional,
    Param,
    Pointer,
    Primitive,
    Set,
    SetLit,
    SliceExpr,
    SliceLit,
    StaticCall,
    StringConcat,
    StringFormat,
    StringLit,
    StringSlice,
    Struct,
    StructLit,
    StructRef,
    InterfaceRef,
    Tuple,
    TupleLit,
    Type,
    TypeAssert,
    Union,
    UnaryOp,
    Var,
    Ternary,
    Slice,
)


class TsBackend(JsLikeBackend):
    """Emit TypeScript code from IR."""

    def __init__(self) -> None:
        super().__init__()
        # Override to store field types for struct literals
        self.struct_fields: dict[str, list[tuple[str, Type]]] = {}

    def emit(self, module: Module) -> str:
        self.indent = 0
        self.lines = []
        self.struct_fields = {}
        self._struct_field_count = {}
        for struct in module.structs:
            self.struct_fields[struct.name] = [(f.name, f.typ) for f in struct.fields]
            self._struct_field_count[struct.name] = len(struct.fields)
        self._emit_module(module)
        return "\n".join(self.lines)

    # --- Preamble ---

    def _emit_preamble(self, module: Module) -> bool:
        emitted = False
        if ir_contains_cast(module, "byte", "string") or ir_contains_cast(module, "string", "byte"):
            self._line("declare var TextEncoder: { new(): { encode(s: string): Uint8Array } };")
            self._line("declare var TextDecoder: { new(): { decode(b: Uint8Array): string } };")
            emitted = True
        if ir_contains_call(module, "range"):
            self._emit_range_function()
            emitted = True
        return emitted

    def _emit_range_function(self) -> None:
        self._line("function range(start: number, end?: number, step?: number): number[] {")
        self.indent += 1
        self._line("if (end === undefined) { end = start; start = 0; }")
        self._line("if (step === undefined) { step = 1; }")
        self._line("const result: number[] = [];")
        self._line("if (step > 0) {")
        self.indent += 1
        self._line("for (let i = start; i < end; i += step) result.push(i);")
        self.indent -= 1
        self._line("} else {")
        self.indent += 1
        self._line("for (let i = start; i > end; i += step) result.push(i);")
        self.indent -= 1
        self._line("}")
        self._line("return result;")
        self.indent -= 1
        self._line("}")

    # --- Interface and Field ---

    def _emit_interface(self, iface: InterfaceDef) -> None:
        self._line(f"interface {_safe_name(iface.name)} {{")
        self.indent += 1
        for fld in iface.fields:
            self._line(f"{_camel(fld.name)}: {self._type(fld.typ)};")
        for method in iface.methods:
            params = self._param_list(method.params)
            ret = self._type(method.ret)
            self._line(f"{_camel(method.name)}({params}): {ret};")
        self.indent -= 1
        self._line("}")

    def _emit_field(self, fld: Field) -> None:
        typ = self._type(fld.typ)
        self._line(f"{_camel(fld.name)}: {typ};")

    def _struct_implements(self, struct: Struct) -> str:
        if struct.implements:
            return f" implements {', '.join(_safe_name(i) for i in struct.implements)}"
        return ""

    # --- Signatures ---

    def _func_signature(self, name: str, params: list[Param], ret: Type) -> str:
        return f"function {_camel(name)}({self._param_list(params)}): {self._type(ret)}"

    def _method_signature(self, name: str, params: list[Param], ret: Type) -> str:
        return f"{_camel(name)}({self._param_list(params)}): {self._type(ret)}"

    def _param_list(self, params: list[Param]) -> str:
        parts = []
        for p in params:
            typ = self._type(p.typ)
            parts.append(f"{_camel(p.name)}: {typ}")
        return ", ".join(parts)

    # --- Calls ---

    def _call_expr(self, func: str, args: list[Expr]) -> str:
        # Handle float() with int/float literals - just output the number
        if func == "float" and len(args) == 1:
            arg = args[0]
            if isinstance(arg, IntLit):
                return str(arg.value)
            if isinstance(arg, FloatLit):
                v = arg.value
                return str(int(v)) if v == int(v) else str(v)
            return f"Number({self._expr(arg)})"
        args_str = ", ".join(self._expr(a) for a in args)
        return f"{_camel(func)}({args_str})"

    # --- Variable declarations ---

    def _var_decl(self, name: str, typ: Type | None, value: Expr | None) -> None:
        ts_type = self._type(typ) if typ else "any"
        if value is not None:
            val = self._expr(value)
            self._line(f"let {_camel(name)}: {ts_type} = {val};")
        else:
            self._line(f"let {_camel(name)}: {ts_type};")

    def _assign_decl(self, lv: str, value: Expr) -> None:
        val = self._expr(value)
        ts_type = self._type(value.typ) if value.typ else "any"
        self._line(f"let {lv}: {ts_type} = {val};")

    def _tuple_assign_decl(self, lvalues: str, value: Expr, value_type: Type | None) -> None:
        val = self._expr(value)
        if isinstance(value_type, Tuple) and value_type.elements:
            elem_types = ", ".join(self._type(t) for t in value_type.elements)
            self._line(f"var [{lvalues}]: [{elem_types}] = {val};")
        else:
            self._line(f"var [{lvalues}] = {val};")

    def _var_decl_inline(self, name: str, typ: Type | None, value: Expr | None) -> str:
        ts_type = self._type(typ) if typ else "any"
        if value is not None:
            return f"let {_camel(name)}: {ts_type} = {self._expr(value)}"
        return f"let {_camel(name)}: {ts_type}"

    def _assign_decl_inline(self, lv: str, value: Expr) -> str:
        ts_type = self._type(value.typ) if value.typ else "any"
        return f"let {lv}: {ts_type} = {self._expr(value)}"

    def _for_value_decl(self, name: str, iter_expr: str, index_name: str | None, elem_type: str) -> None:
        self._line(f"const {name}: {elem_type} = {iter_expr}[{index_name}];")

    # --- Exports ---

    def _emit_exports(self, symbols: list[str]) -> None:
        self._line("// CommonJS exports")
        self._line("declare var module: any;")
        self._line("if (typeof module !== 'undefined') {")
        self.indent += 1
        exports = ", ".join(symbols)
        self._line(f"module.exports = {{ {exports} }};")
        self.indent -= 1
        self._line("}")

    # --- For loops ---

    def _emit_for_of(self, value: str, iter_expr: str, iter_type: Type | None) -> None:
        elem_type = self._element_type_str(iter_type)
        is_string = isinstance(iter_type, Primitive) and iter_type.kind == "string"
        if is_string or elem_type == "any":
            self._line(f"for (const {_camel(value)} of {iter_expr}) {{")
        else:
            self._line(f"for (const {_camel(value)} of {iter_expr} as {elem_type}[]) {{")

    def _element_type_str(self, typ: Type | None) -> str:
        if typ is None:
            return "any"
        match typ:
            case Optional(inner=inner):
                return self._element_type_str(inner)
            case Slice(element=element):
                return self._type(element)
            case Array(element=element):
                return self._type(element)
            case Primitive(kind="string"):
                return "string"
            case _:
                return "any"

    # --- Type switch binding ---

    def _emit_type_switch_binding(self, binding_name: str, var: str, typ: Type) -> None:
        self._line(f"const {binding_name} = {var} as {self._type(typ)};")

    # --- Tuple reassign ---

    def _emit_tuple_reassign(self, stmt, targets, lvalues: str, value: Expr) -> None:
        new_targets = stmt.new_targets
        for name in new_targets:
            self._line(f"var {_camel(name)}: any;")
        val = self._expr(value)
        self._line(f"[{lvalues}] = {val};")

    # --- Type assertions ---

    def _type_assert(self, inner: Expr, asserted: Type) -> str:
        return f"({self._expr(inner)} as unknown as {self._type(asserted)})"

    # --- Cast ---

    def _cast_expr(self, inner: Expr, to_type: Type) -> str:
        # float to int: use Math.trunc
        if (
            isinstance(to_type, Primitive)
            and to_type.kind == "int"
            and isinstance(inner.typ, Primitive)
            and inner.typ.kind == "float"
        ):
            return f"Math.trunc({self._expr(inner)})"
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
        ts_type = self._type(to_type)
        from_type = self._type(inner.typ)
        if from_type == ts_type:
            return self._expr(inner)
        # String indexing to string is redundant in TypeScript
        if (
            isinstance(to_type, Primitive)
            and to_type.kind == "string"
            and isinstance(inner, Index)
            and inner.obj.typ == STRING
        ):
            return self._expr(inner)
        # string to []byte
        if (
            isinstance(to_type, Slice)
            and isinstance(to_type.element, Primitive)
            and to_type.element.kind == "byte"
            and isinstance(inner.typ, Primitive)
            and inner.typ.kind == "string"
        ):
            return f"Array.from(new TextEncoder().encode({self._expr(inner)}))"
        # []byte to string
        if (
            isinstance(to_type, Primitive)
            and to_type.kind == "string"
            and isinstance(inner.typ, Slice)
            and isinstance(inner.typ.element, Primitive)
            and inner.typ.element.kind == "byte"
        ):
            return f"new TextDecoder().decode(new Uint8Array({self._expr(inner)}))"
        # rune/int to string
        if (
            isinstance(to_type, Primitive)
            and to_type.kind == "string"
            and isinstance(inner.typ, Primitive)
            and inner.typ.kind in ("rune", "int")
        ):
            return f"String.fromCodePoint({self._expr(inner)})"
        if isinstance(to_type, Primitive) and to_type.kind == "string":
            return f"String({self._expr(inner)})"
        # Use 'as unknown as' for type conversions
        return f"({self._expr(inner)} as unknown as {ts_type})"

    # --- Struct literal ---

    def _struct_lit(self, struct_name: str, fields: dict[str, Expr]) -> str:
        field_info = self.struct_fields.get(struct_name, [])
        ordered_args = []
        for field_name, field_type in field_info:
            if field_name in fields:
                ordered_args.append(f"{self._expr(fields[field_name])} as any")
            else:
                ordered_args.append(self._default_value(field_type))
        args = ", ".join(ordered_args)
        return f"new {_safe_name(struct_name)}({args})"

    # --- Type generation ---

    def _type(self, typ: Type) -> str:
        match typ:
            case Primitive(kind=kind):
                return _primitive_type(kind)
            case Slice(element=element):
                return f"{self._type(element)}[]"
            case Array(element=element):
                return f"{self._type(element)}[]"
            case Map(key=key, value=value):
                return f"Map<{self._type(key)}, {self._type(value)}>"
            case Set(element=element):
                return f"Set<{self._type(element)}>"
            case Tuple(elements=elements):
                parts = ", ".join(self._type(e) for e in elements)
                return f"[{parts}]"
            case Pointer(target=target):
                return self._type(target)
            case Optional(inner=inner):
                return f"{self._type(inner)} | null"
            case StructRef(name=name):
                return _safe_name(name)
            case InterfaceRef(name=name):
                return _safe_name(name)
            case Union(name=name, variants=variants):
                if name:
                    return name
                parts = " | ".join(self._type(v) for v in variants)
                return f"({parts})"
            case FuncType(params=params, ret=ret):
                params_str = ", ".join(f"p{i}: {self._type(p)}" for i, p in enumerate(params))
                return f"(({params_str}) => {self._type(ret)})"
            case StringSlice():
                return "string"
            case _:
                raise NotImplementedError("Unknown type")

    def _type_name_for_check(self, typ: Type) -> str:
        match typ:
            case StructRef(name=name):
                return _safe_name(name)
            case InterfaceRef(name=name):
                return _safe_name(name)
            case _:
                return self._type(typ)


# --- Helpers ---

def _primitive_type(kind: str) -> str:
    match kind:
        case "string":
            return "string"
        case "int" | "float" | "byte" | "rune":
            return "number"
        case "bool":
            return "boolean"
        case "void":
            return "void"
        case _:
            raise NotImplementedError(f"Unknown primitive: {kind}")
