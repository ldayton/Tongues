"""JavaScript backend: IR → JavaScript code.

Inherits from JsLikeBackend. Adds JS-specific features:
- Preamble helper functions (range, bytes array helpers)
- Hoisted variable tracking for control structures
- Implicit return null for void functions
- Map indexing with .get() method
"""

from __future__ import annotations

from src.backend.jslike import (
    JsLikeBackend,
    _camel,
    _is_array_type,
    _is_bool_int_compare,
    _is_bytes_list_type,
)
from src.backend.util import ir_contains_call, ir_has_bytes_ops, ir_has_tuple_maps, ir_has_tuple_sets, is_bytes_type
from src.ir import (
    BOOL,
    FLOAT,
    INT,
    STRING,
    VOID,
    Array,
    Assign,
    BinaryOp,
    Call,
    Cast,
    Expr,
    Function,
    If,
    IndexLV,
    IntLit,
    InterfaceDef,
    Field,
    ForClassic,
    ForRange,
    Index,
    LValue,
    Map,
    Match,
    MethodCall,
    Module,
    NilLit,
    Optional,
    Param,
    Pointer,
    Primitive,
    Return,
    Set,
    Slice,
    SliceExpr,
    SliceLit,
    Stmt,
    StringLit,
    Ternary,
    TryCatch,
    TupleAssign,
    Type,
    TypeSwitch,
    VarDecl,
    VarLV,
    While,
)


class JsBackend(JsLikeBackend):
    """Emit JavaScript code from IR."""

    def __init__(self) -> None:
        super().__init__()
        self._hoisted_vars: set[str] = set()

    def emit(self, module: Module) -> str:
        self._hoisted_vars = set()
        return super().emit(module)

    # --- Preamble ---

    def _emit_preamble(self, module: Module) -> bool:
        """Emit helper functions needed by generated code."""
        emitted = False
        if ir_contains_call(module, "range"):
            self._emit_range_helper()
            emitted = True
        if ir_contains_call(module, "divmod"):
            self._line("function divmod(a, b) { return [Math.floor(a / b), a % b]; }")
            emitted = True
        if ir_contains_call(module, "pow"):
            self._line("function pow(base, exp) { return Math.pow(base, exp); }")
            emitted = True
        if ir_contains_call(module, "abs"):
            self._line("function abs(x) { return Math.abs(x); }")
            emitted = True
        if ir_contains_call(module, "min"):
            self._line("function min(...args) { return Math.min(...args); }")
            emitted = True
        if ir_contains_call(module, "max"):
            self._line("function max(...args) { return Math.max(...args); }")
            emitted = True
        if ir_contains_call(module, "round"):
            self._emit_round_helper()
            emitted = True
        if ir_contains_call(module, "bytes"):
            self._line(
                "function bytes(x) { return Array.isArray(x) ? x.slice() : new Array(x).fill(0); }"
            )
            emitted = True
        if ir_has_bytes_ops(module) or ir_contains_call(module, "sorted"):
            self._emit_bytes_helpers()
            emitted = True
        if ir_contains_call(module, "sum"):
            self._line("function sum(arr) { return [...arr].reduce((a, b) => a + b, 0); }")
            emitted = True
        if ir_contains_call(module, "all"):
            self._line("function all(arr) { return [...arr].every(Boolean); }")
            emitted = True
        if ir_contains_call(module, "any"):
            self._line("function any(arr) { return [...arr].some(Boolean); }")
            emitted = True
        if ir_contains_call(module, "sorted"):
            self._line(
                "function sorted(arr, reverse) { let r = [...arr].sort((a, b) => a < b ? -1 : a > b ? 1 : 0); return reverse ? r.reverse() : r; }"
            )
            emitted = True
        if ir_contains_call(module, "enumerate"):
            self._line("function enumerate(arr) { return [...arr].map((v, i) => [i, v]); }")
            emitted = True
        if ir_contains_call(module, "list"):
            self._line("function list(x) { return typeof x === 'string' ? [...x] : [...x]; }")
            emitted = True
        if ir_contains_call(module, "zip"):
            self._line(
                "function zip(...arrs) { const len = Math.min(...arrs.map(a => a.length)); return Array.from({length: len}, (_, i) => arrs.map(a => a[i])); }"
            )
            emitted = True
        if ir_contains_call(module, "tuple"):
            self._line("function tuple(x) { if (x === undefined) return []; return typeof x === 'string' ? [...x] : [...x]; }")
            emitted = True
        if ir_contains_call(module, "set"):
            self._line("function set(x) { if (x === undefined) return new Set(); return new Set(x); }")
            emitted = True
        if ir_has_tuple_sets(module):
            self._emit_tuple_set_helpers()
            emitted = True
        if ir_has_tuple_maps(module):
            self._emit_tuple_map_helpers()
            emitted = True
        if ir_contains_call(module, "dict"):
            self._line("function dict(x) { if (x === undefined) return new Map(); return new Map(x); }")
            emitted = True
        if ir_has_bytes_ops(module) or ir_has_tuple_sets(module) or ir_has_tuple_maps(module):
            self._emit_map_helpers()
            emitted = True
        return emitted

    def _emit_tuple_set_helpers(self) -> None:
        """Emit helper functions for sets with tuple elements."""
        self._line("function tupleSetAdd(s, t) {")
        self.indent += 1
        self._line("for (const x of s) if (arrEq(x, t)) return;")
        self._line("s.add(t);")
        self.indent -= 1
        self._line("}")
        self._line("function tupleSetHas(s, t) {")
        self.indent += 1
        self._line("for (const x of s) if (arrEq(x, t)) return true;")
        self._line("return false;")
        self.indent -= 1
        self._line("}")

    def _emit_tuple_map_helpers(self) -> None:
        """Emit helper functions for maps with tuple keys."""
        self._line("function tupleMapGet(m, k) {")
        self.indent += 1
        self._line("for (const [key, val] of m) if (arrEq(key, k)) return val;")
        self._line("return undefined;")
        self.indent -= 1
        self._line("}")
        self._line("function tupleMapHas(m, k) {")
        self.indent += 1
        self._line("for (const [key] of m) if (arrEq(key, k)) return true;")
        self._line("return false;")
        self.indent -= 1
        self._line("}")

    def _emit_map_helpers(self) -> None:
        """Emit helper functions for map operations."""
        self._line("function mapEq(a, b) {")
        self.indent += 1
        self._line("if (a.size !== b.size) return false;")
        self._line("for (const [k, v] of a) {")
        self.indent += 1
        self._line("if (!b.has(k)) return false;")
        self._line("const bv = b.get(k);")
        self._line("if (Array.isArray(v) && Array.isArray(bv)) { if (!arrEq(v, bv)) return false; }")
        self._line("else if (v !== bv) return false;")
        self.indent -= 1
        self._line("}")
        self._line("return true;")
        self.indent -= 1
        self._line("}")

    def _emit_range_helper(self) -> None:
        self._line("function range(start, end, step) {")
        self.indent += 1
        self._line("if (end === undefined) { end = start; start = 0; }")
        self._line("if (step === undefined) { step = 1; }")
        self._line("const result = [];")
        self._line("if (step > 0) {")
        self.indent += 1
        self._line("for (var i = start; i < end; i += step) result.push(i);")
        self.indent -= 1
        self._line("} else {")
        self.indent += 1
        self._line("for (var i = start; i > end; i += step) result.push(i);")
        self.indent -= 1
        self._line("}")
        self._line("return result;")
        self.indent -= 1
        self._line("}")

    def _emit_round_helper(self) -> None:
        self._line("function round(x, n) {")
        self.indent += 1
        self._line("if (n === undefined) {")
        self.indent += 1
        self._line("let f = Math.floor(x), c = Math.ceil(x);")
        self._line("if (Math.abs(x - f) === 0.5) return f % 2 === 0 ? f : c;")
        self._line("return Math.round(x);")
        self.indent -= 1
        self._line("}")
        self._line("let m = Math.pow(10, n), v = x * m;")
        self._line("let f = Math.floor(v), c = Math.ceil(v);")
        self._line("if (Math.abs(v - f) === 0.5) return (f % 2 === 0 ? f : c) / m;")
        self._line("return Math.round(v) / m;")
        self.indent -= 1
        self._line("}")

    def _emit_bytes_helpers(self) -> None:
        """Emit helper functions for byte array operations."""
        self._line("function arrEq(a, b) {")
        self.indent += 1
        self._line("if (a.length !== b.length) return false;")
        self._line("for (let i = 0; i < a.length; i++) {")
        self.indent += 1
        self._line(
            "if (Array.isArray(a[i]) && Array.isArray(b[i])) { if (!arrEq(a[i], b[i])) return false; }"
        )
        self._line("else if (a[i] !== b[i]) return false;")
        self.indent -= 1
        self._line("}")
        self._line("return true;")
        self.indent -= 1
        self._line("}")
        self._line("function arrLt(a, b) {")
        self.indent += 1
        self._line("for (let i = 0; i < Math.min(a.length, b.length); i++) {")
        self.indent += 1
        self._line("if (a[i] < b[i]) return true;")
        self._line("if (a[i] > b[i]) return false;")
        self.indent -= 1
        self._line("}")
        self._line("return a.length < b.length;")
        self.indent -= 1
        self._line("}")
        self._line("function arrConcat(...arrs) { return [].concat(...arrs); }")
        self._line("function arrRepeat(a, n) {")
        self.indent += 1
        self._line("let r = []; for (let i = 0; i < n; i++) r.push(...a); return r;")
        self.indent -= 1
        self._line("}")
        self._line("function arrContains(h, n) {")
        self.indent += 1
        self._line("if (n.length === 0) return true;")
        self._line("outer: for (let i = 0; i <= h.length - n.length; i++) {")
        self.indent += 1
        self._line("for (let j = 0; j < n.length; j++) if (h[i+j] !== n[j]) continue outer;")
        self._line("return true;")
        self.indent -= 1
        self._line("}")
        self._line("return false;")
        self.indent -= 1
        self._line("}")
        self._line("function arrFind(h, n) {")
        self.indent += 1
        self._line("if (n.length === 0) return 0;")
        self._line("outer: for (let i = 0; i <= h.length - n.length; i++) {")
        self.indent += 1
        self._line("for (let j = 0; j < n.length; j++) if (h[i+j] !== n[j]) continue outer;")
        self._line("return i;")
        self.indent -= 1
        self._line("}")
        self._line("return -1;")
        self.indent -= 1
        self._line("}")
        self._line("function arrCount(h, n) {")
        self.indent += 1
        self._line("if (n.length === 0) return 0;")
        self._line("let c = 0, i = 0;")
        self._line("while (i <= h.length - n.length) {")
        self.indent += 1
        self._line("let m = true;")
        self._line("for (let j = 0; j < n.length; j++) if (h[i+j] !== n[j]) { m = false; break; }")
        self._line("if (m) { c++; i += n.length; } else { i++; }")
        self.indent -= 1
        self._line("}")
        self._line("return c;")
        self.indent -= 1
        self._line("}")
        self._line("function arrStartsWith(a, p) {")
        self.indent += 1
        self._line("if (p.length > a.length) return false;")
        self._line("for (let i = 0; i < p.length; i++) if (a[i] !== p[i]) return false;")
        self._line("return true;")
        self.indent -= 1
        self._line("}")
        self._line("function arrEndsWith(a, s) {")
        self.indent += 1
        self._line("if (s.length > a.length) return false;")
        self._line("let o = a.length - s.length;")
        self._line("for (let i = 0; i < s.length; i++) if (a[o+i] !== s[i]) return false;")
        self._line("return true;")
        self.indent -= 1
        self._line("}")
        self._line("function arrUpper(a) { return a.map(b => b >= 97 && b <= 122 ? b - 32 : b); }")
        self._line("function arrLower(a) { return a.map(b => b >= 65 && b <= 90 ? b + 32 : b); }")
        self._line("function arrStrip(a, cs) {")
        self.indent += 1
        self._line("let s = 0, e = a.length;")
        self._line("while (s < e && cs.includes(a[s])) s++;")
        self._line("while (e > s && cs.includes(a[e-1])) e--;")
        self._line("return a.slice(s, e);")
        self.indent -= 1
        self._line("}")
        self._line("function arrLstrip(a, cs) {")
        self.indent += 1
        self._line("let s = 0; while (s < a.length && cs.includes(a[s])) s++; return a.slice(s);")
        self.indent -= 1
        self._line("}")
        self._line("function arrRstrip(a, cs) {")
        self.indent += 1
        self._line(
            "let e = a.length; while (e > 0 && cs.includes(a[e-1])) e--; return a.slice(0, e);"
        )
        self.indent -= 1
        self._line("}")
        self._line("function arrSplit(a, sep) {")
        self.indent += 1
        self._line("let r = [], i = 0;")
        self._line("while (i <= a.length) {")
        self.indent += 1
        self._line("let j = arrFind(a.slice(i), sep);")
        self._line("if (j === -1) { r.push(a.slice(i)); break; }")
        self._line("r.push(a.slice(i, i + j)); i += j + sep.length;")
        self.indent -= 1
        self._line("}")
        self._line("return r;")
        self.indent -= 1
        self._line("}")
        self._line("function arrJoin(arrs, sep) {")
        self.indent += 1
        self._line("if (arrs.length === 0) return [];")
        self._line("let r = arrs[0].slice();")
        self._line("for (let i = 1; i < arrs.length; i++) { r.push(...sep); r.push(...arrs[i]); }")
        self._line("return r;")
        self.indent -= 1
        self._line("}")
        self._line("function arrReplace(a, old, nw) {")
        self.indent += 1
        self._line("if (old.length === 0) return a.slice();")
        self._line("let r = [], i = 0;")
        self._line("while (i < a.length) {")
        self.indent += 1
        self._line("if (arrStartsWith(a.slice(i), old)) { r.push(...nw); i += old.length; }")
        self._line("else { r.push(a[i]); i++; }")
        self.indent -= 1
        self._line("}")
        self._line("return r;")
        self.indent -= 1
        self._line("}")
        self._line("function arrStep(a, lo, hi, step) {")
        self.indent += 1
        self._line("if (lo === null) lo = step > 0 ? 0 : a.length - 1;")
        self._line("if (hi === null) hi = step > 0 ? a.length : -1;")
        self._line("let r = [];")
        self._line("if (step > 0) { for (let i = lo; i < hi; i += step) r.push(a[i]); }")
        self._line("else { for (let i = lo; i > hi; i += step) r.push(a[i]); }")
        self._line("return typeof a === 'string' ? r.join('') : r;")
        self.indent -= 1
        self._line("}")
        self._line("function deepArrEq(a, b) {")
        self.indent += 1
        self._line("if (a.length !== b.length) return false;")
        self._line("for (let i = 0; i < a.length; i++) {")
        self.indent += 1
        self._line("if (Array.isArray(a[i]) && Array.isArray(b[i])) {")
        self.indent += 1
        self._line("if (!arrEq(a[i], b[i])) return false;")
        self.indent -= 1
        self._line("} else if (a[i] !== b[i]) return false;")
        self.indent -= 1
        self._line("}")
        self._line("return true;")
        self.indent -= 1
        self._line("}")

    # --- Interface/Field (no-op for JS) ---

    def _emit_interface(self, iface: InterfaceDef) -> None:
        pass

    def _emit_field(self, fld: Field) -> None:
        pass

    # --- Signatures ---

    def _func_signature(self, name: str, params: list[Param], ret: Type) -> str:
        return f"function {_camel(name)}({self._param_list(params)})"

    def _method_signature(self, name: str, params: list[Param], ret: Type) -> str:
        return f"{_camel(name)}({self._param_list(params)})"

    def _param_list(self, params: list[Param]) -> str:
        return ", ".join(_camel(p.name) for p in params)

    # --- Variable declarations ---

    def _var_decl(self, name: str, typ: Type | None, value: Expr | None) -> None:
        if value is not None:
            val = self._expr(value)
            self._line(f"let {_camel(name)} = {val};")
        else:
            self._line(f"let {_camel(name)};")

    def _assign_decl(self, lv: str, value: Expr) -> None:
        val = self._expr(value)
        self._line(f"let {lv} = {val};")

    def _tuple_assign_decl(self, lvalues: str, value: Expr, value_type: Type | None) -> None:
        val = self._expr(value)
        self._line(f"let [{lvalues}] = {val};")

    def _var_decl_inline(self, name: str, typ: Type | None, value: Expr | None) -> str:
        if value is not None:
            return f"let {_camel(name)} = {self._expr(value)}"
        return f"let {_camel(name)}"

    def _assign_decl_inline(self, lv: str, value: Expr) -> str:
        return f"let {lv} = {self._expr(value)}"

    def _for_value_decl(
        self, name: str, iter_expr: str, index_name: str | None, elem_type: str
    ) -> None:
        self._line(f"const {name} = {iter_expr}[{index_name}];")

    # --- Exports ---

    def _emit_exports(self, symbols: list[str]) -> None:
        self._line("// CommonJS exports")
        self._line("if (typeof module !== 'undefined') {")
        self.indent += 1
        exports = ", ".join(symbols)
        self._line(f"module.exports = {{ {exports} }};")
        self.indent -= 1
        self._line("}")

    # --- Hoisted variables ---

    def _pre_function_hook(self) -> None:
        self._hoisted_vars = set()

    def _post_function_body(self, func: Function) -> None:
        if _is_void_func(func):
            self._line("return null;")

    def _hoisted_vars_hook(self, stmt: Stmt) -> None:
        hoisted = _get_hoisted_vars(stmt)
        for name, _ in hoisted:
            js_name = _camel(name)
            if name not in self._hoisted_vars:
                self._line(f"var {js_name};")
                self._hoisted_vars.add(name)

    def _emit_stmt(self, stmt: Stmt) -> None:
        # Handle Map assignment specially
        match stmt:
            case Assign(target=LValue() as target, value=value):
                if isinstance(target, IndexLV) and isinstance(target.obj.typ, Map):
                    obj_str = self._expr(target.obj)
                    idx_str = self._expr(target.index)
                    val = self._expr(value)
                    self._line(f"{obj_str}.set({idx_str}, {val});")
                    return
                # Check if variable was hoisted
                if isinstance(target, VarLV):
                    var_name = target.name
                    is_hoisted = var_name in self._hoisted_vars
                    if stmt.is_declaration and not is_hoisted:
                        lv = self._lvalue(target)
                        val = self._expr(value)
                        self._line(f"let {lv} = {val};")
                        return
                    elif stmt.is_declaration and is_hoisted:
                        lv = self._lvalue(target)
                        val = self._expr(value)
                        self._line(f"{lv} = {val};")
                        return
            case TupleAssign(targets=targets, value=value):
                lvalues = ", ".join(self._lvalue(t) for t in targets)
                val = self._expr(value)
                all_hoisted = all(
                    isinstance(t, VarLV) and t.name in self._hoisted_vars for t in targets
                )
                if stmt.is_declaration and not all_hoisted:
                    self._line(f"let [{lvalues}] = {val};")
                else:
                    new_targets = stmt.new_targets
                    for name in new_targets:
                        if name not in self._hoisted_vars:
                            self._line(f"let {_camel(name)};")
                    self._line(f"[{lvalues}] = {val};")
                return
        super()._emit_stmt(stmt)

    def _emit_tuple_reassign(
        self, stmt: TupleAssign, targets: list[LValue], lvalues: str, value: Expr
    ) -> None:
        new_targets = stmt.new_targets
        for name in new_targets:
            if name not in self._hoisted_vars:
                self._line(f"let {_camel(name)};")
        val = self._expr(value)
        self._line(f"[{lvalues}] = {val};")

    # --- Expressions ---

    def _slice_expr(
        self, obj: Expr, low: Expr | None, high: Expr | None, step: Expr | None = None
    ) -> str:
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

    def _call_expr(self, func: str, args: list[Expr]) -> str:
        # Handle float() with special string values
        if func == "float" and len(args) == 1:
            arg = args[0]
            if isinstance(arg, StringLit):
                if arg.value == "nan":
                    return "NaN"
                if arg.value == "inf":
                    return "Infinity"
                if arg.value == "-inf":
                    return "-Infinity"
            if isinstance(arg, IntLit):
                return self._expr(arg)
            return f"Number({self._expr(arg)})"
        args_str = ", ".join(self._expr(a) for a in args)
        return f"{_camel(func)}({args_str})"

    def _join_expr(self, sep: Expr, arr: Expr) -> str:
        if _is_bytes_join(sep, arr):
            return f"arrJoin({self._expr(arr)}, {self._expr(sep)})"
        return f"{self._expr(arr)}.join({self._expr(sep)})"

    def _cast_expr(self, inner: Expr, to_type: Type) -> str:
        # Handle float to string with decimal preservation
        if isinstance(to_type, Primitive) and to_type.kind == "string" and inner.typ == FLOAT:
            inner_str = self._expr(inner)
            return f"(Number.isInteger({inner_str}) ? {inner_str}.toFixed(1) : String({inner_str}))"
        # Handle None to string
        if (
            isinstance(inner, NilLit)
            and isinstance(to_type, Primitive)
            and to_type.kind == "string"
        ):
            return '"None"'
        return super()._cast_expr(inner, to_type)


# --- Helpers ---


def _is_void_func(func: Function) -> bool:
    """Check if a function returns void/None and needs implicit return null."""
    if func.ret != VOID:
        return False
    if func.body and isinstance(func.body[-1], Return):
        return False
    return True


def _get_hoisted_vars(stmt: Stmt) -> list[tuple[str, Type]]:
    """Get hoisted variables from a statement."""
    match stmt:
        case If():
            return stmt.hoisted_vars
        case TypeSwitch():
            return stmt.hoisted_vars
        case Match():
            return stmt.hoisted_vars
        case ForRange():
            return stmt.hoisted_vars
        case ForClassic():
            return stmt.hoisted_vars
        case While():
            return stmt.hoisted_vars
        case TryCatch():
            return stmt.hoisted_vars
        case _:
            return []


def _is_bytes_join(sep: Expr, arr: Expr) -> bool:
    """Check if this is a join operation on byte arrays."""
    if _is_bytes_list_type(arr.typ):
        return True
    if isinstance(arr, SliceLit) and arr.elements:
        if isinstance(arr.elements[0], SliceLit):
            return True
    if sep.typ is not None and is_bytes_type(sep.typ):
        return True
    return False


def _is_bool_int_compare_js(left: Expr, right: Expr) -> bool:
    """Extended bool/int comparison check for JavaScript."""
    if _is_bool_int_compare(left, right):
        return True
    l, r = left.typ, right.typ
    if l == BOOL or r == BOOL:
        if isinstance(left, BinaryOp) and left.op in ("+", "-", "*", "/", "//", "%"):
            return True
        if isinstance(right, BinaryOp) and right.op in ("+", "-", "*", "/", "//", "%"):
            return True
    if isinstance(left, Ternary) and _ternary_has_bool_int_mix(left):
        return True
    if isinstance(right, Ternary) and _ternary_has_bool_int_mix(right):
        return True
    return False


def _ternary_has_bool_int_mix(t: Ternary) -> bool:
    """Check if a ternary has both bool and int in its branches."""
    types = {t.then_expr.typ, t.else_expr.typ}
    return BOOL in types and INT in types
