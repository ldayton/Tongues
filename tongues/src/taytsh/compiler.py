"""Bytecode compiler for Taytsh — walks checked AST and emits bytecode."""

from __future__ import annotations
from typing import assert_never

from .ast import (
    TAssignStmt,
    TBinaryOp,
    TBoolLit,
    TBreakStmt,
    TByteLit,
    TBytesLit,
    TCall,
    TContinueStmt,
    TEnumDecl,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFnDecl,
    TFnLit,
    TFloatLit,
    TForStmt,
    TIdentType,
    TIfStmt,
    TIndex,
    TIntLit,
    TInterfaceDecl,
    TLetStmt,
    TListLit,
    TMapLit,
    TMatchStmt,
    TModule,
    TParam,
    TNilLit,
    TOpAssignStmt,
    TPatternEnum,
    TPatternNil,
    TPatternType,
    TPrimitive,
    TRange,
    TReturnStmt,
    TRuneLit,
    TSetLit,
    TSlice,
    TStmt,
    TStringLit,
    TStructDecl,
    TThrowStmt,
    TTernary,
    TTupleAccess,
    TTupleAssignStmt,
    TTupleLit,
    TTryStmt,
    TType,
    TUnaryOp,
    TVar,
    TWhileStmt,
    TListType,
    TMapType,
    TSetType,
    TTupleType,
)
from .check import (
    BOOL_T,
    BYTE_T,
    BYTES_T,
    FLOAT_T,
    INT_T,
    NIL_T,
    RUNE_T,
    STRING_T,
    VOID_T,
    EnumT,
    FnT,
    InterfaceT,
    ListT,
    MapT,
    SetT,
    StructT,
    TupleT,
    Type,
    check_with_info,
    contains_nil,
    type_eq,
)
from .bytecode import (
    CMP_GE,
    CMP_GT,
    CMP_LE,
    CMP_LT,
    CodeObject,
    CompiledModule,
    EnumDef,
    InterfaceDef,
    OP_ADD_BYTE,
    OP_ADD_FLOAT,
    OP_ADD_INT,
    OP_ADD_STRING,
    OP_BIT_AND,
    OP_BIT_NOT,
    OP_BIT_OR,
    OP_BIT_XOR,
    OP_BUILD_LIST,
    OP_BUILD_MAP,
    OP_BUILD_SET,
    OP_BUILD_STRUCT,
    OP_BUILD_TUPLE,
    OP_CALL,
    OP_CALL_BUILTIN,
    OP_CALL_METHOD,
    OP_CMP_BYTE,
    OP_CMP_FLOAT,
    OP_CMP_INT,
    OP_CMP_RUNE,
    OP_CMP_STRING,
    OP_CONST,
    OP_DIV_FLOAT,
    OP_DIV_INT,
    OP_DUP,
    OP_EQ,
    OP_EXTENDED_ARG,
    OP_FALSE,
    OP_FOR_ITER,
    OP_GET_FIELD,
    OP_GET_ITER,
    OP_INDEX,
    OP_IS_TYPE,
    OP_JUMP,
    OP_JUMP_BACK,
    OP_JUMP_IF_FALSE,
    OP_JUMP_IF_TRUE,
    OP_LOAD_BUILTIN,
    OP_LOAD_ENUM,
    OP_LOAD_GLOBAL,
    OP_LOAD_LOCAL,
    OP_STORE_GLOBAL,
    OP_MATCH_TYPE,
    OP_MOD_FLOAT,
    OP_MOD_INT,
    OP_MUL_BYTE,
    OP_MUL_FLOAT,
    OP_MUL_INT,
    OP_NE,
    OP_NEG_FLOAT,
    OP_NEG_INT,
    OP_NIL,
    OP_NOT,
    OP_POP,
    OP_POP_HANDLER,
    OP_PUSH_FINALLY,
    OP_PUSH_HANDLER,
    OP_RETURN,
    OP_RETURN_VOID,
    OP_ROT_TWO,
    OP_SET_FIELD,
    OP_SHIFT_LEFT,
    OP_SHIFT_RIGHT,
    OP_SHIFT_RIGHT_UNSIGNED,
    OP_SLICE,
    OP_STORE_INDEX,
    OP_STORE_LOCAL,
    OP_SUB_BYTE,
    OP_SUB_FLOAT,
    OP_SUB_INT,
    OP_THROW,
    OP_TRUE,
    OP_TUPLE_ACCESS,
    OP_UNPACK,
    OP_INT_ZERO,
    OP_INT_ONE,
    StructDef,
    Val,
    VByte,
    VBytes,
    VFloat,
    VFunc,
    VInt,
    VRune,
    VStr,
)


# ============================================================
# BUILTIN TABLE
# ============================================================

BUILTIN_TABLE: list[str] = [
    "Abs",
    "Add",
    "Append",
    "Args",
    "Assert",
    "ByteToInt",
    "Bytes",
    "BytesFrom",
    "Ceil",
    "Chars",
    "Concat",
    "Contains",
    "Count",
    "Decode",
    "Delete",
    "Difference",
    "DivMod",
    "Encode",
    "EndsWith",
    "Exit",
    "Find",
    "FloatToInt",
    "Floor",
    "FloorDiv",
    "Format",
    "FormatInt",
    "Get",
    "GetEnv",
    "IndexOf",
    "Insert",
    "IntToByte",
    "IntToFloat",
    "Intersection",
    "IsAlnum",
    "IsAlpha",
    "IsDigit",
    "IsInf",
    "IsLower",
    "IsNaN",
    "IsNil",
    "IsSpace",
    "IsType",
    "IsUpper",
    "Items",
    "Join",
    "Keys",
    "Len",
    "ListCompare",
    "ListFrom",
    "Lower",
    "Map",
    "MapFromKeys",
    "MapFromPairs",
    "Max",
    "Merge",
    "Min",
    "ParseFloat",
    "ParseInt",
    "Pop",
    "PopItem",
    "Pow",
    "PythonMod",
    "RFind",
    "RangeList",
    "ReadAll",
    "ReadBytes",
    "ReadBytesN",
    "ReadFile",
    "ReadFileBytes",
    "ReadLine",
    "Remove",
    "RemoveAt",
    "Repeat",
    "Replace",
    "ReplaceCount",
    "ReplaceSlice",
    "Reverse",
    "Reversed",
    "Round",
    "RuneFromInt",
    "RuneToInt",
    "Set",
    "SetFromList",
    "Sorted",
    "Split",
    "SplitN",
    "SplitWhitespace",
    "Sqrt",
    "StartsWith",
    "Sum",
    "ToRepr",
    "ToString",
    "Trim",
    "TrimEnd",
    "TrimStart",
    "Union",
    "Unwrap",
    "Upper",
    "Values",
    "WrappingAdd",
    "WrappingMul",
    "WrappingSub",
    "WriteErr",
    "WriteFile",
    "WriteOut",
    "WritelnErr",
    "WritelnOut",
    "Zip",
]
_BUILTIN_INDEX: dict[str, int] = {}
for _bi, _bname in enumerate(BUILTIN_TABLE):
    _BUILTIN_INDEX[_bname] = _bi


# ============================================================
# SCOPE
# ============================================================


class _Local:
    __slots__ = ("slot", "typ")

    def __init__(self, slot: int, typ: Type) -> None:
        self.slot: int = slot
        self.typ: Type = typ


class _Scope:
    __slots__ = ("locals", "parent")

    def __init__(self, parent: _Scope | None) -> None:
        self.locals: dict[str, _Local] = {}
        self.parent: _Scope | None = parent

    def lookup(self, name: str) -> _Local | None:
        if name in self.locals:
            return self.locals[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None


# ============================================================
# LOOP CONTEXT
# ============================================================


class _LoopCtx:
    __slots__ = ("break_patches", "continue_target", "handler_depth", "match_depth")

    def __init__(self, handler_depth: int, match_depth: int) -> None:
        self.break_patches: list[int] = []
        self.continue_target: int = -1
        self.handler_depth: int = handler_depth
        self.match_depth: int = match_depth


# ============================================================
# FUNCTION COMPILER
# ============================================================


class _FnCompiler:
    """Compiles a single function body into a CodeObject."""

    def __init__(self, compiler: Compiler, name: str) -> None:
        self.compiler: Compiler = compiler
        self.name: str = name
        self.code: list[int] = []
        self.constants: list[Val] = []
        self.lines: list[int] = []
        self.local_names: list[str] = []
        self.next_slot: int = 0
        self.scope: _Scope = _Scope(None)
        self.loop_stack: list[_LoopCtx] = []
        self.handler_depth: int = 0
        self.match_depth: int = 0

    def emit(self, op: int, arg: int, line: int) -> int:
        """Emit a single instruction pair. Returns the offset of the opcode."""
        if arg > 255:
            # Chain EXTENDED_ARG for larger operands
            hi = arg >> 8
            if hi > 255:
                hi2 = hi >> 8
                self.code.append(OP_EXTENDED_ARG)
                self.code.append(hi2 & 0xFF)
                self.lines.append(line)
                self.code.append(OP_EXTENDED_ARG)
                self.code.append(hi & 0xFF)
                self.lines.append(line)
            else:
                self.code.append(OP_EXTENDED_ARG)
                self.code.append(hi)
                self.lines.append(line)
            offset = len(self.code)
            self.code.append(op)
            self.code.append(arg & 0xFF)
            self.lines.append(line)
            return offset
        offset = len(self.code)
        self.code.append(op)
        self.code.append(arg)
        self.lines.append(line)
        return offset

    def emit_const(self, val: Val, line: int) -> None:
        idx = len(self.constants)
        self.constants.append(val)
        self.emit(OP_CONST, idx, line)

    def add_local(self, name: str, typ: Type) -> int:
        slot = self.next_slot
        self.next_slot += 1
        self.scope.locals[name] = _Local(slot, typ)
        if slot >= len(self.local_names):
            self.local_names.append(name)
        return slot

    def patch_jump(self, offset: int) -> None:
        """Patch a forward jump at `offset` to point to current position."""
        target = len(self.code)
        diff = target - offset - 2
        if diff < 0:
            diff = 0
        # Patch the arg byte
        self.code[offset + 1] = diff & 0xFF
        if diff > 255:
            # Need to find the EXTENDED_ARG before and patch it
            if offset >= 2 and self.code[offset - 2] == OP_EXTENDED_ARG:
                self.code[offset - 1] = (diff >> 8) & 0xFF
                if offset >= 4 and self.code[offset - 4] == OP_EXTENDED_ARG:
                    self.code[offset - 3] = (diff >> 16) & 0xFF

    def emit_jump(self, op: int, line: int) -> int:
        """Emit a forward jump with placeholder arg. Returns offset to patch."""
        # Reserve EXTENDED_ARG so patch_jump can fill it in for large offsets
        self.code.append(OP_EXTENDED_ARG)
        self.code.append(0)
        self.lines.append(line)
        offset = len(self.code)
        self.code.append(op)
        self.code.append(0)
        self.lines.append(line)
        return offset

    def current_offset(self) -> int:
        return len(self.code)

    def emit_jump_back(self, target: int, line: int) -> None:
        """Emit a backward jump to target offset, accounting for EXTENDED_ARG sizing."""
        back_dist = self.current_offset() - target + 2
        if back_dist > 255:
            back_dist += 2  # EXTENDED_ARG prefix adds 2 bytes to ip advance
        self.emit(OP_JUMP_BACK, back_dist, line)

    def to_code_object(self, param_count: int) -> CodeObject:
        return CodeObject(
            name=self.name,
            param_count=param_count,
            local_count=self.next_slot,
            code=self.code,
            constants=self.constants,
            lines=self.lines,
            local_names=self.local_names,
        )


# ============================================================
# COMPILER
# ============================================================


class CompileError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class Compiler:
    def __init__(self) -> None:
        self.code_objects: list[CodeObject] = []
        self.global_names: list[str] = []
        self._global_index: dict[str, int] = {}
        self.struct_defs: list[StructDef] = []
        self._struct_index: dict[str, int] = {}
        self.enum_defs: list[EnumDef] = []
        self._enum_index: dict[str, int] = {}
        self.interface_defs: list[InterfaceDef] = []
        self._interface_index: dict[str, int] = {}
        self.entry_index: int = -1
        # Checker state
        self.global_let_types: dict[str, Type] = {}
        self.checker_types: dict[str, Type] = {}
        self.checker_functions: dict[str, FnT] = {}
        self.fn_param_names: dict[str, list[str]] = {}
        self._zero_value_building: set[str] = set()

    def compile_module(self, module: TModule) -> CompiledModule:
        result = check_with_info(module)
        errors = result[0]
        checker = result[1]
        if errors:
            msgs: list[str] = []
            for err in errors:
                msgs.append(repr(err))
            raise CompileError("\n".join(msgs))
        self.checker_types = checker.types
        self.checker_functions = checker.functions
        self.fn_param_names = checker.fn_param_names
        # Collect struct/enum/interface defs
        for decl in module.decls:
            match decl:
                case TStructDecl():
                    self._register_struct(decl)
                case TEnumDecl():
                    self._register_enum(decl)
                case TInterfaceDecl():
                    self._register_interface(decl)
        # Register all top-level function and let names as globals
        for decl in module.decls:
            match decl:
                case TFnDecl():
                    self._ensure_global(decl.name)
                case TLetStmt():
                    self._ensure_global(decl.name)
                    self.global_let_types[decl.name] = self._resolve_ttype(decl.typ)
        # Compile all top-level functions
        for decl in module.decls:
            if isinstance(decl, TFnDecl):
                code_idx = self._compile_fn_decl(decl)
                gidx = self._global_index[decl.name]
                if decl.name == "Main":
                    self.entry_index = code_idx
                # Store function code index in global_names mapping
                self._global_index[decl.name] = gidx
        # Compile struct methods
        for decl in module.decls:
            if isinstance(decl, TStructDecl):
                self._compile_struct_methods(decl)
        # Compile top-level let initializers into a synthetic __init__ function
        init_index = self._compile_top_level_lets(module)
        return CompiledModule(
            code_objects=self.code_objects,
            global_names=self.global_names,
            struct_defs=self.struct_defs,
            enum_defs=self.enum_defs,
            interface_defs=self.interface_defs,
            entry_index=self.entry_index,
            init_index=init_index,
        )

    def _compile_top_level_lets(self, module: TModule) -> int:
        """Compile top-level let initializers into a synthetic function. Returns code index or -1."""
        lets: list[TLetStmt] = []
        for decl in module.decls:
            if isinstance(decl, TLetStmt):
                lets.append(decl)
        if not lets:
            return -1
        fc = _FnCompiler(self, "__init__")
        for stmt in lets:
            gidx = self._global_index[stmt.name]
            if stmt.value is not None:
                self._compile_expr(stmt.value, fc)
            else:
                typ = self._resolve_ttype(stmt.typ)
                self._emit_zero_value(typ, fc, stmt.pos.line)
            fc.emit(OP_STORE_GLOBAL, gidx, stmt.pos.line)
        fc.emit(OP_RETURN_VOID, 0, 0)
        code_idx = len(self.code_objects)
        self.code_objects.append(fc.to_code_object(0))
        return code_idx

    def _ensure_global(self, name: str) -> int:
        if name in self._global_index:
            return self._global_index[name]
        idx = len(self.global_names)
        self.global_names.append(name)
        self._global_index[name] = idx
        return idx

    def _register_struct(self, decl: TStructDecl) -> None:
        ct = self.checker_types.get(decl.name)
        if ct is None or not isinstance(ct, StructT):
            return
        field_names: list[str] = []
        field_types: list[Type] = []
        for f in decl.fields:
            field_names.append(f.name)
            field_types.append(ct.fields[f.name])
        sd = StructDef(
            name=decl.name,
            field_names=field_names,
            field_types=field_types,
            parent=decl.parent,
            method_names=[],
            method_indices=[],
        )
        self._struct_index[decl.name] = len(self.struct_defs)
        self.struct_defs.append(sd)

    def _register_enum(self, decl: TEnumDecl) -> None:
        ed = EnumDef(name=decl.name, variants=list(decl.variants))
        self._enum_index[decl.name] = len(self.enum_defs)
        self.enum_defs.append(ed)

    def _register_interface(self, decl: TInterfaceDecl) -> None:
        ct = self.checker_types.get(decl.name)
        if ct is None or not isinstance(ct, InterfaceT):
            return
        idef = InterfaceDef(
            name=decl.name,
            variant_names=list(ct.variants),
        )
        self._interface_index[decl.name] = len(self.interface_defs)
        self.interface_defs.append(idef)

    def _compile_struct_methods(self, decl: TStructDecl) -> None:
        sidx = self._struct_index.get(decl.name)
        if sidx is None:
            return
        sd = self.struct_defs[sidx]
        ct = self.checker_types[decl.name]
        if not isinstance(ct, StructT):
            return
        for method in decl.methods:
            fc = _FnCompiler(self, decl.name + "." + method.name)
            # 'self' param is slot 0
            fc.add_local("this", ct)
            for p in method.params:
                if p.typ is not None:
                    pt = self._resolve_param_type(p)
                    fc.add_local(p.name, pt)
            self._collect_locals(method.body, fc)
            self._emit_default_preamble(method.params, fc, method.pos.line)
            self._compile_block(method.body, fc)
            mt = ct.methods.get(method.name)
            ret_type = mt.ret if mt is not None else VOID_T
            if type_eq(ret_type, VOID_T):
                fc.emit(OP_RETURN_VOID, 0, method.pos.line)
            code_idx = len(self.code_objects)
            self.code_objects.append(fc.to_code_object(len(method.params) + 1))
            sd.method_names.append(method.name)
            sd.method_indices.append(code_idx)

    def _compile_fn_decl(self, decl: TFnDecl) -> int:
        fc = _FnCompiler(self, decl.name)
        # Register params as locals
        for p in decl.params:
            pt = self._resolve_param_type(p)
            fc.add_local(p.name, pt)
        # Pre-scan body for all let statements to assign slots
        self._collect_locals(decl.body, fc)
        self._emit_default_preamble(decl.params, fc, decl.pos.line)
        # Compile body
        self._compile_block(decl.body, fc)
        # Implicit return void
        fn_type = self.checker_functions.get(decl.name)
        ret_type = fn_type.ret if fn_type is not None else VOID_T
        if type_eq(ret_type, VOID_T):
            fc.emit(OP_RETURN_VOID, 0, decl.pos.line)
        code_idx = len(self.code_objects)
        co = fc.to_code_object(len(decl.params))
        if fn_type is not None:
            co.type_sig = self._format_fn_sig(fn_type)
        self.code_objects.append(co)
        return code_idx

    def _format_fn_sig(self, fn_type: FnT) -> str:
        parts = [self._format_type(p) for p in fn_type.params]
        parts.append(self._format_type(fn_type.ret))
        return "[" + ", ".join(parts) + "]"

    def _format_type(self, t: Type) -> str:
        if type_eq(t, INT_T):
            return "int"
        if type_eq(t, FLOAT_T):
            return "float"
        if type_eq(t, BOOL_T):
            return "bool"
        if type_eq(t, STRING_T):
            return "string"
        if type_eq(t, BYTE_T):
            return "byte"
        if type_eq(t, RUNE_T):
            return "rune"
        if type_eq(t, VOID_T):
            return "void"
        if isinstance(t, FnT):
            return "fn" + self._format_fn_sig(t)
        return "any"

    def _resolve_param_type(self, p: TParam) -> Type:
        """Resolve a TParam's type to a checker Type."""
        if p.typ is None:
            return VOID_T
        return self._resolve_ttype(p.typ)

    def _resolve_ttype(self, tt: TType) -> Type:
        """Resolve a parse-time TType to a checker Type."""
        if isinstance(tt, TPrimitive):
            if tt.kind == "int":
                return INT_T
            if tt.kind == "float":
                return FLOAT_T
            if tt.kind == "bool":
                return BOOL_T
            if tt.kind == "byte":
                return BYTE_T
            if tt.kind == "bytes":
                return BYTES_T
            if tt.kind == "string":
                return STRING_T
            if tt.kind == "rune":
                return RUNE_T
            if tt.kind == "nil":
                return NIL_T
            if tt.kind == "void":
                return VOID_T
        if isinstance(tt, TIdentType):
            t = self.checker_types.get(tt.name)
            if t is not None:
                return t
        if isinstance(tt, TTupleType):
            elts: list[Type] = []
            for e in tt.elements:
                elts.append(self._resolve_ttype(e))
            return TupleT(kind="tuple", elements=elts)
        if isinstance(tt, TListType):
            return ListT(kind="list", element=self._resolve_ttype(tt.element))
        if isinstance(tt, TMapType):
            return MapT(
                kind="map",
                key=self._resolve_ttype(tt.key),
                value=self._resolve_ttype(tt.value),
            )
        if isinstance(tt, TSetType):
            return SetT(kind="set", element=self._resolve_ttype(tt.element))
        return VOID_T

    def _collect_locals(self, stmts: list[TStmt], fc: _FnCompiler) -> None:
        """Pre-scan statements to allocate local slots for let declarations."""
        for stmt in stmts:
            match stmt:
                case TLetStmt():
                    if fc.scope.lookup(stmt.name) is None:
                        typ = self._resolve_ttype(stmt.typ)
                        fc.add_local(stmt.name, typ)
                case TIfStmt():
                    self._collect_locals(stmt.then_body, fc)
                    if stmt.else_body is not None:
                        self._collect_locals(stmt.else_body, fc)
                case TWhileStmt():
                    self._collect_locals(stmt.body, fc)
                case TForStmt():
                    for b in stmt.binding:
                        if b != "_" and fc.scope.lookup(b) is None:
                            fc.add_local(b, INT_T)
                    self._collect_locals(stmt.body, fc)
                case TMatchStmt():
                    for case in stmt.cases:
                        if isinstance(case.pattern, TPatternType):
                            pname = case.pattern.name
                            if fc.scope.lookup(pname) is None:
                                pt = self._resolve_ttype(case.pattern.type_name)
                                fc.add_local(pname, pt)
                        self._collect_locals(case.body, fc)
                    if stmt.default is not None:
                        dname = stmt.default.name
                        if dname is not None:
                            if fc.scope.lookup(dname) is None:
                                fc.add_local(dname, VOID_T)
                        self._collect_locals(stmt.default.body, fc)
                case TTryStmt():
                    self._collect_locals(stmt.body, fc)
                    for catch in stmt.catches:
                        if fc.scope.lookup(catch.name) is None:
                            fc.add_local(catch.name, VOID_T)
                        self._collect_locals(catch.body, fc)
                    if stmt.finally_body is not None:
                        self._collect_locals(stmt.finally_body, fc)

    # ── Statement compilation ─────────────────────────────────

    def _compile_block(self, stmts: list[TStmt], fc: _FnCompiler) -> None:
        for stmt in stmts:
            self._compile_stmt(stmt, fc)

    def _compile_stmt(self, stmt: TStmt, fc: _FnCompiler) -> None:
        match stmt:
            case TLetStmt():
                self._compile_let(stmt, fc)
            case TAssignStmt():
                self._compile_assign(stmt, fc)
            case TOpAssignStmt():
                self._compile_op_assign(stmt, fc)
            case TTupleAssignStmt():
                self._compile_tuple_assign(stmt, fc)
            case TExprStmt():
                if (
                    isinstance(stmt.expr, TCall)
                    and isinstance(stmt.expr.func, TVar)
                    and stmt.expr.func.name == "reveal_type"
                ):
                    return
                self._compile_expr(stmt.expr, fc)
                fc.emit(OP_POP, 0, stmt.pos.line)
            case TReturnStmt():
                if stmt.value is not None:
                    self._compile_expr(stmt.value, fc)
                    fc.emit(OP_RETURN, 0, stmt.pos.line)
                else:
                    fc.emit(OP_RETURN_VOID, 0, stmt.pos.line)
            case TIfStmt():
                self._compile_if(stmt, fc)
            case TWhileStmt():
                self._compile_while(stmt, fc)
            case TForStmt():
                self._compile_for(stmt, fc)
            case TBreakStmt():
                self._compile_break(stmt, fc)
            case TContinueStmt():
                self._compile_continue(stmt, fc)
            case TTryStmt():
                self._compile_try(stmt, fc)
            case TThrowStmt():
                self._compile_expr(stmt.expr, fc)
                fc.emit(OP_THROW, 0, stmt.pos.line)
            case TMatchStmt():
                self._compile_match(stmt, fc)
            case _:
                assert_never(stmt)

    def _compile_let(self, stmt: TLetStmt, fc: _FnCompiler) -> None:
        local = fc.scope.lookup(stmt.name)
        if local is None:
            return
        if stmt.value is not None:
            self._compile_expr(stmt.value, fc)
        else:
            # Zero-value initialization
            typ = local.typ
            self._emit_zero_value(typ, fc, stmt.pos.line)
        fc.emit(OP_STORE_LOCAL, local.slot, stmt.pos.line)

    def _emit_zero_value(self, typ: Type, fc: _FnCompiler, line: int) -> None:
        if type_eq(typ, INT_T):
            fc.emit(OP_INT_ZERO, 0, line)
        elif type_eq(typ, FLOAT_T):
            fc.emit_const(VFloat(0.0), line)
        elif type_eq(typ, BOOL_T):
            fc.emit(OP_FALSE, 0, line)
        elif type_eq(typ, BYTE_T):
            fc.emit_const(VByte(0), line)
        elif type_eq(typ, STRING_T):
            fc.emit_const(VStr(""), line)
        elif type_eq(typ, RUNE_T):
            fc.emit_const(VRune("\x00"), line)
        elif type_eq(typ, BYTES_T):
            fc.emit_const(VBytes(b""), line)
        elif isinstance(typ, TupleT):
            for et in typ.elements:
                self._emit_zero_value(et, fc, line)
            fc.emit(OP_BUILD_TUPLE, len(typ.elements), line)
        elif isinstance(typ, ListT):
            fc.emit(OP_BUILD_LIST, 0, line)
        elif isinstance(typ, MapT):
            fc.emit(OP_BUILD_MAP, 0, line)
        elif isinstance(typ, SetT):
            fc.emit(OP_BUILD_SET, 0, line)
        elif isinstance(typ, StructT):
            sidx = self._struct_index.get(typ.name)
            if sidx is not None and typ.name not in self._zero_value_building:
                self._zero_value_building.add(typ.name)
                sd = self.struct_defs[sidx]
                for ft in sd.field_types:
                    self._emit_zero_value(ft, fc, line)
                fc.emit(OP_BUILD_STRUCT, sidx, line)
                self._zero_value_building.discard(typ.name)
            else:
                fc.emit(OP_NIL, 0, line)
        else:
            fc.emit(OP_NIL, 0, line)

    def _emit_default_preamble(
        self, params: list[TParam], fc: _FnCompiler, line: int
    ) -> None:
        """Emit nil-checks for params with defaults: replace nil with zero value."""
        for p in params:
            if not p.has_default:
                continue
            pt = self._resolve_param_type(p)
            if contains_nil(pt):
                continue
            local = fc.scope.lookup(p.name)
            if local is None:
                continue
            fc.emit(OP_LOAD_LOCAL, local.slot, line)
            fc.emit(OP_NIL, 0, line)
            fc.emit(OP_EQ, 0, line)
            jump = fc.emit_jump(OP_JUMP_IF_FALSE, line)
            self._emit_zero_value(pt, fc, line)
            fc.emit(OP_STORE_LOCAL, local.slot, line)
            fc.patch_jump(jump)

    def _compile_assign(self, stmt: TAssignStmt, fc: _FnCompiler) -> None:
        if isinstance(stmt.target, TVar):
            self._compile_expr(stmt.value, fc)
            local = fc.scope.lookup(stmt.target.name)
            if local is not None:
                fc.emit(OP_STORE_LOCAL, local.slot, stmt.pos.line)
            return
        if isinstance(stmt.target, TIndex):
            self._compile_expr(stmt.target.obj, fc)
            self._compile_expr(stmt.target.index, fc)
            self._compile_expr(stmt.value, fc)
            fc.emit(OP_STORE_INDEX, 0, stmt.pos.line)
            return
        if isinstance(stmt.target, TFieldAccess):
            self._compile_expr(stmt.target.obj, fc)
            self._compile_expr(stmt.value, fc)
            field_name = stmt.target.field
            fidx = self._field_const(field_name, fc, stmt.pos.line)
            fc.emit(OP_SET_FIELD, fidx, stmt.pos.line)
            return

    def _compile_op_assign(self, stmt: TOpAssignStmt, fc: _FnCompiler) -> None:
        if isinstance(stmt.target, TVar):
            local = fc.scope.lookup(stmt.target.name)
            if local is None:
                return
            fc.emit(OP_LOAD_LOCAL, local.slot, stmt.pos.line)
            self._compile_expr(stmt.value, fc)
            self._emit_binop_for_type(stmt.op, local.typ, fc, stmt.pos.line)
            fc.emit(OP_STORE_LOCAL, local.slot, stmt.pos.line)
            return
        if isinstance(stmt.target, TIndex):
            # obj[index] op= value → obj[index] = obj[index] op value
            # Push obj and index for the store (evaluated first)
            self._compile_expr(stmt.target.obj, fc)
            self._compile_expr(stmt.target.index, fc)
            # Load current value (re-evaluate obj[index])
            self._compile_expr(stmt.target, fc)
            # Compute new value
            self._compile_expr(stmt.value, fc)
            typ = self._resolve_expr_type(stmt.target, fc)
            self._emit_binop_for_type(stmt.op, typ, fc, stmt.pos.line)
            # Stack: obj, index, result → OP_STORE_INDEX
            fc.emit(OP_STORE_INDEX, 0, stmt.pos.line)
            return
        if isinstance(stmt.target, TFieldAccess):
            self._compile_expr(stmt.target, fc)
            self._compile_expr(stmt.value, fc)
            typ = self._resolve_expr_type(stmt.target, fc)
            self._emit_binop_for_type(stmt.op, typ, fc, stmt.pos.line)
            self._compile_expr(stmt.target.obj, fc)
            fc.emit(OP_ROT_TWO, 0, stmt.pos.line)
            fidx = self._field_const(stmt.target.field, fc, stmt.pos.line)
            fc.emit(OP_SET_FIELD, fidx, stmt.pos.line)
            return

    def _compile_tuple_assign(self, stmt: TTupleAssignStmt, fc: _FnCompiler) -> None:
        self._compile_expr(stmt.value, fc)
        n = len(stmt.targets)
        fc.emit(OP_UNPACK, n, stmt.pos.line)
        # After UNPACK, stack has n values, last on top — store in reverse
        i = n - 1
        while i >= 0:
            target = stmt.targets[i]
            if isinstance(target, TVar):
                if target.name == "_":
                    fc.emit(OP_POP, 0, stmt.pos.line)
                else:
                    local = fc.scope.lookup(target.name)
                    if local is not None:
                        fc.emit(OP_STORE_LOCAL, local.slot, stmt.pos.line)
                    else:
                        fc.emit(OP_POP, 0, stmt.pos.line)
            else:
                fc.emit(OP_POP, 0, stmt.pos.line)
            i -= 1

    def _compile_if(self, stmt: TIfStmt, fc: _FnCompiler) -> None:
        self._compile_expr(stmt.cond, fc)
        false_jump = fc.emit_jump(OP_JUMP_IF_FALSE, stmt.pos.line)
        self._compile_block(stmt.then_body, fc)
        if stmt.else_body is not None:
            end_jump = fc.emit_jump(OP_JUMP, stmt.pos.line)
            fc.patch_jump(false_jump)
            self._compile_block(stmt.else_body, fc)
            fc.patch_jump(end_jump)
        else:
            fc.patch_jump(false_jump)

    def _compile_while(self, stmt: TWhileStmt, fc: _FnCompiler) -> None:
        loop_start = fc.current_offset()
        loop_ctx = _LoopCtx(fc.handler_depth, fc.match_depth)
        loop_ctx.continue_target = loop_start
        fc.loop_stack.append(loop_ctx)
        self._compile_expr(stmt.cond, fc)
        exit_jump = fc.emit_jump(OP_JUMP_IF_FALSE, stmt.pos.line)
        self._compile_block(stmt.body, fc)
        # Jump back to loop start
        fc.emit_jump_back(loop_start, stmt.pos.line)
        fc.patch_jump(exit_jump)
        # Patch breaks
        for bp in loop_ctx.break_patches:
            fc.patch_jump(bp)
        fc.loop_stack.pop()

    def _compile_for(self, stmt: TForStmt, fc: _FnCompiler) -> None:
        if isinstance(stmt.iterable, TRange):
            self._compile_for_range(stmt, stmt.iterable, fc)
        else:
            self._compile_for_iter(stmt, fc)

    def _compile_for_range(self, stmt: TForStmt, rng: TRange, fc: _FnCompiler) -> None:
        # Compile range args and emit GET_ITER
        for a in rng.args:
            self._compile_expr(a, fc)
        fc.emit(OP_GET_ITER, len(rng.args), stmt.pos.line)
        # Binding variable
        binding = stmt.binding[0] if stmt.binding else "_"
        local = fc.scope.lookup(binding) if binding != "_" else None
        loop_start = fc.current_offset()
        loop_ctx = _LoopCtx(fc.handler_depth, fc.match_depth)
        loop_ctx.continue_target = loop_start
        fc.loop_stack.append(loop_ctx)
        exit_jump = fc.emit_jump(OP_FOR_ITER, stmt.pos.line)
        if local is not None:
            fc.emit(OP_STORE_LOCAL, local.slot, stmt.pos.line)
        else:
            fc.emit(OP_POP, 0, stmt.pos.line)
        self._compile_block(stmt.body, fc)
        fc.emit_jump_back(loop_start, stmt.pos.line)
        fc.patch_jump(exit_jump)
        for bp in loop_ctx.break_patches:
            fc.patch_jump(bp)
        # Pop iterator state (3 values: current, end, step)
        fc.emit(OP_POP, 0, stmt.pos.line)
        fc.emit(OP_POP, 0, stmt.pos.line)
        fc.emit(OP_POP, 0, stmt.pos.line)
        fc.loop_stack.pop()

    def _compile_for_iter(self, stmt: TForStmt, fc: _FnCompiler) -> None:
        # Compile iterable, push index counter
        self._compile_expr(stmt.iterable, fc)
        fc.emit(OP_INT_ZERO, 0, stmt.pos.line)  # index
        loop_start = fc.current_offset()
        loop_ctx = _LoopCtx(fc.handler_depth, fc.match_depth)
        loop_ctx.continue_target = loop_start
        fc.loop_stack.append(loop_ctx)
        # FOR_ITER checks index < len(collection)
        exit_jump = fc.emit_jump(OP_FOR_ITER, stmt.pos.line)
        # After FOR_ITER pushes current element (and optionally index)
        if (
            len(stmt.binding) >= 2
            and stmt.annotations.get("iter_kind") == "tuple_unpack"
        ):
            # for a, b in list_of_tuples — stack has: index_val, tuple_element
            # Unpack tuple, then store each binding, then discard index
            n = len(stmt.binding)
            fc.emit(OP_UNPACK, n, stmt.pos.line)
            # Stack now has: index_val, elem[0], elem[1], ... elem[n-1]
            # Store in reverse order (top of stack = last element)
            i = n - 1
            while i >= 0:
                b = stmt.binding[i]
                local = fc.scope.lookup(b) if b != "_" else None
                if local is not None:
                    fc.emit(OP_STORE_LOCAL, local.slot, stmt.pos.line)
                else:
                    fc.emit(OP_POP, 0, stmt.pos.line)
                i -= 1
            # Discard the index pushed by FOR_ITER
            fc.emit(OP_POP, 0, stmt.pos.line)
        elif len(stmt.binding) == 2:
            # for i, v in collection — stack has: index_val, element
            idx_binding = stmt.binding[0]
            val_binding = stmt.binding[1]
            val_local = fc.scope.lookup(val_binding) if val_binding != "_" else None
            idx_local = fc.scope.lookup(idx_binding) if idx_binding != "_" else None
            if val_local is not None:
                fc.emit(OP_STORE_LOCAL, val_local.slot, stmt.pos.line)
            else:
                fc.emit(OP_POP, 0, stmt.pos.line)
            if idx_local is not None:
                fc.emit(OP_STORE_LOCAL, idx_local.slot, stmt.pos.line)
            else:
                fc.emit(OP_POP, 0, stmt.pos.line)
        else:
            binding = stmt.binding[0] if stmt.binding else "_"
            local = fc.scope.lookup(binding) if binding != "_" else None
            if stmt.annotations.get("iter_kind") == "map":
                # Map 1-var: stack has key, value (top). Discard value, keep key.
                fc.emit(OP_POP, 0, stmt.pos.line)
                if local is not None:
                    fc.emit(OP_STORE_LOCAL, local.slot, stmt.pos.line)
                else:
                    fc.emit(OP_POP, 0, stmt.pos.line)
            else:
                if local is not None:
                    fc.emit(OP_STORE_LOCAL, local.slot, stmt.pos.line)
                else:
                    fc.emit(OP_POP, 0, stmt.pos.line)
                # Discard the extra index/key pushed by FOR_ITER
                fc.emit(OP_POP, 0, stmt.pos.line)
        self._compile_block(stmt.body, fc)
        fc.emit_jump_back(loop_start, stmt.pos.line)
        fc.patch_jump(exit_jump)
        for bp in loop_ctx.break_patches:
            fc.patch_jump(bp)
        # Pop collection and index
        fc.emit(OP_POP, 0, stmt.pos.line)
        fc.emit(OP_POP, 0, stmt.pos.line)
        fc.loop_stack.pop()

    def _compile_break(self, stmt: TBreakStmt, fc: _FnCompiler) -> None:
        if not fc.loop_stack:
            return
        ctx = fc.loop_stack[-1]
        # Pop match scrutinee/dup values pushed inside the loop
        match_diff = fc.match_depth - ctx.match_depth
        i = 0
        while i < match_diff:
            fc.emit(OP_POP, 0, stmt.pos.line)
            i += 1
        # Pop handlers pushed inside the loop
        depth_diff = fc.handler_depth - ctx.handler_depth
        i = 0
        while i < depth_diff:
            fc.emit(OP_POP_HANDLER, 0, stmt.pos.line)
            i += 1
        bp = fc.emit_jump(OP_JUMP, stmt.pos.line)
        ctx.break_patches.append(bp)

    def _compile_continue(self, stmt: TContinueStmt, fc: _FnCompiler) -> None:
        if not fc.loop_stack:
            return
        ctx = fc.loop_stack[-1]
        # Pop match scrutinee/dup values pushed inside the loop
        match_diff = fc.match_depth - ctx.match_depth
        i = 0
        while i < match_diff:
            fc.emit(OP_POP, 0, stmt.pos.line)
            i += 1
        # Pop handlers pushed inside the loop
        depth_diff = fc.handler_depth - ctx.handler_depth
        i = 0
        while i < depth_diff:
            fc.emit(OP_POP_HANDLER, 0, stmt.pos.line)
            i += 1
        fc.emit_jump_back(ctx.continue_target, stmt.pos.line)

    def _compile_try(self, stmt: TTryStmt, fc: _FnCompiler) -> None:
        has_finally = stmt.finally_body is not None
        end_patches: list[int] = []
        if has_finally:
            finally_jump = fc.emit_jump(OP_PUSH_FINALLY, stmt.pos.line)
            fc.handler_depth += 1
        if stmt.catches:
            catch_jump = fc.emit_jump(OP_PUSH_HANDLER, stmt.pos.line)
            fc.handler_depth += 1
            self._compile_block(stmt.body, fc)
            fc.emit(OP_POP_HANDLER, 0, stmt.pos.line)
            fc.handler_depth -= 1
            body_end = fc.emit_jump(OP_JUMP, stmt.pos.line)
            end_patches.append(body_end)
            fc.patch_jump(catch_jump)
            # Compile catch clauses
            for ci, catch in enumerate(stmt.catches):
                local = fc.scope.lookup(catch.name)
                # If not last catch, check type and skip if no match
                if ci < len(stmt.catches) - 1 or catch.types:
                    fc.emit(OP_DUP, 0, catch.pos.line)
                    # Check type match
                    self._emit_catch_type_check(catch.types, fc, catch.pos.line)
                    skip_jump = fc.emit_jump(OP_JUMP_IF_FALSE, catch.pos.line)
                    if local is not None:
                        fc.emit(OP_STORE_LOCAL, local.slot, catch.pos.line)
                    else:
                        fc.emit(OP_POP, 0, catch.pos.line)
                    self._compile_block(catch.body, fc)
                    catch_end = fc.emit_jump(OP_JUMP, catch.pos.line)
                    end_patches.append(catch_end)
                    fc.patch_jump(skip_jump)
                else:
                    # Last catch with no type filter — catch all
                    if local is not None:
                        fc.emit(OP_STORE_LOCAL, local.slot, catch.pos.line)
                    else:
                        fc.emit(OP_POP, 0, catch.pos.line)
                    self._compile_block(catch.body, fc)
                    catch_end = fc.emit_jump(OP_JUMP, catch.pos.line)
                    end_patches.append(catch_end)
            # If no catch matched, rethrow
            fc.emit(OP_THROW, 0, stmt.pos.line)
        else:
            self._compile_block(stmt.body, fc)
        for ep in end_patches:
            fc.patch_jump(ep)
        if stmt.finally_body is not None:
            fc.emit(OP_POP_HANDLER, 0, stmt.pos.line)
            fc.handler_depth -= 1
            # Normal path: run finally then continue
            self._compile_block(stmt.finally_body, fc)
            finally_done = fc.emit_jump(OP_JUMP, stmt.pos.line)
            # Exception path: run finally then rethrow pending exception
            fc.patch_jump(finally_jump)
            self._compile_block(stmt.finally_body, fc)
            fc.emit(OP_NIL, 0, stmt.pos.line)
            fc.emit(OP_THROW, 0, stmt.pos.line)
            fc.patch_jump(finally_done)

    def _emit_catch_type_check(
        self, types: list[TType], fc: _FnCompiler, line: int
    ) -> None:
        if not types:
            fc.emit(OP_POP, 0, line)
            fc.emit(OP_TRUE, 0, line)
            return
        if len(types) == 1:
            tname = self._type_name_str(types[0])
            idx = len(fc.constants)
            fc.constants.append(VStr(tname))
            fc.emit(OP_IS_TYPE, idx, line)
            return
        # Union: for each type except last, DUP + IS_TYPE + short-circuit on true.
        # Last type: just IS_TYPE (consumes the value).
        # Stack: [value]
        # DUP → [value, copy]
        # IS_TYPE → [value, bool]  (IS_TYPE popped copy, pushed bool)
        # If true → pop value, push true, jump to end
        # If false → pop false, try next type
        end_patches: list[int] = []
        for typ_item in types[:-1]:
            tname = self._type_name_str(typ_item)
            idx = len(fc.constants)
            fc.constants.append(VStr(tname))
            fc.emit(OP_DUP, 0, line)
            fc.emit(OP_IS_TYPE, idx, line)
            match_jump = fc.emit_jump(OP_JUMP_IF_TRUE, line)
            # Not matched — continue to next type (value still on stack)
            # Actually JUMP_IF_TRUE already popped the bool.
            # On false path: stack is [value], continue.
            # On true path: we jumped to match_target with stack [value].
            # We need true on stack and value gone. So jump to success block.
            # Let's defer: collect the match jumps, patch them all to one place.
            end_patches.append(match_jump)
        # Last type: IS_TYPE consumes the value
        tname = self._type_name_str(types[-1])
        idx = len(fc.constants)
        fc.constants.append(VStr(tname))
        fc.emit(OP_IS_TYPE, idx, line)
        skip_pop = fc.emit_jump(OP_JUMP, line)
        # Patch all short-circuit jumps here: pop value, push true
        for ep in end_patches:
            fc.patch_jump(ep)
        fc.emit(OP_POP, 0, line)  # pop original value
        fc.emit(OP_TRUE, 0, line)
        fc.patch_jump(skip_pop)

    def _type_name_str(self, ttype: TType) -> str:
        if isinstance(ttype, TIdentType):
            return ttype.name
        if isinstance(ttype, TPrimitive):
            return ttype.kind
        return ""

    def _compile_match(self, stmt: TMatchStmt, fc: _FnCompiler) -> None:
        self._compile_expr(stmt.expr, fc)
        end_patches: list[int] = []
        for case in stmt.cases:
            fc.emit(OP_DUP, 0, case.pos.line)
            if isinstance(case.pattern, TPatternType):
                tname = self._type_name_str(case.pattern.type_name)
                idx = len(fc.constants)
                fc.constants.append(VStr(tname))
                fc.emit(OP_MATCH_TYPE, idx, case.pos.line)
                skip = fc.emit_jump(OP_JUMP_IF_FALSE, case.pos.line)
                local = fc.scope.lookup(case.pattern.name)
                if local is not None:
                    fc.emit(OP_DUP, 0, case.pos.line)
                    fc.emit(OP_STORE_LOCAL, local.slot, case.pos.line)
                fc.match_depth += 2
                self._compile_block(case.body, fc)
                fc.match_depth -= 2
                fc.emit(OP_POP, 0, case.pos.line)
                end_patches.append(fc.emit_jump(OP_JUMP, case.pos.line))
                fc.patch_jump(skip)
                fc.emit(OP_POP, 0, case.pos.line)
            elif isinstance(case.pattern, TPatternEnum):
                idx = len(fc.constants)
                fc.constants.append(
                    VStr(case.pattern.enum_name + "." + case.pattern.variant)
                )
                fc.emit(OP_MATCH_TYPE, idx, case.pos.line)
                skip = fc.emit_jump(OP_JUMP_IF_FALSE, case.pos.line)
                fc.match_depth += 2
                self._compile_block(case.body, fc)
                fc.match_depth -= 2
                fc.emit(OP_POP, 0, case.pos.line)
                end_patches.append(fc.emit_jump(OP_JUMP, case.pos.line))
                fc.patch_jump(skip)
                fc.emit(OP_POP, 0, case.pos.line)
            elif isinstance(case.pattern, TPatternNil):
                fc.emit(OP_NIL, 0, case.pos.line)
                fc.emit(OP_EQ, 0, case.pos.line)
                skip = fc.emit_jump(OP_JUMP_IF_FALSE, case.pos.line)
                fc.match_depth += 1
                self._compile_block(case.body, fc)
                fc.match_depth -= 1
                end_patches.append(fc.emit_jump(OP_JUMP, case.pos.line))
                fc.patch_jump(skip)
        if stmt.default is not None:
            if stmt.default.name is not None:
                local = fc.scope.lookup(stmt.default.name)
                if local is not None:
                    fc.emit(OP_DUP, 0, stmt.default.pos.line)
                    fc.emit(OP_STORE_LOCAL, local.slot, stmt.default.pos.line)
            fc.match_depth += 1  # scrutinee on stack
            self._compile_block(stmt.default.body, fc)
            fc.match_depth -= 1
        for ep in end_patches:
            fc.patch_jump(ep)
        fc.emit(OP_POP, 0, stmt.pos.line)  # pop scrutinee

    # ── Expression compilation ────────────────────────────────

    def _compile_expr(self, expr: TExpr, fc: _FnCompiler) -> None:
        match expr:
            case TIntLit():
                if expr.value == 0:
                    fc.emit(OP_INT_ZERO, 0, expr.pos.line)
                elif expr.value == 1:
                    fc.emit(OP_INT_ONE, 0, expr.pos.line)
                else:
                    fc.emit_const(VInt(expr.value), expr.pos.line)
            case TFloatLit():
                fc.emit_const(VFloat(expr.value), expr.pos.line)
            case TBoolLit():
                if expr.value:
                    fc.emit(OP_TRUE, 0, expr.pos.line)
                else:
                    fc.emit(OP_FALSE, 0, expr.pos.line)
            case TNilLit():
                fc.emit(OP_NIL, 0, expr.pos.line)
            case TStringLit():
                fc.emit_const(VStr(expr.value), expr.pos.line)
            case TByteLit():
                fc.emit_const(VByte(expr.value), expr.pos.line)
            case TRuneLit():
                fc.emit_const(VRune(expr.value), expr.pos.line)
            case TBytesLit():
                fc.emit_const(VBytes(expr.value), expr.pos.line)
            case TVar():
                self._compile_var(expr, fc)
            case TBinaryOp():
                self._compile_binop(expr, fc)
            case TUnaryOp():
                self._compile_unaryop(expr, fc)
            case TCall():
                self._compile_call(expr, fc)
            case TTernary():
                self._compile_ternary(expr, fc)
            case TListLit():
                for e in expr.elements:
                    self._compile_expr(e, fc)
                fc.emit(OP_BUILD_LIST, len(expr.elements), expr.pos.line)
            case TMapLit():
                for k, v in expr.entries:
                    self._compile_expr(k, fc)
                    self._compile_expr(v, fc)
                fc.emit(OP_BUILD_MAP, len(expr.entries), expr.pos.line)
            case TSetLit():
                for e in expr.elements:
                    self._compile_expr(e, fc)
                fc.emit(OP_BUILD_SET, len(expr.elements), expr.pos.line)
            case TTupleLit():
                for e in expr.elements:
                    self._compile_expr(e, fc)
                fc.emit(OP_BUILD_TUPLE, len(expr.elements), expr.pos.line)
            case TIndex():
                self._compile_expr(expr.obj, fc)
                self._compile_expr(expr.index, fc)
                fc.emit(OP_INDEX, 0, expr.pos.line)
            case TSlice():
                self._compile_expr(expr.obj, fc)
                self._compile_expr(expr.low, fc)
                self._compile_expr(expr.high, fc)
                fc.emit(OP_SLICE, 0, expr.pos.line)
            case TFieldAccess():
                self._compile_field_access(expr, fc)
            case TTupleAccess():
                self._compile_expr(expr.obj, fc)
                fc.emit(OP_TUPLE_ACCESS, expr.index, expr.pos.line)
            case TFnLit():
                self._compile_fn_lit(expr, fc)
            case TRange():
                for a in expr.args:
                    self._compile_expr(a, fc)
                fc.emit(OP_GET_ITER, len(expr.args), expr.pos.line)
            case _:
                assert_never(expr)

    def _compile_var(self, expr: TVar, fc: _FnCompiler) -> None:
        local = fc.scope.lookup(expr.name)
        if local is not None:
            fc.emit(OP_LOAD_LOCAL, local.slot, expr.pos.line)
            return
        # Check if it's a global function
        if expr.name in self._global_index:
            gidx = self._global_index[expr.name]
            fc.emit(OP_LOAD_GLOBAL, gidx, expr.pos.line)
            return
        # Check if it's a builtin
        if expr.name in _BUILTIN_INDEX:
            bidx = _BUILTIN_INDEX[expr.name]
            fc.emit(OP_LOAD_BUILTIN, bidx, expr.pos.line)
            return
        # Check if it's a struct constructor or enum
        if expr.name in self._struct_index:
            # Struct name as value — used for constructors
            fc.emit_const(VStr(expr.name), expr.pos.line)
            return
        if expr.name in self.checker_types:
            ct = self.checker_types[expr.name]
            if isinstance(ct, EnumT):
                fc.emit_const(VStr(expr.name), expr.pos.line)
                return

    def _compile_binop(self, expr: TBinaryOp, fc: _FnCompiler) -> None:
        op = expr.op
        # Short-circuit operators
        if op == "&&":
            self._compile_expr(expr.left, fc)
            short = fc.emit_jump(OP_JUMP_IF_FALSE, expr.pos.line)
            self._compile_expr(expr.right, fc)
            end = fc.emit_jump(OP_JUMP, expr.pos.line)
            fc.patch_jump(short)
            fc.emit(OP_FALSE, 0, expr.pos.line)
            fc.patch_jump(end)
            return
        if op == "||":
            self._compile_expr(expr.left, fc)
            short = fc.emit_jump(OP_JUMP_IF_TRUE, expr.pos.line)
            self._compile_expr(expr.right, fc)
            end = fc.emit_jump(OP_JUMP, expr.pos.line)
            fc.patch_jump(short)
            fc.emit(OP_TRUE, 0, expr.pos.line)
            fc.patch_jump(end)
            return
        self._compile_expr(expr.left, fc)
        self._compile_expr(expr.right, fc)
        left_type = self._resolve_expr_type(expr.left, fc)
        if op == "==":
            fc.emit(OP_EQ, 0, expr.pos.line)
        elif op == "!=":
            fc.emit(OP_NE, 0, expr.pos.line)
        elif op == "+":
            self._emit_add(left_type, fc, expr.pos.line)
        elif op == "-":
            self._emit_sub(left_type, fc, expr.pos.line)
        elif op == "*":
            self._emit_mul(left_type, fc, expr.pos.line)
        elif op == "/":
            self._emit_div(left_type, fc, expr.pos.line)
        elif op == "%":
            self._emit_mod(left_type, fc, expr.pos.line)
        elif op == "<":
            self._emit_cmp(left_type, CMP_LT, fc, expr.pos.line)
        elif op == "<=":
            self._emit_cmp(left_type, CMP_LE, fc, expr.pos.line)
        elif op == ">":
            self._emit_cmp(left_type, CMP_GT, fc, expr.pos.line)
        elif op == ">=":
            self._emit_cmp(left_type, CMP_GE, fc, expr.pos.line)
        elif op == "&":
            fc.emit(OP_BIT_AND, 0, expr.pos.line)
        elif op == "|":
            fc.emit(OP_BIT_OR, 0, expr.pos.line)
        elif op == "^":
            fc.emit(OP_BIT_XOR, 0, expr.pos.line)
        elif op == "<<":
            fc.emit(OP_SHIFT_LEFT, 0, expr.pos.line)
        elif op == ">>":
            fc.emit(OP_SHIFT_RIGHT, 0, expr.pos.line)
        elif op == ">>>":
            fc.emit(OP_SHIFT_RIGHT_UNSIGNED, 0, expr.pos.line)

    def _emit_binop_for_type(
        self, op: str, typ: Type, fc: _FnCompiler, line: int
    ) -> None:
        """Emit the binary operator for a compound assignment."""
        if op == "+=":
            self._emit_add(typ, fc, line)
        elif op == "-=":
            self._emit_sub(typ, fc, line)
        elif op == "*=":
            self._emit_mul(typ, fc, line)
        elif op == "/=":
            self._emit_div(typ, fc, line)
        elif op == "%=":
            self._emit_mod(typ, fc, line)
        elif op == "&=":
            fc.emit(OP_BIT_AND, 0, line)
        elif op == "|=":
            fc.emit(OP_BIT_OR, 0, line)
        elif op == "^=":
            fc.emit(OP_BIT_XOR, 0, line)
        elif op == "<<=":
            fc.emit(OP_SHIFT_LEFT, 0, line)
        elif op == ">>=":
            fc.emit(OP_SHIFT_RIGHT, 0, line)

    def _emit_add(self, typ: Type, fc: _FnCompiler, line: int) -> None:
        if type_eq(typ, INT_T):
            fc.emit(OP_ADD_INT, 0, line)
        elif type_eq(typ, FLOAT_T):
            fc.emit(OP_ADD_FLOAT, 0, line)
        elif type_eq(typ, STRING_T):
            fc.emit(OP_ADD_STRING, 0, line)
        elif type_eq(typ, BYTE_T):
            fc.emit(OP_ADD_BYTE, 0, line)
        else:
            fc.emit(OP_ADD_INT, 0, line)

    def _emit_sub(self, typ: Type, fc: _FnCompiler, line: int) -> None:
        if type_eq(typ, FLOAT_T):
            fc.emit(OP_SUB_FLOAT, 0, line)
        elif type_eq(typ, BYTE_T):
            fc.emit(OP_SUB_BYTE, 0, line)
        else:
            fc.emit(OP_SUB_INT, 0, line)

    def _emit_mul(self, typ: Type, fc: _FnCompiler, line: int) -> None:
        if type_eq(typ, FLOAT_T):
            fc.emit(OP_MUL_FLOAT, 0, line)
        elif type_eq(typ, BYTE_T):
            fc.emit(OP_MUL_BYTE, 0, line)
        else:
            fc.emit(OP_MUL_INT, 0, line)

    def _emit_div(self, typ: Type, fc: _FnCompiler, line: int) -> None:
        if type_eq(typ, FLOAT_T):
            fc.emit(OP_DIV_FLOAT, 0, line)
        else:
            fc.emit(OP_DIV_INT, 0, line)

    def _emit_mod(self, typ: Type, fc: _FnCompiler, line: int) -> None:
        if type_eq(typ, FLOAT_T):
            fc.emit(OP_MOD_FLOAT, 0, line)
        else:
            fc.emit(OP_MOD_INT, 0, line)

    def _emit_cmp(self, typ: Type, cmp_kind: int, fc: _FnCompiler, line: int) -> None:
        if type_eq(typ, INT_T):
            fc.emit(OP_CMP_INT, cmp_kind, line)
        elif type_eq(typ, FLOAT_T):
            fc.emit(OP_CMP_FLOAT, cmp_kind, line)
        elif type_eq(typ, STRING_T):
            fc.emit(OP_CMP_STRING, cmp_kind, line)
        elif type_eq(typ, BYTE_T):
            fc.emit(OP_CMP_BYTE, cmp_kind, line)
        elif type_eq(typ, RUNE_T):
            fc.emit(OP_CMP_RUNE, cmp_kind, line)
        else:
            fc.emit(OP_CMP_INT, cmp_kind, line)

    def _compile_unaryop(self, expr: TUnaryOp, fc: _FnCompiler) -> None:
        self._compile_expr(expr.operand, fc)
        if expr.op == "-":
            typ = self._resolve_expr_type(expr.operand, fc)
            if type_eq(typ, FLOAT_T):
                fc.emit(OP_NEG_FLOAT, 0, expr.pos.line)
            else:
                fc.emit(OP_NEG_INT, 0, expr.pos.line)
        elif expr.op == "!":
            fc.emit(OP_NOT, 0, expr.pos.line)
        elif expr.op == "~":
            fc.emit(OP_BIT_NOT, 0, expr.pos.line)

    def _compile_ternary(self, expr: TTernary, fc: _FnCompiler) -> None:
        self._compile_expr(expr.cond, fc)
        false_jump = fc.emit_jump(OP_JUMP_IF_FALSE, expr.pos.line)
        self._compile_expr(expr.then_expr, fc)
        end_jump = fc.emit_jump(OP_JUMP, expr.pos.line)
        fc.patch_jump(false_jump)
        self._compile_expr(expr.else_expr, fc)
        fc.patch_jump(end_jump)

    def _compile_call(self, expr: TCall, fc: _FnCompiler) -> None:
        # Check for struct constructor
        if isinstance(expr.func, TVar):
            name = expr.func.name
            # Struct constructor
            if name in self._struct_index:
                self._compile_struct_constructor(name, expr, fc)
                return
            # Builtin call
            if name in _BUILTIN_INDEX and name not in self.checker_functions:
                self._compile_builtin_call(name, expr, fc)
                return
            # Error struct constructor
            if name in self.checker_types:
                ct = self.checker_types[name]
                if isinstance(ct, StructT):
                    self._compile_error_struct_constructor(name, expr, fc)
                    return
        # Method call
        if isinstance(expr.func, TFieldAccess):
            self._compile_method_call(expr.func, expr, fc)
            return
        # Regular function call
        self._compile_expr(expr.func, fc)
        for a in expr.args:
            self._compile_expr(a.value, fc)
        fc.emit(OP_CALL, len(expr.args), expr.pos.line)

    def _compile_builtin_call(self, name: str, expr: TCall, fc: _FnCompiler) -> None:
        bidx = _BUILTIN_INDEX[name]
        for a in expr.args:
            self._compile_expr(a.value, fc)
        fc.emit(OP_CALL_BUILTIN, bidx, expr.pos.line)
        # Emit argc as a trailing pair so the VM knows how many args to pop
        fc.emit(0, len(expr.args), expr.pos.line)

    def _compile_struct_constructor(
        self, name: str, expr: TCall, fc: _FnCompiler
    ) -> None:
        sidx = self._struct_index[name]
        sd = self.struct_defs[sidx]
        # Build args in field order
        # Support both positional and named args
        has_named = len(expr.args) > 0 and expr.args[0].name is not None
        if has_named:
            # Named args: emit in field order, use defaults for missing
            arg_by_name: dict[str, TExpr] = {}
            for a in expr.args:
                if a.name is not None:
                    arg_by_name[a.name] = a.value
            for fidx, fname in enumerate(sd.field_names):
                if fname in arg_by_name:
                    self._compile_expr(arg_by_name[fname], fc)
                else:
                    self._emit_zero_value(sd.field_types[fidx], fc, expr.pos.line)
        else:
            for a in expr.args:
                self._compile_expr(a.value, fc)
            # Fill remaining with zero values
            for ft in sd.field_types[len(expr.args) :]:
                self._emit_zero_value(ft, fc, expr.pos.line)
        fc.emit(OP_BUILD_STRUCT, sidx, expr.pos.line)

    def _compile_error_struct_constructor(
        self, name: str, expr: TCall, fc: _FnCompiler
    ) -> None:
        """Compile construction of built-in error structs (ValueError, etc.)."""
        # Error structs have a single 'message' field
        if expr.args:
            self._compile_expr(expr.args[0].value, fc)
        else:
            fc.emit_const(VStr(""), expr.pos.line)
        # Encode as a special struct build
        idx = len(fc.constants)
        fc.constants.append(VStr(name))
        fc.emit(OP_BUILD_STRUCT, idx, expr.pos.line)

    def _compile_method_call(
        self, fa: TFieldAccess, expr: TCall, fc: _FnCompiler
    ) -> None:
        self._compile_expr(fa.obj, fc)
        for a in expr.args:
            self._compile_expr(a.value, fc)
        fidx = self._field_const(fa.field, fc, expr.pos.line)
        fc.emit(OP_CALL_METHOD, fidx, expr.pos.line)
        fc.emit(0, len(expr.args), expr.pos.line)

    def _compile_field_access(self, expr: TFieldAccess, fc: _FnCompiler) -> None:
        # Enum variant access: EnumName.Variant
        if isinstance(expr.obj, TVar):
            ct = self.checker_types.get(expr.obj.name)
            if ct is not None and isinstance(ct, EnumT):
                idx = len(fc.constants)
                fc.constants.append(VStr(expr.obj.name))
                fc.constants.append(VStr(expr.field))
                fc.emit(OP_LOAD_ENUM, idx, expr.pos.line)
                return
        self._compile_expr(expr.obj, fc)
        fidx = self._field_const(expr.field, fc, expr.pos.line)
        fc.emit(OP_GET_FIELD, fidx, expr.pos.line)

    def _compile_fn_lit(self, expr: TFnLit, fc: _FnCompiler) -> None:
        # Compile the function literal as a separate CodeObject
        lit_fc = _FnCompiler(self, "<lambda>")
        for p in expr.params:
            pt = self._resolve_param_type(p)
            lit_fc.add_local(p.name, pt)
        is_arrow = expr.annotations.get("fn_lit.arrow") == "true"
        first_stmt = expr.body[0] if len(expr.body) == 1 else None
        if is_arrow and isinstance(first_stmt, TExprStmt):
            # Arrow fn lit: body is a single expression, compile as return
            self._compile_expr(first_stmt.expr, lit_fc)
            lit_fc.emit(OP_RETURN, 0, expr.pos.line)
        else:
            self._collect_locals(expr.body, lit_fc)
            self._emit_default_preamble(expr.params, lit_fc, expr.pos.line)
            self._compile_block(expr.body, lit_fc)
            lit_fc.emit(OP_RETURN_VOID, 0, expr.pos.line)
        code_idx = len(self.code_objects)
        self.code_objects.append(lit_fc.to_code_object(len(expr.params)))
        fc.emit_const(VFunc(code_idx), expr.pos.line)

    def _field_const(self, name: str, fc: _FnCompiler, line: int) -> int:
        """Add a field name to the constant pool and return its index."""
        idx = len(fc.constants)
        fc.constants.append(VStr(name))
        return idx

    # ── Type resolution ───────────────────────────────────────

    def _resolve_expr_type(self, expr: TExpr, fc: _FnCompiler) -> Type:
        """Resolve the type of an expression. Since the checker passed,
        we just need to figure out what type something is."""
        if isinstance(expr, TIntLit):
            return INT_T
        if isinstance(expr, TFloatLit):
            return FLOAT_T
        if isinstance(expr, TBoolLit):
            return BOOL_T
        if isinstance(expr, TNilLit):
            return NIL_T
        if isinstance(expr, TStringLit):
            return STRING_T
        if isinstance(expr, TByteLit):
            return BYTE_T
        if isinstance(expr, TRuneLit):
            return RUNE_T
        if isinstance(expr, TBytesLit):
            return BYTES_T
        if isinstance(expr, TVar):
            local = fc.scope.lookup(expr.name)
            if local is not None:
                return local.typ
            fn = self.checker_functions.get(expr.name)
            if fn is not None:
                return fn
            glt = self.global_let_types.get(expr.name)
            if glt is not None:
                return glt
            return VOID_T
        if isinstance(expr, TBinaryOp):
            op = expr.op
            if op in ("==", "!=", "<", "<=", ">", ">=", "&&", "||"):
                return BOOL_T
            return self._resolve_expr_type(expr.left, fc)
        if isinstance(expr, TUnaryOp):
            if expr.op == "!":
                return BOOL_T
            return self._resolve_expr_type(expr.operand, fc)
        if isinstance(expr, TTernary):
            return self._resolve_expr_type(expr.then_expr, fc)
        if isinstance(expr, TCall):
            return self._resolve_call_type(expr, fc)
        if isinstance(expr, TIndex):
            obj_type = self._resolve_expr_type(expr.obj, fc)
            if isinstance(obj_type, ListT):
                return obj_type.element
            if isinstance(obj_type, MapT):
                return obj_type.value
            if type_eq(obj_type, STRING_T):
                return RUNE_T
            if type_eq(obj_type, BYTES_T):
                return BYTE_T
            return VOID_T
        if isinstance(expr, TSlice):
            return self._resolve_expr_type(expr.obj, fc)
        if isinstance(expr, TFieldAccess):
            return self._resolve_field_type(expr, fc)
        if isinstance(expr, TTupleAccess):
            obj_type = self._resolve_expr_type(expr.obj, fc)
            if isinstance(obj_type, TupleT) and expr.index < len(obj_type.elements):
                return obj_type.elements[expr.index]
            return VOID_T
        if isinstance(expr, TListLit):
            if expr.elements:
                return ListT(
                    kind="list", element=self._resolve_expr_type(expr.elements[0], fc)
                )
            return ListT(kind="list", element=VOID_T)
        if isinstance(expr, TMapLit):
            if expr.entries:
                k, v = expr.entries[0]
                return MapT(
                    kind="map",
                    key=self._resolve_expr_type(k, fc),
                    value=self._resolve_expr_type(v, fc),
                )
            return MapT(kind="map", key=VOID_T, value=VOID_T)
        if isinstance(expr, TSetLit):
            if expr.elements:
                return SetT(
                    kind="set", element=self._resolve_expr_type(expr.elements[0], fc)
                )
            return SetT(kind="set", element=VOID_T)
        if isinstance(expr, TTupleLit):
            elts: list[Type] = []
            for e in expr.elements:
                elts.append(self._resolve_expr_type(e, fc))
            return TupleT(kind="tuple", elements=elts)
        return VOID_T

    def _resolve_call_type(self, expr: TCall, fc: _FnCompiler) -> Type:
        if isinstance(expr.func, TVar):
            name = expr.func.name
            fn = self.checker_functions.get(name)
            if fn is not None:
                return fn.ret
            # Struct constructor
            if name in self._struct_index:
                ct = self.checker_types.get(name)
                if ct is not None:
                    return ct
            if name in self.checker_types:
                return self.checker_types[name]
            # Builtin return types — approximate
            if name in ("ToString", "ToRepr"):
                return STRING_T
            if name == "Len":
                return INT_T
            if name in ("IntToFloat", "Sqrt"):
                return FLOAT_T
            if name in (
                "FloatToInt",
                "Round",
                "Floor",
                "Ceil",
                "ByteToInt",
                "RuneToInt",
                "ParseInt",
            ):
                return INT_T
            if name == "ParseFloat":
                return FLOAT_T
            if name == "IntToByte":
                return BYTE_T
            if name == "RuneFromInt":
                return RUNE_T
            if name in (
                "IsNaN",
                "IsInf",
                "IsNil",
                "IsType",
                "Contains",
                "StartsWith",
                "EndsWith",
                "IsDigit",
                "IsAlpha",
                "IsAlnum",
                "IsSpace",
                "IsUpper",
                "IsLower",
            ):
                return BOOL_T
            if name in (
                "Upper",
                "Lower",
                "Trim",
                "TrimStart",
                "TrimEnd",
                "Join",
                "Replace",
                "ReplaceCount",
                "Repeat",
                "Reverse",
                "Concat",
                "Format",
                "FormatInt",
            ):
                return STRING_T
            if name in ("Abs", "Min", "Max", "Sum", "Pow"):
                # Return type matches first arg
                if expr.args:
                    return self._resolve_expr_type(expr.args[0].value, fc)
                return INT_T
            if name in ("Find", "RFind", "Count", "IndexOf"):
                return INT_T
            if name == "DivMod":
                return TupleT(kind="tuple", elements=[INT_T, INT_T])
            if name in ("Split", "SplitN", "SplitWhitespace", "Chars"):
                return ListT(kind="list", element=STRING_T)
            if name == "Assert":
                return VOID_T
            if name in ("WritelnOut", "WritelnErr", "WriteOut", "WriteErr"):
                return VOID_T
            if name in (
                "Append",
                "Insert",
                "Pop",
                "RemoveAt",
                "Delete",
                "Add",
                "Remove",
            ):
                return VOID_T
            if name == "Get":
                # Map.Get returns value type or nil
                return VOID_T
            if name in ("Keys", "Values", "Items", "Sorted", "Reversed"):
                return VOID_T
            if name == "Exit":
                return VOID_T
        if isinstance(expr.func, TFieldAccess):
            # Method call — resolve through struct type
            obj_type = self._resolve_expr_type(expr.func.obj, fc)
            if isinstance(obj_type, StructT):
                mt = obj_type.methods.get(expr.func.field)
                if mt is not None:
                    return mt.ret
            return VOID_T
        # Function value call
        ft = self._resolve_expr_type(expr.func, fc)
        if isinstance(ft, FnT):
            return ft.ret
        return VOID_T

    def _resolve_field_type(self, expr: TFieldAccess, fc: _FnCompiler) -> Type:
        obj_type = self._resolve_expr_type(expr.obj, fc)
        if isinstance(obj_type, StructT):
            ft = obj_type.fields.get(expr.field)
            if ft is not None:
                return ft
        return VOID_T


def compile_module(module: TModule) -> CompiledModule:
    """Compile a TModule to a CompiledModule."""
    compiler = Compiler()
    return compiler.compile_module(module)
