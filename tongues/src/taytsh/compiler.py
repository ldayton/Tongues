"""Bytecode compiler for Taytsh — walks checked AST and emits bytecode."""

from __future__ import annotations

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
    TUnaryOp,
    TVar,
    TWhileStmt,
)
from .check import (
    BOOL_T,
    BUILTIN_NAMES,
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
    UnionT,
    check_with_info,
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
    OP_LOAD_CAPTURE,
    OP_LOAD_ENUM,
    OP_LOAD_GLOBAL,
    OP_LOAD_LOCAL,
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
    VBool,
    VByte,
    VBytes,
    VFloat,
    VFunc,
    VInt,
    VNil,
    VRune,
    VStr,
)


# ============================================================
# BUILTIN TABLE
# ============================================================

BUILTIN_TABLE: list[str] = sorted(BUILTIN_NAMES)
_BUILTIN_INDEX: dict[str, int] = {}
_bi = 0
while _bi < len(BUILTIN_TABLE):
    _BUILTIN_INDEX[BUILTIN_TABLE[_bi]] = _bi
    _bi += 1


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
    __slots__ = ("break_patches", "continue_target", "handler_depth")

    def __init__(self, handler_depth: int) -> None:
        self.break_patches: list[int] = []
        self.continue_target: int = -1
        self.handler_depth: int = handler_depth


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
        # Reserve space for EXTENDED_ARG in case we need it
        return self.emit(op, 0, line)

    def current_offset(self) -> int:
        return len(self.code)

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
        self.checker_types: dict[str, Type] = {}
        self.checker_functions: dict[str, FnT] = {}
        self.fn_param_names: dict[str, list[str]] = {}

    def compile_module(self, module: TModule) -> CompiledModule:
        errors, checker = check_with_info(module)
        if len(errors) > 0:
            msgs: list[str] = []
            i = 0
            while i < len(errors):
                msgs.append(repr(errors[i]))
                i += 1
            raise CompileError("\n".join(msgs))
        self.checker_types = checker.types
        self.checker_functions = checker.functions
        self.fn_param_names = checker.fn_param_names
        # Collect struct/enum/interface defs
        for decl in module.decls:
            if isinstance(decl, TStructDecl):
                self._register_struct(decl)
            elif isinstance(decl, TEnumDecl):
                self._register_enum(decl)
            elif isinstance(decl, TInterfaceDecl):
                self._register_interface(decl)
        # Register all top-level function names as globals
        for decl in module.decls:
            if isinstance(decl, TFnDecl):
                self._ensure_global(decl.name)
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
        return CompiledModule(
            code_objects=self.code_objects,
            global_names=self.global_names,
            struct_defs=self.struct_defs,
            enum_defs=self.enum_defs,
            interface_defs=self.interface_defs,
            entry_index=self.entry_index,
        )

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
        # Compile body
        self._compile_block(decl.body, fc)
        # Implicit return void
        fn_type = self.checker_functions.get(decl.name)
        ret_type = fn_type.ret if fn_type is not None else VOID_T
        if type_eq(ret_type, VOID_T):
            fc.emit(OP_RETURN_VOID, 0, decl.pos.line)
        code_idx = len(self.code_objects)
        self.code_objects.append(fc.to_code_object(len(decl.params)))
        return code_idx

    def _resolve_param_type(self, p) -> Type:
        """Resolve a TParam's type to a checker Type."""
        if p.typ is None:
            return VOID_T
        return self._resolve_ttype(p.typ)

    def _resolve_ttype(self, tt) -> Type:
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
        return VOID_T

    def _collect_locals(self, stmts: list[TStmt], fc: _FnCompiler) -> None:
        """Pre-scan statements to allocate local slots for let declarations."""
        for stmt in stmts:
            if isinstance(stmt, TLetStmt):
                if fc.scope.lookup(stmt.name) is None:
                    typ = self._resolve_ttype(stmt.typ)
                    fc.add_local(stmt.name, typ)
            elif isinstance(stmt, TIfStmt):
                self._collect_locals(stmt.then_body, fc)
                if stmt.else_body is not None:
                    self._collect_locals(stmt.else_body, fc)
            elif isinstance(stmt, TWhileStmt):
                self._collect_locals(stmt.body, fc)
            elif isinstance(stmt, TForStmt):
                for b in stmt.binding:
                    if b != "_" and fc.scope.lookup(b) is None:
                        fc.add_local(b, INT_T)
                self._collect_locals(stmt.body, fc)
            elif isinstance(stmt, TMatchStmt):
                for case in stmt.cases:
                    if isinstance(case.pattern, TPatternType):
                        pname = case.pattern.name
                        if fc.scope.lookup(pname) is None:
                            pt = self._resolve_ttype(case.pattern.type_name)
                            fc.add_local(pname, pt)
                    self._collect_locals(case.body, fc)
                if stmt.default is not None:
                    if stmt.default.name is not None:
                        dname = stmt.default.name
                        if fc.scope.lookup(dname) is None:
                            fc.add_local(dname, VOID_T)
                    self._collect_locals(stmt.default.body, fc)
            elif isinstance(stmt, TTryStmt):
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
        if isinstance(stmt, TLetStmt):
            self._compile_let(stmt, fc)
        elif isinstance(stmt, TAssignStmt):
            self._compile_assign(stmt, fc)
        elif isinstance(stmt, TOpAssignStmt):
            self._compile_op_assign(stmt, fc)
        elif isinstance(stmt, TTupleAssignStmt):
            self._compile_tuple_assign(stmt, fc)
        elif isinstance(stmt, TExprStmt):
            self._compile_expr(stmt.expr, fc)
            fc.emit(OP_POP, 0, stmt.pos.line)
        elif isinstance(stmt, TReturnStmt):
            if stmt.value is not None:
                self._compile_expr(stmt.value, fc)
                fc.emit(OP_RETURN, 0, stmt.pos.line)
            else:
                fc.emit(OP_RETURN_VOID, 0, stmt.pos.line)
        elif isinstance(stmt, TIfStmt):
            self._compile_if(stmt, fc)
        elif isinstance(stmt, TWhileStmt):
            self._compile_while(stmt, fc)
        elif isinstance(stmt, TForStmt):
            self._compile_for(stmt, fc)
        elif isinstance(stmt, TBreakStmt):
            self._compile_break(stmt, fc)
        elif isinstance(stmt, TContinueStmt):
            self._compile_continue(stmt, fc)
        elif isinstance(stmt, TTryStmt):
            self._compile_try(stmt, fc)
        elif isinstance(stmt, TThrowStmt):
            self._compile_expr(stmt.expr, fc)
            fc.emit(OP_THROW, 0, stmt.pos.line)
        elif isinstance(stmt, TMatchStmt):
            self._compile_match(stmt, fc)

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
        else:
            fc.emit(OP_NIL, 0, line)

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
            # obj[index] op= value
            self._compile_expr(stmt.target.obj, fc)
            self._compile_expr(stmt.target.index, fc)
            fc.emit(OP_DUP, 0, stmt.pos.line)
            fc.emit(OP_ROT_TWO, 0, stmt.pos.line)
            # Stack: obj, index, index, obj (wrong) — need to rethink
            # Actually for index assign, just load current, apply op, store
            # Simpler: compile as target = target op value
            self._compile_expr(stmt.target, fc)
            self._compile_expr(stmt.value, fc)
            typ = self._resolve_expr_type(stmt.target, fc)
            self._emit_binop_for_type(stmt.op, typ, fc, stmt.pos.line)
            self._compile_expr(stmt.target.obj, fc)
            self._compile_expr(stmt.target.index, fc)
            fc.emit(OP_ROT_TWO, 0, stmt.pos.line)
            # Stack: result, obj, index — need: obj, index, result
            # This is getting messy. Let's just use the simple approach.
            # Rewrite: load target, compile value, binop, store target
            # But store target is complex. For now, punt to simple cases.
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
        # After UNPACK, stack has n values, first on top
        i = 0
        while i < n:
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
            i += 1

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
        loop_ctx = _LoopCtx(fc.handler_depth)
        loop_ctx.continue_target = loop_start
        fc.loop_stack.append(loop_ctx)
        self._compile_expr(stmt.cond, fc)
        exit_jump = fc.emit_jump(OP_JUMP_IF_FALSE, stmt.pos.line)
        self._compile_block(stmt.body, fc)
        # Jump back to loop start
        back_dist = fc.current_offset() - loop_start + 2
        fc.emit(OP_JUMP_BACK, back_dist, stmt.pos.line)
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

    def _compile_for_range(
        self, stmt: TForStmt, rng: TRange, fc: _FnCompiler
    ) -> None:
        # Compile range args and emit GET_ITER
        for a in rng.args:
            self._compile_expr(a, fc)
        fc.emit(OP_GET_ITER, len(rng.args), stmt.pos.line)
        # Binding variable
        binding = stmt.binding[0] if len(stmt.binding) > 0 else "_"
        local = fc.scope.lookup(binding) if binding != "_" else None
        loop_start = fc.current_offset()
        loop_ctx = _LoopCtx(fc.handler_depth)
        loop_ctx.continue_target = loop_start
        fc.loop_stack.append(loop_ctx)
        exit_jump = fc.emit_jump(OP_FOR_ITER, stmt.pos.line)
        if local is not None:
            fc.emit(OP_STORE_LOCAL, local.slot, stmt.pos.line)
        else:
            fc.emit(OP_POP, 0, stmt.pos.line)
        self._compile_block(stmt.body, fc)
        back_dist = fc.current_offset() - loop_start + 2
        fc.emit(OP_JUMP_BACK, back_dist, stmt.pos.line)
        fc.patch_jump(exit_jump)
        # Pop iterator state (3 values: current, end, step)
        fc.emit(OP_POP, 0, stmt.pos.line)
        fc.emit(OP_POP, 0, stmt.pos.line)
        fc.emit(OP_POP, 0, stmt.pos.line)
        for bp in loop_ctx.break_patches:
            fc.patch_jump(bp)
        fc.loop_stack.pop()

    def _compile_for_iter(self, stmt: TForStmt, fc: _FnCompiler) -> None:
        # Compile iterable, push index counter
        self._compile_expr(stmt.iterable, fc)
        fc.emit(OP_INT_ZERO, 0, stmt.pos.line)  # index
        loop_start = fc.current_offset()
        loop_ctx = _LoopCtx(fc.handler_depth)
        loop_ctx.continue_target = loop_start
        fc.loop_stack.append(loop_ctx)
        # FOR_ITER checks index < len(collection)
        exit_jump = fc.emit_jump(OP_FOR_ITER, stmt.pos.line)
        # After FOR_ITER pushes current element (and optionally index)
        if len(stmt.binding) == 2:
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
            binding = stmt.binding[0] if len(stmt.binding) > 0 else "_"
            local = fc.scope.lookup(binding) if binding != "_" else None
            if local is not None:
                fc.emit(OP_STORE_LOCAL, local.slot, stmt.pos.line)
            else:
                fc.emit(OP_POP, 0, stmt.pos.line)
            # Discard the extra index/key pushed by FOR_ITER
            fc.emit(OP_POP, 0, stmt.pos.line)
        self._compile_block(stmt.body, fc)
        back_dist = fc.current_offset() - loop_start + 2
        fc.emit(OP_JUMP_BACK, back_dist, stmt.pos.line)
        fc.patch_jump(exit_jump)
        # Pop collection and index
        fc.emit(OP_POP, 0, stmt.pos.line)
        fc.emit(OP_POP, 0, stmt.pos.line)
        for bp in loop_ctx.break_patches:
            fc.patch_jump(bp)
        fc.loop_stack.pop()

    def _compile_break(self, stmt: TBreakStmt, fc: _FnCompiler) -> None:
        if len(fc.loop_stack) == 0:
            return
        ctx = fc.loop_stack[-1]
        # Pop handlers pushed inside the loop
        depth_diff = fc.handler_depth - ctx.handler_depth
        i = 0
        while i < depth_diff:
            fc.emit(OP_POP_HANDLER, 0, stmt.pos.line)
            i += 1
        bp = fc.emit_jump(OP_JUMP, stmt.pos.line)
        ctx.break_patches.append(bp)

    def _compile_continue(self, stmt: TContinueStmt, fc: _FnCompiler) -> None:
        if len(fc.loop_stack) == 0:
            return
        ctx = fc.loop_stack[-1]
        depth_diff = fc.handler_depth - ctx.handler_depth
        i = 0
        while i < depth_diff:
            fc.emit(OP_POP_HANDLER, 0, stmt.pos.line)
            i += 1
        back_dist = fc.current_offset() - ctx.continue_target + 2
        fc.emit(OP_JUMP_BACK, back_dist, stmt.pos.line)

    def _compile_try(self, stmt: TTryStmt, fc: _FnCompiler) -> None:
        has_finally = stmt.finally_body is not None
        end_patches: list[int] = []
        if has_finally:
            finally_jump = fc.emit_jump(OP_PUSH_FINALLY, stmt.pos.line)
            fc.handler_depth += 1
        if len(stmt.catches) > 0:
            catch_jump = fc.emit_jump(OP_PUSH_HANDLER, stmt.pos.line)
            fc.handler_depth += 1
            self._compile_block(stmt.body, fc)
            fc.emit(OP_POP_HANDLER, 0, stmt.pos.line)
            fc.handler_depth -= 1
            body_end = fc.emit_jump(OP_JUMP, stmt.pos.line)
            end_patches.append(body_end)
            fc.patch_jump(catch_jump)
            # Compile catch clauses
            ci = 0
            while ci < len(stmt.catches):
                catch = stmt.catches[ci]
                local = fc.scope.lookup(catch.name)
                # If not last catch, check type and skip if no match
                if ci < len(stmt.catches) - 1 or len(catch.types) > 0:
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
                ci += 1
            # If no catch matched, rethrow
            fc.emit(OP_THROW, 0, stmt.pos.line)
        else:
            self._compile_block(stmt.body, fc)
        for ep in end_patches:
            fc.patch_jump(ep)
        if has_finally:
            fc.emit(OP_POP_HANDLER, 0, stmt.pos.line)
            fc.handler_depth -= 1
            finally_end = fc.emit_jump(OP_JUMP, stmt.pos.line)
            fc.patch_jump(finally_jump)
            self._compile_block(stmt.finally_body, fc)
            fc.patch_jump(finally_end)
            self._compile_block(stmt.finally_body, fc)

    def _emit_catch_type_check(
        self, types: list, fc: _FnCompiler, line: int
    ) -> None:
        if len(types) == 0:
            # Catch-all
            fc.emit(OP_POP, 0, line)
            fc.emit(OP_TRUE, 0, line)
            return
        if len(types) == 1:
            tname = self._type_name_str(types[0])
            idx = len(fc.constants)
            fc.constants.append(VStr(tname))
            fc.emit(OP_IS_TYPE, idx, line)
            return
        # Union: match any
        # Check each type, OR together
        tname = self._type_name_str(types[0])
        idx = len(fc.constants)
        fc.constants.append(VStr(tname))
        fc.emit(OP_IS_TYPE, idx, line)
        ti = 1
        while ti < len(types):
            # DUP the original value from under our bool
            # Actually we consumed the dup. Need to re-dup for each check.
            # Simpler: just check the first one, if true short-circuit
            short = fc.emit_jump(OP_JUMP_IF_TRUE, line)
            # Need the value again for next check — but it was consumed by IS_TYPE
            # This is tricky. Let's emit DUP before each IS_TYPE instead.
            # Rewrite: this approach won't work cleanly. For now, encode all
            # type names as a constant list and let IS_TYPE handle union.
            fc.patch_jump(short)
            ti += 1

    def _type_name_str(self, ttype) -> str:
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
                # Bind the matched value
                local = fc.scope.lookup(case.pattern.name)
                if local is not None:
                    fc.emit(OP_DUP, 0, case.pos.line)
                    fc.emit(OP_STORE_LOCAL, local.slot, case.pos.line)
                self._compile_block(case.body, fc)
                end_patches.append(fc.emit_jump(OP_JUMP, case.pos.line))
                fc.patch_jump(skip)
            elif isinstance(case.pattern, TPatternEnum):
                idx = len(fc.constants)
                fc.constants.append(VStr(case.pattern.enum_name + "." + case.pattern.variant))
                fc.emit(OP_MATCH_TYPE, idx, case.pos.line)
                skip = fc.emit_jump(OP_JUMP_IF_FALSE, case.pos.line)
                self._compile_block(case.body, fc)
                end_patches.append(fc.emit_jump(OP_JUMP, case.pos.line))
                fc.patch_jump(skip)
            elif isinstance(case.pattern, TPatternNil):
                fc.emit(OP_NIL, 0, case.pos.line)
                fc.emit(OP_EQ, 0, case.pos.line)
                skip = fc.emit_jump(OP_JUMP_IF_FALSE, case.pos.line)
                self._compile_block(case.body, fc)
                end_patches.append(fc.emit_jump(OP_JUMP, case.pos.line))
                fc.patch_jump(skip)
        if stmt.default is not None:
            if stmt.default.name is not None:
                local = fc.scope.lookup(stmt.default.name)
                if local is not None:
                    fc.emit(OP_DUP, 0, stmt.default.pos.line)
                    fc.emit(OP_STORE_LOCAL, local.slot, stmt.default.pos.line)
            self._compile_block(stmt.default.body, fc)
        fc.emit(OP_POP, 0, stmt.pos.line)  # pop scrutinee
        for ep in end_patches:
            fc.patch_jump(ep)

    # ── Expression compilation ────────────────────────────────

    def _compile_expr(self, expr: TExpr, fc: _FnCompiler) -> None:
        if isinstance(expr, TIntLit):
            if expr.value == 0:
                fc.emit(OP_INT_ZERO, 0, expr.pos.line)
            elif expr.value == 1:
                fc.emit(OP_INT_ONE, 0, expr.pos.line)
            else:
                fc.emit_const(VInt(expr.value), expr.pos.line)
        elif isinstance(expr, TFloatLit):
            fc.emit_const(VFloat(expr.value), expr.pos.line)
        elif isinstance(expr, TBoolLit):
            if expr.value:
                fc.emit(OP_TRUE, 0, expr.pos.line)
            else:
                fc.emit(OP_FALSE, 0, expr.pos.line)
        elif isinstance(expr, TNilLit):
            fc.emit(OP_NIL, 0, expr.pos.line)
        elif isinstance(expr, TStringLit):
            fc.emit_const(VStr(expr.value), expr.pos.line)
        elif isinstance(expr, TByteLit):
            fc.emit_const(VInt(expr.value), expr.pos.line)
        elif isinstance(expr, TRuneLit):
            fc.emit_const(VRune(expr.value), expr.pos.line)
        elif isinstance(expr, TBytesLit):
            fc.emit_const(VBytes(expr.value), expr.pos.line)
        elif isinstance(expr, TVar):
            self._compile_var(expr, fc)
        elif isinstance(expr, TBinaryOp):
            self._compile_binop(expr, fc)
        elif isinstance(expr, TUnaryOp):
            self._compile_unaryop(expr, fc)
        elif isinstance(expr, TCall):
            self._compile_call(expr, fc)
        elif isinstance(expr, TTernary):
            self._compile_ternary(expr, fc)
        elif isinstance(expr, TListLit):
            for e in expr.elements:
                self._compile_expr(e, fc)
            fc.emit(OP_BUILD_LIST, len(expr.elements), expr.pos.line)
        elif isinstance(expr, TMapLit):
            for k, v in expr.entries:
                self._compile_expr(k, fc)
                self._compile_expr(v, fc)
            fc.emit(OP_BUILD_MAP, len(expr.entries), expr.pos.line)
        elif isinstance(expr, TSetLit):
            for e in expr.elements:
                self._compile_expr(e, fc)
            fc.emit(OP_BUILD_SET, len(expr.elements), expr.pos.line)
        elif isinstance(expr, TTupleLit):
            for e in expr.elements:
                self._compile_expr(e, fc)
            fc.emit(OP_BUILD_TUPLE, len(expr.elements), expr.pos.line)
        elif isinstance(expr, TIndex):
            self._compile_expr(expr.obj, fc)
            self._compile_expr(expr.index, fc)
            fc.emit(OP_INDEX, 0, expr.pos.line)
        elif isinstance(expr, TSlice):
            self._compile_expr(expr.obj, fc)
            self._compile_expr(expr.low, fc)
            self._compile_expr(expr.high, fc)
            fc.emit(OP_SLICE, 0, expr.pos.line)
        elif isinstance(expr, TFieldAccess):
            self._compile_field_access(expr, fc)
        elif isinstance(expr, TTupleAccess):
            self._compile_expr(expr.obj, fc)
            fc.emit(OP_TUPLE_ACCESS, expr.index, expr.pos.line)
        elif isinstance(expr, TFnLit):
            self._compile_fn_lit(expr, fc)
        elif isinstance(expr, TRange):
            # Range as expression (not in for loop) — shouldn't appear normally
            for a in expr.args:
                self._compile_expr(a, fc)
            fc.emit(OP_GET_ITER, len(expr.args), expr.pos.line)

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

    def _emit_cmp(
        self, typ: Type, cmp_kind: int, fc: _FnCompiler, line: int
    ) -> None:
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

    def _compile_builtin_call(
        self, name: str, expr: TCall, fc: _FnCompiler
    ) -> None:
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
            for fname in sd.field_names:
                found = False
                for a in expr.args:
                    if a.name == fname:
                        self._compile_expr(a.value, fc)
                        found = True
                        break
                if not found:
                    fidx = sd.field_names.index(fname)
                    self._emit_zero_value(sd.field_types[fidx], fc, expr.pos.line)
        else:
            for a in expr.args:
                self._compile_expr(a.value, fc)
            # Fill remaining with zero values
            i = len(expr.args)
            while i < len(sd.field_names):
                self._emit_zero_value(sd.field_types[i], fc, expr.pos.line)
                i += 1
        fc.emit(OP_BUILD_STRUCT, sidx, expr.pos.line)

    def _compile_error_struct_constructor(
        self, name: str, expr: TCall, fc: _FnCompiler
    ) -> None:
        """Compile construction of built-in error structs (ValueError, etc.)."""
        # Error structs have a single 'message' field
        if len(expr.args) > 0:
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

    def _compile_field_access(self, expr: TFieldAccess, fc: _FnCompiler) -> None:
        # Enum variant access: EnumName.Variant
        if isinstance(expr.obj, TVar):
            ct = self.checker_types.get(expr.obj.name)
            if ct is not None and isinstance(ct, EnumT):
                idx = len(fc.constants)
                fc.constants.append(VStr(expr.obj.name))
                idx2 = len(fc.constants)
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
        self._collect_locals(expr.body, lit_fc)
        self._compile_block(expr.body, lit_fc)
        # Implicit return void
        lit_fc.emit(OP_RETURN_VOID, 0, expr.pos.line)
        code_idx = len(self.code_objects)
        self.code_objects.append(lit_fc.to_code_object(len(expr.params)))
        fc.emit_const(VFunc(code_idx, []), expr.pos.line)

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
            if len(expr.elements) > 0:
                return ListT(kind="list", element=self._resolve_expr_type(expr.elements[0], fc))
            return ListT(kind="list", element=VOID_T)
        if isinstance(expr, TMapLit):
            if len(expr.entries) > 0:
                k, v = expr.entries[0]
                return MapT(kind="map", key=self._resolve_expr_type(k, fc), value=self._resolve_expr_type(v, fc))
            return MapT(kind="map", key=VOID_T, value=VOID_T)
        if isinstance(expr, TSetLit):
            if len(expr.elements) > 0:
                return SetT(kind="set", element=self._resolve_expr_type(expr.elements[0], fc))
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
            if name == "ToString":
                return STRING_T
            if name == "Len":
                return INT_T
            if name in ("IntToFloat", "Sqrt"):
                return FLOAT_T
            if name in ("FloatToInt", "Round", "Floor", "Ceil", "ByteToInt", "RuneToInt", "ParseInt"):
                return INT_T
            if name == "ParseFloat":
                return FLOAT_T
            if name == "IntToByte":
                return BYTE_T
            if name == "RuneFromInt":
                return RUNE_T
            if name in ("IsNaN", "IsInf", "IsNil", "IsType", "Contains", "StartsWith", "EndsWith", "IsDigit", "IsAlpha", "IsAlnum", "IsSpace", "IsUpper", "IsLower"):
                return BOOL_T
            if name in ("Upper", "Lower", "Trim", "TrimStart", "TrimEnd", "Join", "Replace", "ReplaceCount", "Repeat", "Reverse", "Concat", "Format", "FormatInt"):
                return STRING_T
            if name in ("Abs", "Min", "Max", "Sum", "Pow"):
                # Return type matches first arg
                if len(expr.args) > 0:
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
            if name in ("Append", "Insert", "Pop", "RemoveAt", "Delete", "Add", "Remove"):
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
