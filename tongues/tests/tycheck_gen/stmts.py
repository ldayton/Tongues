"""StmtGen — statement generation with weighted selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.taytsh.ast import (
    TArg,
    TAssignStmt,
    TBinaryOp,
    TBreakStmt,
    TCall,
    TCatch,
    TContinueStmt,
    TDefault,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TIdentType,
    TForStmt,
    TIfStmt,
    TIndex,
    TIntLit,
    TLetStmt,
    TMatchCase,
    TMatchStmt,
    TOpAssignStmt,
    TPrimitive,
    TTupleAssignStmt,
    TPatternEnum,
    TPatternNil,
    TPatternType,
    TRange,
    TReturnStmt,
    TStmt,
    TStringLit,
    TThrowStmt,
    TTryStmt,
    TUnaryOp,
    TVar,
    TWhileStmt,
    TNilLit,
)
from src.taytsh.check import (
    BOOL_T,
    BYTE_T,
    FLOAT_T,
    INT_T,
    NIL_T,
    RUNE_T,
    STRING_T,
    BYTES_T,
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
    contains_nil,
    normalize_union,
    remove_nil,
    type_eq,
)

from .types import make_ttype
from .ast_helpers import P, A

if TYPE_CHECKING:
    from . import Generator

MAX_STMT_DEPTH = 3


class StmtGen:
    def __init__(self, gen: Generator) -> None:
        self.gen = gen
        self.rng = gen.rng
        self._pending_stmts: list[TStmt] = []

    def gen_block(
        self, count: int, must_return: Type | None, *, depth: int = 0
    ) -> list[TStmt]:
        stmts: list[TStmt] = []
        for i in range(count):
            if i == count - 1 and must_return is not None:
                if not type_eq(must_return, VOID_T):
                    ret_expr = self.gen.expr_gen.gen_expr(must_return)
                    stmts.append(TReturnStmt(pos=P, value=ret_expr, annotations=A))
                else:
                    stmts.append(TReturnStmt(pos=P, value=None, annotations=A))
                break
            stmt = self._pick_stmt(depth)
            if stmt is not None:
                if self._pending_stmts:
                    stmts.extend(self._pending_stmts)
                    self._pending_stmts = []
                stmts.append(stmt)
                if isinstance(
                    stmt, (TReturnStmt, TBreakStmt, TContinueStmt, TThrowStmt)
                ):
                    break
        return stmts

    def _pick_stmt(self, depth: int) -> TStmt | None:
        candidates: list[tuple[int, str]] = []
        candidates.append((30, "let"))
        if self.gen.scope.mutable_bindings():
            candidates.append((20, "assign"))
        if self.gen.features.compound_assign and self._find_numeric_binding():
            candidates.append((8, "op_assign"))
        if self.gen.features.tuple_destructure and self._find_tuple_binding():
            candidates.append((5, "tuple_assign"))
        if depth < MAX_STMT_DEPTH:
            candidates.append((15, "if"))
            candidates.append((8, "while"))
            candidates.append((12, "for"))
        matchable = self._find_matchable_binding()
        if matchable is not None and depth < MAX_STMT_DEPTH:
            candidates.append((10, "match"))
        if depth < MAX_STMT_DEPTH:
            candidates.append((5, "try"))
        if not self.gen.in_finally:
            struct_types = self.gen.pool.struct_types()
            if struct_types:
                candidates.append((3, "throw"))
            if self.gen.current_fn_ret is not None:
                candidates.append((5, "return"))
            if self.gen.in_loop:
                candidates.append((3, "break"))
                candidates.append((3, "continue"))
        candidates.append((10, "expr"))

        total = sum(w for w, _ in candidates)
        r = self.rng.randint(0, total - 1)
        acc = 0
        for weight, kind in candidates:
            acc += weight
            if r < acc:
                return self._gen_stmt(kind, depth)
        return self._gen_let(depth)

    def _gen_stmt(self, kind: str, depth: int) -> TStmt | None:
        if kind == "let":
            return self._gen_let(depth)
        if kind == "assign":
            return self._gen_assign()
        if kind == "op_assign":
            return self._gen_op_assign()
        if kind == "tuple_assign":
            return self._gen_tuple_assign()
        if kind == "if":
            return self._gen_if(depth)
        if kind == "while":
            return self._gen_while(depth)
        if kind == "for":
            return self._gen_for(depth)
        if kind == "match":
            return self._gen_match(depth)
        if kind == "try":
            return self._gen_try(depth)
        if kind == "throw":
            return self._gen_throw()
        if kind == "return":
            return self._gen_return()
        if kind == "break":
            return TBreakStmt(pos=P, annotations=A)
        if kind == "continue":
            return TContinueStmt(pos=P, annotations=A)
        if kind == "expr":
            return self._gen_expr_stmt()
        return None

    def _gen_let(self, depth: int = 0) -> TStmt:
        typ = self.gen.pool.random_value_type()
        if self.gen.in_finally:
            if isinstance(typ, FnT):
                typ = INT_T
            elif isinstance(typ, UnionT) and any(
                isinstance(m, FnT) for m in typ.members
            ):
                typ = INT_T
        # Avoid unions where all members have collection invariance issues
        if isinstance(typ, UnionT) and all(
            self.gen.expr_gen._has_invariance_issue(m) for m in typ.members
        ):
            typ = INT_T
        all_names = set()
        for b in self.gen.scope.all_bindings():
            all_names.add(b.name)
        name = self.gen.names.var_name(all_names)

        ttype = make_ttype(typ)

        has_zero = self.gen.pool.has_zero_value(typ)
        omit_init = has_zero and self.rng.random() < 0.3
        if omit_init:
            value = None
        else:
            value = self.gen.expr_gen.gen_expr(typ)

        self.gen.scope.declare(name, typ)
        return TLetStmt(pos=P, name=name, typ=ttype, value=value, annotations=A)

    def _gen_assign(self) -> TStmt:
        can_gen = self.gen.expr_gen._can_gen_type
        bindings = [b for b in self.gen.scope.mutable_bindings() if can_gen(b.typ)]
        if not bindings:
            return self._gen_let()
        options: list[tuple[str, object]] = [("var", None)]
        for b in bindings:
            if isinstance(b.typ, StructT):
                for fname, ftype in b.typ.fields.items():
                    if can_gen(ftype):
                        options.append(("field", (b.name, fname, ftype)))
            if isinstance(b.typ, ListT) and can_gen(b.typ.element):
                options.append(("index", (b.name, b.typ.element, INT_T)))
            elif isinstance(b.typ, MapT) and can_gen(b.typ.value):
                options.append(("index", (b.name, b.typ.value, b.typ.key)))
        kind, info = self.rng.choice(options)
        if kind == "field":
            var_name, fname, ftype = info
            value = self.gen.expr_gen.gen_expr(ftype)
            return TAssignStmt(
                pos=P,
                target=TFieldAccess(
                    pos=P,
                    obj=TVar(pos=P, name=var_name, annotations=A),
                    field=fname,
                    annotations=A,
                ),
                value=value,
                annotations=A,
            )
        if kind == "index":
            var_name, elem_type, key_type = info
            key_expr = self.gen.expr_gen.gen_expr(key_type)
            value = self.gen.expr_gen.gen_expr(elem_type)
            return TAssignStmt(
                pos=P,
                target=TIndex(
                    pos=P,
                    obj=TVar(pos=P, name=var_name, annotations=A),
                    index=key_expr,
                    annotations=A,
                ),
                value=value,
                annotations=A,
            )
        b = self.rng.choice(bindings)
        value = self.gen.expr_gen.gen_expr(b.typ)
        return TAssignStmt(
            pos=P,
            target=TVar(pos=P, name=b.name, annotations=A),
            value=value,
            annotations=A,
        )

    def _find_numeric_binding(self) -> tuple[str, Type] | None:
        for b in self.gen.scope.mutable_bindings():
            if (
                type_eq(b.typ, INT_T)
                or type_eq(b.typ, FLOAT_T)
                or type_eq(b.typ, BYTE_T)
            ):
                return (b.name, b.typ)
        return None

    def _gen_op_assign(self) -> TStmt:
        result = self._find_numeric_binding()
        if result is None:
            return self._gen_let()
        var_name, typ = result
        op = self.rng.choice(["+=", "-=", "*="])
        value = self.gen.expr_gen.gen_expr(typ)
        return TOpAssignStmt(
            pos=P,
            target=TVar(pos=P, name=var_name, annotations=A),
            op=op,
            value=value,
            annotations=A,
        )

    def _find_tuple_binding(self) -> tuple[str, TupleT] | None:
        for b in self.gen.scope.all_bindings():
            if isinstance(b.typ, TupleT) and len(b.typ.elements) >= 2:
                return (b.name, b.typ)
        return None

    def _gen_tuple_assign(self) -> TStmt:
        """Generate let declarations + tuple destructuring assignment."""
        result = self._find_tuple_binding()
        if result is None:
            return self._gen_let()
        var_name, tup_type = result
        all_names = {b.name for b in self.gen.scope.all_bindings()}
        targets: list[TExpr] = []
        preamble: list[TStmt] = []
        for elem_t in tup_type.elements:
            tgt_name = self.gen.names.var_name(all_names)
            all_names.add(tgt_name)
            ttype = make_ttype(elem_t)
            init = self.gen.expr_gen.gen_expr(elem_t)
            self.gen.scope.declare(tgt_name, elem_t)
            preamble.append(
                TLetStmt(pos=P, name=tgt_name, typ=ttype, value=init, annotations=A)
            )
            targets.append(TVar(pos=P, name=tgt_name, annotations=A))
        self._pending_stmts = self._pending_stmts + preamble
        return TTupleAssignStmt(
            pos=P,
            targets=targets,
            value=TVar(pos=P, name=var_name, annotations=A),
            annotations=A,
        )

    def _gen_if(self, depth: int) -> TStmt:
        # Try nil narrowing
        if self.gen.features.nil_narrowing and self.rng.random() < 0.3:
            result = self._gen_if_nil_narrow(depth)
            if result is not None:
                return result
        # Try IsType narrowing
        if self.gen.features.nil_narrowing and self.rng.random() < 0.3:
            result = self._gen_if_istype(depth)
            if result is not None:
                return result
        # Plain if
        cond = self.gen.expr_gen.gen_expr(BOOL_T)
        self.gen.scope.enter_scope()
        then_body = self.gen_block(
            self.rng.randint(1, 2), must_return=None, depth=depth + 1
        )
        self.gen.scope.exit_scope()
        if self.rng.random() < 0.5:
            self.gen.scope.enter_scope()
            else_body = self.gen_block(
                self.rng.randint(1, 2), must_return=None, depth=depth + 1
            )
            self.gen.scope.exit_scope()
        else:
            else_body = None
        return TIfStmt(
            pos=P, cond=cond, then_body=then_body, else_body=else_body, annotations=A
        )

    def _find_optional_binding(self) -> tuple[str, Type] | None:
        """Find a binding with optional type T? in scope (no fn types)."""
        for b in self.gen.scope.all_bindings():
            if isinstance(b.typ, UnionT) and contains_nil(b.typ):
                inner = remove_nil(b.typ)
                if not type_eq(inner, NIL_T) and not self.gen.expr_gen._contains_fn(
                    b.typ
                ):
                    return (b.name, b.typ)
        return None

    def _find_interface_binding(self) -> tuple[str, InterfaceT] | None:
        """Find a binding with interface type in scope."""
        for b in self.gen.scope.all_bindings():
            if isinstance(b.typ, InterfaceT):
                info = self.gen.pool.interface_info_for(b.typ)
                if info is not None and info.variant_names:
                    return (b.name, b.typ)
        return None

    def _gen_if_nil_narrow(self, depth: int) -> TStmt | None:
        """Generate `if x != nil { ... use narrowed x ... }`."""
        found = self._find_optional_binding()
        if found is None:
            return None
        var_name, opt_type = found
        inner = remove_nil(opt_type)
        cond = TBinaryOp(
            pos=P,
            op="!=",
            left=TVar(pos=P, name=var_name, annotations=A),
            right=TNilLit(pos=P, annotations=A),
            annotations=A,
        )
        self.gen.scope.enter_scope()
        self.gen.scope.narrow(var_name, inner)
        then_body = self.gen_block(
            self.rng.randint(1, 2), must_return=None, depth=depth + 1
        )
        self.gen.scope.exit_scope()
        if self.rng.random() < 0.4:
            self.gen.scope.enter_scope()
            self.gen.scope.narrow(var_name, NIL_T)
            else_body = self.gen_block(
                self.rng.randint(1, 2), must_return=None, depth=depth + 1
            )
            self.gen.scope.exit_scope()
        else:
            else_body = None
        # Mirror checker's guard narrowing: if then exits early, narrow to nil
        if else_body is None and self._block_always_exits(then_body):
            self.gen.scope.narrow(var_name, NIL_T)
        return TIfStmt(
            pos=P, cond=cond, then_body=then_body, else_body=else_body, annotations=A
        )

    @staticmethod
    def _block_always_exits(body: list[TStmt]) -> bool:
        if not body:
            return False
        last = body[-1]
        return isinstance(last, (TReturnStmt, TBreakStmt, TContinueStmt, TThrowStmt))

    def _gen_if_istype(self, depth: int) -> TStmt | None:
        """Generate `if IsType(x, "V") { ... use x as V ... }`."""
        found = self._find_interface_binding()
        if found is None:
            return None
        var_name, iface = found
        info = self.gen.pool.interface_info_for(iface)
        vname = self.rng.choice(info.variant_names)
        st = self.gen.pool.struct_for_name(vname)
        if st is None:
            return None
        cond: TExpr = TCall(
            pos=P,
            func=TVar(pos=P, name="IsType", annotations=A),
            args=[
                TArg(pos=P, name=None, value=TVar(pos=P, name=var_name, annotations=A)),
                TArg(
                    pos=P,
                    name=None,
                    value=TStringLit(pos=P, value=vname, annotations=A),
                ),
            ],
            annotations=A,
        )
        # Optionally negate for guard narrowing: if !IsType(x, "V") { return }
        if (
            self.gen.current_fn_ret is not None
            and not self.gen.in_finally
            and self.rng.random() < 0.3
        ):
            neg_cond = TUnaryOp(pos=P, op="!", operand=cond, annotations=A)
            self.gen.scope.enter_scope()
            ret = self.gen.current_fn_ret
            if type_eq(ret, VOID_T):
                guard_body: list[TStmt] = [
                    TReturnStmt(pos=P, value=None, annotations=A)
                ]
            else:
                guard_body = [
                    TReturnStmt(
                        pos=P,
                        value=self.gen.expr_gen.gen_expr(ret),
                        annotations=A,
                    )
                ]
            self.gen.scope.exit_scope()
            # After the guard, x is narrowed in the remaining scope
            self.gen.scope.narrow(var_name, st)
            return TIfStmt(
                pos=P,
                cond=neg_cond,
                then_body=guard_body,
                else_body=None,
                annotations=A,
            )
        # Normal IsType: narrow in then-scope
        self.gen.scope.enter_scope()
        self.gen.scope.narrow(var_name, st)
        then_body = self.gen_block(
            self.rng.randint(1, 2), must_return=None, depth=depth + 1
        )
        self.gen.scope.exit_scope()
        if self.rng.random() < 0.4:
            self.gen.scope.enter_scope()
            else_body = self.gen_block(
                self.rng.randint(1, 2), must_return=None, depth=depth + 1
            )
            self.gen.scope.exit_scope()
        else:
            else_body = None
        return TIfStmt(
            pos=P, cond=cond, then_body=then_body, else_body=else_body, annotations=A
        )

    def _gen_while(self, depth: int) -> TStmt:
        cond = self.gen.expr_gen.gen_expr(BOOL_T)
        old_in_loop = self.gen.in_loop
        self.gen.in_loop = True
        self.gen.scope.enter_scope()
        body = self.gen_block(self.rng.randint(1, 3), must_return=None, depth=depth + 1)
        self.gen.scope.exit_scope()
        self.gen.in_loop = old_in_loop
        return TWhileStmt(pos=P, cond=cond, body=body, annotations=A)

    def _gen_for(self, depth: int) -> TStmt:
        old_in_loop = self.gen.in_loop
        self.gen.in_loop = True
        self.gen.scope.enter_scope()

        use_range = self.rng.random() < 0.5
        if use_range:
            all_names = set()
            for b in self.gen.scope.all_bindings():
                all_names.add(b.name)
            var_name = self.gen.names.var_name(all_names)
            self.gen.scope.declare(var_name, INT_T, is_loop_var=True)
            start = TIntLit(pos=P, value=0, raw="0", annotations=A)
            end = TIntLit(pos=P, value=10, raw="10", annotations=A)
            iterable = TRange(pos=P, args=[start, end], annotations=A)
            binding = [var_name]
        else:
            # Find a collection binding
            collection_bindings = [
                b
                for b in self.gen.scope.all_bindings()
                if isinstance(b.typ, (ListT, MapT, SetT))
                or type_eq(b.typ, STRING_T)
                or type_eq(b.typ, BYTES_T)
            ]
            if collection_bindings:
                cb = self.rng.choice(collection_bindings)
                iterable = TVar(pos=P, name=cb.name, annotations=A)
                binding, bound_types = self._for_binding_types(cb.typ)
                for bname, btype in zip(binding, bound_types):
                    self.gen.scope.declare(bname, btype, is_loop_var=True)
            else:
                # Fall back to range
                all_names = set()
                for b in self.gen.scope.all_bindings():
                    all_names.add(b.name)
                var_name = self.gen.names.var_name(all_names)
                self.gen.scope.declare(var_name, INT_T, is_loop_var=True)
                start = TIntLit(pos=P, value=0, raw="0", annotations=A)
                end = TIntLit(pos=P, value=5, raw="5", annotations=A)
                iterable = TRange(pos=P, args=[start, end], annotations=A)
                binding = [var_name]

        body = self.gen_block(self.rng.randint(1, 2), must_return=None, depth=depth + 1)
        self.gen.scope.exit_scope()
        self.gen.in_loop = old_in_loop
        return TForStmt(
            pos=P, binding=binding, iterable=iterable, body=body, annotations=A
        )

    def _for_binding_types(self, iter_type: Type) -> tuple[list[str], list[Type]]:
        all_names = set()
        for b in self.gen.scope.all_bindings():
            all_names.add(b.name)

        if isinstance(iter_type, ListT):
            name = self.gen.names.var_name(all_names)
            return [name], [iter_type.element]
        if isinstance(iter_type, MapT):
            n1 = self.gen.names.var_name(all_names)
            all_names.add(n1)
            n2 = self.gen.names.var_name(all_names)
            return [n1, n2], [iter_type.key, iter_type.value]
        if isinstance(iter_type, SetT):
            name = self.gen.names.var_name(all_names)
            return [name], [iter_type.element]
        if type_eq(iter_type, STRING_T):
            name = self.gen.names.var_name(all_names)
            return [name], [RUNE_T]
        if type_eq(iter_type, BYTES_T):
            name = self.gen.names.var_name(all_names)
            return [name], [BYTE_T]
        name = self.gen.names.var_name(all_names)
        return [name], [INT_T]

    def _find_matchable_binding(self) -> tuple[str, Type] | None:
        for b in self.gen.scope.all_bindings():
            if isinstance(b.typ, (InterfaceT, EnumT)):
                return (b.name, b.typ)
            if isinstance(b.typ, UnionT):
                enum_count = sum(1 for m in b.typ.members if isinstance(m, EnumT))
                if enum_count <= 1:
                    return (b.name, b.typ)
        return None

    def _gen_match(self, depth: int) -> TStmt:
        result = self._find_matchable_binding()
        if result is None:
            return self._gen_let()
        var_name, scrutinee = result

        cases: list[TMatchCase] = []
        if isinstance(scrutinee, InterfaceT):
            for vname in scrutinee.variants:
                st = self.gen.pool.struct_for_name(vname)
                if st is None:
                    continue
                bind_name = self._fresh_case_bind_name(var_name)
                pat = TPatternType(
                    pos=P,
                    name=bind_name,
                    type_name=make_ttype(st),
                    annotations=A,
                )
                self.gen.scope.enter_scope()
                self.gen.scope.declare(bind_name, st)
                self.gen.scope.narrow(var_name, st)
                body = self.gen_block(
                    self.rng.randint(1, 2), must_return=None, depth=depth + 1
                )
                self.gen.scope.exit_scope()
                cases.append(TMatchCase(pos=P, pattern=pat, body=body, annotations=A))

        elif isinstance(scrutinee, EnumT):
            for variant in scrutinee.variants:
                pat = TPatternEnum(pos=P, enum_name=scrutinee.name, variant=variant)
                self.gen.scope.enter_scope()
                self.gen.scope.narrow(var_name, scrutinee)
                body = self.gen_block(
                    self.rng.randint(1, 2), must_return=None, depth=depth + 1
                )
                self.gen.scope.exit_scope()
                cases.append(TMatchCase(pos=P, pattern=pat, body=body, annotations=A))

        elif isinstance(scrutinee, UnionT):
            # Collect direct struct member names to avoid duplicates when
            # expanding interface variants
            direct_struct_names: set[str] = set()
            for member in scrutinee.members:
                if isinstance(member, StructT):
                    direct_struct_names.add(member.name)
            for member in scrutinee.members:
                if type_eq(member, NIL_T):
                    pat = TPatternNil(pos=P)
                    self.gen.scope.enter_scope()
                    body = self.gen_block(
                        self.rng.randint(1, 2), must_return=None, depth=depth + 1
                    )
                    self.gen.scope.exit_scope()
                    cases.append(
                        TMatchCase(pos=P, pattern=pat, body=body, annotations=A)
                    )
                elif isinstance(member, InterfaceT):
                    for vname in member.variants:
                        if vname in direct_struct_names:
                            continue
                        st = self.gen.pool.struct_for_name(vname)
                        if st is None:
                            continue
                        bind_name = self._fresh_case_bind_name(var_name)
                        pat = TPatternType(
                            pos=P,
                            name=bind_name,
                            type_name=make_ttype(st),
                            annotations=A,
                        )
                        self.gen.scope.enter_scope()
                        self.gen.scope.declare(bind_name, st)
                        self.gen.scope.narrow(var_name, st)
                        body = self.gen_block(
                            self.rng.randint(1, 2), must_return=None, depth=depth + 1
                        )
                        self.gen.scope.exit_scope()
                        cases.append(
                            TMatchCase(pos=P, pattern=pat, body=body, annotations=A)
                        )
                elif isinstance(member, EnumT):
                    for variant in member.variants:
                        pat = TPatternEnum(
                            pos=P, enum_name=member.name, variant=variant
                        )
                        self.gen.scope.enter_scope()
                        body = self.gen_block(
                            self.rng.randint(1, 2), must_return=None, depth=depth + 1
                        )
                        self.gen.scope.exit_scope()
                        cases.append(
                            TMatchCase(pos=P, pattern=pat, body=body, annotations=A)
                        )
                else:
                    bind_name = self._fresh_case_bind_name(var_name)
                    pat = TPatternType(
                        pos=P,
                        name=bind_name,
                        type_name=make_ttype(member),
                        annotations=A,
                    )
                    self.gen.scope.enter_scope()
                    self.gen.scope.declare(bind_name, member)
                    self.gen.scope.narrow(var_name, member)
                    body = self.gen_block(
                        self.rng.randint(1, 2), must_return=None, depth=depth + 1
                    )
                    self.gen.scope.exit_scope()
                    cases.append(
                        TMatchCase(pos=P, pattern=pat, body=body, annotations=A)
                    )

        expr = TVar(pos=P, name=var_name, annotations=A)
        default = None
        if (
            self.gen.features.match_default
            and len(cases) >= 2
            and self.rng.random() < 0.3
        ):
            n_drop = self.rng.randint(1, min(2, len(cases) - 1))
            cases = cases[:-n_drop]
            bind_name: str | None = None
            if self.gen.features.match_default_bind and self.rng.random() < 0.5:
                bind_name = self._fresh_case_bind_name(var_name)
            self.gen.scope.enter_scope()
            if bind_name is not None:
                residual = self._compute_residual(scrutinee, cases)
                self.gen.scope.declare(bind_name, residual)
            body = self.gen_block(
                self.rng.randint(1, 2), must_return=None, depth=depth + 1
            )
            self.gen.scope.exit_scope()
            default = TDefault(pos=P, name=bind_name, body=body, annotations=A)
        return TMatchStmt(pos=P, expr=expr, cases=cases, default=default, annotations=A)

    def _compute_residual(self, scrutinee: Type, cases: list[TMatchCase]) -> Type:
        """Mirror checker's _compute_default_type: scrutinee minus covered cases.

        We collect covered type names matching the checker's _type_key format,
        then subtract from the scrutinee's variants/members.
        """
        covered: set[str] = set()
        for c in cases:
            pat = c.pattern
            if isinstance(pat, TPatternType):
                if isinstance(pat.type_name, TIdentType):
                    covered.add(pat.type_name.name)
                elif isinstance(pat.type_name, TPrimitive):
                    covered.add(pat.type_name.kind)
            elif isinstance(pat, TPatternEnum):
                covered.add(pat.enum_name + "." + pat.variant)
            elif isinstance(pat, TPatternNil):
                covered.add("nil")
        if isinstance(scrutinee, InterfaceT):
            remaining: list[Type] = []
            for vname in scrutinee.variants:
                st = self.gen.pool.struct_for_name(vname)
                if st is None:
                    continue
                if st.name not in covered:
                    remaining.append(st)
            if not remaining:
                return scrutinee
            if len(remaining) == 1:
                return remaining[0]
            return normalize_union(remaining)
        if isinstance(scrutinee, UnionT):
            remaining_u: list[Type] = []
            for m in scrutinee.members:
                if type_eq(m, NIL_T):
                    if "nil" not in covered:
                        remaining_u.append(m)
                elif isinstance(m, InterfaceT):
                    for vname in m.variants:
                        st = self.gen.pool.struct_for_name(vname)
                        if st is None:
                            continue
                        if st.name not in covered:
                            remaining_u.append(st)
                elif isinstance(m, EnumT):
                    has_uncovered = any(
                        m.name + "." + v not in covered for v in m.variants
                    )
                    if has_uncovered:
                        remaining_u.append(m)
                elif isinstance(m, StructT):
                    if m.name not in covered:
                        remaining_u.append(m)
                else:
                    if m.kind not in covered:
                        remaining_u.append(m)
            if len(remaining_u) == 1:
                return remaining_u[0]
            if len(remaining_u) > 1:
                return normalize_union(remaining_u)
            return scrutinee
        return scrutinee

    def _fresh_case_bind_name(self, base: str) -> str:
        all_names = set()
        for b in self.gen.scope.all_bindings():
            all_names.add(b.name)
        return self.gen.names.var_name(all_names)

    def _gen_try(self, depth: int) -> TStmt:
        self.gen.scope.enter_scope()
        body = self.gen_block(self.rng.randint(1, 2), must_return=None, depth=depth + 1)
        self.gen.scope.exit_scope()

        struct_types = self.gen.pool.struct_types()
        catches: list[TCatch] = []
        if struct_types:
            catch_type = self.rng.choice(struct_types)
            all_names = set()
            for b in self.gen.scope.all_bindings():
                all_names.add(b.name)
            catch_name = self.gen.names.var_name(all_names)
            self.gen.scope.enter_scope()
            self.gen.scope.declare(catch_name, catch_type)
            catch_body = self.gen_block(
                self.rng.randint(1, 2), must_return=None, depth=depth + 1
            )
            self.gen.scope.exit_scope()
            catches.append(
                TCatch(
                    pos=P,
                    name=catch_name,
                    types=[make_ttype(catch_type)],
                    body=catch_body,
                    annotations=A,
                )
            )

        finally_body = None
        if self.rng.random() < 0.3:
            old_in_finally = self.gen.in_finally
            self.gen.in_finally = True
            self.gen.scope.enter_scope()
            finally_body = self.gen_block(
                self.rng.randint(1, 2), must_return=None, depth=depth + 1
            )
            self.gen.scope.exit_scope()
            self.gen.in_finally = old_in_finally

        if not catches and finally_body is None:
            # Need at least a catch or finally
            all_names = set()
            for b in self.gen.scope.all_bindings():
                all_names.add(b.name)
            catch_name = self.gen.names.var_name(all_names)
            self.gen.scope.enter_scope()
            self.gen.scope.declare(catch_name, INT_T)  # catch-all
            catch_body = self.gen_block(1, must_return=None, depth=depth + 1)
            self.gen.scope.exit_scope()
            catches.append(
                TCatch(pos=P, name=catch_name, types=[], body=catch_body, annotations=A)
            )

        return TTryStmt(
            pos=P, body=body, catches=catches, finally_body=finally_body, annotations=A
        )

    def _gen_throw(self) -> TStmt:
        struct_types = self.gen.pool.struct_types()
        if not struct_types:
            return self._gen_let()
        st = self.rng.choice(struct_types)
        expr = self.gen.expr_gen.gen_expr(st)
        return TThrowStmt(pos=P, expr=expr, annotations=A)

    def _gen_return(self) -> TStmt:
        ret = self.gen.current_fn_ret
        if ret is None or type_eq(ret, VOID_T):
            return TReturnStmt(pos=P, value=None, annotations=A)
        value = self.gen.expr_gen.gen_expr(ret)
        return TReturnStmt(pos=P, value=value, annotations=A)

    def _gen_expr_stmt(self) -> TStmt:
        options: list[tuple[str, object]] = []
        struct_types = self.gen.pool.struct_types()
        if struct_types:
            options.append(("struct", None))
        if not self.gen.in_fn_lit:
            for fn_name, fn_type in self.gen.functions.items():
                if type_eq(fn_type.ret, VOID_T):
                    if all(
                        self.gen.expr_gen._can_gen_type(pt) for pt in fn_type.params
                    ):
                        options.append(("fn_call", (fn_name, fn_type)))
            for b in self.gen.scope.all_bindings():
                if isinstance(b.typ, StructT):
                    for mname, mtype in b.typ.methods.items():
                        if type_eq(mtype.ret, VOID_T):
                            if all(
                                self.gen.expr_gen._can_gen_type(pt)
                                for pt in mtype.params[1:]
                            ):
                                options.append(("method", (b.name, mname, mtype)))
        # Void builtins
        options.append(("builtin", None))
        kind, info = self.rng.choice(options)
        if kind == "builtin":
            expr = self.gen.builtin_gen.gen_void_builtin(0)
            if expr is not None:
                return TExprStmt(pos=P, expr=expr, annotations=A)
            # Fall through to struct constructor
            if struct_types:
                kind, info = "struct", None
            else:
                return TExprStmt(
                    pos=P,
                    expr=TStringLit(pos=P, value="noop", annotations=A),
                    annotations=A,
                )
        if not options:
            return TExprStmt(
                pos=P,
                expr=TStringLit(pos=P, value="noop", annotations=A),
                annotations=A,
            )
        if kind == "struct":
            st = self.rng.choice(struct_types)
            expr = self.gen.expr_gen._gen_struct_constructor(st, 0)
            return TExprStmt(pos=P, expr=expr, annotations=A)
        if kind == "fn_call":
            fn_name, fn_type = info
            args: list[TArg] = []
            for pt in fn_type.params:
                args.append(
                    TArg(
                        pos=P,
                        name=None,
                        value=self.gen.expr_gen.gen_expr(pt),
                    )
                )
            expr = TCall(
                pos=P,
                func=TVar(pos=P, name=fn_name, annotations=A),
                args=args,
                annotations=A,
            )
            return TExprStmt(pos=P, expr=expr, annotations=A)
        if kind == "method":
            var_name, mname, mtype = info
            args = []
            for pt in mtype.params[1:]:
                args.append(
                    TArg(
                        pos=P,
                        name=None,
                        value=self.gen.expr_gen.gen_expr(pt),
                    )
                )
            expr = TCall(
                pos=P,
                func=TFieldAccess(
                    pos=P,
                    obj=TVar(pos=P, name=var_name, annotations=A),
                    field=mname,
                    annotations=A,
                ),
                args=args,
                annotations=A,
            )
            return TExprStmt(pos=P, expr=expr, annotations=A)
        return TExprStmt(
            pos=P,
            expr=TStringLit(pos=P, value="noop", annotations=A),
            annotations=A,
        )
