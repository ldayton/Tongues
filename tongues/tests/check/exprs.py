"""ExprGen — top-down expression generation from a target type."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.taytsh.ast import (
    Pos,
    TArg,
    TBinaryOp,
    TBoolLit,
    TByteLit,
    TBytesLit,
    TCall,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFloatLit,
    TFnLit,
    TIndex,
    TIntLit,
    TListLit,
    TMapLit,
    TNilLit,
    TParam,
    TReturnStmt,
    TRuneLit,
    TSetLit,
    TSlice,
    TStringLit,
    TTernary,
    TTupleAccess,
    TTupleLit,
    TUnaryOp,
    TVar,
)
from src.taytsh.check import (
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
    UnionT,
    contains_nil,
    remove_nil,
    type_eq,
)

from .types import make_ttype

if TYPE_CHECKING:
    from . import Generator
    from .scope import Binding

P = Pos(1, 1)
A: dict[str, str] = {}

MAX_DEPTH = 4


class ExprGen:
    def __init__(self, gen: Generator) -> None:
        self.gen = gen
        self.rng = gen.rng

    def gen_expr(self, target: Type, depth: int = 0) -> TExpr:
        if depth >= MAX_DEPTH:
            return self._fallback(target)

        productions: list[tuple[int, TExpr | None]] = []

        # Variable reference
        bindings = self.gen.scope.bindings_of_type(target)
        if not self.gen.in_fn_lit:
            if bindings:
                productions.append((20, None))  # sentinel for var
        else:
            fn_bindings = [
                b
                for b in bindings
                if self.gen.scope.lookup(b.name) is not None
                and self._is_fn_lit_local(b.name)
            ]
            bindings = fn_bindings
            if fn_bindings:
                productions.append((20, None))

        # Type-specific productions
        if type_eq(target, INT_T):
            productions.append(
                (
                    10,
                    TIntLit(
                        pos=P, value=self.rng.randint(-100, 100), raw="", annotations=A
                    ),
                )
            )
            if depth < MAX_DEPTH - 1:
                productions.append((5, lambda: self._gen_arith_op(target, depth)))
                productions.append((2, lambda: self._gen_bitwise_op(target, depth)))
                productions.append((2, lambda: self._gen_shift_op(target, depth)))
                productions.append((3, lambda: self._gen_unary_op("-", target, depth)))
                productions.append((2, lambda: self._gen_unary_op("~", target, depth)))
        elif type_eq(target, FLOAT_T):
            v = round(self.rng.uniform(-100.0, 100.0), 2)
            productions.append((10, TFloatLit(pos=P, value=v, raw="", annotations=A)))
            if depth < MAX_DEPTH - 1:
                productions.append((5, lambda: self._gen_arith_op(target, depth)))
                productions.append((3, lambda: self._gen_unary_op("-", target, depth)))
        elif type_eq(target, BOOL_T):
            productions.append(
                (
                    10,
                    TBoolLit(
                        pos=P, value=self.rng.choice([True, False]), annotations=A
                    ),
                )
            )
            if depth < MAX_DEPTH - 1:
                productions.append((3, lambda: self._gen_comparison_op(depth)))
                productions.append((3, lambda: self._gen_equality_op(depth)))
                productions.append((3, lambda: self._gen_logical_op(depth)))
                productions.append(
                    (
                        3,
                        lambda: TUnaryOp(
                            pos=P,
                            op="!",
                            operand=self.gen_expr(BOOL_T, depth + 1),
                            annotations=A,
                        ),
                    )
                )
        elif type_eq(target, BYTE_T):
            productions.append(
                (
                    10,
                    TByteLit(
                        pos=P, value=self.rng.randint(0, 255), raw="", annotations=A
                    ),
                )
            )
            if depth < MAX_DEPTH - 1:
                productions.append((5, lambda: self._gen_arith_op(target, depth)))
                productions.append((2, lambda: self._gen_bitwise_op(target, depth)))
                productions.append((2, lambda: self._gen_shift_op(target, depth)))
                productions.append((3, lambda: self._gen_unary_op("-", target, depth)))
                productions.append((2, lambda: self._gen_unary_op("~", target, depth)))
        elif type_eq(target, BYTES_T):
            productions.append((10, TBytesLit(pos=P, value=b"hello", annotations=A)))
        elif type_eq(target, STRING_T):
            s = self.rng.choice(["hello", "world", "test", "foo", "bar"])
            productions.append((10, TStringLit(pos=P, value=s, annotations=A)))
        elif type_eq(target, RUNE_T):
            productions.append((10, TRuneLit(pos=P, value="a", annotations=A)))
        elif type_eq(target, NIL_T):
            return TNilLit(pos=P, annotations=A)
        elif isinstance(target, ListT):
            productions.append((10, self._gen_list_lit(target, depth)))
        elif isinstance(target, MapT):
            productions.append((10, self._gen_map_lit(target, depth)))
        elif isinstance(target, SetT):
            productions.append((10, self._gen_set_lit(target, depth)))
        elif isinstance(target, TupleT):
            productions.append((10, self._gen_tuple_lit(target, depth)))
        elif isinstance(target, StructT):
            productions.append((10, self._gen_struct_constructor(target, depth)))
        elif isinstance(target, InterfaceT):
            info = self.gen.pool.interface_info_for(target)
            if info is not None and info.variant_names:
                vname = self.rng.choice(info.variant_names)
                st = self.gen.pool.struct_for_name(vname)
                if st is not None:
                    productions.append((10, self._gen_struct_constructor(st, depth)))
        elif isinstance(target, EnumT):
            variant = self.rng.choice(target.variants)
            productions.append(
                (
                    10,
                    TFieldAccess(
                        pos=P,
                        obj=TVar(pos=P, name=target.name, annotations=A),
                        field=variant,
                        annotations=A,
                    ),
                )
            )
        elif isinstance(target, FnT):
            can_fn_lit = not self.gen.in_fn_lit and depth < MAX_DEPTH - 1
            if self.gen.in_finally and not type_eq(target.ret, VOID_T):
                can_fn_lit = False
            if can_fn_lit:
                productions.append((5, self._gen_fn_lit(target, depth)))
        elif isinstance(target, UnionT):
            safe_members = [
                m for m in target.members if not self._has_invariance_issue(m)
            ]
            if self.gen.in_finally:
                safe_members = [m for m in safe_members if not isinstance(m, FnT)]
            if not safe_members:
                safe_members = list(target.members)
            member = self.rng.choice(safe_members)
            if type_eq(member, NIL_T):
                productions.append((10, TNilLit(pos=P, annotations=A)))
            else:
                productions.append((10, self.gen_expr(member, depth + 1)))

        # ── Cross-type productions (depth-gated) ──
        if depth < MAX_DEPTH - 1:
            # Field access on structs
            fa = self._field_access_candidates(target)
            if fa:
                productions.append((8, lambda c=fa: self._gen_field_access_from(c)))
            # Tuple element access
            ta = self._tuple_access_candidates(target)
            if ta:
                productions.append((5, lambda c=ta: self._gen_tuple_access_from(c)))
            # Indexing (list/map/string/bytes)
            ix = self._index_candidates(target)
            if ix:
                productions.append(
                    (5, lambda c=ix, d=depth: self._gen_index_from(c, d))
                )
            # Slicing
            sl = self._slice_candidates(target)
            if sl:
                productions.append(
                    (3, lambda c=sl, d=depth: self._gen_slice_from(c, d))
                )
            # Builtin function calls
            productions.append(
                (5, lambda: self.gen.builtin_gen.gen_builtin(target, depth))
            )
            # Ternary
            if (
                not type_eq(target, NIL_T)
                and not type_eq(target, VOID_T)
                and self._can_gen_type(target)
            ):
                productions.append((3, lambda: self._gen_ternary(target, depth)))
            # Function/method/fn-value calls
            if not self.gen.in_fn_lit:
                fc = self._callable_fn_candidates(target)
                if fc:
                    productions.append(
                        (5, lambda c=fc, d=depth: self._gen_fn_call(c, d))
                    )
                mc = self._method_call_candidates(target)
                if mc:
                    productions.append(
                        (3, lambda c=mc, d=depth: self._gen_method_call(c, d))
                    )
                fvc = self._fn_value_call_candidates(target)
                if fvc:
                    productions.append(
                        (3, lambda c=fvc, d=depth: self._gen_fn_value_call(c, d))
                    )

        # Optional: nil or inner (skip shortcut if inner has invariance issue)
        if isinstance(target, UnionT) and contains_nil(target):
            if self.rng.random() < 0.3:
                return TNilLit(pos=P, annotations=A)
            inner = remove_nil(target)
            if not type_eq(inner, NIL_T) and not self._has_invariance_issue(inner):
                return self.gen_expr(inner, depth + 1)

        if not productions:
            return self._fallback(target)

        # Weighted selection
        total = sum(w for w, _ in productions)
        r = self.rng.randint(0, total - 1)
        acc = 0
        for weight, item in productions:
            acc += weight
            if r < acc:
                if item is None:
                    if bindings:
                        b = self.rng.choice(bindings)
                        return TVar(pos=P, name=b.name, annotations=A)
                    return self._fallback(target)
                if callable(item):
                    result = item()
                    if result is not None:
                        return result
                    return self._fallback(target)
                # Fix up raw for int/float/byte literals
                if isinstance(item, TIntLit):
                    item.raw = str(item.value)
                elif isinstance(item, TFloatLit):
                    item.raw = str(item.value)
                elif isinstance(item, TByteLit):
                    item.raw = "0x" + format(item.value, "02x")
                return item
        return self._fallback(target)

    # ── Phase 1: Binary and unary operators ──

    def _contains_fn(self, t: Type) -> bool:
        """Check if a type is or transitively contains a fn type."""
        if isinstance(t, FnT):
            return True
        if isinstance(t, ListT):
            return self._contains_fn(t.element)
        if isinstance(t, MapT):
            return self._contains_fn(t.key) or self._contains_fn(t.value)
        if isinstance(t, SetT):
            return self._contains_fn(t.element)
        if isinstance(t, TupleT):
            return any(self._contains_fn(e) for e in t.elements)
        if isinstance(t, UnionT):
            return any(self._contains_fn(m) for m in t.members)
        return False

    def _gen_arith_op(self, target: Type, depth: int) -> TExpr:
        op = self.rng.choice(["+", "-", "*", "/", "%"])
        left = self.gen_expr(target, depth + 1)
        right = self.gen_expr(target, depth + 1)
        return TBinaryOp(pos=P, op=op, left=left, right=right, annotations=A)

    def _gen_bitwise_op(self, target: Type, depth: int) -> TExpr:
        op = self.rng.choice(["&", "|", "^"])
        left = self.gen_expr(target, depth + 1)
        right = self.gen_expr(target, depth + 1)
        return TBinaryOp(pos=P, op=op, left=left, right=right, annotations=A)

    def _gen_shift_op(self, target: Type, depth: int) -> TExpr:
        op = self.rng.choice(["<<", ">>"])
        left = self.gen_expr(target, depth + 1)
        right = self.gen_expr(INT_T, depth + 1)
        return TBinaryOp(pos=P, op=op, left=left, right=right, annotations=A)

    def _gen_unary_op(self, op: str, target: Type, depth: int) -> TExpr:
        operand = self.gen_expr(target, depth + 1)
        return TUnaryOp(pos=P, op=op, operand=operand, annotations=A)

    def _gen_comparison_op(self, depth: int) -> TExpr:
        operand_type = self.rng.choice([INT_T, FLOAT_T, BYTE_T, RUNE_T, STRING_T])
        op = self.rng.choice(["<", "<=", ">", ">="])
        left = self.gen_expr(operand_type, depth + 1)
        right = self.gen_expr(operand_type, depth + 1)
        return TBinaryOp(pos=P, op=op, left=left, right=right, annotations=A)

    def _has_nondet_type(self, t: Type) -> bool:
        """Types that resolve non-deterministically (interface/union dispatch)."""
        if isinstance(t, (InterfaceT, UnionT)):
            return True
        if isinstance(t, TupleT):
            return any(self._has_nondet_type(e) for e in t.elements)
        if isinstance(t, ListT):
            return self._has_nondet_type(t.element)
        if isinstance(t, MapT):
            return self._has_nondet_type(t.key) or self._has_nondet_type(t.value)
        if isinstance(t, SetT):
            return self._has_nondet_type(t.element)
        return False

    def _gen_equality_op(self, depth: int) -> TExpr:
        candidates = [
            t
            for t in self.gen.pool.pool
            if not self._contains_fn(t)
            and not type_eq(t, VOID_T)
            and not type_eq(t, NIL_T)
            and not self._has_nondet_type(t)
        ]
        if not candidates:
            candidates = [INT_T]
        operand_type = self.rng.choice(candidates)
        op = self.rng.choice(["==", "!="])
        left = self.gen_expr(operand_type, depth + 1)
        right = self.gen_expr(operand_type, depth + 1)
        return TBinaryOp(pos=P, op=op, left=left, right=right, annotations=A)

    def _gen_logical_op(self, depth: int) -> TExpr:
        op = self.rng.choice(["&&", "||"])
        left = self.gen_expr(BOOL_T, depth + 1)
        right = self.gen_expr(BOOL_T, depth + 1)
        return TBinaryOp(pos=P, op=op, left=left, right=right, annotations=A)

    def _can_gen_type(self, t: Type) -> bool:
        """Check if we can generate a valid expression of this type in current context."""
        if isinstance(t, FnT):
            if self.gen.in_fn_lit:
                return False
            if self.gen.in_finally and not type_eq(t.ret, VOID_T):
                return False
        if isinstance(t, UnionT):
            return all(self._can_gen_type(m) for m in t.members)
        return True

    # ── Phase 2: Field access, tuple access, indexing, slicing ──

    def _accessible_bindings(self) -> list[Binding]:
        all_b = self.gen.scope.all_bindings()
        if not self.gen.in_fn_lit:
            return all_b
        return [b for b in all_b if self._is_fn_lit_local(b.name)]

    def _field_access_candidates(self, target: Type) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for b in self._accessible_bindings():
            if isinstance(b.typ, StructT):
                for fname, ftype in b.typ.fields.items():
                    if type_eq(ftype, target):
                        result.append((b.name, fname))
        return result

    def _tuple_access_candidates(self, target: Type) -> list[tuple[str, int]]:
        result: list[tuple[str, int]] = []
        for b in self._accessible_bindings():
            if isinstance(b.typ, TupleT):
                for i, elem in enumerate(b.typ.elements):
                    if type_eq(elem, target):
                        result.append((b.name, i))
        return result

    def _index_candidates(self, target: Type) -> list[tuple[str, Type]]:
        """Find (var_name, key_type) for indexable bindings yielding target."""
        result: list[tuple[str, Type]] = []
        for b in self._accessible_bindings():
            if isinstance(b.typ, ListT) and type_eq(b.typ.element, target):
                result.append((b.name, INT_T))
            elif isinstance(b.typ, MapT) and type_eq(b.typ.value, target):
                result.append((b.name, b.typ.key))
            elif type_eq(b.typ, STRING_T) and type_eq(target, RUNE_T):
                result.append((b.name, INT_T))
            elif type_eq(b.typ, BYTES_T) and type_eq(target, BYTE_T):
                result.append((b.name, INT_T))
        return result

    def _slice_candidates(self, target: Type) -> list[str]:
        result: list[str] = []
        for b in self._accessible_bindings():
            if isinstance(target, ListT) and isinstance(b.typ, ListT):
                if type_eq(b.typ, target):
                    result.append(b.name)
            elif type_eq(target, STRING_T) and type_eq(b.typ, STRING_T):
                result.append(b.name)
            elif type_eq(target, BYTES_T) and type_eq(b.typ, BYTES_T):
                result.append(b.name)
        return result

    def _gen_field_access_from(self, candidates: list[tuple[str, str]]) -> TExpr:
        var_name, field = self.rng.choice(candidates)
        return TFieldAccess(
            pos=P,
            obj=TVar(pos=P, name=var_name, annotations=A),
            field=field,
            annotations=A,
        )

    def _gen_tuple_access_from(self, candidates: list[tuple[str, int]]) -> TExpr:
        var_name, idx = self.rng.choice(candidates)
        return TTupleAccess(
            pos=P,
            obj=TVar(pos=P, name=var_name, annotations=A),
            index=idx,
            annotations=A,
        )

    def _gen_index_from(self, candidates: list[tuple[str, Type]], depth: int) -> TExpr:
        var_name, key_type = self.rng.choice(candidates)
        key_expr = self.gen_expr(key_type, depth + 1)
        return TIndex(
            pos=P,
            obj=TVar(pos=P, name=var_name, annotations=A),
            index=key_expr,
            annotations=A,
        )

    def _gen_slice_from(self, candidates: list[str], depth: int) -> TExpr:
        var_name = self.rng.choice(candidates)
        low = self.gen_expr(INT_T, depth + 1)
        high = self.gen_expr(INT_T, depth + 1)
        return TSlice(
            pos=P,
            obj=TVar(pos=P, name=var_name, annotations=A),
            low=low,
            high=high,
            annotations=A,
        )

    # ── Phase 3: Ternary, function calls, method calls ──

    def _gen_ternary(self, target: Type, depth: int) -> TExpr:
        # For union targets, pin both branches to the same member to avoid
        # the checker widening to a common interface.
        gen_target = target
        if isinstance(target, UnionT):
            safe = [m for m in target.members if not self._has_invariance_issue(m)]
            if self.gen.in_finally:
                safe = [m for m in safe if not isinstance(m, FnT)]
            if not safe:
                safe = list(target.members)
            gen_target = self.rng.choice(safe)
            if type_eq(gen_target, NIL_T):
                gen_target = target
        cond = self.gen_expr(BOOL_T, depth + 1)
        then_expr = self.gen_expr(gen_target, depth + 1)
        else_expr = self.gen_expr(gen_target, depth + 1)
        return TTernary(
            pos=P,
            cond=cond,
            then_expr=then_expr,
            else_expr=else_expr,
            annotations=A,
        )

    def _callable_fn_candidates(self, target: Type) -> list[tuple[str, FnT]]:
        result: list[tuple[str, FnT]] = []
        for fn_name, fn_type in self.gen.functions.items():
            if type_eq(fn_type.ret, target):
                if all(self._can_gen_type(pt) for pt in fn_type.params):
                    result.append((fn_name, fn_type))
        return result

    def _method_call_candidates(self, target: Type) -> list[tuple[str, str, FnT]]:
        result: list[tuple[str, str, FnT]] = []
        for b in self._accessible_bindings():
            if isinstance(b.typ, StructT):
                for mname, mtype in b.typ.methods.items():
                    if type_eq(mtype.ret, target):
                        if all(self._can_gen_type(pt) for pt in mtype.params[1:]):
                            result.append((b.name, mname, mtype))
        return result

    def _fn_value_call_candidates(self, target: Type) -> list[tuple[str, FnT]]:
        result: list[tuple[str, FnT]] = []
        for b in self._accessible_bindings():
            if isinstance(b.typ, FnT) and type_eq(b.typ.ret, target):
                if all(self._can_gen_type(pt) for pt in b.typ.params):
                    result.append((b.name, b.typ))
        return result

    def _gen_fn_call(self, candidates: list[tuple[str, FnT]], depth: int) -> TExpr:
        fn_name, fn_type = self.rng.choice(candidates)
        args: list[TArg] = []
        for pt in fn_type.params:
            args.append(TArg(pos=P, name=None, value=self.gen_expr(pt, depth + 1)))
        return TCall(
            pos=P,
            func=TVar(pos=P, name=fn_name, annotations=A),
            args=args,
            annotations=A,
        )

    def _gen_method_call(
        self, candidates: list[tuple[str, str, FnT]], depth: int
    ) -> TExpr:
        var_name, method_name, method_type = self.rng.choice(candidates)
        args: list[TArg] = []
        # Skip first param (this)
        for pt in method_type.params[1:]:
            args.append(TArg(pos=P, name=None, value=self.gen_expr(pt, depth + 1)))
        return TCall(
            pos=P,
            func=TFieldAccess(
                pos=P,
                obj=TVar(pos=P, name=var_name, annotations=A),
                field=method_name,
                annotations=A,
            ),
            args=args,
            annotations=A,
        )

    def _gen_fn_value_call(
        self, candidates: list[tuple[str, FnT]], depth: int
    ) -> TExpr:
        var_name, fn_type = self.rng.choice(candidates)
        args: list[TArg] = []
        for pt in fn_type.params:
            args.append(TArg(pos=P, name=None, value=self.gen_expr(pt, depth + 1)))
        return TCall(
            pos=P,
            func=TVar(pos=P, name=var_name, annotations=A),
            args=args,
            annotations=A,
        )

    # ── Existing helpers ──

    def _has_invariance_issue(self, t: Type) -> bool:
        """Check if a type contains a collection whose element is interface-typed."""
        if isinstance(t, ListT):
            return isinstance(t.element, InterfaceT) or self._has_invariance_issue(
                t.element
            )
        if isinstance(t, MapT):
            return (
                isinstance(t.value, InterfaceT)
                or isinstance(t.key, InterfaceT)
                or self._has_invariance_issue(t.value)
                or self._has_invariance_issue(t.key)
            )
        if isinstance(t, SetT):
            return isinstance(t.element, InterfaceT) or self._has_invariance_issue(
                t.element
            )
        if isinstance(t, TupleT):
            return any(self._has_invariance_issue(e) for e in t.elements)
        if isinstance(t, UnionT):
            return any(self._has_invariance_issue(m) for m in t.members)
        return False

    def _is_fn_lit_local(self, name: str) -> bool:
        """Check if name is declared in the innermost scope (fn lit's own scope)."""
        if not self.gen.scope.scopes:
            return False
        return name in self.gen.scope.scopes[-1]

    def _fallback(self, target: Type) -> TExpr:
        """Last resort: literal or zero value."""
        if type_eq(target, INT_T):
            return TIntLit(pos=P, value=0, raw="0", annotations=A)
        if type_eq(target, FLOAT_T):
            return TFloatLit(pos=P, value=0.0, raw="0.0", annotations=A)
        if type_eq(target, BOOL_T):
            return TBoolLit(pos=P, value=False, annotations=A)
        if type_eq(target, BYTE_T):
            return TByteLit(pos=P, value=0, raw="0x00", annotations=A)
        if type_eq(target, BYTES_T):
            return TBytesLit(pos=P, value=b"", annotations=A)
        if type_eq(target, STRING_T):
            return TStringLit(pos=P, value="", annotations=A)
        if type_eq(target, RUNE_T):
            return TRuneLit(pos=P, value="a", annotations=A)
        if type_eq(target, NIL_T):
            return TNilLit(pos=P, annotations=A)
        if type_eq(target, VOID_T):
            return TIntLit(pos=P, value=0, raw="0", annotations=A)
        if isinstance(target, ListT):
            return self._gen_list_lit(target, MAX_DEPTH)
        if isinstance(target, MapT):
            return self._gen_map_lit(target, MAX_DEPTH)
        if isinstance(target, SetT):
            return self._gen_set_lit(target, MAX_DEPTH)
        if isinstance(target, TupleT):
            return self._gen_tuple_lit(target, MAX_DEPTH)
        if isinstance(target, StructT):
            return self._gen_struct_constructor(target, MAX_DEPTH)
        if isinstance(target, InterfaceT):
            info = self.gen.pool.interface_info_for(target)
            if info is not None and info.variant_names:
                st = self.gen.pool.struct_for_name(info.variant_names[0])
                if st is not None:
                    return self._gen_struct_constructor(st, MAX_DEPTH)
        if isinstance(target, EnumT):
            return TFieldAccess(
                pos=P,
                obj=TVar(pos=P, name=target.name, annotations=A),
                field=target.variants[0],
                annotations=A,
            )
        if isinstance(target, UnionT):
            for m in target.members:
                if not type_eq(m, NIL_T) and not self._has_invariance_issue(m):
                    return self._fallback(m)
            # All non-nil members have invariance issues; prefer nil if available
            if contains_nil(target):
                return TNilLit(pos=P, annotations=A)
            for m in target.members:
                if not type_eq(m, NIL_T):
                    return self._fallback(m)
            return TNilLit(pos=P, annotations=A)
        if isinstance(target, FnT):
            if not self.gen.in_finally or type_eq(target.ret, VOID_T):
                return self._gen_fn_lit(target, MAX_DEPTH)
            # Can't generate non-void fn lit in finally — try a variable
            for b in self._accessible_bindings():
                if type_eq(b.typ, target):
                    return TVar(pos=P, name=b.name, annotations=A)
            return self._gen_fn_lit(target, MAX_DEPTH)
        return TIntLit(pos=P, value=0, raw="0", annotations=A)

    def _gen_list_lit(self, target: ListT, depth: int) -> TExpr:
        n = self.rng.randint(1, 3)
        elements = [self.gen_expr(target.element, depth + 1) for _ in range(n)]
        return TListLit(pos=P, elements=elements, annotations=A)

    def _gen_map_lit(self, target: MapT, depth: int) -> TExpr:
        # Single entry to avoid duplicate key errors
        k = self.gen_expr(target.key, depth + 1)
        v = self.gen_expr(target.value, depth + 1)
        return TMapLit(pos=P, entries=[(k, v)], annotations=A)

    def _gen_set_lit(self, target: SetT, depth: int) -> TExpr:
        n = self.rng.randint(1, 3)
        elements = [self.gen_expr(target.element, depth + 1) for _ in range(n)]
        return TSetLit(pos=P, elements=elements, annotations=A)

    def _gen_tuple_lit(self, target: TupleT, depth: int) -> TExpr:
        elements = [self.gen_expr(e, depth + 1) for e in target.elements]
        return TTupleLit(pos=P, elements=elements, annotations=A)

    def _gen_struct_constructor(self, target: StructT, depth: int) -> TExpr:
        args: list[TArg] = []
        for fname, ftype in target.fields.items():
            val = self.gen_expr(ftype, depth + 1)
            args.append(TArg(pos=P, name=fname, value=val))
        return TCall(
            pos=P,
            func=TVar(pos=P, name=target.name, annotations=A),
            args=args,
            annotations=A,
        )

    def _gen_fn_lit(self, fn_type: FnT, depth: int) -> TExpr:
        params: list[TParam] = []
        param_names: list[str] = []
        used: set[str] = set()
        for pt in fn_type.params:
            pname = self.gen.names.var_name(used)
            used.add(pname)
            param_names.append(pname)
            params.append(TParam(pos=P, name=pname, typ=make_ttype(pt), annotations=A))

        # Save/restore generator state
        old_fn_lit = self.gen.in_fn_lit
        self.gen.in_fn_lit = True

        self.gen.scope.enter_scope()
        for pname, pt in zip(param_names, fn_type.params):
            self.gen.scope.declare(pname, pt)

        # Generate body
        if type_eq(fn_type.ret, VOID_T):
            body = self.gen.stmt_gen.gen_block(self.rng.randint(1, 2), must_return=None)
        else:
            ret_expr = self.gen_expr(fn_type.ret, depth + 1)
            body = [TReturnStmt(pos=P, value=ret_expr, annotations=A)]

        self.gen.scope.exit_scope()
        self.gen.in_fn_lit = old_fn_lit

        return TFnLit(
            pos=P,
            params=params,
            ret=make_ttype(fn_type.ret),
            body=body,
            annotations=A,
        )
