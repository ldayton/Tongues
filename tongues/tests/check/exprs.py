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
    TIntLit,
    TListLit,
    TMapLit,
    TNilLit,
    TParam,
    TReturnStmt,
    TRuneLit,
    TSetLit,
    TStringLit,
    TTernary,
    TTupleLit,
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
                productions.append((5, None))  # arithmetic sentinel (tag=arith)
        elif type_eq(target, FLOAT_T):
            v = round(self.rng.uniform(-100.0, 100.0), 2)
            productions.append((10, TFloatLit(pos=P, value=v, raw="", annotations=A)))
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
                productions.append((5, None))  # comparison sentinel
        elif type_eq(target, BYTE_T):
            productions.append(
                (
                    10,
                    TByteLit(
                        pos=P, value=self.rng.randint(0, 255), raw="", annotations=A
                    ),
                )
            )
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
        for weight, expr in productions:
            acc += weight
            if r < acc:
                if expr is not None:
                    # Fix up raw for int/float/byte literals
                    if isinstance(expr, TIntLit):
                        expr.raw = str(expr.value)
                    elif isinstance(expr, TFloatLit):
                        expr.raw = str(expr.value)
                    elif isinstance(expr, TByteLit):
                        expr.raw = "0x" + format(expr.value, "02x")
                    return expr
                # Sentinel — pick a variable
                if bindings:
                    b = self.rng.choice(bindings)
                    return TVar(pos=P, name=b.name, annotations=A)
                return self._fallback(target)
        return self._fallback(target)

    def _has_invariance_issue(self, t: Type) -> bool:
        """Check if a type is a collection whose element is an interface/struct."""
        if isinstance(t, ListT):
            return isinstance(t.element, (InterfaceT, StructT))
        if isinstance(t, MapT):
            return isinstance(t.value, (InterfaceT, StructT)) or isinstance(
                t.key, (InterfaceT, StructT)
            )
        if isinstance(t, SetT):
            return isinstance(t.element, (InterfaceT, StructT))
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
                if not type_eq(m, NIL_T):
                    return self._fallback(m)
            return TNilLit(pos=P, annotations=A)
        if isinstance(target, FnT):
            if not self.gen.in_finally or type_eq(target.ret, VOID_T):
                return self._gen_fn_lit(target, MAX_DEPTH)
            # Can't generate non-void fn lit in finally (return would be flagged)
            return TIntLit(pos=P, value=0, raw="0", annotations=A)
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
