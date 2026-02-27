"""Shared AST factories, module builders, zero-value generation, and error filtering."""

from __future__ import annotations

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
    TFnDecl,
    TIfStmt,
    TIntLit,
    TLetStmt,
    TListLit,
    TMapLit,
    TModule,
    TModuleItem,
    TNilLit,
    TPrimitive,
    TReturnStmt,
    TRuneLit,
    TSetLit,
    TStringLit,
    TTernary,
    TTupleLit,
    TUnaryOp,
    TVar,
    TWhileStmt,
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
    EnumT,
    InterfaceT,
    ListT,
    MapT,
    SetT,
    StructT,
    TupleT,
    Type,
    UnionT,
    type_eq,
)

from .types import make_ttype

P = Pos(1, 1)
A: dict[str, str] = {}


# ── AST node factories ───────────────────────────────────────────


def var(name: str) -> TVar:
    return TVar(pos=P, name=name, annotations=A)


def int_lit(n: int) -> TIntLit:
    return TIntLit(pos=P, value=n, raw=str(n), annotations=A)


def nil_lit() -> TNilLit:
    return TNilLit(pos=P, annotations=A)


def bool_lit(v: bool) -> TBoolLit:
    return TBoolLit(pos=P, value=v, annotations=A)


def str_lit(s: str) -> TStringLit:
    return TStringLit(pos=P, value=s, annotations=A)


def let_stmt(name: str, typ: Type, value=None) -> TLetStmt:
    return TLetStmt(pos=P, name=name, typ=make_ttype(typ), value=value, annotations=A)


def return_stmt(value=None) -> TReturnStmt:
    return TReturnStmt(pos=P, value=value, annotations=A)


def expr_stmt(expr) -> TExprStmt:
    return TExprStmt(pos=P, expr=expr, annotations=A)


def call(func_name: str, *args) -> TCall:
    targs = [TArg(pos=P, name=None, value=a) for a in args]
    return TCall(pos=P, func=var(func_name), args=targs, annotations=A)


def field_access(obj, field: str) -> TFieldAccess:
    return TFieldAccess(pos=P, obj=obj, field=field, annotations=A)


def if_stmt(cond, then_body, else_body=None) -> TIfStmt:
    return TIfStmt(
        pos=P, cond=cond, then_body=then_body, else_body=else_body, annotations=A
    )


def while_stmt(cond, body) -> TWhileStmt:
    return TWhileStmt(pos=P, cond=cond, body=body, annotations=A)


def not_op(expr) -> TUnaryOp:
    return TUnaryOp(pos=P, op="!", operand=expr, annotations=A)


def binop(left, op, right) -> TBinaryOp:
    return TBinaryOp(pos=P, op=op, left=left, right=right, annotations=A)


def ternary(cond, then_expr, else_expr) -> TTernary:
    return TTernary(
        pos=P, cond=cond, then_expr=then_expr, else_expr=else_expr, annotations=A
    )


def filler_stmts(n: int) -> list:
    """Generate n harmless filler statements to increase distance between guard and use."""
    return [let_stmt(f"_filler{i}", INT_T, int_lit(i)) for i in range(n)]


# ── Module builders ──────────────────────────────────────────────


def main_fn(body: list) -> TFnDecl:
    return TFnDecl(
        pos=P,
        name="Main",
        params=[],
        ret=TPrimitive(pos=P, kind="void"),
        body=body,
        annotations=A,
    )


def build_module(decls: list[TModuleItem], body: list) -> TModule:
    return TModule(decls=list(decls) + [main_fn(body)])


def empty_module() -> TModule:
    return build_module([], [])


# ── Zero-value generation ────────────────────────────────────────


def zero_value(t: Type) -> TExpr:
    """Generate a zero/default value expression for a type."""
    if type_eq(t, INT_T):
        return TIntLit(pos=P, value=0, raw="0", annotations=A)
    if type_eq(t, FLOAT_T):
        return TFloatLit(pos=P, value=0.0, raw="0.0", annotations=A)
    if type_eq(t, BOOL_T):
        return TBoolLit(pos=P, value=False, annotations=A)
    if type_eq(t, BYTE_T):
        return TByteLit(pos=P, value=0, raw="0x00", annotations=A)
    if type_eq(t, BYTES_T):
        return TBytesLit(pos=P, value=b"", annotations=A)
    if type_eq(t, STRING_T):
        return TStringLit(pos=P, value="", annotations=A)
    if type_eq(t, RUNE_T):
        return TRuneLit(pos=P, value="a", annotations=A)
    if type_eq(t, NIL_T):
        return TNilLit(pos=P, annotations=A)
    if isinstance(t, StructT):
        args = [TArg(pos=P, name=f, value=zero_value(ft)) for f, ft in t.fields.items()]
        return TCall(
            pos=P,
            func=TVar(pos=P, name=t.name, annotations=A),
            args=args,
            annotations=A,
        )
    if isinstance(t, ListT):
        return TListLit(pos=P, elements=[], annotations=A)
    if isinstance(t, MapT):
        return TMapLit(pos=P, entries=[], annotations=A)
    if isinstance(t, SetT):
        return TSetLit(pos=P, elements=[], annotations=A)
    if isinstance(t, TupleT):
        return TTupleLit(
            pos=P, elements=[zero_value(e) for e in t.elements], annotations=A
        )
    if isinstance(t, EnumT):
        return TFieldAccess(
            pos=P,
            obj=TVar(pos=P, name=t.name, annotations=A),
            field=t.variants[0],
            annotations=A,
        )
    if isinstance(t, UnionT):
        for m in t.members:
            if type_eq(m, NIL_T):
                return TNilLit(pos=P, annotations=A)
        return zero_value(t.members[0])
    return TIntLit(pos=P, value=0, raw="0", annotations=A)


# ── Error filtering ──────────────────────────────────────────────

NOISE_ERRORS = {"missing Main", "variable used before assignment"}


def filter_noise_errors(msgs: list[str]) -> list[str]:
    return [m for m in msgs if not any(n in m for n in NOISE_ERRORS)]
