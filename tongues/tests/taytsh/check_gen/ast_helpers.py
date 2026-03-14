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
    return TVar(pos=P, annotations=A, name=name)


def int_lit(n: int) -> TIntLit:
    return TIntLit(pos=P, annotations=A, value=n, raw=str(n))


def nil_lit() -> TNilLit:
    return TNilLit(pos=P, annotations=A)


def bool_lit(v: bool) -> TBoolLit:
    return TBoolLit(pos=P, annotations=A, value=v)


def str_lit(s: str) -> TStringLit:
    return TStringLit(pos=P, annotations=A, value=s)


def let_stmt(name: str, typ: Type, value=None) -> TLetStmt:
    return TLetStmt(pos=P, annotations=A, name=name, typ=make_ttype(typ), value=value)


def return_stmt(value=None) -> TReturnStmt:
    return TReturnStmt(pos=P, annotations=A, value=value)


def expr_stmt(expr) -> TExprStmt:
    return TExprStmt(pos=P, annotations=A, expr=expr)


def call(func_name: str, *args) -> TCall:
    targs = [TArg(pos=P, name=None, value=a) for a in args]
    return TCall(pos=P, annotations=A, func=var(func_name), args=targs)


def field_access(obj, field: str) -> TFieldAccess:
    return TFieldAccess(pos=P, annotations=A, obj=obj, field=field)


def if_stmt(cond, then_body, else_body=None) -> TIfStmt:
    return TIfStmt(
        pos=P, annotations=A, cond=cond, then_body=then_body, else_body=else_body
    )


def while_stmt(cond, body) -> TWhileStmt:
    return TWhileStmt(pos=P, annotations=A, cond=cond, body=body)


def not_op(expr) -> TUnaryOp:
    return TUnaryOp(pos=P, annotations=A, op="!", operand=expr)


def binop(left, op, right) -> TBinaryOp:
    return TBinaryOp(pos=P, annotations=A, op=op, left=left, right=right)


def ternary(cond, then_expr, else_expr) -> TTernary:
    return TTernary(
        pos=P, annotations=A, cond=cond, then_expr=then_expr, else_expr=else_expr
    )


def filler_stmts(n: int) -> list:
    """Generate n harmless filler statements to increase distance between guard and use."""
    return [let_stmt(f"_filler{i}", INT_T, int_lit(i)) for i in range(n)]


# ── Module builders ──────────────────────────────────────────────


def main_fn(body: list) -> TFnDecl:
    return TFnDecl(
        pos=P,
        annotations=A,
        name="Main",
        params=[],
        ret=TPrimitive(pos=P, kind="void"),
        body=body,
    )


def build_module(decls: list[TModuleItem], body: list) -> TModule:
    return TModule(decls=list(decls) + [main_fn(body)])


def empty_module() -> TModule:
    return build_module([], [])


# ── Zero-value generation ────────────────────────────────────────


def zero_value(t: Type) -> TExpr:
    """Generate a zero/default value expression for a type."""
    if type_eq(t, INT_T):
        return TIntLit(pos=P, annotations=A, value=0, raw="0")
    if type_eq(t, FLOAT_T):
        return TFloatLit(pos=P, annotations=A, value=0.0, raw="0.0")
    if type_eq(t, BOOL_T):
        return TBoolLit(pos=P, annotations=A, value=False)
    if type_eq(t, BYTE_T):
        return TByteLit(pos=P, annotations=A, value=0, raw="0x00")
    if type_eq(t, BYTES_T):
        return TBytesLit(pos=P, annotations=A, value=b"")
    if type_eq(t, STRING_T):
        return TStringLit(pos=P, annotations=A, value="")
    if type_eq(t, RUNE_T):
        return TRuneLit(pos=P, annotations=A, value="a")
    if type_eq(t, NIL_T):
        return TNilLit(pos=P, annotations=A)
    if isinstance(t, StructT):
        args = [TArg(pos=P, name=f, value=zero_value(ft)) for f, ft in t.fields.items()]
        return TCall(
            pos=P,
            annotations=A,
            func=TVar(pos=P, annotations=A, name=t.name),
            args=args,
        )
    if isinstance(t, ListT):
        return TListLit(pos=P, annotations=A, elements=[])
    if isinstance(t, MapT):
        return TMapLit(pos=P, annotations=A, entries=[])
    if isinstance(t, SetT):
        return TSetLit(pos=P, annotations=A, elements=[])
    if isinstance(t, TupleT):
        return TTupleLit(
            pos=P, annotations=A, elements=[zero_value(e) for e in t.elements]
        )
    if isinstance(t, EnumT):
        return TFieldAccess(
            pos=P,
            annotations=A,
            obj=TVar(pos=P, annotations=A, name=t.name),
            field=t.variants[0],
        )
    if isinstance(t, UnionT):
        for m in t.members:
            if type_eq(m, NIL_T):
                return TNilLit(pos=P, annotations=A)
        return zero_value(t.members[0])
    return TIntLit(pos=P, annotations=A, value=0, raw="0")


# ── Error filtering ──────────────────────────────────────────────

NOISE_ERRORS = {"missing Main", "variable used before assignment"}


def filter_noise_errors(msgs: list[str]) -> list[str]:
    return [m for m in msgs if not any(n in m for n in NOISE_ERRORS)]
