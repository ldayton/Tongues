"""Shared helpers for checking whether an expression reads a given variable."""

from __future__ import annotations

from ..taytsh.ast import (
    TBinaryOp,
    TCall,
    TExpr,
    TFieldAccess,
    TIndex,
    TListLit,
    TMapLit,
    TSetLit,
    TSlice,
    TTernary,
    TTupleAccess,
    TTupleLit,
    TUnaryOp,
    TVar,
)


def expr_reads(name: str, expr: TExpr) -> bool:
    if isinstance(expr, TVar):
        return expr.name == name
    if isinstance(expr, TBinaryOp):
        return expr_reads(name, expr.left) or expr_reads(name, expr.right)
    if isinstance(expr, TUnaryOp):
        return expr_reads(name, expr.operand)
    if isinstance(expr, TTernary):
        return (
            expr_reads(name, expr.cond)
            or expr_reads(name, expr.then_expr)
            or expr_reads(name, expr.else_expr)
        )
    if isinstance(expr, TFieldAccess):
        return expr_reads(name, expr.obj)
    if isinstance(expr, TTupleAccess):
        return expr_reads(name, expr.obj)
    if isinstance(expr, TIndex):
        return expr_reads(name, expr.obj) or expr_reads(name, expr.index)
    if isinstance(expr, TSlice):
        return (
            expr_reads(name, expr.obj)
            or expr_reads(name, expr.low)
            or expr_reads(name, expr.high)
        )
    if isinstance(expr, TCall):
        if expr_reads(name, expr.func):
            return True
        for a in expr.args:
            if expr_reads(name, a.value):
                return True
        return False
    if isinstance(expr, TListLit):
        for e in expr.elements:
            if expr_reads(name, e):
                return True
        return False
    if isinstance(expr, TTupleLit):
        for e in expr.elements:
            if expr_reads(name, e):
                return True
        return False
    if isinstance(expr, TSetLit):
        for e in expr.elements:
            if expr_reads(name, e):
                return True
        return False
    if isinstance(expr, TMapLit):
        for k, v in expr.entries:
            if expr_reads(name, k) or expr_reads(name, v):
                return True
        return False
    return False


def target_reads(name: str, target: TExpr) -> bool:
    if isinstance(target, TVar):
        return False
    if isinstance(target, TIndex):
        return expr_reads(name, target.obj) or expr_reads(name, target.index)
    if isinstance(target, TFieldAccess):
        return expr_reads(name, target.obj)
    if isinstance(target, TTupleAccess):
        return expr_reads(name, target.obj)
    if isinstance(target, TSlice):
        return (
            expr_reads(name, target.obj)
            or expr_reads(name, target.low)
            or expr_reads(name, target.high)
        )
    return False
