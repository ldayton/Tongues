"""Analyze which int variables need 64-bit width (long in Java).

Tongues `int` is 64-bit but Java `int` is 32-bit. When strict_math is off,
the Java backend maps int→int for convenience. This pass identifies module-level
variables that genuinely need 64-bit and annotates them so backends can emit `long`.
"""

from __future__ import annotations

from ..taytsh.ast import (
    TBinaryOp,
    TCall,
    TExpr,
    TIntLit,
    TLetStmt,
    TModule,
    TTernary,
    TUnaryOp,
    TVar,
)
from ..taytsh.check import Checker


def analyze_int_width(module: TModule, checker: Checker) -> None:
    """Annotate module-level int variables that need 64-bit width."""
    if module.strict_math:
        return
    wide: set[str] = set()
    for decl in module.decls:
        if isinstance(decl, TLetStmt):
            if decl.value is not None and _is_seed(decl.value):
                wide.add(decl.name)
    changed = True
    while changed:
        changed = False
        for decl in module.decls:
            if isinstance(decl, TLetStmt) and decl.name not in wide:
                if decl.value is not None and _refs_wide(decl.value, wide):
                    wide.add(decl.name)
                    changed = True
    for decl in module.decls:
        if isinstance(decl, TLetStmt) and decl.name in wide:
            decl.annotations["intwidth.wide"] = "true"


def _is_seed(expr: TExpr) -> bool:
    """Check if an expression is inherently 64-bit."""
    if isinstance(expr, TBinaryOp):
        if expr.op == "<<" and isinstance(expr.right, TIntLit):
            if expr.right.value >= 31:
                return True
        return _is_seed(expr.left) or _is_seed(expr.right)
    if isinstance(expr, TUnaryOp):
        return _is_seed(expr.operand)
    if isinstance(expr, TCall) and isinstance(expr.func, TVar):
        if expr.func.name == "Pow":
            return True
    if isinstance(expr, TTernary):
        return _is_seed(expr.then_expr) or _is_seed(expr.else_expr)
    return False


def _refs_wide(expr: TExpr, wide: set[str]) -> bool:
    """Check if expression references any wide variable."""
    if isinstance(expr, TVar):
        return expr.name in wide
    if isinstance(expr, TBinaryOp):
        return _refs_wide(expr.left, wide) or _refs_wide(expr.right, wide)
    if isinstance(expr, TUnaryOp):
        return _refs_wide(expr.operand, wide)
    if isinstance(expr, TTernary):
        return _refs_wide(expr.then_expr, wide) or _refs_wide(expr.else_expr, wide)
    return False
