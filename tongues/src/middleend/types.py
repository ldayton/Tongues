"""Type propagation pass: checker writes types onto AST annotations directly.

The checker's check_expr stores expr.annotations["type"] = type_name(result)
for every expression that has an annotations field and a non-None type.
This module exists solely as the pipeline entry point.
"""

from __future__ import annotations

from ..taytsh.ast import TModule
from ..taytsh.check import Checker


def propagate_types(module: TModule, checker: Checker) -> None:
    """No-op: types are now annotated directly in check_expr."""
    pass
