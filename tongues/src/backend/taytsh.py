"""Taytsh codegen backend — produces valid .ty source from a checked TModule."""

from __future__ import annotations

from ..taytsh.ast import TModule
from ..taytsh.emit import _Emitter


def emit_taytsh(module: TModule) -> str:
    """Emit a TModule as valid Taytsh source."""
    return _Emitter(True).emit(module)
