"""Patch error-typed variable declarations using checker annotations.

When pycheck can't infer a type during hoisting, it sets TPrimitive("error").
The checker later annotates expressions with resolved types. This pass reads
those annotations and patches the TLetStmt.typ so all backends benefit.
"""

from __future__ import annotations

from ..taytsh.ast import (
    Pos,
    TAssignStmt,
    TCatch,  # noqa: F401
    TDefault,  # noqa: F401
    TFnDecl,
    TForStmt,
    TIfStmt,
    TLetStmt,
    TMatchCase,  # noqa: F401
    TMatchStmt,
    TModule,
    TStmt,
    TStructDecl,
    TTryStmt,
    TType,
    TWhileStmt,
    TFuncType,
    TIdentType,
    TListType,
    TMapType,
    TPrimitive,
    TSetType,
    TTupleType,
    TUnionType,
    TVar,
)


_PRIMITIVE_KINDS = frozenset(
    ("int", "float", "bool", "byte", "bytes", "string", "rune", "void", "nil")
)


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split on sep at bracket/paren depth 0."""
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    sep_len = len(sep)
    while i < len(s):
        c = s[i]
        if c in ("[", "("):
            depth += 1
        elif c in ("]", ")"):
            depth -= 1
        elif depth == 0 and s[i : i + sep_len] == sep:
            parts.append(s[start:i].strip())
            i += sep_len
            start = i
            continue
        i += 1
    parts.append(s[start:].strip())
    return parts


def parse_type_ann(ann: str, pos: Pos) -> TType | None:
    """Convert a checker type-annotation string back to a TType node."""
    ann = ann.strip()
    if not ann or ann == "error":
        return None
    if ann in _PRIMITIVE_KINDS:
        return TPrimitive(pos, ann)
    if ann.startswith("list[") and ann.endswith("]"):
        inner = parse_type_ann(ann[5:-1], pos)
        if inner is not None:
            return TListType(pos, inner)
        return None
    if ann.startswith("map[") and ann.endswith("]"):
        parts = _split_top_level(ann[4:-1], ", ")
        if len(parts) == 2:
            k = parse_type_ann(parts[0], pos)
            v = parse_type_ann(parts[1], pos)
            if k is not None and v is not None:
                return TMapType(pos, k, v)
        return None
    if ann.startswith("set[") and ann.endswith("]"):
        inner = parse_type_ann(ann[4:-1], pos)
        if inner is not None:
            return TSetType(pos, inner)
        return None
    if ann.startswith("fn[") and ann.endswith("]"):
        parts = _split_top_level(ann[3:-1], ", ")
        resolved: list[TType] = []
        for p in parts:
            t = parse_type_ann(p, pos)
            if t is None:
                return None
            resolved.append(t)
        return TFuncType(pos, resolved)
    if ann.startswith("(") and ann.endswith(")"):
        parts = _split_top_level(ann[1:-1], ", ")
        resolved: list[TType] = []
        for p in parts:
            t = parse_type_ann(p, pos)
            if t is None:
                return None
            resolved.append(t)
        return TTupleType(pos, resolved)
    if " | " in ann:
        parts = _split_top_level(ann, " | ")
        resolved: list[TType] = []
        for p in parts:
            t = parse_type_ann(p, pos)
            if t is None:
                return None
            resolved.append(t)
        return TUnionType(pos, resolved)
    # Assume it's a struct/interface/enum name
    if ann[0].isupper() or ann[0] == "_":
        return TIdentType(pos, ann)
    return None


def _has_error(typ: TType) -> bool:
    """Check if a TType tree contains any TPrimitive("error")."""
    if isinstance(typ, TPrimitive):
        return typ.kind == "error"
    if isinstance(typ, TListType):
        return _has_error(typ.element)
    if isinstance(typ, TMapType):
        return _has_error(typ.key) or _has_error(typ.value)
    if isinstance(typ, TSetType):
        return _has_error(typ.element)
    if isinstance(typ, TTupleType):
        return any(_has_error(e) for e in typ.elements)
    if isinstance(typ, TFuncType):
        return any(_has_error(p) for p in typ.params)
    if isinstance(typ, TUnionType):
        return any(_has_error(m) for m in typ.members)
    return False


def _find_assign_ann(name: str, stmts: list[TStmt]) -> str:
    """Find the first assignment to name in stmts and return its RHS type annotation."""
    for stmt in stmts:
        if isinstance(stmt, TAssignStmt):
            if isinstance(stmt.target, TVar) and stmt.target.name == name:
                return stmt.value.annotations.get("type", "")
        if isinstance(stmt, TIfStmt):
            result = _find_assign_ann(name, stmt.then_body)
            if result:
                return result
            if stmt.else_body is not None:
                result = _find_assign_ann(name, stmt.else_body)
                if result:
                    return result
    return ""


def _patch_stmts(stmts: list[TStmt]) -> None:
    """Walk a statement list and patch error-typed TLetStmts."""
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, TLetStmt) and _has_error(stmt.typ):
            ann = ""
            if stmt.value is not None:
                ann = stmt.value.annotations.get("type", "")
            if ann:
                resolved = parse_type_ann(ann, stmt.pos)
                if resolved is not None:
                    stmt.typ = resolved
                    continue
            # Fall back: scan following statements for first assignment
            following_ann = _find_assign_ann(stmt.name, stmts[i + 1 :])
            if following_ann:
                resolved = parse_type_ann(following_ann, stmt.pos)
                if resolved is not None:
                    stmt.typ = resolved
        # Recurse into sub-bodies
        if isinstance(stmt, TIfStmt):
            _patch_stmts(stmt.then_body)
            if stmt.else_body is not None:
                _patch_stmts(stmt.else_body)
        elif isinstance(stmt, TForStmt):
            _patch_stmts(stmt.body)
        elif isinstance(stmt, TWhileStmt):
            _patch_stmts(stmt.body)
        elif isinstance(stmt, TTryStmt):
            _patch_stmts(stmt.body)
            for catch in stmt.catches:
                _patch_stmts(catch.body)
            if stmt.finally_body is not None:
                _patch_stmts(stmt.finally_body)
        elif isinstance(stmt, TMatchStmt):
            for case in stmt.cases:
                _patch_stmts(case.body)
            if stmt.default is not None:
                _patch_stmts(stmt.default.body)


def patch_error_types(module: TModule) -> None:
    """Patch all error-typed variable declarations in the module."""
    for decl in module.decls:
        if isinstance(decl, TFnDecl):
            _patch_stmts(decl.body)
        elif isinstance(decl, TStructDecl):
            for method in decl.methods:
                _patch_stmts(method.body)
