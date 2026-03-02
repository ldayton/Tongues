"""Topological ordering of declarations for top-to-bottom languages."""

from __future__ import annotations

from ..taytsh.ast import (
    TEnumDecl,
    TFuncType,
    TIdentType,
    TInterfaceDecl,
    TLetStmt,
    TListType,
    TMapType,
    TModuleItem,
    TOptionalType,
    TSetType,
    TStructDecl,
    TTupleType,
    TType,
    TUnionType,
)


def _collect_type_names(typ: TType) -> set[str]:
    """Recursively collect all TIdentType names from a type tree."""
    if isinstance(typ, TIdentType):
        return {typ.name}
    if isinstance(typ, (TListType, TSetType)):
        return _collect_type_names(typ.element)
    if isinstance(typ, TOptionalType):
        return _collect_type_names(typ.inner)
    if isinstance(typ, TMapType):
        return _collect_type_names(typ.key) | _collect_type_names(typ.value)
    if isinstance(typ, TTupleType):
        out: set[str] = set()
        for el in typ.elements:
            out |= _collect_type_names(el)
        return out
    if isinstance(typ, TFuncType):
        out: set[str] = set()
        for p in typ.params:
            for n in _collect_type_names(p):
                out.add(n)
        return out
    if isinstance(typ, TUnionType):
        out: set[str] = set()
        for m in typ.members:
            for n in _collect_type_names(m):
                out.add(n)
        return out
    return set()


def _pop_min(ready: list[int]) -> int:
    """Remove and return the smallest element from a list."""
    min_i = 0
    for i in range(1, len(ready)):
        if ready[i] < ready[min_i]:
            min_i = i
    val = ready[min_i]
    ready[min_i] = ready[-1]
    ready.pop()
    return val


def _lets_before_fns(decls: list[TModuleItem]) -> list[TModuleItem]:
    """Reorder so module-level lets precede fns (Perl's `my` is lexically scoped)."""
    lets: list[TModuleItem] = []
    rest: list[TModuleItem] = []
    for decl in decls:
        if isinstance(decl, TLetStmt):
            lets.append(decl)
        else:
            rest.append(decl)
    if not lets or not rest:
        return decls
    return lets + rest


def order_decls(
    decls: list[TModuleItem], *, lets_first: bool = False
) -> list[TModuleItem]:
    """Topologically sort type declarations; preserve source order as tiebreaker."""
    type_decls: list[tuple[int, TModuleItem]] = []
    other_decls: list[tuple[int, TModuleItem]] = []
    name_to_idx: dict[str, int] = {}
    for i, decl in enumerate(decls):
        if isinstance(decl, (TStructDecl, TInterfaceDecl, TEnumDecl)):
            type_decls.append((i, decl))
            name_to_idx[decl.name] = i
        else:
            other_decls.append((i, decl))
    if not type_decls:
        return _lets_before_fns(decls) if lets_first else decls
    deps: dict[int, set[int]] = {idx: set() for idx, _ in type_decls}
    for idx, decl in type_decls:
        if isinstance(decl, TStructDecl):
            if decl.parent is not None and decl.parent in name_to_idx:
                deps[idx].add(name_to_idx[decl.parent])
            for field in decl.fields:
                for name in _collect_type_names(field.typ):
                    if name in name_to_idx and name_to_idx[name] != idx:
                        deps[idx].add(name_to_idx[name])
    in_degree: dict[int, int] = {idx: 0 for idx, _ in type_decls}
    for idx, dep_set in deps.items():
        for dep in dep_set:
            if dep in in_degree:
                in_degree[idx] += 1
    ready: list[int] = [idx for idx, deg in in_degree.items() if deg == 0]
    result: list[TModuleItem] = []
    while ready:
        idx = _pop_min(ready)
        result.append(decls[idx])
        for other_idx, dep_set in deps.items():
            if idx in dep_set:
                dep_set.discard(idx)
                in_degree[other_idx] -= 1
                if in_degree[other_idx] == 0:
                    ready.append(other_idx)
    if len(result) < len(type_decls):
        remaining: list[int] = []
        for orig_idx, _ in type_decls:
            if decls[orig_idx] not in result:
                remaining.append(orig_idx)
        remaining.sort()
        for orig_idx in remaining:
            result.append(decls[orig_idx])
    if lets_first:
        for _, decl in other_decls:
            if isinstance(decl, TLetStmt):
                result.append(decl)
        for _, decl in other_decls:
            if not isinstance(decl, TLetStmt):
                result.append(decl)
    else:
        for _, decl in other_decls:
            result.append(decl)
    return result
