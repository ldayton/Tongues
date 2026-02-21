"""Topological ordering of declarations for top-to-bottom languages."""

from __future__ import annotations

from ..taytsh.ast import (
    TEnumDecl,
    TFuncType,
    TIdentType,
    TInterfaceDecl,
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
        out = set()
        for p in typ.params:
            out |= _collect_type_names(p)
        return out
    if isinstance(typ, TUnionType):
        out = set()
        for m in typ.members:
            out |= _collect_type_names(m)
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


def order_decls(decls: list[TModuleItem]) -> list[TModuleItem]:
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
        return decls
    deps: dict[int, set[int]] = {idx: set() for idx, _ in type_decls}
    for idx, decl in type_decls:
        if isinstance(decl, TStructDecl):
            if decl.parent and decl.parent in name_to_idx:
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
        remaining = sorted([idx for idx, _ in type_decls if decls[idx] not in result])
        for idx in remaining:
            result.append(decls[idx])
    for _, decl in other_decls:
        result.append(decl)
    return result
